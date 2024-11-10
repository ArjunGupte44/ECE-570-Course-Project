import os
import warnings
warnings.filterwarnings("ignore")

from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
from models.resnet import blip_resnet
from models.vision_encoder import create_vision_encoder

import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import Transformer

#Note, list below has default 14 diseases, but Vicuna was used to add 4 more diseases, making the total 18
CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]

class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 args,
                 tokenizer=None,
                 image_size = 224,
                 prompt = '',
                 ):
        super().__init__()
        self.args = args
        
        vision_width = 2048
        self.visual_encoder = blip_resnet(args)
        # self.visual_encoder = create_vision_encoder(args)

        self.cls_head = nn.Linear(vision_width+512, 14*4) #18 for 18 cls heads (one for each disease) -> each head outputs probs for 4 options (BLA, POS, NEG, UNC)
        nn.init.normal_(self.cls_head.weight, std=0.001)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)

        self.vision_proj = nn.Linear(vision_width, 512)

        self.tokenizer = tokenizer   
        
        decoder_config = BertConfig.from_json_file('configs/bert_config.json')
        decoder_config.encoder_width = vision_width
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)
        
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        self.memory = Transformer(d_model=512,
                                  num_encoder_layers=2,
                                  num_decoder_layers=2,
                                  num_queries=1)
        
    def forward(self, image, caption, cls_labels, clip_memory, criterion_cls, base_probs):
        image_embeds, avg_embeds = self.visual_encoder(image) #avg_embeds is the average pooled features and is the dotted bxo with green rectangles in it in the paper
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        ##The following block is Cross Modal Feature Enhancement##
        ##########################
        # NxKxC -> KxNxC
        clip_memory = torch.permute(clip_memory, (1, 0, 2))
        query_embed = self.vision_proj(avg_embeds) #This generates the green "query" box in the diagram in the paper
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None) #Retrieval and aggregation of top-k features from report database -> yellow box in paper
        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds, hs), 1) #This is the concat operation in the paper right after CMFE
        ##########################

        #The next two lines perform disease classification and have each of the 18 heads have 4 outputs (each output is a prob for that result (BLA, POS, NEG, UNC) for that disease)
        #This is the salmon colored DDP block in the paper
        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4, 14)

        # logit adjustment
        cls_preds[:, 1, :] += torch.log(torch.from_numpy(base_probs)).view(1, -1).to(image.device)
        loss_cls = criterion_cls(cls_preds, cls_labels) #Compare predicted disease classifications with gt labels
        
        text = self.tokenizer(caption, padding='longest', truncation=True, return_tensors="pt").to(image.device)
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100) 
        decoder_targets[:,:self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
          
        loss_lm = decoder_output.loss                
        return loss_lm, loss_cls
        
    def generate(self, image, clip_memory, sample=False, num_beams=3, max_length=100, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds, avg_embeds = self.visual_encoder(image) 
        
        # NxKxC -> KxNxC
        clip_memory = torch.permute(clip_memory, (1, 0, 2))
        query_embed = self.vision_proj(avg_embeds)
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None)
        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds, hs), 1)

        # classification branch
        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4, 14)
        cls_preds = F.softmax(cls_preds, dim=1)
        cls_preds_logits = cls_preds[:, 1, :14]
        cls_preds = torch.argmax(cls_preds, dim=1).cpu().numpy().tolist()

        prompts = []
        for j in range(len(cls_preds)):
            prompt = ' '.join([SCORES[c] for c in cls_preds[j]])+' '
            prompts.append(prompt)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        text = self.tokenizer(prompts, return_tensors="pt")
        input_ids = text.input_ids.to(image.device)
        attn_masks = text.attention_mask.to(image.device)
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 
        attn_masks = attn_masks[:, :-1] 
        
        #beam search
        outputs = self.text_decoder.generate(input_ids=input_ids,
                                             min_length=min_length, # 4.25 Transformers
                                             max_new_tokens=max_length,
                                             num_beams=num_beams,
                                             eos_token_id=self.tokenizer.sep_token_id,
                                             pad_token_id=self.tokenizer.pad_token_id, 
                                             repetition_penalty=repetition_penalty,
                                             attention_mask = attn_masks,
                                             **model_kwargs)            
            
        captions = []    
        for i, output in enumerate(outputs):
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(prompts[i]):])
        return captions, cls_preds, cls_preds_logits

def blip_decoder(args, tokenizer, **kwargs):
    model = BLIP_Decoder(args, tokenizer, **kwargs)
    return model    
    
