import os
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertTokenizer

from models.transformer import Transformer
from models.med import BertLMHeadModel, BertConfig, BertModel
from models.vision_encoder import create_vision_encoder


class MR_Generator(nn.Module):
    def __init__(self, args, tokenizer, image_size, prompt):
        self.args = args
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.prompt = prompt

        #Define vision encoder
        self.vit_encoder = create_vision_encoder()

        



    #Forward function for training model
    def forward(self):


    #Generate function for inferencing
    def generate(self):





def create_model(args, tokenizer, image_size, prompt):
    mrg_model = MR_Generator(args, tokenizer, image_size, prompt)
    


