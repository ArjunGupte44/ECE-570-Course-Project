import torch
from torch import nn
import torchvision.models as models

#Custom ViT B/16 vision encoder

class ViT_Encoder(nn.Module):
    def __init__(self, args):
        super(ViT_Encoder, self).__init__()
        self.args = args
        self.model = models.vit_b_16(pretrained=True) #Get ViT_b_16 pre-trained on ImageNet
        self.vit_embed_dim = 768
        self.resnet101_output_dim = 2048
        self.patch_size = 16 #Standard ViT 16 patch size
        self.num_patches = int(self.args.image_size / self.patch_size) ** 2

        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=self.vit_embed_dim, 
                                    kernel_size=self.patch_size, stride=self.patch_size)
        self.fc_layer = nn.Linear(self.vit_embed_dim, self.resnet101_output_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))


    def patchify_image(self, x):
        image_patches = self.conv_layer(x)
        image_patches = image_patches.flatten(2)
        image_patches = image_patches.permute(0, 2, 1)
        # print(image_patches.shape)
        return image_patches

    def forward(self, x):
        image_patches = self.patchify_image(x)
        batch_size = image_patches.shape[0]
        # print(image_patches.shape)

        classification_logits = self.model(x)

        # Repeat the CLS token for each image in the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, vit_embed_dim)
        # Concatenate the CLS token with the patch embeddings
        image_patches = torch.cat((cls_tokens, image_patches), dim=1)

        features = self.model.encoder(image_patches)
        global_features = features[:, 0] # = cls_token
        patch_features = features[:, 1:]

        #Current patch_features shape = batch_size x num_patches x embed_dim
        batch_size = patch_features.shape[0]

        # Pool to reduce patches from 196 to 49
        patch_features = patch_features.reshape(batch_size, 14, 14, self.vit_embed_dim)  # Reshape to (batch_size, 14, 14, 768)
        patch_features = nn.AvgPool2d(kernel_size=2, stride=2)(patch_features.permute(0, 3, 1, 2))  # (batch_size, 768, 14, 14) -> (batch_size, 768, 7, 7)
        patch_features = patch_features.permute(0, 2, 3, 1).reshape(batch_size, -1, self.vit_embed_dim)  # Now shape (batch_size, 49, 768)
        # Transform feature dimension from 768 to 2048
        patch_features = self.fc_layer(patch_features)  # Now shape (batch_size, 49, 2048)

        # patch_features = patch_features.reshape(batch_size, self.num_patches, self.vit_embed_dim)
        # # patch_features = patch_features.permute(0, 2, 1)
        global_features = self.fc_layer(global_features)
        # print(patch_features.shape)
        # print(global_features.shape)
        return patch_features, global_features


def create_vision_encoder(args):
    vit_encoder = ViT_Encoder(args)
    return vit_encoder