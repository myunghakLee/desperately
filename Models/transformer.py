# -*- coding: utf-8 -*-
# +
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import torch
import copy

from tqdm import tqdm
import random
# -

repo = 'pytorch/vision'


def get_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# +
class VisionTransformer(nn.Module):
    def __init__(self, class_num, model = None, pretrained = None, is_vanilla=False):
        super(VisionTransformer, self).__init__()
        
        if model == None:
            if pretrained:
                model = torchvision.models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
            else:
                model = torchvision.models.vit_b_16()
            
        self.patch_size = model.patch_size
        self.image_size = model.image_size
        self.hidden_dim = model.hidden_dim
        self.conv_proj = copy.deepcopy(model.conv_proj)
        self.class_token = copy.deepcopy(model.class_token)
        
        self.encoder = copy.deepcopy(model.encoder)

        self.heads = copy.deepcopy(model.heads)
        self.heads.head = nn.Linear(self.heads.head.in_features, class_num)
        
        self.is_vanilla = is_vanilla
        
        del model
        
    def encoder_block(self, inputs, encoder):
        x = encoder.ln_1(inputs)
        x, _ = encoder.self_attention(query=x, key=x, value=x, need_weights=False)
        x = encoder.dropout(x)
        x = x + inputs

        y = encoder.ln_2(x)
        y = encoder.mlp(y)
        return x + y    
    
    def forward(self, x, layer = 0):
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p


        conv_output = self.conv_proj(x)
        if layer == 0:
            feature_knowledge = conv_output

        x = conv_output.reshape(n, self.hidden_dim, n_h * n_w)


        x = x.permute(0, 2, 1)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        if layer == 1:
            feature_knowledge = x


        # ENCODER
        x = x+self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        for i in range(len(self.encoder.layers)):
            x = self.encoder_block(x, self.encoder.layers[i])

            if layer == 2+i:
                feature_knowledge = x

        x = self.encoder.ln(x)
        

        if layer == 2+len(self.encoder.layers):
            feature_knowledge = x


#         if layer == 3+len(self.encoder.layers):  # ????????????
#             feature_knowledge = x
#             feature_knowledge.retain_grad = True

        x = x[:, 0]

        x = self.heads(x)
        
        return x if self.is_vanilla else x, feature_knowledge


# -

def swin(class_num, name, bn = True, checkpoint = None, pretrained = False):
    assert depth in ["b", "s", "t"], "Depth must be select in [18, 34, 50, 101, 152]"
    
    pram_num = {"b" : 87768224, 
                "s" : 49606258, 
                "t" : 28288354}  # 각 vgg의 depth마다의 layer 수

    if checkpoint:
        model = torch.load(checkpoint)
    else:
        model = torch.hub.load(repo, f'swin_{depth}', pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, class_num)
        
    assert get_params(model) == pram_num[depth], "Model depth is wrong"
    
    return model



