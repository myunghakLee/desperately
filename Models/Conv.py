# -*- coding: utf-8 -*-
# +
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import torch
import copy

from tqdm import tqdm
import random

# +
repo = 'pytorch/vision'
model = torch.hub.load('pytorch/vision', 'resnet50', weights=True) # , force_reload=True

model


# -

def get_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# +
# class vgg(nn.Module):
#     def __init__(self, model):
#         pass
        
    
def vgg(class_num, depth, bn = True, checkpoint = None, pretrained = False):
    assert depth in [11, 13, 16, 19], "Depth must be select in [11, 13, 16, 19]"
    
    layer_num = {11 : 21, 
                 13 : 25, 
                 16 : 31, 
                 19 : 37}  # 각 vgg의 depth마다의 layer 수

    if checkpoint:
        model = torch.load(checkpoint)
        assert model.classifier[-1].out_features == class_num, "out_feature is wrong"
    else:
        model = torch.hub.load("pytorch/vision", f'vgg{depth}', pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
        
    assert len(model.features) == layer_num[depth], "Model depth is wrong"
    
    return model


# -

def vgg_bn(class_num, depth, bn = True, checkpoint = None, pretrained = False):
    assert depth in [11, 13, 16, 19], "Depth must be select in [11, 13, 16, 19]"
    
    layer_num = {11 : 29, 
                 13 : 35, 
                 16 : 44, 
                 19 : 53}  # 각 vgg의 depth마다의 layer 수

    if checkpoint:
        model = torch.load(checkpoint)
        assert model.classifier[-1].out_features == class_num, "out_feature is wrong"
    else:
        model = torch.hub.load("pytorch/vision", f'vgg{depth}_bn', pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
        
    assert len(model.features) == layer_num[depth], "Model depth is wrong"
    
    return model


def resnet(class_num, depth, bn = True, checkpoint = None, pretrained = False):
    assert depth in [18, 34, 50, 101, 152], "Depth must be select in [18, 34, 50, 101, 152]"
    
    pram_num = {11 : 11689512, 
                 13 : 21797672, 
                 50 : 25557032, 
                 101 : 44549160,
                 152 : 60192808}  # 각 vgg의 depth마다의 layer 수

    if checkpoint:
        model = torch.load(checkpoint)
    else:
        model = torch.hub.load("pytorch/vision", f'resnet{depth}', pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, class_num)
        
    assert get_params(model) == pram_num[depth], "Model depth is wrong"
    
    return model


def efficientnet(class_num, depth, bn = True, checkpoint = None, pretrained = False):
    assert depth in [0, 1, 2, 3, 4, 5, 6, 7], "Depth must be select in [18, 34, 50, 101, 152]"
    
    pram_num = {0 : 5288548, 
                1 : 7794184, 
                2 : 9109994, 
                3 : 12233232,
                4 : 19341616,
                5 : 30389784,
                6 : 43040704}  # 각 vgg의 depth마다의 layer 수

    if checkpoint:
        model = torch.load(checkpoint)
    else:
        model = torch.hub.load("pytorch/vision", f'efficientnet_b{depth}', pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
        
    assert get_params(model) == pram_num[depth], "Model depth is wrong"
    
    return model



def efficientnetv2(class_num, depth, bn = True, checkpoint = None, pretrained = False):
    assert depth in ["l", "m", "s"], "Depth must be select in [l, m, s]"
    
    pram_num = {"l" : 118515272, 
                "m" : 54139356, 
                "s" : 21458488}  # 각 vgg의 depth마다의 layer 수

    if checkpoint:
        model = torch.load(checkpoint)
    else:
        model = torch.hub.load("pytorch/vision", f'efficientnet_v2_{depth}', pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
        
    assert get_params(model) == pram_num[depth], "Model depth is wrong"
    
    return model


# +
repo = 'pytorch/vision'
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True) # , force_reload=True

model
# -

torch.hub.list('pytorch/vision')
