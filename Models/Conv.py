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


def vgg_bn(class_num, depth, bn = True, checkpoint = None, pretrained = False, weights = None):
    assert depth in [11, 13, 16, 19], "Depth must be select in [11, 13, 16, 19]"
    
    layer_num = {11 : 29, 
                 13 : 35, 
                 16 : 44, 
                 19 : 53}  # 각 vgg의 depth마다의 layer 수

    if checkpoint:
        model = torch.load(checkpoint)
        assert model.classifier[-1].out_features == class_num, "out_feature is wrong"
    else:
        if weights:
            model = torch.hub.load("pytorch/vision", f'vgg{depth}_bn', weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
        else:
            model = torch.hub.load("pytorch/vision", f'vgg{depth}_bn', pretrained=pretrained)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
        
    assert len(model.features) == layer_num[depth], "Model depth is wrong"
    
    return model


def resnet(class_num, depth, bn = True, checkpoint = None, pretrained = False, weights = None):
    assert depth in [18, 34, 50, 101, 152], "Depth must be select in [18, 34, 50, 101, 152]"
    
    pram_num = {18 : 11689512, 
                 34 : 21797672, 
                 50 : 25557032, 
                 101 : 44549160,
                 152 : 60192808}  # 각 vgg의 depth마다의 parameter 수

    if checkpoint:
        model = torch.load(checkpoint)
    else:
        if weights:
            model = torch.hub.load("pytorch/vision", f'resnet{depth}', weights=weights)
            assert get_params(model) == pram_num[depth], "Model depth is wrong"
            model.fc = nn.Linear(model.fc.in_features, class_num)
        else:
            model = torch.hub.load("pytorch/vision", f'resnet{depth}', pretrained=pretrained)
            assert get_params(model) == pram_num[depth], "Model depth is wrong"
            model.fc = nn.Linear(model.fc.in_features, class_num)
    
    
    return model


class vgg_feature(nn.Module):
    def __init__(self, class_num, depth, model = None, pretrained = None, is_vanilla=False):
        super(vgg_feature, self).__init__()
        if model == None:
            if pretrained:
                self.model = torch.hub.load("pytorch/vision", f'vgg{depth}_bn', weights=pretrained)
            else:
                self.model = torch.hub.load("pytorch/vision", f'vgg{depth}_bn')
        
        else:
            self.model = copy.deepcopy(model)
        
        
        if self.model.classifier[-1].out_features != class_num:
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, class_num)
        
        self.bn_pos = {
            11 : [1, 5, 9, 12, 16, 19, 23, 26],
            13 : [1, 4, 8, 11, 15, 18, 22, 25, 29, 32],
            16 : [1, 4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41],
            19 : [1, 4, 8, 11, 15, 18, 21, 24, 28, 31, 34, 37, 41, 44, 47, 50]
        }
        
        self.length = len(self.bn_pos[depth]) + 1 # model내 bn의 개수, 즉 feature knowledge의 개수
        self.depth = depth
        
        del model

    def forward(self, x, layer = 0):
        
        layer_count = 0
        
        for i, feature in enumerate(self.model.features):
            x = feature(x)
            if i in self.bn_pos[self.depth]:
                if layer_count == layer:
                    h = x.clone()
                layer_count += 1
        
        x = self.model.avgpool(x)
        x= torch.flatten(x,1)
        
        for j, features in enumerate(self.model.classifier):
            x = features(x)
            if j % 3 == 2:
                if layer_count == layer:
                    h = x.clone()
                layer_count += 1
        
        return x, h


class resnet_feature(nn.Module):
    def __init__(self, class_num, depth, model = None, pretrained = None, is_vanilla=False):
        super(resnet_feature, self).__init__()
        if model == None:
            if pretrained:
                self.model = torch.hub.load("pytorch/vision", f'resnet{depth}', weights=pretrained)
            else:
                self.model = torch.hub.load()
        
        else:
            self.model = copy.deepcopy(model)
        
        if self.model.fc.out_features != class_num:
            self.model.fc = nn.Linear(self.model.fc.in_features, class_num)
        
        self.layer0 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool
        )

        self.length = 4 # model내 bn의 개수, 즉 feature knowledge의 개수
        self.depth = depth
        
        del model

    def forward(self, inputs, layer = 0):
        
        h0 = self.layer0(inputs)

        h1 = self.model.layer1(h0)

        h2 = self.model.layer2(h1)

        h3 = self.model.layer3(h2)

        h4 = self.model.layer4(h3)

        h5 = self.model.avgpool(h4)
        h5 = torch.flatten(h5, 1)
        h5 = self.model.fc(h5)
        
        return h5, [h0, h1, h2, h3, h4, h5][layer]


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
# repo = 'pytorch/vision'
# model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True) # , force_reload=True

# model

# +
# torch.hub.list('pytorch/vision')
