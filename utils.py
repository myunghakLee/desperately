# -*- coding: utf-8 -*-
from tqdm import tqdm
import torch

import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import torch
import copy


def get_LRP_img(img, label, model, criterion, optimizer, std = 0.1, mean = 1.0):
    img.requires_grad = True
    img.retain_grad = True
    
    output, _ = model(img)

    loss = criterion(output, label)
    loss.backward()
    optimizer.zero_grad()

    with torch.no_grad():

        img_lrp = (img*img.grad).clone()
        img_lrp = f2(img_lrp)

        for i in range(len(img_lrp)):
            img_lrp[i] = to_gaussian(img_lrp[i], std = std, mean = mean)

        img_lrp = img*img_lrp # img_lrp가 음수값인것 지움
    return img_lrp


def train_vanilla(model, data, optimizer, criterion, device, epoch = 0):
    all_data, correct = 0, 0
    model = model.train()
    for img, label in tqdm(data):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(img)
        if isinstance(output, tuple):
            output = output[0]
            
        loss = criterion(output, label)
        loss.backward()
        
        optimizer.step()
        
        correct += sum(label == torch.argmax(output, dim=1))
        all_data += len(img)
            
    print(f"{epoch} \t train acc : {correct / all_data}")
    return correct / all_data


def test(model, data, device, epoch = 0):
    all_data, correct = 0, 0
    model = model.eval()
    for img, label in tqdm(data):
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            if isinstance(output, tuple):
                output = output[0]

            correct += sum(label == torch.argmax(output, dim=1))
            all_data += len(img)
            
    print(f"{epoch} \t test acc : {correct / all_data}")
    return correct / all_data


def normalize_max1(w):
    for i in range(len(w)):
        w[i] = w[i] / torch.max(abs(w[i]))
    return w


softmax = torch.nn.Softmax(dim=1)
softmax2d = lambda b: softmax(torch.flatten(b, start_dim = 1)).reshape(b.shape)
# f2 = lambda w, _=None: softmax2d(normalize_max1(-w)) * len(w[0])
f2 = lambda w, _= None: softmax2d(-w) * len(w[0])

to_gaussian = lambda arr, mean = 1, std = 1: ((arr - torch.mean(arr))/ (torch.std(arr) + 0.00001)) * std + mean


def get_LRP_img(img, label, model, criterion, optimizer, std = 0.1, mean = 1.0):
    img.requires_grad = True
    img.retain_grad = True
    
    output, _ = model(img)

    loss = criterion(output, label)
    loss.backward()
    optimizer.zero_grad()

    with torch.no_grad():

        img_lrp = (img*img.grad).clone()
        img_lrp = f2(img_lrp)

        for i in range(len(img_lrp)):
            img_lrp[i] = to_gaussian(img_lrp[i], std = std, mean = mean)

        img_lrp = img*img_lrp # img_lrp가 음수값인것 지움
    return img_lrp
