# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np


# +
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import torch
import copy

import torch.backends.cudnn as cudnn
import random


# -

def set_seed(seed = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


# +
# def get_LRP_img(img, label, model, criterion, optimizer, std = 0.1, mean = 1.0):
#     img.requires_grad = True
#     img.retain_grad = True
    
#     output, _ = model(img)

#     loss = criterion(output, label)
#     loss.backward()
#     optimizer.zero_grad()

#     with torch.no_grad():

#         img_lrp = (img*img.grad).clone()
#         img_lrp = f2(img_lrp)

#         for i in range(len(img_lrp)):
#             img_lrp[i] = to_gaussian(img_lrp[i], std = std, mean = mean)

#         img_lrp = img*img_lrp # img_lrp가 음수값인것 지움
#     return img_lrp
# -

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


def normalize_max(w, mult = 0.4):
    for i in range(len(w)):
        w[i] = w[i] / torch.max(abs(w[i])) * mult
    return w


softmax = torch.nn.Softmax(dim=1)
softmax2d = lambda b: softmax(torch.flatten(b, start_dim = 1)).reshape(b.shape)
f2 = lambda w, mult=None: softmax2d(normalize_max(-w, mult)) * len(w[0])
# f2 = lambda w, mult = None: softmax2d(-w * mult) * len(w[0])

# +
to_gaussian = lambda arr, mean = 1, std = 1: ((arr - torch.mean(arr))/ (torch.std(arr) + 0.00001)) * std + mean

def to_gaussian_rgb(img, mean, std): # rgb별로 따로 gaussian처리

    for i in range(3):
        img[i] = ((img[i] - torch.mean(img[i])) / (torch.std(img[i]) + 0.00001)) * std[i] + mean[i]  # RGB
    return img


def get_LRP_img(img, label, model, criterion, optimizer, std = 0.1, mean = 1.0, mult = 0.4):
    img.requires_grad = True
    img.retain_grad = True
    
    output = model(img)
    if isinstance(output,tuple):
        output = output[0]

    loss = criterion(output, label)
    loss.backward()
    optimizer.zero_grad()

    with torch.no_grad():
        
        gradient = (img*img.grad).clone()
        gradient = f2(gradient, mult)
#         img.grad *= 50
#         img.grad += 1
        
        for i in range(len(gradient)):
            gradient[i] = to_gaussian(gradient[i], mean = mean, std = std)

#         for i, (m, s) in enumerate(zip(mean,std)):
#             gradient[i] = to_gaussian_rgb(gradient[i], mean = m, std = s)


        mean, std = img.mean(dim=(2,3)), img.std(dim=(2,3))

            
        output_image = img * gradient
        
        for i, (m, s) in enumerate(zip(mean,std)):
            output_image[i] = to_gaussian_rgb(output_image[i], mean = m, std = s)
        
#         print(torch.mean(img.grad))
#         print("="*100)
        
#         img_lrp = f2(img_lrp)

#         for i in range(len(img_lrp)):
#             img_lrp[i] = to_gaussian(img_lrp[i], std = std, mean = mean)

#         img_lrp = img*img_lrp # img_lrp가 음수값인것 지움
    return output_image


# +
def get_LRP_img_plus(img, label, model, criterion, optimizer, mult = 0.4):
    img.requires_grad = True
    img.retain_grad = True
    
    output = model(img)
    if isinstance(output,tuple):
        output = output[0]

    loss = criterion(output, label)
    loss.backward()
    optimizer.zero_grad()

    with torch.no_grad():
        gradient = img.grad
#         gradient = (img*img.grad).clone()
#         gradient = f2(gradient, mult)
#         img.grad *= 50
#         img.grad += 1
        
#         for i in range(len(gradient)):
#             gradient[i] = to_gaussian(gradient[i], mean = 0.0, std = std)

#         for i, (m, s) in enumerate(zip(mean,std)):
#             gradient[i] = to_gaussian_rgb(gradient[i], mean = m, std = s)

    return img + gradient * mult
