#!/usr/bin/env python
# coding: utf-8
# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"


# %%


import torch
model = torch.hub.load("pytorch/vision", "vit_b_16")

import torchvision
from Models.transformer import VisionTransformer as vit
import Models.Conv as conv

from DataLoader import CIFAR100
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

import utils

import numpy as np
import torch.backends.cudnn as cudnn
import random


# %%
train_loader, test_loader = CIFAR100.get_data(58*3)


# %%
from Models import Conv

depth = 101

model = torch.load(f"saved_models/resnet/resnet{depth}.pth").module
teacher = Conv.resnet_feature(100, depth, model)
student = Conv.resnet_feature(100, depth, pretrained="IMAGENET1K_V1")


# %%


device = "cuda"

teacher = teacher.to(device)
teacher = torch.nn.DataParallel(teacher, device_ids=[0, 1])

student = student.to(device)
student = torch.nn.DataParallel(student, device_ids=[0, 1])


# %%


criterion_onlylabel = lambda a,b : mse(a*b, b)
criterion_CE = nn.CrossEntropyLoss()
mse = nn.MSELoss()
softmax = torch.nn.Softmax(dim = 1)
criterion_KLD = torch.nn.KLDivLoss(reduction="batchmean")
criterion_response = lambda a,b : criterion_KLD(torch.log_softmax(a, dim=1),torch.softmax(b, dim=1))


# %%


S_optimizer = optim.SGD(student.parameters(), lr=0.05, momentum=0.9)
T_optimizer = optim.SGD(teacher.parameters(), lr=0.05, momentum=0.9)
CE_loss = nn.CrossEntropyLoss()


# %%


S_scheduler = torch.optim.lr_scheduler.MultiStepLR(S_optimizer, milestones=[9,18,27,28,29,30,31], gamma=0.1)
T_scheduler = torch.optim.lr_scheduler.MultiStepLR(T_optimizer, milestones=[1,2,3,4,5,6,7], gamma=0.1)


# %%


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
best_acc = 0.0
stack = 0

accs_train = []
accs_test = []


# %%
utils.test(teacher, test_loader,device)
utils.test(student, test_loader,device)


# %%
# 1. distill_loss에 2가 곱햊졌던걸 4을 곱함
# 2. 전체 loss를 5로 나눔(원래는 2)
# 3. lr을 0.01에서 0.05로 수정
# 4. 일단 feature knowledge에서 output은 빼자


# %%


student_test_accs = []
layer_num = 4

for epoch in range(100):
    
    print(f"lr : {S_scheduler.get_last_lr()}")
    if S_scheduler.get_last_lr()[0] < 0.000001:
        break
        
    T_correct = 0
    S_correct = 0
    all_data = 0
    
    loss_distill = []
    loss_CE = []
    loss_response = []
    student.train()
    teacher.eval()
    for img, label in tqdm(train_loader):
        input_data = img.to(device)
        label = label.to(device)
        
        
        all_data += len(input_data)
        input_lrp = utils.get_LRP_img(input_data, label, teacher, criterion_CE, T_optimizer, mean=1.5, std = 0.1, mult = 0.4).cuda()
        
        S_optimizer.zero_grad()
        T_optimizer.zero_grad()

        layer = random.randint(0,  layer_num)
        output_s, fk = student(input_data,layer)
        output_t, fk_lrp = teacher(input_lrp,layer)
        
        # channal wise pooling
#         fk = torch.mean(fk, dim=1)
#         fk_lrp = torch.mean(fk_lrp, dim=1)
        
        distill_loss = (mse(torch.mean(fk, dim=1), torch.mean(fk_lrp, dim=1)) + mse(torch.mean(fk, dim=2), torch.mean(fk_lrp, dim=2))) / 2
                    
        CE_loss = criterion_CE(output_s, label)
        
        response_loss = criterion_response(output_s, output_t)
        
        T_correct += sum(label == torch.argmax(output_t, dim=1))
        S_correct += sum(label == torch.argmax(output_s, dim=1))
        
        loss_CE.append(CE_loss.item())
        loss_distill.append(distill_loss.item())
        loss_response.append(response_loss.item())
        
        loss = (distill_loss * 3 + CE_loss + response_loss) / 5
        loss.backward()
        S_optimizer.step()

    print("distill loss : ", sum(loss_distill) / len(loss_distill))
    print("general loss : ", sum(loss_CE) / len(loss_CE))
    print("response loss : ", sum(loss_response) / len(loss_response))
    
    print(f"Teacher acc: {T_correct / all_data}")
    print(f"Student acc: {S_correct / all_data}")

    test_acc = utils.test(student, test_loader,device, epoch) # student도 변하는거 확인 완료함
    
#     if test_acc > best_acc + 0.01:
#         stack = 0
#         best_acc = test_acc
        
#     else:
#         stack+=1
    
#     if stack > 3:  
    S_scheduler.step()
    stack = 0
        
    student_test_accs.append(test_acc.item())
    print("=" * 100)


# %%
# a = torch.zeros((4,5,6,7))
# nn.AvgPool2d(5)(a).shape


# %%


# torch.mean(a, dim=1).shape


# %%


# distill loss를 2배 키워보는것도 좋을지도


# %%


utils.test(teacher, test_loader,device, epoch) # student도 변하는거 확인 완료함
utils.test(student, test_loader,device, epoch) # student도 변하는거 확인 완료함


# %%


torch.save(student, "saved_models/resnet/resnet{depth}_student.pth")


# %%


import json

with open(f"saved_models/resnet/resnet{depth}.json", "w") as f:
    json.dump({"student_test_accs" : student_test_accs}, f)


# %%


# from Models import Conv

# depth = 101

# model = torch.load(f"saved_models/vgg/vgg{depth}.pth").module
# teacher = Conv.resnet_feature(100, depth, model)
# student = Conv.resnet_feature(100, depth, pretrained="IMAGENET1K_V1")


# device = "cuda"

# teacher = teacher.to(device)
# teacher = torch.nn.DataParallel(teacher, device_ids=[0, 1])

# student = student.to(device)
# student = torch.nn.DataParallel(student, device_ids=[0, 1])

# criterion_onlylabel = lambda a,b : mse(a*b, b)
# criterion_CE = nn.CrossEntropyLoss()
# mse = nn.MSELoss()
# softmax = torch.nn.Softmax(dim = 1)
# criterion_KLD = torch.nn.KLDivLoss(reduction="batchmean")
# criterion_response = lambda a,b : criterion_KLD(torch.log_softmax(a, dim=1),torch.softmax(b, dim=1))

# S_optimizer = optim.SGD(student.parameters(), lr=0.05, momentum=0.9)
# T_optimizer = optim.SGD(teacher.parameters(), lr=0.05, momentum=0.9)
# CE_loss = nn.CrossEntropyLoss()

# S_scheduler = torch.optim.lr_scheduler.MultiStepLR(S_optimizer, milestones=[1,2,3,4,5,6,7], gamma=0.1)
# T_scheduler = torch.optim.lr_scheduler.MultiStepLR(T_optimizer, milestones=[1,2,3,4,5,6,7], gamma=0.1)



# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# np.random.seed(0)
# cudnn.benchmark = False
# cudnn.deterministic = True
# random.seed(0)
# best_acc = 0.0
# stack = 0

# accs_train = []
# accs_test = []

# utils.test(teacher, test_loader,device)
# utils.test(student, test_loader,device)

