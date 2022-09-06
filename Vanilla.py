# +
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
# -

train_loader, test_loader = CIFAR100.get_data(117*3)

# +
# model = torch.hub.load("pytorch/vision", f'vgg16', pretrained=True)
# model = vit(model)

model = vit(class_num = 100, pretrained = True, is_vanilla=True)

# +
device = "cuda"

model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

# -

CE = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,3,4,5,6,7], gamma=0.1)

# +
train_acc = []
test_acc = []
best_acc = 0.0
stack = 0
step_epoch = []

for epoch in range(100):
    print(f"lr : {lr_scheduler.get_last_lr()}")
    acc = utils.train_vanilla(model, train_loader, optimizer, CE, device, epoch)
    train_acc.append(acc.item())
    
    acc = utils.test(model, test_loader, device, epoch)
    test_acc.append(acc.item())
    
    if acc > best_acc:
        best_acc = acc
        stack = 0
    else:
        stack+=1
    
    if stack > 4:
        stack = 0
        lr_scheduler.step()
        step_epoch.append(epoch)
        print("STEP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
torch.save(model, "de.pth")
# -


model

# !rm saved_models/vit_b_teacher_16_86_66.pth

torch.save(model, "saved_models/vit_b_teacher_16_86_66.pth")

test_acc

train_acc
