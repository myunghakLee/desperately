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

train_loader, test_loader = CIFAR100.get_data(512)

# +
# model = torch.hub.load("pytorch/vision", f'vgg16', pretrained=True)
# model = vit(model)

model = conv.vgg_bn(class_num = 100, depth = 16)

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
    train_acc.append(acc)
    
    acc = utils.test(model, test_loader, device, epoch)
    test_acc.append(acc)
    
    if acc > best_acc:
        best_acc = acc
        stack = 0
    else:
        stack+=1
    
    if stack > 2:
        stack = 0
        lr_scheduler.step()
        step_epoch.append(epoch)
        print("STEP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
torch.save(model, "de.pth")
# -


outuput[0].shape
