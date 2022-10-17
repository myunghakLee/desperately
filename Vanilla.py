# +
import torch
model = torch.hub.load("pytorch/vision", "vit_l_16")

import torchvision
from Models.transformer import VisionTransformer as vit
import Models.Conv as conv

from DataLoader import CIFAR100
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

import utils

# +
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


# -

train_loader, test_loader = CIFAR100.get_data(70*3)

# +
# model = torch.hub.load("pytorch/vision", f'vgg16', pretrained=True)
# model = vit(model)
depths = [11, 13, 16, 19]

device = "cuda"
CE = nn.CrossEntropyLoss()
iter_per_epoch = len(train_loader)

for layer_num in depths:
    print(f"{layer_num}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    model = conv.vgg_feature(class_num = 100, depth=layer_num, pretrained = None)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) # 
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)
    
    model = model.to(device)

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

#     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,3,4,5,6,7], gamma=0.1)

    train_acc = []
    test_acc = []
    best_acc = 0.0
    stack = 0
    step_epoch = []
    for epoch in range(200):
        if epoch > 1:
            train_scheduler.step(epoch)
            
        print(f"lr : {train_scheduler.get_last_lr()}")
        
        acc = utils.train_vanilla(model, train_loader, optimizer, CE, device, warmup_scheduler, epoch)
        train_acc.append(acc.item())
        
        
        model.eval()
        acc = utils.test(model, test_loader, device, epoch)
        test_acc.append(acc.item())

#         if acc > best_acc + 0.005:
#             best_acc = acc
#             stack = 0
#         else:
#             stack+=1

#         if stack > 4:
#             stack = 0
#             lr_scheduler.step()
#             step_epoch.append(epoch)
#             print("STEP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

#     # torch.save(model, "de.pth")

    torch.save(model, f"saved_models/vgg_nonpretrain/vgg{layer_num}.pth")
    torch.cuda.empty_cache()

    with open(f"saved_models/vgg_nonpretrain/vgg{layer_num}.json", "w") as f:
        json.dump({"student_test_accs" : train_acc,
                  "test_acc" : test_acc}, f)    

    del model
    torch.cuda.empty_cache()
      
