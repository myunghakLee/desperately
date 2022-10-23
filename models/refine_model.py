import torch
import torch.nn as nn

class refine_model(nn.Module):
    def __init__(self, model):
        super(refine_model, self).__init__()
        self.flatten = nn.Flatten()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = nn.Sequential(
            model.avgpool,
            self.flatten,
            model.fc
        )
        self.avgpool = model.avgpool
        self.fc = model.fc
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x, layer = 0, is_feat=False):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        # x = self.relu(x)  # 32x32
        f0 = x0

        x1, f1 = self.layer1(x0)  # 32x32
        f1_act = [self.relu(f) for f in f1]
        x2, f2 = self.layer2(x1)  # 16x16
        f2_act = [self.relu(f) for f in f2]
        x3, f3 = self.layer3(x2)  # 8x8
        f3_act = [self.relu(f) for f in f3]

        x4 = self.avgpool(self.relu(x3))
        x4 = x4.view(x4.size(0), -1)
        f4 = x4
        x5 = self.fc(x4)

        if is_feat:
            return [self.relu(f0)] + f1_act + f2_act + f3_act + [f4], x
        else:
            return x5, [x0, x1, x2, x3, x4][layer] 
