import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import models

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features.children()))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=2, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

def freeze_parameters(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def grad_parameters(model,list):
    for name,param in model.named_parameters():
        if param.requires_grad == False:
            for i in range(len(list)):
                if name == list[i]:
                    param.requires_grad = True

def list_of_features(model):
    list = []
    for name,param in model.named_parameters():
        list.append(name)
    return list

def check_grad(model):
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

model = VGG16()
freeze_parameters(model,feature_extracting=True)
check_grad(model)
list2 = np.array(list_of_features(model))

