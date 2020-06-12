import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import models

class VGG16_modified(nn.Module):
    def __init__(self):
        super(VGG16_modified, self).__init__()
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

if __name__ == "__main__":
    PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt'
    trained_model_RGB = VGG16_modified()
    trained_model_RGB.load_state_dict(torch.load(PATH))
    pass

