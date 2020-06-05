import torch
import torch.nn as nn
import torchvision
from torchvision import models

class VGG16_OneChannel(nn.Module):
    def __init__(self):
        super(VGG16_OneChannel, self).__init__()
        vgg_firstlayer = models.vgg16(pretrained=True).features[0]  # load just the first conv layer
        vgg16 = models.vgg16(pretrained=True)
        w1 = vgg_firstlayer.state_dict()['weight'][:, 0, :, :]
        w2 = vgg_firstlayer.state_dict()['weight'][:, 1, :, :]
        w3 = vgg_firstlayer.state_dict()['weight'][:, 2, :, :]
        w4 = w1 + w2 + w3  # add the three weigths of the channels
        w4 = w4.unsqueeze(1)
        first_conv = nn.Conv2d(1, 64, 3, padding=(1, 1))  # create a new conv layer
        first_conv.weigth = torch.nn.Parameter(w4, requires_grad=True)  # initialize  the conv layer's weigths with w4
        first_conv.bias = torch.nn.Parameter(vgg_firstlayer.state_dict()['bias'],
                                             requires_grad=True)  # initialize  the conv layer's weigths with vgg's first conv bias

        self.first_convlayer = first_conv  # the first layer is 1 channel (Grayscale) conv  layer

        self.features = nn.Sequential(*list(vgg16.features.children())[1:])
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
        x = self.first_convlayer(x)
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

model = VGG16_OneChannel()
freeze_parameters(model,feature_extracting=True)
check_grad(model)
list = list_of_features(model)
grad_parameters(model,list[23:len(list)])
check_grad(model)