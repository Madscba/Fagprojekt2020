import torch
import torch.nn as nn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import preprocessing

#Responsible: Mads Christian
class VGG16_NoSoftmax_OneChannel(nn.Module):
    def __init__(self):
        super(VGG16_NoSoftmax_OneChannel, self).__init__()
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
            nn.ReLU(True))

    def forward(self, x):
        x = self.first_convlayer(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

class VGG16_NoSoftmax_RGB(nn.Module):
    def __init__(self):
        super(VGG16_NoSoftmax_RGB, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = vgg16.classifier


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier[0:5](x)
        return x

def fetchImage(path =r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\Fagprojekt2020\testSpektrograms\test3_or_above1_0.JPG' ):
    img = Image.open(path)
    img = img.resize((224, 224))
    a = np.asarray(img)
    b = np.empty_like(a, dtype=float) # DIM: (224,224,3)
    min_max_scaler = preprocessing.MinMaxScaler() #Rescale values to interval 0-1
    for i in range(a.shape[2]):
        a_stand = min_max_scaler.fit_transform(a[:, :, i])
        b[:, :, i] = a_stand
    b = b.transpose((2, 0, 1))
    img2 = torch.from_numpy(b)
    return img2

if __name__ == "__main__":
    model_gray = VGG16_NoSoftmax_OneChannel()
    model_RGB = VGG16_NoSoftmax_RGB()

    model_gray.eval()
    model_RGB.eval()

    img2 = fetchImage()

    out4 = model_gray(img2[1,:,:].unsqueeze(0).unsqueeze(0).float())
    out2 = model_RGB(img2.unsqueeze_(0).float())


    pass