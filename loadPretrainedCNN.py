import torch
import torch.nn as nn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import preprocessing

def importPretrainedVGG16():
    #Import pretrained VGG16 from pytorchs database.
    return models.vgg16(pretrained=True)

class VGG16_NoSoftMax(nn.Module):
    def __init__(self):
        super(VGG16_NoSoftMax, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features.children())[:])
        self.classifier = nn.Sequential()
    def forward(self, x):
        x = self.features(x)
        return x
class VGG16_NoSoftMax1(nn.Module):
    def __init__(self):
        super(VGG16_NoSoftMax1, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features.children())[:])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
    # def __init__(self):
    #     super(VGG16_NoSoftMax1,self).__init__()
    #     features = list(models.vgg16(pretrained=True).features)
    #     self.features = nn.ModuleList(features).eval()
    # def forward(self,x):
    #     featureVector = self.features[-1](x)
    #     return featureVector

#torch.cat((a,b))
#reshape b = a.reshape(1,8)
if __name__ == "__main__":
    vgg = importPretrainedVGG16()

    # for i, data in enumerate(train_data_loader):
    #     pass

    img = Image.open(r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\Fagprojekt2020\testSpektrograms\test3_or_above1_0.JPG')
    plt.imshow(img)
    img = img.resize((224,224))
    plt.imshow(img)
    a = np.asarray(img)
    b = np.empty_like(a,dtype=float)
    print(type(b))
    min_max_scaler = preprocessing.MinMaxScaler()

    for i in range(a.shape[2]):
        a_stand = min_max_scaler.fit_transform(a[:,:,i])
        b[:,:,i] = a_stand

    model = VGG16_NoSoftMax()
    model1 = VGG16_NoSoftMax1()
    print(model)
    b = b.transpose((2, 0, 1))
    img2 = torch.from_numpy(b)
    #vgg16(img2) #
    ##input should be: (batch size, number of channels, height, width)

    model.eval()
    vgg.eval()
    print(model)
    print(model1)
    print(vgg)

    out0 = model(img2.unsqueeze_(0).float())
    out1 = model1(img2.float())
    out2 = vgg(img2.float())
    #latent = model.forward(img2)

    #
    # for i, data in enumerate(train_data_loader):
    pass