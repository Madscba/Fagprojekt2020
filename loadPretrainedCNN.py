import torch
import torch.nn as nn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import preprocessing

def importPretrained():
    #Import pretrained VGG16 from pytorchs database.
    return models.vgg16(pretrained=True)

class VGG16_NoSoftMax(nn.Module):
    def __init__(self):
        super(VGG16_NoSoftMax, self).__init__()
        self.features = nn.Sequential(*list(vgg16.features.children())[:-1] )
    def forward(self, x):
        x = self.features(x)
        return x

vgg16 = importPretrained()

for i, data in enumerate(train_data_loader):
    pass

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
print(model)
b = b.transpose((2, 0, 1))
img2 = torch.from_numpy(b)
#vgg16(img2) #
##input should be: (batch size, number of channels, height, width)

model.eval()
vgg16.eval()

output = model(img2.unsqueeze_(0).float())
out2 = vgg16(img2.float())
#latent = model.forward(img2)

#
# for i, data in enumerate(train_data_loader):
#     pass