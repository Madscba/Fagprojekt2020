import torch
import torch.nn as nn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image




def importPretrained(modeltype = ''):
    #print("Dimension of vgg16. ",InDim: 224x224x3", 'outDim: 1000')
    if modeltype == 'RES':
        return models.resnet18(pretrained=True)
    else:
        return models.vgg16(pretrained=True)
vgg16 = importPretrained()

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(*list(vgg16.features.children())[:-1] )

    def forward(self, x):
        x = self.features(x)
        return x

#img=mpimg.imread(r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\Fagprojekt2020\testSpektrograms\test3_or_above1_0.JPG')
#plt.axis('off')
#imgplot = plt.imshow(img)

img = Image.open(r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\Fagprojekt2020\testSpektrograms\test3_or_above1_0.JPG')
plt.imshow(img)
img = img.resize((224,224))
plt.imshow(img)
a = np.asarray(img)

model = VGG16()
a = a.reshape((1, a.shape[2], a.shape[0], a.shape[1]))
img2 = torch.from_numpy(a)


vgg16(img2)
model(img2)
#latent = model.forward(img2)
a =2


pass

# for i, data in enumerate(train_data_loader):
#     pass