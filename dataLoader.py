from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Normalize
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
root_dir = r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\Fagprojekt2020\testSpektrograms'
#mean of [red channel, green channel, blue channel]
means = [0.485,0.456,0.405]
#std of [red channel, green channel, blue channel]
std = [0.229,0.224,0.225]
transforms = transforms.Compose([Resize((224,224)),ToTensor(),Normalize(means,std)])
train_data = ImageFolder(root = os.path.join(root_dir,'train_dir'),transform=transforms)
val_data = ImageFolder(root = os.path.join(root_dir,'val_dir'),transform=transforms)

def getDataSetInfo():
    print("picture: ",train_data)
    print("classes: ",train_data.classes)
    print("classes: ", train_data.class_to_idx)

#path to first picture in the training data:
train_data.imgs[0]
#Content of first training image
train_data[0][0]
#Label:
train_data[0][1]

def convertBackToNumpyAndPlot(tensor):
    nump = tensor.numpy().transpose(1,2,0)
    nump = std*nump + means
    nump = np.clip(nump,0,1)
    plt.imshow(nump)
    plt.title(idx_to_class[(tensor[0][1])])

batchSize =3
dataloader = DataLoader(train_data,batch_size=batchSize, shuffle=True)
train_data_loader = iter(dataloader)
x,y = next(train_loader)

#
# for i, data in enumerate(train_data_loader):
#     pass