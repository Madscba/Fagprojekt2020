from flashtorch.saliency import Backprop
from Preprossering.PreprosseringPipeline import preprossingPipeline
import matplotlib.pyplot as plt
import torch
from CNN.modifyCNN import VGG16
from flashtorch.saliency import Backprop
from torchvision import models
from skimage.transform import resize
from sklearn import preprocessing
from flashtorch.utils import format_for_plotting, apply_transforms, denormalize
from torch.autograd import Variable
from flashtorch.activmax import GradientAscent
import numpy as np

test_set = torch.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\test_set_b.pt')
test_labels = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\test_labels.npy')
test_windows = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\test_windows.npy')
cpi = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Kode\Gruppen\Fagprojekt2020\CNN_HPC\cpi.npy')
marks = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Kode\Gruppen\Fagprojekt2020\GMM\marked_windows.npy')
windows = test_windows[cpi]
labels = test_labels[cpi]
data_set = test_set[cpi,:,:,:]
images = data_set[marks,:,:,:]
w = windows[marks]
y = labels[marks]
model = VGG16()
model.load_state_dict(torch.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\model1_l5_b.pt'))
model.eval()

def get_image(image,num):
    img1 = image[num, :, :, :].unsqueeze(0).requires_grad_(requires_grad=True)
    return img1
img1 = get_image(images,0)
img2 = get_image(images,1)
img3 = get_image(images,2)
img4 = get_image(images,3)
img5 = get_image(images,4)
img6 = get_image(images,5)
img7 = get_image(images,6)
img8 = get_image(images,7)

plt.imshow(format_for_plotting(denormalize(img2)))
plt.show()

def visualize_saliency(model, tensor, k=1, guide=True):
    backprop = Backprop(model)
    backprop.visualize(tensor, k, guided=guide)
    plt.show()
def get_image(image,num):
    img1 = image[num, :, :, :].unsqueeze(0).requires_grad_(requires_grad=True)
    return img1
start = 0
def multiple_saliency_maps(model, tensors, start, num, k=0, guide = True):
    backprop = Backprop(model)
    for i in range(start,start+num):
        cur_image = tensors[i,:,:,:].unsqueeze(0).requires_grad_(requires_grad=True)
        backprop.visualize(cur_image, k, guided=guide)
        plt.show()
    print("Range: ", start, " - ", start+num)
    start += num
    return start

s = multiple_saliency_maps(model=model,tensors=data_set,start=start,num=10)
s = multiple_saliency_maps(model=model,tensors=data_set,start=40,num=10)