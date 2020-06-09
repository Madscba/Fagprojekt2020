<<<<<<< HEAD
from CNN.trainCNN2 import model
=======
#from CNN.trainCNN import model
>>>>>>> 5a2ccb471fee206eea1506033e0dc96e03ae9216
from flashtorch.saliency import Backprop
from Preprossering.PreprosseringPipeline import preprossingPipeline
import matplotlib.pyplot as plt
import torch
from torchvision import models
from skimage.transform import resize
from sklearn import preprocessing
from flashtorch.utils import format_for_plotting, apply_transforms, denormalize
from torch.autograd import Variable
from flashtorch.activmax import GradientAscent
import numpy as np

#C = preprossingPipeline(BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")
C=preprossingPipeline(BC_datapath='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG',mac=True)
path = r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\spectograms_rgb'
path1=r'/Volumes/B/spectograms_rgb'
N=2
<<<<<<< HEAD
windows, labels, filenames, window_idx_full = C.make_label_cnn(make_from_filenames=None, quality=None, is_usable='Yes', max_files=N, max_windows = 10,
                   path=path, seed=0, ch_to_include=range(1))
#model = models.vgg16(pretrained=True)
#x_train = np.vstack((spectrogram_is_usable[:10,:],spectrogram_not_usable[:10,:]))
#y_train = np.hstack((labels__is_usable_spec[:10],labels_not_usable_spec[:10]))
#x1 = x_train[1,:]
l1 = preprocessing.LabelEncoder()
t1 = l1.fit_transform(labels)
spec = torch.load(path+'\sbs2data_2018_08_30_19_37_22_286 Part2.edf.pt')
=======
windows, labels, filenames, window_idx_full = C.make_label_cnn(make_from_filenames=None, quality=None, is_usable=None, max_files=N, max_windows = 10,
                   path=path1, seed=0, ch_to_include=range(1))
model = models.vgg16(pretrained=True)
#x_train = np.vstack((spectrogram_is_usable[:10,:],spectrogram_not_usable[:10,:]))
#y_train = np.hstack((labels__is_usable_spec[:10],labels_not_usable_spec[:10]))
#x1 = x_train[1,:]

#spec = torch.load(path+'\sbs2data_2018_08_30_19_37_22_286 Part2.edf.pt')
>>>>>>> 5a2ccb471fee206eea1506033e0dc96e03ae9216
#spec.unsqueeze(0)
#batch_image = np.zeros((1,224,224))
#image_resized = resize(x1, (224, 224), anti_aliasing=True)
#plt.imshow(image_resized)
#plt.show()
#X_image = windows
#plt.imshow(format_for_plotting((spec[:,0,:,:,:].float())))
<<<<<<< HEAD
#backprop = Backprop(model)
#output = model(windows.float())
#gradients = backprop.calculate_gradients(windows[0,:,:,:].unsqueeze(0).float(),t1[0])
#max_gradients = backprop.calculate_gradients(windows[0,:,:,:].unsqueeze(0).float(), t1[0], take_max=True)
#backprop.visualize(windows[0,:,:,:].unsqueeze(0).float(),t1[0], guided=True)
#plt.show()
=======
backprop = Backprop(model)
output = model(windows.float())
gradients = backprop.calculate_gradients(windows[0,:,:,:].unsqueeze(0).float())
max_gradients = backprop.calculate_gradients(windows[0,:,:,:].unsqueeze(0).float(), t1[0], take_max=True)
backprop.visualize(windows[0,:,:,:].unsqueeze(0).float(),1,guided=True)
plt.show()
>>>>>>> 5a2ccb471fee206eea1506033e0dc96e03ae9216
#gradient_ascent.visualize(X_image,gradients, max_gradients)

backprop = Backprop(model)
output = model(windows[0,:,:,:,:].float())
gradients = backprop.calculate_gradients(windows[0,:,:,:,:].float(),t1[0])
max_gradients = backprop.calculate_gradients(windows[0,:,:,:,:].float(), t1[0], take_max=True)
backprop.visualize(windows[0,:,:,:,:].float(),t1[0], guided=True)
plt.show()

