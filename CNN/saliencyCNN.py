from CNN.modifyCNN import model
from flashtorch.saliency import Backprop
from Preprossering.PreprosseringPipeline import preprossingPipeline
import matplotlib.pyplot as plt
import torch
from skimage.transform import resize
from sklearn import preprocessing
from flashtorch.utils import format_for_plotting, apply_transforms, denormalize
from torch.autograd import Variable
from flashtorch.activmax import GradientAscent
import numpy as np

C = preprossingPipeline(BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")

path_spec = r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Spektrograms'
N=2
spectrogram_is_usable,labels__is_usable_spec,_,_= C.make_label(make_from_filenames=False,quality=None,is_usable="Yes",max_files=N,max_windows=20, path = path_spec) #18 files = 2074
spectrogram_not_usable,labels_not_usable_spec,_,_= C.make_label(make_from_filenames=False,quality=None,is_usable='No',max_files=N ,max_windows=20, path = path_spec) #18 files = 1926

x_train = np.vstack((spectrogram_is_usable[:10,:],spectrogram_not_usable[:10,:]))
y_train = np.hstack((labels__is_usable_spec[:10],labels_not_usable_spec[:10]))
x1 = x_train[1,:]
l1 = preprocessing.LabelEncoder()
t1 = l1.fit_transform(y_train)
batch_image = np.zeros((1,224,224))
image_resized = resize(x1, (224, 224), anti_aliasing=True)
plt.imshow(image_resized)
plt.show()
plt.imshow(format_for_plotting(denormalize(X_image.unsqueeze(1).float())))
batch_image[0,:,:] = image_resized
X_image = Variable(torch.from_numpy(batch_image))
backprop = Backprop(model)
output = model(X_image.unsqueeze(1).float())
gradients = backprop.calculate_gradients(X_image.unsqueeze(1).float(),t1[1])
max_gradients = backprop.calculate_gradients(X_image.unsqueeze(1).float(), t1[1], take_max=True)
a = backprop.visualize(X_image.unsqueeze(1).float(),t1[1])
#gradient_ascent.visualize(X_image,gradients, max_gradients)



