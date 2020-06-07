from flashtorch.utils import apply_transforms, load_image
import torchvision.models as models
from flashtorch.saliency import Backprop
from CNN.loadPretrainedCNN2 import VGG16_NoSoftmax_OneChannel
import matplotlib.pyplot as plt
import io
from PIL import Image

model = VGG16_NoSoftmax_OneChannel()
model.eval()
C=preprossingPipeline(BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG",mac=True)
fileNames=C.edfDict.keys()
spec = C.get_spectrogram(list(fileNames)[0])
a=np.array(spec['window_0_15360']['TP10'])
plt.imsave('test_spec.jpg',a)

image = load_image('test_spec.jpg')
img = apply_transforms(image)

def visualize_helper(model_module, tensor=img, k=84):
    model = model_module(pretrained=True).float()
    backprop = Backprop(model)
    backprop.visualize(tensor, k, guided=True)

visualize_helper(models.vgg16)


