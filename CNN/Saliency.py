from flashtorch.utils import apply_transforms, load_image
import torchvision.models as models
from flashtorch.saliency import Backprop
from CNN.loadPretrainedCNN2 import VGG16_NoSoftmax_OneChannel

model = VGG16_NoSoftmax_OneChannel()
model.eval()
C=preprossingPipeline(BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG",mac=True)
fileNames=C.edfDict.keys()
spec = C.get_spectrogram(list(fileNames)[0])



image = load_image('./testSprograms/test3_or_above31_0.jpg')
model = model_module(pretrained=True)
backprop = Backprop(model)
img = apply_transforms(image)


def visualize_helper(model_module, tensor=img, k=84):
    model = model_module(pretrained=True)
    backprop = Backprop(model)
    backprop.visualize(tensor, k, guided=True)




