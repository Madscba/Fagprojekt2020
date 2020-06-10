from flashtorch.utils import apply_transforms, load_image
import torchvision.models as models
from flashtorch.saliency import Backprop
from CNN.loadPretrainedCNN2 import VGG16_NoSoftmax_OneChannel
import matplotlib.pyplot as plt
import io
from PIL import Image


C=preprossingPipeline(BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG",mac=True)
path1=r'/Volumes/B/spectograms_rgb'
N=2
windows, labels, filenames, window_idx_full = C.make_label_cnn(make_from_filenames=None, quality=None, is_usable=None, max_files=N, max_windows = 10,
                   path=path1, seed=0, ch_to_include=range(1))
img=windows[0].unsqueeze(0)
img=load_image('Ricardo_rip.jpg')
img= apply_transforms(img,size=224)
img.detach().requires_grad_(requires_grad=True)

#window = torch.load('/Volumes/B/spectograms_rgb/sbs2data_2018_08_30_19_39_35_288 part 2.edf.pt')
#img=window.detach().requires_grad_(requires_grad=True)[0,0,:,:].unsqueeze(0)

def visualize_helper(model_module, tensor=img, k=854):
    model = model_module(pretrained=True).float()
    backprop = Backprop(model)
    backprop.visualize(tensor, k, guided=True)

model=models.vgg16(pretrained=True)
model.eval()
torch.argmax(model(img))
visualize_helper(models.vgg16,tensor=img)
plt.show()

