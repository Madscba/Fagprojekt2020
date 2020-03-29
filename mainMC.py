from loadPretrainedCNN import VGG16_NoSoftMax
from loadPretrainedCNN import importPretrainedVGG16
#Generate a full VGG model with softmax
vgg16 = importPretrainedVGG16()
#Generate a pretrained VGG model
model = VGG16_NoSoftMax()

print(vgg16)
print(model)
vgg16.eval()
model.eval()


pass