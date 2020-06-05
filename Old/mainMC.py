from loadPretrainedCNN import VGG16_NoSoftmax_OneChannel,VGG16_NoSoftmax_RGB, fetchImage
#Responsible: Mads Christian
import torch
from Preprossering.PreprosseringPipeline import preprossingPipeline, getFeatureVec
if __name__ == "__main__":
    model = VGG16_NoSoftmax_OneChannel()
    model.eval()


    C=preprossingPipeline(mac=True, BC_datapath=r"C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG")
    fileNames = C.edfDict

    randomFile = fileNames.__iter__().__next__()
    spec = C.get_spectrogram(randomFile)
    randomWindow = spec.__iter__().__next__()
    featureVec = getFeatureVec(spec[randomWindow],model)


    #Generate a pretrained VGG model
    # model = VGG16_NoSoftmax_OneChannel()
    # model_rgb = VGG16_NoSoftmax_RGB()
    # path = r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\Fagprojekt2020\testSpektrograms\test3_or_above1_0.JPG'
    # img = fetchImage(path) #Spectrograms should have dim: [3,224,224] for VGG_NoSoftmax_RGB
    # img2 = img[1,:,:] #Spectrograms should have dim: [224,224] for VGG_NoSoftmax_OneChannel
    #
    # model.eval()
    # model_rgb.eval()
    #
    # out1 = model(img2.unsqueeze(0).unsqueeze(0).float()) #input should be tensor with dim: [1,1,224,224] corresponding to [batchsize, channels, breadth, width], [output 1x4096]
    # out2 = model_rgb(img.unsqueeze(0).float()) #input should be tensor with dim: [1,3,224,224] corresponding to [batchsize, channels, breadth, width], output [1x4096]

    pass