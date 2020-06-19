from loadPretrainedCNN import VGG16_NoSoftmax_RGB, fetchImage
from Preprossering.PreprosseringPipeline import preprossingPipeline
import numpy as np
import torch
import os
import gc





def window_vector_loop(windowVec, featureVec):
    if windowVec is 0:
        windowVec = featureVec
        print(i)
    else:
        windowVec = torch.cat((windowVec, featureVec), 0)
        print(i)
    return windowVec.detach()

def feature_vector_loop_inner(tensor_window,model):
    for i in range(14):
        if i==0:
            tempFeatureVec = model(tensor_window[i].unsqueeze(0).float())
            #tempFeatureVec=tensor_window[i].reshape(1,-1)
            featureVec = tempFeatureVec
        else:
            tempFeatureVec = model(tensor_window[i].unsqueeze(0).float())
            #tempFeatureVec = tensor_window[i].reshape(1,-1)
            featureVec = torch.cat((featureVec, tempFeatureVec), 1)
    return featureVec.detach()



C=preprossingPipeline(BC_datapath='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG',mac=True)
fileNames=C.edfDict.keys()
path_new='/Volumes/B/spectograms_rgb'
i=0
file='sbs2data_2018_09_03_11_49_34_148.edf'
model=VGG16_NoSoftmax_RGB()
model.eval()


        #try:
windowVec = 0
tensor, _, _, _ = C.make_label_cnn(make_from_filenames=[file], path=path_new)
tensor.requires_grad_(requires_grad=False)
print(file)
for i in range(int(len(tensor) / 14)):
    feature_vector = feature_vector_loop_inner(tensor[0 + i * 14: 14 + i * 14],model=model)
    windowVec = window_vector_loop(windowVec, feature_vector)
    windowVec=windowVec.detach()
    del feature_vector
    gc.collect()
    print(0 + i * 14, 14 + i * 14)
    print(windowVec.shape)
i += 1
print(i)

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

