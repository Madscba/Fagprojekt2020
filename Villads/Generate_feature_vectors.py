from loadPretrainedCNN import VGG16_NoSoftmax_OneChannel,VGG16_NoSoftmax_RGB, fetchImage
from Preprossering.PreprosseringPipeline import preprossingPipeline, getFeatureVec
import numpy as np
import os



model = VGG16_NoSoftmax_OneChannel()
model.eval()
C=preprossingPipeline(BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG",mac=True)
wdir="/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt"
for file in fileNames:
    if os.path.exists(wdir+r'/feature_vectors/'+file+".npy")==True:
        pass
    else:
        spec = C.get_spectrogram(file)
        i = 0
        for window_value in spec.keys():
            if window_value=='annotations':
                break
            if i == 0:
                featureVec = getFeatureVec(spec[window_value], model).detach().numpy()
                featureVectors = featureVec
            else:
                featureVec = getFeatureVec(spec[window_value],model).detach().numpy()
                featureVectors=np.vstack((featureVectors,featureVec))
            i+=1
            print(i)
        filename=r'/feature_vectors/'+file
        np.save(wdir+filename,featureVectors)








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

