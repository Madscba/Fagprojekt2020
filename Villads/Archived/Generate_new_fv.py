from CNN.loadTrainedCNN import trained_model_RGB
from Preprossering.PreprosseringPipeline import preprossingPipeline, getNewFeatureVec
import numpy as np
import os



model = trained_model_RGB
model.eval()
C=preprossingPipeline(BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")
fileNames=C.edfDict.keys()
wdir="/Users/johan/iCloudDrive/DTU/KID/4. semester/Fagprojekt"
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
                featureVec = getNewFeatureVec(spec[window_value], model).detach().numpy()
                featureVectors = featureVec
            else:
                featureVec = getNewFeatureVec(spec[window_value],model).detach().numpy()
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

