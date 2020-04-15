from loadPretrainedCNN import VGG16_NoSoftmax_OneChannel,VGG16_NoSoftmax_RGB, fetchImage
#Responsible: Mads Christian
import torch
from Preprossering.PreprosseringPipeline import preprossingPipeline


def getFeatureVecWholeFile(filePath):
    spectrogramDict = C.get_spectrogram(filePath)
    windowVec = []
    for windowName in spectrogramDict:
        windowValues = spectrogramDict[windowName]
        featureVec = []
        for channelSpectrogram in windowValues.values():
            #channelValue = windowValues[channelName]
            tempFeatureVec = model(channelSpectrogram.unsqueeze(0).unsqueeze(0).float())
            if len(featureVec)==0:
                featureVec = tempFeatureVec
            else:
                featureVec = torch.cat((featureVec, tempFeatureVec), 1)
        if len(windowVec) ==0:
            windowVec = featureVec
        else:
            windowVec = torch.cat((windowVec,featureVec),0)
                #windowFeatureVec.append(tempFeatureVec)
    return windowVec

def getFeatureVec(windowValues):
    featureVec = []
    for channelSpectrogram in windowValues.values():
        #channelValue = windowValues[channelName]
        tempFeatureVec = model(channelSpectrogram.unsqueeze(0).unsqueeze(0).float())
        if len(featureVec)==0:
            featureVec = tempFeatureVec
        else:
            featureVec = torch.cat((featureVec, tempFeatureVec), 1)
    return featureVec

if __name__ == "__main__":
    model = VGG16_NoSoftmax_OneChannel()
    model.eval()


    C=preprossingPipeline(r"C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG")
    fileNames = C.edfDict

    randomFile = fileNames.__iter__().__next__()
    spec = C.get_spectrogram(randomFile)
    randomWindow = spec.__iter__().__next__()

    featureVec = getFeatureVec(spec[randomWindow])


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