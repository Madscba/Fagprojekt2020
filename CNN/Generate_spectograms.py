from Preprossering.PreprosseringPipeline import preprossingPipeline, getFeatureVec
import numpy as np

C = preprossingPipeline(
    BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")
fileNames=C.edfDict.keys()
wdir="Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt"
for file in fileNames:
    spec = C.get_spectrogram(file)
    filename=r'/spec/'+file
    np.save(wdir+r'/spec/',spec)








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

