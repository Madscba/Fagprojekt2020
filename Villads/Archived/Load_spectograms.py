from Preprossering.PreprosseringPipeline import preprossingPipeline
import os
import numpy as np

model = VGG16_NoSoftmax_OneChannel()
model.eval()
C=preprossingPipeline(BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG",mac=True)
fileNames=C.edfDict.keys()
wdir="/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt"


for file in fileNames:
    if os.path.exists(wdir+r'/feature_vectors/'+file+".npy")==True:
        pass
    else:
        spec = C.get_spectrogram(file)
        break






