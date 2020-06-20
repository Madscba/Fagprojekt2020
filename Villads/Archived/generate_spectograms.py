from Preprossering.PreprosseringPipeline import preprossingPipeline, getFeatureVec
import numpy as np
import os

#Responsible: Mads Christian
import torch

C=preprossingPipeline(BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG",mac=True)
fileNames = C.edfDict.keys()
wdir=os.getcwd()
file_dict=dict()
for file in fileNames:
    i = 0
    file_dict[file]=[]
    if os.path.exists(wdir+r'/spectograms/'+file)==True:
        pass
    else:
        spec = C.get_spectrogram(file)
        for window_value in spec.keys():
            if window_value == 'annotations':
                break
            spectogram=spec[window_value].detach().numpy().reshape((1, 1, -1))
            if spectogram.shape[2] !=  8772:
                print(i)
                pass
            else:
                if i == 0:
                    spectograms = spectogram
                else:
                    spectograms =np.hstack((spectograms,spectogram))
                i+=1
                print(i)
        filename=r'/spectograms/'+file
        np.save(wdir+filename,spectograms)

