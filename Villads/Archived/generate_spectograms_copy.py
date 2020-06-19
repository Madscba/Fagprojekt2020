from Preprossering.PreprosseringPipeline import preprossingPipeline, getFeatureVec
import numpy as np
import os

#Responsible: Mads Christian
import torch

C=preprossingPipeline(BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG",mac=True)
fileNames = C.edfDict.keys()
wdir=os.getcwd()
for file in fileNames:
    i = 0
    if os.path.exists(wdir+r'/spectograms_all_ch/'+file)==True:
        pass
    else:
        spec = C.get_spectrogram(file)
        for window_value in spec.keys():
            j=0
            if window_value=='annotations':
                break
            for channel in spec[window_value].keys():
                if j == 0:
                    channel_spec = spec[window_value][channel].detach().numpy().reshape((1,1,-1))
                    channel_specs = channel_spec
                else:
                    channel_spec = spec[window_value][channel].detach().numpy().reshape((1, 1, -1))
                    channel_specs =np.dstack((channel_specs ,channel_spec))
                j+=1
            print(i)
            if channel_specs.shape[2] != 8772*14:
                pass
            else:
                if i == 0:
                    spectogram = channel_specs
                    spectograms = spectogram
                else:
                    spectogram = channel_specs
                    spectograms =np.hstack((spectograms,spectogram))
            i+=1
        filename=r'/spectogram_all_ch/'+file
        np.save(wdir+filename,spectograms)