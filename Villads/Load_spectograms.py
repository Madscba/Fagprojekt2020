from Preprossering.PreprosseringPipeline import preprossingPipeline
import os
import numpy as np


C = preprossingPipeline(
    BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG", mac=True)
fileNames = C.edfDict.keys()
i = 0
#Change wdir to the directory of the folder 'feature_vectors'
wdir = '/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt'
for filename in fileNames:
    if not os.path.exists(wdir + r'/spectograms_all_ch/' + filename+'.npy'):
        pass
    else:
        print(i)
        if i == 0:

            spectogram = np.load(wdir + r'/spectograms_all_ch/' + filename + '.npy')
            spectograms = spectogram
        else:
            spectogram = np.load(wdir + r'/spectograms_all_ch/' + filename + '.npy')
            spectograms = np.hstack((spectograms, spectogram))
        if i==63
            break



