from Preprossering.PreprosseringPipeline import preprossingPipeline
import os
import numpy as np


C = preprossingPipeline(
    BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG", mac=True)
fileNames = C.edfDict.keys()
i = 0
#Change wdir to the directory of the folder 'feature_vectors'
wdir = os.getcwd()
for filename in fileNames:
    if not os.path.exists(wdir + r'/feature_vectors/' + filename+'.npy'):
        pass
    else:
        if i == 0:
            featureVec = np.load(wdir + r'/feature_vectors/' + filename + '.npy')
            featureVectors = featureVec
        else:
            featureVec = np.load(wdir + r'/feature_vectors/' + filename + '.npy')
            featureVectors = np.vstack((featureVectors, featureVec))
        i+=1



