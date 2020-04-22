import numpy as np
import pickle
import os
from Preprossering.PreprosseringPipeline import preprossingPipeline


pca = pickle.load(open(r'C:\Users\Andreas\Desktop\KID\Fagproject\PCA_TSNE/PCA.sav', 'rb'))
TSNE = pickle.load(open(r'C:\Users\Andreas\Desktop\KID\Fagproject\PCA_TSNE/TSNE.sav', 'rb'))
tsne_fv=np.load(r'C:\Users\Andreas\Desktop\KID\Fagproject\PCA_TSNE\tsne_features.npy')
pca_fv=np.load(r'C:\Users\Andreas\Desktop\KID\Fagproject\PCA_TSNE\pca_features.npy')


#C = preprossingPipeline(
#    BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG", mac=True)
C= preprossingPipeline(r"C:\Users\Andreas\Desktop\KID\Fagproject\Data\BC")
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
