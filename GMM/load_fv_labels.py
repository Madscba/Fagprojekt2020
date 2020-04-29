""" @author: Johannes"""
import numpy as np
import pickle
from Preprossering.PreprosseringPipeline import preprossingPipeline

#pca = pickle.load(open('/Users/johan/iCloudDrive/DTU/KID/4. semester/Fagprojekt/Kode/Johannes/PCA_TSNE/PCA.sav', 'rb'))
#TSNE = pickle.load(open('/Users/johan/iCloudDrive/DTU/KID/4. semester/Fagprojekt/Kode/Johannes/PCA_TSNE/TSNE.sav', 'rb'))
C = preprossingPipeline(
    BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")
path=r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\feature_vectors'

feature_vectors_1,labels_1,filenames= C.make_label(make_spectograms=False,quality=[1],is_usable=None,max_files=5,path = path)
feature_vectors_9_10,labels_9_10= C.make_label(make_spectograms=False,quality=[9,10],is_usable=None,max_files=5,path = path)
feature_vectors=np.vstack(feature_vectors_1,feature_vectors_9_10)