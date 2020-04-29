from Villads.Load_PCA_TSNE import pca
from Preprossering.PreprosseringPipeline import preprossingPipeline
from Villads.PCA_TSNE_classes import scale_data
from Visualization.PCA_TSNE_plot import plot_pca
import numpy as np
import pickle
import os

path='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/feature_vectors/'
C=preprossingPipeline(BC_datapath='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG',mac=True)
path_pca='/Users/villadsstokbro/Dokumenter/DTU/GitHub/Fagprojekt2020/Fagprojekt2020/Villads/PCA_feature_vectors_1.sav'
#path_pca='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/PCA:TSNE/PCA.sav'
pca = pickle.load(open(path_pca, 'rb'))

feature_vectors_1,labels_1,filenames= C.make_label(max_files=5,quality=[1],is_usable=None,make_spectograms=False,path = path,seed=10)
feature_vectors_9_10,labels_9_10, filenames1= C.make_label(max_files=5,quality=[8,9,10],is_usable=None,make_spectograms=True,path = path,seed=10)


feature_vectors=np.vstack((feature_vectors_1,feature_vectors_9_10))
filenames=filenames+filenames1
labels=labels_1+labels_9_10
scaled_feature_vectors=scale_data(feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
plot_pca(pca_vectors,labels,np.unique(labels))


feature_vectors,labels, _= C.make_label(max_files=100,make_from_names=filenames,path = path,make_spectograms=True)
scaled_feature_vectors=scale_data(feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
plot_pca(pca_vectors,labels,np.unique(labels))



feature_vectors_1,labels_1,filenames= C.make_label(max_files=5,is_usable='Yes',path = path, make_spectograms=True)
feature_vectors_9_10,labels_9_10, filenames1= C.make_label(max_files=5,is_usable='No',make_spectograms=True,path = path)
feature_vectors=np.vstack((feature_vectors_1,feature_vectors_9_10))
filenames=filenames+filenames1
labels=labels_1+labels_9_10
scaled_feature_vectors=scale_data(feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
plot_pca(pca_vectors,labels,np.unique(labels))

feature_vectors,labels, _= C.make_label(max_files=100,make_from_names=filenames,path = path,make_spectograms=True)
scaled_feature_vectors=scale_data(feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
plot_pca(pca_vectors,labels,np.unique(labels))
