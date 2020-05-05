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

feature_vectors_1, labels_1, filenames, window_idx= C.make_label(max_files=5,quality=[1],is_usable=None,path = path,seed=10)
feature_vectors_9_10, labels_9_10, filenames1, window_idx1= C.make_label(max_files=5,quality=[8,9,10],is_usable=None,path = path,seed=10)



feature_vectors=np.vstack((feature_vectors_1,feature_vectors_9_10))
filenames=filenames+filenames1
labels=labels_1+labels_9_10
scaled_feature_vectors=scale_data(feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
plot_pca(pca_vectors,labels,np.unique(labels),plot_extremes=True,pca_components=[0,2])


feature_vectors,labels, _,_= C.make_label(max_files=100,make_from_names=filenames,path = path)
scaled_feature_vectors=scale_data(feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
plot_pca(pca_vectors,labels,np.unique(labels),pca_components=[0,2],plot_extremes=True)



feature_vectors_1,labels_1,filenames,_= C.make_label(max_files=5,is_usable='Yes',path = path,seed=1)
feature_vectors_9_10,labels_9_10,filenames1,_= C.make_label(max_files=5,is_usable='No',path = path,seed=1)
feature_vectors=np.vstack((feature_vectors_1,feature_vectors_9_10))
filenames=filenames+filenames1
labels=labels_1+labels_9_10
scaled_feature_vectors=scale_data(feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
plot_pca(pca_vectors,labels,np.unique(labels),plot_extremes=True,pca_components=[0,3])

feature_vectors,labels, _,_= C.make_label(max_files=100,make_from_names=filenames,path = path)
scaled_feature_vectors=scale_data(feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
plot_pca(pca_vectors,labels,np.unique(labels),plot_extremes=True,pca_components=[0,3])
