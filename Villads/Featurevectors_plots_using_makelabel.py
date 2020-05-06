from Preprossering.PreprosseringPipeline import preprossingPipeline, make_pca
from Villads.PCA_TSNE_classes import scale_data
from Visualization.PCA_TSNE_plot import plot_pca
import numpy as np
import pickle


path='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/feature_vectors/'
C=preprossingPipeline(BC_datapath='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG',mac=True)


feature_vectors_1, labels_1, filenames, window_idx= C.make_label(max_files=5,quality=[1],path = path,seed=10)
feature_vectors_9_10, labels_9_10, filenames1, window_idx1= C.make_label(max_files=5,quality=[9,10],path = path,seed=10)


feature_vectors=np.vstack((feature_vectors_1,feature_vectors_9_10))
filenames=filenames+filenames1
labels=labels_1+labels_9_10
pca_vectors=make_pca(feature_vectors)
plot_pca(pca_vectors,labels,np.unique(labels),pca_components=[0,1])


feature_vectors,labels, _,_= C.make_label(max_files=100,make_from_names=filenames,path = path, make_pca=False)
pca_vectors=make_pca(feature_vectors)
plot_pca(pca_vectors,labels,np.unique(labels),pca_components=[0,1])



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
