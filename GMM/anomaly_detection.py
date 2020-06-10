""" @author: Johannes"""
import numpy as np
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from Villads.PCA_TSNE_classes import scale_data
import pandas as pd
from GMM.gmm import gaussian_mixture, plot_cluster, cv_gmm, potential_outliers
from sklearn.preprocessing import StandardScaler
from Preprossering.PreprosseringPipeline import preprossingPipeline
from Visualization.PCA_TSNE_plot import plot_pca
from sklearn.decomposition import PCA

path=r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\feature_vectors'
C=preprossingPipeline(BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")

feature_vectors_1,labels_1,filenames,idx= C.make_label(max_files=10,quality=[1],is_usable=None,path = path, max_windows=20)
feature_vectors_9_10,labels_9_10,filenames_1,idx_1= C.make_label(max_files=10,quality=[9],is_usable=None,path = path, max_windows=20)
feature_vectors_5,labels_5,filenames_2,idx_2= C.make_label(max_files=10,quality=[5],is_usable=None,path = path, max_windows=20)

fv=np.vstack((feature_vectors_1,feature_vectors_5))
feature_vectors=np.vstack((fv,feature_vectors_9_10))
filenames=filenames+filenames_1+filenames_2
labels=labels_1+labels_9_10+labels_5
scaled_feature_vector=scale_data(feature_vectors)

pca = PCA()
pca.fit(scaled_feature_vector)
pca_fv=pca.transform(scaled_feature_vector)

plot_pca(pca_fv,labels,np.unique(labels),model='PCA feature vectors')

cov_type = 'full'

cds, cls, covs, gmm = gaussian_mixture(pca_fv,4,covar_type=cov_type,reps=5,init_procedure='kmeans')

plot_cluster(X=pca_fv,cls=cls,cds=cds,y=labels,covs=covs,idx=[0,1])

cv_gmm(pca_fv,K_range=range(1,11),n_splits=5,covar_type=cov_type,reps=5,init_procedure='kmeans')

outliers, probs = potential_outliers(cluster_model=gmm, data=pca_fv, threshold=0.01)