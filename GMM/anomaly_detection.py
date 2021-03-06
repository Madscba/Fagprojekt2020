""" @author: Johannes"""
import numpy as np
import os
import mne
import matplotlib.pyplot as plt
from Villads.PCA_TSNE_classes import scale_data
from sklearn.metrics import accuracy_score
import pandas as pd
from GMM.gmm import gaussian_mixture, plot_cluster, cv_gmm, potential_outliers
from sklearn.preprocessing import StandardScaler
from Preprossering.PreprosseringPipeline import preprossingPipeline
from sklearn.neighbors import KNeighborsClassifier
from Visualization.PCA_TSNE_plot import plot_pca
from sklearn.decomposition import PCA
from Visualization.plot_inteactive_functions import plot_pca_interactiv


path=r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\feature_vectors'
C=preprossingPipeline(BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")

feature_vectors_1,labels_1,filenames,idx= C.make_label(max_files=15,quality=[1],is_usable=None,path = path, max_windows=60)
feature_vectors_9_10,labels_9_10,filenames_1,idx_1= C.make_label(max_files=15,quality=[9],is_usable=None,path = path, max_windows=60)
feature_vectors_5,labels_5,filenames_2,idx_2= C.make_label(max_files=15,quality=[5],is_usable=None,path = path, max_windows=60)

fv1, l1, f1, i1 =  C.make_label(max_files=70,quality=None,is_usable='Yes',path = path, max_windows=2)
fv2, l2, f2, i2 =  C.make_label(max_files=70,quality=None,is_usable='No',path = path, max_windows=2)



fv=np.vstack((feature_vectors_1,feature_vectors_5))
feature_vectors=np.vstack((fv,feature_vectors_9_10))
fv_new = np.vstack((fv1,fv2))
filenames=filenames+filenames_1+filenames_2
labels=labels_1+labels_9_10+labels_5
index = idx+idx_1+idx_2
l = l1+l2
scaled_feature_vector=scale_data(feature_vectors)
scaled_new = scale_data(fv_new)

pca = PCA()
pca.fit(scaled_feature_vector)
pca_fv=pca.transform(scaled_feature_vector)
pca_fv1 = pca.transform(scaled_new)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(pca_fv, labels)

J = []
for i in range(126):
    J.append(np.argmax(neigh.predict_proba(pca_fv1[i:i+1])))

J2 = []
for i in range(1185):
    J2.append(np.argmax(neigh.predict_proba(pca_fv[i:i+1])))

plot_pca(pca_fv,labels,np.unique(labels),model='PCA feature vectors')
plot_pca(pca_fv,J2,np.unique(labels),model='PCA feature vectors')
plot_pca_interactiv(pca_fv,labels,index,model='PCA feature vectors')

cov_type = 'full'

cds, cls, covs, gmm = gaussian_mixture(pca_fv,3,covar_type=cov_type,reps=5,init_procedure='kmeans')

plot_cluster(X=pca_fv,cls=cls,cds=cds,y=labels,covs=covs,idx=[0,1])

#cv_gmm(pca_fv,K_range=range(1,11),n_splits=5,covar_type=cov_type,reps=5,init_procedure='kmeans')

#outliers, probs = potential_outliers(cluster_model=gmm, data=pca_fv1, threshold=0.1)