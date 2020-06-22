""" @author: Johannes"""
import numpy as np
import os
import mne
import random
import matplotlib.pyplot as plt
from Villads.PCA_TSNE_classes import scale_data
from sklearn.metrics import accuracy_score
import pandas as pd
from GMM.gmm import gaussian_mixture, plot_cluster, cv_gmm, potential_outliers, plot_pca, plot_pca_interactiv
from sklearn.preprocessing import StandardScaler
from Preprossering.PreprosseringPipeline import preprossingPipeline
from sklearn.neighbors import KNeighborsClassifier
#from Visualization.PCA_TSNE_plot import plot_pca
from sklearn.decomposition import PCA
from CNN_HPC.load_results import test_labels, test_windows
from CNN_HPC.load_results_ub import test_labels_ub, test_windows_ub

#from Visualization.plot_inteactive_functions import plot_pca_interactiv
random.seed(42)

new_fv_b = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Kode\Gruppen\Fagprojekt2020\CNN_HPC\new_feature_vectors_b.npy')
cpi = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Kode\Gruppen\Fagprojekt2020\CNN_HPC\cpi.npy')
y_true = np.where(test_labels == 'Yes', 1, 0)
labels = np.array(y_true[cpi],dtype='int64')
windows = test_windows[cpi]

new_fv_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Kode\Gruppen\Fagprojekt2020\CNN_HPC\new_feature_vectors_ub.npy')
cpi_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Kode\Gruppen\Fagprojekt2020\CNN_HPC\cpi_ub.npy')
y_true_ub = np.where(test_labels_ub == 'Yes', 1, 0)
labels_ub = np.array(y_true_ub[cpi_ub],dtype='int64')
windows_ub = test_windows_ub[cpi_ub]

def fit_pca(fv):
    scaled_new_fv = scale_data(fv)
    pca = PCA()
    pca.fit(fv)
    pca_fv = pca.transform(fv)
    return pca_fv

def get_marked_window(mark,windows):
    mark_window = np.unique(np.where((windows[:, 0] == mark[0]) & (windows[:, 1] == mark[1]) & (windows[:, 2] == mark[2])))
    return mark_window
def append_mark(marks):
    mark_list = np.zeros(8)
    for i in range(8):
        mark_list[i] = marks[i]
    return mark_list
pca_fv = fit_pca(new_fv_b)

plot_pca(pca_fv,test_labels[cpi],np.unique(test_labels[cpi]),model='PCA on new feature vectors')
plot_pca_interactiv(pca_fv,labels,test_windows[cpi,:])

cds, cls, covs, gmm = gaussian_mixture(pca_fv,4,covar_type='full',reps=5,init_procedure='kmeans')

plot_cluster(X=pca_fv,cls=cls,cds=cds,y=labels,covs=covs,idx=[0,1])

mark1 = get_marked_window(['sbs2data_2018_09_06_14_06_21_232.edf','15','13'],windows)
mark4 = get_marked_window(['sbs2data_2018_09_04_08_19_37_370.edf','84','3'],windows)
mark6 = get_marked_window(['sbs2data_2018_08_30_19_37_22_286 Part2.edf','18','7'],windows)
mark8 = get_marked_window(['sbs2data_2018_09_06_14_06_21_232.edf','90','12'],windows)
mark2 = get_marked_window(['sbs2data_2018_09_01_09_08_16_335.edf','0','4'],windows)
mark3 = get_marked_window(['sbs2data_2018_09_03_13_54_22_365.edf','0','2'],windows)
mark5 = get_marked_window(['sbs2data_2018_09_01_09_08_16_335.edf','52','12'],windows)
mark7 = get_marked_window(['sbs2data_2018_09_06_11_50_34_411.edf','0','5'],windows)

mark9 = get_marked_window(['sbs2data_2018_09_06_14_06_21_232.edf','45','13'],windows)

marks = append_mark([mark1, mark2, mark3, mark4, mark5, mark6, mark7, mark8])
np.save('marked_windows.npy', np.array(marks,dtype='int64'))