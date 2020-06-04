##Mads Christian
import numpy as np
from sklearn import model_selection
from Preprossering.PreprosseringPipeline import preprossingPipeline
from Classifier_experimentOne_isUsable.trainTestValidateClassifiers import getClassifierAccuracies,tryNewDiv
import random
import os
random.seed(42)

text_file = open("filenames.txt", "r")
x = text_file.read().splitlines() #Read 126 files, first 100 is usable, the next 26 is not usable
text_file.close()

text_file = open("isUsable.txt", "r")
y = text_file.read().splitlines() #Read 126 files, first 100 is usable, the next 26 is not usable
text_file.close()

K = 5
CV = model_selection.KFold(n_splits=K, shuffle=True)

C=preprossingPipeline(mac=False, BC_datapath=r"C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG")
path=r'C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\FeatureVectors'
path_spec = r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\Spektrograms'

N=126
for (train_index, test_index) in CV.split(x,y):
    x_train = list(x[i] for i in train_index)
    x_test = list(x[i] for i in test_index)


    feature_vectors_train,feature_vectors_labels_train,_,_= C.make_label(make_from_filenames=x_train,quality=None,is_usable=None,max_files=N,path = path) #18 files = 2144
    feature_vectors_test,feature_vectors_labels_test,_,_= C.make_label(make_from_filenames=x_test,quality=None,is_usable=None,max_files=N,path = path) #18 files = 2144
    results_feature = tryNewDiv(feature_vectors_train,feature_vectors_labels_train,5,feature_vectors_test,feature_vectors_labels_test)


    spectrograms_test,spectrogram_labels_test, _, _ = C.make_label(make_from_filenames=x_test,quality=None, is_usable=None, max_files=N,path=path)  # 18 files = 1926
    spectrograms_train,spectrogram_labels_train, _, _ = C.make_label(make_from_filenames=x_train,quality=None, is_usable=None, max_files=N,path=path)  # 18 files = 1926
    results_spec = tryNewDiv(spectrograms_train,spectrogram_labels_train,5,spectrograms_test,spectrogram_labels_test)
    # np.save(r'C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\Fagprojekt2020\Classifier_experimentOne_isUsable',results_spec,True)
    # np.save(r'C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\Fagprojekt2020\Classifier_experimentOne_isUsable', results_feature, True)

# spec_is_usable_train = np.array([])
# spec_not_usable_train = np.array([])
# spec_is_usable_test = np.array([])
# spec_not_usable_test = np.array([])
# path_spec = r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\Spektrograms'
#
# used_usable_files = np.unique([x[0] for x in window_idx_full_is_usable[:]])
# used_not_usable_files = np.unique([x[0] for x in window_idx_full_not_usable[:]])
#
# for i in range(N):
#     if (i==0):
#         spec_is_usable_train = np.load(os.path.join(path_spec, used_usable_files[i] + '.npy')).squeeze(0)
#         spec_not_usable_train = np.load(os.path.join(path_spec, used_not_usable_files[i] + '.npy')).squeeze(0)
#     elif (i <18):
#         spec_is_usable_train = np.vstack( (spec_is_usable_train, np.load(os.path.join(path_spec, used_usable_files[i] + '.npy')).squeeze(0)) )
#         spec_not_usable_train = np.vstack( (spec_not_usable_train, np.load(os.path.join(path_spec, used_not_usable_files[i] + '.npy')).squeeze(0)) )
#     elif (i == 18):
#         spec_is_usable_test = np.load(os.path.join(path_spec, used_usable_files[i] + '.npy')).squeeze(0)
#         spec_not_usable_test = np.load(os.path.join(path_spec, used_not_usable_files[i] + '.npy')).squeeze(0)
#     else:
#         spec_is_usable_test = np.vstack((spec_is_usable_test, np.load(os.path.join(path_spec, used_usable_files[i] + '.npy')).squeeze(0)))
#         spec_not_usable_test = np.vstack((spec_not_usable_test, np.load(os.path.join(path_spec, used_not_usable_files[i] + '.npy')).squeeze(0)))
#
# spec_train = np.vstack((spec_is_usable_train,spec_not_usable_train))
# spec_test = np.vstack((spec_is_usable_test,spec_not_usable_test))




#getClassifierAccuracies(feature_vectors,labels,5,x_test,y_test)
#test = np.load(r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\wetransfer-2bf20e\PCA_TSNE\pca_features.npy')
pass



##'sbs2data_2018_09_01_10_51_53_336.edf', 'sbs2data_2018_09_01_10_52_06_332.edf', 'sbs2data_2018_09_01_11_15_15_345DJP.edf', 'sbs2data_2018_09_01_11_33_40_98.edf', 'sbs2data_2018_09_01_11_38_26_333.edf', 'sbs2data_2018_09_01_11_56_10_342.edf', 'sbs2data_2018_09_01_12_24_19_105.edf', 'sbs2data_2018_09_01_12_30_52_341.edf', 'sbs2data_2018_09_01_12_43_34_344.edf



#####################################################
#############Get labels (Is usable)##################
#####################################################

##Setup models

##Divide data into train/test/val (set seed) (perhaps stratified)

##Get accuracies