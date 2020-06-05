##Mads Christian

#######################################################
###########- Load data (Featurevectors)################
#######################################################
import numpy as np
from Preprossering.PreprosseringPipeline import preprossingPipeline
from Classifier_experimentOne_isUsable.trainTestValidateClassifiers import getClassifierAccuracies,tryNewDiv
import random
import os
random.seed(42)

# text_file = open("filenames.txt", "r")
# lines = text_file.readlines()
# text_file.close()




C=preprossingPipeline(mac=False, BC_datapath=r"C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG")
path=r'C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\FeatureVectors'
N=126
feature_vectors_is_usable,labels_is_usable,filenames_is_usable,_= C.make_label(make_from_filenames=False,quality=None,max_files=N,is_usable="Yes",path = path) #18 files = 2144
feature_vectors_not_usable,labels_not_usable,filenames_not_usable,_= C.make_label(make_from_filenames=False,quality=None,is_usable='No',max_files=N ,path = path) #18 files = 1997

l1 = np.append(np.repeat('Yes',len(filenames_is_usable.size)),np.repeat('No',len(filenames_not_usable)))
f=open('isUsable.txt','w')
for ele in l1:
    f.write(ele+'\n')
f.close()
# x_train_feature = np.vstack((feature_vectors_is_usable[:2144,:],feature_vectors_not_usable[:1997,:]))
# y_train_feature = np.hstack((labels__is_usable[:2144],labels_not_usable[:1997]))
#
# x_test_feature = np.vstack((feature_vectors_is_usable[2144:,:],feature_vectors_not_usable[1997:,:]))
# y_test_feature = np.hstack((labels__is_usable[2144:],labels_not_usable[1997:]))
#
# #Free some memory
# del feature_vectors_is_usable
# del feature_vectors_not_usable
# del labels__is_usable
# del labels_not_usable
# #
# results_feature = tryNewDiv(x_train_feature,y_train_feature,5,x_test_feature,y_test_feature)
# np.save(r'C:\Users\Mads-_uop20qq\Documents\fagprojekt',results_feature,True)
# #
# del x_train_feature
# del y_train_feature
# del x_test_feature
# del y_test_feature

path_spec = r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\Spektrograms'

spectrogram_is_usable,labels_is_usable_spec,_,_= C.make_label(make_from_filenames=filenames_is_usable,quality=None,is_usable="Yes",max_files=N,path = path) #18 files = 2074
spectrogram_not_usable,labels_not_usable_spec,_,_= C.make_label(make_from_filenames=filenames_not_usable,quality=None,is_usable='No',max_files=N ,path = path) #18 files = 1926


x_train_spec =np.vstack((spectrogram_is_usable[:2074,:],spectrogram_not_usable[:1926,:]))
y_train_spec = np.hstack((labels_is_usable_spec[:2074],labels_not_usable_spec[:1926]))

x_test_spec = np.vstack((spectrogram_is_usable[2074:,:],spectrogram_not_usable[1926:,:]))
y_test_spec = np.hstack((labels_is_usable_spec[2074:],labels_not_usable_spec[1926:]))

results_spec = tryNewDiv(x_train_spec,y_train_spec,5,x_test_spec,y_test_spec)
np.save(r'C:\Users\Mads-_uop20qq\Documents\fagprojekt',results_spec,True)

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