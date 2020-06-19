import sys
sys.path.append('/zhome/87/9/127623/FagprojektBALANCEDTESTS/Fagprojekt2020')
from CNN.modifyCNN import VGG16
import torch.optim as optim
import torch
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.utils import shuffle
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from Preprossering.PreprosseringPipeline import preprossingPipeline
np.random.seed(42)

def split_dataset_balanced_new(C,path,N,train_split,max_windows,num_channels):
    """ Input: Data and training split (in %)
        Output: Training and test set """
    windows1, labels1, filenames1, window_idx_full1 = C.make_label_cnn(make_from_filenames=None, quality=None,
                                                                       is_usable='Yes', max_files=N, max_windows=max_windows,
                                                                       path=path_s, seed=0, ch_to_include=range(num_channels))
    windows2, labels2, filenames2, window_idx_full2 = C.make_label_cnn(make_from_filenames=None, quality=None,
                                                                       is_usable='No', max_files=N, max_windows=max_windows,
                                                                       path=path_s, seed=0, ch_to_include=range(num_channels))
    key = np.maximum(int(round(len(filenames1) / len(filenames2))),1)
    windows3, labels3, filenames3, window_idx_full3 = C.make_label_cnn(make_from_filenames=None, quality=None,
                                                                       is_usable='No', max_files=N, max_windows=int(round(max_windows*key)),
                                                                       path=path_s, seed=0, ch_to_include=range(num_channels))
    n_train_files1 = int(len(filenames1) / 10 * (train_split/10))
    n_train_files2 = int(len(filenames3) / 10 * (train_split/10))
    a = np.array(window_idx_full1)
    b = np.array(window_idx_full2)
    c = np.array(window_idx_full3)
    j1 = np.unique(a[:, 0], return_counts=True)
    j2 = np.unique(b[:,0], return_counts=True)
    j3 = np.unique(c[:, 0], return_counts=True)
    j1 = np.array(j1)
    j2 = np.array(j2)
    j3 = np.array(j3)
    j1 = np.asarray(j1[1, :], dtype='int64')
    j2 = np.asarray(j2[1,:], dtype='int64')
    j3 = np.asarray(j3[1, :], dtype='int64')
    n1 = np.sum(j1[:n_train_files1])
    n2 = np.sum(j2[:n_train_files2])
    n3 = np.sum(j3[:n_train_files2])
    wt1 = windows1[:n1,:,:,:]
    wt2 = windows3[:n3,:,:,:]
    w1 = torch.cat((wt1,wt2))
    ww1 = windows1[n1:,:,:,:]
    ww2 = windows2[n2:,:,:,:]
    w2 = torch.cat((ww1,ww2))
    l1 = labels3[:n3]+labels2[:n2]
    l2 = labels1[n1:]+labels2[n2:]
    wid = window_idx_full1[n1:]+window_idx_full2[n2:]
    train_windows, train_labels = shuffle(w1,l1)
    test_windows, test_labels, test_id = shuffle(w2,l2,wid)
    return train_windows.detach(), test_windows.detach(), train_labels, test_labels, test_id

def split_dataset_balanced(C,path,N,train_split,max_windows,num_channels):
    """ Input: Data and training split (in %)
        Output: Training and test set """
    windows1, labels1, filenames1, window_idx_full1 = C.make_label_cnn(make_from_filenames=None, quality=None,
                                                                       is_usable='Yes', max_files=N, max_windows=max_windows,
                                                                       path=path_s, seed=0, ch_to_include=range(num_channels))
    windows2, labels2, filenames2, window_idx_full2 = C.make_label_cnn(make_from_filenames=None, quality=None,
                                                                       is_usable='No', max_files=N, max_windows=max_windows,
                                                                       path=path_s, seed=0, ch_to_include=range(num_channels))
    key = np.maximum(int(round(len(filenames1) / len(filenames2))),1)
    windows3, labels3, filenames3, window_idx_full3 = C.make_label_cnn(make_from_filenames=None, quality=None,
                                                                       is_usable='Yes', max_files=N, max_windows=int(round(max_windows/key)),
                                                                       path=path_s, seed=0, ch_to_include=range(num_channels))
    n_train_files1 = int(len(filenames3) / 10 * (train_split/10))
    n_train_files2 = int(len(filenames2) / 10 * (train_split/10))
    a = np.array(window_idx_full1)
    b = np.array(window_idx_full2)
    c = np.array(window_idx_full3)
    j1 = np.unique(a[:, 0], return_counts=True)
    j2 = np.unique(b[:,0], return_counts=True)
    j3 = np.unique(c[:, 0], return_counts=True)
    j1 = np.array(j1)
    j2 = np.array(j2)
    j3 = np.array(j3)
    j1 = np.asarray(j1[1, :], dtype='int64')
    j2 = np.asarray(j2[1,:], dtype='int64')
    j3 = np.asarray(j3[1, :], dtype='int64')
    n1 = np.sum(j1[:n_train_files1])
    n2 = np.sum(j2[:n_train_files2])
    n3 = np.sum(j3[:n_train_files1])
    wt1 = windows3[:n3,:,:,:]
    wt2 = windows2[:n2,:,:,:]
    w1 = torch.cat((wt1,wt2))
    ww1 = windows1[n1:,:,:,:]
    ww2 = windows2[n2:,:,:,:]
    w2 = torch.cat((ww1,ww2))
    l1 = labels3[:n3]+labels2[:n2]
    l2 = labels1[n1:]+labels2[n2:]
    wid = window_idx_full1[n1:]+window_idx_full2[n2:]
    train_windows, train_labels = shuffle(w1,l1)
    test_windows, test_labels, test_id = shuffle(w2,l2,wid)
    return train_windows.detach(), test_windows.detach(), train_labels, test_labels, test_id

def split_dataset(C,path,N,train_split,max_windows,num_channels):
    """ Input: Data and training split (in %)
        Output: Training and test set """
    windows1, labels1, filenames1, window_idx_full1 = C.make_label_cnn(make_from_filenames=None, quality=None,
                                                                       is_usable='Yes', max_files=N, max_windows=max_windows,
                                                                       path=path_s, seed=0, ch_to_include=range(num_channels))
    windows2, labels2, filenames2, window_idx_full2 = C.make_label_cnn(make_from_filenames=None, quality=None,
                                                                       is_usable='No', max_files=N, max_windows=max_windows,
                                                                       path=path_s, seed=0, ch_to_include=range(num_channels))
    n_train_files1 = int(len(filenames1) / 10 * (train_split/10))
    n_train_files2 = int(len(filenames2) / 10 * (train_split/10))
    a = np.array(window_idx_full1)
    b = np.array(window_idx_full2)
    j1 = np.unique(a[:, 0], return_counts=True)
    j2 = np.unique(b[:,0], return_counts=True)
    j1 = np.array(j1)
    j2 = np.array(j2)
    j1 = np.asarray(j1[1, :], dtype='int64')
    j2 = np.asarray(j2[1,:], dtype='int64')
    n1 = np.sum(j1[:n_train_files1])
    n2 = np.sum(j2[:n_train_files2])
    wt1 = windows1[:n1,:,:,:]
    wt2 = windows2[:n2,:,:,:]
    w1 = torch.cat((wt1,wt2))
    ww1 = windows1[n1:,:,:,:]
    ww2 = windows2[n2:,:,:,:]
    w2 = torch.cat((ww1,ww2))
    l1 = labels1[:n1]+labels2[:n2]
    l2 = labels1[n1:]+labels2[n2:]
    wid = window_idx_full1[n1:]+window_idx_full2[n2:]
    train_windows, train_labels = shuffle(w1,l1)
    test_windows, test_labels, test_id = shuffle(w2,l2,wid)
    return train_windows.detach(), test_windows.detach(), train_labels, test_labels, test_id

C = preprossingPipeline(BC_datapath=r"/work3/s173934/Fagprojekt/dataEEG")
path_s = r'/work3/s173934/Fagprojekt/spectograms_rgb'
criterion = nn.CrossEntropyLoss()
_, _, Y_train1, _, _ = split_dataset_balanced(C,path_s,N=120,train_split=80,max_windows=20,num_channels=14)
_, _, Y_train2, _, _ = split_dataset(C,path_s,N=120,train_split=80,max_windows=20,num_channels=14)
_, _, Y_train3, _, _ = split_dataset(C,path_s,N=120,train_split=80,max_windows=20,num_channels=14)
np.save('train_labels.npy',np.asarray(Y_train1))
np.save('train_labels_ub.npy',np.asarray(Y_train2))
np.save('train_labels_b.npy',np.asarray(Y_train3))
