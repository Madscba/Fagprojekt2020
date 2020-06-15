import sys
sys.path.append('/zhome/87/9/127623/Fagprojekt/Fagprojekt2020')
from CNN.modifyCNN import VGG16, freeze_parameters, grad_parameters, list_of_features, check_grad
import torch.optim as optim
import torch
from CNN_HPC.CNN_lr1 import X_train, X_valid, Y_train, Y_valid, windows_id
from CNN_HPC.CNN_lr1 import test_CNN
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.utils import shuffle
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from Preprossering.PreprosseringPipeline import preprossingPipeline
np.random.seed(42)

C = preprossingPipeline(BC_datapath=r"/work3/s173934/Fagprojekt/dataEEG")
path_s = r'/work3/s173934/Fagprojekt/spectograms_rgb'
criterion = nn.CrossEntropyLoss()
modelA = VGG16()
freeze_parameters(modelA,feature_extracting=True)
list2 = np.array(list_of_features(modelA))
modelB = VGG16()
freeze_parameters(modelB,feature_extracting=True)
for i in range(2):
    PATH = '/zhome/87/9/127623/Fagprojekt'
    if i == 0:
        activation_list = np.array([26, 27, 28, 29, 30, 31])
        grad_parameters(modelA, list(list2[activation_list]))
        optimizer = optim.Adam(modelA.parameters(), lr=0.02)
        train_acc, train_loss, val_acc, val_loss, wrong_guesses, wrong_predictions, modelA = test_CNN(modelA, X_train, Y_train, X_valid,
                                                                                  Y_valid, windows_id, batch_size=128,
                                                                                  num_epochs=4, preprocessed=True)
        torch.save(modelA.state_dict(), 'model0_l5.pt')
    else:
        activation_list = np.array([20,21,22,23,24,25,26, 27, 28, 29, 30, 31])
        grad_parameters(modelB, list(list2[activation_list]))
        optimizer = optim.Adam(modelB.parameters(), lr=0.02)
        train_acc, train_loss, val_acc, val_loss, wrong_guesses, wrong_predictions, modelB = test_CNN(modelB, X_train, Y_train, X_valid,
                                                                                  Y_valid, windows_id, batch_size=128,
                                                                                  num_epochs=4, preprocessed=True)
        torch.save(modelB.state_dict(), 'model1_l5.pt')

    train_acc_data = np.asarray(train_acc)
    np.save((f'train_acc_l5_{i}.npy'), train_acc_data)
    print("\n reached: second saving place")
    train_loss_data = np.asarray(train_loss)
    np.save((f'train_loss_l5_{i}.npy'), train_loss_data)
    valid_acc_data = np.asarray(val_acc)
    np.save((f'valid_acc_l5_{i}.npy'), valid_acc_data)
    valid_loss_data = np.asarray(val_loss)
    np.save((f'valid_loss_l5_{i}.npy'), valid_loss_data)
    wrong_guesses_data = np.asarray(wrong_guesses)
    np.save((f'wrong_guesses_l5_{i}.npy'), wrong_guesses_data)
    wrong_pred_data = np.asarray(wrong_predictions)
    np.save((f'wrong_guesses_class_l5_{i}.npy'), wrong_pred_data)