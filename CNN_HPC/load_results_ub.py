import torch
import numpy as np
import torch.nn as nn
import pickle
import sys
from CNN.modifyCNN import VGG16

test_labels_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\test_labels_ub.npy')
test_windows_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\test_windows_ub.npy')


"""Results for CNN trained on imbalanced dataset"""
valid_acc_l1_0_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\valid_acc_l1_0_ub.npy')
valid_acc_l1_1_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\valid_acc_l1_1_ub.npy')
#valid_acc_l2_0_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\valid_acc_l2_0_ub.npy')
#valid_acc_l2_1_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\valid_acc_l2_1_ub.npy')
#valid_acc_l3_0_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\valid_acc_l3_0_ub.npy')
#valid_acc_l3_1_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\valid_acc_l3_1_ub.npy')
valid_acc_l4_0_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\valid_acc_l4_0_ub.npy')
valid_acc_l4_1_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\valid_acc_l4_1_ub.npy')
valid_acc_l5_0_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\valid_acc_l5_0_ub.npy')
valid_acc_l5_1_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\valid_acc_l5_1_ub.npy')


all_guesses_l1_0_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\all_guesses_l1_0_ub.npy')
all_guesses_l1_1_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\all_guesses_l1_1_ub.npy')
#all_guesses_l2_0_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\all_guesses_l2_0_ub.npy')
#all_guesses_l2_1_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\all_guesses_l2_1_ub.npy')
#all_guesses_l3_0_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\all_guesses_l3_0_ub.npy')
#all_guesses_l3_1_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\all_guesses_l3_1_ub.npy')
all_guesses_l4_0_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\all_guesses_l4_0_ub.npy')
all_guesses_l4_1_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\all_guesses_l4_1_ub.npy')
all_guesses_l5_0_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\all_guesses_l5_0_ub.npy')
all_guesses_l5_1_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\all_guesses_l5_1_ub.npy')


def append_results(num_results,results):
    array = np.array([])
    for i in range(num_results):
        array = np.append(array, results[i])
    return array

imbalanced_results = append_results(6, np.array([all_guesses_l1_0_ub,all_guesses_l1_1_ub,all_guesses_l4_0_ub,all_guesses_l4_1_ub,all_guesses_l5_0_ub,all_guesses_l5_1_ub]))


