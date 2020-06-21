import torch
import numpy as np
import torch.nn as nn
import pickle
import sys
from CNN.modifyCNN import VGG16

test_set = torch.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\test_set_b.pt')
test_labels = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\test_labels.npy')
test_windows = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\test_windows.npy')

""" Results for CNN trained on balanced dataset"""
#model1_l2 = VGG16()
#model1_l2.load_state_dict(torch.load(r'D:\Results Johannes\model1_l2.pt'))
#model1_l3 = VGG16()
#model1_l3.load_state_dict(torch.load(r'D:\Results Johannes\model1_l3.pt'))

valid_acc_l1_0 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\valid_acc_l1_0_b.npy')
valid_acc_l1_1 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\valid_acc_l1_1_b.npy')
#valid_acc_l2_0 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\valid_acc_l2_0_b.npy')
#valid_acc_l2_1 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\valid_acc_l2_1_b.npy')
#valid_acc_l3_0 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\valid_acc_l3_0_b.npy')
#valid_acc_l3_1 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\valid_acc_l3_1_b.npy')
valid_acc_l4_0 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\valid_acc_l4_0_b.npy')
valid_acc_l4_1 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\valid_acc_l4_1_b.npy')
valid_acc_l5_0 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\valid_acc_l5_0_b.npy')
valid_acc_l5_1 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\valid_acc_l5_1_b.npy')



all_guesses_l1_0 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\all_guesses_l1_0_b.npy')
all_guesses_l1_1 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\all_guesses_l1_1_b.npy')
#all_guesses_l2_0 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\all_guesses_l2_0_b.npy')
#all_guesses_l2_1 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\all_guesses_l2_1_b.npy')
#all_guesses_l3_0 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\all_guesses_l3_0_b.npy')
#all_guesses_l3_1 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\all_guesses_l3_1_b.npy')
all_guesses_l4_0 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\all_guesses_l4_0_b.npy')
all_guesses_l4_1 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\all_guesses_l4_1_b.npy')
all_guesses_l5_0 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\all_guesses_l5_0_b.npy')
all_guesses_l5_1 = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\all_guesses_l5_1_b.npy')


def append_results(num_results,results):
    array = np.array([])
    for i in range(num_results):
        array = np.append(array, results[i])
    return array

balanced_results = append_results(6,np.array([all_guesses_l1_0,all_guesses_l1_1,all_guesses_l4_0,all_guesses_l4_1,all_guesses_l5_0,all_guesses_l5_1]))

