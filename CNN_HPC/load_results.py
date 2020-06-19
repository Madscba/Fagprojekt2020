import torch
import numpy as np
import torch.nn as nn
import pickle
import sys
from CNN.modifyCNN import VGG16


path_results = r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Kode\Gruppen\Fagprojekt2020\LR1'

model = VGG16()
model.load_state_dict(torch.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Kode\Gruppen\Fagprojekt2020\LR1\model0_l1.pt'))
valid_acc = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\LR1\valid_acc_l1_0.npy')
wrong_guess_class = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\LR1\wrong_guesses_class_l1_0.npy')
