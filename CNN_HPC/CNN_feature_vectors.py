import torch
import numpy as np
import torch.nn as nn
import pickle
import sys
from CNN.modifyCNN import VGG16

def new_forward(model, x):
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.classifier[0:5](x)
    return x

test_set = torch.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\test_set_ub.pt')
test_labels = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\test_labels_ub.npy')
test_windows = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\test_windows_ub.npy')
correct_pred = []
model1_b = VGG16()
model1_b.load_state_dict(torch.load(PATH))

feature_vectors = np.array([1234,4096])

for i in range(np.shape(correct_pred)[0]):
    feature_vectors[i,:] = new_forward(model1_b,test_set[correct_pred])
