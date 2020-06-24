import json ,os
import numpy as np

Kfold_path=r"K-stratified_is_useble_shuffle.json"
edf_path=r"edfFiles.json"

with open(os.path.join(os.getcwd(), Kfold_path), "r") as read_file:
    Kfold = json.load(read_file)

with open(os.path.join(os.getcwd(), edf_path), "r") as read_file:
    edfDict = json.load(read_file)

for n in range(Kfold["N_folds"]):
    Train_labels = [edfDict[lable]["annotation"]["Is Eeg Usable For Clinical Purposes"] for lable in Kfold[f"train_{n}"]]
    Test_labels = [edfDict[lable]["annotation"]["Is Eeg Usable For Clinical Purposes"] for lable in Kfold[f"test_{n}"]]

    trainLen=len(Train_labels)
    testLen=len(Test_labels)

    print(f"Fold {n}")
    print(f"train yes {np.sum(Train_labels==np.full(trainLen,'Yes'))}")
    print(f"train no {np.sum(Train_labels == np.full(trainLen, 'No'))}")
    print(f"train total ={trainLen}")
    print(f"test yes {np.sum(Test_labels==np.full(testLen,'Yes'))}")
    print(f"test no {np.sum(Test_labels == np.full(testLen, 'No'))}")
    print(f"test total ={testLen}")