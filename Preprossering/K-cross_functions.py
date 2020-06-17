


"""
File to gennerate K-corss valdiation splits.


Responsble Andreas
"""

import os, re, glob, json, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from Preprossering.loadData import jsonLoad

def make_fold(stucture,path_edf,path_fold,lable,type):
    """

    :param stucture: list of fold stucture eg: [k1,k2] for two fold.
    :param path_edf: path to experiments edfFiles
    :param path_fold: path and file name .json to save fold
    :param lable: lable of annotaition for stratified split
    :param type: "still", "random", "stratifiedstill", Stratifiedsshuffle"
    :return:
    """
    edfdict=jsonLoad(path_edf)

    filenames=np.array(list(edfdict.keys()))
    lables=np.array([edfdict[key]["annotation"][lable] for key in filenames])

    folds=subfold(stucture,filenames,lables,type)

    with open(os.path.join(os.getcwd(),path_fold), 'w') as fp:
        json.dump(folds, fp, indent=4)

def subfold(stucture,filenames,y,type):
    """
    Recrusive function to gennerate multiple layer k split.
    :param stucture:
    :param edfdict:
    :return: dictionary of subfold
    """
    randstate=0
    if len(stucture) !=0:
        folds={}
        k=stucture[0]


        if type=="shuffle":
            kf = KFold(n_splits=k,shuffle=True,random_state=randstate)
        elif type=="stil":
            kf = KFold(n_splits=k, shuffle=False)
        elif type=="statifiedstill":
            kf = StratifiedKFold(n_splits=k, shuffle=False)
        elif type=="statifiedshuffle":
            kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=randstate)


        i=0
        for train_id,text_id in kf.split(np.array(filenames),np.array(y)):
            folds["N_folds"]=k
            folds[f"train_{i}"]=filenames[train_id].tolist()
            folds[f"test_{i}"]=filenames[text_id].tolist()
            folds[f"subfolds_{i}"]=subfold(stucture[1:],filenames[train_id],y[train_id],type)
            folds[f"train_len_{i}"]=len(train_id)
            folds[f"test_len_{i}"]=len(text_id)
            i+=1
            for l in set(y):
                print(f"lable balance {l} = {np.sum([y==l])/len(y)}")

        if len(set(np.append(folds["train_1"],folds["test_1"])))==len(np.append(folds["train_1"],folds["test_1"])):
            return folds
        else:
            raise Exception("Same index in train and test set")


if __name__ == '__main__':
    make_fold([5],path_edf=os.path.join(os.getcwd(), r"Preprossering/edfFiles.json"),path_fold=r"Preprossering//K-stratified_is_useble_shuffle9.json",lable="Is Eeg Usable For Clinical Purposes",type="statifiedshuffle")
