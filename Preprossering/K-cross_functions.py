


"""
File to gennerate K-corss valdiation splits.


Responsble Andreas
"""

import os, re, glob, json, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from Preprossering.loadData import jsonLoad

def make_fold(stucture,path_edf,path_fold):
    """

    :param stucture: list of fold stucture eg: [k1,k2] for two fold.
    :param path_edf: path to experiments edfFiles
    :param path_fold: path and file name .json to save fold
    :return:
    """
    edfdict=jsonLoad(path_edf)

    filenames=np.array(list(edfdict.keys()))

    folds=subfold(stucture,filenames)

    with open(os.path.join(os.getcwd(),path_fold), 'w') as fp:
        json.dump(folds, fp, indent=4)

def subfold(stucture,filenames):
    """
    Recrusive function to gennerate multiple layer k split.
    :param stucture:
    :param edfdict:
    :return: dictionary of subfold
    """

    if len(stucture) !=0:
        folds={}
        k=stucture[0]
        kf = KFold(n_splits=k,shuffle=True)
        i=0
        for train_id,text_id in kf.split(np.array(filenames)):
            folds["N_folds"]=k
            folds[f"train_{i}"]=filenames[train_id].tolist()
            folds[f"test_{i}"]=filenames[text_id].tolist()
            folds[f"subfolds_{i}"]=subfold(stucture[1:],filenames[train_id])
            folds[f"train_len_{i}"]=len(train_id)
            folds[f"test_len_{i}"]=len(text_id)
            i+=1
        return folds

if __name__ == '__main__':
    make_fold([5,5],path_edf=os.path.join(os.getcwd(), r"Preprossering/edfFiles.json"),path_fold=r"Preprossering//K-fold.json")
