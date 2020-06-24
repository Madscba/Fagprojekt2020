import os, re, glob, json, sys
import pandas as pd
import numpy as np
from result_processing.pairwiseModelComparison import mcnemar
from Preprossering.LoadEXPDICT import load_pred

def mcNemar(true_label, modelA_pred,modelB_pred):
    true_label = np.where(true_label == 'Yes', 1, 0)
    modelA_pred = np.where(modelA_pred == 'Yes', 1, 0)
    modelB_pred = np.where(modelB_pred == 'Yes', 1, 0)
    data, measurements = [], []

    [thetahatA, CIA, p] = mcnemar(true_label,modelA_pred,modelB_pred, alpha=0.05)
    return [thetahatA, CIA,p]


with open(r"Preprossering\\edfFiles.json") as json_file:
    dict = json.load(json_file)
filelits=list(dict.keys())
y_true_bal, test_size_bal, RF_pred_bal, SVM_pred_bal, LDA_pred_bal, GNB_pred_bal, index_bal, filenames_bal=load_pred(r"C:\Users\Mads-\Downloads", r"Spectrograms_bal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'],filelist=filelits)
y_true_unbal, test_size_unbal, RF_pred_unbal, SVM_pred_unbal, LDA_pred_unbal, GNB_pred_unbal, index_unbal, filenames_unbal=load_pred(r"C:\Users\Mads-\Downloads", r"Spectrograms_unbal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'],filelist=filelits)
y_true_unbal_feat, test_size_unbal_feat, RF_pred_unbal_feat, SVM_pred_unbal_feat, LDA_pred_unbal_feat, GNB_pred_unbal_feat, index_unbal_feat, filenames_unbal_feat = load_pred(r"C:\Users\Mads-\Downloads", r"Feature_unbal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'],filelist=filelits)
y_true_bal_feat, test_size_bal_feat, RF_pred_bal_feat, SVM_pred_bal_feat, LDA_pred_bal_feat, GNB_pred_bal_feat, index_bal_feat, filenames_bal_feat = load_pred(r"C:\Users\Mads-\Downloads", r"Feture_bal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'],filelist=filelits)
print(f"Spectrograms bal VS ubal {np.all([index_unbal==index_bal])}")
print(f"Unbalanced spec VS feat {np.all([index_unbal==index_bal_feat])}")
print(f"balanced spec VS feat {np.all([index_bal==index_bal_feat])}")
print(f"Features bal VS unbal {np.all([index_bal_feat==index_unbal_feat])}")

base_svmF_b_est = mcNemar(y_true_bal,np.repeat('Yes',len(y_true_bal)),SVM_pred_bal_feat)
base_svmS_b_est = mcNemar(y_true_bal,np.repeat('Yes',len(y_true_bal)),SVM_pred_bal)
svmF_svmS_b_est = mcNemar(y_true_bal,SVM_pred_bal_feat,SVM_pred_bal)

bal_results = [base_svmF_b_est,base_svmS_b_est,svmF_svmS_b_est]

base_rfF_u_est = mcNemar(y_true_unbal,np.repeat('Yes',len(y_true_bal)),RF_pred_unbal_feat)
base_rfS_u_est = mcNemar(y_true_unbal,np.repeat('Yes',len(y_true_bal)),RF_pred_unbal)
rfF_rfS_u_est = mcNemar(y_true_unbal,RF_pred_unbal_feat,RF_pred_unbal)
unbal_results = [base_rfF_u_est,base_rfS_u_est,rfF_rfS_u_est]

pass

