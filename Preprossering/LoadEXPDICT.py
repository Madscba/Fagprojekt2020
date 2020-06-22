import os, re, glob, json, sys
import pandas as pd
import numpy as np

def load_pred(path, dictname, classifiers, filelist=None):
    """

    :param path:
    :param filename:
    :param classifiers:
    :param filelist:
    :return:
    """
    filepath = os.path.join(path, dictname)
    test_size = np.array([])
    RF_pred = np.array([])
    SVM_pred = np.array([])
    LDA_pred = np.array([])
    GNB_pred = np.array([])
    y_true = np.array([])
    test_size = np.array([])
    index = []

    with open(filepath) as json_file:
        tempData = json.load(json_file)
        for i in range(5):
            y_true = np.append(y_true, tempData[f'fold_{i}']['True'])
            index = index + tempData[f'fold_{i}']["index"]
            test_size = np.append(test_size, np.shape(y_true)[0])
            for c in classifiers:
                if c == 'SVM':
                    SVM_pred = np.append(SVM_pred, tempData[f'fold_{i}'][f'{c}_predict'])
                elif c == 'LDA':

                    LDA_pred = np.append(LDA_pred, tempData[f'fold_{i}'][f'{c}_predict'])
                elif c == 'RF':

                    RF_pred = np.append(RF_pred, tempData[f'fold_{i}'][f'{c}_predict'])
                elif c == 'GNB':
                    GNB_pred = np.append(GNB_pred, tempData[f'fold_{i}'][f'{c}_predict'])

        filenames = np.array(index)[:, 0]
        #filenames = np.unique(filenames)
        # Sorting
        if not filelist is None:
            print("True")

    return y_true, test_size, RF_pred, SVM_pred, LDA_pred, GNB_pred, index, filenames

def allignTestIdx(path, dictnameA,dictnameB, classifiers, filelist=None):
    filepath1 = os.path.join(path, dictnameA)
    filepath2 = os.path.join(path, dictnameB)


    with open(filepath1) as json_file:
        feature_data = json.load(json_file)
    for i in range(5):
        y_true = np.append(y_true, feature_data[f'fold_{i}']['True'])
        index = index + feature_data[f'fold_{i}']["index"]
        test_size = np.append(test_size, np.shape(y_true)[0])
        for c in classifiers:
            if c == 'SVM':
                SVM_pred = np.append(SVM_pred, feature_data[f'fold_{i}'][f'{c}_predict'])
            elif c == 'LDA':

                LDA_pred = np.append(LDA_pred, feature_data[f'fold_{i}'][f'{c}_predict'])
            elif c == 'RF':

                RF_pred = np.append(RF_pred, feature_data[f'fold_{i}'][f'{c}_predict'])
            elif c == 'GNB':
                GNB_pred = np.append(GNB_pred, feature_data[f'fold_{i}'][f'{c}_predict'])
    with open(filepath2) as json_file:
        spec_data = json.load(json_file)

    pass




if __name__ == '__main__':
    allignTestIdx(r"C:\Users\Mads-\Downloads", r"Spectrograms_bal_final1_predict.json",r"Feture_bal_final1_predict.json",['SVM', 'LDA', 'RF', 'GNB'])


    y_true_bal, test_size_bal, RF_pred_bal, SVM_pred_bal, LDA_pred_bal, GNB_pred_bal, index_bal, filenames_bal=load_pred(r"C:\Users\Mads-\Downloads", r"Spectrograms_bal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'])
    y_true_unbal, test_size_unbal, RF_pred_unbal, SVM_pred_unbal, LDA_pred_unbal, GNB_pred_unbal, index_unbal, filenames_unbal=load_pred(r"C:\Users\Mads-\Downloads", r"Spectrograms_unbal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'])
    y_true_unbal_feat, test_size_unbal_feat, RF_pred_unbal_feat, SVM_pred_unbal_feat, LDA_pred_unbal_feat, GNB_pred_unbal_feat, index_unbal_feat, filenames_unbal_feat = load_pred(r"C:\Users\Mads-\Downloads", r"Feature_unbal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'])
    y_true_bal_feat, test_size_bal_feat, RF_pred_bal_feat, SVM_pred_bal_feat, LDA_pred_bal_feat, GNB_pred_bal_feat, index_bal_feat, filenames_bal_feat = load_pred(r"C:\Users\Mads-\Downloads", r"Feture_bal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'])
print(f"Spectrograms bal VS ubal {np.all([filenames_unbal==filenames_bal])}")
print(f"Unbalanced spec VS feat {np.all([filenames_unbal==filenames_unbal_feat])}")
print(f"balanced spec VS feat {np.all([filenames_bal==filenames_bal_feat])}")
print(f"Features bal VS unbal {np.all([filenames_bal_feat==filenames_unbal_feat])}")