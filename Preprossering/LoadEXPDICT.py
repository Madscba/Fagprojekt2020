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
            RF_pred_new = np.array([])
            SVM_pred_new = np.array([])
            LDA_pred_new = np.array([])
            GNB_pred_new = np.array([])
            y_true_new = np.array([])
            index_new = np.empty((0,2))
            filenames_new= np.array([])
            print("True")
            for file in filelist:
                i=np.array(index)[:, 0]==file
                RF_pred_new=np.append(RF_pred_new,RF_pred[i])
                SVM_pred_new = np.append(SVM_pred_new, SVM_pred[i])
                LDA_pred_new = np.append(LDA_pred_new, LDA_pred[i])
                GNB_pred_new = np.append(GNB_pred_new, GNB_pred[i])
                y_true_new = np.append(y_true_new,y_true[i])
                index_new= np.append(index_new,np.array(index)[i,:],axis=0)
                filenames_new=np.append(filenames_new,filenames[i])


            RF_pred=RF_pred_new
            SVM_pred=SVM_pred_new
            LDA_pred=LDA_pred_new
            GNB_pred=GNB_pred_new
            y_true=y_true_new
            index=index_new
            print(f"Sanitychek {np.array(index)[:, 0]==filenames_new}")
            filenames=filenames_new


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
#<<<<<<< HEAD
    with open(r"Preprossering\\edfFiles.json") as json_file:
        dict = json.load(json_file)
    filelits=list(dict.keys())
    y_true_bal, test_size_bal, RF_pred_bal, SVM_pred_bal, LDA_pred_bal, GNB_pred_bal, index_bal, filenames_bal=load_pred(r"ClassifierTestLogs", r"Spectrograms_bal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'],filelist=filelits)
    y_true_unbal, test_size_unbal, RF_pred_unbal, SVM_pred_unbal, LDA_pred_unbal, GNB_pred_unbal, index_unbal, filenames_unbal=load_pred(r"ClassifierTestLogs", r"Spectrograms_unbal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'],filelist=filelits)
    y_true_unbal_feat, test_size_unbal_feat, RF_pred_unbal_feat, SVM_pred_unbal_feat, LDA_pred_unbal_feat, GNB_pred_unbal_feat, index_unbal_feat, filenames_unbal_feat = load_pred(r"ClassifierTestLogs", r"Feature_unbal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'],filelist=filelits)
    y_true_bal_feat, test_size_bal_feat, RF_pred_bal_feat, SVM_pred_bal_feat, LDA_pred_bal_feat, GNB_pred_bal_feat, index_bal_feat, filenames_bal_feat = load_pred(r"ClassifierTestLogs", r"Feture_bal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'],filelist=filelits)
print(f"Spectrograms bal VS ubal {np.all([index_unbal==index_bal])}")
print(f"Unbalanced spec VS feat {np.all([index_unbal==index_bal_feat])}")
print(f"balanced spec VS feat {np.all([index_bal==index_bal_feat])}")
print(f"Features bal VS unbal {np.all([index_bal_feat==index_unbal_feat])}")
#=======
#    allignTestIdx(r"C:\Users\Mads-\Downloads", r"Spectrograms_bal_final1_predict.json",r"Feture_bal_final1_predict.json",['SVM', 'LDA', 'RF', 'GNB'])
#
#
#    y_true_bal, test_size_bal, RF_pred_bal, SVM_pred_bal, LDA_pred_bal, GNB_pred_bal, index_bal, filenames_bal=load_pred(r"C:\Users\Mads-\Downloads", r"Spectrograms_bal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'])
#    y_true_unbal, test_size_unbal, RF_pred_unbal, SVM_pred_unbal, LDA_pred_unbal, GNB_pred_unbal, index_unbal, filenames_unbal=load_pred(r"C:\Users\Mads-\Downloads", r"Spectrograms_unbal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'])
#    y_true_unbal_feat, test_size_unbal_feat, RF_pred_unbal_feat, SVM_pred_unbal_feat, LDA_pred_unbal_feat, GNB_pred_unbal_feat, index_unbal_feat, filenames_unbal_feat = load_pred(r"C:\Users\Mads-\Downloads", r"Feature_unbal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'])
#    y_true_bal_feat, test_size_bal_feat, RF_pred_bal_feat, SVM_pred_bal_feat, LDA_pred_bal_feat, GNB_pred_bal_feat, index_bal_feat, filenames_bal_feat = load_pred(r"C:\Users\Mads-\Downloads", r"Feture_bal_final1_predict.json", ['SVM', 'LDA', 'RF', 'GNB'])
#print(f"Spectrograms bal VS ubal {np.all([filenames_unbal==filenames_bal])}")
#print(f"Unbalanced spec VS feat {np.all([filenames_unbal==filenames_unbal_feat])}")
#print(f"balanced spec VS feat {np.all([filenames_bal==filenames_bal_feat])}")
#print(f"Features bal VS unbal {np.all([filenames_bal_feat==filenames_unbal_feat])}")
#>>>>>>> 2f8d069d4c7eed40424677769c2bdedcc81479ff
