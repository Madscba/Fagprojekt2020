import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
"""
File for visualisation of the outer loop of experiment. aka plotting confusione matrix
Responsble Andreas
"""


def all_matrix(lables,classifiers,EXP):
    fig, axis=plt.subplots( ncols=len(classifiers),figsize=(4*len(classifiers),4))
    #fig.subplots_adjust(top=0.85)

    sns.set(font_scale=1.2)

    y_true, test_size, RF_pred, SVM_pred, LDA_pred, GNB_pred=load_pred(r"ClassifierTestLogs",EXP,classifiers)

    for row,c in enumerate(classifiers):
        if c == 'SVM':
            data=SVM_pred
        elif c == 'LDA':

            data=LDA_pred
        elif c == 'RF':

            data=RF_pred
        elif c == 'GNB':
            data=GNB_pred

        #Print one matrix
        MX=pd.DataFrame()
        for l in lables:
            for j in lables:
                prediction=data==np.full(len(y_true),l)
                true=[y_true==np.full(len(y_true),j)]
                MX.loc[f"% True {j}",f"% Predicted {l}"]=np.sum(np.logical_and(prediction,true))/len(y_true)*100

        #MX.loc[:,"N windows"]=MX.sum(axis=1)
        #MX.loc["N windows", :] = MX.sum(axis=0)
        #MXl.append(MX)
        if len(classifiers)>1:
            sns.heatmap(MX, annot=True,fmt=".1f",cbar=False,ax=axis[row])
            axis[row].set_title(f"{c}")
        else:
            sns.heatmap(MX, annot=True, fmt=".1f", cbar=False)
            axis.set_title(f"{c}")
        print(EXP)
    fig.show()

    #MX=MXl[0]

    #for i in range(1,max(data.index)):
    #    MX+=MXl[i]
    #sns.heatmap(MX, annot=True,fmt=".0f",cbar=False,annot_kws={"size": 20})
   # plt.show()



def load_pred(path,filename,classifiers):
    filepath = os.path.join(path, filename)
    test_size = np.array([])
    RF_pred = np.array([])
    SVM_pred = np.array([])
    LDA_pred = np.array([])
    GNB_pred = np.array([])
    y_true = np.array([])
    test_size = np.array([])
    with open(filepath) as json_file:
        tempData = json.load(json_file)
        for i in range(5):
            y_true = np.append(y_true, tempData[f'fold_{i}']['True'])
            test_size = np.append(test_size, np.shape(y_true)[0])
            for c in classifiers:
                if c == 'SVM':
                    SVM_pred= np.append(SVM_pred, tempData[f'fold_{i}'][f'{c}_predict'])
                elif c == 'LDA':

                    LDA_pred = np.append(LDA_pred,tempData[f'fold_{i}'][f'{c}_predict'])
                elif c == 'RF':

                    RF_pred = np.append(RF_pred, tempData[f'fold_{i}'][f'{c}_predict'])
                elif c == 'GNB':
                    GNB_pred = np.append(GNB_pred, tempData[f'fold_{i}'][f'{c}_predict'])
    return y_true, test_size, RF_pred, SVM_pred, LDA_pred,GNB_pred

if __name__ == '__main__':
    all_matrix(["Yes","No"],['SVM'],"Feture_bal_final1_predict.json")


