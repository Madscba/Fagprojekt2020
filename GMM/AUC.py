import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import json


def roc_plot(p, y):
    '''
    Plots the receiver operating characteristic (ROC) curve and
    calculates the area under the curve (AUC).
    Input:
         p: Estimated probability of class 1. (Between 0 and 1.)
         y: True class indices. (Equal to 0 or 1.)
    Output:
        AUC: The area under the ROC curve
        TPR: True positive rate
        FPR: False positive rate
    '''
    fpr, tpr, thresholds = metrics.roc_curve(y, p)
    AUC = metrics.roc_auc_score(y, p)
    plt.plot(fpr, tpr, 'r', [0, 1], [0, 1], 'k')
    plt.grid()
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC \n AUC={:.3f}'.format(AUC))
    plt.show()

    return AUC, tpr, fpr

with open(filepath) as json_file:
    tempData = json.load(json_file)
    for c in classifiers:
