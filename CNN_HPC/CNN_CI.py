from result_processing.pairwiseModelComparison import jeffrey_interval
import numpy as np
from CNN_HPC.load_results import balanced_results, test_labels
from CNN_HPC.load_results_ub import imbalanced_results, test_labels_ub


y_true = np.where(test_labels == 'Yes', 1, 0)
y_true_ub = np.where(test_labels_ub == 'Yes', 1, 0)


thetahat = np.zeros((12))
CI = np.zeros((12,2))

key = 7700

j = 0
for i in range(12):
    if i < 6:
        print("reached")
        thetahat[i], CI[i,:] = jeffrey_interval(y=y_true,yhat=balanced_results[i*key:(i+1)*key])
    else:
        thetahat[i], CI[i,:] = jeffrey_interval(y=y_true_ub,yhat=imbalanced_results[j*key:(j+1)*key])
        j += 1

np.save('CI_CNN.npy', CI)
np.save('theta.npy', thetahat)