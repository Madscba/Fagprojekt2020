from result_processing.pairwiseModelComparison import jeffrey_interval
import numpy as np
from CNN_HPC.load_results import balanced_results, test_labels
from CNN_HPC.load_results_ub import imbalanced_results


y_true = np.where(test_labels == 'Yes', 1, 0)

thetahat = np.zeros((12))
CI = np.zeros((12,2))

key = 7700

j = 0
for i in range(12):
    if i < 6:
        thetahat[i], CI[i,:] = jeffrey_interval(y=y_true,yhat=balanced_results[i*key:(i+1)*key])
    else:
        thetahat[i], CI[i,:] = jeffrey_interval(y=y_true,yhat=imbalanced_results[j*key:(j+1)*key])
        j += 1

