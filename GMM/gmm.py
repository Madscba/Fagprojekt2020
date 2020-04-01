""" @author: Johannes"""
#from GMM.load_data_bc import *
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from GMM.init import clusterval, clusterplot, gauss_2d
import matplotlib.pyplot as plt
import numpy as np

def gaussian_mixture(X,K,covar_type,reps,init_procedure):
    """""
    K: Number of mixture components
    n_splits: number of splits for cross-validation

    Variables for GMM:
        -X: N x M np.array
        -covar_type: 'full' or 'diag'
        -reps: number of fits with different initalizations, best result will be kept
        -init_procedure:'kmeans' or 'random'
    """
    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=K, covariance_type=covar_type,
                          n_init=reps, init_params=init_procedure,
                          tol=1e-6, reg_covar=1e-6).fit(X)

    # Extract cluster centroids (means of gaussians)
    cds = gmm.means_

    # Extract cluster labels
    cls = gmm.predict(X)

    # Extract covariances of clusters
    covs = gmm.covariances_
    return cds, cls, covs, gmm

def plot_cluster(X,cls,cds,y,covs,idx):
    """" Plot of clusteroids
    idx: feature index, choose two features to use as x and y axis in the plot
    """
    plt.figure(figsize=(14, 9))
    clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
    plt.show()

def cv_gmm(X,K_range,n_splits,covar_type,reps,init_procedure):
    from sklearn import model_selection
    """""
    K_range: Range of K's to try
    n_splits: number of splits for cross-validation

    Variables for GMM:
        -X: N x M np.array
        -covar_type: 'full' or 'diag'
        -reps: number of fits with different initalizations, best result will be kept
        -init_procedure:'kmeans' or 'random'
    """
    T = len(K_range)

    BIC = np.zeros((T,))
    AIC = np.zeros((T,))
    CVE = np.zeros((T,))

    # K-fold crossvalidation
    CV = model_selection.KFold(n_splits=n_splits, shuffle=True)

    # List of means and covariances
    #cds = []
    #cls = []

    for t, K in enumerate(K_range):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm2 = GaussianMixture(n_components=K, covariance_type=covar_type,
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)
        # Get BIC and AIC
        BIC[t,] = gmm2.bic(X)
        AIC[t,] = gmm2.aic(X)
        gmm2.bic()
        #cds.append(gmm.means_)
        #cls.append(gmm.predict(X))

        # For each fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm2 = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm2.score_samples(X_test).sum()

    # Plot results for all K's
    plt.figure(1)
    plt.plot(K_range, BIC, '-*b')
    plt.plot(K_range, AIC, '-xr')
    plt.plot(K_range, 2 * CVE, '-ok')
    plt.legend(['BIC', 'AIC', 'Crossvalidation'])
    plt.xlabel('K')
    plt.ylabel(['Loss'])
    #plt.savefig('GMMs.png')
    plt.show()

def potential_outliers(cluster_model, data, threshold):
    outliers = []
    cluster_model.predict(data)
    probs = cluster_model.predict_proba(data)
    max = 0
    for i in range(np.shape(data)[0]):
        max = np.max(probs[i,:])
        if max > threshold:
            pass
        else:
            outliers.append(np.argmax(probs[i,:]))
    return outliers, probs

