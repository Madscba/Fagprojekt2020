""" @author: Johannes"""
#from GMM.load_data_bc import *
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from GMM.init import clusterval, clusterplot, gauss_2d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


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

    #BIC = np.zeros((T,))
    #AIC = np.zeros((T,))
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
        #BIC[t,] = gmm2.bic(X)
        #AIC[t,] = gmm2.aic(X)
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
    #plt.plot(K_range, BIC, '-*b')
    #plt.plot(K_range, AIC, '-xr')
    plt.plot(K_range, 2 * CVE, '-ok')
    #plt.legend(['BIC', 'AIC', 'Crossvalidation'])
    plt.legend(['Crossvalidation'])
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

def plot_pca(Xdata,ydata,considered_classes,pca_components=[0,1], model='PCA',title='PCA', plot_extremes=True):
    le = LabelEncoder()
    ydata = le.fit_transform(ydata)


    n_label = len(considered_classes)
    colors =  plt.cm.rainbow(np.linspace(0, 1, n_label))


    cdict = {i: colors[i] for i in range(n_label)}
    label_dict = {i: considered_classes[i] for i in range(n_label)}


    f = plt.figure(figsize=(8,8))
    for i in range(n_label):
        indices = np.where(ydata == i)
        plt.scatter(Xdata[indices, pca_components[0]], Xdata[indices, pca_components[1]], color=cdict[i], label=label_dict[i])
        #plt.annotate(label_dict[i], Xdata[indices[0][0], 0:2]) #First point in each class labelled
    if plot_extremes is not True:
        plt.axis(plot_extremes)
    plt.xlabel('PC {:d} '.format(int(pca_components[0])+1))
    plt.ylabel('PC {:d} '.format(int(pca_components[1])+1))
    plt.legend(loc='best')
    plt.title(model)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from Preprossering.PreprosseringPipeline import preprossingPipeline
import pickle
from Villads.PCA_TSNE_classes import scale_data
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.tools as tls
from plotly.subplots import make_subplots
"""
Functions for interactive plots made by Andreas 

"""
#data = np.load(r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\wetransfer-2bf20e\PCA_TSNE\pca_features.npy')
def plot_pca_interactiv(pca_vectors,labels,window_id,pca1=0,pca2=1,model="PCA",plot=True,index=None):
    """

    :param pca_vectors:
    :param labels:
    :param window_id:
    :param pca1:
    :param pca2:
    :param model:
    :param plot: if true plot else returns trace
    :return:
    """
    #make data frame
    df=(pd.DataFrame(window_id,columns=["file","window","channel"]))
    df[int(pca1)]=pca_vectors[:,int(pca1)]
    try:
        df[int(pca2)]=pca_vectors[:,int(pca2)]
    except:
        print("second component not found insert 0")
        df[int(pca2)]=0
    df["label"]=labels
    df["index"]=df.index
    fig=px.scatter(df,x=int(pca1),y=int(pca2),color="label",hover_data=["file","window","channel"])

    if index != None:
        fig.add_trace(go.Scatter(
            x=[df.loc[index,int(pca1)]],y=[df.loc[index,int(pca2)]],
            name=f'{df.loc[index,"file"]} window: {df.loc[index,"window"]}  channel: {df.loc[index,"channel"]}'
            ,line=dict(color='green', width=10, dash='dot')))

    fig.update_layout(
        title=model,
        xaxis_title=f"Component {pca1}",
        yaxis_title=f"Component {pca2}"
    )
    if plot:
        fig.show()
    else:
        return fig['data'][0]

def plot_comand(feature_path,path_pca,BC_datapath,newplot=True):
    pca = pickle.load(open(path_pca, 'rb'))
    get_data=preprossingPipeline(BC_datapath=BC_datapath)
    feature_vectors,labels,filenames,window_id= get_data.make_label( quality=None, is_usable=None, max_files=10,path = feature_path)
    while True:
        print("What you want to plot, options pca or")
        command=input()
        if command== "pca":
            feature_vectors,labels,filenames,window_id= get_data.make_label( quality=None, is_usable=None, max_files=10,path = feature_path)
            scaled_feature_vectors=scale_data(feature_vectors)
            pca_vectors=pca.transform(scaled_feature_vectors)
            print("what components ex 0 1")
            components=input()
            components=components.split(" ")
            plot_pca_interactiv(pca_vectors,labels,window_id,components[0],components[1])

        if command== "break":
            break

        if command== "spec" or command=="EEG":
            print("Insert index")
            idx=int(input())
            if newplot==False:
                get_data.plot_window(window_id[idx][0],window_id[idx][1],type=command)
            else:
                data,time,chanels=get_data.plot_window(window_id[idx][0],window_id[idx][1],type=command,plot=True)
                data=np.array(data).T
                df=pd.DataFrame(data,columns=chanels,index=time)
                fig = make_subplots(rows=len(chanels), cols=1)
                for i,ch in enumerate(chanels):
                    fig.add_trace(
                        px.line(df,y=ch,x=df.index),
                        row=(i+1),col=1)
                fig=px.line(df,y=chanels[0],x=df.index)
                fig.show()
