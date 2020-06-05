import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

"""
Functions for interactive plots made by Andreas 

"""
#data = np.load(r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\wetransfer-2bf20e\PCA_TSNE\pca_features.npy')
def plot_pca_interactiv(Xdata,ydata,considered_classes,pca_components=[0,1], model='PCA',plot_extremes=True):
    if type(ydata[0]) == str: #If ydata is an array of categorical strings, convert them into numerical categorical values
        le = LabelEncoder()
        ydata = le.fit_transform(ydata)
    elif type(ydata[0]==bool): #If ydata is an array of categorical boolean values, convert them into numerical categorical values
        ydata = np.multiply(ydata, 1)

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