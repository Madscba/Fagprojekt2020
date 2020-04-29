from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def scale_data(X):
    standard_scaler = StandardScaler()
    standard_scaler.fit(X)
    scaled_data = standard_scaler.transform(X)
    return scaled_data


def make_pca(X, data_transform=None):
    if data_transform == None:
        data_transform = X
    pca = PCA(n_components=10)
    pca.fit(X)
    return pca


def make_TSNE(X):
    tsne = TSNE()
    tsne_fit = tsne.fit_transform(X)
    return tsne, tsne_fit


def plot_pca_tsne_2D(fitted_data):
    plt.figure(figsize=(8, 8))
    plt.scatter(fitted_data[:, 0], fitted_data[:, 1])
    plt.show()

