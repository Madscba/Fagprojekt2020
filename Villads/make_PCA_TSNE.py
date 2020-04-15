from PCA_TSNE_classes import *
import numpy as np
imgs=np.load('/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/spectograms.npy')
imgs=imgs.reshape(imgs.shape[0],-1)
imgs=scale_data(imgs)
pca_features,_ =make_pca(imgs)
tsne_features=make_TSNE(pca_features[:,0:1000])
plot_pca_tsne_2D(tsne_features)

