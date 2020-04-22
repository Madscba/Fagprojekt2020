
from Villads.PCA_TSNE_classes import *
from Villads.Load_spectograms import spectograms

spectograms=spectograms.squeeze()
spectograms_scaled=scale_data(spectograms)
pca, pca_features=make_pca(spectograms_scaled)
tsne, tsne_features=make_TSNE(pca_features[:,:1000])










