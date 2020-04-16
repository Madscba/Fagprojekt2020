
from Villads.PCA_TSNE_classes import *

feature_vectors_scaled=scale_data(featureVectors)
pca, pca_features=make_pca(feature_vectors_scaled)
tsne, tsne_features=make_TSNE(pca_features[:,:1000])
plot_pca_tsne_2D(tsne_features)


