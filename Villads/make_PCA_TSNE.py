from Villads.Load_FeatureVectors import featureVectors
from Villads.PCA_TSNE_classes import *


spectograms_scaled=scale_data(featureVectors)
pca=make_pca(spectograms_scaled)
filename='PCA_feature_vectors.sav'
pickle.dump(pca, open(filename, 'wb'))
#tsne, tsne_features=make_TSNE(pca_features[:,:1000])










