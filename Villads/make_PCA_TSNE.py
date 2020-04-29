from Villads.Load_FeatureVectors import featureVectors
from Villads.PCA_TSNE_classes import *
from sklearn.decomposition import PCA
import pickle

spectograms_scaled=scale_data(featureVectors)
pca=PCA()
pca.fit(spectograms_scaled)
filename='PCA_feature_vectors_1.sav'
pickle.dump(pca, open(filename, 'wb'))
#tsne, tsne_features=make_TSNE(pca_features[:,:1000])










