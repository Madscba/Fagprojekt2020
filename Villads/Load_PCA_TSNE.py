import pickle
import numpy as np
pca = pickle.load(open('/Users/villadsstokbro/Dokumenter/DTU/GitHub/Fagprojekt2020/Fagprojekt2020/PCA:TSNE/PCA.sav', 'rb'))
TSNE = pickle.load(open('/Users/villadsstokbro/Dokumenter/DTU/GitHub/Fagprojekt2020/Fagprojekt2020/PCA:TSNE_FeatureVectors/TSNE.sav', 'rb'))
tsne_features=np.load('/Users/villadsstokbro/Dokumenter/DTU/GitHub/Fagprojekt2020/Fagprojekt2020/PCA:TSNE/tsne_features.npy')
pca_features=np.load('/Users/villadsstokbro/Dokumenter/DTU/GitHub/Fagprojekt2020/Fagprojekt2020/PCA:TSNE/pca_features.npy')