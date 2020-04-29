import pickle
import numpy as np


path_pca='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/PCA:TSNE/PCA.sav'
path_pca_features='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/PCA:TSNE/pca_features.npy'
path_tsne='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/PCA:TSNE/TSNE.sav'
path_tsne_features='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/PCA:TSNE/tsne_features.npy'

pca = pickle.load(open(path_pca, 'rb'))
tsne = pickle.load(open(path_tsne, 'rb'))
tsne_features=np.load(path_tsne_features)
pca_features=np.load(path_pca_features)