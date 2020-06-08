from CPCA.gmm import gaussian_mixture, gauss_2d, cv_gmm, potential_outliers, plot_cluster
from CPCA.PreprosseringPipeline import preprossingPipeline
from CPCA.PCA_TSNE_classes import scale_data
#from CPCA.fun import cut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from CPCA.PCA_TSNE_plot import plot_pca
from contrastive import CPCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

path_pca_features='/Users/johan/iCloudDrive/DTU/KID/4. semester/Fagprojekt/Kode/Johannes/PCA_TSNE/pca_features.npy'
pca_features=np.load(path_pca_features)

C = preprossingPipeline(
    BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")
path=r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\feature_vectors'

fv,l,fn,w = C.make_label(quality=[2,5,9],is_usable=None,max_files=15,path = path)
df=(pd.DataFrame(w,columns=["file","window"]))
df["label"]=l
df["index"]=df.index
file = df.get("file")
file = np.asarray(file)

X_train = fv[250:,:]
y_train = file[250:]
X_test = fv[:250,:]
#y_test = file[:250]
y_test = np.copy(file)
y_test[0:85] = "1"
y_test[85:209] = "2"
y_test[209:250] = "3"

scaled_feature_vectors=scale_data(X_train)
pca = PCA()
pca_v = pca.fit(X_train)
pca2 = PCA()
pca_v2 = pca2.fit(fv)

clf = LinearDiscriminantAnalysis(n_components=2)
ld = clf.fit(X_train,l[250:])
clf2 = LinearDiscriminantAnalysis(n_components=2)
ld2 = clf2.fit(fv,l)

plot_pca(pca.transform(X_train),y_train,np.unique(y_train),model='PCA')
plot_pca(pca.transform(X_test),y_test,np.unique(y_test),model='PCA')
plot_pca(pca.transform(fv),file,np.unique(file),model='PCA')
plot_pca(pca.transform(fv),file,np.unique(file),model='PCA')

plot_pca(clf.transform(X_test),y_test,np.unique(y_test),model='LDA')