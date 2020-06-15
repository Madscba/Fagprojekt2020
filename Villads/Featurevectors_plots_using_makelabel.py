from Preprossering.PreprosseringPipeline import preprossingPipeline
from Villads.PCA_TSNE_classes import scale_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Visualization.PCA_TSNE_plot import plot_pca
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


path='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/feature_vectors/'
C=preprossingPipeline(BC_datapath='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG',mac=True)

feature_vectors_1,labels_1,filenames,idx= C.make_label(max_files=10,quality=None,is_usable='No',path = path,max_windows=30)
feature_vectors_9_10,labels_9_10,filenames_1,idx_1= C.make_label(max_files=10,quality=None,is_usable='Yes',path = path,max_windows=30)
feature_vectors_lda_1,labels_1,filenames,idx_lda= C.make_label(max_files=20,quality=None,is_usable='No',path = path,max_windows=30)
feature_vectors_lda_9_10,labels_9_10,filenames_1,idx_lda_1= C.make_label(max_files=20,quality=None,is_usable='Yes',path = path,max_windows=30)


feature_vectors=np.vstack((feature_vectors_1,feature_vectors_9_10))
filenames=filenames+filenames_1
labels=labels_1+labels_9_10
scaled_feature_vectors=scale_data(feature_vectors)
pca=PCA()
tsne=TSNE()
pca.fit(scaled_feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
tsne_vectors=tsne.fit_transform(scaled_feature_vectors)
labels_new=[a[0] for a in idx+idx_1]
clf = LinearDiscriminantAnalysis()
ld = clf.fit(scaled_feature_vectors[:30*10],labels)

plot_pca(pca_vectors,labels,np.unique(labels),plot_extremes=[-200,200,-200,200])
plot_pca(tsne_vectors,labels,np.unique(labels),model='TSNE)
plot_pca(pca_vectors,labels_new,np.unique(labels_new),plot_extremes=[-200,200,-200,200])
plot_pca(tsne_vectors,labels_new,np.unique(labels_new),model='TSNE')




