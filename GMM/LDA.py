from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Preprossering.PreprosseringPipeline import preprossingPipeline
from Villads.PCA_TSNE_classes import scale_data
from Visualization.PCA_TSNE_plot import plot_pca
from sklearn import preprocessing
from contrastive import CPCA
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


path2=r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Spektrograms'
path=r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\feature_vectors'
C=preprossingPipeline(BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")

feature_vectors_1,labels_1,filenames,idx= C.make_label(max_files=5,quality=[1],is_usable=None,path = path, max_windows=40)
feature_vectors_9_10,labels_9_10,filenames_1,idx_1= C.make_label(max_files=5,quality=[9],is_usable=None,path = path, max_windows=40)
feature_vectors_5,labels_5,filenames_2,idx_2= C.make_label(max_files=5,quality=[5],is_usable=None,path = path, max_windows=40)

sp1, l1, f1, i1 = C.make_label(max_files=5,quality=[1],is_usable=None,path = path2, max_windows=40)
sp2, l2, f2, i2 = C.make_label(max_files=5,quality=[5],is_usable=None,path = path2, max_windows=40)
sp3, l3, f3, i3 = C.make_label(max_files=5,quality=[9],is_usable=None,path = path2, max_windows=40)

sp = np.vstack((sp1, sp2))
sp = np.vstack((sp,sp3))
f = f1+f2+f3
l = l1+l2+l3
ln =[a[0] for a in i1+i2+i3]
ssp = scale_data(sp)
pca2=PCA()
tsne2=TSNE()
clf2 = LinearDiscriminantAnalysis()
pca2.fit(ssp)
pca_v = pca2.transform(ssp)
tsne_v=tsne2.fit_transform(ssp)
clf2.fit(ssp,l)
lda_v = clf2.transform(ssp)
le2 = preprocessing.LabelEncoder()
le2.fit(ln)

plot_pca(pca_v,l,np.unique(l),model='PCA spectograms')
plot_pca(tsne_v,l,np.unique(l),model='TSNE spectograms')
plot_pca(lda_v,l,np.unique(l),model='LDA spectograms')
plot_pca(pca_v,ln,np.unique(le2.transform(ln)),model='PCA spectograms')
plot_pca(tsne_v,ln,np.unique(le2.transform(ln)),model='TSNE spectograms')
plot_pca(lda_v,ln,np.unique(le2.transform(ln)),model='LDA spectograms')


fv=np.vstack((feature_vectors_1,feature_vectors_5))
feature_vectors=np.vstack((fv,feature_vectors_9_10))
filenames=filenames+filenames_1+filenames_2
labels=labels_1+labels_9_10+labels_5
scaled_feature_vectors=scale_data(feature_vectors)
pca=PCA()
tsne=TSNE()
pca.fit(scaled_feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
tsne_vectors=tsne.fit_transform(scaled_feature_vectors)
labels_new=[a[0] for a in idx+idx_1+idx_2]
clf = LinearDiscriminantAnalysis()
ld = clf.fit(scaled_feature_vectors,labels)
lda_vectors = ld.transform(scaled_feature_vectors)
le = preprocessing.LabelEncoder()
le.fit(labels_new)

plot_pca(pca_vectors,labels,np.unique(labels),model='PCA feature vectors')
plot_pca(tsne_vectors,labels,np.unique(labels),model='TSNE feature vectors')
plot_pca(lda_vectors,labels,np.unique(labels),model='LDA feature vectors')
plot_pca(pca_vectors,labels_new,np.unique(le.transform(labels_new)),model='PCA feature vectors')
plot_pca(tsne_vectors,labels_new,np.unique(le.transform(labels_new)),model='TSNE feature vectors')
plot_pca(lda_vectors,labels_new,np.unique(le.transform(labels_new)),model='LDA feature vectors')