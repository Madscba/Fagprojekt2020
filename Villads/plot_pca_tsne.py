from Villads.Load_PCA_TSNE import pca,tsne
from Preprossering.PreprosseringPipeline import preprossingPipeline
from Villads.PCA_TSNE_classes import scale_data
path='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/feature_vectors/'
C=preprossingPipeline(mac=True)
feature_vectors_1,labels_1= C.make_label(max_files=5,quality=[1],is_usable=None,make_spectograms=False,path = path)
feature_vectors_9_10,labels_9_10= C.make_label(max_files=5,quality=[9,10],is_usable=None,make_spectograms=False,path = path)
feature_vectors=np.vstack(feature_vectors_1,feature_vectors_9_10)
scaled_feature_vectors=scale_data(feature_vectors)
pca_vectors=pca.transform(scaled_feature_vectors)
tsne_vectors=tsne.transform(scaled_feature_vectors)


