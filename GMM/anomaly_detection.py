from GMM.load_data_bc import *
from GMM.gmm import gaussian_mixture, plot_cluster, cv_gmm, potential_outliers
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

imgs=imgs.reshape((imgs.shape[0],-1))
scaler = StandardScaler()
scaler.fit(imgs)
scaled_feature_vector = scaler.transform(imgs)

pca = PCA()
pca.fit(scaled_feature_vector)
pca_fv=pca.transform(scaled_feature_vector)

plt.figure(figsize=(8,8))
plt.scatter(pca_fv[:,0], pca_fv[:,1],color='green')
plt.show()

cov_type = 'full'

cds, cls, covs, gmm = gaussian_mixture(pca_fv,5,covar_type=cov_type,reps=5,init_procedure= 'random')

plot_cluster(X=pca_fv,cls=cls,cds=cds,y='None',covs=covs,idx=[0,1])

cv_gmm(pca_fv,K_range=range(1,11),n_splits=10,covar_type=cov_type,reps=5,init_procedure='random')

outliers, probs = potential_outliers(cluster_model=gmm, data=pca_fv, threshold=0.01)