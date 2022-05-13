from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
import umap


class DimReducer(object):

    def __init__(self, n_components=64, method='pca'):
        self.n_components = n_components
        self.method = method
        self.init_reducer()
    
    def init_reducer(self):
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components)
        elif self.method == 'umap':
            self.reducer = umap.UMAP(n_components=self.n_components, random_state=42)
    
    def fit(self, X):
        return self.fit_transform(X)

    def fit_transform(self, X):
        return self.reducer.fit_transform(X)

class ClusterCls(object):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.classifier = KMeans(n_clusters=self.n_clusters, random_state=42)
    
    def fit(self, X):
        self.classifier.fit(X)
        return self.classifier.predict(X)
    
    def predict(self, X):
        return self.classifier.fit_predict(X)


# class KMeansLabelGenerator:

#     def __init__(self, n_clusters, use_pca=False, n_components=128):
#         self.reducer = PCA(n_components=n_components, random_state=42) if use_pca else umap.UMAP(n_components=n_components, random_state=42) 
#         self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         self.use_pca = use_pca
    
#     def fit(self, data):
#         pca_feature = self.reducer.fit_transform(data)
#         self.kmeans.fit(pca_feature)
#         return self.kmeans.labels_
    