from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf


class KMeansLabelGenerator:

    def __init__(self, n_clusters, use_pca=False, n_components=128):
        self.pca = PCA(n_components=n_components, random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.use_pca = use_pca
    
    def fit(self, data):
        pca_feature = self.pca.fit_transform(data) if self.use_pca else data
        self.kmeans.fit(pca_feature)
        return self.kmeans.labels_
    