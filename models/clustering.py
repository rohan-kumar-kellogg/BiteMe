from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

class TasteClusterer:
    def __init__(self, n_clusters=4):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, user_embeddings):
        self.labels = self.model.fit_predict(user_embeddings)
        score = silhouette_score(user_embeddings, self.labels)
        return score