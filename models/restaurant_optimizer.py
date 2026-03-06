import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RestaurantOptimizer:
    def __init__(self, restaurant_embeddings):
        self.restaurant_embeddings = restaurant_embeddings

    def find_best_restaurant(self, userA, userB):
        joint_vector = (userA + userB) / 2
        
        similarities = cosine_similarity(
            [joint_vector],
            self.restaurant_embeddings
        )[0]
        
        best_index = np.argmax(similarities)
        return best_index