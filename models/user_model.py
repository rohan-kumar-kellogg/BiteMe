import numpy as np

class UserTasteModel:
    def __init__(self):
        self.user_vectors = {}

    def update_user(self, user_id, dish_vector, rating):
        weighted_vector = dish_vector * rating
        
        if user_id not in self.user_vectors:
            self.user_vectors[user_id] = []
        
        self.user_vectors[user_id].append(weighted_vector)

    def get_user_embedding(self, user_id):
        return np.mean(self.user_vectors[user_id], axis=0)