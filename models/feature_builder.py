import numpy as np

class FeatureBuilder:
    def build_dish_vector(self, image_embedding, cuisine_vector, ingredient_vector):
        # Concatenate multimodal signals
        return np.concatenate([
            image_embedding,
            cuisine_vector,
            ingredient_vector
        ])