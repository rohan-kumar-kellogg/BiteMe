from sklearn.linear_model import LinearRegression
import numpy as np

class CompatibilityModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, user_pairs, satisfaction_scores):
        self.model.fit(user_pairs, satisfaction_scores)

    def predict(self, pair_vector):
        return self.model.predict([pair_vector])[0]