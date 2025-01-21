from math import sqrt
import numpy as np
import pandas as pd
from scipy import stats as st

class KNN_D:
    def __init__(self, n_neighbors=5 , dcalc = 'euclidean', p=2 ):
        self.n_neighbors = n_neighbors
        self.dcalc = dcalc
        self.p = p

    def fit(self, X, y):
        self.X = X.to_numpy()  
        self.y = y.to_numpy()  

    def predict(self, to_pred):
        to_pred_np = to_pred.to_numpy()  
        predictions = []
        
        for i in range(len(to_pred_np)):
            distances = self.calc_distance(to_pred_np[i])
            closest_indices = np.argsort(distances)[:self.n_neighbors]
            closest_labels = self.y[closest_indices]
            prediction = np.mean(closest_labels)
            predictions.append(prediction)
        return predictions
    
    def calc_distance(self, to_pred_values):
        if self.dcalc == 'euclidean':
            return self.calc_distance_euclidean(to_pred_values)
        elif self.dcalc == 'manhattan':
            return self.calc_distance_manhattan(to_pred_values)
        elif self.dcalc == 'chebyshev':
            return self.calc_distance_chebyshev(to_pred_values)
        elif self.dcalc == 'minkowski':
            return self.calc_distance_minkowski(to_pred_values)
        elif self.dcalc == 'cosine':
            return self.calc_distance_cosine(to_pred_values)
    
    def calc_distance_euclidean(self, to_pred_values):
        diff = abs(self.X - to_pred_values)
        distances = np.sqrt(np.sum(diff ** 2, axis=1))  
        return distances

    def calc_distance_manhattan(self, to_pred_values):
        diff = np.abs(self.X - to_pred_values)
        distances = np.sum(diff, axis=1)
        return distances
    
    def calc_distance_chebyshev(self, to_pred_values):
        return np.max(np.abs(self.X - to_pred_values), axis=1)
    
    def calc_distance_minkowski(self, to_pred_values):
        diff = np.abs(self.X - to_pred_values)
        distances = np.power(np.sum(np.power(diff, self.p), axis=1), 1/self.p)
        return distances

    def calc_distance_cosine(self, to_pred_values):   
        dot_product = np.dot(self.X, to_pred_values)
        norm_X = np.linalg.norm(self.X, axis=1)
        norm_to_pred = np.linalg.norm(to_pred_values)
        cosine_similarity = dot_product / (norm_X * norm_to_pred)
        cosine_distance = 1 - cosine_similarity
        cosine_distance = np.nan_to_num(cosine_distance)
        return cosine_distance