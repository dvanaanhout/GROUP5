from math import sqrt
import numpy as np
import pandas as pd
from scipy import stats as st


    
class KNN_D:
    def __init__(self, n_neighbors=5 , dcalc = 'euclidean' ):
        self.n_neighbors = n_neighbors
        self.dcalc = dcalc

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
            prediction = st.mode(closest_labels)[0]
            predictions.append(prediction)
        
        return predictions
    
    def calc_distance(self, to_pred_values):
        if self.dcalc == 'euclidean':
            return self.calc_distance_euclidean(to_pred_values)
        elif self.dcalc == 'manhattan':
            return self.calc_distance_manhattan(to_pred_values)
        
    def calc_distance_euclidean(self, to_pred_values):
        diff = np.abs(self.X - to_pred_values)
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        return distances

    def calc_distance_manhatten(self, to_pred_values):
        diff = np.abs(self.X - to_pred_values)
        distances = np.sum(diff, axis=1)
        return distances
    


    


