from math import sqrt
import numpy as np
import pandas as pd


class KNN_D:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self , X , y):
        self.X = X.to_numpy()
        self.y = y

    def predict(self, to_pred):
        predictions = []
        for i in range(len(to_pred)):
            
            percentage_done = (i + 1) / len(to_pred) * 100
            print(f"Predictions done: {percentage_done:.2f}%")

            distances = self.calc_distance(to_pred.iloc[i]) 
            distances_sorted = np.argsort(distances)

            closest_indices = distances_sorted[:self.n_neighbors]  
            closest_labels = self.y.iloc[closest_indices]

            prediction = closest_labels.mode()
            predictions.append(prediction)

        return predictions

    def calc_distance(self, to_pred_values):    
        distances = []  
        to_pred_values_np = np.array(to_pred_values)
        for i in self.X:
            distance = np.sqrt(np.sum((i - to_pred_values_np) ** 2))
            distances.append(distance)
        return distances
    
