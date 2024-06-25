import numpy as np

class KNearestNeighbor():
    def __init__(self, k):
        pass    
    def train(self, x, y):
        self.X_train = x
        self.Y_train = y

    def predict(self, X_test):
        distances = self.compute_distance(X_test)
        return self.predict_labels(distances)
    
    def compute_distance(self, X_test):
        #Naive, inefficient way
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sqrt((np.sum(X_test[i:]-self.X_train[j:]))**2)
        return distances
    
    def predict_labels(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            y_indices = np.argsort(distances[i,:])
            k_closest_classes = self.Y_train[y_indices[:self.k]].astype(int)
            y_pred[i] = np.bincount(k_closest_classes)
        return y_pred

if __name__ == '__main__':
    pass