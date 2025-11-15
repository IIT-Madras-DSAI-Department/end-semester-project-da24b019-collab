

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import time




#                   KNN CLASSIFIER


class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        # Compute Euclidean distances
        distances = cdist(X_test, self.X_train, metric='euclidean')
        predictions = []

        for dist_row in distances:
            k_indices = np.argsort(dist_row)[:self.k]
            k_labels = self.y_train[k_indices]
            predictions.append(np.bincount(k_labels, minlength=10).argmax())

        return np.array(predictions)



#               BAGGING CLASSIFIER (ENSEMBLE)


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators=10, sample_size_ratio=0.8, random_state=42):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sample_size_ratio = sample_size_ratio
        self.random_state = random_state
        self.estimators = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        sample_size = int(self.sample_size_ratio * n_samples)

        print(f"Training Bagging Classifier with {self.n_estimators} estimators...")

        for i in range(self.n_estimators):
            # Bootstrap sampling
            bootstrap_idx = np.random.choice(n_samples, size=sample_size, replace=True)
            X_sample, y_sample = X[bootstrap_idx], y[bootstrap_idx]

            # Clone estimator
            estimator = self.base_estimator.__class__(**self.base_estimator.__dict__)
            estimator.fit(X_sample, y_sample)

            self.estimators.append(estimator)
           

    def predict(self, X_test):
        all_preds = []

        for estimator in self.estimators:
            all_preds.append(estimator.predict(X_test))

        all_preds = np.stack(all_preds, axis=1)

        final_preds = []
        for row in all_preds:
            final_preds.append(np.bincount(row, minlength=10).argmax())

        return np.array(final_preds)



#               F1 SCORE + CONFUSION MATRIX


def calculate_f1_score(y_true, y_pred):
    labels = np.unique(y_true)
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    f1_scores = []
    for i in range(len(labels)):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return np.mean(f1_scores), cm





#             SOFTMAX REGRESSION CLASSIFIER

import numpy as np

class SoftmaxRegression:
    def __init__(self, lr=0.1, epochs=50, reg=0.001):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.W = None
        self.b = None

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Initialize weights
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

        # One-hot encode labels
        y_onehot = np.eye(n_classes)[y]

        for epoch in range(self.epochs):
            scores = X.dot(self.W) + self.b
            probs = self.softmax(scores)

            # Gradient
            dW = (1/n_samples) * (X.T.dot(probs - y_onehot)) + self.reg * self.W
            db = (1/n_samples) * np.sum(probs - y_onehot, axis=0, keepdims=True)

            # Update
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % 10 == 0:
                loss = -np.sum(y_onehot * np.log(probs + 1e-9)) / n_samples


    def predict(self, X):
        scores = X.dot(self.W) + self.b
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)




