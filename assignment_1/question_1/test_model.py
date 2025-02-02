import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np


def cross_validation(x, y, split_size):
    folds = KFold(n_splits=split_size)

    for train_index, test_index in folds.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return x_train, x_test, y_train, y_test


def test_model(x_test, y_test, model):
    with torch.no_grad():
        predictions = model(x_test)
        predictions = torch.round(predictions)
        predictions_np = predictions.numpy()

        # print("Predictions:", predictions)
        
        # accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
        # print(f"Accuracy: {accuracy}")
        mean_prediction = np.mean(predictions_np)

    return mean_prediction
