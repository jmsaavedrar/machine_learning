#coefficient of determination 
import numpy as np

def r2score(y_true, y_pred):
    mu = np.mean(y_true)
    sigma = np.sum(np.square(y_true - mu))
    sigma_r = np.sum(np.square(y_true - y_pred))
    return 1 - sigma_r / sigma

def accuracy(y_true, y_pred):
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    acc = np.mean((y_true == y_pred).astype(dtype = np.int32))
    return acc    