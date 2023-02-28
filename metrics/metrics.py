#coefficient of determination 
import numpy as np

def r2score(y_true, y_pred):
    mu = np.mean(y_true)
    sigma = np.sum(np.square(y_true - mu))
    sigma_r = np.sum(np.square(y_true - y_pred))
    return 1 - sigma_r / sigma

def accuracy(y_true, y_pred):
    acc = np.mean(np.equal(y_true,y_pred).astype(dtype = np.int32))
    return acc    