import numpy as np


def linearRegression(X: np.ndarray, Y : np.ndarray )-> np.ndarray :
    tX = np.transpose(X)
    invX =  np.linalg.pinv(np.matmul(tX, X))
    params = np.matmul(np.matmul(invX, tX), Y)
    return params