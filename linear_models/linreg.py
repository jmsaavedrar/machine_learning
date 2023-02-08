import numpy as np

class LinearRegression :
    def __init__(self):
        self.coeff = []
                
    def fit(self, X_train: np.ndarray, Y : np.ndarray )-> np.ndarray :
        ones = np.ones((X_train.shape[0],1))
        X = np.concatenate([ones, X_train], axis = 1)
        tX = np.transpose(X)
        invX =  np.linalg.pinv(np.matmul(tX, X))
        self.coeff = np.matmul(np.matmul(invX, tX), Y)
        return self.coeff
    
    def predict(self, X: np.ndarray):
        ones = np.ones((X.shape[0],1))
        X = np.concatenate([ones, X], axis = 1)
        y_pred =  np.matmul(X, self.coeff)
        return y_pred
        
