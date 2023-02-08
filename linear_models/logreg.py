import numpy as np
import linear_models.activations as activations 
class LogRegression :
    def __init__(self):
        self.lr = 0.001
        self.coeff = None
        self.dim = None
        self.steps = None
        self.min_error = 0.0001
                
    def fit(self, X_train:np.ndarray, y_train : np.ndarray) -> np.ndarray :
        ones = np.ones((X_train.shape[0],1))
        X = np.concatenate([ones, X_train], axis = 1) 
        self.dim = X_train.shape[1] 
        self.coeff = np.random.normal(loc = 0, scale = 0.1, size = (self.dim, 1))
        for _ in range(self.step) :
            y_pred = self.predict(X_train)
            adjust = (y_train - y_pred) * X
            adjust = np.mean(adjust, axis = 0)
            self.coeff = self.coef + self.lr * adjust
        return self.coeff 
             
           
    def predict(self, X: np.ndarray) -> np.ndarray :
        ones = np.ones((X.shape[0],1))
        X = np.concatenate([ones, X], axis = 1)
        y_pred =  np.matmul(X, self.coeff)
        y_pred = activations.sigmoid(y_pred)
        return y_pred