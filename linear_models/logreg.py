import numpy as np
import linear_models.activations as activations
import metrics.metrics as metrics
 
class LogRegression :
    def __init__(self):
        self.lr = 0.01
        self.coeff = None
        self.dim = None
        self.steps = 100
        self.min_error = 0.0001
        self.print_step = 10
                
    def fit(self, X_train:np.ndarray, y_train : np.ndarray) -> np.ndarray :
        ones = np.ones((X_train.shape[0],1))
        X = np.concatenate([ones, X_train], axis = 1)
        if len(y_train.shape) == 1 :
            y_train = np.expand_dims(y_train, axis = 1)
        print(y_train)    
        self.dim = X.shape[1]
        self.coeff = np.random.normal(loc = 0, scale = 0.1, size = (self.dim,1))
        
        for i in range(self.steps) :                            
            y_pred = self.predict(X, add_ones = False)
            if i % self.print_step  == 0 :                
                acc = metrics.accuracy(y_train, y_pred)
                print('it {} acc {}'.format(i,acc), flush = True)
                                                
            diff = (y_train - y_pred)    
            adjust = diff * X    
            adjust = np.mean(adjust, axis = 0, keepdims = True)
            adjust = np.transpose(adjust)            
            self.coeff = self.coeff + self.lr * adjust            
            
        return self.coeff 
             
           
    def predict(self, X: np.ndarray, add_ones = True) -> np.ndarray :
        if add_ones :
            ones = np.ones((X.shape[0],1))
            X = np.concatenate([ones, X], axis = 1)        
        y_pred =  np.matmul(X, self.coeff)
        y_pred = activations.sigmoid(y_pred)
        return y_pred
    