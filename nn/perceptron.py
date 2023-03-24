"""
This class implements the Perceptron
"""
import numpy as np
import activations.activations as activations
import metrics.metrics as metrics
import losses.losses as losses
 
class Perceptron :
    def __init__(self):
        self.lr = 0.1
        self.coeff = None
        self.dim = None
        self.steps = 1000
        self.min_error = 0.0001
        self.print_steps = 1
        self.loss = 'bce'
        
    def setLoss(self, loss:str):
        if loss  in ['bce', 'mse'] :
            self.loss = loss
    
    def setSteps(self, steps:int):        
        self.steps = steps
    
    def setPrintSteps(self, psteps:int):        
        self.print_steps = psteps
    
    def setLearningRate(self, lr:float):        
        self.lr = lr
        
    def fit(self, X_train:np.ndarray, y_train : np.ndarray) -> np.ndarray :
        ones = np.ones((X_train.shape[0],1))
        X = np.concatenate([ones, X_train], axis = 1)
        
        if len(y_train.shape) == 1 :
            y_train = np.expand_dims(y_train, axis = 1)
        
        self.dim = X.shape[1]
        self.n = X.shape[0]
        self.coeff = np.random.normal(loc = 0, scale = 0.1, size = (self.dim,1))        
        
        
        for i in range(self.steps) :                            
            y_pred = self.predict(X, add_ones = False)
            
            if i % self.print_steps  == 0 :                
                acc = metrics.accuracy(y_train, y_pred.copy())
                if self.loss == 'bce' :
                    loss = losses.bce_loss(y_train, y_pred)
                else :
                    loss = losses.mse_loss(y_train, y_pred)
                print('it {} {} loss {}'.format(i,self.loss, loss), flush = True)
                print('it {} acc {}'.format(i,acc), flush = True)
                            
            diff = (y_pred - y_train)
            
            if self.loss == 'mse' :
                df = y_pred * (1.0 - y_pred)                                                                                         
                diff = diff * df                
                        
            adjust = diff * X               
            adjust = np.mean(adjust, axis = 0, keepdims = True)
            adjust = np.transpose(adjust)                    
            self.coeff = self.coeff - self.lr * adjust
                    
        return self.coeff 
             
           
    def predict(self, X: np.ndarray, add_ones = True) -> np.ndarray :
        if add_ones :
            ones = np.ones((X.shape[0],1))
            X = np.concatenate([ones, X], axis = 1)        
        y_pred =  np.matmul(X, self.coeff)
        y_pred = activations.sigmoid(y_pred)           
        return y_pred
    
if __name__ == '__main__' :    
    X = np.array([[1,0,0],[1,0,1],[1, 1,0],[1, 1,1]], dtype = np.float64)
    y = np.array([[0,0,0,1]], dtype = np.float64)
    y = np.transpose(y)    
    pesos = np.array([[0,0,0.2]], dtype =np.float32)
    for _ in range(10):
        a = np.matmul(X, np.transpose(pesos))    
        pred = activations.sigmoid(a)    
        print(a)
        print(pred)
        print(y)
        diff = (pred - y)
        print(diff)
        inc = np.matmul(np.transpose(X), diff)
        print(inc)
        pesos = pesos - np.transpose(inc)
        print(pesos)
    
    
    