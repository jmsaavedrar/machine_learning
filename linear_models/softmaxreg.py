import numpy as np
import linear_models.activations as activations
import metrics.metrics as metrics

class SoftmaxReg :
    def __init__(self, n_classes):
        self.lr = 0.01
        self.n_classes = n_classes
        self.coeff = None
        self.dim = None
        self.steps = 200
        self.min_error = 0.0001
        self.print_step = 10
    
    def fit(self, X_train:np.ndarray, y_train : np.ndarray) -> np.ndarray :
        n = X_train.shape[0]
        ones = np.ones((n,1))
        X = np.concatenate([ones, X_train], axis = 1)
        if len(y_train.shape) == 1 :
            y_train = np.expand_dims(y_train, axis = 1)        
        self.dim = X.shape[1]
        
        #weights are initialized randomly
        self.coeff = np.random.normal(loc = 0, scale = 0.1, size = (self.dim, self.n_classes))
                
        for i in range(self.steps) :                            
            y_pred = self.predict(X, add_ones = False)
            if i % self.print_step  == 0 :                
                acc = metrics.multiclass_accuracy(y_train, y_pred)
                print('it {} acc {}'.format(i,acc), flush = True)
                                                            
            #softmax = activations.softmax(np.matmul(X, self.coeff), axis = 1)
            #y_pred es softmax
            adjust = - np.matmul(np.transpose(X), y_pred)                                                
            y  = np.squeeze(y_train)                         
            for cl in np.arange(self.n_classes):
                x_mean_class = np.sum(X[np.where(y == cl)[0], :], axis = 0)                            
                adjust[:, cl] = adjust[:, cl] + x_mean_class
                
            #taking the mean    
            adjust = adjust / n
            self.coeff = self.coeff + self.lr * adjust                        
        return self.coeff
     
    
    def predict(self, X: np.ndarray, add_ones = True) -> np.ndarray :
        if add_ones :
            ones = np.ones((X.shape[0],1))
            X = np.concatenate([ones, X], axis = 1)        
        y_pred =  activations.softmax(np.matmul(X, self.coeff), axis = 1)                    
        return y_pred