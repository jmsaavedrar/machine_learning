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

def confusion_matriz(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype = np.int32)
    for cl_true in np.arange(n_classes) :
        y = y_pred[y_true == cl_true]
        for cl_pred in np.arange(n_classes) :            
            cm[cl_true, cl_pred]= np.sum(y==cl_pred)
    return cm