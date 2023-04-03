#coefficient of determination 
import numpy as np

def r2score(y_true, y_pred):
    mu = np.mean(y_true)
    sigma = np.sum(np.square(y_true - mu))
    sigma_r = np.sum(np.square(y_true - y_pred))
    return 1 - sigma_r / sigma

def accuracy(y_true, y_pred):
    y_pred = y_pred.copy()    
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    acc = np.mean(np.equal(y_true,y_pred).astype(dtype = np.int32))
    return acc 

def multiclass_accuracy(y_true, y_pred):
    y_pred = y_pred.copy()
    y_pred = np.argmax(y_pred, axis = 1, keepdims = False)        
    acc = np.mean(np.equal(np.squeeze(y_true),np.squeeze(y_pred)).astype(dtype = np.int32))
    return acc 

def confusion_matrix(y_true, y_pred, n_classes):
    y_pred = np.argmax(y_pred, axis = 1, keepdims = True)
    cm = np.zeros((n_classes, n_classes), dtype = np.int32)
    for cl_true in np.arange(n_classes) :
        y = y_pred[y_true == cl_true]
        for cl_pred in np.arange(n_classes) :            
            cm[cl_true, cl_pred]= np.sum(y==cl_pred)
    return cm