#This is a file containing a collection of activation functions
import numpy as np
def sigmoid(x: np.ndarray) -> np.ndarray :
    out = 1.0 / (1 + np.exp(-x))
    return out

def sigmoid_derivative(x: np.ndarray) -> np.ndarray :
    sig = sigmoid(x)
    out = sig*(1 - sig)
    return out

def softmax(x: np.ndarray, axis = 0):
    e_x = np.exp(x - np.max(x, axis, keepdims = True))
    sm  = e_x / np.sum(e_x, axis = axis, keepdims = True)
    return e_x / sm              