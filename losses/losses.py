import numpy as np
#mse
def mse_loss(y_train, y_pred):   
    y_train = np.squeeze(y_train)
    y_pred = np.squeeze(y_pred)
    loss = np.mean(np.square(y_train - y_pred))
    return loss

#bce
def bce_loss(y_train, y_pred):
    eps = 1e-10
    y_train = np.squeeze(y_train)
    y_pred = np.squeeze(y_pred)    
    loss = (-1) * (y_train * np.log(y_pred + eps) + (1 - y_train) * np.log(1 - y_pred + eps))
    loss = np.mean(loss)
    return loss
