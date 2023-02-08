#coefficient of determination 
import numpy as np

def r2score(y_act, y_pred):
    mu = np.mean(y_act)
    sigma = np.sum(np.square(y_act - mu))
    sigma_r = np.sum(np.square(y_act - y_pred))
    return 1 - sigma_r / sigma