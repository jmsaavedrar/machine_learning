#this is a file for plotting

import matplotlib.pyplot as plt
import numpy as np
import activations as act

if __name__ == '__main__':
    fig, gx = plt.subplots(1,2)
    x = np.arange(-5,5,0.1)
    y = act.sigmoid(x)
    yd = act.sigmoid_derivative(x)
    gx[0].plot(x, y)
    gx[0].axhline(y=0.5, color ='gray', linestyle='--')
    gx[0].axvline(color ='gray', linestyle='--')
    gx[0].set_xticks(np.arange(-5,5))
    gx[0].set_yticks(np.arange(0,1,0.1))
    gx[0].set_xlabel('x')
    gx[0].set_ylabel('f(x)')
    
    gx[1].plot(x, yd)
    gx[1].axvline(color ='gray', linestyle='--')
    gx[1].set_xticks(np.arange(-5,5))
    gx[1].set_yticks(np.arange(0,0.3,0.05))
    gx[1].set_xlabel('x')
    gx[1].set_ylabel('f\'(x)')
    
    plt.show()