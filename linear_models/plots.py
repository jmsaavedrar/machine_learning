#this is a file for plotting

import matplotlib.pyplot as plt
import numpy as np
import activations as act

if __name__ == '__main__':
    x = np.arange(-5,5,0.1)
    y = act.sigmoid(x)
    plt.plot(x, y)
    plt.axhline(y=0.5, color ='gray', linestyle='--')
    plt.axvline(color ='gray', linestyle='--')
    plt.xticks(np.arange(-5,5))
    plt.yticks(np.arange(0,1,0.1))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()