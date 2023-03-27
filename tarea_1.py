import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog


def getSample(n_rows, n_cols, data):
    size = 28
    image = np.ones((n_rows*size, n_cols*size), dtype = np.uint8)*255
    n = n_rows * n_cols
    idx = np.random.randint(data.shape[0], size = n)
    
    i = 0
    for r in np.arange(n_rows) :
            for c in np.arange(n_cols) :
                image[r * size:(r + 1) * size, c * size : (c + 1) * size] = data[idx[i], : , : ]
                i = i + 1
    
    return image
    
if __name__ == '__main__' :
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    print ('{} {}'.format(x_train.shape, x_train.dtype))
    print ('{} {}'.format(x_test.shape, x_train.dtype))
    digit = x_train[10,:,:];
    print(digit.shape)
    fd, hog_image = hog(digit, orientations=8, pixels_per_cell=(7,7),
                    cells_per_block=(1, 1), visualize=True)
    print(fd)
    fig, xs = plt.subplots(1,2)
    xs[0].imshow(digit, cmap = 'gray')
    xs[1].imshow(hog_image, cmap = 'gray')
    print(fd.shape)
    # image = getSample(10,20, x_train)
    #plt.imshow(image, cmap = 'gray')
    # plt.axis('off')
    plt.show()
    
    