import numpy as np
import struct
import matplotlib.pyplot as plt


def readMNISTImages(fname) :
    with open(fname,'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        print('{} {} {}'.format(size, nrows, ncols))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))        
        data = data.reshape((size, nrows, ncols))
        data = np.transpose(data, (0,2,1))
    return data

def readMNISTLabels(fname):
    with open(fname,'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size,)) # (Optional)
    return data
    

def getSampleImage(n_rows, n_cols, data):
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
    data = readMNISTImages('/mnt/hd-data/Datasets/emnist/gzip/emnist-letters-test-images-idx3-ubyte')
    labels = readMNISTLabels('/mnt/hd-data/Datasets/emnist/gzip/emnist-letters-test-labels-idx1-ubyte')
    image = getSampleImage(10, 20, data)
    #print(labels[10000])
    plt.imshow(image, cmap='gray')
    
    plt.show()
    
