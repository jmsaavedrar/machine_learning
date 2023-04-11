from skimage.feature import hog
import numpy as np
import skimage.io as io
import matplotlib.pyplot  as plt
import os

dir_name = '/home/vision/smb-datasets/QuickDraw-10'

def load_save_data(input_name, output_name):
    #fname_train = '/home/vision/smb-datasets/QuickDraw-10/train.txt'
    y = []
    files = []
    with open(input_name) as f:
        for x in f:
            fila = x.split('\t')            
            y.append(int(fila[1]))
            files.append(fila[0])      
     
    x = []
    for i, f in enumerate(files) :
        filename = os.path.join(dir_name, f)
        image = io.imread(filename)      
        fd = hog(image, orientations=8, pixels_per_cell=(32,32),
                        cells_per_block=(1, 1), visualize=False)
        x.append(fd)
        if i % 100 == 0:
            print(i, flush = True)
     
    x = np.array(x)
    y = np.array(y)
    np.save(output_name + '_x.pny', x)
    np.save(output_name +'_y.pny', y)    
 

load_save_data('/home/vision/smb-datasets/QuickDraw-10/train.txt', 'train')
load_save_data('/home/vision/smb-datasets/QuickDraw-10/test.txt', 'test')

#x_train = np.load('train.npy')
#print(x_train.shape)    
# print(fd)
# print(fd.shape)
# fig, xs = plt.subplots(1,2)
# xs[0].imshow(image, cmap = 'gray')
# xs[1].imshow(hog_image, cmap = 'gray')
# plt.show()
