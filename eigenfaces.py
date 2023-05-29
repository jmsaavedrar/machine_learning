import skimage.io as io
import matplotlib.pyplot  as plt
import numpy as np
import skimage.transform as transform
"""
Dataset

Download at https://www.dropbox.com/s/g20qgk3ue48k805/yale.zip

"""
def get_sample(n_rows, n_cols, faces, size, dtype = np.uint8, bg = 255, nosort = False ):    
    image = np.ones((n_rows*size, n_cols*size), dtype = dtype)*bg
    n = n_rows * n_cols
    if nosort :
        idx = range(n)
    else: 
        idx = np.random.randint(faces.shape[0], size = n)    
    i = 0
    for r in np.arange(n_rows) :
            for c in np.arange(n_cols) :
                face = faces[idx[i], : , : ]                                                
                image[r * size:(r + 1) * size, c * size : (c + 1) * size] = face
                i = i + 1    
    return image

def read_faces(filenames, size = 64):
    faces = []
    for fname in filenames:
            face = io.imread(fname)            
            face = transform.resize(face, (size, size))
            face = (face*255).astype(np.uint8)
            faces.append(face)            
    faces = np.array(faces)            
    return faces
    
def read_data(fname):
    with open(fname) as f:
        filenames = []
        labels = []
        for line in f :
            data = line.split('\t')
            filenames.append(data[0].strip())
            labels.append(int(data[1].strip()))
        return filenames, labels
    
def recompose(y, w, mu):
    x = np.zeros(w.shape[0], dtype = np.float32) 
    for i in np.arange(len(y)):
        x += w[:,i]*y[i]
    return x + mu
                
if __name__ == '__main__' :
    list_of_faces = '/mnt/hd-data/Datasets/YALE/faces.txt'    
    filenames, labels = read_data(list_of_faces)
    size = 64
    faces = read_faces(filenames, size)
    x = np.reshape(faces, (faces.shape[0], -1))
    mu = np.mean(x, axis=0, keepdims = True)
    x = x - mu
    print(x.shape)
    print('computing cov + eigen', flush = True)    
    cov = np.cov(x, rowvar = False)
    values, vectors = np.linalg.eig(cov)
    idxs = np.argsort(-values)
    k = 20
    k1 = 60
    w = np.real(vectors[:, idxs])
    y = np.matmul(x, w[:,:k])
    y1 = np.matmul(x, w[:,:k1])
    
    print(y.shape)
    f1 = faces[10, :, :]    
    f2 = recompose(y[10,:], w, mu)
    f2 = np.reshape(f2, (size, size))
    f3 = recompose(y1[10,:], w, mu)
    f3 = np.reshape(f3, (size, size))
    fig, xs = plt.subplots(1,3)
    xs[0].imshow(f1, cmap = 'gray')
    xs[0].set_xlabel('Original')
    xs[0].set_axis_off()
    xs[1].imshow(f2, cmap = 'gray')
    xs[1].set_xlabel('k={}'.format(k))
    xs[1].set_axis_off()
    xs[2].imshow(f3, cmap = 'gray')
    xs[2].set_xlabel('k={}'.format(k1))
    xs[2].set_axis_off()
    plt.show()
#     #drawing eigenfaces    
#     eigenfaces = np.reshape(np.transpose(w), (w.shape[0], size,size))
#     image = get_sample(5, 5, eigenfaces, size, np.float32, 1.0, True)
#     plt.imshow(image, cmap = 'gray')
#     plt.axis('off')
#     plt.show()