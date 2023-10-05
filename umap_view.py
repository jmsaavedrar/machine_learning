import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import matplotlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# x = [4,8,12,16,1,4,9,16]
# y = [1,4,9,16,4,8,12,3]
# label = [0,1,2,3,0,1,2,3]
# colors = ['red','green','blue','purple']
# 
# fig = plt.figure(figsize=(8,8))
# plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))
# plt.show()

if __name__ == '__main__' :
    fv_file = 'fv_emnist.npy'
    lbl_file = 'lbl_emnist.npy'
    reducer = umap.UMAP()
    fv_data = np.load(fv_file)
    lbl_data = np.load(lbl_file)
    n_classes = 10
    n_points_per_class = 100    
    n_labels = 26
    lbl_sample = np.random.permutation(np.arange(n_labels))[:n_classes]            
    embeddings = reducer.fit_transform(fv_data)    
    embedding_sample = np.empty(0)
    lbl_sample1 = []
    print(lbl_sample)
    for i, lbl in enumerate(lbl_sample) :
        idx = np.where(lbl_data == lbl)[0]
        fv = embeddings[idx,:]
        lbl1 = [i for _ in range(fv.shape[0])]                
        if len(embedding_sample) > 0:
            embedding_sample = np.vstack((embedding_sample,fv))
            lbl_sample1 = lbl_sample1 + lbl1
        else :
            embedding_sample = fv
            lbl_sample1 = lbl1
    
    colors = ['red', 'yellow', 'green', 'black', 'lightblue', 'lightgreen', 'blue','darkgreen','darkred','darkblue']                
    plt.scatter(embedding_sample[:, 0], embedding_sample[:, 1], c = lbl_sample1, cmap=matplotlib.colors.ListedColormap(colors))
    plt.title('UMAP projection of EMNIST', fontsize=24);
    plt.show()
    