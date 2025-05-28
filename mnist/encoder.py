
import PIL.Image
import torch
import model
import os
import re
import PIL
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

mnist_model = model.mnist_conv()
model_path = 'model_mnist'
mnist_model.load_state_dict(torch.load(model_path, weights_only=True))
mnist_model.eval()

#summary(mnist_model, input_size= (1,1, 28,28))

feats = {}

def get_gap(model, input, output):            
        bn3 = output.detach()
        x = torch.nn.GELU()(bn3)
        x = torch.nn.MaxPool2d(kernel_size = 3, stride = 2) (x)    
        gap = x.mean(dim = (2,3)) # GAP
        feats['gap'] = gap


mnist_model.bn3.register_forward_hook(get_gap)

def get_feature(fimage) :      
    image = torch.Tensor(np.array(PIL.Image.open(fimage), dtype = np.float32)) / 255.0 
    image = image.view([1,1,image.size()[0], image.size()[1]])

    print(image.shape)
    with torch.no_grad() :
        logits = mnist_model(image)        
    return feats['gap'][0]

datapath = '/hd_data/MNIST-5000'
valid_file = os.path.join(datapath, 'valid.txt')
icls = []
inames = [] 
with open(valid_file) as f :
    for line in f :
        iname_icl = line.split()     
        iname = os.path.join(datapath, iname_icl[0].strip())
        icl = int(iname_icl[1].strip())
        icls.append(icl)
        inames.append(iname) 
dim = 128
# ar_feats = np.zeros((len(inames), dim), dtype = np.float32)
# for i, iname in enumerate(inames) :
#      print(iname)
#      ar_feats[i,:] = get_feature(iname)
# print(ar_feats.shape)     
# np.save('mnist_val_feat.npy', ar_feats)
#icls = np.array(icls)
#np.save('mnist_val_cls.npy', icls)
ar_feats = np.load('mnist_val_feat.npy')
icls = np.load('mnist_val_cls.npy')
# print(ar_feats.shape)     
# # UMAP VIEW
# import umap
# import seaborn as sns

# labels = range(0,10)
# color_palette = sns.color_palette( n_colors=10)
# color_map = dict(zip(labels, color_palette))
# reducer = umap.UMAP(n_components = 2, min_dist = 0.1, n_neighbors = 20)
# embedding = reducer.fit_transform(ar_feats)
# print(embedding.shape)
# for label in labels :
#     ids = np.where(icls == label)[0]    
#     x = embedding[ids, 0]
#     y = embedding[ids, 1]
#     plt.scatter(
#         x,
#         y, color = color_map[label], label = label)


# plt.gca().set_aspect('equal', 'datalim')
# plt.legend(labels)
# plt.title('UMAP projection of the MNIST dataset', fontsize=24)
# plt.show()