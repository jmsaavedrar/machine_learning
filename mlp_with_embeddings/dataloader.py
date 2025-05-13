import torch.utils.data as data
import numpy as np
import os

class SKDataloader(data.Dataset) :
    def __init__(self, datadir, datatype = 'train'):
        assert datatype in ['train', 'val']
        emb_file = os.path.join(datadir, 'embs', '{}_feat.npy'.format(datatype))
        cls_file = os.path.join(datadir, 'cl_{}.txt'.format(datatype))        
        self.embs =  np.load(emb_file)
        print(self.embs.shape)
        with open(cls_file) as f :
            cls = [int(cl) for cl in f]
        self.cls = np.array(cls, dtype = np.long)

    def __getitem__(self, index):
        x = self.embs[index, :] 
        y = self.cls[index]
        return x, y
    
    def __len__(self):
        return self.embs.shape[0]