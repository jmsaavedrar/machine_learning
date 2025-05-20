# 1. Prepare dataset 

import PIL.Image
import torch.utils.data as data
import numpy as np
import os
import PIL

class MNIST_Dataloader(data.Dataset) :
    def __init__(self, datadir, datatype = 'train'):
        assert datatype in ['train', 'valid']
        fname = os.path.join(datadir, '{}.txt'.format(datatype))
        self.inames = []
        self.icls = []
        with open(fname, 'r+') as f :
            for line in f :
                name_cl = line.split('\t')
                self.inames.append(os.path.join(datadir, name_cl[0].strip()))
                self.icls.append(int(name_cl[1].strip()))        
        print('dataset size: {}'.format(len(self.inames)))

    def __getitem__(self, idx):                
        iname = self.inames[idx] 
        y = self.icls[idx]        
        image = PIL.Image.open(iname)    
        image = np.array(image, dtype = np.float32) / 255.0
        return image, y
    
    def __len__(self):
        return len(self.inames)