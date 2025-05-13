
import PIL.Image
import torch
import skmodel
import os
import re
import PIL
import matplotlib.pyplot as plt
import numpy as np

#load util data
datadir = '/hd_data/sketch_eitz'
valdir = os.path.join(datadir, 'val_images')
testfile = os.path.join(datadir, 'test.txt')
mapfile = os.path.join(datadir, 'mapping.txt') 
embfile = os.path.join(datadir, 'embs', 'val_feat.npy')  
file2embs = {}
with open(testfile) as f :
    for i, line in enumerate(f) : 
        filename = line.split()[0]
        cl_id = int(line.split()[1])
        filename = re.sub('png_w256/','', filename)
        filename = re.sub('.*/','', filename)
        filename = re.sub('\.png','', filename)
        print(filename)
        file2embs[filename] = i

cl_name = {}
with open(mapfile) as f :
    for line in f : 
        cname = line.split()[0]
        cid = int(line.split()[1])        
        cl_name[cid] = cname        


emb_file = os.path.join(datadir, 'embs', 'val_feat.npy')
embs = np.load(emb_file)
model = skmodel.skMLP()
model_path = 'modelsk'
model.load_state_dict(torch.load(model_path, weights_only=True))

testfile_name = '10'
testfile_path = os.path.join(valdir, testfile_name + '.png')
image = PIL.Image.open(testfile_path)

with torch.no_grad():       
    model.eval()
    feat = embs[file2embs[testfile_name], :]
    print(feat.shape)
    logits = model(torch.unsqueeze(torch.tensor(feat), 0))
    probs = torch.nn.Softmax()(logits[0])
    cl = np.argmax(probs.numpy())
    print(cl)
    print(cl_name[cl])

##
plt.imshow(image, cmap = 'gray')
plt.text(int(image.size[1]*0.5),1, cl_name[cl] + ' -- {:.4f}'.format(probs[cl]),  ha='center', va='bottom', fontsize = 22)
plt.axis(False)
plt.show()