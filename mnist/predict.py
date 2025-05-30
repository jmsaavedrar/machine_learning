
import PIL.Image
import torch
import model
import os
import re
import PIL
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics

mnist_model = model.mnist_conv()
model_path = 'mnist/model_mnist_full'
mnist_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))

mnist_model.eval()
print(mnist_model)
datapath = '/home/data/MNIST-5000'
fvalid = os.path.join(datapath, 'valid.txt')
data = []
with open(fvalid) as f : 
    for fline in f :
        iname, icl = fline.split()        
        iname = os.path.join(datapath, iname)
        data.append((iname, int(icl)))
def predict(fimage) : 
    image = torch.Tensor(np.array(PIL.Image.open(fimage), dtype = np.float32)) / 255.0 
    image = image.view([1,1,image.size()[0], image.size()[1]])
    with torch.no_grad() :
        logits = mnist_model(image)
        probs = torch.nn.Softmax(dim = 0)(logits[0]).numpy()
        #print(logits)    
        cls = np.argmax(probs)
        # print(probs)
        # print('{} {}'.format(cls, probs[cls]))
    return (cls, probs[cls])

result = []
for fimage, cls in data :
    cls_p, prob_p = predict(fimage)
    result.append([cls, cls_p])

result = np.array(result)


acc_total = np.equal(result[:,0], result[:,1]).astype(np.int32).mean()  
print(acc_total)
acc_per_class = {}
for c in range(0,10) :
    idx = np.where(result[:,0] == c)[0]
    acc = np.equal(result[idx,0], result[idx,1])    
    acc = np.astype(acc, np.int32).mean()    
    acc_per_class[c] = acc
    

print(acc_per_class)

cm = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes = 10, normalize = 'true')
x = cm(torch.Tensor(result[:,1]), torch.Tensor(result[:,0]))
cm.plot(x, cmap = 'jet')

# import seaborn as sn
# import pandas as pd
# sn.heatmap(pd.DataFrame(cm.numpy(), range(10), range(10)), annot=True, annot_kws={"size": 16}) # font size

plt.bar(range(0,10), [item[1] for item in acc_per_class.items()])
plt.show()
          