
import PIL.Image
import torch
import model
import os
import re
import PIL
import matplotlib.pyplot as plt
import numpy as np

mnist_model = model.mnist_conv()
model_path = 'model_mnist'
mnist_model.load_state_dict(torch.load(model_path, weights_only=True))
mnist_model.eval()
print(mnist_model)
datapath = '/hd_data/MNIST-5000'
fimage = os.path.join(datapath, 'valid_images/digit_mnist_00001_7.png')
#fimage = os.path.join(datapath, 'valid_images/digit_mnist_00002_2.png')
#fimage = '/home/data/MNIST-5000/valid_images/digit_mnist_00011_0.png'
#fimage = '/home/data/MNIST-5000/valid_images/digit_mnist_00012_6.png'
image = torch.Tensor(np.array(PIL.Image.open(fimage), dtype = np.float32)) / 255.0 
image = image.view([1,1,image.size()[0], image.size()[1]])

print(image.shape)
with torch.no_grad() :
    logits = mnist_model(image)
    probs = torch.nn.Softmax()(logits[0])
    print(logits)
    
    cls = torch.Tensor.argmax(probs)
    print(probs)
    print('{} {}'.format(cls, probs[cls]))
