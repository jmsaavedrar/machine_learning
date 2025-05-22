import os 
print(os.path.join(os.getcwd(), 'mnist'))
import torch.utils.data as data
import dataloader
import torch 
import model

#download dataset
#clone machine learning repo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#1: dataset loader (defining dataset)
datapath = '/hd_data/MNIST-5000'
#datapath = '/home/data/MNIST-5000'
tr_dataset = dataloader.MNIST_Dataloader(datapath, datatype = 'train')
tr_dataset = data.DataLoader(tr_dataset, batch_size = 64, shuffle = True)

val_dataset = dataloader.MNIST_Dataloader(datapath, datatype = 'valid')
val_dataset = data.DataLoader(val_dataset, batch_size = 64, shuffle = False)

#2: defining the model
mnist_model = model.mnist_conv()
if device == torch.device('cuda') :
    mnist_model.to(device)
    
# defining loss and optimize
loss_fn = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(mnist_model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.AdamW(mnist_model.parameters())


mnist_model.fit(tr_dataset, val_dataset, optimizer, loss_fn, 20)