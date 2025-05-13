import torch
import torch.nn as nn
import torch.utils.data as data
import dataloader
import skmodel
from torch.utils.tensorboard import SummaryWriter

#1: dataset loader (defining dataset)

tr_dataset = dataloader.SKDataloader('/hd_data/sketch_eitz', datatype = 'train')
tr_dataset = data.DataLoader(tr_dataset, batch_size = 64, shuffle = True)

val_dataset = dataloader.SKDataloader('/hd_data/sketch_eitz', datatype = 'val')
val_dataset = data.DataLoader(val_dataset, batch_size = 64, shuffle = False)

#2: defining the model
model = skmodel.skMLP()

#3: defining loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

STEPS_STATS = 100 

#4. defining the training step    
def train_one_epoch(epoch_index) :
    running_loss = 0.
    running_acc = 0.
    last_loss = 0.

    for i, data in enumerate(tr_dataset):
        # every data instance is an input + label pair
        inputs, tr_labels = data
        # set your gradients to zero for every batch
        optimizer.zero_grad()
        # forward phase -> making predictions for this batch 
        outputs = model(inputs)
        # compute the loss and its gradients
        loss = loss_fn(outputs, tr_labels)
        #compute gradientes using backpropagation
        loss.backward()
        # adjust learning weights
        optimizer.step()

        # gather stats and report
        running_loss += loss.item()
        acc = torch.mean(torch.eq(torch.argmax(outputs, dim = 1), tr_labels).float())
        running_acc += acc
        if i % STEPS_STATS == (STEPS_STATS - 1) :
            last_loss = running_loss / STEPS_STATS # loss per batch
            last_acc  = running_acc / STEPS_STATS # acc per batch
            print('  batch {} loss: {} acc: {}'.format(i + 1, last_loss, last_acc))        
            running_loss = 0.
            running_acc = 0.            
    return last_loss, last_acc 


EPOCHS = 32
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # make sure gradient tracking is on, and do a pass over the data
    model.train()
    avg_loss, avg_acc = train_one_epoch(epoch + 1)

    running_vloss = 0.0
    running_vacc = 0.0
    # set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_dataset):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            vacc = torch.mean(torch.eq(torch.argmax(voutputs, dim = 1), vlabels).float())
            running_vacc += vacc  
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    avg_vacc = running_vacc / (i + 1)
    print(' TRAIN: [loss {} acc {}] VAL : [loss {} acc {}]'.format(avg_loss, avg_acc, avg_vloss, avg_vacc))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'modelsk'
        torch.save(model.state_dict(), model_path)    
    
