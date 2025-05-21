import torch.nn as nn 
import torch 

class mnist_conv(nn.Module):
    def __init__(self):
        super().__init__()        

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 'same')
        self.bn1 = nn.BatchNorm2d(num_features = 32)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 'same')
        self.bn2 = nn.BatchNorm2d(num_features = 64)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 'same')
        self.bn3 = nn.BatchNorm2d(num_features = 128)

        self.cl = nn.Linear(in_features = 128, out_features = 10)

        
    def forward(self, x):
        #remember x = [B, C, H, W] = transpose ([B H W C] [0,3,1,2] )
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.GELU()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.GELU()(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.GELU()(x)

        x = x.mean(dim = (2,3)) # GAP
        logits = self.cl(x)

        return logits
    
    #implementing our customized fit
    def fit(self, tr_dataset, val_dataset, optimizer, loss_fn, epochs = 100, device = 'cuda') :
        STEPS_STATS = int(0.1*len(tr_dataset))
        #each epoch
        def train_one_epoch(epoch) :
            running_loss = 0.
            running_acc = 0.
            last_loss = 0.

            for i, data in enumerate(tr_dataset):
                # every data instance is an input + label pair
                inputs, tr_labels = data                                
                #inputs  = inputs.to(device)
                #tr_labels  = tr_labels.to(device)
                # set your gradients to zero for every batch
                optimizer.zero_grad()
                # forward phase -> making predictions for this batch 
                outputs = self.predict(inputs)
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
                
        best_vloss = 1_000_000.
        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch + 1))
            # make sure gradient tracking is on, and do a pass over the data
            self.train()
            avg_loss, avg_acc = train_one_epoch(epoch + 1)
            running_vloss = 0.0
            running_vacc = 0.0
            # set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.eval()

            # disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(val_dataset):
                    vinputs, vlabels = vdata                    
                    # vinputs  = vinputs.to(device)
                    # vlabels  = vlabels.to(device)
                    voutputs = self.predict(vinputs)
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
                model_path = 'model_mnist'
                torch.save(self.state_dict(), model_path)    

    #eval
    def predict(self, inputs) :
        inputs = torch.Tensor.unsqueeze(inputs, dim = 1)
        prediction = self(inputs)
        return prediction