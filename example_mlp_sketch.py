"""
This is an example using MNIST dataset with our own customized MLP
"""

import tensorflow as tf
import numpy as np
import nn.mlp as mlp
import metrics.metrics as metrics
import tensorflow.keras.datasets as datasets
    

# loading dataset
x_train = np.load('/home/jsaavedr/Research/git/courses/machine_learning/labs/train_x.pny.npy')
y_train = np.load('/home/jsaavedr/Research/git/courses/machine_learning/labs/train_y.pny.npy')
x_test = np.load('/home/jsaavedr/Research/git/courses/machine_learning/labs/test_x.pny.npy')
y_test = np.load('/home/jsaavedr/Research/git/courses/machine_learning/labs/test_y.pny.npy') 

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

n_train = x_train.shape[0]
n_test = x_test.shape[0]

#reshape the images to represent 1D arrays -- feature vectors --
  
x_train = np.reshape(x_train, (n_train, -1))
x_test = np.reshape(x_test, (n_test, -1))
 
#converting labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train)#to_one_hot(y_train)
y_test_one_hot = tf.keras.utils.to_categorical(y_test)
 
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
 
mu = np.mean(x_train, axis = 0)
 
# normalize data, just centering
x_train = (x_train - mu) 
x_test = (x_test - mu)  
 
# create the model
mlp = mlp.MLP([512, 256], 10)
input_vector = tf.keras.Input(x_train.shape)
mlp(input_vector)    
mlp.summary()
 
# defining optimizer
opt = tf.keras.optimizers.SGD()
  
# put all together  
mlp.compile(
         optimizer=opt,              
          loss='categorical_crossentropy', 
          metrics=['accuracy'])
  
# training or fitting 
mlp.fit(x_train, 
        y_train_one_hot, 
        batch_size=512,  
        epochs = 100,
        validation_data = (x_test, y_test_one_hot))
  
# prediction using directly the trained model
# there is also a function called -- predict -- , you can check it  
y_pred = mlp(x_test, training = False)
  
# computing confusion_matrix
mc = metrics.confusion_matrix(y_test, y_pred, 10)
  
# print mc
print(mc)
# mc as percentages
rmc = mc.astype(np.float32) / np.sum(mc, axis = 1, keepdims = True)
rmc = (rmc * 100).astype(np.int32) / 100 
print(rmc)