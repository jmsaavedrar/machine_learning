"""
This is an example using MNIST dataset with our own customized MLP
"""

import tensorflow as tf
import numpy as np
import convnet.simple as simple
import metrics.metrics as metrics
import tensorflow.keras.datasets as datasets
    

# loading dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print ('{} {}'.format(x_train.shape, x_train.dtype))
print ('{} {}'.format(x_test.shape, x_train.dtype))
print ('{} {}'.format(y_train.shape, y_train.dtype))
print ('{} {}'.format(y_test.shape, y_train.dtype))

n_train = x_train.shape[0]
n_test = x_test.shape[0]

#reshape the images to represent  BxHxWxC
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)) 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

#converting labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train)#to_one_hot(y_train)
y_test_one_hot = tf.keras.utils.to_categorical(y_test)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)


mu = np.mean(x_train, axis = 0)

# normalize data, just centering
x_train = (x_train - mu) 
x_test = (x_test - mu)  
np.save('mean.npy', mu)
# create the model
model = simple.SimpleModel(10)
model = model.model(x_train.shape[1:])    
model.summary()

# defining optimizer
opt = tf.keras.optimizers.SGD()
 
# put all together  
model.compile(
         optimizer=opt,              
          loss='categorical_crossentropy', 
          metrics=['accuracy'])
 
# training or fitting 
model.fit(x_train, 
        y_train_one_hot, 
        batch_size=256,  
        epochs = 10,
        validation_data = (x_test, y_test_one_hot))
 
# prediction using directly the trained model
# there is also a function called -- predict -- , you can check it  
y_pred = model(x_test, training = False)
 
# computing confusion_matrix
mc = metrics.confusion_matrix(y_test, y_pred, 10)
model_file = 'emnist_model'
model.save(model_file)
print('model was saved at {}'.format(model_file))
# print mc
print(mc)
# mc as percentages
rmc = mc.astype(np.float32) / np.sum(mc, axis = 1, keepdims = True)
rmc = (rmc * 100).astype(np.int32) / 100 
print(rmc)