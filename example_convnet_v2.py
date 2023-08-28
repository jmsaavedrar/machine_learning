"""
This is an example using MNIST dataset with our own customized MLP
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import convnet.simple as simple
import metrics.metrics as metrics
    

def map_fun(sample) :
    image = sample['image']
    image = tf.image.grayscale_to_rgb(image)
    label = sample['label']
    label = tf.one_hot(label, depth = 250)    
    return image, label
    
# loading dataset

# create the model
model = simple.SimpleModel(250)

model = model.model([256,256,3])    
model.summary()

data = tfds.load('tfds_skberlin') 

ds_train = data['train']
ds_train = ds_train.shuffle(1024).map(map_fun).batch(32)
ds_test = data['test']
ds_test = ds_test.shuffle(1024).map(map_fun).batch(32)

val_steps = len(ds_test) / 8
# defining optimizer
opt = tf.keras.optimizers.SGD(momentum = 0.9)
 
# put all together  
model.compile(
         optimizer=opt,              
          loss='categorical_crossentropy', 
          metrics=['accuracy'])
 
# training or fitting 
model.fit(ds_train, epochs = 10, 
          validation_data = ds_test, 
          validation_steps = val_steps)
 
# prediction using directly the trained model
# there is also a function called -- predict -- , you can check it  
#y_pred = model(x_test, training = False)
 
# computing confusion_matrix
#mc = metrics.confusion_matrix(y_test, y_pred, 10)
model_file = 'model'
model.save(model_file)
print('model was saved at {}'.format(model_file))
# print mc)
# mc as percentages
#rmc = mc.astype(np.float32) / np.sum(mc, axis = 1, keepdims = True)
#rmc = (rmc * 100).astype(np.int32) / 100 
print(rmc)