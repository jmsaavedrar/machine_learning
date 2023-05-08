"""
A simple cnn used for training mnist.
"""
import tensorflow as tf


class SimpleModel(tf.keras.Model):
    def __init__(self, number_of_classes):
        super(SimpleModel, self).__init__()        
        self.conv_1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same',  kernel_initializer = 'he_normal', name = 'conv1')
        self.max_pool = tf.keras.layers.MaxPool2D((3,3), strides = 2, padding = 'same')
        self.relu = tf.keras.layers.ReLU();        
        self.bn_conv_1 = tf.keras.layers.BatchNormalization()
        #self.bn_conv_1 = tf.keras.layers.LayerNormalization()        
        self.conv_2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same',  kernel_initializer='he_normal', name = 'conv2')
        self.bn_conv_2 = tf.keras.layers.BatchNormalization()
        #self.bn_conv_2 = tf.keras.layers.LayerNormalization()
        self.conv_3 = tf.keras.layers.Conv2D(128, (3,3), padding = 'same', kernel_initializer='he_normal', name = 'conv3')
        self.bn_conv_3 = tf.keras.layers.BatchNormalization()     
        #self.bn_conv_3 = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(256, kernel_initializer='he_normal', name = 'dense1')
        #self.bn_fc_1 = tf.keras.layers.LayerNormalization()
        self.bn_fc_1 = tf.keras.layers.BatchNormalization(name = 'embedding' )
        self.fc2 = tf.keras.layers.Dense(number_of_classes)

    # here, connecting the modules
    def call(self, inputs):        
        #first block
        x = self.conv_1(inputs)    
        x = self.bn_conv_1(x) 
        x = self.relu(x) 
        x = self.max_pool(x)    #14 X 14
        #second block
        x = self.conv_2(x)  
        x = self.bn_conv_2(x) 
        x = self.relu(x) 
        x = self.max_pool(x)  #7X7
        #third block
        x = self.conv_3(x)  
        x = self.bn_conv_3(x)
        x = self.relu(x)  
        x = self.max_pool(x)  #4X4
        #last block        
        x = tf.keras.layers.Flatten()(x) 
        x = self.fc1(x)  
        x = self.bn_fc_1(x) 
        x = self.relu(x) 
        x = self.fc2(x) 
        x = tf.keras.activations.softmax(x)
        return x
    
    def model(self, input_shape):
        x = tf.keras.layers.Input(shape = input_shape)
        return tf.keras.Model(inputs = [x], outputs = self.call(x))