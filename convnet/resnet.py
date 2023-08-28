"""
 author: jsaavedr
 April, 2020 
 This is a general implementation of ResNet, and it optionally includes SE blocks  
 all layers are initialized as "he_normal"
"""
import tensorflow as tf

# a conv 3x3

def conv3x3(channels, stride = 1, kernel_regularizer = None, **kwargs):
    return tf.keras.layers.Conv2D(channels, (3,3), 
                                  strides = stride, 
                                  padding = 'same', 
                                  kernel_initializer = 'he_normal',
                                  kernel_regularizer = kernel_regularizer, 
                                  **kwargs)

def conv1x1(channels, stride = 1, kernel_regularizer = None, **kwargs):
    return tf.keras.layers.Conv2D(channels, 
                                  (1,1), 
                                  strides = stride, 
                                  padding = 'same', 
                                  kernel_initializer = 'he_normal',
                                  kernel_regularizer = kernel_regularizer,
                                  **kwargs)


class SEBlock(tf.keras.layers.Layer):
    """
    Squeeze and Excitation Block
    r_channels is the factor of reduction
    """
    
    def __init__(self, channels, r_channels, kernel_regularizer = None, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.gap  = tf.keras.layers.GlobalAveragePooling2D(name = 'se_gap')
        self.fc_1 = tf.keras.layers.Dense(r_channels, kernel_regularizer = kernel_regularizer, name = 'se_fc1' )
        self.bn_1 = tf.keras.layers.BatchNormalization(name = 'se_bn1')        
        self.fc_2 = tf.keras.layers.Dense(channels, kernel_regularizer = kernel_regularizer, name = 'se_fc2')    
            
    def call(self, inputs, training = True):       
        y = self.gap(inputs)
        y = tf.keras.activations.relu(self.bn_1(self.fc_1(y), training))
        scale = tf.keras.activations.sigmoid(self.fc_2(y))
        scale = tf.reshape(scale, (-1,1,1,self.channels))
        y = tf.math.multiply(inputs, scale)
        return y        
        

class ResidualBlock(tf.keras.layers.Layer):
    """
    residual block implementated in a full preactivation mode
    input bn-relu-conv1-bn-relu-conv2->y-------------------
      |                                                    |+
      ------------------(projection if necessary)-->shortcut--> y + shortcut
        
    """    
    def __init__(self, filters, stride, use_projection = False, se_factor = 0, kernel_regularizer = None,  **kwargs):        
        super(ResidualBlock, self).__init__(**kwargs)
        self.bn_0 = tf.keras.layers.BatchNormalization(name = 'bn_0')
        self.conv_1 = conv3x3(filters, stride, kernel_regularizer = kernel_regularizer, name = 'conv_1', use_bias = False)
        self.bn_1 = tf.keras.layers.BatchNormalization(name = 'bn_1', )
        self.conv_2 = conv3x3(filters, 1, kernel_regularizer = kernel_regularizer, name = 'conv_2', use_bias = False)
        self.use_projection = use_projection;
        self.projection = 0
        if self.use_projection :                            
            self.projection = conv1x1(filters, stride, kernel_regularizer = kernel_regularizer, name = 'projection', use_bias = False)
        
        self.se = 0
        self.use_se_block = False
        if se_factor > 0 :
            self.se = SEBlock(filters, filters / se_factor, kernel_regularizer = kernel_regularizer)
            self.use_se_block = True
        
    #using full pre-activation mode
    def call(self, inputs, training = True):
        x = self.bn_0(inputs, training)
        x = tf.keras.activations.relu(x)
        if self.use_projection : 
            shortcut = self.projection(x)
        else :
            shortcut = x
        x = self.conv_1(x)
        x = self.bn_1(x, training)
        x = tf.keras.activations.relu(x)
        x = self.conv_2(x)                    
        x = shortcut + x # residual function
        #in the case of using SE block
        if self.use_se_block :
            x = self.se(x)        
        return x


class BottleneckBlock(tf.keras.layers.Layer):
    """
    BottleneckBlock
    expansion rate = x4
    projectio is set  for the first layer o the first resnet block
    """
    
    def __init__(self, filters, stride, use_projection = False, se_factor = 0, kernel_regularizer = None, **kwargs):        
        super(BottleneckBlock, self).__init__(**kwargs)
        self.bn_0 = tf.keras.layers.BatchNormalization(name = 'bn_0')
        #conv_0 is the compression layer
        self.conv_1 = conv1x1(filters, stride, kernel_regularizer = kernel_regularizer, name = 'conv_0', use_bias = False)
        self.bn_1 = tf.keras.layers.BatchNormalization(name = 'bn_1')
        self.conv_2 = conv3x3(filters, 1, kernel_regularizer = kernel_regularizer, name = 'conv_1')
        self.bn_2 = tf.keras.layers.BatchNormalization(name = 'bn_2')
        self.conv_3 = conv1x1(filters * 4, 1, kernel_regularizer = kernel_regularizer, name = 'conv_2', use_bias = False)        
        self.use_projection = use_projection
        self.projection = 0
        if self.use_projection :                            
            self.projection = conv1x1(filters * 4, stride, kernel_regularizer = kernel_regularizer, name = 'projection', use_bias = False)
        self.se = 0
        self.use_se_block = False
        if se_factor > 0 :
            self.se = SEBlock(filters * 4, (filters * 4) / se_factor, kernel_regularizer = kernel_regularizer)
            self.use_se_block = True
        
    #using full pre-activation mode
    def call(self, inputs, training = True):
        #full-preactivation
        x = self.bn_0(inputs, training)
        x = tf.keras.activations.relu(x)
        if self.use_projection :
            shortcut = self.projection(x)
        else :
            shortcut = x            
        x = self.conv_1(x)
        x = self.bn_1(x, training)
        x = tf.keras.activations.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x, training)
        x = tf.keras.activations.relu(x)
        x = self.conv_3(x)                    
        x = shortcut + x # residual function
        if self.use_se_block :
            x = self.se(x)        
        return x

class ResNetBlock(tf.keras.layers.Layer):
    """
    resnet block implementation
    A resnet block contains a set of residual blocks
    Commonly, the residual block of a resnet block starts with a stride = 2, except for the first block
    The number of blocks together with the number of filters used in each block  are defined in __init__
    with_reduction: it is True if the block should apply resolution reduction at the first layer    
    """
    
    def __init__(self, filters,  block_size, kernel_regularizer = None, with_reduction = False, use_bottleneck = False, se_factor = 0, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)        
        self.filters = filters    
        self.block_size = block_size
        if use_bottleneck :
            residual_block = BottleneckBlock
        else:
            residual_block = ResidualBlock            
        #the first block is nos affected by a spatial reduction 
        stride_0 = 1
        #use_projection is True when the input should be projected to match the output dimensions
        use_projection = False
        if with_reduction:
            stride_0 = 2
            use_projection = True
        if use_bottleneck:
            use_projection = True
        self.block_collector = [residual_block(filters = filters, stride = stride_0, use_projection = use_projection, se_factor = se_factor, kernel_regularizer = kernel_regularizer,  name = 'rblock_0')]        
        for idx_block in range(1, block_size) :
            self.block_collector.append(residual_block(filters = filters, stride = 1, se_factor = se_factor, kernel_regularizer = kernel_regularizer, name = 'rblock_{}'.format(idx_block)))
                    
    def call(self, inputs, training):
        x = inputs;
        for block in self.block_collector :
            x = block(x, training)
        return x;


class ResNetBackbone(tf.keras.Model):
    
    def __init__(self, block_sizes,  filters, use_bottleneck = False, se_factor = 0, kernel_regularizer = None, **kwargs) :
        super(ResNetBackbone, self).__init__(**kwargs)
        self.conv_0 = tf.keras.layers.Conv2D(64, (7,7), strides = 2, padding = 'same', 
                                             kernel_initializer = 'he_normal',
                                             kernel_regularizer = kernel_regularizer, 
                                             name = 'conv_0', use_bias = False)
        
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2, padding = 'same')
        self.resnet_blocks = [ResNetBlock(filters = filters[0], 
                                          block_size = block_sizes[0], 
                                          kernel_regularizer = kernel_regularizer,
                                          with_reduction = False,  
                                          use_bottleneck = use_bottleneck, 
                                          se_factor = se_factor, 
                                          name = 'block_0')] 
        for idx_block in range(1, len(block_sizes)) :                     
            self.resnet_blocks.append(ResNetBlock(filters = filters[idx_block], 
                                                  block_size = block_sizes[idx_block],
                                                  kernel_regularizer = kernel_regularizer, 
                                                  with_reduction = True,  
                                                  use_bottleneck = use_bottleneck,
                                                  se_factor = se_factor,
                                                  name = 'block_{}'.format(idx_block)))
        self.bn_last= tf.keras.layers.BatchNormalization(name = 'bn_last')
            
        
    def call(self, inputs, training):
        x = inputs
        x = self.conv_0(x)
        x = self.max_pool(x)                 
        for block in self.resnet_blocks :
            x = block(x, training)      
        x = self.bn_last(x)                
        x = tf.keras.activations.relu(x)  
        return x
    
class ResNet(tf.keras.Model):
    """ 
    ResNet model 
    e.g.    
    block_sizes: it is the number of residual components for each block e.g  [2,2,2] for 3 blocks 
    filters : it is the number of channels within each block [32,64,128]
    number_of_classes: The number of classes of the underlying problem
    use_bottleneck: Is's true when bottleneck blocks are used.
    se_factor : reduction factor in  SE module, 0 if SE is not used
    """        
    
    def __init__(self, block_sizes, filters, number_of_classes, use_bottleneck = False, se_factor = 0, kernel_regularizer = None, **kwargs) :
        super(ResNet, self).__init__(**kwargs)
        self.backbone = ResNetBackbone(block_sizes, filters, use_bottleneck, se_factor, kernel_regularizer = kernel_regularizer, name = 'backbone')                            
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()                     
        self.classifier = tf.keras.layers.Dense(number_of_classes, name='classifier')
        
    def call(self, inputs, training):
        x = inputs
        x = self.backbone(x, training)    
        x = self.avg_pool(x)                
        x = tf.keras.layers.Flatten()(x)                        
        x = self.classifier(x)
        return x
    
    def model(self, input_shape):
        x = tf.keras.layers.Input(shape = input_shape)
        return tf.keras.Model(inputs = [x], outputs = self.call(x))
    
    
    