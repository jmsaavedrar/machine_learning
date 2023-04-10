import tensorflow as tf

class MLP(tf.keras.Model) :
    #defining components
    def __init__(self, layer_size, n_classes):
        super(MLP, self).__init__()         
        self.layer_list = []
        for lsize in layer_size:
            self.layer_list.append(tf.keras.layers.Dense(lsize))
        self.classifier = tf.keras.layers.Dense(n_classes)
        
    # defining architecture
    def call(self, inputs):
        x = inputs
        for mlp_layer in self.layer_list:
            x = mlp_layer(x)
            x = tf.keras.activations.sigmoid(x)
            
        x = self.classifier(x)
        x = tf.keras.activations.softmax(x)
        return x