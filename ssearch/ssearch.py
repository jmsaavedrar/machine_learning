import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class SSearch :
    def __init__(self, model_file, layer_name):
        #loading the model
        model = tf.keras.models.load_model(model_file)         
        model.summary()
        #defining the submodel (embedding layer)                                        
        output = model.get_layer(layer_name).output
        self.sim_model = tf.keras.Model(model.input, output)        
        self.sim_model.summary()
        self.mu = np.load('mean.npy')
        #print('mu {}'.format(self.mu))
        #loading data        
        
    def load_catalog(self, data_file, label_file):
        self.data_catalog = np.load(data_file)
        self.data_labels = np.load(label_file)
        print(self.data_catalog.shape)    

    def prepare_data(self, data):
        prepared_data= np.expand_dims(data, axis = -1)
        prepared_data= prepared_data - self.mu
        return prepared_data
        
    def compute_features(self, data):
        data = self.prepare_data(data)                        
        self.fv = self.sim_model.predict(data)         
        print('FV-shape {}'.format(self.fv.shape))   
        return self.fv
#
    def compute_features_on_catalog(self):
        return self.compute_features(self.data_catalog)
    
    def ssearch_all(self):
        _ = self.compute_features_on_catalog()
        fv = self.fv
        normfv = np.linalg.norm(fv, ord = 2, axis = 1, keepdims = True)        
        fv = fv / normfv
        sim = np.matmul(fv, np.transpose(fv))
        idxq = np.random.randint(self.fv.shape[0]);
        sim_q = sim[idxq, :]
        print('label {}'.format(self.data_labels[idxq]))
        sort_idx = np.argsort(-sim_q)[:10]
        print(self.data_labels[sort_idx])
        self.visualize(sort_idx)
    
    
    def visualize(self, sort_idx):    
        size = 28
        n = 10
        image = np.ones((size, n*size), dtype = np.uint8)*255                        
        i = 0
        for i in np.arange(n) :
            image[:, i * size:(i + 1) * size] = self.data_catalog[sort_idx[i], : , : ]
            i = i + 1   
        plt.axis('off')     
        plt.imshow(image)
        plt.show()       
         