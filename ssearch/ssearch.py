import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import skimage.io as io
import os

class SSearch :
    def __init__(self, model_file, layer_name):
        #loading the model
        self.sim = None
        self.sim_file = 'sim_emnist.npy'
        self.fv_file = 'fv_emnist.npy'
        self.lbl_file = 'lbl_emnist.npy'
        if os.path.exists(self.sim_file) :
            self.sim = np.load(self.sim_file)
        else :
            model = tf.keras.models.load_model(model_file)         
            model.summary()
            #defining the submodel (embedding layer)                                        
            output = model.get_layer(layer_name).output
            self.sim_model = tf.keras.Model(model.input, output)        
            self.sim_model.summary()
            self.mu = np.load('mean.npy')
                                                                
        
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
        np.save(self.fv_file, self.fv)
        np.save(self.lbl_file, self.data_labels)
        print('FV saved at {}'.format(self.fv_file))           
        print('Labels saved at {}'.format(self.lbl_file))
#
    def compute_features_on_catalog(self):
        self.compute_features(self.data_catalog)
    
    def ssearch_all(self):
        if not isinstance(self.sim, np.ndarray): 
            self.compute_features_on_catalog()
            fv = self.fv
            normfv = np.linalg.norm(fv, ord = 2, axis = 1, keepdims = True)        
            fv = fv / normfv
            self.sim = np.matmul(fv, np.transpose(fv))
            np.save(self.sim_file, self.sim)
            print('{} saved'.format(self.sim_file))
    
    def random_save_example(self, n):                                        
        ids = np.random.permutation(self.sim.shape[0])[:n];
        for id_image in ids :          
            sim_q = self.sim[id_image, :]        
            print('label {}'.format(self.data_labels[id_image]))
            sort_idx = np.argsort(-sim_q)[:10]
            print(self.data_labels[sort_idx])
            image = self.get_collage(sort_idx)
            io.imsave('result_{}.png'.format(id_image), image)
    
    def get_collage(self, sort_idx):    
        size = 28
        n = 10
        image = np.ones((size, n*size), dtype = np.uint8)*255                        
        i = 0
        for i in np.arange(n) :
            image[:, i * size:(i + 1) * size] = self.data_catalog[sort_idx[i], : , : ]
            i = i + 1   
        return image       
         