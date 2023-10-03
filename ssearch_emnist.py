import ssearch.ssearch as ss
if __name__ == '__main__' :
    ssearch = ss.SSearch('mnist_model','dense1')
    ssearch.load_catalog('data/test_emnist_images.npy', 'data/test_emnist_labels.npy')
    #ssearch.compute_features_on_catalog()    
    ssearch.ssearch_all()
    #ssearch.random_view()
    #cla = ssearch.getClass(5)    
    