import ssearch.ssearch as ss
if __name__ == '__main__' :
    ssearch = ss.SSearch('mnist_model','embedding')
    ssearch.load_catalog('data/test_emnist_images.npy', 'data/test_emnist_labels.npy')
    #ssearch.compute_features_on_catalog()    
    ssearch.ssearch_all()
    #cla = ssearch.getClass(5)    
    