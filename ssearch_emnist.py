import ssearch.ssearch as ss
if __name__ == '__main__' :
    ssearch = ss.SSearch('emnist_model','embedding')
    ssearch.load_catalog('test_emnist_images.npy', 'test_emnist_labels.npy')
    #ssearch.compute_features_on_catalog()    
    ssearch.ssearch_all()
    #cla = ssearch.getClass(5)    
    