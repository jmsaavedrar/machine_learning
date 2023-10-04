import ssearch.ssearch as ss
if __name__ == '__main__' :
    ssearch = ss.SSearch('mnist_model','dense1')
    ssearch.load_catalog('data/test_emnist_images.npy', 'data/test_emnist_labels.npy')
    ssearch.ssearch_all()        
    ssearch.random_save_example(10)
    #cla = ssearch.getClass(5)    
    