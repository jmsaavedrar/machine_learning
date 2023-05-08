import ssearch.ssearch as ss
if __name__ == '__main__' :
    ssearch = ss.SSearch()
    ssearch.load_catalog('test_emnist_images.npy', 'test_emnist_labels.npy')    
    ssearch.ssearch_all()
    