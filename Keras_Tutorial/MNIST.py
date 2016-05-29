import pickle
import gzip
filename = 'E:/Datasets/MNIST/mnist.pkl.gz'

def get_mnist():
    with gzip.open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
        f.close()
    print('MNIST loaded.')
    return train_set,valid_set,test_set
