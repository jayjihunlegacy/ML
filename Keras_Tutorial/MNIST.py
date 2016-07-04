import pickle
import gzip
filename = 'E:/Datasets/MNIST/mnist.pkl.gz'

def get_mnist():
	print('Loading MNIST...',end='')
	with gzip.open(filename, 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		train_set, valid_set, test_set = u.load()
		f.close()
	print('complete')
	return train_set,valid_set,test_set