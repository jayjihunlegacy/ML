import pickle, gzip, numpy
import theano
import theano.tensor as T
import HeatMap

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

def get_mnist():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
        f.close()


    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    print("Mnist load complete");

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y

train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = get_mnist()
