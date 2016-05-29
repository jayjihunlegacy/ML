import os
os.environ['THEANO_FLAGS']='floatX=float32,device=cpu'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from Learner import SGDLearner
from MNIST import *

class MLP_MNIST(SGDLearner):
    def __init__(self,load_weight=True):
        self.weight_filename = 'MNIST_MLP_weights.txt'
        super.__init__(load_weight)
       
    def build_model(self):
        print('Building model...')
        n_input=28*28
        n_hidden=500
        n_output=10
        self.model = Sequential()
        self.model.add(Dense(output_dim=n_hidden,
                        input_dim=n_input,
                        init='uniform'))

        self.model.add(LeakyReLU(0.1))
        self.model.add(Dense(10,init='uniform'))
        self.model.add(Activation('softmax'))

    def initialize_dataset(self):
        dataset=get_mnist()
        self.train_x, self.train_y = dataset[0]
        self.valid_x, self.valid_y = dataset[1]
        self.test_x, self.test_y = dataset[2]
        self.train_y = np_utils.to_categorical(self.train_y,10)
        self.valid_y = np_utils.to_categorical(self.valid_y,10)
        self.test_y = np_utils.to_categorical(self.test_y,10)

def main():
    #dataset=get_mnist()
    classifier = MLP_MNIST()
    classifier.smart_run()
    #mnist_mlp(dataset)

if __name__=='__main__':
    print('Start Program!')
    main()