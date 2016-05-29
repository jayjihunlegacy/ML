import os
os.environ['THEANO_FLAGS']='floatX=float32,device=gpu'

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from MNIST import *
from Learner import SGDLearner

class CNN_MNIST(SGDLearner):
	def __init__(self,load_weight=True):
		self.weight_filename = 'C:/Users/Jihun/Desktop/Embedded/MNIST_CNN_weights.txt'
		super().__init__(load_weight)

	def build_model(self):
		print('Building model...')
		self.model = Sequential()

		self.model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(1,28,28)))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Convolution2D(32,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Convolution2D(64,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Convolution2D(64,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(Flatten())
			
		self.model.add(Dense(50))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Dense(10))
		self.model.add(Activation('softmax'))


	def initialize_dataset(self):
		dataset=get_mnist()
		self.train_x, self.train_y = dataset[0]
		self.valid_x, self.valid_y = dataset[1]
		self.test_x, self.test_y = dataset[2]

		self.train_x = self.train_x.reshape(self.train_x.shape[0],1,28,28)
		self.valid_x = self.valid_x.reshape(self.valid_x.shape[0],1,28,28)
		self.test_x = self.test_x.reshape(self.test_x.shape[0],1,28,28)

		self.train_y = np_utils.to_categorical(self.train_y,10)
		self.valid_y = np_utils.to_categorical(self.valid_y,10)
		self.test_y = np_utils.to_categorical(self.test_y,10)

def main():
	classifier = CNN_MNIST(load_weight=False)
	#classifier.get_Etest()
	classifier.smart_run()

if __name__=='__main__':
	print('Start Program!')
	main()