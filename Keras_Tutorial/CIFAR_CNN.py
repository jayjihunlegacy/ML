import os
#os.environ['THEANO_FLAGS']='floatX=float32,device=cpu'

from CIFAR import *
from Learner import SGDLearner
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import cifar10
import numpy as np
from theano import config
from sklearn import preprocessing
from keras.regularizers import l1l2

def warn(*args,**kwargs):
	pass
import warnings
warnings.warn = warn

class CNN_CIFAR(SGDLearner):
	def __init__(self,load_weight=True,just_go=False,manual_lr=None,just_go_valid=False):
		self.weight_filename = 'CIFAR_CNN_weights.txt'
		super().__init__(
			load_weight=load_weight,
			just_go=just_go,
			manual_lr=manual_lr,
			just_go_valid=just_go_valid
			)

	def build_model2(self):
		print('Building model...')
		self.model = Sequential()
	
		self.model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(3,32,32)))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Dropout(0.5))
		self.model.add(Convolution2D(32,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Dropout(0.5))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		

		self.model.add(Convolution2D(64,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Dropout(0.5))
		self.model.add(Convolution2D(64,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Dropout(0.5))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(Flatten())

		self.model.add(Dense(100,W_regularizer=l1l2()))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Dense(10))
		self.model.add(Activation('softmax'))

	def build_model(self):
		print('Building model...')
		self.model = Sequential()
	
		self.model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(3,32,32)))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Convolution2D(32,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Convolution2D(32,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Convolution2D(32,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Convolution2D(32,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Convolution2D(32,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Convolution2D(32,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Convolution2D(32,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(Flatten())

		self.model.add(Dense(100,W_regularizer=l1l2()))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Dense(10))
		self.model.add(Activation('softmax'))

	def build_model3(self):
		print('Building model...')
		self.model = Sequential()
	
		self.model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(3,32,32)))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Convolution2D(32,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Convolution2D(16,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Convolution2D(16,3,3,border_mode='same'))
		self.model.add(LeakyReLU(0.1))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(Flatten())

		self.model.add(Dense(100))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Dense(10))
		self.model.add(Activation('softmax'))

	def initialize_dataset(self):
		dataset=get_cifar(10)

		self.train_x, self.train_y = dataset[0]
		self.valid_x, self.valid_y = dataset[1]
		self.test_x, self.test_y = dataset[2]

		#standardizing
		self.train_x = preprocessing.scale(self.train_x)
		self.valid_x = preprocessing.scale(self.valid_x)
		self.test_x = preprocessing.scale(self.test_x)

		self.train_x = self.train_x.reshape(self.train_x.shape[0],3,32,32)
		self.valid_x = self.valid_x.reshape(self.valid_x.shape[0],3,32,32)
		self.test_x = self.test_x.reshape(self.test_x.shape[0],3,32,32)

		self.train_y = np_utils.to_categorical(self.train_y,10)
		self.valid_y = np_utils.to_categorical(self.valid_y,10)
		self.test_y = np_utils.to_categorical(self.test_y,10)

def main():
	classifier = CNN_CIFAR(load_weight=True,manual_lr=0.0001)
	#classifier.get_Etest()
	classifier.smart_run()

if __name__=='__main__':
	print('Start program!')
	main()
