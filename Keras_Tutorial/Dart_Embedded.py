import os
os.environ['THEANO_FLAGS']='floatX=float32,device=cpu'

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from MNIST import *
from Learner import SGDLearner
from keras import backend as K
import numpy as np
from functools import reduce

class CNN_DART(SGDLearner):
	def __init__(self,load_weight=True,just_go=False,manual_lr=None):
		self.weight_filename = r'C:\Users\Jihun\Documents\Visual Studio 2015\Projects\ML1\Keras_Tutorial\DART_Embedded_weights.txt'
		super().__init__(load_weight,just_go=just_go,manual_lr=manual_lr)

	def build_model(self):
		print('Building model...')
		class_num=len(self.valid_y[0])
		self.model = Sequential()

		self.model.add(Convolution2D(10,3,3,input_shape=(1,24,24)))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Convolution2D(10,3,3))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(Flatten())
		 
		self.model.add(Dense(80))
		self.model.add(Activation('relu'))

		self.model.add(Dense(class_num))
		self.model.add(Activation('softmax'))
		

	def initialize_dataset(self):
		folder=r'D:\Downloads\dart_records/'
		file='training_set.txt'
		with open(folder+file,'r') as f:
			tuples=f.readlines()
		splitted=[255-int(value) for value in tuples[0].strip().split(',')]
		train_x=np.array(splitted)
		train_x=train_x.reshape((10000,24,24))
		listized=[mat.transpose() for mat in train_x]
		self.train_x = np.array(listized)
		self.train_x=self.train_x.reshape((10000,1,24,24))
		train_total=reduce(lambda x,y:x+y, self.train_x)
		train_average=train_total/10000
		self.train_x = np.array([image-train_average for image in self.train_x])
		self.train_x=self.train_x.reshape((10000,1,24,24))

		file='test_set.txt'
		with open(folder+file,'r') as f:
			tuples=f.readlines()
		splitted=[255-int(value) for value in tuples[0].strip().split(',')]
		valid_x=np.array(splitted)
		valid_x=valid_x.reshape((2000,24,24))
		listized=[mat.transpose() for mat in valid_x]
		self.valid_x = np.array(listized)
		self.valid_x=self.valid_x.reshape((2000,1,24,24))
		self.valid_x = np.array([image-train_average for image in self.valid_x])
		self.valid_x=self.valid_x.reshape((2000,1,24,24))
		self.test_x=self.valid_x

		file='training_label.txt'
		with open(folder+file,'r') as f:
			tuples=f.readlines()
		splitted=[int(value.strip()) for value in tuples]
		train_y=np.array(splitted)
		self.train_y=np_utils.to_categorical(train_y)

		file='test_label.txt'
		with open(folder+file,'r') as f:
			tuples=f.readlines()
		splitted=[int(value.strip()) for value in tuples]
		valid_y=np.array(splitted)
		self.valid_y=np_utils.to_categorical(valid_y)

		


	def do(self):
		
		for num in range(10):
			x = self.test_x[num]
			y = self.test_y[num]
			for i in range(28):
				for j in range(28):
					print("%.3f"%(x[0][i][j],),end='\t')
				print()
			pred = self.model.predict(x.reshape((1,1,28,28)),1)
			print(pred,y)
			

def main():
	classifier = CNN_DART(load_weight=False,manual_lr=0.01)
	#weights=classifier.model.get_weights()
	#model =classifier.model
	#printing(model)
	#for parameter in weights:
	#	print("WOW~!")
	#	print(parameter)
	#for parameter in weights:
	#	print(parameter.shape)
	#export_weights(weights)
	#classifier.do()
	
	classifier.smart_run()
	#classifier.get_Etest()
	

if __name__=='__main__':
	print('Start Program!')
	main()