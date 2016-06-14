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
import numpy

class CNN_MNIST2(SGDLearner):
	def __init__(self,load_weight=True,just_go=False,manual_lr=None):
		self.weight_filename = 'C:/Users/Jihun/Desktop/Embedded/MNIST_CNN_weights.txt'
		super().__init__(load_weight,just_go=just_go,manual_lr=manual_lr)

	def build_model(self):
		print('Building model...')
		self.model = Sequential()

		self.model.add(Convolution2D(5,5,5,input_shape=(1,28,28)))
		self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		self.model.add(Convolution2D(5,5,5,input_shape=(5,12,12)))
		self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		print(self.model.output_shape)
		self.model.add(Flatten())
		print(self.model.output_shape)
		self.model.add(Dense(40))
		self.model.add(Activation('tanh'))
		print(self.model.output_shape)
		self.model.add(Dense(10))
		self.model.add(Activation('softmax'))
		print(self.model.output_shape)

	def initialize_dataset(self):
		dataset=get_mnist()
		self.train_x, self.train_y = dataset[0]
		self.valid_x, self.valid_y = dataset[1]
		self.test_x = self.importX()
		self.test_y = self.importY()

		self.test_x = numpy.array(self.test_x)
		self.test_y = numpy.array(self.test_y)
		#self.test_x, self.test_y = dataset[2]
		
		self.train_x = self.train_x.reshape(self.train_x.shape[0],1,28,28)
		self.valid_x = self.valid_x.reshape(self.valid_x.shape[0],1,28,28)
		self.test_x = self.test_x.reshape(self.test_x.shape[0],1,28,28)

		self.train_y = np_utils.to_categorical(self.train_y,10)
		self.valid_y = np_utils.to_categorical(self.valid_y,10)
		self.test_y = np_utils.to_categorical(self.test_y,10)

		self.train_y=self.test_y
		self.valid_y = self.test_y

		self.train_x = self.test_x
		self.valid_x = self.test_x
		
	def importX(self):
		feat_num=784
		with open('C:/Users/Jihun/Desktop/Embedded/test_set.h','r') as f:
			tuples = f.readlines()

		tuples = tuples[1:]

		feat_sofar =0
		lis = list()
		total = list()
		for line in tuples:
			modified = line.strip().replace(',','')
			if modified=='};':
				break

			value = float(modified)

			lis.append(value)
			feat_sofar += 1

			if feat_sofar==feat_num:
				total.append(lis)
				lis=list()
				feat_sofar=0

		return total

	def importY(self):
		with open('C:/Users/Jihun/Desktop/Embedded/label.h', 'r') as f:
			tuples = f.readlines()
		
		tuples = tuples[1:]
		total = list()
		for line in tuples:
			modified = line.strip().replace(',','')
			if modified=='};':
				break
			value = int(modified)
			total.append(value)

		return total

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

def export_weights(weights):

	folder = r'C:\Users\Jihun\Documents\Visual Studio 2013\Projects\Embedded\Embedded\data/'
	filenames = ['weights_conv1.h',
			  'bias_conv1.h',
			  'weights_conv2.h',
			  'bias_conv2.h',
			  'weights_ip1.h',
			  'bias_ip1.h',
			  'weights_ip2.h',
			  'bias_ip2.h']

	for i in range(8):
		weight = weights[i]
		filename = filenames[i]
		full = folder+filename
		
		with open(full, 'r') as f:
			firstline = f.readline()
		firstline = firstline.strip()

		if i==0 or i==2:
			shape = weight.shape
			flat=[]
			for output in range(shape[0]):
				for input in range(shape[1]):
					one_kernel = weight[output][input]
					print(one_kernel)
					flat1 = list(one_kernel.flatten('C'))
					flat1.reverse()
					flat+=flat1

		elif i==4 or i==6:
			flat = list(weight.flatten('F'))
		else:
			flat = list(weight.flatten('C'))
		print(len(flat))
		'''
		print(i)
		print(flat[0:5])
		print(flat[-5:])
		print();print()
		'''

		with open(full, 'w') as f:
			f.write(firstline+'\n')
			for values in flat[:-1]:
				f.write(str(values))
				f.write(',\n')
			f.write(str(flat[-1]))
			f.write('\n};')
	print('Weights exported.')

def printing(model):
	print(weights[0])
	get_1rd_layer_output = K.function([model.layers[0].input],[model.layers[0].output])
	get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[1].output])
	X=classifier.test_x[0].reshape((1,1,28,28))
	layer_output1 = get_1rd_layer_output([X])[0]
	layer_output2 = get_3rd_layer_output([X])[0]
	layer_output=[X,layer_output1,layer_output2]
	for layer in layer_output:
		features = layer[0]
		print("NEW LAYER")
		for feature in features:
			for row in feature:
				for value in row:
					print('%.3f\t'%(value,),end='')
				print()
			print()
			print()
		print()


def main():
	classifier = CNN_MNIST2(load_weight=True,manual_lr=0.000001)
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