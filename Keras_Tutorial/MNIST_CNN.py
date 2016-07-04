import os
os.environ['THEANO_FLAGS']='floatX=float32,device=gpu'

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from MNIST import *
from Learner import SGDLearner
from sklearn import preprocessing
from PIL import Image
import numpy as np
from keras import backend as K
from keras.regularizers import l1l2, l1, l2
import matplotlib.pyplot as plt
from functools import reduce

class CNN_MNIST(SGDLearner):
	def __init__(self,load_weight=True,just_go=False,manual_lr=None,just_go_valid=False,batch_size=32):
		self.weight_filename = 'MNIST_CNN_weights.txt'
		super().__init__(
			load_weight=load_weight,
			just_go=just_go,
			manual_lr=manual_lr,
			just_go_valid=just_go_valid,
			batch_size=batch_size
			)

	def build_model(self):
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
			
		self.model.add(Dense(50, W_regularizer=l1(0.01)))
		self.model.add(LeakyReLU(0.2))
		self.model.add(Dense(10, W_regularizer=l1(0.01)))
		self.model.add(Activation('softmax'))


	def initialize_dataset(self):
		dataset=get_mnist()
		self.train_x, self.train_y = dataset[0]
		self.valid_x, self.valid_y = dataset[1]
		self.test_x, self.test_y = dataset[2]

		look_num=20

		#standardizing
		#self.train_x = preprocessing.scale(self.train_x)
		#self.valid_x = preprocessing.scale(self.valid_x)
		#self.test_x = preprocessing.scale(self.test_x)

		self.train_x = self.train_x.reshape(self.train_x.shape[0],1,28,28)
		self.valid_x = self.valid_x.reshape(self.valid_x.shape[0],1,28,28)
		self.test_x = self.test_x.reshape(self.test_x.shape[0],1,28,28)

		self.train_y = np_utils.to_categorical(self.train_y,10)
		self.valid_y = np_utils.to_categorical(self.valid_y,10)
		self.test_y = np_utils.to_categorical(self.test_y,10)

		self.look_x=self.train_x[:look_num]
		self.look_y=self.train_y[:look_num]

	def get_activation(self,input,layer_index):
		layers=self.model.layers
		func = K.function([layers[0].input], [layers[layer_index].output])
		return func([[input]])[0]

	def take_look(self):
		weights=self.model.get_weights()
		
		for idx,weight in enumerate(weights):
			print(idx, weight.shape)

		weight = weights[8]
		#print(weight)
		weight2=np.array(weight)
		weight2=weight2.transpose()
		nonzeros=0
		for idx,weightset in enumerate(weight2):
			print(idx, np.sum(weightset!=0))
			nonzeros+=np.sum(weightset!=0)
			print(weightset)
		#print(weight2)
		print('Total nonzeros:',nonzeros)
		whole=weight.flatten()

		#plt.hist(whole, bins=100)
		#plt.show()
		
		return



		for one in self.look_x:
			data=np.zeros((28,28,3), dtype=np.uint8)
			image=one*256
			data[:,:,0]=image
			data[:,:,1]=image
			data[:,:,2]=image
			img = Image.fromarray(data,'RGB')
			img.show()
			layer_output = self.get_activation(one, 12)
			print(layer_output)


			a=input()

	def clip(self, layer_index, factor=0.1):
		whole_weights = self.model.get_weights()
		target_weight = whole_weights[layer_index]
		if factor>=1:
			target_weight[:]=0
		elif factor<=0:
			return
		else:
			flat = target_weight.flatten()
			flat = abs(flat)

			k = int(len(flat) * factor)
			idx = np.argpartition(flat, k)
			threshold = max(flat[idx[:k]])
			target_weight[abs(target_weight)<threshold]=0

		whole_weights[layer_index] = target_weight
		self.model.set_weights(whole_weights)
		print('Weights in layer %d clipped with a factor of %f'%(layer_index, factor))



class CNN_MNIST2(SGDLearner):
	def __init__(self,load_weight=True,just_go=False,manual_lr=None,just_go_valid=False,batch_size=32):
		self.weight_filename = 'MNIST_CNN2_weights.txt'
		super().__init__(
			load_weight=load_weight,
			just_go=just_go,
			manual_lr=manual_lr,
			just_go_valid=just_go_valid,
			batch_size=batch_size
			)

	def build_model(self):
		self.model = Sequential()

		self.model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(1,28,28)))
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
			
		self.model.add(Dense(50, W_regularizer=l1(0.01)))
		self.model.add(LeakyReLU(0.2))
		self.model.add(Dense(10, W_regularizer=l1(0.01)))
		self.model.add(Activation('softmax'))


	def initialize_dataset(self):
		dataset=get_mnist()
		self.train_x, self.train_y = dataset[0]
		self.valid_x, self.valid_y = dataset[1]
		self.test_x, self.test_y = dataset[2]

		look_num=20

		#standardizing
		#self.train_x = preprocessing.scale(self.train_x)
		#self.valid_x = preprocessing.scale(self.valid_x)
		#self.test_x = preprocessing.scale(self.test_x)

		self.train_x = self.train_x.reshape(self.train_x.shape[0],1,28,28)
		self.valid_x = self.valid_x.reshape(self.valid_x.shape[0],1,28,28)
		self.test_x = self.test_x.reshape(self.test_x.shape[0],1,28,28)

		self.train_y = np_utils.to_categorical(self.train_y,10)
		self.valid_y = np_utils.to_categorical(self.valid_y,10)
		self.test_y = np_utils.to_categorical(self.test_y,10)

		self.look_x=self.train_x[:look_num]
		self.look_y=self.train_y[:look_num]

	def get_activation(self,input,layer_index):
		layers=self.model.layers
		func = K.function([layers[0].input], [layers[layer_index].output])
		return func([[input]])[0]

	def take_look(self):
		weights=self.model.get_weights()
		
		for idx,weight in enumerate(weights):
			print(idx, weight.shape)

		weight = weights[8]
		#print(weight)
		weight2=np.array(weight)
		weight2=weight2.transpose()
		nonzeros=0
		for idx,weightset in enumerate(weight2):
			print(idx, np.sum(weightset!=0))
			nonzeros+=np.sum(weightset!=0)
			
		#print(weight2)
		print('Total nonzeros:',nonzeros)
		whole=weight.flatten()

		#plt.hist(whole, bins=100)
		#plt.show()
		
		return



		for one in self.look_x:
			data=np.zeros((28,28,3), dtype=np.uint8)
			image=one*256
			data[:,:,0]=image
			data[:,:,1]=image
			data[:,:,2]=image
			img = Image.fromarray(data,'RGB')
			img.show()
			layer_output = self.get_activation(one, 12)
			print(layer_output)


			a=input()

	def take_look_FC(self):
		lays = self.model.layers
		layer_index = 12 # first FC layer
		layer_index2 = 14 # last FC layer

		neuron_num1 = 50
		neuron_num2 = 10
		n=len(self.look_x)

		self.typ_clip()

		func=K.function([lays[0].input], [lays[layer_index].output, lays[layer_index2].output])
		 
		result=func([self.look_x])
		history1, history2 = result
		
		#classify result with respect to the answer.
		answersheet = np.array([record.argmax() for record in self.look_y])
		total1=[]
		total2=[]

		for answer in range(10):
			stat1=history1[answersheet==answer]
			stat1=stat1.transpose()

			stat2=history2[answersheet==answer]
			stat2=stat2.transpose()

			total1.append(stat1)
			total2.append(stat2)

		average1 = np.zeros((10, neuron_num1))
		average2 = np.zeros((10, neuron_num2))
		for answer in range(10):
			for n_idx in range(neuron_num1):
				average1[answer][n_idx] = np.average(total1[answer][n_idx])

			for n_idx in range(neuron_num2):
				average2[answer][n_idx] = np.average(total2[answer][n_idx])

		fire_stat1=np.array(average1)
		fire_stat1=average1.transpose()

		fire_stat2=np.array(average2)
		fire_stat2=average1.transpose()
		kill_factor = 0.2
		# if neuron shows similar firing behavior over labels 0~9, that neuron is fire-dead.
		fire_dead1=abs(fire_stat1.max(1)-fire_stat1.min(1))<abs(fire_stat1.max(1))*kill_factor
	
		# if neuron have no outward weight from it, that neuron is weight-dead.
		whole_weights = self.model.get_weights()
		target_weights = whole_weights[10]
		weight_dead1=(target_weights.max(1)==target_weights.min(1))
		print(target_weights.shape)

		for n_idx in range(neuron_num1):
			if fire_dead1[n_idx]:
				continue
			print("Neuron #%i"%(n_idx,),end='\t')
			for a_idx in range(10):
				print('%.4f'%(fire_stat1[n_idx][a_idx],), end='\t')
			if fire_dead1[n_idx]:
				print('dead\t',end='')
			else:
				print('\t',end='')

			if weight_dead1[n_idx]:
				print('dead\t',end='')
			else:
				print('\t',end='')
			print()

		index=[]
		for idx, is_dead in enumerate(fire_dead1):
			if not is_dead:
				index.append(idx)
		
		return np.array(index)

	def clip(self, layer_index, factor=0.1):
		whole_weights = self.model.get_weights()
		target_weight = whole_weights[layer_index]
		if factor>=1:
			target_weight[:]=0
		elif factor<=0:
			return
		else:
			flat = target_weight.flatten()
			flat = abs(flat)

			k = int(len(flat) * factor)
			idx = np.argpartition(flat, k)
			threshold = max(flat[idx[:k]])
			target_weight[abs(target_weight)<threshold]=0

		whole_weights[layer_index] = target_weight
		self.model.set_weights(whole_weights)
		print('Weights in layer %d clipped with a factor of %f'%(layer_index, factor))

	def typ_clip(self):
		self.clip(8, 0.995)
		self.clip(10, 0.95)


class CNN_MNIST3(SGDLearner):
	def __init__(self,**kwargs):
		self.weight_filename = 'MNIST_CNN3_weights.txt'
		super().__init__(**kwargs)

	def build_model(self):
		self.model = Sequential()

		self.model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(1,28,28)))
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
			
		self.model.add(Dense(10))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Dense(10))
		self.model.add(Activation('softmax'))


	def initialize_dataset(self):
		dataset=get_mnist()
		self.train_x, self.train_y = dataset[0]
		self.valid_x, self.valid_y = dataset[1]
		self.test_x, self.test_y = dataset[2]

		look_num=50

		#standardizing
		#self.train_x = preprocessing.scale(self.train_x)
		#self.valid_x = preprocessing.scale(self.valid_x)
		#self.test_x = preprocessing.scale(self.test_x)

		self.train_x = self.train_x.reshape(self.train_x.shape[0],1,28,28)
		self.valid_x = self.valid_x.reshape(self.valid_x.shape[0],1,28,28)
		self.test_x = self.test_x.reshape(self.test_x.shape[0],1,28,28)

		self.train_y = np_utils.to_categorical(self.train_y,10)
		self.valid_y = np_utils.to_categorical(self.valid_y,10)
		self.test_y = np_utils.to_categorical(self.test_y,10)

		self.look_x=self.train_x[:look_num]
		self.look_y=self.train_y[:look_num]

	def take_look(self):
		weights=self.model.get_weights()
		
		for idx,weight in enumerate(weights):
			print(idx, weight.shape)

		weight = weights[8]
		#print(weight)
		weight2=np.array(weight)
		weight2=weight2.transpose()
		nonzeros=0
		for idx,weightset in enumerate(weight2):
			print(idx, np.sum(weightset!=0))
			nonzeros+=np.sum(weightset!=0)
			
		#print(weight2)
		print('Total nonzeros:',nonzeros)
		whole=weight.flatten()

		#plt.hist(whole, bins=100)
		#plt.show()
		
		return



		for one in self.look_x:
			data=np.zeros((28,28,3), dtype=np.uint8)
			image=one*256
			data[:,:,0]=image
			data[:,:,1]=image
			data[:,:,2]=image
			img = Image.fromarray(data,'RGB')
			img.show()
			layer_output = self.get_activation(one, 12)
			print(layer_output)


			a=input()

			
	def take_look_Conv(self):
		layers = self.model.layers
		layer_indices = [1,3,6,8]
		label_num=10
		n=len(self.look_x)
		func=K.function([layers[0].input], [layers[layer_index].output for layer_index in layer_indices])
		 
		# get the activations.
		result=func([self.look_x])

		fire_stats=[]
		# for each activation layer in Neural Network,
		for history in result:
			#classify result with respect to the answer.
			answersheet = np.array([record.argmax() for record in self.look_y])
			total=[]
			for answer in range(label_num):
				stat=history[answersheet==answer]
				stat=np.rollaxis(stat, 0, stat.ndim)
				total.append(stat)

			average = np.zeros((label_num,)+ total[0].shape[:-1])
			for label in range(label_num):
				for n_idx in range(total[0].shape[0]):
					average[label][n_idx] = np.average(total[label][n_idx], axis=total[label][n_idx].ndim-1)

			fire_stat=np.array(average)		
			fire_stats.append(fire_stat)
		

		# Now, visualize.
		for idx,fire_stat in enumerate(fire_stats):
			plt.figure(idx+1)
		
			newdim = (fire_stat.shape[0], fire_stat.shape[1]*fire_stat.shape[2])+fire_stat.shape[3:]
			concat=fire_stat.reshape(newdim)
			concat=np.rollaxis(concat,0,2)
			concat=concat.reshape((concat.shape[0], concat.shape[1]*concat.shape[2]))
			plt.matshow(concat,cmap=plt.cm.gray, fignum=0)
			plt.axis('off')

			plt.tight_layout()
		plt.show()

	def take_look_FC(self):
		lays = self.model.layers
		layer_index = 12 # first FC layer
		neuron_num1 = 50
		n=len(self.look_x)

		self.typ_clip()

		func=K.function([lays[0].input], [lays[layer_index].output])
		 
		result=func([self.look_x])
		history1 = result[0]
		
		#classify result with respect to the answer.
		answersheet = np.array([record.argmax() for record in self.look_y])
		total1=[]

		for answer in range(10):
			stat1=history1[answersheet==answer]
			stat1=stat1.transpose()
			total1.append(stat1)

		average1 = np.zeros((10, neuron_num1))
		for answer in range(10):
			for n_idx in range(neuron_num1):
				average1[answer][n_idx] = np.average(total1[answer][n_idx])

		fire_stat1=np.array(average1)
		fire_stat1=average1.transpose()

		kill_factor = 0.2
		# if neuron shows similar firing behavior over labels 0~9, that neuron is fire-dead.
		fire_dead1=abs(fire_stat1.max(1)-fire_stat1.min(1))<abs(fire_stat1.max(1))*kill_factor
	
		# if neuron have no outward weight from it, that neuron is weight-dead.
		whole_weights = self.model.get_weights()
		target_weights = whole_weights[10]
		weight_dead1=(target_weights.max(1)==target_weights.min(1))
		print(target_weights.shape)

		# get index of non-dead neurons
		index=[]
		for idx in range(neuron_num1):
			if (not fire_dead1[idx]) or (not weight_dead1[idx]):
				index.append(idx)
		
		return np.array(index)

	def clip(self, layer_index, factor=0.1):
		whole_weights = self.model.get_weights()
		target_weight = whole_weights[layer_index]
		if factor>=1:
			target_weight[:]=0
		elif factor<=0:
			return
		else:
			flat = target_weight.flatten()
			flat = abs(flat)

			k = int(len(flat) * factor)
			idx = np.argpartition(flat, k)
			threshold = max(flat[idx[:k]])
			target_weight[abs(target_weight)<threshold]=0

		whole_weights[layer_index] = target_weight
		self.model.set_weights(whole_weights)
		print('Weights in layer %d clipped with a factor of %f'%(layer_index, factor))

	def typ_clip(self):
		self.clip(8, 0.995)
		self.clip(10, 0.95)

def move(classifier, classifier2):
	classifier.get_Etest()

	indices = classifier.take_look_FC()
	whole_weights = classifier.model.get_weights()
	for weights in whole_weights:
		print(weights.shape)
	in_weights = whole_weights[8]
	in_bias = whole_weights[9]
	out_weights = whole_weights[10]

	in_weights = [inner[indices] for inner in in_weights]
	in_weights = np.array(in_weights)

	in_bias = in_bias[indices]

	out_weights = out_weights[indices]
	
	whole_weights[8] = in_weights
	whole_weights[9] = in_bias
	whole_weights[10] = out_weights

	for weights in whole_weights:
		print(weights.shape)

	classifier2.model.set_weights(whole_weights)
	
	classifier2.get_Etest()
	classifier2.save_weights(True)

def main():
	#classifier = CNN_MNIST2(load_weight=True, just_go=True, manual_lr=0.00001)
	#lay=classifier.model.layers
	#for a,idx in enumerate(lay):
	#	print(a,idx)

	#classifier.get_Etest()
	#classifier.clip(8, 0.995)
	#classifier.clip(10, 0.95)	
	#classifier.save_weights(True)
	#classifier.get_Etest()

	#classifier.take_look()
	#classifier.smart_run()
	#classifier.run(1,0.0001)
	#classifier.save_weights(True)

	#while True:
	#	classifier.typ_clip()
	#	classifier.run(5,0.0001)
	#	classifier.save_weights(True)

	classifier = CNN_MNIST3(load_weight=True, just_go=True,manual_lr=0.00001)
	#classifier.smart_run()
	classifier.take_look_Conv()

if __name__=='__main__':
	print('Start Program!')
	main()