import os
os.environ['THEANO_FLAGS']='floatX=float32,device=cpu'
import os.path
import time
import numpy as np
import random
import math
import _pickle
from os import listdir
from functools import reduce
import theano.tensor as T
import theano
from multiprocessing import Process, Manager
class NoWordException(Exception):
	def __init__(self, word):
		self.word = word

	def __str__(self):
		return 'No such word in vocabulary : '+repr(self.word)


class WordVectors(object):
	'''
	Abstract class of any word_vector interface.
	'''

	def __init__(self):
		self.can_onmemory = True # can whole vectors be uploaded on memory?
		self.vocab = None # list of words in vocab.
		self.vectors = None # huge array of wordvectors.

	def word2vec(self, word):
		'''
		return wordvector corresponding to the word.
		raise NoWordError when no such word exists.
		'''
		raise NotImplementedError()

	def vec2word(self, vec):
		'''
		return word closest to the given vector.
		'''
		raise NotImplementedError()

	def vec2word_k(self, vec, k, exc=None):
		'''
		return k closest word to the given vector excluding given words.
		'''
		raise NotImplementedError()

	def get_vocab(self):
		'''
		return list of available words in vocab.
		'''
		raise NotImplementedError()

	def get_vectors(self):
		'''
		return huge array of wordvectors.
		'''
		raise NotImplementedError()


class GloVe(WordVectors):
	folder='E:/Datasets/Glove/'
	cachenum = 4096
	bucket_size = 512
	def __init__(self):
		super().__init__()
		self.vocab = self.get_vocab()
		self.vec_cache={}
		self.corenum = 4

		# initialize cache.
		fname = 'glove.840B.300d[0].pkl'
		with open(GloVe.folder+'buckets/'+fname, 'rb') as f:
			vecs=_pickle.load(f)
		for idx, vec in enumerate(vecs):
			self.vec_cache[idx] = vec
			
	def word2vec(self, word):
		# get index of that word.
		id = self._word2id(word)
		
		# if cache hit!
		if id in self.vec_cache.keys():
			return self.vec_cache[id]
		else:
			# get inter- and intra- bucket index.
			b_idx, idx = self._id2index(id)
		
			# read bucket-sized lines.
			fname = 'glove.840B.300d['+str(b_idx)+'].pkl'
			with open(GloVe.folder+'buckets/'+fname, 'rb') as f:
				vecs=_pickle.load(f)

			# get vector.
			vec = vecs[idx]

			# put vec into cache.
			if len(self.vec_cache) >= GloVe.cachenum:
				idx_kick = random.choice(self.vec_cache.keys())
				self.vec_cache.pop(idx_kick)
			
			self.vec_cache[id] = vec
			return vec

	def vec2word(self, vec):
		return self.vec2word_k(vec,1)

	def vec2word_k(self, vec, k, exc=None):
		if k<=0:
			return None
		if exc is None:
			exc=[]
		# search n words, and find them.
		n = k + len(exc)

		# 1. onmemory
		if self.vectors is not None:
			onetime = 200 # number of words calculated at once.
			
			num_words = len(self.vocab)
			tickle = (num_words % onetime != 0)
			iters = num_words // onetime 
			iters += 1 if tickle else 0
			vec_rep = np.tile(vec, (onetime,1))

			ds = []
			idx = 0
			if tickle:
				for iter in range(iters-1):
					d = np.linalg.norm(self.vectors[idx:idx+onetime] - vec_rep, axis=1).flatten()
					idx+=onetime
					ds.append(d)
				d = np.linalg.norm(self.vectors[idx:] - np.tile(vec, (num_words - idx,1)), axis=1).flatten()
				ds.append(d)
			else:
				for iter in range(iters):
					d = np.linalg.norm(self.vectors[idx:idx+onetime] - vec_rep, axis=1).flatten()
					idx+=onetime
					ds.append(d)
			ds = np.hstack(ds)
			ids = np.argpartition(ds, n)[:n]
			dists = ds[ids]

		# 2. ondisk
		# what to be accelerated by multiprocessing.
		else:
			bucket_num = len(listdir(GloVe.folder+'buckets/'))
			onetime = GloVe.bucket_size # number of words calculated at once.
			vec_rep = np.tile(vec,(onetime,1))
			ds = []
			for b_idx in range(bucket_num):
				fname = 'glove.840B.300d['+str(b_idx)+'].pkl'
				with open(GloVe.folder+'buckets/'+fname, 'rb') as f:
					vecs=_pickle.load(f)
				if b_idx == bucket_num-1:
					vec_rep=np.tile(vec,(vecs.shape[0],1))
				d = np.linalg.norm(vecs-vec_rep, axis=1).flatten()
				ds.append(d)

			ds = np.hstack(ds)
			ids = np.argpartition(ds, n)[:n] # n many smallest IDs. not sorted order.
			dists = ds[ids] # n many smallest distances. not sorted order.

		# kick out exc members.
		for word in exc:
			id = self._word2id(word)
			ids[ids==id] = -1
		dists = dists[ids>=0]
		ids = ids[ids>=0]

		order = np.argsort(dists)
		dists = dists[order][:k]
		words = [self._id2word(id) for id in ids[order][:k]]
		return words, dists

	def get_vocab(self):
		# if self.vocab is already set, return it.
		if self.vocab is not None:
			return self.vocab

		# if not, import from file.
		fname = 'glove.840B.300d[words].txt'
		with open(GloVe.folder+fname, 'r', encoding='latin1') as f:
			tuples=f.readlines()
		return tuples[0].split(' ')

	def get_vectors(self):
		# if cannot on memory,
		if not self.can_onmemory:
			print('This word vectors cannot be on memory')
			return None

		# if can on memory,
		if self.vectors is not None:
			return self.vectors

		# read from pickle files.
		vectors=[]
		bucket_num = len(listdir(GloVe.folder+'buckets/'))
		for b_idx in range(bucket_num):
			fname = 'glove.840B.300d['+str(b_idx)+'].pkl'
			with open(GloVe.folder+'buckets/'+fname, 'rb') as f:
				vecs=_pickle.load(f)
			vectors.append(vecs)
		vectors = np.vstack(vectors)
		self.vectors = vectors
		return vectors

	def clear_cache(self):
		self.vec_cache = {}

	def clear_vectors(self):
		self.vectors = None

	def _word2id(self, word):
		# if word does not exist in vocab, raise exception
		if word not in self.vocab:
			raise NoWordException(word)
		return self.vocab.index(word)

	def _id2word(self, id):
		return self.vocab[id]

	def _id2index(self, id):
		bucket_idx = id//GloVe.bucket_size # inter-bucket index
		idx = id%GloVe.bucket_size # intra-bucket index
		return (bucket_idx, idx)

	# only called very once.
	def _partition(self):
		index = 0
		buc_idx = 0
		fname = 'glove.840B.300d.txt'
		with open(GloVe.folder+fname, 'r', encoding='latin1') as f:
			vectors=[]
			for line in f:
				values = [float(str_value) for str_value in line[:-1].split(' ')[1:]]
				vectors.append(np.array(values))

				index+=1
				if index == GloVe.bucket_size:
					vectors = np.array(vectors,dtype=np.float32)
					fbucname = 'glove.840B.300d['+str(buc_idx)+'].pkl'		
					with open(GloVe.folder+'buckets/'+fbucname, 'wb') as f2:
						_pickle.dump(vectors, f2)

					index=0
					vectors=[]
					buc_idx+=1

			if index != 0:
				vectors = np.array(vectors)
				fbucname = 'glove.840B.300d['+str(buc_idx)+'].pkl'					
				with open(GloVe.folder+'buckets/'+fbucname, 'wb') as f2:
					_pickle.dump(vectors, f2)