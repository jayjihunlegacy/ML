from Glove import GloVe
import time
import _pickle
import numpy as np
from multiprocessing import Process

def main():
	glove = GloVe()

	s=time.time()
	glove.get_vectors()	
	e=time.time()
	print('Vector loaded. Took %.3f seconds'%(e-s,))

	word ='good'
	n=10

	s=time.time()	
	the_vec=glove.word2vec(word)
	words, dists = glove.vec2word_k(the_vec, n, [word])
	for i in range(n):
		print('%s\t\t%.3f'%(words[i],dists[i]))
	
	#vecs=glove.get_vectors()
	#print(vecs.shape)

	e=time.time()
	print('Query complete. Took %.3f seconds'%(e-s,))
	

if __name__=='__main__':
	main()