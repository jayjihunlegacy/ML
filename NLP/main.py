from Glove import GloVe
import time
import _pickle
import numpy as np
def main():
	glove = GloVe()
	the_vec=glove.word2vec('bad')
	s=time.time()
	n=10
	words, dists = glove.vec2word_k(the_vec, n)
	for i in range(n):
		print('%s\t\t%.3f'%(words[i],dists[i]))
	e=time.time()
	print('%.3fs'%(e-s,))
	

if __name__=='__main__':
	main()