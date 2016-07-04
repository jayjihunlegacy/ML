
from keras.optimizers import SGD

class SGDLearner(object):
	def __init__(self,
			  load_weight=True,
			  just_go=False,
			  manual_lr=None,
			  just_go_valid=False,
			  batch_size=32):
		#parameter initialization
		self.just_go=just_go
		self.just_go_valid=just_go_valid
		self.manual_lr=manual_lr
		self.batch_size=batch_size
		self.model = None
		self.learning_rate=None

		# get dataset
		self.initialize_dataset()

		# build model
		self.buildmodel()

		# load weight
		if load_weight:
			self.load_weights()
		
		# compile model
		self.compile_model()

	def buildmodel(self):
		print('Building model...',end='')
		self.build_model()
		print('complete')

	def compile_model(self,learning_rate=0.1):
		if learning_rate==self.learning_rate:
			return
		self.learning_rate=learning_rate
		sgd=SGD(lr=learning_rate)#,momentum=0.9)
		self.model.compile(loss='categorical_crossentropy',
					  optimizer=sgd,
					  metrics=['accuracy'])
		self.learning_rate=learning_rate

	def run(self, n_epoch=None, learning_rate=None):
		if learning_rate is None:
			n_epoch=5
			learning_rate=0.1
		self.compile_model(learning_rate)

		result=self.model.fit(self.train_x, self.train_y,
						 nb_epoch=n_epoch, batch_size=self.batch_size,verbose=2,
						 validation_data=(self.valid_x, self.valid_y))
		return result.history

	def smart_run(self):
		score = self.model.evaluate(self.valid_x,self.valid_y,batch_size=self.batch_size,verbose=0)
		print('		<Start with Valid Accuracy : %.2f%%>'%(score[1]*100,))

		min_lr=0.000001
		epochs_candidates=[2,3,4,5,7,10,20,50,100]

		best_acc=0
		best_val_acc=0
		val_patience=2

		lr=0.1 if self.manual_lr is None else self.manual_lr
				
		while lr >= min_lr:
			for n_epoch in epochs_candidates:	  
				print('Learning rate : %.6f, n_epoch : %i'%(lr,n_epoch))
				result = self.run(n_epoch,lr)

				if self.just_go:
					self.save_weights(True)
					accs=result['acc']
					val_accs=result['val_acc']
					best_acc=max(best_acc, max(accs))
					best_val_acc=max(best_val_acc, max(val_accs))
					continue

				#result is None when epoch is unhealty
				report = self.is_epoch_healthy(result,(best_acc,best_val_acc))
				if report is None:
					print('Abandon epoch!')
					self.load_weights()
					break
				else:
					(best_acc,best_val_acc) = report
					print('Best Accuracy : %.2f%%, Best Valid Accuracy : %.2f%%'%(best_acc*100,best_val_acc*100))
					self.save_weights(True)
			lr/=10			

	def is_epoch_healthy(self,result,best):
		last_acc, last_val_acc = best
		accs=result['acc']
		val_accs=result['val_acc']
		val_patience=2
		for accuracy in accs:
			#if any flip detected
			if accuracy < last_acc:
				return None
			else:
				last_acc=accuracy

		if not self.just_go_valid:
			for val_accuracy in val_accs:
				if val_accuracy < last_val_acc:
					val_patience-=1
					if val_patience==0:
						return None
				else:
					val_patience=2
					last_val_acc=val_accuracy

		#if n_epoch small, strict checking is required.
		n_epoch = len(accs)
		loose=True
		if (n_epoch in [2,3,4]) and (not loose):
			is_accs_sorted = all(accs[i] <= accs[i+1] for i in range(n_epoch-1))
			is_val_accs_sorted = all(val_accs[i] <= val_accs[i+1] for i in range(n_epoch-1))
			if self.just_go_valid:
				is_val_accs_sorted = True
			if (not is_accs_sorted) or (not is_val_accs_sorted):
				return None
			if best[0] > accs[0] or best[1] > val_accs[0]:
				return None
		
		return (last_acc, last_val_acc)
	
	def load_weights(self):
		try:
			print('Loading weights...',end='')
			self.model.load_weights(self.weight_filename)
			print('complete')			
		except:
			print('No weight file found. Initialized randomly.')

	def save_weights(self,over_write=None):
		if over_write is None:
			self.model.save_weights(self.weight_filename)
		else:
			self.model.save_weights(self.weight_filename,overwrite=over_write)
		print('Saving weights...complete')

	def get_Etest(self):
		print('Evaluating Etest...',end='')
		score = self.model.evaluate(self.test_x,self.test_y,batch_size=32,verbose=0)
		e_test = (1-score[1])*100
		print(' %.3f%%'%(e_test,))
		return e_test
