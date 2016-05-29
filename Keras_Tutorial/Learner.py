from keras.optimizers import SGD

class SGDLearner(object):
    '''
    make own definition of 
    1. self.weight_filename
    2. build_model(self)
    3. initialize_dataset(self)
    '''
    def __init__(self,load_weight=True,just_go=False,manual_lr=None):
        self.just_go=just_go
        self.manual_lr=manual_lr
        self.initialize_dataset()
        self.model = None
        self.build_model()
        if load_weight:
            self.load_weights()
        self.compile_model()
       
    def compile_model(self,learning_rate=0.1):
        sgd=SGD(lr=learning_rate)#,momentum=0.9)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    def run(self, n_epoch=None, learning_rate=None):
        if learning_rate is None:
            n_epoch=5
            learning_rate=0.1
        self.compile_model(learning_rate)

        result=self.model.fit(self.train_x, self.train_y,
                         nb_epoch=n_epoch, batch_size=32,verbose=2,
                         validation_data=(self.valid_x, self.valid_y))
        return result.history

    def smart_run(self):
        score = self.model.evaluate(self.valid_x,self.valid_y,batch_size=20,verbose=0)
        print('\t\t<Start with Valid Accuracy : %.2f%%>'%(score[1]*100,))


        lr_candidates=[0.2,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001, 0.00005, 0.00001]
        epochs_candidates=[2,3,4,5,7,10,20,50,100]

        best_acc=0
        best_val_acc=0
        val_patience=2

        if self.manual_lr is not None:
            if self.manual_lr in lr_candidates:
                lr_candidates=lr_candidates[lr_candidates.index(self.manual_lr):]
            else:
                lr_candidates=[self.manual_lr,] * 10


        for lr in lr_candidates:
            for n_epoch in epochs_candidates:      
                print('Learning rate : %.6f, n_epoch : %i'%(lr,n_epoch))
                '''
                1. run for (lr,n_epoch).
                2. observe the result, and see if E_train flips > 0 or E_valid flips > 1.
                3. if flip occurs, abandon generation.
                '''                
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

    def is_epoch_healthy(self,result,best):
        '''
        if healty, return (best_acc,best_val_acc)
        if unhealty, return None

        '''
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
        if n_epoch in [2,3,4]:
            is_accs_sorted = all(accs[i] <= accs[i+1] for i in range(n_epoch-1))
            is_val_accs_sorted = all(val_accs[i] <= val_accs[i+1] for i in range(n_epoch-1))
            if (not is_accs_sorted) or (not is_val_accs_sorted):
                return None
            if best[0] > accs[0] or best[1] > val_accs[0]:
                return None
        
        return (last_acc, last_val_acc)
    
    def load_weights(self):
        try:
            print('Weight imported.')
            self.model.load_weights(self.weight_filename)
        except:
            print('No weight file found. Initialized randomly.')

    def save_weights(self,over_write=None):
        if over_write is None:
            self.model.save_weights(self.weight_filename)
        else:
            self.model.save_weights(self.weight_filename,overwrite=over_write)

    def get_Etest(self):
        score = self.model.evaluate(self.test_x,self.test_y,batch_size=20,verbose=1)
        e_test = (1-score[1])*100
        print('E_test : %.3f%%'%(e_test,))
        return e_test