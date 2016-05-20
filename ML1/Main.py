import pickle, gzip, numpy
import theano
import theano.tensor as T
import LogisticRegression
import MultiLayerPerceptron
import timeit
import CNN
rng = numpy.random.RandomState(23455)


def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

def get_mnist():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
        f.close()

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    print("Mnist load complete");

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def sgd_optimization_mnist(dataset):
    feats = 784
    num_of_train = 50000
    iteration_number = 10;
    learning_rate = 0.01;
    batch_size = 600


    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


    #building models
    print('building models...')
    index = T.lscalar()
    x=T.matrix('x')
    y=T.ivector('y')
    classifier = LogisticRegression.LogisticRegression(input=x, n_in = 28*28, n_out=10)

    test_model = theano.function([index],
                                 classifier.errors(y),
                                 givens={ x:test_set_x[index*batch_size:(index+1)*batch_size],
                                          y:test_set_y[index*batch_size:(index+1)*batch_size]}
                                 )
    valid_model = theano.function([index],
                                  classifier.errors(y),
                                  givens={ x:valid_set_x[index*batch_size:(index+1)*batch_size],
                                           y:valid_set_y[index*batch_size:(index+1)*batch_size]}
                                  )

    g_W = T.grad(cost=classifier.NLL(y), wrt=classifier.W)
    g_b = T.grad(cost=classifier.NLL(y), wrt=classifier.b)

    

    train_model = theano.function([index],
                                  classifier.NLL(y),
                                  updates=[(classifier.W, classifier.W - learning_rate * g_W),
                                           (classifier.b, classifier.b - learning_rate * g_b)],
                                  givens={x:train_set_x[index*batch_size:(index+1)*batch_size],
                                          y:train_set_y[index*batch_size:(index+1)*batch_size]}
                                  )
    #model building completed.
    
    best_validation_loss = numpy.inf

    #start training model.
    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0
    n_epochs = 1000
    patience=5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience//2)
    test_score=0


    print('start training...')


    # index on epoch
    while (epoch<n_epochs) and (not done_looping):
        epoch = epoch +1

        # index on minibatch_index
        for minibatch_index in range(n_train_batches):

            #training
            minibatch_avg_cost = train_model(minibatch_index)

            iter = (epoch-1)*n_train_batches + minibatch_index

            if(iter+1)%validation_frequency == 0:
                #get Evalid. 
                validation_losses = [valid_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error%f %%'%(epoch,minibatch_index+1,n_train_batches,this_validation_loss*100))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter*patience_increase)
                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print('epoch %i, minibatch %i/%i, test error of best model %f %%'%(epoch,minibatch_index+1,n_train_batches,test_score*100))
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier,f)

            if patience <= iter:
                done_looping = True;
                break

    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f %%, with test performance %f %%'%(best_validation_loss*100, test_score*100))
    print('The code run for %d epochs, with %f epochs/sec' %(epoch, epoch/(end_time-start_time)))

def mlp_mnist(dataset):
    learning_rate = 0.01
    L1_reg = 0
    L2_reg = 0.0001
    n_epochs = 1000
    batch_size = 20
    n_hidden = 500

    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    
    rng = numpy.random.RandomState(1234)
    classifier = MultiLayerPerceptron.MLP(rng=rng, input=x, n_in=28*28, n_hidden=n_hidden, n_out=10)

    result = None
    
    cost = (classifier.NLL(y) + L1_reg*classifier.L1 + L2_reg*classifier.L2_sqr)

    print('building model...')
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size : (index+1) * batch_size],
            y: test_set_y[index * batch_size : (index+1) * batch_size]
            }
        )
    print('Test built.')
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size : (index+1) * batch_size],
            y: valid_set_y[index * batch_size : (index+1) * batch_size]
            }
        )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [(param, param - learning_rate * gparam)
               for param,gparam in zip(classifier.params,gparams)]
    print('Valid built.')
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index+1) * batch_size],
            y: train_set_y[index * batch_size : (index+1) * batch_size]
            }
        )
    print('Train built.')
    print('training model...')
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience//2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1)%validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%'%(epoch,minibatch_index+1,n_train_batches,this_validation_loss*100))

                if this_validation_loss < best_validation_loss :
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter*patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses=[test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print('epoch %i, minibatch %i/%i, test error of best model %f %%'%(epoch,minibatch_index+1,n_train_batches,test_score*100))


            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f %%, with test performance %f %%'%(best_validation_loss*100, test_score*100))
    print('The code run for %d epochs, with %f epochs/sec' %(epoch, epoch/(end_time-start_time)))

def cnn_mnist(dataset):
    learning_rate = 0.1
    n_epochs = 200
    batch_size = 500

    #nkerns : number of kernels on each layer
    nkerns = [20,10]


    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    print('Building the model...')

    layer0_input = x.reshape((batch_size, 1, 28, 28))

    layer0 = CNN.ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,1,28,28),
        filter_shape=(nkerns[0],1,5,5),
        poolsize=(2,2)
        )

    layer1 = CNN.ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],12,12),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
        )

    layer2_input = layer1.output.flatten(2)

    layer2 = MultiLayerPerceptron.HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1]*4*4,
        n_out=80,
        activation=T.tanh
        )

    layer3 = LogisticRegression.LogisticRegression(input=layer2.output, n_in=80, n_out=10)

    cost = layer3.NLL(y);

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x : test_set_x[index * batch_size : (index+1) * batch_size],
            y : test_set_y[index * batch_size : (index+1) * batch_size]
            }
        )
    print('1. Test built.')
    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x : valid_set_x[index * batch_size : (index+1) * batch_size],
            y : valid_set_y[index * batch_size : (index+1) * batch_size]
            }
        )
    print('2. Valid built.')
    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost,params)

    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]
    print('3. Derivative calculated.')
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x : train_set_x[index * batch_size : (index+1) * batch_size],
            y : train_set_y[index * batch_size : (index+1) * batch_size]
            }
        )
    print('4. Train built.')
    print('Train model...')
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience//2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

 #           if iter % 100 == 0:
 #               print('training @ iter = ',iter)
            cost_ij = train_model(minibatch_index)

            if(iter+1)%validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %.2f %%'%(epoch, minibatch_index+1, n_train_batches, this_validation_loss*100))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss*improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print('\tepoch %i, minibatch %i/%i, test error of best model %.2f %%'%(epoch, minibatch_index+1, n_train_batches, test_score*100))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete')
    print('Best validation score of %f %% obtained at iteration %i, with test performance %.2f %%'%(best_validation_loss*100, best_iter+1, test_score*100))







def main():
    dataset = get_mnist();
    #sgd_optimization_mnist(dataset)
    #mlp_mnist(dataset)
    cnn_mnist(dataset)

   



if __name__=='__main__':
    main()