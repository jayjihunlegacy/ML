import theano
import theano.tensor as T
import numpy
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):

        '''
        1. filter_shape
        type : tuple of length 4
        param : (num_of_filters, num_input_feature_maps, filter_height, filter_width)

        2. image_shape
        type : tuple of length 4
        param : (batch_size, num_input_feature_maps, image_height, image_width)

        3. poolsize
        type : tuple of length 2
        param : downsampling factor
        '''

        assert image_shape[1] == filter_shape[1]

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolsize)

        W_bound = numpy.sqrt(6 / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(
                    low=-W_bound,
                    high=W_bound,
                    poolsize=filter_shape
                    ),
                dtype=theano.config.floatX
                ),
            borrow=True
            )

        b_values = numpy.zeros((filter_shape[0],), module_type=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
            )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
            )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = input
