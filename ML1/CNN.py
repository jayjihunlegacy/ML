import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy

rng = numpy.random.RandomState(23455)

input = T.tensor4(name='input')

w_shp = (2,3,9,9)
w_bound = numpy.sqrt(3*9*9)

W = theano.shared(
    numpy.asarray(
        rng.uniform(
            low=-1/w_bound,
            high=1/w_bound,
            size=w_shp
            ),
        dtype=input.dtype
        ),
    name='W'
    )

b_shp=(2,)
b = theano.shared(
    numpy.asarray(
        rng.uniform(low=-0.5,high=0.5,size=b_shp),
        dtype=input.dtype
        ),
    name='b'
    )

conv_out = conv2d(input, W)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
f =theano.function([input], output)

import pylab
from PIL import Image

fileName='C:/Users/Jihun/Desktop/Lena.jpg'

img = Image.open(open(fileName,'rb'))
img = numpy.asarray(img, dtype='float32')/256


img_ = img.transpose(2,0,1).reshape(1,3,512,512)

filtered_img = f(img_)

pylab.subplot(1,3,1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()

pylab.subplot(1,3,2); pylab.axis('off'); pylab.imshow(filtered_img[0,0,:,:])
pylab.subplot(1,3,3); pylab.axis('off'); pylab.imshow(filtered_img[0,1,:,:])
pylab.show()