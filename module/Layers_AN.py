import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer

import keras.backend as K
from keras.layers.merge import _Merge
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
# from keras.utils.generic_utils import transpose_shape
from keras.legacy import interfaces
from keras.legacy.layers import AtrousConvolution1D
from keras.legacy.layers import AtrousConvolution2D
from keras.engine.base_layer import InputSpec

from module.tools import psf2otf
from  module.dct2_batch_v0 import get_dct_model, get_idct_model

# Under Construction ....
# class NoiseLevelMap_Layer(Layer):
#     def __init__(self, **kwargs):
#         super(NoiseLevelMap_Layer, self).__init__(**kwargs)

#     def call (self, inputs):
#         x, sigma = inputs
#         N = tf.shape(x)[0]
#         n = tf.shape(x)[1]
#         m = tf.shape(x)[2]
#         noise_map = tf.tile(sigma, [N, n, m])
#         # the shape of noise_map should be [N, n, m, 1]
#         return noise_map

#     def compute_output_shape(self, input_shape):
#         return input_shape[0]


# layers definition
class ConstantLayer_Upperleft(Layer):
    def __init__(self,**kwargs):
        super(ConstantLayer_Upperleft,self).__init__(**kwargs)

    def call(self, inputs):
        y,k=inputs
        y=tf.reduce_sum(y,-1)
        fft_y=tf.fft2d(tf.cast(y,tf.complex64))
        image_size=tf.shape(y)[1:3]#???????????????????????
        k = tf.expand_dims(tf.transpose(k, [1,2,0]), -1)
        otf_k = psf2otf(k, image_size)[:,0,:,:]
        upperleft=fft_y*tf.conj(otf_k)
        upperleft=tf.cast(upperleft,tf.complex64)  

        return upperleft

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class ConstantLayer_Denominator(Layer):
    def __init__(self,**kwargs):
        super(ConstantLayer_Denominator,self).__init__(**kwargs)

    def call(self, inputs):
        y,k=inputs
        image_size=tf.shape(y)[1:3]#???????????????????????
        k = tf.expand_dims(tf.transpose(k, [1,2,0]), -1)
        otf_k = psf2otf(k, image_size)[:,0,:,:]
        denominator=tf.square(tf.abs(otf_k))
        denominator=tf.cast(denominator,tf.complex64)
        return denominator

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class DeconvolutionLayer(Layer):
    modelSigmaS=0

    def __init__(self,iter,**kwargs):
        self.iter=iter
        super(DeconvolutionLayer,self).__init__(**kwargs)

    def call(self, inputs):

        y, denominator, upperleft, sigma=inputs

        sigma=sigma[0]
        lambda_=(sigma**2)/3
        rho=(lambda_*255**2)/(DeconvolutionLayer.modelSigmaS[self.iter]**2)

        y=tf.reduce_sum(y,-1)
        rho=tf.cast(rho,tf.complex64)
        z=(upperleft+rho*tf.fft2d(tf.cast(y,tf.complex64)))/(denominator+rho)
        z=tf.cast(z,tf.complex64)
        z=tf.cast(tf.ifft2d(z), tf.float32)
        z=tf.expand_dims(z,-1)

        return z

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class ReduceNoiseLayer(Layer):
    def __init__(self,**kwargs):
        super(ReduceNoiseLayer,self).__init__(**kwargs)

    def call(self, inputs):
        z,n=inputs
        return z-n

    def compute_output_shape(self, input_shape):
        return input_shape[0]

# C model ---------------------------
def Energy(x, denominator, uppperleft, regularizer_weight):
    x = tf.reduce_sum(x, -1)
    f_x = tf.fft2d(tf.cast(x, tf.complex64))
    res = denominator*f_x - uppperleft
    res = tf.cast(res, tf.complex64)
    # res = tf.to_float(tf.ifft2d(res))
    res = tf.cast(tf.ifft2d(res), tf.float32)
    res = tf.square(res)
    E_fedlity = tf.reduce_sum(res, [1,2])
    x = tf.expand_dims(x, -1)
    [gx, gy] = tf.image.image_gradients(x)
    gradient = gx + gy
    cond_non_0 = tf.not_equal(gradient, 0)
    E_L0_regularizer = regularizer_weight * tf.reduce_sum(tf.cast(cond_non_0, tf.float32),[1,2])

    E = E_fedlity + E_L0_regularizer

    return E

class Energy_Choose_Layer(Layer):
    def __init__(self, regularizer_weight=0.0001, **kwargs):
        super(Energy_Choose_Layer, self).__init__(**kwargs)
        self.regularizer_weight = regularizer_weight

    def call(self, inputs):
        x, x_old, denominator, uppperleft = inputs
        Energy_now = Energy(x, denominator, uppperleft, self.regularizer_weight)
        Energy_old = Energy(x_old, denominator, uppperleft, self.regularizer_weight)
        # signal = tf.sign(Energy_now - Energy_old)
        signal = tf.cast(tf.greater(Energy_now, Energy_old), tf.float32)

        return [(1-signal) * x + signal * x_old, Energy_now, Energy_old]
        return [x, Energy_now, Energy_old]

    def compute_output_shape(self, input_shape):
        return [input_shape[0], [None, None], [None, None]]

def Energy2(x, y, k, regularizer_weight):

    x = tf.reduce_sum(x, -1)
    image_size = tf.shape(x)[1:3]
    f_x = tf.fft2d(tf.cast(x, tf.complex64))
    k = tf.expand_dims(tf.transpose(k, [1,2,0]), -1)
    otf_k = psf2otf(k, image_size)[:,0,:,:]    
    kx = otf_k * f_x
    kx = tf.cast(kx, tf.complex64)
    # kx = tf.to_float(tf.ifft2d(kx))
    kx = tf.cast(tf.ifft2d(kx), tf.float32)
    kx = tf.expand_dims(kx, -1)
    res = kx - y
    res = tf.square(res)
    E_fedlity = tf.reduce_sum(res, [1,2])
    
    x = tf.expand_dims(x, -1)
    [gx, gy] = tf.image.image_gradients(x)
    gradient = gx + gy
    cond_non_0 = tf.not_equal(gradient, 0)
    E_L0_regularizer = regularizer_weight * tf.reduce_sum(tf.cast(cond_non_0, tf.float32),[1,2])

    E = E_fedlity + E_L0_regularizer

    return E

class Energy_Choose_Layer2(Layer):
    def __init__(self, regularizer_weight=0.00001, **kwargs):
        super(Energy_Choose_Layer2, self).__init__(**kwargs)
        self.regularizer_weight = regularizer_weight

    def call(self, inputs):
        x, x_old, y, k = inputs
        Energy_now = Energy2(x, y, k, self.regularizer_weight)
        Energy_old = Energy2(x_old, y, k, self.regularizer_weight)
        
        signal = tf.cast(tf.greater(Energy_now, Energy_old), tf.float32)

        # return [(1-signal) * x + signal * x_old, Energy_now, Energy_old]
        # return [x, Energy_now, Energy_old]
        return (1-signal) * x + signal * x_old

    def compute_output_shape(self, input_shape):
        # return [input_shape[0], [None, None], [None, None]]
        return input_shape[0]

def hard_thresholding(x, thres):
    cond1 = tf.less_equal(x,thres)
    cond2 = tf.greater_equal(x,-thres)
    # cond3 = tf.greater(a,alphav)
    # cond4 = tf.less(a,-alphav)
    step1 = tf.where(tf.logical_and(cond1,cond2), tf.zeros(tf.shape(x)),  x)
    return step1

class Proximal_Layer(Layer):
    def __init__(self, mu, thres,**kwargs):
        super(Proximal_Layer, self).__init__(**kwargs)
        self.mu = mu
        self.thres = thres

    def call(self, inputs):
        x, denominator, uppperleft = inputs
        # gradient decent
        f_x = tf.fft2d(tf.cast(x, tf.complex64))
        f_x = tf.reduce_sum(f_x, -1)
        g = denominator * f_x - uppperleft
        g = tf.cast(g, tf.complex64)
        g = tf.cast(tf.ifft2d(g), tf.float32)
        g = tf.expand_dims(g, -1)
        x = x - self.mu * g
        x  = tf.reduce_sum(x, -1)

        # hard thresholding in idct2 domain:

        dct2d = get_dct_model()
        dct_x = dct2d(x)
        dct_x = hard_thresholding(dct_x, self.thres)
        idct2d = get_idct_model()
        x = idct2d(dct_x)
        x = tf.expand_dims(x, -1)

        # hrad thresholding in gradient domain:
        # can not tuen back to the original domain ...

        # in original domain:
        # x = hard_thresholding(x, 0.001)
        # x = tf.expand_dims(x, -1)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# D ---------------------------------------------------
class DeconvolutionLayer_GDC(Layer):
    modelSigmaS=0
    def __init__(self,iter,**kwargs):
        self.iter=iter
        super(DeconvolutionLayer_GDC,self).__init__(**kwargs)

    def call(self, inputs):

        #epi=rho #------------------------
        epi=0.001*8*(self.iter+1)

        z, u, denominator, upperleft, sigma=inputs
        sigma=sigma[0]
        lambda_=(sigma**2)/3
        rho=(lambda_*255**2)/(DeconvolutionLayer_GDC.modelSigmaS[self.iter]**2)
        z=tf.reduce_sum(z,-1)
        u=tf.reduce_sum(u,-1)
        rho=tf.cast(rho,tf.complex64)
        epi=tf.cast(epi,tf.complex64)
        z=(upperleft+rho*tf.fft2d(tf.cast(z,tf.complex64))+epi*tf.fft2d(tf.cast(u,tf.complex64)))/(denominator+rho+epi)
        z=tf.cast(z,tf.complex64)
        z=tf.to_float(tf.ifft2d(z))
        z=tf.expand_dims(z,-1)
        return z

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class GradientDescentLayer(Layer):
    def __init__(self,iter,dis_prior_weight,**kwargs):
        self.t=0.06
        self.beta=dis_prior_weight
        self.epi=0.001*4*(iter+1)
        super(GradientDescentLayer,self).__init__(**kwargs)

    def call(self, inputs):
        u,x,v = inputs
        dDdu=K.gradients(v,u)[0]
        u=u-self.t*(self.epi*(u-x)+self.beta*dDdu)
        # u=tf.reduce_sum(u,0)
        return u

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha=K.random_normal((2,1,1,1))
        return (alpha*inputs[0])+((1-alpha)*inputs[1])

class Square_Layer(Layer):
    def __init__(self, **kwargs):
        super(Square_Layer, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        out = 0.5 * tf.square(x)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

class Clip_Layer(Layer):
    def __init__(self, **kwargs):
        super(Clip_Layer, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        x = tf.clip_by_value(x, 0, 0.5, name = 'clip_layer')
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class Exp_layer(Layer):
    def __init__(self, **kwargs):
        super(Exp_layer, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        x = tf.exp(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class IDE_layer(Layer):
        def __init__(self, **kwargs):
            super(IDE_layer, self).__init__(**kwargs)

        def call(self, inputs):
            x = inputs
            return x

        def compute_output_shape(self, input_shape):
            return input_shape

# Keras Layers of AN
def l2_normalization(x, eps = 1e-12):
    norm = tf.sqrt(tf.reduce_sum(x * x))
    norm = tf.maximum(norm, eps)
    norm = x / norm
    return norm

class RealSN_Conv_Layer(Layer):
    input_w = 0
    input_h = 0
    batch_size = 0
    
    def __init__(self, power_iter,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(RealSN_Conv_Layer, self).__init__(**kwargs)
        self.power_iter = power_iter
        rank = 2
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})

        # Used for real Spectral Normalization
        U_shape = (RealSN_Conv_Layer.batch_size, RealSN_Conv_Layer.input_h, RealSN_Conv_Layer.input_w, self.filters)
        self.U = tf.random_uniform(U_shape)
        # self.U = self.add_weight(shape = U_shape, name = 'U', trainable = False,
        #              constraint=None, regularizer = None, initializer = initializers.get('random_uniform'))
        self.built = True

    def call(self, inputs):

        # We now not sure the order of normalization and standard convoluton
        # Now: first normalization and then standard convolution
        # Real Spectral Normalization --------------------------------------

        for _ in range(self.power_iter):
            v = l2_normalization(K.conv2d(tf.reverse(self.U, (1,2)), 
                    tf.transpose(self.kernel,(0,1,3,2)), padding = 'same'))
            v = tf.reverse(v, (1,2))
            self.U = l2_normalization(K.conv2d(v, self.kernel, padding = 'same'))
        
        sigma = tf.reduce_sum(self.U * K.conv2d(v, self.kernel, padding = 'same'))
        weight = self.kernel /sigma
        weight = weight * tf.pow(0.4, 1.0/17.0)
        weight = tf.stop_gradient(weight)
        # calulate SVD
        self.kernel.assign(weight)

        outputs = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
     
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return [self.activation(outputs),sigma]
        return [outputs,sigma]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return [(input_shape[0],) + tuple(new_space) + (self.filters,),[None]]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

   
class Weight_D_Layer(Layer):
    def __init__(self, **kwargs):
        super(Weight_D_Layer, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        x = x*0.8
        return x

    def compute_output_shape(self, input_shape):
        return input_shape