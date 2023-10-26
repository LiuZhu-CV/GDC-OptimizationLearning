# Keras (Tensorflow backend) dct2d transform, 
# @Dutmedia Lab, Heyi 2019/4/28

import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.engine.topology import Layer
import numpy as np
import math

# Attention: The DCT coefficients matrix W depends on the shape of batch data only
# So we can calculate W outside cycle to save time 
# just deal with 2D data (grey scale image)


# This is matlab version fft operation
def matlab_fft(y):
    y_t = tf.transpose(y, [0,2,1])
    f_y = tf.fft(tf.cast(y_t, tf.complex64))
    f_y = tf.transpose(f_y, [0,2,1])
    return f_y

def matlab_ifft(y):
    y_t = tf.transpose(y, [0,2,1])
    f_y = tf.ifft(tf.cast(y_t, tf.complex64))
    f_y = tf.transpose(f_y, [0,2,1])
    return f_y   


class transpose_layer(Layer):
    def __init__(self, **kwargs):
        super(transpose_layer, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        x = tf.transpose(x, [0,2,1])
        # x = tf.expand_dims(x, 0)
        return x

    def compute_output_shape(self, input_shape):
        # h = input_shape[1]
        # w = input_shape[2]
        return input_shape


def dct_odd_case(x):
    # x.shape = [N, H, W]
    N = tf.shape(x)[0]
    n = tf.shape(x)[1]
    m = tf.shape(x)[2]

    y = tf.reverse(x, axis = [1])
    y = tf.concat([x, y], axis = 1)
    f_y = matlab_fft(y)
    f_y = f_y[:, 0:n, :]

    t = tf.complex(tf.constant([0.0]), tf.constant([-1.0])) * tf.cast(tf.linspace(0.0, tf.cast(n-1, tf.float32), n), tf.complex64)
    t = t * tf.cast(math.pi / (2.0 * tf.cast(n, tf.float32)), tf.complex64)
    t = tf.exp(t) / tf.cast(tf.sqrt(2.0 * tf.cast(n, tf.float32)), tf.complex64)

    # since tensor obejct does not support item assignment, we have to concat a new tensor
    t0 = t[0] / tf.cast(tf.sqrt(2.0), tf.complex64)
    t0 = tf.expand_dims(t0, 0)
    t = tf.concat([t0, t[1:]], axis = 0)
    t = tf.expand_dims(t, -1)
    t = tf.expand_dims(t, 0)
    W = tf.tile(t, [N,1,m])

    dct_x = W * f_y
    dct_x = tf.cast(dct_x, tf.complex64)
    dct_x = tf.real(dct_x)

    return dct_x

def idct_odd_case(x):
    N = tf.shape(x)[0]
    n = tf.shape(x)[1]
    m = tf.shape(x)[2]

    temp_complex = tf.complex(tf.constant([0.0]), tf.constant([1.0]))
    t = temp_complex * tf.cast(tf.linspace(0.0, tf.cast(n-1, tf.float32), n), tf.complex64)
    t = tf.cast(tf.sqrt(2.0 * tf.cast(n, tf.float32)), tf.complex64) * tf.exp(t * tf.cast(math.pi / (2.0 * tf.cast(n, tf.float32)), tf.complex64))

    t0 = t[0] * tf.cast(tf.sqrt(2.0), tf.complex64)
    t0 = tf.expand_dims(t0, 0)
    t = tf.concat([t0, t[1:]], axis = 0)
    t = tf.expand_dims(t, -1)
    t = tf.expand_dims(t, 0)
    W = tf.tile(t, [N,1,m])

    x = tf.cast(x, tf.complex64)
    yy_up = W * x
    temp_complex = tf.complex(tf.constant([0.0]), tf.constant([-1.0]))
    yy_down = temp_complex * W[:, 1:n, :] * tf.reverse(x[:,1:n, :], axis = [1])
    yy_mid = tf.cast(tf.zeros([N, 1, m]), tf.complex64)
    yy = tf.concat([yy_up, yy_mid, yy_down], axis = 1)
    y = matlab_ifft(yy)
    y = y[:, 0:n, :]
    y = tf.real(y)

    return y


class dct_layer_hy(Layer):
    def __init__(self, **kwargs):
        super(dct_layer_hy, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        dct_x = dct_odd_case(x)
        return dct_x

    def compute_output_shape(self, input_shape):
        return input_shape

class idct_layer_hy(Layer):
    def __init__(self, **kwargs):
        super(idct_layer_hy, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        idct_x = idct_odd_case(x)
        return idct_x

    def compute_output_shape(self, input_shape):
        return input_shape


def get_dct_model():

    x_in = Input(shape=(None, None), name='input_x')
    
    x1 = dct_layer_hy()(x_in)
    x2 = transpose_layer()(x1)
    x3 = dct_layer_hy()(x2)
    x4 = transpose_layer()(x3)
    out = x4
    # out = x1

    return Model([x_in], [out])

def get_idct_model():

    x_in = Input(shape=(None, None), name = 'input_x')
    
    x1 = idct_layer_hy()(x_in)
    x2 = transpose_layer()(x1)
    x3 = idct_layer_hy()(x2)
    x4 = transpose_layer()(x3)
    out = x4

    return Model([x_in], [out])


if __name__ == "__main__":
    from magic import magic
    # x = 2 * np.eye(4,4)
    # x[0,0] = 5
    # x[3,2] = 1.5
    # x = magic(8)
    # x = np.ones((4,4))
    # x = np.random.rand(6,5)
    # x[0,0] = 5
    # x = magic(4)
    # x = magic(6)

    # x = x[np.newaxis, ...]

    # x = np.ones((4,4,4))
    n = 3
    x1 = magic(n)
    x1 = x1[np.newaxis, ...]
    x2 = np.ones((1,n,n))
    x3 = np.eye(n,n)
   
    x3 = x3[np.newaxis, ...]
    x= np.concatenate((x1, x2, x3), axis = 0)

    model = get_dct_model()
    dct_x = model.predict_on_batch([x])
    print(dct_x)
  
    imodel = get_idct_model()
    ix = imodel.predict_on_batch([dct_x])
    print(ix)  # b
    print(x)
    print('done')