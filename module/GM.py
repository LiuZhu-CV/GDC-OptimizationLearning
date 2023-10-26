from keras.layers import Input, Conv2D
from keras.models import Model

from module.Layers import *


# tensorflow: convert PSFs to OTFs
# psf: tensor with shape [height, width, channels_in, channels_out]
# img_shape: pair of integers


def Generative_model():

    x=Input(shape=(None,None,1),name='input_x')

    # n=Conv2D(64,(3,3),padding='same',dilation_rate=1,activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    n=Conv2D(64,(3,3),padding='same',dilation_rate=1,activation='relu')(x)
    n=Conv2D(64,(3,3),padding='same',dilation_rate=2,activation='relu')(n)
    n=Conv2D(64,(3,3),padding='same',dilation_rate=3,activation='relu')(n)
    n=Conv2D(64,(3,3),padding='same',dilation_rate=4,activation='relu')(n)
    n=Conv2D(64,(3,3),padding='same',dilation_rate=3,activation='relu')(n)
    n=Conv2D(64,(3,3),padding='same',dilation_rate=2,activation='relu')(n)
    n=Conv2D(1,(3,3),padding='same',dilation_rate=1)(n)
    z=ReduceNoiseLayer()([x,n])
    return Model([x],z)