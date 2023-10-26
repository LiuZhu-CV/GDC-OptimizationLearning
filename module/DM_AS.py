from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, Conv2D, Dense, LeakyReLU, BatchNormalization
from keras.models import Model

from module.Layers import *

def DM_AN():
    def conv2d_block(input, filters, strides=1, bn=False):
        ds = RealSN_Conv_Layer(1, filters, (3, 3), strides=strides, padding='same',use_bias=False)(input)
        d = LeakyReLU(alpha=0.2)(ds[0])
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    # Input high resolution image
    filters = 16
    img = Input(shape=(None, None, 1))
    x = conv2d_block(img, filters, bn=False)
    x = conv2d_block(x, filters, strides=2)
    x = conv2d_block(x, filters*2)
    x = conv2d_block(x, filters*2, strides=2, bn = True)
    x = conv2d_block(x, filters*4)
    x = conv2d_block(x, filters*4, strides=2, bn = True)
    x = conv2d_block(x, filters*8)
    x = conv2d_block(x, filters*8, strides=2)
    x = Dense(filters*8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)

    # Create model and compile
    model = Model(inputs=img, outputs=x)
    return model