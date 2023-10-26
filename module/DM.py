from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, Conv2D, Dense, LeakyReLU, BatchNormalization
from keras.models import Model


def Discriminative_module():
    def conv2d_block(input, filters, strides=1, bn=False):
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
        d = LeakyReLU(alpha=0.2)(d)
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
    x = Dense(1)(x)


    # Create model and compile
    model = Model(inputs=img, outputs=x)
    return model
    return Model([d0], sigma_map)

