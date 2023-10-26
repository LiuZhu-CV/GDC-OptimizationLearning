from keras.layers import Input, Conv2D
from keras.models import Model

from module.Layers_AN import *


# tensorflow: convert PSFs to OTFs
# psf: tensor with shape [height, width, channels_in, channels_out]
# img_shape: pair of integers



def GM_AN(batch_info):  # batch_info = [N, H, W]

    RealSN_Conv_Layer.batch_size = batch_info[0]
    RealSN_Conv_Layer.input_h = batch_info[1]
    RealSN_Conv_Layer.input_w = batch_info[2]

    x = Input(shape=(None, None, 1), name='input_x')

    ns1 = RealSN_Conv_Layer(20, 64, (3, 3), padding='same', dilation_rate=1, activation='relu')(x)

    ns2 = RealSN_Conv_Layer(20, 64, (3, 3), padding='same', dilation_rate=2, activation='relu')(ns1[0])

    ns3 = RealSN_Conv_Layer(20, 64, (3, 3), padding='same', dilation_rate=3, activation='relu')(ns2[0])

    ns4 = RealSN_Conv_Layer(20, 64, (3, 3), padding='same', dilation_rate=4, activation='relu')(ns3[0])

    ns5 = RealSN_Conv_Layer(20, 64, (3, 3), padding='same', dilation_rate=3, activation='relu')(ns4[0])

    ns6 = RealSN_Conv_Layer(20, 64, (3, 3), padding='same', dilation_rate=2, activation='relu')(ns5[0])

    ns7 = RealSN_Conv_Layer(20, 1, (3, 3), padding='same', dilation_rate=1)(ns6[0])

    z = ReduceNoiseLayer()([x, ns7[0]])

    return Model([x], z)