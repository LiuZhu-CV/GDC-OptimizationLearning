from module.DM_AN  import *
from module.GM_AN  import *
from module.Layers_AN import *
from module.tools import *


# tensorflow: convert PSFs to OTFs
# psf: tensor with shape [height, width, channels_in, channels_out]
# img_shape: pair of integers

def deblur_model_GDC_AN(iter_num, modelSigma1=49, modelSigma2=13, sigma=0.01, dis_prior_weight=0.01,
                                    D_weights='', batch_info=[0, 0, 0]):
    discriminator = DM_AN()


    if iter_num == 1:
        modelSigmaS = modelSigma2
    else:
        modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num)

    DeconvolutionLayer_GDC.modelSigmaS = modelSigmaS

    # build the whole deblur model
    z = []
    u = []

    x_in = Input(shape=(None, None, 1), name='input_x')
    k_in = Input(shape=(None, None), name='input_k')
    sigma_in = Input(shape=(1,), name='input_sigma')

    denominator = ConstantLayer_Denominator()([x_in, k_in])
    upperleft = ConstantLayer_Upperleft()([x_in, k_in])

    net_index = get_net_index(sigma, iter_num)
    print(net_index)

    for j in range(iter_num):

        # GM
        x = DeconvolutionLayer_GDC(j, name=('DeconvLayer_%d' % j))(
            [(x_in if j == 0 else z), (x_in if j == 0 else u), denominator, upperleft, sigma_in])
        gm = GM_AN(batch_info=batch_info)
        z_out = gm(x)
        # DM
        u = x
        for descent_step in range(5):
            v = discriminator(u)
            v = Weight_D_Layer()(v)
            u = GradientDescentLayer(j, dis_prior_weight, name='D_Graient_Descent_Layer_%d_%d' % (j, descent_step))(
                [u, x, v])

        # CM
        z = Proximal_Layer(0.001, 0.0015)([z_out, denominator, upperleft])

    return Model([x_in, k_in, sigma_in], z_out)


if __name__=="__main__":

    deblur_model_total = deblur_model_GDC_AN(15,49,13,D_weights = '',batch_info=[1,255,255])
    deblur_model_total.load_weights('./ckpt/GDC_AN_Joint_Training/model_epoch_007.hdf5')
    target = 'GDC'
    Levin_Result = []
    # BSD68_Result = []
    Noise_Level = [0.01]
    print('Testing Levin Set ...')
    for i in Noise_Level:
        A_Levin_PSNR =test_levin(deblur_model_total, i, target)
        Levin_Result.append(A_Levin_PSNR)

    for i in range(len(Noise_Level)):
           print('Levin %.2f : %.2f' % (Noise_Level[i], Levin_Result[i]))


    print('done')