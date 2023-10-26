from module.DM  import *
from module.GM  import *
from module.Layers import *
from module.tools import *


# tensorflow: convert PSFs to OTFs
# psf: tensor with shape [height, width, channels_in, channels_out]
# img_shape: pair of integers

def deblur_model_GDC(iter_num,modelSigma1=49,modelSigma2=13, sigma = 0.01, dis_prior_weight=0.0001, D_weights=''):

    discriminator = Discriminative_module()
    discriminator.load_weights(D_weights)

    if iter_num==1:
        modelSigmaS=modelSigma2
    else:
        modelSigmaS=np.logspace(np.log10(modelSigma1),np.log10(modelSigma2),iter_num)

    DeconvolutionLayer_GDC.modelSigmaS=modelSigmaS

    # build the whole deblur model
    z=[]
    u=[]

    x_in=Input(shape=(None,None,1),name='input_x')
    k_in=Input(shape=(None,None),name='input_k')
    sigma_in=Input(shape=(1,),name='input_sigma')
    z_old = x_in
    denominator=ConstantLayer_Denominator()([x_in,k_in])
    upperleft=ConstantLayer_Upperleft()([x_in,k_in])

    net_index = get_net_index(sigma, iter_num)
    print(net_index)
   
    for j in range(iter_num):
        
        # warm start
        x=DeconvolutionLayer_GDC(j, name=('DeconvLayer_%d' % j))([(x_in if j==0 else z),(x_in if j==0 else u),denominator,upperleft,sigma_in])
        # Generative Module
        gm = Generative_model()
        print('loading weights of iter: %d' % net_index[j])
        gm.load_weights('ckpt/gms_ckpt/net%d.hdf5'%(net_index[j]))
        z_out = gm(x)
        u=z_out

        # Discriminative Module
        v=discriminator(u)
        u=GradientDescentLayer(j, dis_prior_weight, name = 'D_Graient_Descent_Layer_%d_%d' % (j, 1))([u, x, v])
        # Corrective Module
        z_out = Proximal_Layer(0.001, 0.0015)([u, denominator, upperleft])
        zs= Energy_Choose_Layer2(name='Energy_Choose_Layer_%d' % j)([z_out, z_old, x_in, k_in])
        z =zs [0]
        # Proximal Layer
        z = Proximal_Layer(0.001, 0.0015)([z, denominator, upperleft])
        # z_old = z


    return Model([x_in, k_in, sigma_in], z_out)



if __name__=="__main__":

    D_weights_path = os.path.join('ckpt','dm_ckpt',
                                    'dis_model.hdf5')
    deblur_model_total = deblur_model_GDC(30,49,13,D_weights = D_weights_path)
    # Load the CNN modules
    # deblur_model_total.load_weights('./Experiments/GCD_V1_Large_gp_loss_weight/generator_epoch_019.hdf5')
    # # # # GDC
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