import numpy as np
import math
from scipy.signal import fftconvolve
import tensorflow as tf
from scipy import ndimage

import os
import matplotlib.image as mpimage
from scipy.signal import convolve2d
from scipy.io import loadmat
import time
import decimal
def get_net_index(sigma, iter_num, modelSigma1=49, modelSigma2=13):

    if iter_num==1:
        modelSigmaS=modelSigma2
    else:
        modelSigmaS=np.logspace(np.log10(modelSigma1),np.log10(modelSigma2),iter_num)
    net_index=np.zeros([iter_num,])
    net_index[:]=np.ceil(modelSigmaS/2)
    net_index[:]=np.clip(net_index,1,25)
    net_index=net_index.astype(np.int)

    return net_index

def psnr(img1, img2):
    
    assert img1.dtype==img2.dtype

    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100

    if img1.dtype=='int':
        PIXEL_MAX = 255.0
    if img1.dtype=='float32' or img1.dtype=='float64':
        PIXEL_MAX = 1.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def pad_for_kernel(img,kernel,mode):
    p = [(d-1)//2 for d in kernel.shape]
    padding = [p,p] + (img.ndim-2)*[(0,0)]
    return np.pad(img, padding, mode)


def crop_for_kernel(img,kernel):
    p = [(d-1)//2 for d in kernel.shape]
    r = [slice(p[0],-p[0]),slice(p[1],-p[1])] + (img.ndim-2)*[slice(None)]
    return img[tuple(r)]


def edgetaper_alpha(kernel,img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel,1-i),img_shape[i]-1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z,z[0:1]],0)
        v.append(1 - z/np.max(z))
    return np.outer(*v)


def edgetaper(img,kernel,n_tapers=3):
    alpha = edgetaper_alpha(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[...,np.newaxis]
        alpha  = alpha[...,np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img,_kernel,'wrap'),kernel,mode='valid')
        img = alpha*img + (1-alpha)*blurred
    return img

def psf2otf(psf, img_shape):
    # shape and type of the point spread function(s)
    psf_shape = tf.shape(psf)
    psf_type = psf.dtype

    # coordinates for 'cutting up' the psf tensor
    midH = tf.floor_div(psf_shape[0], 2)
    midW = tf.floor_div(psf_shape[1], 2)

    # slice the psf tensor into four parts
    top_left     = psf[:midH, :midW, :, :]
    top_right    = psf[:midH, midW:, :, :]
    bottom_left  = psf[midH:, :midW, :, :]
    bottom_right = psf[midH:, midW:, :, :]

    # prepare zeros for filler
    zeros_bottom = tf.zeros([psf_shape[0] - midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)
    zeros_top    = tf.zeros([midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)

    # construct top and bottom row of new tensor
    top    = tf.concat([bottom_right, zeros_bottom, bottom_left], 1)
    bottom = tf.concat([top_right,    zeros_top,    top_left],    1)

    # prepare additional filler zeros and put everything together
    zeros_mid = tf.zeros([img_shape[0] - psf_shape[0], img_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)
    pre_otf = tf.concat([top, zeros_mid, bottom], 0)
    # output shape: [img_shape[0], img_shape[1], channels_in, channels_out]

    # fast fourier transform, transposed because tensor must have shape [..., height, width] for this
    otf = tf.fft2d(tf.cast(tf.transpose(pre_otf, perm=[2,3,0,1]), tf.complex64))

    # output shape: [channels_in, channels_out, img_shape[0], img_shape[1]]
    return otf


def test_levin(deblur_model_total, test_sigma = 0.01, target = 'temp'):
    test_gt_s=[]
    test_y_s=[]
    test_k_s=[]
    size_taper = []
    
    dir_path = os.path.join('Vision_Results', target, 'Levin', 'Circular_%.2f' % test_sigma)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('loading test data Levin, and %.1f%% noise is added......'%(test_sigma*100))
    for im_index in range(4):
        gt_name = r'data/levin_data/img_%d.png'% (im_index + 1)
        gt = mpimage.imread(gt_name)
        gt = gt
        for ker_index in range(8):
            im_name = r'data/levin_data/im_%d_ker_%d.png' % (im_index+1, ker_index+1)
            ker_name = r'data/levin_data/kernel_%d.dlm' % (ker_index+1)
            # y = cv2.imread(im_name, 0)
            # y = y / 255
            # y_ = y
            ker = np.loadtxt(ker_name).astype(np.float)
            ker = np.clip(ker, 0, 1)
            ker = ker / np.sum(ker)
            y=ndimage.convolve(gt,ker,mode='wrap')
            np.random.seed(1512818945)
            y = y + test_sigma * np.random.standard_normal(size=y.shape)
            y = np.clip(y, 0, 1.0)
            y = ((y * 255.0).astype(np.uint8) / 255.0).astype(np.float32)
            # y = edgetaper(pad_for_kernel(y, ker, 'edge'), ker)
            # size_taper.append(int((y.shape[0]-y_.shape[0])/2))
            y = y[np.newaxis, ..., np.newaxis]
            ker = ker[np.newaxis, ...]
            test_gt_s.append(gt)
            test_y_s.append(y)
            test_k_s.append(ker)


    # deblur_model_total.load_weights(r'joint_train_model/model_epoch_%d.hdf5'% 51)
    #test levin32 1%--------------------------------------------------------------------------
    print('testing on Levin32 with noise:%.3f%% ......' % (test_sigma*100))
    PSNR_32=0
    SSIM =0
    Time=0
    test_sigma_=np.zeros([1,1])
    test_sigma_[0]=test_sigma
    for i in range(32):
        out=deblur_model_total.predict_on_batch([test_y_s[i], test_k_s[i], test_sigma_])
        pred = out[0,:,:,0]

        # imsave(('Vision_Results/%s/Levin/Circular_%.2f/%03d.png'% (target, test_sigma, i+1)),np.clip(pred,0,1))
        # pred = pred[0, size_taper[i]:pred.shape[1]-size_taper[i], size_taper[i]:pred.shape[1]-size_taper[i], 0]
        PSNR=psnr(test_gt_s[i].astype('float32'), pred)
        print('test pic:%02d, PSNR=%.4f' % ((i+1),PSNR))
        PSNR_32+=PSNR
        #TEST
        # for cc in range(0,45,3):

        #     E_now = out[cc+1]
        #     E_old = out[cc+2]
        #     print('Step:%02d: E_now:%6.4f | E_old:%6.4f'% (cc, E_now, E_old))
            
        #     pred = out[cc][0,:,:,0]
        #     imsave(('test_temp2/%03d.png'% (cc+1)),np.clip(pred,0,1))
        #     # pred = pred[0, size_taper[i]:pred.shape[1]-size_taper[i], size_taper[i]:pred.shape[1]-size_taper[i], 0]
        #     PSNR=psnr(test_gt_s[i].astype('float32'), pred)
        #     print('test pic:%02d, PSNR=%.4f' % ((i+1),PSNR))
        #     PSNR_32+=PSNR
    A_PSNR=PSNR_32/32
    print('A_PSNR=%f' % A_PSNR)
    return A_PSNR



def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))



if __name__ == '__main__':
    
    mse = 0.00188125 

    # if img1.dtype=='int':
    #     PIXEL_MAX = 255.0
    # if img1.dtype=='float32' or img1.dtype=='float64':
    PIXEL_MAX = 1.0

    print(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))