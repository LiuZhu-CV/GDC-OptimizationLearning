%% ----------------------------------------------------------------------------------------------------------------------------
% This is an deblur implementation of paper 
% "Learning Collaborative Generation Correction Modules for Blind Image Deblurring and Beyond"
%  Risheng Liu, Yi He, Shichao Cheng, Xin Fan, Zhongxuan Luo, ACM MM 2018
%
% **Requirement: Matconvnet-1.0-beta24 or later
% **Note:Some paremeters may need finetune
% **% The Code is created based on the method described in the following paper 
%        Jinshan Pan, Zhe Hu, Zhixun Su, and Ming-Hsuan Yang,
%        Deblurring Text Images via L0-Regularized Intensity and Gradient
%        Prior, CVPR, 2014. 
%
% If you find this code is useful, please cite our paper.
%
% @heyi 2018/8/13
%% ----------------------------------------------------------------------------------------------------------------------------
addpath('~/NewDisk/matconvnet-1.0-beta25/matlab/');
addpath(genpath('./.'));
opts.prescale = 1; 
opts.xk_iter = 5;
opts.gamma_correct = 1.0;
opts.k_thresh = 20;
opts.usegpu=1;
global epoch;

lambda_p=0;
lambda_g=4e-3;
SpsPar=0.005;

y=im2double(imread('real_leaffiltered.png'));

if size(y,3)==3
    yg =rgb2gray(y);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for epoch = 75:75
    %%%Loading net
    net_x = LoadNet('model_Noisy_15_to_Direct_ClearGradient_X',epoch,opts.usegpu);
    net_y = LoadNet('model_Noisy_15_to_Direct_ClearGradient_Y',epoch,opts.usegpu);
    net_dis =load_net('binary_classifier', 110, 0);
    opts.net_x=net_x;
    opts.net_y=net_y;
    opts.net_dis =net_dis;
    opts.kernel_size = 65;
    lambda_pixel =lambda_p; lambda_grad = lambda_g;
    tttttt=tic;
    [kernel, interim_latent] = blind_deconv_2(yg, lambda_pixel, lambda_grad, opts);
    time=toc(tttttt);
    x1_r=deconvSps(y(:,:,1),rot90(kernel,2),SpsPar);
    x1_g=deconvSps(y(:,:,2),rot90(kernel,2),SpsPar);
    x1_b=deconvSps(y(:,:,3),rot90(kernel,2),SpsPar);
    x1 = cat(3,x1_r,x1_g,x1_b);
%     x1= whyte_deconv(y, rot90(kernel,0));
    figure,imshow(x1);
    k = kernel - min(kernel(:));
    k = k./max(k(:));
    figure,imshow([y, x1]);
    imwrite(x1,'resultgcd.png');
    imwrite(rot90(k,2),'gcdlekernel.png');
end

