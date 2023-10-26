clear;
warning off;
addpath(genpath('whyte_code'));
addpath(genpath('cho_code'));
addpath(genpath('implit_deconvolution'));
addpath(genpath('L0Smoothing'));
addpath(genpath('fina_deconvolution_code'));
opts.prescale = 1;              %%  downsampling
opts.xk_iter = 5;                %%  5 the iterations
opts.gamma_correct = 1.0;
opts.k_thresh = 20;
% opts.net = load_net('binary_classifier', 110, 0);
lambda_p=4e-3;
lambda_g=4e-3;
SpsPar=0.0002;
PSNR_ALL = 0;
SSIM_ALL = 0;
ER_ALL= 0;
KS_ALL= 0;
TIME_ALL= 0;
global epoch;
opts.usegpu=1;
epoch =75;
recorde_PSNR = [];
    count=0;
    for idx_img = 1:1
        
        for idx_ker = 1:1
            I0 = im2double(imread('manmade_02.png'));
            I0_1 = I0
            if size(I0,3)==3
                I0_1 =rgb2gray(I0);
            end
            count=count+1;
            fprintf('im_%d_ker_%d  \n ',idx_img,idx_ker);
            file = strcat('./levin_data/im01','_ker0',int2str(idx_ker),'.mat');
            A = load(file);
%            I0 =A.x;
              A.x = I0_1;
              A.y = im2double(imread('manmade_02_kernel_03.png'));
            y = A.y;
            opts.kernel_size = 51;
%             gt_kernel = rot90(A.f,2);
%             opts.kernel_size = size(gt_kernel,1);
            opts.A=I0;    
            opts.gamma_correct = 1.0;
            if size(y,3)==3
                yg = im2double(rgb2gray(y));
            else
                yg = im2double(y);
            end
            tic;
            net_x = LoadNet('model_Noisy_15_to_Direct_ClearGradient_X',epoch,opts.usegpu);
            net_y = LoadNet('model_Noisy_15_to_Direct_ClearGradient_Y',epoch,opts.usegpu);
             net_dis =load_net('binary_classifier', 110, 0);
            opts.net_x=net_x;
            opts.net_y=net_y;
             opts.net_dis =net_dis;
            
            lambda_pixel =lambda_p; lambda_grad = lambda_g;
            [kernel, interim_latent] = blind_deconv_2(yg, lambda_pixel, lambda_grad, opts);
            TIME=toc; 
             x1_r=deconvSps(y(:,:,1),rot90(kernel,2),SpsPar);
             x1_g=deconvSps(y(:,:,2),rot90(kernel,2),SpsPar);
             x1_b=deconvSps(y(:,:,3),rot90(kernel,2),SpsPar);
             x1 = cat(3,x1_r,x1_g,x1_b);
              %x1= whyte_deconv(y, rot90(kernel,0));

            figure,imshow(x1);
            [output] =deconvSps(yg,rot90(kernel,2),SpsPar); 
%             [output_gtk] = deconvSps(y,rot90(gt_kernel,2),SpsPar); 
%             figure,imshow(output_gtk);
             [PSNR,SSIM]= comp_quantitative(output,I0,size(kernel,1))
              
            k = kernel - min(kernel(:));
            k = k./max(k(:));
            figure,imshow(rot90(k,2));
            imwrite(rot90(k,2),'example/kernel2.png');
            imwrite(x1,'./GCM_2.png');
%             KS = kernel_similarity(gt_kernel,kernel);
%              [output] =deconvSps(yg,rot90(kernel,2),SpsPar); 
% %             [output_gtk] = deconvSps(y,rot90(gt_kernel,2),SpsPar); 
% %             figure,imshow(output_gtk);
%               [PSNR,SSIM]= comp_quantitative(output,I0,size(kernel,1));
%             ER = comp_error_ratio(output_gtk(1+50:end-50,1+50:end-50),output(1+50:end-50,1+50:end-50),A.x,opts.kernel_size);
%             save(['sun/sun_result/pic_' num2str(idx_img) '_ker_' num2str(idx_ker) '.mat'],'output','output_gtk','kernel','PSNR','SSIM','ER','KS','TIME');    
%             ER_ALL = ER_ALL+ER;
%             KS_ALL = KS_ALL+KS;
%             TIME_ALL = TIME_ALL+TIME;
            PSNR_ALL = PSNR_ALL + PSNR;
            SSIM_ALL = SSIM_ALL + SSIM;
            fprintf('PSNR=%f, Average_PSNR = %f, ',PSNR,PSNR_ALL/count);
            fprintf('SSIM=%f, Average_SSIM = %f \n',SSIM,SSIM_ALL/count);
        end
    end
    
    disp(['Average PSNR : ' num2str(PSNR_ALL/count)]);
    disp(['Average SSIM:' num2str(SSIM_ALL/count)]);
    disp(['Average TiME:' num2str(TIME_ALL/count)]);
    disp(['Average KS:' num2str(KS_ALL/count)]);
    disp(['Average ER:' num2str(ER_ALL/count)]);



