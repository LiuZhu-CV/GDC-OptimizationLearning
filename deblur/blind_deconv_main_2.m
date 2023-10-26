function [k, lambda_pixel, lambda_grad, S] = blind_deconv_main_2(blur_B, k, ...
    lambda_pixel, lambda_grad, threshold, opts,num_scales)

pars.lambda =5e-3;
pars.L=2.8;
lambda = pars.lambda;
L = pars.L;

H = size(blur_B,1);    W = size(blur_B,2);
blur_B_w = wrap_boundary_liu(blur_B, opt_fft_size([H W]+size(k)-1));
blur_B_tmp = blur_B_w(1:H,1:W,:);
[m,n]=size(blur_B_tmp);
[Bx ,By] = gradient(blur_B_tmp);
ppb=1.1;
W_ = @dct2;
WT = @idct2;
if num_scales==1
        opts.xk_iter=opts.xk_iter;
 end
Sx_=Bx;
Sy_=By;
S_=blur_B_tmp;
for iter = 1:opts.xk_iter
    %% Warm Start
    Sbig = psf2otf(k,[m,n]);
    lambda_=8e-4;
    FKernel=psf2otf(k,size(blur_B_tmp));
    KtK=abs(FKernel).^2;  
    Sx=real(ifft2((...
        lambda_*fft2(Sx_)+...
        conj(FKernel).*fft2(Bx))./...
        (KtK+lambda_)));
    Sy=real( ifft2 ((...
        lambda_*fft2(Sy_)+...
        conj(FKernel).*fft2(By))...
        ./(KtK+lambda_)));
    %% Generator
    k = estimate_psf(Bx, By, Sx, Sy, 2, size(k));
    S = L0Restoration(blur_B, k, lambda_grad, 2.0);
%     imshow(S);
    %% Discriminator
        u =S;
        t_max =1;
        elta=0.005;
        beta= 0.05;
        for tt = 1:t_max
            lam=0.006*1*(tt);
            res =net_compute2(u,opts.net_dis,0);
            dfu_gpu=res(1).dzdx;
            dfu=gather(dfu_gpu);
            u=u-elta*(lam*(u-S)+beta*dfu);
        end
        S =u;
%     imshow(S);
    %% Corrector     
    Sx=net_compute(S,opts.net_x,1);
    Sy=net_compute(S,opts.net_y,1);
    
    [SxT,SyT] = threshold_pxpy_my(Sx,Sy,max(size(k)),threshold/2);
    [SxT_,SyT_] = threshold_pxpy_my(Sx_,Sy_,max(size(k)),threshold/2);
    R_now=imfilter(S,rot90(k,2),'replicate');
    R_old=imfilter((S_),rot90(k,2),'replicate');
    [Rx_now,Ry_now]=gradient(R_now);
    [Rx_old,Ry_old]=gradient(R_old);
    S_=S;
    Energe_now=norm((Rx_now+Ry_now)-(Bx+By),'fro')+0.00004*sum(sum((SxT+SyT)~=0));
    Energe_old=norm((Rx_old+Ry_old)-(Bx+By),'fro')+0.00004*sum(sum((SxT_+SyT_)~=0));
    fprintf('Iter=%d | Now=%f | Old=%f \n',iter, Energe_now, Energe_old);
%     if Energe_now>=Energe_old
%          Sx=Sx_;
%         Sy=Sy_;
%      end
    Sx = grad_prox_gradh(Sx,Bx,Sbig,W_,WT,L,lambda);
    Sy = grad_prox_gradv(Sy,By,Sbig,W_,WT,L,lambda);
    Sx_=Sx;
    Sy_=Sy;
    %% end ====================
    [latent_x,latent_y] = threshold_pxpy_my(Sx,Sy,max(size(k)),threshold);
    %% Kernel Estimation
    k_prev = k;
    k = estimate_psf(Bx, By, latent_x, latent_y, 2, size(k_prev));
    figure(10),imshow(imresize(k,4),[]);drawnow;
    %-----------------------------------------------------------------------------------------------------------------------------------
    CC = bwconncomp(k,8);
    for ii=1:CC.NumObjects
        currsum=sum(k(CC.PixelIdxList{ii}));
        if currsum<.1
            k(CC.PixelIdxList{ii}) = 0;
        end
    end
    k(k<0) = 0;
    k=k/sum(k(:)); 
    %% Parameter updating
    if lambda_pixel~=0;
        lambda_pixel = max(lambda_pixel/ppb, 1e-4);
    else
        lambda_pixel = 0;
    end
    lambda_pixel = lambda_pixel/ppb;  %% for natural images
    if lambda_grad~=0;
        lambda_grad = max(lambda_grad/ppb, 1e-4);
    else
        lambda_grad = 0;
    end
    S(S<0) = 0;
    S(S>1) = 1;
end;
k(k<0) = 0;
k = k ./ sum(k(:));
