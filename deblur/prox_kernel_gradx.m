function kernel = prox_kernel_gradx(Bx, By, latent_x, latent_y,k,L)
lambda = 0.01;
[km,kn] = size(k);
[m,n] = size(Bx);
latent_xf = fft2(latent_x);
latent_yf = fft2(latent_y);
Bxf = fft2(Bx);
Byf = fft2(By);
X_t_X = conj(latent_xf).*(latent_xf);
X_t_Y = conj(latent_yf).*(latent_yf);
B_t_X = conj(latent_xf).*Bxf;
B_t_Y = conj(latent_yf).*Byf;
k_otf = psf2otf(k,[m,n]);
grad_k = otf2psf((X_t_X+X_t_Y) .* k_otf -B_t_X-B_t_Y ,[km,kn]);
k = k - 1e-5*grad_k;
%% l1-ball
u = sort(k(:),'descend');
p = zeros(1,1);
rho = km*kn;
for j = 1:km*kn
    p(j)= u(j)-(sum(u(1:j))-1)/j;
    if p(j)<0
        rho = j-1;
        break;
    end
end
theta = (sum(u(1:rho))-1)/rho;
kernel = max(k-theta,0);

