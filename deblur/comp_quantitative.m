function [psnr0,ssim0] = comp_quantitative(fe,sharp_ground,k)
ks = floor(k/2);
fe1 = fe(1+ks:end-ks,1+ks:end-ks);
[m,n] = size(fe1);
psnr0 = 0;
ssim0 = 0;
count=0;
for i = 1:k
    for j = 1:k
        count = count+1;
        %         if mod(count,100)==0
        %         disp(num2str(count));
        %         end
        %         fprintf('%d',count);
        sharp_ground1 = sharp_ground(i:m+i-1,j:j+n-1);
        psnr1 = psnr(fe1,sharp_ground1);
        ssim1 = ssim(fe1,sharp_ground1);
%         ssim1=100;
        if psnr1>psnr0
            psnr0 = psnr1;
        end
        if ssim1>ssim0
            ssim0 = ssim1;
        end
        
    end
end