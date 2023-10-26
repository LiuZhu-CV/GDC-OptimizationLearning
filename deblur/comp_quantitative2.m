function [psnr0,ssim0] = comp_quantitative2(fe,sharp_ground,k)
ks = floor(k/2);
fe1 = fe(1+ks:end-ks,1+ks:end-ks,:);
[m,n,c] = size(fe1);
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
        sharp_ground1 = sharp_ground(i:m+i-1,j:j+n-1,:);
        A=double(fe1); % Ground-truth
        B=double(sharp_ground1); %

        e=A(:)-B(:);
        mse=mean(e.^2);
        psnr0=10*log10(1.0^2/mse);

% if ch==1
%     [ssim_cur, ~] = ssim_index(A, B);
% else
    
%     ssim0 = (ssim_index(A(:,:,1), B(:,:,1)) + ssim_index(A(:,:,2), B(:,:,2)) + ssim_index(A(:,:,3), B(:,:,3)))/3;

    end
end