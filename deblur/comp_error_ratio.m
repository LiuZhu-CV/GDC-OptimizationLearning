function error_ratio = comp_error_ratio(Error_sharp_image,Error_latent_image,sharp_ground,k)

ks = floor(k/2);
Error_sharp_image1 = Error_sharp_image(1+ks:end-ks,1+ks:end-ks);
Error_latent_image1 = Error_latent_image(1+ks:end-ks,1+ks:end-ks);
[m,n] = size(Error_sharp_image1);
d10 = 1000;
d20 = 1000;
for i = 1:k
    for j = 1:k
        sharp_ground1 = sharp_ground(i:m+i-1,j:j+n-1);
        d11 = norm(Error_sharp_image1 -sharp_ground1);
        d21 = norm(Error_latent_image1-sharp_ground1);
        if d11<d10
            d10 = d11;
        end
        if d21<d20
            d20 = d21;
        end
        
    end
end
error_ratio =  d20/d10;