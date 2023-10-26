function s = kernel_similarity(ori_k, esti_k)
% calculate the kernel similarity using the response of convolution
% devided by the norm

ori_k = ori_k/sqrt(sum(sum(ori_k.^2)));
temp = conv2(ori_k, esti_k,'full');
temp2 = sqrt(sum(sum(esti_k.^2)));
s = max(temp(:))/temp2;

end