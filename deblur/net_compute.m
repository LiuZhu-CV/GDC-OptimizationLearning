function output = net_compute(input,net,usegpu)
 input = im2single(input);
%   figure(5),imshow(input,[]),title('net\_input');
if usegpu==1
    input = gpuArray(input);
end
% input=single(input);
 res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
%  output = double(input - res(end).x);
 output = double(res(end).x);
 if   usegpu==1
 output = gather(output);
 end


%  figure(6),imshow(output,[]),title('net\_output');
 
end