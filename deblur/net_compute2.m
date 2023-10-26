function res = net_compute(input,net,usegpu)
% input = im2single(input);
input=single(input);
% figure(5),imshow(input,[]),title('net/_input');
if usegpu==1
    input = gpuArray(input);
end

input=single(input);
% net.layers{1}.precious=true;

% input1(:,:,1) = input;
% input1(:,:,2) = input;

input_1(:,:,1,1) = input;

res    = vl_simplenn(net,input_1,single(1),[],'conserveMemory',true,'mode','test');
 
% output = double(input - res(end).x);
%  output = double(res(end).x);
%  if   usegpu==1
%  output = gather(output);
%  end


% figure(6),imshow(output,[]),title('net\_output');
 
end