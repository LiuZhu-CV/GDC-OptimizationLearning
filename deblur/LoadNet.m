function net = LoadNet(modelName , epoch ,  usegpu)
%RESDUAL �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
addpath('~/NewDisk/matconvnet-1.0-beta25/matlab');
vl_setupnn();
% modelName='model_0_005_gray_Res_Bnorm_Dilate_Adam';
% modelName = strcat('model_Noisy_5_to_ClearGradient_',direction);

% load(fullfile('E:\L0Smoothing_2\data',modelName,[modelName,'-epoch-',num2str(epoch),'.mat']));

load(fullfile('models', [modelName,'-epoch-',num2str(epoch),'.mat']));
net = vl_simplenn_tidy(net); 
net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net); 

if usegpu == 1
%     disp('moving net to GPU.......')
    net = vl_simplenn_move(net,'gpu');
end

end

