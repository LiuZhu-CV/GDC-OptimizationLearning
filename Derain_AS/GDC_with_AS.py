# python 2.7, pytorch 0.3.1

import os, sys
sys.path.insert(1, '../')
import torch
import cv2
import shutil
import torchvision
import numpy as np
import itertools
import subprocess
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from modules_AS import Network_GDC as Network
from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal_gen normal_gen_concat normal_dis normal_dis_concat')

derain = Genotype(normal_gen=[('Residualblocks_3_1', 0), ('Residualblocks_3_1', 1), ('Residualblocks_3_1', 0), ('DilConv_5_1', 0), ('ECAattention_3', 2), ('DilConv_3_1', 3), ('ECAattention_3', 2)], normal_gen_concat=[1, 2, 3,4], normal_dis=[('Residualblocks_3_1', 0), ('Residualblocks_3_1', 1), ('Residualblocks_3_1', 0), ('DilConv_5_1', 0), ('ECAattention_3', 2), ('DilConv_3_1', 3), ('ECAattention_3', 2)], normal_dis_concat=[1, 2, 3,4])
discri = Genotype(normal_gen=[('ECAattention_3', 0), ('Denseblocks_5_1', 1), ('DilConv_5_1', 0), ('SPAattention_3', 1), ('SPAattention_3', 2), ('DilConv_5_1', 1), ('Residualblocks_5_1', 0)], normal_gen_concat=[1, 2, 3,4], normal_dis=[('ECAattention_3', 0), ('Denseblocks_5_1', 1), ('DilConv_5_1', 0), ('SPAattention_3', 1), ('SPAattention_3', 2), ('DilConv_5_1', 1), ('Residualblocks_5_1', 0)], normal_dis_concat=[1, 2, 3,4])

#load GDC_AS Network
genotype = eval("%s" % 'derain')
genotype2 = eval("%s" % 'discri')
model = Network(64,  4, genotype,genotype2,multi=False).cuda()
model_path = 'weights_1299.pt'
model.load_state_dict(torch.load(model_path,map_location='cuda:0'))
model.eval()
#----------------------
def main(argv=None):
    from PIL import Image

    pth_input = './input_1.png'
    pth_output = './output.png'
    im_input = cv2.imread(pth_input)
    im_input =cv2.cvtColor(im_input,cv2.COLOR_BGR2RGB)
    im_input = np.array(im_input)[np.newaxis, :]
    im_input = np.transpose(im_input, (0, 3, 1, 2)).astype(np.float) / 255.
    im_input = torch.tensor(im_input).type(torch.FloatTensor)
    im_input = Variable(im_input, requires_grad=False).cuda()
    with torch.no_grad():
        res= model(im_input)
    res = res.data.cpu().numpy()
    res[res > 1] = 1
    res[res < 0] = 0
    res *= 255
    res = res.astype(np.uint8)[0]
    res = res.transpose((1, 2, 0))
    Image.fromarray(res).save(pth_output)
    print('Test done.')
if __name__=='__main__':

    sys.exit(main())