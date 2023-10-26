import torch
import torch.nn as nn
from Derain_AS.operations_m import *
from torch.autograd import Variable


class MixedOp(nn.Module):

  def __init__(self, C, primitive):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    kernel = 3
    dilation = 1
    if primitive.find('attention') != -1:
        name = primitive.split('_')[0]
        kernel = int(primitive.split('_')[1])
    else:
        name = primitive.split('_')[0]
        kernel = int(primitive.split('_')[1])
        dilation = int(primitive.split('_')[2])
    print(name, kernel, dilation)
    self._op = OPS[name](C, kernel, dilation, False)

  def forward(self, x):
    return self._op(x)


class Cell(nn.Module):

  def __init__(self, genotype, C):
    super(Cell, self).__init__()
    self.preprocess1 = ReLUConvBN(C, C, 1, 1, 0)
    op_names, indices = zip(*genotype.normal_gen)
    concat = genotype.normal_gen_concat
    self._compile(C, op_names, indices, concat)
    # self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2 + 1
    self._concat = concat
    self.multiplier = len(concat)
    self.fusion = SKFF(C,self._steps,reduction=8)
    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = MixedOp(C,name)
      self._ops += [op]
    self._indices = indices

  def forward(self,  inp):
    s1 = self.preprocess1(inp)
    states = [s1]
    for i in range(self._steps):
      if i ==0:
        h1 = states[self._indices[i]]
        op1 = self._ops[i]
        h1 = op1(h1)
        states +=[h1]
      else:
        h1 = states[self._indices[2*i-1]]
        h2 = states[self._indices[2*i ]]
        op1 = self._ops[2*i-1]
        op2 = self._ops[2*i]
        h1 = op1(h1)
        h2 = op2(h2)
        s = h1 + h2
        states += [s]
    res = self.fusion([states[i] for i in self._concat])
    return inp+res



class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
      super(MeanShift, self).__init__(3, 3, kernel_size=1)
      std = torch.Tensor(rgb_std)
      self.weight.data = torch.eye(3).view(3, 3, 1, 1)
      self.weight.data.div_(std.view(3, 1, 1, 1))
      self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
      self.bias.data.div_(std)
      self.requires_grad = False


#GM
class Network_Generator(nn.Module):

  def __init__(self, C,  layers, genotype,multi=False):
    super(Network_Generator, self).__init__()
    self.multi = multi
    self._layers = layers
    self.stem = nn.Sequential(
      nn.Conv2d(3, C, 3, padding=1, bias=True)
    )
    self.stem_out = nn.Sequential(
      nn.Conv2d(C, C, 3, padding=1, bias=True),
      nn.Conv2d(C,3, 3, padding=1, bias=True),
    )

    self.tanh = nn.Tanh()
    # self.cells = nn.ModuleList()

    #Encoders
    self.Cell_encoder_1 = Cell(genotype,C)
    self.Cell_encoder_2 = Cell(genotype,C)
    self.Cell_encoder_3 = Cell(genotype,C)
    self.Cell_encoder_4 = Cell(genotype,C)

  def forward(self, input):

    s1 = self.stem(input)
    s1 = self.Cell_encoder_1(s1)
    s1= self.Cell_encoder_2(s1)
    s1= self.Cell_encoder_3(s1)
    s1= self.Cell_encoder_4(s1)
    res = self.stem_out(s1)
    output = input + self.tanh(res)
    return output


#DM
class Network_DIS(nn.Module):

  def __init__(self, C,  layers, genotype,multi=False):
    super(Network_DIS, self).__init__()
    self.multi = multi
    self._layers = layers
    self.stem = nn.Sequential(
      nn.Conv2d(3, C, 3, padding=1, bias=False),
      nn.Conv2d(C, C, 3, padding=1, bias=False),
    )

    self.Cell_encoder_1 = Cell(genotype,C)
    self.Cell_encoder_2 = Cell(genotype,C)
    self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)

    self.fc = nn.Sequential(
      nn.Linear(C, 64),
      nn.ReLU(inplace=True),
      nn.Linear(64, 1),
      nn.Sigmoid()
    )


  def forward(self, input):
    s1 = self.stem(input)
    s1 = self.Cell_encoder_1(s1)
    s1 = self.Cell_encoder_2(s1)
    b, c, _, _ = s1.size()
    y = self.avg_pool_1(s1).view(b, c)
    # y = self.fc(y).view(b, c, 1, 1)
    return y
class Network_GDC(nn.Module):

  def __init__(self, C,  layers, genotype,genotype2,multi=False, gamma=1e5,step=1e-6):
    super(Network_GDC, self).__init__()
    self.multi = multi
    self._layers = layers
    self.gamma = gamma
    self.step = step
    self.generator = Network_Generator(C,layers,genotype)
    self.generator2 = Network_Generator(C,layers,genotype)
    self.generator3 = Network_Generator(C,layers,genotype)
    self.discriminator = Network_DIS(32,layers,genotype2)
  def forward(self, input):
    # with torch.no_grad():
    x = input
    #GM
    s1 = self.generator(input)
    # s1_d  =self.discriminator(s1)
    # gradient = torch.autograd.grad(s1_d.sum(),s1)
    # s1 = s1 -self.step*gradient[0]
    #CM
    s1_new =(x+self.gamma*(s1))/ (1+self.gamma)
    #GM
    s1 = self.generator2(s1_new)
    #CM
    s1_new = (x + self.gamma* (s1)) / (1 + self.gamma)
    #GM
    s1 = self.generator3(s1_new)
    return s1



