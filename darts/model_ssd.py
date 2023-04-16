import torch
import torch.nn as nn
from torch.autograd import Variable
#from darts.utils import drop_path, count_parameters_in_MB
#from darts.operations import *
#import darts.genotypes
from utils import drop_path, count_parameters_in_MB
from operations import *
import genotypes

class Cell_nor(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction_prev):
    super(Cell_nor, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    op_names, indices = zip(*genotype.normal)
    concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

class Cell_red(nn.Module):

  def __init__(self, C_prev_prev, C_prev, C, ceil):
    super(Cell_red, self).__init__()
    print(C_prev_prev, C_prev, C)
    self.ceil = ceil
    self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

  def forward(self, s0, s1):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    if self.ceil == True :
      h1 = self.pool3(s0)
      h2 = self.pool3(s1)
      s = h1+h2
    else :
      h1 = self.pool1(s0)
      h2 = self.pool1(s1)
      s = h1+h2

    return s

class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    C_curr = 64
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C_curr

    self.cell1 = Cell_nor(genotype, C_prev_prev=64, C_prev=64, C=64, reduction_prev=False) # (N, 64, 300, 300)
    self.cell2 = Cell_nor(genotype, C_prev_prev=64, C_prev=256, C=64, reduction_prev=False) # (N, 64, 300, 300)
    self.cell3 = Cell_red(C_prev_prev= 256, C_prev=256, C=128, ceil=False)
    #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
    #self.cell3 = Cell(genotype, C_prev_prev=256, C_prev=256, C=128, reudction=True, reduction_prev=False) # (N, 128, 150, 150) reduction cell
        
    self.cell4 = Cell_nor(genotype, C_prev_prev=256, C_prev=512, C=128, reduction_prev=True) # (N, 128, 150, 150)
    self.cell5 = Cell_nor(genotype, C_prev_prev=512, C_prev=512, C=128, reduction_prev=False) # (N, 128, 150, 150)
    self.cell6 = Cell_red(C_prev_prev=512, C_prev=512, C=256, ceil=False) # (N, 256, 75, 75)
    #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
    #self.cell6 = Cell(genotype, C_prev_prev=512, C_prev=512, C=256, reudction=True, reduction_prev=False) # (N, 256, 75, 75) reduction cell
        
    self.cell7 = Cell_nor(genotype, C_prev_prev=512, C_prev=1024, C=256, reduction_prev=True) # (N, 256, 75, 75)
    self.cell8 = Cell_nor(genotype, C_prev_prev=1024, C_prev=1024, C=256, reduction_prev=False) # (N, 256, 75, 75)
    self.cell9 = Cell_red(C_prev_prev=1024, C_prev=1024, C=512, ceil = True)
    #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    #self.cell9 = Cell(genotype, C_prev_prev=1024, C_prev=1024, C=512, reudction=True, reduction_prev=False) # (N, 512, 38, 38) reduction cell - ceil_mode
        
    self.cell10 = Cell_nor(genotype, C_prev_prev=1024, C_prev=2048, C=512, reduction_prev=True)
    self.cell11 = Cell_nor(genotype, C_prev_prev=2048, C_prev=2048, C=512, reduction_prev=False)
    self.cell12 = Cell_red(C_prev_prev=2048, C_prev=2048, C=512, ceil=False)
    #self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    #self.cell12 = Cell(genotype, C_prev_prev=2048, C_prev=2048, C=512, reudction=True, reduction_prev=False)
        
    self.cell13 = Cell_nor(genotype, C_prev_prev=2048, C_prev=2048, C=512, reduction_prev=True)
    self.cell14 = Cell_nor(genotype, C_prev_prev=2048, C_prev=2048, C=512, reduction_prev=False)
    #self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)
    
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    s0 = s1 = self.stem(input)
    s0, s1 = s1, self.cell1(s0, s1, self.drop_path_prob) 
    s0, s1 = s1, self.cell2(s0, s1, self.drop_path_prob) 
    s0, s1 = s1, self.cell3(s0, s1)
    
    s0, s1 = s1, self.cell4(s0, s1, self.drop_path_prob) 
    s0, s1 = s1, self.cell5(s0, s1, self.drop_path_prob)
    s0, s1 = s1, self.cell6(s0, s1)

    s0, s1 = s1, self.cell7(s0, s1, self.drop_path_prob)
    s0, s1 = s1, self.cell8(s0, s1, self.drop_path_prob)
    s0, s1 = s1, self.cell9(s0, s1)

    s0, s1 = s1, self.cell10(s0, s1, self.drop_path_prob)
    s0, s1 = s1, self.cell11(s0, s1, self.drop_path_prob)
    s0, s1 = s1, self.cell12(s0, s1)

    s0, s1 = s1, self.cell13(s0, s1, self.drop_path_prob)
    s0, s1 = s1, self.cell14(s0, s1, self.drop_path_prob)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

if __name__ == '__main__':
  device = 'cuda:6'
  torch.cuda.set_device(device)

  init_channels = 64
  num_classes = 21
  layers = 14
  auxiliary = False
  arch = 'PCDARTS'
  genotype = eval("genotypes.%s" % arch)
  model = NetworkCIFAR(init_channels, num_classes, layers, auxiliary, genotype)
  model = model.cuda()
  #print("model = %s", model)
  print("param size = ", count_parameters_in_MB(model), "MB")
