import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="6,3"

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
print('cuda index:', torch.cuda.current_device())
print('gpu 개수:', torch.cuda.device_count())