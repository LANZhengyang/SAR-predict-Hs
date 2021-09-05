import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


data_1516 = torch.load('./dataset/tensors_2015_2016.pt')
data_17 = torch.load('./dataset/tensors_2017.pt')


torch.save({'x1': torch.cat([data_1516['x1'],data_17['x1']],0), 'x2':torch.cat([data_1516['x2'],data_17['x2']],0), 'y': torch.cat([data_1516['y'],data_17['y']],0)}, 'tensors_2015_2017.pt')