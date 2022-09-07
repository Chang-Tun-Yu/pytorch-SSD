import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

# from ..utils import box_utils
# from collections import namedtuple
# GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

def align(n, block):
    # align n to block size
    return n + (block - n % block) % block
def encode_conv2d_IF(tensor):
    ''' encode ifmap tensor
        input:  C-H-W
        output: C-H-W-Hp-Wp-Cp
    '''
    assert len(tensor.shape) == 3
    C, H, W = tensor.shape
    Kp, Cp, Hp, Wp = 1,32,1,1
    Ca = align(C, Cp)
    if (H % Hp + W % Wp):
        print(H, W, Hp, Wp)
    assert (H % Hp + W % Wp) == 0

    z = np.zeros((Ca-C, H, W))
    tensor = np.concatenate((tensor, z), 0)
    
    
    tensor = np.reshape(tensor, (Ca//Cp, Cp, H//Hp, Hp, W//Wp, Wp)) # C-Cp-H-Hp-W-Wp
    tensor = np.transpose(tensor, (0,2,4,3,5,1)) # convert to C-H-W-Hp-Wp-Cp
    tensor = tensor.flatten()
    return tensor

