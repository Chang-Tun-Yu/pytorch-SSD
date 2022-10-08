from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite

import torch
import torch.nn as nn
from torch import Tensor
import torch.quantization as quantization
import torch.nn.qat as nnqat
import torch.nn.functional as F
import numpy as np
import os
import sys


if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

for i in range(1):
    
    image_path = os.path.join("C:/Users/user/Desktop/image_300", "{}.png".format(i+1))
    # os.mkdir("C:/Users/user/Desktop/tensor/{}".format(i+1))

    class_names = [name.strip() for name in open(label_path).readlines()]

    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True, image_i=i+1)
    elif net_type == 'mb3-large-ssd-lite':
        net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'mb3-small-ssd-lite':
        net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)
    net.load(model_path)
    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    net = net.eval()
    modules_to_fuse = []
    with open("fuse_list.txt", "r") as f:
        for l in f:
            l_list = l.strip().split()
            modules_to_fuse.append(l_list)
    torch.ao.quantization.fuse_modules(net, modules_to_fuse, inplace=True)

    for name, layer in net.named_modules():
        print(name, layer)


    # print(net.base_net[0][0].out_channels, net.base_net[0][0].kernel_size,
    #                    net.base_net[0][0].stride, net.base_net[0][0].padding, 
    #                    net.base_net[0][0].groups, net.base_net[0][0].bias,
    #                    net.base_net[0][0].dilation, net.base_net[0][0].padding_mode)
    # print(net.base_net[0][0].weight)
    net.save("./models/mb2-ssd-fusebn.pth")