import torch
# from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
import torch.nn.functional as F
import torch.quantization as quantization
from ..nn.mobilenet_v2 import MobileNetV2, InvertedResidual
import os
from .myssd import MySSD
from .predictor import Predictor
from .config import mobilenetv1_ssd_config as config
from ..quantized_modules import *

class QResidualBlock(nn.Module):
    def __init__(self, conv1_layer, conv1_folder, conv1_name, dwcv_layer, dwcv_folder, dwcv_name, \
        conv2_layer, conv2_folder, conv2_name, add_folder, add_name, add_qconfig):
        super().__init__()
        self._pwcv = QConvReLU2d.from_float(conv1_layer, conv1_folder, conv1_name)
        self._dwcv = QConvReLU2d.from_float(dwcv_layer, dwcv_folder, dwcv_name)
        self._conv = QConv2d.from_float(conv2_layer, conv2_folder, conv2_name)
        self._add = Q_Addition(add_folder, add_name, add_qconfig)        

    def forward(self, x):
        tmp = self._pwcv(x)
        tmp1 = self._dwcv(tmp)
        tmp2 = self._conv(tmp1)
        ret = self._add(x, tmp2)
        return ret

class QNonResidualBlock(nn.Module):
    def __init__(self, conv1_layer, conv1_folder, conv1_name, dwcv_layer, dwcv_folder, dwcv_name, \
        conv2_layer, conv2_folder, conv2_name):
        super().__init__()
        self._pwcv = QConvReLU2d.from_float(conv1_layer, conv1_folder, conv1_name)
        self._dwcv = QConvReLU2d.from_float(dwcv_layer, dwcv_folder, dwcv_name)
        self._conv = QConv2d.from_float(conv2_layer, conv2_folder, conv2_name)

    def forward(self, x):
        tmp = self._pwcv(x)
        tmp1 = self._dwcv(tmp)
        ret = self._conv(tmp1)
        return ret       
  
class QPredictionBlock(nn.Module):
    def __init__(self, dwcv_layer, dwcv_folder, dwcv_name, conv_layer, conv_folder, conv_name):
        super().__init__()
        self._dwcv = QConvReLU2d.from_float(dwcv_layer, dwcv_folder, dwcv_name)
        self._conv = QConv2d.from_float(conv_layer, conv_folder, conv_name)
        
    def forward(self, x):
        tmp = self._dwcv(x)
        ret = self._conv(tmp)
        return ret     

def create_mobilenetv2_ssd_qunat(num_classes, model, dump_folder, is_test=False, hwsim=True, dump=False):

    qconfig = quantization.QConfig(
        activation=quantization.FakeQuantize.with_args(
            observer=quantization.MovingAverageMinMaxObserver.with_args(
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_affine,
                    reduce_range=False,
                    quant_min=0,
                    quant_max=255
            ),
            quant_min=0,
            quant_max=255),
        weight=quantization.FakeQuantize.with_args(
            observer=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_affine,
                    reduce_range=False,
                    quant_min=0,
                    quant_max=255
            ),
            quant_min=0,
            quant_max=255)
    )
    
    # setting quantization

    model.base_net[0][0].qconfig = qconfig
    model.base_net[1].conv[0].qconfig = qconfig
    model.base_net[1].conv[3].qconfig = qconfig
    model.base_net[18][0].qconfig = qconfig

    for i in range(2, 18):
        model.base_net[i].conv[0].qconfig = qconfig
        model.base_net[i].conv[3].qconfig = qconfig
        model.base_net[i].conv[6].qconfig = qconfig

    for i in range(4):
        model.extras[i].conv[0].qconfig = qconfig
        model.extras[i].conv[3].qconfig = qconfig
        model.extras[i].conv[6].qconfig = qconfig

    for i in range(5):
        model.classification_headers[i][0].qconfig = qconfig
        model.classification_headers[i][3].qconfig = qconfig
        model.regression_headers[i][0].qconfig = qconfig
        model.regression_headers[i][3].qconfig = qconfig
    
    model.classification_headers[5].qconfig = qconfig
    model.regression_headers[5].qconfig = qconfig

    mobilenet_0 = nn.Sequential(
        QNonResidualBlock(model.base_net[0][0], os.path.join(dump_folder, str(1)), "Q_Conv2d_0",
                        model.base_net[1].conv[0], os.path.join(dump_folder, str(1)), "Q_Conv2d_1",
                        model.base_net[1].conv[3], os.path.join(dump_folder, str(2)), "Q_Conv2d_2"),
        QNonResidualBlock(model.base_net[2].conv[0], os.path.join(dump_folder, str(3)), "Q_Conv2d_3",
                        model.base_net[2].conv[3], os.path.join(dump_folder, str(3)), "Q_Conv2d_4",
                        model.base_net[2].conv[6], os.path.join(dump_folder, str(4)), "Q_Conv2d_5"),
        QResidualBlock(model.base_net[3].conv[0], os.path.join(dump_folder, str(5)), "Q_Conv2d_6",
                        model.base_net[3].conv[3], os.path.join(dump_folder, str(5)), "Q_Conv2d_7",
                        model.base_net[3].conv[6], os.path.join(dump_folder, str(6)), "Q_Conv2d_8",
                                os.path.join(dump_folder, str(7)), "add_1", qconfig),
        QNonResidualBlock(model.base_net[4].conv[0], os.path.join(dump_folder, str(8)), "Q_Conv2d_9",
                        model.base_net[4].conv[3], os.path.join(dump_folder, str(8)), "Q_Conv2d_10",
                        model.base_net[4].conv[6], os.path.join(dump_folder, str(9)), "Q_Conv2d_11"),
        QResidualBlock(model.base_net[5].conv[0], os.path.join(dump_folder, str(10)), "Q_Conv2d_12",
                        model.base_net[5].conv[3], os.path.join(dump_folder, str(10)), "Q_Conv2d_13",
                        model.base_net[5].conv[6], os.path.join(dump_folder, str(11)), "Q_Conv2d_14",
                                os.path.join(dump_folder, str(12)), "add_2", qconfig),
        QResidualBlock(model.base_net[6].conv[0], os.path.join(dump_folder, str(13)), "Q_Conv2d_15",
                        model.base_net[6].conv[3], os.path.join(dump_folder, str(13)), "Q_Conv2d_16",
                        model.base_net[6].conv[6], os.path.join(dump_folder, str(14)), "Q_Conv2d_17",
                                os.path.join(dump_folder, str(15)), "add_3", qconfig),
        QNonResidualBlock(model.base_net[7].conv[0], os.path.join(dump_folder, str(16)), "Q_Conv2d_18",
                        model.base_net[7].conv[3], os.path.join(dump_folder, str(16)), "Q_Conv2d_19",
                        model.base_net[7].conv[6], os.path.join(dump_folder, str(17)), "Q_Conv2d_20"),
        QResidualBlock(model.base_net[8].conv[0], os.path.join(dump_folder, str(18)), "Q_Conv2d_21",
                        model.base_net[8].conv[3], os.path.join(dump_folder, str(18)), "Q_Conv2d_22",
                        model.base_net[8].conv[6], os.path.join(dump_folder, str(19)), "Q_Conv2d_23",
                                os.path.join(dump_folder, str(20)), "add_4", qconfig),
        QResidualBlock(model.base_net[9].conv[0], os.path.join(dump_folder, str(21)), "Q_Conv2d_24",
                        model.base_net[9].conv[3], os.path.join(dump_folder, str(21)), "Q_Conv2d_25",
                        model.base_net[9].conv[6], os.path.join(dump_folder, str(22)), "Q_Conv2d_26",
                                os.path.join(dump_folder, str(23)), "add_5", qconfig),
        QResidualBlock(model.base_net[10].conv[0], os.path.join(dump_folder, str(24)), "Q_Conv2d_27",
                        model.base_net[10].conv[3], os.path.join(dump_folder, str(24)), "Q_Conv2d_28",
                        model.base_net[10].conv[6], os.path.join(dump_folder, str(25)), "Q_Conv2d_29",
                                os.path.join(dump_folder, str(26)), "add_6", qconfig),
        QNonResidualBlock(model.base_net[11].conv[0], os.path.join(dump_folder, str(27)), "Q_Conv2d_30",
                        model.base_net[11].conv[3], os.path.join(dump_folder, str(27)), "Q_Conv2d_31",
                        model.base_net[11].conv[6], os.path.join(dump_folder, str(28)), "Q_Conv2d_32"),
        QResidualBlock(model.base_net[12].conv[0], os.path.join(dump_folder, str(29)), "Q_Conv2d_33",
                        model.base_net[12].conv[3], os.path.join(dump_folder, str(29)), "Q_Conv2d_34",
                        model.base_net[12].conv[6], os.path.join(dump_folder, str(30)), "Q_Conv2d_35",
                                os.path.join(dump_folder, str(31)), "add_7", qconfig),
        QResidualBlock(model.base_net[13].conv[0], os.path.join(dump_folder, str(32)), "Q_Conv2d_36",
                        model.base_net[13].conv[3], os.path.join(dump_folder, str(32)), "Q_Conv2d_37",
                        model.base_net[13].conv[6], os.path.join(dump_folder, str(33)), "Q_Conv2d_38",
                                os.path.join(dump_folder, str(34)), "add_8", qconfig),
        QConvReLU2d.from_float(model.base_net[14].conv[0], os.path.join(dump_folder, str(35)), "Q_Conv2d_39")
    )

    mobilenet_1 = nn.Sequential(
        QConvReLU2d.from_float(model.base_net[14].conv[3], os.path.join(dump_folder, str(36)), "Q_Conv2d_40"),
        QConv2d.from_float(model.base_net[14].conv[6], os.path.join(dump_folder, str(37)), "Q_Conv2d_41"),
        QResidualBlock(model.base_net[15].conv[0], os.path.join(dump_folder, str(38)), "Q_Conv2d_42",
                        model.base_net[15].conv[3], os.path.join(dump_folder, str(38)), "Q_Conv2d_43",
                        model.base_net[15].conv[6], os.path.join(dump_folder, str(39)), "Q_Conv2d_44",
                                os.path.join(dump_folder, str(40)), "add_9", qconfig),
        QResidualBlock(model.base_net[16].conv[0], os.path.join(dump_folder, str(41)), "Q_Conv2d_45",
                        model.base_net[16].conv[3], os.path.join(dump_folder, str(41)), "Q_Conv2d_46",
                        model.base_net[16].conv[6], os.path.join(dump_folder, str(42)), "Q_Conv2d_47",
                                os.path.join(dump_folder, str(43)), "add_10", qconfig),
        QNonResidualBlock(model.base_net[17].conv[0], os.path.join(dump_folder, str(44)), "Q_Conv2d_48",
                        model.base_net[17].conv[3], os.path.join(dump_folder, str(44)), "Q_Conv2d_49",
                        model.base_net[17].conv[6], os.path.join(dump_folder, str(45)), "Q_Conv2d_50"),
        QConvReLU2d.from_float(model.base_net[18][0], os.path.join(dump_folder, str(46)), "Q_Conv2d_51")   
    )

    extra_net = nn.ModuleList([
            QNonResidualBlock(model.extras[0].conv[0], os.path.join(dump_folder, str(47)), "Q_Conv2d_52",
                        model.extras[0].conv[3], os.path.join(dump_folder, str(47)), "Q_Conv2d_53",
                        model.extras[0].conv[6], os.path.join(dump_folder, str(48)), "Q_Conv2d_54"),
            QNonResidualBlock(model.extras[1].conv[0], os.path.join(dump_folder, str(49)), "Q_Conv2d_55",
                        model.extras[1].conv[3], os.path.join(dump_folder, str(49)), "Q_Conv2d_56",
                        model.extras[1].conv[6], os.path.join(dump_folder, str(50)), "Q_Conv2d_57"),
            QNonResidualBlock(model.extras[2].conv[0], os.path.join(dump_folder, str(51)), "Q_Conv2d_58",
                        model.extras[2].conv[3], os.path.join(dump_folder, str(51)), "Q_Conv2d_59",
                        model.extras[2].conv[6], os.path.join(dump_folder, str(52)), "Q_Conv2d_60"),
            QNonResidualBlock(model.extras[3].conv[0], os.path.join(dump_folder, str(53)), "Q_Conv2d_61",
                        model.extras[3].conv[3], os.path.join(dump_folder, str(53)), "Q_Conv2d_62",
                        model.extras[3].conv[6], os.path.join(dump_folder, str(54)), "Q_Conv2d_63")
    ])

    classification_headers = nn.ModuleList([
            QPredictionBlock(model.classification_headers[0][0], os.path.join(dump_folder, str(55)), "Q_Conv2d_64",
                             model.classification_headers[0][3], os.path.join(dump_folder, str(56)), "Q_Conv2d_65"),
            QPredictionBlock(model.classification_headers[1][0], os.path.join(dump_folder, str(57)), "Q_Conv2d_66",
                             model.classification_headers[1][3], os.path.join(dump_folder, str(58)), "Q_Conv2d_67"),
            QPredictionBlock(model.classification_headers[2][0], os.path.join(dump_folder, str(59)), "Q_Conv2d_68",
                             model.classification_headers[2][3], os.path.join(dump_folder, str(60)), "Q_Conv2d_69"),
            QPredictionBlock(model.classification_headers[3][0], os.path.join(dump_folder, str(61)), "Q_Conv2d_70",
                             model.classification_headers[3][3], os.path.join(dump_folder, str(62)), "Q_Conv2d_71"),
            QPredictionBlock(model.classification_headers[4][0], os.path.join(dump_folder, str(63)), "Q_Conv2d_72",
                             model.classification_headers[4][3], os.path.join(dump_folder, str(64)), "Q_Conv2d_73"),
            QConv2d.from_float(model.classification_headers[5], os.path.join(dump_folder, str(65)), "Q_Conv2d_74")
    ])
    regression_headers = nn.ModuleList([
            QPredictionBlock(model.regression_headers[0][0], os.path.join(dump_folder, str(66)), "Q_Conv2d_75",
                             model.regression_headers[0][3], os.path.join(dump_folder, str(67)), "Q_Conv2d_76"),
            QPredictionBlock(model.regression_headers[1][0], os.path.join(dump_folder, str(68)), "Q_Conv2d_77",
                             model.regression_headers[1][3], os.path.join(dump_folder, str(69)), "Q_Conv2d_78"),
            QPredictionBlock(model.regression_headers[2][0], os.path.join(dump_folder, str(70)), "Q_Conv2d_79",
                             model.regression_headers[2][3], os.path.join(dump_folder, str(71)), "Q_Conv2d_80"),
            QPredictionBlock(model.regression_headers[3][0], os.path.join(dump_folder, str(72)), "Q_Conv2d_81",
                             model.regression_headers[3][3], os.path.join(dump_folder, str(73)), "Q_Conv2d_82"),
            QPredictionBlock(model.regression_headers[4][0], os.path.join(dump_folder, str(74)), "Q_Conv2d_83",
                             model.regression_headers[4][3], os.path.join(dump_folder, str(75)), "Q_Conv2d_84"),
            QConv2d.from_float(model.regression_headers[5], os.path.join(dump_folder, str(76)), "Q_Conv2d_85")
    ])

    ret = MySSD(num_classes, mobilenet_0, mobilenet_1,
               extra_net, classification_headers, regression_headers, is_test=is_test, config=config)
    ret.hw_sim(hwsim, dump)
    return ret

def create_mobilenetv2_ssd_quant_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor