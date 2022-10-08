from base64 import decode
from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
from torchvision import models
from .quantized_modules import *

class DeepLab(BaseModel):
    """
    Simplified model for segmentation
    """
    def __init__(self, num_classes=21, in_channels=3, **_):
        super(DeepLab, self).__init__()
        vgg = models.vgg16(pretrained=True)
        encoder = list(vgg.features.children())
        self.en_conv1 = encoder[0]
        self.en_conv2 = encoder[2]
        self.en_conv3 = encoder[5]
        self.en_conv4 = encoder[7]
        self.en_conv5 = encoder[10]
        self.en_conv6 = encoder[12]
        self.en_conv7 = encoder[14]
        self.en_conv8 = encoder[17]
        self.en_conv9 = encoder[19]
        self.en_conv10 = encoder[21]
        self.en_conv11 = encoder[24]
        self.en_conv12 = encoder[26]
        self.en_conv13 = encoder[28]
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        decoder = list(reversed(encoder))
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                decoder[i-1:i+1] = list(reversed(decoder[i-1:i+1]))
        for i, module in enumerate(decoder):
            if isinstance(module, nn.MaxPool2d):
                decoder[i+1] = nn.ConvTranspose2d(decoder[i+1].out_channels, decoder[i+1].in_channels, kernel_size=2, stride=2)
            elif isinstance(module, nn.Conv2d):
                decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)
        
        self.de_conv1 = decoder[1]
        self.de_conv2 = decoder[3]
        self.de_conv3 = decoder[5]
        self.de_conv4 = decoder[8]
        self.de_conv5 = decoder[10]
        self.de_conv6 = decoder[12]
        self.de_conv7 = decoder[15]
        self.de_conv8 = decoder[17]
        self.de_conv9 = decoder[19]
        self.de_conv10 = decoder[22]
        self.de_conv11 = decoder[24]
        self.de_conv12 = decoder[27]
        self.de_conv13 = decoder[29]
        self.de_conv14 = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)

        self._initialize_weights(self.de_conv1, self.de_conv2, self.de_conv3, self.de_conv4, self.de_conv5, 
            self.de_conv6, self.de_conv7, self.de_conv8, self.de_conv9, self.de_conv10, 
            self.de_conv11, self.de_conv12, self.de_conv13, self.de_conv14)
        ''' Quantization '''
        self.is_quantized = False
        self.HW_SIM = False
        self.DUMP = False
        self.qconfig = quantization.QConfig(
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
    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()

    def sim_hw(self, dump=False):
        assert self.is_quantized
        self.HW_SIM = True
        self.DUMP = True

    def quantize_model(self, dump_folder=None):
        self.is_quantized = True
        self.input_fake_quant = self.qconfig.activation()
        ''' Encoder '''
        self.en_conv1.qconfig = self.qconfig
        self.en_conv2.qconfig = self.qconfig
        self.en_conv3.qconfig = self.qconfig
        self.en_conv4.qconfig = self.qconfig
        self.en_conv5.qconfig = self.qconfig 
        self.en_conv6.qconfig = self.qconfig 
        self.en_conv7.qconfig = self.qconfig 
        self.en_conv8.qconfig = self.qconfig 
        self.en_conv9.qconfig = self.qconfig
        self.en_conv10.qconfig = self.qconfig 
        self.en_conv11.qconfig = self.qconfig 
        self.en_conv12.qconfig = self.qconfig
        self.en_conv13.qconfig = self.qconfig
        self.en_conv1 = QConvReLU2d.from_float(self.en_conv1)
        self.en_conv2 = QConvReLU2d.from_float(self.en_conv2)
        self.en_conv3 = QConvReLU2d.from_float(self.en_conv3)
        self.en_conv4 = QConvReLU2d.from_float(self.en_conv4)
        self.en_conv5 = QConvReLU2d.from_float(self.en_conv5)
        self.en_conv6 = QConvReLU2d.from_float(self.en_conv6)
        self.en_conv7 = QConvReLU2d.from_float(self.en_conv7)
        self.en_conv8 = QConvReLU2d.from_float(self.en_conv8)
        self.en_conv9 = QConvReLU2d.from_float(self.en_conv9)
        self.en_conv10 = QConvReLU2d.from_float(self.en_conv10)
        self.en_conv11 = QConvReLU2d.from_float(self.en_conv11)
        self.en_conv12 = QConvReLU2d.from_float(self.en_conv12)
        self.en_conv13 = QConvReLU2d.from_float(self.en_conv13)
        self.en_conv1.dump_folder = dump_folder 
        self.en_conv2.dump_folder = dump_folder 
        self.en_conv3.dump_folder = dump_folder 
        self.en_conv4.dump_folder = dump_folder 
        self.en_conv5.dump_folder = dump_folder 
        self.en_conv6.dump_folder = dump_folder 
        self.en_conv7.dump_folder = dump_folder 
        self.en_conv8.dump_folder = dump_folder 
        self.en_conv9.dump_folder = dump_folder 
        self.en_conv10.dump_folder = dump_folder 
        self.en_conv11.dump_folder = dump_folder 
        self.en_conv12.dump_folder = dump_folder 
        self.en_conv13.dump_folder = dump_folder 
        self.en_conv1.layer_name = "conv_layer1"
        self.en_conv2.layer_name = "conv_layer2"
        self.en_conv3.layer_name = "conv_layer3"
        self.en_conv4.layer_name = "conv_layer4"
        self.en_conv5.layer_name = "conv_layer5"
        self.en_conv6.layer_name = "conv_layer6"
        self.en_conv7.layer_name = "conv_layer7"
        self.en_conv8.layer_name = "conv_layer8"
        self.en_conv9.layer_name = "conv_layer9"
        self.en_conv10.layer_name = "conv_layer10"
        self.en_conv11.layer_name = "conv_layer11"
        self.en_conv12.layer_name = "conv_layer12"
        self.en_conv13.layer_name = "conv_layer13"
        self.pool = QMaxPool2d(kernel_size=self.pool.kernel_size, stride=self.pool.stride)
        ''' Decoder '''
        self.de_conv1.qconfig = self.qconfig
        self.de_conv2.qconfig = self.qconfig
        self.de_conv3.qconfig = self.qconfig
        self.de_conv4.qconfig = self.qconfig
        self.de_conv5.qconfig = self.qconfig 
        self.de_conv6.qconfig = self.qconfig 
        self.de_conv7.qconfig = self.qconfig 
        self.de_conv8.qconfig = self.qconfig 
        self.de_conv9.qconfig = self.qconfig 
        self.de_conv10.qconfig = self.qconfig 
        self.de_conv11.qconfig = self.qconfig 
        self.de_conv12.qconfig = self.qconfig 
        self.de_conv13.qconfig = self.qconfig
        self.de_conv14.qconfig = self.qconfig
        self.de_conv1 = QConvTransposeReLU2d.from_float(self.de_conv1)
        self.de_conv2 = QConvReLU2d.from_float(self.de_conv2)
        self.de_conv3 = QConvReLU2d.from_float(self.de_conv3)
        self.de_conv4 = QConvTransposeReLU2d.from_float(self.de_conv4)
        self.de_conv5 = QConvReLU2d.from_float(self.de_conv5) 
        self.de_conv6 = QConvReLU2d.from_float(self.de_conv6) 
        self.de_conv7 = QConvTransposeReLU2d.from_float(self.de_conv7) 
        self.de_conv8 = QConvReLU2d.from_float(self.de_conv8) 
        self.de_conv9 = QConvReLU2d.from_float(self.de_conv9) 
        self.de_conv10 = QConvTransposeReLU2d.from_float(self.de_conv10) 
        self.de_conv11 = QConvReLU2d.from_float(self.de_conv11) 
        self.de_conv12 = QConvTransposeReLU2d.from_float(self.de_conv12) 
        self.de_conv13 = QConvReLU2d.from_float(self.de_conv13)
        self.de_conv14 = QConv2d.from_float(self.de_conv14)
        self.de_conv1.dump_folder = dump_folder 
        self.de_conv2.dump_folder = dump_folder 
        self.de_conv3.dump_folder = dump_folder 
        self.de_conv4.dump_folder = dump_folder 
        self.de_conv5.dump_folder = dump_folder 
        self.de_conv6.dump_folder = dump_folder 
        self.de_conv7.dump_folder = dump_folder 
        self.de_conv8.dump_folder = dump_folder 
        self.de_conv9.dump_folder = dump_folder 
        self.de_conv10.dump_folder = dump_folder 
        self.de_conv11.dump_folder = dump_folder 
        self.de_conv12.dump_folder = dump_folder 
        self.de_conv13.dump_folder = dump_folder 
        self.de_conv14.dump_folder = dump_folder
        self.de_conv1.layer_name = "conv_layer14"
        self.de_conv2.layer_name = "conv_layer15"
        self.de_conv3.layer_name = "conv_layer16"
        self.de_conv4.layer_name = "conv_layer17"
        self.de_conv5.layer_name = "conv_layer18"
        self.de_conv6.layer_name = "conv_layer19"
        self.de_conv7.layer_name = "conv_layer20"
        self.de_conv8.layer_name = "conv_layer21"
        self.de_conv9.layer_name = "conv_layer22"
        self.de_conv10.layer_name = "conv_layer23"
        self.de_conv11.layer_name = "conv_layer24"
        self.de_conv12.layer_name = "conv_layer25"
        self.de_conv13.layer_name = "conv_layer26"
        self.de_conv14.layer_name = "conv_layer27"
    
    def forward(self, x):
        if not self.is_quantized:
            # encoder
            x = self.en_conv1(x)
            x = self.relu(x)
            x = self.en_conv2(x)        
            x = self.relu(x)
            x = self.pool(x)
            x = self.en_conv3(x)        
            x = self.relu(x)
            x = self.en_conv4(x)        
            x = self.relu(x)
            x = self.pool(x)
            x = self.en_conv5(x)        
            x = self.relu(x)
            x = self.en_conv6(x)        
            x = self.relu(x)
            x = self.en_conv7(x)        
            x = self.relu(x)
            x = self.pool(x)
            x = self.en_conv8(x)        
            x = self.relu(x)
            x = self.en_conv9(x)        
            x = self.relu(x)
            x = self.en_conv10(x)        
            x = self.relu(x)
            x = self.pool(x)
            x = self.en_conv11(x)        
            x = self.relu(x)
            x = self.en_conv12(x)        
            x = self.relu(x)
            x = self.en_conv13(x)        
            x = self.relu(x)
            x = self.pool(x)
            # decoder
            x = self.de_conv1(x)
            x = self.relu(x)
            x = self.de_conv2(x)        
            x = self.relu(x)
            x = self.de_conv3(x)        
            x = self.relu(x)
            x = self.de_conv4(x)        
            x = self.relu(x)
            x = self.de_conv5(x)        
            x = self.relu(x)
            x = self.de_conv6(x)        
            x = self.relu(x)
            x = self.de_conv7(x)        
            x = self.relu(x)
            x = self.de_conv8(x)        
            x = self.relu(x)
            x = self.de_conv9(x)        
            x = self.relu(x)
            x = self.de_conv10(x)        
            x = self.relu(x)
            x = self.de_conv11(x)        
            x = self.relu(x)
            x = self.de_conv12(x)        
            x = self.relu(x)
            x = self.de_conv13(x)        
            x = self.relu(x)
            x = self.de_conv14(x)
        else:
            if self.training: self.input_fake_quant.enable_observer()
            else: self.input_fake_quant.disable_observer()
            x = self.input_fake_quant(x)
            x.scale = self.input_fake_quant.scale.clone()
            x.zero_point = self.input_fake_quant.zero_point.clone()
            x.HW_SIM = self.HW_SIM
            x.DUMP = self.DUMP
            # encoder
            x = self.en_conv1(x)
            x = self.en_conv2(x)        
            x = self.pool(x)
            x = self.en_conv3(x)        
            x = self.en_conv4(x)        
            x = self.pool(x)
            x = self.en_conv5(x)        
            x = self.en_conv6(x)        
            x = self.en_conv7(x)        
            x = self.pool(x)
            x = self.en_conv8(x)        
            x = self.en_conv9(x)        
            x = self.en_conv10(x)        
            x = self.pool(x)
            x = self.en_conv11(x)        
            x = self.en_conv12(x)        
            x = self.en_conv13(x)        
            x = self.pool(x)
            # decoder
            x = self.de_conv1(x)
            x = self.de_conv2(x)        
            x = self.de_conv3(x)        
            x = self.de_conv4(x)        
            x = self.de_conv5(x)        
            x = self.de_conv6(x)        
            x = self.de_conv7(x)        
            x = self.de_conv8(x)        
            x = self.de_conv9(x)        
            x = self.de_conv10(x)        
            x = self.de_conv11(x)        
            x = self.de_conv12(x)        
            x = self.de_conv13(x)        
            x = self.de_conv14(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


"""
encoder
0	Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
1	ReLU(inplace=True), 
2	Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
3	ReLU(inplace=True), 
4	MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
5	Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
6	ReLU(inplace=True), 
7	Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
8	ReLU(inplace=True), 
9	MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
10	Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
11	ReLU(inplace=True), 
12	Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
13	ReLU(inplace=True), 
14	Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
15	ReLU(inplace=True), 
16	MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
17	Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
18	ReLU(inplace=True), 
19	Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
20	ReLU(inplace=True), 
21	Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
22	ReLU(inplace=True), 
23	MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
24	Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 				
25	ReLU(inplace=True), 
26	Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
27	ReLU(inplace=True), 
28	Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
29	ReLU(inplace=True), 
30	MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

decoder
0	MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
1	ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2)), 
2	ReLU(inplace=True), 
3	Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
4	ReLU(inplace=True), 
5	Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
6	ReLU(inplace=True), 
7	MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
8	ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2)), 
9	ReLU(inplace=True), 
10	Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
11	ReLU(inplace=True), 
12	Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
13	ReLU(inplace=True), 
14	MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
15	ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2)), 
16	ReLU(inplace=True), 
17	Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
18	ReLU(inplace=True), 
19	Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
20	ReLU(inplace=True), 
21	MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
22	ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2)), 
23	ReLU(inplace=True), 
24	Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
25	ReLU(inplace=True), 
26	MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
27	ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2)), 
28	ReLU(inplace=True), 
29	Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
30	ReLU(inplace=True)
"""
