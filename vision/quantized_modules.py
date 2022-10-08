from symbol import shift_expr
from turtle import window_height
from unicodedata import name
import torch
import torch.nn as nn
from torch import Tensor
import torch.quantization as quantization
import torch.nn.qat as nnqat
import torch.nn.functional as F
import numpy as np
import os


class QConvTranspose2d(nn.ConvTranspose2d):

    """ Simulated Quantization Module for TransposeConv2d
    """
    _FLOAT_MODULE = nn.ConvTranspose2d
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, 
                 padding_mode='zeros', qconfig=None, folder_name=None, layer_name=None) -> None:
        super(QConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, output_padding=output_padding, 
                         groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_fake_quant = self.qconfig.activation()
        self.weight_fake_quant = self.qconfig.weight()
        self.layer_name = layer_name
        self.dump_folder = folder_name

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        # if type(mod) == ConvReLU2d:
        #     mod = mod[0]
        qconfig = mod.qconfig
        qat_tconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, output_padding=mod.output_padding, 
                       groups=mod.groups, bias=mod.bias is not None,
                       dilation=mod.dilation, padding_mode=mod.padding_mode, qconfig=qconfig)
        qat_tconv.weight = mod.weight
        qat_tconv.bias = mod.bias
        return qat_tconv

    def hwsim(self, qweight, input_tensor, output_tensor, prequant_output_tensor, dump=False):
        input_scale = input_tensor.scale
        input_zp = input_tensor.zero_point
        output_scale = output_tensor.scale
        output_zp = output_tensor.zero_point
        weight_scale = self.weight_fake_quant.scale
        weight_zp = self.weight_fake_quant.zero_point
        m, shift = get_mult_shift(input_scale, weight_scale, output_scale)
        if dump:
            # Dump weight & bias
            int_weight = torch.round((qweight/weight_scale) + weight_zp)
            int_bias = torch.round(self.bias / (input_scale * weight_scale))
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_weight', to_np(int_weight))
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_bias', to_np(int_bias))
            save_np(os.path.join(self.dump_folder, self.layer_name),'weight_scale', to_np(torch.tensor([weight_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'weight_zp', to_np(torch.tensor([weight_zp])))
            # Dump input feature
            int_input = torch.round((input_tensor / input_scale) + input_zp)
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_input', to_np(int_input))
            save_np(os.path.join(self.dump_folder, self.layer_name),'input_scale', to_np(torch.tensor([input_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'input_zp', to_np(torch.tensor([input_zp])))
        # Dump output feature
        # TODO: Only for 8-bit quantization
        int_output = torch.round(torch.clamp((m * (2 ** (-1 * shift)) * (prequant_output_tensor / (input_scale * weight_scale)) + output_zp), 0, 255))
        hwsim_output = output_scale * (int_output - output_zp)
        hwsim_output.scale = output_scale
        hwsim_output.zero_point = output_zp
        hwsim_output.HW_SIM = input_tensor.HW_SIM
        if dump:
            hwsim_output.DUMP = dump
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_output', to_np(int_output))
            save_np(os.path.join(self.dump_folder, self.layer_name),'output_scale', to_np(torch.tensor([output_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'output_zp', to_np(torch.tensor([output_zp])))
            # Dump multiplier & shifter
            save_np(os.path.join(self.dump_folder, self.layer_name),'multiplier', to_np(torch.tensor([m])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'shifter', to_np(torch.tensor([shift])))
        return hwsim_output

    def forward(self, input: Tensor):#, output_size: Optional[List[int]] = None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
        assert isinstance(self.padding, tuple)
        qweight = self.weight_fake_quant(self.weight)
        weight_scale = self.weight_fake_quant.scale
        qbias = torch.round(self.bias / (input.scale * weight_scale)) * (input.scale * weight_scale)
        x = F.conv_transpose2d(
                input, qweight, qbias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilation
            )
        if self.training: self.act_fake_quant.enable_observer()
        else: self.act_fake_quant.disable_observer()
        x_prequant = x.detach().clone()
        x_quant = self.act_fake_quant(x_prequant)
        x_quant.scale = self.act_fake_quant.scale
        x_quant.zero_point = self.act_fake_quant.zero_point
        ''' HW SIM '''
        if hasattr(input, 'HW_SIM') and input.HW_SIM:
            x_quant = self.hwsim(qweight, input, x_quant, x, dump=(hasattr(input, 'DUMP') and input.DUMP))
        return x_quant


class QConvTransposeReLU2d(nn.ConvTranspose2d):

    """ Simulated Quantization Module for TransposeConv2d + ReLU
    """
    _FLOAT_MODULE = nn.ConvTranspose2d
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, 
                 padding_mode='zeros', qconfig=None, folder_name=None, layer_name=None) -> None:
        super(QConvTransposeReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, output_padding=output_padding, 
                         groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_fake_quant = self.qconfig.activation()
        self.weight_fake_quant = self.qconfig.weight()
        self.dump_folder = folder_name
        self.layer_name = layer_name
    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        qat_tconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, output_padding=mod.output_padding, 
                       groups=mod.groups, bias=mod.bias is not None,
                       dilation=mod.dilation, padding_mode=mod.padding_mode, qconfig=qconfig)
        qat_tconv.weight = mod.weight
        qat_tconv.bias = mod.bias
        return qat_tconv

    def hwsim(self, qweight, input_tensor, output_tensor, prequant_output_tensor, dump=False):
        input_scale = input_tensor.scale
        input_zp = input_tensor.zero_point
        output_scale = output_tensor.scale
        output_zp = output_tensor.zero_point
        weight_scale = self.weight_fake_quant.scale
        weight_zp = self.weight_fake_quant.zero_point
        m, shift = get_mult_shift(input_scale, weight_scale, output_scale)
        if dump:
            # Dump weight & bias
            int_weight = torch.round((qweight/weight_scale) + weight_zp)
            int_bias = torch.round(self.bias / (input_scale * weight_scale))
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_weight', to_np(int_weight))
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_bias', to_np(int_bias))
            save_np(os.path.join(self.dump_folder, self.layer_name),'weight_scale', to_np(torch.tensor([weight_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'weight_zp', to_np(torch.tensor([weight_zp])))
            # Dump input feature
            int_input = torch.round((input_tensor / input_scale) + input_zp)
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_input', to_np(int_input))
            save_np(os.path.join(self.dump_folder, self.layer_name),'input_scale', to_np(torch.tensor([input_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'input_zp', to_np(torch.tensor([input_zp])))
        # Dump output feature
        # TODO: Only for 8-bit quantization
        int_output = torch.round(torch.clamp((m * (2 ** (-1 * shift)) * (prequant_output_tensor / (input_scale * weight_scale)) + output_zp), 0, 255))
        hwsim_output = output_scale * (int_output - output_zp)
        hwsim_output.scale = output_scale
        hwsim_output.zero_point = output_zp
        hwsim_output.HW_SIM = input_tensor.HW_SIM
        if dump:
            hwsim_output.DUMP = dump
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_output', to_np(int_output))
            save_np(os.path.join(self.dump_folder, self.layer_name),'output_scale', to_np(torch.tensor([output_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'output_zp', to_np(torch.tensor([output_zp])))
            # Dump multiplier & shifter
            save_np(os.path.join(self.dump_folder, self.layer_name),'multiplier', to_np(torch.tensor([m])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'shifter', to_np(torch.tensor([shift])))
        return hwsim_output

    def forward(self, input: Tensor):

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        assert isinstance(self.padding, tuple)
        qweight = self.weight_fake_quant(self.weight)
        weight_scale = self.weight_fake_quant.scale
        qbias = torch.round(self.bias / (input.scale * weight_scale)) * (input.scale * weight_scale)
        x = F.relu(
                F.conv_transpose2d(
                    input, qweight, qbias, self.stride, self.padding,
                    self.output_padding, self.groups, self.dilation
                )
            )
        
        if self.training: self.act_fake_quant.enable_observer()
        else: self.act_fake_quant.disable_observer()

        x_prequant = x.detach().clone()
        x_quant = self.act_fake_quant(x_prequant)
        x_quant.scale = self.act_fake_quant.scale
        x_quant.zero_point = self.act_fake_quant.zero_point
        ''' HW_SIM '''
        if hasattr(input, 'HW_SIM') and input.HW_SIM:
            x_quant = self.hwsim(qweight, input, x_quant, x, dump=(hasattr(input, 'DUMP') and input.DUMP))
        return x_quant

class QConv2d(nnqat.Conv2d):

    """ Simulated Quantization Module for Conv2d
    """
    _FLOAT_MODULE = nn.Conv2d
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', qconfig=None,
                 device=None, dtype=None, folder_name=None, layer_name=None) -> None:
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode,
                         qconfig=qconfig)
        self.act_fake_quant = qconfig.activation()
        self.dump_folder = folder_name
        self.layer_name = layer_name

    @classmethod
    def from_float(cls, mod, dump_folder, layer_name):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        # if type(mod) == ConvReLU2d:
        #     mod = mod[0]
        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                       groups=mod.groups, bias=mod.bias is not None, padding_mode=mod.padding_mode,
                       qconfig=qconfig, folder_name=dump_folder, layer_name=layer_name)
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv

    def hwsim(self, qweight, input_tensor, output_tensor, prequant_output_tensor, dump=False):
        input_scale = input_tensor.scale
        input_zp = input_tensor.zero_point
        output_scale = output_tensor.scale
        output_zp = output_tensor.zero_point
        weight_scale = self.weight_fake_quant.scale
        weight_zp = self.weight_fake_quant.zero_point
        m, shift = get_mult_shift(input_scale, weight_scale, output_scale)
        if dump:
            # Dump weight & bias
            int_weight = torch.round((qweight/weight_scale) + weight_zp)
            int_bias = torch.round(self.bias / (input_scale * weight_scale))
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_weight', to_np(int_weight))
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_bias', to_np(int_bias))
            save_np(os.path.join(self.dump_folder, self.layer_name),'weight_scale', to_np(torch.tensor([weight_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'weight_zp', to_np(torch.tensor([weight_zp])))
            # Dump input feature
            int_input = torch.round((input_tensor / input_scale) + input_zp)
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_input', to_np(int_input))
            save_np(os.path.join(self.dump_folder, self.layer_name),'input_scale', to_np(torch.tensor([input_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'input_zp', to_np(torch.tensor([input_zp])))
        # Dump output feature
        # TODO: Only for 8-bit quantization
        # compatable to hw
        _tmp = torch.div((m * 256) * (prequant_output_tensor / (input_scale * weight_scale)), (2 ** (shift)), rounding_mode="floor") / 256
        _mask = torch.logical_or( (torch.frac(_tmp) == 0.5) , (torch.frac(_tmp) == -0.5))
        _tmp[_mask] = _tmp[_mask] + 0.5
        _tmp = torch.round(_tmp) + output_zp
        int_output = torch.clamp(_tmp, 0, 255)
        # int_output = torch.round(torch.clamp((m * (2 ** (-1 * shift)) * (prequant_output_tensor / (input_scale * weight_scale)) + output_zp), 0, 255))
        hwsim_output = output_scale * (int_output - output_zp)
        hwsim_output.scale = output_scale
        hwsim_output.zero_point = output_zp
        hwsim_output.HW_SIM = input_tensor.HW_SIM
        if dump:
            hwsim_output.DUMP = dump
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_output', to_np(int_output))
            save_np(os.path.join(self.dump_folder, self.layer_name),'output_scale', to_np(torch.tensor([output_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'output_zp', to_np(torch.tensor([output_zp])))
            # Dump multiplier & shifter
            save_np(os.path.join(self.dump_folder, self.layer_name),'multiplier', to_np(torch.round(torch.tensor([m * 256]))))
            save_np(os.path.join(self.dump_folder, self.layer_name),'shifter', to_np(torch.tensor([shift])))
        return hwsim_output

    def forward(self, input):
        qweight = self.weight_fake_quant(self.weight)
        weight_scale = self.weight_fake_quant.scale
        qbias = torch.round(self.bias / (input.scale * weight_scale)) * (input.scale * weight_scale)
        x = F.conv2d(input, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
        if self.training: self.act_fake_quant.enable_observer()
        else: self.act_fake_quant.disable_observer()
        x_prequant = x.detach().clone()
        x_quant = self.act_fake_quant(x_prequant)
        x_quant.scale = self.act_fake_quant.scale
        x_quant.zero_point = self.act_fake_quant.zero_point
        ''' HW_SIM '''
        if hasattr(input, 'HW_SIM') and input.HW_SIM:
            x_quant = self.hwsim(qweight, input, x_quant, x, dump=(hasattr(input, 'DUMP') and input.DUMP))
        #TODO: HW_SIM == False, gradient cannot propagate
        return x_quant


class QConvReLU2d(nnqat.Conv2d):

    """ Simulated Quantization Module for Conv2d + ReLU (or relu6)
    """
    _FLOAT_MODULE = nn.Conv2d
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', qconfig=None,
                 device=None, dtype=None, folder_name=None, layer_name=None) -> None:
        super(QConvReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode,
                         qconfig=qconfig)
        self.act_fake_quant = qconfig.activation()
        self.dump_folder = folder_name
        self.layer_name = layer_name

    @classmethod
    def from_float(cls, mod, dump_folder, layer_name):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        # if type(mod) == ConvReLU2d:
        #     mod = mod[0]
        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                       groups=mod.groups, bias=mod.bias is not None, padding_mode=mod.padding_mode,
                       qconfig=qconfig, folder_name=dump_folder, layer_name=layer_name)
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv
    
    def hwsim(self, qweight, input_tensor, output_tensor, prequant_output_tensor, dump=False):
        input_scale = input_tensor.scale
        input_zp = input_tensor.zero_point
        output_scale = output_tensor.scale
        output_zp = output_tensor.zero_point
        weight_scale = self.weight_fake_quant.scale
        weight_zp = self.weight_fake_quant.zero_point
        m, shift = get_mult_shift(input_scale, weight_scale, output_scale)
        if dump:
            # Dump weight & bias
            int_weight = torch.round((qweight/weight_scale) + weight_zp)
            int_bias = torch.round(self.bias / (input_scale * weight_scale))
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_weight', to_np(int_weight))
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_bias', to_np(int_bias))
            save_np(os.path.join(self.dump_folder, self.layer_name),'weight_scale', to_np(torch.tensor([weight_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'weight_zp', to_np(torch.tensor([weight_zp])))
            # Dump input feature
            int_input = torch.round((input_tensor / input_scale) + input_zp)
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_input', to_np(int_input))
            save_np(os.path.join(self.dump_folder, self.layer_name),'input_scale', to_np(torch.tensor([input_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'input_zp', to_np(torch.tensor([input_zp])))
        # Dump output feature
        # TODO: Only for 8-bit quantization
        _tmp = torch.div((m * 256) * (prequant_output_tensor / (input_scale * weight_scale)), (2 ** (shift)), rounding_mode="floor") / 256
        _mask = torch.logical_or( (torch.frac(_tmp) == 0.5) , (torch.frac(_tmp) == -0.5))
        # breakpoint()
        _tmp[_mask] = _tmp[_mask] + 0.5
        _tmp = torch.round(_tmp) + output_zp
        int_output = torch.clamp(_tmp, 0, 255)
        # int_output = torch.round(torch.clamp((m * (2 ** (-1 * shift)) * (prequant_output_tensor / (input_scale * weight_scale)) + output_zp), 0, 255))
        hwsim_output = output_scale * (int_output - output_zp)
        hwsim_output.scale = output_scale
        hwsim_output.zero_point = output_zp
        hwsim_output.HW_SIM = input_tensor.HW_SIM
        if dump:
            hwsim_output.DUMP = dump
            save_np(os.path.join(self.dump_folder, self.layer_name),'int_output', to_np(int_output))
            save_np(os.path.join(self.dump_folder, self.layer_name),'output_scale', to_np(torch.tensor([output_scale])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'output_zp', to_np(torch.tensor([output_zp])))
            # Dump multiplier & shifter
            save_np(os.path.join(self.dump_folder, self.layer_name),'multiplier', to_np(torch.round(torch.tensor([m * 256]))))
            save_np(os.path.join(self.dump_folder, self.layer_name),'shifter', to_np(torch.tensor([shift])))
        return hwsim_output

    def forward(self, input):
        qweight = self.weight_fake_quant(self.weight)
        weight_scale = self.weight_fake_quant.scale
        qbias = torch.round(self.bias / (input.scale * weight_scale)) * (input.scale * weight_scale)
        x = F.relu6(F.conv2d(input, qweight, qbias, self.stride, self.padding, self.dilation, self.groups))
        if self.training: self.act_fake_quant.enable_observer()
        else: self.act_fake_quant.disable_observer()
        x_prequant = x.detach().clone()
        x_quant = self.act_fake_quant(x_prequant)
        x_quant.scale = self.act_fake_quant.scale
        x_quant.zero_point = self.act_fake_quant.zero_point
        ''' HW_SIM '''
        if hasattr(input, 'HW_SIM') and input.HW_SIM:
            x_quant = self.hwsim(qweight, input, x_quant, x, dump=(hasattr(input, 'DUMP') and input.DUMP))
        return x_quant

class QMaxPool2d(nn.MaxPool2d):
    """ Simulated Quantization Module for 2D Max Pooling
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(QMaxPool2d, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                        return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, input):
        x = F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, 
                            self.ceil_mode, self.return_indices)
        x.scale = input.scale
        x.zero_point = input.zero_point
        if hasattr(input, 'HW_SIM') and input.HW_SIM: 
            x.HW_SIM = input.HW_SIM
        if hasattr(input, 'DUMP') and input.DUMP: 
            x.DUMP = input.DUMP
        return x

class QAvgPool2d(nn.AvgPool2d):
    """ Simulated Quantization Module for 2D Average Pooling
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode: bool = False, 
                    count_include_pad: bool = True, divisor_override=None, qconfig=None) -> None:
        super(QAvgPool2d, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                                        count_include_pad=count_include_pad, divisor_override=divisor_override)
        self.qconfig = qconfig
        self.act_fake_quant = qconfig.activation()
        self.weight_fake_quant = qconfig.weight()
    # TODO: hw_sim function
    def forward(self, input):

        x = F.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, 
                self.count_include_pad, self.divisor_override)
        
        if self.training: self.act_fake_quant.enable_observer()
        else: self.act_fake_quant.disable_observer()
        x_quant = self.act_fake_quant(x)
        x_quant.scale = input.scale
        x_quant.zero_point = input.zero_point
        if hasattr(input, 'HW_SIM') and input.HW_SIM: 
            x.HW_SIM = input.HW_SIM
        if hasattr(input, 'DUMP') and input.DUMP: 
            x.DUMP = input.DUMP
        return x_quant

class Q_Addition(nn.Module):
    def __init__(self, dump_folder, layer_name, qconfig):
        super(Q_Addition, self).__init__()
        self.dump_folder = dump_folder
        self.layer_name = layer_name
        self.qconfig = qconfig
        self.act_fake_quant = qconfig.activation()

    def hwsim(self, input1, input2, output, dump=False):
        input1_scale = input1.scale
        input1_zp = input1.zero_point
        input2_scale = input2.scale
        input2_zp = input2.zero_point
        output_scale = output.scale
        output_zp = output.zero_point
        m1, shift1 = get_mult_shift(input1_scale, 1, output_scale)
        m2, shift2 = get_mult_shift(input2_scale, 1, output_scale)
        if dump:
            save_np(os.path.join(self.dump_folder, self.layer_name), "input1_scale", to_np(input1_scale))
            save_np(os.path.join(self.dump_folder, self.layer_name), "input1_zp", to_np(input1_zp))
            save_np(os.path.join(self.dump_folder, self.layer_name), "input2_scale", to_np(input2_scale))
            save_np(os.path.join(self.dump_folder, self.layer_name), "input2_zp", to_np(input2_zp))
        # hw compatable
        int_input1 = torch.round(input1 / input1_scale + input1_zp)
        int_input2 = torch.round(input2 / input2_scale + input2_zp)
        _tmp1 = torch.div((int_input1 - input1_zp) * (m1 * 256), 2 ** (shift1), rounding_mode="floor") / 256
        _tmp2 = torch.div((int_input2 - input2_zp) * (m2 * 256), 2 ** (shift2), rounding_mode="floor") / 256
        _tmp_out = _tmp1 + _tmp2
        _mask = torch.logical_or((torch.frac(_tmp_out) == 0.5), (torch.frac(_tmp_out)==-0.5))
        _tmp_out[_mask] = _tmp_out[_mask] + 0.5
        int_output = torch.clamp(torch.round(_tmp_out) + output_zp, 0, 255)
        hwsim_output = output_scale * (int_output - output_zp)
        hwsim_output.scale = output_scale
        hwsim_output.zero_point = output_zp
        hwsim_output.HW_SIM = input1.HW_SIM
        if dump:
            hwsim_output.DUMP = dump
            save_np(os.path.join(self.dump_folder, self.layer_name), "int_output", to_np(int_output))
            save_np(os.path.join(self.dump_folder, self.layer_name), "output_scale", to_np(output_scale))
            save_np(os.path.join(self.dump_folder, self.layer_name), "output_zp", to_np(output_zp))
            # Dump multiplier & shifter
            save_np(os.path.join(self.dump_folder, self.layer_name),'1_multiplier', to_np(torch.round(torch.tensor([m1 * 256]))))
            save_np(os.path.join(self.dump_folder, self.layer_name),'1_shifter', to_np(torch.tensor([shift1])))
            save_np(os.path.join(self.dump_folder, self.layer_name),'2_multiplier', to_np(torch.round(torch.tensor([m2 * 256]))))
            save_np(os.path.join(self.dump_folder, self.layer_name),'2_shifter', to_np(torch.tensor([shift2])))
        return hwsim_output

    def forward(self, input_1, input_2):        
        output = input_1 + input_2
        if self.training: self.act_fake_quant.enable_observer()
        else: self.act_fake_quant.disable_observer()
        output_prequant = output.detach().clone()
        output_qaunt = self.act_fake_quant(output_prequant)
        output_qaunt.scale = self.act_fake_quant.scale
        output_qaunt.zero_point = self.act_fake_quant.zero_point
        ''' HW_SIM ''' 
        # we assert hw_sim is always true
        if hasattr(input_1, 'HW_SIM') and input_1.HW_SIM:
            output_qaunt = self.hwsim(input_1, input_2, output_qaunt, dump=(hasattr(input_1, 'DUMP') and input_1.DUMP))
        return output_qaunt


''' Auxiliary Functions for Hardware Simulation '''
def get_mult_shift(if_scale, w_scale, of_scale):
    m = if_scale * w_scale / of_scale
    shift = 0
    while m < 0.5:
        m *= 2
        shift += 1
    m = int(torch.round(m*256)) / 256
    return m, shift

def save_np(folder_path, file_name, input):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(os.path.join(folder_path, file_name+'.npy'), to_np(input))

def to_np(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    else:
        return input
