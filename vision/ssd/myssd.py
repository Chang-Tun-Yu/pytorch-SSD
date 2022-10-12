import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.quantization as quantization
import numpy as np
from typing import List, Tuple

from ..utils import box_utils
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

class TransformInput(nn.Module):
    def __init__(self):
        super(TransformInput, self).__init__()
    
    def forward(sefl, x):
        return (x - 127.0) / 128.0
class MySSD(nn.Module):
    def __init__(self, num_classes: int, mobilenet_0: nn.Sequential, mobilenet_1: nn.Sequential,
                 extra_net: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(MySSD, self).__init__()
        self.transform = TransformInput()
        self.num_classes = num_classes
        self.mobilenet_0 = mobilenet_0
        self.mobilenet_1 = mobilenet_1
        self.extras = extra_net
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config
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

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

        self.is_quantized = False
        self.input_fake_quant = self.qconfig.activation()
        self.HW_SIM = False
        self.DUMP = False
        
    def hw_sim(self, hwsim, dump):
        self.HW_SIM = hwsim
        self.DUMP = dump
            
    def forward(self, x: torch.Tensor): # -> Tuple[torch.Tensor, torch.Tensor]:
        # quantization setting
        # if self.training: self.input_fake_quant.enable_observer()
        # else: self.input_fake_quant.disable_observer()
        # x = self.input_fake_quant(x)
        # x.scale = self.input_fake_quant.scale.clone()
        # x.zero_point = self.input_fake_quant.zero_point.clone()
        # x.HW_SIM = self.HW_SIM
        # x.DUMP = self.DUMP 

        
        x_t = self.transform(x)
        x_t.scale = 1 / 128.0
        x_t.zero_point = 127
        x_t.HW_SIM = self.HW_SIM
        x_t.DUMP = self.DUMP
        x_t.testing = self.is_test        

        confidences = []
        locations = []
        header_index = 0
        # TODO: quantize of input
        y = self.mobilenet_0(x_t)
        confidence, location = self.compute_header(header_index, y)
        confidences.append(confidence)
        locations.append(location)
        header_index += 1

        z = self.mobilenet_1(y)
        confidence, location = self.compute_header(header_index, z)
        confidences.append(confidence)
        locations.append(location)
        header_index += 1
        
        for i in range(4):
            z = self.extras[i](z)
            confidence, location = self.compute_header(header_index, z)
            confidences.append(confidence)
            locations.append(location)
            header_index += 1

        
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)

        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


