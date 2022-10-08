import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

from ..utils import box_utils
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

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
class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None, image_i=0):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config
        self.image_i = image_i

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            # with open("./tensor_comp/config.txt", "w") as f:
            #     f.write("image_size ")
            #     f.write(str(config.image_size))
            #     f.write(str("\n"))
            #     f.write("image_mean ")
            #     f.write(str(config.image_mean[0]))
            #     f.write(str(" "))
            #     f.write(str(config.image_mean[1]))
            #     f.write(str(" "))
            #     f.write(str(config.image_mean[2]))
            #     f.write(str("\n"))
            #     f.write("image_std ")
            #     f.write(str(config.image_std))
            #     f.write(str("\n"))
            #     f.write("iou_threshold ")
            #     f.write(str(config.iou_threshold))
            #     f.write(str("\n"))
            #     f.write("center_variance ")
            #     f.write(str(config.center_variance))
            #     f.write(str("\n"))
            #     f.write("size_variance ")
            #     f.write(str(config.size_variance))
            #     f.write(str("\n"))
            # save prior information
            # with open("./tensor_comp/prior.txt", "w") as f:
            #     f.write(str(config.priors.shape[0]))
            #     f.write("\n")
            #     for i in range(config.priors.shape[0]):
            #         f.write(str(config.priors[i, 0].item()))
            #         f.write(" ")
            #         f.write(str(config.priors[i, 1].item()))
            #         f.write(" ")
            #         f.write(str(config.priors[i, 2].item()))
            #         f.write(" ")
            #         f.write(str(config.priors[i, 3].item()))
            #         f.write(" ")
            #         f.write("\n")
            self.priors = config.priors.to(self.device)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        # _confidence = confidences.clone().detach()
        # _confidence = _confidence.cpu().detach().numpy()
        # with open("./tensor_comp/output/conf.txt", "w") as f:
        #     for k in range(_confidence.shape[1]):
        #         for j in range(self.num_classes):
        #             f.write(str(_confidence[0, k, j]))
        #             f.write("\n")
        # _location = locations.clone().detach()
        # _location = _location.cpu().detach().numpy()
        # with open("./tensor_comp/output/loc.txt", "w") as f:
        #     for k in range(_location.shape[1]):
        #         for j in range(4):
        #             f.write(str(_location[0, k, j]))
        #             f.write("\n")
        # print("confidences dimenetion:", confidences.size())
        # print("locations dimenetion:", locations.size())
        
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            # output
            # _confidence = confidences.clone().detach()
            # _confidence = _confidence.cpu().detach().numpy()
            # with open("./tensor_comp/ssd/conf.txt", "w") as f:
            #     for k in range(_confidence.shape[1]):
            #         for j in range(self.num_classes):
            #             f.write(str(_confidence[0, k, j]))
            #             f.write("\n")
            # _location = boxes.clone().detach()
            # _location = _location.cpu().detach().numpy()
            # with open("./tensor_comp/ssd/loc.txt", "w") as f:
            #     for k in range(_location.shape[1]):
            #         for j in range(4):
            #             f.write(str(_location[0, k, j]))
            #             f.write("\n")
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        _type = "orig"
        confidence = self.classification_headers[i](x)
        # torch.save(confidence, "tensor_comp/{}/{}.pt".format(_type, "confidence"+str(i)))
        _confidence = confidence.clone().detach()
        _confidence = _confidence.cpu().detach().numpy()
        _confidence = encode_conv2d_IF(_confidence[0])
        # with open("C:/Users/user/Desktop/tensor/{}/conf_{}.txt".format(self.image_i ,i), "w") as f:
        #     for j in range(_confidence.shape[0]):
        #         f.write(str(_confidence[j]))
        #         f.write("\n")
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        # torch.save(location, "tensor_comp/{}/{}.pt".format(_type, "location"+str(i)))
        _location = location.clone().detach()
        _location = _location.cpu().detach().numpy()
        _location = encode_conv2d_IF(_location[0])
        # with open("C:/Users/user/Desktop/tensor/{}/loc_{}.txt".format(self.image_i, i), "w") as f:
        #     for j in range(_location.shape[0]):
        #         f.write(str(_location[j]))
        #         f.write("\n")
        # print("location {} size:".format(i), location.size())
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

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


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
