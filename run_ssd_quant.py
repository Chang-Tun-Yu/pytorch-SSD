from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.mobilenet_v2_ssd_quant import create_mobilenetv2_ssd_qunat, create_mobilenetv2_ssd_quant_predictor

from vision.utils.misc import Timer
import cv2
import sys
import os
import numpy
import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn

if len(sys.argv) < 3:
    print('Usage: python run_ssd_quant.py <model path> <label path>')
    sys.exit(0)
# net_type = sys.argv[1]
model_path = sys.argv[1]
label_path = sys.argv[2]

for i in range(1, 10):
    
    image_path = os.path.join("D:/10_dataset/VOC/VOC2007_test/JPEGImages", f'{i:06}.jpg')
    # os.mkdir("C:/Users/user/Desktop/tensor/{}".format(i+1))

    class_names = [name.strip() for name in open(label_path).readlines()]

    float_net = create_mobilenetv2_ssd_lite(21, is_test=True, image_i=0)
    float_net.load("./models/mb2-ssd-lite-mp-0_686.pth")
    float_net = float_net.eval()
    modules_to_fuse = []
    with open("fuse_list.txt", "r") as f:
        for l in f:
            l_list = l.strip().split()
            modules_to_fuse.append(l_list)
    torch.ao.quantization.fuse_modules(float_net, modules_to_fuse, inplace=True)
    net = create_mobilenetv2_ssd_qunat(len(class_names), float_net, "./quant_dump", is_test=True)


    # if net_type == 'vgg16-ssd':
    #     net = create_vgg_ssd(len(class_names), is_test=True)
    # elif net_type == 'mb1-ssd':
    #     net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    # elif net_type == 'mb1-ssd-lite':
    #     net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    # elif net_type == 'mb2-ssd-lite':
    #     net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True, image_i=i+1)
    # elif net_type == 'mb3-large-ssd-lite':
    #     net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
    # elif net_type == 'mb3-small-ssd-lite':
    #     net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
    # elif net_type == 'sq-ssd-lite':
    #     net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    # else:
    #     print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    #     sys.exit(1)
    net.load(model_path)



#     if net_type == 'vgg16-ssd':
#         predictor = create_vgg_ssd_predictor(net, candidate_size=200)
#     elif net_type == 'mb1-ssd':
#         predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
#     elif net_type == 'mb1-ssd-lite':
#         predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
#     elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
#         predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
#     elif net_type == 'sq-ssd-lite':
#         predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
#     else:
#         predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    predictor = create_mobilenetv2_ssd_quant_predictor(net, candidate_size=200)
    if not os.path.isfile(image_path):
        continue
    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)

    boxes = boxes.cpu().detach().numpy().astype(numpy.int32)
    labels = labels.cpu().detach().numpy()
    probs = probs.cpu().detach().numpy()

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(orig_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    path = "run_ssd_example_output.jpg"
    cv2.imshow("fjfadojf", orig_image)
    cv2.waitKey(0)
    print(f"Found {len(probs)} objects. The output image is {path}")


# print(class_names)