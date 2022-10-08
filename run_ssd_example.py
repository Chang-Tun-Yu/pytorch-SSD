from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.utils.misc import Timer
import cv2
import sys
import os
import numpy


# if len(sys.argv) < 5:
#     print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
#     sys.exit(0)
# net_type = sys.argv[1]
# model_path = sys.argv[2]
# label_path = sys.argv[3]
# image_path = sys.argv[4]

# class_names = [name.strip() for name in open(label_path).readlines()]

# if net_type == 'vgg16-ssd':
#     net = create_vgg_ssd(len(class_names), is_test=True)
# elif net_type == 'mb1-ssd':
#     net = create_mobilenetv1_ssd(len(class_names), is_test=True)
# elif net_type == 'mb1-ssd-lite':
#     net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
# elif net_type == 'mb2-ssd-lite':
#     net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
# elif net_type == 'mb3-large-ssd-lite':
#     net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
# elif net_type == 'mb3-small-ssd-lite':
#     net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
# elif net_type == 'sq-ssd-lite':
#     net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
# else:
#     print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
#     sys.exit(1)
# net.load(model_path)

# if net_type == 'vgg16-ssd':
#     predictor = create_vgg_ssd_predictor(net, candidate_size=200)
# elif net_type == 'mb1-ssd':
#     predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
# elif net_type == 'mb1-ssd-lite':
#     predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
# elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
#     predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
# elif net_type == 'sq-ssd-lite':
#     predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
# else:
#     predictor = create_vgg_ssd_predictor(net, candidate_size=200)

# orig_image = cv2.imread(image_path)
# image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
# boxes, labels, probs = predictor.predict(image, 10, 0.4)

# boxes = boxes.cpu().detach().numpy()
# labels = labels.cpu().detach().numpy()
# probs = probs.cpu().detach().numpy()

# print(boxes)
# print(labels)
# print(probs)

# for i in range(boxes.shape[0]):
#     box = boxes[i, :].astype(int)
#     cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
#     #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
#     label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
#     cv2.putText(orig_image, label,
#                 (box[0] + 20, box[1] + 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,  # font scale
#                 (255, 0, 255),
#                 2)  # line type
# path = "run_ssd_example_output.jpg"
# cv2.imwrite(path, orig_image)
# print(f"Found {len(probs)} objects. The output image is {path}")


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

    for name, layer in net.named_modules():
        print(name, layer)

    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    else:
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)

    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)

    boxes = boxes.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    probs = probs.cpu().detach().numpy()

    for i in range(boxes.size(0)):
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
    cv2.imwrite(path, orig_image)
    print(f"Found {len(probs)} objects. The output image is {path}")

    # with open(os.path.join("C:/Users/user/Desktop/ans", "{}.txt".format(i+1)), "w") as f:
    #     f.write(str(boxes.shape[0]))
    #     f.write("\n")
    #     for j in range(boxes.shape[0]):
    #         f.write(str(boxes[j][0]))
    #         f.write(" ")
    #         f.write(str(boxes[j][1]))
    #         f.write(" ")
    #         f.write(str(boxes[j][2]))
    #         f.write(" ")
    #         f.write(str(boxes[j][3]))
    #         f.write(" ")
    #         f.write(str(labels[j]))
    #         f.write(" ")
    #         f.write(str(probs[j]))
    #         f.write("\n")

print(class_names)