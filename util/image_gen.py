import cv2
orig_image = cv2.imread("D:/10_dataset/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000018.jpg")
orig_image = cv2.resize(orig_image, (300, 300))
# image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
padding_image = cv2.copyMakeBorder(orig_image, 0, 20, 0, 20, cv2.BORDER_CONSTANT, value=0)
cv2.imwrite("../orig.jpg", orig_image)
cv2.imwrite("../padd.jpg", padding_image)