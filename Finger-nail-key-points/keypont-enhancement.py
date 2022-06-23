import os
import cv2
import matplotlib.pyplot as plt

data_path = '/home/zhenyuzhou/Desktop/YOLOv5_OBB/imgs/blood_vessel/images/'

image_name = os.listdir(data_path)
alpha = 0.5
beta = 100
for i in image_name:
    image_path = os.path.join(data_path, i)
    src_image = cv2.imread(image_path)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    scale_image = cv2.convertScaleAbs(src_image, alpha=alpha, beta=beta)
    # the opencv read image with BGR
    hsv_image = cv2.cvtColor(scale_image, cv2.COLOR_BGR2HSV)
    plt.subplot(2, 1, 1)
    plt.imshow(src_image)
    plt.subplot(2, 1, 2)
    plt.imshow(hsv_image)
    plt.show()

