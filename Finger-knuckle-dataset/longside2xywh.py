# convert the data label format
# from x, y, long-side, short-side, angle [0, 180)
# to x, y, w, h


import os
import numpy as np

import cv2

image_path = "/home/zhenyuzhou/Desktop/YOLOv5/YOLOv5_OBB/imgs/img_longside_cw/test/images/"
label_path = "/home/zhenyuzhou/Desktop/YOLOv5/YOLOv5_OBB/imgs/img_longside_cw/test/angle_labels/"
new_label_path = "/home/zhenyuzhou/Desktop/YOLOv5/YOLOv5_OBB/imgs/img_longside_cw/test/xywh_labels/"

image_name = os.listdir(image_path)
label_name = os.listdir(label_path)


def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
    """
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最长边
    @param shortside: 最短边
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x轴逆时针旋转碰到的第一条边最长边
            height: 与width不同的边
            theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    """
    if ((theta_longside >= -180) and (theta_longside < -90)):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height = shortside
        theta = theta_longside

    if (theta < -90) or (theta >= 0):
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)


def minAreaRect2boundingRect(rect):
    poly = np.float32(cv2.boxPoints(rect))
    poly = np.int0(poly)
    # cv2.boundingRect will return the left-top position (x, y) and (w, h)
    boudning_box = cv2.boundingRect(poly)
    return poly, boudning_box


for l in label_name:
    label_file = os.path.join(label_path, l)
    image_file = os.path.join(image_path, l.split('.')[0] + '.jpg')
    image = cv2.imread(image_file)
    h, w, c = image.shape
    new_label_file = os.path.join(new_label_path, l)
    with open(new_label_file, 'a+') as write_f:
        with open(label_file, 'r') as read_f:
            lines = read_f.readlines()
            for l in lines:
                class_n, x_c, y_c, longside, shortside, theta_longside = l.strip('\n').split(' ')
                # change the position value to image pixel
                x_c, y_c, longside, shortside, theta_longside = float(x_c) * w, float(y_c) * h, float(longside) * w, \
                                                                float(shortside) * h, float(theta_longside) - 179.9

                minarea_rect = longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside)
                poly, bounding_box = minAreaRect2boundingRect(minarea_rect)
                # the bounding box will return  the top-left position (x, y) and the (w, h)
                lefttop_x = bounding_box[0]
                lefttop_y = bounding_box[1]
                bbox_w = bounding_box[2]
                bbox_h = bounding_box[3]
                center_x = lefttop_x + bbox_w / 2
                center_y = lefttop_y + bbox_h / 2
                bounding_box_label = class_n + ' ' + str(center_x / w) + ' ' + str(center_y / h) + ' ' + \
                                     str(bbox_w / w) + ' ' + str(bbox_h / h) + '\n'
                write_f.write(bounding_box_label)

                pt1 = (int(center_x - bbox_w / 2), int(center_y - bbox_h / 2))
                pt2 = (int(center_x + bbox_w / 2), int(center_y + bbox_h / 2))
                cv2.rectangle(image, pt1, pt2, [0, 0, 255], thickness=3)
                # cv2.drawContours(image=image, contours=[poly], contourIdx=-1, color=[0, 0, 255], thickness=2)

    cv2.namedWindow('Bounding Box', cv2.WINDOW_NORMAL)
    cv2.imshow('Bounding Box', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
