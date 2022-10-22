import os

import cv2
import cv2 as cv
import sys

src_path = "/media/zhenyuzhou/My Passport/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/right-yolov5x-csl/right-index/"
dst_path = "/media/zhenyuzhou/My Passport/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/right-yolov5x-csl/right-index-resize/"
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

subject_name = os.listdir(src_path)
resize = 0


def show_image(src_image):
    cv.imshow('show image', src_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


for s in subject_name:
    subject_path = os.path.join(src_path, s)
    dst_subject_path = os.path.join(dst_path, s)
    if not os.path.exists(dst_subject_path):
        os.mkdir(dst_subject_path)
    image_list = os.listdir(subject_path)
    for i in image_list:
        src_image_path = os.path.join(subject_path, i)
        image = cv.imread(src_image_path)
        H, W, C = image.shape
        center_x = W/2
        center_y = H/2
        if H > W:
            size = W
        else:
            size = H
        crop_size = size/2
        if center_y-crop_size < 0:
            y = int(0)
        else:
            y = int(center_y-crop_size)

        if center_x-crop_size<0:
            x = int(0)
        else:
            x = int(center_x-crop_size)

        if center_y+crop_size > H:
            y1= int(H)
        else:
            y1 = int(center_y+crop_size)

        if center_x+crop_size > W:
            x1 = int(W)
        else:
            x1 = int(center_x+crop_size)

        square_img = image[y:y1, x:x1]
        resize_img = cv2.resize(square_img, (128,128))
        # show_image(flip_image)
        dst_image_path = os.path.join(dst_subject_path, i.split('.')[0]+'.jpg')
        cv.imwrite(dst_image_path, resize_img)
        resize += 1

print(resize)



