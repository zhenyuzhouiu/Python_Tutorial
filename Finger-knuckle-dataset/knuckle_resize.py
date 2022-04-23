import os

import cv2
import cv2 as cv
import sys

src_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/Database/Segmented/Session_2/"
dst_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/Database/Segmented/Session_2_128/"
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

image_name = os.listdir(src_path)
flip = 0


def show_image(src_image):
    cv.imshow('show image', src_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


for i in image_name:
    image_path = os.path.join(src_path, i)
    image_list = os.listdir(image_path)
    for j in image_list:
        file_type = j.split('.')[-1]
        if file_type == "bmp":
            src_image_path = os.path.join(image_path, j)
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
            dst_image_folder = os.path.join(dst_path, i)
            if not os.path.exists(dst_image_folder):
                os.mkdir(dst_image_folder)
            dst_image_path = os.path.join(dst_image_folder, j)
            cv.imwrite(dst_image_path, resize_img)
            flip += 1

print(flip)



