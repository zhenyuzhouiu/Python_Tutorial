import os

import cv2
import cv2 as cv
import sys

src_path = "/home/zhenyuzhou/Pictures/Ranking-system/Flip_PolyUKnuckleV1/major/Flip_image/"
dst_path = "/home/zhenyuzhou/Pictures/Ranking-system/Flip_PolyUKnuckleV1/major/hist_Flip_image/"
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

image_name = os.listdir(src_path)
flip = 0


def show_image(src_image):
    cv.imshow('show image', src_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


for i in image_name:
    src_image_path = os.path.join(src_path, i)
    image = cv.imread(src_image_path, cv2.IMREAD_GRAYSCALE)
    # show_image(image)
    # flip_image = cv.flip(image, 1)
    hist_image = cv.equalizeHist(image)
    # show_image(flip_image)
    dst_image_path = os.path.join(dst_path, i)
    cv.imwrite(dst_image_path, hist_image)
    flip += 1

print(flip)



