import os, shutil

import cv2
import cv2 as cv
import sys

src_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV1/Segmented/GUI_Seg/major/Source_image/"
dst_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV1/Segmented/GUI_Seg/major/sub_image/"
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

image_name = os.listdir(src_path)

save = 0
for i in image_name:
    src_image_path = os.path.join(src_path, i)
    subject_name = i.split('_')[0]
    save_path = os.path.join(dst_path, subject_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    move_path = os.path.join(save_path, i)
    shutil.move(src_image_path, move_path)

    save += 1

print(save)



