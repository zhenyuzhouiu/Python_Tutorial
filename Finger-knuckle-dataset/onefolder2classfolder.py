import os
import shutil

src_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/HD/Original Images/502-712(segment)/major/502-712/"
dst_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/HD/Original Images/502-712(segment)/major/class/"
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

image_name = os.listdir(src_path)

for i in image_name:
    class_name = i.split("_")[0]
    class_path = os.path.join(dst_path, class_name)
    if not os.path.exists(class_path):
        os.mkdir(class_path)
    src_image = os.path.join(src_path, i)
    dst_image = os.path.join(class_path, i)
    shutil.copy(src_image, dst_image)
