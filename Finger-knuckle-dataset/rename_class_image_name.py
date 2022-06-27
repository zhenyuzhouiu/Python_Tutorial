import os
import shutil

src_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/HD/Original Images/class/"
dst_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/HD/Original Images/re_class/"
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

sub_name = os.listdir(src_path)
sub_no = 1
for s in sub_name:
    sub_path = os.path.join(src_path, s)
    dst_sub_path = os.path.join(dst_path, str(sub_no))
    if not os.path.exists(dst_sub_path):
        os.mkdir(dst_sub_path)
    image_name = os.listdir(sub_path)
    img_no = 1
    for i in image_name:
        image_path = os.path.join(sub_path, i)
        dst_image_path = os.path.join(dst_sub_path, str(img_no)+'.jpg')
        shutil.copy(image_path, dst_image_path)
        img_no += 1
    sub_no += 1