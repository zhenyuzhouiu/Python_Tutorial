import os
import shutil

src_path = "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/CodeTest_Kevin_0926/processed_database/" \
           "data_label_190/data_render2_190_with1_down_s2_ori/"
dst_path ="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Recognition/dataset/3DFingerKnuckle/3D/"
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

subject_name = os.listdir(src_path)

for i in subject_name:
    for j in ['set1', 'set2', 'set3', 'set4', 'set5', 'set6']:
        image_file = os.path.join(src_path, i, 'session2', 'forefinger', j, 'img_0.bmp')
        dst_subject = os.path.join(dst_path, i)
        if not os.path.exists(dst_subject):
            os.mkdir(dst_subject)
        dst_file = os.path.join(dst_subject, "image"+j+".bmp")
        shutil.copy(image_file, dst_file)
