import os
import cv2

data_path = "/media/zhenyuzhou/9096E35096E3357E/timeF60/007/kumar/"
dst_path = "/media/zhenyuzhou/9096E35096E3357E/timeF60/007/rotate/"

subjects_name = os.listdir(data_path)

for s in subjects_name:
    subject_path = os.path.join(data_path, s)
    dst_subject_path = os.path.join(dst_path, s)
    if not os.path.exists(dst_subject_path):
        os.mkdir(dst_subject_path)
    image_name = os.listdir(subject_path)
    for i in image_name:
        image_path = os.path.join(subject_path, i)
        dst_image_path = os.path.join(dst_subject_path, i)
        image = cv2.imread(image_path)
        rotate_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(dst_image_path, rotate_image)