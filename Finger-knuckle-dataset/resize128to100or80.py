import os
import cv2

data_path = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg-128/01/"

dst_path_test = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg-128/01_100_70/"
dst_path_train = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg-128/01_80_48/"
if not os.path.exists(dst_path_test):
    os.mkdir(dst_path_test)
if not os.path.exists(dst_path_train):
    os.mkdir(dst_path_train)

subject_path = os.listdir(data_path)

subject_number = 1
for s in subject_path:
    image_path = os.listdir(os.path.join(data_path, s))
    image_number = 1
    for i in image_path:
        image_file = os.path.join(data_path, s, i)
        image = cv2.imread(image_file)
        # change image 128x128 to 100x70
        # step1:-> rotation 90 degree
        image_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # image_90 = image
        # step2:-> keep ration 128x128->128x90
        h, w, _ = image_90.shape
        center_w = int(w/2)
        center_h = int(h/2)
        keep_ration_h = int((w * 70) / 100)
        crop_image = image_90[int((h-keep_ration_h)/2):int((h-keep_ration_h)/2)+keep_ration_h, 0:w]
        image_100_70 = cv2.resize(crop_image, (100, 70))
        save_path = os.path.join(dst_path_test, str(subject_number))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file = os.path.join(save_path, str(image_number) + '.' + i.split('.')[-1])
        cv2.imwrite(save_file, image_100_70)

        # step3:-> generate the 80x48 training image
        image_80_64 = image_100_70[10:58, 9:89]
        save_path = os.path.join(dst_path_train, str(subject_number))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file = os.path.join(save_path, str(image_number) + '.' + i.split('.')[-1])
        cv2.imwrite(save_file, image_80_64)
        image_number += 1
    subject_number += 1







