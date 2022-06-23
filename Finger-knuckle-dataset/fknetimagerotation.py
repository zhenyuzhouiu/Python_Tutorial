import os
import cv2

src_path = "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Recognition/dataset/PolyUKnuckleV3/yolov5/caffe/train/Session_2-80-48/"

subject_name = os.listdir(src_path)

for s in subject_name:
    subject_path = os.path.join(src_path, s)
    image_list = os.listdir(subject_path)
    for i in image_list:
        image_path = os.path.join(subject_path, i)
        image = cv2.imread(image_path)
        for a in range(-10, 11):
            if a == 0:
                continue
            h, w, _ = image.shape
            matrix = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=a, scale=1)
            r_image = cv2.warpAffine(image, M=matrix, dsize=(w, h))
            save_path = os.path.join(subject_path, i.split(".")[0] + "_" + str(a) + "." + i.split(".")[-1])
            cv2.imwrite(save_path, r_image)
