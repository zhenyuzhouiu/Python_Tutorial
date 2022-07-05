import os
import cv2

session1 = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/GUI_Segment/Session_1/"

session_subject = os.listdir(session1)
min_width = 99999999
min_height = 99999999
sum_width = 0
sum_height = 0
sum_num = 0
for s in session_subject:
    subject_path = os.path.join(session1, s)
    images = os.listdir(subject_path)
    for i in images:
        image_path = os.path.join(subject_path, i)
        read_image = cv2.imread(image_path)
        h, w, c = read_image.shape
        sum_width = sum_width + w
        sum_height = sum_height + h
        sum_num += 1
        if h < min_height:
            min_height = h
        if w < min_width:
            min_width = w

mean_width = sum_width / sum_num
mean_height = sum_height / sum_num
print("for the first session samples")
print(min_width)
print(min_height)
print(mean_width)
print(mean_height)
print(mean_width/mean_height)
print("-----------------------")

session2 = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/GUI_Segment/Session_2/"

session_subject = os.listdir(session2)
min_width = 99999999
min_height = 99999999
sum_width = 0
sum_height = 0
sum_num = 0
for s in session_subject:
    subject_path = os.path.join(session2, s)
    images = os.listdir(subject_path)
    for i in images:
        image_path = os.path.join(subject_path, i)
        read_image = cv2.imread(image_path)
        h, w, c = read_image.shape
        sum_width = sum_width + w
        sum_height = sum_height + h
        sum_num += 1
        if h < min_height:
            min_height = h
        if w < min_width:
            min_width = w

mean_width = sum_width / sum_num
mean_height = sum_height / sum_num
print("for the second session samples")
print(min_width)
print(min_height)
print(mean_width)
print(mean_height)
print(mean_width/mean_height)
print("-----------------------")
