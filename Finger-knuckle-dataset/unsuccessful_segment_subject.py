import os
import shutil

samples = 6
data_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/GUI_Segment/Session_2/"
subjects = os.listdir(data_path)

less_subjects = []
more_subjects = []

for s in subjects:
    subject_path = os.path.join(data_path, s)
    images = os.listdir(subject_path)
    if len(images) > 6:
        more_subjects.append(s)
    elif len(images) < 6:
        less_subjects.append(s)

print("more subjects:_____________________")
print(more_subjects)
print('___________________________________')
print("less subjects:_____________________")
print(less_subjects)


source_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/Session_2/"
dst_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/unsuccessful/"
for less_s in less_subjects:
    source_subject = os.path.join(source_path, less_s)
    dst_subject = os.path.join(dst_path, less_s)
    if not os.path.exists(dst_subject):
        os.mkdir(dst_subject)
    images = os.listdir(source_subject)
    for i in images:
        image_file = os.path.join(source_subject, i)
        dst_image_file = os.path.join(dst_subject, i)
        shutil.copy(image_file, dst_image_file)

