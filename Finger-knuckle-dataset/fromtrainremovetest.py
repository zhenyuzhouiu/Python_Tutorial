import os

train_path = "/home/zhenyuzhou/Desktop/YOLOv5/datasets/finger_knuckle/labels/train/"
test_path = "/home/zhenyuzhou/Desktop/YOLOv5/datasets/finger_knuckle/labels/test/"

train_name = os.listdir(train_path)
test_name = os.listdir(test_path)

remove_number = 0

for train_n in train_name:
    if train_n in test_name:
        train_file = os.path.join(train_path, train_n)
        os.remove(train_file)
        remove_number += 1

print(remove_number)