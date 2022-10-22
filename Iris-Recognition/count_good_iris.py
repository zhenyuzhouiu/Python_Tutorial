import os

src_path = "/home/zhenyuzhou/Desktop/Iris-Recognition/Eddie_ARVR_Code/Code/Train_Test/Preprocessing/Quality/Result/S1"
part_name = os.listdir(src_path)
L = 0
R = 0
for p in part_name:
    part_path = os.path.join(src_path, p)
    subject_name = os.listdir(part_path)
    for s in subject_name:
        subject_path = os.path.join(part_path, s)
        left_right_name = os.listdir(subject_path)
        for l_r in left_right_name:
            lr_path = os.path.join(subject_path, l_r)
            image_name = os.listdir(lr_path)
            image_number = len(image_name)
            if l_r == "L":
                L += image_number
            elif l_r == "R":
                R += image_number
            else:
                print("wrong left right information")

print('L: ' + str(L))
print('R: ' + str(R))
