import os, shutil

src_path = '/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/HD/Original Images/GUI_Segment/sub_major'
train_path = '/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/HD/Original Images/GUI_Segment/train_set'
test_path = '/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/HD/Original Images/GUI_Segment/test_set'

sub_name = os.listdir(src_path)

for i in sub_name:
    sub_path = os.path.join(src_path, i)
    img_name = os.listdir(sub_path)
    num = 0
    for j in img_name:
        if num == 5:
            break
        img_path = os.path.join(sub_path, j)
        if num < 1:
            train_sub = os.path.join(train_path, i)
            if not os.path.exists(train_sub):
                os.mkdir(train_sub)
            train_img = os.path.join(train_sub, j)
            shutil.copy(img_path, train_img)
        else:
            test_sub = os.path.join(test_path, i)
            if not os.path.exists(test_sub):
                os.mkdir(test_sub)
            test_img = os.path.join(test_sub, j)
            shutil.copy(img_path, test_img)
        num += 1









