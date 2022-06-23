import os

data_path = "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/CodeTest_Kevin_0926/code/" \
            "4b-TIFS20/data_label_190/fkv3_yolov5_session2_80_48/"

subject_path = os.listdir(data_path)

list_image = []
list_label = []

for s in subject_path:
    image_path = os.path.join(data_path, s)
    image_name = os.listdir(image_path)
    for i in image_name:
        image_file = os.path.join(image_path, i)
        label = ''.join(x for x in s if x.isdigit())
        list_image.append("fkv3_yolov5_session2_80_48/" + s + "/" + i)
        list_label.append(int(label) - 1)


txt_path = "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/CodeTest_Kevin_0926/code/4b-TIFS20/" \
            "data_label_190/fkv3_yolov5_session2_80_48.txt"

with open(txt_path, 'a+') as f:
    for i in range(len(list_image)):
        f.write("{} {}\n".format(list_image[i], str(list_label[i])))
