import os

data_path = "/media/zhenyuzhou/Data/Project/Finger-Knuckle-Contactless/deep-learning/codekevin/CodeTest_Kevin_0926/code/4b-TIFS20/data_label_190/assistant-03-80-48/"

subject_path = os.listdir(data_path)

list_image = []
list_label = []

for s in subject_path:
    image_path = os.path.join(data_path, s)
    image_name = os.listdir(image_path)
    for i in image_name:
        image_file = os.path.join(image_path, i)
        label = ''.join(x for x in s if x.isdigit())
        list_image.append("assistant-03-80-48/" + s + "/" + i)
        list_label.append(int(label) - 1)


txt_path = "/media/zhenyuzhou/Data/Project/Finger-Knuckle-Contactless/deep-learning/codekevin/CodeTest_Kevin_0926/code/4b-TIFS20/data_label_190/assistant-03-80-48.txt"

with open(txt_path, 'a+') as f:
    for i in range(len(list_image)):
        f.write("{} {}\n".format(list_image[i], str(list_label[i])))
