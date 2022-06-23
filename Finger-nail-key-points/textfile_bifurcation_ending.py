import os

bifurcation_path = "/home/zhenyuzhou/Desktop/YOLOv5_OBB/imgs/blood_vessel/textfile/train-textfile/result_center_bifurcation/"
ending_path = "/home/zhenyuzhou/Desktop/YOLOv5_OBB/imgs/blood_vessel/textfile/train-textfile/result_center_ending/"

text_file = os.listdir(bifurcation_path)

for t in text_file:
    src_text_path = os.path.join(bifurcation_path, t)
    with open(src_text_path, 'r') as open_file:
        lines = open_file.readlines()
        dst_text_path = os.path.join(ending_path, t)
        with open(dst_text_path, 'a+') as write_file:
            contents = ''.join(lines)
            write_file.write(contents)
