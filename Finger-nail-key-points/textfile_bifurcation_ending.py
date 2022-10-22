import os

bifurcation_path = "/home/zhenyuzhou/Desktop/YOLOv5/YOLOv5_OBB/imgs/blood_vessel/ending/02-09-2022-textfile-iou-0.4-conf-0.1/detection-bifurcation/result_txt/result_center/"
ending_path = "/home/zhenyuzhou/Desktop/YOLOv5/YOLOv5_OBB/imgs/blood_vessel/ending/02-09-2022-textfile-iou-0.4-conf-0.1/detection/result_txt/result_center/"

text_file = os.listdir(bifurcation_path)

for t in text_file:
    src_text_path = os.path.join(bifurcation_path, t)
    with open(src_text_path, 'r') as open_file:
        lines = open_file.readlines()
        dst_text_path = os.path.join(ending_path, t)
        with open(dst_text_path, 'a+') as write_file:
            contents = ''.join(lines)
            write_file.write(contents)
