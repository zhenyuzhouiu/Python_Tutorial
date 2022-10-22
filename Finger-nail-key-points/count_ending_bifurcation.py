import os

src_path = "/home/zhenyuzhou/Desktop/YOLOv5/YOLOv5_OBB/imgs/blood_vessel/ending/02-09-2022-textfile-iou-0.4-conf-0.1/detection/result_txt/result_center/"
ending = []
bifurcation = []
text_name = os.listdir(src_path)
for t in text_name:
    text_file = os.path.join(src_path, t)
    with open(text_file, 'r') as open_file:
        lines = open_file.readlines()
        e = 0
        b = 0
        for l in lines:
            type = int(float(l.split(' ')[3]))
            if type == 1:
                e += 1
            if type == 2:
                b += 1

        ending.append(e)
        bifurcation.append(b)


print("Total Ending Point Number: " + str(sum(ending)) +
      "Min Ending Point for One Image: " + str(min(ending)) +
      "Max Ending Point for One Image: " + str(max(ending)) +
      "Average Ending Point for One Image: " + str(sum(ending)/75))

print("Total Bifurcation Number: " + str(sum(bifurcation)) +
      "Min Bifurcation for One Image: " + str(min(bifurcation)) +
      "Max Bifurcation for One Image: " + str(max(bifurcation)) +
      "Average Bifurcation for One Image: " + str(sum(bifurcation)/75))

