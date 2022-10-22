import json
import os

label_path = '/home/zhenyuzhou/Desktop/YOLOv5/YOLOv5_OBB/imgs/blood_vessel/label_json/'

list_path = os.listdir(label_path)

endpoints = []
bifurcation = []
dots = []

for i in list_path:
    json_path = os.path.join(label_path, i)
    if json_path.endswith(".json"):
        with open(json_path, 'rb') as load_f:
            load_dict = json.load(load_f)
            label_name = load_dict["shapes"]

            i, j, k = 0, 0, 0
            for _ in label_name:
                if _["label"] == "End" or _["label"] == "end":
                    i = i + 1
                if _["label"] == "Junction" or _["label"] == "junction":
                    j = j + 1
                if _["label"] == "Dot" or _["label"] == "Point":
                    k = k + 1

            if i == 550:
                print(json_path)
            if j == 358:
                print(json_path)

            # endpoints = endpoints + i
            # bifurcation = bifurcation + j
            # dots = dots + k
            endpoints.append(i)
            bifurcation.append(j)
            dots.append(k)
    else:
        pass

print("Total Ending Point Number: " + str(sum(endpoints)) +
      "Min Ending Point for One Image: " + str(min(endpoints)) +
      "Max Ending Point for One Image: " + str(max(endpoints)) +
      "Average Ending Point for One Image: " + str(sum(endpoints)/84))

print("Total Bifurcation Number: " + str(sum(bifurcation)) +
      "Min Bifurcation for One Image: " + str(min(bifurcation)) +
      "Max Bifurcation for One Image: " + str(max(bifurcation)) +
      "Average Bifurcation for One Image: " + str(sum(bifurcation)/84))

print("Total Dot Number: " + str(sum(dots)) +
      "Min Dot for One Image: " + str(min(dots)) +
      "Max Dot for One Image: " + str(max(dots)) +
      "Average Dot for One Image: " + str(sum(dots)/84))
