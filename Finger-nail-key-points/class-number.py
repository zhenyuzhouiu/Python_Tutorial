import json
import os

label_path = '/home/zhenyuzhou/Pictures/Blood-Vessel-Key-Points/labeled-images/'

list_path = os.listdir(label_path)

endpoints = 0
bifurcation = 0
dots = 0

for i in list_path:
    json_path = os.path.join(label_path, i)
    if json_path.endswith(".json"):
        with open(json_path, 'rb') as load_f:
            load_dict = json.load(load_f)
            label_name = load_dict["shapes"]

            i, j, k = 0, 0, 0
            for _ in label_name:
                i = i + 1 if _["label"] == "End" else i
                j = j + 1 if _["label"] == "Junction" else j
                k = k + 1 if _["label"] == "Dot" or "Point" else k

            endpoints = endpoints + i
            bifurcation = bifurcation + j
            dots = dots + k
    else:
        pass

print("endpoints, bifurcation, dots = " + str(endpoints) + ', ' + str(bifurcation) + ', ' + str(dots))
