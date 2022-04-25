import random

train_txt = '/home/zhenyuzhou/Desktop/darknet/data/train.txt'
new_train_txt = '/home/zhenyuzhou/Desktop/darknet/data/new_train.txt'
new_valid_txt = '/home/zhenyuzhou/Desktop/darknet/data/new_valid.txt'

num_lines = 0
list_lines = []
valid_lines = []
with open(train_txt, 'r') as f:
    num_lines = len(f.readlines())
    list_lines = []
    x = 0
    for i in range(num_lines):
        list_lines.append(x)
        x += 1
    valid_lines = random.sample(list_lines, int(num_lines*0.2))

with open(train_txt, 'r') as f:
    for i in range(num_lines):
        # 去除x64/release
        data = f.readline()[12:]
        if i in valid_lines:
            with open(new_valid_txt, 'a') as valid_f:
                valid_f.write(data)
        else:
            with  open(new_train_txt, 'a') as train_f:
                train_f.write(data)
