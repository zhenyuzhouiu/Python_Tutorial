import cv2
import numpy as np
import torch
from scipy.ndimage import rotate
import datetime

image_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV1/Segmented/GUI_Seg/major/dataset/train_set/1/1_1-0.jpg"
image_path1 = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV1/Segmented/GUI_Seg/major/dataset/train_set/2/2_3-0.jpg"
image_path2 = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV1/Segmented/GUI_Seg/major/dataset/train_set/3/3_1-0.jpg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = np.expand_dims(image, axis=2)
image = image.repeat(512, axis=2)
image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
image1 = np.expand_dims(image1, axis=2)
image1 = image1.repeat(512, axis=2)
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
image2 = np.expand_dims(image2, axis=2)
image2 = image2.repeat(512, axis=2)

h, w, _ = image.shape



# src = np.concatenate((image, image1, image2), axis=2)
src = [image, image1, image2]
#
# for i in range (3):
#     cv2.imshow('src', src[:, :, 0 + 1*i:1 + 1*i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# src = image.repeat(3000, axis=2)
# src = src.astype(np.float64)
# mask = np.ones([w, h], dtype=np.float)

# rot = 5*np.pi/180
# centered_trans = np.float64([[ np.cos(rot), -np.sin(rot), 0],
#                              [np.sin(rot), np.cos(rot), 0],
#                              [ 0,           0,           1]])

starttime = datetime.datetime.now()
rotate_matrix = cv2.getRotationMatrix2D(center=(w/2, h/2), angle=5, scale=1)
mask = np.ones([w, h])
r_mask = cv2.warpAffine(mask, M=rotate_matrix, dsize=(w, h))

# overlap_h, overlap_w, overlap_bs = src.shape
# if overlap_bs > 512:
#     num_chuncks = overlap_bs // 512
#     num_reminder = overlap_bs % 512
#     r_ref2 = np.zeros(src.shape)
#     for nc in range(num_chuncks):
#         nc_ref2 = src[:, :, 0 + nc * 512:512 + nc * 512]
#         r_nc_ref2 = cv2.warpAffine(nc_ref2, M=M, dsize=[overlap_w, overlap_h])
#         r_ref2[:, :, 0 + nc * 512:512 + nc * 512] = r_nc_ref2
#     if num_reminder > 0:
#         nc_ref2 = src[:, :, 512 + nc * 512:]
#         r_nc_ref2 = cv2.warpAffine(nc_ref2, M=M, dsize=[overlap_w, overlap_h])
#         r_ref2[:, :, 512 + nc * 512:] = r_nc_ref2
# else:
#     r_ref2 = cv2.warpAffine(src, M=M, dsize=[overlap_w, overlap_h])
rotated_image = cv2.warpAffine(src=src, M=rotate_matrix, dsize=(w, h))
rotated_image = cv2
endtime = datetime.datetime.now()

# rotated_image = rotate(src, angle=5, reshape=False)
endtime2 = datetime.datetime.now()

print("cv rotation time: {}".format({(endtime-starttime).microseconds}))
print("scipy rotation time: {}".format({(endtime2-endtime).microseconds}))


for i in range (3):
    cv2.imshow('src', src[:, :, 0 + 1*i:1 + 1*i])
    cv2.imshow('rotate', rotated_image[:, :, 0+1*i:1+1*i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()






