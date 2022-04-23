# step 1: read image
# step 2: upload image data to gpu
# step 3: image processing with cuda
# step 4: download image from gpu to cpu

import cv2
import numpy as np
import datetime

def first_demo():

    # how to use gpu to accelerate opencv
    # step 1: read image
    frame=cv2.imread('/home/zhenyuzhou/Pictures/HR.png')

    # step 2: upload image data to gpu
    gpu_frame=cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    print(gpu_frame.cudaPtr())

    # step 3: image processing with cuda
    gpu_resframe=cv2.cuda.resize(gpu_frame,(1024,512))

    # step 4: download image from gpu to cpu
    cpu_resfram=gpu_resframe.download()
    print(cpu_resfram.shape)


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

src = np.concatenate((image, image1, image2), axis=2)
rotate_matrix = cv2.getRotationMatrix2D(center=(w/2, h/2), angle=5, scale=1)

# opencv with cpu to warpaffine images
starttime = datetime.datetime.now()
overlap_h, overlap_w, overlap_bs = src.shape
if overlap_bs > 512:
    num_chuncks = overlap_bs // 512
    num_reminder = overlap_bs % 512
    rotated_image = np.zeros(src.shape)
    for nc in range(num_chuncks):
        nc_ref2 = src[:, :, 0 + nc * 512:512 + nc * 512]
        r_nc_ref2 = cv2.warpAffine(nc_ref2, M=rotate_matrix, dsize=[overlap_w, overlap_h])
        rotated_image[:, :, 0 + nc * 512:512 + nc * 512] = r_nc_ref2
    if num_reminder > 0:
        nc_ref2 = src[:, :, 512 + nc * 512:]
        r_nc_ref2 = cv2.warpAffine(nc_ref2, M=rotate_matrix, dsize=[overlap_w, overlap_h])
        rotated_image[:, :, 512 + nc * 512:] = r_nc_ref2
else:
    rotated_image = cv2.warpAffine(src, M=rotate_matrix, dsize=[overlap_w, overlap_h])
# rotated_image = cv2.warpAffine(src=src, M=rotate_matrix, dsize=(w, h))

endtime = datetime.datetime.now()


# opencv with gpu to warpaffine images
overlap_h, overlap_w, overlap_bs = src.shape
if overlap_bs > 5:
    num_chuncks = overlap_bs // 5
    num_reminder = overlap_bs % 5
    rotated_image = np.zeros(src.shape)
    for nc in range(num_chuncks):
        nc_ref2 = src[:, :, 0 + nc * 5:5 + nc * 5]
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(nc_ref2)
        gpu_resframe = cv2.cuda.warpAffine(gpu_frame, M=rotate_matrix, dsize=[overlap_w, overlap_h])
        cpu_resfram = gpu_resframe.download()
        rotated_image[:, :, 0 + nc * 5:5 + nc * 5] = cpu_resfram
    if num_reminder > 0:
        nc_ref2 = src[:, :, 5 + nc * 5:]
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(nc_ref2)
        gpu_resframe = cv2.cuda.warpAffine(gpu_frame, M=rotate_matrix, dsize=[overlap_w, overlap_h])
        cpu_resfram = gpu_resframe.download()
        rotated_image[:, :, 5 + nc * 5:] = cpu_resfram
else:
    rotated_image = cv2.warpAffine(src, M=rotate_matrix, dsize=[overlap_w, overlap_h])




endtime2 = datetime.datetime.now()

print("cv rotation time: {}".format({(endtime-starttime).microseconds}))
print("cv with gpu acceleration rotation time: {}".format({(endtime2-endtime).microseconds}))



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

# for i in range (3):
#     cv2.imshow('src', src[:, :, 0 + 1*i:1 + 1*i])
#     cv2.imshow('rotate', rotated_image[:, :, 0+1*i:1+1*i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()






