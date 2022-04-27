# Pytorch中的仿射变换(affine_grid)
# https://www.jianshu.com/p/723af68beb2e

# 下面我们将通过分别通过手动编码和pytorch方式对该图片进行平移、旋转、转置、缩放等操作，这些操作的数学原理在本文中不会详细讲解。

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
from torch.nn import functional

# 旋转操作

image_path = "/home/zhenyuzhou/Pictures/timberlake.jpg"
# transforms.ToTensor()
# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.
# FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes
# (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
# In the other cases, tensors are returned without scaling.
image_torch = transforms.ToTensor()(Image.open(image_path))
plt.subplot(1, 4, 1)
plt.imshow(image_torch.numpy().transpose(1, 2, 0))

rotate_angle = 30 * math.pi / 180
rotate_matrix = np.array([[math.cos(rotate_angle), -math.sin(rotate_angle), 0],
                          [math.sin(rotate_angle), math.cos(rotate_angle), 0]])

# # 普通操作
# t1 = rotate_matrix[:, 0:2]
# t2 = rotate_matrix[:, 2]
# rotate_image = torch.zeros_like(image_torch)
# # image_torch.size() or image_torch.shape
# c, h, w = image_torch.shape
# for y in range(h):
#     for x in range(w):
#         pos = np.array([[x],
#                         [y]])
#         new_pos = t1 @ pos + t2
#         new_x = int(new_pos[0][0])
#         new_y = int(new_pos[1][0])
#         if 0 <= new_x < w and 0 <= new_y < h:
#             rotate_image[:, new_y, new_x] = image_torch[:, y, x]
# plt.imshow(rotate_image.numpy().transpose(1, 2, 0))
# plt.show()

# pytorch操作
# 要使用 pytorch 的平移操作，只需要两步：
#   1. 创建 grid：grid = torch.nn.functional.affine_grid(theta, size)，
#   其实我们可以通过调节 size 设置所得到的图像的大小(相当于resize)；
#   2. grid_sample 进行重采样：outputs = torch.nn.functional.grid_sample(inputs, grid, mode='bilinear')
# rotate_matrix 的第三列为平移比例，向右为负，向下为负；
c, h, w = image_torch.shape
rotate_matrix = np.array([[math.cos(rotate_angle), -math.sin(rotate_angle), 0],
                          [math.sin(rotate_angle), math.cos(rotate_angle), 0]])
theta = torch.tensor(rotate_matrix, dtype=torch.float)
# align_corners: 角点是否对齐
# https://blog.csdn.net/shwan_ma/article/details/108830991?utm_medium=distribute.pc_relevant.none-task-blog-title-6&spm=1001.2101.3001.4242
# Here is a simple illustration I made showing how a 4x4 image is upsampled to 8x8.
# When align_corners=True, pixels are regarded as a grid of points. Points at the corners are aligned.
# When align_corners=False, pixels are regarded as 1x1 areas. Area boundaries, rather than their centers, are aligned.
grid = functional.affine_grid(theta.unsqueeze(0), image_torch.unsqueeze(0).size(), align_corners=False)
outputs = functional.grid_sample(image_torch.unsqueeze(0), grid, align_corners=False)
plt.subplot(1, 4, 2)
plt.imshow(outputs[0].numpy().transpose(1, 2, 0))

c1, h1, w1 = image_torch.shape
rotate_matrix1 = np.array([[math.cos(rotate_angle), -math.sin(rotate_angle)*h1/w1, 0],
                          [math.sin(rotate_angle)*w1/h1, math.cos(rotate_angle), 0]])
theta1 = torch.tensor(rotate_matrix1, dtype=torch.float)
grid1 = functional.affine_grid(theta1.unsqueeze(0), image_torch.unsqueeze(0).size(), align_corners=True)
outputs1 = functional.grid_sample(image_torch.unsqueeze(0), grid1, align_corners=True)
plt.subplot(1, 4, 3)
plt.imshow(outputs1[0].numpy().transpose(1, 2, 0))


c2, h2, w2 = image_torch.shape
rotate_angle = 0
tx = 2*175/w2
ty = 2*100/h2
# theta的tx，ty是和图片w/2和h/2的比值
rotate_matrix2 = np.array([[math.cos(rotate_angle), -math.sin(rotate_angle)*h1/w1, tx],
                          [math.sin(rotate_angle)*w1/h1, math.cos(rotate_angle), ty]])
theta2 = torch.tensor(rotate_matrix2, dtype=torch.float)
grid2 = functional.affine_grid(theta2.unsqueeze(0), image_torch.unsqueeze(0).size(), align_corners=True)
outputs2 = functional.grid_sample(image_torch.unsqueeze(0), grid2, align_corners=True)
plt.subplot(1, 4, 4)
plt.imshow(outputs2[0].numpy().transpose(1, 2, 0))
plt.show()
