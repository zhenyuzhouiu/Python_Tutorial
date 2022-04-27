import torch
from torch.nn import functional as F
import numpy as np
import math
from torch.autograd.gradcheck import gradcheck


class GridSampleRotationGradient(torch.nn.Module):
    def __init__(self, i_angle):
        super(GridSampleRotationGradient, self).__init__()
        self.radian = i_angle * math.pi / 180

    def forward(self, i_fm1, i_fm2):
        b, c, h, w = i_fm1.shape
        mask = torch.ones_like(i_fm2, dtype=torch.double)
        theta = torch.tensor([[math.cos(self.radian), math.sin(-self.radian) * h / w, 0],
                              [math.sin(self.radian) * w / h, math.cos(self.radian), 0]],
                             dtype=torch.double).unsqueeze(0).repeat(b, 1, 1)
        grid = F.affine_grid(theta, i_fm2.size(), align_corners=True)
        r_fm2 = F.grid_sample(i_fm2, grid, align_corners=True)
        r_mask = F.grid_sample(mask, grid, align_corners=True)
        square_err = torch.mul(torch.pow((i_fm2 - r_fm2), 2), r_mask)
        mean_se = square_err.view(b, -1).sum(1) / r_mask.view(b, -1).sum(1)
        return mean_se


input1 = torch.randn([5, 5], dtype=torch.double, requires_grad=False).unsqueeze(0).unsqueeze(0)
input2 = torch.randn([5, 5], dtype=torch.double, requires_grad=False).unsqueeze(0).unsqueeze(0)
gsrg_loss = GridSampleRotationGradient(i_angle=5)
loss = gsrg_loss(input1, input2)
print(loss)


input1 = torch.randn([5, 5], dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0)
input2 = torch.randn([5, 5], dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0)
test = gradcheck(gsrg_loss, [input1, input2])
print("Are the gradients correct: ", test)