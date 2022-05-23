import torch
import math
import torch.nn.functional as F


def generate_theta_undeformable(i_radian, i_tx, i_ty, i_batch_size, i_h, i_w, i_dtype):
    theta = torch.tensor([[math.cos(i_radian), math.sin(-i_radian) * i_h / i_w, i_tx],
                          [math.sin(i_radian) * i_w / i_h, math.cos(i_radian), i_ty]],
                         dtype=i_dtype).unsqueeze(0).repeat(i_batch_size, 1, 1)
    return theta

def generate_theta_deformable(i_radian, i_tx, i_ty, i_batch_size, i_h, i_w, i_dtype):
    theta = torch.tensor([[math.cos(i_radian), math.sin(-i_radian), i_tx],
                          [math.sin(i_radian), math.cos(i_radian), i_ty]],
                         dtype=i_dtype).unsqueeze(0).repeat(i_batch_size, 1, 1)
    return theta


def rotate_mse_loss(i_fm1, i_fm2, i_mask):
    # the input feature map shape is (bs, 1, h, w)
    square_err = torch.mul(torch.pow((i_fm1 - i_fm2), 2), i_mask)
    mean_se = square_err.view(i_fm1.size(0), -1).sum(1) / i_mask.view(i_fm1.size(0), -1).sum(1)
    return mean_se


def mse_loss(src, target):
    if isinstance(src, torch.autograd.Variable):
        return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
    else:
        return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)


class WholeImageRotationAndTranslation_Undeformable(torch.nn.Module):
    def __init__(self, i_v_shift, i_h_shift, i_angle):
        super(WholeImageRotationAndTranslation_Undeformable, self).__init__()
        self.v_shift = i_v_shift
        self.h_shift = i_h_shift
        self.angle = i_angle

    def forward(self, i_fm1, i_fm2):
        b, c, h, w = i_fm1.shape
        mask = torch.ones_like(i_fm2, device=i_fm1.device)
        n_affine = 0
        if self.training:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=True, device=i_fm1.device)
        else:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=False, device=i_fm1.device)

        if self.v_shift == self.h_shift == 0:
            min_dist = mse_loss(i_fm1, i_fm2).cuda()
            return min_dist
        for tx in range(-self.h_shift, self.h_shift + 1):
            for ty in range(-self.v_shift, self.v_shift + 1):
                for a in range(-self.angle, self.angle + 1):
                    # input self.ange is angel ont radian
                    radian_a = a * math.pi / 180.
                    # the shift tx and ty is a ratio to w/2 and h/2
                    ratio_tx = 2 * tx / w
                    ratio_ty = 2 * ty / h
                    theta = generate_theta_undeformable(radian_a, ratio_tx, ratio_ty, b, h, w, i_fm1.dtype)
                    grid = F.affine_grid(theta, i_fm2.size(), align_corners=True).to(i_fm1.device)
                    r_fm2 = F.grid_sample(i_fm2, grid, align_corners=True)
                    r_mask = F.grid_sample(mask, grid, align_corners=True)
                    # mean_se.shape: -> (bs, )
                    mean_se = rotate_mse_loss(i_fm1, r_fm2, r_mask)
                    if n_affine == 0:
                        min_dist = mean_se
                    else:
                        min_dist = torch.vstack([min_dist, mean_se])
                    n_affine += 1

        min_dist, _ = torch.min(min_dist, dim=0)
        return min_dist




if __name__ == "__main__":
    loss = WholeImageRotationAndTranslation_Undeformable(i_v_shift=3, i_h_shift=3, i_angle=5).train().cuda()
    input1 = torch.randn([32, 32], dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0).cuda()
    input2 = torch.randn([32, 32], dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0).cuda()
    min_dist = loss(input1, input2)
    print(min_dist)