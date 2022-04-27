import torch
from torch.nn import Module
from scipy.ndimage import rotate
import numpy as np
from torch.autograd.gradcheck import gradcheck
from torch.autograd import Function


class ImageRotationLoss(Function):
    # def __init__(self, i_angle):
    #     super(ImageRotationLoss, self).__init__()
    #     self.angle = float(i_angle)
    @staticmethod
    def forward(ctx, fm1, fm2):
        # fm1 = i_fm1.detach().numpy()
        # .new() just create a new Tensor which has a same type and device as original Tensor
        # fm1 = i_fm1.new(fm1)
        fm1 = fm1.detach()
        fm2 = fm2.detach()
        r_fm2 = rotate(fm2.numpy(), angle=float(5.0), reshape=False)
        # r_fm2 = torch.from_numpy(r_fm2)
        # r_fm2 = torch.tensor(r_fm2, dtype=torch.double, requires_grad=True)
        mask = np.ones(shape=fm1.shape)
        r_mask = rotate(mask, angle=float(5.0), reshape=False)
        # r_mask = i_fm1.new(r_mask)
        ctx.save_for_backward(fm1, fm2)
        loss = np.sum(((fm1.numpy() - r_fm2)**2)) / np.sum(r_mask)
        return torch.as_tensor(loss, dtype=fm1.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        fm1, fm2 = ctx.saved_tensors
        return grad_output, grad_output

class ScipyRotation(Module):
    def __init__(self):
        super(ScipyRotation, self).__init__()

    def forward(self, fm1, fm2):
        return ImageRotationLoss.apply(fm1, fm2)


input1 = torch.rand([3, 3], dtype=torch.double, requires_grad=True)
input2 = torch.rand([3, 3], dtype=torch.double, requires_grad=True)
scipy_loss = ScipyRotation()
output = scipy_loss(input1, input2)
print(output)

test = gradcheck(scipy_loss, [input1, input2])
print("Are the gradients correct: ", test)
