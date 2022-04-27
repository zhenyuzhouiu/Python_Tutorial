# Create extensions using numpy and scipy
# https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html
#
# In deep learning literature, this layer is confusingly referred to as convolution
# while the actual operation is cross-correlation (the only difference is that filter
# is flipped for convolution, which is not the case for cross-correlation).
#
# Implementation of a layer with learnable weights, where cross-correlation has a
# filter (kernel) that represents weights.
#
# The backward pass computes the gradient wrt the input and the gradient wrt the filter.
#

from torch.autograd import Function
import torch
from numpy import flip
import numpy as np
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class ScipyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        # the "mode" will decide the output size
        # "mode"-> full: output is the full discrete linear cross-correlation of the input
        # valid: The output contains only those elements that do not rely on zero padding.
        #  In 'valid' mode, either in1 or in2 must be at least as large as the other in each dimension.
        # same: the output is the same size as in1, centered on the "full" output
        # the "boundary" and "fill-value" will decide the boundary condition correlation
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        test = torch.as_tensor(result, dtype=input.dtype)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        # keep numpy dimension,
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(
            grad_bias).to(torch.float)


class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter, self.bias)


# Example usage
module = ScipyConv2d(3, 3)
# print("Filter and bias: ", list(module.parameters()))
input = torch.randn(10, 10, requires_grad=True)
output = module(input)
print("Output from the convolution: ", output)
# the output size is (w-f+2p)/s+1: => (10-3+0)/1+1=8
output.backward(torch.randn(8, 8))
print("Gradient for the input map: ", input.grad)
print("Gradient for the filter map: ", module.filter.grad)
print("Gradient for the bias: ", module.bias.grad)

# check the gradients
from torch.autograd.gradcheck import gradcheck

moduleConv = ScipyConv2d(3, 3)

input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
# test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
test = gradcheck(moduleConv, input)
print("Are the gradients correct: ", test)
