# Create extensions using numpy and scipy
# https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html

import torch
from torch.autograd import Function
from numpy.fft import rfft2, irfft2

class BadFFTFunction(Function):
    @staticmethod
    def forward(ctx, input):
        numpy_input = input.detach().numpy()
        result = abs(rfft2(numpy_input))
        return input.new(result)

    @staticmethod
    def backward(ctx, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return grad_output.new(result)

def incorrect_fft(input):
    return BadFFTFunction.apply(input)

input = torch.randn(8, 8, requires_grad=True)
result = incorrect_fft(input)
print(result)
result.backward(torch.randn(result.size()))
print(input)