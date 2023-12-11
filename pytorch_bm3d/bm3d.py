import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

import bm3d_cuda


class bm3d_function(Function):

    @staticmethod
    def forward(ctx, im, variance, two_step=True):
        '''
        im: 4D tensor with shape (batch_size, channel, height, width)
        variance: noise variance
        two_step: whether to use two step method
        '''
        assert len(im.shape) == 4, "BM3D forward input must be 4D tensor with shape (batch_size, channel, height, width)"
        assert im.shape[0] == 1, "Only support batch_size=1"
        if im.shape[0] == 3:
            print("Warning: We do not support RGB image, inference as multiple gray images")
        
        assert im.dtype == torch.int, "BM3D forward input must be int tensor"

        assert im.min() >= 0, "Input should be larger than zero"

        output = torch.zeros_like(im)
        for c in range(im.shape[1]):
            output[0, c, :, :] = bm3d_cuda.forward(im[0, c, :, :], variance, two_step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''
        grad_output: gradient of output
        '''
        raise NotImplementedError("BM3D backward is not implemented")


class BM3D(Module):

    def __init__(self, two_step=True):
        '''
        Support interpolation mode with Bilinear and Nearest.
        '''
        super().__init__()
        self._two_step = two_step

    def forward(self, input, variance):
        return bm3d_function.apply(input, variance, self._two_step)
