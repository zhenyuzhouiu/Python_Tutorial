import torch
from torch import nn
from collections import OrderedDict
from typing import Optional, Callable
from torch.nn import functional as F


def drop_path(x, drop_prob: float = 0., training: bool = True):
    if drop_prob == 0 or training is False:
        return x

    keep_p = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_p + torch.rand(shape)
    random_tensor = random_tensor.floor_()
    output = x.div(keep_p) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=0., training=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.training = training

    def forward(self, x):
        return drop_path(x, drop_prob=self.drop_prob, training=self.training)


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 input_ch: int,
                 output_ch: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 normal_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if normal_layer is None:
            normal_layer = nn.BatchNorm2d(num_features=output_ch)
        if activation_layer is None:
            activation_layer = nn.SiLU

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=input_ch,
                                                         out_channels=output_ch,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups),
                                               normal_layer(output_ch),
                                               activation_layer())


class SqueezeExcitation(nn.Module):
    def __init__(self, input_ch: int,
                 expand_ch: int,
                 squeeze_rate: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_ch = input_ch // squeeze_rate
        self.fc1 = nn.Conv2d(in_channels=expand_ch, out_channels=squeeze_ch, kernel_size=1)
        self.ac1 = nn.SiLU()
        self.fc2 = nn.Conv2d(in_channels=squeeze_ch, out_channels=expand_ch, kernel_size=1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.ac1(self.fc1(scale))
        scale = self.ac2(self.fc2(scale))
        return x * scale


class MBConv(nn.Module):
    def __init__(self,
                 input_ch: int,
                 expand_rate: int,
                 output_ch: int,
                 kernel_size: int,
                 stride: int,
                 use_se: bool,
                 drop_prob: float,
                 index: str,
                 width_coefficient: int,
                 normal_layer=Optional[Callable[..., nn.Module]],
                 activation_layer=Optional[Callable[..., nn.Module]]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        expand_ch = input_ch * expand_rate

        layers = OrderedDict()

        if expand_ch != input_ch:
            expand_conv = ConvBNActivation(input_ch=input_ch,
                                           output_ch=expand_ch,
                                           kernel_size=1,
                                           normal_layer=normal_layer,
                                           activation_layer=nn.SiLU)

            layers.update({'expand_conv': expand_conv})

        dw_conv = ConvBNActivation(input_ch=expand_ch,
                                   output_ch=expand_ch,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   groups=expand_ch,
                                   normal_layer=normal_layer,
                                   activation_layer=nn.SiLU)

        layers.update({'depthwise_conv': dw_conv})

        if use_se:
            se_conv = SqueezeExcitation(input_ch=input_ch,
                                        expand_ch=expand_ch,
                                        squeeze_rate=4)

        layers.update({'se_conv': se_conv})

        project_conv = ConvBNActivation(input_ch=expand_ch,
                                        output_ch=output_ch,
                                        kernel_size=1,
                                        normal_layer=normal_layer,
                                        activation_layer=nn.Identity)

        layers.update({'project_conv': project_conv})

        if input_ch == output_ch and stride == 1:
            self.use_res_path = True

        self.block = nn.Sequential(layers)

        if drop_prob != 0. and self.use_res_path:
            self.drop_path = DropPath(drop_prob=drop_prob)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out = self.drop_path(out)
        if self.use_res_path:
            out += x
            
        return out
            

class EfficientNet(nn.Module):
    def __init__(self,
                 input_ch: int,
                 num_class: int,
                 width_coefficient: int,
                 depth_coefficient: int,
                 drop_out_rate: float,
                 drop_path_rate: float,
                 mbblock: Optional[Callable[..., nn.Module]],
                 normal_layer: Optional[Callable[..., nn.Module]]):
        super(EfficientNet, self).__init__()

        self.stage1 = ConvBNActivation(input_ch=3, output_ch=32)