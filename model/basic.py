import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
import math



class SDifferenceConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SDifferenceConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)       # kernel_size*kernel_size [3,3]
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        self.weight = Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))  # [out_channels, in_channels, 3, 3, 3]
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        grad_weight = -self.weight.clone()
        hw = self.weight.size(-1)
        grad_weight[:, :, int((hw-1)/2), int((hw-1)/2)] = torch.sum(self.weight, dim=[2, 3])

        if self.padding_mode != "zeros":
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            grad_weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, grad_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)




class DyfConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DyfConv, self).__init__()
        self.kernel_s_ope = kernel_size
        self.ope_channels = self.kernel_s_ope*self.kernel_s_ope  # kernel_size*kernel_size
        self.operator_conv = nn.Sequential(nn.Linear(self.ope_channels, self.ope_channels*self.ope_channels), nn.Tanh())

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)       # kernel_size*kernel_size [3,3]
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.weight = Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))  # [out_channels:1, in_channels:1, 3, 3]
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        b, c, h, w = input.size()
        h1 = h//self.stride[0]
        w1 = w//self.stride[0]
        x_unfold = F.unfold(input, kernel_size=self.kernel_s_ope, stride=self.stride, padding=self.padding, dilation=self.dilation)
        x_unfold = x_unfold.view(b, c, -1, h1, w1).permute(0,1,3,4,2).unsqueeze(4)  # b, c, kk, h, w -> b, c, h, w, 1, kk
        operator = self.operator_conv(torch.norm(x_unfold, p=2, dim=1)).view(b,1,h1,w1,self.ope_channels,self.ope_channels)  # b, 1, h, w, kk,kk
        x_operator = torch.einsum('bchwij, blhwjk -> bchwik', x_unfold, operator).squeeze(4).permute(0,1,4,2,3).contiguous()  # b, c, h, w, 1, kk -> b, c, kk, h, w
        self.weight_reshape = self.weight.view(self.in_channels, self.out_channels, -1).permute(1,0,2)  # c, out_c, k, k -> out_c, c, kk
        if self.bias is not None:
            output = torch.einsum('bckhw, ock -> bohw', x_operator+x_unfold.squeeze(4).permute(0,1,4,2,3),
                                  self.weight_reshape).contiguous() + self.bias[None, :, None, None]
        else:
            output = torch.einsum('bckhw, ock -> bohw', x_operator+x_unfold.squeeze(4).permute(0,1,4,2,3),
                                  self.weight_reshape).contiguous()

        return output




class RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        #nn.init.zeros_(self.convmap.weight)
        self.bias = None#nn.Parameter(torch.zeros(out_channels), requires_grad=True)     # must have a bias for identical initialization
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)


# m = FDConv(in_channels=128, out_channels=64, kernel_num=8, kernel_size=3, padding=1, bias=True)



