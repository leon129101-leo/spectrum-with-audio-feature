from astropy.io import fits
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.ndimage import median_filter
import os
import torchaudio
from torch.nn import Softmax
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import math
import torch.fft
import torch.nn.functional as F
class SE(nn.Module):
    def __init__(self, channels=60, r=15):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // r, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # squeeze
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.relu(y)
        # excitation
        y = self.fc2(y)
        y = self.sigmoid(y)
        # reshape to broadcast
        y = y.view(y.size(0), y.size(1), 1, 1)
        # scale the input by the excitation weights
        return x * y.expand_as(x)
class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, H, W, C = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        return x


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x
class GCT(nn.Module):
    def __init__(self, channels=60, beta_wd_mult=0.0, epsilon=1e-5):
        super(GCT, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)
        self.beta_wd_mult = beta_wd_mult

    def forward(self, x):
        embedding = torch.sqrt(torch.sum(x ** 2, dim=(2, 3), keepdim=True) + self.epsilon) * self.alpha
        norm = torch.sqrt(torch.mean(embedding ** 2, dim=1, keepdim=True) + self.epsilon)
        embedding = embedding / norm
        gate = 1.0 + torch.tanh(self.gamma * embedding + self.beta)
        return x * gate


class ECA(nn.Module):
    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global spatial average pooling
        y = self.avg_pool(x)
        print(y.shape)  # Before squeeze and transpose

        # Ensure y has at least 2 dimensions before squeeze and transpose operations
        if y.dim() == 3:
            y = y.unsqueeze(1)  # Add a channel dimension if missing
        # Transpose to fit Conv1d input format
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)

        # Transpose back and unsqueeze for broadcasting
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Sigmoid activation
        y = self.sigmoid(y)
        print(y.shape)  # Before squeeze and transpose
        # Apply attention weights
        return x * y.expand_as(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        # self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0',
                               nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i,
                                   nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                             kernel_size=3, \
                                             padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor):
        att = 1 + F.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor


class CBAM_Module(nn.Module):

    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class EPSABlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(EPSABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class EPSANet(nn.Module):
    def __init__(self, block, layers, num_classes=8):
        super(EPSANet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layers(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layers(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def epsanet50():
    model = EPSANet(EPSABlock, [3, 4, 6, 3], num_classes=256)
    return model

class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=(1, 1), stride=1),
            torch.nn.BatchNorm2d(10),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 10, kernel_size=(1, 2), stride=1),
            torch.nn.BatchNorm2d(10),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=(1, 3), stride=1),
            torch.nn.BatchNorm2d(20),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 30, kernel_size=(1, 4), stride=1),
            torch.nn.BatchNorm2d(30),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 40, kernel_size=(1, 5), stride=1),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(40, 50, kernel_size=(1, 7), stride=1),
            torch.nn.BatchNorm2d(50),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(50, 60, kernel_size=(1, 8), stride=1),
            torch.nn.BatchNorm2d(60),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(60, 60, kernel_size=(1, 9), stride=1),
            torch.nn.BatchNorm2d(60),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.dense1 = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(60 * 1 * 219, 1024),
        )
        self.dense2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
        )
        self.sf = torch.nn.Sequential(
            torch.nn.Softmax())
        self.coord = CoordAtt(inp=60, oup=60, reduction=10)

    # 前向传播
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = self.conv7(x6)
        x7 = self.coord(x6)
        x7 = x7.view(-1, 60 * 1 * 219)
        x8 = self.dense1(x7)
        x9 = self.dense2(x8)
        return x9
class audio_mfcc(nn.Module):
    def __init__(self):
        super(audio_mfcc, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.drop_out = nn.Dropout(0.4)
        self.fc1 = nn.Linear(10304, 1024)
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class audio_mel(nn.Module):
    def __init__(self):
        super(audio_mel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.drop_out = nn.Dropout(0.4)
        self.fc1 = nn.Linear(15360, 1024)
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class audio_lfcc(nn.Module):
    def __init__(self):
        super(audio_lfcc, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.drop_out = nn.Dropout(0.4)
        self.fc1 = nn.Linear(5824, 1024)
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class audioModel2(nn.Module):
    def __init__(self):
        super(audioModel2, self).__init__()
        self.cnn1 = CNN_Model()
        self.cnn2 = audio_mel()
        self.cnn3 = audio_mfcc()
        self.cnn4 = audio_lfcc()
        self.cnn5 = epsanet50()
        self.dense = nn.Linear(256 * 5, 8)  # 输出分类结果的全连接层

    def forward(self, x1, x2, x3, x4, x5):
        out1 = self.cnn1(x1)
        out2 = self.cnn2(x2)
        out3 = self.cnn3(x3)
        out4 = self.cnn4(x4)
        out5 = self.cnn5(x5)

        combined_output = torch.cat((out1, out2, out3, out4, out5), dim=1)
        output = self.dense(combined_output)
        return output


dr11train = pd.read_csv('/nfsdata/users/swzhang/11lamost6000.csv')
label_mapping = {"O": 0, "B": 1, "A": 2, "F": 3, "G": 4, "K": 5, "M": 6, "C": 7}

# 定义 Dataset
class FITSDataset(Dataset):
    def __init__(self):
        self.data = dr11train
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filepath = self.data.iloc[idx]['filepath']
        label = self.data.iloc[idx]['SUBCLASS']

        # 读取FITS数据
        with fits.open(filepath) as hdul:
            flux = hdul[1].data[0][0]
            flux = flux.ravel()[:3700]
            flux_nor = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
            flux_nor = median_filter(flux_nor, size=9, mode='reflect')

            # 转换为图像
            fig = plt.figure(figsize=(16, 16), dpi=32)
            plt.plot(wavelength111, flux_nor, color='black')
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            canvas_width, canvas_height = fig.canvas.get_width_height()
            buf.shape = (canvas_height, canvas_width, 4)
            buf = np.roll(buf, 3, axis=2)
            image = Image.frombytes("RGBA", (512, 512), buf.tobytes()).convert('L')
            plt.close()

            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image_tensor = transform(image)

            # 光谱转特征
            flux_interpolated = torch.tensor(flux_nor[:3700], dtype=torch.float32).unsqueeze(0)
            MELSpec = torchaudio.transforms.MelSpectrogram(n_fft=512, win_length=60, hop_length=60, n_mels=64)(
                flux_interpolated)
            MELSpec = torch.squeeze(MELSpec, dim=2)
            MFCC = torchaudio.transforms.MFCC(n_mfcc=30,
                                              melkwargs={"n_fft": 512, "win_length": 50, "hop_length": 40, "n_mels": 64}
                                              )(flux_interpolated)
            MFCC = torch.squeeze(MFCC, dim=2)
            LFCC = torchaudio.transforms.LFCC(
                n_lfcc=30,
                speckwargs={"n_fft": 512, "hop_length": 60, "center": False},
            )(flux_interpolated)

            # 标签
            label = self.transform_label(label)
            label = self.label_mapping[label]
            label_one_hot = F.one_hot(torch.LongTensor([label]), num_classes=8).float()

        return flux_interpolated, image_tensor, label_one_hot, MELSpec, MFCC, LFCC

    def transform_label(self, label):
        if label.startswith(('A',)):
            return 'A'
        elif label.startswith(('B',)):
            return 'B'
        elif label.startswith(('F',)):
            return 'F'
        elif label.startswith(('G',)):
            return 'G'
        elif label.startswith(('K',)):
            return 'K'
        elif label.startswith(('C',)):
            return 'C'
        elif label == 'O':
            return 'O'
        elif label.startswith(('dM', 'gM', 'sdM')):
            return 'M'
        else:
            return 'Unknown'

# wavelength111 (一次性读一个文件就行)
file_sample = '/nfsdata/users/swzhang/lamost_data/B/B9/fitspng439940143/spec-55959-GAC_094N27_V1_sp02-060.fits.gz'
with fits.open(file_sample) as hdulist:
    wavelength111 = hdulist[1].data['WAVELENGTH'].ravel()[:3700]

# 训练准备
device = torch.device('cuda:0')
model = audioModel2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 数据划分
dataset = FITSDataset()
train_img, val_img = train_test_split(dataset, test_size=0.2, random_state=39)
test_img, val_img = train_test_split(val_img, test_size=0.5, random_state=39)

batch_size = 256
train_loader = DataLoader(train_img, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_img, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_img, batch_size=batch_size, shuffle=True)

# 训练循环
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    for flux, image, labels, MELSpec, MFCC, LFCC in train_loader:
        if flux.size(0) != batch_size:  # 保证batch一致
            continue
        flux, image, MELSpec, MFCC, LFCC = flux.to(device), image.to(device), MELSpec.to(device), MFCC.to(device), LFCC.to(device)
        labels = torch.argmax(labels, dim=2).ravel().to(device)
        flux = flux.unsqueeze(1).to(torch.float32)

        optimizer.zero_grad()
        outputs = model(flux, MELSpec, MFCC, LFCC, image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for flux, image, labels, MELSpec, MFCC, LFCC in val_loader:
            if flux.size(0) != batch_size:
                continue
            flux, image, MELSpec, MFCC, LFCC = flux.to(device), image.to(device), MELSpec.to(device), MFCC.to(device), LFCC.to(device)
            labels = torch.argmax(labels, dim=2).ravel().to(device)
            flux = flux.unsqueeze(1).to(torch.float32)
            outputs = model(flux, MELSpec, MFCC, LFCC, image)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Val Accuracy: {accuracy:.2f}%, loss: {loss:.2f}')

# 测试
model.eval()
correct, total = 0, 0
all_true_labels, all_predicted_labels = [], []
with torch.no_grad():
    for flux, image, labels, MELSpec, MFCC, LFCC in test_loader:
        if flux.size(0) != batch_size:
            continue
        flux, image, MELSpec, MFCC, LFCC = flux.to(device), image.to(device), MELSpec.to(device), MFCC.to(device), LFCC.to(device)
        labels = torch.argmax(labels, dim=2).ravel().to(device)
        flux = flux.unsqueeze(1).to(torch.float32)
        outputs = model(flux, MELSpec, MFCC, LFCC, image)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_true_labels.extend(labels.cpu().numpy())
        all_predicted_labels.extend(predicted.cpu().numpy())
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
confusion = confusion_matrix(all_true_labels, all_predicted_labels)