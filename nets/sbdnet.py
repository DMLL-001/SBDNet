# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class DN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DN, self).__init__()
        # 第一个转置卷积层，使用 4x4 卷积核
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(output_channels)

        # 第二个转置卷积层，使用 16x16 卷积核
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        # 通过第一个转置卷积层
        x = self.deconv1(x)
        # 可选：添加激活函数，例如 ReLU
        x = self.bn1(x)
        x = nn.ReLU()(x)

        # 通过第二个转置卷积层
        x = self.deconv2(x)
        # 可选：再次添加激活函数
        x = self.bn2(x)
        x = nn.ReLU()(x)

        return x


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class NetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


#SE注意力机制
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = ConvBlock(in_channels=self.in_channels,
                                   out_channels=num_classes,
                                   stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(
            1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

######CBAM注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



#scce注意力模块
class cSE(nn.Module):  # noqa: N801
    """
    The channel-wise SE (Squeeze and Excitation) block from the
    `Squeeze-and-Excitation Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
    and
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class sSE(nn.Module):  # noqa: N801
    """
    The sSE (Channel Squeeze and Spatial Excitation) block from the
    `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class scSE(nn.Module):  # noqa: N801
    """
    The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation)
    block from the `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE(in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=2, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        # self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        # self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        # self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        # self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_c4 = nn.Conv2d(c4_in_channels, 256, kernel_size=1)
        self.linear_c3 = nn.Conv2d(c3_in_channels, 256, kernel_size=1)
        self.linear_c2 = nn.Conv2d(c2_in_channels, 256, kernel_size=1)
        self.linear_c1 = nn.Conv2d(c1_in_channels, 256, kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.sge = SpatialGroupEnhance(32)

        self.tconv1 = Conv3BN(256, 64)
        self.tdn1 = DN(64, embedding_dim)
        self.tconv2 = Conv3BN(256, 64)
        self.tdn2 = DN(64, embedding_dim)
        self.tconv3 = Conv3BN(256, 64)
        self.tdn3 = DN(64, embedding_dim)
        self.tconv4 = Conv3BN(256, 64)
        self.tdn4 = DN(64, embedding_dim)

        self.bconv1 = Conv3BN(256, 64)
        self.bdn1 = DN(64, embedding_dim)
        self.bconv2 = Conv3BN(256, 64)
        self.bdn2 = DN(64, embedding_dim)
        self.bconv3 = Conv3BN(256, 64)
        self.bdn3 = DN(64, embedding_dim)
        self.bconv4 = Conv3BN(256, 64)
        self.bdn4 = DN(64, embedding_dim)

        self.conv4 = NetModule(832, 320)
        self.conv3 = NetModule(448, 128)
        self.conv2 = NetModule(192, 64)
        self.conv1 = NetModule(64, 32)

        self.sge1 = SpatialGroupEnhance(32)
        self.ffm1 = FeatureFusionModule(320, 832)
        self.sge2 = SpatialGroupEnhance(32)
        self.ffm2 = FeatureFusionModule(128, 448)
        self.sge3 = SpatialGroupEnhance(32)
        self.ffm3 = FeatureFusionModule(64, 192)
        self.sge4 = SpatialGroupEnhance(32)
        self.ffm4 = FeatureFusionModule(32, 64)


        # self.linear_fuse = ConvModule(
        #     c1=8,
        #     c2=1,
        #     k=1,
        # )

        self.linear_fuse = ConvModule(
            c1=8*embedding_dim,
            c2=embedding_dim,
            k=1,
        )

    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4)#.permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3)#.permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2)#.permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1)#.permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # x4 = self.upsample1(c4)
        # x4 = self.conv4(torch.cat([c3, x4], 1))
        #
        # x3 = self.upsample1(x4)
        # x3 = self.conv3(torch.cat([c2, x3], 1))
        #
        # x2 = self.upsample1(x3)
        # x2 = self.conv2(torch.cat([c1, x2], 1))
        #
        # x1 = self.upsample2(x2)
        # x1 = self.conv1(x1)
        # x_out1 = self.sge(x1)

        x4 = self.sge1(c4)
        x4 = self.upsample1(x4)

        x3 = self.sge1(c3)
        x3 = self.ffm1(x3, x4)
        x3 = self.upsample1(x3)

        # x3 = self.upsample1(c3)
        x2 = self.sge2(c2)
        x2 = self.ffm2(x2, x3)
        x2 = self.upsample1(x2)

        # x2 = self.upsample1(c2)
        x1 = self.sge3(c1)
        x1 = self.ffm3(x1, x2)
        x1 = self.upsample2(x1)
        x_out1 = self.conv1(x1)

        t_out1 = self.tconv1(_c1)
        t_out1 = self.tdn1(t_out1)
        t_out2 = self.tconv2(_c2+_c1)
        t_out2 = self.tdn2(t_out2)
        t_out3 = self.tconv3(_c3+_c2+_c1)
        t_out3 = self.tdn3(t_out3)
        t_out4 = self.tconv4(_c4+_c3+_c2+_c1)
        t_out4 = self.tdn4(t_out4)

        d_out4 = self.bconv4(_c4)
        d_out4 = self.bdn4(d_out4)
        d_out3 = self.bconv3(_c4 + _c3)
        d_out3 = self.bdn3(d_out3)
        d_out2 = self.bconv2(_c4 + _c3 + _c2)
        d_out2 = self.bdn2(d_out2)
        d_out1 = self.bconv1(_c4 + _c3 + _c2 + _c1)
        d_out1 = self.bdn1(d_out1)


        fuse = self.linear_fuse(torch.cat([t_out1, t_out2, t_out3, t_out4, d_out1, d_out2, d_out3, d_out4], dim=1))

        x_out2 = [t_out1, t_out2, t_out3, t_out4, d_out1, d_out2, d_out3, d_out4, fuse]

        return x_out1, x_out2

class SBDNet(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
        super(SBDNet, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512 ],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 32,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]

        self.conv6 = NetModule(768, 256)
        self.conv7 = NetModule(384, 128)
        self.conv8 = NetModule(192, 64)
        self.conv9 = NetModule(96, 32)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)


        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

        # if add_output:
        self.conv_final1 = nn.Conv2d(32, num_classes, 1)
        self.conv_final2 = nn.Conv2d(32, num_classes, 1)
        self.conv_final3 = nn.Conv2d(32, 1, 1)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)
        x_out, x_out2 = self.decode_head.forward(x)

        x_out1 = self.conv_final1(x_out)
        x_out1 = F.log_softmax(x_out1, dim=1)

        x_out2 = [self.conv_final2(out) for out in x_out2]
        x_out2 = [F.log_softmax(out, dim=1) for out in x_out2]

        x_out3 = self.conv_final3(x_out)
        x_out3 = torch.sigmoid(x_out3)
        # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x_out1, x_out2, x_out3


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.rand((8, 3, 256, 256))
    # print(input)
    model = SBDNet(num_classes=2, phi='b2').to(device)
    outputs = model(input.to(device))
    print('---------')
