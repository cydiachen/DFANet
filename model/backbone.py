import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding = 0, dilation=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

# This is a simplify version of channel attention
class fc_attention(nn.Module):
    def __init__(self,in_channels,out_channels=192):
        super(fc_attention,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels,1000,bias = False),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1000,out_channels,bias = False,kernel_size = 1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self,x):
        residual = x
        route1 = self.pooling(x)
        route1 = self.fc(route1)
        route1 = self.conv(route1)

        # Equals to channel attention
        y = residual * route1
        return route1,y

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        # for inbalance channel numbers
        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_channels
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_channels))
            filters = out_channels

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_channels))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x

class XceptionA(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    Modified Xception A architecture, as specified in
    https://arxiv.org/pdf/1904.02216.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(XceptionA, self).__init__()
        self.num_classes = num_classes

        # preprocess
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, stride = 2, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # conv for reducing channel size in input for non-first backbone stages
        # To be viewed
        self.enc2_conv = nn.Conv2d(in_channels = (192+48), out_channels = 8, kernel_size = 1, stride = 1, bias=False) # bias=False?

        # where to put the stride=2 is a problem
        self.enc2_1 = Block(in_channels=8, out_channels=12, reps=4, stride=1, start_with_relu=True, grow_first=True)
        self.enc2_2 = Block(in_channels=12, out_channels=12, reps=4, stride=1, start_with_relu=True, grow_first=True)
        self.enc2_3 = Block(in_channels=12, out_channels=48, reps=4, stride=2, start_with_relu=True, grow_first=True)

        self.enc2 = nn.Sequential(self.enc2_1, self.enc2_2, self.enc2_3)

        self.enc3_conv = nn.Conv2d(in_channels = (96+48), out_channel = 48, kernel_size = 1, stride = 1, bias=False)

        self.enc3_1 = Block(in_channels=48, out_channels=24, reps=6, stride=1, start_with_relu=True, grow_first=True)
        self.enc3_2 = Block(in_channels=24, out_channels=24, reps=6, stride=1, start_with_relu=True, grow_first=True)
        self.enc3_3 = Block(in_channels=24, out_channels=96, reps=6, stride=2, start_with_relu=True, grow_first=True)
        self.enc3 = nn.Sequential(self.enc3_1, self.enc3_2, self.enc3_3)

        self.enc4_conv = nn.Conv2d(in_channels=(96+192), out_channels=96, reps=1, stride=1, bias=False)
        self.enc4_1 = Block(in_channels=96, out_channels=48, reps=4, stride=1, start_with_relu=True, grow_first=True)
        self.enc4_2 = Block(in_channels=48, out_channels=48, reps=4, strides=1, start_with_relu=True, grow_first=True)
        self.enc4_3 = Block(in_channels=48, out_channels=192, reps=4, strides=2, start_with_relu=True, grow_first=True)
        self.enc4 = nn.Sequential(self.enc4_1, self.enc4_2, self.enc4_3)

        self.fc_attention = nn.Sequential(
            fc_attention(in_channels=192,out_channels=192)
        )

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        enc2 = self.enc2(x)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        pool = self.pooling(enc4)

        fc,fca = fc_attention(pool)

        return enc2, enc3, enc4, fc, fca

    def forward_concat(self, fca_concat, enc2_concat, enc3_concat, enc4_concat):
        """For second and third stage."""
        # 这里以第二stage的enc2为例子：前两个变量分别是上一层的输出以及在上面一层的输入，我个人是觉得有点类似skip结构
        enc2 = self.enc2(self.enc2_conv(torch.cat((fca_concat, enc2_concat), dim=1)))
        enc3 = self.enc3(self.enc3_conv(torch.cat((enc2, enc3_concat), dim=1)))
        enc4 = self.enc4(self.enc4_conv(torch.cat((enc3, enc4_concat), dim=1)))
        pool = self.pooling(enc4)
        fc,fca = fc_attention(pool)

        return enc2, enc3, enc4, fc, fca

class XceptionB(nn.Module):
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(XceptionA, self).__init__()
        self.num_classes = num_classes

        # preprocess
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, stride = 2, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # conv for reducing channel size in input for non-first backbone stages
        # To be viewed
        self.enc2_conv = nn.Conv2d(in_channels = (128+32), out_channels = 8, kernel_size = 1, stride = 1, bias=False) # bias=False?

        # where to put the stride=2 is a problem
        self.enc2_1 = Block(in_channels=8, out_channels=8, reps=4, stride=1, start_with_relu=True, grow_first=True)
        self.enc2_2 = Block(in_channels=8, out_channels=8, reps=4, stride=1, start_with_relu=True, grow_first=True)
        self.enc2_3 = Block(in_channels=8, out_channels=32, reps=4, stride=2, start_with_relu=True, grow_first=True)

        self.enc2 = nn.Sequential(self.enc2_1, self.enc2_2, self.enc2_3)

        self.enc3_conv = nn.Conv2d(in_channels = (64+32), out_channel = 32, kernel_size = 1, stride = 1, bias=False)

        self.enc3_1 = Block(in_channels=32, out_channels=16, reps=6, stride=1, start_with_relu=True, grow_first=True)
        self.enc3_2 = Block(in_channels=16, out_channels=16, reps=6, stride=1, start_with_relu=True, grow_first=True)
        self.enc3_3 = Block(in_channels=16, out_channels=64, reps=6, stride=2, start_with_relu=True, grow_first=True)
        self.enc3 = nn.Sequential(self.enc3_1, self.enc3_2, self.enc3_3)

        self.enc4_conv = nn.Conv2d(in_channels=(64+128), out_channels=64, reps=1, stride=1, bias=False)
        self.enc4_1 = Block(in_channels=64, out_channels=32, reps=4, stride=1, start_with_relu=True, grow_first=True)
        self.enc4_2 = Block(in_channels=32, out_channels=32, reps=4, strides=1, start_with_relu=True, grow_first=True)
        self.enc4_3 = Block(in_channels=32, out_channels=128, reps=4, strides=2, start_with_relu=True, grow_first=True)
        self.enc4 = nn.Sequential(self.enc4_1, self.enc4_2, self.enc4_3)

        self.fc_attention = nn.Sequential(
            fc_attention(in_channels=128,out_channels=128)
        )

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        enc2 = self.enc2(x)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        pool = self.pooling(enc4)

        fc,fca = fc_attention(pool)

        return enc2, enc3, enc4, fc, fca

    def forward_concat(self, fca_concat, enc2_concat, enc3_concat, enc4_concat):
        """For second and third stage."""
        # 这里以第二stage的enc2为例子：前两个变量分别是上一层的输出以及在上面一层的输入，我个人是觉得有点类似skip结构
        enc2 = self.enc2(self.enc2_conv(torch.cat((fca_concat, enc2_concat), dim=1)))
        enc3 = self.enc3(self.enc3_conv(torch.cat((enc2, enc3_concat), dim=1)))
        enc4 = self.enc4(self.enc4_conv(torch.cat((enc3, enc4_concat), dim=1)))
        pool = self.pooling(enc4)
        fc,fca = fc_attention(pool)

        return enc2, enc3, enc4, fc, fca


def backboneA(pretrained = False,**kwargs):
    '''
    construct Xception like model
    '''

    model_url = './checkpoints/XceptionA_best.pth.tar'

    model = XceptionA(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_url),strict=False)
    return model

def backboneB(pretrained = False,**kwargs):
    model_url = './checkpoints/XceptionB_best.pth.tar'

    model = XceptionB(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_url),strict=False)
    return model
