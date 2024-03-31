from torch import nn
import torch

Pool = nn.MaxPool2d


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Deconv(nn.Module):
    def __init__(self,f):
        super(Deconv, self).__init__()
        self.dec = nn.ConvTranspose2d(
                    in_channels=f,
                    out_channels=f,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False)
        self.bn = nn.BatchNorm2d(f, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dec(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2


class VModule(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(VModule, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = VModule(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = Deconv(f)

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2


class conv_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, 
            has_bn=True, has_relu=True, efficient=False,groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding,groups=groups)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func 

        func = _func_factory(
                self.conv, self.bn, self.relu, self.has_bn, self.has_relu)
        x = func(x)

        return x


class PRM(nn.Module):

    def __init__(self, output_chl_num, efficient=False):
        super(PRM, self).__init__()
        self.output_chl_num = output_chl_num
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.res_prm_1 = Residual(self.output_chl_num, self.output_chl_num)
        self.res_prm_2 = Residual(self.output_chl_num, self.output_chl_num)
        # self.conv_bn_relu_prm_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=3,
        #         stride=1, padding=1, has_bn=True, has_relu=True,
        #         efficient=efficient) 
        self.conv_bn_relu_prm_2_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.conv_bn_relu_prm_2_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.sigmoid2 = nn.Sigmoid()
        self.conv_bn_relu_prm_3_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.conv_bn_relu_prm_3_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=9,
                stride=1, padding=4, has_bn=True, has_relu=True,
                efficient=efficient,groups=self.output_chl_num)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        out = self.res_prm_1(x)
        out_1 = out
        out_2 = self.avg_pool(out_1)
        # out_2 = torch.nn.functional.adaptive_avg_pool2d(out_1, (1,1))
        out_2 = self.conv_bn_relu_prm_2_1(out_2)
        out_2 = self.conv_bn_relu_prm_2_2(out_2)
        out_2 = self.sigmoid2(out_2)
        out_2 = out_2.mul(out_1)

        out_3 = self.conv_bn_relu_prm_3_1(out_1)
        out_3 = self.conv_bn_relu_prm_3_2(out_3)
        out_3 = out_1.mul(out_3)
        out_3 = self.sigmoid3(out_3)
 
        out = self.res_prm_2(out_1 + out_2 + out_3)

        return out

# class PRM(nn.Module):

#     def __init__(self, output_chl_num, efficient=False):
#         super(PRM, self).__init__()
#         self.output_chl_num = output_chl_num
#         self.conv_bn_relu_prm_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=3,
#                 stride=1, padding=1, has_bn=True, has_relu=True,
#                 efficient=efficient) 
#         self.conv_bn_relu_prm_2_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
#                 stride=1, padding=0, has_bn=True, has_relu=True,
#                 efficient=efficient)
#         self.conv_bn_relu_prm_2_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
#                 stride=1, padding=0, has_bn=True, has_relu=True,
#                 efficient=efficient)
#         self.sigmoid2 = nn.Sigmoid()
#         self.conv_bn_relu_prm_3_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
#                 stride=1, padding=0, has_bn=True, has_relu=True,
#                 efficient=efficient)
#         self.conv_bn_relu_prm_3_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=9,
#                 stride=1, padding=4, has_bn=True, has_relu=True,
#                 efficient=efficient,groups=self.output_chl_num)
#         self.sigmoid3 = nn.Sigmoid()

#     def forward(self, x):
#         out = self.conv_bn_relu_prm_1(x)
#         out_1 = out
#         out_2 = torch.nn.functional.adaptive_avg_pool2d(out_1, (1,1))
#         out_2 = self.conv_bn_relu_prm_2_1(out_2)
#         out_2 = self.conv_bn_relu_prm_2_2(out_2)
#         out_2 = self.sigmoid2(out_2)
#         out_3 = self.conv_bn_relu_prm_3_1(out_1)
#         out_3 = self.conv_bn_relu_prm_3_2(out_3)
#         out_3 = self.sigmoid3(out_3)
#         out = out_1.mul(1 + out_2.mul(out_3))
#         return out
