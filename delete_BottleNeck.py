import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(sci_mode=False)


class WrapReLU(nn.ReLU):
    def __init__(self, mid_channel: int, inplace=True):
        super(WrapReLU, self).__init__(inplace)
        self.mid_channel = mid_channel

    def forward(self, x):
        out = torch.cat([super(WrapReLU, self).forward(x[:, :self.mid_channel]), x[:, self.mid_channel:]], dim=1)
        return out


class SimplifyRMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, delete_attr=True):
        self.in_channel = inplanes
        self.out_channel = planes * 4
        self.channel = planes
        self.delete_attr = delete_attr

        super(SimplifyRMBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or self.in_channel != self.out_channel:
            down_sample_kernel = 1
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * 4, down_sample_kernel, stride, down_sample_kernel // 2, bias=False),
                                            nn.BatchNorm2d(planes * 4))
        else:
            self.downsample = None

        self.stride = stride

        self.wrap_relu = WrapReLU(mid_channel=planes, inplace=True)
        self.deploy_flag = False

        self.delete_list = (nn.BatchNorm2d, nn.Conv2d, nn.Sequential, nn.ModuleList)
        self.attr_list = list(self.__dict__['_modules'].items())

    def forward(self, x):
        if self.training or not self.deploy_flag:
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)
            return out
        else:
            x = self.rm_conv1(x)
            x = self.rm_bn1(x)
            x = self.wrap_relu(x)
            x = self.rm_conv2(x)
            x = self.rm_bn2(x)
            x = self.wrap_relu(x)
            x = self.rm_conv3(x)
            x = self.relu(x)
            return x

    def deploy(self, verbose=False):
        if self.deploy_flag:
            return
        self.deploy_flag = True
        device = self.conv1.weight.device

        out_channel_1, in_channel_1 = self.conv1.weight.size()[:2]
        # for conv1
        self.rm_conv1 = nn.Conv2d(in_channel_1, out_channel_1 + self.in_channel, 1, bias=False).to(device).eval()
        self.rm_conv1.weight.data[:out_channel_1] = self.conv1.weight.data.clone().detach()
        nn.init.dirac_(self.rm_conv1.weight[out_channel_1:])

        # for bn1
        self.rm_bn1 = nn.BatchNorm2d(out_channel_1 + self.in_channel).to(device).eval()
        self.rm_bn1.weight.data[:out_channel_1] = self.bn1.weight.data.clone().detach()
        self.rm_bn1.bias.data[:out_channel_1] = self.bn1.bias.data.clone().detach()
        self.rm_bn1.running_mean[:out_channel_1] = self.bn1.running_mean.clone().detach()
        self.rm_bn1.running_var[:out_channel_1] = self.bn1.running_var.clone().detach()
        self.rm_bn1.weight.data[out_channel_1:] = 1
        self.rm_bn1.bias.data[out_channel_1:] = 0
        self.rm_bn1.running_mean[out_channel_1:] = 0
        self.rm_bn1.running_var[out_channel_1:] = 1

        # for conv2
        self.rm_conv2 = nn.Conv2d(out_channel_1 + self.in_channel, out_channel_1 + self.out_channel, kernel_size=3, stride=self.stride, padding=1,
                                  bias=True).to(device).eval()
        nn.init.zeros_(self.rm_conv2.weight)
        nn.init.zeros_(self.rm_conv2.bias)
        self.rm_conv2.weight.data[:out_channel_1, :out_channel_1] = self.conv2.weight.data.clone().detach()
        if hasattr(self, 'downsample') and self.downsample is not None:
            fused_downsample_conv = nn.utils.fuse_conv_bn_eval(self.downsample[0], self.downsample[1]).eval()
            fused_weight = fused_downsample_conv.weight.data.clone().detach()
            if fused_downsample_conv.kernel_size == (1, 1):
                fused_weight = F.pad(fused_weight, [1, 1, 1, 1], value=0)

            self.rm_conv2.weight.data[out_channel_1:, out_channel_1:] = fused_weight
            self.rm_conv2.bias.data[out_channel_1:] = fused_downsample_conv.bias.data.clone().detach()
            del fused_downsample_conv
        else:
            self.rm_conv2.weight.data[:out_channel_1, :out_channel_1] = self.conv2.weight.data.clone().detach()
            nn.init.dirac_(self.rm_conv2.weight[out_channel_1:, out_channel_1:])

        # for bn2
        self.rm_bn2 = nn.BatchNorm2d(out_channel_1 + self.out_channel).to(device).eval()
        self.rm_bn2.weight.data[:out_channel_1] = self.bn2.weight.data.clone().detach()
        self.rm_bn2.bias.data[:out_channel_1] = self.bn2.bias.data.clone().detach()
        self.rm_bn2.running_mean[:out_channel_1] = self.bn2.running_mean.clone().detach()
        self.rm_bn2.running_var[:out_channel_1] = self.bn2.running_var.clone().detach()
        self.rm_bn2.weight.data[out_channel_1:] = 1
        self.rm_bn2.bias.data[out_channel_1:] = 0
        self.rm_bn2.running_mean[out_channel_1:] = 0
        self.rm_bn2.running_var[out_channel_1:] = 1

        # for conv3 and bn3
        fused_conv3_bn3 = nn.utils.fuse_conv_bn_eval(self.conv3, self.bn3)
        self.rm_conv3 = nn.Conv2d(out_channel_1 + self.out_channel, self.out_channel, 1, 1, 0, bias=True).to(device).eval()
        self.rm_conv3.weight.data[:, :out_channel_1] = fused_conv3_bn3.weight.data.clone().detach()
        self.rm_conv3.bias.data = fused_conv3_bn3.bias.data.clone().detach()
        nn.init.dirac_(self.rm_conv3.weight.data[:, out_channel_1:])
        del fused_conv3_bn3

        print('deploy successfully......')

        if self.delete_attr:
            self.delete_raw_attribute(verbose=verbose)

    def delete_raw_attribute(self, verbose=False):
        for (key, value) in list(self.attr_list):
            if isinstance(value, self.delete_list):
                if verbose:
                    print('self.{} will be deleted when call deploy func......'.format(key))
                delattr(self, key)


class NormalRMBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, delete_attr=True):
        super(NormalRMBottleNeck, self).__init__()

        self.in_channel = inplanes
        self.out_channel = planes * 4
        self.channel = planes
        self.delete_attr = delete_attr
        self.stride = stride

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or self.in_channel != self.out_channel:
            down_sample_kernel = 1
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * 4, down_sample_kernel, stride, down_sample_kernel // 2, bias=False),
                                            nn.BatchNorm2d(planes * 4))
        else:
            self.downsample = None

        self.deploy_flag = False
        self.delete_list = (nn.BatchNorm2d, nn.Conv2d, nn.Sequential, nn.ModuleList)
        self.attr_list = list(self.__dict__['_modules'].items())

        self.running1 = nn.BatchNorm2d(inplanes, affine=False)
        self.running2 = nn.BatchNorm2d(self.out_channel, affine=False)

    def forward(self, x):
        if self.training or not self.deploy_flag:
            self.running1(x)

            residual = x
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            if self.downsample is not None:
                residual = self.downsample(residual)
            x += residual
            self.running2(x)
            x = self.relu(x)
            return x
        else:
            x = self.relu(self.rm_bn1(self.rm_conv1(x)))
            x = self.relu(self.rm_bn2(self.rm_conv2(x)))
            x = self.relu(self.rm_bn3(self.rm_conv3(x)))
            return x

    def deploy(self, verbose=False):
        if self.deploy_flag is True:
            return

        self.deploy_flag = True
        device = self.conv1.weight.device

        out_channel_1, in_channel_1 = self.conv1.weight.size()[:2]

        # for conv1
        self.rm_conv1 = nn.Conv2d(in_channel_1, out_channel_1 + self.in_channel, 1, bias=False).to(device).eval()
        self.rm_conv1.weight.data[:out_channel_1] = self.conv1.weight.data.clone().detach()
        nn.init.dirac_(self.rm_conv1.weight[out_channel_1:])

        # for bn1
        self.rm_bn1 = nn.BatchNorm2d(out_channel_1 + self.in_channel).to(device).eval()
        self.rm_bn1.weight.data[:out_channel_1] = self.bn1.weight.data.clone().detach()
        self.rm_bn1.bias.data[:out_channel_1] = self.bn1.bias.data.clone().detach()
        self.rm_bn1.running_mean.data[:out_channel_1] = self.bn1.running_mean.data.clone().detach()
        self.rm_bn1.running_var.data[:out_channel_1] = self.bn1.running_var.data.clone().detach()
        self.rm_bn1.running_mean.data[out_channel_1:] = self.running1.running_mean.data.clone().detach()
        self.rm_bn1.running_var.data[out_channel_1:] = self.running1.running_var.data.clone().detach()
        self.rm_bn1.weight.data[out_channel_1:] = torch.sqrt(self.running1.running_var + self.running1.eps).clone().detach()
        self.rm_bn1.bias.data[out_channel_1:] = self.running1.running_mean.data.clone().detach()

        # for conv2
        self.rm_conv2 = nn.Conv2d(out_channel_1 + self.in_channel, out_channel_1 + self.in_channel, 3, self.stride, 1, bias=False).to(device).eval()
        nn.init.zeros_(self.rm_conv2.weight)
        self.rm_conv2.weight.data[:out_channel_1, :out_channel_1] = self.conv2.weight.data.clone().detach()
        nn.init.dirac_(self.rm_conv2.weight[out_channel_1:, out_channel_1:])

        # for bn2
        self.rm_bn2 = nn.BatchNorm2d(out_channel_1 + self.in_channel).to(device).eval()
        self.rm_bn2.weight.data[:out_channel_1] = self.bn2.weight.data.clone().detach()
        self.rm_bn2.bias.data[:out_channel_1] = self.bn2.bias.data.clone().detach()
        self.rm_bn2.running_mean.data[:out_channel_1] = self.bn2.running_mean.data.clone().detach()
        self.rm_bn2.running_var.data[:out_channel_1] = self.bn2.running_var.data.clone().detach()
        self.rm_bn2.running_mean.data[out_channel_1:] = self.running1.running_mean.data.clone().detach()
        self.rm_bn2.running_var.data[out_channel_1:] = self.running1.running_var.data.clone().detach()
        self.rm_bn2.weight.data[out_channel_1:] = torch.sqrt(self.running1.running_var + self.running1.eps).clone().detach()
        self.rm_bn2.bias.data[out_channel_1:] = self.running1.running_mean.data.clone().detach()

        # for conv3
        fused_conv3_bn3 = nn.utils.fuse_conv_bn_eval(self.conv3, self.bn3)
        self.rm_conv3 = nn.Conv2d(out_channel_1 + self.in_channel, self.out_channel, 1, 1, 0, bias=True).to(device).eval()
        self.rm_conv3.weight.data[:, :out_channel_1] = fused_conv3_bn3.weight.data.clone().detach()
        self.rm_conv3.bias.data = fused_conv3_bn3.bias.data.clone().detach()
        if hasattr(self, 'downsample') and self.downsample is not None:
            fuse_downsample_conv = nn.utils.fuse_conv_bn_eval(self.downsample[0], self.downsample[1])
            fuse_weight = fuse_downsample_conv.weight.data.clone().detach()
            self.rm_conv3.weight.data[:, out_channel_1:] = fuse_weight
            self.rm_conv3.bias.data += fuse_downsample_conv.bias.data.clone().detach()
            del fuse_downsample_conv
        else:
            nn.init.dirac_(self.rm_conv3.weight[:, out_channel_1:])
        del fused_conv3_bn3

        # for bn3
        self.rm_bn3 = nn.BatchNorm2d(self.out_channel).to(device).eval()
        self.rm_bn3.running_mean.data = self.running2.running_mean.data.clone().detach()
        self.rm_bn3.running_var.data = self.running2.running_var.data.clone().detach()
        self.rm_bn3.weight.data = torch.sqrt(self.running2.running_var + self.running2.eps).clone().detach()
        self.rm_bn3.bias.data = self.running2.running_mean.data.clone().detach()

        print('deploy successfully......')
        if self.delete_attr:
            self.delete_raw_attribute(verbose)

        return nn.ModuleList([self.rm_conv1, self.rm_bn1, self.relu, self.rm_conv2, self.rm_bn2, self.relu, self.rm_conv3, self.rm_bn3, self.relu])

    def delete_raw_attribute(self, verbose=False):
        for (key, value) in list(self.attr_list):
            if isinstance(value, self.delete_list):
                if verbose:
                    print('self.{} will be deleted when call deploy func......'.format(key))
                delattr(self, key)


def testBottleNeck():
    model = SimplifyRMBottleneck(inplanes=4, planes=3, stride=1).train()
    for i in range(10):
        x = torch.rand(6, 4, 4, 4)
        model(x)

    model.eval()
    print(model)
    x = torch.rand(1, 4, 4, 4) - 0.5

    y = model(x).flatten()
    print(y)

    model.deploy()
    y2 = model(x).flatten()
    print(y2)
    print(model)


def testNormalRMBottleNeck():
    model = NormalRMBottleNeck(inplanes=4, planes=1, stride=2)
    for _ in range(5):
        x = torch.rand(6, 4, 4, 4)
        model(x)

    model.eval()

    x = torch.rand(1, 4, 4, 4)
    y = model(x).flatten()
    print(y)

    model.deploy(True)
    y2 = model(x).flatten()
    print(y2)


if __name__ == '__main__':
    testNormalRMBottleNeck()
