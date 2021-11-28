import torch
import torch.nn as nn
import torch.nn.functional as F


class WrapReLU(nn.ReLU):
    def __init__(self, inplace=True):
        super(WrapReLU, self).__init__(inplace)

    def forward(self, x):
        channel = int(x.size()[1])
        out = torch.cat([super(WrapReLU, self).forward(x[:, :channel // 2]), x[:, channel // 2:]], dim=1)
        return out


class SimplifyRMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, delete_attr=True):
        super(SimplifyRMBasicBlock, self).__init__()
        self.stride = stride
        self.delete_attr = delete_attr

        self.conv = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.act = nn.ReLU(True)
        self.act2 = WrapReLU(True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False), nn.BatchNorm2d(planes))
        else:
            self.downsample = None

        self.deploy_flag = False

        self.delete_list = (nn.BatchNorm2d, nn.Conv2d, nn.Sequential, nn.ModuleList)
        self.attr_list = list(self.__dict__['_modules'].items())

    def forward(self, x):
        if self.training or not self.deploy_flag:
            residual = x
            x = self.conv(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)

            if self.downsample is not None:
                residual = self.downsample(residual)

            x = x + residual
            x = self.act(x)
            return x
        else:
            x = self.rm_conv(x)
            x = self.rm_bn(x)
            x = self.act2(x)
            x = self.rm_conv2(x)
            x = self.act(x)
            return x

    def deploy(self, verbose=False):
        # for first conv
        if self.deploy_flag:
            print('model has been already deployed......')
            return
        self.deploy_flag = True
        device = self.conv.weight.device

        out_channel, in_channel = self.conv.weight.data.size()[:2]
        self.rm_conv = nn.Conv2d(in_channel, out_channel * 2, 3, self.stride, 1, bias=True).to(device).eval()
        nn.init.zeros_(self.rm_conv.bias)
        self.rm_conv.weight.data[:out_channel] = self.conv.weight.data
        if hasattr(self, 'downsample') and self.downsample is not None:
            fuse_downsample_conv = nn.utils.fuse_conv_bn_eval(self.downsample[0], self.downsample[1])
            self.rm_conv.weight.data[out_channel:] = fuse_downsample_conv.weight.data.clone().detach()
            self.rm_conv.bias.data[out_channel:] = fuse_downsample_conv.bias.data.clone().detach()
            del fuse_downsample_conv
        else:
            nn.init.dirac_(self.rm_conv.weight[out_channel:])

        # for first bn
        self.rm_bn = nn.BatchNorm2d(2 * out_channel).to(device).eval()
        self.rm_bn.weight.data[:out_channel] = self.bn1.weight.data
        self.rm_bn.bias.data[:out_channel] = self.bn1.bias.data
        self.rm_bn.running_mean[:out_channel] = self.bn1.running_mean
        self.rm_bn.running_var[:out_channel] = self.bn1.running_var

        self.rm_bn.running_mean[out_channel:] = 0
        self.rm_bn.running_var[out_channel:] = 1
        self.rm_bn.weight.data[out_channel:] = 1
        self.rm_bn.bias.data[out_channel:] = 0

        # for second conv and bn
        fused_conv = nn.utils.fuse_conv_bn_eval(self.conv2, self.bn2)
        self.rm_conv2 = nn.Conv2d(out_channel * 2, out_channel, 3, 1, 1, bias=True).to(device).eval()
        self.rm_conv2.weight.data[:, :out_channel] = fused_conv.weight.data.clone().detach()
        self.rm_conv2.bias.data = fused_conv.bias.data.clone().detach()
        nn.init.dirac_(self.rm_conv2.weight[:, out_channel:])
        del fused_conv
        print('deploy successfully......')

        if self.delete_attr:
            self.delete_raw_attribute(verbose=verbose)

        return nn.ModuleList([self.rm_conv, self.rm_bn, self.act2, self.rm_conv2, self.act])

    def delete_raw_attribute(self, verbose=False):
        for (key, value) in list(self.attr_list):
            if isinstance(value, self.delete_list):
                if verbose:
                    print('self.{} will be deleted when call deploy func......'.format(key))
                delattr(self, key)


class NormalRMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, delete_attr=True):
        super(NormalRMBasicBlock, self).__init__()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        self.delete_attr = delete_attr

        self.conv = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.act = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, 0, bias=False), nn.BatchNorm2d(planes))
        else:
            self.downsample = None

        # 存在的原因：收集统计量。论文中提到了channel prune，因此需要BN的参数，这里进行统计，在删除residual后添加BN
        self.running_bn1 = nn.BatchNorm2d(inplanes, affine=False)
        self.running_bn2 = nn.BatchNorm2d(planes, affine=False)

        self.deploy_flag = False

        self.delete_list = (nn.BatchNorm2d, nn.Conv2d, nn.Sequential, nn.ModuleList)
        self.attr_list = list(self.__dict__['_modules'].items())

    def forward(self, x):
        if self.training or not self.deploy_flag:
            self.running_bn1(x)
            residual = x
            x = self.conv(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)

            if self.downsample is not None:
                residual = self.downsample(residual)

            x = x + residual
            self.running_bn2(x)

            x = self.act(x)
            return x
        else:
            x = self.rm_conv(x)
            x = self.rm_bn(x)
            x = self.act(x)

            x = self.rm_conv2(x)
            x = self.rm_bn2(x)
            x = self.act(x)

            return x

    def deploy(self, verbose=False):
        if self.deploy_flag:
            return

        self.deploy_flag = True
        device = self.conv.weight.device

        # for first conv
        out_channel, in_channel = self.conv.weight.data.size()[:2]
        self.rm_conv = nn.Conv2d(in_channel, out_channel + self.inplanes, 3, self.stride, 1, bias=False).to(device).eval()
        self.rm_conv.weight.data[:out_channel] = self.conv.weight.data
        nn.init.dirac_(self.rm_conv.weight[out_channel:])

        # for first bn
        self.rm_bn = nn.BatchNorm2d(out_channel + self.inplanes).to(device).eval()
        self.rm_bn.weight.data[:out_channel] = self.bn1.weight.data
        self.rm_bn.bias.data[:out_channel] = self.bn1.bias.data
        self.rm_bn.running_mean[:out_channel] = self.bn1.running_mean
        self.rm_bn.running_var[:out_channel] = self.bn1.running_var

        self.rm_bn.running_mean[out_channel:] = self.running_bn1.running_mean
        self.rm_bn.running_var[out_channel:] = self.running_bn1.running_var
        self.rm_bn.weight.data[out_channel:] = torch.sqrt(self.running_bn1.running_var + self.running_bn1.eps)
        self.rm_bn.bias.data[out_channel:] = self.running_bn1.running_mean

        # for second conv and bn
        fused_conv = nn.utils.fuse_conv_bn_eval(self.conv2, self.bn2)
        self.rm_conv2 = nn.Conv2d(out_channel + self.inplanes, out_channel, 3, 1, 1, bias=True).to(device).eval()
        self.rm_conv2.weight.data[:, :out_channel] = fused_conv.weight.data.clone().detach()
        self.rm_conv2.bias.data = fused_conv.bias.data.clone().detach()

        if hasattr(self, 'downsample') and self.downsample is not None:
            fuse_downsample_conv = nn.utils.fuse_conv_bn_eval(self.downsample[0], self.downsample[1])
            fuse_weight = fuse_downsample_conv.weight.data.clone().detach()
            if fuse_downsample_conv.kernel_size == (1, 1):
                fuse_weight = F.pad(fuse_weight, [1, 1, 1, 1], value=0)
            self.rm_conv2.weight.data[:, out_channel:] = fuse_weight
            self.rm_conv2.bias.data += fuse_downsample_conv.bias.data.clone().detach()
            del fuse_downsample_conv
        else:
            nn.init.dirac_(self.rm_conv2.weight[:, out_channel:])
        del fused_conv

        self.rm_bn2 = nn.BatchNorm2d(out_channel).to(device).eval()
        self.rm_bn2.running_mean = self.running_bn2.running_mean
        self.rm_bn2.running_var = self.running_bn2.running_var
        self.rm_bn2.weight.data = torch.sqrt(self.running_bn2.running_var + self.running_bn2.eps)
        self.rm_bn2.bias.data = self.running_bn2.running_mean

        print('deploy successfully......')

        if self.delete_attr:
            self.delete_raw_attribute(verbose=verbose)

        return nn.ModuleList([self.rm_conv, self.rm_bn, self.act, self.rm_conv2, self.rm_bn2, self.act])

    def delete_raw_attribute(self, verbose=False):
        for (key, value) in list(self.attr_list):
            if isinstance(value, self.delete_list):
                if verbose:
                    print('self.{} will be deleted when call deploy func......'.format(key))
                delattr(self, key)


def test():
    model = NormalRMBasicBlock(inplanes=2, planes=2, stride=1).train()
    for i in range(5):
        x = torch.rand(6, 2, 4, 4)
        model(x)

    model.eval()
    x = torch.rand(1, 2, 4, 4)
    assert (x > 0).all()

    y = model(x).flatten()
    print(y)

    model.deploy()
    y2 = model(x).flatten()
    print(y2)


if __name__ == '__main__':
    test()
