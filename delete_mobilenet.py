import torch
import torch.nn as nn

torch.set_printoptions(sci_mode=False)


class WrapReLU6(nn.ReLU6):
    def __init__(self, mid_channel: int, inplace=True):
        super(WrapReLU6, self).__init__(inplace)
        self.mid_channel = mid_channel

    def forward(self, x):
        out = torch.cat([super(WrapReLU6, self).forward(x[:, :self.mid_channel]), x[:, self.mid_channel:]], dim=1)
        return out


class SimplifyInvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(SimplifyInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_channel = int(round(in_channel * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channel == out_channel

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channel = hidden_channel

        self.conv1 = nn.Conv2d(in_channel, hidden_channel, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.act = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, 3, stride, 1, groups=hidden_channel, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channel)
        self.conv3 = nn.Conv2d(hidden_channel, out_channel, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.deploy_flag = False

        self.delete_list = (nn.BatchNorm2d, nn.Conv2d, nn.Sequential, nn.ModuleList)
        self.attr_list = list(self.__dict__['_modules'].items())

    def forward(self, x):
        if self.training or not self.deploy_flag:
            residual = x if self.use_res_connect else None
            x = self.act(self.bn1(self.conv1(x)))
            x = self.act(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            if residual is not None:
                x += residual
            return x
        else:
            x = self.rm_act(self.rm_bn1(self.rm_conv1(x)))
            x = self.rm_act(self.rm_bn2(self.rm_conv2(x)))
            x = self.rm_conv3(x)
            return x

    def deploy(self, verbose=False):
        if not self.use_res_connect:
            assert self.deploy_flag is False
            print('no need to deploy this module......')
            return

        if self.deploy_flag:
            return

        self.deploy_flag = True
        device = self.conv1.weight.device
        out_channel_1, in_channel_1 = self.conv1.weight.data.size()[:2]

        # for conv1
        self.rm_conv1 = nn.Conv2d(in_channel_1, out_channel_1 + self.in_channel, 1, bias=False).to(device).eval()
        self.rm_conv1.weight.data[:out_channel_1] = self.conv1.weight.data.clone().detach()
        nn.init.dirac_(self.rm_conv1.weight[out_channel_1:])

        # for bn1
        self.rm_bn1 = nn.BatchNorm2d(out_channel_1 + self.in_channel).to(device).eval()
        self.rm_bn1.weight.data[:out_channel_1] = self.bn1.weight.data[:out_channel_1].clone().detach()
        self.rm_bn1.bias.data[:out_channel_1] = self.bn1.bias.data[:out_channel_1].clone().detach()
        self.rm_bn1.running_mean[:out_channel_1] = self.bn1.running_mean[:out_channel_1].clone().detach()
        self.rm_bn1.running_var[:out_channel_1] = self.bn1.running_var[:out_channel_1].clone().detach()
        self.rm_bn1.weight.data[out_channel_1:] = 1
        self.rm_bn1.bias.data[out_channel_1:] = 0
        self.rm_bn1.running_mean[out_channel_1:] = 0
        self.rm_bn1.running_var[out_channel_1:] = 1

        # for relu
        self.rm_act = WrapReLU6(mid_channel=out_channel_1)

        # for conv2
        self.rm_conv2 = nn.Conv2d(out_channel_1 + self.in_channel, out_channel_1 + self.in_channel, 3, self.stride, 1,
                                  groups=out_channel_1 + self.in_channel, bias=False).to(device).eval()
        self.rm_conv2.weight.data[:out_channel_1] = self.conv2.weight.data.clone().detach()
        nn.init.zeros_(self.rm_conv2.weight[out_channel_1:])
        self.rm_conv2.weight.data[out_channel_1:, :, 1, 1] = 1

        # for bn2
        self.rm_bn2 = nn.BatchNorm2d(out_channel_1 + self.in_channel).to(device).eval()
        self.rm_bn2.weight.data[:out_channel_1] = self.bn2.weight.data.clone().detach()
        self.rm_bn2.bias.data[:out_channel_1] = self.bn2.bias.data.clone().detach()
        self.rm_bn2.running_mean[:out_channel_1] = self.bn2.running_mean.clone().detach()
        self.rm_bn2.running_var[:out_channel_1] = self.bn2.running_var.clone().detach()
        self.rm_bn2.weight.data[out_channel_1:] = 1
        self.rm_bn2.bias.data[out_channel_1:] = 0
        self.rm_bn2.running_mean[out_channel_1:] = 0
        self.rm_bn2.running_var[out_channel_1:] = 1

        # for conv3  self.conv3 = nn.Conv2d(hidden_channel, out_channel, 1, 1, 0, bias=False)
        fused_conv_bn = nn.utils.fuse_conv_bn_eval(self.conv3, self.bn3)
        self.rm_conv3 = nn.Conv2d(out_channel_1 + self.in_channel, self.out_channel, 1, 1, 0, bias=True).to(device).eval()
        self.rm_conv3.weight.data[:, :out_channel_1] = fused_conv_bn.weight.data.clone().detach()
        self.rm_conv3.bias.data = fused_conv_bn.bias.data.clone().detach()
        nn.init.dirac_(self.rm_conv3.weight[:, out_channel_1:])
        del fused_conv_bn
        print('deploy successfully......')

        if self.delete_list:
            self.delete_raw_attribute(verbose)

    def delete_raw_attribute(self, verbose=False):
        for (key, value) in list(self.attr_list):
            if isinstance(value, self.delete_list):
                if verbose:
                    print('self.{} will be deleted when call deploy func......'.format(key))
                delattr(self, key)


class NormalInvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(NormalInvertedResidual, self).__init__()
        assert stride in (1, 2)

        hidden_channel = int(round(in_channel * expand_ratio))
        self.use_res_connect = stride == 1 and in_channel == out_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channel = hidden_channel

        self.conv1 = nn.Conv2d(in_channel, hidden_channel, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.act = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, 3, stride, 1, groups=hidden_channel, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channel)
        self.conv3 = nn.Conv2d(hidden_channel, out_channel, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.deploy_flag = False

        self.delete_list = (nn.BatchNorm2d, nn.Conv2d, nn.Sequential, nn.ModuleList)
        self.attr_list = list(self.__dict__['_modules'].items())

        self.running1 = nn.BatchNorm2d(in_channel, affine=False) if self.use_res_connect else nn.Identity
        self.running2 = nn.BatchNorm2d(out_channel, affine=False) if self.use_res_connect else nn.Identity

    def forward(self, x):
        if not self.use_res_connect or self.training or not self.deploy_flag:
            self.running1(x)
            residual = x if self.use_res_connect else None
            x = self.act(self.bn1(self.conv1(x)))
            x = self.act(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            if residual is not None:
                x += residual
            self.running2(x)
            return x
        else:
            x = self.rm_act1(self.rm_bn1(self.rm_conv1(x)))
            x = self.rm_act2(self.rm_bn2(self.rm_conv2(x)))
            x = self.rm_bn3(self.rm_conv3(x))
            return x

    def deploy(self, verbose=False):
        if self.deploy_flag or not self.use_res_connect:
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
        self.rm_bn1.weight.data[:out_channel_1] = self.bn1.weight.data[:out_channel_1].clone().detach()
        self.rm_bn1.bias.data[:out_channel_1] = self.bn1.bias.data[:out_channel_1].clone().detach()
        self.rm_bn1.running_mean[:out_channel_1] = self.bn1.running_mean[:out_channel_1].clone().detach()
        self.rm_bn1.running_var[:out_channel_1] = self.bn1.running_var[:out_channel_1].clone().detach()
        self.rm_bn1.weight.data[out_channel_1:] = torch.sqrt(self.running1.running_mean + self.running1.eps).clone().detach()
        self.rm_bn1.bias.data[out_channel_1:] = self.running1.running_mean.clone().detach()
        self.rm_bn1.running_mean[out_channel_1:] = self.running1.running_mean.clone().detach()
        self.rm_bn1.running_var[out_channel_1:] = self.running1.running_var.clone().detach()

        # for act1
        self.rm_act1 = nn.PReLU(out_channel_1 + self.in_channel, init=1)

        # for conv2
        self.rm_conv2 = nn.Conv2d(out_channel_1 + self.in_channel, out_channel_1 + self.in_channel, 3, self.stride, 1,
                                  groups=out_channel_1 + self.in_channel, bias=False).to(device).eval()
        self.rm_conv2.weight.data[:out_channel_1] = self.conv2.weight.data.clone().detach()
        nn.init.zeros_(self.rm_conv2.weight[out_channel_1:])
        self.rm_conv2.weight.data[out_channel_1:, :, 1, 1] = 1

        # for bn2
        self.rm_bn2 = nn.BatchNorm2d(out_channel_1 + self.in_channel).to(device).eval()
        self.rm_bn2.weight.data[:out_channel_1] = self.bn2.weight.data.clone().detach()
        self.rm_bn2.bias.data[:out_channel_1] = self.bn2.bias.data.clone().detach()
        self.rm_bn2.running_mean[:out_channel_1] = self.bn2.running_mean.clone().detach()
        self.rm_bn2.running_var[:out_channel_1] = self.bn2.running_var.clone().detach()
        self.rm_bn2.weight.data[out_channel_1:] = torch.sqrt(self.running1.running_mean + self.running1.eps).clone().detach()
        self.rm_bn2.bias.data[out_channel_1:] = self.running1.running_mean.clone().detach()
        self.rm_bn2.running_mean[out_channel_1:] = self.running1.running_mean.clone().detach()
        self.rm_bn2.running_var[out_channel_1:] = self.running1.running_var.clone().detach()

        # for act2
        self.rm_act2 = nn.PReLU(out_channel_1 + self.in_channel, init=1)

        # for conv3
        fused_conv_bn = nn.utils.fuse_conv_bn_eval(self.conv3, self.bn3)
        self.rm_conv3 = nn.Conv2d(out_channel_1 + self.in_channel, self.out_channel, 1, 1, 0, bias=True).to(device).eval()
        self.rm_conv3.weight.data[:, :out_channel_1] = fused_conv_bn.weight.data.clone().detach()
        self.rm_conv3.bias.data = fused_conv_bn.bias.data.clone().detach()
        nn.init.dirac_(self.rm_conv3.weight[:, out_channel_1:])
        del fused_conv_bn

        # for bn3
        self.rm_bn3 = nn.BatchNorm2d(self.out_channel).to(device).eval()
        self.rm_bn3.weight.data = torch.sqrt(self.running2.running_mean + self.running2.eps).clone().detach()
        self.rm_bn3.bias.data = self.running2.running_mean.clone().detach()
        self.rm_bn3.running_mean = self.running2.running_mean.clone().detach()
        self.rm_bn3.running_var = self.running2.running_var.clone().detach()

        print('deploy successfully......')

        if self.delete_list:
            self.delete_raw_attribute(verbose)

        return nn.ModuleList([self.rm_conv1, self.rm_bn1, self.rm_act1, self.rm_conv2, self.rm_bn2, self.rm_act2, self.rm_conv3, self.rm_bn3])

    def delete_raw_attribute(self, verbose=False):
        for (key, value) in list(self.attr_list):
            if isinstance(value, self.delete_list):
                if verbose:
                    print('self.{} will be deleted when call deploy func......'.format(key))
                delattr(self, key)


def test():
    model = SimplifyInvertedResidual(in_channel=2, out_channel=2, stride=2, expand_ratio=1)
    x = torch.rand(2, 2, 4, 4)
    for _ in range(10):
        model(x)

    x = torch.rand(1, 2, 4, 4)
    model.eval()
    y1 = model(x)
    model.deploy(True)
    y2 = model(x)

    print(y1.flatten())
    print(y2.flatten())


def testNormalInvertedResidual():
    model = NormalInvertedResidual(in_channel=2, out_channel=2, stride=2, expand_ratio=1)
    x = torch.rand(2, 2, 4, 4)
    for _ in range(5):
        model(x)

    model.eval()
    x = torch.rand(1, 2, 4, 4) - 0.5
    y1 = model(x)
    model.deploy(True)
    y2 = model(x)

    print(y1.flatten())
    print(y2.flatten())


if __name__ == '__main__':
    testNormalInvertedResidual()
