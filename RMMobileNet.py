import torch
import torch.nn as nn
from delete_mobilenet import SimplifyInvertedResidual


class Head(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Head, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        out = self.classifier(x)
        return out


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                                         nn.BatchNorm2d(out_planes),
                                         nn.ReLU6(inplace=True))


class RMMobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None, use_head=False):

        super(RMMobileNetV2, self).__init__()
        self.classifier = Head(1280, 10) if use_head else nn.Identity()

        if block is None:
            block = SimplifyInvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1],
                                         [6, 24, 2, 2],
                                         [6, 32, 3, 2],
                                         [6, 64, 4, 2],
                                         [6, 96, 3, 1],
                                         [6, 160, 3, 2],
                                         [6, 320, 1, 1]
                                         ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _deploy(self):
        if self.training:
            raise RuntimeError('make sure your model is in eval mode when you call "deploy" function......')
        for module in self.modules():
            if hasattr(module, 'deploy'):
                module.deploy()


def rmmobilenet(block=SimplifyInvertedResidual, use_head=False):
    model = RMMobileNetV2(block=block, width_mult=1., round_nearest=8, use_head=use_head)
    return model


def test():
    model = rmmobilenet(use_head=True).eval()
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.size())


if __name__ == '__main__':
    test()
