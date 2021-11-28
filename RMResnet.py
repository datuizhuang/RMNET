from delete_BottleNeck import NormalRMBottleNeck
from delete_BasicBlock import NormalRMBasicBlock

import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Head, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        out = self.classifier(x)
        return out


class RMReset(nn.Module):
    def __init__(self, block, layers, last_stride=2, width=1, head_dim=0, stem=False, **kwargs):
        super(RMReset, self).__init__()
        self.inplanes = 64

        if stem:
            self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.base = int(64 * width)

        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, self.base * 1, layers[0])
        self.layer2 = self.make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self.make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self.make_layer(block, self.base * 8, layers[3], stride=last_stride)

        self.head = Head(in_channel=head_dim, out_channel=10) if head_dim > 0 else nn.Identity()

        self.deploy_features = None

    def make_layer(self, block, planes, blocks, stride=1):
        layers = [block(inplanes=self.inplanes, planes=planes, stride=stride)]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        assert (x >= 0).all()

        if self.deploy_features is not None:
            x = self.deploy_features(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.head(x)

        return x

    def _deploy(self):
        if self.training:
            raise RuntimeError('make sure your model is in eval mode when you call "deploy" function......')

        module_list = nn.ModuleList()
        for module in self.modules():
            if hasattr(module, 'deploy'):
                cur_module_list = module.deploy()
                module_list.extend(cur_module_list)
        [delattr(self, "layer{}".format(i)) for i in range(1, 5)]

        self.deploy_features = nn.Sequential(*module_list)


def rmresnet(block=NormalRMBottleNeck, layers=[1, 1, 1, 1], last_stride=2, width=1, head_dim=0):
    model = RMReset(block, layers, last_stride, width, head_dim=head_dim)
    return model


def test():
    model = RMReset(block=NormalRMBottleNeck, layers=[2, 2, 2, 2], last_stride=2, width=1, head_dim=2048)
    t = torch.rand(2, 3, 224, 224) - 0.5
    for _ in range(5):
        model(t)
    model.eval()

    x = torch.rand(1, 3, 224, 224) - 0.5
    y = model(x)
    print(y)
    model._deploy()
    y2 = model(x)
    print(y2)

    print(model)


if __name__ == '__main__':
    test()
