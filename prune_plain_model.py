import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from RMResnet import rmresnet, Head


class ConvBn(nn.Module):
    def __init__(self, conv_module, bn_module, act_module=None, conv_index=-1, bn_index=-1, block_index=-1) -> None:
        super(ConvBn, self).__init__()
        self.conv = conv_module
        self.bn = bn_module  # if one channel's bn weight.abs() is lower than self.bn, it will be pruned
        self.act = act_module
        self.input_mask = None
        self.output_mask = None
        self.conv_index = conv_index
        self.bn_index = bn_index
        self.block_index = block_index

    def __repr__(self) -> str:
        str_format = "block index: {}\tconv: {}\tbn: {}\tconv_index: {}\tbn_index: {}"
        return str_format.format(self.block_index, self.conv, self.bn, self.conv_index, self.bn_index)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x

    def add_act(self, module: nn.Module):
        assert isinstance(module, (nn.ReLU, nn.PReLU))
        self.act = module


class PruneRMModel(nn.Module):
    def __init__(self, model: nn.Module, prune_percentage=0.5, channel_limit=8):
        super(PruneRMModel, self).__init__()
        self.bn_thresh = -1
        self.prune_percentage = prune_percentage
        self.channel_limit = channel_limit
        self.has_prune = False
        self.model = deepcopy(model)
        self.model.eval()

        if not hasattr(self.model, 'deploy_features'):
            raise RuntimeError('cur model has not been converted to plain model, you should convert to plain model......')
        deploy_features = getattr(self.model, 'deploy_features')

        self.wrap_list = nn.ModuleList()
        for i in range(len(deploy_features) - 1):
            if not isinstance(deploy_features[i], nn.Conv2d) or not isinstance(deploy_features[i + 1], nn.BatchNorm2d):
                continue
            module = ConvBn(deploy_features[i], deploy_features[i + 1])
            if i + 2 < len(deploy_features) and isinstance(deploy_features[i + 2], nn.PReLU):
                module.add_act(deploy_features[i + 2])
            self.wrap_list.append(module)

        self.generate_mask()

    def get_bn_thresh(self):
        print('before calc, bn thresh is: {}'.format(self.bn_thresh))
        weight_list = []

        for module in self.wrap_list:
            weight_list.append(module.bn.weight.data.abs().clone().detach())
        weight_list = torch.cat(weight_list, dim=0)
        weight_list = torch.sort(weight_list, dim=0)[0]

        index = int(weight_list.size(0) * self.prune_percentage)
        self.bn_thresh = float(weight_list[index].item())
        print('after calc, bn thresh is: {}'.format(self.bn_thresh))

    def generate_mask(self):
        if self.bn_thresh == -1:
            self.get_bn_thresh()

        input_mask = None
        output_mask = None
        device = self.wrap_list[0].conv.weight.device

        for i, module in enumerate(self.wrap_list):
            if i == 0:
                input_mask = torch.ones(module.conv.weight.data.size(1), device=device)
            else:
                input_mask = output_mask

            bn_weight = module.bn.weight
            if i == len(self.wrap_list) - 1:
                output_mask = torch.ones(bn_weight.data.size(0), device=device)
            else:
                remain_channel = bn_weight.data.abs().ge(self.bn_thresh).sum().item()
                real_remain_channel = remain_channel // self.channel_limit * self.channel_limit + (
                    0 if remain_channel % self.channel_limit == 0 else self.channel_limit)
                if real_remain_channel == 0:
                    real_remain_channel = self.channel_limit
                output_mask = torch.ones(bn_weight.size(0), device=device)
                real_prune_channel = bn_weight.size(0) - real_remain_channel
                index = bn_weight.data.abs().clone().detach().argsort(dim=0)
                output_mask[index[:real_prune_channel]] = 0
                print('total channel: {}\tprune channel: {}\tremain channel: {}'.format(bn_weight.size(0), real_prune_channel, real_remain_channel))

            module.input_mask = input_mask
            module.output_mask = output_mask

    def prune_model_keep_size(self):
        for i in range(len(self.wrap_list) - 1):
            cur_conv_bn = self.wrap_list[i]
            cur_output_mask = cur_conv_bn.output_mask
            cur_conv_bn.bn.weight.data.mul_(cur_output_mask)

            activation = F.relu((1 - cur_output_mask) * cur_conv_bn.bn.bias.data)
            next_conv_bn = self.wrap_list[i + 1]
            conv_sum = next_conv_bn.conv.weight.data.sum(dim=(2, 3))
            offset = torch.matmul(conv_sum, activation.view(-1, 1)).view(-1)
            next_conv_bn.bn.running_mean.data.sub_(offset)
            cur_conv_bn.bn.bias.data.mul_(cur_output_mask)

        self.has_prune = True

    def get_compact_model(self, verbose=False):
        if not self.has_prune:
            self.prune_model_keep_size()

        compact_model = deepcopy(self.model)
        deploy_features = getattr(compact_model, 'deploy_features')
        wrap_list = list()
        for i in range(len(deploy_features) - 1):
            if not isinstance(deploy_features[i], nn.Conv2d) or not isinstance(deploy_features[i + 1], nn.BatchNorm2d):
                continue
            wrap_list.append(ConvBn(deploy_features[i], deploy_features[i + 1]))

        assert len(wrap_list) == len(self.wrap_list)

        for i in range(len(wrap_list)):
            compact_conv_bn = wrap_list[i]
            raw_conv_bn = self.wrap_list[i]

            input_mask = raw_conv_bn.input_mask
            output_mask = raw_conv_bn.output_mask

            # prune bn
            compact_conv_bn.bn.weight.data = raw_conv_bn.bn.weight.data[output_mask.bool()].clone()
            compact_conv_bn.bn.bias.data = raw_conv_bn.bn.bias.data[output_mask.bool()].clone()
            compact_conv_bn.bn.running_mean.data = raw_conv_bn.bn.running_mean.data[output_mask.bool()].clone()
            compact_conv_bn.bn.running_var.data = raw_conv_bn.bn.running_var.data[output_mask.bool()].clone()

            # prune conv
            compact_conv_bn.conv.weight.data = raw_conv_bn.conv.weight.data[output_mask.bool()][:, input_mask.bool(), ...].clone().detach()
            if compact_conv_bn.conv.bias is not None:
                compact_conv_bn.conv.bias.data = raw_conv_bn.conv.bias.data[output_mask.bool()]

            if verbose:
                raw_size = raw_conv_bn.conv.weight.size()
                combine_size = compact_conv_bn.conv.weight.size()
                print('weight transfer {} to {}'.format(raw_size, combine_size))

        return compact_model

    def forward(self, x):
        return self.model(x)


def build_model(device='cpu'):
    model = rmresnet(layers=[1, 1, 1, 1], last_stride=2, head_dim=2048).to(device)
    x = torch.rand(2, 3, 32, 32)
    for _ in range(20):
        _ = model(x)

    model.eval()
    deploy_model = deepcopy(model)
    deploy_model._deploy()
    model.eval()
    deploy_model.eval()
    return model, deploy_model


def test():
    device = 'cpu'
    model, deploy_model = build_model(device=device)

    prune_model = PruneRMModel(deploy_model, prune_percentage=0.3)
    compact_model = prune_model.get_compact_model(False)

    model.eval()
    deploy_model.eval()
    compact_model.eval()
    x = torch.rand(1, 3, 224, 224)
    y1 = model(x)
    y2 = deploy_model(x)
    y3 = compact_model(x)
    y4 = prune_model(x)
    print(y1)
    print(y2)
    print(y3)
    print(y4)


if __name__ == '__main__':
    test()
