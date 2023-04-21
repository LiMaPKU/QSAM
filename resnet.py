import torch
import torch.nn as nn
from functools import partial
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import math

__all__ = ['ResNet', 'resnet18', ]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


class MultiBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, bn_num=1):
        super(MultiBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.bns = nn.ModuleList()

        for i in range(bn_num):
            self.bns.append(nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                           track_running_stats=track_running_stats).cuda())
        self.batch_type = None
        self.bn_num = bn_num

    def forward(self, input):
        if isinstance(self.batch_type, int):
            if self.batch_type == 0:
                input = super(MultiBatchNorm2d, self).forward(input)
            else:
                input = self.bns[self.batch_type - 1](input)
        elif isinstance(self.batch_type, list):
            assert input.shape[0] % self.bn_num == 0
            batch_size = input.shape[0] // self.bn_num
            tmp = []
            for i in range(self.bn_num):
                if i == 0:
                    tmp.append(super(MultiBatchNorm2d, self).forward(input[:batch_size]))
                else:
                    tmp.append(self.bns[i - 1](input[i * batch_size: (i + 1) * batch_size]))
            input = torch.cat(tmp, 0)
        else:
            assert isinstance(self.batch_type, torch.Tensor)
            tmp = []
            for i in range(self.bn_num):
                if i == 0:
                    tmp.append(super(MultiBatchNorm2d, self).forward(input).unsqueeze(dim=-1))
                else:
                    tmp.append(self.bns[i - 1](input).unsqueeze(dim=-1))
            input = torch.cat(tmp, dim=-1)
            input = input.mul(self.batch_type.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(input)).sum(-1)

        return input


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bn_num=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, bn_num=bn_num)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, bn_num=bn_num)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bn_num=1):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, bn_num=bn_num)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, bn_num=bn_num)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, bn_num=bn_num)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, bn_num=1):
        super(ResNet, self).__init__()

        self._norm_layer = norm_layer
        self.bn_num = bn_num

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes, bn_num=self.bn_num)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, bn_num=self.bn_num),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, bn_num=self.bn_num))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, bn_num=self.bn_num))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        y = self.avgpool(x)
        x = torch.flatten(y, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class AdvResNet(ResNet):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, bn_num=1):
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
                         groups=groups, width_per_group=width_per_group,
                         replace_stride_with_dilation=replace_stride_with_dilation,
                         norm_layer=norm_layer, bn_num=bn_num)
        self.multibn = 0

    def set_multibn(self, multibn):
        self.multibn = multibn

    def forward(self, x):
        if isinstance(self.multibn, int):
            self.apply(partial(to_status, status=self.multibn))
            return self._forward_impl(x)
        elif isinstance(self.multibn, list):
            self.apply(partial(to_status, status=self.multibn))
            return self._forward_impl(x)
        elif isinstance(self.multibn, torch.Tensor):
            self.apply(partial(to_status, status=self.multibn))
            return self._forward_impl(x)
        else:
            raise NotImplementedError


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = AdvResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError('do not set pretrained as True, since we aim at training from scratch')
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True)[0].data, requires_grad=False)

            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.linear(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt)
    else:
        return F.linear(inputs, weight, bias)


class MetaMLP(nn.Module):
    def __init__(self, out_channel, channel=2, norm='linear'):
        super(MetaMLP, self).__init__()
        self.fc1 = nn.Linear(64 * channel, 64 * channel)
        self.fc2 = nn.Linear(64 * channel, 128 * channel)
        self.fc3 = nn.Linear(128 * channel, out_channel)
        self.norm = norm

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):
        x = torch.flatten(x, start_dim=1)
        x = linear(inputs=x,
                   weight=self.fc1.weight,
                   bias=self.fc1.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        x = F.relu(x, inplace=True)
        x = linear(inputs=x,
                   weight=self.fc2.weight,
                   bias=self.fc2.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        x = F.relu(x, inplace=True)
        x = linear(inputs=x,
                   weight=self.fc3.weight,
                   bias=self.fc3.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        if self.norm == 'softmax':
            x = F.softmax(x)
        elif self.norm == 'linear':
            x = x / (torch.sum(x, dim=1, keepdim=True).clamp_max(-1e-6) + torch.sum(x, dim=1, keepdim=True).clamp_min(
                1e-5))
        else:
            raise NotImplementedError
        return x
