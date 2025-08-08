from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from experiments.models import ct_model

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, activation_function=F.relu):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.activation_function = activation_function

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.activation_function(self.bn1(x))))
        out = self.conv2(self.activation_function(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation_function=F.relu):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.activation_function = activation_function

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.activation_function(self.bn1(self.conv1(x)))
        out = self.activation_function(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation_function(out)
        return out

class WideResNet(ct_model.CtModel):
    activation_function: object

    def __init__(self, depth, widen_factor, dataset, normalized, dropout_rate=0.0, num_classes=10,
                 factor=1, block=WideBasic, activation_function='relu', **kwargs):
        super(WideResNet, self).__init__(dataset=dataset, normalized=normalized, num_classes=num_classes)
        self.in_planes = 16
        self.activation_function = getattr(F, activation_function)

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (int)((depth-4)/6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0], stride=1)
        self.layer1 = self._wide_layer(block, nStages[1], n, dropout_rate, stride=factor, activation_function=self.activation_function)
        self.layer2 = self._wide_layer(block, nStages[2], n, dropout_rate, stride=2, activation_function=self.activation_function)
        self.layer3 = self._wide_layer(block, nStages[3], n, dropout_rate, stride=2, activation_function=self.activation_function)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(nStages[3], num_classes)
        self.blocks = [self.conv1, self.layer1, self.layer2, self.layer3]

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, activation_function):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, activation_function))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, targets=None, robust_samples=0, corruptions=None, mixup_alpha=0.0, mixup_p=0.0, manifold=False,
                manifold_noise_factor=1, cutmix_alpha=0.0, cutmix_p=0.0, noise_minibatchsize=1,
                concurrent_combinations=1, noise_sparsity=0.0, noise_patch_lower_scale = 1.0,
                noise_patch_upper_scale=1.0, generated_ratio=0.0, n2n_deepaugment=False, style_feats=None, **kwargs):
        
        x = super(WideResNet, self).forward_handle_greyscale(x)
        out = super(WideResNet, self).forward_normalize(x)
        out, mixed_targets = super(WideResNet, self).forward_noise_mixup(out, targets, robust_samples, corruptions,
                                        mixup_alpha, mixup_p, manifold, manifold_noise_factor, cutmix_alpha, cutmix_p,
                                        noise_minibatchsize, concurrent_combinations, noise_sparsity, noise_patch_lower_scale, 
                                        noise_patch_upper_scale, generated_ratio, n2n_deepaugment, style_feats, **kwargs)
        out = self.activation_function(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.training == True:
            return out, mixed_targets
        else:
            return out

def WideResNet_28_2(num_classes, factor, dataset, normalized, block=WideBasic, dropout_rate=0.0, activation_function='relu'):
    return WideResNet(depth=28, widen_factor=2, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, activation_function=activation_function)

def WideResNet_28_4(num_classes, factor, dataset, normalized, block=WideBasic, dropout_rate=0.0, activation_function='relu', **kwargs):
    return WideResNet(depth=28, widen_factor=4, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, activation_function=activation_function, **kwargs)

def WideResNet_28_10(num_classes, factor, dataset, normalized, block=WideBasic, dropout_rate=0.0, activation_function='relu'):
    return WideResNet(depth=28, widen_factor=10, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, activation_function=activation_function)

def WideResNet_28_12(num_classes, factor, dataset, normalized, block=WideBasic, dropout_rate=0.0, activation_function='relu'):
    return WideResNet(depth=28, widen_factor=12, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, activation_function=activation_function)

def WideResNet_40_10(num_classes, factor, dataset, normalized, block=WideBasic, dropout_rate=0.0, activation_function='relu'):
    return WideResNet(depth=40, widen_factor=10, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, activation_function=activation_function)

def WideResNet_70_16(num_classes, factor, dataset, normalized, block=WideBasic, dropout_rate=0.0, activation_function='relu'):
    return WideResNet(depth=70, widen_factor=16, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, activation_function=activation_function)
