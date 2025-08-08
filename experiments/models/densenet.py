'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.models import ct_model


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(ct_model.CtModel):

    def __init__(self, block, nblocks, growth_rate, dataset, normalized, reduction=0.5, num_classes=10, factor=1):
        super(DenseNet, self).__init__(dataset=dataset, normalized=normalized, num_classes=num_classes)
        self.growth_rate = growth_rate
        self.factor = factor

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False, stride=factor)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.blocks = [self.conv1,
                       nn.Sequential(self.dense1, self.trans1),
                       nn.Sequential(self.dense2, self.trans2),
                       nn.Sequential(self.dense3, self.trans3)]

        self.bn = nn.BatchNorm2d(num_planes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, targets=None, robust_samples=0, corruptions=None, mixup_alpha=0.0, mixup_p=0.0, manifold=False,
                manifold_noise_factor=1, cutmix_alpha=0.0, cutmix_p=0.0, noise_minibatchsize=1,
                concurrent_combinations=1, noise_sparsity=0.0, noise_patch_lower_scale=0.3, noise_patch_upper_scale=1.0,
                generated_ratio=0.0, n2n_deepaugment=False, style_feats=None, **kwargs):
        
        x = super(DenseNet, self).forward_handle_greyscale(x)
        out = super(DenseNet, self).forward_normalize(x)
        out, mixed_targets = super(DenseNet, self).forward_noise_mixup(out, targets, robust_samples, corruptions,
                                        mixup_alpha, mixup_p, manifold, manifold_noise_factor, cutmix_alpha, cutmix_p,
                                        noise_minibatchsize, concurrent_combinations, noise_sparsity,
                                        noise_patch_lower_scale, noise_patch_upper_scale, generated_ratio, n2n_deepaugment, 
                                        style_feats, **kwargs)
        out = self.dense4(out)
        out = self.avgpool(F.relu(self.bn(out)))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.training == True:
            return out, mixed_targets
        else:
            return out

def DenseNet121_32(dataset, normalized, num_classes, factor):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, dataset=dataset, normalized=normalized,
                    num_classes=num_classes, factor=factor)

def DenseNet169_32(dataset, normalized, num_classes, factor):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, dataset=dataset, normalized=normalized,
                    num_classes=num_classes, factor=factor)

def DenseNet201_32(dataset, normalized, num_classes, factor):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, dataset=dataset, normalized=normalized,
                    num_classes=num_classes, factor=factor)

def DenseNet161_48(dataset, normalized, num_classes, factor):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, dataset=dataset, normalized=normalized,
                    num_classes=num_classes, factor=factor)

def DenseNet121_12(dataset, normalized, num_classes, factor):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, dataset=dataset, normalized=normalized,
                    num_classes=num_classes, factor=factor)

def DenseNet201_12(dataset, normalized, num_classes, factor):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=12, dataset=dataset, normalized=normalized,
                    num_classes=num_classes, factor=factor)
