import torch
import torch.nn as nn
import torch.nn.functional as F

from abstract import *


class ResNeXtBottleneck(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor) -> None:
        super(ResNeXtBottleneck, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        width_ratio = out_channels / (widen_factor*64.)
        self.D = cardinality*int(base_width*width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, self.D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(self.D)
        
        self.conv_split = nn.Conv2d(self.D, self.D, kernel_size=3, stride=stride, padding=1, group=cardinality, bias=False)
        self.bn_split = nn.BatchNorm2d(self.D)
        
        self.conv_expand = nn.Conv2d(self.D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        
        self.shortcut = self.short_cut(stride)

    def short_cut(self, stride):

        shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        
        return shortcut

    def forward(self, input):
        bottle_neck = self.conv_reduce(input)
        bottle_neck = self.bn_reduce.forward(bottle_neck)
        bottle_neck = F.relu(bottle_neck, inplace=True)
        
        bottle_neck = self.conv_split(bottle_neck)
        bottle_neck = self.bn_split(bottle_neck)
        bottle_neck = F.relu(bottle_neck, inplace=True)
        
        bottle_neck = self.conv_expand(bottle_neck)
        bottle_neck = self.bn_expand(bottle_neck)
        
        residual = self.shortcut(input)
        
        return F.relu(residual + bottle_neck, inplace=True)
    
    
class ResNeXtBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality,
                 base_width,
                 widen_factor,
                 block_depth) -> None:
        super(ResNeXtBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.block_depth = block_depth
        
        block_list = []
        
        for bottle_neck in range(block_depth):
            if bottle_neck == 0:
                block_list.append(ResNeXtBottleneck(in_channels, out_channels, stride, cardinality, base_width, widen_factor))
            else:
                block_list.append(ResNeXtBottleneck(in_channels, out_channels, 1, cardinality, base_width, widen_factor))
        
        self.block = nn.Sequential(*block_list)

    def forward(self, input):
        return self.block(input)

    
class ResNeXt(nn.Module):
    
    def __init__(self,
                 cardinality,
                 depth,
                 nlabels,
                 base_width,
                 widen_factor=4,
                 pretrained=False) -> None:
        super(ResNeXt, self).__init__()
        
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        
        self.conv_stem = nn.Conv2d(3, 64, 7, 1, 1, bias=False)
        self.bn_stem = nn.BatchNorm2d(64)
        
        self.stage_1 = ResNeXtBlock(self.stages[0], self.stages[1], 1, self.cardinality, self.base_width, self.widen_factor, self.block_depth)
        self.stage_2 = ResNeXtBlock(self.stages[1], self.stages[2], 2, self.cardinality, self.base_width, self.widen_factor, self.block_depth)
        self.stage_3 = ResNeXtBlock(self.stages[2], self.stages[3], 2, self.cardinality, self.base_width, self.widen_factor, self.block_depth)
        
        self.classifier = nn.Linear(self.stages[3], self.nlabels)
        
        if pretrained:
            self.load()
        else:
            self._init_weight()
    
    def forward(self, input):
        x = self.conv_stem(input)
        x = self.bn_stem(x)
        x = F.relu(x, inplace=True)
        
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def remove_cl(state_dict):
        for key in list(state_dict.keys()):
            if key.startswith('fc.'):
                del state_dict[key]
        return state_dict


def build_resnext(cfg):
    model = ResNeXt()
    return model