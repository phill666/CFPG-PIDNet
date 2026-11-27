"""Exact STDC main-logit path without unused detail, auxiliary or classifier heads."""
import torch.nn as nn
import torch.nn.functional as F
import torch


class StageBottleneck(nn.Module):
    """Avoid constructing and registering a sliced ModuleList on every forward."""
    def __init__(self, block):
        super().__init__()
        self.conv_list = block.conv_list
        self.stride = block.stride
        self.avd_layer = block.avd_layer if block.stride==2 else None
        self.skip = block.skip if block.stride==2 else None

    def forward(self,x):
        first = self.conv_list[0](x)
        outputs = []
        for index in range(1,len(self.conv_list)):
            if index==1:
                value = self.conv_list[index](self.avd_layer(first) if self.stride==2 else first)
            else:
                value = self.conv_list[index](value)
            outputs.append(value)
        outputs.insert(0,self.skip(first) if self.stride==2 else first)
        return torch.cat(outputs,dim=1)


def remove_dynamic_slices(module):
    for name,child in list(module.named_children()):
        if type(child).__name__=='CatBottleneck':
            setattr(module,name,StageBottleneck(child))
        else:
            remove_dynamic_slices(child)


class SegmentationStages(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        for name in ['x2','x4','x8','x16','x32']:
            setattr(self,name,getattr(backbone,name))
        remove_dynamic_slices(self)
        self.conv_last = backbone.conv_last if backbone.use_conv_last else None

    def forward(self,x):
        x2 = self.x2(x)
        x4 = self.x4(x2)
        x8 = self.x8(x4)
        x16 = self.x16(x8)
        x32 = self.x32(x16)
        if self.conv_last is not None: x32 = self.conv_last(x32)
        return x2,x4,x8,x16,x32


class STDCMainInference(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.cp = wrapper.model.cp
        self.cp.backbone = SegmentationStages(self.cp.backbone)
        self.ffm = wrapper.model.ffm
        self.conv_out = wrapper.model.conv_out

    def forward(self,x):
        features = self.cp(x)
        fused = self.ffm(features[2],features[4])
        return F.interpolate(self.conv_out(fused),size=x.shape[-2:],mode='bilinear',align_corners=True)
