# Native PIDNet by Jiacong Xu; final CFPG integration by Tianyu Liu et al.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
from .cfpg import IdentityCFPG
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

class PIDNet(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True, input_channels=3, use_cfpg: bool=False):
        super(PIDNet, self).__init__()
        self.use_cfpg = use_cfpg
        self.augment = augment
        self.last_debug_shapes = {}
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, planes, kernel_size=3, stride=2, padding=1), BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1), BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.tgfe = None
        self.wtra = None
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        self.compression3 = nn.Sequential(nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom))
        self.compression4 = nn.Sequential(nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom))
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        self.gar3 = nn.Identity()
        self.gar4 = nn.Identity()
        self.gshalf3 = nn.Identity()
        self.gshalf4 = nn.Identity()
        self.dac3 = nn.Identity()
        self.dac4 = nn.Identity()
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False), BatchNorm2d(planes, momentum=bn_mom))
            self.diff4 = nn.Sequential(nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom))
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom))
            self.diff4 = nn.Sequential(nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom))
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        self.final_d_gate = None
        self.final_dwgca = None
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)
        self.cfpg = IdentityCFPG(planes * 4) if use_cfpg else nn.Identity()
        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.tgfe is not None:
            self.tgfe.reset_gate()
        if self.wtra is not None:
            self.wtra.reset_gate()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(inplanes, planes, stride=1, no_relu=True, **kwargs))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False, **kwargs))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1, **kwargs):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom))
        layer = block(inplanes, planes, stride, downsample, no_relu=True, **kwargs)
        return layer

    def forward(self, x):
        debug_shapes = {}
        input_image = x
        x = self.conv1(x)
        x = self.layer1(x)
        if self.tgfe is not None:
            thermal_gate = self.tgfe(input_image, output_size=x.shape[-2:])
            x = x * (1.0 + thermal_gate)
            debug_shapes['tgfe_gate'] = tuple(thermal_gate.shape)
        if self.wtra is not None:
            x = self.wtra(x, input_image)
            debug_shapes['wtra_output'] = tuple(x.shape)
        x = self.relu(self.layer2(self.relu(x)))
        x_p3 = self.layer3_(x)
        x_ = self.gar3(x_p3)
        x_d = self.layer3_d(x)
        branch_size = x_d.shape[-2:]
        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        if False and hasattr(self.pag3, 'last_debug_shapes'):
            for (key, value) in self.pag3.last_debug_shapes.items():
                debug_shapes['hfigate3_{}'.format(key)] = value
        if False and hasattr(self.pag3, 'last_debug_shapes'):
            for (key, value) in self.pag3.last_debug_shapes.items():
                debug_shapes['csgrm3_{}'.format(key)] = value
        x_d = x_d + F.interpolate(self.diff3(x), size=branch_size, mode='bilinear', align_corners=algc)
        if self.augment and self.training:
            temp_p = x_
        x = self.relu(self.layer4(x))
        x_p4 = self.layer4_(self.relu(x_))
        x_ = self.gar4(x_p4)
        x_d = self.layer4_d(self.relu(x_d))
        branch_size = x_d.shape[-2:]
        x_ = self.pag4(x_, self.compression4(x))
        if False and hasattr(self.pag4, 'last_debug_shapes'):
            for (key, value) in self.pag4.last_debug_shapes.items():
                debug_shapes['hfigate4_{}'.format(key)] = value
        if False and hasattr(self.pag4, 'last_debug_shapes'):
            for (key, value) in self.pag4.last_debug_shapes.items():
                debug_shapes['csgrm4_{}'.format(key)] = value
        x_d = x_d + F.interpolate(self.diff4(x), size=branch_size, mode='bilinear', align_corners=algc)
        if self.augment and self.training:
            temp_d = x_d
        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        branch_size = x_d.shape[-2:]
        context = self.spp(self.layer5(x))
        debug_shapes['native_spp'] = tuple(context.shape)
        x = F.interpolate(context, size=branch_size, mode='bilinear', align_corners=algc)
        debug_shapes['pre_final_p'] = tuple(x_.shape)
        debug_shapes['pre_final_i'] = tuple(x.shape)
        debug_shapes['pre_final_d'] = tuple(x_d.shape)
        if self.final_d_gate is not None:
            x_d = self.final_d_gate(x_, x, x_d)
            debug_shapes['final_d_hfigate_output'] = tuple(x_d.shape)
            if self.final_d_gate.last_gate_shape is not None:
                debug_shapes['final_d_gate_map'] = self.final_d_gate.last_gate_shape
        if self.final_dwgca is not None:
            debug_shapes['dwgca_input'] = tuple(x_d.shape)
            x_d = self.final_dwgca(x_d)
            debug_shapes['dwgca_output'] = tuple(x_d.shape)
            if getattr(self.final_dwgca, 'last_debug_shapes', None):
                for (key, value) in self.final_dwgca.last_debug_shapes.items():
                    debug_shapes['dwgca_{}'.format(key)] = value
        self.last_debug_shapes = debug_shapes
        x_ = self.final_layer(self.cfpg(self.dfm(x_, x, x_d)))
        if self.augment and self.training:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_
