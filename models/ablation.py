"""Controlled DFM-output mechanisms sharing the final CFPG initialization."""
import torch
from torch import nn
from .cfpg import CFPG

INTERNAL_VARIANTS = ['ctrl_conv', 'cfpg_periph_only', 'cfpg_plus_csa', 'cfpg_plus_entropy_no_grn']


class CoarseOnlyCFPG(CFPG):
    def __init__(self, dim, center_suppress):
        super().__init__(dim, k_list=(7, 15, 31), center_suppress=center_suppress, use_grn=False)
        del self.refine_gate, self.confidence_head

    def forward(self, x):
        freq_maps = self._build_freq_maps(x)
        alpha = torch.softmax(self.coarse_gate(freq_maps).float(), dim=1)
        y = sum(branch(x) * alpha[:, i:i+1] for i, branch in enumerate(self.branches))
        y = self.out_proj(self.channel_reweight(y.to(x.dtype)))
        return y + x


class ConvControl(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(.1))
        self.block = nn.Sequential(
            nn.Conv2d(dim, 44, 3, padding=1, bias=False),
            nn.BatchNorm2d(44, momentum=.1), nn.GELU(),
            nn.Conv2d(44, dim, 1, bias=True))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return x + self.gamma.to(x.dtype) * self.block(x)


def build_final_model(cfg, augment=False):
    from .pidnet import PIDNet
    variant = cfg.get('experiment', {}).get('variant', cfg.get('variant', 'cfpg'))
    if variant not in ['baseline','cfpg',*INTERNAL_VARIANTS]:
        raise ValueError(variant)
    model = PIDNet(planes=32,ppm_planes=96,head_planes=128,num_classes=2,
                   augment=augment,use_cfpg=variant!='baseline')
    if variant not in INTERNAL_VARIANTS:
        return model
    with torch.random.fork_rng(devices=[]):
        if variant=='ctrl_conv':
            model.cfpg=ConvControl(128)
        else:
            replacement=(CFPG(128,center_suppress=True,use_grn=False)
                if variant=='cfpg_plus_entropy_no_grn' else
                CoarseOnlyCFPG(128,center_suppress=variant=='cfpg_plus_csa'))
            state=model.cfpg.cfpg.state_dict()
            replacement.load_state_dict({k:state[k] for k in replacement.state_dict()},strict=True)
            model.cfpg.cfpg=replacement
    return model
