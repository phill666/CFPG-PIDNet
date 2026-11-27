import torch.nn as nn
import torch.nn.functional as F

from .ddrnet_core import BasicBlock, DualResNet


class DDRNet23Slim(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, augment=False):
        super().__init__()
        self.augment = bool(augment)
        self.backbone = DualResNet(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=int(num_classes),
            planes=32,
            spp_planes=128,
            head_planes=64,
            augment=self.augment,
            input_channels=int(input_channels),
        )

    def _resize_output(self, output, size):
        if output.shape[-2:] == size:
            return output
        return F.interpolate(output, size=size, mode="bilinear", align_corners=True)

    def forward(self, x):
        size = x.shape[-2:]
        outputs = self.backbone(x)
        if isinstance(outputs, (list, tuple)):
            return [self._resize_output(output, size) for output in outputs]
        return self._resize_output(outputs, size)


def build_ddrnet23_slim(num_classes=2, augment=False):
    return DDRNet23Slim(num_classes=int(num_classes), input_channels=3, augment=bool(augment))
