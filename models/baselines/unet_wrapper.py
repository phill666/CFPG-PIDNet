import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


class UNetWrapper(nn.Module):
    def __init__(self, n_channels=3, num_classes=2, bilinear=True):
        super().__init__()
        self.model = UNet(n_channels=int(n_channels), n_classes=int(num_classes), bilinear=bool(bilinear))
        self.n_channels = int(n_channels)
        self.num_classes = int(num_classes)
        self.bilinear = bool(bilinear)

    def forward(self, x):
        input_size = x.shape[-2:]
        logits = self.model(x)
        if isinstance(logits, dict):
            logits = logits.get("out", next(iter(logits.values())))
        elif isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=True)
        return logits


def build_unet(n_channels=3, num_classes=2, bilinear=True):
    return UNetWrapper(n_channels=n_channels, num_classes=num_classes, bilinear=bilinear)
