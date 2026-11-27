import torch.nn as nn
import torch.nn.functional as F

class SafeInPlaceABNSync(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, activation='leaky_relu', slope=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.activation = activation
        self.slope = slope

    def forward(self, x):
        x = self.bn(x)
        if self.activation == 'relu':
            return F.relu(x, inplace=True)
        if self.activation == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=self.slope, inplace=True)
        if self.activation == 'elu':
            return F.elu(x, inplace=True)
        return x
