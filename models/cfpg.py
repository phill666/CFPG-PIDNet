"""CFPG-PIDNet multiscale peripheral-center gating and entropy feedback."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRN(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x.float(), p=2, dim=(2, 3), keepdim=True)
        nx = x / (gx + self.eps)
        return (x + self.gamma * nx + self.beta).to(x.dtype)

class ChannelReweight(nn.Module):

    def __init__(self, dim: int, hidden_ratio: float=0.25):
        super().__init__()
        hidden_dim = max(8, int(dim * hidden_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True), nn.GELU(), nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mlp(self.pool(x))
        return x * w

class CFPG(nn.Module):

    class Branch(nn.Module):

        def __init__(self, dim: int, k: int, center_suppress: bool=True):
            super().__init__()
            self.k = k
            self.center_suppress = center_suppress
            self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, k), padding=(0, k // 2), groups=dim, bias=False)
            self.dw_v = nn.Conv2d(dim, dim, kernel_size=(k, 1), padding=(k // 2, 0), groups=dim, bias=False)
            if self.center_suppress:
                self.dw_c = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
                self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
            else:
                self.dw_c = None
                self.register_parameter('beta', None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.dw_v(self.dw_h(x))
            if self.center_suppress:
                center = self.dw_c(x)
                y = y - torch.tanh(self.beta) * center
            return y

    def __init__(self, dim: int, k_list=(7, 15, 31), center_suppress: bool=True, use_grn: bool=True, use_residual: bool=True):
        super().__init__()
        self.dim = dim
        self.k_list = k_list
        self.num_scales = len(k_list)
        self.use_grn = use_grn
        self.use_residual = use_residual
        self.branches = nn.ModuleList([CFPG.Branch(dim, k, center_suppress=center_suppress) for k in k_list])
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.repeat(dim, 1, 1, 1), persistent=False)
        self.register_buffer('sobel_y', sobel_y.repeat(dim, 1, 1, 1), persistent=False)
        self.register_buffer('laplace', laplace.repeat(dim, 1, 1, 1), persistent=False)
        self.coarse_gate = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1, bias=True), nn.GELU(), nn.Conv2d(16, self.num_scales, kernel_size=1, bias=True))
        self.refine_gate = nn.Sequential(nn.Conv2d(3 + self.num_scales, 32, kernel_size=1, bias=True), nn.GELU(), nn.Conv2d(32, self.num_scales, kernel_size=1, bias=True))
        self.confidence_head = nn.Sequential(nn.Conv2d(1, 8, kernel_size=1, bias=True), nn.GELU(), nn.Conv2d(8, 1, kernel_size=1, bias=True), nn.Sigmoid())
        self.channel_reweight = ChannelReweight(dim)
        self.out_proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=True), nn.GELU(), nn.Conv2d(dim, dim, kernel_size=1, bias=True))
        if self.use_grn:
            self.grn = GRN(dim)

    def _depthwise_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        (b, c, h, w) = x.shape
        weight = kernel.to(device=x.device, dtype=x.dtype)
        return F.conv2d(x, weight, padding=1, groups=self.dim)

    def _build_freq_maps(self, x: torch.Tensor) -> torch.Tensor:
        gx = self._depthwise_filter(x, self.sobel_x)
        gy = self._depthwise_filter(x, self.sobel_y)
        lap = self._depthwise_filter(x, self.laplace)
        grad_mag = torch.sqrt(gx.float().pow(2) + gy.float().pow(2) + 1e-06)
        mean = F.avg_pool2d(x.float(), kernel_size=3, stride=1, padding=1)
        mean2 = F.avg_pool2d(x.float() * x.float(), kernel_size=3, stride=1, padding=1)
        var = torch.clamp(mean2 - mean * mean, min=0.0)
        f1 = grad_mag.mean(dim=1, keepdim=True)
        f2 = lap.abs().mean(dim=1, keepdim=True)
        f3 = var.mean(dim=1, keepdim=True)
        return torch.cat([f1, f2, f3], dim=1).to(x.dtype)

    def _branch_response_maps(self, branch_outputs):
        energy_maps = []
        for y in branch_outputs:
            e = y.abs().mean(dim=1, keepdim=True)
            energy_maps.append(e)
        return torch.cat(energy_maps, dim=1)

    def _entropy_map(self, prob: torch.Tensor) -> torch.Tensor:
        prob = prob.float()
        entropy = -(prob * torch.log(prob.clamp_min(1e-08))).sum(dim=1, keepdim=True)
        entropy = entropy / max(math.log(self.num_scales), 1e-06)
        return entropy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_maps = self._build_freq_maps(x)
        coarse_logits = self.coarse_gate(freq_maps)
        coarse_alpha = torch.softmax(coarse_logits.float(), dim=1)
        branch_outputs = [branch(x) for branch in self.branches]
        branch_energy = self._branch_response_maps(branch_outputs)
        refine_input = torch.cat([freq_maps, branch_energy], dim=1)
        refine_logits = self.refine_gate(refine_input)
        refine_alpha = torch.softmax(refine_logits.float(), dim=1)
        entropy_map = self._entropy_map(coarse_alpha)
        refine_ratio = self.confidence_head(entropy_map.to(x.dtype))
        final_alpha = (1.0 - refine_ratio) * coarse_alpha + refine_ratio * refine_alpha
        final_alpha = final_alpha / (final_alpha.sum(dim=1, keepdim=True) + 1e-08)
        y = 0.0
        for (i, branch_y) in enumerate(branch_outputs):
            y = y + branch_y * final_alpha[:, i:i + 1, :, :]
        y = self.channel_reweight(y.to(x.dtype))
        y = self.out_proj(y)
        if self.use_grn:
            y = self.grn(y)
        if self.use_residual:
            y = y + x
        return y

class IdentityCFPG(nn.Module):
    """Preserve the DFM feature with an initial correction scale of 0.1."""
    def __init__(self, dim, gamma=0.1):
        super().__init__()
        self.cfpg = CFPG(dim, k_list=(7, 15, 31), center_suppress=True, use_grn=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, x):
        return x + self.gamma.to(x.dtype) * (self.cfpg(x) - x)
