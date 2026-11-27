import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, height, width):
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=32, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=1, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.sr_ratio = int(sr_ratio)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(batch, channels, height, width)
            x_ = self.sr(x_).reshape(batch, channels, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x
        kv = self.kv(x_).reshape(batch, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, sr_ratio=1, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio, attn_drop=drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.ffn(self.norm2(x), height, width))
        return x


class MiTB0(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.patch_embed1 = OverlapPatchEmbed(in_channels, embed_dims[0], kernel_size=7, stride=4, padding=3)
        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1)
        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1)
        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1)

        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()
        cur = 0
        self.blocks1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0], drop_rate, dpr[cur + i])
            for i in range(depths[0])
        ])
        cur += depths[0]
        self.blocks2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1], drop_rate, dpr[cur + i])
            for i in range(depths[1])
        ])
        cur += depths[1]
        self.blocks3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2], drop_rate, dpr[cur + i])
            for i in range(depths[2])
        ])
        cur += depths[2]
        self.blocks4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3], drop_rate, dpr[cur + i])
            for i in range(depths[3])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward_stage(self, x, patch_embed, blocks, norm):
        x, height, width = patch_embed(x)
        for block in blocks:
            x = block(x, height, width)
        x = norm(x)
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        return x

    def forward(self, x):
        c1 = self._forward_stage(x, self.patch_embed1, self.blocks1, self.norm1)
        c2 = self._forward_stage(c1, self.patch_embed2, self.blocks2, self.norm2)
        c3 = self._forward_stage(c2, self.patch_embed3, self.blocks3, self.norm3)
        c4 = self._forward_stage(c3, self.patch_embed4, self.blocks4, self.norm4)
        return [c1, c2, c3, c4]


class SegFormerMLPDecoder(nn.Module):
    def __init__(self, in_channels=(32, 64, 160, 256), decoder_dim=256, num_classes=2, drop_rate=0.0):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(ch, decoder_dim) for ch in in_channels])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(drop_rate)
        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        target_size = c1.shape[-2:]
        outputs = []
        for feature, proj in zip(features, self.proj):
            batch, channels, height, width = feature.shape
            x = feature.flatten(2).transpose(1, 2)
            x = proj(x)
            x = x.transpose(1, 2).reshape(batch, -1, height, width)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            outputs.append(x)
        x = self.linear_fuse(torch.cat(outputs, dim=1))
        x = self.dropout(x)
        return self.linear_pred(x)


class SegFormerB0(nn.Module):
    def __init__(
        self,
        num_classes=2,
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        decoder_dim=256,
        drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.encoder = MiTB0(
            in_channels=3,
            embed_dims=tuple(embed_dims),
            num_heads=tuple(num_heads),
            mlp_ratios=(4, 4, 4, 4),
            depths=tuple(depths),
            sr_ratios=tuple(sr_ratios),
            drop_rate=float(drop_rate),
            drop_path_rate=float(drop_path_rate),
        )
        self.decode_head = SegFormerMLPDecoder(
            in_channels=tuple(embed_dims),
            decoder_dim=int(decoder_dim),
            num_classes=int(num_classes),
            drop_rate=float(drop_rate),
        )
        self.num_classes = int(num_classes)

    def forward(self, x):
        input_size = x.shape[-2:]
        logits = self.decode_head(self.encoder(x))
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def build_segformer_b0(**kwargs):
    return SegFormerB0(**kwargs)
