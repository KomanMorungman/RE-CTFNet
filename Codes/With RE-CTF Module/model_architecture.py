"""Restoration-enhanced HOGformer and RDMNet-style YOLOX model."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class LearnableHOG(nn.Module):
    def __init__(self, channels: int, bins: int = 9, cell_size: int = 8) -> None:
        super().__init__()
        self.bins = bins
        self.cell_size = cell_size
        self.gradient_x = nn.Conv2d(
            channels // 2, channels // 2, 3, padding=1, groups=channels // 2, bias=False
        )
        self.gradient_y = nn.Conv2d(
            channels // 2, channels // 2, 3, padding=1, groups=channels // 2, bias=False
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        sobel_y = sobel_x.t()
        feature_channels = channels // 2
        with torch.no_grad():
            self.gradient_x.weight.copy_(
                sobel_x.view(1, 1, 3, 3).repeat(feature_channels, 1, 1, 1)
            )
            self.gradient_y.weight.copy_(
                sobel_y.view(1, 1, 3, 3).repeat(feature_channels, 1, 1, 1)
            )
        self.bin_projection = nn.Conv2d(bins, channels // 2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        half_channels = x.shape[1] // 2
        source = x[:, :half_channels]
        gx = self.gradient_x(source).mean(dim=1, keepdim=True).float()
        gy = self.gradient_y(source).mean(dim=1, keepdim=True).float()
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
        # Bin assignment and interpolation weights are treated as constants:
        # the backward of atan2 is 0/0 = NaN wherever gx = gy = 0 (flat or padded
        # regions), which made every gradient non-finite and every optimizer step skipped.
        with torch.no_grad():
            angle = torch.atan2(gy, gx).remainder(math.pi) * (self.bins / math.pi)
            lower = angle.floor().long().remainder(self.bins)
            upper = (lower + 1).remainder(self.bins)
            upper_weight = angle - angle.floor()
        histogram = magnitude.new_zeros((x.shape[0], self.bins, *x.shape[-2:]))
        histogram.scatter_add_(1, lower, magnitude * (1 - upper_weight))
        histogram.scatter_add_(1, upper, magnitude * upper_weight)

        height, width = x.shape[-2:]
        pad_h = (-height) % self.cell_size
        pad_w = (-width) % self.cell_size
        if pad_h or pad_w:
            histogram = F.pad(histogram, (0, pad_w, 0, pad_h))
        histogram = F.avg_pool2d(histogram, self.cell_size, self.cell_size)
        histogram = F.normalize(histogram, p=2, dim=1, eps=1e-6)
        hog_map = self.bin_projection(histogram)
        hog_map = F.interpolate(hog_map, size=(height, width), mode="bilinear", align_corners=False)
        return torch.cat((source + hog_map, x[:, half_channels:]), dim=1)


class Attention_DHOGSA(nn.Module):
    """Dual channel-attention branches conditioned on learnable HOG features."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.hog = LearnableHOG(dim)
        self.qkv = nn.Conv2d(dim, dim * 5, 1, bias=bias)
        self.qkv_depthwise = nn.Conv2d(
            dim * 5, dim * 5, 3, padding=1, groups=dim * 5, bias=bias
        )
        self.projection = nn.Conv2d(dim, dim, 1, bias=bias)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        batch, _, height, width = q.shape
        spatial = height * width
        channels_per_head = self.dim // self.num_heads
        q = q.reshape(batch, self.num_heads, channels_per_head, spatial)
        k = k.reshape(batch, self.num_heads, channels_per_head, spatial)
        v = v.reshape(batch, self.num_heads, channels_per_head, spatial)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature.unsqueeze(0)
        return (attention.softmax(dim=-1) @ v).reshape(batch, self.dim, height, width)

    def forward(self, x: Tensor) -> Tensor:
        x = self.hog(x)
        batch, channels, height, width = x.shape
        qkv = self.qkv_depthwise(self.qkv(x))
        q1, k1, q2, k2, value = qkv.chunk(5, dim=1)

        with torch.no_grad():  # the sort order is not differentiable anyway
            gx = value.detach().float().mean(dim=1, keepdim=True)
            grad_x = F.pad(gx[..., 1:] - gx[..., :-1], (0, 1, 0, 0))
            grad_y = F.pad(gx[..., 1:, :] - gx[..., :-1, :], (0, 0, 0, 1))
            magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
            order = (
                magnitude * torch.atan2(grad_y, grad_x).remainder(math.pi)
            ).flatten(1).argsort(dim=-1)
        order = order.unsqueeze(1).expand(batch, channels, height * width)

        def reorder(feature: Tensor) -> Tensor:
            return torch.gather(feature.flatten(2), 2, order).view_as(feature)

        attended = self._attend(reorder(q1), reorder(k1), reorder(value))
        attended2 = self._attend(reorder(q2), reorder(k2), reorder(value))
        attended = attended * attended2
        attended = torch.zeros_like(attended).flatten(2).scatter(
            2, order, attended.flatten(2)
        ).view(batch, channels, height, width)
        return self.projection(attended)


class FFN_DIFF(nn.Module):
    def __init__(self, dim: int, expansion: float = 8 / 3, bias: bool = False) -> None:
        super().__init__()
        hidden = max(4, math.ceil(dim * expansion / 4) * 4)
        branch_channels = hidden // 4
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.depthwise_5 = nn.Conv2d(
            branch_channels, branch_channels, 5, padding=2, groups=branch_channels, bias=bias
        )
        self.depthwise_dilated = nn.Conv2d(
            branch_channels,
            branch_channels,
            3,
            padding=2,
            dilation=2,
            groups=branch_channels,
            bias=bias,
        )
        self.scale = nn.Parameter(torch.full((1, branch_channels, 1, 1), 1e-5))
        self.decompose = nn.Conv2d(branch_channels, 1, 1)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pixel_shuffle(self.project_in(x), 2)
        first, second = x.chunk(2, dim=1)
        first = self.depthwise_5(first)
        second = self.depthwise_dilated(second)
        x = F.mish(second) * first
        x = x + self.scale * (x - F.gelu(self.decompose(x)))
        return self.project_out(F.pixel_unshuffle(x, 2))


class HOGTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm_attention = LayerNorm2d(dim)
        self.attention = Attention_DHOGSA(dim, heads)
        self.norm_ffn = LayerNorm2d(dim)
        self.ffn = FFN_DIFF(dim)
        self.use_checkpoint = False

    def _forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.ffn(self.norm_ffn(x))

    def forward(self, x: Tensor) -> Tensor:
        # Activation checkpointing: recompute this block in backward to save GPU memory.
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class HOGDownsample(nn.Module):
    def __init__(self, channels: int, factor: int = 2) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError("Restoration downsampling factors must be positive integers.")
        projected_channels, remainder = divmod(channels * 2, factor * factor)
        if remainder:
            raise ValueError(
                f"Feature width {channels} cannot be downsampled by {factor} while doubling channels."
            )
        self.factor = factor
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, projected_channels, 3, padding=1, bias=False),
            nn.PixelUnshuffle(factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(x)


class RestorationEncoder(nn.Module):
    """HOGformer encoder returning features at 1/2 through 1/32 scales."""

    def __init__(
        self,
        dim: int = 36,
        block_depths: Sequence[int] = (4, 6, 6, 6, 8),
        heads: Sequence[int] = (1, 2, 4, 8, 16),
        downsample_factors: Sequence[int] = (2, 2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(block_depths) != 5 or len(heads) != 5:
            raise ValueError("Restoration encoder expects 5 block depths and 5 head counts.")
        if len(downsample_factors) != 5 or any(factor < 1 for factor in downsample_factors):
            raise ValueError("Restoration encoder expects 5 positive downsampling factors.")
        if dim % 2:
            raise ValueError("HOG dimension must be even.")
        self.downsample_factors = tuple(downsample_factors)
        self.patch_embed = nn.Conv2d(3, dim, 3, stride=downsample_factors[0], padding=1)
        self.level1 = nn.Sequential(*(HOGTransformerBlock(dim, heads[0]) for _ in range(block_depths[0])))
        self.down1 = HOGDownsample(dim, downsample_factors[1])
        self.level2 = nn.Sequential(*(HOGTransformerBlock(dim * 2, heads[1]) for _ in range(block_depths[1])))
        self.down2 = HOGDownsample(dim * 2, downsample_factors[2])
        self.level3 = nn.Sequential(*(HOGTransformerBlock(dim * 4, heads[2]) for _ in range(block_depths[2])))
        self.down3 = HOGDownsample(dim * 4, downsample_factors[3])
        self.level4 = nn.Sequential(*(HOGTransformerBlock(dim * 8, heads[3]) for _ in range(block_depths[3])))
        self.down4 = HOGDownsample(dim * 8, downsample_factors[4])
        self.level5 = nn.Sequential(*(HOGTransformerBlock(dim * 16, heads[4]) for _ in range(block_depths[4])))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        half = self.level1(self.patch_embed(image))
        quarter = self.level2(self.down1(half))
        eighth = self.level3(self.down2(quarter))
        sixteenth = self.level4(self.down3(eighth))
        thirtysecond = self.level5(self.down4(sixteenth))
        return half, quarter, eighth, sixteenth, thirtysecond


class ChannelPixelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.pixel_attention = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.channel_attention(x), self.pixel_attention(x)


class RestorationEnhancedCrossTaskFusion(nn.Module):
    def __init__(self, restoration_channels: int, detection_channels: int) -> None:
        super().__init__()
        self.restoration_projection = nn.Sequential(
            nn.Conv2d(restoration_channels, detection_channels, 1),
            LayerNorm2d(detection_channels),  # bounds the scale of restoration features
        )
        self.concat_projection = nn.Conv2d(detection_channels * 2, detection_channels, 1)
        self.depthwise = nn.Conv2d(
            detection_channels,
            detection_channels,
            3,
            padding=1,
            groups=detection_channels,
        )
        self.cpa = ChannelPixelAttention(detection_channels)
        self.lambda_logit = nn.Parameter(torch.tensor(0.0))
        self.beta_logit = nn.Parameter(torch.tensor(-2.0))
        self.gamma = nn.Parameter(torch.tensor(0.1))  # start gentle; was 1.0

    def forward(self, restoration: Tensor, detection: Tensor) -> Tensor:
        restoration = self.restoration_projection(restoration)
        if restoration.shape[-2:] != detection.shape[-2:]:
            restoration = F.interpolate(
                restoration, size=detection.shape[-2:], mode="bilinear", align_corners=False
            )
        fused = self.depthwise(self.concat_projection(torch.cat((restoration, detection), dim=1)))
        channel_weights, pixel_weights = self.cpa(fused)
        blend = torch.sigmoid(self.lambda_logit)
        adaptive_weights = blend * channel_weights + (1 - blend) * pixel_weights
        lower_bound = torch.sigmoid(self.beta_logit)
        gate = lower_bound + (1 - lower_bound) * adaptive_weights
        return detection + self.gamma * (restoration * gate)


class BaseConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel: int, stride: int = 1, groups: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            padding=(kernel - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(x)))


class Focus(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: int = 2) -> None:
        super().__init__()
        if downsample < 1:
            raise ValueError("Detection downsampling factors must be positive integers.")
        self.downsample = downsample
        if downsample == 2:
            self.conv = BaseConv(in_channels * 4, out_channels, 3)
        else:
            self.conv = BaseConv(in_channels, out_channels, 3, downsample)

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample != 2:
            return self.conv(x)
        patches = (
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        )
        return self.conv(torch.cat(patches, dim=1))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True) -> None:
        super().__init__()
        self.conv1 = BaseConv(channels, channels, 1)  # YOLOX expansion = 1.0
        self.conv2 = BaseConv(channels, channels, 3)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        result = self.conv2(self.conv1(x))
        return result + x if self.shortcut else result


class CSPLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, depth: int, shortcut: bool = True
    ) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.branch1 = BaseConv(in_channels, hidden, 1)
        self.branch2 = BaseConv(in_channels, hidden, 1)
        self.blocks = nn.Sequential(*(Bottleneck(hidden, shortcut) for _ in range(depth)))
        self.merge = BaseConv(hidden * 2, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.merge(torch.cat((self.blocks(self.branch1(x)), self.branch2(x)), dim=1))


class SPPBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.conv1 = BaseConv(channels, hidden, 1)
        self.pools = nn.ModuleList(
            nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2) for size in (5, 9, 13)
        )
        self.conv2 = BaseConv(hidden * 4, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return self.conv2(torch.cat((x, *(pool(x) for pool in self.pools)), dim=1))


class SCConv(nn.Module):
    def __init__(self, channels: int, pooling: int = 4) -> None:
        super().__init__()
        self.context_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.pooling = pooling
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1)
        self.output = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(0.2))

    def forward(self, x: Tensor) -> Tensor:
        pool_h = min(self.pooling, x.shape[-2])
        pool_w = min(self.pooling, x.shape[-1])
        context = F.avg_pool2d(x, (pool_h, pool_w), (pool_h, pool_w))
        context = self.context_conv(context)
        context = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.spatial(x) * torch.sigmoid(x + context))


class SCBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        half = channels // 2
        self.path1 = nn.Sequential(
            nn.Conv2d(channels, half, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(half, half, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.path2 = nn.Sequential(nn.Conv2d(channels, half, 1), nn.LeakyReLU(0.2), SCConv(half))
        self.merge = nn.Conv2d(channels, channels, 1)
        self.activation = nn.LeakyReLU(0.2)
        # Start near identity so the pretrained neck path is not perturbed.
        nn.init.zeros_(self.merge.weight)
        nn.init.zeros_(self.merge.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.merge(torch.cat((self.path1(x), self.path2(x)), dim=1)) + x)


class DetectionEncoder(nn.Module):
    """RDMNet Focus/CSPDarknet stages with cross-task injection at 1/4 and 1/16."""

    def __init__(
        self,
        base_channels: int,
        hog_dim: int,
        depth: float = 0.33,
        downsample_factors: Sequence[int] = (2, 2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(downsample_factors) != 5 or any(factor < 1 for factor in downsample_factors):
            raise ValueError("Detection encoder expects 5 positive downsampling factors.")
        self.downsample_factors = tuple(downsample_factors)
        blocks = max(round(depth * 3), 1)
        self.focus = Focus(3, base_channels, downsample_factors[0])
        self.dark2_down = BaseConv(base_channels, base_channels * 2, 3, downsample_factors[1])
        self.dark2 = CSPLayer(base_channels * 2, base_channels * 2, blocks)
        self.fusion_quarter = RestorationEnhancedCrossTaskFusion(hog_dim * 2, base_channels * 2)
        self.dark3_down = BaseConv(base_channels * 2, base_channels * 4, 3, downsample_factors[2])
        self.dark3 = CSPLayer(base_channels * 4, base_channels * 4, blocks * 3)
        self.dark4_down = BaseConv(base_channels * 4, base_channels * 8, 3, downsample_factors[3])
        self.dark4 = CSPLayer(base_channels * 8, base_channels * 8, blocks * 3)
        self.fusion_sixteenth = RestorationEnhancedCrossTaskFusion(hog_dim * 8, base_channels * 8)
        self.dark5_down = BaseConv(base_channels * 8, base_channels * 16, 3, downsample_factors[4])
        self.spp = SPPBottleneck(base_channels * 16)
        self.dark5 = CSPLayer(
            base_channels * 16, base_channels * 16, blocks, shortcut=False
        )

    def forward(
        self, image: Tensor, restoration_quarter: Tensor, restoration_sixteenth: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = self.focus(image)
        x = self.dark2(self.dark2_down(x))
        x = self.fusion_quarter(restoration_quarter, x)
        feature_eighth = self.dark3(self.dark3_down(x))
        x = self.dark4(self.dark4_down(feature_eighth))
        x = self.fusion_sixteenth(restoration_sixteenth, x)
        feature_thirtysecond = self.dark5(self.spp(self.dark5_down(x)))
        return feature_eighth, x, feature_thirtysecond


class YOLOPAFPN(nn.Module):
    def __init__(self, base_channels: int, depth: float = 0.33) -> None:
        super().__init__()
        c3, c4, c5 = base_channels * 4, base_channels * 8, base_channels * 16
        blocks = max(round(depth * 3), 1)
        self.lateral5 = BaseConv(c5, c4, 1)
        self.top4 = CSPLayer(c4 * 2, c4, blocks, shortcut=False)
        self.reduce4 = BaseConv(c4, c3, 1)
        self.top3_sc = SCBottleneck(c3)
        self.top3 = CSPLayer(c3 * 2, c3, blocks, shortcut=False)
        self.down3 = BaseConv(c3, c3, 3, 2)
        self.bottom4_sc = SCBottleneck(c3)
        self.bottom4 = CSPLayer(c3 * 2, c4, blocks, shortcut=False)
        self.down4 = BaseConv(c4, c4, 3, 2)
        self.bottom5_sc = SCBottleneck(c4)
        self.bottom5 = CSPLayer(c4 * 2, c5, blocks, shortcut=False)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        feature3, feature4, feature5 = features
        p5 = self.lateral5(feature5)
        p5_upsampled = F.interpolate(p5, size=feature4.shape[-2:], mode="nearest")
        p4_top = self.top4(torch.cat((p5_upsampled, feature4), dim=1))
        p4_reduced = self.reduce4(p4_top)
        p4_upsampled = F.interpolate(p4_reduced, size=feature3.shape[-2:], mode="nearest")
        p3_top = self.top3(torch.cat((self.top3_sc(p4_upsampled), feature3), dim=1))
        p3_downsampled = F.interpolate(self.down3(p3_top), size=p4_reduced.shape[-2:], mode="nearest")
        p4_out = self.bottom4(
            torch.cat((self.bottom4_sc(p3_downsampled), p4_reduced), dim=1)
        )
        p4_downsampled = F.interpolate(self.down4(p4_out), size=p5.shape[-2:], mode="nearest")
        p5_out = self.bottom5(
            torch.cat((self.bottom5_sc(p4_downsampled), p5), dim=1)
        )
        return p3_top, p4_out, p5_out


class YOLOXHead(nn.Module):
    def __init__(self, num_classes: int, base_channels: int) -> None:
        super().__init__()
        channels = (base_channels * 4, base_channels * 8, base_channels * 16)
        hidden = base_channels * 4
        self.stems = nn.ModuleList(BaseConv(c, hidden, 1) for c in channels)
        self.classification = nn.ModuleList(
            nn.Sequential(BaseConv(hidden, hidden, 3), BaseConv(hidden, hidden, 3))
            for _ in channels
        )
        self.regression = nn.ModuleList(
            nn.Sequential(BaseConv(hidden, hidden, 3), BaseConv(hidden, hidden, 3))
            for _ in channels
        )
        self.classification_output = nn.ModuleList(nn.Conv2d(hidden, num_classes, 1) for _ in channels)
        self.box_output = nn.ModuleList(nn.Conv2d(hidden, 4, 1) for _ in channels)
        self.objectness_output = nn.ModuleList(nn.Conv2d(hidden, 1, 1) for _ in channels)
        # YOLOX prior-probability bias init (p = 0.01) for class/objectness logits.
        prior_bias = -math.log((1 - 0.01) / 0.01)
        for conv in (*self.classification_output, *self.objectness_output):
            nn.init.constant_(conv.bias, prior_bias)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> list[Tensor]:
        outputs: list[Tensor] = []
        for index, feature in enumerate(features):
            shared = self.stems[index](feature)
            cls_feature = self.classification[index](shared)
            reg_feature = self.regression[index](shared)
            outputs.append(
                torch.cat(
                    (
                        self.box_output[index](reg_feature),
                        self.objectness_output[index](reg_feature),
                        self.classification_output[index](cls_feature),
                    ),
                    dim=1,
                )
            )
        return outputs


class RestorationDecoder(nn.Module):
    def __init__(self, dim: int = 36) -> None:
        super().__init__()
        self.reduce32 = nn.Conv2d(dim * 16, dim * 8, 1)
        self.fuse16 = nn.Sequential(
            nn.Conv2d(dim * 16, dim * 8, 1), nn.Conv2d(dim * 8, dim * 8, 3, padding=1), nn.GELU()
        )
        self.reduce16 = nn.Conv2d(dim * 8, dim * 4, 1)
        self.fuse8 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 4, 1), nn.Conv2d(dim * 4, dim * 4, 3, padding=1), nn.GELU()
        )
        self.reduce8 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.fuse4 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, 1), nn.Conv2d(dim * 2, dim * 2, 3, padding=1), nn.GELU()
        )
        self.up2 = nn.Sequential(nn.Conv2d(dim * 2, dim, 3, padding=1), nn.GELU())
        self.fuse2 = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Conv2d(dim, dim, 3, padding=1), nn.GELU())
        self.full_skip = nn.Conv2d(3, dim // 2, 3, padding=1)
        self.up1 = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1), nn.GELU())
        self.output = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(
        self,
        image: Tensor,
        half: Tensor,
        quarter: Tensor,
        eighth: Tensor,
        sixteenth: Tensor,
        thirtysecond: Tensor,
    ) -> Tensor:
        x = F.interpolate(self.reduce32(thirtysecond), size=sixteenth.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse16(torch.cat((x, sixteenth), dim=1))
        x = F.interpolate(self.reduce16(x), size=eighth.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse8(torch.cat((x, eighth), dim=1))
        x = F.interpolate(self.reduce8(x), size=quarter.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse4(torch.cat((x, quarter), dim=1))
        x = F.interpolate(self.up2(x), size=half.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat((x, half), dim=1))
        x = F.interpolate(self.up1(x), size=image.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, self.full_skip(image)), dim=1)
        return image + self.output(x)


class REHOGModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 32,
        hog_dim: int = 36,
        depth: float = 0.33,
        block_depths: Sequence[int] = (4, 6, 6, 6, 8),
        heads: Sequence[int] = (1, 2, 4, 8, 16),
        restoration_downsample_factors: Sequence[int] = (2, 2, 2, 2, 2),
        detection_downsample_factors: Sequence[int] = (2, 2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.hog_dim = hog_dim
        self.depth = depth
        self.block_depths = tuple(block_depths)
        self.heads = tuple(heads)
        self.restoration_downsample_factors = tuple(restoration_downsample_factors)
        self.detection_downsample_factors = tuple(detection_downsample_factors)
        self.detection_strides = tuple(
            math.prod(self.detection_downsample_factors[:length])
            for length in (3, 4, 5)
        )
        self.restoration_encoder = RestorationEncoder(
            hog_dim, block_depths, heads, restoration_downsample_factors
        )
        self.detection_encoder = DetectionEncoder(
            base_channels, hog_dim, depth, detection_downsample_factors
        )
        self.detection_neck = YOLOPAFPN(base_channels, depth)
        self.detection_head = YOLOXHead(num_classes, base_channels)
        self.restoration_decoder = RestorationDecoder(hog_dim)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        for module in self.modules():
            if isinstance(module, HOGTransformerBlock):
                module.use_checkpoint = enabled

    def forward(self, image: Tensor) -> dict[str, Tensor | list[Tensor]]:
        required_divisor = math.lcm(
            math.prod(self.restoration_downsample_factors),
            math.prod(self.detection_downsample_factors),
        )
        if image.shape[-2] % required_divisor or image.shape[-1] % required_divisor:
            raise ValueError(
                f"Input height and width must both be divisible by {required_divisor} "
                "for the configured encoder downsampling factors."
            )
        half, quarter, eighth, sixteenth, thirtysecond = self.restoration_encoder(image)
        encoded = self.detection_encoder(image, quarter, sixteenth)
        detections = self.detection_head(self.detection_neck(encoded))
        restored = self.restoration_decoder(image, half, quarter, eighth, sixteenth, thirtysecond)
        return {"detections": detections, "restored": restored}


def decode_predictions(
    outputs: Sequence[Tensor],
    num_classes: int,
    strides: Sequence[int] = (8, 16, 32),
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
    max_candidates: int = 1000,
) -> list[Tensor]:
    """Decode YOLOX outputs and apply class-aware greedy non-maximum suppression."""
    batch_size = outputs[0].shape[0]
    decoded: list[Tensor] = []
    for output, stride in zip(outputs, strides):
        _, _, height, width = output.shape
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=output.device, dtype=output.dtype),
            torch.arange(width, device=output.device, dtype=output.dtype),
            indexing="ij",
        )
        grid = torch.stack((x_grid, y_grid), dim=-1).view(1, -1, 2)
        prediction = output.flatten(2).transpose(1, 2).clone()
        prediction[..., :2] = (prediction[..., :2] + grid) * stride
        prediction[..., 2:4] = prediction[..., 2:4].clamp(max=10).exp() * stride
        decoded.append(prediction)

    results: list[Tensor] = []
    all_predictions = torch.cat(decoded, dim=1)
    for batch_index in range(batch_size):
        prediction = all_predictions[batch_index]
        class_probabilities = prediction[:, 5 : 5 + num_classes].sigmoid()
        scores, classes = class_probabilities.max(dim=1)
        scores = scores * prediction[:, 4].sigmoid()
        selected = scores >= confidence_threshold
        if not selected.any():
            results.append(prediction.new_zeros((0, 6)))
            continue
        candidate_indices = torch.where(selected)[0]
        if candidate_indices.numel() > max_candidates:
            candidate_indices = candidate_indices[
                scores[candidate_indices].topk(max_candidates).indices
            ]
        boxes = prediction[candidate_indices, :4]
        scores = scores[candidate_indices]
        classes = classes[candidate_indices]
        boxes_xyxy = torch.cat(
            (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1
        )
        keep: list[Tensor] = []
        for class_id in classes.unique():
            class_indices = torch.where(classes == class_id)[0]
            order = scores[class_indices].argsort(descending=True)
            while order.numel():
                chosen = order[0]
                keep.append(class_indices[chosen])
                if order.numel() == 1:
                    break
                remaining = order[1:]
                chosen_box = boxes_xyxy[class_indices[chosen]]
                other_boxes = boxes_xyxy[class_indices[remaining]]
                top_left = torch.maximum(chosen_box[:2], other_boxes[:, :2])
                bottom_right = torch.minimum(chosen_box[2:], other_boxes[:, 2:])
                intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
                area_chosen = (chosen_box[2:] - chosen_box[:2]).clamp(min=0).prod()
                area_other = (other_boxes[:, 2:] - other_boxes[:, :2]).clamp(min=0).prod(dim=1)
                overlap = intersection / (area_chosen + area_other - intersection + 1e-7)
                order = remaining[overlap <= iou_threshold]
        keep_tensor = torch.stack(keep) if keep else classes.new_empty((0,))
        if keep_tensor.numel():
            keep_tensor = keep_tensor[scores[keep_tensor].argsort(descending=True)[:max_detections]]
            results.append(
                torch.cat(
                    (
                        boxes_xyxy[keep_tensor],
                        scores[keep_tensor, None],
                        classes[keep_tensor, None].to(boxes_xyxy.dtype),
                    ),
                    dim=1,
                )
            )
        else:
            results.append(prediction.new_zeros((0, 6)))
    return results
