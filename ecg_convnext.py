import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class Block(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding="same", groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x, mask=None):
        y = apply_mask(x, mask)
        y = self.dwconv(y)
        y = apply_mask(y, mask)
        y = y.permute(0, 2, 1)
        y = self.norm(y)
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.grn(y)
        y = self.pwconv2(y)
        y = y.permute(0, 2, 1)
        y = x + y
        return y


class GRN(nn.Module):
    """Global Response Normalization"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


class ECGConvNeXt(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 320],
        kernel_size=7,
        stem_size=4,
        downsample_factor=2,
    ):
        super().__init__()

        if len(dims) != len(depths):
            raise ValueError("dims and depths should have the same size.")

        self.depths = depths
        self.dims = dims
        self.stem_size = stem_size
        self.n_stages = len(depths)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=dims[0],
                kernel_size=stem_size,
                stride=stem_size,
            ),
            LayerNorm(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(self.n_stages - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i]),
                nn.Conv1d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=downsample_factor,
                    stride=downsample_factor,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(self.n_stages):
            stage = nn.Sequential(
                *[Block(dim=dims[i], kernel_size=kernel_size) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.head = nn.Identity()

    def upsample_mask(self, mask, scale):
        return mask.repeat_interleave(scale, axis=1)

    def forward(self, x, mask=None):
        if mask is None:
            masks = [None] * self.n_stages
        else:
            masks = [
                self.upsample_mask(mask, 2 ** (self.n_stages - 1 - i))
                .unsqueeze(1)
                .type_as(x)
                for i in range(self.n_stages)
            ]

        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            for block in stage:
                x = block(x, mask=masks[i])

        x = self.head(x)

        return x


def apply_mask(x, mask=None):
    if mask is None:
        return x
    x = x * (1.0 - mask)
    return x


def ecg_convnext(pretrained=True, **kwargs):
    if pretrained:
        return ECGConvNeXt.from_pretrained("cfauchereau/ecg-convnext")

    return ECGConvNeXt(**kwargs)
