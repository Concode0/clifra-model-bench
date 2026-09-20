"""Clifra algebraic lowering composed with ordinary PyTorch spatial operations."""

from functools import lru_cache

import torch
from clifra import make_algebra
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=3)
def product_basis_2d(input_blades: int, output_blades: int) -> torch.Tensor:
    """Evaluate the fixed input-times-weight product on basis elements once."""
    algebra = make_algebra(2, 0)
    full = algebra.layout((0, 1, 2))
    scalar_vector = algebra.layout((0, 1))
    source = scalar_vector if input_blades == 3 else full
    target = scalar_vector if output_blades == 3 else full
    product = algebra.plan_product(left=source, right=full, output=target, pairwise=True)
    return product(torch.eye(input_blades), torch.eye(4)).detach()


class CliffordConv2d(nn.Module):
    """Right Clifford multiplication at each spatial kernel position, then conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int,
        input_blades: int = 4,
        output_blades: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_blades = input_blades
        self.output_blades = output_blades
        self.padding = padding
        self.weight = nn.Parameter(
            torch.empty(4, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(output_blades, out_channels)) if bias else None
        self.register_buffer(
            "product_basis", product_basis_2d(input_blades, output_blades).clone(), persistent=False
        )
        # Match CliffordLayers' per-blade fan-in, including all four learned blades.
        bound = 1 / (in_channels * 4 * kernel_size**2) ** 0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width, _ = x.shape
        # Basis axes are input blade, learned weight blade, output blade.
        kernel = torch.einsum("koihw,bka->aobihw", self.weight, self.product_basis)
        kernel = kernel.reshape(
            self.output_blades * self.out_channels,
            self.input_blades * self.in_channels,
            *self.weight.shape[-2:],
        )
        field = x.permute(0, 4, 1, 2, 3).reshape(
            batch, self.input_blades * self.in_channels, height, width
        )
        result = F.conv2d(
            field,
            kernel,
            None if self.bias is None else self.bias.reshape(-1),
            padding=self.padding,
        )
        return result.reshape(
            batch, self.output_blades, self.out_channels, *result.shape[-2:]
        ).permute(0, 2, 3, 4, 1)


class CliffordGroupNorm2d(nn.Module):
    """The reference's channelwise four-coefficient whitening and affine map."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.eye(4).unsqueeze(-1).expand(4, 4, channels).clone())
        self.bias = nn.Parameter(torch.zeros(4, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width, blades = x.shape
        centered = x - x.mean(dim=(2, 3), keepdim=True)
        rows = centered.permute(0, 1, 4, 2, 3).reshape(batch * channels, blades, -1)
        covariance = rows @ rows.transpose(-1, -2) / rows.shape[-1]
        eye = torch.eye(blades, dtype=x.dtype, device=x.device)
        upper = torch.linalg.cholesky(covariance + self.eps * eye).mH
        whitened = torch.linalg.solve_triangular(upper, rows, upper=True)
        whitened = whitened.reshape(batch, channels, blades, height, width).permute(0, 1, 3, 4, 2)
        affine = self.weight.permute(2, 0, 1)
        repeated_bias = self.bias.repeat(1, batch).reshape(batch, channels, 1, 1, blades)
        return torch.einsum("coi,bchwi->bchwo", affine, whitened) + repeated_bias


class CliffordBasicBlock2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = CliffordConv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = CliffordConv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = CliffordGroupNorm2d(channels)
        self.norm2 = CliffordGroupNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.gelu(self.norm1(x)))
        out = self.conv2(F.gelu(self.norm2(out)))
        return out + x


class ClifraFluidNet2d(nn.Module):
    """The official fluid model's boundary maps, padding, and block sequence."""

    padding = 9

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        blocks: int,
        block_type=CliffordBasicBlock2d,
    ):
        super().__init__()
        self.encoder = CliffordConv2d(
            in_channels, hidden_channels, kernel_size=1, padding=0, input_blades=3
        )
        self.decoder = CliffordConv2d(
            hidden_channels, out_channels, kernel_size=1, padding=0, output_blades=3, bias=False
        )
        self.layers = nn.ModuleList(
            nn.Sequential(block_type(hidden_channels)) for _ in range(blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(F.gelu(x))
        if self.padding:
            x = F.pad(x.permute(0, 4, 1, 2, 3), (0, self.padding, 0, self.padding)).permute(
                0, 2, 3, 4, 1
            )
        for layer in self.layers:
            x = layer(x)
        if self.padding:
            x = x[:, :, : -self.padding, : -self.padding, :]
        return self.decoder(x)


class ClifraResNet2d(ClifraFluidNet2d):
    pass
