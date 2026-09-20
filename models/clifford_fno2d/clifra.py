"""Clifford Fourier pairing and planned-product lowering with PyTorch FFT."""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from models.clifford_resnet2d.clifra import CliffordConv2d, ClifraFluidNet2d, product_basis_2d


def pair_fourier(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual pairs (scalar,bivector) and (e1,e2) in the paper's FFT convention."""
    return torch.complex(x[..., 0], x[..., 3]), torch.complex(x[..., 1], x[..., 2])


def unpair_fourier(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.stack((first.real, second.real, second.imag, first.imag), dim=-1)


class CliffordSpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, modes: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(4, out_channels, in_channels, 2 * modes, 2 * modes)
        )
        self.register_buffer("product_basis", product_basis_2d(4, 4).clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width, _ = x.shape
        first, second = pair_fourier(x)
        first_ft, second_ft = torch.fft.fft2(first), torch.fft.fft2(second)
        modes = self.modes
        transformed = torch.cat(
            (first_ft.real, second_ft.real, second_ft.imag, first_ft.imag), dim=1
        )
        retained = torch.cat(
            (
                torch.cat(
                    (transformed[:, :, :modes, :modes], transformed[:, :, :modes, -modes:]), dim=-1
                ),
                torch.cat(
                    (transformed[:, :, -modes:, :modes], transformed[:, :, -modes:, -modes:]),
                    dim=-1,
                ),
            ),
            dim=-2,
        )
        kernel = torch.einsum("koihw,bka->aobihw", self.weights, self.product_basis)
        kernel = kernel.reshape(4 * self.out_channels, 4 * self.in_channels, 2 * modes, 2 * modes)
        multiplied = torch.einsum("bixy,oixy->boxy", retained, kernel)
        result = x.new_zeros(batch, 4 * self.out_channels, height, width)
        result[:, :, :modes, :modes] = multiplied[:, :, :modes, :modes]
        result[:, :, -modes:, :modes] = multiplied[:, :, -modes:, :modes]
        result[:, :, :modes, -modes:] = multiplied[:, :, :modes, -modes:]
        result[:, :, -modes:, -modes:] = multiplied[:, :, -modes:, -modes:]
        result = result.reshape(batch, 4, self.out_channels, height, width).permute(0, 2, 3, 4, 1)
        first_out, second_out = pair_fourier(result)
        return unpair_fourier(
            torch.fft.ifft2(first_out, s=(height, width)),
            torch.fft.ifft2(second_out, s=(height, width)),
        )


class CliffordFourierBasicBlock2d(nn.Module):
    def __init__(self, channels: int, *, modes: int = 16):
        super().__init__()
        self.fourier = CliffordSpectralConv2d(channels, channels, modes=modes)
        self.conv = CliffordConv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.fourier(x) + self.conv(x))


class ClifraFNO2d(ClifraFluidNet2d):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        blocks: int,
        modes: int = 16,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            blocks=blocks,
            block_type=partial(CliffordFourierBasicBlock2d, modes=modes),
        )
