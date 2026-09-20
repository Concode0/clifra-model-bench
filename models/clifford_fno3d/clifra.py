"""Cl(3,0) Maxwell Fourier model from planned products and PyTorch spatial operations."""

from functools import lru_cache, partial
from itertools import product as corners

import torch
from clifra import make_algebra
from torch import nn
from torch.nn import functional as F

FULL = (0, 1, 2, 3)
MAXWELL = (1, 2)


@lru_cache(maxsize=3)
def product_basis_3d(input_grades: tuple[int, ...], output_grades: tuple[int, ...]):
    """Lower input-times-weight multiplication on canonical Clifra basis lanes once."""
    algebra = make_algebra(3, 0)
    source = algebra.layout(input_grades)
    weight = algebra.layout(FULL)
    target = algebra.layout(output_grades)
    operation = algebra.plan_product(left=source, right=weight, output=target, pairwise=True)
    return operation(torch.eye(source.dim), torch.eye(weight.dim)).detach()


class CliffordConv3d(nn.Module):
    """Right Clifford multiplication lowered to a PyTorch conv3d kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 1,
        input_grades: tuple[int, ...] = FULL,
        output_grades: tuple[int, ...] = FULL,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        basis = product_basis_3d(input_grades, output_grades)
        self.input_blades, _, self.output_blades = basis.shape
        self.weight = nn.Parameter(
            torch.empty(8, out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(self.output_blades, out_channels)) if bias else None
        self.register_buffer("product_basis", basis.clone(), persistent=False)
        # Pinned CliffordLayers scales the bound by all eight learned blades.
        bound = 1 / (in_channels * 8 * kernel_size**3) ** 0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, depth, height, width, _ = x.shape
        kernel = torch.einsum("koidhw,bka->aobidhw", self.weight, self.product_basis)
        kernel = kernel.reshape(
            self.output_blades * self.out_channels,
            self.input_blades * self.in_channels,
            *self.weight.shape[-3:],
        )
        field = x.permute(0, 5, 1, 2, 3, 4).reshape(
            batch, self.input_blades * self.in_channels, depth, height, width
        )
        result = F.conv3d(field, kernel, None if self.bias is None else self.bias.reshape(-1))
        return result.reshape(
            batch, self.output_blades, self.out_channels, *result.shape[-3:]
        ).permute(0, 2, 3, 4, 5, 1)


def pair_fourier(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Pinned dual pairs, expressed in Clifra's canonical [1,e1,e2,e12,e3,...] order."""
    return (
        torch.complex(x[..., 0], x[..., 7]),
        torch.complex(x[..., 1], x[..., 6]),
        torch.complex(x[..., 2], x[..., 5]),
        torch.complex(x[..., 4], x[..., 3]),
    )


def unpair_fourier(pairs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    first, second, third, fourth = pairs
    return torch.stack(
        (
            first.real,
            second.real,
            third.real,
            fourth.imag,
            fourth.real,
            third.imag,
            second.imag,
            first.imag,
        ),
        dim=-1,
    )


class CliffordSpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(8, out_channels, in_channels, 2 * modes, 2 * modes, 2 * modes)
        )
        self.register_buffer(
            "product_basis", product_basis_3d(FULL, FULL).clone(), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, depth, height, width, _ = x.shape
        modes = self.modes
        first, second, third, fourth = (
            torch.fft.fftn(pair, dim=(-3, -2, -1)) for pair in pair_fourier(x)
        )
        transformed = torch.cat(
            (
                first.real,
                second.real,
                third.real,
                fourth.imag,
                fourth.real,
                third.imag,
                second.imag,
                first.imag,
            ),
            dim=1,
        )
        retained = transformed
        for axis in (-1, -2, -3):
            retained = torch.cat(
                (
                    retained.narrow(axis, 0, modes),
                    retained.narrow(axis, retained.shape[axis] - modes, modes),
                ),
                dim=axis,
            )
        kernel = torch.einsum("koidhw,bka->aobidhw", self.weights, self.product_basis)
        kernel = kernel.reshape(
            8 * self.out_channels, 8 * self.in_channels, 2 * modes, 2 * modes, 2 * modes
        )
        multiplied = torch.einsum("bidhw,oidhw->bodhw", retained, kernel)
        restored = x.new_zeros(batch, 8 * self.out_channels, depth, height, width)
        for corner in corners((0, 1), repeat=3):
            destination = tuple(
                slice(0, modes) if side == 0 else slice(-modes, None) for side in corner
            )
            source = tuple(slice(0, modes) if side == 0 else slice(modes, None) for side in corner)
            if corner == (1, 0, 1):
                # This one block uses the first depth slice in the pinned CliffordLayers source.
                source = (slice(0, modes), source[1], source[2])
            restored[(slice(None), slice(None), *destination)] = multiplied[
                (slice(None), slice(None), *source)
            ]
        restored = restored.reshape(batch, 8, self.out_channels, depth, height, width)
        restored = restored.permute(0, 2, 3, 4, 5, 1)
        return unpair_fourier(
            tuple(
                torch.fft.ifftn(pair, s=(depth, height, width)) for pair in pair_fourier(restored)
            )
        )


class CliffordFourierBasicBlock3d(nn.Module):
    def __init__(self, channels: int, *, modes: int = 8):
        super().__init__()
        self.fourier = CliffordSpectralConv3d(channels, channels, modes=modes)
        self.conv = CliffordConv3d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.fourier(x) + self.conv(x))


class ClifraFNO3d(nn.Module):
    padding = 2

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        blocks: int,
        modes: int = 8,
    ):
        super().__init__()
        self.encoder = CliffordConv3d(in_channels, hidden_channels, input_grades=MAXWELL)
        self.decoder = CliffordConv3d(
            hidden_channels, out_channels, output_grades=MAXWELL, bias=False
        )
        self.layers = nn.ModuleList(
            nn.Sequential(partial(CliffordFourierBasicBlock3d, modes=modes)(hidden_channels))
            for _ in range(blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(F.gelu(x))
        x = F.pad(x.permute(0, 5, 1, 2, 3, 4), (0, self.padding, 0, self.padding, 0, self.padding))
        x = x.permute(0, 2, 3, 4, 5, 1)
        for layer in self.layers:
            x = layer(x)
        x = x[:, :, : -self.padding, : -self.padding, : -self.padding, :]
        return self.decoder(x)
