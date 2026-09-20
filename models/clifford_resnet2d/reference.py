"""Official CliffordLayers 2D fluid ResNet, at the pinned Microsoft revision."""

import torch
from cliffordlayers.models.models_2d import CliffordBasicBlock2d, CliffordFluidNet2d
from cliffordlayers.nn.modules.groupnorm import CliffordGroupNorm2d
from torch.nn import functional as F


class MPSCliffordGroupNorm2d(CliffordGroupNorm2d):
    """Preserve upstream whitening while avoiding a broken MPS solve layout.

    On PyTorch 2.14 MPS, upstream's one-RHS-per-pixel triangular solve can
    return zeros. Solving the same covariance against all pixels at once agrees
    with the upstream implementation on CPU, including its gradients.
    """

    def forward(self, x):
        batch, channels, height, width, blades = x.shape
        centered = x - x.mean(dim=(2, 3), keepdim=True)
        rows = centered.permute(0, 1, 4, 2, 3).reshape(batch * channels, blades, -1)
        covariance = rows @ rows.transpose(-1, -2) / rows.shape[-1]
        eye = torch.eye(blades, dtype=x.dtype, device=x.device)
        upper = torch.linalg.cholesky(covariance + self.eps * eye).mH
        whitened = torch.linalg.solve_triangular(upper, rows, upper=True)
        whitened = whitened.reshape(batch, channels, blades, height, width).permute(
            0, 1, 3, 4, 2
        )
        affine = self.weight.permute(2, 0, 1)
        repeated_bias = self.bias.repeat(1, batch).reshape(batch, channels, 1, 1, blades)
        return torch.einsum("coi,bchwi->bchwo", affine, whitened) + repeated_bias


class ReferenceResNet2d(CliffordFluidNet2d):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        blocks: int,
        mps_norm_patch: bool = False,
    ):
        super().__init__(
            g=[1, 1],
            block=CliffordBasicBlock2d,
            num_blocks=[1] * blocks,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            activation=F.gelu,
            rotation=False,
            norm=True,
            num_groups=1,
        )
        if mps_norm_patch:
            for layer in self.layers:
                for block in layer:
                    for name in ("norm1", "norm2"):
                        original = getattr(block, name)
                        corrected = MPSCliffordGroupNorm2d([1, 1], 1, hidden_channels)
                        corrected.load_state_dict(original.state_dict())
                        setattr(block, name, corrected)
