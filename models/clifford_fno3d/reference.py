"""Pinned Microsoft CliffordLayers 3D Maxwell Fourier architecture."""

from cliffordlayers.models.models_3d import CliffordFourierBasicBlock3d, CliffordMaxwellNet3d
from cliffordlayers.models.utils import partialclass
from torch.nn import functional as F


class ReferenceFNO3d(CliffordMaxwellNet3d):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        blocks: int,
        modes: int = 8,
    ):
        super().__init__(
            g=[1, 1, 1],
            block=partialclass(
                "CliffordFourierBasicBlock3dConfigured",
                CliffordFourierBasicBlock3d,
                modes1=modes,
                modes2=modes,
                modes3=modes,
            ),
            num_blocks=[1] * blocks,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            activation=F.gelu,
            norm=False,
        )
