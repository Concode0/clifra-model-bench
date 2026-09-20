"""Official CliffordLayers 2D fluid Fourier model at the pinned Microsoft revision."""

from cliffordlayers.models.models_2d import CliffordFluidNet2d, CliffordFourierBasicBlock2d
from cliffordlayers.models.utils import partialclass
from torch.nn import functional as F


class ReferenceFNO2d(CliffordFluidNet2d):
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
            g=[1, 1],
            block=partialclass(
                "CliffordFourierBasicBlock2dConfigured",
                CliffordFourierBasicBlock2d,
                modes1=modes,
                modes2=modes,
            ),
            num_blocks=[1] * blocks,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            activation=F.gelu,
            rotation=False,
            norm=False,
        )
