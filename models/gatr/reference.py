"""The pinned, installed Qualcomm GATr model with an optional xFormers import bridge.

The pinned GATr source imports xFormers even when unmasked attention uses PyTorch SDPA.
The bridge supplies only that unused import surface on systems without xFormers.
"""

import sys
from types import ModuleType


def _allow_sdpa_without_xformers() -> None:
    try:
        import xformers.ops  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "xformers":
            raise
        package = ModuleType("xformers")
        operators = ModuleType("xformers.ops")

        class AttentionBias:
            """Import-only placeholder; this benchmark does not use attention masks."""

        def unavailable_attention(*args, **kwargs):
            raise RuntimeError("xFormers attention was requested but is not installed")

        operators.AttentionBias = AttentionBias
        operators.memory_efficient_attention = unavailable_attention
        package.ops = operators
        sys.modules["xformers"] = package
        sys.modules["xformers.ops"] = operators


_allow_sdpa_without_xformers()

from gatr import GATr, MLPConfig, SelfAttentionConfig  # noqa: E402


def make_reference(config):
    return GATr(
        in_mv_channels=config.in_mv_channels,
        out_mv_channels=config.out_mv_channels,
        hidden_mv_channels=config.hidden_mv_channels,
        in_s_channels=config.in_s_channels,
        out_s_channels=config.out_s_channels,
        hidden_s_channels=config.hidden_s_channels,
        attention=SelfAttentionConfig(num_heads=config.heads, pos_encoding=False),
        mlp=MLPConfig(),
        num_blocks=config.blocks,
        dropout_prob=None,
        checkpoint=None,
    )
