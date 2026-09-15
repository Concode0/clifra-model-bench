"""Minimal GCA-MLP built from the official cliffordlayers GCAN primitives."""

from __future__ import annotations

import torch
from cliffordlayers.cliffordalgebra import CliffordAlgebra
from cliffordlayers.nn.modules.gcan import MultiVectorAct, PGAConjugateLinear
from torch import nn


class ReferenceGCAMLP(nn.Module):
    """Upstream GATr GCA-MLP semantics with ``flatten=False``.

    Inputs have shape ``(..., in_channels, 16)`` and outputs have shape
    ``(..., out_channels, 16)``. Leading dimensions are folded only because
    the official ``PGAConjugateLinear`` primitive accepts one batch dimension.
    """

    action_blades = (0, 5, 6, 7, 8, 9, 10, 15)
    all_blades = tuple(range(16))

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        hidden_layers: int,
        act_agg: str = "linear",
    ) -> None:
        super().__init__()
        if hidden_layers <= 0:
            raise ValueError("hidden_layers must be positive")
        if act_agg not in {"linear", "sum", "mean"}:
            raise ValueError(f"unsupported activation aggregation: {act_agg}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pga = CliffordAlgebra((0, 1, 1, 1))
        widths = [in_channels, *([hidden_channels] * hidden_layers), out_channels]
        self.linears = nn.ModuleList(
            PGAConjugateLinear(
                source,
                target,
                self.pga,
                input_blades=self.all_blades,
                action_blades=self.action_blades,
            )
            for source, target in zip(widths, widths[1:])
        )
        self.activations = nn.ModuleList(
            MultiVectorAct(
                hidden_channels,
                self.pga,
                input_blades=self.all_blades,
                agg=act_agg,
            )
            for _ in range(hidden_layers)
        )

    def _apply(self, fn):
        module = super()._apply(fn)
        parameter = next(self.parameters())
        self.pga.to(parameter.device)
        return module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != (self.in_channels, 16):
            raise ValueError(
                f"expected (..., {self.in_channels}, 16), got {tuple(inputs.shape)}"
            )
        leading = inputs.shape[:-2]
        values = inputs.reshape(-1, self.in_channels, 16)
        values = self.linears[0](values)
        for activation, linear in zip(self.activations, self.linears[1:]):
            values = linear(activation(values))
        return values.reshape(*leading, self.out_channels, 16)
