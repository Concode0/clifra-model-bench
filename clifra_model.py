"""Clifra-native, ahead-of-time-lowered implementation of GCA-MLP."""

from __future__ import annotations

import math

import torch
from clifra import make_algebra
from torch import nn


class _PGAConjugateLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        operator_basis: torch.Tensor,
        norm_signs: torch.Tensor,
        action_basis_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.action_basis_indices = action_basis_indices
        self._action = nn.Parameter(torch.empty(out_features, in_features, 8))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.embed_e0 = nn.Parameter(torch.zeros(in_features, 1))
        self.register_buffer("operator_basis", operator_basis, persistent=False)
        pairs = torch.triu_indices(8, 8)
        self.register_buffer("monomial_left", pairs[0], persistent=False)
        self.register_buffer("monomial_right", pairs[1], persistent=False)
        # e123 is lane 14 in cliffordlayers and canonical blade 0b0111 here.
        self.embedded_blade = 0b0111
        self.reset_parameters(norm_signs)

    def reset_parameters(self, norm_signs: torch.Tensor) -> None:
        with torch.no_grad():
            self._action.zero_()
            # Canonical (0,2,4)-grade order: scalar, three Euclidean rotation
            # bivectors, three ideal translation bivectors, pseudoscalar.
            self._action[..., :4].uniform_(-1.0, 1.0)
            norm = (self._action.square() * norm_signs).sum(-1, keepdim=True).abs().sqrt()
            self._action.div_(norm)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        action_products = (
            self._action[..., self.monomial_left] * self._action[..., self.monomial_right]
        )
        coefficients = action_products * self.weight[..., None]
        kernel = torch.matmul(coefficients, self.operator_basis).reshape(
            self.out_features, self.in_features, 16, 16
        )
        outputs = torch.einsum("...ir,oipr->...op", inputs, kernel)
        replacement = self.embed_e0[..., 0] - inputs[..., self.embedded_blade]
        return outputs + torch.einsum(
            "...i,oip->...op", replacement, kernel[..., self.embedded_blade]
        )


class _MultiVectorAct(nn.Module):
    def __init__(self, channels: int, agg: str, full_basis: tuple[int, ...]) -> None:
        super().__init__()
        self.agg = agg
        if agg == "linear":
            self.conv = nn.Conv1d(channels, channels, kernel_size=16, groups=channels)
        elif agg in {"sum", "mean"}:
            # The source architecture sums null-first-oriented PGA coordinates.
            # This covector is its exact expression on Clifra's canonical,
            # null-last basis; coefficient tensors themselves remain canonical.
            null_bit = 1 << 3
            signs = [
                -1.0 if blade & null_bit and (blade ^ null_bit).bit_count() % 2 else 1.0
                for blade in full_basis
            ]
            self.register_buffer("aggregate_form", torch.tensor(signs), persistent=False)
        else:
            raise ValueError(f"unsupported activation aggregation: {agg}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.agg == "linear":
            gate = self.conv(inputs)
        else:
            gate = torch.sum(inputs * self.aggregate_form, dim=-1, keepdim=True)
            if self.agg == "mean":
                gate = gate / 16.0
        return inputs * torch.sigmoid(gate)


class ClifraGCAMLP(nn.Module):
    """The reference GCA-MLP in Clifra's canonical PGA convention.

    All Clifford structure is planned and lowered during construction. Runtime
    linear layers form a 16x16 kernel from 36 quadratic action monomials and a
    fixed operator basis, then apply that kernel with ordinary PyTorch tensors.
    """

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
        self.algebra = make_algebra(3, 0, 1)
        full = self.algebra.layout()
        action = self.algebra.layout((0, 2, 4))
        operator_basis = self._lower_conjugation_basis(full, action)
        norm_plan = self.algebra.plan_signature_norm_squared(input=action)
        norm_signs = norm_plan(torch.eye(action.dim)).squeeze(-1)

        widths = [in_channels, *([hidden_channels] * hidden_layers), out_channels]
        self.linears = nn.ModuleList(
            _PGAConjugateLinear(
                source,
                target,
                operator_basis,
                norm_signs,
                action.basis_indices,
            )
            for source, target in zip(widths, widths[1:])
        )
        self.activations = nn.ModuleList(
            _MultiVectorAct(hidden_channels, act_agg, full.basis_indices)
            for _ in range(hidden_layers)
        )

    def _lower_conjugation_basis(self, full, action) -> torch.Tensor:
        """Use public plans to lower ``k x reverse(k)`` into 36 fixed maps."""
        left = self.algebra.plan_product(left=action, right=full, output=full)
        right = self.algebra.plan_product(left=full, right=action, output=full)
        reverse = self.algebra.plan_unary(op="reverse", input=action, output=action)
        action_eye = torch.eye(action.dim)
        value_eye = torch.eye(full.dim)
        first = left(action_eye[:, None, None, :], value_eye[None, None, :, :])
        maps = right(first, reverse(action_eye)[None, :, None, :]).permute(0, 1, 3, 2)
        lowered = []
        for left_index in range(action.dim):
            for right_index in range(left_index, action.dim):
                value = maps[left_index, right_index]
                if left_index != right_index:
                    value = value + maps[right_index, left_index]
                lowered.append(value)
        return torch.stack(lowered).reshape(36, 16 * 16)

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
