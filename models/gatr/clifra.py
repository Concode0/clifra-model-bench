"""GATr from Clifra PGA plans and ordinary PyTorch transformer operations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from clifra import make_algebra
from torch import nn


class PGAStructure(nn.Module):
    """Fixed Cl(3,0,1) maps, lowered once from Clifra's public algebra plans."""

    def __init__(self):
        super().__init__()
        algebra = make_algebra(3, 0, 1)
        full = algebra.layout()
        self.gp = algebra.plan_product(left=full, right=full, output=full)
        self.wedge = algebra.plan_product(op="wedge", left=full, right=full, output=full)
        blades = full.basis_indices
        positions = {blade: i for i, blade in enumerate(blades)}
        identity = torch.eye(full.dim)
        wedge_table = algebra.plan_product(
            op="wedge", left=full, right=full, output=full, pairwise=True
        )(identity, identity)

        # Poincare complement: e_A maps to the complementary blade with the
        # orientation of e_A wedge e_complement = I. This is not the degenerate
        # algebra's pseudoscalar product.
        dual_source = []
        dual_signs = []
        for target in blades:
            source = ((1 << algebra.n) - 1) ^ target
            source_lane = positions[source]
            target_lane = positions[target]
            dual_source.append(source_lane)
            # GATr orients the reference pseudoscalar e0e1e2e3; in
            # Clifra's null-last basis that pseudoscalar is -e1e2e3e0.
            dual_signs.append(-float(wedge_table[source_lane, target_lane, positions[15]]))
        self.register_buffer("dual_source", torch.tensor(dual_source), persistent=False)
        self.register_buffer("dual_signs", torch.tensor(dual_signs), persistent=False)

        # Five grade projections, followed by four null-generator insertion maps.
        # The latter arise from e_null wedge e_E, with E Euclidean. Normalize
        # each basis map exactly as the official Pin-linear parameterization.
        basis = []
        for grade in range(5):
            matrix = torch.diag(full.grade_mask((grade,)).float())
            basis.append(matrix / torch.linalg.norm(matrix))
        null_bit = 1 << algebra.p
        for grade in range(4):
            matrix = torch.zeros(full.dim, full.dim)
            for euclidean_blade in blades:
                if euclidean_blade & null_bit or euclidean_blade.bit_count() != grade:
                    continue
                source = euclidean_blade | null_bit
                matrix[positions[source], positions[euclidean_blade]] = wedge_table[
                    positions[null_bit], positions[euclidean_blade], positions[source]
                ]
            basis.append(matrix / torch.linalg.norm(matrix))
        self.register_buffer("linear_basis", torch.stack(basis), persistent=False)
        norm_signs = algebra.plan_signature_norm_squared(input=full)(identity).flatten()
        self.register_buffer("inner_mask", norm_signs, persistent=False)
        self.register_buffer(
            "attention_inner_indices",
            torch.tensor([positions[b] for b in (0, 1, 2, 4, 3, 5, 6)]),
            persistent=False,
        )
        self.register_buffer(
            "trivector_indices",
            torch.tensor([positions[b] for b in (11, 13, 14, 7)]),
            persistent=False,
        )
        self.reference_lane = positions[7]
        self.scalar_lane = positions[0]

    def dual(self, values):
        return values.index_select(-1, self.dual_source) * self.dual_signs

    def join(self, left, right, reference):
        result = self.dual(self.wedge(self.dual(left), self.dual(right)))
        return reference[..., self.reference_lane : self.reference_lane + 1] * result

    def inner(self, left, right):
        return (left * right * self.inner_mask).sum(dim=-1, keepdim=True)


class ClifraEquiLinear(nn.Module):
    def __init__(self, structure, in_mv, out_mv, in_s, out_s, bias=True):
        super().__init__()
        self.structure = structure
        self.weight = nn.Parameter(torch.empty(out_mv, in_mv, 9))
        self.bias = nn.Parameter(torch.zeros(out_mv, 1)) if bias and in_s is None else None
        self.s2mvs = nn.Linear(in_s, out_mv, bias=bias) if in_s else None
        self.mvs2s = nn.Linear(in_mv, out_s, bias=bias) if out_s else None
        self.s2s = (
            nn.Linear(in_s, out_s, bias=False) if in_s is not None and out_s is not None else None
        )
        nn.init.normal_(self.weight, std=1 / math.sqrt(in_mv * 9))

    def forward(self, multivectors, scalars=None):
        basis = self.structure.linear_basis
        outputs_mv = torch.einsum("oia,axy,...iy->...ox", self.weight, basis, multivectors)
        scalar_part = None
        if self.bias is not None:
            scalar_part = self.bias.squeeze(-1)
        if self.s2mvs is not None and scalars is not None:
            addition = self.s2mvs(scalars)
            scalar_part = addition if scalar_part is None else scalar_part + addition
        if scalar_part is not None:
            outputs_mv = outputs_mv + F.pad(scalar_part.unsqueeze(-1), (0, 15))
        outputs_s = self.mvs2s(multivectors[..., 0]) if self.mvs2s is not None else None
        if self.s2s is not None and scalars is not None:
            outputs_s = outputs_s + self.s2s(scalars)
        return outputs_mv, outputs_s


class ClifraEquiLayerNorm(nn.Module):
    def __init__(self, structure):
        super().__init__()
        self.structure = structure

    def forward(self, multivectors, scalars):
        squared = self.structure.inner(multivectors, multivectors).mean(dim=-2, keepdim=True)
        mv = multivectors / torch.sqrt(torch.clamp(squared, min=0.01))
        return mv, F.layer_norm(scalars, scalars.shape[-1:])


class ClifraGeometricBilinear(nn.Module):
    def __init__(self, structure, in_mv, out_mv, in_s, out_s):
        super().__init__()
        self.structure = structure
        half = out_mv // 2
        if 2 * half != out_mv:
            raise ValueError("bilinear hidden width must be even")
        self.linear_left = ClifraEquiLinear(structure, in_mv, half, in_s, None)
        self.linear_right = ClifraEquiLinear(structure, in_mv, half, in_s, None)
        self.linear_join_left = ClifraEquiLinear(structure, in_mv, half, in_s, None)
        self.linear_join_right = ClifraEquiLinear(structure, in_mv, half, in_s, None)
        self.linear_out = ClifraEquiLinear(structure, out_mv, out_mv, in_s, out_s)

    def forward(self, multivectors, reference_mv, scalars):
        left, _ = self.linear_left(multivectors, scalars)
        right, _ = self.linear_right(multivectors, scalars)
        gp = self.structure.gp(left, right)
        left, _ = self.linear_join_left(multivectors, scalars)
        right, _ = self.linear_join_right(multivectors, scalars)
        joined = self.structure.join(left, right, reference_mv)
        return self.linear_out(torch.cat((gp, joined), dim=-2), scalars)


class ClifraScalarGatedNonlinearity(nn.Module):
    def forward(self, multivectors, scalars):
        gate = multivectors[..., :1]
        return F.gelu(gate, approximate="tanh") * multivectors, F.gelu(scalars)


class ClifraGeoMLP(nn.Module):
    def __init__(self, structure, mv_channels, s_channels):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                ClifraGeometricBilinear(
                    structure, mv_channels, 2 * mv_channels, s_channels, 2 * s_channels
                ),
                ClifraScalarGatedNonlinearity(),
                ClifraEquiLinear(
                    structure, 2 * mv_channels, mv_channels, 2 * s_channels, s_channels
                ),
            )
        )

    def forward(self, multivectors, scalars, reference_mv):
        mv, s = self.layers[0](multivectors, reference_mv, scalars)
        mv, s = self.layers[1](mv, s)
        return self.layers[2](mv, s)


class ClifraMultiQueryQKV(nn.Module):
    def __init__(self, structure, mv_channels, s_channels, heads, hidden_mv, hidden_s):
        super().__init__()
        self.q_linear = ClifraEquiLinear(
            structure, mv_channels, heads * hidden_mv, s_channels, heads * hidden_s
        )
        self.k_linear = ClifraEquiLinear(structure, mv_channels, hidden_mv, s_channels, hidden_s)
        self.v_linear = ClifraEquiLinear(structure, mv_channels, hidden_mv, s_channels, hidden_s)
        self.heads = heads
        self.hidden_mv = hidden_mv
        self.hidden_s = hidden_s

    def forward(self, multivectors, scalars):
        q_mv, q_s = self.q_linear(multivectors, scalars)
        k_mv, k_s = self.k_linear(multivectors, scalars)
        v_mv, v_s = self.v_linear(multivectors, scalars)
        batch, items = q_mv.shape[:2]
        q_mv = q_mv.reshape(batch, items, self.hidden_mv, self.heads, 16).permute(0, 3, 1, 2, 4)
        q_s = q_s.reshape(batch, items, self.hidden_s, self.heads).permute(0, 3, 1, 2)
        k_mv, v_mv = k_mv.unsqueeze(1), v_mv.unsqueeze(1)
        k_s, v_s = k_s.unsqueeze(1), v_s.unsqueeze(1)
        return q_mv, k_mv, v_mv, q_s, k_s, v_s


class ClifraGeometricAttention(nn.Module):
    def __init__(self, structure, heads, hidden_mv, epsilon=1e-3):
        super().__init__()
        self.structure = structure
        self.log_weights = nn.Parameter(torch.zeros(heads, 1, hidden_mv))
        self.epsilon = epsilon

    def features(self, q_mv, k_mv, v_mv, q_s, k_s, v_s):
        q_tri = q_mv.index_select(-1, self.structure.trivector_indices)
        k_tri = k_mv.index_select(-1, self.structure.trivector_indices)

        def distance_vector(tri, query):
            tri = tri * (tri[..., 3:4] / (tri[..., 3:4].square() + self.epsilon))
            spatial, homogeneous = tri[..., :3], tri[..., 3:4]
            if query:
                return torch.cat(
                    (
                        spatial.square().sum(-1, keepdim=True),
                        homogeneous.square(),
                        spatial * homogeneous,
                    ),
                    dim=-1,
                )
            return torch.cat(
                (
                    -homogeneous.square(),
                    -spatial.square().sum(-1, keepdim=True),
                    2 * spatial * homogeneous,
                ),
                dim=-1,
            )

        q_dist = distance_vector(q_tri, True) * self.log_weights.exp().unsqueeze(0).unsqueeze(-1)
        k_dist = distance_vector(k_tri, False)
        q_inner = q_mv.index_select(-1, self.structure.attention_inner_indices).flatten(-2)
        k_inner = k_mv.index_select(-1, self.structure.attention_inner_indices).flatten(-2)
        q_width = q_inner.shape[-1] + q_dist.shape[-2] * 5 + q_s.shape[-1]
        v_width = v_mv.shape[-2] * 16 + v_s.shape[-1]
        width = math.ceil(max(q_width, v_width) / 8) * 8
        q = F.pad(torch.cat((q_inner, q_dist.flatten(-2), q_s), dim=-1), (0, width - q_width))
        k = F.pad(torch.cat((k_inner, k_dist.flatten(-2), k_s), dim=-1), (0, width - q_width))
        v = F.pad(torch.cat((v_mv.flatten(-2), v_s), dim=-1), (0, width - v_width))
        return q, k * math.sqrt(width / q_width), v

    def forward(self, q_mv, k_mv, v_mv, q_s, k_s, v_s):
        q, k, v = self.features(q_mv, k_mv, v_mv, q_s, k_s, v_s)
        k = k.expand(k.shape[0], q.shape[1], k.shape[2], k.shape[3])
        v = v.expand(v.shape[0], q.shape[1], v.shape[2], v.shape[3])
        result = F.scaled_dot_product_attention(q, k, v)
        mv_width = v_mv.shape[-2] * 16
        mv = result[..., :mv_width].reshape(*result.shape[:-1], v_mv.shape[-2], 16)
        scalars = result[..., mv_width : mv_width + v_s.shape[-1]]
        return mv, scalars


class ClifraSelfAttention(nn.Module):
    def __init__(self, structure, mv_channels, s_channels, heads):
        super().__init__()
        hidden_mv = max(2 * mv_channels // heads, 1)
        hidden_s = max(2 * s_channels // heads, 4)
        self.qkv_module = ClifraMultiQueryQKV(
            structure, mv_channels, s_channels, heads, hidden_mv, hidden_s
        )
        self.attention = ClifraGeometricAttention(structure, heads, hidden_mv)
        self.out_linear = ClifraEquiLinear(
            structure, heads * hidden_mv, mv_channels, heads * hidden_s, s_channels
        )

    def forward(self, multivectors, scalars):
        qkv = self.qkv_module(multivectors, scalars)
        mv, s = self.attention(*qkv)
        batch, heads, items, channels, _ = mv.shape
        mv = mv.permute(0, 2, 1, 3, 4).reshape(batch, items, heads * channels, 16)
        s = s.permute(0, 2, 1, 3).reshape(batch, items, -1)
        return self.out_linear(mv, s)


class ClifraGATrBlock(nn.Module):
    def __init__(self, structure, mv_channels, s_channels, heads):
        super().__init__()
        self.norm = ClifraEquiLayerNorm(structure)
        self.attention = ClifraSelfAttention(structure, mv_channels, s_channels, heads)
        self.mlp = ClifraGeoMLP(structure, mv_channels, s_channels)

    def forward(self, multivectors, scalars, reference_mv):
        mv, s = self.norm(multivectors, scalars)
        mv, s = self.attention(mv, s)
        multivectors, scalars = multivectors + mv, scalars + s
        mv, s = self.norm(multivectors, scalars)
        mv, s = self.mlp(mv, s, reference_mv)
        return multivectors + mv, scalars + s


class ClifraGATr(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.structure = PGAStructure()
        st = self.structure
        self.linear_in = ClifraEquiLinear(
            st,
            config.in_mv_channels,
            config.hidden_mv_channels,
            config.in_s_channels,
            config.hidden_s_channels,
        )
        self.blocks = nn.ModuleList(
            ClifraGATrBlock(st, config.hidden_mv_channels, config.hidden_s_channels, config.heads)
            for _ in range(config.blocks)
        )
        self.linear_out = ClifraEquiLinear(
            st,
            config.hidden_mv_channels,
            config.out_mv_channels,
            config.hidden_s_channels,
            config.out_s_channels,
        )

    def forward(self, multivectors, scalars):
        # The data reference is part of the established forward computation.
        reference_mv = multivectors.mean(dim=tuple(range(1, multivectors.ndim - 1)), keepdim=True)
        mv, s = self.linear_in(multivectors, scalars)
        for block in self.blocks:
            mv, s = block(mv, s, reference_mv)
        return self.linear_out(mv, s)
