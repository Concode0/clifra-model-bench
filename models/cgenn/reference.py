"""Execution-only port of DavidRuhe/CGENN at 08f2f2cb.

The pinned tree has no installable package and its algebra module has an indentation
error in ``b``. This port retains the learned modules and their forward operations,
omitting only trainer, dataset, and metric wrappers. The algebra operation below uses
the source's shortlex Cayley convention and its camera-ready formulas. The fixed
product-path mask is registered as a buffer so it follows module device movement.
"""

import itertools
import math

import torch
from torch import nn


class CliffordAlgebra(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.register_buffer("metric", torch.as_tensor(metric))
        self.dim = len(metric)
        blades = [
            sum(1 << i for i in choice)
            for grade in range(self.dim + 1)
            for choice in itertools.combinations(range(self.dim), grade)
        ]
        self.index_to_bitmap = tuple(blades)
        self.bitmap_to_index = {blade: i for i, blade in enumerate(blades)}
        self.n_blades = len(blades)
        self.grades = tuple(range(self.dim + 1))
        self.n_subspaces = self.dim + 1
        self.register_buffer(
            "subspaces", torch.tensor([math.comb(self.dim, g) for g in self.grades])
        )
        starts = [sum(math.comb(self.dim, k) for k in range(g)) for g in self.grades]
        self.grade_to_slice = [
            slice(start, start + math.comb(self.dim, g)) for g, start in enumerate(starts)
        ]
        for grade, span in enumerate(self.grade_to_slice):
            self.register_buffer(
                f"_grade_to_index_{grade}", torch.arange(span.start, span.stop), persistent=False
            )
        self.register_buffer(
            "bbo_grades",
            torch.tensor([b.bit_count() for b in blades], dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "_beta_signs", (-1) ** (self.bbo_grades * (self.bbo_grades - 1) // 2), persistent=False
        )
        self.register_buffer("even_grades", self.bbo_grades % 2 == 0)
        self.register_buffer("odd_grades", ~self.even_grades)
        cayley = torch.zeros(self.n_blades, self.n_blades, self.n_blades)
        for left, a in enumerate(blades):
            for right, b in enumerate(blades):
                sign = 1.0
                for i in range(self.dim):
                    if a & (1 << i):
                        sign *= (-1.0) ** (b & ((1 << i) - 1)).bit_count()
                        if b & (1 << i):
                            sign *= float(metric[i])
                cayley[left, self.bitmap_to_index[a ^ b], right] = sign
        self.register_buffer("cayley", cayley)

    @property
    def grade_to_index(self):
        return [getattr(self, f"_grade_to_index_{g}") for g in self.grades]

    @property
    def geometric_product_paths(self):
        paths = torch.zeros((self.dim + 1,) * 3, dtype=torch.bool, device=self.cayley.device)
        for i, j, k in itertools.product(self.grades, repeat=3):
            paths[i, j, k] = (
                self.cayley[self.grade_to_slice[i], self.grade_to_slice[j], self.grade_to_slice[k]]
                != 0
            ).any()
        return paths

    def geometric_product(self, left, right):
        return torch.einsum("...i,ijk,...k->...j", left, self.cayley, right)

    def embed(self, values, indices):
        result = values.new_zeros(*values.shape[:-1], self.n_blades)
        result[..., indices] = values
        return result

    def embed_grade(self, values, grade):
        result = values.new_zeros(*values.shape[:-1], self.n_blades)
        result[..., self.grade_to_slice[grade]] = values
        return result

    def get_grade(self, values, grade):
        return values[..., self.grade_to_slice[grade]]

    def q(self, values, blades=None):
        if blades is None:
            indices = torch.arange(self.n_blades, device=values.device)
        else:
            indices = blades
        scalar = self.cayley[indices[:, None], 0, indices]
        reversed_values = values * self._beta_signs[indices]
        return torch.einsum("...i,ij,...j->...", reversed_values, scalar, values).unsqueeze(-1)

    def norm(self, values, blades=None):
        return (self.q(values, blades=blades).square() + 1e-16).pow(0.25)

    def norms(self, values, grades=None):
        grades = self.grades if grades is None else grades
        return [
            self.norm(self.get_grade(values, grade), self.grade_to_index[grade]) for grade in grades
        ]

    def qs(self, values, grades=None):
        grades = self.grades if grades is None else grades
        return [
            self.q(self.get_grade(values, grade), self.grade_to_index[grade]) for grade in grades
        ]

    def rho(self, versor, values):
        odd = bool(torch.all(versor[..., self.even_grades] == 0))
        even = bool(torch.all(versor[..., self.odd_grades] == 0))
        if odd == even:
            raise ValueError("versor must have one parity")
        adjusted = values * (torch.where(self.odd_grades, -1.0, 1.0) if odd else 1.0)
        inverse = versor * self._beta_signs / self.q(versor)
        return self.geometric_product(self.geometric_product(versor, adjusted), inverse)


class MVLinear(nn.Module):
    def __init__(self, algebra, in_features, out_features, subspaces=True, bias=True):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.subspaces = subspaces
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, algebra.n_subspaces)
            if subspaces
            else torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.empty(1, out_features, 1)) if bias else None
        nn.init.normal_(self.weight, std=1 / math.sqrt(in_features))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, values):
        if self.subspaces:
            weight = self.weight.repeat_interleave(self.algebra.subspaces, dim=-1)
            result = torch.einsum("bm...i,nmi->bn...i", values, weight)
        else:
            result = torch.einsum("bm...i,nm->bn...i", values, self.weight)
        if self.bias is not None:
            bias = self.algebra.embed(self.bias, (0,))
            while bias.ndim < result.ndim:
                bias = bias.unsqueeze(2)
            result = result + bias
        return result


class NormalizationLayer(nn.Module):
    def __init__(self, algebra, features, init=0):
        super().__init__()
        self.algebra = algebra
        self.in_features = features
        self.a = nn.Parameter(torch.zeros(features, algebra.n_subspaces) + init)

    def forward(self, values):
        norms = torch.cat(self.algebra.norms(values), dim=-1)
        norms = torch.sigmoid(self.a) * (norms - 1) + 1
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        return values / (norms + 1e-6)


class MVSiLU(nn.Module):
    def __init__(self, algebra, channels, invariant="mag2", exclude_dual=False):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.invariant = invariant
        self.exclude_dual = exclude_dual
        self.a = nn.Parameter(torch.ones(1, channels, algebra.n_subspaces))
        self.b = nn.Parameter(torch.zeros(1, channels, algebra.n_subspaces))

    def forward(self, values):
        get = self.algebra.norms if self.invariant == "norm" else self.algebra.qs
        invariants = torch.cat(
            [values[..., :1], *get(values, grades=self.algebra.grades[1:])], dim=-1
        )
        a, b = self.a, self.b
        while a.ndim < invariants.ndim:
            a, b = a.unsqueeze(2), b.unsqueeze(2)
        gate = (a * invariants + b).repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.sigmoid(gate) * values


class MVLayerNorm(nn.Module):
    def __init__(self, algebra, channels):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.a = nn.Parameter(torch.ones(1, channels))

    def forward(self, values):
        norm = self.algebra.norm(values)[..., :1].mean(dim=1, keepdim=True) + 1e-6
        a = self.a
        while a.ndim < norm.ndim:
            a = a.unsqueeze(2)
        return a * values / norm


class _ProductLayer(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        out_features,
        *,
        connected,
        include_first_order=True,
        normalization_init=0,
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.include_first_order = include_first_order
        self.normalization = (
            NormalizationLayer(algebra, in_features, normalization_init)
            if normalization_init is not None
            else nn.Identity()
        )
        self.linear_right = MVLinear(algebra, in_features, in_features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(algebra, in_features, out_features, bias=True)
        # The pinned source stores this mask as an attribute; registering the same
        # fixed mask lets module.to(device) move it with the learned weights.
        self.register_buffer("product_paths", algebra.geometric_product_paths, persistent=False)
        shape = (out_features, in_features) if connected else (in_features,)
        self.weight = nn.Parameter(torch.empty(*shape, int(self.product_paths.sum())))
        std = 1 / math.sqrt((in_features if connected else 1) * (algebra.dim + 1))
        nn.init.normal_(self.weight, std=std)
        self.connected = connected

    def _get_weight(self):
        weight = self.weight.new_zeros(*self.weight.shape[:-1], *self.product_paths.shape)
        weight[..., self.product_paths] = self.weight
        for axis in (-3, -2, -1):
            weight = weight.repeat_interleave(self.algebra.subspaces, dim=axis)
        return self.algebra.cayley * weight

    def forward(self, values):
        right = self.normalization(self.linear_right(values))
        weight = self._get_weight()
        if self.connected:
            product = torch.einsum("bni,mnijk,bnk->bmj", values, weight, right)
        else:
            product = torch.einsum("bni,nijk,bnk->bnj", values, weight, right)
        if self.include_first_order:
            return (self.linear_left(values) + product) / math.sqrt(2)
        return product


class SteerableGeometricProductLayer(_ProductLayer):
    def __init__(self, algebra, features, include_first_order=True, normalization_init=0):
        super().__init__(
            algebra,
            features,
            features,
            connected=False,
            include_first_order=include_first_order,
            normalization_init=normalization_init,
        )


class FullyConnectedSteerableGeometricProductLayer(_ProductLayer):
    def __init__(
        self, algebra, in_features, out_features, include_first_order=True, normalization_init=0
    ):
        super().__init__(
            algebra,
            in_features,
            out_features,
            connected=True,
            include_first_order=include_first_order,
            normalization_init=normalization_init,
        )


class O3CGMLP(nn.Module):
    def __init__(self, hidden_features=16, num_layers=3, in_features=3, out_features=1):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        net = [
            FullyConnectedSteerableGeometricProductLayer(self.algebra, in_features, hidden_features)
        ]
        net.extend(MVSiLU(self.algebra, hidden_features) for _ in range(num_layers - 1))
        net.append(
            FullyConnectedSteerableGeometricProductLayer(
                self.algebra, hidden_features, out_features
            )
        )
        self.net = nn.Sequential(*net)

    def forward(self, values):
        return self.net(values)[:, 0, -1]


class O5CGMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0,) * 5)
        self.gp = nn.Sequential(
            MVLinear(self.algebra, 2, 8, subspaces=False),
            SteerableGeometricProductLayer(self.algebra, 8),
        )
        self.mlp = nn.Sequential(
            nn.Linear(8, 580), nn.ReLU(), nn.Linear(580, 580), nn.ReLU(), nn.Linear(580, 1)
        )

    def forward(self, values):
        return self.mlp(self.gp(values)[..., 0])


class ConvexHullCGMLP(nn.Module):
    def __init__(self, in_features=16, hidden_features=8, out_features=1, num_layers=4):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0,) * 5)
        self.net = nn.Sequential(
            MVLinear(self.algebra, in_features, hidden_features, subspaces=False),
            *(
                SteerableGeometricProductLayer(self.algebra, hidden_features)
                for _ in range(num_layers)
            ),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, values):
        return self.mlp(self.net(values).norm(dim=-1)).squeeze(-1)


def unsorted_segment_mean(data, segment_ids, num_segments):
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full((num_segments, data.size(1)), 0)
    count = data.new_full((num_segments, data.size(1)), 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class CEMLP(nn.Module):
    def __init__(
        self, algebra, in_features, hidden_features, out_features, n_layers=2, normalization_init=0
    ):
        super().__init__()
        layers = []
        for _ in range(n_layers - 1):
            layers.append(
                nn.Sequential(
                    MVLinear(algebra, in_features, hidden_features),
                    MVSiLU(algebra, hidden_features),
                    SteerableGeometricProductLayer(
                        algebra, hidden_features, normalization_init=normalization_init
                    ),
                    MVLayerNorm(algebra, hidden_features),
                )
            )
            in_features = hidden_features
        layers.append(
            nn.Sequential(
                MVLinear(algebra, in_features, out_features),
                MVSiLU(algebra, out_features),
                SteerableGeometricProductLayer(
                    algebra, out_features, normalization_init=normalization_init
                ),
                MVLayerNorm(algebra, out_features),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, values):
        for layer in self.layers:
            values = layer(values)
        return values


class EGCL(nn.Module):
    def __init__(
        self, algebra, features, edge_attr_features=1, residual=True, normalization_init=0
    ):
        super().__init__()
        self.residual = residual
        self.edge_model = CEMLP(
            algebra,
            features + edge_attr_features,
            features,
            features,
            normalization_init=normalization_init,
        )
        self.node_model = CEMLP(
            algebra, 2 * features, features, features, normalization_init=normalization_init
        )

    def forward(self, h, edge_index, edge_attr=None):
        rows, cols = edge_index
        message = h[rows] - h[cols]
        if edge_attr is not None:
            message = torch.cat([message, edge_attr], dim=1)
        h_msg = self.edge_model(message)
        aggregate = unsorted_segment_mean(h_msg.flatten(1), rows, num_segments=len(h))
        aggregate = aggregate.view(len(h), *h_msg.shape[1:])
        out_h = self.node_model(torch.cat([h, aggregate], dim=1))
        return h + out_h if self.residual else out_h


class NBodyCGGNN(nn.Module):
    def __init__(self, in_features=3, hidden_features=8, out_features=1, n_layers=2, residual=True):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        self.embedding = MVLinear(self.algebra, in_features, hidden_features, subspaces=False)
        self.layers = nn.Sequential(
            *(EGCL(self.algebra, hidden_features, residual=residual) for _ in range(n_layers))
        )
        self.projection = nn.Sequential(MVLinear(self.algebra, hidden_features, out_features))

    def forward(self, h, edges, edge_attr):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges, edge_attr=edge_attr)
        return self.projection(h)


def get_invariants(algebra, values):
    return torch.cat([values[..., :1], *algebra.qs(values, grades=algebra.grades[1:])], dim=-1)


def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


class CGLayer(nn.Module):
    def __init__(
        self,
        algebra,
        in_features_x,
        hidden_features_x,
        out_features_x,
        in_features_h,
        hidden_features_h,
        out_features_h,
        edge_attr_x=3,
        edge_attr_h=0,
        node_attr_x=1,
        node_attr_h=2,
        aggregation="mean",
        use_invariants_to_update=True,
        residual=False,
        normalization_init=None,
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features_x = in_features_x
        self.out_features_x = out_features_x
        self.in_features_h = in_features_h
        self.out_features_h = out_features_h
        self.aggregation = aggregation
        self.use_invariants_to_update = use_invariants_to_update
        self.residual = residual
        invariant_width = out_features_x * algebra.n_subspaces
        self.phi_h = nn.Sequential(
            nn.Linear(
                3 * in_features_h + edge_attr_h + invariant_width, hidden_features_h, bias=False
            ),
            nn.BatchNorm1d(hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
        )
        self.phi_x = nn.Sequential(
            FullyConnectedSteerableGeometricProductLayer(
                algebra,
                edge_attr_x + 3 * in_features_x,
                hidden_features_x,
                normalization_init=normalization_init,
            ),
            MVLayerNorm(algebra, hidden_features_x),
        )
        self.theta_x = nn.Sequential(
            FullyConnectedSteerableGeometricProductLayer(
                algebra,
                node_attr_x + in_features_x + hidden_features_x,
                out_features_x,
                normalization_init=normalization_init,
            ),
            MVLayerNorm(algebra, out_features_x),
        )
        self.theta_h = nn.Sequential(
            nn.Linear(
                node_attr_h
                + algebra.n_subspaces * hidden_features_x
                + in_features_h
                + hidden_features_h,
                hidden_features_h,
            ),
            nn.BatchNorm1d(hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_h),
        )
        self.psi_x = nn.Sequential(
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_x * algebra.n_subspaces),
        )
        self.chi_x = nn.Sequential(
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_x * algebra.n_subspaces),
        )

    def reduce(self, values, rows, nodes):
        if self.aggregation == "mean":
            return unsorted_segment_mean(values, rows, nodes)
        return unsorted_segment_sum(values, rows, nodes)

    def forward(self, h, x, edges, node_attr_h, node_attr_x, edge_attr_h, edge_attr_x):
        i, j = edges
        x_diff = x[i] - x[j]
        geometric = [x[i], x[j], x_diff]
        if edge_attr_x is not None:
            geometric.append(edge_attr_x)
        m_x = self.phi_x(torch.cat(geometric, dim=1))
        m_invariants = get_invariants(self.algebra, m_x).flatten(1)
        scalar = [m_invariants, h[i], h[j], h[i] - h[j]]
        if edge_attr_h is not None:
            scalar.append(edge_attr_h)
        m_h = self.phi_h(torch.cat(scalar, dim=1))
        if self.use_invariants_to_update:
            weights = self.psi_x(m_h).view(len(m_h), self.out_features_x, self.algebra.n_subspaces)
            weights = weights.repeat_interleave(self.algebra.subspaces, dim=2)
            m_x = m_x * torch.sigmoid(weights)
        x_red = self.reduce(m_x.flatten(1), i, len(x)).view(len(x), *m_x.shape[1:])
        h_red = self.reduce(m_h, i, len(h))
        x_inputs = [x, x_red]
        if node_attr_x is not None:
            x_inputs.append(node_attr_x)
        x_u = self.theta_x(torch.cat(x_inputs, dim=1))
        u_invariants = get_invariants(self.algebra, x).flatten(1)
        h_inputs = [h, h_red, u_invariants]
        if node_attr_h is not None:
            h_inputs.append(node_attr_h)
        h_u = self.theta_h(torch.cat(h_inputs, dim=1))
        if self.use_invariants_to_update:
            weights = self.chi_x(h_u).view(len(h_u), self.out_features_x, self.algebra.n_subspaces)
            weights = weights.repeat_interleave(self.algebra.subspaces, dim=2)
            x_u = x_u * torch.sigmoid(weights)
        h = h + h_u if self.residual and self.in_features_h == self.out_features_h else h_u
        x = x + x_u if self.residual and self.in_features_x == self.out_features_x else x_u
        return h, x


class LorentzCGGNN(nn.Module):
    def __init__(
        self,
        in_features_h=2,
        hidden_features_h=8,
        in_features_x=1,
        hidden_features_x=4,
        decoder_features=8,
        n_class=2,
        n_layers=2,
        dropout=0.2,
        use_invariants_to_update=True,
        residual=False,
        aggregation="mean",
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
        self.hidden_features_h = hidden_features_h
        self.hidden_features_x = hidden_features_x
        self.n_layers = n_layers
        self.embedding_h = nn.Linear(in_features_h, hidden_features_h)
        self.embedding_x = MVLinear(self.algebra, in_features_x, hidden_features_x, subspaces=False)
        self.CGLs = nn.ModuleList(
            CGLayer(
                self.algebra,
                hidden_features_x,
                hidden_features_x,
                hidden_features_x,
                hidden_features_h,
                hidden_features_h,
                hidden_features_h,
                use_invariants_to_update=use_invariants_to_update,
                residual=residual,
                aggregation=aggregation,
            )
            for _ in range(n_layers)
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(
                hidden_features_h + hidden_features_x * self.algebra.n_subspaces, decoder_features
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_features, n_class),
        )

    def forward(self, h, x, edge_attr_x, node_attr_x, node_attr_h, edges, node_mask, n_nodes):
        h = self.embedding_h(h)
        x = self.embedding_x(x)
        for layer in self.CGLs:
            h, x = layer(h, x, edges, node_attr_h, node_attr_x, None, edge_attr_x)
        invariants = get_invariants(self.algebra, x).flatten(1)
        h = torch.cat([h, invariants], dim=1)
        h = (h * node_mask).view(-1, n_nodes, h.shape[-1]).mean(dim=1)
        return self.graph_dec(h).squeeze(1)
