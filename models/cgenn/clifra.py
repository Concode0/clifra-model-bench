"""CGENN primitives from Clifra algebra plans and ordinary PyTorch composition."""

import itertools
import math

import torch
from clifra import make_algebra
from torch import nn


class CGStructure:
    """Fixed grade and product data derived once from a Clifra algebra."""

    def __init__(self, algebra):
        self.algebra = algebra
        self.full = algebra.layout()
        self.grades = tuple(range(algebra.n + 1))
        self.grade_layouts = tuple(algebra.layout((grade,)) for grade in self.grades)
        self.grade_indices = tuple(layout.indices_tensor() for layout in self.grade_layouts)
        self.grade_sizes = tuple(layout.dim for layout in self.grade_layouts)
        self.blade_grades = self.full.grade_indices_tensor().long()
        self.dim = self.full.dim

        # The public plan supplies both signs and grade support. No Cayley table is written here.
        product = algebra.plan_product(
            left=self.full, right=self.full, output=self.full, pairwise=True
        )
        multiplication = product(torch.eye(self.dim), torch.eye(self.dim)).detach()
        self.product_basis = multiplication.permute(0, 2, 1).contiguous()  # left, output, right
        norm = algebra.plan_signature_norm_squared(input=self.full)
        self.norm_signs = norm(torch.eye(self.dim)).flatten().detach()

        paths = []
        path_basis = []
        for left_grade, right_grade, output_grade in itertools.product(self.grades, repeat=3):
            left = self.blade_grades == left_grade
            right = self.blade_grades == right_grade
            output = self.blade_grades == output_grade
            restricted = self.product_basis * (
                left[:, None, None] & output[None, :, None] & right[None, None, :]
            )
            if torch.any(restricted):
                paths.append((left_grade, right_grade, output_grade))
                path_basis.append(restricted)
        self.path_triples = tuple(paths)
        self.path_basis = torch.stack(path_basis)


def _invariants(values, blade_grades, norm_signs, n_grades, *, smooth_norm=False):
    signed_squares = values.square() * norm_signs
    forms = torch.stack(
        [signed_squares[..., blade_grades == grade].sum(dim=-1) for grade in range(n_grades)],
        dim=-1,
    )
    if smooth_norm:
        return (forms.square() + 1e-16).pow(0.25)
    return forms


class MVLinear(nn.Module):
    def __init__(self, structure, in_features, out_features, subspaces=True, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.subspaces = subspaces
        width = len(structure.grades)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, width)
            if subspaces
            else torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.empty(1, out_features, 1)) if bias else None
        self.register_buffer("blade_grades", structure.blade_grades.clone(), persistent=False)
        self.dim = structure.dim
        nn.init.normal_(self.weight, std=1 / math.sqrt(in_features))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, values):
        if self.subspaces:
            weight = self.weight.index_select(-1, self.blade_grades)
            result = torch.einsum("bm...i,nmi->bn...i", values, weight)
        else:
            result = torch.einsum("bm...i,nm->bn...i", values, self.weight)
        if self.bias is not None:
            scalar_bias = torch.cat(
                (self.bias, self.bias.new_zeros(1, self.out_features, self.dim - 1)), dim=-1
            )
            while scalar_bias.ndim < result.ndim:
                scalar_bias = scalar_bias.unsqueeze(2)
            result = result + scalar_bias
        return result


class NormalizationLayer(nn.Module):
    def __init__(self, structure, features, init=0):
        super().__init__()
        self.in_features = features
        self.n_grades = len(structure.grades)
        self.a = nn.Parameter(torch.zeros(features, self.n_grades) + init)
        self.register_buffer("blade_grades", structure.blade_grades.clone(), persistent=False)
        self.register_buffer("norm_signs", structure.norm_signs.clone(), persistent=False)

    def forward(self, values):
        norms = _invariants(
            values, self.blade_grades, self.norm_signs, self.n_grades, smooth_norm=True
        )
        scale = torch.sigmoid(self.a) * (norms - 1) + 1
        return values / (scale.index_select(-1, self.blade_grades) + 1e-6)


class MVSiLU(nn.Module):
    def __init__(self, structure, channels, invariant="mag2", exclude_dual=False):
        super().__init__()
        self.channels = channels
        self.invariant = invariant
        self.exclude_dual = exclude_dual
        self.n_grades = len(structure.grades)
        self.a = nn.Parameter(torch.ones(1, channels, self.n_grades))
        self.b = nn.Parameter(torch.zeros(1, channels, self.n_grades))
        self.register_buffer("blade_grades", structure.blade_grades.clone(), persistent=False)
        self.register_buffer("norm_signs", structure.norm_signs.clone(), persistent=False)

    def forward(self, values):
        forms = _invariants(
            values,
            self.blade_grades,
            self.norm_signs,
            self.n_grades,
            smooth_norm=self.invariant == "norm",
        )
        invariants = torch.cat((values[..., :1], forms[..., 1:]), dim=-1)
        a, b = self.a, self.b
        while a.ndim < invariants.ndim:
            a, b = a.unsqueeze(2), b.unsqueeze(2)
        gate = (a * invariants + b).index_select(-1, self.blade_grades)
        return torch.sigmoid(gate) * values


class MVLayerNorm(nn.Module):
    def __init__(self, structure, channels):
        super().__init__()
        self.channels = channels
        self.a = nn.Parameter(torch.ones(1, channels))
        self.register_buffer("norm_signs", structure.norm_signs.clone(), persistent=False)

    def forward(self, values):
        form = (values.square() * self.norm_signs).sum(dim=-1, keepdim=True)
        norm = (form.square() + 1e-16).pow(0.25).mean(dim=1, keepdim=True) + 1e-6
        a = self.a
        while a.ndim < norm.ndim:
            a = a.unsqueeze(2)
        return a * values / norm


class _ProductLayer(nn.Module):
    def __init__(
        self,
        structure,
        in_features,
        out_features,
        *,
        connected,
        include_first_order=True,
        normalization_init=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_first_order = include_first_order
        self.connected = connected
        self.path_triples = structure.path_triples
        self.register_buffer("path_basis", structure.path_basis, persistent=False)
        self.normalization = (
            NormalizationLayer(structure, in_features, normalization_init)
            if normalization_init is not None
            else nn.Identity()
        )
        self.linear_right = MVLinear(structure, in_features, in_features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(structure, in_features, out_features, bias=True)
        shape = (out_features, in_features) if connected else (in_features,)
        self.weight = nn.Parameter(torch.empty(*shape, len(self.path_triples)))
        std = 1 / math.sqrt((in_features if connected else 1) * len(structure.grades))
        nn.init.normal_(self.weight, std=std)

    def forward(self, values):
        right = self.normalization(self.linear_right(values))
        if self.connected:
            # Contract blade products before mixing output channels. This avoids
            # materializing an out×in×blade³ kernel (128 MiB at 32×32×32³).
            path_values = torch.einsum("bni,pijk,bnk->bnpj", values, self.path_basis, right)
            product = torch.einsum("bnpj,mnp->bmj", path_values, self.weight)
        else:
            kernel = torch.einsum("...p,pijk->...ijk", self.weight, self.path_basis)
            product = torch.einsum("bni,nijk,bnk->bnj", values, kernel, right)
        if self.include_first_order:
            return (self.linear_left(values) + product) / math.sqrt(2)
        return product


class SteerableGeometricProductLayer(_ProductLayer):
    def __init__(self, structure, features, include_first_order=True, normalization_init=0):
        super().__init__(
            structure,
            features,
            features,
            connected=False,
            include_first_order=include_first_order,
            normalization_init=normalization_init,
        )


class FullyConnectedSteerableGeometricProductLayer(_ProductLayer):
    def __init__(
        self, structure, in_features, out_features, include_first_order=True, normalization_init=0
    ):
        super().__init__(
            structure,
            in_features,
            out_features,
            connected=True,
            include_first_order=include_first_order,
            normalization_init=normalization_init,
        )


class O3CGMLP(nn.Module):
    def __init__(self, hidden_features=16, num_layers=3, in_features=3, out_features=1):
        super().__init__()
        structure = CGStructure(make_algebra(3, 0))
        net = [
            FullyConnectedSteerableGeometricProductLayer(structure, in_features, hidden_features)
        ]
        net.extend(MVSiLU(structure, hidden_features) for _ in range(num_layers - 1))
        net.append(
            FullyConnectedSteerableGeometricProductLayer(structure, hidden_features, out_features)
        )
        self.net = nn.Sequential(*net)

    def forward(self, values):
        return self.net(values)[:, 0, -1]


class O5CGMLP(nn.Module):
    def __init__(self):
        super().__init__()
        structure = CGStructure(make_algebra(5, 0))
        self.gp = nn.Sequential(
            MVLinear(structure, 2, 8, subspaces=False),
            SteerableGeometricProductLayer(structure, 8),
        )
        self.mlp = nn.Sequential(
            nn.Linear(8, 580), nn.ReLU(), nn.Linear(580, 580), nn.ReLU(), nn.Linear(580, 1)
        )

    def forward(self, values):
        return self.mlp(self.gp(values)[..., 0])


class ConvexHullCGMLP(nn.Module):
    def __init__(self, in_features=16, hidden_features=8, out_features=1, num_layers=4):
        super().__init__()
        structure = CGStructure(make_algebra(5, 0))
        self.net = nn.Sequential(
            MVLinear(structure, in_features, hidden_features, subspaces=False),
            *(
                SteerableGeometricProductLayer(structure, hidden_features)
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
    expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.scatter_add_(0, expanded, data)
    count.scatter_add_(0, expanded, torch.ones_like(data))
    return result / count.clamp(min=1)


class CEMLP(nn.Module):
    def __init__(
        self,
        structure,
        in_features,
        hidden_features,
        out_features,
        n_layers=2,
        normalization_init=0,
    ):
        super().__init__()
        layers = []
        for _ in range(n_layers - 1):
            layers.append(
                nn.Sequential(
                    MVLinear(structure, in_features, hidden_features),
                    MVSiLU(structure, hidden_features),
                    SteerableGeometricProductLayer(
                        structure, hidden_features, normalization_init=normalization_init
                    ),
                    MVLayerNorm(structure, hidden_features),
                )
            )
            in_features = hidden_features
        layers.append(
            nn.Sequential(
                MVLinear(structure, in_features, out_features),
                MVSiLU(structure, out_features),
                SteerableGeometricProductLayer(
                    structure, out_features, normalization_init=normalization_init
                ),
                MVLayerNorm(structure, out_features),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, values):
        for layer in self.layers:
            values = layer(values)
        return values


class EGCL(nn.Module):
    def __init__(
        self, structure, features, edge_attr_features=1, residual=True, normalization_init=0
    ):
        super().__init__()
        self.residual = residual
        self.edge_model = CEMLP(
            structure,
            features + edge_attr_features,
            features,
            features,
            normalization_init=normalization_init,
        )
        self.node_model = CEMLP(
            structure, 2 * features, features, features, normalization_init=normalization_init
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
        structure = CGStructure(make_algebra(3, 0))
        self.embedding = MVLinear(structure, in_features, hidden_features, subspaces=False)
        self.layers = nn.Sequential(
            *(EGCL(structure, hidden_features, residual=residual) for _ in range(n_layers))
        )
        self.projection = nn.Sequential(MVLinear(structure, hidden_features, out_features))

    def forward(self, h, edges, edge_attr):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges, edge_attr=edge_attr)
        return self.projection(h)


def get_invariants(values, blade_grades, norm_signs, n_grades):
    forms = _invariants(values, blade_grades, norm_signs, n_grades)
    return torch.cat((values[..., :1], forms[..., 1:]), dim=-1)


def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


class CGLayer(nn.Module):
    def __init__(
        self,
        structure,
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
        self.in_features_x = in_features_x
        self.out_features_x = out_features_x
        self.in_features_h = in_features_h
        self.out_features_h = out_features_h
        self.aggregation = aggregation
        self.use_invariants_to_update = use_invariants_to_update
        self.residual = residual
        self.n_grades = len(structure.grades)
        self.register_buffer("blade_grades", structure.blade_grades.clone(), persistent=False)
        self.register_buffer("norm_signs", structure.norm_signs.clone(), persistent=False)
        invariant_width = out_features_x * self.n_grades
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
                structure,
                edge_attr_x + 3 * in_features_x,
                hidden_features_x,
                normalization_init=normalization_init,
            ),
            MVLayerNorm(structure, hidden_features_x),
        )
        self.theta_x = nn.Sequential(
            FullyConnectedSteerableGeometricProductLayer(
                structure,
                node_attr_x + in_features_x + hidden_features_x,
                out_features_x,
                normalization_init=normalization_init,
            ),
            MVLayerNorm(structure, out_features_x),
        )
        self.theta_h = nn.Sequential(
            nn.Linear(
                node_attr_h + self.n_grades * hidden_features_x + in_features_h + hidden_features_h,
                hidden_features_h,
            ),
            nn.BatchNorm1d(hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_h),
        )
        self.psi_x = nn.Sequential(
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_x * self.n_grades),
        )
        self.chi_x = nn.Sequential(
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_x * self.n_grades),
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
        m_invariants = get_invariants(
            m_x, self.blade_grades, self.norm_signs, self.n_grades
        ).flatten(1)
        scalar = [m_invariants, h[i], h[j], h[i] - h[j]]
        if edge_attr_h is not None:
            scalar.append(edge_attr_h)
        m_h = self.phi_h(torch.cat(scalar, dim=1))
        if self.use_invariants_to_update:
            weights = self.psi_x(m_h).view(len(m_h), self.out_features_x, self.n_grades)
            weights = weights.index_select(2, self.blade_grades)
            m_x = m_x * torch.sigmoid(weights)
        x_red = self.reduce(m_x.flatten(1), i, len(x)).view(len(x), *m_x.shape[1:])
        h_red = self.reduce(m_h, i, len(h))
        x_inputs = [x, x_red]
        if node_attr_x is not None:
            x_inputs.append(node_attr_x)
        x_u = self.theta_x(torch.cat(x_inputs, dim=1))
        u_invariants = get_invariants(x, self.blade_grades, self.norm_signs, self.n_grades).flatten(
            1
        )
        h_inputs = [h, h_red, u_invariants]
        if node_attr_h is not None:
            h_inputs.append(node_attr_h)
        h_u = self.theta_h(torch.cat(h_inputs, dim=1))
        if self.use_invariants_to_update:
            weights = self.chi_x(h_u).view(len(h_u), self.out_features_x, self.n_grades)
            weights = weights.index_select(2, self.blade_grades)
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
        structure = CGStructure(make_algebra(1, 3))
        self.hidden_features_h = hidden_features_h
        self.hidden_features_x = hidden_features_x
        self.n_layers = n_layers
        self.n_grades = len(structure.grades)
        self.register_buffer("blade_grades", structure.blade_grades.clone(), persistent=False)
        self.register_buffer("norm_signs", structure.norm_signs.clone(), persistent=False)
        self.embedding_h = nn.Linear(in_features_h, hidden_features_h)
        self.embedding_x = MVLinear(structure, in_features_x, hidden_features_x, subspaces=False)
        self.CGLs = nn.ModuleList(
            CGLayer(
                structure,
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
            nn.Linear(hidden_features_h + hidden_features_x * self.n_grades, decoder_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_features, n_class),
        )

    def forward(self, h, x, edge_attr_x, node_attr_x, node_attr_h, edges, node_mask, n_nodes):
        h = self.embedding_h(h)
        x = self.embedding_x(x)
        for layer in self.CGLs:
            h, x = layer(h, x, edges, node_attr_h, node_attr_x, None, edge_attr_x)
        invariants = get_invariants(x, self.blade_grades, self.norm_signs, self.n_grades).flatten(1)
        h = torch.cat([h, invariants], dim=1)
        h = (h * node_mask).view(-1, n_nodes, h.shape[-1]).mean(dim=1)
        return self.graph_dec(h).squeeze(1)
