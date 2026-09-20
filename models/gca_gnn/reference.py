"""Qualcomm GATr GCA-GNN semantics using CliffordLayers and PyG."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from models.gca_mlp.reference import ReferenceGCAMLP


class ReferenceGCAGNNLayer(MessagePassing):
    """Add messages from source nodes to targets, then update each target."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        message_channels: int,
        mlp_hidden_channels: int,
        mlp_hidden_layers: int,
    ) -> None:
        super().__init__(aggr="add", flow="source_to_target", node_dim=-3)
        shared = dict(hidden_channels=mlp_hidden_channels, hidden_layers=mlp_hidden_layers)
        self.message_mlp = ReferenceGCAMLP(2 * in_channels, message_channels, **shared)
        self.update_mlp = ReferenceGCAMLP(in_channels + message_channels, out_channels, **shared)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.message_mlp(torch.cat([x_i, x_j], dim=1))

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.update_mlp(torch.cat([x, aggr_out], dim=1))


class ReferenceGCAGNN(nn.Module):
    """The pinned GATr sequence of initial, intermediate, and final GCA-GNN layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        node_channels: int,
        message_channels: int,
        mlp_hidden_channels: int,
        mlp_hidden_layers: int,
        message_passing_steps: int,
    ) -> None:
        super().__init__()
        shared = dict(
            message_channels=message_channels,
            mlp_hidden_channels=mlp_hidden_channels,
            mlp_hidden_layers=mlp_hidden_layers,
        )
        self.layers = nn.ModuleList(
            [ReferenceGCAGNNLayer(in_channels, node_channels, **shared)]
            + [
                ReferenceGCAGNNLayer(node_channels, node_channels, **shared)
                for _ in range(message_passing_steps - 2)
            ]
            + [ReferenceGCAGNNLayer(node_channels, out_channels, **shared)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index=edge_index)
        return x
