"""Graph neural networks for propagation-based misinformation detection.

Graph-level classification of propagation cascades (fake / real):
- ``GCN`` / ``GAT``: standard message-passing baselines.
- ``BiGCN``: bidirectional model reading the cascade top-down (propagation)
  and bottom-up (dispersion), following Bian et al. (2020).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GAT(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1)
        self.lin = Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class BiGCN(torch.nn.Module):
    """Bidirectional GCN: a top-down branch on the propagation edges and a
    bottom-up branch on the reversed edges, pooled and concatenated."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.td1 = GCNConv(in_dim, hidden)
        self.td2 = GCNConv(hidden, hidden)
        self.bu1 = GCNConv(in_dim, hidden)
        self.bu2 = GCNConv(hidden, hidden)
        self.lin = Linear(hidden * 2, num_classes)

    def forward(self, x, edge_index, batch):
        rev = edge_index.flip(0)
        td = F.relu(self.td1(x, edge_index))
        td = F.relu(self.td2(td, edge_index))
        bu = F.relu(self.bu1(x, rev))
        bu = F.relu(self.bu2(bu, rev))
        td = global_mean_pool(td, batch)
        bu = global_mean_pool(bu, batch)
        return self.lin(torch.cat([td, bu], dim=1))


MODELS = {"GCN": GCN, "GAT": GAT, "Bi-GCN": BiGCN}
