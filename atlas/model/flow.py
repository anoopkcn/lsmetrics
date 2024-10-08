import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
import pytorch_lightning as pl
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


class GraphCouplingLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim):
        super(GraphCouplingLayer, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_dim // 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim // 2),
        )

    def forward(self, x, edge_index, edge_attr):
        x1, x2 = torch.chunk(x, 2, dim=-1)

        # Aggregate edge features
        row, col = edge_index
        edge_features = (
            torch.mean(edge_attr, dim=0) if edge_attr.dim() > 1 else edge_attr
        )
        edge_features = edge_features.repeat(x1.size(0), 1)

        # Concatenate node features with edge features
        nn_input = torch.cat([x1, edge_features], dim=-1)

        t = self.nn(nn_input)
        y2 = x2 * torch.exp(t) + t

        return torch.cat([x1, y2], dim=-1), torch.sum(t, dim=-1)


class RealNVPCouplingLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim):
        super(RealNVPCouplingLayer, self).__init__()
        self.in_dim = in_dim
        self.half_dim = in_dim // 2

        self.s_network = nn.Sequential(
            nn.Linear(self.half_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half_dim),
        )

        self.t_network = nn.Sequential(
            nn.Linear(self.half_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half_dim),
        )

    def forward(self, x, edge_index, edge_attr, reverse=False):
        x1, x2 = torch.split(x, [self.half_dim, self.half_dim], dim=-1)

        # Aggregate edge features
        edge_features = (
            torch.mean(edge_attr, dim=0) if edge_attr.dim() > 1 else edge_attr
        )
        edge_features = edge_features.repeat(x1.size(0), 1)

        if not reverse:
            s = self.s_network(torch.cat([x1, edge_features], dim=-1))
            t = self.t_network(torch.cat([x1, edge_features], dim=-1))
            y2 = x2 * torch.exp(s) + t
            log_det = torch.sum(s, dim=-1)
        else:
            s = self.s_network(torch.cat([x1, edge_features], dim=-1))
            t = self.t_network(torch.cat([x1, edge_features], dim=-1))
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -torch.sum(s, dim=-1)

        return torch.cat([x1, y2], dim=-1), log_det


class GlowCouplingLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim):
        super(GlowCouplingLayer, self).__init__()
        self.in_dim = in_dim
        self.half_dim = in_dim // 2

        self.nn = nn.Sequential(
            nn.Linear(self.half_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, self.half_dim * 2
            ),  # Output both scale and translation
        )

    def forward(self, x, edge_index, edge_attr, reverse=False):
        x1, x2 = torch.split(x, [self.half_dim, self.half_dim], dim=-1)

        # Aggregate edge features
        edge_features = (
            torch.mean(edge_attr, dim=0) if edge_attr.dim() > 1 else edge_attr
        )
        edge_features = edge_features.repeat(x1.size(0), 1)

        nn_out = self.nn(torch.cat([x1, edge_features], dim=-1))
        s, t = torch.split(nn_out, [self.half_dim, self.half_dim], dim=-1)

        if not reverse:
            y2 = x2 * torch.exp(s) + t
            log_det = torch.sum(s, dim=-1)
        else:
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -torch.sum(s, dim=-1)

        return torch.cat([x1, y2], dim=-1), log_det


class GNFlow(pl.LightningModule):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels,
        num_layers,
        learning_rate,
        num_flows=3,
    ):
        super(GNFlow, self).__init__()
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.learning_rate = learning_rate

        # Initial node embedding
        self.node_embedding = nn.Linear(
            num_node_features, hidden_channels, dtype=torch.float32
        )
        self.edge_embedding = nn.Linear(
            num_edge_features, num_edge_features, dtype=torch.float32
        )

        # Initial GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(
                CGConv(hidden_channels, num_edge_features, bias=True).to(torch.float32)
            )
            self.batch_norms.append(
                nn.BatchNorm1d(
                    hidden_channels, track_running_stats=False, dtype=torch.float32
                )
            )

        # Flow layers
        self.flows = nn.ModuleList(
            [
                RealNVPCouplingLayer(
                    hidden_channels, hidden_channels, num_edge_features
                )
                for _ in range(num_flows)
            ]
        )

        # Final prediction layer
        self.predict = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial node embedding
        x = self.node_embedding(x)
        # Initial edge embedding
        edge_attr = self.edge_embedding(edge_attr)

        # Initial GNN embedding
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            if x.size(0) > 1:
                x = self.batch_norms[i](x)

        # Apply flow layers
        log_det_sum = 0
        for flow in self.flows:
            x, log_det = flow(x, edge_index, edge_attr, reverse=False)
            log_det_sum += log_det

        # Global pooling
        x = global_mean_pool(x, batch)

        # Predict bandgap
        bandgap = self.predict(x)

        return bandgap.squeeze(), log_det_sum

    # You might also want to add a reverse method for generation tasks
    def reverse(self, z, edge_index, edge_attr):
        for flow in reversed(self.flows):
            z, _ = flow(z, edge_index, edge_attr, reverse=True)
        return z

    def training_step(self, batch, batch_idx):
        y_hat, log_det = self(batch)
        loss = F.mse_loss(y_hat, batch.y) - 0.05 * log_det.mean()
        mae = F.l1_loss(y_hat, batch.y)
        # self.clip_gradients(
        #     optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        # )
        self.log(
            "train_loss",
            loss,
            batch_size=batch.num_graphs,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_mae", mae, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, log_det = self(batch)
        loss = F.mse_loss(y_hat, batch.y) - 0.05 * log_det.mean()
        mae = F.l1_loss(y_hat, batch.y)

        self.log(
            "val_loss", loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
        )
        self.log(
            "val_mae", mae, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
