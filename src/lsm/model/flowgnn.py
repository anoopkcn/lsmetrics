import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
import pytorch_lightning as pl
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from typing import Optional


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))
        self.u = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        wx = torch.sum(self.w * x, dim=-1, keepdim=True) + self.b
        zwx = torch.tanh(wx)
        u = self.u + (
            (-1 + F.softplus(self.w @ self.u) + self.w @ self.u)
            * self.w
            / torch.sum(self.w**2)
        )
        z = x + u * zwx
        log_det = torch.log(torch.abs(1 + u @ self.w * (1 - zwx**2)))
        return z, log_det


class FlowGNN(pl.LightningModule):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        num_layers: int,
        num_flow_layers: int = 4,
        learning_rate: float = 0.01,
        pretrained_path: Optional[str] = None,
        flow_weight: float = 0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_flow_layers = num_flow_layers
        self.learning_rate = learning_rate
        self.flow_weight = flow_weight

        # Initial node embedding
        self.node_embedding = nn.Linear(
            num_node_features, hidden_channels, dtype=torch.float32
        )

        # CGConv layers
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
            [PlanarFlow(hidden_channels) for _ in range(self.num_flow_layers)]
        )

        # Regression head
        self.regression_head = nn.Linear(hidden_channels, 1)

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x.float(),
            data.edge_index,
            data.edge_attr.float(),
            data.batch,
        )

        # Initial node embedding
        x = self.node_embedding(x)

        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            if x.size(0) > 1:
                x = self.batch_norms[i](x)

        # Global pooling
        z = global_mean_pool(x, batch)

        # Apply flow layers
        log_det_sum = 0
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det

        # Regression
        prediction = self.regression_head(z).squeeze(-1)

        return z, log_det_sum, prediction

    def encode(self, data):
        z, _, _ = self(data)
        return z

    def flow(self, data):
        z, log_det_sum, _ = self(data)
        return z, log_det_sum

    def inverse(self, z):
        for flow in reversed(self.flows):
            z = self.inverse_flow(z, flow)
        return z

    @staticmethod
    def inverse_flow(z, flow):
        x = z.clone()
        for _ in range(100):  # Max iterations
            fx, _ = flow(x)
            if torch.allclose(fx, z, atol=1e-6):
                return x
            x = x - (fx - z)
        return x

    def training_step(self, batch, batch_idx):
        z, log_det, prediction = self(batch)

        # Flow loss
        prior_ll = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.size(1) * torch.log(
            torch.tensor(2 * torch.pi)
        )
        flow_loss = -(prior_ll + log_det).mean()

        # Regression loss
        regression_loss = F.l1_loss(prediction, batch.y)

        # Combined loss
        loss = regression_loss + self.flow_weight * flow_loss

        self.log(
            "train_loss",
            loss,
            batch_size=batch.num_graphs,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_regression_loss",
            regression_loss,
            batch_size=batch.num_graphs,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_flow_loss",
            flow_loss,
            batch_size=batch.num_graphs,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        z, log_det, prediction = self(batch)

        prior_ll = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.size(1) * torch.log(
            torch.tensor(2 * torch.pi)
        )
        flow_loss = -(prior_ll + log_det).mean()
        regression_loss = F.l1_loss(prediction, batch.y)
        loss = regression_loss + self.flow_weight * flow_loss

        self.log(
            "val_loss", loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
        )
        self.log(
            "val_regression_loss",
            regression_loss,
            batch_size=batch.num_graphs,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_flow_loss",
            flow_loss,
            batch_size=batch.num_graphs,
            prog_bar=True,
            sync_dist=True,
        )

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

    def load_pretrained(self, pretrained_path):
        if not pretrained_path:
            return

        try:
            state_dict = torch.load(pretrained_path, map_location=self.device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained model from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained model: {str(e)}")

    def freeze_encoder(self):
        for param in self.node_embedding.parameters():
            param.requires_grad = False
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False
        for bn in self.batch_norms:
            for param in bn.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.node_embedding.parameters():
            param.requires_grad = True
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = True
        for bn in self.batch_norms:
            for param in bn.parameters():
                param.requires_grad = True
