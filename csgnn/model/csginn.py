# csgnn/model/csgin.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool
import pytorch_lightning as pl
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch_geometric.typing import OptTensor


class GINConvWithEdgeFeatures(MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False):
        super(GINConvWithEdgeFeatures, self).__init__(aggr="add")
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.nn((1 + self.eps) * x + out)

    def message(self, x_j: Tensor, edge_attr: OptTensor = None) -> Tensor:
        if edge_attr is None or edge_attr.shape[0] != x_j.shape[0]:
            return x_j
        return torch.cat([x_j, edge_attr], dim=1)


class GINLayerWithEdgeFeatures(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim):
        super(GINLayerWithEdgeFeatures, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINConvWithEdgeFeatures(self.mlp, train_eps=True)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class CSGINN(pl.LightningModule):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels,
        num_layers,
        learning_rate=0.01,
        pretrained_path=None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_layers = num_layers

        # Initial node embedding
        self.node_embedding = nn.Linear(num_node_features, hidden_channels)

        # Edge embedding
        self.edge_embedding = nn.Linear(num_edge_features, hidden_channels)

        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GINLayerWithEdgeFeatures(
                    hidden_channels, hidden_channels, hidden_channels
                )
            )
            self.batch_norms.append(
                nn.BatchNorm1d(hidden_channels, track_running_stats=False)
            )

        # Output layers
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)  # For property prediction

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def forward(self, data, mode="regression"):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial node and edge embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            if x.size(0) > 1:
                x = self.batch_norms[i](x)

        # Global pooling
        graph_embedding = global_add_pool(x, batch)

        if mode == "encoder":
            return graph_embedding
        elif mode == "regression":
            # Final layers for property prediction
            x = self.linear1(graph_embedding)
            x = F.relu(x)
            property_prediction = self.linear2(x).view(-1)
            return property_prediction
        else:
            raise ValueError("Invalid mode. Use 'encoder' or 'regression'.")

    def encode(self, data):
        return self.forward(data, mode="encoder")

    def predict_property(self, data):
        return self.forward(data, mode="regression")

    def custom_loss(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true)
        l1_loss = F.l1_loss(y_pred, y_true)
        huber_loss = F.smooth_l1_loss(y_pred, y_true)
        return 0.4 * mse_loss + 0.4 * l1_loss + 0.2 * huber_loss

    def training_step(self, batch, batch_idx):
        y_hat = self.predict_property(batch)
        y_true = batch.y.float().view(-1)

        loss = self.custom_loss(y_hat, y_true)
        mae = F.l1_loss(y_hat, y_true)

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
        y_hat = self.predict_property(batch)
        y_true = batch.y.float().view(-1)

        loss = self.custom_loss(y_hat, y_true)
        mae = F.l1_loss(y_hat, y_true)

        self.log(
            "val_loss", loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
        )
        self.log(
            "val_mae", mae, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        y_hat = self.predict_property(batch)
        y_true = batch.y.float().view(-1)

        mae = F.l1_loss(y_hat, y_true)

        self.log(
            "test_mae", mae, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
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
