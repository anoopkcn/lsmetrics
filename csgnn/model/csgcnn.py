import torch
import torch.nn as nn
from torch.optim.adam import Adam
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


class CSGCNN(pl.LightningModule):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels,
        num_layers,
        learning_rate=0.01,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_layers = num_layers

        # Initial node embedding
        self.node_embedding = nn.Linear(num_node_features, hidden_channels)

        # CGConv layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(CGConv(hidden_channels, num_edge_features, bias=True))
            self.batch_norms.append(
                nn.BatchNorm1d(hidden_channels, track_running_stats=False)
            )

        # Output layers
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)  # For property prediction

    def forward(self, data, mode="regression"):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
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
        graph_embedding = global_mean_pool(x, batch)

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

    def training_step(self, batch, batch_idx):
        y_hat = self.predict_property(batch)
        loss = F.l1_loss(y_hat, batch.y.view(-1))
        self.log("train_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.predict_property(batch)
        loss = F.l1_loss(y_hat, batch.y.view(-1))
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)

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


class CSGANN(pl.LightningModule):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels,
        num_layers,
        learning_rate=0.01,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.node_embedding = nn.Linear(num_node_features, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, edge_dim=num_edge_features)
            )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(0.2)

        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.node_embedding(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
            x = self.layer_norms[i](x)
            # x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return x

    def encode(self, data):
        """
        Encode the input data into a graph embedding.
        """
        return self.forward(data)

    def predict_property(self, data):
        """
        Predict the property using the graph embedding.
        """
        x = self.encode(data)
        x = self.linear1(x)
        x = F.elu(x)
        # x = self.dropout(x)
        x = self.linear2(x).view(-1)
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self.predict_property(batch)
        loss = F.l1_loss(y_hat, batch.y.view(-1))
        self.log("train_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.predict_property(batch)
        loss = F.l1_loss(y_hat, batch.y.view(-1))
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)

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
