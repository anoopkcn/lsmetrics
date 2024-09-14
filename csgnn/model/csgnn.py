import torch
import torch.nn as nn
from torch.optim.adam import Adam
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
import pytorch_lightning as pl


class CSGNN(pl.LightningModule):
    def __init__(
        self, num_node_features, num_edge_features, hidden_channels, num_layers
    ):
        super().__init__()
        self.num_layers = num_layers

        # Initial node embedding
        self.node_embedding = nn.Linear(num_node_features, hidden_channels)

        # CGConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(CGConv(hidden_channels, num_edge_features, bias=True))

        # Output layers
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)  # For property prediction

        # Batch normalization layers (optional)
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)]
        )

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
            x = self.batch_norms[i](x)  # Apply batch normalization instead of dropout

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
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y.view(-1))
        self.log("train_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y.view(-1))
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.01)
