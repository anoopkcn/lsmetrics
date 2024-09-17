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
        pretrained_path=None,
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

        if pretrained_path:
            self.load_pretrained(pretrained_path)

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
        self.log(
            "train_loss",
            loss,
            batch_size=batch.num_graphs,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.predict_property(batch)
        loss = F.l1_loss(y_hat, batch.y.view(-1))
        self.log(
            "val_loss", loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
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
        """
        Load pretrained weights from a file.
        """
        if not pretrained_path:
            return

        try:
            # Load the state dict
            state_dict = torch.load(pretrained_path, map_location=self.device)

            # If it's a checkpoint file, extract just the model state dict
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Load the state dict, allowing for missing or unexpected keys
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained model from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained model: {str(e)}")

    def freeze_encoder(self):
        """
        Freeze the encoder part of the model (useful for fine-tuning).
        """
        for param in self.node_embedding.parameters():
            param.requires_grad = False
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False
        for bn in self.batch_norms:
            for param in bn.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        """
        Unfreeze the encoder part of the model.
        """
        for param in self.node_embedding.parameters():
            param.requires_grad = True
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = True
        for bn in self.batch_norms:
            for param in bn.parameters():
                param.requires_grad = True
