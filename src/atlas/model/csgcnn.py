from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import CGConv, global_mean_pool


class CSGCNN(pl.LightningModule):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        num_layers: int,
        learning_rate: float = 0.01,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self._dtype = torch.bfloat16

        # Initial node embedding
        self.node_embedding = nn.Linear(
            num_node_features, hidden_channels, dtype=self._dtype
        )

        # CGConv layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(
                CGConv(hidden_channels, num_edge_features, bias=True).to(self._dtype)
            )
            self.batch_norms.append(
                nn.BatchNorm1d(
                    hidden_channels, track_running_stats=False, dtype=self.float32
                )
            )

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, dtype=self._dtype),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1, dtype=self._dtype),
        )

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def forward(self, data, mode="regression"):
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
        graph_embedding = global_mean_pool(x, batch)

        if mode == "encoder":
            return graph_embedding
        elif mode == "regression":
            return self.output_layer(graph_embedding).view(-1)
        else:
            raise ValueError("Invalid mode. Use 'encoder' or 'regression'.")

    def encode(self, data):
        return self.forward(data, mode="encoder")

    def regression(self, data):
        return self.forward(data, mode="regression")

    def custom_loss(self, y_pred, y_true):
        # Your custom loss function
        # mse_loss = F.mse_loss(y_pred, y_true)
        mae_loss = F.l1_loss(y_pred, y_true)
        # huber_loss = F.smooth_l1_loss(y_pred, y_true)
        return mae_loss

    def training_step(self, batch, batch_idx):
        y_hat = self.regression(batch)
        y_true = batch.y.float().view(-1)

        # Calculate custom loss for training
        loss = self.custom_loss(y_hat, y_true)

        # Calculate MAE for comparison
        mae = F.l1_loss(y_hat, y_true)

        # Log both losses
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
        y_hat = self.regression(batch)
        y_true = batch.y.float().view(-1)

        # Calculate custom loss
        loss = self.custom_loss(y_hat, y_true)

        # Calculate MAE for comparison
        mae = F.l1_loss(y_hat, y_true)

        # Log both losses
        self.log(
            "val_loss", loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
        )
        self.log(
            "val_mae", mae, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self.regression(batch)
        y_true = batch.y.float().view(-1)

        # Calculate MAE for final evaluation
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
