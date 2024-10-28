from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
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
        contrastive_weight: float = 0.5,
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
                    hidden_channels, track_running_stats=False, dtype=torch.float32
                )
            )

        # Output layers
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, dtype=self._dtype),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1, dtype=self._dtype),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, dtype=self._dtype),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels, dtype=self._dtype),
        )

        self.contrastive_weight = contrastive_weight

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def forward(self, data, mode="encoder"):
        x, edge_index, edge_attr, batch = (
            data.x.to(torch.bfloat16),
            data.edge_index,
            data.edge_attr.to(torch.bfloat16),
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
        elif mode == "projection":
            return self.projection_head(graph_embedding)
        elif mode == "regression":
            return self.regression_head(graph_embedding).view(-1)
        else:
            raise ValueError(
                "Invalid mode. Use 'encoder', 'projection', or 'regression'."
            )

    def encode(self, data):
        return self.forward(data, mode="encoder")

    def projection(self, data):
        return self.forward(data, mode="projection")

    def regression(self, data):
        return self.forward(data, mode="regression")

    def contrastive_loss(self, z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / temperature
        sim_i_j = torch.diag(sim, N)
        sim_j_i = torch.diag(sim, -N)
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 2)
        negative_samples = sim[
            torch.logical_not(torch.eye(2 * N, dtype=torch.bool))
        ].reshape(2 * N, -1)
        labels = torch.zeros(N, dtype=torch.long).to(positive_samples.device)
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        loss = F.cross_entropy(logits, labels)
        return loss

    def custom_loss(self, y_pred, y_true):
        # Your custom loss function
        # mse_loss = F.mse_loss(y_pred, y_true)
        mae_loss = F.l1_loss(y_pred, y_true)
        # huber_loss = F.smooth_l1_loss(y_pred, y_true)
        return mae_loss

    def training_step(self, batch, batch_idx):
        y_hat = self.regression(batch)
        y_true = batch.y.view(-1)

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
        y_true = batch.y.view(-1)

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
        y_true = batch.y.view(-1)

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

    def from_pretrained(self, ckpt_path, remove_head: bool = True):
        """
        Load pretrained weights from a file.
        """
        if not ckpt_path:
            return

        try:
            # Load the state dict
            state_dict = torch.load(ckpt_path, weights_only=True)

            # remove the output layer if needed
            if remove_head:
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("output_layer")
                }

            # If it's a checkpoint file, extract just the model state dict
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Load the state dict, allowing for missing or unexpected keys
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained model from {ckpt_path}")
        except Exception as e:
            logger.info(f"Error loading pretrained model: {e!s}")

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
