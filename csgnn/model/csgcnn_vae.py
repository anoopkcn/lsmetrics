import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pytorch_lightning as pl
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from typing import Optional


class CSGCNN_VAE(pl.LightningModule):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        num_layers: int,
        latent_dim: int = 64,
        learning_rate: float = 0.01,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim

        # Encoder
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            in_channels = num_node_features if i == 0 else hidden_channels
            self.conv_layers.append(GCNConv(in_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.fc_mu = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_channels)
        self.conv_decode = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_decode.append(GCNConv(hidden_channels, hidden_channels))
        self.fc_output = nn.Linear(hidden_channels, num_node_features)

        # Property Predictor
        self.fc_predict = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),  # Assuming single property prediction
        )

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def encode(self, x, edge_index, batch):
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = F.relu(bn(conv(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, edge_index, batch):
        x = self.fc_decode(z)
        x = x[batch]  # Map back to graph structure
        for conv in self.conv_decode:
            x = F.relu(conv(x, edge_index))
        return self.fc_output(x)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, edge_index, batch)
        pred_property = self.fc_predict(z).squeeze(-1)
        return recon_x, pred_property, mu, logvar

    def custom_loss(
        self, recon_x, x, pred_property, true_property, mu, logvar, alpha=0.05, beta=1.0
    ):
        BCE = F.mse_loss(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        PROP_MSE = F.mse_loss(pred_property, true_property)
        return BCE + alpha * KLD + beta * PROP_MSE

    def training_step(self, batch, batch_idx):
        recon_x, pred_property, mu, logvar = self(batch)
        loss = self.custom_loss(recon_x, batch.x, pred_property, batch.y, mu, logvar)

        mae = F.l1_loss(pred_property, batch.y)

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
        recon_x, pred_property, mu, logvar = self(batch)
        loss = self.custom_loss(recon_x, batch.x, pred_property, batch.y, mu, logvar)

        mae = F.l1_loss(pred_property, batch.y)

        self.log(
            "val_loss", loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
        )
        self.log(
            "val_mae", mae, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        _, pred_property, _, _ = self(batch)
        mae = F.l1_loss(pred_property, batch.y)
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
        for layer in self.conv_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.fc_mu.parameters():
            param.requires_grad = False
        for param in self.fc_logvar.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for layer in self.conv_layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.fc_mu.parameters():
            param.requires_grad = True
        for param in self.fc_logvar.parameters():
            param.requires_grad = True
