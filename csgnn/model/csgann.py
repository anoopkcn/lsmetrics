import torch
import torch.nn as nn
from torch.optim.adam import Adam
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


class CSGANN(pl.LightningModule):
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
        self.node_embedding = nn.Linear(
            num_node_features, hidden_channels, dtype=torch.float32
        )

        # GATConv layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    hidden_channels, hidden_channels, edge_dim=num_edge_features
                ).to(torch.float32)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_channels, dtype=torch.float32))

        self.dropout = nn.Dropout(0.2)

        # Output layers
        self.linear1 = nn.Linear(hidden_channels, hidden_channels, dtype=torch.float32)
        self.linear2 = nn.Linear(hidden_channels, 1, dtype=torch.float32)

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
            x = F.elu(x)
            x = self.layer_norms[i](x)
            x = self.dropout(x)

        # Global pooling
        graph_embedding = global_mean_pool(x, batch)

        if mode == "encoder":
            return graph_embedding
        elif mode == "regression":
            # Final layers for property prediction
            x = self.linear1(graph_embedding)
            x = F.elu(x)
            x = self.dropout(x)
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
        for ln in self.layer_norms:
            for param in ln.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.node_embedding.parameters():
            param.requires_grad = True
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = True
        for ln in self.layer_norms:
            for param in ln.parameters():
                param.requires_grad = True
