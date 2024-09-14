# train.py
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.optim.adam import Adam  # Correct import for Adam
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from csgnn.data.dataloader_json import CrystalStructureDataset
from csgnn.model.csgnn import CSGNN
from csgnn.utils.checkpoint import load_checkpoint

from torch.utils.data import Subset as TorchSubset
from torch_geometric.data import Dataset as PyGDataset

class CustomSubset(TorchSubset, PyGDataset):
    def __init__(self, dataset, indices):
        TorchSubset.__init__(self, dataset, indices)
        PyGDataset.__init__(self)

    def get(self, idx):
        return self.dataset[self.indices[idx]]

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.05
NUM_EPOCHS = 100
HIDDEN_CHANNELS = 128
NUM_LAYERS = 3
CHECKPOINT_DIR = '../checkpoints'
DATAFILE = 'combined_data.json'

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
def main():
    full_dataset = CrystalStructureDataset(DATAFILE, radius=10, target_property='band_gap')

    train_indices, test_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    train_dataset = CustomSubset(full_dataset, train_indices)
    test_dataset = CustomSubset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Print diagnostic information
    print(f"Dataset length: {len(full_dataset)}")
    sample_data = full_dataset.get(0)
    print(f"Sample data: {sample_data}")

    if sample_data is None or not hasattr(sample_data, 'x') or not hasattr(sample_data, 'edge_attr'):
        print("Error: Unable to access required attributes from the dataset.")
        return

    num_node_features = sample_data.x.size(1) if sample_data.x is not None else 0
    num_edge_features = sample_data.edge_attr.size(1) if sample_data.edge_attr is not None else 0

    print(f"Number of node features: {num_node_features}")
    print(f"Number of edge features: {num_edge_features}")

    if num_node_features == 0 or num_edge_features == 0:
        print("Error: Invalid number of features. Check your dataset implementation.")
        return

    model = CSGNN(num_node_features=num_node_features,
                  num_edge_features=num_edge_features,
                  hidden_channels=HIDDEN_CHANNELS,
                  num_layers=NUM_LAYERS)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss'
    )

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='auto',
        devices="auto",  # Use all available GPUs
        strategy='auto',  # Use DistributedDataParallel
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
