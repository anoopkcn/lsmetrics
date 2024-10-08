import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.optim.adam import Adam
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from csgnn.data.edge_features import (
    TruncatedCoulombCalculator,
    RBFCalculator,
    GaussianDistanceCalculator,
    WeightedGaussianDistanceCalculator,
    PeriodicWeightedGaussianCalculator,
    AtomSpecificGaussianCalculator,
    CosineSimilarityCalculator,
    ScreenedCoulombCalculator,
)

from csgnn.data.node_features import atom_custom_json_initializer
from csgnn.data import CrystalStructureGraphDataset
from csgnn.model import get_model, get_available_models
from csgnn.utils.checkpoint import load_checkpoint
from torch.utils.data import Subset as TorchSubset
from torch_geometric.data import Dataset as PyGDataset

torch.set_default_dtype(torch.float32)


class CustomSubset(TorchSubset, PyGDataset):
    def __init__(self, dataset, indices):
        TorchSubset.__init__(self, dataset, indices)
        PyGDataset.__init__(self)

    def get(self, idx):
        return self.dataset[self.indices[idx]]


def main(
    datafile,
    num_epochs=100,
    lr=0.05,
    model_name="CSGCNN",
    resume=None,
    batch_size=32,
    hidden_channels=128,
    num_layers=3,
    checkpoint_dir="checkpoints",
    test_size=0.2,
    random_state=42,
    num_workers=4,
    num_nodes=1,
    node_rank=0,
):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    full_dataset = CrystalStructureGraphDataset(
        datafile,
        # calculators=[RBFCalculator()],
        # atom_initializer=atom_custom_json_initializer,
        radius=8,
        target_property="band_gap",
    )

    train_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=test_size, random_state=random_state
    )
    train_dataset = CustomSubset(full_dataset, train_indices)
    test_dataset = CustomSubset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    # print("\nChecking first batch from val_loader:")
    # try:
    #     first_batch = next(iter(val_loader))
    #     print("First batch type:", type(first_batch))
    #     print("First batch content:", first_batch)
    #     if hasattr(first_batch, "x"):
    #         print("first_batch.x type:", type(first_batch.x))
    #         print("first_batch.x content:", first_batch.x)
    #     else:
    #         print("first_batch has no 'x' attribute")
    # except Exception as e:
    #     print(f"Exception when checking val_loader: {str(e)}")
    #     print(f"Exception type: {type(e)}")
    #     import traceback

    #     traceback.print_exc()

    # Print diagnostic information
    print(f"Dataset length: {len(full_dataset)}")
    sample_data = full_dataset.get(0)
    print(f"Sample data: {sample_data}")

    if (
        sample_data is None
        or not hasattr(sample_data, "x")
        or not hasattr(sample_data, "edge_attr")
    ):
        print("Error: Unable to access required attributes from the dataset.")
        return

    num_node_features = sample_data.x.size(1) if sample_data.x is not None else 0
    num_edge_features = (
        sample_data.edge_attr.size(1) if sample_data.edge_attr is not None else 0
    )

    print(f"Number of node features: {num_node_features}")
    print(f"Number of edge features: {num_edge_features}")

    if num_node_features == 0 or num_edge_features == 0:
        print("Error: Invalid number of features. Check your dataset implementation.")
        return

    model_class = get_model(model_name)

    if resume:
        print(f"Resuming from checkpoint: {resume}")
        model = model_class.load_from_checkpoint(
            resume,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            learning_rate=lr,
        )
    else:
        model = model_class(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            learning_rate=lr,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{model_name}" + "-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices="auto",
        # num_nodes=num_nodes,
        strategy="auto",  # DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume if resume else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSGNN model")
    parser.add_argument("--datafile", type=str, help="Path to the data file")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to the checkpoint file to resume training from",
        default=None,
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument(
        "--model",
        type=str,
        choices=get_available_models(),
        default="CSGCNN",
        help="Model type to use",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--hidden_channels", type=int, default=128, help="Hidden channels"
    )
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="Number of nodes to use"
    )
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node")

    args = parser.parse_args()

    main(
        args.datafile,
        args.num_epochs,
        lr=args.lr,
        model_name=args.model,
        resume=args.resume,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        checkpoint_dir=args.checkpoint_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        num_workers=args.num_workers,
        num_nodes=args.num_nodes,
        node_rank=args.node_rank,
    )

# Example usage: python train.py --datafile path/to/data --num_epochs 200 --lr 0.01 --model CSGCNN --batch_size 64 --hidden_channels 256 --num_layers 4 --checkpoint_dir my_checkpoints --test_size 0.15 --random_state 123 --num_workers 8 --num_nodes 2 --node_rank 0
