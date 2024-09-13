# train.py

import torch
import torch.nn as nn
from torch.optim.adam import Adam  # Correct import for Adam
from torch_geometric.loader import DataLoader
from csgnn.data.dataloader_json import CrystalStructureDataset
from csgnn.model.csgnn import CSGNN
import os
import argparse
from csgnn.utils.checkpoint import load_checkpoint
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
    parser = argparse.ArgumentParser(description='Train CSGNN model')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    args = parser.parse_args()

    print("Loading dataset...")
    full_dataset = CrystalStructureDataset(DATAFILE, radius=10, target_property='band_gap')

    # Split the dataset into training and testing
    train_indices, test_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get a sample to determine feature dimensions
    sample_data = full_dataset[0]

    # Initialize model
    model = CSGNN(num_node_features=sample_data.x.size(1),
                  num_edge_features=sample_data.edge_attr.size(1),
                  hidden_channels=HIDDEN_CHANNELS,
                  num_layers=NUM_LAYERS)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Load checkpoint if resuming training
    start_epoch = 100
    end_epoch = start_epoch+NUM_EPOCHS
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(start_epoch,end_epoch):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{end_epoch}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model.predict_property(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{end_epoch}, Training Loss: {avg_loss:.4f}')

        # Validation on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model.predict_property(batch)
                loss = criterion(out, batch.y)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f'Epoch {epoch+1}/{end_epoch}, Test Loss: {avg_test_loss:.4f}')

        # Save checkpoint
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'test_loss': avg_test_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

    print('Training completed.')

if __name__ == '__main__':
    main()
