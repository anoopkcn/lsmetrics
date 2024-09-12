# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from csgnn.data.dataloader_json import CrystalStructureDataset
from csgnn.model.csgnn import CSGNN
import os
import argparse
from csgnn.utils.checkpoint import load_checkpoint

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 100
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
CHECKPOINT_DIR = '../checkpoints'
DATAFILE = 'path/to/your/json_file.json'

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Train CSGNN model')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    args = parser.parse_args()

    # Load dataset
    dataset = CrystalStructureDataset(DATAFILE, target_property='band_gap')
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get the dimensions from the first item in the dataset
    first_data = dataset[0]  # This returns a Data object
    num_node_features = first_data.x.size(1)
    num_edge_features = first_data.edge_attr.size(1)

    # Initialize model
    model = CSGNN(num_node_features=num_node_features,
                  num_edge_features=num_edge_features,
                  hidden_channels=HIDDEN_CHANNELS,
                  num_layers=NUM_LAYERS)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load checkpoint if resuming training
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model.predict_property(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}')

        # Save checkpoint
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

    print('Training completed.')

if __name__ == '__main__':
    main()
