import pytest
import torch
from torch_geometric.data import Data, Batch
from csgnn.model.csgnn import CSGNN

def test_csgnn_forward():
    model = CSGNN(num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3)

    # Create a dummy batch
    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 5)
    batch = torch.zeros(100, dtype=torch.long)
    y = torch.randn(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=y)
    batch = Batch.from_data_list([data])

    # Test forward pass
    output = model(batch)
    assert output.shape == torch.Size([1])

def test_csgnn_encode():
    model = CSGNN(num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3)

    # Create a dummy batch
    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 5)
    batch = torch.zeros(100, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    batch = Batch.from_data_list([data])

    # Test encode method
    encoding = model.encode(batch)
    assert encoding.shape == (1, 32)  # (batch_size, hidden_channels)

def test_csgnn_training_step():
    model = CSGNN(num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3)
    # Temporarily disable logging
    model.log = lambda *args, **kwargs: None

    # Create a dummy batch
    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 5)
    batch = torch.zeros(100, dtype=torch.long)
    y = torch.randn(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=y)
    batch = Batch.from_data_list([data])

    # Test training step
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # scalar
