import pytest
import torch
from torch_geometric.data import Data, Batch
from csgnn.model.csgnn import CSGCNN, CSGANN
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any


def test_csgcnn_initialization():
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
    assert isinstance(model, CSGCNN)
    assert len(model.convs) == 3
    assert len(model.batch_norms) == 3


def test_csgcnn_forward():
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )

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


def test_csgcnn_encode():
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )

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


def test_csgcnn_predict_property():
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )

    # Create a dummy batch
    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 5)
    batch = torch.zeros(100, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    batch = Batch.from_data_list([data])

    # Test predict_property method
    prediction = model.predict_property(batch)
    assert prediction.shape == torch.Size([1])  # (batch_size,)


def test_csgcnn_training_step():
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
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


def test_csgcnn_validation_step():
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
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

    # Test validation step
    model.validation_step(batch, 0)
    # No assertion needed as validation_step doesn't return anything


def test_csgcnn_configure_optimizers():
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )

    config = model.configure_optimizers()
    assert isinstance(config, dict)

    assert "optimizer" in config
    assert isinstance(config["optimizer"], Adam)

    assert "lr_scheduler" in config
    assert isinstance(config["lr_scheduler"], dict)

    lr_scheduler_config = config["lr_scheduler"]
    assert "scheduler" in lr_scheduler_config
    assert isinstance(lr_scheduler_config["scheduler"], ReduceLROnPlateau)

    # Check for optional keys without assuming they exist
    if "monitor" in lr_scheduler_config:
        assert lr_scheduler_config["monitor"] == "val_loss"
    if "interval" in lr_scheduler_config:
        assert lr_scheduler_config["interval"] == "epoch"
    if "frequency" in lr_scheduler_config:
        assert lr_scheduler_config["frequency"] == 1


def test_csgann_forward():
    model = CSGANN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )

    # Create a dummy batch
    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 5)
    batch = torch.zeros(100, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    batch = Batch.from_data_list([data])

    # Test forward pass
    output = model(batch)
    assert output.shape == (1, 32)  # (batch_size, hidden_channels)


def test_csgann_encode():
    model = CSGANN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )

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


def test_csgann_predict_property():
    model = CSGANN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )

    # Create a dummy batch
    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 5)
    batch = torch.zeros(100, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    batch = Batch.from_data_list([data])

    # Test predict_property method
    prediction = model.predict_property(batch)
    assert prediction.shape == torch.Size([1])  # (batch_size,)


def test_csgann_training_step():
    model = CSGANN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
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


def test_csgann_validation_step():
    model = CSGANN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
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

    # Test validation step
    model.validation_step(batch, 0)
    # No assertion needed as validation_step doesn't return anything


def test_csgann_configure_optimizers():
    model = CSGANN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )

    config = model.configure_optimizers()
    assert isinstance(config, dict)

    assert "optimizer" in config
    assert isinstance(config["optimizer"], Adam)

    assert "lr_scheduler" in config
    assert isinstance(config["lr_scheduler"], dict)

    lr_scheduler_config = config["lr_scheduler"]
    assert "scheduler" in lr_scheduler_config
    assert isinstance(lr_scheduler_config["scheduler"], ReduceLROnPlateau)

    # Check for optional keys without assuming they exist
    if "monitor" in lr_scheduler_config:
        assert lr_scheduler_config["monitor"] == "val_loss"
    if "interval" in lr_scheduler_config:
        assert lr_scheduler_config["interval"] == "epoch"
    if "frequency" in lr_scheduler_config:
        assert lr_scheduler_config["frequency"] == 1


@pytest.mark.parametrize("num_nodes", [1, 10, 1000])
def test_csgcnn_different_graph_sizes(num_nodes):
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
    x = torch.randn(num_nodes, 10)
    edge_index = torch.randint(0, num_nodes, (2, max(1, num_nodes - 1)))
    edge_attr = torch.randn(max(1, num_nodes - 1), 5)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    batch = Batch.from_data_list([data])
    output = model(batch)
    assert output.shape == torch.Size([1])


@pytest.mark.parametrize("hidden_channels,num_layers", [(16, 2), (64, 4), (128, 5)])
def test_csgcnn_different_hyperparameters(hidden_channels, num_layers):
    model = CSGCNN(
        num_node_features=10,
        num_edge_features=5,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
    )
    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 5)
    batch = torch.zeros(100, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    batch = Batch.from_data_list([data])

    output = model(batch)
    assert output.shape == torch.Size([1])


def test_csgcnn_no_edges():
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
    x = torch.randn(10, 10)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 5))
    batch = torch.zeros(10, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    batch = Batch.from_data_list([data])
    output = model(batch)
    assert output.shape == torch.Size([1])


def test_csgcnn_save_load(tmp_path):
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 5)
    batch = torch.zeros(100, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    batch = Batch.from_data_list([data])

    original_output = model(batch)

    # Save the model
    torch.save(model.state_dict(), tmp_path / "model.pt")

    # Load the model
    loaded_model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
    loaded_model.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    loaded_output = loaded_model(batch)

    assert torch.allclose(original_output, loaded_output)


def test_csgcnn_gradient_flow():
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
    x = torch.randn(100, 10, requires_grad=True)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 5)
    batch = torch.zeros(100, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    batch = Batch.from_data_list([data])

    output = model(batch)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert all(param.grad is not None for param in model.parameters())


def test_csgcnn_numerical_stability():
    model = CSGCNN(
        num_node_features=10, num_edge_features=5, hidden_channels=32, num_layers=3
    )
    x = torch.randn(100, 10) * 1e6  # Very large values
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 5) * 1e-6  # Very small values
    batch = torch.zeros(100, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    batch = Batch.from_data_list([data])

    output = model(batch)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
