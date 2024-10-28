import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from lsmetrics.analysis import LatentSpaceMetrics
from torch_geometric.nn.conv.rgcn_conv import torch_sparse

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset as TorchSubset
    from torch_geometric.data import Dataset as PyGDataset

    from lsmetrics.data.dataloader import CrystalStructureGraphDataset
    from lsmetrics.model.csgcnn import CSGCNN

    class CustomSubset(TorchSubset, PyGDataset):
        def __init__(self, dataset, indices):
            TorchSubset.__init__(self, dataset, indices)
            PyGDataset.__init__(self)

        def get(self, idx):
            return self.dataset[self.indices[idx]]

    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pretrin_model_path = "pretrained_crystal_structure.ckpt"
    data_file = "inorganic_materials_small.json"
    full_dataset = CrystalStructureGraphDataset(
        data_file,
        radius=8,
        target_property="band_gap",
    )

    train_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=0.2, random_state=42
    )
    train_dataset = CustomSubset(full_dataset, train_indices)
    test_dataset = CustomSubset(full_dataset, test_indices)

    # get node and edge feature dimensions
    sample_data = full_dataset.get(0)
    num_node_features = sample_data.x.size(1) if sample_data.x is not None else 0
    num_edge_features = (
        sample_data.edge_attr.size(1) if sample_data.edge_attr is not None else 0
    )

    model = CSGCNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_channels=128,
        num_layers=6,
        learning_rate=0.05,
    )
    model = model.to(device)

    model.from_pretrained(pretrin_model_path)

    metrics = LatentSpaceMetrics()
    embeddings = []

    with torch.no_grad():
        for batch in test_dataset:
            # Get embeddings using the encoder mode
            batch = batch.to(str(device))
            emb = model(batch)
            embeddings.append(emb.float().cpu())

    # Concatenate all embeddings and labels
    embeddings = torch.vstack(embeddings)

    eee_metrics = metrics.eee(embeddings)
    print("EEE metrics:", eee_metrics)

    # Calculate VRM
    vrm_metrics = metrics.vrm(embeddings)
    print("VRM metrics:", vrm_metrics)
