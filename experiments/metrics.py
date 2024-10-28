import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def visualize_embeddings(
    encoder, dataloader, batch_size=32, perplexity=30, n_components=2
):
    # Collect embeddings and labels
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Get embeddings using the encoder mode
            emb = encoder(batch)
            embeddings.append(emb.float().cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    # Concatenate all embeddings and labels
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)

    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Create the visualization
    plt.figure(figsize=(10, 8))

    if n_components == 2:
        scatter = plt.scatter(
            embeddings_tsne[:, 0],
            embeddings_tsne[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Band Gap (eV)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
    else:  # 3D plot
        ax = plt.axes(projection="3d")
        scatter = ax.scatter(
            embeddings_tsne[:, 0],
            embeddings_tsne[:, 1],
            embeddings_tsne[:, 2],
            c=labels,
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Band Gap (eV)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        # ax.set_zlabel('t-SNE 3')

    plt.title("t-SNE Visualization of Crystal Structure Embeddings")
    plt.tight_layout()
    plt.savefig("embedding_tsne.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Calculate and print some metrics about the embedding space
    print("\nEmbedding Space Analysis:")
    print(f"Number of samples: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Calculate average distance between points
    from scipy.spatial.distance import pdist

    distances = pdist(embeddings)
    print(f"Average distance between points: {np.mean(distances):.4f}")
    print(f"Standard deviation of distances: {np.std(distances):.4f}")

    # Calculate correlation between embedding distances and label differences
    label_diffs = pdist(labels.reshape(-1, 1))
    correlation = np.corrcoef(distances, label_diffs)[0, 1]
    print(f"Correlation between distances and label differences: {correlation:.4f}")


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
    model.from_pretrained(pretrin_model_path)
    # print(encoder)

    visualize_embeddings(model, test_dataset)
