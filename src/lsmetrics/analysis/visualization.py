import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, Optional
from matplotlib.axes import Axes

class LatentSpaceVisualizer:
    @staticmethod
    def plot_tsne(
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ax: Optional[Axes] = None
    ):

        """Create t-SNE visualization of embeddings"""
        if ax is None:
            _, ax = plt.subplots()

        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(embeddings.numpy())

        if labels is not None:
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap='viridis'
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        ax.set_title('t-SNE Visualization')

        return ax

    @staticmethod
    def plot_distance_distribution(
        embeddings: torch.Tensor,
        ax: Optional[Axes] = None
    ):
        """Plot distribution of pairwise distances"""
        if ax is None:
            _, ax = plt.subplots()

        distances = torch.cdist(embeddings, embeddings)
        sns.histplot(distances.flatten().numpy(), ax=ax)
        ax.set_title('Pairwise Distances Distribution')

        return ax

    @staticmethod
    def plot_metrics_summary(
        metrics: Dict[str, float],
        ax: Optional[Axes] = None
    ):
        """Create text summary of metrics"""
        if ax is None:
            _, ax = plt.subplots()

        ax.axis('off')
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        ax.text(0.1, 0.9, metrics_text, fontsize=10)
        ax.set_title('Metrics Summary')

        return ax

    @staticmethod
    def create_summary_plot(
        embeddings: torch.Tensor,
        metrics: Dict[str, float],
        labels: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ):
        """Create comprehensive visualization of latent space analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # t-SNE visualization
        LatentSpaceVisualizer.plot_tsne(embeddings, labels, axes[0,0])

        # Local smoothness distribution
        if 'local_smoothness_mean' in metrics:
            # Convert float to list/array if it's a single value
            data = [metrics['local_smoothness_mean']] if isinstance(metrics['local_smoothness_mean'], float) else metrics['local_smoothness_mean']
            data_array = np.array(data).flatten()
            sns.histplot(data=data_array, ax=axes[0,1])
            axes[0,1].set_title('Local Smoothness Distribution')

        # Pairwise distances
        LatentSpaceVisualizer.plot_distance_distribution(embeddings, axes[1,0])

        # Metrics summary
        LatentSpaceVisualizer.plot_metrics_summary(metrics, axes[1,1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()

        return fig, axes
