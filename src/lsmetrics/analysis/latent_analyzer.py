import torch
from typing import Optional, Dict, Callable
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from .metrics import LatentSpaceMetrics
from .visualization import LatentSpaceVisualizer

class LatentSpaceAnalyzer:
    def __init__(
        self,
        model: torch.nn.Module,
        encoder_function: Optional[Callable] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the latent space analyzer.

        Args:
            model: The model to analyze
            encoder_function: Optional function to get embeddings from model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        if encoder_function is None:
            if hasattr(model, 'encode'):
                self.encoder_function = model.encode
            else:
                self.encoder_function = model.forward
        else:
            self.encoder_function = encoder_function

        self.metrics = LatentSpaceMetrics()
        self.visualizer = LatentSpaceVisualizer()

    def get_embeddings(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Get embeddings for data"""
        embeddings = []
        sample_count = 0

        with torch.no_grad():
            for batch in dataloader:
                if not isinstance(batch, (Data, Batch)):
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                z = self.encoder_function(batch)
                embeddings.append(z.cpu())

                sample_count += z.size(0)
                if max_samples and sample_count >= max_samples:
                    break

        return torch.cat(embeddings, dim=0)[:max_samples]

    def analyze_latent_space(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        original_distances: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Perform comprehensive latent space analysis"""
        embeddings = self.get_embeddings(dataloader, max_samples)

        metrics = {}
        metrics.update(self.metrics.local_smoothness(embeddings))
        metrics.update(self.metrics.global_structure(embeddings, original_distances))
        metrics.update(self.metrics.manifold_quality(embeddings))
        metrics.update(self.metrics.isotropy(embeddings))

        # Sample some points for interpolation
        z1, z2 = embeddings[0:1], embeddings[1:2]
        metrics.update(self.metrics.interpolation_smoothness(z1, z2))

        return metrics

    def visualize_latent_space(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        save_path: Optional[str] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Create visualization of latent space analysis"""
        embeddings = self.get_embeddings(dataloader, max_samples)
        metrics = self.analyze_latent_space(dataloader, max_samples)

        return self.visualizer.create_summary_plot(
            embeddings,
            metrics,
            labels,
            save_path
        )


# training script
# analyzer = LatentSpaceAnalyzer(model, encoder_function=model.encode)

# # Create callback
# class LatentSpaceAnalysisCallback(pl.Callback):
#     def __init__(self, analyzer, dataloader, frequency=10):
#         self.analyzer = analyzer
#         self.dataloader = dataloader
#         self.frequency = frequency

#     def on_validation_epoch_end(self, trainer, pl_module):
#         if trainer.current_epoch % self.frequency == 0:
#             metrics = self.analyzer.analyze_latent_space(self.dataloader)
#             self.analyzer.visualize_latent_space(
#                 self.dataloader,
#                 save_path=f'latent_space_epoch_{trainer.current_epoch}.png'
#             )
