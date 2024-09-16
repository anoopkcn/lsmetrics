import os
import json
import torch
import warnings
from torch_geometric.data import Dataset, Data
from pymatgen.core import Structure
from .base_dataloader import AtomCustomJSONInitializer
from .utils import GaussianDistance


class GaussianDistanceGraphDataset(Dataset):
    def __init__(
        self,
        json_file,
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        target_property=None,
        all_neighbors=False,
    ):
        super().__init__()
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.all_neighbors = all_neighbors
        self.ari = AtomCustomJSONInitializer()
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.target_property = target_property
        if not os.path.isfile(json_file):
            raise FileNotFoundError(f"The JSON file '{json_file}' does not exist.")
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def len(self):
        return len(self.data)

    def get(self, idx):
        item = self.data[idx]
        cif_id = item["material_id"]
        structure = Structure.from_dict(item["crystal_structure"])

        # Extract node features (e.g., atomic numbers)
        node_features = torch.stack(
            [self.ari.get_atom_features(atom.specie.number) for atom in structure]
        )

        if self.all_neighbors:
            # Connect all atoms to each other
            num_atoms = len(structure)
            edge_index = []
            edge_attr = []
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        edge_index.append([i, j])
                        distance = structure.get_distance(i, j)
                        edge_attr.append([distance])
        else:
            all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            edge_index, edge_attr = [], []

            for center, nbr in enumerate(all_nbrs):
                if len(nbr) < self.max_num_nbr:
                    warnings.warn(
                        f"{cif_id} not find enough neighbors to build graph. "
                        f"If it happens frequently, consider increase radius."
                    )
                    nbr = nbr + [(structure[center], self.radius + 1.0, center)] * (
                        self.max_num_nbr - len(nbr)
                    )
                else:
                    nbr = nbr[: self.max_num_nbr]

                for neighbor in nbr:
                    # neighbor is a tuple: (Site, distance, index)
                    edge_index.append([center, neighbor[2]])
                    edge_attr.append([neighbor[1]])  # distance

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Apply Gaussian distance expansion
        edge_attr = self.gdf.expand(edge_attr)

        # Create PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        if self.target_property is not None:
            if self.target_property in item:
                target = torch.tensor([item[self.target_property]], dtype=torch.float)
                data.y = target
            else:
                warnings.warn(
                    f"Target property '{self.target_property}' not found for material {cif_id}"
                )
        return data
