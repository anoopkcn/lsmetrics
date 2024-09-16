import torch
from torch_geometric.data import Dataset, Data
from pymatgen.core import Structure
from .base_dataloader import AtomCustomJSONInitializer
import os
import json
import warnings


class SimpleCoulombGraphDataset(Dataset):
    """
    Simplified Coulomb-like potential based on atomic numbers(instead of using EwaldSummation).
    We'll use the atomic numbers as a proxy for charge, which isn't entirely accurate ...
    ...but can serve as a reasonable approximation for our graph representation.
    """

    def __init__(
        self,
        json_file,
        radius=10,
        target_property=None,
    ):
        super().__init__()
        self.radius = radius
        self.ari = AtomCustomJSONInitializer()
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

        num_atoms = len(structure)
        edge_index = []
        edge_attr = []

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distance = structure.get_distance(i, j)
                if distance <= self.radius:
                    edge_index.extend([[i, j], [j, i]])  # Add both directions

                    # Calculate Coulomb-like potential
                    z_i = structure[i].specie.Z
                    z_j = structure[j].specie.Z
                    coulomb_potential = (z_i * z_j) / distance

                    edge_attr.extend(
                        [[coulomb_potential], [coulomb_potential]]
                    )  # Same potential for both directions

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

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
