import os
import json
import torch
import warnings
from torch_geometric.data import Dataset, Data
from pymatgen.core import Structure
from csgnn.dataloaders.utils import (
    GaussianDistanceCalculator,
    EwaldSummationCalculator,
    AtomCustomJSONInitializer,
)


class CrystalStructureGraphDataset(Dataset):
    def __init__(
        self,
        json_file,
        energy_calculator,
        gaussian_distance_calculator=None,
        max_num_nbr=12,
        radius=8,
        target_property=None,
    ):
        super().__init__()
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.ari = AtomCustomJSONInitializer()
        self.target_property = target_property
        self.energy_calculator = energy_calculator
        self.gaussian_distance_calculator = gaussian_distance_calculator
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
        node_features = []
        for atom in structure:
            atom_number = atom.specie.number
            if atom_number is not None and isinstance(atom_number, int):
                node_features.append(self.ari.get_atom_features(atom_number))
            else:
                print(f"Warning: Invalid atom number for atom: {atom}")
                default_feature = torch.zeros(self.ari.get_atom_features(1).shape[0])
                node_features.append(default_feature)

        node_features = torch.stack(node_features)

        all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        edge_index = []
        distances = []

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
                edge_index.append([center, neighbor[2]])
                distances.append(neighbor[1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        distances = torch.tensor(distances, dtype=torch.float).unsqueeze(1)

        # Calculate energies using the provided energy calculator
        energies = self.energy_calculator.calculate_pairwise(structure, edge_index)

        # Determine edge attributes based on the presence of Gaussian distance calculator
        if self.gaussian_distance_calculator:
            expanded_distances = self.gaussian_distance_calculator.expand(distances)
            edge_attr = torch.cat([expanded_distances, energies], dim=1)
        else:
            edge_attr = energies

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
