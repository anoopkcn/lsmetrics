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
        calculators=None,
        max_num_nbr=12,
        radius=8,
        target_property=None,
    ):
        super().__init__()
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.ari = AtomCustomJSONInitializer()
        self.target_property = target_property
        self.dtype = torch.float32

        # Set up calculators
        self.gaussian_calculator = GaussianDistanceCalculator(
            dmin=0, dmax=radius, step=0.2
        )
        if calculators is None:
            self.calculators = [self.gaussian_calculator]
        elif isinstance(calculators, list):
            self.calculators = calculators
        else:
            self.calculators = [calculators]

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
                default_feature = torch.zeros(
                    self.ari.get_atom_features(1).shape[0], dtype=self.dtype
                )
                node_features.append(default_feature)

        node_features = torch.stack(node_features).to(self.dtype)

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
        distances = torch.tensor(distances, dtype=self.dtype).unsqueeze(1)

        # Calculate edge attributes using all provided calculators
        edge_attrs = []
        for calculator in self.calculators:
            if isinstance(calculator, GaussianDistanceCalculator):
                edge_attr = calculator.expand(distances)
            else:
                edge_attr = calculator.calculate_pairwise(structure, edge_index)
            edge_attrs.append(edge_attr.to(self.dtype))

        # Concatenate all edge attributes
        edge_attr = torch.cat(edge_attrs, dim=1)

        # Create PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        if self.target_property is not None:
            if self.target_property in item:
                target = torch.tensor([item[self.target_property]], dtype=self.dtype)
                data.y = target
            else:
                warnings.warn(
                    f"Target property '{self.target_property}' not found for material {cif_id}"
                )
        return data
