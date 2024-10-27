import json
import os
import warnings

import torch
from pymatgen.core import Structure
from torch_geometric.data import Data, Dataset

from lsm.data.edge_features import (
    GaussianDistanceCalculator,
)
from lsm.data.node_features import (
    generate_site_species_vector,
)


class CrystalStructureGraphDataset(Dataset):
    def __init__(
        self,
        json_file,
        calculators=None,
        atom_initializer=None,
        max_num_nbr=12,
        radius=8,
        target_property=None,
    ):
        super().__init__()
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.atom_initializer = atom_initializer
        self.target_property = target_property
        self.dtype = torch.float32

        # Set up calculators
        if calculators is None:
            self.calculators = [
                GaussianDistanceCalculator(dmin=0, dmax=radius, step=0.2)
            ]
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

        # Extract node features using the atom initializer
        if self.atom_initializer:
            node_features = self.atom_initializer(structure)
        else:
            node_features = generate_site_species_vector(structure, 96)

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
            edge_attr = calculator.calculate_pairwise(structure, edge_index)
            edge_attrs.append(edge_attr.to(self.dtype))

        # Concatenate all edge attributes
        edge_attr = torch.cat(edge_attrs, dim=1)

        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=torch.zeros(node_features.size(0), dtype=torch.long),
        )

        if self.target_property is not None:
            if self.target_property in item:
                target = torch.tensor([item[self.target_property]], dtype=self.dtype)
                data.y = target
            else:
                warnings.warn(
                    f"Target property '{self.target_property}' not found for material {cif_id}"
                )
        return data
