import torch
import json
import os
import warnings
from pymatgen.core import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.periodic_table import Element
from torch_geometric.data import Dataset, Data
from .base_dataloader import AtomCustomJSONInitializer


class AdvancedCrystalGraphDataset(Dataset):
    def __init__(self, json_file, radius=10, target_property=None):
        super().__init__()
        self.radius = radius
        self.ari = AtomCustomJSONInitializer()
        self.target_property = target_property
        self.voronoi = VoronoiNN(cutoff=radius)
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

        # Enhanced node features
        node_features = []
        for i, site in enumerate(structure):
            elem = site.specie
            node_feat = [
                elem.number,
                elem.electron_affinity,
                elem.ionization_energy,
                elem.atomic_radius,
                elem.group,
                elem.row,
                len(self.voronoi.get_nn_info(structure, i)),  # coordination number
            ]
            node_features.append(node_feat)
        node_features = torch.tensor(node_features, dtype=torch.float)

        num_atoms = len(structure)
        edge_index = []
        edge_attr = []

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distance = structure.get_distance(i, j)
                if distance <= self.radius:
                    edge_index.extend([[i, j], [j, i]])

                    z_i, z_j = structure[i].specie.Z, structure[j].specie.Z
                    coulomb_potential = (z_i * z_j) / distance

                    # Calculate relative coordinates
                    delta = structure[j].coords - structure[i].coords

                    edge_feat = [
                        coulomb_potential,
                        distance,
                        delta[0],
                        delta[1],
                        delta[2],  # relative coordinates
                        z_i * z_j,  # Coulomb matrix element
                    ]
                    edge_attr.extend([edge_feat, edge_feat])

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
