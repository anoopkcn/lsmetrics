import os
import json
import torch
import warnings
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from pymatgen.core import Structure
from csgnn.data.atom_init import atom_init

class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_features(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a dictionary mapping
    from element number to a list representing the
    feature vector of the element.
    """
    def __init__(self):
        elem_embedding = atom_init
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance expansion to distances.

        Args:
        distances (torch.Tensor): A tensor of shape [num_edges, 1] containing the distances.

        Returns:
        torch.Tensor: A tensor of shape [num_edges, num_gaussian_filters] containing the expanded distances.
        """
        distances = distances.view(-1, 1)  # Ensure shape is [num_edges, 1]

        # Convert filter and var to torch tensors
        filter_tensor = torch.tensor(self.filter, dtype=torch.float, device=distances.device)
        var_tensor = torch.tensor(self.var, dtype=torch.float, device=distances.device)

        # Compute Gaussian expansion
        return torch.exp(-(distances - filter_tensor).pow(2) / var_tensor.pow(2))


class CrystalStructureDataset(Dataset):
    def __init__(self, json_file, max_num_nbr=12, radius=8, dmin=0, step=0.2, target_property=None):
        super().__init__()
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.ari = AtomCustomJSONInitializer()
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.target_property = target_property
        if not os.path.isfile(json_file):
            raise FileNotFoundError(f"The JSON file '{json_file}' does not exist.")
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def len(self):
        return len(self.data)

    def get(self, idx):
        item = self.data[idx]
        cif_id = item['material_id']
        structure = Structure.from_dict(item['crystal_structure'])

        # Extract node features (e.g., atomic numbers)
        node_features = np.array([self.ari.get_atom_features(atom.specie.number) for atom in structure])
        node_features = torch.tensor(node_features, dtype=torch.float)

        all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        edge_index, edge_attr = [], []

        for center, nbr in enumerate(all_nbrs):
            if len(nbr) < self.max_num_nbr:
                warnings.warn(f'{cif_id} not find enough neighbors to build graph. '
                              f'If it happens frequently, consider increase radius.')
                nbr = nbr + [(structure[center], self.radius + 1., center)] * (self.max_num_nbr - len(nbr))
            else:
                nbr = nbr[:self.max_num_nbr]

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
                warnings.warn(f"Target property '{self.target_property}' not found for material {cif_id}")

        return data
