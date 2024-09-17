import numpy as np
import torch
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.ewald import EwaldSummation
from scipy.constants import epsilon_0
from pymatgen.core import Structure
from csgnn.dataloaders.atom_init import atom_init
import torch
from typing import Dict, Set


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
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a dictionary mapping
    from element number to a list representing the
    feature vector of the element.
    """

    def __init__(self):
        elem_embedding: Dict[int, torch.Tensor] = {
            int(key): torch.tensor(value, dtype=torch.float)
            for key, value in atom_init.items()
        }
        atom_types: Set[int] = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        self._embedding: Dict[int, torch.Tensor] = elem_embedding
        self._decodedict: Dict[int, int] = {
            idx: atom_type for atom_type, idx in enumerate(self._embedding.keys())
        }

    def get_atom_features(self, atom_type: int) -> torch.Tensor:
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict: Dict[int, torch.Tensor]) -> None:
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for idx, atom_type in enumerate(self._embedding.keys())
        }

    def state_dict(self) -> Dict[int, torch.Tensor]:
        return self._embedding

    def decode(self, idx: int) -> int:
        return self._decodedict[idx]


class GaussianDistanceCalculator(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = torch.arange(dmin, dmax + step, step)
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

        # Compute Gaussian expansion
        return torch.exp(-(distances - self.filter).pow(2) / self.var**2)


class EwaldSummationCalculator:
    def __init__(self, accuracy=4.0):
        self.accuracy = accuracy
        self.bv = BVAnalyzer()

    def add_oxidation_states(self, structure):
        try:
            return self.bv.get_oxi_state_decorated_structure(structure)
        except ValueError:
            return structure.add_oxidation_state_by_guess()

    def calculate_pairwise(
        self, structure: Structure, edge_index: torch.Tensor
    ) -> torch.Tensor:
        # Add oxidation states if they're not present
        if not all(hasattr(site.specie, "oxi_state") for site in structure):
            structure = self.add_oxidation_states(structure)

        ewald = EwaldSummation(structure, acc_factor=self.accuracy)

        # Compute the full energy matrix
        energy_matrix = ewald.total_energy_matrix

        num_edges = edge_index.shape[1]
        pairwise_energies = torch.zeros(num_edges, 1)

        for i in range(num_edges):
            atom1_index = edge_index[0, i].item()
            atom2_index = edge_index[1, i].item()

            # Get the pairwise energy from the energy matrix
            energy = energy_matrix[atom1_index, atom2_index]
            pairwise_energies[i] = energy

        return pairwise_energies


class TruncatedCoulombCalculator:
    def __init__(self, cutoff_radius=10.0):
        self.cutoff_radius = cutoff_radius
        # Convert to atomic units (Hartree atomic units)
        self.conversion_factor = 1 / (4 * np.pi * epsilon_0) * 1.602176634e-19 * 1e10

    def calculate_pairwise(self, structure, edge_index):
        energies = []
        for i, j in edge_index.t():
            site_i = structure[i.item()]
            site_j = structure[j.item()]
            distance = structure.get_distance(i.item(), j.item())

            if distance > self.cutoff_radius:
                energies.append(0.0)
            else:
                charge_i = site_i.specie.oxi_state
                charge_j = site_j.specie.oxi_state
                energy = (charge_i * charge_j) / distance * self.conversion_factor
                energies.append(energy)

        return torch.tensor(energies, dtype=torch.float)
