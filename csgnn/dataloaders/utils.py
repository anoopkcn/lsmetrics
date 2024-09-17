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
            int(key): torch.tensor(value, dtype=torch.float32)
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
        self.filter = torch.arange(dmin, dmax + step, step, dtype=torch.float32)
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
        if not all(hasattr(site.specie, "oxi_state") for site in structure):
            structure = self.add_oxidation_states(structure)

        ewald = EwaldSummation(structure, acc_factor=self.accuracy)
        energy_matrix = torch.tensor(ewald.total_energy_matrix, dtype=torch.float32)

        pairwise_energies = energy_matrix[edge_index[0], edge_index[1]].unsqueeze(1)
        return pairwise_energies


class TruncatedCoulombCalculator:
    def __init__(self, cutoff_radius=10.0):
        self.cutoff_radius = cutoff_radius
        self.conversion_factor = 1 / (4 * np.pi * epsilon_0) * 1.602176634e-19 * 1e10
        self.bv_analyzer = BVAnalyzer()

    def calculate_pairwise(self, structure, edge_index):
        if not all(hasattr(site.specie, "oxi_state") for site in structure):
            structure = self.add_oxidation_states(structure)

        charges = torch.tensor(
            [site.specie.oxi_state for site in structure], dtype=torch.float32
        )
        distances = torch.tensor(
            [structure.get_distance(i.item(), j.item()) for i, j in edge_index.t()],
            dtype=torch.float32,
        )

        charge_products = charges[edge_index[0]] * charges[edge_index[1]]
        energies = torch.where(
            (distances > self.cutoff_radius) | (distances < 1e-10),
            torch.zeros_like(distances),
            charge_products / distances * self.conversion_factor,
        )

        return energies.unsqueeze(1)

    def add_oxidation_states(self, structure):
        try:
            return self.bv_analyzer.get_oxi_state_decorated_structure(structure)
        except ValueError:
            return structure.add_oxidation_state_by_guess()


class ScreenedCoulombCalculator:
    def __init__(self, screening_length=1.0):
        self.screening_length = screening_length
        self.conversion_factor = 14.4  # eV * Å

    def calculate_pairwise(self, structure, edge_index):
        if not all(hasattr(site.specie, "oxi_state") for site in structure):
            structure = structure.add_oxidation_state_by_guess()

        charges = torch.tensor(
            [site.specie.oxi_state for site in structure], dtype=torch.float32
        )
        distances = torch.tensor(
            [structure.get_distance(i.item(), j.item()) for i, j in edge_index.t()],
            dtype=torch.float32,
        )

        charge_products = charges[edge_index[0]] * charges[edge_index[1]]
        energies = (
            self.conversion_factor
            * charge_products
            * torch.exp(-distances / self.screening_length)
            / distances
        )

        return energies.unsqueeze(1)


class RBFCalculator:
    def __init__(self, num_rbf=10, cutoff=8.0):
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.centers = torch.linspace(0, cutoff, num_rbf, dtype=torch.float32)
        self.widths = (self.centers[1] - self.centers[0]) * torch.ones_like(
            self.centers, dtype=torch.float32
        )

    def calculate_pairwise(self, structure, edge_index):
        distances = torch.tensor(
            [structure.get_distance(i.item(), j.item()) for i, j in edge_index.t()],
            dtype=torch.float32,
        )
        rbf_output = torch.exp(
            -((distances.unsqueeze(1) - self.centers) ** 2) / (self.widths**2)
        )
        return rbf_output


class CosineSimilarityCalculator:
    def __init__(self):
        pass

    def calculate_pairwise(self, structure, edge_index):
        positions = torch.tensor(
            [site.coords for site in structure], dtype=torch.float32
        )
        start, end = edge_index
        vec1 = positions[start]
        vec2 = positions[end]

        dot_product = (vec1 * vec2).sum(dim=1)
        norm1 = torch.norm(vec1, dim=1)
        norm2 = torch.norm(vec2, dim=1)

        similarity = dot_product / (norm1 * norm2)
        return similarity.unsqueeze(1)
