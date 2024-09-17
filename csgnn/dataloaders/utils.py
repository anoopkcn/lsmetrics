import numpy as np
import torch
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.periodic_table import Element
from scipy.constants import epsilon_0
from pymatgen.core import Structure
from csgnn.dataloaders.atom_init import atom_init
import torch
from typing import Dict, Set, Mapping, Any


class AtomInitializer(object):
    """
    Base class for initializing the vector representation for atoms.

    This class provides a framework for creating and managing atom embeddings.
    It supports loading and saving state dictionaries, as well as encoding
    and decoding atom types to and from indices.

    Attributes:
        atom_types (set): A set of atom types supported by the initializer.
        _embedding (dict): A dictionary mapping atom types to their vector representations.
        _decodedict (dict): A dictionary for decoding indices back to atom types.

    Note:
        It is recommended to use one AtomInitializer per dataset for consistency.
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_features(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = {
            int(k): torch.tensor(v, dtype=torch.float32) for k, v in state_dict.items()
        }
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type
            for idx, atom_type in enumerate(sorted(self._embedding.keys()))
        }

    def state_dict(self) -> Mapping[int, Any]:
        return {k: v.tolist() for k, v in self._embedding.items()}

    def decode(self, idx):
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initializes atom feature vectors using a custom JSON dictionary.

    This class extends AtomInitializer to provide initialization of atom
    feature vectors based on a predefined JSON dictionary (atom_init).
    It maps atomic numbers to corresponding feature vectors.
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
            idx: atom_type
            for idx, atom_type in enumerate(sorted(self._embedding.keys()))
        }

    def get_atom_features(self, atom_type: int) -> torch.Tensor:
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict: Dict[int, torch.Tensor]) -> None:
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type
            for idx, atom_type in enumerate(sorted(self._embedding.keys()))
        }

    def state_dict(self) -> Mapping[int, Any]:
        return self._embedding

    def decode(self, idx: int) -> int:
        return self._decodedict[idx]


class AtomFeatureCalculator:
    def __init__(self):
        self.properties = [
            "atomic_number",
            "electronegativity",
            "atomic_radius",
            "ionization_energy",
            "electron_affinity",
            "valence_electrons",
        ]

    def calculate_features(self, atomic_number):
        element = Element.from_Z(atomic_number)
        features = []

        for prop in self.properties:
            value = None
            if prop == "atomic_number":
                value = atomic_number
            elif prop == "electronegativity":
                value = element.X
            elif prop == "atomic_radius":
                value = element.atomic_radius
            elif prop == "ionization_energy":
                value = element.ionization_energy
            elif prop == "electron_affinity":
                value = element.electron_affinity
            elif prop == "valence_electrons":
                value = (
                    element.common_oxidation_states[0]
                    if element.common_oxidation_states
                    else None
                )

            features.append(float(value) if value is not None else 0.0)

        return torch.tensor(features, dtype=torch.float32)


class GaussianDistanceCalculator:
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """

    def __init__(self, dmin=0, dmax=8, step=0.2, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = torch.arange(dmin, dmax + step, step, dtype=torch.float32)
        if var is None:
            var = step
        self.var = var

    def calculate_pairwise(self, structure, edge_index):
        """
        Calculate pairwise Gaussian distance expansion.

        Args:
        structure (Structure): A pymatgen Structure object.
        edge_index (torch.Tensor): A tensor of shape [2, num_edges] containing the edge indices.

        Returns:
        torch.Tensor: A tensor of shape [num_edges, num_gaussian_filters] containing the expanded distances.
        """
        distances = torch.tensor(
            [structure.get_distance(i.item(), j.item()) for i, j in edge_index.t()],
            dtype=torch.float32,
        )
        distances = distances.view(-1, 1)  # Ensure shape is [num_edges, 1]

        # Compute Gaussian expansion
        return torch.exp(-(distances - self.filter).pow(2) / self.var**2)


class EwaldSummationCalculator:
    """
    A class for calculating pairwise Ewald summation energies for crystal structures.

    This calculator uses the pymatgen EwaldSummation class to compute the
    electrostatic interactions between ions in a periodic system. It can
    handle structures with or without pre-assigned oxidation states.
    """

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
    """
    Calculates pairwise energies using a truncated Coulomb potential.

    This class implements a simple Coulomb interaction between pairs of atoms,
    truncated at a specified cutoff radius. It handles structures with or without
    pre-assigned oxidation states.

    Attributes:
        cutoff_radius (float): The distance beyond which Coulomb interactions are ignored.
        conversion_factor (float): Factor to convert Coulomb's law to the desired units.
        bv_analyzer (BVAnalyzer): Tool for estimating oxidation states if not provided.
    """

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
    """
    This class calculates pairwise screened Coulomb interactions for a given structure.
    It uses a screening length to model the shielding effect in materials, which modifies
    the standard Coulomb interaction. The calculated energies are in units of eV.
    """

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
    """
    Radial Basis Function (RBF) calculator for pairwise atomic interactions.

    This class implements a Radial Basis Function (RBF) calculator that can be used
    to featurize pairwise atomic distances in crystal structures. It creates a set
    of Gaussian basis functions centered at evenly spaced points between 0 and a
    specified cutoff distance.

    The RBF expansion provides a smooth, continuous representation of atomic
    distances, which can be useful in various machine learning models for materials
    science applications.
    """

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
    """
    Calculates the cosine similarity between pairs of atoms in a structure.

    This class computes the cosine similarity between the position vectors of atom pairs
    specified by the edge_index. The cosine similarity is a measure of the angle between
    two vectors, ranging from -1 (opposite directions) to 1 (same direction), with 0
    indicating orthogonality.
    """

    def __init__(self):
        pass

    def calculate_pairwise(self, structure, edge_index):
        # Convert list of coords to a single numpy array first
        positions = np.array([site.coords for site in structure])
        # Then convert to a torch tensor
        positions = torch.from_numpy(positions).float()

        start, end = edge_index
        vec1 = positions[start]
        vec2 = positions[end]

        dot_product = (vec1 * vec2).sum(dim=1)
        norm1 = torch.norm(vec1, dim=1)
        norm2 = torch.norm(vec2, dim=1)

        # Avoid division by zero
        denominator = norm1 * norm2
        denominator = torch.where(
            denominator == 0, torch.ones_like(denominator), denominator
        )

        similarity = dot_product / denominator
        return similarity.unsqueeze(1)
