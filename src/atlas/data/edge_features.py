import numpy as np
import torch
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.periodic_table import Element
from scipy.constants import epsilon_0
from pymatgen.core import Structure
from atlas.data.atom_init import atom_init
from typing import Dict, Set, Mapping, Any


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


class WeightedGaussianDistanceCalculator:
    """
    Expands the distance by Gaussian basis with atom-specific weights.
    Unit: angstrom

    This class extends the GaussianDistanceCalculator by incorporating
    atom-specific weights in the distance expansion. The weights are
    calculated based on atomic properties such as mass, radius, and
    electronegativity.

    Attributes:
        filter (torch.Tensor): A tensor of evenly spaced points representing
            the centers of the Gaussian functions.
        var (float): The variance of the Gaussian functions.
        atomic_weights (dict): A dictionary mapping atomic numbers to
            their calculated weights.

    Methods:
        calculate_pairwise: Computes the weighted Gaussian distance expansion
            for pairs of atoms in a crystal structure.
        calculate_weight: Calculates the weight for a given atomic number
            based on its properties.
    """

    def __init__(self, dmin=0, dmax=8, step=0.2, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = torch.arange(dmin, dmax + step, step, dtype=torch.float32)
        if var is None:
            var = step
        self.var = var
        self.atomic_weights = {i: self.calculate_weight(i) for i in range(1, 119)}

    def calculate_pairwise(self, structure, edge_index):
        distances = torch.tensor(
            [structure.get_distance(i.item(), j.item()) for i, j in edge_index.t()],
            dtype=torch.float32,
        )

        # Get atomic numbers for each atom in the edge pairs
        atom_i = torch.tensor(
            [structure[i.item()].specie.number for i in edge_index[0]]
        )
        atom_j = torch.tensor(
            [structure[j.item()].specie.number for j in edge_index[1]]
        )

        # Calculate weights for each pair
        weights_i = torch.tensor([self.atomic_weights[int(a.item())] for a in atom_i])
        weights_j = torch.tensor([self.atomic_weights[int(a.item())] for a in atom_j])
        pair_weights = (weights_i + weights_j) / 2  # or some other combination

        # Apply Gaussian expansion
        gaussian_expansion = torch.exp(
            -(distances.unsqueeze(1) - self.filter).pow(2) / self.var**2
        )

        # Weight the expansion
        weighted_expansion = gaussian_expansion * pair_weights.unsqueeze(1)

        return weighted_expansion

    def calculate_weight(self, atomic_number):
        elem = Element.from_Z(atomic_number)
        mass = elem.atomic_mass or 1.0
        radius = elem.atomic_radius or 1.0
        electronegativity = elem.X or 1.0
        return (mass * radius * electronegativity) ** (1 / 3)


class AtomSpecificGaussianCalculator:
    def __init__(self, dmin=0, dmax=8, step=0.2):
        self.filter = torch.arange(dmin, dmax + step, step)
        # self.atom_specific_var = {
        #     i: (Element.from_Z(i).atomic_mass / 300) ** 0.5  # Square root to dampen the effect
        #     for i in range(1, 119)
        # }
        self.atom_specific_var = {i: self.calculate_variance(i) for i in range(1, 119)}

    def calculate_pairwise(self, structure, edge_index):
        distances = torch.tensor(
            [structure.get_distance(i.item(), j.item()) for i, j in edge_index.t()],
            dtype=torch.float32,
        )

        atom_i = torch.tensor(
            [structure[i.item()].specie.number for i in edge_index[0]]
        )
        atom_j = torch.tensor(
            [structure[j.item()].specie.number for j in edge_index[1]]
        )

        var_i = torch.tensor([self.atom_specific_var[int(a.item())] for a in atom_i])
        var_j = torch.tensor([self.atom_specific_var[int(a.item())] for a in atom_j])
        pair_var = (var_i + var_j) / 2  # or some other combination

        # Apply Gaussian expansion with pair-specific variance
        gaussian_expansion = torch.exp(
            -(distances.unsqueeze(1) - self.filter).pow(2) / pair_var.unsqueeze(1) ** 2
        )

        return gaussian_expansion

    def calculate_variance(self, atomic_number):
        elem = Element.from_Z(atomic_number)
        radius_factor = (elem.atomic_radius or 1.0) / 2
        electronegativity_factor = 1 / (elem.X or 1.0)
        mass_factor = (elem.atomic_mass / 200) ** 0.5
        return (radius_factor + electronegativity_factor + mass_factor) / 3


class PeriodicWeightedGaussianCalculator:
    """
    1. **Distance decay**: It uses a Gaussian function to model the decay of interatomic
    influences as distance increases. The `gaussian_width` parameter controls how quickly
    this decay occurs.

    2. **Atom size influence**: It incorporates the atomic radii of both atoms in each pair.
    Larger atoms will have a bigger influence on their neighbors.

    3. **Gaussian distance decay**: The core of the calculation is based on a Gaussian function,
    which provides a smooth decay of influence with distance.

    4. **Periodicity**: It takes into account the periodic boundary conditions of the crystal
    structure. This is done by working with fractional coordinates and applying the minimum
    image convention (subtracting the rounded difference from the original difference).

    5. **Cutoff**: It includes a cutoff distance beyond which interactions are set to zero.
    This can help limit the range of interactions and improve computational efficiency for
    large structures.
    """

    def __init__(self, cutoff=10.0, gaussian_width=1.0):
        self.cutoff = cutoff
        self.gaussian_width = gaussian_width

    def calculate_pairwise(
        self, structure: Structure, edge_index: torch.Tensor
    ) -> torch.Tensor:
        # Get lattice matrix
        lattice_matrix = torch.tensor(structure.lattice.matrix, dtype=torch.float32)

        # Get fractional coordinates
        frac_coords = np.array([site.frac_coords for site in structure])
        frac_coords = torch.tensor(frac_coords, dtype=torch.float32)
        # Get atomic radii
        atomic_radii = torch.tensor(
            [Element(site.specie.symbol).atomic_radius or 1.0 for site in structure],
            dtype=torch.float32,
        )

        # Calculate pairwise distances and vectors
        start, end = edge_index
        diff = frac_coords[end] - frac_coords[start]
        diff = diff - torch.round(diff)  # Apply periodic boundary conditions
        cart_diff = torch.matmul(diff, lattice_matrix)
        distances = torch.norm(cart_diff, dim=1)

        # Calculate Gaussian decay
        gaussian_values = torch.exp(-(distances**2) / (2 * self.gaussian_width**2))

        # Apply cutoff
        gaussian_values = torch.where(
            distances > self.cutoff, torch.zeros_like(gaussian_values), gaussian_values
        )

        # Incorporate atom size influence
        start_radii = atomic_radii[start]
        end_radii = atomic_radii[end]
        size_influence = (start_radii + end_radii) / 2

        # Combine all factors
        final_values = gaussian_values * size_influence

        return final_values.unsqueeze(1)


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

    def __init__(self, num_rbf=10, cutoff=8.0, normalize=True):
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.centers = torch.linspace(0, cutoff, num_rbf, dtype=torch.float32)
        self.widths = (self.centers[1] - self.centers[0]) * torch.ones_like(
            self.centers, dtype=torch.float32
        )
        self.normalize = normalize

    def calculate_pairwise(self, structure, edge_index):
        distances = torch.tensor(
            [structure.get_distance(i.item(), j.item()) for i, j in edge_index.t()],
            dtype=torch.float32,
        )
        rbf_output = torch.exp(
            -((distances.unsqueeze(1) - self.centers) ** 2) / (self.widths**2)
        )

        if self.normalize:
            # Normalize along the RBF dimension
            rbf_sum = rbf_output.sum(dim=1, keepdim=True)
            rbf_output = rbf_output / rbf_sum.clamp(min=1e-10)  # Avoid division by zero

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
