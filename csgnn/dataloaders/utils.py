import torch
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core import Structure


class GaussianDistance(object):
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
