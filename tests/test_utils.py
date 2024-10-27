import pytest
import torch
import numpy as np
from pymatgen.core import Structure, Lattice
from lsm.data.edge_features import (
    EwaldSummationCalculator,
    GaussianDistanceCalculator,
    TruncatedCoulombCalculator,
    ScreenedCoulombCalculator,
    RBFCalculator,
    CosineSimilarityCalculator,
)


@pytest.fixture
def simple_structure():
    lattice = Lattice.cubic(4.0)
    return Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def complex_structure():
    lattice = Lattice.cubic(4.2)
    return Structure(
        lattice,
        ["Si", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
    )


def test_gaussian_distance_calculator(simple_structure):
    calculator = GaussianDistanceCalculator(dmin=0, dmax=5, step=0.5, var=0.5)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    gaussian_distances = calculator.calculate_pairwise(simple_structure, edge_index)

    assert isinstance(gaussian_distances, torch.Tensor)
    assert gaussian_distances.shape == (
        2,
        11,
    )  # 11 Gaussian filters from 0 to 5 with step 0.5
    assert torch.all(gaussian_distances >= 0) and torch.all(gaussian_distances <= 1)

    # Check symmetry
    assert torch.allclose(gaussian_distances[0], gaussian_distances[1])

    # Check peak location
    actual_distance = simple_structure.get_distance(0, 1)
    peak_index = torch.argmax(gaussian_distances[0])
    assert abs(calculator.filter[peak_index].item() - actual_distance) < calculator.var


def test_ewald_summation_calculator_initialization():
    calculator = EwaldSummationCalculator()
    assert calculator.accuracy == 4.0

    calculator = EwaldSummationCalculator(accuracy=5.0)
    assert calculator.accuracy == 5.0


def test_add_oxidation_states(simple_structure):
    calculator = EwaldSummationCalculator()
    decorated_structure = calculator.add_oxidation_states(simple_structure)

    assert all(hasattr(site.specie, "oxi_state") for site in decorated_structure)


def test_calculate_pairwise(simple_structure):
    calculator = EwaldSummationCalculator()
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    pairwise_energies = calculator.calculate_pairwise(simple_structure, edge_index)

    assert isinstance(pairwise_energies, torch.Tensor)
    assert pairwise_energies.shape == (2, 1)
    assert torch.all(pairwise_energies != 0)
    assert torch.allclose(pairwise_energies[0], pairwise_energies[1])


def test_calculate_pairwise_complex(complex_structure):
    calculator = EwaldSummationCalculator()
    num_atoms = len(complex_structure)
    edge_index = torch.tensor(
        np.array(np.meshgrid(range(num_atoms), range(num_atoms))).reshape(2, -1)
    )

    pairwise_energies = calculator.calculate_pairwise(complex_structure, edge_index)

    assert isinstance(pairwise_energies, torch.Tensor)
    assert pairwise_energies.shape == (num_atoms * num_atoms, 1)
    assert torch.all(pairwise_energies != 0)


def test_calculate_pairwise_symmetry(simple_structure):
    calculator = EwaldSummationCalculator()
    num_atoms = len(simple_structure)
    edge_index = torch.tensor(
        np.array(np.meshgrid(range(num_atoms), range(num_atoms))).reshape(2, -1)
    )

    pairwise_energies = calculator.calculate_pairwise(simple_structure, edge_index)

    energy_matrix = pairwise_energies.reshape(num_atoms, num_atoms)
    assert torch.allclose(energy_matrix, energy_matrix.t())


def test_ewald_summation_calculator_edge_cases():
    calculator = EwaldSummationCalculator()

    # Test with a single-atom structure
    single_atom_structure = Structure(Lattice.cubic(4.0), ["H"], [[0, 0, 0]])
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    pairwise_energies = calculator.calculate_pairwise(single_atom_structure, edge_index)
    assert pairwise_energies.shape == (1, 1)
    assert (
        pairwise_energies[0, 0] == 0
    )  # Self-interaction for a neutral atom should be zero

    # Test with a single-atom structure with non-zero oxidation state
    single_ion_structure = Structure(Lattice.cubic(4.0), ["Na1+"], [[0, 0, 0]])
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    pairwise_energies = calculator.calculate_pairwise(single_ion_structure, edge_index)
    assert pairwise_energies.shape == (1, 1)
    assert (
        pairwise_energies[0, 0] != 0
    )  # Self-interaction for an ion should be non-zero

    # Test with an empty edge index
    edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
    pairwise_energies = calculator.calculate_pairwise(single_atom_structure, edge_index)
    assert pairwise_energies.shape == (0, 1)


def test_truncated_coulomb_calculator(simple_structure):
    calculator = TruncatedCoulombCalculator(cutoff_radius=5.0)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    energies = calculator.calculate_pairwise(simple_structure, edge_index)
    assert energies.shape == (2, 1)


def test_screened_coulomb_calculator(simple_structure):
    calculator = ScreenedCoulombCalculator(screening_length=1.0)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    energies = calculator.calculate_pairwise(simple_structure, edge_index)
    assert energies.shape == (2, 1)


def test_rbf_calculator(simple_structure):
    calculator = RBFCalculator(num_rbf=10, cutoff=8.0)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    rbf_output = calculator.calculate_pairwise(simple_structure, edge_index)
    assert rbf_output.shape == (2, 10)


def test_cosine_similarity_calculator(simple_structure):
    calculator = CosineSimilarityCalculator()
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    similarity = calculator.calculate_pairwise(simple_structure, edge_index)
    assert similarity.shape == (2, 1)
    assert torch.all(torch.isfinite(similarity))  # Check for NaN and infinity
    assert torch.all(similarity >= -1 - 1e-6) and torch.all(similarity <= 1 + 1e-6)


if __name__ == "__main__":
    pytest.main()
