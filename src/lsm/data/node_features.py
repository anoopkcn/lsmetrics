import numpy as np
import torch
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.periodic_table import (
    Element,
    Specie,
    Species,
    DummySpecie,
    DummySpecies,
)
from scipy.constants import epsilon_0
from pymatgen.core import Structure, Composition
from lsm.data.atom_init import atom_init
from typing import Dict, Set, Mapping, Any


def onehot_encode_atoms(structure: Structure, ATOM_NUM_UPPER: int = 96) -> torch.Tensor:
    """
    One-hot encodes atoms in a structure based on their atomic numbers.

    Args:
        structure (Structure): A pymatgen Structure object.
        ATOM_NUM_UPPER (int): The maximum atomic number to consider (default is 118).

    Returns:
        torch.Tensor: A tensor of shape (num_atoms, ATOM_NUM_UPPER) containing
                      one-hot encoded vectors for each atom in the structure.
    """
    num_atoms = len(structure)
    encoding = torch.zeros((num_atoms, ATOM_NUM_UPPER), dtype=torch.float32)

    for i, site in enumerate(structure):
        atomic_number = site.specie.number
        if 1 <= atomic_number <= ATOM_NUM_UPPER:
            encoding[i, atomic_number - 1] = 1.0

    return encoding


def atom_to_bit_features(
    structure: Structure, ATOM_NUM_UPPER: int = 96
) -> torch.Tensor:
    """
    Converts atoms in a structure to bit representations and returns them as features.

    Args:
        structure (Structure): A pymatgen Structure object.
        ATOM_NUM_UPPER (int): The maximum atomic number to consider (default is 96).

    Returns:
        torch.Tensor: A tensor of shape (num_atoms, ATOM_NUM_UPPER) representing
                      the atoms' features in bit representation.
    """
    num_atoms = len(structure)
    features = torch.zeros((num_atoms, ATOM_NUM_UPPER), dtype=torch.float32)

    for i, site in enumerate(structure):
        atomic_number = site.specie.number
        if 1 <= atomic_number <= ATOM_NUM_UPPER:
            # Convert to binary representation
            binary = format(atomic_number, f"0{ATOM_NUM_UPPER}b")

            # Convert binary string to list of integers
            bit_list = [int(bit) for bit in binary]

            # Pad with zeros if necessary
            bit_list = [0] * (ATOM_NUM_UPPER - len(bit_list)) + bit_list

            # Assign to the feature tensor
            features[i] = torch.tensor(bit_list, dtype=torch.float32)

    return features


def atom_custom_json_initializer(structure: Structure) -> torch.Tensor:
    """
    Generate feature vectors for atoms in the given structure using the
    custom JSON dictionary (atom_init).

    Args:
        structure (Structure): A pymatgen Structure object.

    Returns:
        torch.Tensor: A tensor of shape (num_atoms, feature_size) containing
                      feature vectors for each atom in the structure.
    """
    elem_features = {
        int(key): torch.tensor(value, dtype=torch.float32)
        for key, value in atom_init.items()
    }
    feature_size = len(next(iter(elem_features.values())))

    features = []
    for site in structure:
        atom_number = site.specie.number
        if atom_number in elem_features:
            features.append(elem_features[atom_number])
        else:
            # If atom type is not in our dictionary, use a zero vector
            features.append(torch.zeros(feature_size, dtype=torch.float32))

    return torch.stack(features)


def generate_site_species_vector(
    structure: Structure, ATOM_NUM_UPPER: int
) -> torch.Tensor:
    """
    from: https://github.com/omron-sinicx/crystalformer/blob/de25debae22b450e0116b8806f48b2786b3dbe87/dataloaders/common.py#L41
    """
    if hasattr(structure, "species"):
        atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        atom_num = torch.tensor(structure.atomic_numbers, dtype=torch.long).unsqueeze_(
            -1
        )
        x_species_vector = torch.eye(ATOM_NUM_UPPER)[atom_num - 1].squeeze()

    else:
        x_species_vector = []
        for site in structure.species_and_occu:
            site_species_and_occupancy = []
            for elem in site.elements:
                if type(elem) == Element:
                    occupancy = site.element_composition[elem]
                elif type(elem) == Specie or type(elem) == Species:
                    occupancy = site.element_composition[elem.element]
                elif type(elem) == Composition:
                    occupancy = site.element_composition[elem.element]
                    # print(elem, occupancy)
                elif type(elem) == DummySpecie or type(elem) == DummySpecies:
                    raise ValueError(f"Unsupported specie: {site}! Skipped")
                else:
                    print(site, type(site))
                    raise AttributeError
                atom_num = torch.tensor(elem.Z, dtype=torch.long)
                elem_onehot = torch.eye(ATOM_NUM_UPPER)[atom_num - 1]
                site_species_and_occupancy.append(elem_onehot * occupancy)
            site_species_and_occupancy_sum = torch.stack(
                site_species_and_occupancy
            ).sum(0)
            x_species_vector.append(site_species_and_occupancy_sum)
        x_species_vector = torch.stack(x_species_vector, 0)

    if x_species_vector.dim() == 1:
        x_species_vector.unsqueeze_(0)
    return x_species_vector
