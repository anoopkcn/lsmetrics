import pytest
import torch
import tempfile
import json
from csgnn.dataloaders.csdataloader import CrystalStructureGraphDataset
from csgnn.dataloaders.utils import GaussianDistanceCalculator, EwaldSummationCalculator


def test_crystal_structure_dataset():
    # Create a small test JSON file
    test_data = [
        {
            "material_id": "test1",
            "crystal_structure": {
                "@module": "pymatgen.core.structure",
                "@class": "Structure",
                "charge": 0,
                "lattice": {
                    "matrix": [
                        [3.85863387, -0.0, 0.0],
                        [-0.0, 3.85863387, 0.0],
                        [-0.0, -0.0, 3.85863387],
                    ],
                    "pbc": [True, True, True],
                    "a": 3.85863387,
                    "b": 3.85863387,
                    "c": 3.85863387,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                    "volume": 57.4514132376898,
                },
                "properties": {},
                "sites": [
                    {
                        "species": [{"element": "Ac", "occu": 1}],
                        "abc": [-0.0, 0.0, 0.0],
                        "properties": {"magmom": 0.0},
                        "label": "Ac",
                        "xyz": [0.0, 0.0, 0.0],
                    },
                    {
                        "species": [{"element": "Al", "occu": 1}],
                        "abc": [0.5, 0.5, 0.5],
                        "properties": {"magmom": -0.0},
                        "label": "Al",
                        "xyz": [1.929316935, 1.929316935, 1.929316935],
                    },
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.5, 0.5, 0.0],
                        "properties": {"magmom": 0.0},
                        "label": "O",
                        "xyz": [1.929316935, 1.929316935, 0.0],
                    },
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.5, 0.0, 0.5],
                        "properties": {"magmom": 0.0},
                        "label": "O",
                        "xyz": [1.929316935, 0.0, 1.929316935],
                    },
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.0, 0.5, 0.5],
                        "properties": {"magmom": 0.0},
                        "label": "O",
                        "xyz": [0.0, 1.929316935, 1.929316935],
                    },
                ],
            },
            "target_property": 1.0,
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        json.dump(test_data, tmp)
        tmp_path = tmp.name

    # Initialize required calculators
    energy_calculator = EwaldSummationCalculator()
    gaussian_calculator = GaussianDistanceCalculator(dmin=0, dmax=5, step=0.5, var=0.5)

    dataset = CrystalStructureGraphDataset(
        tmp_path,
        energy_calculator=energy_calculator,
        gaussian_distance_calculator=gaussian_calculator,
        target_property="target_property",
    )

    assert len(dataset) == 1

    data = dataset.get(0)
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)
    assert isinstance(data.edge_attr, torch.Tensor)
    assert isinstance(data.y, torch.Tensor)


def test_gaussian_distance_calculator():
    gd = GaussianDistanceCalculator(dmin=0, dmax=5, step=0.5, var=0.5)
    distances = torch.tensor([[1.0], [2.0], [3.0]])
    expanded = gd.expand(distances)

    assert expanded.shape == (3, 11)  # 3 distances, 11 Gaussian filters
    assert torch.all(expanded >= 0) and torch.all(expanded <= 1)
