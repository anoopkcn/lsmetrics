# %%
import os
import json
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")


def create_json_from_cif_and_dataframe(cif_directory, dataframe_file):
    # Read the dataframe from the txt file
    df = pd.read_csv(dataframe_file, sep="\\s+")

    # Create a dictionary mapping compound names to band gaps
    bandgap_dict = dict(zip(df["compound"], df["Eg_fund"]))

    json_data = []

    # Iterate through CIF files in the directory
    for filename in os.listdir(cif_directory):
        if filename.endswith(".cif"):
            material_id = os.path.splitext(filename)[0]
            cif_path = os.path.join(cif_directory, filename)

            # Parse CIF file
            parser = CifParser(cif_path)
            structure = parser.parse_structures(primitive=True)[0]

            # Create crystal structure dictionary
            crystal_structure = structure.as_dict()

            # Get band gap from the dataframe
            band_gap = bandgap_dict.get(material_id, None)

            # Create entry for this material
            entry = {
                "material_id": material_id,
                "band_gap": band_gap,
                "crystal_structure": crystal_structure,
            }

            json_data.append(entry)

    # Write to JSON file
    with open("perovskites_halide.json", "w") as f:
        json.dump(json_data, f, indent=2)


# %%
# Usage
cif_directory = "../../data/hybrid_perovskites_3D_CIF"
dataframe_file = "../../data/hybrid_perovskites_3D_CIF/interesting_candidates_hybrid_perovskites_3D.txt"
create_json_from_cif_and_dataframe(cif_directory, dataframe_file)
