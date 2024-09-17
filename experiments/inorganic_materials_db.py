# %%
import pandas as pd
import glob
import os, json
from pymatgen.io.vasp import Poscar


path_to_json = r"../../data/hybrid_bandgap_database/"
json_files = glob.glob(os.path.join(path_to_json, "*.json"))
# convert all files to dataframe
df = pd.concat((pd.read_json(f, lines=True) for f in json_files))

# Remove materials with zero Band_gap_HSE and materials with Band_gap_HSE > 5
df = df[(df["Band_gap_HSE"] > 0) & (df["Band_gap_HSE"] <= 5)]

# %%
# print(df.iloc[10]["Band_gap_HSE"], df.iloc[10]["SNUMAT_id"])

# %%
# # Create a list to store the data for each material
materials_data = []

df["Structure_rlx"] = df["Structure_rlx"].astype(str)

# Iterate through the dataframe
for _, row in df.iterrows():
    # Convert Structure_rlx to a pymatgen structure
    structure = Poscar.from_str(str(row["Structure_rlx"])).structure

    # Create a dictionary for the current material
    material_dict = {
        "material_id": row["SNUMAT_id"],
        "band_gap": row["Band_gap_HSE"],
        "crystal_structure": structure.as_dict(),
    }

    # Append the dictionary to the list
    materials_data.append(material_dict)

# Write the data to a JSON file
with open("inorganic_materials_small.json", "w") as f:
    json.dump(materials_data, f, indent=4)

print("JSON file 'materials_data.json' has been created.")
