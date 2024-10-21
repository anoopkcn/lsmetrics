# see https://docs.materialsproject.org/downloading-data/using-the-api/examples for example queries
import json
import os

from mp_api.client import MPRester
from tqdm import tqdm

API_KEY = os.environ.get("MP_API_KEY")


def save_data(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def load_data(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Load previously fetched data if exists
all_data = load_data("perovskite_data.json")


with MPRester(API_KEY) as mpr:
    results = mpr.materials.summary.search(
        formula="ABC3", fields=["material_id", "formula_pretty", "band_gap"]
    )

    # Filter out materials already processed
    material_ids = {doc.material_id for doc in results}
    print(f"Found {len(material_ids)} perovskite materials.")
    processed_ids = {entry["material_id"] for entry in all_data}
    new_material_ids = list(material_ids - processed_ids)

    for result in tqdm(results, desc="Fetching data"):
        entry_data = {"material_id": result.material_id}
        material_id = result.material_id
        band_gap = result.band_gap
        entry_data["band_gap"] = band_gap

        # Querying crystal structure
        structure = mpr.get_structure_by_material_id(material_id)
        entry_data["crystal_structure"] = structure.as_dict()

        all_data.append(entry_data)
        # Save after processing each material to minimize loss in case of interruption
        save_data(all_data, "perovskite_data.json")

print("Data fetching and serialization complete.")
