import json


def save_data(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def load_data(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Load the first JSON file with material_id and crystal_structure
with open("perovskite_data.json", "r") as f:
    crystal_data = json.load(f)

# Load the second JSON file with material_id and bandgap
with open("new_band_gap.json", "r") as f:
    bandgap_data = json.load(f)

# Create a dictionary to store the combined data
all_data = load_data("combined_data.json")

# Iterate through the crystal structure data
for item in crystal_data:
    material_id = item["material_id"]
    entry_data = {"material_id": material_id}
    crystal_structure = item["crystal_structure"]
    entry_data["crystal_structure"] = crystal_structure

    # Find the corresponding bandgap
    band_gap = next(
        (bg["band_gap"] for bg in bandgap_data if bg["material_id"] == material_id),
        None,
    )

    if band_gap is not None:
        entry_data["band_gap"] = band_gap

    all_data.append(entry_data)

    save_data(all_data, "combined_data.json")

print("Combined data has been written to 'combined_data.json'")
