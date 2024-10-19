import os

from pymatgen.io.vasp import Poscar


def convert_poscar_to_cif(from_file):
    if os.path.exists(from_file):
        poscar = Poscar.from_file(from_file)
        structure = poscar.structure
        # formula = structure.composition.reduced_formula
        # num_atoms = len(structure)
        # unique_id = f"{formula}_{num_atoms}"
        unique_id = os.path.splitext(os.path.basename(from_file))[0]
        structure.to(filename=unique_id + ".cif", fmt="cif")
    else:
        raise FileNotFoundError(f"The file {from_file} does not exist.")


def convert_all_vasp_to_cif(directory):
    """
    Example usage: convert_all_vasp_to_cif("./hybrid_perovskites_3D")
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    for filename in os.listdir(directory):
        if filename.endswith(".vasp"):
            full_path = os.path.join(directory, filename)
            try:
                convert_poscar_to_cif(full_path)
                print(f"Converted {filename} to CIF successfully.")
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")
