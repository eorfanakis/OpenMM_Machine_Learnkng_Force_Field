import random, os
from ase.io import read, write, iread
from ase import Atoms
import numpy as np
from tqdm import tqdm

out_dir = "05_final_datasets"
os.makedirs(out_dir, exist_ok=True)
random.seed(42)

def load_subset_stream(filename, limit, desc):
    data = []
    with open(filename, 'r') as f:
        for i, a in tqdm(enumerate(iread(f, format='extxyz')), total=limit, desc=desc):
            try: en = a.get_potential_energy()
            except: en = a.info.get("energy", 0.0)
            try: fcs = a.get_forces()
            except: fcs = a.arrays.get("forces", np.zeros((len(a), 3)))
            
            a.calc = None
            a.info["REF_energy"] = en
            a.arrays["REF_forces"] = fcs
            if "energy" in a.info: del a.info["energy"]
            if "forces" in a.arrays: del a.arrays["forces"]
            
            data.append(a)
            if i >= limit - 1: break
    return data

def split_data(data, frac=0.9):
    random.shuffle(data)
    cut = int(len(data) * frac)
    return data[:cut], data[cut:]

print("1/4: Διαβάζουμε και διορθώνουμε ORCA Scans...")
orca_all = load_subset_stream("03_orca_references/train_sigma_hole_orca.extxyz", 6000, "ORCA")
orca_tr, orca_val = split_data(orca_all, frac=0.9)
orca_tr_weighted = orca_tr * 10 

print("2/4: Διαβάζουμε και διορθώνουμε SPICE...")
hal_all = load_subset_stream("01_spice_subsets/subset_halogens.extxyz", 50000, "Halogens")
gen_all = load_subset_stream("01_spice_subsets/subset_general.extxyz", 50000, "General")

hal_tr, hal_val = split_data(hal_all, frac=0.9)
gen_tr, gen_val = split_data(gen_all, frac=0.9)

print("3/4: Διαβάζουμε τα Isolated Atoms...")
# ΑΣΦΑΛΗΣ ΜΕΘΟΔΟΣ: Ορίζουμε αυστηρά ποια άτομα ψάχνουμε
elements = ["H", "C", "N", "O", "P", "Cl", "Br", "I", "B", "F", "Si", "S"]
atoms_data = []

for el in elements:
    out_file = f"04_isolated_atoms/iso_{el}.out"
    if os.path.exists(out_file):
        en = 0.0
        with open(out_file, "r") as f:
            for line in f:
                if "FINAL SINGLE POINT ENERGY" in line:
                    en = float(line.split()[-1]) * 27.211386245988
        if en != 0.0:
            a = Atoms(el, positions=[[0, 0, 0]])
            a.info["REF_energy"] = en
            a.info["config_type"] = "IsolatedAtom"
            a.arrays["REF_forces"] = np.zeros((1, 3))
            atoms_data.append(a)

print("4/4: Ενώνουμε τα δεδομένα και αποθηκεύουμε...")
train_final = orca_tr_weighted + hal_tr + gen_tr + atoms_data
val_final = orca_val + hal_val + gen_val

random.shuffle(train_final)
random.shuffle(val_final)

write(f"{out_dir}/train_perfect.extxyz", train_final, format="extxyz")
write(f"{out_dir}/val_perfect.extxyz", val_final, format="extxyz")
print("✅ ΤΕΛΕΙΑ ΔΕΔΟΜΕΝΑ ΕΤΟΙΜΑ!")
