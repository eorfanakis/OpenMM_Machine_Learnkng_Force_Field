import random, os
from ase.io import read, write, iread
from tqdm import tqdm

out_dir = "05_final_datasets"
os.makedirs(out_dir, exist_ok=True)
random.seed(42)

def load_subset(filename, limit, desc):
    data = []
    for i, a in tqdm(enumerate(iread(filename)), total=limit, desc=desc):
        data.append(a)
        if i >= limit - 1: break
    return data

def split_data(data, frac=0.9):
    random.shuffle(data)
    cut = int(len(data) * frac)
    return data[:cut], data[cut:]

print("Διαβάζουμε τις κβαντικές δομές του ORCA...")
orca_all = read("03_orca_references/train_sigma_hole_orca.extxyz", index=":")
orca_tr, orca_val = split_data(orca_all, frac=0.9)
# Oversampling x10!
orca_tr_weighted = orca_tr * 10 

hal_all = load_subset("01_spice_subsets/subset_halogens.extxyz", 50000, "🧪 SPICE Halogens")
gen_all = load_subset("01_spice_subsets/subset_general.extxyz", 50000, "💧 SPICE General")

hal_tr, hal_val = split_data(hal_all, frac=0.9)
gen_tr, gen_val = split_data(gen_all, frac=0.9)

atoms_data = read("04_isolated_atoms/isolated_atoms.extxyz", index=":")

print("Ενώνουμε τα δεδομένα και τα ανακατεύουμε...")
train_final = orca_tr_weighted + hal_tr + gen_tr + atoms_data
val_final = orca_val + hal_val + gen_val

random.shuffle(train_final)
random.shuffle(val_final)

write(f"{out_dir}/train_pilot_v1.extxyz", train_final, format="extxyz")
write(f"{out_dir}/val_pilot_v1.extxyz", val_final, format="extxyz")
print(f"✅ Έτοιμα! Train: {len(train_final)} δομές | Valid: {len(val_final)} δομές")
