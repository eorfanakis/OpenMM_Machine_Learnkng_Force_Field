import os
from ase.io import read, write
from ase.calculators.orca import ORCA
from tqdm import tqdm

out_dir = "03_orca_references"
orca_exe = "/media/manos/c46e35b0-80bd-43a2-9d0c-2ccd0ca4463f/ff/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca"
os.environ["ASE_ORCA_COMMAND"] = f"{orca_exe} PREFIX.inp > PREFIX.out"

out_file = os.path.join(out_dir, "train_sigma_hole_orca.extxyz")
all_structures = read("02_sigma_hole_geoms/sigma_hole_input.xyz", index=":")

# Έλεγχος προόδου
start_idx = 0
if os.path.exists(out_file):
    try:
        existing_data = read(out_file, index=":")
        start_idx = len(existing_data)
        print(f"📊 Βρέθηκαν {start_idx} ήδη υπολογισμένες δομές. Συνεχίζουμε από την {start_idx + 1}...")
    except:
        print("⚠️ Το αρχείο εξόδου είναι κατεστραμμένο, ξεκινάμε από την αρχή.")
        os.remove(out_file)

# Εκτέλεση για τις υπόλοιπες δομές
for i, atoms in tqdm(enumerate(all_structures[start_idx:], start=start_idx+1), 
                     total=len(all_structures)-start_idx, desc="⚛️ ORCA DFT Resume"):
    calc = ORCA(label=f"orca_scan_{i:04d}", directory=out_dir,
                orcasimpleinput="wB97X-D4 def2-TZVP TightSCF",
                orcablocks="%pal nprocs 4 end\n%scf MaxIter 300 end")
    atoms.calc = calc
    atoms.info["energy"] = atoms.get_potential_energy()
    atoms.arrays["forces"] = atoms.get_forces()
    write(out_file, [atoms], format="extxyz", append=True)
