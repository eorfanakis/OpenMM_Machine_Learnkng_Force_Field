import os, subprocess, numpy as np
from ase.io import read, write
from ase import Atoms

print("1/3: Υπολογισμός Baseline για B, F, Si, S (UKS)...")
orca_exe = "/media/manos/c46e35b0-80bd-43a2-9d0c-2ccd0ca4463f/ff/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca"
specs = {"B":(0,2), "F":(0,2), "Si":(0,3), "S":(0,3)}

# Διαβάζουμε τα παλιά Isolated Atoms και φτιάχνουμε τα ονόματά τους
old_isos = read("04_isolated_atoms/isolated_atoms.extxyz", ":")
all_isos = []
for a in old_isos:
    a.info["REF_energy"] = a.info.get("energy", 0.0)
    a.arrays["REF_forces"] = np.zeros((1,3))
    a.calc = None
    all_isos.append(a)

# Υπολογίζουμε τα νέα άτομα με ORCA
for el, (chg, mult) in specs.items():
    inp = f"04_isolated_atoms/iso_{el}.inp"
    out = f"04_isolated_atoms/iso_{el}.out"
    with open(inp, "w") as f:
        f.write(f"! UKS wB97X-D4 def2-TZVP TightSCF\n%pal nprocs 1 end\n* xyz {chg} {mult}\n{el} 0 0 0\n*")
    subprocess.run([orca_exe, inp], stdout=open(out, "w"), stderr=subprocess.STDOUT)
    en = 0.0
    for line in open(out):
        if "FINAL SINGLE POINT ENERGY" in line:
            en = float(line.split()[-1]) * 27.211386245988
    a = Atoms(el, positions=[[0,0,0]])
    a.info["REF_energy"] = en
    a.info["config_type"] = "IsolatedAtom"
    a.arrays["REF_forces"] = np.zeros((1,3))
    all_isos.append(a)

def fix_dataset(in_file, out_file, is_train=False):
    print(f"Διόρθωση του {in_file}...")
    data = read(in_file, ":")
    
    # Πετάμε τα παλιά Isolated Atoms που μπήκαν με λάθος keys
    data = [a for a in data if a.info.get("config_type") != "IsolatedAtom"]
    
    for a in data:
        try: en = a.get_potential_energy()
        except: en = a.info.get("energy", 0.0)
        try: f = a.get_forces()
        except: f = a.arrays.get("forces", np.zeros((len(a), 3)))
        
        a.calc = None # Καθαρισμός εσωτερικής μνήμης ASE
        a.info["REF_energy"] = en
        a.arrays["REF_forces"] = f
        
    if is_train:
        data.extend(all_isos) # Προσθήκη ΟΛΩΝ των Isolated Atoms στο τέλος
    write(out_file, data, format="extxyz")

print("2/3: Διόρθωση Train Set...")
fix_dataset("05_final_datasets/train_pilot_v1.extxyz", "05_final_datasets/train_fixed.extxyz", is_train=True)
print("3/3: Διόρθωση Valid Set...")
fix_dataset("05_final_datasets/val_pilot_v1.extxyz", "05_final_datasets/val_fixed.extxyz", is_train=False)
print("✅ Όλα διορθώθηκαν τέλεια!")
