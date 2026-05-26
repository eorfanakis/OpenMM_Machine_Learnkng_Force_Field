import os
from ase.io import read, write
from ase.calculators.orca import ORCA
from tqdm import tqdm

orca_exe = "/media/manos/c46e35b0-80bd-43a2-9d0c-2ccd0ca4463f/ff/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca"
os.environ["ASE_ORCA_COMMAND"] = f"{orca_exe} PREFIX.inp > PREFIX.out"

out_file = "03_orca_references/train_sigma_hole_orca.extxyz"
if os.path.exists(out_file): os.remove(out_file)

structures = read("02_sigma_hole_geoms/sigma_hole_input.xyz", index=":")

for i, atoms in tqdm(enumerate(structures, start=1), total=len(structures), desc="⚛️ ORCA DFT", unit=" calc"):
    calc = ORCA(
        label=f"orca_scan_{i:04d}", directory="03_orca_references",
        orcasimpleinput="wB97X-D4 def2-TZVP TightSCF", 
        orcablocks="%pal nprocs 4 end\n%scf MaxIter 300 end"
    )
    atoms.calc = calc
    atoms.info["energy"] = atoms.get_potential_energy()
    atoms.arrays["forces"] = atoms.get_forces()
    write(out_file, [atoms], format="extxyz", append=True)
