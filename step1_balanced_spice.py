import h5py, os
from ase import Atoms
from ase.io import write
from tqdm import tqdm

BOHR_TO_ANG = 0.529177210903
HART_TO_EV = 27.211386245988
GRAD_TO_FORC = -(HART_TO_EV / BOHR_TO_ANG)
halogens = {17, 35, 53}

out_hal = "01_spice_subsets/subset_halogens.extxyz"
out_gen = "01_spice_subsets/subset_general.extxyz"
for f_out in [out_hal, out_gen]:
    if os.path.exists(f_out): os.remove(f_out)

with h5py.File("SPICE-2.0.1.hdf5", "r") as f:
    items = list(f.items())
    for name, group in tqdm(items, desc="📊 SPICE Parsing", unit=" molecule"):
        z = group['atomic_numbers'][:]
        conf = group['conformations'][:] * BOHR_TO_ANG
        en = group['dft_total_energy'][:] * HART_TO_EV
        f_grad = group['dft_total_gradient'][:] * GRAD_TO_FORC
        
        is_hal = any(atom_z in halogens for atom_z in z)
        out_file = out_hal if is_hal else out_gen
        
        chunk = []
        for i in range(len(conf)):
            a = Atoms(numbers=z, positions=conf[i])
            a.info['energy'], a.arrays['forces'] = en[i], f_grad[i]
            chunk.append(a)
        write(out_file, chunk, format="extxyz", append=True)
