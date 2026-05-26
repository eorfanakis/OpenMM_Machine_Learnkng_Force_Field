import numpy as np, itertools
from ase import Atoms
from ase.io import write
from tqdm import tqdm

# Ορισμός των 3 Βάσεων κατά Lewis (Συντεταγμένες προσανατολισμένες στο Z)
bases = {
    'N': Atoms(['N', 'H', 'H', 'H'], positions=[[0,0,0], [0.94, 0, 0.38], [-0.47, 0.81, 0.38], [-0.47, -0.81, 0.38]]),
    'O': Atoms(['O', 'H', 'H'], positions=[[0,0,0], [0, 0.76, 0.59], [0, -0.76, 0.59]]),
    'P': Atoms(['P', 'H', 'H', 'H'], positions=[[0,0,0], [1.20, 0, 0.77], [-0.60, 1.04, 0.77], [-0.60, -1.04, 0.77]])
}

# Μήκη δεσμών C-X σε Angstroms
halogens = {'Cl': 1.78, 'Br': 1.93, 'I': 2.14}

distances = np.arange(2.5, 4.6, 0.25)
angles = np.arange(180, 119, -1)
combinations = list(itertools.product(distances, angles))

all_geoms = []

for h_elem, bond_len in halogens.items():
    # Μόριο Δότη: CH3-X
    ch3x = Atoms(['C', 'H', 'H', 'H', h_elem], 
                 positions=[[0,0,0], [1.09, 0, -0.36], [-0.54, 0.94, -0.36], [-0.54, -0.94, -0.36], [0, 0, bond_len]])
    x_pos = np.array([0.0, 0.0, bond_len])
    
    for b_elem, base_mol in bases.items():
        for dist, angle in tqdm(combinations, desc=f"📐 Scans: {h_elem} vs {b_elem}", unit=" geom"):
            frag = base_mol.copy()
            target_pos = x_pos + np.array([0.0, 0.0, dist])
            frag.translate(target_pos - frag.positions[0])
            frag.rotate(180.0 - angle, 'y', center=x_pos)
            
            dimer = ch3x + frag
            dimer.info.update({
                "scan_halogen": h_elem,
                "scan_base": b_elem,
                "scan_distance": float(dist), 
                "scan_angle": float(angle)
            })
            all_geoms.append(dimer)

write("02_sigma_hole_geoms/sigma_hole_input.xyz", all_geoms)
print(f"✅ Συνολικές γεωμετρίες υψηλής ακρίβειας που δημιουργήθηκαν: {len(all_geoms)}")
