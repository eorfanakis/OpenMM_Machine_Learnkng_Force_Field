# Mixed ML/MM Molecular Dynamics Pipeline for Halogen Bonding

## Overview
This repository contains a hybrid Machine Learning / Molecular Mechanics (ML/MM) pipeline designed to accurately simulate and track **halogen bonding** of a ligand (Alprazolam) within a lipid membrane. 

Traditional empirical force fields (like CHARMM) struggle to correctly represent the quantum mechanical nuances of halogens (e.g., the $\sigma$-hole effect), often requiring arbitrary virtual sites and parameter tweaking. To solve this, we replaced the classical representation of the ligand with a **Machine Learning Force Field (MLFF)** trained on Density Functional Theory (DFT) data, while keeping the rest of the massive biological system (lipids + water) in the classical MM regime.

## What the am I using? (The Software Stack)
Bridging Quantum Mechanics, Machine Learning, and Classical MD in a single simulation is notoriously complex. Here is the stack that makes it possible:

* **OpenMM (v8.1+)**: The core Molecular Dynamics engine, chosen for its high customizability and excellent GPU acceleration.
* **MACE (Machine Learning form of a Coulomb Environment)**: The Equivariant Graph Neural Network used to predict energies and forces for the ligand. It was pre-trained on DFT data to intrinsically capture the $\sigma$-hole without explicit virtual sites.
* **OpenMM-ML & OpenMM-Torch**: The C++ and Python bridge. `openmm-ml` handles the creation of the "Mixed System" (stripping classical parameters from the ligand), while `openmm-torch` allows OpenMM to send atomic coordinates to PyTorch and receive MACE-predicted forces at every integration step.
* **PyTorch (CUDA 12.1)**: The backend executing the MACE model on the GPU.

## The Simulation Approach (SMD)
We perform a Steered Molecular Dynamics (SMD) simulation (umbrella pulling). The Alprazolam molecule is harmonically restrained and pulled along the Z-axis into the lipid bilayer. 
At every integration step:
1. OpenMM calculates the Classical forces for the membrane.
2. The PyTorch/MACE backend calculates the ML forces for Alprazolam.
3. A custom Python tracker dynamically scans for halogen bonds (Cl...O distance < 3.5 Å, angle > 140°) and logs the events.

## ⚠️ Challenges & "Dependency Hell"
Setting up this pipeline is computationally highly non-trivial. If you are trying to reproduce this environment, be prepared for intense Dependency Conflicts (ABI mismatches). 

**Known Problems Faced:**
1. **NumPy 1.x vs 2.x Conflict**: `openmm` is strictly compiled against NumPy 1.x. Installing newer scientific libraries forces NumPy 2.x, completely breaking the OpenMM C++ bindings.
2. **C++ ABI Mismatches (`libc10_cuda.so`)**: Mixing `pip` and `conda` channels for PyTorch and OpenMM-Torch results in C++ compiler conflicts. OpenMM-Torch expects a specific PyTorch C++ backend. 
3. **The `openmm-ml` Code Bug**: The current version of `openmm-ml` contains a hardcoded typo (`force.addExclusion(a1, a2, True)` instead of `(a1, a2)`). We had to manually patch (`sed`) the source code in the conda environment to make it run.

*Recommendation:* Strictly use `conda-forge` for `openmm`, `openmm-torch`, and `pytorch`, enforcing `pytorch=*=*cuda*` to avoid the CPU fallback, and pin `numpy<2`.
