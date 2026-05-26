# Development of a Membrane-Oriented MLFF for Halogen Bonding using SPICE, ORCA and MACE

## Overview

This repository documents a workflow I developed for training a machine learning force field (MLFF) with enhanced representation of halogen-bonding interactions in drug-like molecules.

The project started from a problem that is particularly relevant to medicinal chemistry and membrane simulations. Many biologically active molecules contain chlorine, bromine or iodine. These atoms can form directional interactions with electron-rich atoms through the so-called σ-hole. In a conventional atom-centred representation, this anisotropic interaction is not always straightforward to describe, especially when the geometry of the interaction matters.

My long-term goal is to investigate halogenated ligands in lipid membrane environments. For that reason, the MLFF was designed as a membrane-oriented model: not because it has already been trained on complete membrane configurations, but because it was constructed with the interactions likely to matter in those applications in mind.

The dataset development strategy combines two complementary components:

1. publicly available DFT-level molecular data, providing broad chemical coverage;
2. targeted ORCA calculations on simple σ-hole / halogen-bond model systems, providing additional reference information for the directional interaction of interest.

The overall workflow is:

```text
Public DFT molecular data
        +
Targeted ORCA calculations for halogen bonding
        ↓
Chemically enriched training dataset
        ↓
MACE training
        ↓
Specialised MLFF for future membrane-oriented testing
```

The intention is not to impose an empirical correction for halogens, but to expose the machine learning model to quantum-mechanical reference configurations in which halogen-bond geometry is explicitly sampled.

This repository currently represents a model-development workflow. The resulting model is suitable for further validation and testing, but should not yet be interpreted as a quantitatively validated force field for membrane permeation or permeability prediction.

---

## What am I using? (The Software Stack)

This workflow combines quantum chemistry, molecular dataset preparation and equivariant machine learning.

### SPICE 2.0.1

SPICE is used as the publicly available source of molecular conformations with DFT-level reference energies and gradients. It provides the broad chemical background required for developing a transferable molecular potential.

### Python and ASE

Python scripts are used throughout the workflow for data preparation and automation. ASE, the Atomic Simulation Environment, is used to:

- represent molecular structures;
- read and write molecular configurations;
- convert configurations to `.extxyz` format;
- interface generated geometries with ORCA calculations.

### ORCA

ORCA is used to produce additional quantum-mechanical reference data for specifically designed halogen-bond geometries. These calculations provide energies and atomic forces for σ-hole scans that complement the broader public dataset.

The custom ORCA calculations use:

```text
wB97X-D4 def2-TZVP TightSCF
```

### MACE

MACE stands for **Message Passing Atomic Cluster Expansion**. It is an equivariant machine learning interatomic potential framework that learns molecular energies and atomic forces from quantum-mechanical reference configurations.

In this project, MACE is trained on a dataset that combines general molecular structures with targeted halogen-bond configurations, so that the model receives additional information about the directional interaction that motivated the workflow.

### PyTorch and CUDA

MACE training was performed through a GPU-accelerated PyTorch environment.

The training run documented here used:

```text
MACE version:  0.3.15
CUDA version:  12.1
GPU device:    0
```

---

## Workflow Summary

The project is organised into sequential stages:

```text
Step 1: Public DFT data preprocessing
        ↓
Step 2: Targeted σ-hole geometry generation
        ↓
Step 3: ORCA reference energy and force calculations
        ↓
Step 4: Isolated atom reference calculations
        ↓
Step 5: Dataset assembly and normalisation
        ↓
Step 6: MACE training and validation monitoring
```

---

## Step 1 — Preparing Publicly Available DFT Data

### Purpose

The first step establishes the broad chemical foundation of the MLFF using existing quantum-mechanical data.

The source file is:

```text
SPICE-2.0.1.hdf5
```

This dataset contains molecular conformations together with DFT reference energies and gradients. Rather than using all structures as a single undifferentiated collection, the script reorganises the data according to the chemistry relevant to this project.

In particular, structures containing chlorine, bromine or iodine are separated from more general molecular structures. This allows the later training strategy to retain broad chemical coverage while explicitly identifying halogen-containing configurations.

### Script

Run from the project root directory:

```bash
python 01_spice_subsets/step1_balanced_spice.py
```

### What the script does

The script performs the following operations:

1. Reads atomic numbers, molecular conformations, DFT total energies and DFT total gradients from the SPICE HDF5 file.
2. Converts coordinates from Bohr to Ångström.
3. Converts energies from Hartree to electronvolts.
4. Converts gradients to atomic forces in eV/Å.
5. Identifies configurations containing `Cl`, `Br` or `I`.
6. Writes the processed data into separate `.extxyz` files.

### Output files

```text
01_spice_subsets/subset_halogens.extxyz
01_spice_subsets/subset_general.extxyz
```

The first file contains configurations involving the halogens of greatest interest in this project. The second file provides the broader chemical background needed to avoid training a model that is narrowly restricted to a single type of interaction.

The purpose of this step can be summarised as:

```text
Public DFT Data
        ↓
Unit Conversion and Chemical Filtering
        ↓
Halogen-Containing and General Molecular Subsets
        ↓
MLFF Training Data Foundation
```

---

## Step 2 — Generating Targeted σ-Hole and Halogen-Bond Geometries

### Purpose

Although public DFT datasets provide broad chemical diversity, they do not necessarily sample directional halogen-bond geometries with the density required for a specialised model.

For this reason, I generated an additional collection of simple model complexes designed specifically to explore the geometry of the σ-hole interaction.

### Model systems

The halogen-bond donor molecules are methyl halides:

```text
CH3–Cl
CH3–Br
CH3–I
```

Each donor is paired with a representative Lewis base:

```text
NH3   → nitrogen-based acceptor model
H2O   → oxygen-based acceptor model
PH3   → phosphorus-based acceptor model
```

These systems are intentionally simple. They are not intended to represent an entire drug molecule or a complete lipid membrane. Their purpose is to isolate the directional interaction between a carbon-bound halogen and an electron-rich acceptor atom.

### Script

Run from the project root directory:

```bash
python 02_sigma_hole_geoms/step2_generate_scans.py
```

### Scan design

The script systematically varies both intermolecular distance and interaction angle:

```text
Halogens:      Cl, Br, I
Lewis bases:   N, O, P
Distance:      2.50–4.50 Å, in 0.25 Å increments
Angle:         180°–120°, in 1° increments
```

The angular scan is particularly important. Halogen bonding is associated with a directional region of positive electrostatic potential along the extension of the covalent C–X bond. Near-linear geometries therefore represent the preferred σ-hole approach direction, while more bent geometries provide contrasting examples.

The total number of generated geometries is:

```text
3 halogens × 3 Lewis bases × 9 distances × 61 angles = 4,941 geometries
```

### Output file

```text
02_sigma_hole_geoms/sigma_hole_input.xyz
```

At this stage, the output contains molecular coordinates and scan metadata only. New quantum-mechanical energies and atomic forces are calculated in the next step.

This stage can be summarised as:

```text
Scientific question:
How can directional halogen bonding be represented in the training data?

        ↓

Construction of CH3–X···Lewis Base model complexes

        ↓

Systematic distance and angular sampling

        ↓

Targeted σ-hole geometry library
```

---

## Step 3 — Calculating DFT Reference Energies and Forces with ORCA

### Purpose

The structures generated in Step 2 become useful for MLFF training only after assigning quantum-mechanical reference energies and atomic forces.

This is the role of the ORCA stage.

### Script

Run from the project root directory:

```bash
python 03_orca_references/step3_run_orca.py
```

A resume-capable version is also available for interrupted calculations:

```bash
python 03_orca_references/step3_resume_orca.py
```

### What the script does

The script reads:

```text
02_sigma_hole_geoms/sigma_hole_input.xyz
```

and performs an independent ORCA calculation for each generated σ-hole geometry.

The calculations use:

```text
wB97X-D4 def2-TZVP TightSCF
```

with the following ORCA settings:

```text
%pal nprocs 4 end
%scf MaxIter 300 end
```

For each geometry, the workflow calculates:

- a DFT reference energy;
- the corresponding atomic forces.

### Output file

```text
03_orca_references/train_sigma_hole_orca.extxyz
```

This file is the custom reference dataset produced specifically for halogen-bond geometry sampling.

The two main sources of molecular information are therefore complementary:

```text
SPICE-derived data:
Broad chemical coverage

ORCA-derived σ-hole data:
Dense sampling of the directional interaction of interest
```

The ORCA stage can be summarised as:

```text
Targeted σ-hole geometries
        ↓
DFT calculations with ORCA
        ↓
Reference energies and atomic forces
        ↓
Specialised halogen-bond training data
```

---

## Step 4 — Calculating Isolated Atom Reference Energies

### Purpose

MACE training requires a consistent energetic treatment of the chemical elements represented in the dataset. Isolated atom calculations provide atomic reference energies that can be used as energetic baselines during model training.

### Script

Run from the project root directory:

```bash
python step4_isolated_atoms.py
```

### Initial atomic references

The initial isolated atom calculations include:

```text
H, C, N, O, P, Cl, Br, I
```

Additional atomic references were subsequently added for elements appearing in the combined public dataset:

```text
B, F, Si, S
```

The final element coverage is therefore:

```text
H, B, C, N, O, F, Si, P, S, Cl, Br, I
```

### Quantum-mechanical level

The isolated atom calculations use:

```text
UKS wB97X-D4 def2-TZVP TightSCF
```

### Output file

```text
04_isolated_atoms/isolated_atoms.extxyz
```

Conceptually, the atomic references allow the model to handle the total energy as:

```text
Total energy ≈ sum of atomic reference energies
               + environment-dependent learned energy
```

or, more compactly:

```text
E_total ≈ Σ E_atomic_reference + E_learned(environment)
```

The isolated atom configurations are assigned zero forces, since a single isolated atom has no internal geometric degrees of freedom to optimise.

---

## Step 5 — Dataset Assembly and Label Normalisation

### Purpose

The final training data must combine general chemistry, halogen-containing public configurations, targeted ORCA reference structures and atomic energy baselines in a consistent format.

The four data sources are:

```text
1. SPICE general molecular configurations
2. SPICE halogen-containing configurations
3. ORCA σ-hole / halogen-bond reference configurations
4. Isolated atom reference configurations
```

---

### Initial assembly script

```bash
python step5_merge_datasets.py
```

This script combines the different components and separates them into training and validation subsets.

A central design choice is the increased representation of the ORCA-generated σ-hole data in the training set:

```python
orca_tr_weighted = orca_tr * 10
```

This tenfold oversampling does not mean that the model is trained only on halogen bonding. Instead, it increases the frequency with which the model encounters the directional interaction that is central to this project, while retaining the broader chemical information derived from SPICE.

The initial output files are:

```text
05_final_datasets/train_pilot_v1.extxyz
05_final_datasets/val_pilot_v1.extxyz
```

---

### Dataset repair and normalisation

During development, the assembled datasets required normalisation of the energy and force labels used by the MACE training setup.

This correction is performed with:

```bash
python fix_and_resume.py
```

The script standardises the reference keys:

```text
energy  → REF_energy
forces  → REF_forces
```

It also:

- removes inconsistent isolated atom records from earlier dataset versions;
- clears stored ASE calculator objects;
- adds missing isolated atom references for `B`, `F`, `Si` and `S`;
- generates corrected training and validation datasets.

The corrected output files are:

```text
05_final_datasets/train_fixed.extxyz
05_final_datasets/val_fixed.extxyz
```

---

### Final dataset assembly

The final, consolidated assembly script is:

```bash
python step5_ultimate_merge.py
```

This script directly prepares the MACE-compatible datasets by:

- reading the ORCA σ-hole reference configurations;
- reading the SPICE halogen and general subsets;
- normalising energies and forces to `REF_energy` and `REF_forces`;
- adding isolated atom references;
- applying the intended ORCA σ-hole oversampling in the training set;
- writing the final training and validation files.

### Final output files

```text
05_final_datasets/train_perfect.extxyz
05_final_datasets/val_perfect.extxyz
```

These are the files used as the final training-data products of the workflow.

The full dataset-construction strategy can be summarised as:

```text
General public DFT molecular data
        +
Halogen-containing public DFT configurations
        +
Purpose-built ORCA σ-hole reference calculations
        +
Isolated atom energy baselines
        ↓
Normalised and chemically enriched MACE datasets
```

### Development note

`step5_merge_datasets.py` and `fix_and_resume.py` document the earlier assembly and correction route used during development. `step5_ultimate_merge.py` is the consolidated script intended for preparing the final `train_perfect.extxyz` and `val_perfect.extxyz` datasets from the source components.

---

## Step 6 — Training the MLFF with MACE

### Purpose

The final datasets are used to train a MACE machine learning interatomic potential.

A force field must provide both molecular energies and forces. In this workflow, MACE learns an energy model from the quantum-mechanical reference data:

```text
E = E(atomic coordinates)
```

Atomic forces are obtained from the derivative of the learned energy:

```text
F_i = - dE / dR_i
```

where:

```text
E    = predicted molecular energy
F_i  = predicted force on atom i
R_i  = coordinates of atom i
```

In practical terms, the model is trained to reproduce the DFT-level energy and force information present in the assembled dataset.

### Training environment

The recorded training run used:

```text
MACE version:  0.3.15
CUDA version:  12.1
GPU device:    0
```

### Dataset size

```text
Training configurations:    134,460
Validation configurations:   10,495
```

### Chemical elements represented during training

```text
H, B, C, N, O, F, Si, P, S, Cl, Br, I
```

The training set therefore contains broad molecular coverage together with increased exposure to purpose-built σ-hole configurations.

---

## Training Performance

The validation curves show a marked reduction in both energy and force errors during the early part of training, followed by a lower-error region with fluctuations between epochs.

The best validation values recorded in the available training log were:

```text
Lowest recorded energy RMSE:
52.05 meV/atom at epoch 69

Lowest recorded force RMSE:
51.86 meV/Å at epoch 71
```

At the final recorded epoch shown in the log:

```text
Epoch 74:
Energy RMSE = 52.29 meV/atom
Force RMSE  = 56.90 meV/Å
```

These values indicate that the model learned a stable representation of a substantial part of the training and validation reference data.

![MACE validation energy and force error curves](<img width="2460" height="3095" alt="mace_validation_energy_force_combined" src="https://github.com/user-attachments/assets/5ff4bd46-e725-4018-84d4-8f833a27a9cf" />s/mace_validation_energy_force_combined.png)

**Figure 1.** Validation errors during MACE training. The upper panel shows the root mean square error for energies per atom, and the lower panel shows the root mean square error for atomic forces. Both quantities decrease substantially during the initial training period and remain in a lower-error range at later epochs.

It is important to distinguish validation performance from final physical validation. The curves show that the model fits the assembled validation set reasonably well. They do not yet demonstrate that it accurately predicts membrane permeation, lipid interactions or every possible halogen-bond environment.

---

## Generated Model Files

The workflow produced the following model-related artifacts:

```text
06_mace_training/halogen_membrane_best.pt
06_mace_training/checkpoints/halogen_membrane_v1_run-123_epoch-71.pt
07_md_simulation/mace_model.pt
07_md_simulation/mace_model_deployed.pt
```

These `.pt` files contain trained PyTorch/MACE model parameters intended for subsequent testing and deployment.

---

## Repository Structure

```text
.
├── 01_spice_subsets/
│   ├── step1_balanced_spice.py
│   ├── subset_general.extxyz
│   └── subset_halogens.extxyz
│
├── 02_sigma_hole_geoms/
│   ├── step2_generate_scans.py
│   └── sigma_hole_input.xyz
│
├── 03_orca_references/
│   ├── step3_run_orca.py
│   ├── step3_resume_orca.py
│   └── train_sigma_hole_orca.extxyz
│
├── 04_isolated_atoms/
│   └── isolated_atoms.extxyz
│
├── 05_final_datasets/
│   ├── train_pilot_v1.extxyz
│   ├── val_pilot_v1.extxyz
│   ├── train_fixed.extxyz
│   ├── val_fixed.extxyz
│   ├── train_perfect.extxyz
│   └── val_perfect.extxyz
│
├── 06_mace_training/
│   ├── mace_train.log
│   ├── halogen_membrane_best.pt
│   └── checkpoints/
│
├── 07_md_simulation/
│   ├── mace_model.pt
│   └── mace_model_deployed.pt
│
├── figures/
│   └── mace_validation_energy_force_combined.png
│
├── fix_and_resume.py
├── step4_isolated_atoms.py
├── step5_merge_datasets.py
└── step5_ultimate_merge.py
```

---

## Current Status

At its present stage, the project has achieved the following:

- preprocessing of publicly available DFT molecular data;
- separation of halogen-containing and general molecular configurations;
- generation of targeted σ-hole scan geometries;
- ORCA calculations for halogen-bond reference energies and forces;
- isolated atom reference calculations;
- assembly of normalised MACE training and validation datasets;
- GPU-accelerated training of a first MACE MLFF;
- production of trained model files and validation-error curves.

The current model is best described as:

```text
A quantum-data-driven, halogen-bond-enriched MLFF
developed for future testing in membrane-oriented simulations.
```

---

## Next Steps

The most important next stage is independent validation.

Planned validation tasks include:

1. evaluating the model on new σ-hole geometries not present in the training dataset;
2. comparing predicted energies and forces against additional ORCA calculations for halogenated drug-like molecules;
3. examining whether the model remains stable during molecular dynamics tests;
4. testing the model in more realistic membrane-oriented systems;
5. determining whether targeted halogen-bond enrichment improves the treatment of ligand interactions in lipid environments.

---

## Methodological Note

The custom σ-hole reference calculations were generated with ORCA using:

```text
wB97X-D4 def2-TZVP TightSCF
```

The publicly available SPICE structures originate from an external quantum-chemical dataset. Because public reference data and newly generated ORCA calculations may not necessarily be based on identical quantum-chemical settings, their compatibility should be examined carefully before presenting the merged model as a quantitatively validated production potential.

For this reason, the present repository should be viewed as a documented model-development workflow and a foundation for further validation, rather than as a completed membrane force field.

---

## Citation and Data Availability

The SPICE dataset used in this workflow is publicly available and should be cited according to its original publication and distribution terms.

ORCA is used for the additional quantum-chemical calculations and should be cited according to the official ORCA citation guidelines.

MACE is used as the machine learning interatomic potential framework and should be cited according to the original MACE publications.

Large raw datasets, ORCA output files and trained model artifacts may be omitted from the repository if file-size or licensing constraints apply. The repository is intended primarily to document the scripts, workflow and validation outputs required to reproduce the model-development process.
