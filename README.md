# Mixed ML/MM Molecular Dynamics Pipeline for Halogen Bonding
## Overview

This repository contains the workflow I developed for training a machine learning force field (MLFF) with increased sensitivity to halogen-bonding interactions in drug-like molecules.

The project was motivated by a practical problem in molecular simulation. Many pharmaceutically relevant compounds contain chlorine, bromine or iodine, and these atoms can participate in directional interactions through the so-called σ-hole. This is especially relevant when considering the behaviour of halogenated ligands in polar biological environments, including lipid membrane interfaces.

The long-term aim of this work is to develop a model suitable for testing in membrane-oriented simulations of halogenated drug molecules. At the present stage, the model has not been trained directly on complete lipid bilayer configurations. Instead, it was built from a combination of:

1. publicly available DFT-level molecular data, used to provide broad chemical coverage;
2. targeted quantum-mechanical calculations designed specifically to sample σ-hole and halogen-bond geometries;
3. a MACE-based training workflow for learning energies and atomic forces.

The central idea is simple:

```text
Public DFT data
        +
Targeted ORCA calculations for halogen bonding
        ↓
Chemically enriched training dataset
        ↓
MACE training
        ↓
Specialized MLFF for future membrane-oriented applications

Rather than introducing an explicit empirical σ-hole correction, the aim is to expose the model to quantum-mechanical reference data that contain the directional interaction of interest. The resulting MLFF can then be assessed in future studies involving halogenated ligands and lipid environments.

What am I using? (The Software Stack)

The workflow combines quantum chemistry, molecular dataset preparation and equivariant machine learning.

SPICE 2.0.1
A publicly available DFT-level molecular dataset used as the broad chemical foundation of the training data. It provides molecular conformations together with reference energies and gradients.
Python and ASE (Atomic Simulation Environment)
Used for reading and writing molecular structures, converting units, generating scan geometries and assembling .extxyz datasets.
ORCA
Used to calculate additional DFT energies and forces for purpose-built σ-hole / halogen-bond geometries. The targeted calculations were performed at the wB97X-D4 def2-TZVP TightSCF level of theory.
MACE (Message Passing Atomic Cluster Expansion)
The machine learning interatomic potential used to train the MLFF. MACE learns energies and atomic forces from quantum-mechanical reference configurations while preserving the geometric relationships required for molecular systems.
PyTorch with CUDA acceleration
Used as the computational backend for MACE training. The training reported here was performed with MACE v0.3.15 on a CUDA 12.1 GPU environment.
Project Workflow

The workflow is organised into five main data-preparation stages followed by MACE training.

SPICE preprocessing
        ↓
Targeted σ-hole geometry generation
        ↓
ORCA reference calculations
        ↓
Isolated atom reference calculations
        ↓
Dataset assembly and normalisation
        ↓
MACE training
Step 1 — Preparing Publicly Available DFT Data
Purpose

The first stage uses an existing quantum-chemical dataset to provide a broad molecular background for the MLFF.

The SPICE-2.0.1.hdf5 file contains molecular conformations together with DFT total energies and gradients. Since the present project focuses on halogenated molecules, these data are processed and divided into two chemically meaningful groups:

configurations containing chlorine, bromine or iodine;
general molecular configurations without these halogens.

This separation does not discard the general chemistry present in the public dataset. Instead, it allows halogen-containing configurations to be handled explicitly during later dataset assembly.

Script
python 01_spice_subsets/step1_balanced_spice.py
What the script does

The script:

reads atomic numbers, coordinates, DFT energies and DFT gradients from the SPICE HDF5 file;
converts coordinates from Bohr to Ångström;
converts energies from Hartree to eV;
converts gradients into atomic forces in eV/Å;
detects structures containing Cl, Br or I;
exports the processed configurations in .extxyz format.
Output files
01_spice_subsets/subset_halogens.extxyz
01_spice_subsets/subset_general.extxyz

Conceptually, this stage is:

Public DFT Data → Preprocessing → Chemically Organised ML Training Data

These data provide the general quantum-mechanical foundation on which the more targeted halogen-bond information is later added.

Step 2 — Generating Targeted σ-Hole Geometries
Purpose

A broad public dataset is useful, but it does not guarantee dense coverage of the precise geometrical arrangements associated with directional halogen bonding. For that reason, I generated an additional collection of simple model complexes designed specifically to sample σ-hole interactions.

The donor molecules are methyl halides:

CH3–Cl
CH3–Br
CH3–I

Each donor is paired with a representative Lewis base:

NH3   → nitrogen-based acceptor model
H2O   → oxygen-based acceptor model
PH3   → phosphorus-based acceptor model

These are intentionally simple model systems. Their role is not to reproduce a whole drug or a membrane, but to isolate the geometric features of a halogen bond in a controlled way.

Script
python 02_sigma_hole_geoms/step2_generate_scans.py
Scan design

The script varies both distance and angle:

Halogens:      Cl, Br, I
Lewis bases:   N, O, P
Distance:      2.50–4.50 Å in 0.25 Å increments
Angle:         180°–120° in 1° increments

The angular sampling is particularly important because halogen bonding is directional. Configurations close to 180° correspond to an acceptor approaching along the extension of the C–X bond, which is the direction associated with the σ-hole. Bent geometries provide contrasting configurations and help the future model distinguish a directional interaction from a generic contact.

The total number of generated structures is:

3 halogens × 3 Lewis bases × 9 distances × 61 angles = 4,941 geometries
Output file
02_sigma_hole_geoms/sigma_hole_input.xyz

At this point, only coordinates and scan metadata are generated. Energies and forces are obtained in the next step by running DFT calculations with ORCA.

Conceptually:

Directional Halogen-Bond Question
        ↓
CH3–X···Lewis Base Model Systems
        ↓
Distance and Angular Scans
        ↓
Targeted σ-Hole Geometry Library
Step 3 — Calculating DFT Reference Energies and Forces with ORCA
Purpose

The targeted geometries created in Step 2 are converted into useful ML training data by evaluating them at the quantum-mechanical level.

Script
python 03_orca_references/step3_run_orca.py

A resume-capable version of the workflow is also available for interrupted calculations:

python step3_resume_orca.py
What the script does

The script reads:

02_sigma_hole_geoms/sigma_hole_input.xyz

and performs a separate ORCA calculation for each generated geometry using:

wB97X-D4 def2-TZVP TightSCF

with four CPU processes per calculation and an increased SCF iteration limit:

%pal nprocs 4 end
%scf MaxIter 300 end

For every σ-hole geometry, the calculation provides:

a DFT reference energy;
atomic forces derived from the quantum-mechanical calculation.
Output file
03_orca_references/train_sigma_hole_orca.extxyz

This is the custom reference dataset generated specifically for the interaction of interest. Whereas the SPICE-derived data provide broad chemical coverage, the ORCA-derived structures provide dense sampling of halogen-bond geometries.

Conceptually:

Targeted σ-Hole Geometries
        ↓
ORCA DFT Calculations
        ↓
Reference Energies and Forces
        ↓
Specialised Halogen-Bond Training Data
Step 4 — Calculating Isolated Atom Reference Energies
Purpose

MACE training benefits from consistent atomic reference energies for the elements present in the dataset. These isolated atom calculations provide the energetic baseline used during the training process.

Script
python step4_isolated_atoms.py
Initial elements

The initial isolated atom calculations include:

H, C, N, O, P, Cl, Br, I

Additional elements required by the combined training dataset were subsequently included:

B, F, Si, S

Therefore, the full element coverage of the final training workflow is:

H, B, C, N, O, F, Si, P, S, Cl, Br, I
Quantum-mechanical level

The isolated atom reference calculations were performed using:

UKS wB97X-D4 def2-TZVP TightSCF
Output file
04_isolated_atoms/isolated_atoms.extxyz

The role of these references can be represented conceptually as:

E
total
	​

≈
i
∑
	​

E
i
reference
	​

+E
environment
	​

(R)

where the model learns the environment-dependent part of the energy while using consistent atomic reference contributions.

Step 5 — Assembling and Normalising the Final Training Dataset
Purpose

The final dataset combines general molecular information with enhanced sampling of the chemistry that matters most for this project.

The dataset is assembled from four sources:

SPICE general configurations
SPICE halogen-containing configurations
ORCA σ-hole / halogen-bond configurations
Isolated atom reference configurations
Initial dataset assembly
python step5_merge_datasets.py

This script combines the different data sources and separates them into training and validation sets.

A central design choice is the increased representation of the ORCA-generated σ-hole structures in the training data:

orca_tr_weighted = orca_tr * 10

This tenfold oversampling increases the exposure of the model to directional halogen-bond geometries during training. The aim is not to replace general molecular chemistry, but to ensure that the model pays sufficient attention to the interaction that motivated the project.

The initial assembled datasets are:

05_final_datasets/train_pilot_v1.extxyz
05_final_datasets/val_pilot_v1.extxyz
Dataset correction and label normalisation
python fix_and_resume.py

This script performs corrections required for consistent MACE training. In particular, it standardises the reference labels:

energy  → REF_energy
forces  → REF_forces

It also:

removes inconsistent isolated atom entries;
resets stored ASE calculators;
includes missing isolated atom references;
creates corrected training and validation files.

Output files:

05_final_datasets/train_fixed.extxyz
05_final_datasets/val_fixed.extxyz
Final dataset preparation
python step5_ultimate_merge.py

This script generates the final MACE-ready datasets by combining the normalised reference data, including isolated atom baselines and the oversampled ORCA halogen-bond configurations.

Final output files:

05_final_datasets/train_perfect.extxyz
05_final_datasets/val_perfect.extxyz

The final dataset design can be summarised as:

General DFT Molecular Coverage
        +
Halogen-Containing Public DFT Configurations
        +
Targeted ORCA σ-Hole Reference Data
        +
Atomic Energy Baselines
        ↓
MACE-Ready Training and Validation Datasets
MACE Training

The final datasets were used to train a MACE machine learning interatomic potential.

MACE learns an energy function from the quantum-mechanical reference configurations:

E=E(R)

and obtains atomic forces through differentiation:

F
i
	​

=−∇
i
	​

E(R)

This means that the trained model is intended to reproduce both the energies and the forces associated with the reference data, including the targeted halogen-bond geometries generated in this work.

Training environment
MACE version:   0.3.15
CUDA version:   12.1
GPU device:     0
Dataset sizes
Training configurations:    134,460
Validation configurations:   10,495
Chemical elements represented
H, B, C, N, O, F, Si, P, S, Cl, Br, I

The trained model was therefore exposed to both general molecular configurations and an enriched subset of geometries related to halogen bonding.

Training Performance

The validation errors decreased substantially over the course of training. The model rapidly improved during the early epochs and subsequently stabilised in a lower-error region.

Representative values observed in the later part of training were approximately:

Energy RMSE:   52–54 meV/atom
Force RMSE:    52–60 meV/Å

The lowest energy validation error observed in the available training record was:

RMSE_E_per_atom = 52.05 meV/atom

at epoch 69.

The training curves are shown below:

Figure 1. Validation error during MACE training. The upper panel shows the energy RMSE per atom, while the lower panel shows the force RMSE. Both metrics decrease markedly during the early training period and remain within a comparatively stable low-error region at later epochs.

These validation results indicate that the model has learned a substantial part of the energy and force patterns contained in the assembled dataset. However, this result should not be interpreted as full validation for lipid membrane simulations. Independent testing on unseen halogen-bond configurations and membrane-relevant molecular systems remains necessary.

Trained Model Files

The workflow produced the following model-related files:

06_mace_training/halogen_membrane_best.pt
06_mace_training/checkpoints/halogen_membrane_v1_run-123_epoch-71.pt
07_md_simulation/mace_model.pt
07_md_simulation/mace_model_deployed.pt

These .pt files are PyTorch/MACE model artifacts containing trained neural-network parameters for subsequent testing and deployment.
