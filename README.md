# Mixed ML/MM Molecular Dynamics Pipeline for Halogen Bonding

## Overview

This repository documents the development of a membrane-oriented Machine Learning Force Field (MLFF) for halogenated drug-like molecules, with a particular focus on the accurate treatment of halogen bonding and directional σ-hole interactions.

The project was motivated by the intended application of molecular simulations to drug permeation through lipid membranes. In this context, halogenated compounds may form directional interactions with polar acceptor groups, including those present in lipid headgroups and other biologically relevant environments. These interactions are challenging to describe using purely isotropic classical atom-centered parameters, because the electrostatic behaviour of a covalently bound halogen is not spatially uniform.

To address this problem, the MLFF development strategy combines two complementary sources of quantum-mechanical information:

1. **Publicly available DFT-level molecular data**, used to provide broad chemical coverage and to expose the model to both halogen-containing and general drug-like molecular configurations.
2. **Targeted ORCA reference calculations**, specifically generated for σ-hole and halogen-bond geometries involving chlorine, bromine and iodine interacting with representative Lewis bases.

The resulting workflow is designed to produce a chemically informed training dataset for MACE, with increased emphasis on the interactions most relevant to halogenated ligands in membrane-oriented applications.

At this stage, the model should be understood as a specialized MLFF developed for future testing and validation in membrane simulation workflows, rather than as a fully validated production force field for quantitative permeability predictions.

## What am I using? (The Software Stack)

The development of this MLFF combines quantum chemistry, dataset engineering and equivariant machine learning:

- **SPICE 2.0.1 dataset**  
  Used as the publicly available source of DFT-level molecular configurations, energies and atomic forces. The dataset provides broad chemical coverage and forms the general foundation of the training data.

- **ASE (Atomic Simulation Environment)**  
  Used to read, manipulate and write molecular structures, generate `.extxyz` datasets, and interface the generated geometries with quantum-mechanical calculations.

- **ORCA**  
  Used to calculate additional DFT reference energies and forces for specifically designed σ-hole / halogen-bond geometries. The reference calculations were performed using the `wB97X-D4 def2-TZVP TightSCF` level of theory.

- **MACE (Message Passing Atomic Cluster Expansion)**  
  Used to train the Machine Learning Force Field. MACE is an equivariant graph-neural-network-based interatomic potential that learns molecular energies and atomic forces from quantum-mechanical reference data.

- **PyTorch with CUDA acceleration**  
  Used as the GPU-accelerated backend for MACE training. The training reported in this repository was executed using MACE v0.3.15 with CUDA 12.1.

## Workflow

### Step 1 — Preparing publicly available DFT data

The first step of the workflow uses publicly available quantum-mechanical data as the broad chemical foundation of the MLFF.

The `SPICE-2.0.1.hdf5` dataset contains DFT-level molecular conformations together with reference energies and atomic gradients. These data are converted into the units required for MLFF training and reorganized into chemically meaningful subsets:

- `subset_halogens.extxyz` — configurations containing `Cl`, `Br` or `I`
- `subset_general.extxyz` — configurations without these halogens

This separation allows the training strategy to retain broad chemical diversity while explicitly identifying molecular structures relevant to halogen chemistry.

Conceptually, this first step can be summarized as:

```text
Public DFT Data → Preprocessing and Chemical Filtering → ML Training Dataset
