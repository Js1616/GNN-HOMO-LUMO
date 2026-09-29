# GNN for HOMO-LUMO Gap Prediction

**Graph Neural Network pipeline for predicting molecular HOMO-LUMO gaps using edge-aware GINE on ~400,000 molecules.**

## Overview

This project investigates **Graph Isomorphism Networks with Edge features (GINE)** for predicting the **HOMO-LUMO gap** of molecules from molecular graph representations.

The model is evaluated using **Murcko scaffold-based splitting** to test generalization to structurally different molecular cores, rather than relying only on random train/test splits.

> **Project report:** The complete methodology, experiments, visualizations, error analysis, and discussion are provided in the accompanying research report.

## Key Results

| Metric                |                 Result |
| --------------------- | ---------------------: |
| **Test MAE**          |           **0.069 eV** |
| **Test Median Error** |           **0.049 eV** |
| **Test R²**           |             **0.9971** |
| Training MAE          |              0.0907 eV |
| Validation MAE        |              0.0989 eV |
| Train–Validation Gap  |              ~0.008 eV |
| Dataset Size          | **~400,000 molecules** |
| Throughput            | **1,000+ molecules/s** |

The model achieves **sub-0.1 eV MAE** while maintaining strong performance across different molecular sizes.

## Dataset & Preprocessing

The dataset follows the **OGB PCQM4Mv2-style molecular graph representation**, with DFT-computed HOMO-LUMO gaps.

### Molecular Graph

**Node features — 9D**

```text
Atomic number
Hydrogen count
Formal charge
Radical electrons
Hybridization
Aromaticity
Ring membership
Chirality
Degree
```

**Edge features — 3D**

```text
Bond type
Conjugation
Ring membership
```

### Scaffold Split

Molecules are divided using **Murcko scaffolds**, ensuring that validation and test sets contain chemically different molecular cores.

This reduces structural data leakage and provides a more meaningful test of inductive generalization.

## Model Architecture

```text
Molecular Graph
      │
      ▼
9D Node Features + 3D Edge Features
      │
      ▼
GINE × 6 Layers
      │
      ▼
256D Molecular Representations
      │
      ├──────────────┐
      ▼              ▼
 Global Add Pool   Global Max Pool
      │              │
      └──────┬───────┘
             ▼
        Concatenation
             │
             ▼
            MLP
             │
             ▼
     HOMO-LUMO Gap (eV)
```

**Model size:** ~2.1M parameters

## Training

* **Loss:** L1 / MAE
* **Optimizer:** AdamW
* **Learning rate:** `1e-3`
* **Weight decay:** `1e-2`
* **Batch size:** `128`
* **Epochs:** `50`
* **Warmup:** 5 epochs
* **Scheduler:** Cosine decay
* **Mixed precision:** AMP
* **Gradient clipping:** `1.0`
* **Random seed:** `42`

Target values are transformed using **log1p + standardization** to handle the skewed HOMO-LUMO gap distribution.

## Error Analysis

The model shows:

```text
Median error   : 0.049 eV
75th percentile: 0.070 eV
90th percentile: 0.150 eV
95th percentile: 0.200 eV
```

Performance remains relatively stable across molecules ranging from approximately **10–75 atoms**.

Higher errors are mainly associated with:

* High-gap molecules
* Rare molecular scaffolds
* Charged species
* Fused-ring systems
* Stereochemical and 3D effects not represented by the 2D graph

## Limitations

* Uses **2D molecular topology** rather than explicit 3D geometry
* No bond lengths, angles, or dihedral information
* Single-task HOMO-LUMO gap prediction
* Point predictions without uncertainty estimates
* Limited to the chemical/domain distribution represented in the dataset
* Generalization to organometallics, very large molecules, and unusual chemistries remains untested

## Future Work

* 3D GNN architectures such as **SchNet, DimeNet++, and GemNet**
* Multi-task prediction of HOMO, LUMO, IP, EA, dipole, and polarizability
* Active learning for difficult molecular chemistries
* Transfer learning from larger molecular datasets
* Uncertainty quantification using ensembles or conformal prediction
* Molecular subgraph attribution and explainability

## Tech Stack

**Python · PyTorch · PyTorch Geometric · GINE · RDKit · Graph Neural Networks · Scientific Machine Learning**

## Repository

This repository contains the implementation and experiments for **HOMO-LUMO gap prediction using GINE**.

The detailed methodology, complete results, figures, training analysis, and scientific discussion are available in the **project report**.

## Keywords

`GNN` `GINE` `PyTorch Geometric` `HOMO-LUMO Gap` `Molecular Property Prediction` `Quantum Chemistry` `DFT` `Scientific ML` `Molecular Graphs` `Drug Discovery` `Materials Science`
