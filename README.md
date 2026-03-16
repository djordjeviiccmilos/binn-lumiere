# BINN — Biology-Informed Neural Network for Tumor Growth Modeling

A biology-informed neural network that models tumor growth dynamics using the LUMIERE dataset. The model learns patient-specific biological parameters and integrates them through an ODE solver to predict tumor volume trajectories over time.

---

## Overview

Standard neural networks treat tumor growth as a pure data-fitting problem and ignore known biology. This project takes a different approach: a small neural network predicts three biological parameters per patient, and those parameters are fed into a Gompertz ODE that handles the actual prediction. This way the model is constrained to produce biologically plausible trajectories.

The extended Gompertz equation used:

```
dV/dt = α · V · ln(K / V) - β · V
```

Where:
- `α` — tumor growth rate
- `K` — carrying capacity (maximum tumor volume)
- `β` — therapy effect (how much treatment slows or shrinks the tumor)

High `α`, low `β` → aggressive tumor, therapy ineffective  
Low `α`, high `β` → tumor shrinking, therapy working

---

## Project Structure

```
├── data/
│   └── processed/
│       ├── tumor_volumes.csv        # Extracted volumes (raw)
│       └── tumor_volumes_clean.csv  # Cleaned and normalized
├── models/
│   └── binn.py                      # BINN model, GompertzODE, BINNLoss
├── preprocessing/
│   ├── extract_volumes.py           # Extract tumor volumes from NIfTI masks
│   ├── prepare_data.py              # Clean, normalize, remove short sequences
│   ├── explore_data.py              # Visualize patient trajectories
│   └── graphs/                      # Generated plots
├── training/
│   ├── binn_trainer.py              # Train on full dataset
│   ├── kfold_trainer.py             # 5-fold cross-validation
│   └── outputs/                     # Saved models and training logs
└── results_summary.ipynb            # Results and analysis
```

---

## Model Architecture

**Input features (per patient):**
1. Normalized initial volume (always 1.0)
2. Log of actual initial volume in cm³
3. Number of measurements / 20
4. Max follow-up week / 173

**Parameter network:**  
4 → 64 → 64 → 3 (fully connected, Tanh activations)

The three outputs are passed through Softplus to ensure α, K, β are always positive.

**ODE solving:**  
For each patient, a `GompertzODE` module is constructed with the predicted parameters and integrated using `dopri5` (Dormand-Prince RK45) via `torchdiffeq`.

---

## Loss Function

```
total_loss = data_loss + 0.05 * biology_loss
```

- `data_loss` — MSE on log-transformed volumes (log1p), makes the loss scale-invariant
- `biology_loss` — penalty if parameters go outside biologically realistic ranges:
  - α > 1.0 is penalized
  - K > 10.0 is penalized
  - β > 1.0 is penalized

---

## Training

**Full training:**
```bash
cd training
python binn_trainer.py
```
Trains on all patients. Saves best model to `outputs/model.pt`. Uses early stopping (patience=50).

**5-fold cross-validation:**
```bash
cd training
python kfold_trainer.py
```
Runs 5-fold CV, saves per-fold logs and models, prints mean ± std of val loss, MAE, and MAPE. Best fold model is copied to `outputs/best_model.pt`.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Hidden size | 64 |
| Learning rate | 1e-3 (full) / 5e-3 (kfold) |
| Epochs | 500 |
| Biology weight | 0.05 (full) / 0.01 (kfold) |
| Early stopping patience | 50 |
| ODE solver | dopri5 |
| Optimizer | Adam + weight decay 1e-4 |
| LR scheduler | ReduceLROnPlateau |

---

## Data Pipeline

**Step 1 — Extract volumes:**
```bash
cd preprocessing
python extract_volumes.py
```
Reads all `seg_mask.nii` files, counts tumor voxels, converts to cm³, outputs `data/processed/tumor_volumes.csv`.

**Step 2 — Clean and normalize:**
```bash
cd preprocessing
python prepare_data.py
```
Removes patients with fewer than 4 measurements, averages duplicate weeks, normalizes time and volume, outputs `data/processed/tumor_volumes_clean.csv`.

**Step 3 — Explore (optional):**
```bash
cd preprocessing
python explore_data.py
```
Plots tumor trajectories for first 12 patients and measurement distribution histogram. Saves to `preprocessing/graphs/`.

---

## Dependencies

```
torch
torchdiffeq
pandas
numpy
matplotlib
scikit-learn
nibabel
```

Install:
```bash
pip install torch torchdiffeq pandas numpy matplotlib scikit-learn nibabel
```
