# Computational Physics & (MDP) Reinforcement Learning — Projects (UniTrento, 2025)

This is the final project for the **Computational Physics** and **Markov Decision
Processes & Reinforcement Learning** courses held at the University of Trento in
2025 by Prof. Pederiva and Prof. Cordoni, respectively.

The work is primarily focused on **Physics-Informed Neural Networks (PINNs)** 
applied to various problems related to the Gross-Pitaevskii Equation (GPE) and
in reproducing a result from the _Raissi, M., P. Perdikaris, and G.E. Karniadakis 
(2019) paper on PINNs_.

> Note: the directory structure is organized as a set of sub-projects
(`ex0_*`, `ex1_*`, …), each mostly self-contained.

---

## Repository layout

- `ex0_sin/`  
  Introductory PINN example solving a simple boundary value problem whose analytical solution is `sin(x)`.

- `ex1_GPE_simple/`  
  PINN solver for a simple (time-independent) GPE-like setup (1D).

- `ex2_GPE_time_dependent/`  
  PINN solver for a **time-dependent** GPE-like problem.

- `ex3_GPE_double_well/`  
  PINN solver for a GPE in a **double-well potential**, including utilities such as Gaussian initialization.

- `ex3.5_eigenvalue_problem/`  
  Materials related to an eigenvalue-style problem (naming indicates an intermediate assignment).

- `ex4_GPE_vortex/`  
  PINN solver for a **vortex** / rotating-state style configuration (complex field represented via real+imaginary outputs).

- `report/` 
  latex source and compiled PDF of the final report summarizing the work, results, and conclusions.

---

## Requirements

The code is written in ```python3.13.12``` and uses common scientific/ML packages. 
From the code, the main dependencies are:

- `pytorch`
- `numpy`
- `scipy`
- `matplotlib`

An AMD RX 6750 XT GPU was used for development, but the code should run on any
CUDA/ROCm-compatible GPU or even CPU (with slower performance) selecting
the device automatically.

---

## How to run

Each exercise has a `src/` folder containing scripts (often a `main.py`) that 
define the domain/problem configuration and start training.

Examples:

### `ex0_sin` (intro PINN)
Run the example training script:

```bash
cd ex1_GPE_simple
python src/main.py
```

There are also plotting/benchmark utilities (e.g., `plot.py`, `gradgrad_vs_hessian.py`).

### `ex1_GPE_simple`
```bash
cd ex2_GPE_time_dependent
python plot.py
```

Many scripts save artifacts such as:
- trained model weights (`.pth`)
- training history (`*_history.csv`)
- parameter snapshots (`*_param.json`)

These are typically stored in `models/` subfolders created by the scripts.

---

## Notes on structure & conventions

- Most solvers follow a pattern:
  1. Define a **domain** (collocation points, boundaries, potential, coupling constants, etc.)
  2. Define a **model** (MLP architecture: layers, neurons, activations)
  3. Define a **training config** (optimizer choice: Adam / L-BFGS, stopping criteria, logging)
  4. Train and save results

- The PINN loss typically encodes:
  - PDE residual terms at sampled collocation points
  - boundary condition penalties (Dirichlet / periodic / custom constraints)
  - sometimes normalization or physics constraints (depending on the exercise)

---

## Results / Reports

The final report is available here: [report/main.pdf](report/main.pdf). 
