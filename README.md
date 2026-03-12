# PyOscProb

Python package for computing **neutrino oscillation probabilities** in media modeled as **slabs of constant density**.

The code uses:

- **PyTorch** for efficient **vectorized Hamiltonian diagonalisation** over many neutrino energies.
- **Astropy Units** (`astropy.units`) to provide a **clear and safe physical interface** for energies, distances, and densities.

---

## Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yaml
```

### 2. Install PyOscProb as editable package

```bash
conda activate PyOscProb_env
pip install -e .
```

3, 2, 1 Oscillate!


