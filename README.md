# BML — Constant-SGD and Effective Rank Experiments

This folder contains two main scripts for Bayesian-style analysis of constant-step SGD and gradient noise:

- **`central_code.py`** — Runs all experiments: SGD-sampler comparisons and gradient-noise diagnostics.
- **`effective_rank_experiment.py`** — Studies how the effective rank of the gradient noise covariance \(BB^\top\) evolves during Adam training for MLPs of different depths.

Both rely on the same core types and utilities (datasets, models, \(BB^\top\) estimation) defined in `central_code.py`.

---

## 1. `central_code.py`

**Purpose:** Implements constant-step SGD as a Bayesian sampler and runs a full suite of experiments in one go.

### What it does

- **Models:** Linear regression, logistic regression, and PyTorch MLPs (regression and classification), all behind a common `ModelInterface` (MAP, Hessian, stochastic gradient).
- **Datasets:** Wine, Skin, Protein (CASP), California Housing, Adult Income; optional Digits with PCA. Data are normalized or scaled as needed.
- **Gradient noise:** Computes \(BB^\top = S \cdot \text{Cov}(\hat{g}_{\mathcal{S}})\) exactly (over all samples) or via Monte Carlo when `n_mc` is set.
- **SGD variants:**
  - **Constant SGD** — step size \(\varepsilon^* = 2DS/(N\,\text{Tr}(BB^\top))\).
  - **Constant SGD-d** — diagonal preconditioner from \(BB^\top\).
  - **Constant SGD-f** — full preconditioner (ridge-regularized inverse of \(BB^\top\)).
- **Theory:** Stationary covariance of the OU process and KL divergence between the SGD stationary distribution and the Laplace posterior.
- **Outputs:**
  - KL table for all dataset/model pairs and the three SGD variants.
  - Figure comparing posterior vs SGD iterates (e.g. California Housing, Adult Income) saved as `figure_california_income.pdf`.
  - Gradient-noise norm histograms and log-log CCDFs for a long list of dataset/model/batch-size configurations (linear, logistic, MLPs; optionally with bias).

### How to run

From the project folder:

```bash
python central_code.py
```

No arguments; it runs all experiments and writes the figure and per-configuration PDFs (e.g. `*_gaussian_test.pdf`) into the current directory.

### Dependencies

- NumPy, SciPy, pandas, matplotlib, scikit-learn, tqdm.
- PyTorch (and optionally `torchinfo`) for neural networks.

---

## 2. `effective_rank_experiment.py`

**Purpose:** Measures how the **effective rank** of \(BB^\top\) (number of eigenvalues needed to explain a given fraction of total variance, e.g. 99%) changes as a function of **Adam training steps**, for MLPs of different **depths** on two benchmarks.

### What it does

- **Benchmarks:** California Housing (regression), Adult Income (binary classification).
- **Setup:** One Adam run per (seed, depth). Training is checkpointed at a fixed set of step counts (e.g. 50, 100, …, 1000). At each checkpoint:
  - Current parameters \(\theta\) are read.
  - \(BB^\top\) is estimated at \(\theta\) using a Monte Carlo estimate (minibatches of size \(S\), `n_mc` samples) with regularization \(\lambda\).
  - Effective rank = smallest \(k\) such that the top-\(k\) eigenvalues of \(BB^\top\) explain \(\geq\) a chosen variance fraction (e.g. 99%).
- **Outputs:** One figure with two panels (California Housing, Adult Income), plus separate PDFs per benchmark. Plots show mean ± std of effective rank over seeds vs Adam steps (log scale), one curve per depth.

### How to run

From the project folder (with `central_code.py` in the same directory):

```bash
python effective_rank_experiment.py
```

Configurable at the top of `if __name__ == "__main__"`: depths, training steps, seeds, batch size \(S\), `n_mc` for \(BB^\top\), \(\lambda\), and variance-explained fraction.

### Dependencies

- Same as `central_code.py` (NumPy, matplotlib, PyTorch, etc.).
- **Imports:** `Dataset`, `NeuralNetModel`, `load_california`, `load_adult`, `estimate_noise_cov`, `explained_variance_threshold` from **`central_code`** (not from any other module).

---

## Summary

| Script                     | Role                                                                 |
|----------------------------|----------------------------------------------------------------------|
| **central_code.py**        | Single entry point to run all SGD-sampler and gradient-noise experiments. |
| **effective_rank_experiment.py** | Tracks effective rank of \(BB^\top\) vs Adam steps and depth; depends on `central_code.py`. |

Data files (e.g. UCI) are downloaded automatically into the current directory when first needed.
