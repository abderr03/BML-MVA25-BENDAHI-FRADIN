"""
effective_rank_experiment.py
============================
Tracks the effective rank of BB^T (gradient noise covariance) as a function
of Adam training steps, for MLPs of varying depth on California Housing
and Adult Income.

One Adam run per (seed, depth), checkpointed at each milestone; BB^T is
estimated via Monte Carlo and effective rank is the number of eigenvalues
needed to explain a given fraction of variance.

Usage
-----
    python effective_rank_experiment.py   # requires central_code.py in same folder
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

import torch
import torch.nn as nn

from central_code import (
    Dataset,
    NeuralNetModel,
    load_california,
    load_adult,
    estimate_noise_cov,
    explained_variance_threshold,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NETWORK FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_mlp_factory(input_dim: int, depth: int, hidden_dim: int = 16) -> callable:
    """Zero-argument factory → MLP with `depth` hidden Tanh layers."""
    def factory() -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)
    return factory


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EFFECTIVE RANK
# ══════════════════════════════════════════════════════════════════════════════

def effective_rank(M: np.ndarray, variance_explained: float = 0.99) -> int:
    """Smallest k s.t. top-k eigenvalues of M explain `variance_explained` of Tr(M)."""
    eigs = np.linalg.eigvalsh(M)
    _, k = explained_variance_threshold(eigs, variance_explained)
    return k


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SINGLE SEED RUN  —  one training, checkpointed
# ══════════════════════════════════════════════════════════════════════════════

def run_single_seed(
    dataset: Dataset,
    depth: int,
    hidden_dim: int,
    task: str,
    lam: float,
    training_steps: list[int],
    seed: int,
    S: int,
    n_mc_bbt: int,
    variance_explained: float,
) -> dict[int, int]:
    """
    Single Adam run from scratch (seed-controlled), pausing at each milestone
    to measure rank(BB^T).  Resumes training — never restarts.

    Returns  { n_steps: rank_BBT }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, D = dataset.X.shape

    factory         = make_mlp_factory(D, depth=depth, hidden_dim=hidden_dim)
    model           = NeuralNetModel(net_factory=factory, task=task, adam_lr=1e-3)
    net             = factory()
    model._net      = net
    model.nb_params = sum(p.numel() for p in net.parameters())

    X_t   = torch.tensor(dataset.X, dtype=torch.float32)
    y_t   = torch.tensor(dataset.y, dtype=torch.float32)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    milestones = sorted(set(training_steps))
    results: dict[int, int] = {}
    step = 0

    for target in milestones:
        # Resume training up to this milestone
        while step < target:
            optim.zero_grad()
            pred = net(X_t).squeeze(-1)
            data_loss = (
                0.5 * ((pred - y_t) ** 2).mean()
                if task == "regression"
                else nn.functional.binary_cross_entropy_with_logits(pred, y_t)
            )
            reg = 0.5 * lam * sum((p ** 2).sum() for p in net.parameters())
            (data_loss + reg).backward()
            optim.step()
            step += 1

        # Snapshot flat parameters
        theta = np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in net.parameters()])

        # BB^T and its effective rank
        BBT    = estimate_noise_cov(theta, dataset, model, S, lam=lam, n_mc=n_mc_bbt)
        rank_B = effective_rank(BBT, variance_explained)
        results[target] = rank_B
        print(f"      step={target:>6}  rank(BB^T)={rank_B:>4}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SWEEP  (depths × seeds)
# ══════════════════════════════════════════════════════════════════════════════

def sweep_effective_ranks(
    dataset: Dataset,
    depths: list[int],
    training_steps: list[int],
    seeds: list[int],
    task: str,
    lam: float,
    S: int,
    hidden_dim: int           = 16,
    n_mc_bbt: int             = 2_000,
    variance_explained: float = 0.99,
) -> dict:
    """
    Returns
    -------
    results[depth][n_steps] = np.ndarray of shape (n_seeds,)   — rank(BB^T) values
    """
    results: dict = {d: {t: [] for t in training_steps} for d in depths}
    total = len(depths) * len(seeds)
    run   = 0

    for depth in depths:
        for seed in seeds:
            run += 1
            print(f"\n  [{run}/{total}]  depth={depth}  seed={seed}")
            try:
                sr = run_single_seed(
                    dataset=dataset, depth=depth, hidden_dim=hidden_dim,
                    task=task, lam=lam, training_steps=training_steps,
                    seed=seed, S=S, n_mc_bbt=n_mc_bbt,
                    variance_explained=variance_explained,
                )
                for t in training_steps:
                    results[depth][t].append(sr[t])
            except Exception as e:
                print(f"    ERROR: {e}")
                for t in training_steps:
                    results[depth][t].append(np.nan)

    # Lists → arrays
    for depth in depths:
        for t in training_steps:
            results[depth][t] = np.array(results[depth][t], dtype=float)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

_COLORS  = ["#648FFF", "#FE6100", "#785EF0"]
_MARKERS = ["o", "s", "^"]
_BG      = "#f7f8fc"
_GRID    = "#dde1ee"


def _draw_panel(
    ax: plt.Axes,
    results: dict,
    training_steps: list[int],
    depths: list[int],
    n_seeds: int,
    variance_explained: float,
    title: str,
) -> None:
    """Draw one benchmark panel onto `ax`."""
    x = np.array(training_steps)

    ax.set_facecolor(_BG)
    ax.grid(True, linestyle="--", linewidth=0.45, color=_GRID, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aaaaaa")

    for i, depth in enumerate(depths):
        vals  = np.array([results[depth][t] for t in training_steps])   # (n_steps, n_seeds)
        means = np.nanmean(vals, axis=1)
        stds  = np.nanstd(vals,  axis=1)

        color  = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]

        ax.fill_between(x, means - stds, means + stds,
                        alpha=0.15, color=color, zorder=1)
        ax.plot(x, means,
                color=color, marker=marker, markersize=5.5,
                linewidth=2.0, label=f"{depth} hidden layer{'s' if depth > 1 else ''}",
                zorder=3)

    ax.set_xscale("log")
    ax.set_title(title, fontsize=10, fontweight="semibold", pad=5)
    ax.set_xlabel("Adam steps", fontsize=8.5)
    ax.set_ylabel(
        r"eff. rank of $BB^\top$"
        f"  (≥{int(100*variance_explained)}% var.)",
        fontsize=8.5,
    )
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(labelsize=8)


def build_legend_handles(depths: list[int]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=_COLORS[i % len(_COLORS)],
               marker=_MARKERS[i % len(_MARKERS)], markersize=6,
               linewidth=2.0,
               label=f"{d} hidden layer{'s' if d > 1 else ''}")
        for i, d in enumerate(depths)
    ]


def plot_all(
    res_cal: dict,
    res_adult: dict,
    training_steps: list[int],
    depths: list[int],
    n_seeds: int,
    variance_explained: float,
    save_combined: str  = "effective_rank_combined.pdf",
    save_cal:      str  = "effective_rank_california.pdf",
    save_adult:    str  = "effective_rank_adult.pdf",
    dpi: int            = 220,
) -> plt.Figure:
    """
    Build one figure with two side-by-side panels (one per benchmark).
    Also saves each panel individually.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.2),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    _draw_panel(axes[0], res_cal,   training_steps, depths, n_seeds,
                variance_explained, title="California Housing")
    _draw_panel(axes[1], res_adult, training_steps, depths, n_seeds,
                variance_explained, title="Adult Income")

    # Shared legend below both panels
    handles = build_legend_handles(depths)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(depths),
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.07),
    )

    # ── Save combined ─────────────────────────────────────────────────────
    fig.savefig(save_combined, dpi=dpi, bbox_inches="tight")
    print(f"  Saved combined  -> {save_combined}")

    # ── Save California panel individually ────────────────────────────────
    fig_cal, ax_cal = plt.subplots(figsize=(6, 4.2))
    fig_cal.patch.set_facecolor("white")
    _draw_panel(ax_cal, res_cal, training_steps, depths, n_seeds,
                variance_explained, title="California Housing")
    ax_cal.legend(handles=build_legend_handles(depths),
                  fontsize=8.5, framealpha=0.9, loc="best")
    fig_cal.savefig(save_cal, dpi=dpi, bbox_inches="tight")
    plt.close(fig_cal)
    print(f"  Saved California -> {save_cal}")

    # ── Save Adult panel individually ─────────────────────────────────────
    fig_ad, ax_ad = plt.subplots(figsize=(6, 4.2))
    fig_ad.patch.set_facecolor("white")
    _draw_panel(ax_ad, res_adult, training_steps, depths, n_seeds,
                variance_explained, title="Adult Income")
    ax_ad.legend(handles=build_legend_handles(depths),
                 fontsize=8.5, framealpha=0.9, loc="best")
    fig_ad.savefig(save_adult, dpi=dpi, bbox_inches="tight")
    plt.close(fig_ad)
    print(f"  Saved Adult      -> {save_adult}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    DEPTHS         = [1, 2, 3]
    HIDDEN_DIM     = 16
    TRAINING_STEPS = list(range(50, 1001, 50)) # [50, 100, 250, 300, 500, 1_000] #, 2_500, 5_000, 10_000]
    SEEDS          = [0, 1, 2, 3, 4]
    S              = 64
    N_MC_BBT       = 1_000    # raise to ~5_000 for publication quality
    LAM            = 1e-2
    VARIANCE_EXP   = 0.99

    print("Loading datasets ...")
    ds_cal   = load_california(normalize=True)
    ds_adult = load_adult(normalize=True)

    # ── California ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  California Housing\n{'='*60}")
    res_cal = sweep_effective_ranks(
        dataset=ds_cal, depths=DEPTHS, training_steps=TRAINING_STEPS,
        seeds=SEEDS, task="regression", lam=LAM, S=S,
        hidden_dim=HIDDEN_DIM, n_mc_bbt=N_MC_BBT,
        variance_explained=VARIANCE_EXP,
    )

    # ── Adult Income ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  Adult Income\n{'='*60}")
    res_adult = sweep_effective_ranks(
        dataset=ds_adult, depths=DEPTHS, training_steps=TRAINING_STEPS,
        seeds=SEEDS, task="binary_classification", lam=LAM, S=S,
        hidden_dim=HIDDEN_DIM, n_mc_bbt=N_MC_BBT,
        variance_explained=VARIANCE_EXP,
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plot_all(
        res_cal=res_cal, res_adult=res_adult,
        training_steps=TRAINING_STEPS, depths=DEPTHS,
        n_seeds=len(SEEDS), variance_explained=VARIANCE_EXP,
    )
    plt.show()
    print("\nDone.")