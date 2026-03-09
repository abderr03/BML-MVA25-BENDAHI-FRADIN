"""Constant-SGD as Bayesian sampler. Runs all experiments: SGD variants + gradient-noise diagnostics."""

from __future__ import annotations
import os
import urllib.request
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import minimize
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from tqdm.auto import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd.functional import hessian as torch_hessian
    from torch.nn.utils.stateless import functional_call
except ImportError:
    torch = None
    nn = None
    F = None
    torch_hessian = None
    functional_call = None

try:
    from torchinfo import summary as torchinfo_summary
except ImportError:
    torchinfo_summary = None

warnings.filterwarnings("ignore")


def make_mlp(D_in: int, K: int = 1, hidden_layers: list[int] = None, activation=None, output_activation=None):
    """Module-level MLP factory for use in main (requires torch)."""
    if nn is None:
        raise ImportError("torch required for make_mlp")
    if hidden_layers is None:
        hidden_layers = []
    if activation is None:
        activation = nn.GELU()
    layers = []
    prev_dim = D_in
    for h in hidden_layers:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation)
        prev_dim = h
    layers.append(nn.Linear(prev_dim, K))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray


class ModelInterface(ABC):
    @abstractmethod
    def init_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def loss_and_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]: ...

    @abstractmethod
    def stoch_grad(self, theta: np.ndarray, Xb: np.ndarray, yb: np.ndarray, N: int, lam: float) -> np.ndarray: ...

    @abstractmethod
    def hessian(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray: ...

    def find_map(self, X: np.ndarray, y: np.ndarray, lam: float, **kw) -> np.ndarray:
        res = minimize(
            lambda th: self.loss_and_grad(th, X, y, lam),
            self.init_params(X, y), jac=True, method="L-BFGS-B",
            options={"maxiter": kw.get("max_iter", 10_000), "gtol": kw.get("tol", 1e-10)},
        )
        return res.x


class LinearModel(ModelInterface):
    def __init__(self, bias: bool = False):
        self.bias = bias

    def _aug(self, X: np.ndarray) -> np.ndarray:
        if not self.bias:
            return X
        return np.concatenate([X, np.ones((len(X), 1))], axis=1)

    def _lam_vec(self, D_aug: int, lam: float) -> np.ndarray:
        v = np.full(D_aug, lam)
        if self.bias:
            v[-1] = 0.0
        return v

    def init_params(self, X, y):
        return np.zeros(X.shape[1] + int(self.bias))

    def loss_and_grad(self, theta, X, y, lam):
        Xa = self._aug(X)
        lv = self._lam_vec(len(theta), lam)
        N = len(y)
        r = Xa @ theta - y
        loss = 0.5 * np.dot(r, r) / N + 0.5 * np.dot(lv * theta, theta)
        grad = Xa.T @ r / N + lv * theta
        return float(loss), grad

    def stoch_grad(self, theta, Xb, yb, N, lam):
        Xa = self._aug(Xb)
        lv = self._lam_vec(len(theta), lam)
        r = Xa @ theta - yb
        return Xa.T @ r / len(yb) + lv * theta

    def hessian(self, theta, X, y, lam):
        Xa = self._aug(X)
        lv = self._lam_vec(len(theta), lam)
        return Xa.T @ Xa / len(X) + np.diag(lv)

    def find_map(self, X, y, lam, **kw):
        Xa = self._aug(X)
        lv = self._lam_vec(Xa.shape[1], lam)
        return np.linalg.solve(Xa.T @ Xa / len(X) + np.diag(lv), Xa.T @ y / len(X))


class LogisticModel(ModelInterface):
    def __init__(self, bias: bool = False):
        self.bias = bias

    def _aug(self, X: np.ndarray) -> np.ndarray:
        if not self.bias:
            return X
        return np.concatenate([X, np.ones((len(X), 1))], axis=1)

    def _lam_vec(self, D_aug: int, lam: float) -> np.ndarray:
        v = np.full(D_aug, lam)
        if self.bias:
            v[-1] = 0.0
        return v

    def init_params(self, X, y):
        return np.zeros(X.shape[1] + int(self.bias))

    def loss_and_grad(self, theta, X, y, lam):
        Xa  = self._aug(X)
        lv  = self._lam_vec(len(theta), lam)
        N   = len(y)
        p   = self._sigmoid(Xa @ theta)
        nll = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
        grad = Xa.T @ (p - y) / N + lv * theta
        return float(nll + 0.5 * np.dot(lv * theta, theta)), grad

    def stoch_grad(self, theta, Xb, yb, N, lam):
        Xa  = self._aug(Xb)
        lv  = self._lam_vec(len(theta), lam)
        p   = self._sigmoid(Xa @ theta)
        return Xa.T @ (p - yb) / len(yb) + lv * theta

    def hessian(self, theta, X, y, lam):
        Xa  = self._aug(X)
        lv  = self._lam_vec(len(theta), lam)
        p   = self._sigmoid(Xa @ theta)
        w   = p * (1 - p)
        return (Xa.T * w) @ Xa / len(X) + np.diag(lv)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

class NeuralNetModel(ModelInterface):
    def __init__(
        self,
        net_factory: Callable,
        task:            str   = "regression",
        adam_lr:         float = 1e-3,
        adam_steps:      int   = 5_000,
        hessian_approx:  str   = "full",
    ):
        if torch is None:
            raise ImportError("pip install torch (required for NeuralNetModel)")
        self.net_factory   = net_factory
        self.task          = task
        self.adam_lr       = adam_lr
        self.adam_steps    = adam_steps
        self.hessian_approx = hessian_approx
        self._net = None

    def _net_(self):
        if self._net is None:
            self._net = self.net_factory()
        return self._net

    def _load(self, theta: np.ndarray):
        net = self._net_()
        idx = 0
        for p in net.parameters():
            n = p.numel()
            p.data.copy_(torch.tensor(
                theta[idx:idx+n].reshape(p.shape), dtype=torch.float32))
            idx += n
        return net

    def _dump(self) -> np.ndarray:
        return np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self._net_().parameters()])

    def _n_params(self, X: np.ndarray) -> int:
        net = self._net_()
        with torch.no_grad():
            net(torch.zeros(1, X.shape[1], dtype=torch.float32))
        return sum(p.numel() for p in net.parameters())

    def _task_loss(self, net, X_t, y_t):
        pred = net(X_t)
        if self.task == "regression":
            return 0.5 * ((pred.squeeze(-1) - y_t) ** 2).mean()
        elif self.task == "binary_classification":
            return F.binary_cross_entropy_with_logits(pred.squeeze(-1), y_t)
        elif self.task == "multiclass":
            return F.cross_entropy(pred, y_t.long())
        else:
            raise ValueError(f"Unknown task: {self.task!r}")

    def init_params(self, X, y):
        self._n_params(X)
        return self._dump()

    def loss_and_grad(self, theta, X, y, lam):
        net = self._load(theta)
        X_t  = torch.tensor(X, dtype=torch.float32)
        y_t  = torch.tensor(y, dtype=torch.float32)
        for p in net.parameters():
            p.requires_grad_(True)
        loss = self._task_loss(net, X_t, y_t)
        if lam > 0.0:
            loss = loss + 0.5 * lam * sum(
                (p**2).sum() for p in net.parameters())
        loss.backward()
        grad = np.concatenate([
            p.grad.detach().cpu().numpy().ravel() for p in net.parameters()])
        net.zero_grad()
        return float(loss.item()), grad

    def stoch_grad(self, theta, Xb, yb, N, lam):
        net = self._load(theta)
        X_t  = torch.tensor(Xb, dtype=torch.float32)
        y_t  = torch.tensor(yb, dtype=torch.float32)
        for p in net.parameters():
            p.requires_grad_(True)
        loss = self._task_loss(net, X_t, y_t)
        if lam > 0.0:
            loss = loss + 0.5 * lam * sum(
                (p**2).sum() for p in net.parameters())
        loss.backward()
        grad = np.concatenate([
            p.grad.detach().cpu().numpy().ravel() for p in net.parameters()])
        net.zero_grad()
        return grad

    def hessian(self, theta, X, y, lam):
        theta = np.asarray(theta, dtype=np.float32).ravel()
        net = self.net_factory()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        def loss_fn(flat):
            params_dict, idx = {}, 0
            for name, p in net.named_parameters():
                n = p.numel()
                params_dict[name] = flat[idx:idx+n].view_as(p)
                idx += n
            pred = functional_call(net, params_dict, X_t)
            if self.task == "regression":
                loss = 0.5 * ((pred.squeeze(-1) - y_t)**2).mean()
            elif self.task == "binary_classification":
                loss = F.binary_cross_entropy_with_logits(pred.squeeze(-1), y_t)
            elif self.task == "multiclass":
                loss = F.cross_entropy(pred, y_t.long())
            return loss + 0.5 * lam * (flat**2).sum()

        th_flat = torch.tensor(theta, dtype=torch.float32, requires_grad=True)
        H_torch = torch_hessian(loss_fn, th_flat)
        return H_torch.detach().cpu().numpy()

    def find_map(self, X, y, lam, **kw):
        net = self._net_()
        X_t   = torch.tensor(X, dtype=torch.float32)
        y_t   = torch.tensor(y, dtype=torch.float32)
        optim = torch.optim.Adam(net.parameters(), lr=kw.get("adam_lr", self.adam_lr))
        steps = kw.get("adam_steps", self.adam_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps, eta_min=1e-5)
        pbar = tqdm(range(steps), desc="Steps")
        for step in pbar:
            optim.zero_grad()
            loss = self._task_loss(net, X_t, y_t) + 0.5 * lam * sum((p**2).sum() for p in net.parameters())
            loss.backward()
            optim.step()
            scheduler.step()
            fl = float(loss)
            if (step + 1) % 50 == 0:
                pbar.set_postfix({"loss": f"{fl:.6f}"})
        return self._dump()

    def predict(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        net = self._load(theta)
        X_t    = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = net(X_t)
        if self.task == "multiclass":
            return F.softmax(out, dim=-1).numpy()
        elif self.task == "binary_classification":
            return torch.sigmoid(out.squeeze(-1)).numpy()
        return out.squeeze(-1).numpy()

    def accuracy(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict(theta, X)
        if self.task == "multiclass":
            preds = probs.argmax(axis=1)
        else:
            preds = (probs > 0.5).astype(int)
        return float((preds == y.astype(int)).mean())


def _download(url: str, path: str) -> None:
    if not os.path.exists(path):
        print(f"  Downloading {path} ...")
        urllib.request.urlretrieve(url, path)

def _scale_unit_length(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    return X / np.where(norms == 0, 1.0, norms)

def _preprocess(X_raw, normalize, y_raw=None, normalize_y=False):
    X = StandardScaler().fit_transform(X_raw) if normalize else _scale_unit_length(X_raw)
    if y_raw is not None and normalize_y:
        y = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-12)
        return X, y
    return X, y_raw

def load_wine(normalize: bool = True) -> Dataset:
    path = "winequality-white.csv"
    _download("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", path)
    df = pd.read_csv(path, sep=";")
    X, y = _preprocess(df.iloc[:,:-1].values.astype(float), normalize, y_raw=df.iloc[:,-1].values.astype(float), normalize_y=False)
    return Dataset("Wine", X, y)

def load_skin(normalize: bool = True) -> Dataset:
    path = "Skin_NonSkin.txt"
    _download("https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt", path)
    df = pd.read_csv(path, sep="\t", header=None)
    X, y = _preprocess(df.iloc[:, :3].values.astype(float), normalize, (df.iloc[:, 3].values == 1).astype(float))
    return Dataset("Skin", X, y)

def load_protein(normalize: bool = True) -> Dataset:
    path = "CASP.csv"
    _download("https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv", path)
    df = pd.read_csv(path)
    X, y = _preprocess(df.iloc[:, 1:9].values.astype(float), normalize, df.iloc[:, 0].values.astype(float), normalize_y=False)
    return Dataset("Protein", X, y)

def load_california(normalize: bool = True) -> Dataset:
    data = fetch_california_housing()
    X, y = _preprocess(data.data.astype(float), normalize, data.target.astype(float), normalize_y=False)
    return Dataset("California", X, y)

def load_adult(normalize: bool = True) -> Dataset:
    train_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
    test_url  = ("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test")

    _download(train_url, "adult.data")
    _download(test_url,  "adult.test")

    cols = ["age","workclass","fnlwgt","education","education-num",
            "marital","occupation","relationship","race","sex",
            "capital-gain","capital-loss","hours-per-week","country","income"]

    train = pd.read_csv("adult.data", header=None, names=cols, skipinitialspace=True, na_values="?").dropna()
    test  = pd.read_csv("adult.test", header=None, names=cols, skipinitialspace=True, na_values="?", skiprows=1).dropna()
    test["income"] = test["income"].str.rstrip(".")  # test set has trailing dot

    df = pd.concat([train, test], ignore_index=True)

    numeric_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", "fnlwgt"]
    df["sex_bin"]      = (df["sex"] == "Male").astype(float)
    df["marital_bin"]  = df["marital"].isin(["Married-civ-spouse", "Married-AF-spouse"]).astype(float)
    df["race_white"]   = (df["race"] == "White").astype(float)
    df["us_born"]      = (df["country"] == "United-States").astype(float)
    df["self_emp"]     = df["workclass"].isin(["Self-emp-not-inc", "Self-emp-inc"]).astype(float)
    df["exec_mgr"]     = df["occupation"].isin(["Exec-managerial", "Prof-specialty"]).astype(float)
    df["husband_wife"] = df["relationship"].isin(["Husband", "Wife"]).astype(float)

    feature_cols = numeric_cols + ["sex_bin", "marital_bin", "race_white", "us_born", "self_emp", "exec_mgr", "husband_wife"]

    X, y = _preprocess(df[feature_cols].values.astype(float), normalize, (df["income"].str.contains(">50K")).astype(float).values)
    return Dataset("Adult Income", X, y)

def load_digits_dataset(normalize: bool = True, binary: bool = False, n_pca: int = None) -> Dataset:
    raw = load_digits()
    X, y = raw.data / 16.0, raw.target.astype(float)

    if binary:
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]
        name = "Digits (0 vs 1)"
    else:
        name = "Digits"

    if n_pca is not None:
        pca = PCA(n_components=n_pca)
        X   = pca.fit_transform(X)
        var = pca.explained_variance_ratio_.sum()
        print(f"  PCA({n_pca}): {100*var:.1f}% variance retained")
        name += f" PCA-{n_pca}"

    X = _preprocess(X, normalize)
    return Dataset(name, X, y)

def load_custom(name: str, X: np.ndarray, y: np.ndarray, normalize: bool = True) -> Dataset:
    return Dataset(name, _preprocess(X, normalize), y)


def estimate_noise_cov(
    theta: np.ndarray,
    dataset: Dataset,
    model: ModelInterface,
    S: int,
    lam: float = 0.0,
    n_mc: Optional[int] = None,
) -> np.ndarray:
    N = len(dataset.y)
    P  = model.nb_params
    F  = np.zeros((P, P))
    g0 = np.zeros(P)

    if n_mc is None:
        n_used = N
        desc = "Computing BB^T"
    else:
        n_used = n_mc
        desc = "Computing BB^T (MC)"

    for i in tqdm(range(n_used), desc=desc):
        if n_mc is None:
            idx = i
            Xb = dataset.X[idx : idx + 1]
            yb = dataset.y[idx : idx + 1]
            batch_N = 1
        else:
            batch_idx = np.random.randint(0, N, size=min(S, N))
            Xb = dataset.X[batch_idx]
            yb = dataset.y[batch_idx]
            batch_N = N
        g = model.stoch_grad(theta, Xb, yb, N=batch_N, lam=lam)
        g = np.asarray(g).ravel()
        g0 += g
        F += np.outer(g, g)

    g0 /= n_used
    c   = (N - S) / (N - 1.0) if N > 1 else 1.0
    BBT = c * (F / n_used - np.outer(g0, g0))
    return 0.5 * (BBT + BBT.T)

def estimate_BBT_eigenvalues(
    theta: np.ndarray, dataset: Dataset, model: ModelInterface, S: int,
    k: int = None, batch: int = 256,
) -> tuple:
    N, P = len(dataset.y), model.nb_params
    c = (N - S) / (N - 1.0)
    G = np.empty((N, P), dtype=np.float32)
    idx = 0
    for start in tqdm(range(0, N, batch), desc="Collecting gradients"):
        end = min(start + batch, N)
        for i in range(end - start):
            n = start + i
            G[idx] = model.stoch_grad(theta, dataset.X[n:n+1], dataset.y[n:n+1], N=1, lam=0.0)
            idx += 1
    g0 = G.mean(axis=0)
    if k is not None:
        G_centered = G - g0[None, :]
        _, svals, _ = randomized_svd(G_centered, n_components=k, n_iter=4, random_state=0)
        eigs = c * svals**2 / N
    elif N < P:
        G_centered = G - g0[None, :]
        gram = G_centered @ G_centered.T
        eigs_gram = np.linalg.eigvalsh(gram)
        eigs = np.maximum(c * eigs_gram / N, 0.0)
    else:
        G_centered = G - g0[None, :]
        BBT = c * (G_centered.T @ G_centered) / N
        BBT = 0.5 * (BBT + BBT.T)
        eigs = np.maximum(np.linalg.eigvalsh(BBT), 0.0)
    eigs = np.sort(eigs)[::-1]
    total = eigs.sum()
    rank_99 = int(np.searchsorted(np.cumsum(eigs) / total, 0.99)) + 1 if total > 0 else 0
    print(f"  BBT eigenvalues: max={eigs.max():.3e} rank(99% var)={rank_99}/{len(eigs)}")
    return eigs, g0


def ou_stationary_cov(A_eff: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Sigma = solve_continuous_lyapunov(A_eff, Q)
    Sigma = 0.5 * (Sigma + Sigma.T)
    lmin = np.linalg.eigvalsh(Sigma).min()
    if lmin < 0:
        Sigma -= 2 * lmin * np.eye(len(Sigma))
    return Sigma


def kl_divergence(Sigma: np.ndarray, A: np.ndarray, N: int) -> float:
    D = A.shape[0]
    sgn, ldNA = np.linalg.slogdet(N * A)
    if sgn <= 0:
        return float("nan")
    sgn, ldS = np.linalg.slogdet(Sigma)
    if sgn <= 0:
        return float("nan")
    return 0.5 * (N * np.trace(A @ Sigma) - ldNA - ldS - D)


def optimal_eps(S: int, D: int, N: int, BBT: np.ndarray) -> float:
    return 2.0 * D * S / (N * np.trace(BBT))


def optimal_H_diagonal(S: int, N: int, BBT: np.ndarray, eps_jitter: float = 1e-6) -> np.ndarray:
    diag_noise = np.diag(BBT) + eps_jitter
    return np.diag(2.0 * S / (N * diag_noise))


def optimal_H_fullrank(S: int, N: int, BBT: np.ndarray, eps_jitter: float = 1e-4) -> np.ndarray:
    D = BBT.shape[0]
    reg = eps_jitter * np.trace(BBT) / D * np.eye(D)
    return (2.0 * S / N) * np.linalg.inv(BBT + reg)


def run_sgd(
    theta_star: np.ndarray, dataset: Dataset, model: ModelInterface, S: int, lam: float,
    eps: float = 1.0, H: Optional[np.ndarray] = None,
    n_runs: int = 1, n_steps_per_run: int = 50_000, n_samples_per_run: int = 20_000,
    grad_clip: Optional[float] = None, log_every: int = 5_000,
) -> Optional[np.ndarray]:
    N = len(dataset.y)
    burn = n_steps_per_run - n_samples_per_run
    if burn < 0:
        raise ValueError("n_steps_per_run must be >= n_samples_per_run")
    collected = []
    for run in tqdm(range(n_runs), desc="Runs", position=0):
        theta = theta_star.copy()
        samples = np.empty((n_samples_per_run, len(theta)))
        s_idx = 0
        ok = True
        pbar = tqdm(range(n_steps_per_run), desc="Steps", position=1, leave=False)
        for step in pbar:
            idx = np.random.randint(0, N, S)
            g = model.stoch_grad(theta, dataset.X[idx], dataset.y[idx], N, lam)
            if grad_clip is not None:
                g_norm = np.linalg.norm(g)
                if g_norm > grad_clip:
                    g = g * (grad_clip / g_norm)
            theta = theta - (H @ g if H is not None else eps * g)
            if not np.all(np.isfinite(theta)):
                print(f"    !! run {run+1}: diverged at step {step}")
                ok = False
                break
            if log_every and step > 0 and step % log_every == 0:
                pbar.set_postfix({"drift": f"{np.linalg.norm(theta - theta_star):.4f}"})
            if step >= burn:
                samples[s_idx] = theta
                s_idx += 1
        if ok:
            collected.append(samples)
    return np.concatenate(collected, axis=0) if collected else None


def sample_posterior(
    theta_star: np.ndarray, A: np.ndarray, N: int, n_samples: int = 5_000, seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(N * A)
    z = rng.standard_normal((len(theta_star), n_samples))
    return theta_star + np.linalg.solve(L.T, z).T


@dataclass
class SGDConfig:
    n_runs: int = 1
    n_steps_per_run: int = 50_000
    n_samples_per_run: int = 20_000
    n_posterior: int = 5_000


def run_experiment(
    dataset: Dataset, model: ModelInterface, S: int, lam: float = 1.0, cfg: Optional[SGDConfig] = None,
) -> dict:
    np.random.seed(42)
    if cfg is None:
        cfg = SGDConfig()
    N, D = dataset.X.shape
    if isinstance(model, NeuralNetModel):
        model.nb_params = model._n_params(dataset.X)
    else:
        model.nb_params = D
    print(f"\n{'='*62}\n  {dataset.name}  |  N={N}, D={D}, S={S}, model={type(model).__name__}, lam={lam}\n{'='*62}")
    print("  Finding MAP …")
    theta_star = model.find_map(dataset.X, dataset.y, lam)
    lmap, gmap = model.loss_and_grad(theta_star, dataset.X, dataset.y, lam)
    print(f"      L at MAP = {lmap:.2e}  |grad| = {np.linalg.norm(gmap):.2e}")
    print("  Computing Hessian …")
    A = model.hessian(theta_star, dataset.X, dataset.y, lam)
    A = 0.5 * (A + A.T)
    lmin = np.linalg.eigvalsh(A).min()
    if lmin < 1e-12:
        A += (1e-8 - lmin) * np.eye(model.nb_params)
    eigs_A = np.linalg.eigvalsh(A)
    print(f"    eig(A): min={eigs_A.min():.3e} max={eigs_A.max():.3e}")
    print("  Computing BBT …")
    BBT = estimate_noise_cov(theta_star, dataset, model, S)
    eigs_BBT = np.linalg.eigvalsh(BBT)
    print(f"    Tr(BBT)={np.trace(BBT):.4e}  eig(BBT): min={eigs_BBT.min():.3e} max={eigs_BBT.max():.3e}")
    posterior_samples = sample_posterior(theta_star, A, N, cfg.n_posterior)
    eps_raw = optimal_eps(S, model.nb_params, N, BBT)
    tau_slow = 1.0 / max(eps_raw * eigs_A.min(), 1e-30)
    tau_fast = 1.0 / max(eps_raw * eigs_A.max(), 1e-30)
    print(f"    mixing time approx [{tau_fast:.1f}, {tau_slow:.1f}] steps")
    sgd_kw = dict(dataset=dataset, model=model, S=S, lam=lam, n_runs=cfg.n_runs,
                  n_steps_per_run=cfg.n_steps_per_run, n_samples_per_run=cfg.n_samples_per_run)
    variants = {}
    print("  [SGD] constant SGD …")
    eps1 = optimal_eps(S, model.nb_params, N, BBT)
    s1 = run_sgd(theta_star, eps=eps1, **sgd_kw)
    emp1 = np.cov(s1, rowvar=False) if s1 is not None else None
    th1 = ou_stationary_cov(A, eps1 * BBT / S)
    kl1 = kl_divergence(emp1 if emp1 is not None else th1, A, N)
    print(f"    eps* = {eps_raw:.3e}  used {eps1:.3e}  KL = {kl1:.4f}")
    variants["constant SGD"] = dict(samples=s1, sigma_empirical=emp1, sigma_theory=th1, kl=kl1, eps=eps1)
    print("  [SGD-d] …")
    Hd = optimal_H_diagonal(S, N, BBT)
    sd = run_sgd(theta_star, H=Hd, **sgd_kw)
    empd = np.cov(sd, rowvar=False) if sd is not None else None
    thd = ou_stationary_cov(Hd @ A, Hd @ BBT @ Hd.T / S)
    variants["constant SGD-d"] = dict(samples=sd, sigma_empirical=empd, sigma_theory=thd, kl=kl_divergence(empd if empd is not None else thd, A, N))
    print("  [SGD-f] …")
    Hf = optimal_H_fullrank(S, N, BBT)
    sf = run_sgd(theta_star, H=Hf, **sgd_kw)
    empf = np.cov(sf, rowvar=False) if sf is not None else None
    thf = ou_stationary_cov(Hf @ A, Hf @ BBT @ Hf.T / S)
    variants["constant SGD-f"] = dict(samples=sf, sigma_empirical=empf, sigma_theory=thf, kl=kl_divergence(empf if empf is not None else thf, A, N))
    return dict(name=dataset.name, model_name=type(model).__name__, N=N, D=D, S=S, lam=lam,
               theta_star=theta_star, A=A, BBT=BBT, posterior_samples=posterior_samples,
               mixing_times={"tau_slowest": tau_slow, "tau_fastest": tau_fast},
               variants=variants, model=model, dataset=dataset)


def explained_variance_threshold(eigs: np.ndarray, variance_explained: float = 0.99, squared: bool = False) -> tuple[float, int]:
    eigs_sorted = np.sort(np.maximum(eigs, 0))[::-1]
    if squared:
        eigs_sorted = eigs_sorted**2
    total = eigs_sorted.sum()
    if total <= 0:
        return 0.0, 0
    cumvar = np.cumsum(eigs_sorted) / total
    k = min(int(np.searchsorted(cumvar, variance_explained)) + 1, len(eigs_sorted))
    return float(eigs_sorted[k - 1]), k


def compute_jacobian(theta: np.ndarray, dataset: Dataset, model: ModelInterface, batch: int = 128) -> np.ndarray:
    N = len(dataset.y)
    P = model.nb_params
    if isinstance(model, LinearModel):
        return dataset.X.astype(np.float64)
    if isinstance(model, LogisticModel):
        p  = LogisticModel._sigmoid(dataset.X @ theta)   # (N,)
        w  = p * (1.0 - p)                               # σ'
        return (dataset.X * w[:, None]).astype(np.float64)
    if isinstance(model, NeuralNetModel):
        net = model._load(theta)
        net.eval()
        x0 = torch.tensor(dataset.X[:1], dtype=torch.float32)
        with torch.no_grad():
            out0 = net(x0)
        C = out0.shape[-1] if out0.ndim > 1 else 1

        m = N * C
        J = np.zeros((m, P), dtype=np.float64)

        for start in tqdm(range(0, N, batch), desc="Jacobian"):
            end  = min(start + batch, N)
            X_b  = torch.tensor(dataset.X[start:end], dtype=torch.float32)
            b    = end - start

            for i in range(b):
                x_i = X_b[i:i+1]   # (1, D)

                def f_flat(flat_theta):
                    pd, idx = {}, 0
                    for name, p in net.named_parameters():
                        n = p.numel()
                        pd[name] = flat_theta[idx:idx+n].view_as(p)
                        idx += n
                    out = functional_call(net, pd, x_i)
                    return out.squeeze(0)   # (C,) or scalar

                jac = torch.autograd.functional.jacobian(
                    f_flat,
                    torch.tensor(theta, dtype=torch.float32,
                                 requires_grad=True),
                    vectorize=True,
                )
                if jac.ndim == 1:
                    jac = jac.unsqueeze(0)
                J[(start + i) * C:(start + i + 1) * C, :] = jac.detach().cpu().numpy()
        return J
    raise NotImplementedError(f"compute_jacobian not implemented for {type(model).__name__}")

def project_BBT_to_tangent(
    BBT: np.ndarray, J: np.ndarray, theta_star: np.ndarray = None,
    rank: int = None, tol: float = 1e-8, verbose: bool = True,
) -> dict:
    m, P = J.shape
    if m >= P:
        _, s, Vt = np.linalg.svd(J, full_matrices=False)
        V_full = Vt.T
    else:
        K = J @ J.T
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        s = np.sqrt(np.maximum(eigvals, 0.0))
        nonzero = s > tol * s[0]
        U_nz = eigvecs[:, nonzero]
        s_nz = s[nonzero]
        V_full = J.T @ (U_nz / s_nz[None, :])
        V_full, _ = np.linalg.qr(V_full)
        s = s_nz
    if rank is not None:
        r = min(int(rank), V_full.shape[1])
    else:
        r = int((s > tol * s[0]).sum())
        r = max(r, 1)

    V = V_full[:, :r]
    singulars = s[:r]
    VtB = V.T @ BBT
    BBT_red = VtB @ V
    BBT_proj = V @ BBT_red @ V.T
    tr_total = np.trace(BBT)
    tr_tangent = np.trace(BBT_red)
    leakage = 1.0 - tr_tangent / max(tr_total, 1e-30)
    theta_red = V.T @ theta_star if theta_star is not None else None
    if verbose:
        eigs_red = np.linalg.eigvalsh(BBT_red)
        print(f"\n  Tangent: J {J.shape} rank {r}/{P}  Tr(BBT) {tr_total:.4e} in tangent {tr_tangent:.4e}  leakage {100*leakage:.2f}%")
        if leakage > 0.01:
            print("  !! Leakage > 1%")

    return dict(
        V         = V,
        BBT_red   = BBT_red,
        BBT_proj  = BBT_proj,
        r         = r,
        theta_red = theta_red,
        singulars = singulars,
        leakage   = leakage,
    )

def run_sgd_tangent(
    theta_star: np.ndarray, V: np.ndarray, A_tan: np.ndarray, BBT_tan: np.ndarray,
    dataset: Dataset, model: ModelInterface, S: int, lam: float,
    eps: float = None, H_tan: np.ndarray = None,
    n_runs: int = 1, n_steps_per_run: int = 50_000, n_samples_per_run: int = 20_000,
    grad_clip: float = None, log_every: int = 5_000,
) -> np.ndarray:
    N = len(dataset.y)
    r = V.shape[1]
    burn = n_steps_per_run - n_samples_per_run
    collected = []
    for run in tqdm(range(n_runs), desc="Runs"):
        phi = np.zeros(r)
        samples = np.empty((n_samples_per_run, r))
        s_idx = 0
        ok = True
        pbar = tqdm(range(n_steps_per_run), desc="Steps", position=1, leave=False)
        for step in pbar:
            theta = theta_star + V @ phi
            idx = np.random.randint(0, N, S)
            g_full = model.stoch_grad(theta, dataset.X[idx], dataset.y[idx], N, lam)
            g_tan = V.T @ g_full
            if grad_clip is not None:
                gnorm = np.linalg.norm(g_tan)
                if gnorm > grad_clip:
                    g_tan = g_tan * (grad_clip / gnorm)
            phi = phi - (H_tan @ g_tan if H_tan is not None else eps * g_tan)
            if not np.all(np.isfinite(phi)):
                print(f"  run {run+1}: diverged at step {step}")
                ok = False
                break
            if log_every and step > 0 and step % log_every == 0:
                pbar.set_postfix({"drift": f"{np.linalg.norm(theta - theta_star):.4f}"})
            if step >= burn:
                samples[s_idx] = phi
                s_idx += 1
        if ok:
            collected.append(samples)
    return np.concatenate(collected, axis=0) if collected else None


def _cov_ellipse(ax, center, cov2d, n_std=2.0, **kw):
    vals, vecs = np.linalg.eigh(cov2d)
    vals = np.maximum(vals, 0.0)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse(xy=center, width=2*n_std*np.sqrt(vals[0]), height=2*n_std*np.sqrt(vals[1]), angle=angle, **kw))


def _pick_pc_directions(post_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(post_cov)
    return vecs[:, -1], vecs[:, 0]


def _draw_panel(ax, theta_star, post_samples, sgd_samples, sigma_theory, sigma_empirical, post_cov, dims, n_std, max_scatter, title):
    v_small, v_large = dims
    mu = np.zeros(2)
    V = np.column_stack([v_small, v_large])
    ax.set_facecolor("#eef2ff")

    def proj(s):
        c = s - theta_star
        return np.column_stack([c @ v_small, c @ v_large])

    rng = np.random.default_rng(7)
    ps = proj(post_samples)
    ax.scatter(ps[rng.choice(len(ps), min(max_scatter, len(ps)), replace=False), 0],
               ps[rng.choice(len(ps), min(max_scatter, len(ps)), replace=False), 1],
               c="#3355dd", marker="x", s=12, alpha=0.35, linewidths=0.7, zorder=2, label="posterior")
    label = "iterates"
    if sgd_samples is not None:
        ss = proj(sgd_samples)
    else:
        try:
            L = np.linalg.cholesky(sigma_theory + 1e-10*np.eye(len(sigma_theory)))
            z = rng.standard_normal((max_scatter, len(sigma_theory)))
            ss = proj(theta_star + z @ L.T)
            label = "iterates (theory)"
        except np.linalg.LinAlgError:
            ss = None
    if ss is not None:
        idx = rng.choice(len(ss), min(max_scatter, len(ss)), replace=False)
        ax.scatter(ss[idx, 0], ss[idx, 1], c="#00bb99", marker="+", s=14, alpha=0.35, linewidths=0.7, zorder=3, label=label)
    _cov_ellipse(ax, mu, V.T @ post_cov @ V, n_std=n_std, fill=False, edgecolor="black", linestyle="--", linewidth=1.8, zorder=5)
    _cov_ellipse(ax, mu, V.T @ sigma_theory @ V, n_std=n_std, fill=False, edgecolor="#ee2222", linestyle="--", linewidth=1.8, zorder=5)
    if sigma_empirical is not None:
        _cov_ellipse(ax, mu, V.T @ sigma_empirical @ V, n_std=n_std, fill=False, edgecolor="#ddaa00", linestyle="-", linewidth=1.8, zorder=6)
    ax.scatter(*mu, c="black", s=30, zorder=7)
    ax.set_title(title, fontsize=10, pad=5)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Smallest PC", fontsize=8)
    ax.set_ylabel("Largest PC", fontsize=8)


def plot_figure1(
    experiments: list[dict], methods: tuple[str, ...] = ("constant SGD", "constant SGD-d", "constant SGD-f"),
    n_std: float = 2.0, max_scatter: int = 3_000, figsize: Optional[tuple] = None,
    save_path: Optional[str] = None, dpi: int = 150,
) -> Optional[plt.Figure]:
    if not experiments:
        return None
    n_rows, n_cols = len(experiments), len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (4.8*n_cols, 4.4*n_rows), squeeze=False)
    fig.patch.set_facecolor("white")
    for row, exp in enumerate(experiments):
        post_cov = np.linalg.inv(exp["N"] * exp["A"])
        dims = _pick_pc_directions(post_cov)
        for col, method in enumerate(methods):
            var = exp["variants"][method]
            _draw_panel(ax=axes[row][col], theta_star=exp["theta_star"], post_samples=exp["posterior_samples"],
                        sgd_samples=var["samples"], sigma_theory=var["sigma_theory"],
                        sigma_empirical=var.get("sigma_empirical"), post_cov=post_cov,
                        dims=dims, n_std=n_std, max_scatter=max_scatter, title=f"{method}\n({exp['name']})")
    axes[0][0].legend(handles=[
        Line2D([0],[0], marker="x", color="#3355dd", linestyle="None", markersize=8, label="posterior"),
        Line2D([0],[0], marker="+", color="#00bb99", linestyle="None", markersize=8, label="iterates"),
        Line2D([0],[0], color="black", linestyle="--", lw=1.5, label="posterior ellipse"),
        Line2D([0],[0], color="#ddaa00", linestyle="-", lw=1.5, label="empirical cov"),
        Line2D([0],[0], color="#ee2222", linestyle="--", lw=1.5, label="OU theory ellipse"),
    ], fontsize=7.5, loc="upper left", framealpha=0.85)
    plt.tight_layout(pad=1.5)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Figure saved -> {save_path}")
    return fig


def plot_gradient_noise(
    model: ModelInterface, theta_star: np.ndarray, dataset: Dataset,
    BBT_eigs: np.ndarray, grad_mean: np.ndarray, S: int = 100,
    n_samples: int = 50_000, n_pca: int = 5_000, batch_size: int = 512,
    save_path: str = None, figsize: tuple = (11, 3.2), noise_norm_only: bool = False,
) -> plt.Figure:
    N, P = len(dataset.y), len(theta_star)
    n_pca = min(n_pca, n_samples)
    grad_norms = np.empty(n_samples, dtype=np.float64)
    grads_pca = np.empty((n_pca, P), dtype=np.float32)
    for i in tqdm(range(n_samples), desc="Sample gradient minibatches"):
        idx = np.random.choice(N, S, replace=False)
        g = model.stoch_grad(theta_star, dataset.X[idx], dataset.y[idx], N, lam=0.0)
        gc = g - grad_mean
        grad_norms[i] = np.linalg.norm(gc)
        if i < n_pca:
            grads_pca[i] = gc.astype(np.float32)
    if not noise_norm_only:
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(grads_pca.astype(np.float64))[:, 0]
        pc1_n = pc1 / (pc1.std() + 1e-12)
        del grads_pca
    k = min(len(BBT_eigs), P)
    eigs_full = np.zeros(P, dtype=np.float32)
    eigs_full[:k] = np.sort(np.maximum(BBT_eigs[:k], 0.0).astype(np.float32))[::-1]
    eigs_full /= float(S)
    sqrt_eigs = np.sqrt(eigs_full)
    norms_gauss = np.empty(n_samples, dtype=np.float64)
    rng = np.random.default_rng(0)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        z_b = rng.standard_normal((end - start, P)).astype(np.float32)
        norms_gauss[start:end] = np.linalg.norm(z_b * sqrt_eigs, axis=1)
    has_ref = norms_gauss.std() > 1e-10
    C = {"data": "#4477CC", "ref": "#009988", "norm": "#AA3377"}
    t_std = np.linspace(-5, 5, 400)
    fig, axes = plt.subplots(1, 2 if noise_norm_only else 4, figsize=figsize)
    i = 0
    if not noise_norm_only:
        i = 2
        ax = axes[0]
        ax.hist(pc1_n, bins=80, density=True, color=C["data"], alpha=0.35, edgecolor="none", label="Empirical")
        ax.plot(t_std, scipy_stats.norm.pdf(t_std), color=C["ref"], lw=1.5, ls="--", label="N(0,1)")
        ax.set_title(f"Gradient noise PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)", fontsize=8.5)
        ax.set_xlabel("Gradient noise on PC 1", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_xlim(-5, 5)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.tick_params(labelsize=7)
        ax = axes[1]
        (osm, osr), (slope, intercept, _) = scipy_stats.probplot(pc1_n, dist="norm")
        ax.scatter(osm, osr, s=2.0, alpha=0.25, color=C["data"], rasterized=True)
        ax.plot(osm, slope * np.array(osm) + intercept, color=C["ref"], lw=0.8)
        ax.set_title("QQ plot (PC 1)", fontsize=8.5)
        ax.set_xlabel("Theoretical quantiles", fontsize=8)
        ax.set_ylabel("Sample quantiles", fontsize=8)
        ax.tick_params(labelsize=7)
    ax = axes[i]
    tmax = np.percentile(grad_norms, 99.5)
    t_n = np.linspace(1e-12, tmax, 400)
    ax.hist(grad_norms, bins=100, density=True, color=C["norm"], alpha=0.35, edgecolor="none", label="Empirical")
    if has_ref:
        ax.plot(t_n, gaussian_kde(norms_gauss, bw_method="scott")(t_n), color=C["ref"], lw=1.5, ls="--", label="Gaussian ref")
    ax.set_title("Gradient-noise norm", fontsize=8.5)
    ax.set_xlabel("Norm of gradient noise", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_xlim(0, tmax)
    ax.legend(fontsize=7, framealpha=0.7, loc="upper right")
    ax.tick_params(labelsize=7)
    ax = axes[i + 1]
    sn = np.sort(grad_norms)[::-1]
    ranks = np.arange(1, len(sn) + 1) / len(sn)
    ax.loglog(sn, ranks, color=C["norm"], lw=1.2, label="Empirical CCDF")
    if has_ref:
        sn_g = np.sort(norms_gauss)[::-1]
        ax.loglog(sn_g, np.arange(1, len(sn_g) + 1) / len(sn_g), color=C["ref"], lw=1.2, ls="--", label="Gaussian ref")
    ax.set_title("Log-log tail (CCDF)", fontsize=8.5)
    ax.set_xlabel("Norm of gradient noise", fontsize=8)
    ax.set_ylabel("P(norm > t)", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.show()
    return fig


def print_table(experiments: list[dict]) -> None:
    names = [f"{e['name']} ({type(e['model']).__name__[:6]})" for e in experiments]
    W     = 60 + 20 * len(experiments)
    
    print(f"\n{'═'*W}\n  Table 1 — KL Divergences  (KL / KL per dim)\n{'═'*W}")
    print(f"  {'Method':<22}" + "".join(f"  {n:>22}" for n in names))
    print("─" * W)
    
    for m in ["constant SGD", "constant SGD-d", "constant SGD-f"]:
        row = []
        for e in experiments:
            kl   = e["variants"][m]["kl"]
            d    = e.get("d_joint", e.get("D"))   # subspace dim or full dim
            kpd  = kl / max(d, 1)
            if np.isfinite(kl):
                row.append(f"{kl:.4f} / {kpd:.4f}")
            else:
                row.append("n/a")
        print("  " + f"{m:<22}" + "".join(f"  {r:>22}" for r in row))
        print("─" * W)
    
    print("  " + f"{'dim used':<22}" + "".join(
        f"  {e.get('d_joint', e.get('D')):>22}" for e in experiments))
    print("═" * W)


if __name__ == "__main__":
    print("Loading datasets ...")
    ds_wine = load_wine()
    ds_skin = load_skin()
    ds_protein = load_protein()
    ds_california = load_california()
    ds_adults = load_adult()

    cfg = SGDConfig(n_runs=10, n_steps_per_run=5_000, n_samples_per_run=100, n_posterior=5_000)
    print("\n" + "=" * 60 + "\n  Main experiments\n" + "=" * 60)
    exp_wine = run_experiment(ds_wine, LinearModel(), S=100, lam=1.0, cfg=cfg)
    exp_skin = run_experiment(ds_skin, LogisticModel(), S=1_000, lam=1.0, cfg=cfg)
    exp_protein = run_experiment(ds_protein, LinearModel(), S=100, lam=1.0, cfg=cfg)
    exp_california = run_experiment(ds_california, LinearModel(), S=100, lam=1.0, cfg=cfg)
    exp_adults = run_experiment(ds_adults, LogisticModel(), S=100, lam=1.0, cfg=cfg)
    print_table([exp_wine, exp_skin, exp_protein, exp_california, exp_adults])
    plot_figure1(experiments=[exp_california, exp_adults], save_path="figure_california_income.pdf", dpi=300,
                 methods=("constant SGD", "constant SGD-d", "constant SGD-f"))
    _d = [
        (ds_wine, LinearModel(), 100, 1.0), (ds_wine, LinearModel(bias=True), 100, 1.0),
        (ds_skin, LogisticModel(), 1_000, 1.0), (ds_skin, LogisticModel(bias=True), 1_000, 1.0),
        (ds_protein, LinearModel(), 100, 1.0), (ds_protein, LinearModel(bias=True), 100, 1.0),
        (ds_california, LinearModel(), 100, 1.0), (ds_california, LinearModel(bias=True), 100, 1.0),
        (ds_adults, LogisticModel(), 100, 1.0), (ds_adults, LogisticModel(bias=True), 100, 1.0),
        (ds_wine, NeuralNetModel(lambda: make_mlp(D_in=11, K=1, hidden_layers=[16], activation=nn.ReLU()), task="regression", adam_lr=3e-3, adam_steps=30_000, hessian_approx="full"), 100, 4e-2),
        (ds_wine, NeuralNetModel(lambda: make_mlp(D_in=11, K=1, hidden_layers=[16, 16], activation=nn.ReLU()), task="regression", adam_lr=3e-3, adam_steps=30_000, hessian_approx="full"), 100, 4e-2),
        (ds_wine, NeuralNetModel(lambda: make_mlp(D_in=11, K=1, hidden_layers=[16, 16, 16], activation=nn.ReLU()), task="regression", adam_lr=3e-3, adam_steps=30_000, hessian_approx="full"), 100, 4e-2),
        (ds_skin, NeuralNetModel(lambda: make_mlp(D_in=3, K=1, hidden_layers=[8], activation=nn.ReLU()), task="binary_classification", adam_lr=3e-3, adam_steps=30_000, hessian_approx="full"), 1_000, 4e-2),
        (ds_skin, NeuralNetModel(lambda: make_mlp(D_in=3, K=1, hidden_layers=[8, 8], activation=nn.ReLU()), task="binary_classification", adam_lr=3e-3, adam_steps=30_000, hessian_approx="full"), 1_000, 4e-2),
        (ds_skin, NeuralNetModel(lambda: make_mlp(D_in=3, K=1, hidden_layers=[8, 8, 8], activation=nn.ReLU()), task="binary_classification", adam_lr=3e-3, adam_steps=30_000, hessian_approx="full"), 1_000, 4e-2),
        (ds_protein, NeuralNetModel(lambda: make_mlp(D_in=8, K=1, hidden_layers=[16], activation=nn.ReLU()), task="regression", adam_lr=3e-3, adam_steps=30_000, hessian_approx="full"), 100, 4e-2),
        (ds_protein, NeuralNetModel(lambda: make_mlp(D_in=8, K=1, hidden_layers=[16, 16], activation=nn.ReLU()), task="regression", adam_lr=3e-3, adam_steps=30_000, hessian_approx="full"), 100, 4e-2),
        (ds_protein, NeuralNetModel(lambda: make_mlp(D_in=8, K=1, hidden_layers=[16, 16, 16], activation=nn.ReLU()), task="regression", adam_lr=3e-3, adam_steps=30_000, hessian_approx="full"), 100, 4e-2),
    ]
    for ds, model, S, lbd in _d:
        if isinstance(model, NeuralNetModel):
            model.nb_params = model._n_params(ds.X)
            if torchinfo_summary is not None:
                try:
                    torchinfo_summary(model._net_(), input_size=(1, ds.X.shape[1]))
                except Exception:
                    pass
        else:
            model.nb_params = ds.X.shape[1] if not hasattr(model, 'bias') else (ds.X.shape[1] + model.bias)
        print(f"Dataset: {ds.name} N={ds.X.shape[0]} D={ds.X.shape[1]} Model={model.__class__.__name__} P={model.nb_params}")
        np.random.seed(42)
        theta_star = model.find_map(ds.X, ds.y, lam=lbd)
        lmap, gmap = model.loss_and_grad(theta_star, ds.X, ds.y, lbd)
        print(f"  L at MAP = {lmap:.2e}  |grad| = {np.linalg.norm(gmap):.2e}")
        BBT_eigs, g0 = estimate_BBT_eigenvalues(theta_star, ds, model, S=S)
        file_name = f"{ds.name}_{model.__class__.__name__}_{model.nb_params}_gaussian_test.pdf"
        if isinstance(model, NeuralNetModel):
            file_name = f"{ds.name}_{model.__class__.__name__}_{model.nb_params}_{model.adam_steps}_gaussian_test.pdf"
        plot_gradient_noise(model, theta_star, ds, BBT_eigs, grad_mean=g0, S=S, n_samples=100_000, save_path=file_name, noise_norm_only=True)
    print("\nDone.")