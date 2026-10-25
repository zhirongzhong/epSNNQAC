"""Training-only running covariance and Bussgang-LMMSE reconstruction."""

import numpy as np
import torch


class RunningCov:
    def __init__(self, dim, alpha=0.05):
        self.dim = dim
        self.alpha = alpha
        self.mean = None
        self.C = None

    def update(self, x):
        mean = x.mean(0)
        centered = x - mean
        covariance = centered.T @ centered / len(x) + 1e-6 * np.eye(self.dim)
        if self.mean is None:
            self.mean, self.C = mean, covariance
        else:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * mean
            self.C = (1 - self.alpha) * self.C + self.alpha * covariance


class BussgangLMMSE:
    def __init__(self, dim):
        self.x = RunningCov(dim)
        self.r = RunningCov(dim)

    def update(self, x, r):
        self.x.update(x)
        self.r.update(r)

    def infer(self, r):
        if self.x.C is None:
            raise RuntimeError("Bussgang statistics must be fitted on training data first")
        cx, cr = self.x.C, self.r.C
        b = np.diag(np.sqrt(2 / np.pi) / np.sqrt(np.diag(cx) + 1e-9))
        reconstruction = cx @ b.T @ np.linalg.pinv(cr)
        mean = (r - self.r.mean) @ reconstruction.T + self.x.mean
        posterior = cx - reconstruction @ cr @ reconstruction.T
        return mean.astype(np.float32), np.diag(posterior).astype(np.float32)

    def state_dict(self):
        if self.x.C is None:
            raise RuntimeError("Cannot save unfitted Bussgang statistics")
        return {
            "x_mean": torch.from_numpy(self.x.mean.copy()),
            "x_C": torch.from_numpy(self.x.C.copy()),
            "r_mean": torch.from_numpy(self.r.mean.copy()),
            "r_C": torch.from_numpy(self.r.C.copy()),
        }

    def load_state_dict(self, state):
        for prefix, cov in (("x", self.x), ("r", self.r)):
            cov.mean = state[f"{prefix}_mean"].cpu().numpy().copy()
            cov.C = state[f"{prefix}_C"].cpu().numpy().copy()
            if cov.mean.shape != (cov.dim,) or cov.C.shape != (cov.dim, cov.dim):
                raise ValueError("Bussgang state dimensions do not match the model")
            if not np.isfinite(cov.mean).all() or not np.isfinite(cov.C).all():
                raise ValueError("Bussgang state contains nonfinite values")
