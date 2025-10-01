import math
import numpy as np
from typing import Optional, List
from dppy.finite_dpps import FiniteDPP

class DPPStreamingSelector:
    """Streaming DPP (MAP-like) with Cholesky updates and a geometry-rule threshold.

    Works best with L2-normalized embeddings and a cosine kernel:
        k(x, y) = <x, y>  (on unit-norm vectors → PSD Gram)

    Accept rule (geometry): accept candidate i iff d_i^2 >= tau,
        where d_i^2 = k_ii - k_iS^T K_SS^{-1} k_iS
        and tau = sin^2(angle_threshold).

    Maintains the Cholesky factor R of K_SS (upper-triangular: K_SS = R^T R)
    so each marginal gain check is O(|S|)–O(|S|^2) without inverses.

    Parameters
    ----------
    angle_threshold_deg : float
        Minimum novelty angle (degrees). tau = sin^2(angle).
        e.g., 30° -> tau=0.25 (good default).
    jitter : float
        Small diagonal regularizer to keep K well-conditioned.
    max_selected : Optional[int]
        If set, stop accepting when |S| reaches this cap.
    """

    def __init__(self,
                 angle_threshold_deg: float,
                 jitter: float = 1e-6,
                 max_selected: Optional[int] = None):
        self.tau = float(np.sin(np.deg2rad(angle_threshold_deg)) ** 2)
        self.jitter = float(jitter)
        self.max_selected = max_selected

        # State
        self.R: Optional[np.ndarray] = None         # upper-triangular Cholesky of K_SS
        self.X: Optional[np.ndarray] = None         # selected feature matrix (n_sel, d), unit-norm rows
        self.n_sel: int = 0
        self.dim: Optional[int] = None

    # ---------- Public API ----------

    def bootstrap_with(self, feats: np.ndarray):
        """Seed the selector with already-saved items (unit-normalized rows).

        Used to rebuild state from disk on startup.
        """
        if feats is None or len(feats) == 0:
            return
        for f in feats:
            self._accept_direct(f.astype(np.float32, copy=False))

    def consider(self, feat: np.ndarray) -> bool:
        """Decide whether to accept a candidate feature (unit-normalized row).

        Returns
        -------
        bool
            True if accepted, False otherwise.
        """
        feat = self._as_unit_row(feat)

        if self.max_selected is not None and self.n_sel >= self.max_selected:
            return False, None, None

        # First item: accept immediately
        if self.n_sel == 0:
            self._accept_direct(feat)
            return True, None, None

        # k_iS = X @ feat  (cosine kernel on unit vectors)
        k_iS = self.X @ feat
        k_ii = 1.0 + self.jitter

        # Solve R^T w = k_iS  (forward substitution on lower-triangular R^T)
        w = self._solve_lower(RT=self.R.T, b=k_iS)

        # Schur complement d^2
        d2 = float(k_ii - np.dot(w, w))
        angle_deg = self.d2_to_angle_deg(d2)
        # Geometry threshold
        if d2 >= self.tau:
            # Rank-1 Cholesky append:
            # R_new = [[R, w],
            #          [0, sqrt(d2)]]
            d = math.sqrt(max(d2, self.jitter))
            self.R = np.block([
                [self.R,                w[:, None]],
                [np.zeros((1, self.n_sel), dtype=np.float32), np.array([[d]], dtype=np.float32)]
            ])
            # Append to X
            self.X = np.vstack([self.X, feat[None, :]])
            self.n_sel += 1
            return True, d2, angle_deg

        return False, d2, angle_deg

    def d2_to_angle_deg(self, d2):
        d2 = float(np.clip(d2, 0.0, 1.0))
        return np.degrees(np.arcsin(np.sqrt(d2)))

    def size(self) -> int:
        return self.n_sel

    def set_angle_threshold(self, angle_deg: float):
        """Update geometry threshold online."""
        self.tau = float(np.sin(np.deg2rad(angle_deg)) ** 2)

    # ---------- Optional batch k-DPP refinement using DPPy ----------
    def refine_with_k_dpp(self, pool_feats: np.ndarray, k: int) -> List[int]:
        """(Optional) On a pool, pick a diverse subset via k-DPP using DPPy (offline/batch).

        Returns indices of chosen items within pool_feats.
        """

        pool = self._as_unit_matrix(pool_feats)
        # Cosine kernel L (PSD since Gram of unit vectors)
        L = pool @ pool.T
        L = (L + L.T) * 0.5
        np.fill_diagonal(L, np.diag(L) + self.jitter)

        dpp = FiniteDPP('likelihood', **{'L': L})
        # k-DPP sampling by conditioning on |S|=k
        dpp.flush_samples()
        dpp.sample_exact_k_dpp(size=k)
        return sorted(list(dpp.list_of_samples[-1]))

    # ---------- Internals ----------

    def _accept_direct(self, feat: np.ndarray):
        """Accept without thresholding (used for bootstrapping)."""
        feat = self._as_unit_row(feat)
        if self.n_sel == 0:
            # R = [sqrt(k_ii)]
            d = math.sqrt(1.0 + self.jitter)
            self.R = np.array([[d]], dtype=np.float32)
            self.X = feat[None, :]
            self.n_sel = 1
            self.dim = feat.shape[0]
            return

        # Standard rank-1 append path
        k_iS = self.X @ feat
        w = self._solve_lower(RT=self.R.T, b=k_iS)
        d2 = float(1.0 + self.jitter - np.dot(w, w))
        d = math.sqrt(max(d2, self.jitter))

        self.R = np.block([
            [self.R,                w[:, None]],
            [np.zeros((1, self.n_sel), dtype=np.float32), np.array([[d]], dtype=np.float32)]
        ])
        self.X = np.vstack([self.X, feat[None, :]])
        self.n_sel += 1

    @staticmethod
    def _as_unit_row(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n <= 0:
            raise ValueError("Zero/NaN vector given to DPPStreamingSelector")
        return v / n

    @staticmethod
    def _as_unit_matrix(M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=np.float32)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return M / norms

    @staticmethod
    def _solve_lower(RT: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve (R^T) w = b, where R is upper-triangular → R^T is lower-triangular."""
        L = RT  # lower-triangular
        b = b.astype(np.float32, copy=False)
        n = L.shape[0]
        w = np.empty_like(b)
        for i in range(n):
            s = b[i]
            if i > 0:
                s -= np.dot(L[i, :i], w[:i])
            w[i] = s / (L[i, i] if L[i, i] != 0 else 1e-12)
        return w
