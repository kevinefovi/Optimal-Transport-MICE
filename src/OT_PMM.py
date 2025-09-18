from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Literal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge

def _as_np(X_df: pd.DataFrame) -> Tuple[np.ndarray, list[str], pd.Index]:
    X = X_df.to_numpy(dtype=float, copy=True)
    return X, list(X_df.columns), X_df.index

def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(0 if seed is None else seed)

def _cap(n: int, mx: int) -> int:
    return max(1, min(int(n), int(mx)))

def _pos_scale(v: np.ndarray) -> float:
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0: return 1.0
    pos = v[v > 0]
    if pos.size:
        s = float(np.median(pos)); return s if s > 1e-12 else 1.0
    s = float(np.median(np.abs(v - np.median(v))))
    return s if s > 1e-12 else 1.0

def _row_weights(cost_row: np.ndarray, temp: float | str = "auto") -> np.ndarray:
    c = np.asarray(cost_row, float).ravel()
    c = np.where(np.isfinite(c), c, np.inf)
    if isinstance(temp, str) and temp == "auto":
        finite = c[np.isfinite(c)]
        if finite.size:
            med = np.median(finite); mad = np.median(np.abs(finite - med))
            t = float(mad if mad > 1e-12 else 1.0)
        else:
            t = 1.0
    else:
        t = float(temp)
    c0 = c - np.nanmin(c)
    z = np.exp(-c0 / max(t, 1e-8))
    s = z.sum()
    if not np.isfinite(s) or s <= 0: return np.ones_like(c) / c.size
    w = z / s; w = np.where(np.isfinite(w), w, 0.0); s2 = w.sum()
    return w / s2 if s2 > 0 else np.ones_like(c) / c.size

def _init_fill_mean_noise(X: np.ndarray, M: np.ndarray, noise: float, rng: np.random.Generator):
    col_mean = np.nanmean(np.where(M, np.nan, X), axis=0)
    col_std  = np.nanstd (np.where(M, np.nan, X), axis=0, ddof=0)
    col_std[col_std == 0] = 1.0
    r, c = np.where(M)
    X[M] = col_mean[c] + noise * col_std[c] * rng.standard_normal(len(r))
    return X

def ot_pmm_impute_transductive_ols(
    X_df: pd.DataFrame,
    M: np.ndarray,
    *,
    sweeps: int = 5,
    pca_dims: Optional[int] = 20,
    k_like: Optional[int] = None,
    top_k: int = 64,
    alpha: float = 1.0,
    beta: float = 0.25,
    init_noise: float = 0.0,
    random_state: int = 42,
) -> pd.DataFrame:
    if not isinstance(X_df, pd.DataFrame):
        raise TypeError("X_df must be a pandas DataFrame.")
    X, cols, idx = _as_np(X_df)
    if M.shape != X.shape:
        raise ValueError(f"M shape {M.shape} != X shape {X.shape}")

    rng = _rng(random_state)
    X = _init_fill_mean_noise(X, M, noise=init_noise, rng=rng)

    n, d = X.shape
    top_k = _cap(top_k, n)
    alpha = float(max(0.0, alpha))
    beta = float(max(0.0, beta))

    for _ in range(int(sweeps)):
        for j in range(d):
            obs = ~M[:, j]
            mis = M[:, j]
            idx_mis = np.flatnonzero(mis)
            n_obs, n_mis = int(obs.sum()), idx_mis.size
            if n_mis == 0 or n_obs < 5:
                continue

            X_obs_pred = np.delete(X[obs, :], j, axis=1)
            X_mis_pred = np.delete(X[mis, :], j, axis=1)
            y_obs = X[obs, j]

            est = LinearRegression()
            est.fit(X_obs_pred, y_obs)

            yhat_obs = est.predict(X_obs_pred)
            yhat_mis = est.predict(X_mis_pred)

            Z_obs_raw, Z_mis_raw = X_obs_pred, X_mis_pred
            use_d2 = (Z_obs_raw.shape[1] > 0) and (beta > 0.0)

            if use_d2 and pca_dims is not None:
                n_comp = _cap(pca_dims, min(Z_obs_raw.shape[1], Z_obs_raw.shape[0]))
                if n_comp > 0:
                    pca = PCA(n_components=n_comp, random_state=random_state)
                    Z_obs = pca.fit_transform(Z_obs_raw)
                    Z_mis = pca.transform(Z_mis_raw)
                else:
                    Z_obs, Z_mis = Z_obs_raw, Z_mis_raw
            else:
                Z_obs, Z_mis = Z_obs_raw, Z_mis_raw

            if use_d2:
                donor_norm2 = (Z_obs ** 2).sum(axis=1)

            # per missing row (use integer row index i for assignment)
            for r_idx, i in enumerate(idx_mis):
                gap_all = np.abs(yhat_mis[r_idx] - yhat_obs)

                k_eff = _cap(k_like, n_obs)
                idx_gap = np.argpartition(gap_all, k_eff - 1)[:k_eff]

                gap = gap_all[idx_gap]
                gap_scaled = gap / _pos_scale(gap)

                if use_d2:
                    z = Z_mis[r_idx]
                    d2_all = donor_norm2 - 2.0 * (Z_obs @ z) + float((z ** 2).sum())
                    d2 = d2_all[idx_gap]
                    d2_scaled = d2 / _pos_scale(d2)
                else:
                    d2_scaled = 0.0

                c = beta * d2_scaled + alpha * gap_scaled

                K_eff = _cap(top_k, len(idx_gap))
                idx_top_local = np.argpartition(c, K_eff - 1)[:K_eff]
                donors = idx_gap[idx_top_local]
                cost_slice = c[idx_top_local]

                w = _row_weights(cost_slice, temp="auto")

                X[i, j] = float(y_obs[rng.choice(donors, p=w)])

    return pd.DataFrame(X, columns=cols, index=idx)