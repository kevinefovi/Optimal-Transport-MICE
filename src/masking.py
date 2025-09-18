import pandas as pd
import numpy as np

# ---
# Masking functions (MCAR, MAR, MNAR_quantile)

def calibrate_intercept(logits, target, tol=1e-6, maxit=50):
    lo, hi = -20.0, 20.0
    for _ in range(maxit):
        mid = (lo + hi) / 2
        p = 1.0 / (1.0 + np.exp(-(logits + mid)))
        m = p.mean()
        if abs(m - target) < tol:
            return mid
        if m < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def mask_mcar(X, p, rng):
    M = rng.random(X.shape) < p
    return M

def mask_mar(X, overall_p=0.30, rng=None, maskable_frac=0.70, k_predictors=None):
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X)
    n, d = X.shape
    Z = (X - X.mean(0)) / (X.std(0, ddof=0) + 1e-12)

    d_mask = max(1, int(round(maskable_frac * d)))
    maskable = rng.choice(d, size=d_mask, replace=False)
    always_obs = np.setdiff1d(np.arange(d), maskable)

    M = np.zeros((n, d), dtype=bool)

    for j in maskable:
        preds = always_obs
        if k_predictors is not None and len(always_obs) > k_predictors:
            preds = rng.choice(always_obs, size=k_predictors, replace=False)

        w = rng.normal(size=len(preds))
        w /= (np.linalg.norm(w) + 1e-12)

        logits = (Z[:, preds] @ w)
        logits /= (logits.std() + 1e-12)

        b = calibrate_intercept(logits, target=overall_p)
        p_row = 1 / (1 + np.exp(-(logits + b)))

        M[:, j] = rng.random(n) < p_row

    return M

def mask_mnar(X, overall_p=0.30, rng=None, frac_cols=0.30,
                             lower_q=0.25, upper_q=0.75):
    
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X); n, d = X.shape
    M = np.zeros((n, d), dtype=bool)

    d_sel = max(1, int(round(frac_cols * d)))
    cols = rng.choice(d, size=d_sel, replace=False)

    for j in cols:
        ql = np.quantile(X[:, j], lower_q)
        qu = np.quantile(X[:, j], upper_q)
        tails = (X[:, j] <= ql) | (X[:, j] >= qu)
        frac_tails = max(tails.mean(), 1e-12)
        r = min(1.0, overall_p / frac_tails)
        M[:, j] = tails & (rng.random(n) < r)
    return M