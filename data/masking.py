import pandas as pd
import numpy as np
from pathlib import Path

# ---
# Masking functions (MCAR, MAR, MNAR_logit, MNAR_quantile)

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

def mask_mnar_logistic(X, overall_p=0.30, rng=None,
                             input_frac=0.30, maskable_frac=0.70, mcar_on_inputs=0.30):
    
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X); n, d = X.shape
    Z = (X - X.mean(0)) / (X.std(0, ddof=0) + 1e-12)

    d_inputs = max(1, int(round(input_frac * d)))
    inputs = rng.choice(d, size=d_inputs, replace=False)
    remaining = np.setdiff1d(np.arange(d), inputs)
    d_mask = max(1, int(round(maskable_frac * d)))
    maskable = rng.choice(remaining, size=min(d_mask, len(remaining)), replace=False)

    w = rng.normal(size=len(inputs)); w /= (np.linalg.norm(w) + 1e-12)
    logits = Z[:, inputs] @ w
    logits /= (logits.std() + 1e-12)

    b = calibrate_intercept(logits, target=overall_p)
    p_row = 1 / (1 + np.exp(-(logits + b)))

    M = np.zeros((n, d), dtype=bool)
    U = rng.random((n, len(maskable)))
    M[:, maskable] = U < p_row[:, None]

    if mcar_on_inputs > 0:
        M[:, inputs] |= (rng.random((n, len(inputs))) < mcar_on_inputs)

    return M

def mask_mnar_quantile(X, overall_p=0.30, rng=None, frac_cols=0.30,
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

# --- 
# Creating and storing multiple iterations of masked datasets

def save_mask_npz(dataset, mech, mask, meta, seed):
    outdir = Path("data/masks")/dataset/mech
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{dataset}_{mech}_p{int(meta.get('p', 0)*100):02d}_seed{seed:03d}.npz"
    np.savez_compressed(outdir/fname, mask=mask.astype(bool), seed=np.array([seed]),
                        **{k: np.asarray(v) for k,v in meta.items() if v is not None})

def load_mask_npz(path):
    f = np.load(path, allow_pickle=False)
    mask = f["mask"].astype(bool)
    meta = {k: f[k] for k in f.files if k != "mask"}
    return mask, meta

def generate_masks_for_dataset(dataset_name, csv_path, seeds_base=42, p=0.30):

    df = pd.read_csv(csv_path)
    X = df.to_numpy(dtype=float)

    rng0 = np.random.default_rng(seeds_base)
    seeds = rng0.integers(0, 2**31 - 1, size=30)

    for seed in seeds:
        rng_mcar = np.random.default_rng(seed + 1000)
        rng_mar  = np.random.default_rng(seed + 2000)
        rng_mnrl = np.random.default_rng(seed + 3000)
        rng_mnrq = np.random.default_rng(seed + 4000)

        M = mask_mcar(X, p=p, rng=rng_mcar)
        save_mask_npz(dataset_name, "MCAR", M, {"p": p}, seed)

        M = mask_mar(X, overall_p=p, rng=rng_mar, maskable_frac=0.70)
        save_mask_npz(dataset_name, "MAR", M, {"p": p}, seed)

        M = mask_mnar_logistic(X, overall_p=p, rng=rng_mnrl,
                               input_frac=0.30, maskable_frac=0.70, mcar_on_inputs=p)
        save_mask_npz(dataset_name, "MNAR_log", M, {"p": p}, seed)

        M = mask_mnar_quantile(X, overall_p=p, rng=rng_mnrq,
                               frac_cols=0.30, lower_q=0.25, upper_q=0.75)
        save_mask_npz(dataset_name, "MNAR_quant", M,
                      {"p": p, "lower_q": 0.25, "upper_q": 0.75}, seed)
        
generate_masks_for_dataset("parkinsons", "data/processed/parkinsons.data")
generate_masks_for_dataset("transfusion", "data/processed/transfusion.data")
generate_masks_for_dataset("breast_cancer", "data/processed/wdbc.data")

# ---