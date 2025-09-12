# src/main.py
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

# local imports work whether run from repo root or src/
THIS = Path(__file__).resolve()
sys.path.append(str(THIS.parent))

from baseline_PMM import statsmodels_pmm_impute_ols
from OT_PMM import ot_pmm_impute_transductive_ols

# ------------- config -------------
PROCESSED = {
    "parkinsons":     Path("data/processed/parkinsons.data"),
    "breast_cancer":  Path("data/processed/breast_cancer.data"),
    "transfusion":    Path("data/processed/transfusion.data"),
}
DATASETS   = ["transfusion", "parkinsons", "breast_cancer"]
MECHANISMS = ["MCAR", "MAR", "MNAR_log", "MNAR_quant"]

PMM_ITERS  = 5
PMM_K      = 5

# Fair OT preset: same conditional model, same donor sampling, same sweeps
OTP_KW = dict(
    sweeps=PMM_ITERS,
    pca_dims=10,       # OT's extra signal (can set None/5/10/20)
    beta=0.10,         # small multivariate weight
    k_like=PMM_K,      # same k as PMM
    top_k=PMM_K,       # same donor pool size as PMM for strict fairness
    alpha=2.0,         # keep |yhat gap| dominant
    barycentric=False, # sampling (same as PMM)
    cond_model="ols",  # SAME conditional model as PMM
    init_noise=0.0,    # match statsmodels mean init
    ridge_alpha=1.0,   # unused when cond_model="ols"
)

# ------------- helpers -------------
def load_processed_df(dataset: str) -> pd.DataFrame:
    fp = PROCESSED[dataset]
    if not fp.exists():
        raise FileNotFoundError(f"Processed file not found: {fp}")
    df = pd.read_csv(fp)
    if df.isna().any().any():
        raise ValueError(f"{fp} contains NaNs; processed features must be fully observed.")
    return df

def iter_masks(dataset: str, mechanism: str):
    """
    Yield (seed, mask) from:
      data/masks/{dataset}/{mechanism}/{dataset}_{mechanism}_p*_seed*.npz
    """
    mask_dir = Path(f"data/masks/{dataset}/{mechanism}")
    if not mask_dir.exists():
        return
    pattern = f"{dataset}_{mechanism}_p*_seed*.npz"
    for fp in sorted(mask_dir.glob(pattern)):
        z = np.load(fp)
        M = z["mask"].astype(bool)
        if "seed" in z.files:
            seed = int(np.atleast_1d(z["seed"])[0])
        else:
            m = re.search(r"seed(\d+)", fp.stem)
            seed = int(m.group(1)) if m else 0
        yield seed, M

def drop_all_missing_rows(X_df: pd.DataFrame, M: np.ndarray):
    """Remove rows where all features are masked (True across all columns)."""
    all_miss = M.all(axis=1)
    if all_miss.any():
        X_df = X_df.loc[~all_miss].copy()
        M = M[~all_miss, :]
    return X_df, M, int(all_miss.sum())

def rmse_mae(X_true: np.ndarray, X_imp: np.ndarray, M: np.ndarray):
    diff = (X_imp - X_true)[M]
    rmse = float(np.sqrt((diff**2).mean()))
    mae  = float(np.abs(diff).mean())
    return rmse, mae

# ------------- main -------------
def run():
    for dataset in DATASETS:
        X_df_full = load_processed_df(dataset)

        print(f"\n================ {dataset} ================")
        for mech in MECHANISMS:
            rmses_pmm, maes_pmm = [], []
            rmses_otp, maes_otp = [], []

            any_masks = False
            for seed, M in iter_masks(dataset, mech):
                any_masks = True

                # Align inputs by removing rows that are entirely masked
                X_df, M_run, n_drop = drop_all_missing_rows(X_df_full, M)
                if n_drop:
                    print(f"  [info] {dataset}/{mech}/seed{seed}: dropped {n_drop} all-missing rows")

                X_true = X_df.to_numpy(copy=True)

                # PMM baseline
                X_pmm_df = statsmodels_pmm_impute_ols(
                    X_df, M_run, iters=PMM_ITERS, k_pmm=PMM_K, seed=seed
                )
                rmse_p, mae_p = rmse_mae(X_true, X_pmm_df.to_numpy(), M_run)
                rmses_pmm.append(rmse_p); maes_pmm.append(mae_p)

                # OT-PMM (fair config)
                X_otp_df = ot_pmm_impute_transductive_ols(
                    X_df, M_run, random_state=seed, **OTP_KW
                )
                rmse_o, mae_o = rmse_mae(X_true, X_otp_df.to_numpy(), M_run)
                rmses_otp.append(rmse_o); maes_otp.append(mae_o)

            if not any_masks:
                print(f"[WARN] No masks for mechanism {mech}")
                continue

            rp_m, rp_s = np.mean(rmses_pmm), np.std(rmses_pmm, ddof=1)
            mp_m, mp_s = np.mean(maes_pmm),  np.std(maes_pmm,  ddof=1)
            ro_m, ro_s = np.mean(rmses_otp), np.std(rmses_otp, ddof=1)
            mo_m, mo_s = np.mean(maes_otp),  np.std(maes_otp,  ddof=1)

            print(f"\nMechanism: {mech}")
            print(f"  PMM(OLS)         RMSE: {rp_m:.4f} ± {rp_s:.4f}   MAE: {mp_m:.4f} ± {mp_s:.4f}   (n={len(rmses_pmm)})")
            print(f"  OT-PMM(OLS,fair) RMSE: {ro_m:.4f} ± {ro_s:.4f}   MAE: {mo_m:.4f} ± {mo_s:.4f}   (n={len(rmses_otp)})")

if __name__ == "__main__":
    run()