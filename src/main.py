from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

from baseline_PMM import statsmodels_pmm_impute_ols
from OT_PMM import ot_pmm_impute_transductive_ols

datasets   = ["transfusion", "parkinsons", "breast_cancer"]
mechanisms = ["MCAR", "MAR", "MNAR_quant"]

# parameters that OT_PMM shares with normal PMM are equal for a fair comparison
otp = dict(
    sweeps=5,
    pca_dims=10,       # OT's denoising strength
    beta=0.10,         # small multivariate weight
    k_like=5,          # same k as PMM
    top_k=5,           # same donor pool size as PMM
    alpha=2.0,         # keepng gap dominant (balance between alpha and beta)
    init_noise=0.0,   
)

def iter_masks(dataset: str, mechanism: str):
    # allows parsing of whatever file in directory (w/o needing exact fp)
    mask_dir = Path(f"data/masks/{dataset}/{mechanism}")
    pattern = f"{dataset}_{mechanism}_p*_seed*.npz"

    for fp in sorted(mask_dir.glob(pattern)):
        z = np.load(fp)
        M = z["mask"].astype(bool)
        if "seed" in z.files:
            seed = int(np.atleast_1d(z["seed"])[0])
        yield seed, M

def drop_all_missing_rows(X_df: pd.DataFrame, M: np.ndarray):
    # matching can't happen on all NaN rows
    all_miss = M.all(axis=1)
    if all_miss.any():
        X_df = X_df.loc[~all_miss].copy()
        M = M[~all_miss, :]
    return X_df, M, int(all_miss.sum())

def rmse_mae(X_true: np.ndarray, X_imp: np.ndarray, M: np.ndarray):
    # benchmark metrics 
    diff = (X_imp - X_true)[M]
    rmse = float(np.sqrt((diff**2).mean()))
    mae  = float(np.abs(diff).mean())
    return rmse, mae

# main
def run():
    results = []

    for dataset in datasets:
        X_df_full = pd.read_csv(f"data/processed/{dataset}.data")

        for mech in mechanisms:
            any_masks = False
            for seed, M in iter_masks(dataset, mech):
                any_masks = True

                # remove full NaN rows (only the parkinsons dataset suffers from this
                # due to low number of features)
                X_df, M_run, n_drop = drop_all_missing_rows(X_df_full, M)

                X_true = X_df.to_numpy(copy=True)

                # PMM baseline
                X_pmm_df = statsmodels_pmm_impute_ols(
                    X_df, M_run, iters=5, k_pmm=5, seed=seed
                )
                rmse_p, mae_p = rmse_mae(X_true, X_pmm_df.to_numpy(), M_run)

                # OT PMM
                X_otp_df = ot_pmm_impute_transductive_ols(
                    X_df, M_run, random_state=seed, **otp
                )
                rmse_o, mae_o = rmse_mae(X_true, X_otp_df.to_numpy(), M_run)

                # building data for dataframe
                results += [
                    dict(ds=dataset, mech=mech, seed=seed, method="PMM", metric="RMSE", value=rmse_p),
                    dict(ds=dataset, mech=mech, seed=seed, method="PMM", metric="MAE", value=mae_p),
                    dict(ds=dataset, mech=mech, seed=seed, method="OT_PMM", metric="RMSE", value=rmse_o),
                    dict(ds=dataset, mech=mech, seed=seed, method="OT_PMM", metric="MAE", value=mae_o)
                ]

    return results

if __name__ == "__main__":
    res = run()
    df = pd.DataFrame(res)
    df.to_csv("data/results/raw_results.csv")