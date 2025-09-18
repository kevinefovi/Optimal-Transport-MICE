from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

from baseline_PMM import statsmodels_pmm_impute_ols
from OT_PMM import ot_pmm_impute_transductive_ols

datasets   = ["parkinsons", "transfusion", "breast_cancer"]
mechanisms = ["MCAR", "MAR", "MNAR"]

# parameters that OT_PMM shares with normal PMM are equal for a fair comparison
otp = dict(
    sweeps=5,
    pca_dims=10,       # OT's denoising strength
    beta=0.10,         # small multivariate weight
    k_like=5,          # same k as PMM
    top_k=5,           # same donor pool size as PMM
    alpha=2.0,         # keep |yhat gap| dominant over multivariate weight
    init_noise=0.0,    # match statsmodels mean init
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

        print(f"\n================ {dataset} ================")
        for mech in mechanisms:
            rmses_pmm, maes_pmm = [], []
            rmses_otp, maes_otp = [], []

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
                rmses_pmm.append(rmse_p); maes_pmm.append(mae_p)

                # OT PMM
                X_otp_df = ot_pmm_impute_transductive_ols(
                    X_df, M_run, random_state=seed, **otp
                )
                rmse_o, mae_o = rmse_mae(X_true, X_otp_df.to_numpy(), M_run)
                rmses_otp.append(rmse_o); maes_otp.append(mae_o)

                results += [
                    dict(ds=dataset, mech=mech, seed=seed, method="PMM", metric="RMSE", value=rmse_p),
                    dict(ds=dataset, mech=mech, seed=seed, method="PMM", metric="MAE", value=mae_p),
                    dict(ds=dataset, mech=mech, seed=seed, method="OT_PMM", metric="RMSE", value=rmse_o),
                    dict(ds=dataset, mech=mech, seed=seed, method="OT_PMM", metric="MAE", value=mae_o)
                ]

            rp_m, rp_s = np.mean(rmses_pmm), np.std(rmses_pmm, ddof=1)
            mp_m, mp_s = np.mean(maes_pmm),  np.std(maes_pmm,  ddof=1)
            ro_m, ro_s = np.mean(rmses_otp), np.std(rmses_otp, ddof=1)
            mo_m, mo_s = np.mean(maes_otp),  np.std(maes_otp,  ddof=1)

            print(f"\nMechanism: {mech}")
            print(f"  PMM(OLS)         RMSE: {rp_m:.4f} ± {rp_s:.4f}   MAE: {mp_m:.4f} ± {mp_s:.4f}   (n={len(rmses_pmm)})")
            print(f"  OT-PMM(OLS,fair) RMSE: {ro_m:.4f} ± {ro_s:.4f}   MAE: {mo_m:.4f} ± {mo_s:.4f}   (n={len(rmses_otp)})")
    return results

if __name__ == "__main__":
    res = run()
    df = pd.DataFrame(res)
    df.to_csv("data/test.csv")