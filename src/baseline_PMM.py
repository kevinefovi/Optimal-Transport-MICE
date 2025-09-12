# src/baseline_PMM.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd

def _patsy_safe_cols(cols):
    safe = []
    for c in cols:
        # replace any non-alphanumeric/underscore with underscore
        s = re.sub(r'[^0-9a-zA-Z_]', '_', c)
        # collapse repeats and trim edges
        s = re.sub(r'_+', '_', s).strip('_')
        # ensure not starting with a digit
        if s and s[0].isdigit():
            s = 'f_' + s
        # fallback if name becomes empty
        if not s:
            s = 'var'
        safe.append(s)
    # avoid accidental duplicates by uniquifying
    seen = {}
    uniq = []
    for s in safe:
        k = s
        i = seen.get(s, 0)
        if i:
            k = f"{s}_{i}"
        seen[s] = i + 1
        uniq.append(k)
    return uniq

def statsmodels_pmm_impute_ols(
    X_df: pd.DataFrame,
    M: np.ndarray,
    *,
    iters: int = 5,
    k_pmm: int | None = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Single-imputation with statsmodels' MICE using OLS + PMM for continuous vars."""
    from statsmodels.imputation.mice import MICEData  # lazy import

    # Apply mask -> NaNs
    df_nan = X_df.copy()
    df_nan.values[M] = np.nan

    # Make a patsy-safe copy of the data for statsmodels
    orig_cols = list(df_nan.columns)
    safe_cols = _patsy_safe_cols(orig_cols)
    df_safe = df_nan.copy()
    df_safe.columns = safe_cols
    # force numeric dtypes (object columns can also upset patsy)
    df_safe = df_safe.apply(pd.to_numeric, errors="coerce")

    # Reproducibility: statsmodels uses numpy's global RNG
    np.random.seed(seed)

    imp = MICEData(df_safe)

    # Try to set PMM donor count for this statsmodels version
    if k_pmm is not None:
        for attr in ("k_pmm", "_k_pmm", "k_pmm_mean_match"):
            if hasattr(imp, attr):
                try:
                    setattr(imp, attr, int(k_pmm))
                except Exception:
                    pass

    for _ in range(int(iters)):
        imp.update_all()

    out = imp.data.copy()
    out.columns = orig_cols  # restore original names for the caller
    return out