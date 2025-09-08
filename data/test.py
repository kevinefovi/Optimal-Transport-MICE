import pandas as pd
from pathlib import Path

bcd = Path("data") / "raw" / "wdbc.data"

base=["radius","texture","perimeter","area","smoothness","compactness",
    "concavity","concave_points","symmetry","fractal_dimension"]
stats=["mean","se","worst"]

COLS = ["ID","Diagnosis"] + [f"{feature}_{stat}" for stat in stats for feature in base]

df = pd.read_csv(bcd, sep=",", header=None, names=COLS)
print(df.head())