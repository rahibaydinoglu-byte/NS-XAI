"""Task B diagnostic: magnitude of the learned corrections to the TSO forecast and their
correlation with the TSO error (seed-averaged, test year 2018). Reported in Section 5.6."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "energy_dataset.csv"))
df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Madrid")
df = df.set_index("time").sort_index(); df = df[~df.index.duplicated()]
tso = df["total load forecast"].interpolate(limit_direction="both")
for m in ["ridge", "tsk8", "tsk", "lgbm", "mlp", "transformer", "lstm"]:
    d = np.load(os.path.join(ROOT, "results_B", f"pred_{m}.npz"), allow_pickle=True)
    t = pd.DatetimeIndex(pd.to_datetime(d["time"], utc=True)).tz_convert("Europe/Madrid")
    T = tso.reindex(t).values; c = d["pred"].mean(0) - T; e = d["y"] - T
    print(f"{m:12s} sd(correction) {c.std():6.1f} MW  sd(TSO error) {e.std():6.1f} MW  corr {np.corrcoef(c, e)[0, 1]:.3f}")
