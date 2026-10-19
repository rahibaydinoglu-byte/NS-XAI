"""Explanation cost and explanation stability for every model (seed-0 fitted model).
Explains 30 test days (720 hourly day-ahead forecasts) with the explainer that is natural for
each model family, measures CPU time, and computes stability as the mean Spearman rank
correlation between attributions for x and for a perturbed x' = x(1+eps), eps~N(0,0.01) on
load-derived inputs."""
import sys, pickle, warnings
import numpy as np
from scipy.stats import spearmanr
import shap
from common import *
warnings.filterwarnings("ignore")

df, load, X, y, tr, va, te = make_splits()
cols = list(X.columns)
Xs = Scaler().fit(X.values[tr]); ys = Scaler().fit(y.values[tr])
load_n = ((load.values - ys.m) / ys.s).astype(np.float32)
cal_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "weekend"]
if TASK == "B": cal_cols = cal_cols + ["tso", "tso_err24", "tso_err48", "tso_err168"]
cal = (Xs.t(X.values)[:, [cols.index(c) for c in cal_cols]] if TASK == "B" else X[cal_cols].values).astype(np.float32)
itr, ite = np.where(tr)[0], np.where(te)[0]
rng = np.random.default_rng(7)
days = np.sort(rng.choice(365, 30, replace=False))
sel = np.concatenate([ite[d * 24:(d + 1) * 24] for d in days])
bg_idx = rng.choice(itr, 100, replace=False)

def tab(idx, noise=0.0, fuzzy=False):
    Xr = X.values[idx].copy()
    if noise:
        lc = [i for i, c in enumerate(cols) if c.startswith(("lag", "roll"))]
        Xr[:, lc] *= (1 + np.random.default_rng(11).normal(0, noise, size=(len(idx), len(lc))))
    Z = Xs.t(Xr).astype(np.float32)
    return Z[:, [cols.index(c) for c in FUZZY_INPUTS]] if fuzzy else Z

def seq(idx, noise=0.0):
    s, c = seq_windows(load_n, cal, idx)
    if noise:
        raw = s * ys.s + ys.m
        raw = raw * (1 + np.random.default_rng(11).normal(0, noise, size=raw.shape))
        s = ((raw - ys.m) / ys.s).astype(np.float32)
    return s, c

class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, *a): return self.m(*a).unsqueeze(-1)

def stability(a, b):
    return float(np.nanmean([spearmanr(np.abs(a[i]), np.abs(b[i]))[0] for i in range(len(a))]))

out = {}
for name in ["ridge", "lgbm", "tsk", "tsk8", "mlp", "transformer", "lstm"]:
    path = os.path.join(OUT, f"model_{name}.pkl")
    if not os.path.exists(path): continue
    m = pickle.load(open(path, "rb"))["model"]
    def explain(noise):
        if name == "ridge":
            Z = tab(sel, noise); return Z * m.coef_[None, :]           # exact linear attribution
        if name == "lgbm":
            Z = tab(sel, noise); return m.predict(Z, pred_contrib=True)[:, :-1]   # TreeSHAP
        if name.startswith("tsk"):
            Z = torch.tensor(tab(sel, noise, fuzzy=True))
            with torch.no_grad():
                w = m.firing(Z)                                           # (n, R)
                contrib = (w[:, :, None] * m.A[None] * Z[:, None, :]).sum(1)  # intrinsic local-linear
            return contrib.numpy()
        if name == "mlp":
            e = shap.GradientExplainer(Wrap(m), torch.tensor(tab(bg_idx)))
            return np.array(e.shap_values(torch.tensor(tab(sel, noise)), nsamples=200)).reshape(len(sel), -1)
        s_bg, c_bg = seq(bg_idx); s, c = seq(sel, noise)
        e = shap.GradientExplainer(Wrap(m), [torch.tensor(s_bg), torch.tensor(c_bg)])
        sv = e.shap_values([torch.tensor(s), torch.tensor(c)], nsamples=200)
        sv = [np.asarray(v) for v in sv]
        return np.concatenate([sv[0].reshape(len(sel), -1), sv[1].reshape(len(sel), -1)], 1)
    if hasattr(m, "eval"): m.eval()
    with CPUMeter() as cm:
        a = explain(0.0)
    b = explain(0.01)
    out[name] = dict(cpu_s_720=cm.cpu, wall_s_720=cm.wall, cpu_s_per_day=cm.cpu / 30,
                     stability=stability(a, b), n_attr=int(a.shape[1]))
    print(name, out[name], flush=True)
save_json(out, "xai.json")
