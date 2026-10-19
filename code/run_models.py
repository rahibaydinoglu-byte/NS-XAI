"""Hyper-parameter search + final multi-seed training + inference cost + robustness, per model.
Usage: python run_models.py <model> [n_seeds]
Every training/inference call is wrapped in CPUMeter so that the full development cost
(search + final training) is recorded, not only the final fit."""
import sys, pickle, itertools
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
import lightgbm as lgb
from common import *

name = sys.argv[1]; NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 5
df, load, X, y, tr, va, te = make_splits()
cols = list(X.columns)
Xs = Scaler().fit(X.values[tr])
# Task B learns a correction of the TSO forecast (residual target); Task A predicts the load itself
base = df["total load forecast"].interpolate(limit_direction="both").values if TASK == "B" else np.zeros(len(y))
resid = y.values - base
ys = Scaler().fit(resid[tr]); yl = Scaler().fit(y.values[tr])
Xn = Xs.t(X.values).astype(np.float32); yn = ys.t(resid).astype(np.float32)
load_n = ((load.values - yl.m) / yl.s).astype(np.float32)
cal_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "weekend"]
if TASK == "B": cal_cols = cal_cols + ["tso", "tso_err24", "tso_err48", "tso_err168"]
cal = (Xs.t(X.values)[:, [cols.index(c) for c in cal_cols]] if TASK == "B" else X[cal_cols].values).astype(np.float32)
itr, iva, ite = np.where(tr)[0], np.where(va)[0], np.where(te)[0]
ytrue_te = y.values[ite]
rng = np.random.default_rng(123)

def inputs(idx, noise=0.0):
    """Model-specific inputs; optional multiplicative noise on all load-derived inputs."""
    if name in ("ridge", "lgbm", "mlp", "tsk", "tsk8"):
        Xr = X.values[idx].copy()
        if noise:
            lc = [i for i, c in enumerate(cols) if c.startswith(("lag", "roll"))]
            Xr[:, lc] *= (1 + rng.normal(0, noise, size=(len(idx), len(lc))))
        Z = Xs.t(Xr).astype(np.float32)
        if name.startswith("tsk"): Z = Z[:, [cols.index(c) for c in FUZZY_INPUTS]]
        return (Z,)
    s, c = seq_windows(load_n, cal, idx)
    if noise:
        raw = s * yl.s + yl.m
        raw = raw * (1 + rng.normal(0, noise, size=raw.shape))
        s = ((raw - yl.m) / yl.s).astype(np.float32)
    return (s, c)

if name == "ridge":
    grid = [dict(alpha=a) for a in [0.1, 1, 10, 100, 1000, 10000]]
elif name == "lgbm":
    grid = [dict(num_leaves=l, learning_rate=r, min_child_samples=m)
            for l, r, m in [(31, .05, 20), (63, .05, 20), (31, .1, 20), (63, .1, 50), (127, .05, 50), (15, .1, 20)]]
elif name == "mlp":
    grid = [dict(h=h, depth=d) for h, d in itertools.product([64, 128, 256], [2, 3])]
elif name == "lstm":
    grid = [dict(h=h, lr=lr) for h, lr in itertools.product([32, 64, 128], [1e-3, 3e-3])]
elif name == "transformer":
    grid = [dict(d=d, layers=l) for d, l in itertools.product([16, 32, 64], [1, 2])]
elif name == "tsk8":
    grid = [dict(R=8, lr=1e-2)]
elif name == "tsk":
    grid = [dict(R=R, lr=lr) for R, lr in [(4, 1e-2), (8, 1e-2), (16, 1e-2), (32, 1e-2), (16, 3e-3), (32, 3e-3)]]

def fit(cfg, seed):
    Itr, Iva = inputs(itr), inputs(iva)
    if name == "ridge":
        m = Ridge(alpha=cfg["alpha"]).fit(Itr[0], yn[itr]); return m, None
    if name == "lgbm":
        m = lgb.LGBMRegressor(n_estimators=3000, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              objective="l1", random_state=seed, verbose=-1, n_jobs=2,
                              **cfg)
        m.fit(Itr[0], yn[itr], eval_set=[(Iva[0], yn[iva])], callbacks=[lgb.early_stopping(50, verbose=False)])
        return m, m.best_iteration_
    torch.manual_seed(seed)
    if name == "mlp": net = MLP(Itr[0].shape[1], cfg["h"], cfg["depth"]); lr = 1e-3
    elif name == "lstm": net = LSTMNet(len(cal_cols), cfg["h"]); lr = cfg["lr"]
    elif name == "transformer": net = TransformerNet(len(cal_cols), cfg["d"], 4, cfg["layers"]); lr = 1e-3
    elif name.startswith("tsk"):
        km = KMeans(cfg["R"], n_init=3, random_state=seed).fit(Itr[0])
        net = TSK(Itr[0].shape[1], cfg["R"], km.cluster_centers_); lr = cfg["lr"]
    net, best, ep = train_torch(net, Itr, yn[itr], Iva, yn[iva], lr=lr, seed=seed)
    return net, ep

def predict(m, I):
    if name in ("ridge", "lgbm"): return m.predict(I[0])
    return predict_torch(m, I)

res = dict(model=name, hpo=[], final=[])
# ---- hyper-parameter search (seed 0) -----------------------------------------
best_cfg, best_v = None, 1e18
for cfg in grid:
    with CPUMeter() as cm:
        m, ep = fit(cfg, 0)
    v = mae(ys.inv(predict(m, inputs(iva))) + base[iva], y.values[iva])
    res["hpo"].append(dict(cfg=cfg, val_mae=v, cpu_s=cm.cpu, wall_s=cm.wall, epochs=ep))
    print(name, cfg, f"val MAE {v:.1f}  cpu {cm.cpu:.1f}s", flush=True)
    if v < best_v: best_v, best_cfg = v, cfg
res["best_cfg"] = best_cfg
# ---- final training, several seeds -------------------------------------------
preds, preds_noisy = [], []
Ite = inputs(ite)
for seed in range(NSEED):
    with CPUMeter() as cm:
        m, ep = fit(best_cfg, seed)
    p = ys.inv(predict(m, Ite)) + base[ite]
    # inference cost: full test year (8760 day-ahead hourly forecasts), 5 repetitions
    reps = []
    for _ in range(5):
        with CPUMeter() as ci: predict(m, Ite)
        reps.append(ci.cpu)
    pn = ys.inv(predict(m, inputs(ite, noise=0.03))) + base[ite]
    n_par = (sum(p.numel() for p in m.parameters()) if hasattr(m, "parameters") and callable(m.parameters)
             else (m.coef_.size + 1 if name == "ridge" else int(m.booster_.num_trees())))
    res["final"].append(dict(seed=seed, cpu_train_s=cm.cpu, wall_train_s=cm.wall, epochs=ep,
                             cpu_infer_year_s=float(np.median(reps)), test_mae=mae(p, ytrue_te),
                             test_rmse=rmse(p, ytrue_te), test_mape=mape(p, ytrue_te),
                             test_mae_noise3=mae(pn, ytrue_te), n_params=n_par))
    preds.append(p); preds_noisy.append(pn)
    print(name, "seed", seed, res["final"][-1], flush=True)
    if seed == 0:
        with open(os.path.join(OUT, f"model_{name}.pkl"), "wb") as f:
            pickle.dump(dict(model=m, cfg=best_cfg), f)
np.savez(os.path.join(OUT, f"pred_{name}.npz"), pred=np.array(preds), pred_noisy=np.array(preds_noisy),
         y=ytrue_te, time=np.array(load.index[ite].astype(str)))
save_json(res, f"res_{name}.json")
