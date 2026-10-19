"""Shared data preparation, models and measurement utilities for the NS-XAI case study.

Task: day-ahead (24 h-ahead) hourly national load forecasting for Spain (ENTSO-E data,
2015-2018, distributed with the darts library). Train 2015-2016, validation 2017, test 2018.
"""
import os, time, json, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "energy_dataset.csv")
TASK = os.environ.get("TASK", "A")   # A: forecast from history; B: post-process the TSO day-ahead forecast
OUT = os.path.join(ROOT, "results" if TASK == "A" else "results_B")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2)

H = 24          # forecast horizon (day-ahead)
SEQ = 168       # look-back window for sequence models (one week)

# ---------------------------------------------------------------- data
def load_raw():
    df = pd.read_csv(DATA)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Madrid")
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated()]
    load = df["total load actual"].interpolate(limit_direction="both")
    return df, load

def tabular_features(load, tso=None):
    """Features available at issue time t-H for target hour t (no leakage)."""
    f = pd.DataFrame(index=load.index)
    if TASK == "B":  # TSO day-ahead forecast for the target hour is published before issue time
        f["tso"] = tso
        f["tso_err24"] = (load - tso).shift(24)
        f["tso_err48"] = (load - tso).shift(48)
        f["tso_err168"] = (load - tso).shift(168)
    for lag in [24, 25, 26, 48, 72, 96, 120, 144, 168, 336]:
        f[f"lag{lag}"] = load.shift(lag)
    r = load.shift(24)
    f["roll24_mean"] = r.rolling(24).mean()
    f["roll24_max"] = r.rolling(24).max()
    f["roll24_min"] = r.rolling(24).min()
    f["roll168_mean"] = r.rolling(168).mean()
    idx = load.index
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24); f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7); f["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25); f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["weekend"] = (idx.dayofweek >= 5).astype(float)
    return f

FUZZY_INPUTS = (["lag24", "lag168", "roll24_mean", "hour_sin", "hour_cos", "weekend"] if TASK == "A"
                else ["tso", "tso_err24", "tso_err168", "lag24", "hour_sin", "weekend"])

def make_splits():
    df, load = load_raw()
    tso = df["total load forecast"].interpolate(limit_direction="both")
    X = tabular_features(load, tso)
    y = load.rename("y")
    ok = X.notna().all(axis=1)
    # sequence windows need SEQ+H history
    ok &= np.arange(len(load)) >= (SEQ + H + 336)
    yr = load.index.year
    tr = ok & (yr <= 2016); va = ok & (yr == 2017); te = ok & (yr == 2018)
    return df, load, X, y, tr.values, va.values, te.values

class Scaler:
    def fit(self, a):
        self.m = a.mean(0); self.s = a.std(0) + 1e-9; return self
    def t(self, a): return (a - self.m) / self.s
    def inv(self, a): return a * self.s + self.m

def seq_windows(load_np, cal_np, idx):
    """For target index i: loads[i-H-SEQ+1 .. i-H] (scaled) + target calendar features."""
    w = np.stack([load_np[i - H - SEQ + 1:i - H + 1] for i in idx])
    return w[..., None].astype(np.float32), cal_np[idx].astype(np.float32)

# ---------------------------------------------------------------- torch models
class MLP(nn.Module):
    def __init__(self, d, h=128, depth=2, drop=0.1):
        super().__init__()
        layers, k = [], d
        for _ in range(depth):
            layers += [nn.Linear(k, h), nn.ReLU(), nn.Dropout(drop)]; k = h
        layers += [nn.Linear(k, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

class LSTMNet(nn.Module):
    def __init__(self, dcal, h=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, h, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h + dcal, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, s, c):
        o, _ = self.lstm(s)
        return self.head(torch.cat([o[:, -1], c], -1)).squeeze(-1)

class TransformerNet(nn.Module):
    def __init__(self, dcal, d=32, heads=4, layers=2, patch=24):
        super().__init__()
        self.patch = patch
        self.inp = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ // patch, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d * (SEQ // patch) + dcal, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, s, c):
        b = s.shape[0]
        p = s.reshape(b, SEQ // self.patch, self.patch)
        z = self.enc(self.inp(p) + self.pos)
        return self.head(torch.cat([z.reshape(b, -1), c], -1)).squeeze(-1)

class TSK(nn.Module):
    """First-order Takagi-Sugeno-Kang neuro-fuzzy model (ANFIS-type) with R rules and
    Gaussian antecedents; product t-norm; normalised firing strengths."""
    def __init__(self, d, R=16, centers=None):
        super().__init__()
        c = torch.tensor(centers, dtype=torch.float32) if centers is not None else torch.randn(R, d)
        self.c = nn.Parameter(c)
        self.log_s = nn.Parameter(torch.zeros(R, d))
        self.A = nn.Parameter(torch.zeros(R, d)); self.b = nn.Parameter(torch.zeros(R))
    def firing(self, x):
        z = (x[:, None, :] - self.c[None]) / torch.exp(self.log_s)[None]
        logw = -0.5 * (z ** 2).sum(-1)
        return torch.softmax(logw, dim=1)           # normalised firing strengths
    def forward(self, x):
        w = self.firing(x)
        f = (x[:, None, :] * self.A[None]).sum(-1) + self.b[None]
        return (w * f).sum(-1)

def train_torch(model, train_inputs, ytr, val_inputs, yva, lr=1e-3, bs=256, epochs=40, patience=5, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_t = [torch.tensor(a) for a in train_inputs]; ytr_t = torch.tensor(ytr, dtype=torch.float32)
    va_t = [torch.tensor(a) for a in val_inputs]; yva_t = torch.tensor(yva, dtype=torch.float32)
    n = len(ytr_t); best, best_state, bad, ep_run = 1e18, None, 0, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for k in range(0, n, bs):
            b = perm[k:k + bs]
            opt.zero_grad()
            loss = nn.functional.l1_loss(model(*[t[b] for t in tr_t]), ytr_t[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.l1_loss(model(*va_t), yva_t).item()
        ep_run = ep + 1
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model, best, ep_run

def predict_torch(model, inputs, bs=4096):
    model.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(inputs[0]), bs):
            out.append(model(*[torch.tensor(a[k:k + bs]) for a in inputs]).numpy())
    return np.concatenate(out)

# ---------------------------------------------------------------- measurement
class CPUMeter:
    """Measures process CPU time (all threads) and wall time. Energy is derived later from a
    declared per-core power model, because the container exposes no RAPL/power telemetry."""
    def __enter__(self):
        self.c0, self.w0 = time.process_time(), time.perf_counter(); return self
    def __exit__(self, *a):
        self.cpu = time.process_time() - self.c0; self.wall = time.perf_counter() - self.w0

def mae(a, b): return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b): return float(np.mean(np.abs(a - b) / np.abs(a)) * 100)

def save_json(obj, name):
    with open(os.path.join(OUT, name), "w") as f: json.dump(obj, f, indent=2, default=float)
