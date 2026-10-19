"""NS-XAI analysis: carbon intensity, fuzzy energy/carbon accounting, fuzzy SBCR, break-even
scale, possibility/necessity of net-positivity, IBCR, Pareto analysis, statistics, carbon-aware
scheduling. All numbers reported in the manuscript are produced by this script."""
import json, itertools
import numpy as np, pandas as pd
from scipy import stats
from common import *

MODELS = ["ridge", "tsk8", "tsk", "lgbm", "mlp", "transformer", "lstm"]
LABEL = dict(naive="Seasonal naive", ridge="Ridge", tsk8="TSK-8 (compact)", tsk="TSK-32", lgbm="LightGBM",
             mlp="MLP", transformer="Transformer", lstm="LSTM")
df, load, X, y, tr, va, te = make_splits()
ite = np.where(te)[0]
ytrue = y.values[ite]
times = load.index[ite]
A = np.linspace(0, 1, 101)                       # alpha grid

# ---------------------------------------------------------------- fuzzy helpers (TFN = (l, m, u))
def cut(t, a): l, m, u = t; return np.array([l + a * (m - l), u - a * (u - m)])
def tfn_prod(*ts):      # alpha-cuts of a product of positive TFNs (exact for alpha-cuts)
    lo = np.prod([cut(t, A)[0] for t in ts], 0); hi = np.prod([cut(t, A)[1] for t in ts], 0); return lo, hi
def crisp(t): return t if np.isscalar(t) else None

# ---------------------------------------------------------------- 1. hourly carbon intensity (gCO2e/kWh)
EF = {  # IPCC AR5 WGIII Annex III lifecycle medians; 'assumed' marks categories without an IPCC entry
 "generation biomass": (230, "IPCC"), "generation fossil brown coal/lignite": (820, "IPCC coal PC (lower bound for lignite)"),
 "generation fossil coal-derived gas": (820, "assumed = coal"), "generation fossil gas": (490, "IPCC gas CC"),
 "generation fossil hard coal": (820, "IPCC"), "generation fossil oil": (650, "assumed = IPCC gas-CC upper bound"),
 "generation geothermal": (38, "IPCC"), "generation hydro run-of-river and poundage": (24, "IPCC"),
 "generation hydro water reservoir": (24, "IPCC"), "generation marine": (17, "IPCC"), "generation nuclear": (12, "IPCC"),
 "generation other": (490, "assumed = gas CC"), "generation other renewable": (230, "assumed = biomass"),
 "generation solar": (48, "IPCC PV utility"), "generation waste": (490, "assumed = gas CC"),
 "generation wind offshore": (12, "IPCC"), "generation wind onshore": (11, "IPCC")}
gen = df[list(EF)].fillna(0).clip(lower=0)
ci = (gen * pd.Series({k: v[0] for k, v in EF.items()})).sum(axis=1) / gen.sum(axis=1).replace(0, np.nan)
N_CI_MISSING = int(ci.loc[load.index[ite]].isna().sum())
ci = ci.interpolate(limit_direction='both')
ci_te = ci.loc[times]
assumed = [k for k, v in EF.items() if "assumed" in v[1]]
share_assumed = float(gen.loc[times, assumed].sum().sum() / gen.loc[times].sum().sum())
CI_MEAN = float(ci_te.mean())                    # g/kWh
ci_info = dict(mean=CI_MEAN, min=float(ci_te.min()), max=float(ci_te.max()), p5=float(ci_te.quantile(.05)),
               p95=float(ci_te.quantile(.95)), share_generation_assumed_EF=share_assumed, n_hours_interpolated=N_CI_MISSING)

# ---------------------------------------------------------------- 2. accuracy
res = {m: json.load(open(os.path.join(OUT, f"res_{m}.json"))) for m in MODELS}
pred = {m: np.load(os.path.join(OUT, f"pred_{m}.npz")) for m in MODELS}
xai = json.load(open(os.path.join(OUT, "xai.json")))
naive = load.shift(168).values[ite]
tso = df["total load forecast"].reindex(times).interpolate().values
acc = {"naive": dict(mae=mae(naive, ytrue), rmse=rmse(naive, ytrue), mape=mape(naive, ytrue)),
       "tso": dict(mae=mae(tso, ytrue), rmse=rmse(tso, ytrue), mape=mape(tso, ytrue))}
rows = []
for m in MODELS:
    f = pd.DataFrame(res[m]["final"])
    hpo_cpu = sum(h["cpu_s"] for h in res[m]["hpo"])
    if m == "tsk8": hpo_cpu = sum(h["cpu_s"] for h in res["tsk"]["hpo"])   # shares the TSK search
    rows.append(dict(model=m, mae=f.test_mae.mean(), mae_sd=f.test_mae.std(ddof=1), rmse=f.test_rmse.mean(),
                     mape=f.test_mape.mean(), mae_noise=f.test_mae_noise3.mean(),
                     degr_pct=100 * (f.test_mae_noise3.mean() / f.test_mae.mean() - 1),
                     n_params=int(f.n_params.median()), cpu_train=f.cpu_train_s.mean(), cpu_hpo=hpo_cpu,
                     cpu_infer_day=f.cpu_infer_year_s.mean() / 365, cpu_xai_day=xai[m]["cpu_s_per_day"],
                     stability=xai[m]["stability"], best_cfg=str(res[m]["best_cfg"])))
T = pd.DataFrame(rows).set_index("model")

# ---------------------------------------------------------------- 3. fuzzy AI-side energy & carbon (annual)
K_RETRAIN = 12                                   # monthly retraining
P_CORE = (4.0, 8.0, 15.0)                        # W per busy core (TFN, declared assumption)
PUE = (1.1, 1.2, 1.6)                            # facility overhead (TFN, declared assumption)
T["cpu_year"] = T.cpu_hpo + K_RETRAIN * T.cpu_train + 365 * (T.cpu_infer_day + T.cpu_xai_day)
T["share_xai"] = 365 * T.cpu_xai_day / T.cpu_year
T["share_dev"] = (T.cpu_hpo + K_RETRAIN * T.cpu_train) / T.cpu_year
def energy_kwh(cpu_s):  # alpha-cuts (lo, hi) of annual energy in kWh
    lo, hi = tfn_prod(P_CORE, PUE); return cpu_s * lo / 3.6e6, cpu_s * hi / 3.6e6
for m in MODELS:
    lo, hi = energy_kwh(T.loc[m, "cpu_year"])
    T.loc[m, "E_kwh_mode"] = lo[-1]; T.loc[m, "E_kwh_lo"] = lo[0]; T.loc[m, "E_kwh_hi"] = hi[0]
    T.loc[m, "C_kg_mode"] = lo[-1] * CI_MEAN / 1000; T.loc[m, "C_kg_lo"] = lo[0] * CI_MEAN / 1000
    T.loc[m, "C_kg_hi"] = hi[0] * CI_MEAN / 1000

# ---------------------------------------------------------------- 4. benefit model & fuzzy SBCR
S_FOSSIL = (0.05, 0.25, 0.60)                    # share of imbalance energy met by additional fossil output
EF_BAL = (0.41, 0.49, 0.65)                      # tCO2e/MWh, IPCC gas-CC min/median/max
L_ES = float(load.loc[times].mean())             # MW, mean Spanish load in 2018
HOURS = 8760
REF = "naive" if TASK == "A" else "tso"           # A: seasonal naive; B: operational TSO forecast
T["dmae"] = acc[REF]["mae"] - T.mae              # MW, improvement over the reference forecast
T["delta_rel"] = T.dmae / L_ES                   # scale-free improvement (fraction of mean load)
phi_lo, phi_hi = tfn_prod(S_FOSSIL, EF_BAL)      # tCO2e per MWh of avoided imbalance

# statistical uncertainty of delta: moving-block bootstrap (7-day blocks) of the mean daily MAE difference
ref_pred = naive if REF == "naive" else tso
ndays = len(ytrue) // 24
def dmae_daily(p): return (np.abs(ref_pred - ytrue) - np.abs(p - ytrue))[:ndays * 24].reshape(ndays, 24).mean(1)
brng = np.random.default_rng(2024); BLK = 7; NB = 2000
starts = np.arange(ndays - BLK + 1)
for m in MODELS:
    d = np.mean([dmae_daily(p) for p in pred[m]["pred"]], 0)
    bs = []
    for _ in range(NB):
        idx = np.concatenate([np.arange(s0, s0 + BLK) for s0 in brng.choice(starts, int(np.ceil(ndays / BLK)))])[:ndays]
        bs.append(d[idx].mean())
    T.loc[m, "dmae_lo"], T.loc[m, "dmae_hi"] = np.percentile(bs, [2.5, 97.5])
T["delta_lo"] = T.dmae_lo / L_ES; T["delta_hi"] = T.dmae_hi / L_ES
def delta_cuts(m):   # TFN (lower CI, mean, upper CI)
    return cut((T.loc[m, "delta_lo"], T.loc[m, "delta_rel"], T.loc[m, "delta_hi"]), A)

def benefit_cuts(m, L_mw):   # kgCO2e/yr; exact alpha-cuts, delta may have negative support
    dl, dh = delta_cuts(m); k = L_mw * HOURS * 1000
    lo = np.where(dl >= 0, dl * phi_lo, dl * phi_hi) * k
    hi = np.where(dh >= 0, dh * phi_hi, dh * phi_lo) * k
    return lo, hi
def ratio_cuts(B_lo, B_hi, C_lo, C_hi):
    return np.where(B_lo >= 0, B_lo / C_hi, B_lo / C_lo), np.where(B_hi >= 0, B_hi / C_lo, B_hi / C_hi)
def lstar_cuts(m, C_lo, C_hi):  # MW; upper end infinite when delta support reaches zero
    dl, dh = delta_cuts(m)
    lo = np.where(dh > 0, C_lo / (np.maximum(dh, 1e-300) * HOURS * 1000 * phi_hi), np.inf)
    hi = np.where(dl > 0, C_hi / (np.maximum(dl, 1e-300) * HOURS * 1000 * phi_lo), np.inf)
    return lo, hi
def pos_nec(lo, hi, thr=1.0):
    """Pos(SBCR>=thr) = sup{a: upper_a >= thr};  Nec(SBCR>=thr) = 1 - sup{a: lower_a < thr}."""
    pos = A[hi >= thr].max() if (hi >= thr).any() else 0.0
    nec = 1.0 - A[lo < thr].max() if (lo < thr).any() else 1.0
    return float(pos), float(nec)
def cost_cuts(m, host):
    e_lo, e_hi = energy_kwh(T.loc[m, "cpu_year"]); C_lo, C_hi = e_lo * CI_MEAN / 1000, e_hi * CI_MEAN / 1000
    if host:
        h_lo, h_hi = cut(P_HOST, A); C_lo = C_lo + h_lo * HOURS / 1000 * CI_MEAN / 1000; C_hi = C_hi + h_hi * HOURS / 1000 * CI_MEAN / 1000
    return C_lo, C_hi

P_HOST = (2.0, 5.0, 15.0)                        # W, dedicated always-on edge host (TFN, declared assumption)
SCALES = {"1 kW": 1e-3, "10 kW": 1e-2, "100 kW": 0.1, "1 MW": 1.0, "10 MW": 10.0, "National (ES)": L_ES}
sb, sbx = [], []
for m in MODELS:
    for host, col, lst in [(False, "Lstar", sb), (True, "LstarX", sbx)]:
        C_lo, C_hi = cost_cuts(m, host)
        Llo, Lhi = lstar_cuts(m, C_lo, C_hi)
        mode = Llo[-1] if T.loc[m, "delta_rel"] > 0 else np.inf
        T.loc[m, f"{col}_mode_kW"] = mode * 1000; T.loc[m, f"{col}_lo_kW"] = Llo[0] * 1000; T.loc[m, f"{col}_hi_kW"] = Lhi[0] * 1000
        for s_, L in SCALES.items():
            B_lo, B_hi = benefit_cuts(m, L); lo, hi = ratio_cuts(B_lo, B_hi, C_lo, C_hi); p, n = pos_nec(lo, hi)
            lst.append(dict(model=m, scale=s_, L_MW=L, sbcr_mode=lo[-1], sbcr_lo=lo[0], sbcr_hi=hi[0], pos=p, nec=n))
    for s_, L in SCALES.items():   # compute budget (modal, model-only boundary), kWh/yr
        T.loc[m, f"Ebudget_kWh_{s_}"] = T.loc[m, "delta_rel"] * L * HOURS * S_FOSSIL[1] * EF_BAL[1] * 1000 / (CI_MEAN / 1000)
SB = pd.DataFrame(sb); SBX = pd.DataFrame(sbx); SBX.to_csv(os.path.join(OUT, "table_sbcr_host.csv"), index=False)
host_fraction = {m: float((P_HOST[1] * HOURS / 1000) / (P_HOST[1] * HOURS / 1000 + T.loc[m, "E_kwh_mode"])) for m in MODELS}

# ---------------------------------------------------------------- 5. Pareto analysis
crit = pd.DataFrame({"MAE": T.mae, "CO2": T.C_kg_mode, "XAIcost": T.cpu_xai_day, "Instab": 1 - T.stability,
                     "Degr": T.degr_pct})
def pareto(Dm):
    nd = []
    for i in Dm.index:
        dom = any(((Dm.loc[j] <= Dm.loc[i]).all() and (Dm.loc[j] < Dm.loc[i]).any()) for j in Dm.index if j != i)
        if not dom: nd.append(i)
    return nd
pareto_full = pareto(crit); pareto_2d = pareto(crit[["MAE", "CO2"]])
# incremental benefit-cost ratio along the 2-D front ordered by cost (renamed from 'MSE')
front = T.loc[pareto_2d].sort_values("C_kg_mode")
ibcr = []
for a, b in zip(front.index[:-1], front.index[1:]):
    dB_mwh = (T.loc[b, "dmae"] - T.loc[a, "dmae"]) * HOURS                       # MWh/yr imbalance at ES scale
    dB_kg_mode = dB_mwh * S_FOSSIL[1] * EF_BAL[1] * 1000
    dC = T.loc[b, "C_kg_mode"] - T.loc[a, "C_kg_mode"]
    dB_rel = (T.loc[b, "delta_rel"] - T.loc[a, "delta_rel"])
    ibcr.append(dict(frm=a, to=b, dB_MWh_ES=dB_mwh, dC_kg=dC, IBCR_ES=dB_kg_mode / dC,
                     breakeven_kW=1000 * dC / (dB_rel * HOURS * S_FOSSIL[1] * EF_BAL[1] * 1000)))
# rankings
rank_acc = T.mae.rank()
rank_sbcr = SB[SB.scale == "1 MW"].set_index("model").sbcr_mode.rank(ascending=False)   # scale-invariant (Prop. 2)
NBr = {}
for s, L in SCALES.items():
    nb = T.delta_rel * L * HOURS * S_FOSSIL[1] * EF_BAL[1] * 1000 - T.C_kg_mode      # kgCO2e/yr, modal
    NBr[s] = nb.rank(ascending=False)
    SB.loc[SB.scale == s, "NB_kg_mode"] = SB.loc[SB.scale == s, "model"].map(nb).values
kendall = dict(acc_vs_sbcr=float(stats.kendalltau(rank_acc[MODELS], rank_sbcr[MODELS])[0]),
               **{f"acc_vs_NB_{s}": float(stats.kendalltau(rank_acc[MODELS], r[MODELS])[0]) for s, r in NBr.items()})

# ---------------------------------------------------------------- 6. statistics on daily MAE
def daily_mae(p): return np.abs(p - ytrue).reshape(-1, 24).mean(1)
nd = len(ytrue) // 24
D = {m: np.mean([daily_mae(p[:nd * 24]) for p in pred[m]["pred"]], 0) for m in MODELS}
D[REF] = daily_mae((naive if REF == "naive" else tso)[:nd * 24])
names = list(D); M = np.column_stack([D[k] for k in names])
fr = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
ranks = stats.rankdata(M, axis=1).mean(0)
k = len(names); q05 = {8: 3.031, 7: 2.949}[k]                     # Nemenyi critical values (Demsar 2006)
CD = q05 * np.sqrt(k * (k + 1) / (6 * nd))
best = min(MODELS, key=lambda m: T.loc[m, "mae"])
def hln_dm(d):  # Diebold-Mariano with Harvey-Leybourne-Newbold correction, h = 1 on daily losses
    n = len(d); dm = d.mean() / np.sqrt(d.var(ddof=1) / n); dm *= np.sqrt((n - 1) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), n - 1))
tests = []
for m in names:
    if m == best: continue
    d = D[m] - D[best]
    dm, p = hln_dm(d); w = stats.wilcoxon(D[m], D[best])
    tests.append(dict(model=m, mean_diff=float(d.mean()), dm=dm, p_dm=p, p_wilcoxon=float(w.pvalue)))
tests = pd.DataFrame(tests)
# Holm correction
for col in ["p_dm", "p_wilcoxon"]:
    o = np.argsort(tests[col].values); adj = np.empty(len(o)); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(o) - r) * tests[col].values[i])); adj[i] = run
    tests[col + "_holm"] = adj

# ---------------------------------------------------------------- 7. carbon-aware scheduling of deferrable jobs
ci_te = ci.loc[times]
cid = ci_te.values[:nd * 24].reshape(nd, 24)
hours_local = times[:24].hour
sched = dict(fixed_12h=float(cid[:, list(hours_local).index(12)].mean()),
             fixed_20h=float(cid[:, list(hours_local).index(20)].mean()),
             daily_mean=float(cid.mean()), min_hour_in_day=float(cid.min(1).mean()))
sched["reduction_vs_mean_pct"] = 100 * (1 - sched["min_hour_in_day"] / sched["daily_mean"])
sched["reduction_vs_20h_pct"] = 100 * (1 - sched["min_hour_in_day"] / sched["fixed_20h"])
# only development energy (HPO + retraining) is deferrable; inference+XAI must run at issue time
T["C_kg_mode_carbonaware"] = T.E_kwh_mode * (T.share_dev * sched["min_hour_in_day"] + (1 - T.share_dev) * CI_MEAN) / 1000

# ---------------------------------------------------------------- 7b. sensitivity: retraining frequency, XAI on/off
sens = []
for Kr in [1, 4, 12, 52, 365]:
    for xai_on in [True, False]:
        for m in MODELS:
            cpu = T.loc[m, "cpu_hpo"] + Kr * T.loc[m, "cpu_train"] + 365 * (T.loc[m, "cpu_infer_day"] + xai_on * T.loc[m, "cpu_xai_day"])
            Ekwh = cpu * P_CORE[1] * PUE[1] / 3.6e6
            Lstar = Ekwh * CI_MEAN / 1000 / (T.loc[m, "delta_rel"] * HOURS * S_FOSSIL[1] * EF_BAL[1] * 1000)
            sens.append(dict(K=Kr, xai=xai_on, model=m, E_kwh=Ekwh, Lstar_kW=1000 * Lstar))
pd.DataFrame(sens).to_csv(os.path.join(OUT, "table_sensitivity.csv"), index=False)

# ---------------------------------------------------------------- 8. training-vs-inference crossover (Prop. 3)
T["crossover_days"] = (T.cpu_hpo + T.cpu_train) / (T.cpu_infer_day + T.cpu_xai_day)

summary = dict(task=TASK, ref=REF, ci=ci_info, acc_ref=acc, L_ES_MW=L_ES, scales=SCALES, sched=sched, friedman=dict(stat=fr.statistic,
               p=fr.pvalue, k=k, n=nd, CD=CD, ranks=dict(zip(names, ranks.tolist()))), best=best,
               pareto_full=pareto_full, pareto_2d=pareto_2d, ibcr=ibcr, kendall=kendall,
               host_fraction=host_fraction, assumptions=dict(P_HOST=P_HOST, K_RETRAIN=K_RETRAIN, P_CORE=P_CORE, PUE=PUE, S_FOSSIL=S_FOSSIL, EF_BAL=EF_BAL))
save_json(summary, "summary.json")
T.to_csv(os.path.join(OUT, "table_models.csv")); SB.to_csv(os.path.join(OUT, "table_sbcr.csv"), index=False)
tests.to_csv(os.path.join(OUT, "table_tests.csv"), index=False)
ci_te.to_csv(os.path.join(OUT, "ci_2018.csv"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(json.dumps({k: v for k, v in summary.items() if k not in ("scales",)}, indent=1, default=float))
print(T.round(4).T); print(SB.round(4).to_string()); print(tests.round(4))
