"""Result figures for the manuscript (both tasks)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "axes.labelcolor": "#222222", "xtick.color": "#444444", "ytick.color": "#444444",
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
MODELS = ["ridge", "tsk8", "tsk", "lgbm", "mlp", "transformer", "lstm"]
LAB = dict(ridge="Ridge", tsk8="TSK-8", tsk="TSK-32", lgbm="LightGBM", mlp="MLP", transformer="Transformer", lstm="LSTM")
COL = dict(ridge="#2a78d6", tsk8="#1baf7a", tsk="#008300", lgbm="#eda100", mlp="#e87ba4", transformer="#6250d6", lstm="#eb6834")
MRK = dict(ridge="o", tsk8="^", tsk="v", lgbm="s", mlp="D", transformer="P", lstm="X")
R = {t: os.path.join(ROOT, "results" if t == "A" else "results_B") for t in "AB"}
T = {t: pd.read_csv(os.path.join(R[t], "table_models.csv"), index_col=0) for t in "AB" if os.path.exists(os.path.join(R[t], "table_models.csv"))}
S = {t: json.load(open(os.path.join(R[t], "summary.json"))) for t in T}

# ---- Fig 2: annual AI-side CPU time decomposition (Task A), log scale stacked is misleading -> grouped shares
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
t = T["A"]; comp = pd.DataFrame({"Search": t.cpu_hpo, "Retraining (12×)": 12 * t.cpu_train,
                                 "Inference (365×)": 365 * t.cpu_infer_day, "Explanation (365×)": 365 * t.cpu_xai_day}).loc[MODELS]
ax = axes[0]; y = np.arange(len(MODELS)); ax.barh(y, comp.sum(1) / 3600, color="#8c8c8c", height=0.6)
for i, m in enumerate(MODELS): ax.text(comp.sum(1)[m] / 3600 * 1.15, i, f"{comp.sum(1)[m]/3600:.2g} h", va="center", fontsize=7, color="#222")
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([LAB[m] for m in MODELS]); ax.invert_yaxis()
ax.set_xlabel("Annual AI-side CPU time (core-hours, log scale)"); ax.set_title("(a) Total burden, Task A", loc="left", fontsize=8.5)
ax.grid(axis="y", visible=False)
ax = axes[1]; sh = comp.div(comp.sum(1), axis=0) * 100; left = np.zeros(len(MODELS))
cols = ["#184f95", "#5598e7", "#b7d3f6", "#eb6834"]
for c, k in zip(cols, sh.columns):
    ax.barh(y, sh[k], left=left, color=c, height=0.6, label=k, edgecolor="white", linewidth=1); left += sh[k].values
ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Share of annual CPU time (%)")
ax.set_title("(b) Composition", loc="left", fontsize=8.5); ax.grid(axis="y", visible=False)
ax.legend(ncol=2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.25), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_cost.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: break-even scale intervals (support and modal value), both tasks
fig, axes = plt.subplots(1, len(T), figsize=(7.2, 2.9), sharey=True)
axes = np.atleast_1d(axes)
for ax, (tk, t) in zip(axes, T.items()):
    for i, m in enumerate(MODELS):
        if t.loc[m, "delta_rel"] <= 0:
            ax.text(0.05, i, "no benefit (δ ≤ 0)", va="center", fontsize=6.8, color="#6b6b6b"); continue
        XMAX = 3e4; hi = t.loc[m, "LstarX_hi_kW"]; open_end = not np.isfinite(hi); hi = min(hi, XMAX)
        ax.plot([t.loc[m, "LstarX_lo_kW"], hi], [i, i], color=COL[m], lw=2.2, solid_capstyle="round")
        if open_end: ax.plot(hi, i, ">", color=COL[m], ms=5)
        ax.plot(t.loc[m, "LstarX_mode_kW"], i, MRK[m], color=COL[m], ms=6, mec="white", mew=0.8)
    for v, lab in [(0.4, "household"), (100, "building"), (1e4, "district")]:
        ax.axvline(v, color="#bbbbbb", lw=0.7, ls=":"); ax.text(v * 1.15, len(MODELS) - 0.35, lab, fontsize=6.3, color="#6b6b6b", va="top")
    ax.set_xscale("log"); ax.set_xlim(0.01, 5e4); ax.set_ylim(len(MODELS) + 0.2, -0.6); ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([LAB[m] for m in MODELS])
    ax.set_xlabel("Break-even mean load L* incl. host (kW, log)"); ax.grid(axis="y", visible=False)
    ax.set_title(f"({'a' if tk=='A' else 'b'}) Task {tk}: reference = {'seasonal naive' if tk=='A' else 'TSO forecast'}", loc="left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_breakeven.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: accuracy vs annual AI carbon (Task A), Pareto front highlighted
fig, axes = plt.subplots(1, len(T), figsize=(7.2, 3.0))
axes = np.atleast_1d(axes)
for ax, (tk, t) in zip(axes, T.items()):
    pf = S[tk]["pareto_2d"]
    fr = t.loc[pf].sort_values("C_kg_mode"); ax.step(fr.C_kg_mode * 1000, fr.mae, where="post", color="#bbbbbb", lw=1, zorder=1)
    for m in MODELS:
        ax.errorbar(t.loc[m, "C_kg_mode"] * 1000, t.loc[m, "mae"], yerr=t.loc[m, "mae_sd"], fmt=MRK[m], color=COL[m], ms=6,
                    mec="white", mew=0.8, capsize=2, elinewidth=0.8, zorder=3)
        OFF = dict(A=dict(ridge=(5, 4), tsk8=(-30, 7), tsk=(6, 5), lgbm=(-8, -13), mlp=(4, 6), transformer=(4, -11), lstm=(5, 4)),
                   B=dict()).get(tk, {})
        ax.annotate(LAB[m] + ("*" if m in pf else ""), (t.loc[m, "C_kg_mode"] * 1000, t.loc[m, "mae"]), xytext=OFF.get(m, (5, 3)),
                    textcoords="offset points", fontsize=6.8, color="#222")
    ax.axhline(S[tk]["acc_ref"][S[tk]["ref"]]["mae"], color="#6b6b6b", lw=0.8, ls="--")
    ax.text(ax.get_xlim()[0] if False else t.C_kg_mode.min() * 1000, S[tk]["acc_ref"][S[tk]["ref"]]["mae"],
            " reference", va="bottom", fontsize=6.5, color="#6b6b6b")
    ax.set_xscale("log"); ax.set_xlabel("Annual AI-side emissions (gCO₂e, modal, log scale)"); ax.set_ylabel("Test MAE (MW)")
    ax.set_title(f"({'a' if tk=='A' else 'b'}) Task {tk}", loc="left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_pareto.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig 5: hourly carbon intensity profile 2018 + carbon-aware slot
ci = pd.read_csv(os.path.join(R["A"], "ci_2018.csv"), index_col=0).iloc[:, 0]
ci.index = pd.to_datetime(ci.index, utc=True).tz_convert("Europe/Madrid")
prof = ci.groupby(ci.index.hour).agg(["mean", lambda x: x.quantile(.1), lambda x: x.quantile(.9)]); prof.columns = ["mean", "p10", "p90"]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.fill_between(prof.index, prof.p10, prof.p90, color="#cde2fb", label="10–90th percentile")
ax.plot(prof.index, prof["mean"], color="#2a78d6", lw=2, label="mean")
ax.set_xlabel("Hour of day (local time)"); ax.set_ylabel("gCO₂e/kWh"); ax.set_xticks(range(0, 24, 3))
ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.set_title("Carbon intensity of Spanish electricity, 2018", loc="left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_ci.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("figures ok")
