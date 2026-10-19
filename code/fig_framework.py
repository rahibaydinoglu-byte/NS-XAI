"""Fig. 1: NS-XAI system boundary and evaluation flow (schematic)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.6})
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#6b6b6b"

fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(-1.5, 58); ax.axis("off")
def box(x, y, w, h, title, lines, ec, fc="#ffffff", lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", ec=ec, fc=fc, lw=lw, ls=ls))
    ax.text(x + 1.2, y + h - 2.2, title, weight="bold", color=INK, va="top", fontsize=6.8)
    for i, l in enumerate(lines):
        ax.text(x + 1.2, y + h - 6.0 - i * 3.3, l, color=INK, va="top", fontsize=6.3)
def arrow(x0, y0, x1, y1, c=MUTED):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1))

# declared boundary
ax.add_patch(FancyBboxPatch((1, 13), 98, 43, boxstyle="round,pad=0.3,rounding_size=1.5", ec=MUTED, fc="none", lw=1, ls=(0, (4, 3))))
ax.text(2.5, 54.6, "Declared assessment boundary (horizon H, deployment profile K, N)", color=MUTED, fontsize=6.5, va="top")
box(3, 16, 29, 34, r"AI-side burden $\tilde C$", [r"Hyper-parameter search $S^{hpo}$", r"Training $\times$ K retrainings", r"Inference $N\, s^{inf}$",
    r"Explanation $N\, s^{xai}$", r"$\times$ core power $\tilde\kappa$ $\times$ PUE $\tilde\rho$", r"$\times$ carbon intensity $CI_t$"], ORANGE)
box(68, 16, 29, 34, r"Application-side benefit $\tilde B$", ["Reference (counterfactual) forecast", r"$\delta = \Delta MAE\, /\, \bar L$", r"Avoided imbalance $\delta L H$",
    r"$\times$ fossil share $\tilde s$", r"$\times$ balancing factor $\tilde e_{bal}$", r"$\rightarrow$ avoided CO$_2$e"], AQUA)
box(36, 33, 28, 17, "Candidate model m", ["accuracy · cost · explainability", "stability · robustness"], BLUE)
box(36, 16, 28, 14, "Explainability criteria", ["intrinsic vs post-hoc", r"cost $s^{xai}$, stability"], BLUE, ls="--")
arrow(36, 41, 32.6, 41); arrow(64, 41, 67.4, 41)
# outputs
box(3, 0, 94, 10.5, "Outputs", [], INK, fc="#f4f4f2")
ax.text(4.2, 5.2, r"Fuzzy SBCR $\alpha$-cuts (Prop. 1)  |  break-even scale $\tilde L^*$ (Prop. 2)  |  ranking-reversal scales (Prop. 3)", fontsize=6.2, va="center", color=INK)
ax.text(4.2, 1.9, r"Possibility / necessity of net-positivity  |  Pareto set (Lemma 1)  |  incremental benefit-cost ratio (IBCR)", fontsize=6.2, va="center", color=INK)
arrow(50, 13, 50, 11.0)
fig.savefig(os.path.join(OUT, "fig1_framework.png"), dpi=300, bbox_inches="tight"); print("ok")
