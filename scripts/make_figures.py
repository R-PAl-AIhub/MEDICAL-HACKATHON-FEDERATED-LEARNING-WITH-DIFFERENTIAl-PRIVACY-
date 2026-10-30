"""
Builds every chart in assets/charts/ from numbers logged in main.ipynb.

No model is re-run here. Each number below is copied from a printed output in
main.ipynb (the cell number is in the comment next to it). The confusion
matrices are derived exactly from the logged per-class F1 scores, the logged
error count, and the class counts of the test split (234 NORMAL / 390 PNEUMONIA
in total, 78 / 130 per hospital). The script asserts that each derivation
lands on whole numbers and that the three hospitals add up to the global model.

Usage:  python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- logged data
# Global model on GLOBAL_TEST_DATASET (624 images)          -> main.ipynb cell 72
GLOBAL = dict(acc=0.8237179517745972, loss=0.40240588784217834,
              f1=(0.7544642, 0.8624999), n_normal=234, n_pneu=390)

# Global model on each hospital's test split (208 images)   -> main.ipynb cell 73
FED = {
    "Hospital 1": dict(acc=0.8125, loss=0.41457879543304443, f1=(0.73103446, 0.8560885)),
    "Hospital 2": dict(acc=0.8365384340286255, loss=0.39400988817214966, f1=(0.7671232, 0.87407404)),
    "Hospital 3": dict(acc=0.8221153616905212, loss=0.3986290395259857, f1=(0.76433116, 0.85714275)),
}
# Each hospital's own local model, before federation         -> cells 30, 44, 58
LOCAL = {
    "Hospital 1": dict(acc=0.7884615659713745, loss=0.6040990948677063, f1=(0.66666657, 0.84507036)),
    "Hospital 2": dict(acc=0.754807710647583, loss=0.7065857648849487, f1=(0.6277372, 0.81720424)),
    "Hospital 3": dict(acc=0.75, loss=1.1526081562042236, f1=(0.67499995, 0.79687494)),
}
HOSP_TEST = dict(n_normal=78, n_pneu=130)

# Local training loss in each federated round (1 epoch / hospital) -> cell 69
ROUND_LOSS = {
    "Hospital 1": [0.3223, 0.3008, 0.2842, 0.2679, 0.2759],
    "Hospital 2": [0.3073, 0.2816, 0.2667, 0.2684, 0.2373],
    "Hospital 3": [0.3037, 0.2949, 0.2703, 0.2491, 0.2484],
}


def f1d(x):
    """Format a percentage with one decimal, rounding halves up (81.25 -> 81.3)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(round(x, 9))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


# ------------------------------------------------------------ derivation
def confusion(acc, f1, n_normal, n_pneu):
    """Exact 2x2 confusion matrix from accuracy, per-class F1 and class counts.

    For a binary task, FP + FN of either class equals the total error count E,
    so F1_c = 2*TP_c / (2*TP_c + E)  =>  TP_c = F1_c * E / (2 * (1 - F1_c)).
    """
    n = n_normal + n_pneu
    errors = round(n * (1 - acc))
    tp_n = f1[0] * errors / (2 * (1 - f1[0]))
    tp_p = f1[1] * errors / (2 * (1 - f1[1]))
    assert abs(tp_n - round(tp_n)) < 0.05 and abs(tp_p - round(tp_p)) < 0.05, (tp_n, tp_p)
    tp_n, tp_p = round(tp_n), round(tp_p)
    assert tp_n + tp_p == n - errors
    # rows = true class, cols = predicted class (NORMAL, PNEUMONIA)
    return [[tp_n, n_normal - tp_n], [n_pneu - tp_p, tp_p]]


def summary(cm):
    (tn, fp), (fn, tp) = cm
    return dict(sens=tp / (tp + fn), spec=tn / (tn + fp),
                acc=(tn + tp) / (tn + fp + fn + tp))


CM_GLOBAL = confusion(GLOBAL["acc"], GLOBAL["f1"], GLOBAL["n_normal"], GLOBAL["n_pneu"])
CM_FED = {h: confusion(v["acc"], v["f1"], **HOSP_TEST) for h, v in FED.items()}
CM_LOCAL = {h: confusion(v["acc"], v["f1"], **HOSP_TEST) for h, v in LOCAL.items()}
# the global test set is the three hospital test splits combined
assert [[sum(CM_FED[h][r][c] for h in CM_FED) for c in range(2)] for r in range(2)] == CM_GLOBAL

# ------------------------------------------------------------------ styling
for f in Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Inter"   # falls back to the default sans if Inter is missing

THEMES = {
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", gray="#b4b2ab",
                  series=["#2a78d6", "#eb6834", "#1baf7a"],
                  seq=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark": dict(bg="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#262b33", axis="#3a3f47", gray="#5c5b56",
                 series=["#3987e5", "#d95926", "#199e70"],
                 seq=["#162a46", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"]),
}


def base(theme, w, h):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": "Inter", "font.size": 11,
        "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    return fig, ax, t


def title(fig, text, sub, t):
    fig.text(0.03, 0.95, text, fontsize=15, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.03, 0.865, sub, fontsize=10.5, color=t["ink2"], va="top")


def save(fig, name, theme):
    fig.savefig(OUT / f"{name}-{theme}.png", facecolor=fig.get_facecolor())
    plt.close(fig)


# ------------------------------------------------- 1. local vs federated
def chart_local_vs_fed(theme):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), dpi=200, sharey=True)
    t = THEMES[theme]
    base(theme, 1, 1); plt.close()  # apply rcParams
    fig.patch.set_facecolor(t["bg"])
    hosp = list(FED)
    panels = [("Test accuracy", lambda d: d["acc"]),
              ("Macro F1 (Normal + Pneumonia)", lambda d: sum(d["f1"]) / 2)]
    for ax, (name, fn) in zip(axes, panels):
        ax.set_facecolor(t["bg"])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        for i, h in enumerate(hosp):
            y = len(hosp) - 1 - i
            a, b = fn(LOCAL[h]) * 100, fn(FED[h]) * 100
            ax.plot([a, b], [y, y], color=t["grid"], lw=6, solid_capstyle="round", zorder=1)
            ax.scatter([a], [y], s=90, color=t["gray"], zorder=3, edgecolor=t["bg"], linewidth=2)
            ax.scatter([b], [y], s=110, color=t["series"][0], zorder=4, edgecolor=t["bg"], linewidth=2)
            ax.text(a - 0.8, y, f1d(a), ha="right", va="center", fontsize=10, color=t["ink2"])
            ax.text(b + 0.8, y, f1d(b), ha="left", va="center", fontsize=10.5,
                    color=t["ink"], fontweight="bold")
            ax.text(b + 4.6, y, f"+{f1d(b - a)} pts", ha="left", va="center", fontsize=9.5,
                    color=t["ink2"])
        ax.set_xlim(66, 92)
        ax.set_ylim(-0.6, len(hosp) - 0.4)
        ax.set_yticks(range(len(hosp)))
        ax.set_yticklabels(hosp[::-1], fontsize=11, color=t["ink"])
        ax.xaxis.grid(True, color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("%", color=t["muted"], fontsize=9.5)
        ax.set_title(name, loc="left", fontsize=11.5, color=t["ink"], fontweight="bold", pad=10)
    fig.text(0.03, 0.97, "Federation beat local-only training at every hospital",
             fontsize=15, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.03, 0.885, "Same 208-image test split per hospital.  ", fontsize=10.5,
             color=t["ink2"], va="top")
    # legend
    lx = 0.43
    fig.lines += []
    fig.text(lx, 0.885, "●", color=t["gray"], fontsize=13, va="top")
    fig.text(lx + 0.02, 0.885, "Local model, own data only", color=t["ink2"], fontsize=10.5, va="top")
    fig.text(lx + 0.26, 0.885, "●", color=t["series"][0], fontsize=13, va="top")
    fig.text(lx + 0.28, 0.885, "Federated global model", color=t["ink2"], fontsize=10.5, va="top")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.70, bottom=0.14, wspace=0.12)
    save(fig, "local_vs_federated", theme)


# ------------------------------------------------- 2. confusion matrix
def chart_confusion(theme):
    t = THEMES[theme]
    base(theme, 1, 1); plt.close()
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=200)
    fig.patch.set_facecolor(t["bg"]); ax.set_facecolor(t["bg"])
    cm = CM_GLOBAL
    labels = ["Normal", "Pneumonia"]
    rows = [sum(r) for r in cm]
    for r in range(2):
        for c in range(2):
            v = cm[r][c]
            share = v / rows[r]
            correct = r == c
            fill = t["seq"][3] if correct else t["seq"][1]
            if theme == "dark":
                fill = t["seq"][2] if correct else t["seq"][1]
            ax.add_patch(plt.Rectangle((c + 0.02, 1 - r + 0.02), 0.96, 0.96, color=fill,
                                       lw=0, joinstyle="round"))
            txt = "#ffffff" if (correct or theme == "dark") else "#0b0b0b"
            ax.text(c + 0.5, 1 - r + 0.56, f"{v}", ha="center", va="center", fontsize=26,
                    fontweight="bold", color=txt)
            ax.text(c + 0.5, 1 - r + 0.30, f"{share * 100:.1f}%",
                    ha="center", va="center", fontsize=11, color=txt)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(labels, fontsize=11, color=t["ink"])
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(labels, fontsize=11, color=t["ink"], rotation=90,
                                                   va="center")
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("Predicted", fontsize=10, color=t["muted"], labelpad=8)
    ax.set_ylabel("Actual", fontsize=10, color=t["muted"], labelpad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    s = summary(cm)
    fig.text(0.5, 0.06, f"Sensitivity {s['sens'] * 100:.1f}%   ·   Specificity {s['spec'] * 100:.1f}%"
             f"   ·   Accuracy {s['acc'] * 100:.1f}%", ha="center", fontsize=10.5, color=t["ink2"])
    fig.text(0.04, 0.965, "Global model on 624 test X-rays", fontsize=14, fontweight="bold",
             color=t["ink"], va="top")
    fig.text(0.04, 0.905, "Count, and share of each actual class", fontsize=10, color=t["ink2"], va="top")
    fig.subplots_adjust(left=0.13, right=0.96, top=0.74, bottom=0.13)
    save(fig, "confusion_matrix", theme)


# ------------------------------------------------- 3. per-hospital consistency
def chart_consistency(theme):
    t = THEMES[theme]
    base(theme, 1, 1); plt.close()
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    fig.patch.set_facecolor(t["bg"]); ax.set_facecolor(t["bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    metrics = [("Accuracy", lambda h: FED[h]["acc"], GLOBAL["acc"]),
               ("Macro F1", lambda h: sum(FED[h]["f1"]) / 2, sum(GLOBAL["f1"]) / 2),
               ("Sensitivity (Pneumonia recall)", lambda h: summary(CM_FED[h])["sens"],
                summary(CM_GLOBAL)["sens"]),
               ("Specificity (Normal recall)", lambda h: summary(CM_FED[h])["spec"],
                summary(CM_GLOBAL)["spec"])]
    offsets = [0.17, 0.0, -0.17]
    for i, (name, fn, g) in enumerate(metrics):
        y = len(metrics) - 1 - i
        vals = [fn(h) * 100 for h in FED]
        ax.plot([min(vals), max(vals)], [y, y], color=t["grid"], lw=14, solid_capstyle="round", zorder=1)
        for j, h in enumerate(FED):
            ax.scatter(vals[j], y + offsets[j] * 0, s=95, color=t["series"][j], zorder=3,
                       edgecolor=t["bg"], linewidth=2, label=h if i == 0 else None)
        ax.scatter(g * 100, y, marker="|", s=420, color=t["ink"], zorder=4, linewidth=2,
                   label="All hospitals combined" if i == 0 else None)
        ax.text(max(vals) + 1.2, y, f"spread {max(vals) - min(vals):.1f} pts", va="center",
                fontsize=9.5, color=t["ink2"])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[0] for m in metrics][::-1], fontsize=10.5, color=t["ink"])
    ax.set_xlim(60, 100)
    ax.set_ylim(-0.6, len(metrics) - 0.4)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("%", color=t["muted"], fontsize=9.5)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.005, 1.17), ncol=4, frameon=False,
                    fontsize=10, handletextpad=0.3, columnspacing=1.4)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.text(0.03, 0.97, "How the one global model scores at each hospital",
             fontsize=15, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.03, 0.885, "Accuracy and F1 stay within 3 points. Specificity varies the most.",
             fontsize=10.5, color=t["ink2"], va="top")
    fig.subplots_adjust(left=0.27, right=0.95, top=0.70, bottom=0.13)
    save(fig, "hospital_consistency", theme)


# ------------------------------------------------- 4. federated rounds
def chart_rounds(theme):
    fig, ax, t = base(theme, 8.4, 4.0)
    rounds = [1, 2, 3, 4, 5]
    for j, (h, vals) in enumerate(ROUND_LOSS.items()):
        ax.plot(rounds, vals, color=t["series"][j], lw=2, zorder=3)
        ax.scatter(rounds, vals, s=34, color=t["series"][j], zorder=4, edgecolor=t["bg"], linewidth=1.5)
        ax.text(5.12, vals[-1] + (0.002 if j == 2 else (-0.004 if j == 1 else 0.0)), h,
                va="center", fontsize=10, color=t["ink"])
    ax.set_xticks(rounds); ax.set_xticklabels([f"Round {r}" for r in rounds], color=t["muted"])
    ax.set_xlim(0.8, 5.75)
    ax.set_ylim(0.22, 0.335)
    ax.yaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.spines["bottom"].set_color(t["axis"])
    ax.set_ylabel("Local training loss", fontsize=10, color=t["muted"])
    title(fig, "Local training loss trended down over 5 federated rounds",
          "1 local epoch per round, then weighted FedAvg on the server.", t)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.74, bottom=0.12)
    save(fig, "federated_rounds", theme)


if __name__ == "__main__":
    for th in THEMES:
        chart_local_vs_fed(th)
        chart_confusion(th)
        chart_consistency(th)
        chart_rounds(th)
    print("Global confusion matrix [[TN, FP], [FN, TP]]:", CM_GLOBAL, summary(CM_GLOBAL))
    for h in FED:
        print(h, "local", CM_LOCAL[h], {k: round(v, 3) for k, v in summary(CM_LOCAL[h]).items()},
              "| federated", CM_FED[h], {k: round(v, 3) for k, v in summary(CM_FED[h]).items()})
    print("charts written to", OUT)
    print("\n| Model | Evaluated on | Accuracy | Macro F1 | Sensitivity | Specificity | Loss |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for h in FED:
        for kind, src, cm in (("Local only", LOCAL, CM_LOCAL), ("**Federated global**", FED, CM_FED)):
            d, sm = src[h], summary(cm[h])
            b = "**" if kind.startswith("**") else ""
            print(f"| {kind} | {h} test (208) | {b}{f1d(d['acc']*100)}%{b} | {b}{sum(d['f1'])/2:.3f}{b} | "
                  f"{f1d(sm['sens']*100)}% | {f1d(sm['spec']*100)}% | {d['loss']:.3f} |")
    sm = summary(CM_GLOBAL)
    print(f"| **Federated global** | All 3 test splits (624) | **{f1d(GLOBAL['acc']*100)}%** | "
          f"**{sum(GLOBAL['f1'])/2:.3f}** | {f1d(sm['sens']*100)}% | {f1d(sm['spec']*100)}% | {GLOBAL['loss']:.3f} |")
