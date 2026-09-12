#!/usr/bin/env python3
# ================================================================================================
# STAGE     Earlier two-panel layer figure (superseded by make_mech_figures mech3)
# RAN ON    laptop
# NEEDS     all_results.csv, geom.csv
# PRODUCES  fig_layers
# FIGURE    —
# NOTE      code unchanged from the run that produced the paper's numbers; only this header was added
# ================================================================================================
"""
Layer figure from saved data. No model.

Left panel  (from all_results.csv, experiment E1_layers + E0_specificity):
    attack success when the role subspace is restored at ONE layer only, per family, with Wilson CIs,
    each point annotated with the size of the edit. Baselines dashed. The 3-layer restore shown for reference.
Right panel (from geom.csv, Step 1):
    how far style pushes a token along the tag axis, as % of the full tag effect, per layer — where style merges into tag.

Reading: the intervention only works at layer 8 (readout window 8-12), yet the style/tag merge keeps growing to layer 20.
So the model consults "who said this" early, and later confusion is irrelevant to whether it acts.

    python make_layer_figure.py --results all_results.csv --geom geom.csv --outdir figs
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(forgery="#e67e22", prefix="#16a085", style="#8e44ad")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300, "pdf.fonttype": 42})


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--geom", default=None); ap.add_argument("--outdir", default="figs")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.results)

    # ---- single-layer restore, per family ----
    e1 = df[df.experiment == "E1_layers"]
    rows = []
    for fam in ("forgery", "prefix"):
        d = e1[e1.family == fam]
        base = d[d.cond == "none"]
        for cond, g in d[d.cond != "none"].groupby("cond"):
            L = int(cond.split("_L")[-1]); k, n = int(g.attack.sum()), len(g)
            rows.append(dict(family=fam, layer=L, asr=k / n, lo=wilson(k, n)[0], hi=wilson(k, n)[1], n=n,
                             delta=g.delta_norm_mean.mean(), base=base.attack.mean(), base_n=len(base)))
    r = pd.DataFrame(rows).sort_values(["family", "layer"])
    e0 = df[df.experiment == "E0_specificity"]
    full = {fam: e0[(e0.family == fam) & (e0.cond == "restore_role")].attack.mean() for fam in ("forgery", "prefix")}
    full_base = {fam: e0[(e0.family == fam) & (e0.cond == "none")].attack.mean() for fam in ("forgery", "prefix")}

    have_geom = a.geom and os.path.exists(a.geom)
    fig, axes = plt.subplots(1, 2 if have_geom else 1, figsize=(9.2 if have_geom else 5, 3.4), squeeze=False)
    ax = axes[0][0]
    for fam, off in (("forgery", -0.35), ("prefix", 0.35)):
        s = r[r.family == fam]
        ax.errorbar(s.layer + off, s.asr, yerr=[s.asr - s.lo, s.hi - s.asr], fmt="o", color=C[fam], ms=7, capsize=3, lw=1, label=f"{fam}: restore at this layer only")
        ax.axhline(s.base.iloc[0], color=C[fam], ls="--", lw=.9, alpha=.8)
        ax.text(r.layer.max() + 1.2, s.base.iloc[0], f"{fam} baseline", color=C[fam], fontsize=7, va="center")
    Ls = sorted(r.layer.unique())
    dl = r.groupby("layer").delta.agg(["min", "max"])
    ax.set_xticks(Ls); ax.set_xticklabels([f"{L}\n‖Δh‖ {dl.loc[L,'min']:.0f}–{dl.loc[L,'max']:.0f}" for L in Ls], fontsize=8)
    ax.set_xlabel("layer at which the role subspace is restored (single layer), and size of the edit")
    ax.set_ylabel("attack success rate"); ax.set_ylim(-0.03, 1.0); ax.set_xlim(r.layer.min() - 2, r.layer.max() + 6.5)
    ax.set_title("Where the role coordinates are read", loc="left", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.88))
    ax.text(0.98, 0.98, "layer 8 alone: 0/16 on both families\nlayer 16 alone: no effect, largest edit\n"
            f"3-layer restore, for reference: {full['forgery']:.2f} / {full['prefix']:.2f}", transform=ax.transAxes, fontsize=7, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="none"))

    if have_geom:
        g = pd.read_csv(a.geom); g = g.rename(columns={g.columns[0]: "layer"}) if "layer" not in g.columns else g; g = g[g.layer > 0]
        ax2 = axes[0][1]
        y = 100 * g["style / tag effect along tag axis"]
        ax2.scatter(g.layer, y, s=45, color=C["style"], zorder=3, label="style's push along the tag axis\n(% of the full tag effect)")
        if "cos(style_A, style_B)" in g:
            rel = g["cos(style_A, style_B)"].clip(0.05, 1); ax2.fill_between(g.layer, y / np.sqrt(rel), y * np.sqrt(rel), color=C["style"], alpha=.12, lw=0)
        ax2.axvspan(8, 12, color="#1f77b4", alpha=.10, lw=0); ax2.text(10, 8, "readout\nwindow", ha="center", fontsize=7, color="#1f77b4")
        ax2.set_xlabel("layer"); ax2.set_ylabel("% of tag effect"); ax2.set_ylim(0, 105); ax2.set_xticks(sorted(g.layer.unique()))
        ax2.set_title("Where style merges into tag", loc="left", fontsize=9.5); ax2.legend(fontsize=7, loc="upper left")
        fig.suptitle("The model decides who said it at layers 8-12; the style/tag merge keeps growing afterwards, and it no longer matters", fontsize=9.5, y=1.02)

    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.outdir, f"fig_layers.{ext}"), bbox_inches="tight")
    print(r[["family", "layer", "asr", "lo", "hi", "n", "delta", "base"]].round(2).to_string(index=False))
    print(f"\nsaved {a.outdir}/fig_layers.png/.pdf")


if __name__ == "__main__":
    main()
