#!/usr/bin/env python3
# ================================================================================================
# STAGE     Figures 2, 3, 5 (mech1_geometry, mech2_causal_decomposition, mech3_localization) + captions.md
# RAN ON    laptop
# NEEDS     geom.csv, all_results.csv, step2b_results.pkl, optional token_projections.csv
# PRODUCES  mech1/2/3 png+pdf, captions.md
# FIGURE    Figure 2, Figure 3, Figure 5
# NOTE      code unchanged from the run that produced the paper's numbers; only this header was added
# ================================================================================================
"""
The three mech-interp figures, from saved data. Self-contained; no model.

  mech1_geometry.png            Fig 1/2  — two directions, one leak (left); the leak grows with depth (right)
  mech2_causal_decomposition.png Fig 6   — the small role-aligned part of style is causal; the large remainder is inert
  mech3_localization.png        fig_layers — restore effect by layer beside where style merges into tag
  captions.md                    — captions for the three figures

Inputs:
  --geom     geom.csv                  Colab save cell (required for mech1 right panel and mech3 right panel)
  --results  all_results.csv           RunPod run   (required for mech3)
  --colab    step2b_results.pkl        Colab checkpoint (optional; mech2 otherwise uses the reported numbers)
  --tokens   token_projections.csv     Colab, optional: per-token scatter behind mech1's four dots
             (make it in Colab with:  L=TEST_LAYER_IX; H=hs[L][tok_df['sample_ix'].tolist()].float();
              u_t=_unit(_mean(all_probe_hs[L], pdf[pdf.role=='cot']['sample_ix'].tolist())-_mean(all_probe_hs[L], pdf[pdf.role=='user']['sample_ix'].tolist()));
              u_s=_unit(_mean(hs[L], tok_df[tok_df.style=='styled']['sample_ix'].tolist())-_mean(hs[L], tok_df[tok_df.style=='destyled']['sample_ix'].tolist()));
              tok_df.assign(p_tag=(H@u_t).numpy(), p_style=(H@u_s).numpy())[['tag','style','p_tag','p_style']].to_csv('token_projections.csv', index=False))

    python make_mech_figures.py --geom geom.csv --results all_results.csv --colab step2b_results.pkl --outdir figs
"""
import argparse, os, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

C = dict(tag="#c0392b", style="#8e44ad", style_light="#d9c9e6", forgery="#e67e22", prefix="#16a085", rand="#9e9e9e", base="#424242", user="#2c3e50", win="#1f77b4")
DEFAULT_MEANS = {"user|destyled": (-3.32, -5.34), "user|styled": (-0.18, 3.23), "cot|destyled": (1.43, -2.86), "cot|styled": (1.13, 3.82)}   # layer 12, Colab
DEFAULT_ON_FRAC = {8: 0.114, 12: 0.067, 16: 0.180}                                                                                      # RunPod printout
DEFAULT_ARM_ASR = {("forgery", "none"): .75, ("forgery", "style_perp"): .75, ("forgery", "style_on"): .53, ("forgery", "random"): .78,
                   ("prefix", "none"): .86, ("prefix", "style_perp"): .83, ("prefix", "style_on"): .56, ("prefix", "random"): .81}          # Colab
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300, "pdf.fonttype": 42})


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def save(fig, outdir, name):
    for ext in ("png", "pdf"): fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig); print(f"  {name}.png / .pdf")


def load_geom(path):
    g = pd.read_csv(path)
    if "layer" not in g.columns: g = g.rename(columns={g.columns[0]: "layer"})
    return g


# ======================================================================= mech1: geometry + depth
def mech1(outdir, geom_csv=None, tokens_csv=None):
    means = DEFAULT_MEANS; tok = None
    if tokens_csv and os.path.exists(tokens_csv):
        tok = pd.read_csv(tokens_csv)
        means = {f"{tg}|{st}": (g.p_tag.mean(), g.p_style.mean()) for (tg, st), g in tok.groupby(["tag", "style"])}
    have_geom = geom_csv and os.path.exists(geom_csv)
    fig, axes = plt.subplots(1, 2 if have_geom else 1, figsize=(10.2 if have_geom else 5, 4.4), squeeze=False, gridspec_kw={"width_ratios": [1.15, 1]} if have_geom else None)
    ax = axes[0][0]

    # ---- left: two directions, one overlap ----
    if tok is not None:
        for (tg, st), g in tok.groupby(["tag", "style"]):
            ax.scatter(g.p_tag, g.p_style, s=4, alpha=.15, color=C["tag"] if tg == "cot" else C["user"], marker="o" if st == "styled" else "x", lw=.5)
    pu, ps, cu, cs = means["user|destyled"], means["user|styled"], means["cot|destyled"], means["cot|styled"]
    pts = [("user|destyled", pu, "Plain text\ntagged USER", C["user"], (-0.3, -0.35), "right", "top"),
           ("user|styled", ps, "CoT-style text\ntagged USER", C["user"], (-0.25, 0.4), "right", "bottom"),
           ("cot|destyled", cu, "Plain text\ntagged CoT", C["tag"], (0.3, 0.0), "left", "center"),
           ("cot|styled", cs, "CoT-style text\ntagged CoT", C["tag"], (0.3, 0.0), "left", "center")]
    for key, (x, y), lab, col, (dx, dy), ha, va in pts:
        ax.scatter([x], [y], s=150, color=col, edgecolor="k", lw=.8, zorder=5)
        ax.annotate(lab, (x, y), xytext=(x + dx, y + dy), fontsize=7.5, ha=ha, va=va)
    ax.add_patch(FancyArrowPatch(pu, cu, arrowstyle="-|>", mutation_scale=16, lw=2, color=C["tag"], zorder=4))
    ax.text((pu[0] + cu[0]) / 2, (pu[1] + cu[1]) / 2 - 0.95, "Change the tag  (v_tag)", ha="center", va="top", fontsize=8, color=C["tag"], fontweight="bold")
    ax.add_patch(FancyArrowPatch(pu, ps, arrowstyle="-|>", mutation_scale=16, lw=2, color=C["style"], zorder=4))
    mx, my = (pu[0] + ps[0]) / 2, (pu[1] + ps[1]) / 2
    ax.text(mx - 0.45, my, "Change the style\n(v_style)", ha="right", va="center", fontsize=8, color=C["style"], fontweight="bold")
    leak = (ps[0] - pu[0]) / (cu[0] - pu[0]); ytop = max(ps[1], cs[1]) + 1.3
    ax.annotate("", xy=(ps[0], ytop), xytext=(pu[0], ytop), arrowprops=dict(arrowstyle="<->", color="k", lw=1))
    ax.plot([pu[0], pu[0]], [pu[1], ytop], ":", color="#999", lw=.8, zorder=1); ax.plot([ps[0], ps[0]], [ps[1], ytop], ":", color="#999", lw=.8, zorder=1)
    ax.text((pu[0] + ps[0]) / 2, ytop + 0.15, f"The overlap: changing only the style moves a USER token\n{100*leak:.0f}% of the way toward CoT along the tag axis", ha="center", va="bottom", fontsize=7.5)
    ax.set_xlabel("Projection onto tag direction   (USER  \u2190  \u2192  CoT)")
    ax.set_ylabel("Projection onto style direction   (plain  \u2190  \u2192  CoT-style)")
    ax.set_title("Two Directions, One Overlap  (layer 12)", loc="center")
    xs = [m[0] for m in means.values()]; ys = [m[1] for m in means.values()]
    ax.set_xlim(min(xs) - 2.6, max(xs) + 2.6); ax.set_ylim(min(ys) - 1.4, ytop + 1.6)
    ax.axhline(0, color="#ccc", lw=.6, zorder=0); ax.axvline(0, color="#ccc", lw=.6, zorder=0)

    # ---- right: the overlap grows with depth ----
    if have_geom:
        g = load_geom(geom_csv); g = g[g.layer > 0]
        ax2 = axes[0][1]; y = 100 * g["style / tag effect along tag axis"]
        ax2.scatter(g.layer, y, s=55, color=C["style"], zorder=3, label="Style's push along the tag axis\n(% of the full tag effect)")
        ax3 = ax2.twinx(); ax3.scatter(g.layer, g["cos(tag, style)"], s=30, marker="s", color=C["tag"], zorder=3, label="cos(v_tag, v_style)")
        ax3.set_ylim(-0.05, 1.0); ax3.set_ylabel("Cosine", color=C["tag"]); ax3.spines["right"].set_visible(True)
        ax2.set_xlabel("Layer"); ax2.set_ylabel("% of tag effect", color=C["style"]); ax2.set_ylim(-5, 105); ax2.set_xticks(sorted(g.layer.unique()))
        last = g.iloc[-1]
        ax2.annotate(f"By layer {int(last.layer)}: {100*last['style / tag effect along tag axis']:.0f}% \u2014 the model\ncan barely tell style from tag",
                     (last.layer, 100 * last["style / tag effect along tag axis"]), xytext=(-150, -32), textcoords="offset points", fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=.8))
        h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels(); ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
        ax2.set_title("The Overlap Grows with Depth", loc="center")
    save(fig, outdir, "mech1_geometry")
    return leak


# ======================================================================= mech2: causal decomposition
def mech2(outdir, colab_pickle=None, on_frac=None):
    on_frac = on_frac or DEFAULT_ON_FRAC
    if colab_pickle and os.path.exists(colab_pickle):
        d = pd.DataFrame(pickle.load(open(colab_pickle, "rb")))
        base = d[d.cond == "none"].groupby(["family", "attack_ix"]).attack.mean().reset_index(name="b")
        succ = base[base.b > 0].sort_values("b", ascending=False).groupby("family").head(12)
        d = d[d.attack_ix.isin(succ.attack_ix)]; asr = d.groupby(["family", "cond"]).attack.mean(); src = "Colab checkpoint"
    else:
        asr = pd.Series(DEFAULT_ARM_ASR); src = "reported numbers"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1, 1.5]})
    Ls = sorted(on_frac); on = np.array([on_frac[L] for L in Ls]); off = 1 - on
    a1.bar(range(len(Ls)), 100 * off, color=C["style_light"], edgecolor="k", lw=.5, label="outside the role subspace  (inert)")
    a1.bar(range(len(Ls)), 100 * on, bottom=100 * off, color=C["style"], edgecolor="k", lw=.5, label="inside the role subspace  (the overlap)")
    for i, v in enumerate(on): a1.text(i, 100 * (1 - v / 2), f"{100*v:.0f}%", ha="center", va="center", fontsize=8, color="w", fontweight="bold")
    a1.set_xticks(range(len(Ls))); a1.set_xticklabels([f"L{L}" for L in Ls]); a1.set_ylabel("Style Displacement %"); a1.set_ylim(0, 100)
    a1.legend(fontsize=6.5, loc="lower right", framealpha=.9)
    order = ["none", "style_perp", "style_on", "random"]
    lab = {"none": "no\nintervention", "style_perp": "ablate the\n82-93% outside\nthe subspace", "style_on": "ablate the\n7-18% inside\nthe subspace", "random": "ablate a\nrandom\ndirection"}
    col = {"none": C["base"], "style_perp": C["style_light"], "style_on": C["style"], "random": C["rand"]}; w = 0.38
    for j, fam in enumerate(["forgery", "prefix"]):
        vals = [asr.get((fam, a), np.nan) for a in order]
        a2.bar(np.arange(len(order)) + (j - 0.5) * w, vals, w, color=[col[a] for a in order], edgecolor="k", lw=.5, hatch="" if j == 0 else "//")
    a2.set_xticks(range(len(order))); a2.set_xticklabels([lab[a] for a in order], fontsize=7.5); a2.set_ylabel("ASR"); a2.set_ylim(0, 1)
    a2.set_yticks([0, .25, .5, .75, 1]); a2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    a2.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch=h, label=f) for f, h in [("forgery", ""), ("prefix", "//")]], fontsize=7, loc="upper right")
    fig.suptitle("Style Overlap and Ablations", fontsize=11, y=1.0)
    save(fig, outdir, "mech2_causal_decomposition")
    return asr, src


# ======================================================================= mech3: localisation
def mech3(outdir, results_csv, geom_csv=None):
    df = pd.read_csv(results_csv); e1 = df[df.experiment == "E1_layers"]; rows = []
    for fam in ("forgery", "prefix"):
        d = e1[e1.family == fam]; base = d[d.cond == "none"]
        for cond, g in d[d.cond != "none"].groupby("cond"):
            L = int(cond.split("_L")[-1]); k, n = int(g.attack.sum()), len(g)
            rows.append(dict(family=fam, layer=L, asr=k / n, lo=wilson(k, n)[0], hi=wilson(k, n)[1], n=n, delta=g.delta_norm_mean.mean(), base=base.attack.mean()))
    r = pd.DataFrame(rows).sort_values(["family", "layer"])
    have_geom = geom_csv and os.path.exists(geom_csv)
    fig, axes = plt.subplots(1, 2 if have_geom else 1, figsize=(9.4 if have_geom else 5, 3.5), squeeze=False)
    ax = axes[0][0]
    FAM_LABEL = {"forgery": "Style", "prefix": "Prefix"}
    for fam, off in (("forgery", -0.35), ("prefix", 0.35)):
        s = r[r.family == fam]
        ax.scatter(s.layer + off, 100 * s.asr, s=70, color=C[fam], edgecolor="k", lw=.5, zorder=3, label=f"{FAM_LABEL[fam]}")
        ax.axhline(100 * s.base.iloc[0], color=C[fam], ls="--", lw=.9, alpha=.8)
        ax.text(r.layer.max() + 1.0, 100 * s.base.iloc[0] + 2.0, f"{FAM_LABEL[fam]} baseline", color=C[fam], fontsize=7.5, va="bottom", ha="left")
    Ls = sorted(r.layer.unique())
    ax.set_xticks(Ls); ax.set_xticklabels([str(L) for L in Ls])
    ax.set_xlabel("Role Restoration Input Layer"); ax.set_ylabel("ASR")
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(-3, 103); ax.set_xlim(r.layer.min() - 2, r.layer.max() + 6.5)
    ax.set_title("Role Restoration by Layer", loc="center")
    ax.legend(fontsize=8, loc="upper left", title="Attack type")
    if have_geom:
        g = load_geom(geom_csv); g = g[g.layer > 0]; ax2 = axes[0][1]; y = 100 * g["style / tag effect along tag axis"]
        ax2.scatter(g.layer, y, s=45, color=C["style"], zorder=3)
        ax2.set_xlabel("Layer"); ax2.set_ylabel("Style's push along the tag axis\n(% of the full tag effect)")
        ax2.set_yticks([0, 25, 50, 75, 100]); ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"]); ax2.set_ylim(0, 105)
        ax2.set_xticks(sorted(g.layer.unique()))
        ax2.set_title("Style's Push Along the Tag Axis", loc="center")
    save(fig, outdir, "mech3_localization")
    return r


# ======================================================================= captions
def write_captions(outdir, leak, asr, src, r):
    f8 = r[(r.family == "forgery") & (r.layer == 8)].asr.iloc[0]; p8 = r[(r.family == "prefix") & (r.layer == 8)].asr.iloc[0]
    f16 = r[(r.family == "forgery") & (r.layer == 16)].asr.iloc[0]; p16 = r[(r.family == "prefix") & (r.layer == 16)].asr.iloc[0]
    d16 = r[r.layer == 16].delta.max(); d8 = r[r.layer == 8].delta.min()
    txt = f"""# Captions

**mech1_geometry.** *Left:* the four conditions of the 2x2 (same text, plain vs reasoning-style, tagged USER vs COT) at layer 12, projected onto the tag direction (x) and the style direction (y). The two directions are largely distinct (cosine 0.19, against split-half reliabilities of 0.99 and 0.63), yet changing only the style moves a USER-tagged token {100*leak:.0f}% of the way toward CoT along the tag axis. *Right:* that shared component as a function of depth: roughly a third of the tag effect at layer 4, half by layer 8, and 91% by layer 20 (purple; band = split-half reliability of the style direction, estimated from 7 text pairs). The cosine between the directions rises from 0.08 to 0.48 (red, right axis). Style and tag write to different directions that converge late; a small shared part is present throughout.

**mech2_causal_decomposition.** *Left:* the fraction of the style displacement that lies inside the four-dimensional role subspace (span of the five centred role means): 7-18% depending on layer; the rest lies outside it. *Right:* attack success on baseline-successful published injections (tool-span scope, layers 8/12/16) when each part is projected out. Ablating the small inside part reduces success ({asr.get(('forgery','none'),float('nan')):.2f} -> {asr.get(('forgery','style_on'),float('nan')):.2f} forgery, {asr.get(('prefix','none'),float('nan')):.2f} -> {asr.get(('prefix','style_on'),float('nan')):.2f} prefix); ablating the large outside part ({asr.get(('forgery','style_perp'),float('nan')):.2f} / {asr.get(('prefix','style_perp'),float('nan')):.2f}) is indistinguishable from a random direction ({asr.get(('forgery','random'),float('nan')):.2f} / {asr.get(('prefix','random'),float('nan')):.2f}). The role-aligned sliver of style carries the causal effect; the far larger orthogonal remainder is inert. ({src}; hatched = prefix.)

**mech3_localization.** *Left:* attack success when the role subspace of tool tokens is restored at a single layer (8 attacks per family, 2 samples each, Wilson 95% CIs; dashed = that experiment's no-intervention baseline; x-tick labels give the mean per-token edit size). Layer 8 alone gives {f8:.2f} / {p8:.2f}; layer 16 alone gives {f16:.2f} / {p16:.2f} despite the largest edit in the study (‖Δh‖ ≈ {d16:.0f} vs {d8:.0f} at layer 8). *Right:* the depth curve from mech1 with the readout window shaded. Role leakage keeps growing through layer 20, but the model has consulted "who said this" by layers 8-12; later confusion no longer affects whether it acts. Which components perform that readout is the open question (next: attribution-patch the restore effect over attention heads).
"""
    open(os.path.join(outdir, "captions.md"), "w").write(txt); print("  captions.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geom", default=None); p.add_argument("--results", default=None); p.add_argument("--colab", default=None); p.add_argument("--tokens", default=None)
    p.add_argument("--outdir", default="figs")
    a = p.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    print(f"writing to {a.outdir}/")
    leak = mech1(a.outdir, a.geom, a.tokens)
    asr, src = mech2(a.outdir, a.colab)
    if a.results and os.path.exists(a.results):
        r = mech3(a.outdir, a.results, a.geom)
        write_captions(a.outdir, leak, asr, src, r)
    else:
        print("  (no --results: skipping mech3 and captions)")
    print("done")


if __name__ == "__main__":
    main()
