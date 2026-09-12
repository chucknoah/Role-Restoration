#!/usr/bin/env python3
# ================================================================================================
# STAGE     Headline figures incl. Figure 4 (fig3_slopes_before_after); also schematic, where-not-how-much, reads-not-obeys, small-part-big-effect
# RAN ON    laptop
# NEEDS     all_results.csv, geom.csv, optional step2b_results.pkl, token_projections.csv
# PRODUCES  fig0..fig6 png+pdf
# FIGURE    Figure 4
# NOTE      code unchanged from the run that produced the paper's numbers; only this header was added
# ================================================================================================
"""
Make the six figures from saved artefacts. No model, no GPU.

Inputs (all optional except --results):
  --results   all_results.csv            from the RunPod run            -> Figs 3, 4, 5
  --geom      geom.csv                   from the Colab save cell        -> Fig 2
  --tokens    token_projections.csv      per-token (tag, style, p_tag, p_style) at layer 12 -> Fig 1 scatter
              If absent, Fig 1 is drawn from the 2x2 cluster means (defaults below, override with --means).

To produce token_projections.csv in Colab (Step 1 state must be live), run:
    L = TEST_LAYER_IX; H = hs[L][tok_df['sample_ix'].tolist()].float()
    u_t = _unit(_mean(all_probe_hs[L], pdf[pdf.role=='cot']['sample_ix'].tolist()) - _mean(all_probe_hs[L], pdf[pdf.role=='user']['sample_ix'].tolist()))
    u_s = _unit(_mean(hs[L], tok_df[tok_df.style=='styled']['sample_ix'].tolist()) - _mean(hs[L], tok_df[tok_df.style=='destyled']['sample_ix'].tolist()))
    tok_df.assign(p_tag=(H@u_t).numpy(), p_style=(H@u_s).numpy())[['tag','style','p_tag','p_style']].to_csv('token_projections.csv', index=False)

Usage:
    python make_figures.py --results all_results.csv --geom geom.csv --outdir figs
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 2x2 cluster means at layer 12 from the Colab run (projection onto unit v_tag, unit v_style)
DEFAULT_MEANS = {"user|destyled": (-3.32, -5.34), "user|styled": (-0.18, 3.23),
                 "cot|destyled": (1.43, -2.86), "cot|styled": (1.13, 3.82)}

C = dict(restore="#1f77b4", rand="#9e9e9e", rand_dark="#616161", base="#424242", style="#8e44ad", tag="#c0392b",
         forgery="#e67e22", prefix="#16a085", stopped="#1f77b4", notstopped="#bdbdbd")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300, "pdf.fonttype": 42, "figure.dpi": 110})


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


# ---------------------------------------------------------------- Fig 0: schematic
def fig0_schematic(outdir):
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    W_ROLE, W_REST, X0 = 1.1, 6.6, 1.2

    def row(y, role_color, role_label, title):
        ax.add_patch(FancyBboxPatch((X0, y), W_ROLE, 0.55, boxstyle="round,pad=0.02", fc=role_color, ec="k", lw=.7))
        ax.add_patch(FancyBboxPatch((X0 + W_ROLE + 0.08, y), W_REST, 0.55, boxstyle="round,pad=0.02", fc="#eeeeee", ec="k", lw=.7))
        ax.text(X0 + W_ROLE / 2, y + 0.27, role_label, ha="center", va="center", fontsize=7.5, color="w", fontweight="bold")
        ax.text(X0 + W_ROLE + 0.08 + W_REST / 2, y + 0.27, "everything else: meaning, commands, code, tone  (untouched)",
                ha="center", va="center", fontsize=7.5)
        ax.text(X0 - 0.15, y + 0.27, title, ha="right", va="center", fontsize=8.5)

    row(2.2, C["tag"], "who said\nthis: USER", "forged tool text")
    row(0.9, "#7f8c8d", "who said\nthis: TOOL", "after restore")
    ax.add_patch(FancyArrowPatch((X0 + W_ROLE / 2, 2.15), (X0 + W_ROLE / 2, 1.5), arrowstyle="-|>", mutation_scale=14, lw=1.2, color="k"))
    ax.text(X0 + W_ROLE / 2 + 0.15, 1.83, "swap only this\n(4 of 2880 dims, layer 8)", fontsize=7.5, va="center")
    ax.text(X0, 2.95, "A token's activation, split into the part that encodes who is speaking and the rest", fontsize=8.5)
    ax.text(X0, 0.45, "The harness wrote the tag, so it knows the true speaker. Restore writes that truth back where the model keeps it.",
            fontsize=7.5, color="#333")
    save(fig, outdir, "fig0_restore_schematic")


# ---------------------------------------------------------------- Fig 1: geometry
def fig1_geometry(outdir, tokens_csv=None, means=None):
    means = means or DEFAULT_MEANS
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    tok = None
    if tokens_csv and os.path.exists(tokens_csv):
        tok = pd.read_csv(tokens_csv)
        for (tg, st), g in tok.groupby(["tag", "style"]):
            ax.scatter(g.p_tag, g.p_style, s=4, alpha=.18, color=C["tag"] if tg == "cot" else "#2c3e50", marker="o" if st == "styled" else "x", lw=.5)
        means = {f"{tg}|{st}": (g.p_tag.mean(), g.p_style.mean()) for (tg, st), g in tok.groupby(["tag", "style"])}

    pu, ps, cu, cs = means["user|destyled"], means["user|styled"], means["cot|destyled"], means["cot|styled"]
    pts = [("user|destyled", pu, "plain text\ntagged USER", "#2c3e50", (-0.3, -0.35), "right", "top"),
           ("user|styled", ps, "reasoning-style text\ntagged USER", "#2c3e50", (-0.25, 0.4), "right", "bottom"),
           ("cot|destyled", cu, "plain text\ntagged COT", C["tag"], (0.3, 0.0), "left", "center"),
           ("cot|styled", cs, "reasoning-style text\ntagged COT", C["tag"], (0.3, 0.0), "left", "center")]
    for key, (x, y), lab, col, (dx, dy), ha, va in pts:
        ax.scatter([x], [y], s=150, color=col, edgecolor="k", lw=.8, zorder=5)
        ax.annotate(lab, (x, y), xytext=(x + dx, y + dy), fontsize=7.5, ha=ha, va=va)

    ax.add_patch(FancyArrowPatch(pu, cu, arrowstyle="-|>", mutation_scale=16, lw=2, color=C["tag"], zorder=4))
    ax.text((pu[0] + cu[0]) / 2, (pu[1] + cu[1]) / 2 - 0.95, "change the TAG  (v_tag)", ha="center", va="top", fontsize=8, color=C["tag"], fontweight="bold")
    ax.add_patch(FancyArrowPatch(pu, ps, arrowstyle="-|>", mutation_scale=16, lw=2, color=C["style"], zorder=4))
    mx, my = (pu[0] + ps[0]) / 2, (pu[1] + ps[1]) / 2
    ax.text(mx - 0.45, my, "change the STYLE\n(v_style)", ha="right", va="center", fontsize=8, color=C["style"], fontweight="bold")

    leak = (ps[0] - pu[0]) / (cu[0] - pu[0])
    ytop = max(ps[1], cs[1]) + 1.3
    ax.annotate("", xy=(ps[0], ytop), xytext=(pu[0], ytop), arrowprops=dict(arrowstyle="<->", color="k", lw=1))
    ax.plot([pu[0], pu[0]], [pu[1], ytop], ":", color="#999", lw=.8, zorder=1)
    ax.plot([ps[0], ps[0]], [ps[1], ytop], ":", color="#999", lw=.8, zorder=1)
    ax.text((pu[0] + ps[0]) / 2, ytop + 0.15, f"the leak: changing only the style moves a USER token\n{100*leak:.0f}% of the way toward COT along the tag axis",
            ha="center", va="bottom", fontsize=7.5)
    ax.set_xlabel("projection onto tag direction   (USER  \u2190  \u2192  COT)")
    ax.set_ylabel("projection onto style direction   (plain  \u2190  \u2192  reasoning-style)")
    ax.set_title("Two directions, one leak  (layer 12)", loc="left")
    xs = [m[0] for m in means.values()]; ys = [m[1] for m in means.values()]
    ax.set_xlim(min(xs) - 2.6, max(xs) + 2.6); ax.set_ylim(min(ys) - 1.4, ytop + 1.6)
    ax.axhline(0, color="#ccc", lw=.6, zorder=0); ax.axvline(0, color="#ccc", lw=.6, zorder=0)
    save(fig, outdir, "fig1_geometry_leak")


# ---------------------------------------------------------------- Fig 2: leak by depth
def fig2_leak_by_depth(outdir, geom_csv):
    g = pd.read_csv(geom_csv)
    if "layer" not in g.columns:
        g = g.rename(columns={g.columns[0]: "layer"})
    g = g[g.layer > 0]                        # embedding layer: no computation yet, style/tag ratio is meaningless there
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    y = 100 * g["style / tag effect along tag axis"]
    ax.plot(g.layer, y, "o", color=C["style"], ms=6, label="style's push along the tag axis\n(% of the full tag effect)")
    if "cos(style_A, style_B)" in g:
        rel = g["cos(style_A, style_B)"].clip(0.05, 1)
        ax.fill_between(g.layer, y / np.sqrt(rel), y * np.sqrt(rel), color=C["style"], alpha=.12, lw=0, label="split-half reliability band")
    ax2 = ax.twinx()
    ax2.plot(g.layer, g["cos(tag, style)"], "s", color=C["tag"], ms=4.5, label="cos(v_tag, v_style)")
    ax2.set_ylim(-0.05, 1.0); ax2.set_ylabel("cosine", color=C["tag"]); ax2.spines["right"].set_visible(True)
    ax.set_xlabel("layer"); ax.set_ylabel("% of tag effect", color=C["style"]); ax.set_ylim(-5, 105)
    last = g.iloc[-1]
    ax.annotate(f"by layer {int(last.layer)}: {100*last['style / tag effect along tag axis']:.0f}% \u2014 the model\ncan barely tell style from tag",
                (last.layer, 100 * last["style / tag effect along tag axis"]), xytext=(-140, -30), textcoords="offset points", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=.8))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    ax.set_title("The leak grows with depth")
    save(fig, outdir, "fig2_leak_by_depth")


# ---------------------------------------------------------------- helpers for results
def per_attack(df, experiment):
    d = df[df.experiment == experiment]
    return d.groupby(["family", "attack_ix", "cond"]).attack.mean().unstack("cond")


# ---------------------------------------------------------------- Fig 3: slope chart
def fig3_slopes(outdir, df):
    pa = per_attack(df, "E0_specificity")
    fams = [f for f in ("forgery", "prefix") if f in pa.index.get_level_values(0)]
    FAM_LABEL = {"forgery": "Style", "prefix": "Prefix"}
    arms = [("restore_role", "Role Restoration"), ("restore_random_MM", "Role Restoration Control")]
    fig, axes = plt.subplots(len(fams), len(arms), figsize=(3.4 * len(arms), 2.7 * len(fams)), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    palette = plt.cm.tab10.colors + plt.cm.Set2.colors
    for i, fam in enumerate(fams):
        d = pa.loc[fam]
        for j, (arm, lab) in enumerate(arms):
            ax = axes[i][j]
            if arm not in d:
                ax.axis("off"); continue
            z = d[["none", arm]].dropna()
            for k, (_, r) in enumerate(z.iterrows()):
                jit = rng.uniform(-0.015, 0.015)
                ax.plot([0, 1], [100 * (r["none"] + jit), 100 * (r[arm] + jit)], "-o", color=palette[k % len(palette)], lw=1.4, ms=4, alpha=.9)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["No intervention", "After intervention"], fontsize=8)
            ax.set_xlim(-0.25, 1.25); ax.set_ylim(-3, 103)
            ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
            if j == 0:
                ax.set_ylabel(f"{FAM_LABEL[fam]} ASR")
            if i == 0:
                ax.set_title(lab, fontsize=9.5)
    save(fig, outdir, "fig3_slopes_before_after")


# ---------------------------------------------------------------- Fig 4: where, not how much
def fig4_where_not_how_much(outdir, df):
    rows = []
    for exp, arms in [("E0_specificity", ["restore_random_subspace", "restore_random_MM", "restore_role"]),
                      ("E1_layers", ["restore_L8", "restore_L12", "restore_L16"])]:
        d = df[df.experiment == exp]
        if d.empty:
            continue
        for fam in d.family.unique():
            base = d[(d.family == fam) & (d.cond == "none")].attack.mean()
            for a in arms:
                s = d[(d.family == fam) & (d.cond == a)]
                if len(s):
                    rows.append(dict(family=fam, arm=a, delta=s.delta_norm_mean.mean(), dASR=(s.attack.mean() / base) if base > 0 else np.nan))
    r = pd.DataFrame(rows)
    if r.empty:
        warnings.warn("no data for fig 4"); return
    LAB = {"restore_random_subspace": "small random\nsubspace", "restore_random_MM": "matched random\n(same size as restore)",
           "restore_role": "restore, layers 8+12+16", "restore_L8": "restore, layer 8 only", "restore_L12": "restore, layer 12 only",
           "restore_L16": "restore, layer 16 only"}
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for a, g in r.groupby("arm"):                                     # join the two family points of the same arm
        if len(g) == 2:
            ax.plot(g.delta, g.dASR, "-", color="#bbb", lw=.8, zorder=1)
    for fam, mk in [("forgery", "o"), ("prefix", "s")]:
        s = r[r.family == fam]
        if s.empty:
            continue
        col = [C["restore"] if ("role" in a or "L8" in a) else (C["rand_dark"] if "random" in a else "#7f8c8d") for a in s.arm]
        ax.scatter(s.delta, s.dASR, marker=mk, s=70, c=col, edgecolor="k", lw=.6, label=fam, zorder=3)
    OFF = {"restore_random_subspace": (8, 10), "restore_random_MM": (10, -6), "restore_role": (10, 8),
           "restore_L8": (-6, 10), "restore_L12": (8, 8), "restore_L16": (-8, -10)}
    for a, g in r.groupby("arm"):
        anchor = g.sort_values("dASR").iloc[0] if a in ("restore_role", "restore_L8", "restore_random_MM") else g.sort_values("dASR").iloc[-1]
        dx, dy = OFF.get(a, (8, 8))
        ax.annotate(LAB.get(a, a), (anchor.delta, anchor.dASR), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.85))
    ax.axhline(1, color="k", ls="--", lw=.8, zorder=0)
    ax.text(r.delta.max(), 1.02, "no change", ha="right", va="bottom", fontsize=7, color="#555")
    ax.axhline(0, color="#999", lw=.6, zorder=0)
    ax.set_xlabel("size of the edit   (mean per-token \u2016\u0394h\u2016)")
    ax.set_ylabel("attack success remaining\n(fraction of no-intervention rate)")
    ax.set_title("Where, not how much: the largest edit does nothing; a small one at layer 8 does everything", loc="left", fontsize=9)
    ax.set_ylim(-0.08, max(1.15, r.dASR.max() + 0.12))
    ax.legend(loc="upper right", title="attack family")
    save(fig, outdir, "fig4_where_not_how_much")


# ---------------------------------------------------------------- Fig 5: reads it, doesn't obey
def fig5_reads_not_obeys(outdir, df):
    d = df[(df.experiment == "E0_specificity") & (df.cond.isin(["none", "restore_role"]))]
    if d.empty:
        warnings.warn("no data for fig 5"); return
    s = d.groupby(["family", "cond"]).agg(mentions=("cot_mentions", "mean"), acts=("attack", "mean")).reset_index()
    fams = [f for f in ("forgery", "prefix") if f in s.family.values]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.4 * len(fams), 3.0), sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        z = s[s.family == fam].set_index("cond").reindex(["none", "restore_role"])
        x = np.arange(2); w = 0.36
        ax.bar(x - w / 2, z.mentions, w, color="#95a5a6", edgecolor="k", lw=.5, label="mentions the injected\ninstruction in its reasoning")
        ax.bar(x + w / 2, z.acts, w, color=C["tag"], edgecolor="k", lw=.5, label="acts on it")
        gap = z.mentions.iloc[1] - z.acts.iloc[1]
        ax.annotate("", xy=(1 + w / 2, z.acts.iloc[1]), xytext=(1 + w / 2, z.mentions.iloc[1]), arrowprops=dict(arrowstyle="<->", lw=1))
        ax.text(1 + w / 2 + 0.06, (z.mentions.iloc[1] + z.acts.iloc[1]) / 2, f"authority\nwithdrawn\n(\u2212{gap:.2f})", fontsize=7.5, va="center")
        ax.set_xticks(x); ax.set_xticklabels(["no intervention", "restore"]); ax.set_title(fam)
        ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("fraction of runs"); axes[0][0].legend(loc="upper left", fontsize=7)
    fig.suptitle("The model still reads the instruction; it no longer believes a user gave it", y=1.02, fontsize=9.5)
    save(fig, outdir, "fig5_reads_not_obeys")



# ---------------------------------------------------------------- Fig 6: small part, big effect
def fig6_small_part_big_effect(outdir, colab_pickle=None, on_frac=None):
    """Left: fraction of the style displacement inside the role subspace, per layer.
    Right: ASR when ablating that small on-subspace part vs the large off-subspace part vs random (Colab, tool-span, 3 layers)."""
    on_frac = on_frac or {8: 0.114, 12: 0.067, 16: 0.180}          # RunPod printout: on-role-subspace fraction of |v_style|^2
    arms = {"none": None, "style_perp": None, "style_on": None, "random": None}
    if colab_pickle and os.path.exists(colab_pickle):
        import pickle
        d = pd.DataFrame(pickle.load(open(colab_pickle, "rb")))
        base = d[d.cond == "none"].groupby(["family", "attack_ix"]).attack.mean().reset_index(name="b")
        succ = base[base.b > 0].sort_values("b", ascending=False).groupby("family").head(12)   # same top-12/family subset the arms ran on
        d = d[d.attack_ix.isin(succ.attack_ix)]
        asr = d.groupby(["family", "cond"]).attack.mean()
    else:                                                          # numbers from the Colab run reported in chat
        asr = pd.Series({("forgery", "none"): .75, ("forgery", "style_perp"): .75, ("forgery", "style_on"): .53, ("forgery", "random"): .78,
                         ("prefix", "none"): .86, ("prefix", "style_perp"): .83, ("prefix", "style_on"): .56, ("prefix", "random"): .81})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.1), gridspec_kw={"width_ratios": [1, 1.5]})
    Ls = sorted(on_frac); on = np.array([on_frac[L] for L in Ls]); off = 1 - on
    a1.bar(range(len(Ls)), 100 * off, color="#d9c9e6", edgecolor="k", lw=.5, label="off the role subspace (inert)")
    a1.bar(range(len(Ls)), 100 * on, bottom=100 * off, color=C["style"], edgecolor="k", lw=.5, label="inside the role subspace (the leak)")
    for i, v in enumerate(on): a1.text(i, 100 * (1 - v / 2), f"{100*v:.0f}%", ha="center", va="center", fontsize=8, color="w", fontweight="bold")
    a1.set_xticks(range(len(Ls))); a1.set_xticklabels([f"L{L}" for L in Ls]); a1.set_ylabel("% of the style displacement"); a1.set_ylim(0, 100)
    a1.legend(fontsize=6.5, loc="lower right", framealpha=.9); a1.set_title("How much of style is role?", loc="left", fontsize=9)
    order = ["none", "style_perp", "style_on", "random"]; lab = {"none": "no\nintervention", "style_perp": "ablate the\n82-93% OFF\nthe subspace", "style_on": "ablate the\n7-18% INSIDE\nthe subspace", "random": "ablate a\nrandom\ndirection"}
    col = {"none": C["base"], "style_perp": "#d9c9e6", "style_on": C["style"], "random": C["rand"]}
    w = 0.38
    for j, fam in enumerate(["forgery", "prefix"]):
        vals = [asr.get((fam, a), np.nan) for a in order]
        a2.bar(np.arange(len(order)) + (j - 0.5) * w, vals, w, color=[col[a] for a in order], edgecolor="k", lw=.5, hatch="" if j == 0 else "//", label=fam)
    a2.set_xticks(range(len(order))); a2.set_xticklabels([lab[a] for a in order], fontsize=7.5); a2.set_ylabel("attack success rate"); a2.set_ylim(0, 1)
    a2.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch=h, label=f) for f, h in [("forgery", ""), ("prefix", "//")]], fontsize=7, loc="upper right")
    a2.set_title("Remove the small part: attacks drop. Remove the large part: nothing.", loc="left", fontsize=9)
    fig.suptitle("Small part, big effect: the sliver of style that enters the role subspace is what carries authority", fontsize=9.5, y=1.04)
    save(fig, outdir, "fig6_small_part_big_effect")

# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="all_results.csv from the RunPod run")
    p.add_argument("--geom", default=None, help="geom.csv from the Colab save cell")
    p.add_argument("--tokens", default=None, help="token_projections.csv (optional, adds scatter to Fig 1)")
    p.add_argument("--means", default=None, help="JSON file overriding the 2x2 cluster means for Fig 1")
    p.add_argument("--outdir", default="figs")
    p.add_argument("--colab-pickle", default=None, help="step2b_results.pkl from Colab (optional; else uses reported numbers)")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.results)
    means = json.load(open(a.means)) if a.means else None
    if means:
        means = {k: tuple(v) for k, v in means.items()}
    print(f"writing to {a.outdir}/")
    fig0_schematic(a.outdir)
    fig1_geometry(a.outdir, a.tokens, means)
    if a.geom and os.path.exists(a.geom):
        fig2_leak_by_depth(a.outdir, a.geom)
    else:
        print("  (skipped fig2: pass --geom geom.csv)")
    fig3_slopes(a.outdir, df)
    fig4_where_not_how_much(a.outdir, df)
    fig5_reads_not_obeys(a.outdir, df)
    fig6_small_part_big_effect(a.outdir, a.colab_pickle)
    print("done")


if __name__ == "__main__":
    main()
