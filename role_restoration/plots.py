"""
plots.py — the figures and tables in the paper. No model needed; everything reads saved results.

  figure1_text_and_trace   Figure 1: the attack as colour-coded text + CoTness/Userness trace, before/after (from textfig JSON)
  figure2_geometry         Figure 2: Two Directions, One Overlap | The Overlap Grows with Depth   (geom.csv, token_projections.csv)
  figure3_overlap_ablation Figure 3: Style Overlap and Ablations                                  (results, geom)
  figure4_slopes           Figure 4: every attack before/after, Role Restoration | Role Restoration Control
  figure5_layers           Figure 5: Role Restoration by Layer
  table1_asr               Table 1: attack success rate by condition
Conventions: "CoT" (not COT), "overlap" (not leak), percentage axes, attack types "Style" and "Prefix".
"""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.lines import Line2D

C = dict(tag="#c0392b", style="#8e44ad", style_light="#d9c9e6", forgery="#e67e22", prefix="#16a085", rand="#9e9e9e", base="#424242", user="#2c3e50", point="#333333")
PERCEIVED = {"cot": "#d35400", "user": "#1f77b4", "tool": "#7f8c8d", "assistant": "#27ae60", "system": "#95a5a6", "other": "#222222"}
TRUE_BAND = {"user task": "#dbeafe", "tool: page": "#ececec", "tool: injected": "#fde2df", "model CoT": "#fff1d6", "model tool call": "#dcf5e3", "model final": "#dcf5e3", "other": "#ffffff"}
NICE = {"cot": "CoTness", "user": "Userness", "tool": "Toolness"}
FAM = {"forgery": "Style", "prefix": "Prefix"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})
PCT = ([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])

def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig); print(f"  {name}.png / .pdf")

# ============================================================================= Figure 1
def _draw_text_block(ax, toks, perceived, bands, chars_per_line=96, fontsize=9.2):
    ax.axis("off"); ax.set_xlim(0, chars_per_line); col, row = 0, 0
    for t, pr, bd in zip(toks, perceived, bands):
        s = t.replace("\n", "\u23ce")
        if col + len(s) > chars_per_line and col > 0: col, row = 0, row + 1
        ax.text(col, -row, s, fontsize=fontsize, family="monospace", color=PERCEIVED.get(pr, "k"), va="top", ha="left", bbox=dict(boxstyle="square,pad=0.02", fc=TRUE_BAND.get(bd, "#fff"), ec="none"))
        col += len(s)
        if "\n" in t: col, row = 0, row + 1
    ax.set_ylim(-(row + 1.2), 0.6)

def figure1_text_and_trace(data: dict, outdir: str, metric="auto", chars_per_line=96):
    """Two files per attack: figure1_<family>_<id>_text and _trace."""
    fam, aix, EL, PL = data["family"], data["attack_ix"], data["edit_layer"], data["probe_layer"]
    flab = data.get("family_label", fam); metric = ("cot" if fam == "forgery" else "user") if metric == "auto" else metric
    blocks = data["blocks"]; red, blue = "#c0392b", "#1f77b4"
    title = lambda bi, b: ("No intervention" if bi == 0 else f"Role restoration @ layer {EL}") + ("  \u2014  model exfiltrated" if b["attack"] else "  \u2014  model did not act")
    # text
    fig = plt.figure(figsize=(12.5, 9)); gs = fig.add_gridspec(2, 1, hspace=0.35)
    for bi, b in enumerate(blocks):
        ax = fig.add_subplot(gs[bi, 0]); _draw_text_block(ax, b["toks"], b["perceived"], b["labels"], chars_per_line)
        col = red if b["attack"] else blue; ax.set_title(title(bi, b), loc="left", fontsize=10.5, color=col, pad=6)
        ax.text(0, ax.get_ylim()[0] - 0.2, "MODEL ACTION  \u2192  " + b["action"], fontsize=8, family="monospace", va="top", ha="left", color=col, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f5" if b["attack"] else "#f2f7fd", ec=col, lw=.8))
    h = [Line2D([], [], color=PERCEIVED[k], lw=4, label=f"text: perceived as {k}") for k in ("cot", "user", "tool", "assistant")]
    h += [Patch(fc=TRUE_BAND[k], ec="#999", label=f"band: actually {k}") for k in ("user task", "tool: page", "tool: injected", "model CoT")]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.99, 0.5), fontsize=7.5, borderaxespad=0)
    fig.suptitle(f"{flab} attack {aix}  \u00b7  text colour = what the probe (layer {PL}) thinks each token is; band = what it actually is", fontsize=10, y=0.995)
    _save(fig, outdir, f"figure1_{flab}_{aix}_text")
    # trace
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), gridspec_kw={"hspace": 0.45})
    for bi, (ax, b) in enumerate(zip(axes, blocks)):
        P = {k: np.array([np.nan if v is None else v for v in vals], dtype=float) for k, vals in b["P"].items()}
        x = np.arange(len(b["toks"])); i0, i1 = b["inj"]; cut = b["cut"]; u = b.get("usr", 0); p0 = u + (1 if u else 0)
        if u: ax.axvspan(0, u, color=TRUE_BAND["user task"], lw=0, zorder=0)
        ax.axvspan(p0, i0, color=TRUE_BAND["tool: page"], lw=0, zorder=0); ax.axvspan(i0, i1, color=TRUE_BAND["tool: injected"], lw=0, zorder=0); ax.axvspan(cut, len(x), color=TRUE_BAND["model CoT"], lw=0, zorder=0)
        ax.scatter(x, 100 * P[metric], s=10, color=C["point"], zorder=3); ax.axvline(cut - 0.5, color="k", lw=.6, ls=":")
        m = 100 * float(np.nanmean(P[metric][i0:i1])); ax.hlines(m, i0, i1, color=C["point"], lw=1.2, ls="--", zorder=4)
        ax.text((i0 + i1) / 2, min(97, m + 6), f"injected text: mean {NICE[metric]} {m:.0f}%", ha="center", fontsize=7.5)
        ax.set_ylim(0, 102); ax.set_yticks(*PCT); ax.set_ylabel(f"{NICE[metric]} (%)")
        ax.set_xticks(([u / 2] if u else []) + [(p0 + i0) / 2, (i0 + i1) / 2, (cut + len(x)) / 2]); ax.set_xticklabels((["User task"] if u else []) + ["Tool (page)", "Tool (injected)", "Model's own CoT \u2192"], fontsize=8)
        ax.set_title(title(bi, b), loc="left", fontsize=10, color=red if b["attack"] else blue)
    h = [Line2D([], [], color=C["point"], ls="none", marker="o", ms=4, label=f"point: {NICE[metric]} of the token")] + [Patch(fc=TRUE_BAND[k], ec="#999", label=f"band: actually {k}") for k in ("user task", "tool: page", "tool: injected", "model CoT")]
    for ax in axes: ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5, borderaxespad=0)
    fig.suptitle(f"{flab} attack {aix}  \u00b7  {NICE[metric]} of every token, before and after restoring the role subspace at layer {EL}", fontsize=10, y=1.0)
    _save(fig, outdir, f"figure1_{flab}_{aix}_trace")

# ============================================================================= Figure 2
def figure2_geometry(geom: pd.DataFrame, outdir: str, tokens: pd.DataFrame | None = None, proj_layer=12):
    g = geom[geom.layer > 0]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    if tokens is not None:
        for (tg, st), gg in tokens.groupby(["tag", "style"]):
            ax.scatter(gg.p_tag, gg.p_style, s=4, alpha=.15, color=C["tag"] if tg == "cot" else C["user"], marker="o" if st == "styled" else "x", lw=.5)
        means = {f"{tg}|{st}": (gg.p_tag.mean(), gg.p_style.mean()) for (tg, st), gg in tokens.groupby(["tag", "style"])}
    else:
        means = {"user|destyled": (-3.32, -5.34), "user|styled": (-0.18, 3.23), "cot|destyled": (1.43, -2.86), "cot|styled": (1.13, 3.82)}
    pu, ps, cu, cs = means["user|destyled"], means["user|styled"], means["cot|destyled"], means["cot|styled"]
    for key, (x, y), lab, col, (dx, dy), ha, va in [("user|destyled", pu, "Plain text\ntagged USER", C["user"], (-0.3, -0.35), "right", "top"),
                                                  ("user|styled", ps, "CoT-style text\ntagged USER", C["user"], (-0.25, 0.4), "right", "bottom"),
                                                  ("cot|destyled", cu, "Plain text\ntagged CoT", C["tag"], (0.3, 0.0), "left", "center"),
                                                  ("cot|styled", cs, "CoT-style text\ntagged CoT", C["tag"], (0.3, 0.0), "left", "center")]:
        ax.scatter([x], [y], s=150, color=col, edgecolor="k", lw=.8, zorder=5); ax.annotate(lab, (x, y), xytext=(x + dx, y + dy), fontsize=7.5, ha=ha, va=va)
    ax.add_patch(FancyArrowPatch(pu, cu, arrowstyle="-|>", mutation_scale=16, lw=2, color=C["tag"], zorder=4))
    ax.text((pu[0] + cu[0]) / 2, (pu[1] + cu[1]) / 2 - 0.95, "Change the tag  (v_tag)", ha="center", va="top", fontsize=8, color=C["tag"], fontweight="bold")
    ax.add_patch(FancyArrowPatch(pu, ps, arrowstyle="-|>", mutation_scale=16, lw=2, color=C["style"], zorder=4))
    ax.text((pu[0] + ps[0]) / 2 - 0.45, (pu[1] + ps[1]) / 2, "Change the style\n(v_style)", ha="right", va="center", fontsize=8, color=C["style"], fontweight="bold")
    leak = (ps[0] - pu[0]) / (cu[0] - pu[0]); ytop = max(ps[1], cs[1]) + 1.3
    ax.annotate("", xy=(ps[0], ytop), xytext=(pu[0], ytop), arrowprops=dict(arrowstyle="<->", color="k", lw=1))
    ax.plot([pu[0], pu[0]], [pu[1], ytop], ":", color="#999", lw=.8, zorder=1); ax.plot([ps[0], ps[0]], [ps[1], ytop], ":", color="#999", lw=.8, zorder=1)
    ax.text((pu[0] + ps[0]) / 2, ytop + 0.15, f"The overlap: changing only the style moves a USER token\n{100*leak:.0f}% of the way toward CoT along the tag axis", ha="center", va="bottom", fontsize=7.5)
    ax.set_xlabel("Projection onto tag direction   (USER  \u2190  \u2192  CoT)"); ax.set_ylabel("Projection onto style direction   (plain  \u2190  \u2192  CoT-style)")
    ax.set_title(f"Two Directions, One Overlap  (layer {proj_layer})", loc="center")
    xs = [m[0] for m in means.values()]; ys = [m[1] for m in means.values()]; ax.set_xlim(min(xs) - 2.6, max(xs) + 2.6); ax.set_ylim(min(ys) - 1.4, ytop + 1.6)
    ax.axhline(0, color="#ccc", lw=.6, zorder=0); ax.axvline(0, color="#ccc", lw=.6, zorder=0)
    y = 100 * g["style / tag effect along tag axis"]
    ax2.scatter(g.layer, y, s=55, color=C["style"], zorder=3, label="Style's push along the tag axis\n(% of the full tag effect)")
    ax3 = ax2.twinx(); ax3.scatter(g.layer, g["cos(tag, style)"], s=30, marker="s", color=C["tag"], zorder=3, label="cos(v_tag, v_style)")
    ax3.set_ylim(-0.05, 1.0); ax3.set_ylabel("Cosine", color=C["tag"]); ax3.spines["right"].set_visible(True)
    ax2.set_xlabel("Layer"); ax2.set_ylabel("% of tag effect", color=C["style"]); ax2.set_ylim(-5, 105); ax2.set_xticks(sorted(g.layer.unique()))
    last = g.iloc[-1]
    ax2.annotate(f"By layer {int(last.layer)}: {100*last['style / tag effect along tag axis']:.0f}% \u2014 the model\ncan barely tell style from tag", (last.layer, 100 * last["style / tag effect along tag axis"]), xytext=(-150, -32), textcoords="offset points", fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=.8))
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels(); ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    ax2.set_title("The Overlap Grows with Depth", loc="center")
    _save(fig, outdir, "figure2_geometry")

# ============================================================================= Figure 3
def figure3_overlap_ablation(results: pd.DataFrame, experiment: str, on_frac: dict, outdir: str):
    """results rows must have cond in {none, style, style_perp, style_on, random} for `experiment`; on_frac: {layer: fraction of |v_style|^2 inside Q}."""
    d = results[results.experiment == experiment]; asr = d.groupby(["family", "cond"]).attack.mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1, 1.5]})
    Ls = sorted(on_frac); on = np.array([on_frac[L] for L in Ls]); off = 1 - on
    a1.bar(range(len(Ls)), 100 * off, color=C["style_light"], edgecolor="k", lw=.5, label="outside the role subspace  (inert)")
    a1.bar(range(len(Ls)), 100 * on, bottom=100 * off, color=C["style"], edgecolor="k", lw=.5, label="inside the role subspace  (the overlap)")
    for i, v in enumerate(on): a1.text(i, 100 * (1 - v / 2), f"{100*v:.0f}%", ha="center", va="center", fontsize=8, color="w", fontweight="bold")
    a1.set_xticks(range(len(Ls))); a1.set_xticklabels([f"L{L}" for L in Ls]); a1.set_ylabel("Style Displacement %"); a1.set_ylim(0, 100); a1.legend(fontsize=6.5, loc="lower right", framealpha=.9)
    order = ["none", "style_perp", "style_on", "random"]
    lab = {"none": "no\nintervention", "style_perp": f"ablate the\n{100*(1-max(on)):.0f}-{100*(1-min(on)):.0f}% outside\nthe subspace", "style_on": f"ablate the\n{100*min(on):.0f}-{100*max(on):.0f}% inside\nthe subspace", "random": "ablate a\nrandom\ndirection"}
    col = {"none": C["base"], "style_perp": C["style_light"], "style_on": C["style"], "random": C["rand"]}; w = 0.38
    for j, fam in enumerate(["forgery", "prefix"]):
        a2.bar(np.arange(len(order)) + (j - 0.5) * w, [asr.get((fam, a), np.nan) for a in order], w, color=[col[a] for a in order], edgecolor="k", lw=.5, hatch="" if j == 0 else "//")
    a2.set_xticks(range(len(order))); a2.set_xticklabels([lab[a] for a in order], fontsize=7.5); a2.set_ylabel("ASR"); a2.set_ylim(0, 1)
    a2.set_yticks([0, .25, .5, .75, 1]); a2.set_yticklabels(PCT[1])
    a2.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch=h, label=FAM[f]) for f, h in [("forgery", ""), ("prefix", "//")]], fontsize=7, loc="upper right")
    fig.suptitle("Style Overlap and Ablations", fontsize=11, y=1.0)
    _save(fig, outdir, "figure3_overlap_ablation")

# ============================================================================= Figure 4
def figure4_slopes(results: pd.DataFrame, experiment: str, outdir: str, arms=(("restore_role", "Role Restoration"), ("restore_random_MM", "Role Restoration Control"))):
    d = results[results.experiment == experiment]; pa = d.groupby(["family", "attack_ix", "cond"]).attack.mean().unstack("cond")
    fams = [f for f in ("forgery", "prefix") if f in pa.index.get_level_values(0)]
    fig, axes = plt.subplots(len(fams), len(arms), figsize=(3.4 * len(arms), 2.7 * len(fams)), sharey=True, squeeze=False)
    rng = np.random.default_rng(0); palette = plt.cm.tab10.colors + plt.cm.Set2.colors
    for i, fam in enumerate(fams):
        dd = pa.loc[fam]
        for j, (arm, lab) in enumerate(arms):
            ax = axes[i][j]
            if arm not in dd: ax.axis("off"); continue
            z = dd[["none", arm]].dropna()
            for k, (_, r) in enumerate(z.iterrows()):
                jit = rng.uniform(-0.015, 0.015); ax.plot([0, 1], [100 * (r["none"] + jit), 100 * (r[arm] + jit)], "-o", color=palette[k % len(palette)], lw=1.4, ms=4, alpha=.9)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["No intervention", "After intervention"], fontsize=8); ax.set_xlim(-0.25, 1.25); ax.set_ylim(-3, 103); ax.set_yticks(*PCT)
            if j == 0: ax.set_ylabel(f"{FAM[fam]} ASR")
            if i == 0: ax.set_title(lab, fontsize=9.5)
    _save(fig, outdir, "figure4_slopes")

# ============================================================================= Figure 5
def figure5_layers(results: pd.DataFrame, experiment: str, outdir: str):
    e1 = results[results.experiment == experiment]; rows = []
    for fam in ("forgery", "prefix"):
        d = e1[e1.family == fam]; base = d[d.cond == "none"].attack.mean()
        for cond, g in d[d.cond != "none"].groupby("cond"):
            if "_L" not in cond: continue
            rows.append(dict(family=fam, layer=int(cond.split("_L")[-1]), asr=g.attack.mean(), base=base))
    r = pd.DataFrame(rows).sort_values(["family", "layer"])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for fam, off in (("forgery", -0.35), ("prefix", 0.35)):
        s = r[r.family == fam]
        ax.scatter(s.layer + off, 100 * s.asr, s=70, color=C[fam], edgecolor="k", lw=.5, zorder=3, label=FAM[fam])
        ax.axhline(100 * s.base.iloc[0], color=C[fam], ls="--", lw=.9, alpha=.8)
        ax.text(r.layer.max() + 1.0, 100 * s.base.iloc[0] + 2.0, f"{FAM[fam]} baseline", color=C[fam], fontsize=7.5, va="bottom", ha="left")
    Ls = sorted(r.layer.unique()); ax.set_xticks(Ls); ax.set_xticklabels([str(L) for L in Ls])
    ax.set_xlabel("Role Restoration Input Layer"); ax.set_ylabel("ASR"); ax.set_yticks(*PCT); ax.set_ylim(-3, 103); ax.set_xlim(r.layer.min() - 2, r.layer.max() + 6.5)
    ax.set_title("Role Restoration by Layer", loc="center"); ax.legend(fontsize=8, loc="upper left", title="Attack type")
    _save(fig, outdir, "figure5_layers")

# ============================================================================= Table 1
def table1_asr(results: pd.DataFrame, rows: list, outdir: str) -> pd.DataFrame:
    """rows: list of (experiment, cond, label). Cells are ASR with (k/n)."""
    out = []
    for exp, cond, label in rows:
        d = results[(results.experiment == exp) & (results.cond == cond)]
        if d.empty: continue
        cell = lambda x: f"{100*x.attack.mean():.0f}% ({int(x.attack.sum())}/{len(x)})" if len(x) else "\u2014"
        out.append({"Intervention Type": label, "Prefix ASR": cell(d[d.family == "prefix"]), "Style ASR": cell(d[d.family == "forgery"]), "Total ASR": cell(d)})
    t = pd.DataFrame(out); os.makedirs(outdir, exist_ok=True)
    t.to_csv(os.path.join(outdir, "table1_asr.csv"), index=False)
    try: open(os.path.join(outdir, "table1_asr.md"), "w").write(t.to_markdown(index=False))
    except Exception: pass
    return t
