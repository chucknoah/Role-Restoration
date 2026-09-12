#!/usr/bin/env python3
# ================================================================================================
# STAGE     Figure 1 renderer from textfig JSON: scatter trace, legend at right, model-action box
# RAN ON    laptop
# NEEDS     textfig_*.json
# PRODUCES  textfig_*_text.png, textfig_*_trace.png
# FIGURE    Figure 1
# NOTE      code unchanged from the run that produced the paper's numbers; only this header was added
# ================================================================================================
"""
Re-render the paper-style text + CoTness figure from saved data. No model, no GPU, instant.

Input: textfig_<family>_<attack>.json files written by role_cotness_text_figure.py (in <outdir>/figs/).
Output: textfig_<family>_<attack>_text.png/.pdf and _trace.png/.pdf next to them (or in --outdir).

    python render_textfig.py --figs-dir /path/to/role_mech/figs
    python render_textfig.py --json textfig_forgery_1.json --also-tool --chars-per-line 90

Points are drawn with scatter and are never connected.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

NICE = {"cot": "CoTness", "user": "Userness", "tool": "Toolness"}
PERCEIVED = {"cot": "#d35400", "user": "#1f77b4", "tool": "#7f8c8d", "assistant": "#27ae60", "system": "#95a5a6", "other": "#222222"}
POINT = "#333333"
TRUE_BAND = {"user task": "#dbeafe", "tool: page": "#ececec", "tool: injected": "#fde2df", "model CoT": "#fff1d6",
             "model tool call": "#dcf5e3", "model final": "#dcf5e3", "other": "#ffffff"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300, "pdf.fonttype": 42})


def draw_text_block(ax, toks, perceived, bands, chars_per_line, fontsize=7.4):
    ax.axis("off"); ax.set_xlim(0, chars_per_line); col, row = 0, 0
    for t, pr, bd in zip(toks, perceived, bands):
        s = t.replace("\n", "\u23ce")
        if col + len(s) > chars_per_line and col > 0:
            col, row = 0, row + 1
        ax.text(col, -row, s, fontsize=fontsize, family="monospace", color=PERCEIVED.get(pr, "k"), va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.02", fc=TRUE_BAND.get(bd, "#fff"), ec="none"))
        col += len(s)
        if "\n" in t:
            col, row = 0, row + 1
    ax.set_ylim(-(row + 1.2), 0.6)


def render(d, outdir, chars_per_line=96, also_tool=False, dpi=300, metric="auto"):
    """Writes TWO figures per attack:
         textfig_<family>_<id>_text.png/.pdf   : colored text + model action, both blocks
         textfig_<family>_<id>_trace.png/.pdf  : CoTness/Userness scatter, both blocks, legend at right
    """
    fam, aix, EL, PL, acc = d["family"], d["attack_ix"], d["edit_layer"], d["probe_layer"], d["probe_acc"]
    flab = d.get("family_label", {"forgery": "style-forgery", "prefix": "prefix"}.get(fam, fam))
    metric = ("cot" if fam == "forgery" else "user") if metric == "auto" else metric
    blocks = d["blocks"]; red, blue = "#c0392b", "#1f77b4"
    os.makedirs(outdir, exist_ok=True)

    def title_of(bi, b):
        return ("NO INTERVENTION" if bi == 0 else f"RESTORE @ layer {EL}") + ("  —  model exfiltrated" if b["attack"] else "  —  model did not act")

    # ---------------- figure 1: text + model action ----------------
    fig = plt.figure(figsize=(12.5, 9))
    gs = fig.add_gridspec(2, 1, hspace=0.35)
    for bi, b in enumerate(blocks):
        axT = fig.add_subplot(gs[bi, 0])
        draw_text_block(axT, b["toks"], b["perceived"], b["labels"], chars_per_line, fontsize=9.2)
        col = red if b["attack"] else blue
        axT.set_title(title_of(bi, b), loc="left", fontsize=10.5, color=col, pad=6)
        axT.text(0, axT.get_ylim()[0] - 0.2, "MODEL ACTION  →  " + b["action"], fontsize=8, family="monospace", va="top", ha="left",
                 color=col, clip_on=False, bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f5" if b["attack"] else "#f2f7fd", ec=col, lw=.8))
    h = [Line2D([], [], color=PERCEIVED[k], lw=4, label=f"text: perceived as {k}") for k in ("cot", "user", "tool", "assistant")]
    h += [Patch(fc=TRUE_BAND[k], ec="#999", label=f"band: actually {k}") for k in ("user task", "tool: page", "tool: injected", "model CoT")]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.99, 0.5), fontsize=7.5, borderaxespad=0)
    fig.suptitle(f"{flab} attack {aix}  ·  text colored by what the probe (layer {PL}, held-out acc {acc:.2f}) thinks each token is; bands show what it actually is",
                 fontsize=10, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"textfig_{flab}_{aix}_text.{ext}"), bbox_inches="tight", dpi=dpi if ext == "png" else None)
    plt.close(fig)

    # ---------------- figure 2: the trace ----------------
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=False, gridspec_kw={"hspace": 0.45})
    for bi, (ax, b) in enumerate(zip(axes, blocks)):
        P = {k: np.array([np.nan if v is None else v for v in vals], dtype=float) for k, vals in b["P"].items()}
        x = np.arange(len(b["toks"])); i0, i1 = b["inj"]; cut = b["cut"]; u = b.get("usr", 0); p0 = u + (1 if u else 0)
        if u: ax.axvspan(0, u, color=TRUE_BAND["user task"], lw=0, zorder=0)
        ax.axvspan(p0, i0, color=TRUE_BAND["tool: page"], lw=0, zorder=0)
        ax.axvspan(i0, i1, color=TRUE_BAND["tool: injected"], lw=0, zorder=0)
        ax.axvspan(cut, len(x), color=TRUE_BAND["model CoT"], lw=0, zorder=0)
        ax.scatter(x, 100 * P[metric], s=10, color=POINT, zorder=3)
        if also_tool:
            ax.scatter(x, 100 * P["tool"], s=5, color=PERCEIVED["tool"], alpha=.5, zorder=2)
        ax.axvline(cut - 0.5, color="k", lw=.6, ls=":")
        inj_mean = 100 * float(np.nanmean(P[metric][i0:i1]))
        ax.hlines(inj_mean, i0, i1, color=POINT, lw=1.2, ls="--", zorder=4)
        ax.text((i0 + i1) / 2, min(97, inj_mean + 6), f"injected text: mean {NICE[metric]} {inj_mean:.0f}%", ha="center", fontsize=7.5)
        ax.set_ylim(0, 102); ax.set_ylabel(f"{NICE[metric]} (%)")
        ticks = ([u / 2] if u else []) + [(p0 + i0) / 2, (i0 + i1) / 2, (cut + len(x)) / 2]
        ax.set_xticks(ticks); ax.set_xticklabels((["User task"] if u else []) + ["Tool (page)", "Tool (injected)", "Model's own CoT →"], fontsize=8)
        ax.set_title(title_of(bi, b), loc="left", fontsize=10, color=red if b["attack"] else blue)
        ax.text(0.01, 0.97, f"probe reads layer {PL}", transform=ax.transAxes, fontsize=7, va="top", color="#666")
    h = [Line2D([], [], color=POINT, ls="none", marker="o", ms=4, label=f"point: {NICE[metric]} of the token = P({metric} | token)")]
    if also_tool: h.append(Line2D([], [], color=PERCEIVED["tool"], ls="none", marker="o", ms=3, label="point: P(tool | token)"))
    h += [Patch(fc=TRUE_BAND[k], ec="#999", label=f"band: actually {k}") for k in ("user task", "tool: page", "tool: injected", "model CoT")]
    axes[0].legend(handles=h, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5, borderaxespad=0)
    axes[1].legend(handles=h, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5, borderaxespad=0)
    fig.suptitle(f"{flab} attack {aix}  ·  {NICE[metric]} of every token, before and after restoring the role subspace at layer {EL}", fontsize=10, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"textfig_{flab}_{aix}_trace.{ext}"), bbox_inches="tight", dpi=dpi if ext == "png" else None)
    plt.close(fig)
    print(f"  rendered textfig_{flab}_{aix}_text + _trace   injected-token {NICE[metric]}: " +
          "  ".join(f"{b['cond']}={100*np.nanmean([np.nan if v is None else v for v in b['P'][metric][b['inj'][0]:b['inj'][1]]]):.0f}%" for b in blocks))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figs-dir", default=None, help="directory containing textfig_*.json")
    p.add_argument("--json", nargs="*", default=None, help="explicit json files")
    p.add_argument("--outdir", default=None, help="where to write png/pdf (default: next to the json)")
    p.add_argument("--chars-per-line", type=int, default=96)
    p.add_argument("--also-tool", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--metric", default="auto", choices=["auto", "cot", "user", "tool"], help="auto = CoTness for style-forgery, Userness for prefix")
    a = p.parse_args()
    files = a.json or sorted(glob.glob(os.path.join(a.figs_dir or ".", "textfig_*.json")))
    if not files:
        raise SystemExit("no textfig_*.json found; run role_cotness_text_figure.py on the pod once to produce them")
    for f in files:
        d = json.load(open(f))
        render(d, a.outdir or os.path.dirname(f) or ".", a.chars_per_line, a.also_tool, a.dpi, a.metric)


if __name__ == "__main__":
    main()
