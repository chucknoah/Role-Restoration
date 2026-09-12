#!/usr/bin/env python3
# ================================================================================================
# STAGE     Driver: make_figures + make_layer_figure + render_textfig; writes index.md
# RAN ON    laptop
# NEEDS     the inputs of those three
# PRODUCES  figs/ + index.md
# FIGURE    all
# NOTE      code unchanged from the run that produced the paper's numbers; only this header was added
# ================================================================================================
"""
Regenerate every figure from saved data. No model, no GPU.

Needs, in the same directory as this file:
    make_figures.py        make_layer_figure.py        render_textfig.py

Inputs (pass what you have; missing ones are skipped with a note):
    --results   all_results.csv            RunPod run                    -> fig3, fig4, fig5, fig_layers
    --geom      geom.csv                   Colab save cell               -> fig1 (means fallback), fig2, fig_layers right panel
    --colab     step2b_results.pkl         Colab checkpoint              -> fig6 (else reported numbers are used)
    --tokens    token_projections.csv      Colab, optional               -> fig1 token scatter
    --textfig   directory with textfig_*.json  from role_cotness_text_figure.py -> text + trace figures

    python regenerate_all_figures.py --results all_results.csv --geom geom.csv --colab step2b_results.pkl \
        --textfig role_mech/figs --outdir figs
"""
import argparse, os, subprocess, sys, glob, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))

INDEX = {
    "fig0_restore_schematic": "What restore does: swap 4 of 2,880 dims (who said this) at layer 8, leave the rest.",
    "fig1_geometry_leak": "Tag and style are different directions (cos 0.19); style alone moves a USER token ~66% of the way to COT on the tag axis.",
    "fig2_leak_by_depth": "The shared component grows with depth: ~30% of the tag effect early, 91% by layer 20.",
    "fig3_slopes_before_after": "Every baseline-successful attack, before vs restore (7/8 per family fully stopped) and before vs matched-random (scatter).",
    "fig4_where_not_how_much": "Layer 8 with a small edit -> 0; layer 16 with the largest edit -> no change. Where, not how much.",
    "fig5_reads_not_obeys": "Under restore the model still mentions the injected instruction (~0.65) but acts on it ~0.04.",
    "fig6_small_part_big_effect": "Ablate the 7-18% of style inside the role subspace: attacks drop. Ablate the other 82-93%: nothing.",
    "fig_layers": "Single-layer restore vs layer (readout window 8-12) beside where style merges into tag.",
    "textfig_*_text": "The attack as colored text (perceived role) over true-provenance bands, with the model's action, before/after.",
    "textfig_*_trace": "CoTness/Userness of every token before/after, the paper's own plot type.",
}


def run(cmd, label):
    print(f"\n== {label}\n   {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"   !! {label} failed (exit {r.returncode}); continuing")
    return r.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None)
    p.add_argument("--geom", default=None)
    p.add_argument("--colab", default=None, help="step2b_results.pkl")
    p.add_argument("--tokens", default=None, help="token_projections.csv")
    p.add_argument("--textfig", default=None, help="directory containing textfig_*.json")
    p.add_argument("--outdir", default="figs")
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    py = sys.executable
    made = []

    for f in ("make_figures.py", "make_layer_figure.py", "render_textfig.py"):
        if not os.path.exists(os.path.join(HERE, f)):
            sys.exit(f"missing {f} next to this script")

    if a.results:
        cmd = [py, "make_figures.py", "--results", a.results, "--outdir", a.outdir]
        if a.geom: cmd += ["--geom", a.geom]
        if a.colab: cmd += ["--colab-pickle", a.colab]
        if a.tokens: cmd += ["--tokens", a.tokens]
        if run(cmd, "headline figures (fig0-fig6)"):
            made += [k for k in INDEX if k.startswith("fig") and k != "fig_layers"]
        cmd = [py, "make_layer_figure.py", "--results", a.results, "--outdir", a.outdir]
        if a.geom: cmd += ["--geom", a.geom]
        if run(cmd, "layer figure"):
            made.append("fig_layers")
    else:
        print("\n(no --results: skipping fig3/4/5/6 and fig_layers)")

    if a.textfig:
        js = sorted(glob.glob(os.path.join(a.textfig, "textfig_*.json")))
        if js:
            if run([py, "render_textfig.py", "--json", *js, "--outdir", a.outdir, "--dpi", str(a.dpi)], f"text + trace figures ({len(js)} attacks)"):
                made += ["textfig_*_text", "textfig_*_trace"]
        else:
            print(f"\n(no textfig_*.json in {a.textfig}; run role_cotness_text_figure.py on the pod first)")
    else:
        print("\n(no --textfig: skipping text/trace figures)")

    # index
    lines = ["# Figure index", ""]
    for k, v in INDEX.items():
        status = "made" if k in made else "skipped"
        files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.outdir, k.replace("*", "*") + ".png")))
        lines.append(f"- **{k}** ({status}): {v}" + (f"  \n  files: {', '.join(files)}" if files else ""))
    lines += ["", "Suggested mech-interp set: fig1 -> fig6 -> fig_layers (distinct directions -> the shared part is causal -> where it is read).",
              "Suggested defense set: fig0 -> fig3 -> fig4 -> fig5 (what it does -> it works -> not brute force -> reads but doesn't obey).",
              "Comparison to the paper's own plot type: textfig_*_trace."]
    with open(os.path.join(a.outdir, "index.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {a.outdir}/index.md\n")
    print(textwrap.indent("\n".join(lines[2:]), "  "))


if __name__ == "__main__":
    main()
