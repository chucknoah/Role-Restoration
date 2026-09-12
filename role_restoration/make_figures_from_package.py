#!/usr/bin/env python3
"""
Regenerate every figure and Table 1 from saved results. No model.

From a run of scripts/run_experiments.py:
    python scripts/make_figures.py --results-dir results --outdir figures

From the original artefacts (Colab pickle + RunPod csv + geom csv):
    python scripts/make_figures.py --legacy-colab step2b_results.pkl --legacy-runpod all_results.csv \
        --geom geom.csv --tokens token_projections.csv --figure1-dir role_mech/figs --outdir figures
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from role_restoration import plots as P, legacy as Lg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=None); p.add_argument("--outdir", default="figures")
    p.add_argument("--legacy-colab", default=None); p.add_argument("--legacy-runpod", default=None)
    p.add_argument("--geom", default=None); p.add_argument("--tokens", default=None); p.add_argument("--figure1-dir", default=None)
    p.add_argument("--on-frac", default=None, help="JSON {layer: fraction} if not in results dir")
    a = p.parse_args(); os.makedirs(a.outdir, exist_ok=True)

    if a.results_dir:                                                    # package run
        R = a.results_dir; res = pd.read_csv(os.path.join(R, "results.csv"))
        geom = pd.read_csv(os.path.join(R, "geom.csv")) if os.path.exists(os.path.join(R, "geom.csv")) else None
        tokens = pd.read_csv(os.path.join(R, "token_projections.csv")) if os.path.exists(os.path.join(R, "token_projections.csv")) else None
        on_frac = {int(k): v for k, v in json.load(open(os.path.join(R, "style_inside_frac.json"))).items()} if os.path.exists(os.path.join(R, "style_inside_frac.json")) else None
        fig1_dir = a.figure1_dir or os.path.join(R, "figure1")
        ABL, RES, LAY = "ablation", "restoration", "layers"
    else:                                                                # original artefacts
        parts = []
        if a.legacy_colab: parts.append(Lg.load_colab_pickle(a.legacy_colab))
        if a.legacy_runpod: parts.append(Lg.load_runpod_csv(a.legacy_runpod))
        res = pd.concat(parts, ignore_index=True)
        geom = pd.read_csv(a.geom) if a.geom else None; tokens = pd.read_csv(a.tokens) if a.tokens else None
        on_frac = {int(k): v for k, v in json.load(open(a.on_frac)).items()} if a.on_frac else {8: 0.114, 12: 0.067, 16: 0.180}
        fig1_dir = a.figure1_dir; ABL, RES, LAY = "ablation", "E0_specificity", "E1_layers"

    if geom is not None: P.figure2_geometry(geom, a.outdir, tokens)
    if on_frac and ABL in res.experiment.values: P.figure3_overlap_ablation(res, ABL, on_frac, a.outdir)
    if RES in res.experiment.values: P.figure4_slopes(res, RES, a.outdir)
    if LAY in res.experiment.values: P.figure5_layers(res, LAY, a.outdir)
    if fig1_dir:
        for f in sorted(glob.glob(os.path.join(fig1_dir, "*.json"))): P.figure1_text_and_trace(json.load(open(f)), a.outdir)
    rows = [(RES, "none", "None"), (ABL, "style", "Ablate All Style"), (ABL, "style_on", "Ablate Style\u2013Tag Overlap"), (ABL, "random", "Ablation Control"),
            (RES, "restore_random_subspace", "Role Restoration Control (small edit)"), (RES, "restore_random_MM", "Role Restoration Control (magnitude matched)"),
            (RES, "restore_role", "Role Restoration at Layers 8, 12, and 16"), (LAY, "restore_L8", "Role Restoration at Layer 8")]
    t = P.table1_asr(res, rows, a.outdir); print(t.to_string(index=False)); print("figures ->", a.outdir)

if __name__ == "__main__":
    main()
