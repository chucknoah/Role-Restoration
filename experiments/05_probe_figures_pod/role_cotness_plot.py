#!/usr/bin/env python3
# ================================================================================================
# STAGE     6. Ye et al.'s probe + CoTness/Userness per token, before/after restore
# RAN ON    RunPod (imports 03, 04)
# NEEDS     followup_results.jsonl
# PRODUCES  cotness_*.png, cotness_summary.csv
# FIGURE    appendix
# NOTE      code unchanged from the run that produced the paper's numbers; only this header was added
# ================================================================================================
"""
CoTness plot: does the injected text stop "looking like CoT" after restoration?

1. Picks, from followup_results.jsonl (E0_specificity), attacks that succeeded at baseline and were fully stopped by restore.
2. Trains the paper's role probe (multinomial logistic regression on per-token activations, content tokens only,
   identical text under each of the five tags) at the probe layer.
3. For each chosen attack runs one forward pass under none / matched-random@L8 / restore@L8, reads the probe at the
   probe layer (downstream of the edit), and plots P(cot), P(user), P(tool) at every token of the tool span with the
   injected region shaded. Also writes the baseline-vs-restored transcripts side by side.

Needs role_mech_followups_v2.py and role_mech_plots.py in the same directory.  pip install scikit-learn

    python role_cotness_plot.py --outdir /workspace/role_mech --n-per-family 2
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import role_mech_followups_v2 as V
from role_mech_plots import Intervene3, Spec3, capture_under

ROLES = V.ROLES
COL = {"cot": "#c0392b", "user": "#2c3e50", "tool": "#7f8c8d", "assistant": "#8e44ad", "system": "#bdc3c7"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300, "pdf.fonttype": 42})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="openai/gpt-oss-20b")
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/workspace/hf"))
    p.add_argument("--outdir", default="/workspace/role_mech")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--edit-layer", type=int, default=8)
    p.add_argument("--probe-layer", type=int, default=12)
    p.add_argument("--n-per-family", type=int, default=2)
    p.add_argument("--n-carriers", type=int, default=3)
    p.add_argument("--n-benign-ref", type=int, default=8)
    p.add_argument("--carrier-chars", type=int, default=6000)
    p.add_argument("--n-prefix-user", type=int, default=8)
    p.add_argument("--n-prefix-other", type=int, default=4)
    p.add_argument("--role-texts", type=int, default=40, help="more texts than the followups: the probe needs tokens")
    p.add_argument("--role-max-tokens", type=int, default=160)
    p.add_argument("--role-batch-size", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("--attack-ids", default=None, help="override selection, comma-separated attack_ix")
    return p.parse_args()


# ---------------- pick attacks: succeeded at baseline, stopped by restore ----------------
def pick_attacks(outdir, n_per_family):
    path = Path(outdir) / "followup_results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    d = pd.DataFrame(rows); d = d[d.experiment == "E0_specificity"]
    per = d.groupby(["family", "attack_ix", "cond"]).attack.mean().unstack("cond")
    per = per[(per["none"] > 0) & (per["restore_role"] == 0)].reset_index()
    per = per.sort_values(["family", "none"], ascending=[True, False]).groupby("family").head(n_per_family)
    print("chosen (baseline ASR -> restore ASR):")
    for _, r in per.iterrows():
        print(f"  {r.family:8s} attack {int(r.attack_ix):3d}   {r['none']:.2f} -> {r['restore_role']:.2f}")
    logs = d[(d.attack_ix.isin(per.attack_ix)) & (d.cond.isin(["none", "restore_role"]))][["attack_ix", "family", "cond", "sample", "attack", "log"]]
    return per.attack_ix.astype(int).tolist(), logs


# ---------------- the paper's probe ----------------
def train_role_probe(model, tok, texts, layer, args, seed):
    """Multinomial LR on content tokens of identical text under each tag. Returns pipeline and held-out accuracy."""
    recs = [(r, V.render_single_role(r, t), i) for i, t in enumerate(texts) for r in ROLES]
    X, y, g = [], [], []
    for st in tqdm(range(0, len(recs), args.role_batch_size), desc=f"probe data @L{layer}"):
        batch = recs[st:st + args.role_batch_size]
        store, ids = V.capture_batch(model, tok, [b[1] for b in batch], [layer], args.role_max_tokens + 32)
        for bi, (role, _, ti) in enumerate(batch):
            s, e = V.content_slice_from_ids(ids[bi], tok)
            H = store[layer][bi, s:e]
            X.append(H); y += [role] * (e - s); g += [ti] * (e - s)
    X = torch.cat(X).numpy(); y = np.array(y); g = np.array(g)
    rng = np.random.default_rng(seed); test_texts = set(rng.choice(np.unique(g), size=max(2, len(np.unique(g)) // 5), replace=False))
    tr, te = ~np.isin(g, list(test_texts)), np.isin(g, list(test_texts))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    clf.fit(X[tr], y[tr])
    acc = float((clf.predict(X[te]) == y[te]).mean())
    print(f"probe @L{layer}: {tr.sum()} train / {te.sum()} held-out tokens, held-out accuracy {acc:.3f}  (chance 0.20)")
    return clf, acc


def probe_probs(clf, H):
    P = clf.predict_proba(H.numpy()); cls = list(clf.classes_)
    return {r: P[:, cls.index(r)] for r in ROLES}


# ---------------- plot ----------------
def cotness_figure(tok, msgs, ts, sp, probs_by_arm, mean_by_arm, fam, aix, edit_layer, probe_layer, outdir):
    s0, e0 = ts[0]; n = e0 - s0
    arms = list(probs_by_arm)
    fig, axes = plt.subplots(len(arms), 1, figsize=(8.5, 1.9 * len(arms) + 0.6), sharex=True, squeeze=False)
    key = "cot" if fam == "forgery" else "user"
    for ax, arm in zip(axes[:, 0], arms):
        P = probs_by_arm[arm]
        x = np.arange(n)
        ax.axvspan(sp[0] - s0, sp[1] - s0, color="#c0392b", alpha=.10, lw=0)
        for r in ("cot", "user", "tool"):
            ax.plot(x, P[r], ls="none", marker="o", ms=2.2, color=COL[r], label=f"P({r})", alpha=.9)
        m = mean_by_arm[arm]
        ax.text(0.995, 0.92, f"{arm}   mean over injected tokens:  P(cot)={m['cot']:.2f}  P(user)={m['user']:.2f}  P(tool)={m['tool']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.9))
        ax.set_ylim(-0.02, 1.02); ax.set_ylabel("probe\nprobability")
    axes[0, 0].legend(loc="upper left", ncol=3, fontsize=7.5)
    axes[-1, 0].set_xlabel("token position in the tool span (shaded = injected text)")
    fig.suptitle(f"{fam} attack {aix}: what the role probe reads at layer {probe_layer}, before and after editing layer {edit_layer}\n"
                 f"baseline: model exfiltrated  |  after restore: it did not", fontsize=9.5, y=1.02)
    fd = Path(outdir) / "figs"; fd.mkdir(exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(fd / f"cotness_{fam}_{aix}.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"  fig -> cotness_{fam}_{aix}")

    # zoom on the injected region with the actual tokens along the axis
    ids = tok("".join(msgs) + "<|start|>assistant", add_special_tokens=False).input_ids
    toks = [tok.decode([t]) for t in ids[sp[0]:sp[1]]]
    k = len(toks); step = max(1, k // 40)
    fig, ax = plt.subplots(figsize=(min(14, 0.25 * k / step + 3), 2.8))
    for arm, ls in [("none", "-"), (arms[-1], "--")]:
        P = probs_by_arm[arm]; seg = slice(sp[0] - s0, sp[1] - s0)
        mk = "o" if arm == "none" else "^"
        ax.plot(np.arange(k), P[key][seg], ls="none", marker=mk, ms=3.2, color=COL[key], label=f"P({key}) — {arm}")
        ax.plot(np.arange(k), P["tool"][seg], ls="none", marker=mk, ms=3.2, mfc="none", color=COL["tool"], label=f"P(tool) — {arm}")
    ax.set_xticks(np.arange(0, k, step)); ax.set_xticklabels([t.replace("\n", "\\n")[:10] for t in toks[::step]], rotation=90, fontsize=6)
    ax.set_ylim(-0.02, 1.02); ax.set_ylabel("probe probability"); ax.legend(fontsize=7, ncol=2, loc="center right")
    ax.set_title(f"{fam} attack {aix}: injected tokens only  (circles = no intervention, triangles = restore@L{edit_layer})", loc="left", fontsize=9)
    for ext in ("png", "pdf"): fig.savefig(fd / f"cotness_{fam}_{aix}_zoom.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"  fig -> cotness_{fam}_{aix}_zoom")


# =====================================================================================
def main():
    args = parse_args(); V.set_seed(args.seed)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    EL, PL = args.edit_layer, args.probe_layer

    if args.attack_ids:
        chosen, logs = [int(x) for x in args.attack_ids.split(",")], pd.DataFrame()
    else:
        chosen, logs = pick_attacks(out, args.n_per_family)

    model, tok = V.load_model_and_tokenizer(args)
    carriers, benign, role_texts = V.load_wikipedia_material(args)
    layers = tuple(sorted({EL, PL}))
    role_means = V.estimate_role_means(model, tok, role_texts, layers, args)
    mu_ref = V.benign_tool_reference(model, tok, benign, layers)
    dirs = V.build_dirs(role_means, mu_ref, layers, args.seed)
    clf, acc = train_role_probe(model, tok, role_texts, PL, args, args.seed)
    attacks = V.load_published_attacks(carriers, args)
    sel = attacks[attacks.attack_ix.isin(chosen)]

    S_NONE = Spec3("none", (), "none"); S_RND = Spec3(f"random_MM_L{EL}", (EL,), "restore_rand_mm"); S_RES = Spec3(f"restore_L{EL}", (EL,), "restore")
    summary = []
    for r in sel.to_dict("records"):
        msgs = V.scaffold(r["task"], r["html"]); ts = V.tool_spans(tok, msgs); sp = V.injected_span(tok, msgs, r["inj"])
        if sp is None:
            print(f"attack {r['attack_ix']}: could not locate injected span, skipping"); continue
        prompt = "".join(msgs) + "<|start|>assistant"; s0, e0 = ts[0]
        probs_by_arm, mean_by_arm = {}, {}
        for spec in (S_NONE, S_RND, S_RES):
            H, _ = capture_under(model, tok, prompt, spec, ts, dirs, [PL])
            P = probe_probs(clf, H[PL][s0:e0]); probs_by_arm[spec.name] = P
            seg = slice(sp[0] - s0, sp[1] - s0); rest = np.ones(e0 - s0, bool); rest[seg] = False
            mean_by_arm[spec.name] = {k: float(P[k][seg].mean()) for k in ROLES}
            summary.append({"attack_ix": r["attack_ix"], "family": r["family"], "cond": spec.name,
                            **{f"inj_P({k})": float(P[k][seg].mean()) for k in ROLES},
                            **{f"page_P({k})": float(P[k][rest].mean()) for k in ("cot", "user", "tool")}})
        cotness_figure(tok, msgs, ts, sp, probs_by_arm, mean_by_arm, r["family"], r["attack_ix"], EL, PL, out)

    sm = pd.DataFrame(summary).round(3); sm.to_csv(out / "cotness_summary.csv", index=False)
    print("\nMean probe probability over the INJECTED tokens (probe at L%d, edit at L%d, held-out probe acc %.2f):" % (PL, EL, acc))
    print(sm.pivot_table(index=["family", "attack_ix"], columns="cond", values=["inj_P(cot)", "inj_P(user)", "inj_P(tool)"]).round(2).to_string())

    if len(logs):
        with open(out / "cotness_transcripts.md", "w") as f:
            for aix in chosen:
                sub = logs[logs.attack_ix == aix]
                if sub.empty: continue
                f.write(f"\n\n# attack {aix} ({sub.family.iloc[0]})\n")
                for cond in ("none", "restore_role"):
                    ex = sub[sub.cond == cond].sort_values("attack", ascending=(cond != "none")).iloc[0]
                    f.write(f"\n## {cond}  (attack={ex.attack})\n```\n{ex.log[:1800]}\n```\n")
        print("transcripts -> cotness_transcripts.md")
    print("done ->", out)


if __name__ == "__main__":
    main()
