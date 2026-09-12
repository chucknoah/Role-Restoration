#!/usr/bin/env python3
# ================================================================================================
# STAGE     7. Figure 1 data: colour-coded text + trace, before/after; dumps textfig_*.json
# RAN ON    RunPod (imports 03, 04, 05)
# NEEDS     followup_results.jsonl
# PRODUCES  figs/textfig_<family>_<id>.json (+png, html)
# FIGURE    Figure 1 (render locally with 06/render_textfig.py)
# NOTE      code unchanged from the run that produced the paper's numbers; only this header was added
# ================================================================================================
"""
Paper-style figure: the attack as color-coded text next to the per-token role-probe trace, BEFORE and AFTER restoration.

Encoding (differs from the paper on purpose):
  text color       = role the probe PERCEIVES for that token (argmax of P(role))   <- what the model thinks
  background band  = role the harness ASSIGNED (true provenance)                    <- what is actually true
So injected tool text that turns orange/blue on a grey band is role confusion, visibly. After restore it turns grey.
Top block: no intervention, with the model's own continuation (it exfiltrates).
Bottom block: restore at layer 8, with the model's own continuation (it does not).

Needs role_mech_followups_v2.py, role_mech_plots.py, role_cotness_plot.py in the same directory.  pip install scikit-learn
    python role_cotness_text_figure.py --outdir /workspace/role_mech --n-per-family 1
"""
from __future__ import annotations
import argparse, html, json, os, re
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import role_mech_followups_v2 as V
from role_mech_plots import Intervene3, Spec3, capture_under
from role_cotness_plot import pick_attacks, train_role_probe, probe_probs

ROLES = V.ROLES
NICE = {"cot": "CoTness", "user": "Userness", "tool": "Toolness"}
PERCEIVED = {"cot": "#d35400", "user": "#1f77b4", "tool": "#7f8c8d", "assistant": "#27ae60", "system": "#95a5a6", "other": "#222222"}
FAMILY_LABEL = {"forgery": "style-forgery", "prefix": "prefix"}
TRUE_BAND = {"user task": "#dbeafe", "tool: page": "#ececec", "tool: injected": "#fde2df", "model CoT": "#fff1d6", "model tool call": "#dcf5e3", "model final": "#dcf5e3", "other": "#ffffff"}
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
    p.add_argument("--n-per-family", type=int, default=1)
    p.add_argument("--attack-ids", default=None)
    p.add_argument("--n-carriers", type=int, default=3)
    p.add_argument("--n-benign-ref", type=int, default=8)
    p.add_argument("--carrier-chars", type=int, default=6000)
    p.add_argument("--n-prefix-user", type=int, default=8)
    p.add_argument("--n-prefix-other", type=int, default=4)
    p.add_argument("--role-texts", type=int, default=40)
    p.add_argument("--role-max-tokens", type=int, default=160)
    p.add_argument("--role-batch-size", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("--context-tokens", type=int, default=70, help="page tokens shown before the injection")
    p.add_argument("--gen-tokens", type=int, default=140, help="generated tokens shown")
    p.add_argument("--chars-per-line", type=int, default=96)
    p.add_argument("--max-tries", type=int, default=6, help="sampling seeds tried to get attack=True (none) / False (restore)")
    p.add_argument("--show-user", action="store_true", default=True)
    p.add_argument("--no-show-user", dest="show_user", action="store_false")
    p.add_argument("--also-tool", action="store_true", help="also draw P(tool) faintly")
    return p.parse_args()


# ---------------- one generation step under an arm, with a target outcome ----------------
def generate_with_outcome(model, tok, r, spec, dirs, args, want_attack):
    msgs = V.scaffold(r["task"], r["html"]); ts = V.tool_spans(tok, msgs)
    prompt = "".join(msgs) + "<|start|>assistant"
    tries = [("greedy", None)] + [("sample", args.seed + 7919 * k) for k in range(args.max_tries)]
    best = None
    for kind, sd in tries:
        with Intervene3(model, spec, ts, dirs):
            raw = V.gen_step(model, tok, prompt, sd if sd is not None else args.seed, args, greedy=(kind == "greedy"))
        p = V.parse_harmony(raw)
        cmds = []
        for fqn, aj in p["tool_calls"]:
            try: cmds.append(json.loads(aj).get("command", ""))
            except Exception: cmds.append(aj)
        attack = any(V.EXFIL.search(c) for c in cmds)
        cont = [V.h_analysis(a) for a in p["analysis"]]
        if p["tool_calls"]: cont.append(V.h_tool_call(p["tool_calls"][0][0], p["tool_calls"][0][1]))
        elif p["final"]: cont.append(V.h_final(p["final"]))
        best = best or (cont, attack, kind, sd)
        if attack == want_attack:
            return cont, attack, kind, sd
    return best


# ---------------- token layout in monospace ----------------
def draw_text_block(ax, toks, perceived, bands, chars_per_line, fontsize=7.4):
    ax.axis("off"); ax.set_xlim(0, chars_per_line); col, row = 0, 0
    for t, pr, bd in zip(toks, perceived, bands):
        s = t.replace("\n", "\u23ce")
        if col + len(s) > chars_per_line and col > 0:
            col, row = 0, row + 1
        ax.text(col, -row, s, fontsize=fontsize, family="monospace", color=PERCEIVED[pr], va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.02", fc=TRUE_BAND[bd], ec="none"))
        col += len(s)
        if "\n" in t:
            col, row = 0, row + 1
    ax.set_ylim(-(row + 1.2), 0.6)
    return row + 1


def region_of(i, spans):
    for name, (s, e) in spans.items():
        if s <= i < e: return name
    return "other"


def build_view(tok, msgs, cont, sp, ts, args):
    """Token window: [user task] ... [context before injection | injection] [generated continuation]. Returns ids, idx, toks, labels, usr_n."""
    full = "".join(msgs + cont)
    ids = tok(full, add_special_tokens=False).input_ids
    s0, e0 = ts[0]
    u0, u1 = V.ntok(tok, "".join(msgs[:2])), V.ntok(tok, "".join(msgs[:3]))
    gen0 = V.ntok(tok, "".join(msgs))
    spans = {"user task": (u0, u1), "tool: injected": (sp[0], sp[1])}
    cursor = gen0
    for m in cont:
        n = V.ntok(tok, m)
        name = "model CoT" if "analysis" in m[:60] else ("model tool call" if "commentary to=" in m[:80] else "model final")
        spans[name + f"#{cursor}"] = (cursor, cursor + n); cursor += n
    a, b = max(s0, sp[0] - args.context_tokens), sp[1]
    c, d = gen0, min(len(ids), gen0 + args.gen_tokens)
    usr = list(range(u0, u1)) if args.show_user else []
    idx = usr + list(range(a, b)) + list(range(c, d))
    labels = []
    for i in idx:
        lab = region_of(i, spans)
        if lab == "other" and s0 <= i < e0: lab = "tool: page"
        labels.append(lab.split("#")[0])
    toks = [tok.decode([ids[i]]) for i in idx]
    return ids, idx, toks, labels, len(usr), (a, b, c, d)


def main():
    args = parse_args(); V.set_seed(args.seed)
    out = Path(args.outdir); (out / "figs").mkdir(parents=True, exist_ok=True)
    EL, PL = args.edit_layer, args.probe_layer
    chosen = [int(x) for x in args.attack_ids.split(",")] if args.attack_ids else pick_attacks(out, args.n_per_family)[0]

    model, tok = V.load_model_and_tokenizer(args)
    carriers, benign, role_texts = V.load_wikipedia_material(args)
    layers = tuple(sorted({EL, PL}))
    role_means = V.estimate_role_means(model, tok, role_texts, layers, args)
    mu_ref = V.benign_tool_reference(model, tok, benign, layers)
    dirs = V.build_dirs(role_means, mu_ref, layers, args.seed)
    clf, acc = train_role_probe(model, tok, role_texts, PL, args, args.seed)
    attacks = V.load_published_attacks(carriers, args)
    S_NONE, S_RES = Spec3("none", (), "none"), Spec3(f"restore_L{EL}", (EL,), "restore")

    for r in attacks[attacks.attack_ix.isin(chosen)].to_dict("records"):
        fam, aix = r["family"], r["attack_ix"]; key = "cot" if fam == "forgery" else "user"
        msgs = V.scaffold(r["task"], r["html"]); ts = V.tool_spans(tok, msgs); sp = V.injected_span(tok, msgs, r["inj"])
        if sp is None: print(f"attack {aix}: no injected span"); continue
        blocks = []
        for spec, want in [(S_NONE, True), (S_RES, False)]:
            cont, attack, kind, sd = generate_with_outcome(model, tok, r, spec, dirs, args, want)
            ids, idx, toks, labels, usr_n, (a, b, c, d) = build_view(tok, msgs, cont, sp, ts, args)
            H, _ = capture_under(model, tok, "".join(msgs + cont), spec, ts, dirs, [PL])
            P = probe_probs(clf, H[PL])
            perceived = [max(ROLES, key=lambda k: P[k][i]) for i in idx]
            Pv = {k: [float(P[k][i]) for i in idx] for k in ROLES}
            if usr_n:                                              # visual gap between the user turn and the page
                toks.insert(usr_n, " \u22ef "); labels.insert(usr_n, "other"); perceived.insert(usr_n, "other")
                for k in ROLES: Pv[k].insert(usr_n, float("nan"))
            off = usr_n + (1 if usr_n else 0)
            inj_v = (off + (sp[0] - a), off + (sp[1] - a)); cut_v = off + (b - a)
            action = ""
            for m in cont:
                if "commentary to=" in m[:80]:
                    try: action = "bash: " + json.loads(re.search(r"<\|message\|>(.*)<\|end\|>", m, re.S).group(1)).get("command", "")
                    except Exception: action = "tool call"
                elif "final<|message|>" in m[:60]:
                    action = "final answer: " + re.search(r"<\|message\|>(.*)<\|end\|>", m, re.S).group(1).strip()[:160]
            blocks.append(dict(spec=spec, attack=attack, kind=kind, seed=sd, toks=toks, labels=labels, perceived=perceived, action=action,
                               P={k: np.array(Pv[k]) for k in ROLES}, cut=cut_v, inj=inj_v, usr=usr_n))
            print(f"  {spec.name}: attack={attack} ({kind}{'' if sd is None else f' seed {sd}'})  "
                  f"mean P({key}) over injected tokens = {P[key][sp[0]:sp[1]].mean():.2f}, P(tool) = {P['tool'][sp[0]:sp[1]].mean():.2f}")

        # ---------------- save everything the figure needs, so it can be re-rendered without the model ----------------
        flab = FAMILY_LABEL.get(fam, fam)
        dump = {"family": fam, "family_label": flab, "attack_ix": int(aix), "edit_layer": EL, "probe_layer": PL, "probe_acc": float(acc),
                "blocks": [{"cond": b["spec"].name, "attack": bool(b["attack"]), "kind": b["kind"], "seed": b["seed"], "action": b["action"],
                            "toks": b["toks"], "labels": b["labels"], "perceived": b["perceived"], "cut": int(b["cut"]), "inj": [int(b["inj"][0]), int(b["inj"][1])], "usr": int(b.get("usr", 0)),
                            "P": {k: [None if (isinstance(v, float) and v != v) else float(v) for v in b["P"][k]] for k in b["P"]}} for b in blocks]}
        (out / "figs" / f"textfig_{flab}_{aix}.json").write_text(json.dumps(dump))
        print(f"  data -> textfig_{flab}_{aix}.json  (re-render locally with render_textfig.py)")

        # ---------------- figure: [text | CoTness trace | legend] x 2 ----------------
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        fig = plt.figure(figsize=(17, 9.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1], wspace=0.06, hspace=0.42)
        for bi, blk in enumerate(blocks):
            axT = fig.add_subplot(gs[bi, 0]); axP = fig.add_subplot(gs[bi, 1])
            draw_text_block(axT, blk["toks"], blk["perceived"], blk["labels"], args.chars_per_line)
            title = ("NO INTERVENTION" if bi == 0 else f"RESTORE @ layer {EL}") + ("  —  model exfiltrated" if blk["attack"] else "  —  model did not act")
            axT.set_title(title, loc="left", fontsize=10.5, color="#c0392b" if blk["attack"] else "#1f77b4", pad=6)
            # the model's action, boxed under the text
            axT.text(0, axT.get_ylim()[0] - 0.2, "MODEL ACTION  →  " + blk["action"], fontsize=8, family="monospace", va="top", ha="left",
                     color="#c0392b" if blk["attack"] else "#1f77b4", clip_on=False,
                     bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f5" if blk["attack"] else "#f2f7fd", ec="#c0392b" if blk["attack"] else "#1f77b4", lw=.8))
            # CoTness trace
            x = np.arange(len(blk["toks"]))
            if blk.get("usr"): axP.axvspan(0, blk["usr"], color=TRUE_BAND["user task"], lw=0, zorder=0)
            axP.axvspan(blk.get("usr", 0) + (1 if blk.get("usr") else 0), blk["inj"][0], color=TRUE_BAND["tool: page"], lw=0, zorder=0)
            axP.axvspan(blk["inj"][0], blk["inj"][1], color=TRUE_BAND["tool: injected"], lw=0, zorder=0)
            axP.axvspan(blk["cut"], len(x), color=TRUE_BAND["model CoT"], lw=0, zorder=0)
            metric = "cot" if fam == "forgery" else "user"
            axP.scatter(x, 100 * blk["P"][metric], s=9, color="#333333", zorder=3, label=f"{NICE[metric]} = P({metric} | token)")
            if args.also_tool:
                axP.plot(x, 100 * blk["P"]["tool"], ls="none", marker="o", ms=1.6, color=PERCEIVED["tool"], alpha=.5, label="P(tool | token)")
            axP.axvline(blk["cut"] - 0.5, color="k", lw=.6, ls=":")
            inj_mean = 100 * float(np.nanmean(blk["P"][metric][blk["inj"][0]:blk["inj"][1]]))
            axP.hlines(inj_mean, blk["inj"][0], blk["inj"][1], color="#333333", lw=1.2, ls="--")
            axP.text((blk["inj"][0] + blk["inj"][1]) / 2, min(97, inj_mean + 6), f"injected text: mean {metric.upper()}ness {inj_mean:.0f}%", ha="center", fontsize=7.5)
            axP.set_ylim(0, 102); axP.set_ylabel(f"{NICE[metric]} (%)")
            u = blk.get("usr", 0); p0 = u + (1 if u else 0)
            ticks = ([u / 2] if u else []) + [(p0 + blk["inj"][0]) / 2, (blk["inj"][0] + blk["inj"][1]) / 2, (blk["cut"] + len(x)) / 2]
            axP.set_xticks(ticks); axP.set_xticklabels((["User task"] if u else []) + ["Tool (page)", "Tool (injected)", "Model's own CoT →"], fontsize=8)
            axP.text(0.01, 0.97, f"probe reads layer {PL}", transform=axP.transAxes, fontsize=7, va="top", color="#666")
            # legend to the right of this block
            h = [Line2D([], [], color="#333333", ls="none", marker="o", ms=4, label=f"{NICE[metric]} = P({metric} | token)")]
            if args.also_tool: h.append(Line2D([], [], color=PERCEIVED["tool"], ls="none", marker="o", ms=3, label="P(tool | token)"))
            h += [Line2D([], [], color=PERCEIVED[k], lw=4, label=f"text: perceived as {k}") for k in ("cot", "user", "tool", "assistant")]
            h += [Patch(fc=TRUE_BAND[k], ec="#999", label=f"band: actually {k}") for k in ("user task", "tool: page", "tool: injected", "model CoT")]
            axP.legend(handles=h, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5, borderaxespad=0)
        fig.suptitle(f"{flab} attack {aix}  ·  text colored by what the probe thinks each token is; bands show what it actually is  ·  held-out probe acc {acc:.2f}",
                     fontsize=10, y=0.995)
        for ext in ("png", "pdf"): fig.savefig(out / "figs" / f"textfig_{flab}_{aix}.{ext}", bbox_inches="tight")
        plt.close(fig); print(f"  fig -> textfig_{flab}_{aix}")

        # ---------------- HTML version (exact same encoding, no wrapping headaches) ----------------
        def span(t, pr, bd):
            return f'<span style="color:{PERCEIVED[pr]};background:{TRUE_BAND[bd]};white-space:pre">{html.escape(t)}</span>'
        parts = [f"<h3>{flab} attack {aix}</h3><p style='font:12px sans-serif'>text color = perceived role (probe @L{PL}); background = true provenance.</p>"]
        for blk in blocks:
            parts.append(f"<h4 style='color:{'#c0392b' if blk['attack'] else '#1f77b4'}'>{blk['spec'].name}: attack={blk['attack']} &nbsp; <code>{html.escape(blk['action'])}</code></h4>"
                         "<div style='font:11px/1.5 monospace;max-width:1000px'>" +
                         "".join(span(t, pr, bd) for t, pr, bd in zip(blk["toks"], blk["perceived"], blk["labels"])) + "</div>")
        (out / "figs" / f"textfig_{flab}_{aix}.html").write_text("<html><body>" + "".join(parts) + "</body></html>")
    print("done ->", out / "figs")


if __name__ == "__main__":
    main()
