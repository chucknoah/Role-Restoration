#!/usr/bin/env python3
# ================================================================================================
# STAGE     5. Mech plots: token-level mismatch traces, persistence across depth, attention to the injection; optional dose / sweep / leave-one-out
# RAN ON    RunPod (imports 03)
# NEEDS     role_mech_followups_v2.py in the same directory; --successful-ids
# PRODUCES  M1_*.png, M2_persistence.csv, M3_attention.csv
# FIGURE    supporting
# NOTE      code unchanged from the run that produced the paper's numbers; only this header was added
# ================================================================================================
"""
Mechanistic plots for the role-restoration result. Imports the machinery from role_mech_followups_v2.py
(keep both files in the same directory).

Forward-pass only (minutes):
  M1  token-level role-mismatch TRACE along the tool span, injected region shaded, under none / restore_L8 / random_MM_L8
  M2  PERSISTENCE: edit at layer 8 only, measure mismatch at layers 8..22 (fixes the "measured at the edit layer" tautology)
  M3  ATTENTION from the model's first planning-CoT tokens onto {injected span, rest of page, user turn, other}, by layer,
      under none / restore_L8   (requires eager attention; uses shorter carriers to keep attention matrices small)
Generation (flags, ~30-40 min each on an A100):
  M4  DOSE-RESPONSE: partial restore alpha in {0.25, 0.5, 0.75} at layer 8
  M5  FINER LAYER SWEEP: restore at layers 2,4,6,10 (merge with 8/12/16 from E1)
  M6  LEAVE-ONE-OUT: restore 3 of 4 role coordinates, four ways
Also writes: table_dataset.md, table_condition_ladder.md

Usage:
    python role_mech_plots.py --outdir /workspace/role_mech --successful-ids 0,1,2,7,8,10,13,14,16,28,30,32,34,35,37,38
    add --run-dose --run-sweep --run-loo for the generation experiments
"""
from __future__ import annotations
import argparse, json, os, re, math, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np, pandas as pd, torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import role_mech_followups_v2 as V

C = dict(none="#424242", restore="#1f77b4", rand="#9e9e9e", inj="#c0392b", forgery="#e67e22", prefix="#16a085")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300, "pdf.fonttype": 42})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="openai/gpt-oss-20b")
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/workspace/hf"))
    p.add_argument("--outdir", default="/workspace/role_mech")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--edit-layer", type=int, default=8)
    p.add_argument("--measure-layers", default="8,10,12,14,16,18,20,22")
    p.add_argument("--sweep-layers", default="2,4,6,10")
    p.add_argument("--successful-ids", required=True)
    p.add_argument("--n-trace", type=int, default=3, help="attacks per family for the token traces")
    p.add_argument("--n-carriers", type=int, default=3)
    p.add_argument("--n-benign-ref", type=int, default=8)
    p.add_argument("--carrier-chars", type=int, default=6000)
    p.add_argument("--attn-carrier-chars", type=int, default=2200, help="shorter pages for M3 so attention matrices fit")
    p.add_argument("--n-prefix-user", type=int, default=8)
    p.add_argument("--n-prefix-other", type=int, default=4)
    p.add_argument("--role-texts", type=int, default=24)
    p.add_argument("--role-max-tokens", type=int, default=160)
    p.add_argument("--role-batch-size", type=int, default=6)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("--run-dose", action="store_true")
    p.add_argument("--run-sweep", action="store_true")
    p.add_argument("--run-loo", action="store_true")
    p.add_argument("--skip-attn", action="store_true")
    return p.parse_args()


# ---------------- extended intervention: partial restore, coordinate subsets ----------------
@dataclass
class Spec3:
    name: str
    layers: Tuple[int, ...]
    mode: str                                   # none | restore | restore_rand_mm
    alpha: float = 1.0                          # fraction of the way to the reference
    dims: Optional[Tuple[int, ...]] = None      # subset of role-subspace columns; None = all
    scope: str = "tool_span"


class Intervene3(V.Intervene):
    def __enter__(self):
        if self.spec.mode == "none":
            return self
        L_mods = V.get_layers(self.model)
        for L in self.spec.layers:
            d = self.dirs[L]

            def hook(mod, inp, out, d=d):
                t = V.first_tensor(out); N = t.shape[1]
                if N == 1 or not self.spans:
                    return out
                mask = torch.zeros(N, dtype=torch.bool, device=t.device)
                for s, e in self.spans:
                    mask[s:min(e, N)] = True
                if not mask.any():
                    return out
                x = t[:, mask].float()
                Q = d["Q"].to(x.device); tgt_full = d["target"].to(x.device)
                if self.spec.dims is not None:
                    Q = Q[:, list(self.spec.dims)]
                    tgt = Q @ (Q.T @ tgt_full)
                else:
                    tgt = tgt_full
                ref = -((x @ Q) @ Q.T) + tgt
                if self.spec.mode == "restore":
                    delta = self.spec.alpha * ref
                elif self.spec.mode == "restore_rand_mm":
                    Qr = d["Q_rand"].to(x.device); tr = d["target_rand"].to(x.device)
                    raw = -((x @ Qr) @ Qr.T) + tr
                    delta = self.spec.alpha * raw * (ref.norm(dim=-1, keepdim=True) / raw.norm(dim=-1, keepdim=True).clamp_min(1e-8))
                else:
                    raise ValueError(self.spec.mode)
                self.delta_norms.extend(delta.norm(dim=-1).detach().cpu().flatten().tolist())
                nt = t.clone(); nt[:, mask] = (x + delta).to(t.dtype)
                return (nt,) + tuple(out[1:]) if isinstance(out, tuple) else nt
            self.handles.append(L_mods[L].register_forward_hook(hook))
        return self


def run_attack3(model, tok, row, spec, dirs, sample_seed, args):
    msgs = V.scaffold(row["task"], row["html"]); attack, broken, dn = False, False, []
    for step in range(args.max_steps):
        ctx = Intervene3(model, spec, V.spans_for(tok, msgs, row["inj"], spec.scope), dirs)
        with ctx:
            raw = V.gen_step(model, tok, "".join(msgs) + "<|start|>assistant", sample_seed + 1000 * step, args)
        dn.extend(ctx.delta_norms)
        p = V.parse_harmony(raw)
        if not (p["tool_calls"] or p["final"] or p["analysis"]):
            broken = True; break
        for a in p["analysis"]: msgs.append(V.h_analysis(a))
        if p["final"]: msgs.append(V.h_final(p["final"]))
        for fqn, aj in p["tool_calls"]:
            try: cmd = json.loads(aj).get("command", "")
            except Exception: cmd = aj
            attack |= bool(V.EXFIL.search(cmd))
            msgs.append(V.h_tool_call(fqn, aj)); msgs.append(V.h_tool_res(fqn, json.dumps(V.fake_bash(cmd, row["html"]), ensure_ascii=False)))
        if attack or not p["tool_calls"]: break
    return {"attack": bool(attack), "broken": bool(broken), "delta_norm_mean": float(np.mean(dn)) if dn else 0.0}


def grid3(model, tok, subset, specs, dirs, n_samples, experiment, store, args, seed_offset):
    todo = [(sp, r, s) for sp in specs for r in subset.to_dict("records") for s in range(n_samples)
            if (experiment, int(r["attack_ix"]), sp.name, s) not in store.done]
    for sp, r, s in tqdm(todo, desc=experiment):
        out = run_attack3(model, tok, r, sp, dirs, args.seed + seed_offset + s, args)
        store.append({"experiment": experiment, "attack_ix": int(r["attack_ix"]), "family": r["family"], "type": r["type"],
                      "cond": sp.name, "scope": sp.scope, "sample": s, **out, "cot_mentions": None, "log": ""})


# ---------------- capture with intervention ----------------
@torch.inference_mode()
def capture_under(model, tok, prompt, spec, spans, dirs, layers, want_attn=False, q_range=None):
    """One forward pass under `spec`; returns per-layer residuals (N,D) and, optionally, attention rows for q_range."""
    store, attn, handles = {}, {}, []
    L_mods = V.get_layers(model)
    with Intervene3(model, spec, spans, dirs):
        for L in layers:
            handles.append(L_mods[L].register_forward_hook(lambda m, i, o, L=L: store.__setitem__(L, V.first_tensor(o)[0].detach().float().cpu())))
        if want_attn:
            for L in range(len(L_mods)):
                def ah(m, i, o, L=L):
                    w = o[1] if isinstance(o, tuple) and len(o) > 1 else None
                    if w is not None and q_range is not None:
                        attn[L] = w[0, :, q_range[0]:q_range[1], :].float().mean(0).detach().cpu()   # (nq, N) averaged over heads
                handles.append(L_mods[L].self_attn.register_forward_hook(ah))
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        model(**enc, use_cache=False, output_attentions=want_attn)
        for h in handles: h.remove()
    return store, attn


def mismatch_per_token(H, d):
    return ((H @ d["Q"]) - (d["Q"].T @ d["target"])).norm(dim=-1).numpy()


def savefig(fig, outdir, name):
    fd = Path(outdir) / "figs"; fd.mkdir(exist_ok=True, parents=True)
    for ext in ("png", "pdf"): fig.savefig(fd / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"  fig -> {name}")


# =====================================================================================
def main():
    args = parse_args()
    V.set_seed(args.seed)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    EL = args.edit_layer
    MEAS = tuple(int(x) for x in args.measure_layers.split(","))
    SWEEP = tuple(int(x) for x in args.sweep_layers.split(","))
    ALL_L = tuple(sorted(set(MEAS) | set(SWEEP) | {EL, 12, 16}))

    model, tok = V.load_model_and_tokenizer(args)
    carriers, benign, role_texts = V.load_wikipedia_material(args)
    print("role means at layers", ALL_L)
    role_means = V.estimate_role_means(model, tok, role_texts, ALL_L, args)
    mu_ref = V.benign_tool_reference(model, tok, benign, ALL_L)
    dirs = V.build_dirs(role_means, mu_ref, ALL_L, args.seed)
    attacks = V.load_published_attacks(carriers, args)
    ids = [int(x) for x in args.successful_ids.split(",") if x.strip()]
    sel = attacks[attacks.attack_ix.isin(ids)].reset_index(drop=True)
    sel = sel.assign(inj=sel["inj"])
    print("selected", sel.groupby("family").size().to_dict())
    store = V.Store(out)
    S_NONE = Spec3("none", (), "none"); S_RES = Spec3(f"restore_L{EL}", (EL,), "restore"); S_RND = Spec3(f"random_MM_L{EL}", (EL,), "restore_rand_mm")

    # ---------------- M1: token traces ----------------
    print("\nM1 token-level mismatch traces")
    trace_rows = []
    for fam in ("forgery", "prefix"):
        for r in sel[sel.family == fam].head(args.n_trace).to_dict("records"):
            msgs = V.scaffold(r["task"], r["html"]); ts = V.tool_spans(tok, msgs); sp = V.injected_span(tok, msgs, r["inj"])
            if sp is None: continue
            prompt = "".join(msgs) + "<|start|>assistant"
            fig, axes = plt.subplots(2, 1, figsize=(7.5, 3.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            s0, e0 = ts[0]
            for spec, col, lab in [(S_NONE, C["none"], "no intervention"), (S_RND, C["rand"], f"matched random @L{EL}"), (S_RES, C["restore"], f"restore @L{EL}")]:
                H, _ = capture_under(model, tok, prompt, spec, ts, dirs, [12])
                m = mismatch_per_token(H[12][s0:e0], dirs[12])
                axes[0].plot(np.arange(s0, e0), m, ls="none", marker="o", ms=1.8, color=col, label=lab, alpha=.9 if spec is S_RES else .7)
                trace_rows.append({"attack_ix": r["attack_ix"], "family": fam, "cond": spec.name,
                                   "mean_inj": float(m[sp[0]-s0:sp[1]-s0].mean()), "mean_rest": float(np.delete(m, np.arange(sp[0]-s0, sp[1]-s0)).mean())})
            for ax in axes: ax.axvspan(sp[0], sp[1], color=C["inj"], alpha=.12, lw=0)
            axes[0].set_ylabel("distance from\nhonest-tool role coords (L12)"); axes[0].legend(loc="upper left", ncol=3, fontsize=7)
            axes[0].set_title(f"{fam} attack {r['attack_ix']}: the injection lights up in role space; restore flattens it", loc="left")
            # projections onto user/cot axes for the no-intervention run
            H, _ = capture_under(model, tok, prompt, S_NONE, ts, dirs, [12]); Hs = H[12][s0:e0]
            axes[1].plot(np.arange(s0, e0), (Hs @ dirs[12]["axes"]["user"]).numpy() - float(mu_ref[12] @ dirs[12]["axes"]["user"]), ls="none", marker="o", ms=1.6, color="#2c3e50", label="user-ness (rel. to honest tool)")
            axes[1].plot(np.arange(s0, e0), (Hs @ dirs[12]["axes"]["cot"]).numpy() - float(mu_ref[12] @ dirs[12]["axes"]["cot"]), ls="none", marker="o", ms=1.6, color=C["inj"], label="cot-ness (rel. to honest tool)")
            axes[1].axhline(0, color="#999", lw=.5); axes[1].legend(fontsize=6.5, ncol=2, loc="upper left"); axes[1].set_xlabel("token position in tool span"); axes[1].set_ylabel("proj.")
            savefig(fig, out, f"M1_trace_{fam}_{r['attack_ix']}")
    tr = pd.DataFrame(trace_rows); tr.to_csv(out / "M1_trace_summary.csv", index=False)
    print(tr.groupby(["family", "cond"])[["mean_inj", "mean_rest"]].mean().round(1))

    # ---------------- M2: persistence across depth ----------------
    print("\nM2 persistence of the layer-8 edit across depth")
    pers = []
    for r in tqdm(sel.to_dict("records"), desc="M2"):
        msgs = V.scaffold(r["task"], r["html"]); ts = V.tool_spans(tok, msgs); sp = V.injected_span(tok, msgs, r["inj"])
        if sp is None: continue
        prompt = "".join(msgs) + "<|start|>assistant"
        for spec in (S_NONE, S_RND, S_RES):
            H, _ = capture_under(model, tok, prompt, spec, ts, dirs, list(MEAS))
            for L in MEAS:
                m = mismatch_per_token(H[L][sp[0]:sp[1]], dirs[L])
                pers.append({"attack_ix": r["attack_ix"], "family": r["family"], "cond": spec.name, "layer": L, "mismatch": float(m.mean()),
                             "mismatch_rel": float(m.mean() / max(1e-6, mismatch_per_token(H[L][ts[0][0]:ts[0][1]], dirs[L]).mean()))})
    pdf_ = pd.DataFrame(pers); pdf_.to_csv(out / "M2_persistence.csv", index=False)
    g = pdf_.groupby(["cond", "layer"]).mismatch.agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3))
    for spec, col in [(S_NONE, C["none"]), (S_RND, C["rand"]), (S_RES, C["restore"])]:
        d = g[g.cond == spec.name]; ax.errorbar(d.layer, d["mean"], yerr=d["sem"], fmt="o", color=col, label=spec.name, ms=5, capsize=2)
    ax.axvline(EL, color="#bbb", ls=":", lw=1); ax.text(EL + 0.2, ax.get_ylim()[1] * 0.95, "edit here", fontsize=7, color="#666", va="top")
    ax.set_xlabel("layer measured"); ax.set_ylabel("injected span: distance from\nhonest-tool role coords"); ax.legend(fontsize=7)
    ax.set_title(f"An edit at layer {EL} stays put; the matched random edit never touches role space", loc="left", fontsize=9)
    savefig(fig, out, "M2_persistence")
    print(g.pivot(index="layer", columns="cond", values="mean").round(1))

    # ---------------- M3: attention from planning tokens to the injected span ----------------
    if not args.skip_attn:
        print("\nM3 attention from the first planning-CoT tokens to the injection (short carriers)")
        # short carriers: truncate body text so attention matrices fit in memory
        short_pages = []
        for page in carriers:
            body = re.search(r"<p>(.*)</p>", page, re.S).group(1)[:args.attn_carrier_chars]
            short_pages.append(re.sub(r"<p>.*</p>", f"<p>{body}</p>", page, flags=re.S))
        att_rows = []
        for r in tqdm(sel.to_dict("records"), desc="M3"):
            page = short_pages[r["carrier_ix"]]; html = page.replace("</body>", r["inj"] + "</body>", 1)
            msgs = V.scaffold(r["task"], html); ts = V.tool_spans(tok, msgs); sp = V.injected_span(tok, msgs, r["inj"])
            if sp is None: continue
            # one greedy planning step with no intervention gives the CoT we then query from, identical across arms
            raw = V.gen_step(model, tok, "".join(msgs) + "<|start|>assistant", args.seed, args, greedy=True)
            p = V.parse_harmony(raw); cot = p["analysis"][0] if p["analysis"] else None
            if not cot: continue
            msgs_q = msgs + [V.h_analysis(cot)]
            q0 = V.ntok(tok, "".join(msgs)); q1 = V.ntok(tok, "".join(msgs_q))
            prompt = "".join(msgs_q)
            u0, u1 = V.ntok(tok, "".join(msgs[:2])), V.ntok(tok, "".join(msgs[:3]))     # user turn
            regions = {"injected": (sp[0], sp[1]), "rest_of_page": None, "user_turn": (u0, u1)}
            for spec in (S_NONE, S_RES):
                try:
                    _, attn = capture_under(model, tok, prompt, spec, ts, dirs, [], want_attn=True, q_range=(q0, q1))
                except Exception as e:
                    warnings.warn(f"attention capture failed ({e}); skipping M3"); attn = {}
                    break
                for L, A in attn.items():
                    tot = A.sum(-1).mean().item()
                    inj = A[:, sp[0]:sp[1]].sum(-1).mean().item()
                    page_m = A[:, ts[0][0]:ts[0][1]].sum(-1).mean().item() - inj
                    usr = A[:, u0:u1].sum(-1).mean().item()
                    att_rows.append({"attack_ix": r["attack_ix"], "family": r["family"], "cond": spec.name, "layer": L,
                                     "injected": inj / tot, "rest_of_page": page_m / tot, "user_turn": usr / tot,
                                     "injected_per_token": inj / max(1, sp[1] - sp[0]) / tot, "page_per_token": page_m / max(1, (ts[0][1] - ts[0][0]) - (sp[1] - sp[0])) / tot})
        if att_rows:
            ad = pd.DataFrame(att_rows); ad.to_csv(out / "M3_attention.csv", index=False)
            g = ad.groupby(["cond", "layer"])[["injected_per_token", "page_per_token"]].mean().reset_index()
            fig, ax = plt.subplots(figsize=(4.8, 3))
            for spec, col in [(S_NONE, C["none"]), (S_RES, C["restore"])]:
                d = g[g.cond == spec.name]
                ax.plot(d.layer, d.injected_per_token / d.page_per_token, "o", color=col, ms=4, label=f"{spec.name}")
            ax.axhline(1, color="#999", ls="--", lw=.8); ax.axvline(EL, color="#bbb", ls=":", lw=1)
            ax.set_xlabel("layer"); ax.set_ylabel("attention to injected tokens\n÷ attention to other page tokens\n(from the model's planning CoT)")
            ax.set_title("Does the model keep looking at the injection after restoration?", loc="left", fontsize=9); ax.legend(fontsize=7)
            savefig(fig, out, "M3_attention_to_injection")
            print(g.pivot(index="layer", columns="cond", values="injected_per_token").round(4).head(24))

    # ---------------- M4: dose-response ----------------
    if args.run_dose:
        specs = [S_NONE] + [Spec3(f"restore_L{EL}_a{a}", (EL,), "restore", alpha=a) for a in (0.25, 0.5, 0.75, 1.0)]
        grid3(model, tok, sel, specs, dirs, args.samples, "M4_dose", store, args, 200000)
        d = store.frame(); d = d[d.experiment == "M4_dose"]
        g = d.groupby(["family", "cond"]).attack.mean().reset_index()
        g["alpha"] = g.cond.map(lambda c: 0.0 if c == "none" else float(c.split("_a")[-1]))
        fig, ax = plt.subplots(figsize=(4.2, 3))
        for fam, col in [("forgery", C["forgery"]), ("prefix", C["prefix"])]:
            s = g[g.family == fam].sort_values("alpha"); ax.plot(s.alpha, s.attack, "o", color=col, ms=6, label=fam)
        ax.set_xlabel(f"fraction of the way to honest-tool role coordinates (layer {EL})"); ax.set_ylabel("attack success rate"); ax.legend()
        ax.set_title("Dose-response of restoration", loc="left"); savefig(fig, out, "M4_dose_response")
        print(g.pivot(index="alpha", columns="family", values="attack").round(2))

    # ---------------- M5: finer layer sweep ----------------
    if args.run_sweep:
        specs = [S_NONE] + [Spec3(f"restore_L{L}", (L,), "restore") for L in SWEEP]
        grid3(model, tok, sel, specs, dirs, args.samples, "M5_sweep", store, args, 300000)
        d = store.frame()
        prev = d[d.experiment == "E1_layers"] if "E1_layers" in d.experiment.values else pd.DataFrame()
        cur = d[d.experiment == "M5_sweep"]
        rows = []
        for src in (prev, cur):
            if src.empty: continue
            base = src[src.cond == "none"].groupby("family").attack.mean()
            for (fam, cond), g in src[src.cond != "none"].groupby(["family", "cond"]):
                rows.append({"family": fam, "layer": int(cond.split("_L")[-1]), "asr_rel": g.attack.mean() / max(1e-6, base[fam])})
        sw = pd.DataFrame(rows).groupby(["family", "layer"]).asr_rel.mean().reset_index(); sw.to_csv(out / "M5_layer_sweep.csv", index=False)
        fig, ax = plt.subplots(figsize=(4.4, 3))
        for fam, col in [("forgery", C["forgery"]), ("prefix", C["prefix"])]:
            s = sw[sw.family == fam].sort_values("layer"); ax.plot(s.layer, s.asr_rel, "o", color=col, ms=6, label=fam)
        ax.axhline(1, color="#999", ls="--", lw=.8); ax.set_xlabel("layer restored (single layer)"); ax.set_ylabel("attack success\n(fraction of baseline)"); ax.legend()
        ax.set_title("Where the role coordinates are read", loc="left"); savefig(fig, out, "M5_layer_sweep")
        print(sw.pivot(index="layer", columns="family", values="asr_rel").round(2))

    # ---------------- M6: leave-one-coordinate-out ----------------
    if args.run_loo:
        specs = [S_NONE, S_RES] + [Spec3(f"restore_L{EL}_drop{k}", (EL,), "restore", dims=tuple(j for j in range(4) if j != k)) for k in range(4)]
        grid3(model, tok, sel, specs, dirs, args.samples, "M6_loo", store, args, 400000)
        d = store.frame(); d = d[d.experiment == "M6_loo"]
        g = d.groupby(["family", "cond"]).attack.mean().unstack("family").round(2); g.to_csv(out / "M6_leave_one_out.csv")
        fig, ax = plt.subplots(figsize=(4.8, 2.8)); g.plot.bar(ax=ax, color=[C["forgery"], C["prefix"]], edgecolor="k", lw=.5)
        ax.set_ylabel("attack success rate"); ax.set_xlabel(""); ax.set_title("Which role coordinates are necessary? (restore 3 of 4)", loc="left")
        savefig(fig, out, "M6_leave_one_out"); print(g)

    # ---------------- tables ----------------
    ds = attacks.groupby("family").agg(n_attacks=("attack_ix", "size"), n_templates=("type", "nunique")).reset_index()
    ds["source"] = ds.family.map({"forgery": "cot-forgery-agent-evals/prompts/injections.yaml (5 forged-CoT injections)",
                                  "prefix": "agent-injections: injection.yaml wrapped in 12 of 210 role-declaration templates"})
    ds["paper_ASR"] = ds.family.map({"forgery": "56-70% (gpt-oss-20b)", "prefix": "~26% avg; user-declaring templates highest"})
    (out / "table_dataset.md").write_text(ds.to_markdown(index=False))
    d = store.frame()
    lad = []
    for exp, cond, what in [("E0_specificity", "none", "no intervention"),
                            ("E0_specificity", "restore_random_subspace", "random 4-D subspace, own magnitude"),
                            ("E0_specificity", "restore_random_MM", "random 4-D subspace, per-token magnitude matched"),
                            ("E2_axis_validate", "clamp_system_L8", "single best role axis, layer 8"),
                            ("E1_layers", "restore_L16", "role subspace, layer 16 only"),
                            ("E1_layers", "restore_L12", "role subspace, layer 12 only"),
                            ("E1_layers", "restore_L8", "role subspace, layer 8 only"),
                            ("E0b_scope", "restore_injected_only", "role subspace, injected tokens only"),
                            ("E0_specificity", "restore_role", "role subspace, layers 8+12+16 (full)")]:
        s = d[(d.experiment == exp) & (d.cond == cond)]
        if len(s):
            lad.append({"condition": what, "forgery ASR": round(s[s.family == "forgery"].attack.mean(), 2),
                        "prefix ASR": round(s[s.family == "prefix"].attack.mean(), 2), "‖Δh‖": round(s.delta_norm_mean.mean(), 0), "broken": round(s.broken.mean(), 2)})
    if lad:
        (out / "table_condition_ladder.md").write_text(pd.DataFrame(lad).to_markdown(index=False))
    print("\ntables -> table_dataset.md, table_condition_ladder.md")
    print("done ->", out)


if __name__ == "__main__":
    main()
