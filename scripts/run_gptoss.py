#!/usr/bin/env python3
"""
run_gptoss.py — the gpt-oss-20b experiments, runnable one stage at a time.

Every stage reads what earlier stages saved in --outdir and writes its own outputs there, so you can
run any subset, in any session, without recomputing what's already on disk.

STAGES (in order)                                              needs GPU   reads                         writes
  data           Wikipedia pages + the paper's attack strings      no       —                             data.json, attacks.parquet
  geometry       role means, Q, v_tag, v_style, role probe         yes      data.json                     geom.csv, token_projections.csv, style_pairs.json,
                                                                                                          style_directions.pt, probe.pkl, style_inside_frac.json
  reference      honest-tool reference mu_tool                     yes      data.json, style_directions.pt directions.pt
  baseline       every attack with no intervention; pick test set  yes      attacks.parquet, directions.pt runs.jsonl, test_set_attack_ix.json
  interventions  ablations, role restoration, controls, layer sweep yes      + test set                    runs.jsonl
  extras         benign utility, mediation, Figure-1 payloads      yes      + probe.pkl                   benign_utility.csv, mediation.csv, figure1/*.json
  stats          results table, ASR summary, paired bootstrap CIs  no       runs.jsonl                    results.csv, summary.csv, paired_*.csv

EXAMPLES
  python scripts/run_gptoss.py                                   # everything
  python scripts/run_gptoss.py --stages data geometry            # just probes + geometry (Figure 2 data)
  python scripts/run_gptoss.py --stages reference baseline       # tool reference + baseline ASR
  python scripts/run_gptoss.py --stages interventions            # only the interventions (needs the stages above done once)
  python scripts/run_gptoss.py --stages interventions --conditions restore_L8 restore_random_MM
  python scripts/run_gptoss.py --stages stats                    # recompute tables/CIs from runs.jsonl, no GPU
  python scripts/run_gptoss.py --stages geometry --style-pairs results_gptoss/style_pairs.json   # skip style-pair generation

Episode-running stages (baseline, interventions) are resumable: runs.jsonl is append-only and keyed on
(experiment, attack, condition, sample), so re-running skips finished episodes.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, ".."))
import pandas as pd

STAGES = ["data", "geometry", "reference", "baseline", "interventions", "extras", "stats"]
NEEDS_MODEL = {"geometry", "reference", "baseline", "interventions", "extras"}
NEEDS = {  # files each stage requires from earlier stages
    "geometry": ["data.json"],
    "reference": ["data.json", "style_directions.pt"],
    "baseline": ["attacks.parquet", "directions.pt"],
    "interventions": ["attacks.parquet", "directions.pt"],
    "extras": ["data.json", "attacks.parquet", "directions.pt", "probe.pkl"],
    "stats": ["runs.jsonl"],
}
MADE_BY = {"data.json": "data", "attacks.parquet": "data", "style_directions.pt": "geometry", "probe.pkl": "geometry",
           "directions.pt": "reference", "runs.jsonl": "baseline", "test_set_attack_ix.json": "baseline"}


def parse_args():
    p = argparse.ArgumentParser(description="gpt-oss role-restoration experiments, stage by stage.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("EXAMPLES")[1])
    p.add_argument("--stages", nargs="+", default=["all"], help=f"any of {STAGES}, or 'all'")
    p.add_argument("--outdir", default="results_gptoss")
    p.add_argument("--model-id", default="openai/gpt-oss-20b")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20], help="layers to build directions at")
    p.add_argument("--proj-layer", type=int, default=12, help="layer the role probe is trained/read at")
    p.add_argument("--n-role-texts", type=int, default=40)
    p.add_argument("--n-style-pairs", type=int, default=24)
    p.add_argument("--style-pairs", default=None, help="reuse a style_pairs.json (skips generating them)")
    p.add_argument("--n-samples", type=int, default=3, help="attempts per (attack, condition) at temperature 1")
    p.add_argument("--per-family", type=int, default=8, help="baseline-successful attacks kept per family")
    p.add_argument("--test-set", default="successful", help="successful | all | path/to/test_set_attack_ix.json")
    p.add_argument("--ablate-layer", type=int, default=8)
    p.add_argument("--restore-layers", type=int, nargs="+", default=[8, 12, 16], help="multi-layer restoration")
    p.add_argument("--sweep-layers", type=int, nargs="+", default=[8, 12, 16], help="single-layer restorations (Figure 5)")
    p.add_argument("--conditions", nargs="+", default=None, help="only run these intervention conditions (by name)")
    p.add_argument("--max-new", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    a = p.parse_args()
    a.stages = STAGES if "all" in a.stages else a.stages
    bad = [s for s in a.stages if s not in STAGES]
    if bad: p.error(f"unknown stage(s) {bad}; choose from {STAGES} or 'all'")
    a.stages = [s for s in STAGES if s in a.stages]          # always run in pipeline order
    return a


class Ctx:
    """Shared state; the model and directions are loaded lazily, only if a stage needs them."""
    def __init__(self, a):
        self.a, self.out = a, a.outdir
        self._model = self._tok = self._dirs = None
    def path(self, f): return os.path.join(self.out, f)
    def has(self, f): return os.path.exists(self.path(f))
    def require(self, stage):
        missing = [f for f in NEEDS.get(stage, []) if not self.has(f)]
        if missing:
            # full chain of prerequisite stages whose outputs are absent, in pipeline order
            todo, frontier = set(), {MADE_BY[f] for f in missing if f in MADE_BY}
            while frontier:
                s = frontier.pop(); todo.add(s)
                frontier |= {MADE_BY[f] for f in NEEDS.get(s, []) if not self.has(f) and f in MADE_BY} - todo
            how = [s for s in STAGES if s in todo]
            sys.exit(f"[{stage}] missing {missing} in {self.out} -> run --stages {' '.join(how)} first")
    @property
    def model(self):
        if self._model is None:
            from role_restoration import base as B
            print(f"[model] loading {self.a.model_id} ...")
            self._model, self._tok = B.load_model_and_tokenizer(self.a.model_id, cache_dir=self.a.cache_dir)
        return self._model
    @property
    def tok(self):
        _ = self.model; return self._tok
    def data(self): return json.load(open(self.path("data.json")))
    def attacks(self): return pd.read_parquet(self.path("attacks.parquet"))
    def dirs(self):
        if self._dirs is None:
            import torch
            from role_restoration import intervention as I
            d = torch.load(self.path("directions.pt"), weights_only=False)
            self.layers = tuple(d["layers"])
            self._dirs = I.build_dirs(d["role_means"], d["mu_tool"], self.layers, self.a.seed, v_style=d["v_style"])
        return self._dirs
    def store(self):
        from role_restoration import intervention as I
        return I.Store(self.path("runs.jsonl"))
    def test_set(self, attacks):
        ts = self.a.test_set
        if ts == "all": ix = attacks.attack_ix.astype(int).tolist()
        elif ts == "successful":
            if not self.has("test_set_attack_ix.json"): sys.exit("no test_set_attack_ix.json -> run --stages baseline first (or --test-set all)")
            ix = json.load(open(self.path("test_set_attack_ix.json")))
        else: ix = json.load(open(ts))
        return attacks[attacks.attack_ix.isin(ix)].reset_index(drop=True)
    def gen_kw(self): return dict(max_new=self.a.max_new, temperature=self.a.temperature)


def specs(a):
    from role_restoration.intervention import Spec
    none = Spec("none", (), "none")
    ablate = [Spec(n, (a.ablate_layer,), "ablate", direction=n) for n in ("style", "style_on", "style_perp", "random")]
    restore = [Spec("restore_role", tuple(a.restore_layers), "restore"),
               Spec("restore_random_subspace", tuple(a.restore_layers), "restore_rand"),
               Spec("restore_random_MM", tuple(a.restore_layers), "restore_rand_mm")]
    sweep = [Spec(f"restore_L{L}", (L,), "restore") for L in a.sweep_layers]
    if a.conditions:
        keep = set(a.conditions); ablate = [s for s in ablate if s.name in keep]
        restore = [s for s in restore if s.name in keep]; sweep = [s for s in sweep if s.name in keep]
    return none, ablate, restore, sweep


# ============================================================================== stages

def stage_data(c):
    from role_restoration import base as B
    a = c.a; B.set_seed(a.seed)
    carriers, benign, role_texts = B.load_wikipedia_material(a.seed, n_carriers=3, n_benign_ref=8, n_role_texts=a.n_role_texts)
    json.dump({"carriers": carriers, "benign": benign, "role_texts": role_texts}, open(c.path("data.json"), "w"))
    att = B.load_published_attacks(carriers, a.seed, n_prefix_user=8, n_prefix_other=4)
    att.to_parquet(c.path("attacks.parquet"))
    print(f"[data] {len(att)} attacks ({(att.family=='forgery').sum()} forgery, {(att.family=='prefix').sum()} prefix), "
          f"{len(benign)} clean pages, {len(role_texts)} probe texts")

def stage_geometry(c):
    from role_restoration import base as B, geometry as G
    a = c.a; B.set_seed(a.seed); layers = tuple(a.layers)
    pairs = json.load(open(a.style_pairs)) if a.style_pairs else None
    geo = G.run_geometry(c.model, c.tok, c.out, a.seed, layers=layers, role_texts=c.data()["role_texts"],
                         n_pairs=a.n_style_pairs, proj_layer=a.proj_layer, pairs=pairs)
    pickle.dump({"probe": geo["probe"], "acc": geo["probe_acc"], "proj_layer": a.proj_layer}, open(c.path("probe.pkl"), "wb"))
    frac = {}
    for L in layers:
        Q, v = geo["Q"][L], geo["styd"][L]["v_style"].float()
        frac[str(L)] = float((Q @ (Q.T @ v)).norm() ** 2 / v.norm() ** 2)
    json.dump(frac, open(c.path("style_inside_frac.json"), "w"), indent=1)
    print(f"[geometry] probe held-out accuracy {geo['probe_acc']:.3f}; geom.csv written")

def stage_reference(c):
    import torch
    from role_restoration import base as B
    a = c.a; sd = torch.load(c.path("style_directions.pt"), weights_only=False); layers = tuple(sd["layers"])
    mu = B.benign_tool_reference(c.model, c.tok, c.data()["benign"], layers)
    torch.save({"layers": list(layers), "role_means": sd["role_means"], "v_style": sd["v_style"], "mu_tool": mu}, c.path("directions.pt"))
    print(f"[reference] mu_tool at layers {list(layers)} -> directions.pt")

def stage_baseline(c):
    from role_restoration import intervention as I
    a = c.a; none, *_ = specs(a); att = c.attacks(); st = c.store()
    I.run_grid(c.model, c.tok, att, [none], c.dirs(), a.n_samples, "restoration", st, a.seed, **c.gen_kw())
    keep = I.select_successful(st, "restoration", a.per_family)
    json.dump(sorted(int(x) for x in keep), open(c.path("test_set_attack_ix.json"), "w"))
    d = st.frame(); b = d[(d.experiment == "restoration") & (d.cond == "none")]
    print(f"[baseline] ASR {100*b.attack.mean():.0f}% over {len(b)} episodes; test set = {len(keep)} attacks "
          f"(top {a.per_family}/family that succeeded at least once)")

def stage_interventions(c):
    from role_restoration import intervention as I
    a = c.a; none, ablate, restore, sweep = specs(a); test = c.test_set(c.attacks()); st = c.store(); kw = c.gen_kw()
    for L in set(a.restore_layers) | set(a.sweep_layers) | {a.ablate_layer}:
        if L not in c.dirs(): sys.exit(f"layer {L} has no directions; rebuild geometry/reference with --layers including {L}")
    print(f"[interventions] {len(test)} attacks x {a.n_samples} samples; conditions: "
          f"{[s.name for s in ablate + restore + sweep]}")
    if restore: I.run_grid(c.model, c.tok, test, restore, c.dirs(), a.n_samples, "restoration", st, a.seed, **kw)
    if ablate:  I.run_grid(c.model, c.tok, test, [none] + ablate, c.dirs(), a.n_samples, "ablation", st, a.seed, **kw)
    if sweep:   I.run_grid(c.model, c.tok, test, [none] + sweep, c.dirs(), a.n_samples, "layers", st, a.seed, **kw)
    # 'none' on the test set for the restoration experiment already exists from the baseline stage when test-set=successful;
    # for test-set=all/path make sure it's there too
    I.run_grid(c.model, c.tok, test, [none], c.dirs(), a.n_samples, "restoration", st, a.seed, **kw)

def stage_extras(c):
    from role_restoration import intervention as I
    from role_restoration.intervention import Spec
    a = c.a; none, *_ = specs(a); test = c.test_set(c.attacks()); dirs = c.dirs()
    probe = pickle.load(open(c.path("probe.pkl"), "rb"))["probe"]
    rr, r8 = Spec("restore_role", tuple(a.restore_layers), "restore"), Spec("restore_L8", (8,), "restore")
    I.benign_utility(c.model, c.tok, c.data()["benign"][:6], [none, rr, r8], dirs, a.seed, max_new=a.max_new) \
     .to_csv(c.path("benign_utility.csv"), index=False)
    I.mediation(c.model, c.tok, test, [none, rr, r8], dirs, measure_layers=[L for L in (8, 12, 16) if L in dirs]) \
     .to_csv(c.path("mediation.csv"), index=False)
    os.makedirs(c.path("figure1"), exist_ok=True)
    for fam in ("forgery", "prefix"):
        sub = test[test.family == fam]
        if not len(sub): continue
        d = I.textfig_data(c.model, c.tok, sub.iloc[0].to_dict(), dirs, probe, edit_layer=8, probe_layer=a.proj_layer, seed=a.seed)
        if d: json.dump(d, open(c.path(f"figure1/figure1_{d['family_label']}_{d['attack_ix']}.json"), "w"), indent=1)
    print("[extras] benign_utility.csv, mediation.csv, figure1/*.json written")

def stage_stats(c):
    from role_restoration import intervention as I
    a = c.a; df = c.store().frame(); df.to_csv(c.path("results.csv"), index=False)
    pd.concat([I.summarize(df, e).assign(experiment=e) for e in df.experiment.unique()]).to_csv(c.path("summary.csv"))
    rep = I.paired_report(df, "restoration", seed=a.seed)
    for fam in ("forgery", "prefix"):
        (rep[rep.family == fam] if len(rep) else rep).to_csv(c.path(f"paired_{fam}.csv"), index=False)
    print(f"[stats] {len(df)} episodes -> results.csv, summary.csv, paired_*.csv\n")
    print(df.groupby(["experiment", "cond"]).attack.agg(ASR=lambda s: round(100 * s.mean(), 1), n="size").to_string())

RUN = {"data": stage_data, "geometry": stage_geometry, "reference": stage_reference, "baseline": stage_baseline,
       "interventions": stage_interventions, "extras": stage_extras, "stats": stage_stats}


def main():
    a = parse_args(); os.makedirs(a.outdir, exist_ok=True); c = Ctx(a)
    print(f"stages: {a.stages}   outdir: {a.outdir}" + ("   (no GPU needed)" if not NEEDS_MODEL & set(a.stages) else ""))
    produced = set()
    for s in a.stages:
        # requirements may be satisfied by an earlier stage in this same invocation
        missing = [f for f in NEEDS.get(s, []) if not c.has(f) and MADE_BY.get(f) not in produced]
        if missing: c.require(s)
        print(f"\n===== {s} ====="); RUN[s](c); produced.add(s)
    print(f"\ndone. figures (no GPU): python scripts/make_figures.py --results-dir {a.outdir} --outdir {a.outdir}/figures")


if __name__ == "__main__":
    main()