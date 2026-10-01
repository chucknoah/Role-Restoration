#!/usr/bin/env bash
# run_all.sh — run the role-restoration pipeline on the SMALL Qwen, then the LARGE Qwen:
#   geometry (01_prepare_qwen.py)  ->  interventions (02_run_intervention_qwen.py)
# for each model, with IDENTICAL seed / attack set / conditions / sampling, so the two runs are
# directly comparable. All data lands under $RESULTS_ROOT (default: role_restoration_results).
#
# What "exact same attacks" means here and how it is enforced:
#   * same --seed  -> identical Wikipedia carriers + identical sampled prefix templates -> identical
#     attacks.parquet for both models (model-independent). We VERIFY this with a content hash.
#   * --test-set all -> the intervention grid runs on EVERY rendered attack for both models (the
#     default "successful" filter is model-dependent and would give the two models different sets).
#     Baseline rows for all attacks are saved, so the paper's "succeeded-at-baseline" filter can
#     still be applied post-hoc in analysis.
#
# Usage (from the folder that contains scripts/ and role_restoration/):
#   bash run_all.sh                 # both models, full grid
#   SKIP_SMALL=1 bash run_all.sh    # only the large model
#   SKIP_LARGE=1 bash run_all.sh    # only the small model
#   SMOKE=1 bash run_all.sh         # tiny condition set, 1 sample, greedy -- pipeline check only
#
# Every variable below can be overridden from the environment, e.g.
#   N_SAMPLES=5 RESULTS_ROOT=/workspace/role_restoration_results bash run_all.sh
set -euo pipefail

# ----------------------------------------------------------------------------- locate the package
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Two supported layouts:
#   nested:  ROOT/role_restoration/base_qwen.py  + ROOT/scripts/01_prepare_qwen.py   (as delivered)
#   flat:    ROOT/base_qwen.py                   + ROOT/01_prepare_qwen.py or ROOT/scripts/01_prepare_qwen.py
#            (ROOT itself is the package; it must be named role_restoration and contain __init__.py)
if   [[ -f "$ROOT/role_restoration/base_qwen.py" ]]; then PKG_PARENT="$ROOT"
elif [[ -f "$ROOT/base_qwen.py" ]]; then
  PKG_PARENT="$(dirname "$ROOT")"
  [[ "$(basename "$ROOT")" == "role_restoration" ]] || { echo "ERROR: flat layout requires this folder to be named role_restoration (it is $(basename "$ROOT"))"; exit 1; }
  [[ -f "$ROOT/__init__.py" ]] || { echo "ERROR: flat layout needs $ROOT/__init__.py"; exit 1; }
else
  echo "ERROR: cannot find base_qwen.py under $ROOT (nested: role_restoration/base_qwen.py; flat: base_qwen.py)"; exit 1
fi
for cand in "$ROOT/scripts" "$ROOT"; do
  if [[ -f "$cand/01_prepare_qwen.py" && -f "$cand/02_run_intervention_qwen.py" ]]; then SCRIPTS="$cand"; break; fi
done
[[ -n "${SCRIPTS:-}" ]] || { echo "ERROR: cannot find 01_prepare_qwen.py / 02_run_intervention_qwen.py under $ROOT or $ROOT/scripts"; exit 1; }
PREP="$SCRIPTS/01_prepare_qwen.py"; RUN="$SCRIPTS/02_run_intervention_qwen.py"
export PYTHONPATH="$PKG_PARENT${PYTHONPATH:+:$PYTHONPATH}"
echo "layout: package parent=$PKG_PARENT | scripts=$SCRIPTS"

# ----------------------------------------------------------------------------- config (override via env)
if [[ -d /workspace ]]; then DEFAULT_RESULTS=/workspace/role_restoration_results; else DEFAULT_RESULTS="$PWD/role_restoration_results"; fi
RESULTS_ROOT="${RESULTS_ROOT:-$DEFAULT_RESULTS}"
SEED="${SEED:-1234}"
LAYERS="${LAYERS:-4 8 12 16 20}"        # valid for both the 36-layer 4B and the 48-layer 30B
PROJ_LAYER="${PROJ_LAYER:-12}"
N_SAMPLES="${N_SAMPLES:-3}"             # stochastic attempts per (attack, condition); temp 1
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_STEPS="${MAX_STEPS:-3}"
THINK_BUDGET="${THINK_BUDGET:-1024}"    # max thinking tokens per step before force-close (Thinking-2507 reasons at length)
MAX_NEW="${MAX_NEW:-512}"               # answer/tool_call tokens after thinking
N_ROLE_TEXTS="${N_ROLE_TEXTS:-40}"
N_STYLE_PAIRS="${N_STYLE_PAIRS:-24}"
N_CARRIERS="${N_CARRIERS:-3}"
N_PREFIX_USER="${N_PREFIX_USER:-8}"
N_PREFIX_OTHER="${N_PREFIX_OTHER:-4}"
TEST_SET="${TEST_SET:-all}"             # all | successful | /path/to/test_set_attack_ix.json
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
SMALL_TAG="${SMALL_TAG:-qwen3-4b-thinking}"
LARGE_TAG="${LARGE_TAG:-qwen3-30b-a3b-thinking}"
SKIP_SMALL="${SKIP_SMALL:-0}"
SKIP_LARGE="${SKIP_LARGE:-0}"
SMOKE="${SMOKE:-0}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
[[ -f "$ROOT/../venv/bin/activate" ]] && source "$ROOT/../venv/bin/activate" || true
[[ -f /workspace/venv/bin/activate ]] && source /workspace/venv/bin/activate || true

mkdir -p "$RESULTS_ROOT/logs"
LOG="$RESULTS_ROOT/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== run_all.sh =="; echo "root=$ROOT"; echo "results=$RESULTS_ROOT"; echo "log=$LOG"
echo "seed=$SEED layers=[$LAYERS] n_samples=$N_SAMPLES temp=$TEMPERATURE think_budget=$THINK_BUDGET max_new=$MAX_NEW test_set=$TEST_SET smoke=$SMOKE"
python -c "import torch,transformers;print('torch',torch.__version__,'| transformers',transformers.__version__,'| cuda',torch.cuda.is_available())"

RUN_EXTRA=()
if [[ "$SMOKE" == "1" ]]; then RUN_EXTRA=(--smoke --n-samples 1 --greedy); fi

# ----------------------------------------------------------------------------- helpers
attack_hash () {   # content hash of the rendered attack set (ignores parquet metadata)
  python - "$1" <<'PY'
import sys, hashlib, pandas as pd
d = pd.read_parquet(sys.argv[1]).sort_values("attack_ix")
h = hashlib.sha256("\n".join(f"{r.attack_ix}|{r.family}|{r.type}|{r.carrier_ix}|{r.inj}" for r in d.itertuples()).encode()).hexdigest()
print(h)
PY
}

run_model () {     # $1 = model id, $2 = tag
  local MODEL="$1" TAG="$2" OUT="$RESULTS_ROOT/$2"
  mkdir -p "$OUT"
  echo; echo "################ $TAG  ($MODEL) ################"; echo "outdir=$OUT"

  echo; echo "---- [01] geometry / directions / probe / attacks ----"
  python "$PREP" --model-id "$MODEL" --outdir "$OUT" --seed "$SEED" \
      --layers $LAYERS --proj-layer "$PROJ_LAYER" \
      --n-role-texts "$N_ROLE_TEXTS" --n-style-pairs "$N_STYLE_PAIRS" \
      --n-carriers "$N_CARRIERS" --n-benign-ref 8 \
      --n-prefix-user "$N_PREFIX_USER" --n-prefix-other "$N_PREFIX_OTHER"

  # guard: the exact failure you hit before -- never start 02 without 01's outputs
  for f in directions.pt probe.pkl attacks.parquet geom.csv; do
    [[ -f "$OUT/$f" ]] || { echo "ERROR: 01 did not produce $OUT/$f -- check the [01] log above"; exit 2; }
  done
  echo "[01] OK: $(ls "$OUT" | tr '\n' ' ')"

  echo; echo "---- [02] interventions (test-set=$TEST_SET, n_samples=$N_SAMPLES, temp=$TEMPERATURE) ----"
  python "$RUN" --model-id "$MODEL" --outdir "$OUT" --seed "$SEED" \
      --n-samples "$N_SAMPLES" --temperature "$TEMPERATURE" --max-steps "$MAX_STEPS" \
      --think-budget "$THINK_BUDGET" --max-new "$MAX_NEW" \
      --test-set "$TEST_SET" "${RUN_EXTRA[@]}"

  for f in results.csv summary.csv runs.jsonl test_set_attack_ix.json; do
    [[ -f "$OUT/$f" ]] || { echo "ERROR: 02 did not produce $OUT/$f"; exit 3; }
  done
  echo "[02] OK"
}

# ----------------------------------------------------------------------------- run: small, then large
if [[ "$SKIP_SMALL" != "1" ]]; then run_model "$SMALL_MODEL" "$SMALL_TAG"; fi
if [[ "$SKIP_LARGE" != "1" ]]; then run_model "$LARGE_MODEL" "$LARGE_TAG"; fi

# ----------------------------------------------------------------------------- verify the two runs used the SAME attacks
if [[ "$SKIP_SMALL" != "1" && "$SKIP_LARGE" != "1" ]]; then
  echo; echo "---- verifying identical attack sets across models ----"
  HS=$(attack_hash "$RESULTS_ROOT/$SMALL_TAG/attacks.parquet")
  HL=$(attack_hash "$RESULTS_ROOT/$LARGE_TAG/attacks.parquet")
  echo "small attacks.parquet hash: $HS"; echo "large attacks.parquet hash: $HL"
  if [[ "$HS" != "$HL" ]]; then
    echo "ERROR: rendered attack sets differ between models (seed/config mismatch?) -- results are NOT comparable"; exit 4
  fi
  python - "$RESULTS_ROOT/$SMALL_TAG/test_set_attack_ix.json" "$RESULTS_ROOT/$LARGE_TAG/test_set_attack_ix.json" <<'PY'
import sys, json
a, b = (json.load(open(p))["attack_ix"] for p in sys.argv[1:3])
print("small test set:", len(a), "attacks | large test set:", len(b), "attacks")
if a != b: print("ERROR: intervention test sets differ -- set TEST_SET=all (or point both at one file)"); sys.exit(5)
print("OK: both models ran the intervention grid on the identical", len(a), "attacks")
PY
fi

# ----------------------------------------------------------------------------- headline numbers for analysis
echo; echo "---- headline numbers (also in each outdir: geom.csv, summary.csv, paired_*.csv) ----"
for TAG in "$SMALL_TAG" "$LARGE_TAG"; do
  OUT="$RESULTS_ROOT/$TAG"; [[ -f "$OUT/results.csv" ]] || continue
  python - "$OUT" "$TAG" <<'PY'
import sys, pandas as pd, json
out, tag = sys.argv[1], sys.argv[2]
print(f"\n=== {tag} ===")
g = pd.read_csv(f"{out}/geom.csv")
cols = ["layer","cos(tag, style)","style / tag effect along tag axis","style inside role subspace (frac of |v_style|^2)"]
print("geometry (overlap + cosine, per layer):"); print(g[[c for c in cols if c in g]].round(3).to_string(index=False))
rp = json.load(open(f"{out}/role_probe_report.json"))
print(f"probe held-out acc: {rp['held_out_acc']:.3f} | cos(user_mean, tool_mean): " +
      ", ".join(f"L{k}={v:+.2f}" for k,v in rp["tool_user_cos_by_layer"].items()))
d = pd.read_csv(f"{out}/results.csv")
d = d[d.experiment.isin(["restoration","layers","ablation"])]
asr = d.groupby(["cond","family"]).attack.mean().unstack("family").mul(100).round(1)
asr["total"] = d.groupby("cond").attack.mean().mul(100).round(1)
print("intervention ASR % (attack success rate; want none high, restore_* low, random controls ~none):")
print(asr.to_string())
PY
done

# ----------------------------------------------------------------------------- top-level manifest
python - "$RESULTS_ROOT" "$SEED" "$N_SAMPLES" "$TEMPERATURE" "$TEST_SET" "$SMALL_MODEL" "$LARGE_MODEL" "$SMALL_TAG" "$LARGE_TAG" "$LAYERS" <<'PY'
import sys, json, os, datetime
root, seed, ns, temp, ts, sm, lm, st, lt, layers = sys.argv[1:11]
json.dump({"written": datetime.datetime.now().isoformat(), "seed": int(seed), "n_samples": int(ns), "temperature": float(temp),
           "test_set": ts, "layers": [int(x) for x in layers.split()],
           "models": {st: sm, lt: lm}, "outdirs": {st: os.path.join(root, st), lt: os.path.join(root, lt)},
           "note": "same seed -> identical attacks.parquet; test_set=all -> identical intervention set across models"},
          open(os.path.join(root, "run_all_manifest.json"), "w"), indent=1)
print("wrote", os.path.join(root, "run_all_manifest.json"))
PY

echo; echo "== done. everything is under $RESULTS_ROOT =="
echo "   $SMALL_TAG/  $LARGE_TAG/   each with: geom.csv token_projections.csv directions.pt (geometry: overlap + cosine)"
echo "                                  results.csv summary.csv paired_*.csv runs.jsonl transcripts/ (intervention ASR)"
echo "   plot later (no GPU): python $SCRIPTS/make_figures.py --results-dir $RESULTS_ROOT/<tag> --outdir $RESULTS_ROOT/<tag>/figures"
