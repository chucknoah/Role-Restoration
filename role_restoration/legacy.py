"""legacy.py — load results from the original Colab / RunPod runs into the package's results format."""
from __future__ import annotations
import pickle
import pandas as pd

def load_colab_pickle(path: str, per_family=12, experiment="ablation") -> pd.DataFrame:
    """step2b_results.pkl -> results frame (experiment=`experiment`), restricted to the top-`per_family` baseline-successful attacks."""
    d = pd.DataFrame(pickle.load(open(path, "rb")))
    d = d.rename(columns={"cot_mentions_exfil": "cot_mentions"})
    b = d[d.cond == "none"].groupby(["family", "attack_ix"]).attack.mean().reset_index(name="b")
    keep = b[b.b > 0].sort_values("b", ascending=False).groupby("family").head(per_family).attack_ix
    d = d[d.attack_ix.isin(keep)].copy(); d["experiment"] = experiment
    if "delta_norm_mean" not in d: d["delta_norm_mean"] = float("nan")
    return d

def load_runpod_csv(path: str) -> pd.DataFrame:
    """all_results.csv from the follow-up run (experiments E0_specificity, E0b_scope, E1_layers, ...)."""
    return pd.read_csv(path)
