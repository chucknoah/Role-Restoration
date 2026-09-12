"""
geometry.py — the role-leakage experiment (Section: Role Leakage).

Builds two difference-of-means directions per layer and measures their geometry:
    v_tag   = E[h | text in CoT tag]      - E[h | same text in User tag]      (content held fixed)
    v_style = E[h | CoT-style text]       - E[h | plain rewrite, same tag]     (tag held fixed)
and reports cosine, split-half reliabilities, the "% of tag effect" overlap, and the fraction of v_style
inside the role subspace. Also writes per-token projections at one layer for the geometry figure.

Outputs (in outdir): style_pairs.json, geom.csv, token_projections.csv, style_directions.pt
"""
from __future__ import annotations
import json, os
from typing import Dict, List, Sequence
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from . import base as B

DESTYLE_SYS = ("Reasoning: low\nRewrite the user's text in plain, ordinary prose. Keep every piece of content and meaning "
               "identical. Use full natural sentences, no terse note-like fragments, no 'we need to' phrasing. Output only the rewritten text.")

QUESTIONS = [
    "Should I take an umbrella if there is a 40% chance of rain?", "Is it cheaper to buy or rent a home for 3 years?",
    "How many buses do I need for 130 students if each bus holds 40?", "What's a good weekly plan to learn basic Spanish?",
    "Should I refinance a 6% mortgage to 5.2% with $3000 in fees?", "Which is heavier, a litre of water or a litre of oil?",
    "How should I split a $180 restaurant bill among 7 people with one paying double?", "Is it worth buying an annual gym pass at $400 if I go twice a week?",
    "What time should I leave for a 9am flight if the airport is 45 minutes away?", "How many 2.5kg bags of rice do I need for 30 people over a week?",
    "Should I charge my phone overnight?", "What's the fastest way to defrost chicken safely?",
    "If a train leaves at 3:10 and takes 2h50m, when does it arrive?", "Is a 15% tip on $64 more or less than $10?",
    "How do I decide between two job offers with different salaries and commutes?", "What's the best order to visit three shops to minimise walking?",
    "How much paint do I need for a 4m x 3m wall with two coats?", "Should I water my plants every day in summer?",
    "What's a sensible monthly budget on a $3200 income?", "How many hours of sleep should a 30-year-old get?",
    "Is it better to pay off a credit card or save an emergency fund first?", "How do I convert 68F to Celsius?",
    "What's the best way to pack a suitcase for a 5-day trip?", "Should I take the stairs or the lift to the 4th floor?",
]

# ----------------------------------------------------------------------------- style pairs

def make_style_pairs(model, tok, seed: int, n_pairs=24, max_new=768, max_txt_tokens=256, min_words=15) -> List[dict]:
    """The model's own chain-of-thought (CoT style) and a plain-prose rewrite of it with the same content."""
    trunc = lambda t: tok.decode(tok(t, add_special_tokens=False, truncation=True, max_length=max_txt_tokens).input_ids)
    pairs = []
    for i, q in enumerate(tqdm(QUESTIONS[:n_pairs], desc="style pairs")):
        styled, _ = B.gen_chat(model, tok, [{"role": "user", "content": q}], seed + i, max_new=max_new)
        if len(styled.split()) < min_words: continue
        _, plain = B.gen_chat(model, tok, [{"role": "system", "content": DESTYLE_SYS}, {"role": "user", "content": styled}], seed + i, max_new=max_new)
        if len(plain.split()) < min_words: continue
        pairs.append({"pair_ix": len(pairs), "question": q, "styled": trunc(styled), "destyled": trunc(plain)})
    return pairs

# ----------------------------------------------------------------------------- directions

def _mean_content(model, tok, prompts: Sequence[str], layers, max_len, keep_tokens=False):
    """Mean over content tokens of each prompt; optionally keep per-token activations."""
    sums = {L: torch.zeros(1) for L in layers}; n = 0; toks = {L: [] for L in layers}
    for st in range(0, len(prompts), 8):
        store, ids = B.capture_batch(model, tok, prompts[st:st + 8], layers, max_len)
        for bi in range(ids.shape[0]):
            s, e = B.content_slice_from_ids(ids[bi], tok)
            for L in layers:
                H = store[L][bi, s:e]
                sums[L] = sums[L] + H.sum(0); 
                if keep_tokens: toks[L].append(H)
            n += (e - s)
    means = {L: sums[L] / n for L in layers}
    return (means, {L: torch.cat(toks[L]) for L in layers}) if keep_tokens else means

def style_directions(model, tok, pairs: List[dict], layers: Sequence[int], max_len=300):
    """v_style per layer (averaged over both tags), split-half versions, and the 2x2 per-token activations."""
    tags = ["user", "cot"]
    cond = {(t, s): [B.render_single_role(t, p[s]) for p in pairs] for t in tags for s in ("styled", "destyled")}
    half = {(t, s, h): [B.render_single_role(t, p[s]) for p in pairs if p["pair_ix"] % 2 == h] for t in tags for s in ("styled", "destyled") for h in (0, 1)}
    M, T = {}, {}
    for k, ps in cond.items(): M[k], T[k] = _mean_content(model, tok, ps, layers, max_len, keep_tokens=True)
    Mh = {k: _mean_content(model, tok, ps, layers, max_len) for k, ps in half.items()}
    out = {}
    for L in layers:
        v = 0.5 * ((M[("user", "styled")][L] - M[("user", "destyled")][L]) + (M[("cot", "styled")][L] - M[("cot", "destyled")][L]))
        vA = 0.5 * ((Mh[("user", "styled", 0)][L] - Mh[("user", "destyled", 0)][L]) + (Mh[("cot", "styled", 0)][L] - Mh[("cot", "destyled", 0)][L]))
        vB = 0.5 * ((Mh[("user", "styled", 1)][L] - Mh[("user", "destyled", 1)][L]) + (Mh[("cot", "styled", 1)][L] - Mh[("cot", "destyled", 1)][L]))
        v_u = M[("user", "styled")][L] - M[("user", "destyled")][L]; v_c = M[("cot", "styled")][L] - M[("cot", "destyled")][L]
        out[L] = {"v_style": v, "v_style_A": vA, "v_style_B": vB, "v_style_user": v_u, "v_style_cot": v_c,
                  "tokens": {k: T[k][L] for k in cond}, "means": {k: M[k][L] for k in cond}}
    return out

def tag_directions(X: Dict[int, torch.Tensor], y: np.ndarray, groups: np.ndarray, layers: Sequence[int]):
    """v_tag = mean(cot) - mean(user) on identical text, plus split-half versions (from estimate_role_means tokens)."""
    out = {}
    for L in layers:
        H = X[L]; cot, usr = (y == "cot"), (y == "user"); ev = (groups % 2 == 0)
        v = H[cot].mean(0) - H[usr].mean(0)
        vA = H[cot & ev].mean(0) - H[usr & ev].mean(0); vB = H[cot & ~ev].mean(0) - H[usr & ~ev].mean(0)
        out[L] = {"v_tag": v, "v_tag_A": vA, "v_tag_B": vB}
    return out

# ----------------------------------------------------------------------------- metrics

def cos(a, b): return float(B.unit(a) @ B.unit(b))

def geometry_table(tagd, styd, Q: Dict[int, torch.Tensor], probe_axes: Dict[int, torch.Tensor] | None, layers) -> pd.DataFrame:
    rows = []
    D = tagd[layers[0]]["v_tag"].shape[0]
    rand_cos = float(torch.stack([torch.abs(B.unit(torch.randn(D)) @ B.unit(torch.randn(D))) for _ in range(200)]).mean())
    for L in layers:
        vt, vs = tagd[L]["v_tag"], styd[L]["v_style"]; ut = B.unit(vt)
        along = float(vs @ ut); on = Q[L] @ (Q[L].T @ vs)
        row = {"layer": L, "cos(tag, style)": cos(vt, vs), "cos(tag_A, tag_B)": cos(tagd[L]["v_tag_A"], tagd[L]["v_tag_B"]),
               "cos(style_A, style_B)": cos(styd[L]["v_style_A"], styd[L]["v_style_B"]),
               "cos(style|user, style|cot)": cos(styd[L]["v_style_user"], styd[L]["v_style_cot"]),
               "style / tag effect along tag axis": along / float(vt.norm()), "style gap off tag axis": 1 - along ** 2 / float(vs.norm() ** 2),
               "style inside role subspace (frac of |v_style|^2)": float(on.norm() ** 2 / vs.norm() ** 2),
               "|v_tag|": float(vt.norm()), "|v_style|": float(vs.norm()), "random |cos| baseline": rand_cos}
        if probe_axes and L in probe_axes:
            row["cos(probe, tag)"] = cos(probe_axes[L], vt); row["cos(probe, style)"] = cos(probe_axes[L], vs)
        rows.append(row)
    return pd.DataFrame(rows)

def token_projections(styd, tagd, L: int) -> pd.DataFrame:
    """Per-token projections of the 2x2 onto unit v_tag and unit v_style at layer L (for the geometry figure)."""
    ut, us = B.unit(tagd[L]["v_tag"]), B.unit(styd[L]["v_style"]); rows = []
    for (tag, style), H in styd[L]["tokens"].items():
        rows.append(pd.DataFrame({"tag": tag, "style": style, "p_tag": (H @ ut).numpy(), "p_style": (H @ us).numpy()}))
    return pd.concat(rows, ignore_index=True)

# ----------------------------------------------------------------------------- driver

def run_geometry(model, tok, outdir: str, seed: int, layers=(4, 8, 12, 16, 20), role_texts=None, n_pairs=24, proj_layer=12, pairs=None):
    """Full geometry experiment. Returns dict with everything downstream needs.
    pairs: optional list of {"pair_ix","styled","destyled"} (e.g. pairs.json from an earlier run) to skip generation."""
    os.makedirs(outdir, exist_ok=True)
    if role_texts is None:
        _, _, role_texts = B.load_wikipedia_material(seed, n_role_texts=40)
    means, X, y, g = B.estimate_role_means(model, tok, role_texts, layers, return_tokens=True)
    Q = {L: B.role_basis_from_means(means[L]) for L in layers}
    tagd = tag_directions(X, y, g, layers)
    if pairs is None:
        pairs = make_style_pairs(model, tok, seed, n_pairs=n_pairs)
    else:
        pairs = [{"pair_ix": i, "styled": p["styled"], "destyled": p["destyled"], "question": p.get("question", "")} for i, p in enumerate(pairs)]
    json.dump(pairs, open(os.path.join(outdir, "style_pairs.json"), "w"), indent=1)
    print(f"{len(pairs)} usable style pairs")
    styd = style_directions(model, tok, pairs, layers)
    # probe axis (cot vs user) at the projection layer, for the "probe is blind to style" claim
    clf, acc = B.train_role_probe(X[proj_layer], y, g, seed)
    W = clf.named_steps["logisticregression"].coef_; cls = list(clf.classes_)
    probe_axes = {proj_layer: torch.tensor(W[cls.index("cot")] - W[cls.index("user")]).float()}
    geom = geometry_table(tagd, styd, Q, probe_axes, layers); geom.to_csv(os.path.join(outdir, "geom.csv"), index=False)
    tp = token_projections(styd, tagd, proj_layer); tp.to_csv(os.path.join(outdir, "token_projections.csv"), index=False)
    torch.save({"layers": layers, "role_means": means, "Q": Q, "v_tag": {L: tagd[L]["v_tag"] for L in layers},
                "v_style": {L: styd[L]["v_style"] for L in layers}, "probe_acc": acc}, os.path.join(outdir, "style_directions.pt"))
    print(geom.round(3).to_string(index=False))
    return {"means": means, "Q": Q, "tagd": tagd, "styd": styd, "geom": geom, "token_projections": tp, "probe": clf, "probe_acc": acc, "X": X, "y": y, "groups": g}
