# ================================================================================================
# STAGE     1b. Geometry only (fixes base_seq_ix KeyError); reuses pairs/tok_df/hs from step 1, no model calls
# RAN ON    Colab
# NEEDS     step 1 state
# PRODUCES  geom (table), proj_tables  ->  geom.csv via 08_save
# FIGURE    Figure 2
# NOTE      code unchanged from the run that produced the paper's numbers; only this header was added
# ================================================================================================
"""
STEP 1 (continued) — geometry only. Reuses `pairs`, `tok_df`, `hs` from the previous cell; no generation, no forward pass.
Fix: base_seq_ix comes from input_df (prompt_ix -> base_seq_ix), not probe_sample_df.
"""
import torch, numpy as np, pandas as pd, cupy, plotly.express as px

# ---- recover base_seq_ix for split-half reliability of v_tag ----
if 'base_seq_ix' not in probe_sample_df.columns:
    if 'input_df' in globals():
        pdf = probe_sample_df.merge(input_df[['prompt_ix', 'base_seq_ix']], on='prompt_ix', how='left')
    else:
        pdf = probe_sample_df.assign(base_seq_ix=lambda d: d['prompt_ix'] // 5)   # input_df was built base_seq x 5 roles, in order
else:
    pdf = probe_sample_df

_unit = lambda v: v / (v.norm() + 1e-8)
_cos  = lambda a, b: float(_unit(a) @ _unit(b))
_mean = lambda H, idx: H[idx].float().mean(0)

n_pairs = tok_df['pair_ix'].nunique()
print(f"{n_pairs} pairs, {len(tok_df)} content tokens across {tok_df['tag'].nunique()}x{tok_df['style'].nunique()} conditions")
display(tok_df.groupby(['tag', 'style']).size().rename('tokens').to_frame().T)

D = hs[LAYERS_TO_PROBE[0]].shape[1]
rand_cos = float(torch.stack([torch.abs(_unit(torch.randn(D)) @ _unit(torch.randn(D))) for _ in range(200)]).mean())

rows, proj_tables = [], {}
for L in LAYERS_TO_PROBE:
    Hp, H = all_probe_hs[L], hs[L]
    sel_p = lambda m, r: pdf[m & (pdf['role'] == r)]['sample_ix'].tolist()
    sel_t = lambda m, s: tok_df[m & (tok_df['style'] == s)]['sample_ix'].tolist()
    allp, allt = pd.Series(True, index=pdf.index), pd.Series(True, index=tok_df.index)
    evP, evT = (pdf['base_seq_ix'] % 2 == 0), (tok_df['pair_ix'] % 2 == 0)

    # v_tag: same content, cot tag minus user tag (C4/Dolma probe data)
    v_tag   = _mean(Hp, sel_p(allp, 'cot')) - _mean(Hp, sel_p(allp, 'user'))
    v_tag_A = _mean(Hp, sel_p(evP,  'cot')) - _mean(Hp, sel_p(evP,  'user'))
    v_tag_B = _mean(Hp, sel_p(~evP, 'cot')) - _mean(Hp, sel_p(~evP, 'user'))
    v_tag_gen = (_mean(H, tok_df[tok_df['tag'] == 'cot']['sample_ix'].tolist())
                 - _mean(H, tok_df[tok_df['tag'] == 'user']['sample_ix'].tolist()))

    # v_style: same tag, styled minus destyled (generated pairs)
    v_style   = _mean(H, sel_t(allt, 'styled')) - _mean(H, sel_t(allt, 'destyled'))
    v_style_A = _mean(H, sel_t(evT,  'styled')) - _mean(H, sel_t(evT,  'destyled'))
    v_style_B = _mean(H, sel_t(~evT, 'styled')) - _mean(H, sel_t(~evT, 'destyled'))
    v_style_u = _mean(H, sel_t(tok_df['tag'] == 'user', 'styled')) - _mean(H, sel_t(tok_df['tag'] == 'user', 'destyled'))
    v_style_c = _mean(H, sel_t(tok_df['tag'] == 'cot',  'styled')) - _mean(H, sel_t(tok_df['tag'] == 'cot',  'destyled'))

    # probe's own cot-vs-user axis
    pr = [p for p in all_probes if p['layer_ix'] == L][0]
    W = cupy.asnumpy(pr['probe'].named_steps['clf'].coef_)
    w_probe = torch.tensor(W[0] if W.shape[0] == 1 else W[pr['roles_map']['cot']] - W[pr['roles_map']['user']]).float()

    u_tag, u_style, u_probe = _unit(v_tag), _unit(v_style), _unit(w_probe)
    style_along_tag = float(v_style @ u_tag)
    rows.append({
        'layer': L,
        'cos(tag, style)':            _cos(v_tag, v_style),
        'cos(tag_A, tag_B)':          _cos(v_tag_A, v_tag_B),
        'cos(style_A, style_B)':      _cos(v_style_A, v_style_B),
        'cos(tag_c4, tag_gen)':       _cos(v_tag, v_tag_gen),
        'cos(style|user, style|cot)': _cos(v_style_u, v_style_c),
        'cos(probe, tag)':            _cos(w_probe, v_tag),
        'cos(probe, style)':          _cos(w_probe, v_style),
        'style gap off tag axis':     1 - style_along_tag**2 / float(v_style.norm()**2),
        'style / tag effect along tag axis': style_along_tag / float(v_tag.norm()),
        '|v_tag|': float(v_tag.norm()), '|v_style|': float(v_style.norm()),
    })
    Ht = H[tok_df['sample_ix'].tolist()].float()
    proj_tables[L] = (tok_df.assign(p_tag=(Ht @ u_tag).numpy(), p_style=(Ht @ u_style).numpy(), p_probe=(Ht @ u_probe).numpy())
                      .groupby(['tag', 'style'])[['p_tag', 'p_style', 'p_probe']].mean().round(2))

geom = pd.DataFrame(rows).set_index('layer').round(3)
print(f"\nrandom-direction |cos| baseline in D={D}: {rand_cos:.3f}")
display(geom)
print(f"\nMean projection onto unit v_tag / v_style / probe axis, layer {TEST_LAYER_IX}:")
display(proj_tables[TEST_LAYER_IX])

fig = px.line(geom.reset_index().melt(id_vars='layer',
              value_vars=['cos(tag, style)', 'cos(tag_A, tag_B)', 'cos(style_A, style_B)', 'cos(probe, style)', 'cos(probe, tag)']),
              x='layer', y='value', color='variable', markers=True,
              title=f'Tag vs style geometry by layer (n={n_pairs} style pairs) — cross-cosine vs split-half reliability')
fig.add_hline(y=rand_cos, line_dash='dot', annotation_text='random baseline')
fig.update_yaxes(range=[-0.2, 1]); fig.show()

r = geom.loc[TEST_LAYER_IX]
print(f"\nLayer {TEST_LAYER_IX}: cos(tag,style)={r['cos(tag, style)']:.2f} | reliabilities tag={r['cos(tag_A, tag_B)']:.2f}, style={r['cos(style_A, style_B)']:.2f} | "
      f"{100*r['style gap off tag axis']:.0f}% of the style gap is off the tag axis | "
      f"styling moves a token {100*r['style / tag effect along tag axis']:.0f}% of the full tag effect along the tag axis.")
print("Separable if cos(tag,style) sits well below both reliabilities; shared if it approaches them. "
      f"With only {n_pairs} pairs the style reliability will be noisy — treat it as a bound, not a point estimate.")
