# ================================================================================================
# STAGE     1. Role leakage: build v_tag and v_style, measure their geometry
# RAN ON    Colab (A100), as a cell after the paper's role-probe demo notebook
# NEEDS     demo-notebook state: model, tokenizer, all_probe_hs, probe_sample_df, all_probes, run_and_export_states, render_single_role_gptoss, ...
# PRODUCES  pairs (style pairs), tok_df, hs, cond_df; prints the geometry table
# FIGURE    Figure 2 (rendered by 06_figures_local)
# NOTE      code unchanged from the run that produced the paper's numbers; only this header was added
# ================================================================================================
"""
STEP 1 — Are TAG and STYLE the same direction?

Builds two directions at every probed layer and measures their geometry:
  v_tag   : same content, different tag   (cot - user)      <- reuses probe-training data already in memory
  v_style : same tag, different style     (styled - destyled) <- generated here, no attacks involved

Reliability controls: split-half cosine of each direction with itself (the ceiling any cross-cosine
should be judged against), a random-direction baseline, and the probe's own cot-vs-user weight axis.

Assumes from earlier cells: model, tokenizer, main_device, seed, run_gptoss_custom, run_and_export_states,
ReconstructableTextDataset, stack_collate, label_gptoss_content_roles, render_single_role_gptoss,
all_probe_hs, probe_sample_df, LAYERS_TO_PROBE, all_probes, TEST_LAYER_IX
"""
import re, gc, torch, numpy as np, pandas as pd, cupy, plotly.express as px
from tqdm import tqdm
from torch.utils.data import DataLoader

N_PAIRS, MAX_GEN_TOKS, MAX_TXT_TOKS, TAGS = 24, 384, 256, ['user', 'cot']
torch.manual_seed(seed)

# ---------------- helpers ----------------
@torch.no_grad()
def _gen(messages, max_new_tokens):
    inp = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors='pt').to(main_device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                         pad_token_id=tokenizer.eos_token_id)
    txt = tokenizer.decode(out[0, inp['input_ids'].shape[-1]:], skip_special_tokens=False)
    def ch(c):
        m = re.search(rf"<\|channel\|>{c}<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|<\|return\|>|$)", txt, re.DOTALL)
        return m.group(1).strip() if m else ""
    return ch('analysis'), ch('final')

def _trunc(t):
    return tokenizer.decode(tokenizer(t, add_special_tokens=False, truncation=True, max_length=MAX_TXT_TOKS).input_ids)

def _unit(v): return v / (v.norm() + 1e-8)
def _cos(a, b): return float(_unit(a) @ _unit(b))
def _mean(H, idx): return H[idx].float().mean(0)

# ---------------- 1. generate styled / destyled pairs (benign, model-authored) ----------------
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
DESTYLE_SYS = ("Reasoning: low\nRewrite the user's text in plain, ordinary prose. Keep every piece of content and meaning "
               "identical. Use full natural sentences, no terse note-like fragments, no 'we need to' phrasing. Output only the rewritten text.")
pairs = []
for q in tqdm(QUESTIONS[:N_PAIRS], desc='generating styled/destyled pairs'):
    styled, _ = _gen([{'role': 'user', 'content': q}], MAX_GEN_TOKS)              # genuine model CoT = reasoning style
    if len(styled.split()) < 15: continue
    _, destyled = _gen([{'role': 'system', 'content': DESTYLE_SYS}, {'role': 'user', 'content': styled}], MAX_GEN_TOKS)
    if len(destyled.split()) < 15: continue
    pairs.append({'pair_ix': len(pairs), 'question': q, 'styled': _trunc(styled), 'destyled': _trunc(destyled)})
print(f"\n{len(pairs)} usable pairs. Example:\n[STYLED]   {pairs[0]['styled'][:200]}\n[DESTYLED] {pairs[0]['destyled'][:200]}\n")

# ---------------- 2. 2x2 conditions: {styled, destyled} x {user tag, cot tag}; one forward pass ----------------
cond_df = pd.DataFrame([{'pair_ix': p['pair_ix'], 'style': s, 'tag': t, 'prompt': render_single_role_gptoss(t, p[s])}
                        for p in pairs for s in ['styled', 'destyled'] for t in TAGS]).assign(prompt_ix=lambda d: range(len(d)))
max_len = int(tokenizer(cond_df['prompt'].tolist(), padding=True, return_tensors='pt')['attention_mask'].sum(1).max())
dl = DataLoader(ReconstructableTextDataset(cond_df['prompt'].tolist(), tokenizer, max_length=max_len, prompt_ix=cond_df['prompt_ix'].tolist()),
                batch_size=16, shuffle=False, collate_fn=stack_collate)
gc.collect(); torch.cuda.empty_cache()
out = run_and_export_states(model, tokenizer, run_model_return_states=run_gptoss_custom, dl=dl, layers_to_keep_acts=LAYERS_TO_PROBE)
tok_df = (label_gptoss_content_roles(out['sample_df'].assign(sample_ix=lambda d: range(len(d))))
          .pipe(lambda d: d[(d['is_content'] == True) & d['role'].notna()])
          .merge(cond_df[['prompt_ix', 'pair_ix', 'style', 'tag']], on='prompt_ix'))
hs = {L: out['all_hs'][:, i, :].to(torch.float16) for i, L in enumerate(LAYERS_TO_PROBE)}

# ---------------- 3. geometry per layer ----------------
pdf = probe_sample_df
D = hs[LAYERS_TO_PROBE[0]].shape[1]
rand_cos = float(torch.stack([torch.abs(_unit(torch.randn(D)) @ _unit(torch.randn(D))) for _ in range(200)]).mean())
rows, proj_tables = [], {}
for L in LAYERS_TO_PROBE:
    Hp, H = all_probe_hs[L], hs[L]
    sel_p = lambda m, r: pdf[m & (pdf['role'] == r)]['sample_ix'].tolist()
    sel_t = lambda m, s: tok_df[m & (tok_df['style'] == s)]['sample_ix'].tolist()
    allp, allt = pd.Series(True, index=pdf.index), pd.Series(True, index=tok_df.index)
    evP, evT = (pdf['base_seq_ix'] % 2 == 0), (tok_df['pair_ix'] % 2 == 0)

    v_tag   = _mean(Hp, sel_p(allp, 'cot')) - _mean(Hp, sel_p(allp, 'user'))          # content-controlled (C4/Dolma)
    v_tag_A = _mean(Hp, sel_p(evP,  'cot')) - _mean(Hp, sel_p(evP,  'user'))
    v_tag_B = _mean(Hp, sel_p(~evP, 'cot')) - _mean(Hp, sel_p(~evP, 'user'))
    v_tag_gen = (_mean(H, tok_df[tok_df['tag'] == 'cot']['sample_ix'].tolist())
                 - _mean(H, tok_df[tok_df['tag'] == 'user']['sample_ix'].tolist()))      # same direction, on generated text

    v_style   = _mean(H, sel_t(allt, 'styled')) - _mean(H, sel_t(allt, 'destyled'))     # tag-controlled
    v_style_A = _mean(H, sel_t(evT,  'styled')) - _mean(H, sel_t(evT,  'destyled'))
    v_style_B = _mean(H, sel_t(~evT, 'styled')) - _mean(H, sel_t(~evT, 'destyled'))
    v_style_u = _mean(H, sel_t(tok_df['tag'] == 'user', 'styled')) - _mean(H, sel_t(tok_df['tag'] == 'user', 'destyled'))
    v_style_c = _mean(H, sel_t(tok_df['tag'] == 'cot',  'styled')) - _mean(H, sel_t(tok_df['tag'] == 'cot',  'destyled'))

    pr = [p for p in all_probes if p['layer_ix'] == L][0]
    W = cupy.asnumpy(pr['probe'].named_steps['clf'].coef_)
    w_probe = torch.tensor(W[0] if W.shape[0] == 1 else W[pr['roles_map']['cot']] - W[pr['roles_map']['user']]).float()

    u_tag, u_style = _unit(v_tag), _unit(v_style)
    style_along_tag = float(v_style @ u_tag)
    rows.append({
        'layer': L,
        'cos(tag, style)':        _cos(v_tag, v_style),
        'cos(tag_A, tag_B)':      _cos(v_tag_A, v_tag_B),        # reliability ceiling for v_tag
        'cos(style_A, style_B)':  _cos(v_style_A, v_style_B),    # reliability ceiling for v_style
        'cos(tag_c4, tag_gen)':   _cos(v_tag, v_tag_gen),        # does the tag direction transfer to this text?
        'cos(style|user, style|cot)': _cos(v_style_u, v_style_c),# is the style direction the same under both tags?
        'cos(probe, tag)':        _cos(w_probe, v_tag),
        'cos(probe, style)':      _cos(w_probe, v_style),
        'style gap off tag axis': 1 - style_along_tag**2 / float(v_style.norm()**2),   # Mogford's ~95% number
        'style effect / tag effect (along tag axis)': style_along_tag / float(v_tag.norm()),  # footnote 12 quantified
        '|v_tag|': float(v_tag.norm()), '|v_style|': float(v_style.norm()),
    })
    # 2x2 table of mean projections (raw units) — shows what moves what
    proj_tables[L] = (tok_df.assign(p_tag=(H[tok_df['sample_ix'].tolist()].float() @ u_tag).numpy(),
                                    p_style=(H[tok_df['sample_ix'].tolist()].float() @ u_style).numpy())
                      .groupby(['tag', 'style'])[['p_tag', 'p_style']].mean().round(2))

geom = pd.DataFrame(rows).set_index('layer').round(3)
print(f"random-direction |cos| baseline in D={D}: {rand_cos:.3f}\n")
display(geom)
print(f"\nMean projection onto unit v_tag / unit v_style at layer {TEST_LAYER_IX} (2x2):")
display(proj_tables[TEST_LAYER_IX])

fig = px.line(geom.reset_index().melt(id_vars='layer',
              value_vars=['cos(tag, style)', 'cos(tag_A, tag_B)', 'cos(style_A, style_B)', 'cos(probe, style)']),
              x='layer', y='value', color='variable', markers=True,
              title='Tag vs style geometry by layer — cross-cosine vs split-half reliability')
fig.add_hline(y=rand_cos, line_dash='dot', annotation_text='random baseline')
fig.update_yaxes(range=[-0.2, 1]); fig.show()

r = geom.loc[TEST_LAYER_IX]
print(f"\nLayer {TEST_LAYER_IX}: cos(tag,style)={r['cos(tag, style)']:.2f} vs reliabilities "
      f"{r['cos(tag_A, tag_B)']:.2f}/{r['cos(style_A, style_B)']:.2f}; "
      f"{100*r['style gap off tag axis']:.0f}% of the style gap lies OFF the tag axis; "
      f"styling moves a token {100*r['style effect / tag effect (along tag axis)']:.0f}% of the way along the tag axis.")
print("Read: separable if cos(tag,style) is well below both split-half reliabilities; shared if it approaches them.")
