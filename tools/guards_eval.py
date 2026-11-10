"""Re-rank saved ARC-AGI-1 candidates with symbolic guards and likelihood; print the scores.

The paper's 71.88 (d384 x6 with scale augmentation and scale views) and 70.88 (d384 x6 baseline)
come from this script: likelihood ranking of the 20 most-voted candidates, then the symbolic
guards. The guard rules follow transductARC's demo-derived candidate guards (MIT licence);
src/common.py holds the decoder's own copy of the same rules.

Inputs: a full decode's candidates.json.gz, the ARC-AGI-1 evaluation challenges and solutions,
and the rescore.json of a "rescore" evaluation over the same candidates. For the paper rows:

    python tools/guards_eval.py \
        runs/loopL-d384-exit-mid6-scale12-e330/evals/final_scale_views/candidates.json.gz \
        assets_tmp/ARC-1/arc1_eval_challenges.json assets_tmp/ARC-1/arc1_eval_solutions.json \
        runs/loopL-d384-exit-mid6-scale12-e330-views-rescore20/evals/rescore_views/rescore.json

    python tools/guards_eval.py \
        runs/loopL-d384-exit-mid6-e330/evals/final/candidates.json.gz \
        assets_tmp/ARC-1/arc1_eval_challenges.json assets_tmp/ARC-1/arc1_eval_solutions.json \
        runs/loopL-d384-exit-mid6-e330-rescore20/evals/rescore/rescore.json

The "ll_top20+guards" line is the reported top-2 score; "oracle all" is the oracle score.
"""
import argparse
import json, gzip, sys
from collections import Counter
from fractions import Fraction
import numpy as np

MAX_GRID_DIM = 30
_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
for _name in ("candidates", "challenges", "solutions", "rescore"):
    _parser.add_argument(_name)
_args = _parser.parse_args()
cands_path, chall_path, sol_path, rescore_path = _args.candidates, _args.challenges, _args.solutions, _args.rescore

def colors(g): return {int(v) for r in g for v in r}
def fg_counts(g): return Counter(int(v) for r in g for v in r if v != 0)
def is_sub(sub, grid):
    a = np.asarray(grid, dtype=np.int16); b = np.asarray(sub, dtype=np.int16)
    if a.ndim != 2 or b.ndim != 2 or b.shape[0] > a.shape[0] or b.shape[1] > a.shape[1]: return False
    w = np.lib.stride_tricks.sliding_window_view(a, b.shape)
    return bool((w == b).all(axis=(2, 3)).any())

def build(demos):
    g = dict(forbid_empty=True, forbid_copy=True, rr=None, cr=None, cset=False, ccount=False,
             crop=False, single=False, dropped=None)
    if not demos or any(not i or not i[0] or not o or not o[0] for i, o in demos): return g
    g['forbid_empty'] = not any(all(v == 0 for r in o for v in r) for _, o in demos)
    g['forbid_copy'] = not any(i == o for i, o in demos)
    rr = {Fraction(len(o), len(i)) for i, o in demos}; cr = {Fraction(len(o[0]), len(i[0])) for i, o in demos}
    if len(rr) == 1: g['rr'] = next(iter(rr))
    if len(cr) == 1: g['cr'] = next(iter(cr))
    g['cset'] = all(colors(i) == colors(o) for i, o in demos)
    g['ccount'] = all(fg_counts(i) == fg_counts(o) for i, o in demos)
    g['crop'] = all(is_sub(o, i) for i, o in demos)
    g['single'] = all(len(colors(o) - {0}) == 1 for _, o in demos)
    dropped = []
    for i, o in demos:
        ci, co = colors(i), colors(o); d = ci - co
        if len(d) != 1 or not co <= ci: dropped = []; break
        dropped.append(next(iter(d)))
    if dropped and len(set(dropped)) == 1: g['dropped'] = dropped[0]
    return g

def rejects(g, grid, inp):
    """Returns the name of the first rule that rejects, or None."""
    if not grid or not grid[0]: return 'empty_grid'
    if g['forbid_empty'] and not any(v for r in grid for v in r): return 'all_background'
    if g['forbid_copy'] and inp and grid == inp: return 'input_copy'
    for cd, idim, ratio, name in ((len(grid), len(inp), g['rr'], 'row_ratio'),
                                  (len(grid[0]), len(inp[0]) if inp else 0, g['cr'], 'col_ratio')):
        if ratio is None or idim <= 0: continue
        exp = idim * ratio
        if exp.denominator != 1 or not 1 <= exp <= MAX_GRID_DIM: continue
        if cd != exp: return name
    if g['single'] and len(colors(grid) - {0}) > 1: return 'single_color'
    if inp and inp[0]:
        if g['crop'] and not is_sub(grid, inp): return 'crop'
        if g['cset'] and colors(grid) != colors(inp): return 'color_set'
        if g['ccount'] and fg_counts(grid) != fg_counts(inp): return 'color_counts'
        if g['dropped'] is not None:
            exp = colors(inp) - {g['dropped']}; c = colors(grid)
            if c - exp or exp - c - {0}: return 'dropped_color'
    return None

C = json.load(gzip.open(cands_path))['pairs']
CH = json.load(open(chall_path)); SOL = json.load(open(sol_path))
R = {(p['task_id'], p['pair_index']): p for p in json.load(open(rescore_path))['pairs']}
tasks = sorted(CH)
n_out = {t: len(CH[t]['test']) for t in tasks}
assert len(tasks) == 400

guards = {t: build([(p['input'], p['output']) for p in CH[t]['train']]) for t in tasks}
active = Counter()
for t in tasks:
    g = guards[t]
    for k in ('rr', 'cr', 'dropped'):
        if g[k] is not None: active[k] += 1
    for k in ('cset', 'ccount', 'crop', 'single'):
        if g[k]: active[k] += 1
print('guards active (tasks of 400):', dict(active))

def score(picks):  # picks[(t,i)] = list of grids (top-2)
    s = 0.0
    for t in tasks:
        ok = sum(1 for i in range(n_out[t]) if any(gr == SOL[t][i] for gr in picks.get((t, i), [])[:2]))
        s += ok / n_out[t]
    return s

rules = {}
gen = gen_kept = 0; rej_reasons = Counter(); correct_rejected = []
oracle = {'all': {}, 'guarded': {}}
picks = {k: {} for k in ['votes', 'votes_noinputcopy', 'votes+guards', 'll_top20', 'll_top20+guards']}
mismatch = 0
for p in C:
    t, i = p['task_id'], p['pair_index']
    inp = CH[t]['test'][i]['input']; truth = SOL[t][i]
    cands = p['candidates']  # assumed already ordered by votes
    order = sorted(range(len(cands)), key=lambda j: -cands[j]['count'])  # stable: keeps file order on ties
    if order != list(range(len(cands))): mismatch += 1
    grids = [cands[j]['grid'] for j in order]; counts = [cands[j]['count'] for j in order]
    copyflag = [cands[j].get('input_copy', False) for j in order]
    gen += sum(counts)
    keep = []
    for g_, c_, cf in zip(grids, counts, copyflag):
        why = rejects(guards[t], g_, inp)
        if why: 
            rej_reasons[why] += c_
            if g_ == truth: correct_rejected.append((t, i, why, c_))
        else:
            keep.append((g_, c_)); gen_kept += c_
    picks['votes'][(t, i)] = grids
    picks['votes_noinputcopy'][(t, i)] = [g_ for g_, cf in zip(grids, copyflag) if not cf]
    picks['votes+guards'][(t, i)] = [g_ for g_, _ in keep]
    oracle['all'][(t, i)] = grids; oracle['guarded'][(t, i)] = [g_ for g_, _ in keep]
    # likelihood: rescore candidates = top-20 by votes with input copies dropped (verify via counts/correct)
    r = R[(t, i)]['candidates']
    pool = [(g_, c_) for g_, c_, cf in zip(grids, counts, copyflag) if not cf][:len(r)]
    assert [c_ for _, c_ in pool] == [x['count'] for x in r], (t, i)
    assert [g_ == truth for g_, _ in pool] == [bool(x['correct']) for x in r], (t, i)
    ll = sorted(range(len(r)), key=lambda j: -r[j]['log_likelihood'])
    picks['ll_top20'][(t, i)] = [pool[j][0] for j in ll]
    picks['ll_top20+guards'][(t, i)] = [pool[j][0] for j in ll if not rejects(guards[t], pool[j][0], inp)]

print('candidate order already by votes in file for all but', mismatch, 'outputs')
print(f'generated votes {gen}, rejected {gen-gen_kept} ({100*(gen-gen_kept)/gen:.2f}%)')
print('rejections by rule (votes):', dict(rej_reasons.most_common()))
print('correct answers rejected by a guard:', correct_rejected)
def orc(d):
    return score({k: v for k, v in d.items()} ) if False else sum(
        sum(1 for i in range(n_out[t]) if any(g_ == SOL[t][i] for g_ in d.get((t, i), []))) / n_out[t] for t in tasks)
for k, v in picks.items():
    s = score(v); print(f'{k:22s} top-2 {s:6.2f}/400 = {100*s/400:6.3f}%')
for k, v in oracle.items():
    s = orc(v); print(f'oracle {k:15s} {s:6.2f}/400 = {100*s/400:6.3f}%')
# per-output changes vs votes
def solved(v):
    return {(t, i) for t in tasks for i in range(n_out[t]) if any(g_ == SOL[t][i] for g_ in v.get((t, i), [])[:2])}
base = solved(picks['votes'])
for k in ['votes+guards', 'll_top20', 'll_top20+guards']:
    s = solved(picks[k]); print(f'{k}: +{len(s-base)} / -{len(base-s)} test outputs vs votes', sorted(s-base)[:12], sorted(base-s)[:12])
