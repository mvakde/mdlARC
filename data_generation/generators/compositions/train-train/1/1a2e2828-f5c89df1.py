"""Registered ribbons: complete, translate, and re-layer an anchored weave.

Only 0 (background), 8 (destination), and 9 (per-ribbon source) have fixed
roles. Ribbon colors 1..7 are interchangeable. No parent code is imported.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np


HOST_SEEDS = (0,175032568,788329201,6652785,838725861,976152075,128157282,
              888440571,860747138,451272315,194024017,7750631,690184338,
              977735189,985868420,316547161,253771604,56219914,65241186,
              202225125,315845623,285974717,684913034,380700377,444194649,
              627233538,641634232,246634944,172872110,874158413,997633366,
              464423218,298807121,401509033,846075089,605468501,955416615,
              707315508,228203269,754775451,809115690,646262199)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer grid required')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'colors outside 0..9')
    return a


def _runs(mask):
    result = []
    for i, on in enumerate(list(mask) + [False]):
        if on and (i == 0 or not mask[i-1]):
            start = i
        if not on and i and mask[i-1]:
            result.append((start, i))
    return result


def _rectangle(r0, r1, c0, c1):
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _decode(value):
    """Recover rectangles, their pins, and the complete depth order from cells."""
    a = _grid(value)
    material = ((a >= 1) & (a <= 7)) | (a == 9)
    pos = np.argwhere(material)
    _require(len(pos) > 0, 'no swatch')
    r0, c0 = map(int, pos.min(axis=0))
    rb, cb = map(int, pos.max(axis=0) + 1)
    crop = a[r0:rb, c0:cb]
    mat = material[r0:rb, c0:cb]
    hm, vm = np.all(mat, axis=1), np.all(mat, axis=0)
    hr, vr = _runs(hm), _runs(vm)
    _require(hr and vr and np.any(~hm) and np.any(~vm), 'missing bands or gaps')
    _require(not hm[0] and not hm[-1] and not vm[0] and not vm[-1], 'swatch margins')
    layers = []
    for orientation, runs in (('h', hr), ('v', vr)):
        for start, end in runs:
            rect = (_rectangle(r0+start, r0+end, c0, cb) if orientation == 'h'
                    else _rectangle(r0, rb, c0+start, c0+end))
            exposed = (crop[start:end, :][:, ~vm] if orientation == 'h'
                       else crop[~hm, :][:, start:end])
            colors = set(map(int, np.unique(exposed))) - {9}
            _require(len(colors) == 1 and 1 <= next(iter(colors)) <= 7, 'band identity')
            color = colors.pop()
            pins = [(r, c) for r, c in sorted(rect) if a[r, c] == 9]
            _require(len(pins) == 1, 'one source pin on each ribbon required')
            pin = pins[0]
            rr, cc = pin[0]-r0, pin[1]-c0
            _require(bool(hm[rr]) != bool(vm[cc]), 'source pin at crossing')
            layers.append({'orientation': orientation, 'color': color,
                           'pin': pin, 'cells': rect - {pin}, 'rect': rect,
                           'thickness': end-start})
    _require(2 <= len(layers) <= 7, 'ribbon count')
    colors = [x['color'] for x in layers]
    _require(len(set(colors)) == len(colors), 'distinct ribbon colors required')
    edges = [set() for _ in layers]
    indeg = [0] * len(layers)
    for hi, (ra, rz) in enumerate(hr):
        for vi, (ca, cz) in enumerate(vr):
            j = len(hr)+vi
            crossing = crop[ra:rz, ca:cz]
            color = int(crossing.flat[0])
            _require(bool(np.all(crossing == color)), 'nonuniform crossing')
            _require(color in (colors[hi], colors[j]), 'invalid crossing color')
            above, below = (hi, j) if color == colors[hi] else (j, hi)
            edges[above].add(below)
            indeg[below] += 1
    order, remaining = [], set(range(len(layers)))
    while remaining:
        top = [i for i in sorted(remaining) if indeg[i] == 0]
        _require(len(top) == 1, 'crossings must determine one complete depth order')
        i = top[0]
        remaining.remove(i)
        order.append(i)
        for j in edges[i]:
            indeg[j] -= 1
    # Regenerate the entire source drawing, including all occluded material.
    weave = np.zeros(a.shape, dtype=np.int8)
    for i in reversed(order):
        for r, c in layers[i]['rect']:
            weave[r, c] = colors[i]
    expected = weave.copy()
    for layer in layers:
        expected[layer['pin']] = 9
    destinations = [tuple(map(int, p)) for p in np.argwhere(a == 8)]
    _require(1 <= len(destinations) <= 3, 'destination count')
    for d in destinations:
        _require(expected[d] == 0, 'destination on source material')
        expected[d] = 8
    _require(np.array_equal(a, expected), 'unexplained cells or broken ribbons')
    return {'layers': layers, 'order': order, 'destinations': destinations,
            'weave': weave, 'shape': a.shape, 'swatch_shape': (rb-r0, cb-c0)}


def _placements(model, *, visible_only=False, common_pin=False):
    result = []
    reference = model['layers'][0]['pin']
    for layer in model['layers']:
        sr, sc = reference if common_pin else layer['pin']
        cells = layer['cells']
        if visible_only:
            cells = {(r,c) for r,c in cells if model['weave'][r,c] == layer['color']}
        result.append([{(r-sr+dr, c-sc+dc) for r,c in cells}
                       for dr,dc in model['destinations']])
    return result


def _paint(model, placements=None, order=None, strict=True):
    if placements is None:
        placements = _placements(model)
    if order is None:
        order = model['order']
    h, w = model['shape']
    out = np.zeros((h,w), dtype=np.int8)
    for i in reversed(order):
        for copy in placements[i]:
            for r,c in copy:
                if strict:
                    _require(0 <= r < h and 0 <= c < w, 'full copy out of bounds')
                if 0 <= r < h and 0 <= c < w:
                    out[r,c] = model['layers'][i]['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All interpretation uses only this input, including transitive depth."""
    return _paint(_decode(input_grid))


def _features(a, model=None):
    m = _decode(a) if model is None else model
    placements = _placements(m)
    correct = _paint(m, placements)
    visible = _paint(m, _placements(m, visible_only=True))
    reversed_depth = _paint(m, placements, list(reversed(m['order'])))
    common = _paint(m, _placements(m, common_pin=True), strict=False)
    coverage = {}
    for i, copies in enumerate(placements):
        for cells in copies:
            for cell in cells:
                coverage.setdefault(cell, set()).add(i)
    rank = {i:k for k,i in enumerate(m['order'])}
    parallel = 0
    for owners in coverage.values():
        top = min(owners, key=rank.get)
        parallel += any(i != top and m['layers'][i]['orientation'] ==
                        m['layers'][top]['orientation'] for i in owners)
    within_layer_overlap = sum(sum(map(len, copies)) - len(set().union(*copies))
                               for copies in placements)
    target_holes_covered = sum(correct[d] != 0 for d in m['destinations'])
    # A whole assembled stamp pasted last must not substitute for global layers.
    objectwise_differences = []
    for target_order in permutations(range(len(m['destinations']))):
        shortcut = np.zeros(m['shape'],np.int8)
        for target in target_order:
            for i in reversed(m['order']):
                for r,c in placements[i][target]:
                    shortcut[r,c] = m['layers'][i]['color']
        objectwise_differences.append(not np.array_equal(correct,shortcut))
    return {'ribbons': len(m['layers']), 'targets': len(m['destinations']),
            'top_orientation': m['layers'][m['order'][0]]['orientation'],
            'depth_matters': not np.array_equal(correct, reversed_depth),
            'completion_matters': not np.array_equal(correct, visible),
            'registration_matters': not np.array_equal(correct, common),
            'parallel_witness_cells': int(parallel), 'copy_overlap': within_layer_overlap,
            'covered_target_holes': int(target_holes_covered),
            'layerwise_union_matters': all(objectwise_differences),
            'output_colors': len(set(map(int,np.unique(correct))) - {0}),
            'thicknesses': tuple(x['thickness'] for x in m['layers']),
            'swatch_shape': m['swatch_shape'], 'shape': m['shape']}


def _axis(rng, count):
    thickness = [int(rng.choice((1,1,2,2,3))) for _ in range(count)]
    gaps = [int(rng.choice((1,1,2,3))) for _ in range(count+1)]
    runs, p = [], gaps[0]
    for i,t in enumerate(thickness):
        runs.append((p,p+t))
        p += t+gaps[i+1]
    return p, runs


def _attempt(rng, count, target_count, top_orientation, colors):
    # Alternation gives a crossing path between every pair, including parallels.
    orientations = [top_orientation if i % 2 == 0 else
                    ('v' if top_orientation == 'h' else 'h') for i in range(count)]
    nh, nv = orientations.count('h'), orientations.count('v')
    h, hr = _axis(rng, nh)
    w, vr = _axis(rng, nv)
    if max(h,w) > 16:
        return None
    hm, vm = np.zeros(h,bool), np.zeros(w,bool)
    for x,y in hr:
        hm[x:y] = True
    for x,y in vr:
        vm[x:y] = True
    # Node ids follow spatial order; depth independently permutes each orientation.
    rects = [_rectangle(x,y,0,w) for x,y in hr] + [_rectangle(0,h,x,y) for x,y in vr]
    hp, vp = list(rng.permutation(nh)), list(nh+rng.permutation(nv))
    order = [int((hp if o == 'h' else vp).pop()) for o in orientations]
    pins = []
    for i,rect in enumerate(rects):
        exposed = sorted((r,c) for r,c in rect if not (hm[r] and vm[c]))
        pins.append(exposed[int(rng.integers(len(exposed)))])
    weave = np.zeros((h,w),np.int8)
    for i in reversed(order):
        for r,c in rects[i]:
            weave[r,c] = colors[i]
    source = weave.copy()
    for pin in pins:
        source[pin] = 9
    # Targets are ordinary blank points in or around the swatch, without a panel.
    pool = [(r,c) for r in range(-5,h+5) for c in range(-5,w+5)
            if not (0 <= r < h and 0 <= c < w and source[r,c] != 0)]
    first = pool[int(rng.integers(len(pool)))]
    # Close target groups make overlap common; free groups provide separation.
    radius = int(rng.integers(2,7))
    nearby = [(r,c) for r,c in pool if (r,c) != first and
              max(abs(r-first[0]),abs(c-first[1])) <= radius]
    candidate_pool = nearby if rng.random() < .75 and len(nearby) >= target_count-1 else pool
    if len(candidate_pool) < target_count-1:
        return None
    destinations = [first]
    if target_count > 1:
        for k in rng.permutation(len(candidate_pool)):
            if candidate_pool[int(k)] not in destinations:
                destinations.append(candidate_pool[int(k)])
            if len(destinations) == target_count:
                break
    all_cells = set().union(*rects) | set(destinations)
    for rect, (sr,sc) in zip(rects,pins):
        for dr,dc in destinations:
            all_cells |= {(r-sr+dr,c-sc+dc) for r,c in rect if (r,c) != (sr,sc)}
    lo_r, hi_r = min(r for r,c in all_cells), max(r for r,c in all_cells)
    lo_c, hi_c = min(c for r,c in all_cells), max(c for r,c in all_cells)
    bh, bw = hi_r-lo_r+1, hi_c-lo_c+1
    if max(bh,bw) > 30:
        return None
    # Swatch corners remain blank; tight output edges are allowed independently.
    pads = [int(rng.integers(0,min(3,30-bh)+1)), 0,
            int(rng.integers(0,min(3,30-bw)+1)), 0]
    pads[1] = int(rng.integers(0,min(3,30-bh-pads[0])+1))
    pads[3] = int(rng.integers(0,min(3,30-bw-pads[2])+1))
    off_r, off_c = pads[0]-lo_r, pads[2]-lo_c
    inp = np.zeros((bh+pads[0]+pads[1],bw+pads[2]+pads[3]),np.int8)
    inp[off_r:off_r+h,off_c:off_c+w] = source
    for r,c in destinations:
        inp[off_r+r,off_c+c] = 8
    # Construction-side oracle: use the sampled rectangles and pins directly,
    # before any input decoding. This does not call solve or its paint helper.
    out = np.zeros_like(inp)
    for i in reversed(order):
        sr,sc = pins[i]
        for dr,dc in destinations:
            for r,c in rects[i]:
                if (r,c) != (sr,sc):
                    out[off_r+r-sr+dr,off_c+c-sc+dc] = colors[i]
    return {'input':inp,'output':out}


def _pair(rng, count, targets, orientation):
    for _ in range(1500):
        colors = [int(x) for x in rng.choice(np.arange(1,8),count,replace=False)]
        pair = _attempt(rng,count,targets,orientation,colors)
        if pair is None:
            continue
        inp = pair['input']
        model = _decode(inp)
        _require(np.array_equal(_paint(model),pair['output']), 'construction oracle disagrees')
        f = _features(inp,model)
        if not all(f[x] for x in ('depth_matters','completion_matters','registration_matters')):
            continue
        if f['output_colors'] < 2 or (count >= 3 and not f['parallel_witness_cells']):
            continue
        if targets > 1 and not (f['copy_overlap'] and f['layerwise_union_matters']
                                and f['covered_target_holes']):
            continue
        return pair
    raise RuntimeError('could not construct informative registered weave')


def _counts(num_train, num_test):
    for x, lo, hi in ((num_train,2,10),(num_test,1,4)):
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not lo <= int(x) <= hi:
            raise ValueError('illegal example counts')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train),int(num_test)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(map(int,a.flat))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a,b = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    first = str(rng.choice(('h','v')))
    specs = [(3,2,first),(4,1,'v' if first == 'h' else 'h')]
    specs += [(int(rng.integers(2,5)),int(rng.integers(1,4)),str(rng.choice(('h','v'))))
              for _ in range(a-2)]
    rng.shuffle(specs)
    tests = [(int(rng.integers(5,7)),int(rng.integers(1,4)),str(rng.choice(('h','v'))))
             for _ in range(b)]
    puzzle, seen = {'train':[],'test':[]}, set()
    for split, plan in (('train',specs),('test',tests)):
        for count,targets,orientation in plan:
            for _ in range(30):
                pair = _pair(rng,count,targets,orientation)
                k = _key(pair['input'])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('duplicate input retries exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in ('train','test')), 'split lists')
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen, features = set(), {'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'}, 'pair keys')
            inp,out = _grid(pair['input']),_grid(pair['output'])
            _require(inp.shape == out.shape, 'output shape')
            k = _key(inp)
            _require(k not in seen, 'duplicate input')
            seen.add(k)
            m = _decode(inp)
            _require(np.array_equal(_paint(m),out), 'incorrect registered layering')
            _require(not np.any((out == 8) | (out == 9)), 'markers copied')
            f = _features(inp,m)
            _require(all(f[x] for x in ('depth_matters','completion_matters','registration_matters')),
                     'parent ablation does not change answer')
            _require(f['output_colors'] >= 2, 'only uppermost color survives')
            if f['ribbons'] >= 3:
                _require(f['parallel_witness_cells'] > 0, 'no transitive parallel-layer witness')
            if f['targets'] > 1:
                _require(f['copy_overlap'] > 0, 'multiple registration lacks overlap')
                _require(f['layerwise_union_matters'], 'whole assemblies can substitute for layers')
                _require(f['covered_target_holes'] > 0, 'no transparent-hole demonstration')
            features[split].append(f)
    train = features['train']
    _require({f['top_orientation'] for f in train} == {'h','v'}, 'orientation demonstrations')
    _require(any(f['ribbons'] == 3 and f['targets'] == 2 for f in train), 'three-ribbon multi-target demo')
    _require(any(f['ribbons'] == 4 and f['targets'] == 1 for f in train), 'four-ribbon single-target demo')
    _require(all(f['ribbons'] <= 4 for f in train), 'training count regime')
    _require(all(f['ribbons'] in (5,6) for f in features['test']), 'test depth extension')


def self_test():
    """Bounded: 42 specified seeds plus count boundaries and a tiny analytic case."""
    signatures, shapes, motifs, pins, palettes, top = set(),set(),set(),set(),set(),set()
    counts, thicknesses = set(),set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = tuple((_key(p['input']),_key(p['output']))
                          for s in ('train','test') for p in puzzle[s])
        signatures.add(signature)
        for split in ('train','test'):
            for pair in puzzle[split]:
                m = _decode(pair['input'])
                _require(np.array_equal(solve(pair['input']),pair['output']), 'solve disagreement')
                shapes.add(m['shape'])
                palettes.add(tuple(sorted(x['color'] for x in m['layers'])))
                counts.add((len(m['layers']),len(m['destinations'])))
                top.add(m['layers'][m['order'][0]]['orientation'])
                for layer in m['layers']:
                    sr,sc = layer['pin']
                    offsets = tuple(sorted((r-sr,c-sc) for r,c in layer['cells']))
                    motifs.add(offsets)
                    pins.add((layer['orientation'],min(r for r,c in offsets),
                              max(r for r,c in offsets),min(c for r,c in offsets),
                              max(c for r,c in offsets)))
                    thicknesses.add(layer['thickness'])
    _require(len(signatures) == 42 and len(shapes) >= 50, 'shape/seed diversity')
    _require(len(motifs) >= 150 and len(pins) >= 150 and len(palettes) >= 25, 'structural diversity')
    _require({n for n,k in counts} == {2,3,4,5,6} and {k for n,k in counts} == {1,2,3}, 'count diversity')
    _require(top == {'h','v'} and thicknesses == {1,2,3}, 'geometry diversity')
    p = generate_puzzle(17,2,1)
    q = generate_puzzle(17,2,1)
    _require(all(_key(x[k]) == _key(y[k]) for s in p for x,y in zip(p[s],q[s])
                 for k in ('input','output')), 'determinism')
    for a,b in ((2,1),(10,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(113+a+b,a,b))
    for a,b in ((1,1),(11,1),(4,0),(5,5),(2,2),(3,4),(2.5,1),(4,True),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Hand-calculated registration/compositing, independent of the sampler.
    model = {'shape':(5,5),'order':[0,1], 'destinations':[(2,2)],
             'layers':[{'color':2,'pin':(0,1),'cells':{(0,0),(0,2)}},
                       {'color':3,'pin':(1,0),'cells':{(0,0),(1,1),(2,0)}}]}
    expected = np.zeros((5,5),np.int8)
    expected[2,1],expected[2,3],expected[1,2],expected[3,2] = 2,2,3,3
    _require(np.array_equal(_paint(model),expected), 'analytic registration case')
    # Hand-solved H(1) > V(2) > H(3) swatch. After registration both
    # horizontal ribbons share row 7, so their new collision uses transitivity.
    fixture = np.zeros((12,11),np.int8)
    for r,row in enumerate(('00900','91111','00200','39233','00200')):
        fixture[r,:5] = list(map(int,row))
    fixture[7,5] = 8
    expected = np.zeros_like(fixture)
    expected[7,4],expected[7,6:10],expected[8:12,5] = 3,1,2
    _require(np.array_equal(solve(fixture),expected), 'hand-solved transitive weave')
    thick = np.zeros((16,16),np.int8)
    thick[:6,1:5],thick[1:5,:6] = 4,3
    thick[2,0],thick[0,2],thick[10,10] = 9,9,8
    expected = np.zeros_like(thick)
    expected[10:16,9:13],expected[9:13,10:16] = 4,3
    expected[10,10] = 0
    _require(np.array_equal(solve(thick),expected), 'thicker-ribbon support')
    # Exercise the documented seven-ribbon extension beyond sampled tests.
    seven = _pair(np.random.default_rng(2718),7,1,'h')
    _require(len(_decode(seven['input'])['layers']) == 7 and
             np.array_equal(solve(seven['input']),seven['output']), 'seven-ribbon support')
    return {'seeds':42,'puzzles':len(signatures),'shapes':len(shapes),
            'ribbon_offset_shapes':len(motifs),'pin_geometries':len(pins),
            'palettes':len(palettes),'ribbon_target_counts':sorted(counts)}


if __name__ == '__main__':
    print(self_test())
