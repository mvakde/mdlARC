"""Boundary-clipped pattern transfers. Standard library and NumPy only.

No files are read by generation or solving. Diagnostics stay in memory.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

D = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0,950251740,269111610,997622850,794007739,173908428,933223910,
    117997615,959484320,511402292,869599426,340080216,594167912,247183956,
    64712846,864335437,983126277,893176211,415696494,28856140,884470126,
    961248071,26274864,410312767,17492719,628968441,982838786,752981192,
    497630465,134682967,13946740,544333526,572038317,36381894,945859478,
    633512793,110026244,419971251,284273593,420316764,32275563,525419349)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid dimensions')
    _need(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _need(bool(np.all((0 <= a) & (a <= 9))), 'invalid colors')
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in D:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _regions(clean):
    ids = np.full(clean.shape, -1, dtype=np.int16)
    colors, cells = [], []
    for color in np.unique(clean):
        for component in _components(clean == color):
            label = len(colors)
            colors.append(int(color))
            cells.append(component)
            for p in component:
                ids[p] = label
    return ids, colors, cells


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _plus(p, d, n=1):
    return p[0] + n*d[0], p[1] + n*d[1]


def _geometry(center, direction, shape):
    perpendicular = -direction[1], direction[0]
    tip = _plus(center, direction)
    arms = (_plus(center, perpendicular), _plus(center, perpendicular, -1))
    sides = (_plus(tip, perpendicular), _plus(tip, perpendicular, -1))
    landing = _plus(center, direction, 2)
    write = {center, tip, *arms}
    if not all(_inside(p, shape) for p in write | set(sides) | {landing}):
        return None
    return dict(center=center, direction=direction, tip=tip, arms=arms,
                sides=sides, landing=landing, write=write)


def _paint(clean, ids, templates, arrows, ink, *, decorations=True, clipping=True):
    assignments = {}
    for arrow in arrows:
        target, payload = arrow['target'], arrow['payload']
        _need(target not in assignments or assignments[target] == payload,
              'conflicting incoming keys')
        assignments[target] = payload
    out = clean.copy()
    for target, payload in assignments.items():
        out[ids == target] = payload
    if decorations:
        for arrow in arrows:
            ar, ac = arrow['landing']
            for dr, dc in templates[arrow['payload']]:
                if (dr, dc) == (0, 0):
                    continue
                p = ar + dr, ac + dc
                if _inside(p, clean.shape) and (not clipping or ids[p] == arrow['target']):
                    out[p] = ink
    return out


def _analyze(input_grid):
    """Reconstruct the entire interpretation using this input alone."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _need(all(np.all(edge == bg) for edge in
              (grid[0], grid[-1], grid[:, 0], grid[:, -1])), 'missing exterior margin')
    frames = []
    for color in np.unique(grid):
        if color == bg:
            continue
        for cells in _components(grid == color):
            r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
            c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
            h, w = r1-r0+1, c1-c0+1
            if min(h, w) >= 7 and len(cells) == 2*h+2*w-4 and all(
                    r in (r0, r1) or c in (c0, c1) for r, c in cells):
                frames.append((r0, c0, h, w, int(color)))
    _need(len(frames) == 1, 'map frame must be unique')
    r0, c0, h, w, frame = frames[0]
    exterior = grid != bg
    exterior[r0:r0+h, c0:c0+w] = False
    references = _components(exterior)
    _need(bool(references), 'no visible dictionary')
    possible_inks = []
    for color in np.unique(grid[exterior]):
        if all(len(comp) >= 4 and sum(grid[p] == color for p in comp) == len(comp)-1
               for comp in references):
            possible_inks.append(int(color))
    _need(len(possible_inks) == 1, 'dictionary must identify one drawing ink')
    ink = possible_inks[0]
    _need(len({bg, frame, ink}) == 3, 'structural roles must be distinct')
    templates, repetitions = {}, {}
    for comp in references:
        anchor = next(p for p in comp if grid[p] != ink)
        key = int(grid[anchor])
        _need(key not in (bg, frame, ink), 'key collides with a structural role')
        offsets = tuple(sorted((r-anchor[0], c-anchor[1]) for r, c in comp))
        _need(key not in templates or templates[key] == offsets, 'inconsistent dictionary')
        templates[key] = offsets
        repetitions[key] = repetitions.get(key, 0) + 1
    _need(2 <= len(templates) <= 5, 'dictionary needs two to five keys')
    board = grid[r0+1:r0+h-1, c0+1:c0+w-1].copy()
    _need(set(map(int, np.unique(board))) <= set(templates) | {ink}, 'unknown map color')
    arrows = []
    for p in np.ndindex(board.shape):
        for d in D:
            geo = _geometry(p, d, board.shape)
            if geo is None or not all(board[q] == ink for q in (*geo['arms'], geo['tip'])):
                continue
            a, b = geo['sides']
            host, targetcolor = int(board[a]), int(board[geo['landing']])
            if host not in templates or board[b] != host or targetcolor not in templates or targetcolor == host:
                continue
            centercolor = int(board[p])
            if centercolor not in set(templates) | {ink}:
                continue
            arrows.append(dict(geo, host=host, explicit=centercolor != ink,
                               payload=host if centercolor == ink else centercolor))
    _need(bool(arrows), 'map has no arrows')
    used = set()
    clean = board.copy()
    for arrow in arrows:
        _need(not used & arrow['write'], 'ambiguous or overlapping T annotations')
        used.update(arrow['write'])
        for p in arrow['write']:
            clean[p] = arrow['host']
    _need({p for p in used if board[p] == ink} == set(map(tuple, np.argwhere(board == ink))),
          'unexplained map ink')
    ids, colors, cells = _regions(clean)
    for arrow in arrows:
        source, target = int(ids[arrow['center']]), int(ids[arrow['landing']])
        _need(source != target and all(ids[p] == source for p in arrow['write'] | set(arrow['sides'])),
              'arrow must lie within its original host')
        arrow.update(source=source, target=target)
    painted = _paint(clean, ids, templates, arrows, ink)
    out = np.full((h, w), frame, dtype=np.int8)
    out[1:-1, 1:-1] = painted
    return out, dict(grid=grid, box=(r0,c0,h,w), bg=bg, frame=frame, ink=ink,
                     templates=templates, repetitions=repetitions, clean=clean,
                     ids=ids, colors=colors, cells=cells, arrows=arrows, painted=painted)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _analyze(input_grid)[0]


def _shape(rng, extent):
    """Grow a new connected, asymmetric, nonsolid polyomino, then choose its root."""
    for _ in range(80):
        height, width = int(rng.integers(2, extent+1)), int(rng.integers(2, extent+1))
        count = int(rng.integers(4, min(12, height*width)+1))
        points = {(int(rng.integers(height)), int(rng.integers(width)))}
        while len(points) < count:
            frontier = sorted({_plus(p, d) for p in points for d in D
                               if _inside(_plus(p, d), (height, width))} - points)
            points.add(frontier[int(rng.integers(len(frontier)))])
        root = sorted(points)[int(rng.integers(len(points)))]
        offsets = tuple(sorted((r-root[0], c-root[1]) for r, c in points))
        rr, cc = zip(*offsets)
        if ((max(rr)-min(rr)+1)*(max(cc)-min(cc)+1) > count
                and set(offsets) != {(-c, r) for r,c in offsets}
                and set(offsets) != {(r, -c) for r,c in offsets}):
            return offsets
    return ((0,0),(0,1),(1,0),(2,0),(2,1))


def _legend(rng, templates, repetitions):
    ids = [key for key in templates for _ in range(repetitions[key])]
    rng.shuffle(ids)
    places, r, c, rowheight = [], 0, 0, 0
    for key in ids:
        rr, cc = zip(*templates[key])
        h, w = max(rr)-min(rr)+1, max(cc)-min(cc)+1
        if c+w > 28:
            r, c, rowheight = r+rowheight+1, 0, 0
        places.append((key, r-min(rr), c-min(cc)))
        c += w+1
        rowheight = max(rowheight, h)
    return (r+rowheight, max(ac+max(c for r,c in templates[k])+1 for k,ar,ac in places)), places


def _partition(rng, h, w, count, subdivide):
    bands = count-1 if subdivide else count
    widths = np.full(bands, 3, dtype=int)
    for _ in range(w-3*bands):
        widths[int(rng.integers(bands))] += 1
    cuts = np.cumsum(widths)[:-1]
    shift = np.zeros(h, dtype=int)
    if rng.random() < .8:
        block = int(rng.integers(2, 5))
        for r in range(0, h, block):
            shift[r:r+block] = int(rng.integers(-1, 2))
    labels = np.zeros((h, w), dtype=np.int16)
    for r in range(h):
        labels[r] = np.searchsorted(cuts+shift[r], np.arange(w), side='right')
    if subdivide:
        host = int(rng.integers(bands))
        cut = int(rng.integers(4, h-3))
        labels[(labels == host) & (np.indices((h,w))[0] >= cut)] = count-1
    return labels


def _candidates(labels):
    result = []
    for p in np.ndindex(labels.shape):
        for d in D:
            geo = _geometry(p, d, labels.shape)
            if geo is None:
                continue
            source, target = int(labels[p]), int(labels[geo['landing']])
            if source != target and all(labels[q] == source for q in geo['write'] | set(geo['sides'])):
                result.append(dict(geo, source=source, target=target))
    return result


def _plan(rng, labels, colors, templates, mode):
    candidates = _candidates(labels)
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    arrows, assignments, blocked, contexts = [], {}, set(), set()

    def append(candidate, payload, explicit):
        target = candidate['target']
        if payload == colors[target] or (target in assignments and assignments[target] != payload):
            return False
        if candidate['write'] & (blocked | contexts):
            return False
        used = {p for a in arrows for p in a['write']}
        if (set(candidate['sides']) | {candidate['landing']}) & used:
            return False
        ar, ac = candidate['landing']
        offsets = templates[payload]
        retained = sum(_inside((ar+r,ac+c), labels.shape) and labels[ar+r,ac+c] == target
                       for r,c in offsets if (r,c) != (0,0))
        if not 0 < retained < len(offsets)-1:
            return False
        arrows.append(dict(candidate, payload=int(payload), explicit=bool(explicit), host=int(colors[candidate['source']])))
        assignments[target] = payload
        for p in candidate['write']:
            blocked.add(p)
            blocked.update(_plus(p,d) for d in D)
        contexts.update(candidate['sides'])
        contexts.add(candidate['landing'])
        return True

    if mode == 'chain':
        for source, target in ((0,1),(1,2)):
            if not any(append(c, colors[source], False) for c in candidates
                       if c['source'] == source and c['target'] == target):
                return None
    elif mode == 'local':
        novel = next(k for k in templates if k not in colors)
        for _ in range(2):
            if not any(append(c, novel, True) for c in candidates if c['source'] == 1 and c['target'] == 0):
                return None
        if not any(append(c, colors[2], False) for c in candidates if c['source'] == 2 and c['target'] == 3):
            return None
    else:
        desired = int(rng.integers(2, min(7, len(colors)+2)))
        for c in candidates:
            if len(arrows) >= desired:
                break
            explicit = rng.random() < .35
            payload = int(rng.choice(list(templates))) if explicit else colors[c['source']]
            append(c, payload, explicit)
        if len(arrows) < 2:
            return None
    return arrows


def _pair(rng, mode, is_test):
    for _ in range(100):
        nkeys = 3 if mode == 'chain' else 4 if mode == 'local' else int(rng.integers(2,5))
        palette = list(map(int, rng.permutation(10)))
        bg, frame, ink = palette[:3]
        keys = palette[3:3+nkeys]
        extent = 5 if is_test else int(rng.integers(3,5))
        templates = {}
        for key in keys:
            for _ in range(20):
                shape = _shape(rng, extent)
                if shape not in templates.values():
                    templates[key] = shape
                    break
            else:
                break
        if len(templates) != nkeys:
            continue
        repetitions = {k: 2 if mode == 'chain' else int(rng.integers(1,3)) for k in keys}
        (lh,lw), places = _legend(rng, templates, repetitions)
        maxh = 30-lh-5  # outer margins, one gap, map frame
        if maxh < 10:
            continue
        h = int(rng.integers(10, min(20,maxh)+1))
        count = 3 if mode == 'chain' else 4 if mode == 'local' else int(rng.integers(4 if is_test else 2,7))
        w = int(rng.integers(max(17,3*count),27))
        subdivide = mode not in ('chain','local') and count >= 4 and rng.random() < .65
        labels = _partition(rng,h,w,count,subdivide)
        if mode == 'chain':
            colors = keys[:3]
        elif mode == 'local':
            colors = [keys[0],keys[1],keys[0],keys[2]]
        else:
            colors = [-1]*count
            adjacency = {i:set() for i in range(count)}
            for a,b in ((labels[:-1],labels[1:]),(labels[:,:-1],labels[:,1:])):
                for x,y in zip(a[a != b],b[a != b]):
                    adjacency[int(x)].add(int(y))
                    adjacency[int(y)].add(int(x))
            for i in range(count):
                legal = [k for k in keys if k not in {colors[j] for j in adjacency[i]}]
                if not legal:
                    break
                colors[i] = int(rng.choice(legal))
            if -1 in colors:
                continue
        arrows = _plan(rng,labels,colors,templates,mode)
        if arrows is None:
            continue
        clean = np.asarray(colors,dtype=np.int8)[labels]
        board = clean.copy()
        for arrow in arrows:
            for p in arrow['write']:
                board[p] = ink
            if arrow['explicit']:
                board[arrow['center']] = arrow['payload']
        out = np.full((h+2,w+2),frame,dtype=np.int8)
        out[1:-1,1:-1] = _paint(clean,labels,templates,arrows,ink)
        gh, gw = lh+h+5, max(lw,w+2)+2
        grid = np.full((gh,gw),bg,dtype=np.int8)
        legendleft = int(rng.integers(1,gw-lw))
        for key,ar,ac in places:
            for dr,dc in templates[key]:
                grid[1+ar+dr,legendleft+ac+dc] = key if (dr,dc) == (0,0) else ink
        mr, mc = lh+2, int(rng.integers(1,gw-w-1))
        grid[mr:mr+h+2,mc:mc+w+2] = frame
        grid[mr+1:mr+h+1,mc+1:mc+w+1] = board
        # Rotate the entire scene and dictionary together; stamping itself is
        # always translation in the orientation actually visible in this input.
        turn = int(rng.integers(4))
        grid, out = np.rot90(grid,turn).copy(), np.rot90(out,turn).copy()
        try:
            prediction, info = _analyze(grid)
            _need(np.array_equal(prediction,out), 'constructor/reader mismatch')
            features = _features(info)
            _need(features['visible_clip'], 'no observable clipping contrast')
            if is_test:
                used_shapes = [templates[a['payload']] for a in arrows]
                _need(any(max(max(r for r,c in s)-min(r for r,c in s)+1,
                                  max(c for r,c in s)-min(c for r,c in s)+1) == 5
                          for s in used_shapes), 'test must use a five-cell-extent stencil')
            if mode == 'chain':
                _need(all(features[k] for k in ('chain','cascade_diff','keyed','transparent',
                                               'original_boundary_matters')), 'missing chain curriculum contrast')
            if mode == 'local':
                _need(features['locality'] and features['novel'] and features['multi'], 'missing local curriculum')
            return dict(input=grid, output=out)
        except AssertionError:
            continue
    raise RuntimeError('bounded pair construction exhausted')


def _features(info):
    arrows, colors, templates = info['arrows'], info['colors'], info['templates']
    assignments = {a['target']: a['payload'] for a in arrows}
    final = [assignments.get(i,c) for i,c in enumerate(colors)]
    plain = _paint(info['clean'],info['ids'],templates,arrows,info['ink'],decorations=False)
    unclip = _paint(info['clean'],info['ids'],templates,arrows,info['ink'],clipping=False)
    # These alternatives isolate conceptual shortcuts, rather than merely
    # comparing a label with the same renderer that constructed it.
    no_a = [dict(a,payload=colors[a['target']]) for a in arrows]
    ignore_a = _paint(info['clean'],info['ids'],templates,no_a,info['ink'])
    cascaded = [dict(a,payload=(a['payload'] if a['explicit'] else
                               assignments.get(a['source'],a['payload']))) for a in arrows]
    try:
        cascade_diff = not np.array_equal(_paint(info['clean'],info['ids'],templates,cascaded,info['ink']),info['painted'])
    except AssertionError:
        cascade_diff = True  # A previously consistent instruction set conflicts.
    newids, _, _ = _regions(plain)
    merged = [dict(a,target=int(newids[a['landing']])) for a in arrows]
    merged_diff = not np.array_equal(_paint(plain,newids,templates,merged,info['ink']),info['painted'])
    keyed = all(not np.array_equal(_paint(info['clean'],info['ids'],
                        {k:shape for k in templates},arrows,info['ink']),info['painted'])
                for shape in templates.values())
    opaque = info['painted'].copy()
    for a in arrows:
        shape = templates[a['payload']]
        rr, cc = zip(*shape)
        for r in range(min(rr),max(rr)+1):
            for c in range(min(cc),max(cc)+1):
                p = a['landing'][0]+r,a['landing'][1]+c
                if ((r,c) not in shape and _inside(p,opaque.shape)
                        and info['ids'][p] == a['target']):
                    opaque[p] = info['bg']
    nonrect = 0
    for component in info['cells']:
        rows, cols = zip(*component)
        nonrect += len(component) != (max(rows)-min(rows)+1)*(max(cols)-min(cols)+1)
    return dict(chain=any(not a['explicit'] and a['source'] in assignments
                          and assignments[a['source']] != a['payload'] for a in arrows),
                locality=any(colors[i] == colors[j] and final[i] != final[j]
                             for i in range(len(colors)) for j in range(i)),
                novel=any(a['explicit'] and a['payload'] not in colors for a in arrows),
                multi=len(assignments) < len(arrows),
                visible_clip=not np.array_equal(unclip,info['painted']),
                motif_matters=not np.array_equal(plain,info['painted']),
                transfer_matters=not np.array_equal(ignore_a,info['painted']),
                cascade_diff=cascade_diff,original_boundary_matters=merged_diff,
                keyed=keyed,transparent=not np.array_equal(opaque,info['painted']),
                directions={D.index(a['direction']) for a in arrows},
                territories=len(colors), arrows=len(arrows), keys=len(templates),
                nonrect=nonrect)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _int(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _int(num_train) or not 2 <= num_train <= 10 or not _int(num_test)
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train':[], 'test':[]}, set()
    for split,count in (('train',num_train),('test',num_test)):
        for i in range(count):
            mode = 'chain' if split == 'train' and i == 0 else 'local' if split == 'train' and i == 1 else 'free'
            for _ in range(8):
                pair = _pair(rng,mode,split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
    # The two compact demonstrations precede optional mixed examples.
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, 'wrong puzzle schema')
    _need(all(isinstance(puzzle[s],list) for s in puzzle), 'splits must be lists')
    a,b = len(puzzle['train']),len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal counts')
    seen, training = set(), []
    for split in ('train','test'):
        for pair in puzzle[split]:
            _need(isinstance(pair,dict) and set(pair) == {'input','output'}, 'wrong pair schema')
            inp,out = _grid(pair['input']),_grid(pair['output'])
            _need(_key(inp) not in seen, 'duplicate inputs')
            seen.add(_key(inp))
            prediction, info = _analyze(inp)
            _need(np.array_equal(prediction,out), 'incorrect transfer, stencil, clipping or frame')
            f = _features(info)
            _need(f['motif_matters'] and f['visible_clip'] and f['transfer_matters'],
                  'degenerate transfer, pattern or clipping')
            _need(np.count_nonzero(info['painted'] != info['ink']) > len(info['arrows']), 'fills obscured')
            for arrow in info['arrows']:
                ar,ac = arrow['landing']
                kept = [(_inside((ar+r,ac+c),info['clean'].shape)
                         and info['ids'][ar+r,ac+c] == arrow['target'])
                        for r,c in info['templates'][arrow['payload']] if (r,c) != (0,0)]
                _need(any(kept) and not all(kept), 'every generated stamp must demonstrate clipping')
            if split == 'train':
                training.append(f)
    for feature in ('chain','locality','novel','multi','cascade_diff','keyed','transparent',
                    'original_boundary_matters'):
        _need(any(f[feature] for f in training), 'training lacks '+feature)


def self_test():
    """42 fixed seeds, grid-derived coverage, counterfactuals, and metamorphic tests.

    Retain only bounded hashes/features and one current puzzle. No disk writes.
    """
    signatures, shapes, geometries, counts, directions = set(),set(),set(),set(),set()
    totals = dict(chain=0,locality=0,novel=0,multi=0,visible_clip=0,nonrect=0,
                  transfer_matters=0,motif_matters=0,cascade_diff=0,keyed=0,
                  transparent=0,original_boundary_matters=0)
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp, out = pair['input'],pair['output']
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                prediction, info = _analyze(inp)
                _need(np.array_equal(prediction,out), 'input-only solving failed')
                f = _features(info)
                for k in totals:
                    totals[k] += bool(f[k])
                shapes.add(inp.shape)
                counts.add((f['territories'],f['arrows'],f['keys']))
                directions.update(f['directions'])
                geometries.update(info['templates'].values())
        signatures.add(digest.hexdigest())
        if seed == 0:
            again = generate_puzzle(seed)
            _need(all(_key(p[k]) == _key(q[k]) for s in puzzle
                      for p,q in zip(puzzle[s],again[s]) for k in ('input','output')), 'nondeterminism')
            # Palette and whole-scene symmetries must not encode hidden rules.
            palette = np.array([7,3,0,9,1,8,4,2,6,5],dtype=np.int8)
            for p in puzzle['train']:
                for turn in range(4):
                    _need(np.array_equal(solve(np.rot90(p['input'],turn)),np.rot90(p['output'],turn)), 'rotation covariance')
                _need(np.array_equal(solve(palette[p['input']]),palette[p['output']]), 'color covariance')
                _need(np.array_equal(solve(np.fliplr(p['input'])),np.fliplr(p['output'])), 'reflection covariance')
            bad = {s:[{k:v.copy() for k,v in p.items()} for p in puzzle[s]] for s in puzzle}
            bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('corrupt output accepted')
    _need(len(signatures) == 42 and len(shapes) >= 20 and len(geometries) >= 120,
          'insufficient measured structural variation')
    _need(len(counts) >= 25 and directions == set(range(4)), 'collapsed graphs or directions')
    _need(all(v >= 20 for v in totals.values()), 'missing family-specific variation')
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    return dict(seeds=42,pairs=210,input_shapes=len(shapes),template_geometries=len(geometries),
                graph_count_profiles=len(counts),directions=sorted(directions),features=totals)


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
