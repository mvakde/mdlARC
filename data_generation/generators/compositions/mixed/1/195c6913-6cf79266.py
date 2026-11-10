"""Cyclic staircases consume blank space before exhaustive 3x3 union marking.

Only NumPy and the standard library are used. No files, caches, or traces are
written by this module. Semantic roles and both directions are input-visible.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 / 1..4 with num_train > num_test")
    return int(a), int(b)


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _orient(g, code):
    return np.rot90(g[:, ::-1] if code >= 4 else g, code % 4).copy()


def _undo(g, code):
    g = np.rot90(g, -(code % 4))
    return (g[:, ::-1] if code >= 4 else g).copy()


def _components(g, color):
    seen = np.zeros(g.shape, bool)
    result = []
    for rr, cc in np.argwhere(g == color):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if _inside(g.shape, p) and not seen[p] and g[p] == color:
                    seen[p] = True
                    stack.append(p)
        result.append(cells)
    return result


def _swatches(g, terrains):
    found = []
    for color in np.unique(g):
        if int(color) in terrains:
            continue
        for cells in _components(g, color):
            r, c = min(p[0] for p in cells), min(p[1] for p in cells)
            if set(cells) == {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}:
                found.append((r, c, int(color)))
    return found


def _decode(input_grid):
    g = np.asarray(input_grid)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer)
            or not all(1 <= n <= 30 for n in g.shape)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("invalid grid")
    colors, counts = np.unique(g, return_counts=True)
    if len(colors) < 5:
        raise ValueError("missing terrain or annotation roles")
    terrains = set(map(int, colors[np.argsort(counts)[-2:]]))
    symbols = _swatches(g, terrains)
    annotation = {(r+dr, c+dc) for r, c, _ in symbols
                  for dr in range(2) for dc in range(2)}
    starts = [tuple(map(int, p)) for p in np.argwhere(~np.isin(g, list(terrains)))
              if tuple(p) not in annotation]
    if not (3 <= len(symbols) <= 7 and 1 <= len(starts) <= 3
            and len({int(g[p]) for p in starts}) == 1):
        raise ValueError("unrecognizable swatches or singleton starts")
    rings = {int(g[p]) for r, c in annotation
             for p in ((r-1,c), (r+1,c), (r,c-1), (r,c+1))
             if _inside(g.shape, p) and p not in annotation}
    if len(rings) != 1 or not rings <= terrains:
        raise ValueError("annotations must have one obstacle-terrain surround")
    obstacle = next(iter(rings))
    blank = next(iter(terrains - {obstacle}))
    restored, labels = g.copy(), np.zeros(g.shape, np.int8)
    for p in annotation:
        restored[p] = obstacle
    for p in starts:
        restored[p] = blank
        labels[p] = 1
    candidates = []
    for code in range(8):
        terrain = _undo(restored, code)
        cg = _undo(g, code)
        cs = tuple(map(tuple, np.argwhere(_undo(labels, code))))
        ss = _swatches(cg, terrains)
        mask = terrain == blank
        for marker in ss:
            legend = sorted((s for s in ss if s != marker), key=lambda s: s[1])
            if len({s[0] for s in legend}) != 1:
                continue
            if any(b[1] - a[1] < 3 for a, b in zip(legend, legend[1:])):
                continue
            word = tuple(s[2] for s in legend)
            if marker[2] in word or word[0] != int(g[starts[0]]):
                continue
            if not all(r >= legend[0][0] + 4 and c+1 < mask.shape[1]
                       and mask[r,c+1] and (c == 0 or not mask[r,c-1]) for r,c in cs):
                continue
            candidates.append(dict(code=code, terrain=terrain, mask=mask,
                                   starts=cs, word=word, marker=marker[2],
                                   legend=legend, marker_swatch=marker,
                                   blank=blank, obstacle=obstacle))
    if len(candidates) != 1:
        raise ValueError("legend reading and travel directions must be unique")
    return candidates[0]


def _trace(mask, start):
    """Maximal right/up legs; a bend cell occurs once; an edge exit is final."""
    p, direction = start, 0
    path, hits, legs = [], [], []
    while True:
        dr, dc = ((0, 1), (-1, 0))[direction]
        leg = []
        while _inside(mask.shape, p) and mask[p]:
            leg.append(p)
            p = (p[0]+dr, p[1]+dc)
        assert leg
        path.extend(leg)
        legs.append(len(leg))
        if not _inside(mask.shape, p):
            end = "edge"
            break
        hits.append(p)
        direction = 1-direction
        dr, dc = ((0, 1), (-1, 0))[direction]
        p = (leg[-1][0]+dr, leg[-1][1]+dc)
        if not _inside(mask.shape, p) or not mask[p]:
            end = "blocked"
            break
    return dict(path=tuple(path), hits=tuple(hits), legs=tuple(legs), end=end)


def _walk(mask, start):
    """Cellwise validation of the ray-based tracer, with separate control flow."""
    p, vertical, path, hits, legs = start, False, [], [], [0]
    while True:
        assert mask[p] and p not in path
        path.append(p)
        legs[-1] += 1
        ahead = (p[0]-1, p[1]) if vertical else (p[0], p[1]+1)
        if not _inside(mask.shape, ahead):
            end = "edge"
            break
        if mask[ahead]:
            p = ahead
            continue
        hits.append(ahead)
        nxt = (p[0], p[1]+1) if vertical else (p[0]-1, p[1])
        if not _inside(mask.shape, nxt) or not mask[nxt]:
            end = "blocked"
            break
        p, vertical = nxt, not vertical
        legs.append(0)
    return dict(path=tuple(path), hits=tuple(hits), legs=tuple(legs), end=end)


def _windows(mask):
    h, w = mask.shape
    if min(h, w) < 3:
        return []
    good = np.ones((h-2, w-2), bool)
    for dr in range(3):
        for dc in range(3):
            good &= mask[dr:dr+h-2, dc:dc+w-2]
    return [tuple(map(int, p)) for p in np.argwhere(good)]


def _union(shape, starts):
    result = np.zeros(shape, bool)
    for r, c in starts:
        result[r:r+3, c:c+3] = True
    return result


def _alternative_square_union(mask, size):
    """Only for checking that demonstrations distinguish other square sizes."""
    h, w = mask.shape
    good = np.ones((h-size+1,w-size+1),bool)
    for dr in range(size):
        for dc in range(size):
            good &= mask[dr:dr+h-size+1,dc:dc+w-size+1]
    result = np.zeros(mask.shape,bool)
    for r,c in np.argwhere(good):
        result[r:r+size,c:c+size] = True
    return result


def _paint(info, records):
    stage = info['terrain'].copy()
    word = info['word']
    for rec in records:
        for i, p in enumerate(rec['path']):
            stage[p] = word[i % len(word)]
        for p in rec['hits']:
            stage[p] = info['marker']
    return stage


def _analyze(g):
    info = _decode(g)
    records = [_trace(info['mask'], s) for s in info['starts']]
    stage = _paint(info, records)
    windows = _windows(stage == info['blank'])
    footprint = _union(stage.shape, windows)
    result = stage.copy()
    result[footprint] = info['marker']
    info.update(records=records, stage=stage, windows=windows,
                footprint=footprint, result=_orient(result, info['code']))
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone, including legend removal before scanning."""
    return _analyze(input_grid)['result']


def _evidence(info):
    word, records = info['word'], info['records']
    unique = tuple(dict.fromkeys(word))
    evidence, offset = set(), 0
    for rec in records:
        n = len(rec['path'])
        actual = [word[i % len(word)] for i in range(n)]
        if n >= 2*len(word):
            evidence.add('cycling')
        if actual != [unique[i % len(unique)] for i in range(n)]:
            evidence.add('multiplicity')
        reset, consume, phase = [], [], 0
        for j, length in enumerate(rec['legs']):
            reset.extend(word[i % len(word)] for i in range(length))
            consume.extend(word[(phase+i+j) % len(word)] for i in range(length))
            phase += length
        if actual != reset:
            evidence.add('bend_phase')
        if actual != consume:
            evidence.add('marker_phase')
        if actual != [word[(offset+i) % len(word)] for i in range(n)]:
            evidence.add('restart')
        offset += n
    footprint, windows, stage = info['footprint'], info['windows'], info['stage']
    before = _union(stage.shape, _windows(info['mask']))
    if np.any(before != footprint):
        evidence.add('route_dependency')
    if windows and np.any(footprint != _union(stage.shape, windows[:1])):
        evidence.add('all_windows')
    if 9*len(windows) > int(footprint.sum()):
        evidence.add('overlap')
    greedy = np.zeros(stage.shape, bool)
    for r,c in windows:
        if not np.any(greedy[r:r+3,c:c+3]):
            greedy[r:r+3,c:c+3] = True
    if np.any(greedy != footprint):
        evidence.add('overlap_union')
    if np.any((stage == info['blank']) & ~footprint):
        evidence.add('preserve_fringe')
    # Even selecting just qualifying connected components and filling them all
    # must give a wrong answer in an informative demonstration.
    component_fill = np.zeros(stage.shape, bool)
    for cells in _components(stage, info['blank']):
        if any(footprint[p] for p in cells):
            for p in cells:
                component_fill[p] = True
    if np.any(component_fill != footprint):
        evidence.add('component_not_target')
    for size in (2,4):
        if np.any(_alternative_square_union(stage == info['blank'],size) != footprint):
            evidence.add('size_' + str(size) + '_rejected')
    return evidence


_REQUIRED = {'cycling', 'multiplicity', 'bend_phase', 'marker_phase', 'restart',
             'route_dependency', 'all_windows', 'overlap', 'overlap_union',
             'preserve_fringe', 'component_not_target',
             'size_2_rejected', 'size_4_rejected'}


def _landscape(rng, h, w, clear, complex_scene):
    """Independent irregular boundaries plus bays, rooms and thin appendages."""
    k = int(rng.integers(4, 9))
    knots = np.linspace(4, h-1, k)
    left = np.rint(np.interp(np.arange(h), knots,
                            rng.integers(-3, max(2,w//2), k))).astype(int)
    right = np.rint(np.interp(np.arange(h), knots,
                             rng.integers(w//2, w+5, k))).astype(int)
    mask = np.zeros((h, w), bool)
    for r in range(4, h):
        mask[r, max(0,left[r]):min(w,max(left[r]+2,right[r]))] = True
    for _ in range(int(rng.integers(3, 10 if complex_scene else 7))):
        r, c = int(rng.integers(4,h-1)), int(rng.integers(w-1))
        dh, dw = int(rng.integers(1,7)), int(rng.integers(1,9))
        mask[r:r+dh, c:c+dw] = bool(rng.integers(2))
    mask[clear] = False
    return mask


def _backstop(rng, h, w, clear):
    """Finite constructive backup: wide upper staircase, shorter lower route."""
    a, d = int(rng.integers(5,11)), int(rng.integers(4,10))
    mask = np.zeros((h,w), bool)
    mask[5:h-6, :a] = True
    mask[5:8, :] = True
    mask[5:, :2] = True
    mask[h-4:, :d] = True
    mask[clear] = False
    return mask


def _choose_routes(rng, mask, row, count, witness):
    options = []
    for r in range(row+4, mask.shape[0]):
        for c in range(mask.shape[1]-1):
            if mask[r,c] and mask[r,c+1] and (c == 0 or not mask[r,c-1]):
                rec = _trace(mask, (r,c))
                if len(rec['path']) >= 3:
                    options.append(((r,c), rec))
    rng.shuffle(options)
    if witness:
        options.sort(key=lambda x: (len(x[1]['legs']) > 1, len(x[1]['path'])), reverse=True)
    picked, occupied = [], set()
    for start, rec in options:
        cells = set(rec['path']) | set(rec['hits'])
        if cells & occupied or any(abs(start[0]-s[0])+abs(start[1]-s[1]) < 3 for s,_ in picked):
            continue
        picked.append((start,rec))
        occupied |= cells
        if len(picked) == count:
            return sorted(picked)
    return []


def _sample_word(rng, n, witness):
    distinct = int(rng.integers(2, n if witness else n+1))
    word = list(range(distinct)) + list(map(int,rng.integers(distinct, size=n-distinct)))
    rng.shuffle(word)
    return tuple(word)


def _make_pair(rng, test, shift, code, witness):
    if shift == 0:
        low, high = (26,30) if test else (20,25)
    else:
        low, high = 20,30
    h, w = int(rng.integers(low,high+1)), int(rng.integers(low,high+1))
    n = int(rng.integers(5,7)) if test and shift == 1 else int(rng.integers(3 if witness else 2,5))
    palette = list(map(int, rng.permutation(10)))
    blank, obstacle, marker = palette[:3]
    word = tuple(palette[3+x] for x in _sample_word(rng,n,witness))
    count = 2 if witness else int(rng.choice([1,2,3], p=[.43,.43,.14]))
    for attempt in range(160):
        # A fresh word on retry also avoids a degenerate repeated word as the
        # sole witness. No phase equality pattern is fixed by pair position.
        if witness and attempt:
            word = tuple(palette[3+x] for x in _sample_word(rng,n,True))
        row = int(rng.integers(0,3))
        gaps = [int(rng.integers(1,3)) for _ in range(n-1)]
        required_width = 2*n + sum(gaps)
        if required_width > w-1:
            gaps = [1]*(n-1)
            required_width = 3*n-1
        left = int(rng.integers(0,min(4,w-required_width)))
        cols = [left]
        for gap in gaps:
            cols.append(cols[-1]+2+gap)
        clear = np.zeros((h,w), bool)
        clear[max(0,row-1):row+3, max(0,left-1):cols[-1]+3] = True
        mask = (_landscape(rng,h,w,clear,test) if attempt < 100 else
                _backstop(rng,h,w,clear))
        # At most 27 cells use any annotation color. Both terrain roles are
        # deliberately more frequent, so identification never uses area rank.
        if min(int(mask.sum())-count, int((~mask).sum())-4*(n+1)) < 40:
            continue
        chosen = _choose_routes(rng,mask,row,count,witness)
        if not chosen:
            continue
        terrain = np.where(mask,blank,obstacle).astype(np.int8)
        records = [rec for _,rec in chosen]
        info = dict(terrain=terrain, mask=mask, word=word, marker=marker, blank=blank,
                    records=records)
        stage = _paint(info,records)
        windows = _windows(stage == blank)
        footprint = _union(stage.shape,windows)
        info.update(stage=stage,windows=windows,footprint=footprint)
        evidence = _evidence(info)
        if not windows or 'route_dependency' not in evidence or (witness and not _REQUIRED <= evidence):
            continue
        occupied = clear.copy()
        for _,rec in chosen:
            for r,c in rec['path'] + rec['hits']:
                occupied[max(0,r-1):r+2,max(0,c-1):c+2] = True
        marker_options = [(r,c) for r in range(1,h-2) for c in range(1,w-2)
                          if abs(r-row) >= 4
                          and not np.any(mask[r-1:r+3,c-1:c+3])
                          and not np.any(occupied[r:r+2,c:c+2])]
        if not marker_options:
            continue
        mr,mc = marker_options[int(rng.integers(len(marker_options)))]
        inp = terrain.copy()
        for c,color in zip(cols,word):
            inp[row:row+2,c:c+2] = color
        inp[mr:mr+2,mc:mc+2] = marker
        for start,_ in chosen:
            inp[start] = word[0]
        out = stage.copy()
        out[footprint] = marker
        inp, out = _orient(inp,code), _orient(out,code)
        try:
            decoded = _decode(inp)
        except ValueError:
            continue
        if (decoded['code'] != code or decoded['word'] != word
                or decoded['starts'] != tuple(s for s,_ in chosen)):
            continue
        return {'input':inp,'output':out}
    raise RuntimeError("bounded construction could not find a decodable informative scene")


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train,num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift, base = int(rng.integers(2)), int(rng.integers(8))
    mixed = bool(rng.integers(2))
    witness_slot = int(rng.integers(nt))
    puzzle, seen = {'train':[], 'test':[]}, set()
    for split,count in (('train',nt),('test',ne)):
        for index in range(count):
            code = int(rng.integers(8)) if mixed else base
            for _ in range(12):
                pair = _make_pair(rng,split == 'test',shift,code,
                                  split == 'train' and index == witness_slot)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input retries exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen, evidence = set(), set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for g in (inp,out):
                assert g.ndim == 2 and np.issubdtype(g.dtype,np.integer)
                assert all(1 <= n <= 30 for n in g.shape) and np.all((0 <= g) & (g <= 9))
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert len(set(info['word'])) >= 2
            assert all(np.count_nonzero(inp == color) >= 40
                       for color in (info['blank'],info['obstacle']))
            canonical_inp = _undo(inp,info['code'])
            for r,c in info['starts']:
                assert all(canonical_inp[p] != info['word'][0]
                           for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                           if _inside(canonical_inp.shape,p))
            occupied = set()
            ann = {(r+dr,c+dc) for r,c,_ in info['legend']+[info['marker_swatch']]
                   for dr in range(2) for dc in range(2)}
            for start,rec in zip(info['starts'],info['records']):
                assert rec == _walk(info['mask'],start)
                cells = set(rec['path']) | set(rec['hits'])
                assert not (cells & occupied) and not (cells & ann)
                occupied |= cells
                assert all(info['terrain'][p] == info['obstacle'] for p in rec['hits'])
            stage, blank = info['stage'], info['blank']
            # Exhaustive reference scan checks full footprints, not centers,
            # greedy disjoint targets, or mutation during the scan.
            direct = np.zeros(stage.shape,bool)
            direct_starts = []
            for r in range(stage.shape[0]-2):
                for c in range(stage.shape[1]-2):
                    if np.all(stage[r:r+3,c:c+3] == blank):
                        direct_starts.append((r,c))
                        direct[r:r+3,c:c+3] = True
            assert direct_starts == info['windows'] and np.array_equal(direct,info['footprint'])
            canonical_out = _undo(out,info['code'])
            assert np.all(canonical_out[direct] == info['marker'])
            assert np.array_equal(canonical_out[~direct],stage[~direct])
            assert np.array_equal(out,info['result'])
            assert direct.any() and 'route_dependency' in _evidence(info)
            assert all(not direct[p] for p in occupied | ann)
            if split == 'train':
                evidence |= _evidence(info)
    assert _REQUIRED <= evidence, _REQUIRED-evidence


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train','test'):
        for pair in puzzle[split]:
            for role in ('input','output'):
                g = np.asarray(pair[role])
                digest.update(bytes(g.shape))
                digest.update(g.astype(np.uint8).tobytes())
    return digest.hexdigest()


def self_test():
    # Explicit semantic fixtures independent of the scene sampler.
    for shape,expected in (((3,4),12),((4,4),16),((2,8),0),
                           ((2,3),0),((3,2),0),((3,3),9)):
        mask = np.ones(shape,bool)
        assert int(_union(shape,_windows(mask)).sum()) == expected
    mask = np.ones((3,3),bool)
    mask[1,1] = False
    assert not _windows(mask)
    horizontal = np.zeros((5,8),bool)
    horizontal[3,:] = True
    horizontal[2,7] = True
    assert _trace(horizontal,(3,0))['path'] == tuple((3,c) for c in range(8))
    vertical = np.zeros((7,9),bool)
    vertical[5,:4] = True
    vertical[:6,3] = True
    vertical[0,3:] = True
    rec = _trace(vertical,(5,0))
    assert rec == _walk(vertical,(5,0)) and rec['path'][-1] == (0,3)
    assert (0,4) not in rec['path'] and rec['hits'] == ((5,4),)
    signatures, shapes, palettes, words, counts, orientations, windows, legs = (set() for _ in range(8))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ('train','test'):
            for pair in puzzle[split]:
                info = _analyze(pair['input'])
                assert np.array_equal(solve(pair['input']),pair['output'])
                shapes.add(pair['input'].shape)
                palettes.add((info['blank'],info['obstacle'],info['marker']))
                mapping = {}
                words.add(tuple(mapping.setdefault(c,len(mapping)) for c in info['word']))
                counts.add(len(info['starts']))
                orientations.add(info['code'])
                windows.add(len(info['windows']))
                legs.update(len(r['legs']) for r in info['records'])
    assert len(signatures) == 42 and len(shapes) >= 35 and len(palettes) >= 100
    assert len(words) >= 12 and counts == {1,2,3} and orientations == set(range(8))
    assert len(windows) >= 25 and {1,2,3} <= legs
    assert _signature(generate_puzzle(314159,7,3)) == _signature(generate_puzzle(314159,7,3))
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(9000+a*10+b,a,b)
            assert len(p['train']) == a and len(p['test']) == b
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(2.0,1),(3,True),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds; all legal count combinations)')
