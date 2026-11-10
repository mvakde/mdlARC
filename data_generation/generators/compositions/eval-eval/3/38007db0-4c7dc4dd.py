"""Echo brushes: locally demonstrated toggle stencils acting on tile differences.

Only NumPy and the Python standard library are used. No parent module or data is
loaded. solve() is input-only; generation builds labels from its logical masks.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "bad color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _assemble(rows, lattice):
    h, w = rows[0][0].shape
    columns = len(rows[0])
    out = np.full((len(rows) * (h + 1) + 1, columns * (w + 1) + 1),
                  lattice, dtype=np.int8)
    for b, row in enumerate(rows):
        assert len(row) == columns
        for c, tile in enumerate(row):
            assert tile.shape == (h, w)
            r0, c0 = b * (h + 1) + 1, c * (w + 1) + 1
            out[r0:r0+h, c0:c0+w] = tile
    return out


def _connected(points):
    todo = set(points)
    stack = [todo.pop()]
    while stack:
        r, c = stack.pop()
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if p in todo:
                todo.remove(p)
                stack.append(p)
    return not todo


def _parse(input_grid):
    a = _grid(input_grid)
    lattice = int(a[0, 0])
    rs = np.flatnonzero(np.all(a == lattice, axis=1)).tolist()
    cs = np.flatnonzero(np.all(a == lattice, axis=0)).tolist()
    _require(len(rs) >= 3 and len(cs) >= 4, "need calibration, queries, and a strict majority")
    _require(rs[0] == cs[0] == 0 and rs[-1] == a.shape[0]-1
             and cs[-1] == a.shape[1]-1, "incomplete outer lattice")
    _require(len(set(np.diff(rs))) == len(set(np.diff(cs))) == 1, "irregular lattice")
    h, w = rs[1]-1, cs[1]-1
    _require(h >= 4 and w >= 4, "tiles too small")
    bands = []
    for top, bottom in zip(rs, rs[1:]):
        row = [a[top+1:bottom, left+1:right].copy() for left, right in zip(cs, cs[1:])]
        _require(all(np.all(p != lattice) for p in row), "lattice color inside tile")
        counts = Counter(_key(p) for p in row)
        _require(len(counts) == 2 and sorted(counts.values()) == [1, len(row)-1],
                 "band must have exactly one visible exception")
        common_key = max(counts, key=counts.get)
        column = next(i for i, p in enumerate(row) if _key(p) != common_key)
        common = next(p for p in row if _key(p) == common_key)
        bands.append(dict(common=common, exception=row[column], column=column, tiles=row))
    colors = set(map(int, np.unique(a)))
    _require(len(colors) == 4, "need exactly four distinct color roles")

    candidates = []
    for b, band in enumerate(bands):
        common, exception = band['common'], band['exception']
        values, frequencies = np.unique(common, return_counts=True)
        if len(values) != 2 or sorted(map(int, frequencies)) != [1, h*w-1]:
            continue
        pivot = int(values[np.argmin(frequencies)])
        neutral = int(values[np.argmax(frequencies)])
        if any(np.any(p == pivot) for j, other in enumerate(bands) if j != b
               for p in other['tiles']):
            continue
        anchor = tuple(map(int, np.argwhere(common == pivot)[0]))
        if not all(np.count_nonzero(p == pivot) == 1 and p[anchor] == pivot for p in band['tiles']):
            continue
        remaining = colors - {lattice, pivot, neutral}
        if len(remaining) != 1:
            continue
        ink = remaining.pop()
        if set(map(int, np.unique(exception))) != {neutral, ink, pivot}:
            continue
        brush = {(0, 0)} | {(int(r)-anchor[0], int(c)-anchor[1])
                            for r, c in np.argwhere(exception == ink)}
        if not 3 <= len(brush) <= 7 or any(max(abs(r), abs(c)) > 2 for r, c in brush):
            continue
        if not _connected(brush):
            continue
        query_ok = all(set(map(int, np.unique(other['common']))) == {neutral, ink}
                       and set(map(int, np.unique(other['exception']))) <= {neutral, ink}
                       for j, other in enumerate(bands) if j != b)
        if query_ok:
            candidates.append(dict(reference=b, pivot=pivot, neutral=neutral, ink=ink,
                                   anchor=anchor, brush=tuple(sorted(brush))))
    _require(len(candidates) == 1, "calibration roles must be uniquely readable")
    scene = candidates[0]
    scene.update(grid=a, lattice=lattice, h=h, w=w, copies=len(cs)-1, bands=bands)
    return scene


def _coverage(shape, triggers, brush):
    """Integer counts, including separate contributions that coincide."""
    h, w = shape
    hits = np.zeros(shape, dtype=np.int16)
    clipped = 0
    for r, c in triggers:
        for dr, dc in brush:
            y, x = r+dr, c+dc
            if 0 <= y < h and 0 <= x < w:
                hits[y, x] += 1
            else:
                clipped += 1
    return hits, clipped


def _solve_scene(scene):
    rows = []
    for b, band in enumerate(scene['bands']):
        if b == scene['reference']:
            continue
        common, exception = band['common'], band['exception']
        triggers = [tuple(map(int, p)) for p in np.argwhere(common != exception)]
        hits, _ = _coverage(common.shape, triggers, scene['brush'])
        result = common.copy()
        odd = hits % 2 == 1
        result[odd] = np.where(common[odd] == scene['neutral'], scene['ink'], scene['neutral'])
        rows.append([result])
    return _assemble(rows, scene['lattice'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive a missing companion for each query using only the input pixels."""
    return _solve_scene(_parse(input_grid))


def _geometry(rng, queries=None, witness=False):
    choices = []
    for h in range(4, 8):
        for w in range(4, 9):
            for q in ([3] if witness else ([queries] if queries is not None else range(1, 5))):
                for c in range(3, 6):
                    if (q+1)*(h+1)+1 <= 30 and c*(w+1)+1 <= 30:
                        choices.append((h, w, q, c))
    return choices[int(rng.integers(len(choices)))]


def _brush(rng, shape, anchor, size=None):
    h, w = shape
    ar, ac = anchor
    brush = {(0, 0)}
    target = int(rng.integers(3, 8)) if size is None else size
    while len(brush) < target:
        border = sorted({(r+dr, c+dc) for r, c in brush
                         for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
                         if -2 <= r+dr <= 2 and -2 <= c+dc <= 2
                         and 0 <= ar+r+dr < h and 0 <= ac+c+dc < w} - brush)
        # Even a corner has at least nine feasible offsets, so target <= 7 fits.
        assert border
        brush.add(border[int(rng.integers(len(border)))])
    return tuple(sorted(brush))


def _mask(rng, shape):
    h, w = shape
    rr, cc = np.indices(shape)
    mode = int(rng.integers(6))
    if mode == 0:
        result = rng.random(shape) < float(rng.uniform(.15, .85))
    elif mode == 1:
        result = np.zeros(shape, bool)
        result[int(rng.integers(h)), int(rng.integers(w))] = True
        target = int(rng.integers(3, h*w-1))
        while int(result.sum()) < target:
            border = [(r, c) for r in range(h) for c in range(w) if not result[r, c]
                      and any(0 <= y < h and 0 <= x < w and result[y, x]
                              for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))]
            result[border[int(rng.integers(len(border)))]] = True
    elif mode == 2:
        r0, r1 = sorted(map(int, rng.choice(h, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(w, 2, replace=False)))
        result = (((rr == r0) | (rr == r1)) & (cc >= c0) & (cc <= c1))
        result |= (((cc == c0) | (cc == c1)) & (rr >= r0) & (rr <= r1))
    elif mode == 3:
        period = int(rng.integers(2, 4))
        result = (rr * int(rng.integers(3)) + cc) % period == int(rng.integers(period))
    elif mode == 4:
        spine = int(rng.integers(w))
        result = cc == spine
        for r in rng.choice(h, int(rng.integers(1, h+1)), replace=False):
            end = int(rng.integers(w))
            result[r, min(spine, end):max(spine, end)+1] = True
    else:
        result = ((rr+cc) % 2 == int(rng.integers(2))) & (rng.random(shape) < .75)
    if rng.random() < .2:
        result |= result[:, ::-1]
    if rng.random() < .2:
        result |= result[::-1]
    if rng.random() < .25:
        result = ~result
    if result.all():
        result[0, 0] = False
    if not result.any():
        result[-1, -1] = True
    return result


def _overlap_pair(rng, shape, brush):
    """Pick two actual triggers with a double hit and a nontrivial extension."""
    positions = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    order = rng.permutation(len(positions))
    for i in order:
        p = positions[int(i)]
        for j in rng.permutation(len(positions)):
            q = positions[int(j)]
            if p == q:
                continue
            hits, _ = _coverage(shape, (p, q), brush)
            delta = np.zeros(shape, bool)
            delta[p] = delta[q] = True
            if np.any(hits == 2) and not np.array_equal(hits % 2 == 1, delta):
                return (p, q)
    raise AssertionError("connected nontrivial brush has no overlap witness")


def _triple(rng, shape, brush):
    centers = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    for index in rng.permutation(len(centers)):
        r, c = centers[int(index)]
        sources = sorted((r-dr, c-dc) for dr, dc in brush
                         if 0 <= r-dr < shape[0] and 0 <= c-dc < shape[1])
        if len(sources) >= 3:
            selected = [sources[int(i)] for i in rng.choice(len(sources), 3, replace=False)]
            # Extra events must not turn this three-hit witness into an even
            # four-hit cell. Higher multiplicities may still occur elsewhere.
            extras = [p for p in centers if p not in selected
                      and (r-p[0], c-p[1]) not in brush]
            more = int(rng.integers(0, 4))
            selected.extend(extras[int(i)] for i in rng.choice(len(extras), more, replace=False))
            return tuple(selected)
    raise AssertionError("brush cannot make triple overlap")


def _edge(rng, shape, brush):
    h, w = shape
    candidates = [(r, c) for r in range(h) for c in range(w)
                  if r in (0, h-1) or c in (0, w-1)]
    candidates = [p for p in candidates if _coverage(shape, (p,), brush)[1] > 0]
    return candidates[int(rng.integers(len(candidates)))]


def _query(rng, shape, triggers, forced=None):
    master = _mask(rng, shape)
    for p, value in (forced or {}).items():
        master[p] = value
    # Keep both roles visible even after forced values on tiny sparse silhouettes.
    free = [(r, c) for r in range(shape[0]) for c in range(shape[1])
            if (r, c) not in (forced or {})]
    if master.all():
        master[free[0]] = False
    if not master.any():
        master[free[-1]] = True
    exception = master.copy()
    for p in triggers:
        exception[p] = not exception[p]
    return master, exception


def _render(geometry, palette, brush, anchor, queries, reference, columns):
    h, w, q, copies = geometry
    lattice, neutral, ink, pivot = palette
    calibration = np.full((h, w), neutral, dtype=np.int8)
    calibration[anchor] = pivot
    companion = calibration.copy()
    for dr, dc in brush:
        if (dr, dc) != (0, 0):
            companion[anchor[0]+dr, anchor[1]+dc] = ink
    row_tiles, answers = [], []
    query_index = 0
    for b in range(q+1):
        if b == reference:
            common, exception = calibration, companion
        else:
            master_mask, exception_mask = queries[query_index]
            query_index += 1
            common = np.where(master_mask, ink, neutral).astype(np.int8)
            exception = np.where(exception_mask, ink, neutral).astype(np.int8)
            # Label construction: apply each event successively as an involution.
            # solve instead extracts pixels and uses integer coverage parity.
            answer = master_mask.copy()
            for r, c in np.argwhere(master_mask != exception_mask):
                for dr, dc in brush:
                    y, x = int(r)+dr, int(c)+dc
                    if 0 <= y < h and 0 <= x < w:
                        answer[y, x] = not answer[y, x]
            answers.append([np.where(answer, ink, neutral).astype(np.int8)])
        row = [common.copy() for _ in range(copies)]
        row[columns[b]] = exception.copy()
        row_tiles.append(row)
    return dict(input=_assemble(row_tiles, lattice), output=_assemble(answers, lattice))


def _twins(rng, palette):
    geometry = _geometry(rng, witness=True)
    h, w, q, copies = geometry
    shape = h, w
    anchor = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
    brush1 = _brush(rng, shape, anchor)
    # Guaranteed finite different footprint, without a color-only escape hatch.
    brush2 = None
    for _ in range(32):
        candidate = _brush(rng, shape, anchor)
        if candidate != brush1:
            brush2 = candidate
            break
    if brush2 is None:
        # Connected subset of three cells differs from a 4..7 cell brush;
        # otherwise grow its frontier by one cell.
        if len(brush1) > 3:
            for subset in combinations(brush1, 3):
                if (0, 0) in subset and _connected(subset):
                    brush2 = tuple(sorted(subset))
                    break
        else:
            for r, c in brush1:
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    p = r+dr, c+dc
                    if p not in brush1 and max(abs(p[0]), abs(p[1])) <= 2 \
                            and 0 <= anchor[0]+p[0] < h and 0 <= anchor[1]+p[1] < w:
                        brush2 = tuple(sorted((*brush1, p)))
                        break
                if brush2:
                    break
    assert brush2 is not None
    both = set(brush1) | set(brush2)
    safe = [(r, c) for r in range(h) for c in range(w)
            if all(0 <= r+dr < h and 0 <= c+dc < w for dr, dc in both)]
    preferred = [p for p in safe if p != anchor]
    p = (preferred or safe)[int(rng.integers(len(preferred or safe)))]
    x, y = _overlap_pair(rng, shape, brush1)
    edge = _edge(rng, shape, brush1)
    first_added = bool(rng.integers(2))
    queries = [_query(rng, shape, (p,), {p: not first_added}),
               _query(rng, shape, (x, y), {x: False, y: True}),
               _query(rng, shape, (edge,), {edge: first_added})]
    # Ensure a target-foreground shortcut is defeated, even if a randomly
    # sampled baseline happens to lie in this brush's binary null space.
    master, exception = queries[0]
    old_hits, _ = _coverage(shape, [tuple(map(int, z)) for z in np.argwhere(master)], brush1)
    if not np.any(old_hits % 2):
        for cell in [(r, c) for r in range(h) for c in range(w) if (r, c) != p]:
            changed = master.copy()
            changed[cell] = not changed[cell]
            if 0 < changed.sum() < changed.size:
                partner = exception.copy()
                partner[cell] = not partner[cell]
                queries[0] = changed, partner
                break
    rng.shuffle(queries)
    reference = int(rng.integers(q+1))
    columns = list(map(int, rng.integers(copies, size=q+1)))
    # At least these demonstrations must reject fixed-column extraction.
    if len(set(columns)) == 1:
        columns[-1] = (columns[-1]+1) % copies
    return [_render(geometry, palette, b, anchor, queries, reference, columns)
            for b in (brush1, brush2)]


def _ordinary(rng, palette, test=False):
    geometry = _geometry(rng)
    h, w, q, copies = geometry
    shape = h, w
    anchor = int(rng.integers(h)), int(rng.integers(w))
    brush = _brush(rng, shape, anchor)
    queries = []
    triple_index = int(rng.integers(q)) if test else -1
    for j in range(q):
        if j == triple_index:
            triggers = _triple(rng, shape, brush)
        elif rng.random() < .4:
            triggers = _overlap_pair(rng, shape, brush)
        else:
            count = int(rng.integers(1, 3))
            triggers = tuple(divmod(int(p), w) for p in rng.choice(h*w, count, replace=False))
        queries.append(_query(rng, shape, triggers))
    columns = list(map(int, rng.integers(copies, size=q+1)))
    return _render(geometry, palette, brush, anchor, queries, int(rng.integers(q+1)), columns)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _count(num_train, 'num_train', 2, 10)
    b = _count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    shared_palette = bool(rng.integers(2))
    puzzle = {'train': _twins(rng, palette), 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, remaining in (('train', a-2), ('test', b)):
        for _ in range(remaining):
            for attempt in range(64):
                roles = palette if shared_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
                pair = _ordinary(rng, roles, test=split == 'test')
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct-input construction exhausted")
    return puzzle


def _features(input_grid):
    scene = _parse(input_grid)
    queries = []
    for b, band in enumerate(scene['bands']):
        if b == scene['reference']:
            continue
        common, exception = band['common'], band['exception']
        delta = common != exception
        points = [tuple(map(int, p)) for p in np.argwhere(delta)]
        hits, clipped = _coverage(common.shape, points, scene['brush'])
        answer_mask = (common == scene['ink']) ^ (hits % 2 == 1)
        foreground_hits, _ = _coverage(common.shape,
                                      [tuple(map(int, p)) for p in np.argwhere(exception == scene['ink'])],
                                      scene['brush'])
        foreground_answer = (common == scene['ink']) ^ (foreground_hits % 2 == 1)
        queries.append(dict(triggers=len(points), additions=int(np.count_nonzero(delta & (common == scene['neutral']))),
                            removals=int(np.count_nonzero(delta & (common == scene['ink']))),
                            double=int(np.count_nonzero(hits == 2)), triple=int(np.count_nonzero(hits == 3)),
                            higher=int(np.count_nonzero(hits >= 4)),
                            clipped=clipped, unexpanded_diff=not np.array_equal(answer_mask, exception == scene['ink']),
                            ignores_A_diff=not np.array_equal(answer_mask, foreground_answer),
                            common=_key(common == scene['ink']), delta=_key(delta),
                            density=float(np.mean(common == scene['ink']))))
    return scene, queries


def _are_twins(first, second):
    s, t = _parse(first['input']), _parse(second['input'])
    if s['grid'].shape != t['grid'].shape or s['reference'] != t['reference']:
        return False
    if any(s[k] != t[k] for k in ('anchor', 'lattice', 'neutral', 'ink', 'pivot')):
        return False
    if s['brush'] == t['brush'] or np.array_equal(first['output'], second['output']):
        return False
    for b, (x, y) in enumerate(zip(s['bands'], t['bands'])):
        if x['column'] != y['column'] or not np.array_equal(x['common'], y['common']):
            return False
        if b != s['reference'] and not np.array_equal(x['exception'], y['exception']):
            return False
    return True


def validate_generated_puzzle(puzzle):
    """Check returned pixels and the promised demonstration/shift evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "pairs must be lists")
    _count(len(puzzle['train']), 'num_train', 2, 10)
    _count(len(puzzle['test']), 'num_test', 1, 4)
    _require(len(puzzle['train']) > len(puzzle['test']), "bad pair counts")
    seen, train_features, train_scenes = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            scene, features = _features(inp)
            _require(np.array_equal(_solve_scene(scene), out), "wrong output content or framing")
            _require(scene['pivot'] not in out, "calibration marker leaked")
            if split == 'train':
                train_features.extend(features)
                train_scenes.append(scene)
                _require(all(f['triggers'] <= 2 for f in features), "training trigger shift violated")
            else:
                _require(any(f['triggers'] >= 3 and f['triple'] > 0 for f in features),
                         "test needs triple-overlap transfer")
    _require(any(_are_twins(p, q) for p, q in combinations(puzzle['train'], 2)),
             "training needs reference twins")
    _require(any(f['triggers'] == 1 and f['clipped'] == 0 and f['unexpanded_diff'] for f in train_features),
             "need full single-event demonstration")
    _require(any(f['additions'] == f['removals'] == 1 for f in train_features), "need balanced comparison")
    _require(any(f['double'] > 0 for f in train_features), "need cancellation evidence")
    _require(any(f['clipped'] > 0 for f in train_features), "need clipping evidence")
    _require(any(f['ignores_A_diff'] for f in train_features), "foreground shortcut must fail")
    _require(any(len({b['column'] for b in s['bands']}) > 1 for s in train_scenes), "fixed-column shortcut")


def _encode(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
                       for s in ('train', 'test')}, separators=(',', ':'))


def _diagnostics():
    # Hand-calculated horizontal three-cell toggle brush.
    geometry = (4, 4, 1, 3)
    palette = (9, 0, 2, 5)
    brush = ((0, -1), (0, 0), (0, 1))
    base = np.zeros((4, 4), bool)
    base[1, 2] = True
    delta = base.copy()
    delta[1, 1] = True
    delta[1, 2] = False
    pair = _render(geometry, palette, brush, (1, 1), [(base, delta)], 0, [2, 0])
    # Stamps at c=1 and c=2 leave the center pair unchanged and toggle c=0,c=3.
    expected = np.array([[0, 0, 0, 0], [2, 0, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    assert np.array_equal(solve(pair['input']), _assemble([[expected]], 9))
    # The exact same selected exception, but a different common baseline.
    changed_base = base.copy()
    changed_base[2, 1] = True
    other = _render(geometry, palette, brush, (1, 1), [(changed_base, delta)], 0, [2, 0])
    assert np.array_equal(_parse(pair['input'])['bands'][1]['exception'],
                          _parse(other['input'])['bands'][1]['exception'])
    assert not np.array_equal(solve(pair['input']), solve(other['input']))
    # Same queries, different complete source/companion analogy.
    vertical = ((-1, 0), (0, 0), (1, 0))
    other = _render(geometry, palette, vertical, (1, 1), [(base, delta)], 0, [2, 0])
    assert not np.array_equal(solve(pair['input']), solve(other['input']))
    # Triple hits count as one; clipping has no opposite-edge effect.
    triggers = [(1, 0), (1, 1), (1, 2)]
    hits, clipped = _coverage((4, 4), triggers, brush)
    assert hits[1].tolist() == [2, 3, 2, 1] and clipped == 1
    triple_exception = base.copy()
    for cell in triggers:
        triple_exception[cell] = not triple_exception[cell]
    triple_pair = _render(geometry, palette, brush, (1, 1), [(base, triple_exception)], 0, [2, 0])
    triple_expected = np.zeros((4, 4), dtype=np.int8)
    triple_expected[1] = [0, 2, 2, 2]
    assert np.array_equal(solve(triple_pair['input']), _assemble([[triple_expected]], 9))
    # Move only the reference pivot and footprint, leaving the relative brush.
    moved = _render(geometry, palette, brush, (2, 2), [(base, delta)], 0, [2, 0])
    assert np.array_equal(solve(pair['input']), solve(moved['input']))
    # A column permutation carries no operation or selection meaning.
    scene = _parse(pair['input'])
    permuted = _assemble([[b['tiles'][i] for i in (2, 0, 1)] for b in scene['bands']], 9)
    assert np.array_equal(solve(pair['input']), solve(permuted))
    lookup = np.array([6, 0, 8, 1, 3, 2, 7, 9, 5, 4], dtype=np.int8)
    assert np.array_equal(solve(lookup[pair['input']]), lookup[solve(pair['input'])])
    # Exercise a legal family geometry beyond the sampler's usual tile heights.
    large = np.indices((13, 8)).sum(axis=0) % 3 == 0
    partner = large.copy()
    for cell in ((0, 7), (12, 0), (8, 4), (8, 5)):
        partner[cell] = not partner[cell]
    large_pair = _render((13, 8, 1, 3), palette,
                         ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
                         (0, 0), [(large, partner)], 1, [2, 1])
    assert large_pair['input'].shape == (29, 28)
    assert np.array_equal(solve(large_pair['input']), large_pair['output'])


def self_test():
    """Bounded 42-seed sweep plus count, causality and mutation checks."""
    signatures, shapes, geometries, brushes, palettes, masks, deltas = (set() for _ in range(7))
    reference_rows, columns, densities, sizes = set(), set(), [], set()
    counters = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_encode(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene, features = _features(pair['input'])
                shapes.add(pair['input'].shape)
                geometries.add((scene['h'], scene['w'], len(features), scene['copies']))
                brushes.add(scene['brush'])
                sizes.add(len(scene['brush']))
                palettes.add(tuple(scene[k] for k in ('lattice', 'neutral', 'ink', 'pivot')))
                reference_rows.add(scene['reference'])
                columns.add(tuple(b['column'] for b in scene['bands']))
                counters['rectangular'] += scene['h'] != scene['w']
                counters['repeated_columns'] += len({b['column'] for b in scene['bands']}) < len(scene['bands'])
                counters['symmetric_brush'] += set(scene['brush']) == {(-r, -c) for r, c in scene['brush']}
                for f in features:
                    masks.add(f['common'])
                    deltas.add(f['delta'])
                    densities.append(f['density'])
                    counters['double'] += f['double'] > 0
                    counters['triple'] += f['triple'] > 0
                    counters['clipped'] += f['clipped'] > 0
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(geometries) >= 40
    assert len(brushes) >= 65 and sizes == set(range(3, 8))
    assert len(palettes) >= 35 and len(masks) >= 200 and len(deltas) >= 100
    assert len(reference_rows) >= 4 and len(columns) >= 60
    assert min(densities) <= .15 and max(densities) >= .85
    assert all(counters[k] > 0 for k in counters)
    assert _encode(generate_puzzle(197260663)) == _encode(generate_puzzle(197260663))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(1000+10*a+b, a, b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _diagnostics()
    # A damaged label and a deleted exception must fail for their actual pixels.
    for mode in ('label', 'missing'):
        bad = generate_puzzle(0)
        pair = bad['train'][0]
        if mode == 'label':
            pair['output'][1, 1] = (int(pair['output'][1, 1])+1) % 10
        else:
            scene = _parse(pair['input'])
            b = next(i for i in range(len(scene['bands'])) if i != scene['reference'])
            column = scene['bands'][b]['column']
            r, c = b*(scene['h']+1)+1, column*(scene['w']+1)+1
            pair['input'][r:r+scene['h'], c:c+scene['w']] = scene['bands'][b]['common']
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("mutation was accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, reference and majority diagnostics')
