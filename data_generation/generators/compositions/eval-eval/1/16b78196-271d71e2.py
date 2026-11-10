"""Fueled telescoping jigsaws. Only standard library and NumPy are required.

Background is 0. The spanning anchor's color means spent; the other nonzero
color means fuel. Fragment masks are conserved while their counters advance.
No parent module, file, example, RNG state, or training pair enters solve().
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid size")
    _need(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= g) & (g <= 9))), "invalid color")
    return g.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _scan(cells, side):
    # Toward-anchor travel is upward for bottom fragments (side +1). For
    # top fragments rotate 180 degrees to make travel upward, then rasterize.
    return np.array(sorted(map(tuple, cells), key=lambda p: (side*p[0], side*p[1])), dtype=int)


def _profile(cells, side, start, width):
    lo, hi = [], []
    for x in range(start, start+width):
        ys = np.sort(side*cells[cells[:, 1] == x, 0])
        _need(len(ys) > 0, "fragment must span every lane column")
        _need(np.array_equal(ys, np.arange(ys[0], ys[-1]+1)), "non-convex column")
        lo.append(int(ys[0]))
        hi.append(int(ys[-1])+1)
    return np.array(lo), np.array(hi)


def _read(input_grid):
    """Read all roles and lanes from geometry, using rotations, never transpose."""
    raw = _grid(input_grid)
    components = _components(raw != 0)
    candidates = []
    for cells in components:
        if len(set(cells[:, 1])) == raw.shape[1]:
            candidates.append((0, cells))
        if len(set(cells[:, 0])) == raw.shape[0]:
            candidates.append((1, cells))
    _need(len(candidates) == 1, "expected one uniquely spanning anchor")
    turns = candidates[0][0]
    g = np.rot90(raw, turns).copy()
    components = _components(g != 0)
    bands = [c for c in components if len(set(c[:, 1])) == g.shape[1]]
    _need(len(bands) == 1, "ambiguous anchor")
    band = bands[0]
    colors = np.unique(g[band[:, 0], band[:, 1]])
    _need(len(colors) == 1 and colors[0] != 0, "anchor must be monochrome")
    spent = int(colors[0])
    fuels = set(map(int, np.unique(g))) - {0, spent}
    _need(len(fuels) <= 1, "expected one fuel color")
    fuel = next(iter(fuels), None)
    band_mask = np.zeros(g.shape, dtype=bool)
    band_mask[band[:, 0], band[:, 1]] = True
    lower, upper = _profile(band, 1, 0, g.shape[1])
    lanes = {}
    for cells in components:
        if np.any(band_mask[cells[:, 0], cells[:, 1]]):
            continue
        start, end = int(cells[:, 1].min()), int(cells[:, 1].max())+1
        _need(0 < start < end < g.shape[1], "lanes need side margins")
        xs, ys = cells[:, 1], cells[:, 0]
        if np.all(ys < lower[xs]):
            side = -1
        elif np.all(ys >= upper[xs]):
            side = 1
        else:
            raise AssertionError("fragment straddles anchor")
        lo, hi = _profile(cells, side, start, end-start)
        order = _scan(cells, side)
        values = g[order[:, 0], order[:, 1]]
        advanced = int(np.count_nonzero(values == spent))
        _need(bool(np.all(values[:advanced] == spent)), "spent cells must be a prefix")
        _need(advanced == len(values) or bool(np.all(values[advanced:] == fuel)), "fuel must be a suffix")
        fragment = dict(cells=cells, order=order, lo=lo, hi=hi, advanced=advanced,
                        fuel=len(cells)-advanced, capacity=len(cells))
        lanes.setdefault((side, start, end-start), []).append(fragment)
    _need(bool(lanes), "no fragments")
    keys = sorted(lanes)
    for i, (s, a, w) in enumerate(keys):
        for t, b, v in keys[:i]:
            _need(s != t or a+w < b or b+v < a, "same-side lanes must be separated")
    for key, fragments in lanes.items():
        side, start, width = key
        edge = upper[start:start+width] if side == 1 else 1-lower[start:start+width]
        fragments.sort(key=lambda f: int(f['lo'].min()))
        frontier = edge.copy()
        last_hi = None
        for f in fragments:
            if last_hi is not None:
                _need(int(f['lo'].min()) > int(last_hi.max()), "input fragments must have separate boxes")
            else:
                _need(int(f['lo'].min()) > int(frontier.max()), "anchor clearance missing")
            difference = f['lo']-frontier
            _need(len(set(map(int, difference))) == 1 and difference[0] > 0,
                  "adjacent complementary profiles must admit exact assembly")
            f['ideal_distance'] = int(difference[0])
            frontier = f['hi']-difference[0]
            last_hi = f['hi']
        _need(len(set(map(int, frontier))) == 1, "fully docked lane must end flat")
    return g, turns, spent, fuel, band_mask, (lower, upper), lanes


def _transform(input_grid, *, unlimited=False, boxes=False):
    g, turns, spent, fuel, band_mask, edges, lanes = _read(input_grid)
    out = np.zeros_like(g)
    out[band_mask] = spent
    records = []
    for (side, start, width), fragments in sorted(lanes.items()):
        frontier = (edges[1][start:start+width] if side == 1
                    else 1-edges[0][start:start+width]).copy()
        previous_residual = 0
        for index, f in enumerate(fragments):
            distance = int(np.min(f['lo']-frontier))
            if boxes:
                distance = max(0, int(f['lo'].min()-frontier.max()))
            movement = distance if unlimited else min(distance, f['fuel'])
            cells = f['cells'].copy()
            cells[:, 0] -= side*movement
            order = f['order'].copy()
            order[:, 0] -= side*movement
            values = np.full(len(order), fuel if fuel is not None else spent, dtype=np.int8)
            values[:min(len(order), f['advanced']+movement)] = spent
            _need(not np.any(out[cells[:, 0], cells[:, 1]]), "settled fragments overlap")
            out[order[:, 0], order[:, 1]] = values
            residual = f['ideal_distance']-movement
            records.append(dict(side=side, start=start, width=width, index=index,
                                pieces=len(fragments), capacity=f['capacity'], fuel=f['fuel'],
                                needed=distance, move=movement, ideal=f['ideal_distance'],
                                residual=residual, inherited=previous_residual,
                                jam=previous_residual > 0 and movement == distance,
                                stop=('zero' if f['fuel'] == 0 else 'fuel' if f['fuel'] < distance
                                      else 'exact' if f['fuel'] == distance else 'contact')))
            frontier = f['hi']-movement
            previous_residual = residual
    return np.rot90(out, -turns).copy(), records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Settle each lane from the anchor outward, spending one cell per step."""
    return _transform(input_grid)[0]


def _cuts(rng, width, count):
    """Connected, full-width, complementary strips, with a flat final cap."""
    cuts = [rng.integers(0, 3, size=width)]
    if np.ptp(cuts[0]) == 0:
        cuts[0][int(rng.integers(width))] += 1
    for _ in range(count-1):
        shape = rng.integers(0, int(rng.integers(2, 4)), size=width)
        if np.ptp(shape) == 0:
            shape[int(rng.integers(width))] += 1
        old = cuts[-1]
        offset = max(int(np.max(old+1-shape)),
                     int(np.max(np.maximum(old[:-1], old[1:])+1-
                                np.minimum(shape[:-1], shape[1:]))))
        cuts.append(shape+offset+int(rng.integers(0, 2)))
    cuts.append(np.full(width, int(cuts[-1].max())+int(rng.integers(1, 4))))
    return cuts


def _lane(rng, width, count, lesson):
    cuts = _cuts(rng, width, count)
    shifts, source_hi = [], int(cuts[0].max())
    for i in range(count):
        gap = int(rng.integers(2, 5)) if i == 0 else int(rng.integers(1, 4))
        shift = source_hi+gap-int(cuts[i].min())
        shifts.append(shift)
        source_hi = int(cuts[i+1].max())+shift
    capacities = [int(np.sum(b-a)) for a, b in zip(cuts, cuts[1:])]
    fuel_counts, residual = [], 0
    if lesson == 'contact':
        bottlenecks = set()
    elif lesson in ('jam', 'zero'):
        bottlenecks = {int(rng.integers(0, count-1))}
    elif lesson == 'double':
        bottlenecks = set(map(int, rng.choice(count-1, size=2, replace=False)))
    else:  # Exact contact without a new bottleneck.
        bottlenecks = set()
    for i, (shift, capacity) in enumerate(zip(shifts, capacities)):
        needed = shift-residual
        if i in bottlenecks:
            if lesson == 'zero':
                available = 0
            else:
                if needed < 2 or capacity < 2:
                    raise ValueError('insufficient room for informative bottleneck')
                available = int(rng.integers(1, min(needed-1, capacity-1)+1))
        elif lesson == 'exact' and i == 0:
            if needed > capacity:
                raise ValueError('exact counter too small')
            available = needed
        else:
            # A follower with surplus fuel makes inherited jamming observable.
            low = needed+1
            high = capacity-1 if lesson == 'contact' and i == 0 else capacity
            if low > high:
                raise ValueError('counter too small for surplus')
            available = int(rng.integers(low, high+1))
        fuel_counts.append(available)
        residual = shift-min(available, needed)
    return dict(cuts=cuts, shifts=shifts, fuels=fuel_counts, extent=source_hi, width=width)


def _make_pair(rng, lesson, rotation, palette):
    # Construction retries keep only one candidate in memory, never on disk.
    spent, fuel = palette
    main_count = int(rng.integers(3, 5)) if lesson == 'double' else int(rng.integers(2, 4))
    number = int(rng.choice((1, 2, 3), p=(.50, .40, .10)))
    descriptions = []
    main_side = int(rng.choice((-1, 1)))
    for j in range(number):
        width = int(rng.integers(3, 8))
        count = main_count if j == 0 else int(rng.integers(1, 3))
        kind = lesson if j == 0 else str(rng.choice(('contact', 'exact')))
        item = _lane(rng, width, count, kind)
        item['side'] = main_side if j == 0 else int(rng.choice((-1, 1)))
        descriptions.append(item)
    minimum_width = max(11, sum(d['width']+1 for d in descriptions)+1)
    if minimum_width > 30:
        raise ValueError('crosswise space')
    W = int(rng.integers(minimum_width, min(30, minimum_width+8)+1))
    # Interleave freely positioned disjoint lane spans; opposite lanes may align.
    reserved = {-1: set(), 1: set()}
    for d in descriptions:
        choices = [x for x in range(1, W-d['width'])
                   if not set(range(x-1, x+d['width']+1)) & reserved[d['side']]]
        if not choices:
            raise ValueError('lane separation')
        d['start'] = int(rng.choice(choices))
        reserved[d['side']].update(range(d['start'], d['start']+d['width']))
    boundary = {s: rng.integers(0, 2, size=W) for s in (-1, 1)}
    for d in descriptions:
        x, w = d['start'], d['width']
        boundary[d['side']][x:x+w] = d['cuts'][0]
    extents = {s: max([int(boundary[s].max())]+[d['extent'] for d in descriptions if d['side'] == s])
               for s in (-1, 1)}
    core = int(rng.integers(2, 5))
    minimum_height = extents[-1]+core+extents[1]
    if minimum_height > 30:
        raise ValueError('axial space')
    H = int(rng.integers(minimum_height, min(30, minimum_height+4)+1))
    center = extents[-1]+int(rng.integers(H-minimum_height+1))
    g = np.zeros((H, W), dtype=np.int8)
    for x in range(W):
        g[center-boundary[-1][x]:center+core+boundary[1][x], x] = spent
    for d in descriptions:
        side, x, width = d['side'], d['start'], d['width']
        for i, available in enumerate(d['fuels']):
            a, b = d['cuts'][i:i+2]
            cells = np.array([(center-1-u if side == -1 else center+core+u, x+c)
                              for c in range(width)
                              for u in range(int(a[c])+d['shifts'][i], int(b[c])+d['shifts'][i])])
            order = _scan(cells, side)
            _need(not np.any(g[order[:, 0], order[:, 1]]), 'construction overlap')
            advanced = len(order)-available
            g[order[:, 0], order[:, 1]] = fuel
            g[order[:advanced, 0], order[:advanced, 1]] = spent
    g = np.rot90(g, rotation).copy()
    result, records = _transform(g)
    if lesson == 'double' and sum(r['stop'] == 'fuel' for r in records) < 2:
        raise ValueError('double bottleneck not exposed')
    if lesson in ('jam', 'zero', 'double') and not any(r['jam'] and r['fuel'] > r['needed'] for r in records):
        raise ValueError('jam not exposed')
    return dict(input=g, output=result)


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(np.arange(1, 10), 2, replace=False)))
    stable_palette = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    lessons = ['contact', 'jam']+[str(rng.choice(('contact', 'jam', 'zero', 'exact'))) for _ in range(ntrain-2)]
    rng.shuffle(lessons)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            lesson = lessons[i] if split == 'train' else 'double'
            k = (rotation+i) % 4 if split == 'train' else int(rng.integers(4))
            colors = palette if stable_palette else tuple(map(int, rng.choice(np.arange(1, 10), 2, replace=False)))
            for _ in range(600):
                try:
                    pair = _make_pair(rng, lesson, k, colors)
                except ValueError:
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded geometric construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _unit_steps(input_grid):
    """Check movement by actual cell collisions and single-cell counter ticks."""
    g, turns, spent, fuel, band_mask, _, lanes = _read(input_grid)
    out = np.zeros_like(g)
    out[band_mask] = spent
    for (side, _, _), fragments in sorted(lanes.items()):
        for f in fragments:
            order = f['order'].copy()
            values = g[order[:, 0], order[:, 1]].copy()
            for _ in range(30):
                pending = np.flatnonzero(values != spent)
                shifted = order.copy()
                shifted[:, 0] -= side
                inside = bool(np.all((shifted[:, 0] >= 0) & (shifted[:, 0] < g.shape[0])))
                if not len(pending) or not inside or np.any(out[shifted[:, 0], shifted[:, 1]]):
                    break
                order = shifted
                values[int(pending[0])] = spent
            out[order[:, 0], order[:, 1]] = values
    return np.rot90(out, -turns).copy()


def _validate_pair(pair):
    _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _need(inp.shape == out.shape, 'shape changed')
    predicted, records = _transform(inp)
    _need(np.array_equal(out, predicted), 'incorrect settled output')
    _need(np.array_equal(out, _unit_steps(inp)), 'profile engine disagrees with cell steps')
    g, turns, spent, fuel, band, _, _ = _read(inp)
    canonical_out = np.rot90(out, turns)
    _need(bool(np.all(canonical_out[band] == spent)), 'anchor changed')
    _need(np.count_nonzero(inp) == np.count_nonzero(out), 'occupied area changed')
    if fuel is not None:
        _need(np.count_nonzero(inp == fuel)-np.count_nonzero(out == fuel) == sum(r['move'] for r in records),
              'one fuel cell per step invariant failed')
    return records


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            records = _validate_pair(pair)
            key = _key(_grid(pair['input']))
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            if split == 'train':
                training.extend(records)
    _need(any(r['stop'] == 'contact' and r['move'] >= 2 for r in training), 'missing contact-limited lesson')
    _need(any(r['stop'] == 'fuel' and r['move'] > 0 for r in training), 'missing fuel-limited lesson')
    _need(any(r['jam'] and r['fuel'] > r['needed'] for r in training), 'missing inherited jam lesson')
    # Omitting fuel or replacing complementary contact by box contact must
    # produce a different labeled answer in the actual training demonstrations.
    _need(any(not np.array_equal(_transform(p['input'], unlimited=True)[0], p['output'])
              for p in puzzle['train']), 'fuel concept is dispensable')
    _need(any(not np.array_equal(_transform(p['input'], boxes=True)[0], p['output'])
              for p in puzzle['train']), 'complementarity concept is dispensable')


def self_test():
    """42 streaming seeds, mutation checks, rotations, and count boundaries.

    Retains hashes and scalar feature sets, never a corpus of grids or traces.
    Returns a small JSON-compatible report; writes no files and prints nothing.
    """
    shapes, widths, counts, capacities, roles, structures, stops = set(), set(), set(), set(), set(), set(), Counter()
    puzzles, total_pairs, jams, fresh, exhausted = set(), 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                signature.update(str(inp.shape).encode()+inp.tobytes()+out.tobytes())
                records = _validate_pair(pair)
                total_pairs += 1
                shapes.add(inp.shape)
                roles.add(tuple(sorted(set(map(int, np.unique(inp)))-{0})))
                mapping = {0: 0}
                normalized = bytes(mapping.setdefault(int(v), len(mapping)) for v in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).hexdigest()))
                for r in records:
                    widths.add(r['width'])
                    counts.add(r['pieces'])
                    capacities.add(r['capacity'])
                    stops[r['stop']] += 1
                    jams += r['jam']
                    fresh += r['fuel'] == r['capacity']
                    exhausted += r['fuel'] == 0
                for k in (1, 2, 3):
                    _need(np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k)), 'rotation covariance failed')
        puzzles.add(signature.hexdigest())
        repeated = generate_puzzle(seed)
        _need(all(np.array_equal(a[k], b[k]) for s in puzzle for a, b in zip(puzzle[s], repeated[s])
                  for k in ('input', 'output')), 'nondeterministic puzzle')
    _need(len(puzzles) == 42 and len(structures) >= 205, 'seed variation collapsed')
    _need(len(shapes) >= 40 and widths == set(range(3, 8)), 'geometry variation collapsed')
    _need({1, 2, 3, 4} <= counts and len(capacities) >= 15, 'fragment variation collapsed')
    _need(set(stops) == {'contact', 'exact', 'fuel', 'zero'} and jams > 42 and fresh > 0 and exhausted > 0,
          'counter or interaction coverage missing')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(4200+a*10+b, a, b)
            _need(len(p['train']) == a and len(p['test']) == b, 'explicit counts ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0, 2, 1)
    bad = {s: [{k: g.copy() for k, g in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    return dict(seeds=42, pairs=total_pairs, distinct_puzzles=len(puzzles), normalized_inputs=len(structures),
                shapes=len(shapes), widths=sorted(widths), pieces_per_lane=sorted(counts),
                capacities=len(capacities), stops=dict(stops), inherited_jams=jams,
                fresh_fragments=fresh, initially_exhausted_fragments=exhausted)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
