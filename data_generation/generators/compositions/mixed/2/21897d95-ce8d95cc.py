"""T territory transfers followed by exact adjacent-vector contraction.

Only NumPy and the standard library are used. No files, parent modules, seed,
demonstrations, or generation metadata are consulted by solve().
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _add(p, d, distance=1):
    return p[0] + distance * d[0], p[1] + distance * d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(a):
    ids = np.full(a.shape, -1, dtype=np.int16)
    colors, areas, boxes = [], [], []
    h, w = a.shape
    for r in range(h):
        for c in range(w):
            if ids[r, c] >= 0:
                continue
            label, color = len(colors), int(a[r, c])
            colors.append(color)
            stack, area, r0, r1, c0, c1 = [(r, c)], 0, r, r, c, c
            ids[r, c] = label
            while stack:
                row, col = stack.pop()
                area += 1
                r0, r1 = min(r0, row), max(r1, row)
                c0, c1 = min(c0, col), max(c1, col)
                for dr, dc in _DIRECTIONS:
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < h and 0 <= cc < w and ids[rr, cc] < 0 and a[rr, cc] == color:
                        ids[rr, cc] = label
                        stack.append((rr, cc))
            areas.append(area)
            boxes.append((r0, r1, c0, c1))
    return ids, np.array(colors, dtype=np.int8), areas, boxes


def _geometry(center, direction, shape):
    perpendicular = (-direction[1], direction[0])
    arms = (_add(center, perpendicular), _add(center, perpendicular, -1))
    tip, target = _add(center, direction), _add(center, direction, 2)
    sides = (_add(tip, perpendicular), _add(tip, perpendicular, -1))
    write = frozenset((center, *arms, tip))
    if not all(_inside(p, shape) for p in (*write, *sides, target)):
        return None
    return dict(center=center, direction=direction, arms=arms, tip=tip,
                targetcell=target, sides=sides, write=write)


def _run_starts(a, axis):
    vectors = a if axis == 0 else a.T
    keep = np.ones(len(vectors), dtype=bool)
    keep[1:] = np.any(vectors[1:] != vectors[:-1], axis=1)
    return np.flatnonzero(keep)


def _collapse(a):
    # Both equivalence tests refer to the same fully painted grid. Either
    # sequential axis order is equivalent, but no transfer may follow this.
    return a[np.ix_(_run_starts(a, 0), _run_starts(a, 1))].copy()


def _decode(a):
    """Infer the unique complete T interpretation from visible input cells."""
    raw_ids, raw_colors, raw_areas, _ = _components(a)
    answers = []
    for ink_value in np.unique(a):
        ink = int(ink_value)
        ink_ids = np.flatnonzero(raw_colors == ink)
        # Ink may form four-cell implicit Ts or three separated singletons
        # around a literal junction. No large fill component can be ink.
        if any(raw_areas[int(i)] > 4 for i in ink_ids):
            continue
        ink_cells = set(map(tuple, np.argwhere(a == ink)))
        if len(ink_cells) < 3:
            continue
        centers = {q for p in ink_cells for d in _DIRECTIONS
                   if _inside(q := _add(p, d), a.shape)}
        arrows = []
        for center in sorted(centers):
            for direction in _DIRECTIONS:
                geo = _geometry(center, direction, a.shape)
                if geo is None or not all(a[p] == ink for p in (*geo['arms'], geo['tip'])):
                    continue
                # A candidate must contain the complete raw ink component of
                # each non-junction marker cell, rather than a fragment of a T.
                groups = {int(raw_ids[p]) for p in (*geo['arms'], geo['tip'])}
                if any(sum(int(raw_ids[p]) == i for p in geo['write']) != raw_areas[i] for i in groups):
                    continue
                flank_a, flank_b = geo['sides']
                host = int(a[flank_a])
                if host == ink or a[flank_b] != host or int(a[geo['targetcell']]) in (host, ink):
                    continue
                explicit = int(a[center]) != ink
                payload = int(a[center]) if explicit else host
                arrows.append(dict(geo, host=host, explicit=explicit, payload=payload))
        if not arrows:
            continue
        used, halo, context, valid = set(), set(), set(), True
        for arrow in arrows:
            read = {*arrow['sides'], arrow['targetcell']}
            if arrow['write'] & (halo | context) or read & used:
                valid = False
                break
            used.update(arrow['write'])
            context.update(read)
            halo.update(arrow['write'])
            halo.update(_add(p, d) for p in arrow['write'] for d in _DIRECTIONS)
        if not valid or used & ink_cells != ink_cells:
            continue
        clean = a.copy()
        for arrow in arrows:
            for p in arrow['write']:
                clean[p] = arrow['host']
        if bool(np.any(clean == ink)):
            continue
        ids, colors, areas, boxes = _components(clean)
        assignments = {}
        for arrow in arrows:
            source = int(ids[arrow['center']])
            target = int(ids[arrow['targetcell']])
            if (source == target or int(colors[source]) != arrow['host']
                    or any(ids[p] != source for p in (*arrow['write'], *arrow['sides']))
                    or target in assignments and assignments[target] != arrow['payload']):
                valid = False
                break
            arrow.update(source=source, target=target)
            assignments[target] = arrow['payload']
        if not valid:
            continue
        final_colors = colors.copy()
        for target, payload in assignments.items():
            final_colors[target] = payload
        painted = final_colors[ids]
        answers.append(dict(ink=ink, arrows=arrows, clean=clean, ids=ids,
                            colors=colors, areas=areas, boxes=boxes,
                            final_colors=final_colors, assignments=assignments,
                            painted=painted, output=_collapse(painted)))
    _require(len(answers) == 1, "input must have one unambiguous complete T interpretation")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore, transfer original component colors simultaneously, then contract."""
    return _decode(_array(input_grid))['output'].copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        h.update(bytes((len(puzzle[split]),)))
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _runs(rng, count, total, minimum=3):
    _require(total >= minimum * count, "not enough space for variable bands")
    result = np.full(count, minimum, dtype=int)
    extra = total - minimum * count
    if extra:
        weights = rng.dirichlet(np.ones(count))
        result += rng.multinomial(extra, weights)
    return result


def _oriented(a, turns, mirror):
    b = np.rot90(a, turns)
    return (np.fliplr(b) if mirror else b).copy()


def _finish(inp, clean, painted, rng):
    turns, mirror = int(rng.integers(4)), bool(rng.integers(2))
    inp, clean, painted = [_oriented(a, turns, mirror) for a in (inp, clean, painted)]
    out = _collapse(painted)
    info = _decode(inp)
    _require(np.array_equal(info['clean'], clean), "restoration differs from constructed map")
    _require(np.array_equal(info['painted'], painted), "decoded transfers differ from construction")
    _require(np.array_equal(info['output'], out), "incorrect contraction")
    _require(not np.array_equal(_collapse(clean), out), "transfer must affect the answer")
    _require(out.shape[0] < inp.shape[0] and out.shape[1] < inp.shape[1], "sample must contract both axes")
    _require(min(out.shape) >= 2, "sample must retain visible structure")
    return dict(input=inp, output=out)


def _draw(inp, arrow, ink, payload=None):
    for p in arrow['write']:
        inp[p] = ink
    if payload is not None:
        inp[arrow['center']] = payload


def _anchor(rng, kind):
    palette = rng.permutation(10)
    ink, fills = int(palette[0]), palette[1:]
    heights = _runs(rng, 3 if kind == 0 else 2, int(rng.integers(10, 22)))
    widths = _runs(rng, 4, int(rng.integers(13, 25)))
    if kind == 0:
        logical = np.array([[0, 1, 3, 0], [2, 2, 1, 2], [0, 0, 3, 0]])
        clean = np.repeat(np.repeat(fills[logical], heights, axis=0), widths, axis=1).astype(np.int8)
        painted, inp = clean.copy(), clean.copy()
        row = int(rng.integers(1, heights[0] - 1))
        center = row, int(widths[0] - 2)
        arrow = _geometry(center, (0, 1), clean.shape)
        _draw(inp, arrow, ink)
        painted[:heights[0], widths[0]:widths[0] + widths[1]] = fills[0]
    else:
        logical = np.array([[0, 1, 2, 3], [4, 4, 5, 4]])
        clean = np.repeat(np.repeat(fills[logical], heights, axis=0), widths, axis=1).astype(np.int8)
        painted, inp = clean.copy(), clean.copy()
        row = int(rng.integers(1, heights[0] - 1))
        edges = np.cumsum(np.r_[0, widths])
        for index in range(3):
            arrow = _geometry((row, int(edges[index + 1] - 2)), (0, 1), clean.shape)
            _draw(inp, arrow, ink, int(fills[6]) if index == 2 else None)
            painted[:heights[0], edges[index + 1]:edges[index + 2]] = fills[6] if index == 2 else fills[index]
    return _finish(inp, clean, painted, rng)


def _mosaic(rng, shape, palette, test):
    h, w = shape
    m = int(rng.integers(2 if not test else 3, min(5 if test else 4, h // 3) + 1))
    n = int(rng.integers(2 if not test else 3, min(5 if test else 4, w // 3) + 1))
    heights, widths = _runs(rng, m, h), _runs(rng, n, w)
    colors = rng.choice(palette, size=(m, n)).astype(np.int8)
    for r in range(m):
        for c in range(n):
            if r and rng.random() < .22:
                colors[r, c] = colors[r - 1, c]
            elif c and rng.random() < .22:
                colors[r, c] = colors[r, c - 1]
    a = np.empty(shape, dtype=np.int8)
    base_edges, row = np.cumsum(np.r_[0, widths]), 0
    for r in range(m):
        edges = base_edges.copy()
        # A shared color across adjoining bands can become a stepped or
        # concave component; there is no fixed 3x3 or 5x5 territory lattice.
        for c in range(1, n):
            low, high = int(edges[c - 1] + 3), int(base_edges[c + 1] - 3)
            edges[c] = int(np.clip(base_edges[c] + rng.integers(-2, 3), low, high))
        for c in range(n):
            a[row:row + heights[r], edges[c]:edges[c + 1]] = colors[r, c]
        row += int(heights[r])
    return a


def _nested(rng, shape, palette, test):
    a = np.full(shape, palette[0], dtype=np.int8)
    r0, r1, c0, c1 = 0, shape[0], 0, shape[1]
    maximum = min(4 if test else 3, (min(shape) - 3) // 6)
    layers = int(rng.integers(1, max(1, maximum) + 1))
    for layer in range(layers):
        # Each new interior has a visible collar. Its outer host remains
        # connected, even when a later interior reuses an exterior color.
        r0 += int(rng.integers(2, 4))
        r1 -= int(rng.integers(2, 4))
        c0 += int(rng.integers(2, 4))
        c1 -= int(rng.integers(2, 4))
        if min(r1 - r0, c1 - c0) < 3:
            break
        choices = [int(c) for c in palette if c != a[r0, c0]]
        a[r0:r1, c0:c1] = int(rng.choice(choices))
    return a


def _crossings(rng, shape, palette, test):
    h, w = shape
    horizontal = int(rng.integers(1, min(3 if test else 2, (h // 3 - 1) // 2) + 1))
    vertical = int(rng.integers(1, min(3 if test else 2, (w // 3 - 1) // 2) + 1))
    row_runs, col_runs = _runs(rng, 2 * horizontal + 1, h), _runs(rng, 2 * vertical + 1, w)
    core = np.full((len(row_runs), len(col_runs)), palette[0], dtype=np.int8)
    hcolors = rng.choice(palette[1:], horizontal)
    vcolors = rng.choice(palette[1:], vertical)
    core[1::2, :] = hcolors[:, None]
    core[:, 1::2] = vcolors[None, :]
    mode = int(rng.integers(3))
    for r in range(horizontal):
        for c in range(vertical):
            bit = mode if mode < 2 else int(rng.integers(2))
            core[2 * r + 1, 2 * c + 1] = hcolors[r] if bit == 0 else vcolors[c]
    return np.repeat(np.repeat(core, row_runs, axis=0), col_runs, axis=1)


def _boundary_candidates(ids):
    result = []
    for center in np.ndindex(ids.shape):
        for direction in _DIRECTIONS:
            geo = _geometry(center, direction, ids.shape)
            if geo is None:
                continue
            source, target = int(ids[center]), int(ids[geo['targetcell']])
            if source != target and all(ids[p] == source for p in (*geo['write'], *geo['sides'])):
                result.append(dict(geo, source=source, target=target))
    return result


def _free_pair(rng, test):
    h, w = int(rng.integers(12, 25)), int(rng.integers(12, 25))
    if test:
        h, w = int(rng.integers(15, 31)), int(rng.integers(25, 31))
    shape = h, w
    colors = rng.permutation(10)
    ink = int(colors[0])
    palette = colors[1:int(rng.integers(5, 9))]
    kind = int(rng.integers(3))
    clean = (_mosaic, _nested, _crossings)[kind](rng, shape, palette, test)
    ids, fills, _, _ = _components(clean)
    _require(len(fills) >= 2, "map has too few territories")
    candidates = _boundary_candidates(ids)
    _require(bool(candidates), "no readable T positions")
    desired = int(rng.integers(2 if test else 1, 8 if test else 5))
    chosen, assignments, used, halo, context = [], {}, set(), set(), set()
    for index in rng.permutation(len(candidates)):
        arrow = candidates[int(index)]
        write, read = arrow['write'], {*arrow['sides'], arrow['targetcell']}
        if write & (halo | context) or read & used:
            continue
        source, target = arrow['source'], arrow['target']
        if target in assignments:
            if rng.random() > .18:
                continue
            payload = assignments[target]
            explicit = payload != fills[source]
        else:
            explicit = bool(rng.random() < .4)
            if explicit:
                pool = [c for c in range(10) if c not in (ink, int(fills[source]), int(fills[target]))]
                payload = int(rng.choice(pool))
            else:
                payload = int(fills[source])
        if payload == fills[target]:
            continue
        chosen.append(dict(arrow, payload=payload, explicit=explicit))
        assignments[target] = payload
        used.update(write)
        halo.update(write)
        halo.update(_add(p, d) for p in write for d in _DIRECTIONS)
        context.update(read)
        if len(chosen) == desired:
            break
    _require(bool(chosen), "could not place effective instructions")
    inp, final = clean.copy(), fills.copy()
    for arrow in chosen:
        _draw(inp, arrow, ink, arrow['payload'] if arrow['explicit'] else None)
    for target, payload in assignments.items():
        final[target] = payload
    return _finish(inp, clean, final[ids], rng)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not 2 <= num_train <= 10
            or not _integer(num_test) or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require integer counts 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            for _ in range(80):
                try:
                    pair = _anchor(rng, index) if split == 'train' and index < 2 else _free_pair(rng, split == 'test')
                except ValueError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction could not produce a distinct readable pair")
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    clean, painted, out = info['clean'], info['painted'], info['output']
    colors, final, arrows = info['colors'], info['final_colors'], info['arrows']
    before_rows, before_cols = _run_starts(clean, 0), _run_starts(clean, 1)
    after_rows, after_cols = _run_starts(painted, 0), _run_starts(painted, 1)
    original_coordinates = painted[np.ix_(before_rows, before_cols)]
    rows = [row.tobytes() for row in out]
    cols = [col.tobytes() for col in out.T]
    return dict(
        territories=len(colors), arrows=len(arrows), ink=info['ink'],
        explicit=sum(a['explicit'] for a in arrows),
        literal=sum(a['explicit'] and a['payload'] != colors[a['source']] for a in arrows),
        singleton=sum(a['explicit'] and a['payload'] not in colors for a in arrows),
        locality=any(colors[i] == colors[j] and final[i] != final[j]
                     for i in range(len(colors)) for j in range(i)),
        chain=any(not a['explicit'] and not b['explicit'] and a['target'] == b['source']
                  and a['source'] != b['target'] and a['payload'] != b['payload']
                  for a in arrows for b in arrows),
        cycle=any(a['source'] == b['target'] and a['target'] == b['source'] for a in arrows for b in arrows),
        fanout=len({a['source'] for a in arrows}) < len({(a['source'], a['target']) for a in arrows}),
        agreeing_incoming=len(info['assignments']) < len(arrows),
        dependency=not np.array_equal(original_coordinates, out),
        new_row_runs=len(before_rows) - len(after_rows),
        new_col_runs=len(before_cols) - len(after_cols),
        separated_rows=len(rows) > len(set(rows)),
        separated_cols=len(cols) > len(set(cols)),
        nonrectangular=sum(area != (b[1] - b[0] + 1) * (b[3] - b[2] + 1)
                           for area, b in zip(info['areas'], info['boxes'])),
        directions={a['direction'] for a in arrows})


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "wrong pair keys")
            inp, out = _array(pair['input']), _array(pair['output'])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _decode(inp)
            _require(np.array_equal(info['output'], out), "wrong transfer or collapse output")
            _require(not np.array_equal(_collapse(info['clean']), out), "omitting A leaves answer unchanged")
            _require(out.shape[0] < inp.shape[0] and out.shape[1] < inp.shape[1], "omitting B leaves dimensions unchanged")
            painted = info['painted']
            rs, cs = _run_starts(painted, 0), _run_starts(painted, 1)
            row_lengths = np.diff(np.r_[rs, painted.shape[0]])
            col_lengths = np.diff(np.r_[cs, painted.shape[1]])
            reconstructed = np.repeat(np.repeat(out, row_lengths, axis=0), col_lengths, axis=1)
            _require(np.array_equal(reconstructed, painted), "retained cell or crossing changed")
            _require(np.array_equal(_collapse(out), out), "output still contains duplicate neighboring vectors")
            # A second axis order checks commutation, separately from _collapse.
            col_first = painted[:, cs]
            col_first = col_first[_run_starts(col_first, 0)]
            _require(np.array_equal(col_first, out), "axis contractions do not commute")
            if split == 'train':
                _require(max(inp.shape) <= 24, "training sample exceeds scale regime")
                training.append(_features(info))
            else:
                _require(max(inp.shape) >= 25, "test lacks its demonstrated scale shift")
    for feature in ('dependency', 'locality', 'chain', 'literal', 'singleton', 'separated_rows', 'separated_cols'):
        _require(any(f[feature] for f in training), "missing training contrast: " + feature)
    _require(any(f['dependency'] and f['locality'] and f['separated_rows'] and f['separated_cols']
                 for f in training), "missing combined seam witness")


def _regressions():
    # Whole-vector recurrence and crossing values, with unequal expansion.
    small = np.array([[2, 7, 2], [4, 6, 4], [2, 7, 2]], dtype=np.int8)
    expanded = np.repeat(np.repeat(small, [2, 1, 4], axis=0), [3, 2, 1], axis=1)
    _require(np.array_equal(_collapse(expanded), small), "separated recurrence was lost")
    # Minimal boundary T: two-cell host depth, one-cell target width.
    inp = np.array([[8, 3, 6], [8, 8, 6], [8, 3, 6]], dtype=np.int8)
    _require(np.array_equal(solve(inp), np.array([[3]], dtype=np.int8)), "edge/thin target regression")
    for kind in (0, 1):
        pair = _anchor(np.random.default_rng(101 + kind), kind)
        info = _decode(pair['input'])
        features = _features(info)
        if kind == 0:
            _require(features['dependency'] and features['locality'], "seam witness missing")
            _require(features['separated_rows'] and features['separated_cols'], "recurrences missing")
        else:
            _require(features['chain'] and features['singleton'], "chain/payload witness missing")
            # Deliberately cascade in each direction. At least one execution
            # order reads a newly changed host, contrary to simultaneity.
            wrong = []
            for arrows in (info['arrows'], info['arrows'][::-1]):
                final = info['colors'].copy()
                for arrow in arrows:
                    final[arrow['target']] = arrow['payload'] if arrow['explicit'] else final[arrow['source']]
                wrong.append(_collapse(final[info['ids']]))
            _require(any(not np.array_equal(w, pair['output']) for w in wrong), "chain does not distinguish cascade")
            # Mutate only the literal junction. Output must change accordingly.
            literal = next(a for a in info['arrows'] if a['explicit'])
            mutated = pair['input'].copy()
            replacement = next(c for c in range(10) if c not in (info['ink'], literal['payload']))
            mutated[literal['center']] = replacement
            _require(not np.array_equal(solve(mutated), pair['output']), "literal payload ignored")


def self_test():
    """Bounded, sequential 42-seed checks; no files or execution traces saved."""
    _regressions()
    hashes, inputs, outputs, structures, inks, counts, directions = set(), set(), set(), set(), set(), set(), set()
    totals = {name: 0 for name in ('pairs', 'dependency', 'locality', 'chain', 'literal', 'singleton',
                                   'separated_rows', 'separated_cols', 'nonrectangular', 'cycle',
                                   'fanout', 'agreeing_incoming')}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_digest(puzzle))
        if seed in (0, 41):
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "seed is not deterministic")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                f = _features(info)
                totals['pairs'] += 1
                for name in totals:
                    if name != 'pairs':
                        totals[name] += int(bool(f[name]))
                inputs.add(pair['input'].shape)
                outputs.add(pair['output'].shape)
                inks.add(f['ink'])
                counts.add(f['arrows'])
                directions.update(f['directions'])
                # Store a digest of a color-normalized partition, not its cells.
                rename, normalized = {}, []
                for value in info['clean'].flat:
                    normalized.append(rename.setdefault(int(value), len(rename)))
                structures.add(hashlib.sha256(bytes(info['clean'].shape) + bytes(normalized)).hexdigest())
    _require(len(hashes) == 42, "seed diversity collapsed")
    _require(len(inputs) >= 40 and len(outputs) >= 12 and len(structures) >= 120, "geometry variation collapsed")
    _require(len(inks) >= 8 and directions == set(_DIRECTIONS), "palette/orientation variation collapsed")
    _require(len(counts) >= 4 and totals['nonrectangular'] >= 10, "transfer/territory variation collapsed")
    for counts_request in ((2, 1), (10, 1), (10, 4), (6, 3)):
        generate_puzzle(900 + sum(counts_request), *counts_request)
    for train, test in ((1, 1), (11, 1), (4, 0), (5, 5), (3, 4), (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, distinct_puzzles=len(hashes), input_shapes=len(inputs), output_shapes=len(outputs),
                normalized_maps=len(structures), ink_colors=sorted(inks), arrow_counts=sorted(counts),
                directions=[list(d) for d in sorted(directions)], **totals)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
