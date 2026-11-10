"""Complete a miniature motif and register its possibly absent handle to an edge guide.

Only NumPy and the standard library are used. No generation state enters solve.
Experiments retain aggregate statistics, not execution traces or puzzle caches.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'invalid color')
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(cells)
    return result


def _bbox(points):
    p = np.asarray(points)
    return tuple(map(int, (*p.min(axis=0), *(p.max(axis=0) + 1))))


def _apart(a, b, gap=1):
    """Half-open rectangles have at least gap empty rows or columns between them."""
    return a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1]


def _matches(plan, sequence, axis):
    length = len(sequence)
    found = []
    for r in range(plan.shape[0] - (length - 1 if axis == 0 else 0)):
        for c in range(plan.shape[1] - (length - 1 if axis == 1 else 0)):
            if tuple(int(plan[r + (axis == 0) * k, c + (axis == 1) * k]) for k in range(length)) == sequence:
                found.append((r, c))
    return found


def _paint(grid, plan, base, side, pitch, bg, sites=None):
    """Paint complete squares, clipping exactly at the canvas boundary."""
    h, w = grid.shape
    for (r, c), color in np.ndenumerate(plan):
        if int(color) == bg or sites is not None and (r, c) not in sites:
            continue
        y, x = base[0] + r * pitch, base[1] + c * pitch
        y0, x0, y1, x1 = max(0, y), max(0, x), min(h, y + side), min(w, x + side)
        if y0 < y1 and x0 < x1:
            grid[y0:y1, x0:x1] = color


def _decode(input_grid):
    inp = _grid(input_grid)
    h, w = inp.shape
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > inp.size / 2, 'background must exceed half the input')
    singletons, squares = [], []
    for cells in _components(inp != bg):
        if len(cells) == 1:
            singletons.append(cells[0])
        else:
            r, c, er, ec = _bbox(cells)
            side = er - r
            _require(side in (3, 5) and ec - c == side and len(cells) == side * side,
                     'enlarged clues must be separated odd-sided squares')
            color = int(inp[r, c])
            _require(bool(np.all(inp[r:er, c:ec] == color)), 'non-solid clue square')
            squares.append((r, c, side, color))
    anchors = [p for p in singletons if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    swatches = [p for p in singletons if p not in anchors]
    _require(1 <= len(anchors) <= 2, 'one registration guide is required')
    marker = int(inp[anchors[0]])
    _require(all(int(inp[p]) == marker for p in anchors), 'multiple guide colors')
    rows, cols = {p[0] for p in anchors}, {p[1] for p in anchors}
    horizontal = len(rows) == 1 and all(p[1] in (0, w - 1) for p in anchors)
    vertical = len(cols) == 1 and all(p[0] in (0, h - 1) for p in anchors)
    _require(horizontal != vertical, 'ambiguous guide or corner anchor')
    axis = 0 if horizontal else 1  # changed coordinate
    guide = next(iter(rows if horizontal else cols))
    _require(0 < guide < inp.shape[axis] - 1, 'corner guide')
    _require(len(swatches) >= 3, 'miniature missing')
    mr, mc, me, ce = _bbox(swatches)
    _require(all((r - mr) % 2 == 0 and (c - mc) % 2 == 0 for r, c in swatches), 'miniature pitch is two')
    plan = inp[mr:me:2, mc:ce:2].copy()
    _require(all(2 <= d <= 5 for d in plan.shape), 'miniature dimensions must be 2..5')
    mask = np.zeros((me - mr, ce - mc), dtype=bool)
    mask[::2, ::2] = True
    _require(bool(np.all(inp[mr:me, mc:ce][~mask] == bg)), 'miniature separators changed')
    _require(np.count_nonzero(plan != bg) == len(swatches), 'unassigned singleton')
    _require(np.count_nonzero(plan == marker) == 1, 'miniature must have one handle swatch')
    handle = tuple(map(int, np.argwhere(plan == marker)[0]))
    _require(2 <= len(squares) <= 5, 'clue requires at least two squares')
    if len({s[0] for s in squares}) == 1:
        seed_axis = 1
    else:
        _require(len({s[1] for s in squares}) == 1, 'clue squares must form one straight run')
        seed_axis = 0
    squares.sort(key=lambda s: s[seed_axis])
    side = squares[0][2]
    _require(all(s[2] == side for s in squares), 'inconsistent square sizes')
    ds = np.diff([s[seed_axis] for s in squares])
    pitch = int(ds[0])
    _require(bool(np.all(ds == pitch)) and 1 <= pitch - side <= 3, 'inconsistent pitch/gutter')
    sequence = tuple(s[3] for s in squares)
    matches = _matches(plan, sequence, seed_axis)
    _require(matches, 'clue has no miniature match')
    bases = set()
    raw_bases = []
    for match in matches:
        base = [squares[0][d] - match[d] * pitch for d in (0, 1)]
        raw_bases.append(tuple(base))
        base[axis] = guide - handle[axis] * pitch - side // 2
        bases.add(tuple(base))
    _require(len(bases) == 1, 'matches disagree after registration')
    base = next(iter(bases))
    extent = tuple((d - 1) * pitch + side for d in plan.shape)
    p = 1 - axis
    _require(2 <= base[p] and base[p] + extent[p] <= inp.shape[p] - 2,
             'completed motif must fit along the guide, with a two-cell edge margin')
    target_box = (max(0, base[0]), max(0, base[1]), min(h, base[0] + extent[0]), min(w, base[1] + extent[1]))
    mini_box = (mr, mc, me, ce)
    seed_box = (min(s[0] for s in squares), min(s[1] for s in squares),
                max(s[0] + side for s in squares), max(s[1] + side for s in squares))
    _require(_apart(mini_box, target_box) and _apart(mini_box, seed_box), 'miniature must remain separate')
    clean = inp.copy()
    for r, c, s, _ in squares:
        clean[r:r + s, c:c + s] = bg
    out = clean.copy()
    _paint(out, plan, base, side, pitch, bg)
    _require(all(int(out[q]) == marker for q in anchors), 'motif overwrites anchor')
    return dict(inp=inp, output=out, clean=clean, background=bg, marker=marker, anchors=anchors,
                plan=plan, handle=handle, side=side, pitch=pitch, axis=axis, guide=guide,
                seed_axis=seed_axis, sequence=sequence, squares=squares, matches=matches,
                raw_bases=raw_bases, base=base, extent=extent, miniature=mini_box,
                clipped=base[axis] < 0 or base[axis] + extent[axis] > inp.shape[axis],
                hidden=marker not in sequence)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all geometry and registration from this input alone."""
    return _decode(input_grid)['output']


def _ablation_checks(d):
    """Every possible source match needs both alignment and completion."""
    for match, raw in zip(d['matches'], d['raw_bases']):
        no_a = d['clean'].copy()
        _paint(no_a, d['plan'], raw, d['side'], d['pitch'], d['background'])
        _require(not np.array_equal(no_a, d['output']), 'guide alignment is dispensable')
        sites = {(match[0] + (d['seed_axis'] == 0) * k,
                  match[1] + (d['seed_axis'] == 1) * k) for k in range(len(d['sequence']))}
        no_b = d['clean'].copy()
        _paint(no_b, d['plan'], d['base'], d['side'], d['pitch'], d['background'], sites)
        _require(not np.array_equal(no_b, d['output']), 'motif completion is dispensable')


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'invalid pair counts')
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'canvas changed')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            d = _decode(inp)
            _require(np.array_equal(d['output'], out), 'incorrect registration, completion, erasure or preservation')
            _ablation_checks(d)
            stats[split].append(d)
    train = stats['train']
    _require({d['axis'] for d in train} == {0, 1}, 'both guide orientations must be taught')
    _require(any(not d['hidden'] for d in train), 'no visible-handle witness')
    _require(any(d['hidden'] and len(d['matches']) >= 2 for d in train), 'no collapsed-ambiguity witness')
    _require(any(d['clipped'] for d in train) and any(not d['clipped'] for d in train), 'clipping and full completion must be taught')
    _require(any(np.any(d['plan'] == d['background']) for d in train), 'no blank-site witness')
    _require(len({d['side'] for d in train}) >= 2, 'scale inference not taught')
    _require(len({d['background'] for d in train}) >= 2, 'local palette roles not taught')


def _asymmetric(plan):
    variants = [np.flip(plan, 0), np.flip(plan, 1), np.flip(plan, (0, 1))]
    if plan.shape[0] == plan.shape[1]:
        variants += [plan.T, np.rot90(plan), np.rot90(plan, 3), np.flip(plan.T, (0, 1))]
    return all(not np.array_equal(plan, x) for x in variants)


def _sample_plan(rng, shape, bg, marker, payload, axis, mode, test):
    """Bounded plan search; all match locations are checked, not just the chosen one."""
    for _ in range(160):
        plan = rng.choice(payload, size=shape).astype(np.int8)
        seed_axis = axis if mode == 'collapsed' else int(rng.integers(2))
        if mode == 'collapsed':
            cross = int(rng.integers(shape[1 - axis]))
            period = 2 if shape[axis] >= 4 and rng.random() < .55 else 1
            length = int(rng.integers(2, shape[axis] - period + 1))
            seq_values = list(map(int, rng.choice(payload, period, replace=False)))
            line = np.array([seq_values[k % period] for k in range(shape[axis])], dtype=np.int8)
            if axis == 0:
                plan[:, cross] = line
            else:
                plan[cross, :] = line
            allowed = [(r, c) for r in range(shape[0]) for c in range(shape[1]) if (r, c)[1 - axis] != cross]
        else:
            allowed = list(np.ndindex(shape))
        handle = allowed[int(rng.integers(len(allowed)))]
        # Holes can lie at a corner or inside; each extreme must still have a witness.
        holes = rng.random(shape) < float(rng.choice([0.0, .12, .25, .35]))
        if mode == 'direct':
            holes[allowed[int(rng.integers(len(allowed)))]] = True
        if mode == 'collapsed':
            if axis == 0:
                holes[:, cross] = False
            else:
                holes[cross, :] = False
        plan[holes] = bg
        plan[handle] = marker
        if any(np.all(edge == bg) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
            continue
        if mode == 'direct' and not np.any(plan == bg):
            continue
        if not _asymmetric(plan):
            continue
        choices = []
        for length in range(2, shape[seed_axis] + 1):
            sequences = {}
            for r in range(shape[0] - (length - 1 if seed_axis == 0 else 0)):
                for c in range(shape[1] - (length - 1 if seed_axis == 1 else 0)):
                    seq = tuple(int(plan[r + (seed_axis == 0) * k, c + (seed_axis == 1) * k]) for k in range(length))
                    if bg not in seq:
                        sequences.setdefault(seq, []).append((r, c))
            for seq, matches in sequences.items():
                if len({m[1 - axis] for m in matches}) != 1:
                    continue
                if mode == 'direct' and marker not in seq:
                    continue
                if mode != 'direct' and marker in seq:
                    continue
                if mode == 'collapsed' and len(matches) < 2:
                    continue
                if mode == 'hidden' and len(matches) != 1:
                    continue
                # An unobserved occupied site is needed, even before clipping.
                if np.count_nonzero(plan != bg) <= length:
                    continue
                choices.append((seq, matches))
        if choices:
            weights = np.array([1 + (len(seq) >= 3) * (2 if test else 1) for seq, _ in choices], dtype=float)
            seq, matches = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
            return plan, handle, seed_axis, seq, matches
    return None


def _rect_positions(shape, size, forbidden):
    positions = []
    for r in range(1, shape[0] - size[0]):
        for c in range(1, shape[1] - size[1]):
            box = (r, c, r + size[0], c + size[1])
            if all(_apart(box, other) for other in forbidden):
                positions.append((r, c))
    return positions


def _fallback(rng, axis, mode, side, clipped, bg=None, variant=0):
    """A constructive layout for every requested convention after bounded search.

    Fourteen miniature offsets exceed the thirteen possible preceding inputs.
    This fallback changes sampling geometry, never the transformation rule.
    """
    colors = list(map(int, rng.permutation(10)))
    if bg is not None:
        colors.remove(bg)
        colors.insert(0, bg)
    background, marker, a, b = colors[:4]
    plan = np.array([[marker, background, b], [a, a, a]] if mode == 'collapsed'
                    else [[marker, b, a], [b, background, a]], dtype=np.int8)
    seq = (a, a) if mode == 'collapsed' else (marker, b) if mode == 'direct' else (b, a)
    pitch = side + 1
    guide = 28 if clipped else 10
    base = (3, guide - side // 2)
    mini = (22, 1 + variant)
    inp = np.full((30, 30), background, dtype=np.int8)
    inp[mini[0]:mini[0] + 3:2, mini[1]:mini[1] + 5:2] = plan
    inp[0, guide] = marker
    clean = inp.copy()
    source = (3 + (pitch if mode == 'collapsed' else 0), 3)
    for k, color in enumerate(seq):
        y, x = source[0], source[1] + k * pitch
        inp[y:y + side, x:x + side] = color
    out = clean.copy()
    _paint(out, plan, base, side, pitch, background)
    if axis == 0:
        inp, out = inp.T.copy(), out.T.copy()
    d = _decode(inp)
    _require(np.array_equal(d['output'], out), 'invalid constructive fallback')
    _ablation_checks(d)
    return {'input': inp, 'output': out}


def _make_pair(rng, axis, mode, side, clipped, bg=None):
    # At most 100 small proposals; no payloads or retries are persisted.
    for _ in range(100):
        colors = list(map(int, rng.permutation(10)))
        if bg is not None:
            colors.remove(bg)
            colors.insert(0, bg)
        background, marker = colors[:2]
        payload = colors[2:2 + int(rng.integers(2, 6))]
        gutter = int(rng.integers(1, 4))
        pitch = side + gutter
        shapes = [(h, w) for h in range(2, 6) for w in range(2, 6)
                  if ((h, w)[1 - axis] - 1) * pitch + side <= 26
                  and (mode != 'collapsed' or (h, w)[axis] >= 3)
                  and ((h, w)[axis] - 1) * pitch + side <= 37]
        shape = shapes[int(rng.integers(len(shapes)))]
        sampled = _sample_plan(rng, shape, background, marker, payload, axis, mode, mode != 'direct')
        if sampled is None:
            continue
        plan, handle, seed_axis, seq, matches = sampled
        length = len(seq)
        extent = tuple((d - 1) * pitch + side for d in shape)
        mini_size = tuple(2 * d - 1 for d in shape)
        seed_size = tuple((length - 1) * pitch + side if d == seed_axis else side for d in (0, 1))
        if max(seed_size) > 28:
            continue
        lower = [max(13, mini_size[d] + 2, seed_size[d] + 2) for d in (0, 1)]
        lower[1 - axis] = max(lower[1 - axis], extent[1 - axis] + 4)
        if max(lower) > 30:
            continue
        canvas = tuple(int(rng.integers(n, 31)) for n in lower)
        guides = [g for g in range(1, canvas[axis] - 1)
                  if ((g - handle[axis] * pitch - side // 2 < 0 or
                       g - handle[axis] * pitch - side // 2 + extent[axis] > canvas[axis]) == clipped)]
        if not guides:
            continue
        guide = int(rng.choice(guides))
        base = [0, 0]
        base[axis] = guide - handle[axis] * pitch - side // 2
        p = 1 - axis
        base[p] = int(rng.integers(2, canvas[p] - extent[p] - 1))
        base = tuple(base)
        target_box = (max(0, base[0]), max(0, base[1]), min(canvas[0], base[0] + extent[0]), min(canvas[1], base[1] + extent[1]))
        ends = [0, 1] if rng.random() < .35 else [int(rng.integers(2))]
        anchors = [(guide, end * (canvas[1] - 1)) if axis == 0 else (end * (canvas[0] - 1), guide) for end in ends]
        anchor_boxes = [(r, c, r + 1, c + 1) for r, c in anchors]
        minis = _rect_positions(canvas, mini_size, [target_box] + anchor_boxes)
        if not minis:
            continue
        mini = minis[int(rng.integers(len(minis)))]
        mini_box = (mini[0], mini[1], mini[0] + mini_size[0], mini[1] + mini_size[1])
        starts = _rect_positions(canvas, seed_size, [mini_box] + anchor_boxes)
        # No candidate source match is already aligned. This directly defeats
        # completing at the unregistered source origin, even with repeated matches.
        starts = [q for q in starts
                  if q[p] == base[p] + matches[0][p] * pitch
                  and all(q[axis] - m[axis] * pitch != base[axis] for m in matches)]
        if not starts:
            continue
        source = starts[int(rng.integers(len(starts)))]
        inp = np.full(canvas, background, dtype=np.int8)
        inp[mini[0]:mini[0] + mini_size[0]:2, mini[1]:mini[1] + mini_size[1]:2] = plan
        for q in anchors:
            inp[q] = marker
        clean = inp.copy()
        for k, color in enumerate(seq):
            y = source[0] + (seed_axis == 0) * k * pitch
            x = source[1] + (seed_axis == 1) * k * pitch
            inp[y:y + side, x:x + side] = color
        out = clean.copy()
        _paint(out, plan, base, side, pitch, background)
        pair = {'input': inp, 'output': out}
        try:
            d = _decode(inp)
            _require(np.array_equal(d['output'], out), 'construction/recovery disagree')
            _ablation_checks(d)
            _require(len(set(int(v) for v in out.flat) - {background}) >= 2, 'unhelpful clipping')
        except AssertionError:
            continue
        return pair
    return _fallback(rng, axis, mode, side, clipped, bg)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    axis = int(rng.integers(2))
    side = int(rng.choice([3, 5]))
    first_bg = int(rng.integers(10))
    second_bg = int(rng.choice([c for c in range(10) if c != first_bg]))
    plans = [(axis, 'direct', side, False, first_bg),
             (1 - axis, 'collapsed', 8 - side, True, second_bg)]
    for _ in range(nt - 2):
        plans.append((int(rng.integers(2)), str(rng.choice(['direct', 'hidden', 'collapsed'])),
                      int(rng.choice([3, 5])), bool(rng.random() < .4), None))
    for _ in range(ns):
        plans.append((int(rng.integers(2)), str(rng.choice(['hidden', 'collapsed'], p=[.4, .6])),
                      int(rng.choice([3, 5])), bool(rng.random() < .65), None))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, plan in enumerate(plans):
        for _ in range(16):
            pair = _make_pair(rng, *plan)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            for variant in range(14):
                pair = _fallback(rng, *plan, variant=variant)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError('fourteen distinct constructive layouts exhausted')
        seen.add(key)
        puzzle['train' if i < nt else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streaming seeds; at most two 14-pair puzzles live at once, no disk writes."""
    fingerprints, structures, shapes, palettes = set(), set(), set(), set()
    axes, sides, scales, gutters, dimensions, lengths, repetitions = set(), set(), set(), set(), set(), set(), set()
    metrics = dict(pairs=0, clipped=0, hidden_handles=0, collapsed_matches=0, holes=0,
                   cross_axis_seeds=0, repeated_multicolor_sequences=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                _ablation_checks(d)
                metrics['pairs'] += 1
                metrics['clipped'] += int(d['clipped'])
                metrics['hidden_handles'] += int(d['hidden'])
                metrics['collapsed_matches'] += int(len(d['matches']) > 1)
                metrics['holes'] += int(np.any(d['plan'] == d['background']))
                metrics['cross_axis_seeds'] += int(d['axis'] != d['seed_axis'])
                metrics['repeated_multicolor_sequences'] += int(len(d['matches']) > 1 and len(set(d['sequence'])) > 1)
                axes.add(d['axis'])
                scales.add(d['side'])
                gutters.add(d['pitch'] - d['side'])
                dimensions.add(d['plan'].shape)
                lengths.add(len(d['sequence']))
                repetitions.add(len(d['matches']))
                shapes.add(pair['input'].shape)
                palettes.add(d['background'])
                roles = {}
                normalized = tuple(roles.setdefault(int(v), len(roles)) for v in pair['input'].flat)
                structures.add((pair['input'].shape, normalized))
                for r, c in d['anchors']:
                    sides.add('top' if r == 0 else 'bottom' if r == pair['input'].shape[0] - 1 else 'left' if c == 0 else 'right')
        if seed < 3:
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), 'nondeterminism')
    _require(len(fingerprints) == 42 and len(structures) >= 160 and len(shapes) >= 35, 'poor structural diversity')
    _require(axes == {0, 1} and scales == {3, 5} and gutters == {1, 2, 3}, 'geometry coverage')
    _require(sides == {'left', 'right', 'top', 'bottom'} and palettes == set(range(10)), 'palette/edge coverage')
    _require(len(dimensions) >= 10 and max(lengths) >= 3 and max(repetitions) >= 3, 'plan/sequence diversity')
    _require(all(metrics[k] >= 10 for k in ('clipped', 'hidden_handles', 'collapsed_matches', 'holes', 'cross_axis_seeds')),
             'missing conceptual variation')
    _require(metrics['repeated_multicolor_sequences'] >= 3, 'only uniform ambiguity demonstrated')
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ns, nt, ns)
            _require((len(p['train']), len(p['test'])) == (nt, ns), 'counts ignored')
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Color roles are entirely input-local. Reflection changes coordinates but
    # preserves the registration rule; this also exercises bottom/right anchors.
    p = generate_puzzle(222, 2, 1)
    rng = np.random.default_rng(333)
    q = {'train': [], 'test': []}
    for split in q:
        for pair in p[split]:
            perm = rng.permutation(10)
            q[split].append({k: perm[np.flip(v, (0, 1))].astype(np.int8) for k, v in pair.items()})
    validate_generated_puzzle(q)
    for turns in (1, 3):
        rotated = {s: [{k: np.rot90(v, turns).copy() for k, v in pair.items()}
                       for pair in p[s]] for s in p}
        validate_generated_puzzle(rotated)
    for axis in (0, 1):
        for side in (3, 5):
            for clipped in (False, True):
                for mode in ('direct', 'hidden', 'collapsed'):
                    pair = _fallback(rng, axis, mode, side, clipped)
                    d = _decode(pair['input'])
                    _require(d['axis'] == axis and d['clipped'] == clipped, 'fallback convention')
    # Repetition in the perpendicular coordinate collapses; repetition along
    # the guide does not. The latter must be rejected, not silently tie-broken.
    pair = _fallback(rng, 1, 'collapsed', 3, False)
    d = _decode(pair['input'])
    ambiguous = pair['input'].copy()
    mr, mc = d['miniature'][:2]
    ambiguous[mr, mc + 2] = ambiguous[mr, mc + 4] = d['sequence'][0]
    try:
        _decode(ambiguous)
    except AssertionError as error:
        _require('matches disagree' in str(error), 'ambiguity rejected for the wrong reason')
    else:
        raise AssertionError('unresolved matching ambiguity accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, puzzles=len(fingerprints), grid_shapes=len(shapes),
                color_normalized_inputs=len(structures), plan_shapes=len(dimensions),
                scales=sorted(scales), gutters=sorted(gutters), seed_lengths=sorted(lengths),
                match_counts=sorted(repetitions), **metrics)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
