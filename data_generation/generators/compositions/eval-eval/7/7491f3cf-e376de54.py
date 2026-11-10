"""Borrow the opposite region's ordinal-middle ruler.

Two congruent framed boards: a cut/marker map and a parallel-stroke bank.
The map classifies anchors, not stroke pixels. Only marked-region strokes
are resized, using the middle stroke among opposite-region anchors.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int64, copy=False)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or not (r and c))]
    groups = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        group, stack = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    group.append(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _cut_info(cut):
    cells = set(map(tuple, np.argwhere(cut)))
    _require(len(cells) >= 3, "cut too short")
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if (dr or dc) and q in cells:
                    # Suppress diagonal shortcuts at an orthogonal corner.
                    if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                        continue
                    neighbors.append(q)
        graph[r, c] = neighbors
    ends = [p for p in graph if len(graph[p]) == 1]
    _require(len(ends) == 2 and all(1 <= len(v) <= 2 for v in graph.values()), "branched or cyclic cut")
    h, w = cut.shape
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(border(p) == (p in ends) for p in cells), "cut has invalid boundary contacts")
    _require(len(_components(cut, True)) == 1, "disconnected cut")
    regions = _components(~cut)
    _require(len(regions) == 2, "cut must create two regions")
    labels = np.full(cut.shape, -1, dtype=int)
    for i, region in enumerate(regions):
        for p in region:
            labels[p] = i
    sites = [[], []]
    for p in map(tuple, np.argwhere(~cut)):
        if min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for q in cells) >= 2:
            sites[int(labels[p])].append(p)
    _require(all(sites), "both regions need a separate marker site")
    return labels, sites


def _panels(g):
    frame = int(g[0, 0])
    _require(np.all(g[[0, -1], :] == frame) and np.all(g[:, [0, -1]] == frame), "broken frame")
    rows = list(map(int, np.flatnonzero(np.all(g == frame, axis=1))))
    cols = list(map(int, np.flatnonzero(np.all(g == frame, axis=0))))
    h, w = g.shape
    if rows == [0, h - 1] and w % 2 == 1 and cols == [0, w // 2, w - 1]:
        slices = [(slice(1, h - 1), slice(1, w // 2)),
                  (slice(1, h - 1), slice(w // 2 + 1, w - 1))]
    elif cols == [0, w - 1] and h % 2 == 1 and rows == [0, h // 2, h - 1]:
        slices = [(slice(1, h // 2), slice(1, w - 1)),
                  (slice(h // 2 + 1, h - 1), slice(1, w - 1))]
    else:
        raise AssertionError("expected two congruent framed boards")
    return frame, slices, [g[s] for s in slices]


def _line_coords(axis, r, c):
    if axis == (0, 1):
        return r, c, 1
    if axis == (1, 0):
        return c, r, 1
    if axis == (1, 1):
        return r - c, r + c, 2
    return r + c, r - c, 2


def _strokes(bank, bg, labels, marked):
    candidates = []
    for axis in ((0, 1), (1, 0), (1, 1), (1, -1)):
        raw = {}
        for r, c in map(tuple, np.argwhere(bank != bg)):
            key, t, step = _line_coords(axis, r, c)
            raw.setdefault(key, []).append((t, int(r), int(c), int(bank[r, c])))
        keys = sorted(raw)
        if len(keys) < 6 or len(keys) % 2:
            continue
        gaps = np.diff(keys)
        diagonal = all(axis)
        if np.any(gaps < (4 if diagonal else 2)) or (diagonal and np.any(gaps % 2)):
            continue
        groups = [sorted(raw[k]) for k in keys]
        if any(len({v[3] for v in group}) != 1 or
               any(b[0] - a[0] != step for a, b in zip(group, group[1:])) for group in groups):
            continue
        low = len({v[0][0] for v in groups}) == 1
        high = len({v[-1][0] for v in groups}) == 1
        if low == high:
            continue
        direction = axis if low else tuple(-x for x in axis)
        records = []
        for key, group in zip(keys, groups):
            ordered = group if low else group[::-1]
            _, r, c, color = ordered[0]
            region = int(labels[r, c])
            if region < 0:
                break
            records.append(dict(key=int(key), anchor=(r, c), color=color,
                                length=len(group), region=region,
                                cells=[(v[1], v[2]) for v in ordered]))
        else:
            recipients = [s for s in records if s['region'] == marked]
            donors = [s for s in records if s['region'] != marked]
            if all(len(x) >= 3 and len(x) % 2 for x in (recipients, donors)):
                ref = donors[len(donors) // 2]
                if ref['length'] >= 2:
                    fits = all(0 <= s['anchor'][k] + (ref['length'] - 1) * direction[k] < bank.shape[k]
                               for s in recipients for k in (0, 1))
                    if fits:
                        candidates.append(dict(strokes=records, recipients=recipients, donors=donors,
                                               reference=ref, target=ref['length'], direction=direction))
    _require(len(candidates) == 1, "ambiguous stroke bank or invalid regional populations")
    return candidates[0]


def _decode(input_grid):
    g = _grid(input_grid)
    frame, slices, panels = _panels(g)
    interpretations = []
    for ci in (0, 1):
        ctrl, bank = panels[ci], panels[1 - ci]
        if len(np.unique(ctrl)) != 2:
            continue
        for bg in map(int, np.unique(ctrl)):
            components = sorted(_components(ctrl != bg, True), key=len)
            if len(components) != 2 or len(components[0]) != 1 or len(components[1]) < 3:
                continue
            marker = components[0][0]
            cut = np.zeros(ctrl.shape, dtype=bool)
            for p in components[1]:
                cut[p] = True
            control = int(ctrl[marker])
            if len({bg, frame, control}) != 3 or np.any((bank == frame) | (bank == control)) or not np.any(bank == bg):
                continue
            try:
                labels, sites = _cut_info(cut)
                marked = int(labels[marker])
                _require(marked in (0, 1) and marker in sites[marked], "marker touches cut")
                analysis = _strokes(bank, bg, labels, marked)
            except AssertionError:
                continue
            analysis.update(grid=g, frame=frame, background=bg, control=control, marker=marker,
                            marked=marked, labels=labels, sites=sites, cut=cut, panels=panels,
                            slices=slices, control_index=ci, bank_index=1 - ci, bank=bank)
            interpretations.append(analysis)
    _require(len(interpretations) == 1, "input must have one complete interpretation")
    return interpretations[0]


def _ray(stroke, direction, length):
    r, c = stroke['anchor']
    return [(r + i * direction[0], c + i * direction[1]) for i in range(length)]


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    out = d['grid'].copy()
    bank = out[d['slices'][d['bank_index']]]
    for stroke in d['recipients']:
        for p in stroke['cells']:
            bank[p] = d['background']
        for p in _ray(stroke, d['direction'], d['target']):
            bank[p] = stroke['color']
    return out


def _pack(ctrl, bank, frame):
    h, w = ctrl.shape
    g = np.full((h + 2, 2 * w + 3), frame, dtype=np.int64)
    g[1:-1, 1:w + 1] = ctrl
    g[1:-1, w + 2:-1] = bank
    return g


def _transform(g, code):
    a = np.rot90(g, code % 4)
    return (np.fliplr(a) if code >= 4 else a).copy()


def _key(g):
    a = np.asarray(g, dtype=np.int64)
    return a.shape, a.tobytes()


def _sample_lengths(rng, count, target, capacity, donor, informative=False, mode='mixed'):
    middle = count // 2
    if donor:
        if informative:
            # Nonmedian reference; every other donor differs from the reference.
            short_majority = bool(rng.integers(2))
            lengths = [int(rng.integers(1, target)) if short_majority else
                       int(rng.integers(target + 1, capacity + 1)) for _ in range(count)]
        elif rng.random() < .25:
            lengths = [target] * count
        else:
            lengths = list(map(int, rng.integers(1, capacity + 1, count)))
        lengths[middle] = target
        return lengths
    lengths = [target] * count
    if mode == 'mixed':
        lengths = [int(rng.integers(1, capacity + 1)) for _ in range(count)]
        lengths[middle] = int(rng.integers(2, target))
        lengths[(middle + 1) % count] = int(rng.integers(target + 1, capacity + 1))
    else:
        indices = [int(rng.integers(count))] if mode.startswith('one_') else list(range(count))
        for i in indices:
            if len(indices) == 1 or rng.random() < .75:
                lengths[i] = (int(rng.integers(1, target)) if mode.endswith('extend') else
                              int(rng.integers(target + 1, capacity + 1)))
        if all(v == target for v in lengths):
            lengths[middle] = target - 1 if mode.endswith('extend') else target + 1
    return lengths


def _new_pair(rng, *, kind, marked, donor_count, receiver_count, code,
              palette, mono=False, witness=False, avoid_target=None, min_switches=2,
              crossing=False, max_switches=None):
    n = donor_count + receiver_count
    if kind == 'straight':
        membership = ([marked] * receiver_count + [1 - marked] * donor_count
                      if marked == 0 else [1 - marked] * donor_count + [marked] * receiver_count)
    elif min_switches == max_switches == 2:
        counts = {marked: receiver_count, 1 - marked: donor_count}
        outside = int(rng.integers(2))
        prefix = int(rng.integers(1, counts[outside]))
        membership = ([outside] * prefix + [1 - outside] * counts[1 - outside]
                      + [outside] * (counts[outside] - prefix))
    else:
        for _ in range(200):
            membership = [marked] * receiver_count + [1 - marked] * donor_count
            rng.shuffle(membership)
            switches = sum(a != b for a, b in zip(membership, membership[1:]))
            if switches >= min_switches and (max_switches is None or switches <= max_switches):
                break
        else:
            # At least three members per side admit five alternating switches.
            # This constructive fallback retains the required populations.
            counts = {marked: receiver_count, 1 - marked: donor_count}
            membership = [0, 1, 0, 1, 0, 1]
            membership += [1] * (counts[1] - 3) + [0] * (counts[0] - 3)
            _require(max_switches is None, "unsupported switch limit")
    gaps = [2] * (n - 1)
    if witness:
        donor_indices = [i for i, v in enumerate(membership) if v != marked]
        edge = donor_indices[0] if rng.integers(2) else donor_indices[-2]
        gaps[edge] += int(rng.choice([5, 7]))
    else:
        budget = int(rng.integers(0, min(7, 25 - sum(gaps)) + 1))
        for _ in range(budget):
            gaps[int(rng.integers(n - 1))] += 1
    span = sum(gaps)
    h = int(rng.integers(span + 3, 29))
    start = int(rng.integers(1, h - span - 1))
    rows = [start] + [start + int(v) for v in np.cumsum(gaps)]
    w = int(rng.integers(11 if crossing else 9 if kind != 'straight' else 8, 14))
    sign = (1 if marked == 0 else -1) if crossing else int(rng.choice([-1, 1]))
    if kind == 'straight':
        anchor = int(rng.integers(0, w - 4)) if sign == 1 else int(rng.integers(4, w))
    elif sign == 1:
        # Both extreme cut columns leave two background columns at the edges.
        # Thus every profile has isolated marker sites, even with short runs.
        anchor = int(rng.integers(4, w - (5 if crossing else 4)))
    else:
        anchor = int(rng.integers(5 if crossing else 4, w - 4))
    capacity = w - anchor if sign == 1 else anchor + 1
    choices = [t for t in range(4 if crossing else 3, capacity) if t != avoid_target]
    _require(choices, "no reference length fits")
    target = int(rng.choice(choices))
    donor_lengths = _sample_lengths(rng, donor_count, target, capacity, True, informative=witness)
    mode = 'mixed' if witness or crossing else str(rng.choice(['mixed', 'extend', 'trim', 'one_extend', 'one_trim']))
    receiver_lengths = _sample_lengths(rng, receiver_count, target, capacity, False, mode=mode)
    lengths = []
    di = ri = 0
    for member in membership:
        if member == marked:
            lengths.append(receiver_lengths[ri])
            ri += 1
        else:
            lengths.append(donor_lengths[di])
            di += 1
    cut = np.zeros((h, w), dtype=bool)
    if kind == 'straight':
        split = next(i for i in range(n - 1) if membership[i] != membership[i + 1])
        cut[(rows[split] + rows[split + 1]) // 2, :] = True
    else:
        # A monotone row traversal; its column swings across the anchor baseline.
        columns = [anchor + 2 if member == 0 else anchor - 2 for member in membership]
        last = 0
        col = columns[0]
        for i in range(n - 1):
            if columns[i + 1] != col:
                turn = (rows[i] + rows[i + 1]) // 2
                cut[last:turn + 1, col] = True
                cut[turn, min(col, columns[i + 1]):max(col, columns[i + 1]) + 1] = True
                last, col = turn, columns[i + 1]
        cut[last:, col] = True
    labels, sites = _cut_info(cut)
    _require(all(labels[r, anchor] == m for r, m in zip(rows, membership)), "incorrect constructed membership")
    frame, bg, control = palette[:3]
    marker = sites[marked][int(rng.integers(len(sites[marked])))]
    ctrl = np.full((h, w), bg, dtype=np.int64)
    ctrl[cut] = control
    ctrl[marker] = control
    colors = [palette[3]] * n if mono else [int(rng.choice(palette[3:])) for _ in range(n)]
    if not mono:
        ri = [i for i, m in enumerate(membership) if m == marked]
        colors[ri[0]], colors[ri[-1]] = palette[3], palette[4]
    bank = np.full((h, w), bg, dtype=np.int64)
    answer = bank.copy()
    for row, color, length, member in zip(rows, colors, lengths, membership):
        for step in range(length):
            bank[row, anchor + sign * step] = color
        final_length = target if member == marked else length
        for step in range(final_length):
            answer[row, anchor + sign * step] = color
    return {'input': _transform(_pack(ctrl, bank, frame), code),
            'output': _transform(_pack(ctrl, answer, frame), code)}


def _opposite(pair, rng):
    d = _decode(pair['input'])
    inp = d['grid'].copy()
    ctrl = inp[d['slices'][d['control_index']]]
    ctrl[d['marker']] = d['background']
    sites = d['sites'][1 - d['marked']]
    ctrl[sites[int(rng.integers(len(sites)))]] = d['control']
    # This controlled counterfactual is constructed from visible input data.
    return {'input': inp, 'output': solve(inp)}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    base_code = int(rng.integers(8))
    side = int(rng.integers(2))
    regime = str(rng.choice(['switches', 'larger_donor', 'rotate']))
    base_palette = tuple(map(int, rng.permutation(10)))
    fixed_palette = bool(rng.random() < .3)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for attempt in range(100):
                palette = base_palette if fixed_palette or (split == 'train' and index == 0) else tuple(map(int, rng.permutation(10)))
                code = base_code
                if split == 'train' and index == 0:
                    kwargs = dict(kind='straight', marked=side, donor_count=5, receiver_count=3,
                                  mono=True, witness=True)
                elif split == 'train' and index == 1:
                    first = _decode(puzzle['train'][0]['input'])
                    # After D4 transformation, component zero is the region
                    # containing the top-left corner. Teach both actual sides.
                    corner_side = int(_transform(np.array([[0, 1], [0, 1]]), code)[0, 0])
                    second_side = (1 - first['marked']) ^ corner_side
                    kwargs = dict(kind='profile', marked=second_side, donor_count=3, receiver_count=3,
                                  crossing=True, avoid_target=first['target'], max_switches=2)
                elif split == 'train' and index == 2 and nt >= 4 and rng.random() < .35:
                    pair = _opposite(puzzle['train'][0], rng)
                    if _key(pair['input']) not in seen:
                        break
                    continue
                else:
                    donor_count = 7 if split == 'test' and regime == 'larger_donor' else int(rng.choice([3, 5]))
                    receiver_count = int(rng.choice([3, 5]))
                    kind = 'profile' if split == 'test' and regime == 'switches' else str(rng.choice(['straight', 'profile'], p=[.35, .65]))
                    if split == 'test' and regime == 'rotate':
                        code = (base_code // 4) * 4 + (base_code + int(rng.choice([1, 3]))) % 4
                    elif split == 'train' and index >= 2 and regime != 'rotate' and rng.random() < .3:
                        code = int(rng.integers(8))
                    kwargs = dict(kind=kind, marked=int(rng.integers(2)), donor_count=donor_count,
                                  receiver_count=receiver_count, mono=bool(rng.random() < .25),
                                  min_switches=3 if split == 'test' and regime == 'switches' else 2,
                                  max_switches=2 if split == 'train' and regime == 'switches' else None)
                pair = _new_pair(rng, code=code, palette=palette, **kwargs)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError("unique input sampling exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(d):
    donor = d['donors']
    keys = np.array([s['key'] for s in donor])
    distances = abs(2 * keys - keys[0] - keys[-1])
    nearest = np.flatnonzero(distances == distances.min())
    geometric = (len(nearest) == 1 and donor[int(nearest[0])]['length'] != d['target'])
    crossing = any(d['labels'][p] == 1 - d['marked'] for s in d['recipients']
                   for p in _ray(s, d['direction'], d['target']))
    membership = [s['region'] for s in d['strokes']]
    switches = sum(a != b for a, b in zip(membership, membership[1:]))
    return dict(nonmedian=sorted(s['length'] for s in donor)[len(donor) // 2] != d['target'],
                geometric=bool(geometric), crossing=crossing, interleaved=switches >= 2,
                extend=any(s['length'] < d['target'] for s in d['recipients']),
                trim=any(s['length'] > d['target'] for s in d['recipients']),
                wrong_own_reference=d['recipients'][len(d['recipients']) // 2]['length'] != d['target'])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and _key(inp) not in seen, "shape mismatch or duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            bank_out = out[d['slices'][d['bank_index']]]
            expected_bank = d['bank'].copy()
            for s in d['recipients']:
                # Cellwise length predicate, rather than invoking solve.
                for p in s['cells']:
                    expected_bank[p] = d['background']
                for p in _ray(s, d['direction'], d['target']):
                    expected_bank[p] = s['color']
                for p in _ray(s, d['direction'], min(s['length'], d['target'])):
                    _require(bank_out[p] == s['color'], "changed anchor or shared prefix")
            _require(np.array_equal(bank_out, expected_bank), "incorrect recipient resizing")
            restored = out.copy()
            restored[d['slices'][d['bank_index']]] = d['bank']
            _require(np.array_equal(restored, inp), "changed map, frame or divider")
            for s in d['donors']:
                _require(all(bank_out[p] == s['color'] for p in s['cells']), "changed donor")
            _require(not np.array_equal(inp, out), "identity example")
            if split == 'train':
                train.append(d)
    flags = [_evidence(d) for d in train]
    for name in ('nonmedian', 'geometric', 'crossing', 'interleaved', 'extend', 'trim', 'wrong_own_reference'):
        _require(any(f[name] for f in flags), f"training lacks {name} evidence")
    _require(len({d['target'] for d in train}) >= 2, "training has only one ruler length")
    _require(len({len(d['donors']) for d in train}) >= 2, "training has only one donor count")
    _require({d['marked'] for d in train} == {0, 1}, "training does not select both map components")
    _require(any(len({s['color'] for s in d['strokes']}) == 1 for d in train), "missing monochrome demonstration")
    _require(any(len({s['color'] for s in d['strokes']}) > 1 for d in train), "missing color-preservation demonstration")
    _require(any(len({s['color'] for s in d['recipients']}) > 1 for d in train), "recipient color copying not taught")


def self_test():
    """Bounded structural, counterfactual, equivariance and 42-seed campaign."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({tuple(_key(p['input']) for p in task['train'] + task['test']) for task in puzzles}) == 42,
             "seed diversity collapsed")
    shapes, maps, arrangements, donor_counts, targets, backgrounds = set(), set(), set(), set(), set(), set()
    shifts = set()
    rng = np.random.default_rng(99137)
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        if seed < 3:
            other = generate_puzzle(seed)
            _require(all(np.array_equal(a[k], b[k]) for split in puzzle
                         for a, b in zip(puzzle[split], other[split]) for k in a), "nondeterminism")
        by_split = {s: [_decode(p['input']) for p in puzzle[s]] for s in puzzle}
        def switches(d):
            members = [s['region'] for s in d['strokes']]
            return sum(a != b for a, b in zip(members, members[1:]))
        observed = set()
        if max(switches(d) for d in by_split['train']) <= 2 and min(switches(d) for d in by_split['test']) >= 3:
            observed.add('switches')
        if max(len(d['donors']) for d in by_split['train']) <= 5 and all(len(d['donors']) == 7 for d in by_split['test']):
            observed.add('larger_donor')
        axes = {s: {tuple(abs(x) for x in d['direction']) for d in by_split[s]} for s in puzzle}
        if axes['train'].isdisjoint(axes['test']):
            observed.add('rotate')
        _require(observed, "missing structural train/test transfer")
        shifts |= observed
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                _require(np.array_equal(solve(pair['input']), pair['output']), "solve mismatch")
                shapes.add(pair['input'].shape)
                maps.add((d['cut'].shape, d['cut'].tobytes()))
                arrangements.add((tuple(s['key'] for s in d['strokes']),
                                  tuple(s['region'] for s in d['strokes']),
                                  tuple(s['length'] for s in d['strokes'])))
                donor_counts.add(len(d['donors']))
                targets.add(d['target'])
                backgrounds.add(d['background'])
        pair = puzzle['train'][0]
        for code in range(8):
            _require(np.array_equal(solve(_transform(pair['input'], code)), _transform(pair['output'], code)),
                     "rotation/reflection does not commute")
        palette = rng.permutation(10)
        _require(np.array_equal(solve(palette[pair['input']]), palette[pair['output']]), "palette does not commute")
        opposite = _opposite(pair, rng)
        _require(not np.array_equal(opposite['output'], pair['output']), "marker has no effect")
        # A donor-reference edit changes recipients, even though donor editing
        # is not itself the rule's output operation.
        d = _decode(pair['input'])
        altered = pair['input'].copy()
        bank = altered[d['slices'][d['bank_index']]]
        ref = d['reference']
        bank[ref['cells'][-1]] = d['background']
        changed = solve(altered)
        before = pair['output'][d['slices'][d['bank_index']]]
        after = changed[d['slices'][d['bank_index']]]
        _require(any(before[p] != after[p] for s in d['recipients']
                     for p in _ray(s, d['direction'], d['target'])), "reference perturbation ignored")
    _require(len(shapes) >= 20 and len(maps) >= 100 and len(arrangements) >= 150, "structural variation collapsed")
    _require(donor_counts == {3, 5, 7} and len(targets) >= 6 and backgrounds == set(range(10)), "parameter range collapsed")
    _require(shifts == {'switches', 'larger_donor', 'rotate'}, "transfer regimes collapsed")
    for nt in range(2, 11):
        for nq in range(1, min(nt - 1, 4) + 1):
            p = generate_puzzle(7301, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "counts ignored")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted frame accepted")
    for role in ('recipient', 'donor', 'map'):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        pair = damaged['train'][0]
        d = _decode(pair['input'])
        if role == 'map':
            pair['output'][d['slices'][d['control_index']]][d['marker']] = d['background']
        else:
            stroke = d['recipients'][0] if role == 'recipient' else d['donors'][0]
            pair['output'][d['slices'][d['bank_index']]][stroke['anchor']] = d['background']
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"corrupted {role} accepted")
    return {'seeds': 42, 'shapes': len(shapes), 'cut_structures': len(maps),
            'anchor_length_structures': len(arrangements), 'donor_counts': sorted(donor_counts),
            'target_lengths': sorted(targets), 'backgrounds': sorted(backgrounds), 'shift_regimes': sorted(shifts)}


if __name__ == '__main__':
    print(self_test())
