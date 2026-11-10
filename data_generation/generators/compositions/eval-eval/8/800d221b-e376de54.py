"""Elected ruler: port voting chooses the ordered subset that supplies a ruler.

No parent imports, generation records, or puzzle state are used by solve().
Only the standard library and NumPy are required.
"""
from __future__ import annotations

import itertools
import json
import numpy as np


_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _check_counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    seen = set()
    components = []
    h, w = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        p = int(r), int(c)
        if p in seen:
            continue
        seen.add(p)
        todo, cells = [p], []
        while todo:
            x, y = todo.pop()
            cells.append((x, y))
            for dr, dc in _DIRS:
                q = x + dr, y + dc
                if (0 <= q[0] < h and 0 <= q[1] < w
                        and mask[q] and q not in seen):
                    seen.add(q)
                    todo.append(q)
        components.append(cells)
    return components


def _interpret(g, background, scaffold):
    """Parse one candidate oriented with fingers pointing to the right."""
    signals = tuple(int(c) for c in np.unique(g)
                    if c not in (background, scaffold))
    assert len(signals) == 2
    fingers = []
    for cells in _components(g == scaffold):
        rows = {p[0] for p in cells}
        cols = sorted(p[1] for p in cells)
        assert len(rows) == 1 and cols == list(range(cols[0], cols[-1] + 1))
        fingers.append((next(iter(rows)), cols[0], len(cols)))
    fingers.sort()
    assert 4 <= len(fingers) <= 9
    anchor = fingers[0][1]
    assert all(c == anchor for r, c, length in fingers)
    assert all(b[0] - a[0] >= 2 for a, b in zip(fingers, fingers[1:]))
    panel_ids = np.full(g.shape, -1, dtype=int)
    panels = []
    mask = (g == signals[0]) | (g == signals[1])
    for cells in _components(mask):
        rs, cs = zip(*cells)
        box = min(rs), max(rs) + 1, min(cs), max(cs) + 1
        r0, r1, c0, c1 = box
        assert r1 - r0 >= 2 and len(cells) == (r1 - r0) * (c1 - c0)
        assert c1 == anchor and len(cells) >= 3
        counts = tuple(sum(g[p] == s for p in cells) for s in signals)
        assert min(counts) > 0 and counts[0] != counts[1]
        label = signals[int(counts[1] > counts[0])]
        for p in cells:
            panel_ids[p] = len(panels)
        panels.append(dict(box=box, cells=cells, counts=counts, label=label))
    assert 2 <= len(panels) <= 6
    owners = [int(panel_ids[r, c - 1]) for r, c, length in fingers]
    assert min(owners) >= 0 and set(owners) == set(range(len(panels)))
    labels = [panels[i]['label'] for i in owners]
    votes = [labels.count(s) for s in signals]
    assert min(votes) > 0 and votes[0] != votes[1]
    winner = signals[int(votes[1] > votes[0])]
    eligible = [i for i, s in enumerate(labels) if s == winner]
    assert len(eligible) >= 3 and len(eligible) % 2 == 1
    reference_index = eligible[len(eligible) // 2]
    reference = fingers[reference_index][2]
    assert reference >= 2 and anchor + reference < g.shape[1]
    return dict(background=background, scaffold=scaffold, signals=signals,
                fingers=fingers, panels=panels, owners=owners, labels=labels,
                votes=votes, winner=winner, eligible=eligible,
                reference_index=reference_index, reference=reference)


def _analyze(input_grid):
    g = _grid(input_grid)
    background = int(g[0, 0])
    assert (np.all(g[0] == background) and np.all(g[-1] == background)
            and np.all(g[:, 0] == background) and np.all(g[:, -1] == background))
    values = tuple(int(c) for c in np.unique(g))
    assert len(values) == 4
    candidates = []
    for turns in range(4):
        oriented = np.rot90(g, turns)
        for scaffold in values:
            if scaffold == background:
                continue
            try:
                info = _interpret(oriented, background, scaffold)
            except (AssertionError, IndexError):
                continue
            info['turns'] = turns
            info['oriented'] = oriented
            candidates.append(info)
    assert len(candidates) == 1, "input needs one panel-and-finger interpretation"
    return candidates[0]


def _paint(info, reference=None):
    """Apply the rule in parsed coordinates, then undo the parsing rotation."""
    g = info['oriented']
    out = g.copy()
    out[out == info['scaffold']] = info['background']
    span = info['reference'] if reference is None else int(reference)
    for (r, c, length), label in zip(info['fingers'], info['labels']):
        assert c + span < g.shape[1]
        out[r, c:c + span] = label
    return np.rot90(out, -info['turns']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, both votes, the filtered middle and its length from input."""
    return _paint(_analyze(input_grid))


def _compositions(total, count):
    if count == 1:
        yield (total,)
    else:
        for first in range(1, total - count + 2):
            for rest in _compositions(total - first, count - 1):
                yield (first,) + rest


def _topology(rng, kind, allowed_counts):
    if kind == 'weighted':
        # Five elected fingers on one panel defeat panel/area counting. The
        # group can be at either end; all geometry and contents are resampled.
        n = int(rng.choice([n for n in (7, 8) if n in allowed_counts]))
        weights = [5] + [1] * (n - 5)
        labels = [True] + [False] * (n - 5)
        if rng.integers(2):
            weights.reverse()
            labels.reverse()
        return weights, labels
    if kind == 'separated':
        # Elected ports straddle a losing panel. A global geometric/ordinal
        # center therefore cannot substitute for the filtered middle.
        weights, labels = [2, 1, 1, 1], [True, False, True, False]
        if rng.integers(2):
            weights.reverse()
            labels.reverse()
        return weights, labels
    n = int(rng.choice(allowed_counts))
    count = int(rng.integers(2, min(6, n) + 1))
    options = []
    for weights in _compositions(n, count):
        for labels in itertools.product((False, True), repeat=count):
            elected = sum(w for w, label in zip(weights, labels) if label)
            if n / 2 < elected < n and elected >= 3 and elected % 2 == 1:
                options.append((weights, labels))
    weights, labels = options[int(rng.integers(len(options)))]
    return list(weights), list(labels)


def _layout(rng, weights, kind):
    """Fit independently composed panel heights, port gaps and blank separators."""
    gaps = [[2] * (w - 1) for w in weights]
    pads = [[0, 0] for w in weights]
    for i, w in enumerate(weights):
        if w == 1:
            pads[i][int(rng.integers(2))] = 1
    separators = [1] * (len(weights) - 1)
    if kind == 'weighted':
        heavy = weights.index(5)
        end = 0 if rng.integers(2) else 3
        gaps[heavy][end] = int(rng.integers(7, 11))
    used = (sum(1 + sum(gs) + sum(ps) for gs, ps in zip(gaps, pads))
            + sum(separators) + 2)
    budget = int(rng.integers(0, min(7, 30 - used) + 1))
    choices = [('pad', i, j) for i in range(len(weights)) for j in range(2)]
    choices += [('sep', i, 0) for i in range(len(separators))]
    if kind != 'weighted':
        choices += [('gap', i, j) for i, gs in enumerate(gaps) for j in range(len(gs))]
    for _ in range(budget):
        typ, i, j = choices[int(rng.integers(len(choices)))]
        if typ == 'pad':
            pads[i][j] += 1
        elif typ == 'sep':
            separators[i] += 1
        else:
            gaps[i][j] += 1
    y, panels, port_rows = 0, [], []
    for i, (gs, ps) in enumerate(zip(gaps, pads)):
        rows = [y + ps[0]]
        for gap in gs:
            rows.append(rows[-1] + gap)
        height = 1 + sum(gs) + sum(ps)
        panels.append((y, height, rows))
        port_rows.extend(rows)
        y += height + (separators[i] if i < len(separators) else 0)
    return panels, port_rows, y


def _pattern(rng, height, width, majority, minority, strength=None):
    area = height * width
    assert area >= 3
    if strength == 'close':
        nminor = (area - 1) // 2
    elif strength == 'strong':
        nminor = 1
    else:
        nminor = int(rng.integers(1, (area - 1) // 2 + 1))
    rr, cc = np.indices((height, width))
    style = int(rng.integers(4))
    if style == 0:
        order = rng.permutation(area)
    elif style == 1:
        x, y = int(rng.integers(height)), int(rng.integers(width))
        order = np.argsort((abs(rr - x) + abs(cc - y) + rng.random((height, width)) * .4).ravel())
    elif style == 2:
        order = np.argsort(((rr + cc) % 2 * area + rr * width + cc).ravel())
    else:
        order = np.argsort((rr * width + cc if rng.integers(2) else cc * height + rr).ravel())
    flat = np.full(area, majority, dtype=np.int8)
    flat[order[:nminor]] = minority
    return flat.reshape(height, width)


def _make_pair(rng, kind, roles, winner_index, turns, allowed_counts,
               reference_choices, avoid_reference=None):
    background, scaffold, s0, s1 = roles
    signals = sorted((s0, s1))
    winner, loser = signals[winner_index], signals[1 - winner_index]
    for _ in range(100):
        weights, elected_panels = _topology(rng, kind, allowed_counts)
        panel_layout, rows, height = _layout(rng, weights, kind)
        widths = [int(rng.integers(2, 10)) for _ in weights]
        if kind == 'weighted':
            widths = [int(rng.integers(1, 3)) if label else int(rng.integers(7, 11))
                      for label in elected_panels]
            areas = [w * p[1] for w, p in zip(widths, panel_layout)]
            if sum(a if label else -a for a, label in zip(areas, elected_panels)) >= 0:
                continue
        labels = [winner if label else loser for w, label in zip(weights, elected_panels)
                  for _ in range(w)]
        eligible = [i for i, c in enumerate(labels) if c == winner]
        reference_index = eligible[len(eligible) // 2]
        choices = [r for r in reference_choices if r != avoid_reference]
        reference = int(rng.choice(choices))
        capacity = int(rng.integers(max(reference + 3, 12), 20))
        if max(widths) + capacity + 2 > 30:
            capacity = 28 - max(widths)
        lengths = [reference] * len(rows)
        if kind == 'weighted':
            for i in range(len(rows)):
                if i != reference_index:
                    lengths[i] = (int(rng.integers(1, reference // 2 + 1))
                                  if labels[i] == winner else capacity)
            if sum(length if label == winner else -length
                   for length, label in zip(lengths, labels)) >= 0:
                continue
        elif kind == 'separated':
            for i in range(len(rows)):
                if i != reference_index:
                    lengths[i] = (int(rng.integers(1, reference)) if labels[i] == winner
                                  else int(rng.integers(reference + 1, capacity + 1)))
        else:
            mode = int(rng.integers(5))  # mixed, extend, trim, one change, identity lengths
            others = [i for i in range(len(rows)) if i != reference_index]
            for i in others:
                if mode == 4 or rng.random() < .23:
                    continue
                short = mode == 1 or (mode not in (1, 2) and rng.integers(2))
                lengths[i] = (int(rng.integers(1, reference)) if short
                              else int(rng.integers(reference + 1, capacity + 1)))
            if mode == 3:
                changed = int(rng.choice(others))
                lengths = [reference] * len(rows)
                lengths[changed] = int(rng.choice([1, capacity]))
        used_width = max(widths) + max(max(lengths), reference)
        extra_h = int(rng.integers(0, min(5, 28 - height) + 1))
        extra_w = int(rng.integers(0, min(5, 28 - used_width) + 1))
        top = 1 + int(rng.integers(extra_h + 1))
        left = 1 + int(rng.integers(extra_w + 1))
        shape = height + extra_h + 2, used_width + extra_w + 2
        anchor = left + max(widths)
        inp = np.full(shape, background, dtype=np.int8)
        out = inp.copy()
        for (y, ph, group_rows), width, elected in zip(panel_layout, widths, elected_panels):
            major, minor = (winner, loser) if elected else (loser, winner)
            strength = ('close' if elected else 'strong') if kind == 'weighted' else None
            pattern = _pattern(rng, ph, width, major, minor, strength)
            if kind == 'weighted' and elected:
                # The ruler's attachment deliberately touches a minority cell.
                # Swap within the panel so its vote and all area witnesses stay
                # unchanged. Copying the nearest pixel must fail in training.
                attachment = (rows[reference_index] - y, width - 1)
                if pattern[attachment] != minor:
                    minorities = np.argwhere(pattern == minor)
                    source = tuple(minorities[int(rng.integers(len(minorities)))])
                    pattern[attachment], pattern[source] = minor, major
            inp[top + y:top + y + ph, anchor - width:anchor] = pattern
            out[top + y:top + y + ph, anchor - width:anchor] = pattern
        for row, length, label in zip(rows, lengths, labels):
            inp[top + row, anchor:anchor + length] = scaffold
            out[top + row, anchor:anchor + reference] = label
        # This ground truth is drawn from sampled geometry, independently of
        # the input parser. A transverse reflection changes no ordinal middle.
        if rng.integers(2):
            inp, out = np.flipud(inp), np.flipud(out)
        inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
        try:
            info = _analyze(inp)
            assert np.array_equal(_paint(info), out)
            if kind == 'weighted':
                features = _features(info)
                assert all(features[k] for k in ('wrong_panel_vote', 'wrong_raw_vote',
                            'wrong_area_vote', 'wrong_route_area_vote', 'nonmedian',
                            'filtered_center_witness', 'geometric_witness',
                            'minority_at_reference'))
            return {'input': inp, 'output': out}, reference
        except AssertionError:
            continue
    raise RuntimeError("bounded panel/finger construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # directed-axis, count, or length transfer
    varying_palette = bool(rng.random() < .7)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    base_turns, phase = int(rng.integers(4)), int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    references = []
    first_winner = None
    for split, count in (('train', ntrain), ('test', ntest)):
        for index in range(count):
            kind = ('weighted' if index == 0 else 'separated' if index == 1 else 'normal') if split == 'train' else 'normal'
            allowed = list(range(4, 10))
            if regime == 1:
                allowed = list(range(4, 8)) if split == 'train' else [8, 9]
            choices = list(range(3, 10))
            if regime == 2:
                choices = list(range(3, 7)) if split == 'train' else list(range(7, 13))
            turns = (base_turns if split == 'train' else (base_turns + 1 + 2 * int(rng.integers(2))) % 4) if regime == 0 else int(rng.integers(4))
            winner_index = (phase if index == 0 else 1 - phase) if split == 'train' and index < 2 else int(rng.integers(2))
            avoid = references[0] if split == 'train' and index == 1 else None
            for _ in range(20):
                roles = tuple(map(int, rng.choice(10, 4, replace=False))) if varying_palette else palette
                if split == 'train' and index == 1 and sorted(roles[2:])[winner_index] == first_winner:
                    continue
                pair, reference = _make_pair(rng, kind, roles, winner_index, turns,
                                             allowed, choices, avoid)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness retry exhausted")
            puzzle[split].append(pair)
            seen.add(key)
            if split == 'train':
                references.append(reference)
                if index == 0:
                    first_winner = sorted(roles[2:])[winner_index]
    # Witness positions are not a convention for the solver.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(g):
    a = np.asarray(g)
    return a.shape, a.astype(np.int8).tobytes()


def _features(info):
    panels, winner = info['panels'], info['winner']
    weights = [info['owners'].count(i) for i in range(len(panels))]
    lengths = [f[2] for f in info['fingers']]
    rows = [info['fingers'][i][0] for i in info['eligible']]
    distances = [abs(2 * r - rows[0] - rows[-1]) for r in rows]
    nearest = [i for i, d in enumerate(distances) if d == min(distances)]
    reference = info['reference']
    reference_row, reference_col, _ = info['fingers'][info['reference_index']]
    global_centers = [(len(lengths) - 1) // 2, len(lengths) // 2]
    return dict(count=len(lengths), panel_count=len(panels), weights=tuple(weights),
                direction=info['turns'],
                reference=reference, winner_index=info['signals'].index(winner),
                eligible_count=len(info['eligible']),
                extensions=sum(n < reference for n in lengths),
                trims=sum(n > reference for n in lengths),
                minority_at_reference=bool(info['oriented'][reference_row, reference_col - 1]
                                           != winner),
                nonmedian=int(np.median(lengths)) != reference,
                filtered_center_witness=all(lengths[i] != reference for i in global_centers),
                geometric_witness=(len(nearest) == 1 and nearest[0] != len(rows) // 2
                                   and lengths[info['eligible'][nearest[0]]] != reference),
                wrong_panel_vote=sum(1 if p['label'] == winner else -1 for p in panels) < 0,
                wrong_area_vote=sum(len(p['cells']) * (1 if p['label'] == winner else -1) for p in panels) < 0,
                wrong_raw_vote=sum((p['counts'][info['signals'].index(winner)]
                                    - p['counts'][1 - info['signals'].index(winner)]) for p in panels) < 0,
                wrong_route_area_vote=sum(n * (1 if label == winner else -1)
                                          for n, label in zip(lengths, info['labels'])) < 0)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, measures = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(_paint(info), out), 'incorrect elected-ruler output'
            assert not np.array_equal(inp, out)
            # Preservation and free-tail invariants in the parsed coordinate system.
            canonical_out = np.rot90(out, info['turns'])
            assert not np.any(canonical_out == info['scaffold'])
            for panel in info['panels']:
                for p in panel['cells']:
                    assert canonical_out[p] == info['oriented'][p]
            for (r, c, n), label in zip(info['fingers'], info['labels']):
                assert np.all(canonical_out[r, c:c + info['reference']] == label)
                assert np.all(canonical_out[r, c + info['reference']:c + max(n, info['reference'])] == info['background'])
            measures[split].append(_features(info))
    train, test = measures['train'], measures['test']
    assert len({f['reference'] for f in train}) >= 2
    assert len({f['count'] for f in train}) >= 2
    assert {f['winner_index'] for f in train} == {0, 1}
    assert any(f['extensions'] and f['trims'] for f in train)
    assert any(all(f[k] for k in ('wrong_panel_vote', 'wrong_raw_vote', 'wrong_area_vote',
                                  'wrong_route_area_vote', 'nonmedian',
                                  'filtered_center_witness', 'geometric_witness',
                                  'minority_at_reference')) for f in train)
    assert any(len(set(f['weights'])) > 1 for f in train)
    assert len({_analyze(p['input'])['winner'] for p in puzzle['train']}) >= 2
    # Verify an observable held-out feature without needing the sampled regime.
    train_directions = {f['direction'] for f in train}
    direction_shift = (len(train_directions) == 1 and all(
        (f['direction'] - next(iter(train_directions))) % 2 == 1 for f in test))
    count_shift = max(f['count'] for f in train) < min(f['count'] for f in test)
    length_shift = max(f['reference'] for f in train) < min(f['reference'] for f in test)
    assert direction_shift or count_shift or length_shift, 'missing train/test shift'


def _boundary_fixture(lengths):
    """Hand-built family member; not called by generation or input inference."""
    # Panel labels W,L,W,L with weights 2,1,1,1; elected ranks 0,1,3.
    assert len(lengths) == 5
    anchor, rows, labels = 4, [1, 3, 6, 10, 14], [1, 1, 2, 1, 2]
    inp = np.zeros((17, anchor + max(lengths) + 1), dtype=np.int8)
    for start, label in ((1, 1), (5, 2), (9, 1), (13, 2)):
        inp[start:start + 3, 1:anchor] = label
        inp[start + 2, anchor - 1] = 3 - label
    # The first panel's minority is directly behind its elected ruler.
    assert inp[3, anchor - 1] == 2
    out = inp.copy()
    for row, length, label in zip(rows, lengths, labels):
        inp[row, anchor:anchor + length] = 9
        out[row, anchor:anchor + lengths[1]] = label
    return {'input': inp, 'output': out}


def self_test():
    """Bounded 42-seed campaign, count boundaries, symmetries and ablations."""
    shapes, structures, palettes, counts, references, weight_profiles = [set() for _ in range(6)]
    rows = []
    rng = np.random.default_rng(18329)
    # A two-cell ruler, all-correct lengths, and a long reference outside the
    # sampler's range are legal under the specification and solved from input.
    for lengths in ([1, 2, 4, 1, 3], [2, 2, 2, 2, 2], [1, 24, 25, 12, 25]):
        fixture = _boundary_fixture(lengths)
        assert np.array_equal(solve(fixture['input']), fixture['output'])
        for k in range(4):
            assert np.array_equal(solve(np.rot90(fixture['input'], k)), np.rot90(fixture['output'], k))
    fixture = _boundary_fixture([1, 2, 4, 1, 3])
    moved = fixture['input'].copy()
    moved[3, 3], moved[2, 1] = 1, 2
    expected = fixture['output'].copy()
    expected[3, 3], expected[2, 1] = 1, 2
    assert np.array_equal(solve(moved), expected)
    assert fixture['output'][3, 4] != fixture['input'][3, 3]
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], again[s]) for k in a)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info = _analyze(inp)
                features = _features(info)
                counts.add(features['count'])
                references.add(features['reference'])
                weight_profiles.add(tuple(sorted(features['weights'])))
                shapes.add(inp.shape)
                palettes.add(tuple(info[k] for k in ('background', 'scaffold')) + info['signals'])
                # Crop and canonicalize literal color roles, rather than relying on hashes.
                g = info['oriented']
                coords = np.argwhere(g != info['background'])
                lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
                norm = np.zeros(g.shape, dtype=np.int8)
                for c, role in ((info['scaffold'], 1), (info['winner'], 2),
                                (next(s for s in info['signals'] if s != info['winner']), 3)):
                    norm[g == c] = role
                structures.add(_key(norm[lo[0]:hi[0], lo[1]:hi[1]]))
                rows.append(dict(seed=seed, split=split, shape=inp.shape, **features))
                if split == 'test':
                    # Literal roles and all four directed cardinal axes are arbitrary.
                    perm = rng.permutation(10)
                    assert np.array_equal(solve(perm[inp]), perm[out])
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                    assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
        witness = next(pair for pair in puzzle['train']
                       if all(_features(_analyze(pair['input']))[k] for k in
                              ('wrong_panel_vote', 'wrong_route_area_vote',
                               'geometric_witness', 'filtered_center_witness')))
        info = _analyze(witness['input'])
        # Omitting A's fan-out weighting chooses the other signal. Every losing
        # finger in this witness is long, so either middle for an even losing
        # subset gives the same wrong ruler without an arbitrary tie-break.
        losing_lengths = [finger[2] for finger, label in zip(info['fingers'], info['labels'])
                          if label != info['winner']]
        assert len(set(losing_lengths)) == 1
        assert not np.array_equal(_paint(info, losing_lengths[0]), witness['output'])
        # Even with the correct ruler granted, copying each attachment pixel
        # instead of counting its panel gives a wrong propagated color.
        no_local_vote = info['oriented'].copy()
        no_local_vote[no_local_vote == info['scaffold']] = info['background']
        for r, c, n in info['fingers']:
            no_local_vote[r, c:c + info['reference']] = info['oriented'][r, c - 1]
        assert not np.array_equal(np.rot90(no_local_vote, -info['turns']), witness['output'])
        # Change two equal-weight panels' local majorities in a valid input.
        # Port totals stay fixed, but the resulting finger labels must change.
        local_infos = [_analyze(pair['input']) for pair in puzzle['train']]
        local_info = next(a for a in local_infos if len(a['fingers']) == 5
                          and sorted(a['owners'].count(i) for i in range(len(a['panels']))) == [1, 1, 1, 2]
                          and a['owners'].count(a['owners'][a['reference_index']]) == 2
                          and a['reference_index'] in (1, 3))
        panel_pair = next((a, b) for a, pa in enumerate(local_info['panels'])
                          for b, pb in enumerate(local_info['panels']) if a != b
                          and pa['label'] != pb['label']
                          and local_info['owners'].count(a) == local_info['owners'].count(b) == 1)
        altered = local_info['oriented'].copy()
        for pid in panel_pair:
            for p in local_info['panels'][pid]['cells']:
                altered[p] = next(s for s in local_info['signals'] if s != int(altered[p]))
        changed_output = np.rot90(solve(np.rot90(altered, -local_info['turns'])), local_info['turns'])
        for finger, owner, old_label in zip(local_info['fingers'], local_info['owners'], local_info['labels']):
            if owner in panel_pair:
                assert changed_output[finger[0], finger[1]] != old_label
        # Omitting B: retain original lengths; the guaranteed mixed witness differs.
        no_b = info['oriented'].copy()
        for (r, c, n), label in zip(info['fingers'], info['labels']):
            no_b[r, c:c + n] = label
        assert not np.array_equal(np.rot90(no_b, -info['turns']), witness['output'])
        # Omitting the election filter: the whole-array middle is wrong.
        whole_middle = info['fingers'][len(info['fingers']) // 2][2]
        assert not np.array_equal(_paint(info, whole_middle), witness['output'])
        # Changing the visible elected ruler length changes all target lengths.
        changed = info['oriented'].copy()
        r, c, n = info['fingers'][info['reference_index']]
        changed[r, c + n - 1] = info['background']
        assert not np.array_equal(solve(np.rot90(changed, -info['turns'])), witness['output'])
    assert counts == set(range(4, 10)) and len(references) >= 8
    assert len(shapes) >= 40 and len(structures) >= 180 and len(palettes) >= 50
    assert len(weight_profiles) >= 15
    # Exercise every explicit legal count combination, and common illegal types.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(3201, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4., 1), (4, False), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = {s: [{k: np.asarray(g).copy() for k, g in pair.items()} for pair in puzzle[s]] for s in puzzle}
    pair = broken['test'][0]
    info = _analyze(pair['input'])
    damaged = np.rot90(pair['output'], info['turns']).copy()
    r, c, n = info['fingers'][info['reference_index']]
    damaged[r, c] = info['background']
    pair['output'] = np.rot90(damaged, -info['turns']).copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged finger accepted')
    return dict(seeds=42, pairs=len(rows), shapes=len(shapes),
                canonical_structures=len(structures), palettes=len(palettes),
                counts=sorted(counts), references=sorted(references),
                weight_profiles=len(weight_profiles))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
