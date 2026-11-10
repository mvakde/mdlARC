"""Periodic dye signals: input-visible clocks, shape-qualified gates and splashes.

Only standard-library types and NumPy are used. No examples, files, seeds or
construction metadata are consulted by solve().
"""
from __future__ import annotations

import numpy as np

_ARMS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=True)


def _points(r, c):
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _analyze(value):
    g = _grid(value)
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(counts.max()) * 2 <= g.size:
        raise ValueError("background must be a strict majority")
    others = [(int(n), int(c)) for c, n in zip(colors, counts) if c != bg]
    if not others:
        raise ValueError("missing scaffold")
    others.sort(reverse=True)
    if len(others) > 1 and others[0][0] == others[1][0]:
        raise ValueError("scaffold plurality is tied")
    fg = others[0][1]
    h, w = g.shape
    adjacent = np.zeros(g.shape, dtype=bool)
    adjacent[1:] |= g[:-1] == fg
    adjacent[:-1] |= g[1:] == fg
    adjacent[:, 1:] |= g[:, :-1] == fg
    adjacent[:, :-1] |= g[:, 1:] == fg
    beads = {}
    for r, c in np.argwhere((g != bg) & (g != fg) & ~adjacent):
        beads.setdefault(int(g[r, c]), []).append((int(r), int(c)))
    if not beads or any(len(v) != 2 for v in beads.values()):
        raise ValueError("each reference hue needs exactly two scaffold-free beads")
    nodes = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if g[r, c] != bg and all(g[r + dr, c + dc] == fg for dr, dc in _ARMS):
                nodes[(r, c)] = int(g[r, c])
    lanes, axes = [], set()
    for color, positions in sorted(beads.items()):
        (r1, c1), (r2, c2) = positions
        if r1 == r2:
            axis, line, a, b, length = "H", r1, min(c1, c2), max(c1, c2), w
        elif c1 == c2:
            axis, line, a, b, length = "V", c1, min(r1, r2), max(r1, r2), h
        else:
            raise ValueError("reference beads must be aligned")
        axes.add(axis)
        p = b - a
        local = {c if axis == "H" else r: dye for (r, c), dye in nodes.items()
                 if (r if axis == "H" else c) == line}
        terminals = [x for x, dye in local.items() if dye not in (fg, color)
                     and (x - a) % p == 0 and (x < a or x > b)]
        if len(terminals) > 1:
            raise ValueError("multiple eligible gates on a lane")
        t = terminals[0] if terminals else None
        lo, hi = (0, length - 1) if t is None else (min(a, t), max(b, t))
        lanes.append(dict(axis=axis, line=line, a=a, b=b, p=p, length=length,
                          ref=color, t=t, dye=color if t is None else local[t],
                          lo=lo, hi=hi, nodes=local))
    if len(axes) != 1 or len({s['line'] for s in lanes}) != len(lanes):
        raise ValueError("one common axis and one pair per lane required")
    return g, bg, fg, nodes, lanes


def _paint(g, nodes, lanes, *, dense=False, splash=True, any_dye=False, bg=None, fg=None):
    requested = np.full(g.shape, -1, dtype=np.int16)
    for s in lanes:
        lo, hi, dye = s['lo'], s['hi'], s['dye']
        if any_dye:
            values = g[s['line']] if s['axis'] == "H" else g[:, s['line']]
            candidates = [x for x in range(s['length'])
                          if int(values[x]) not in (bg, fg, s['ref'])
                          and (x - s['a']) % s['p'] == 0
                          and (x < s['a'] or x > s['b'])]
            if candidates:
                t = min(candidates, key=lambda x: (min(abs(x-s['a']), abs(x-s['b'])), x))
                lo, hi, dye = min(s['a'], t), max(s['b'], t), int(values[t])
            else:
                lo, hi, dye = 0, s['length'] - 1, s['ref']
        for x in range(lo, hi + 1):
            if not dense and (x - s['a']) % s['p']:
                continue
            r, c = (s['line'], x) if s['axis'] == "H" else (x, s['line'])
            footprint = _points(r, c) if splash and (r, c) in nodes else {(r, c)}
            for point in footprint:
                if requested[point] not in (-1, dye):
                    raise ValueError("conflicting lane paint requests")
                requested[point] = dye
    out = g.copy()
    mask = requested >= 0
    out[mask] = requested[mask]
    return out, requested


def solve(input_grid: np.ndarray) -> np.ndarray:
    g, _, _, nodes, lanes = _analyze(input_grid)
    return _paint(g, nodes, lanes)[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _key(g):
    return (g.shape, g.tobytes())


def _plant(g, blocked, r, c, kind, center_color, bg, fg):
    """Atomic node insertion; protected missing cells stay genuinely missing."""
    arms = [(r + dr, c + dc) for dr, dc in _ARMS]
    if not all(0 <= y < g.shape[0] and 0 <= x < g.shape[1] for y, x in arms):
        return False
    filled = arms if kind != "T" else arms[1:]
    if kind != "ring":
        filled = filled + [(r, c)]
    missing = [(r, c)] if kind == "ring" else [arms[0]] if kind == "T" else []
    if any(g[p] != bg for p in missing):
        return False
    for point in filled:
        color = center_color if point == (r, c) else fg
        if blocked[point] or g[point] not in (bg, color):
            return False
    for point in filled:
        g[point] = center_color if point == (r, c) else fg
    for point in missing:
        blocked[point] = True
    return True


def _geometries(length, p, bounded, direction, core=False):
    result = []
    for a in range(length - p):
        b = a + p
        phase = list(range(a % p, length, p))
        if not bounded:
            if len(phase) < 3:
                continue
            if core and not (a >= p and b + p < length):
                continue
            result.append((a, b, None))
        else:
            for t in phase:
                if not 1 <= t < length - 1:
                    continue
                if (direction < 0 and t >= a) or (direction > 0 and t <= b):
                    continue
                distance = min(abs(t-a), abs(t-b))
                if distance < (9 if core else 3):
                    continue
                if not 0 <= t + direction * max(p, 2) < length:
                    continue
                result.append((a, b, t))
    return result


def _scene(rng, kind, colors, richness):
    bg, fg = int(colors[0]), int(colors[1])
    if kind == "boundary":
        count, length = int(rng.integers(1, 3)), int(rng.integers(28, 31))
        flags = [False] * count
    elif kind == "gated":
        count, length = int(rng.integers(2, 4)), int(rng.integers(26, 31))
        flags = [True] * count
    else:
        count = int(rng.integers(2, 6)) if kind == "test" else int(rng.integers(1, 7))
        length = int(rng.integers(16, 31))
        if kind == "test":
            flags = [bool(rng.integers(2)) for _ in range(count)]
            flags[0], flags[1] = False, True
            rng.shuffle(flags)
        else:
            # Demonstrate each behavior alone; tests compose them within a scene.
            flags = [bool(rng.integers(2))] * count
    minimum = max(5, 4 * (count - 1) + 3)
    height = int(rng.integers(minimum, 31))
    gaps = [4] * (count - 1)
    spare = height - 3 - sum(gaps)
    for i in rng.permutation(len(gaps)):
        extra = int(rng.integers(min(4, spare) + 1))
        gaps[int(i)] += extra
        spare -= extra
    start = int(rng.integers(1, height - 1 - sum(gaps)))
    rows = [start] + [start + int(x) for x in np.cumsum(gaps)]
    ref_colors = [int(x) for x in rng.choice(colors[2:8], count, replace=False)]
    g = np.full((height, length), bg, dtype=np.int8)
    blocked = np.zeros(g.shape, dtype=bool)
    plans = []
    for i, (r, bounded, ref) in enumerate(zip(rows, flags, ref_colors)):
        direction = (-1 if i % 2 == 0 else 1) if kind == "gated" else int(rng.choice((-1, 1)))
        core_boundary = kind == "boundary" and i == 0
        core_gate = kind == "gated" and i == 0
        p = 2 if core_boundary else 1 if core_gate else int(rng.integers(1, 8))
        feasible = _geometries(length, p, bounded, direction, core=core_boundary or core_gate)
        if not feasible:
            p = 1
            feasible = _geometries(length, p, bounded, direction)
        if core_boundary:
            feasible = [s for s in feasible if length // 3 <= s[0] <= 2 * length // 3]
        if not feasible:
            return None
        a, b, t = _pick(rng, feasible)
        for x in (a, b):
            g[r, x] = ref
            for y, z in _points(r, x):
                if 0 <= y < height and 0 <= z < length:
                    blocked[y, z] = True
        dyes = [int(c) for c in colors[2:] if c != ref]
        dye = int(_pick(rng, dyes))
        plans.append((r, a, b, p, t, ref, dye))
    for i, (r, a, b, p, t, ref, dye) in enumerate(plans):
        def places(phase=None, interval=None):
            xs = [x for x in range(1, length - 1) if min(abs(x-a), abs(x-b)) >= 3]
            if phase is not None:
                xs = [x for x in xs if ((x-a) % p == 0) == phase]
            if interval is not None:
                xs = [x for x in xs if interval[0] <= x <= interval[1]]
            return [int(x) for x in rng.permutation(xs)]

        def add(kind2, color, options):
            for x in options:
                if _plant(g, blocked, r, x, kind2, color, bg, fg):
                    return x
            return None

        if t is not None and not _plant(g, blocked, r, t, "plus", dye, bg, fg):
            return None
        if kind == "boundary" and i == 0:
            # Two cadence-selected crosses sharing their intermediate arm.
            choices = [x for x in places(True) if x + 2 < length - 1
                       and min(abs(x+2-a), abs(x+2-b)) >= 3]
            if not choices:
                return None
            x = _pick(rng, choices)
            if not (_plant(g, blocked, r, x, "plus", fg, bg, fg)
                    and _plant(g, blocked, r, x+2, "plus", fg, bg, fg)):
                return None
            # A diagonal attached to the selected union is not part of it.
            if g[r-1, x-1] == bg and not blocked[r-1, x-1]:
                g[r-1, x-1] = fg
            if add("ring", fg, places(True)) is None:
                return None
            if add("plus", dye, places(False)) is None:
                return None
        if kind == "gated" and i == 0:
            lo, hi = sorted((t, a if t < a else b))
            options = [x for x in places(True, (lo+3, hi-3)) if abs(x-t) >= 3]
            wrong_dye = int(_pick(rng, [int(c) for c in colors[2:] if c not in (ref, dye)]))
            if add("T", wrong_dye, options) is None:
                return None
        # Free structural variation, independent of the required diagnostic nodes.
        for _ in range(int(rng.integers(1, 7))):
            add("plus", fg, places(bool(rng.integers(2))))
        if p > 1 and rng.random() < .6:
            add("plus", int(_pick(rng, [int(c) for c in colors[2:] if c != ref])), places(False))
        if rng.random() < .5:
            add("T", dye, places())
        if rng.random() < .4:
            add("ring", fg, places())
    # Background gaps protected above also prevent clutter from repairing T/ring
    # witnesses or making scaffold-touching reference beads.
    budget = int(g.size * rng.uniform(0, (.07, .17, .28)[richness]))
    for _ in range(budget // 3):
        r, c = int(rng.integers(height)), int(rng.integers(length))
        mode = int(rng.integers(3))
        offsets = [(0, 0)] if mode == 0 else [(0, k) for k in range(int(rng.integers(2, 6)))] if mode == 1 else [(y, x) for y in range(2) for x in range(2)]
        for dr, dc in offsets:
            y, x = r + dr, c + dc
            if 0 <= y < height and 0 <= x < length and not blocked[y, x] and g[y, x] == bg:
                g[y, x] = fg
    return g


def _features(value):
    g, bg, fg, nodes, lanes = _analyze(value)
    out, requests = _paint(g, nodes, lanes)
    flags = dict(boundary=False, gate=False, bidirectional=False, stop=False,
                 phase_gate=False, incomplete=False, ring=False, overlap=False,
                 attached=False)
    for s in lanes:
        flags['boundary'] |= s['t'] is None
        flags['gate'] |= s['t'] is not None
        flags['bidirectional'] |= s['t'] is None and s['a'] >= s['p'] and s['b'] + s['p'] < s['length']
        if s['t'] is not None:
            direction = -1 if s['t'] < s['a'] else 1
            for x in range(s['length']):
                r, c = (s['line'], x) if s['axis'] == "H" else (x, s['line'])
                flags['stop'] |= ((x-s['a']) % s['p'] == 0
                                  and (x-s['t']) * direction > 0
                                  and requests[r, c] < 0 and g[r, c] == bg)
        cells, number = set(), 0
        for x in range(s['length']):
            r, c = (s['line'], x) if s['axis'] == "H" else (x, s['line'])
            on_phase = (x-s['a']) % s['p'] == 0
            active = on_phase and s['lo'] <= x <= s['hi']
            if (r, c) in nodes:
                if active:
                    cells.update(_points(r, c))
                    number += 1
                flags['phase_gate'] |= not on_phase and nodes[(r, c)] not in (fg, s['ref']) and (x < s['a'] or x > s['b'])
            elif active and 0 < r < g.shape[0]-1 and 0 < c < g.shape[1]-1:
                flags['ring'] |= g[r, c] == bg and all(g[r+dr, c+dc] == fg for dr, dc in _ARMS)
                if s['t'] is not None and g[r, c] not in (bg, fg, s['ref']):
                    flags['incomplete'] |= min(s['a'], s['t']) < x < max(s['b'], s['t']) and (x < s['a'] or x > s['b'])
        flags['overlap'] |= 5 * number > len(cells)
        for r, c in cells:
            for dr, dc in _ARMS:
                y, x = r + dr, c + dc
                if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
                    flags['attached'] |= g[y, x] == fg and requests[y, x] < 0
    flags['ablate_a'] = not np.array_equal(out, _paint(g, nodes, lanes, dense=True)[0])
    flags['ablate_b_spread'] = not np.array_equal(out, _paint(g, nodes, lanes, splash=False)[0])
    flags['ablate_b_gate'] = not np.array_equal(out, _paint(g, nodes, lanes, any_dye=True, bg=bg, fg=fg)[0])
    return flags, (g, bg, fg, nodes, lanes, out, requests)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .25)
    colors = rng.permutation(10)
    richness = int(rng.integers(3))
    first_axis = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            kind = ("boundary" if index == 0 else "gated" if index == 1 else "free") if split == "train" else "test"
            transpose = first_axis if kind == "boundary" else not first_axis if kind == "gated" else bool(rng.integers(2))
            for _ in range(240):
                palette = colors if stable else rng.permutation(10)
                g = _scene(rng, kind, palette, richness)
                if g is None:
                    continue
                if transpose:
                    g = g.T.copy()
                if rng.random() < .5:
                    g = g[::-1].copy()
                if rng.random() < .5:
                    g = g[:, ::-1].copy()
                try:
                    flags, data = _features(g)
                except ValueError:
                    continue
                required = ('bidirectional', 'phase_gate', 'ring', 'overlap', 'attached', 'ablate_a', 'ablate_b_spread') if kind == "boundary" else ('stop', 'incomplete', 'ablate_b_gate') if kind == "gated" else ()
                if not all(flags[k] for k in required) or _key(g) in seen:
                    continue
                out = data[5]
                if np.array_equal(g, out):
                    continue
                seen.add(_key(g))
                puzzle[split].append({"input": g, "output": out})
                break
            else:
                raise RuntimeError(f"bounded scene construction exhausted for {kind}")
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            flags, data = _features(g)
            expected, requests = data[5], data[6]
            assert np.array_equal(expected, out), "wrong cadence, gate, footprint, or preservation"
            assert np.array_equal(g[requests < 0], out[requests < 0])
            assert not np.array_equal(g, out)
            if split == 'train':
                assert not (flags['boundary'] and flags['gate']), "training scenes isolate the two completion modes"
                training.append((flags, data))
            else:
                assert flags['boundary'] and flags['gate'], "test must combine the demonstrated modes"
    for feature in ('boundary', 'gate', 'bidirectional', 'stop', 'phase_gate', 'incomplete', 'ring',
                    'overlap', 'attached', 'ablate_a', 'ablate_b_spread', 'ablate_b_gate'):
        assert any(flags[feature] for flags, _ in training), f"missing training witness: {feature}"
    assert {data[4][0]['axis'] for _, data in training} == {'H', 'V'}
    directions = {-1 if s['t'] < s['a'] else 1 for _, data in training for s in data[4] if s['t'] is not None}
    assert directions == {-1, 1}, "training must expose gates on both sides"


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                 for split in ('train', 'test') for p in puzzle[split])


def _hand_checks():
    # Expected labels are specified by coordinates, not by a production solver.
    g = np.zeros((5, 25), dtype=np.int8)
    g[2, [9, 11]] = 2
    for x in (3, 5):
        for point in _points(2, x):
            g[point] = 1
    g[1, 2] = 1
    for point in _points(2, 16):
        g[point] = 1
    g[2, 16] = 3
    for point in _points(2, 21) - {(2, 21)}:
        g[point] = 1
    expected = g.copy()
    expected[2, 1::2] = 2
    for x in (3, 5):
        for point in _points(2, x):
            expected[point] = 2
    assert np.array_equal(solve(g), expected)
    assert expected[1, 21] == 1 and expected[2, 21] == 2  # no new-plus activation
    assert expected[2, 16] == 3 and expected[1, 2] == 1
    b = np.zeros((5, 24), dtype=np.int8)
    b[2, [4, 5]] = 2
    for point in _points(2, 18):
        b[point] = 1
    b[2, 18] = 3
    for point in _points(2, 10) - {(1, 10)}:
        b[point] = 1
    b[2, 10] = 4
    expected_b = b.copy()
    expected_b[2, 4:19] = 3
    for point in _points(2, 18):
        expected_b[point] = 3
    assert np.array_equal(solve(b), expected_b)
    assert expected_b[3, 10] == 1 and expected_b[2, 20] == 0
    for inp, out in ((g, expected), (b, expected_b)):
        for turns in range(4):
            assert np.array_equal(solve(np.rot90(inp, turns)), np.rot90(out, turns))
            assert np.array_equal(solve(9-np.rot90(inp, turns)), 9-np.rot90(out, turns))
    # A clipped boundary center cannot be a complete five-cell plus.
    clipped = g.copy()
    clipped[0, 20:23] = 1
    clipped[0, 21] = 4
    clipped[1, 21] = 1
    assert (0, 21) not in _analyze(clipped)[3]


def self_test():
    _hand_checks()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    shapes, layouts, backgrounds, scaffolds, periods, counts, directions = set(), set(), set(), set(), set(), set(), set()
    edge, overlaps, rings = 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle['train'] + puzzle['test']:
            flags, data = _features(pair['input'])
            g, bg, fg, nodes, lanes = data[:5]
            shapes.add(g.shape)
            backgrounds.add(bg)
            scaffolds.add(fg)
            counts.add(len(lanes))
            periods.update(s['p'] for s in lanes)
            directions.update(-1 if s['t'] < s['a'] else 1 for s in lanes if s['t'] is not None)
            layouts.add((g.shape, tuple((s['axis'], s['line'], s['a'], s['b'], s['t'], len(s['nodes'])) for s in lanes)))
            edge += any(r in (1, g.shape[0]-2) or c in (1, g.shape[1]-2) for r, c in nodes)
            overlaps += flags['overlap']
            rings += flags['ring']
    assert len(shapes) >= 60 and len(layouts) >= 180
    assert {int(np.sign(h-w)) for h, w in shapes} == {-1, 0, 1}
    assert any(30 in shape for shape in shapes)
    assert backgrounds == scaffolds == set(range(10))
    assert set(range(1, 8)) <= periods and set(range(1, 7)) <= counts
    assert directions == {-1, 1} and min(edge, overlaps, rings) >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1) + 1):
            p = generate_puzzle(12345, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    g = damaged['test'][0]['output']
    g[0, 0] = (int(g[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged label")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, family invariants, counts and coordinate fixtures')
