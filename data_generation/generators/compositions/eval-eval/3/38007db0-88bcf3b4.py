"""Consensus loom: bitmap discrepancies become one labeled, anchored strand.

Only standard Python and NumPy are used. Generation and solving do not read
files, retain examples, or share hidden state. self_test streams its corpus.
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _present(a, pose):
    a = np.rot90(a, pose % 4)
    return (np.fliplr(a) if pose >= 4 else a).copy()


def _unpresent(a, pose):
    return np.rot90(np.fliplr(a) if pose >= 4 else a, -(pose % 4)).copy()


def _connected(cells):
    if not cells:
        return False
    visited = {min(cells)}
    todo = list(visited)
    while todo:
        r, c = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = (r + dr, c + dc)
                if p in cells and p not in visited:
                    visited.add(p)
                    todo.append(p)
    return visited == cells


def _analytic_route(obstacle, length):
    """Input-only clearance envelope, including the parallel incoming tangent."""
    _require(bool(obstacle) and min(u for u, v in obstacle) >= 1, "bad obstacle")
    peak = max(v for u, v in obstacle)
    tangent = max(v - u for u, v in obstacle)
    _require(peak == 0 or (peak > 0 and tangent == 0), "unsupported tangent")
    first = min(u for u, v in obstacle if v == peak)
    far = max(u for u, v in obstacle)
    route = [(u, 1 if peak == 0 and u <= first else
              max(v + 1 - abs(u - x) for x, v in obstacle))
             for u in range(max(length, far + 4))]
    slopes = [q[1] - p[1] for p, q in zip(route, route[1:])]
    _require(route[0] == (0, 1), "bad attachment")
    _require(all(s in (-1, 0, 1) for s in slopes) and
             slopes == sorted(slopes, reverse=True), "unsupported contour")
    _require(not (set(route) & obstacle), "route hits obstacle")
    return route[:length]


def _sweep_route(obstacle, length):
    """Construction via two clearance sweeps; not used by solve."""
    n = max(length, max(u for u, v in obstacle) + 3)
    profile = [-1000] * n
    for u, v in obstacle:
        profile[u] = max(profile[u], v + 1)
    for u in range(1, n):
        profile[u] = max(profile[u], profile[u - 1] - 1)
    for u in range(n - 2, -1, -1):
        profile[u] = max(profile[u], profile[u + 1] - 1)
    if max(v for u, v in obstacle) == 0:
        first = min(u for u, v in obstacle if v == 0)
        profile[:first + 1] = [1] * (first + 1)
    return list(enumerate(profile[:length]))


def _location(info, p):
    u, v = p
    return 1 + info['L'] + u, 1 + info['s'] + v


def _inside(info, p):
    return 0 < p[0] < info['H'] - 1 and 0 < p[1] < info['pw'] + 1


def _capacity(info):
    # With convex slopes and a fully visible obstacle, exits are permanent.
    route = _analytic_route(info['obstacle'], 64)
    visible = [_inside(info, _location(info, p)) for p in route]
    first = visible.index(False)
    _require(all(visible[:first]) and not any(visible[first:]), "route re-entry")
    return first


def _draw(info, lengths=None, straight=None, construction=False):
    lengths = info['lengths'] if lengths is None else lengths
    sequence = [color for color, n in zip(info['tags'], lengths) for _ in range(n)]
    out = np.full((info['H'], info['pw'] + 2), info['bg'], dtype=np.int8)
    out[[0, -1], :] = info['frame']
    out[:, [0, -1]] = info['frame']
    out[1:info['ph'] + 1, 1:info['pw'] + 1] = info['master']
    if straight:
        route = [(u, 1 if straight == 'parallel' else u + 1) for u in range(len(sequence))]
    else:
        route = (_sweep_route if construction else _analytic_route)(info['obstacle'], len(sequence))
    for point, color in zip(route, sequence):
        p = _location(info, point)
        if _inside(info, p):
            if not straight:
                _require(out[p] == info['bg'], "thread hits preserved material")
            out[p] = color
    return out


def _parse_canonical(g):
    H, W = g.shape
    frame = int(g[0, 0])
    _require(np.all(g[0] == frame) and np.all(g[-1] == frame) and
             np.all(g[:, -1] == frame), "not canonical frame")
    rs = list(map(int, np.flatnonzero(np.all(g == frame, axis=1))))
    cs = list(map(int, np.flatnonzero(np.all(g == frame, axis=0))))
    _require(len(rs) >= 2 and len(cs) >= 3 and rs[0] == 0, "missing lattice")
    ph, pw = rs[1] - 1, cs[0] - 1
    _require(6 <= ph <= 13 and 5 <= pw <= 8, "unsupported panel dimensions")
    _require(rs == list(range(0, H, ph + 1)) and rs[-1] == H - 1, "bad row spacing")
    _require(cs == list(range(pw + 1, W, pw + 1)) and cs[-1] == W - 1, "bad column spacing")
    bands, copies = len(rs) - 1, len(cs)
    _require(1 <= bands <= 4 and 3 <= copies <= 4, "bad panel counts")
    tag_rows = [1 + k * (ph + 1) for k in range(bands)]
    _require(list(map(int, np.flatnonzero(g[:, 0] != frame))) == tag_rows, "bad tags")
    tags = [int(g[r, 0]) for r in tag_rows]
    rows, masters, columns, variants = [], [], [], []
    for b in range(bands):
        r0 = 1 + b * (ph + 1)
        row = [g[r0:r0 + ph, 1 + c * (pw + 1):1 + c * (pw + 1) + pw].copy()
               for c in range(copies)]
        _require(all(np.all(p != frame) for p in row), "lattice inside panel")
        counts = Counter(p.tobytes() for p in row)
        winners = [k for k, v in counts.items() if v == copies - 1]
        _require(len(counts) == 2 and len(winners) == 1, "not one visible exception")
        master = next(p for p in row if p.tobytes() == winners[0])
        index = next(c for c, p in enumerate(row) if p.tobytes() != winners[0])
        masters.append(master)
        columns.append(index)
        variants.append(row[index])
        rows.append(row)
    master = masters[0]
    _require(all(np.array_equal(p, master) for p in masters), "references disagree")
    bg = int(master[0, 0])
    _require(np.all(master[[0, -1], :] == bg) and np.all(master[:, [0, -1]] == bg), "bad perimeter")
    roles = set(map(int, np.unique(master)))
    _require(len(roles) == 3 and frame not in roles and not (set(tags) & (roles | {frame})), "bad role palette")
    support_choices = []
    for color in roles - {bg}:
        cells = {tuple(map(int, p)) for p in np.argwhere(master == color)}
        if (2 <= len(cells) <= 4 and len({c for r, c in cells}) == 1
                and {r for r, c in cells} == set(range(1, len(cells) + 1))):
            support_choices.append((color, len(cells), next(iter(cells))[1]))
    _require(len(support_choices) == 1, "ambiguous support")
    support, L, s = support_choices[0]
    obstacle_color = next(iter(roles - {bg, support}))
    obstacle_global = {tuple(map(int, p)) for p in np.argwhere(master == obstacle_color)}
    _require(_connected(obstacle_global), "disconnected reference obstacle")
    obstacle = {(r - L, c - s) for r, c in obstacle_global}
    _require(1 <= s <= pw - 3 and master[L, s + 1] == bg, "bad attachment position")
    _analytic_route(obstacle, 1)
    additions, removals, lengths, difference_masks = [], [], [], []
    for variant in variants:
        _require(set(map(int, np.unique(variant))) <= roles, "extra interior color")
        _require(np.all(variant[[0, -1], :] == bg) and np.all(variant[:, [0, -1]] == bg), "changed perimeter")
        _require(np.array_equal(variant == support, master == support) and
                 variant[L, s + 1] == bg, "changed support/attachment")
        changed = master != variant
        n = int(changed.sum())
        _require(1 <= n <= 8, "bad discrepancy count")
        _require(all({int(master[r, c]), int(variant[r, c])} == {bg, obstacle_color}
                     for r, c in np.argwhere(changed)), "not a two-role toggle")
        additions.append(int(np.count_nonzero((master == bg) & (variant == obstacle_color))))
        removals.append(int(np.count_nonzero((master == obstacle_color) & (variant == bg))))
        lengths.append(n)
        difference_masks.append(changed)
    info = dict(H=H, W=W, ph=ph, pw=pw, bands=bands, copies=copies, frame=frame,
                bg=bg, support=support, obstacle_color=obstacle_color, L=L, s=s,
                master=master, obstacle=obstacle, tags=tags, lengths=lengths,
                columns=columns, additions=additions, removals=removals,
                variants=variants, difference_masks=difference_masks)
    cap = _capacity(info)
    _require(sum(lengths[:-1]) < cap, "a segment is wholly clipped")
    info['capacity'] = cap
    return info


def _parse(input_grid):
    g = _grid(input_grid)
    candidates = []
    for pose in range(8):
        try:
            info = _parse_canonical(_unpresent(g, pose))
        except ValueError:
            continue
        info['pose'] = pose
        candidates.append(info)
    _require(len(candidates) == 1, "input lacks a unique tagged orientation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from this input, with no seed or examples."""
    info = _parse(input_grid)
    return _present(_draw(info), info['pose'])


def _raw_obstacle(rng, height, width):
    kind = int(rng.integers(6))
    if kind == 0:
        raw = {(0, 0)}
    elif kind == 1:
        if rng.random() < .5:
            raw = {(u, 0) for u in range(int(rng.integers(1, height + 1)))}
        else:
            raw = {(0, v) for v in range(int(rng.integers(1, width + 1)))}
    elif kind == 2:
        h, w = int(rng.integers(1, height + 1)), int(rng.integers(1, width + 1))
        raw = {(u, v) for u in range(h) for v in range(w)}
    elif kind == 3 and height >= 3:
        radius = int(rng.integers(1, min(2, (height - 1) // 2) + 1))
        thin = rng.random() < .5
        raw = {(u, v) for u in range(2 * radius + 1)
               for v in (range(radius - abs(u - radius) + 1) if not thin else
                         [radius - abs(u - radius)])}
    else:
        plateau = int(rng.integers(1, min(3, height) + 1))
        ramp = int(rng.integers(0, min(3, height - plateau) + 1))
        thick = rng.random() < .5
        raw = {(u, v) for u in range(plateau + ramp)
               for v in (range(ramp - max(0, u - plateau + 1) + 1) if thick else
                         [ramp - max(0, u - plateau + 1)])}
    if rng.random() < .5:
        end = max(u for u, v in raw)
        raw = {(end - u, v) for u, v in raw}
    return raw


def _scene(rng, ph, pw, approach, colors, fallback=False):
    frame, bg, support, obstacle_color = colors[:4]
    if fallback:
        L, s = 2, (min(3, pw - 3) if approach == 'parallel' else 1)
        obstacle = {(2, 0 if approach == 'parallel' else 2)}
    else:
        L = int(rng.integers(2, min(4, ph - 3) + 1))
        extent = ph - 2 - L
        raw = _raw_obstacle(rng, min(5, extent), min(4, pw - 2))
        height = max(u for u, v in raw) + 1
        near = int(rng.integers(1, extent - height + 2))
        offset = -max(v for u, v in raw) if approach == 'parallel' else near - max(v - u for u, v in raw)
        obstacle = {(u + near, v + offset) for u, v in raw}
        _analytic_route(obstacle, 1)
        columns = [c for c in range(1, pw - 2)
                   if all(1 <= c + v <= pw - 2 for u, v in obstacle)]
        _require(bool(columns), "obstacle too wide")
        s = int(rng.choice(columns))
    master = np.full((ph, pw), bg, dtype=np.int8)
    master[1:L + 1, s] = support
    for u, v in obstacle:
        master[L + u, s + v] = obstacle_color
    return dict(ph=ph, pw=pw, frame=frame, bg=bg, support=support,
                obstacle_color=obstacle_color, L=L, s=s, obstacle=obstacle, master=master)


def _editable(info):
    m, bg, oc = info['master'], info['bg'], info['obstacle_color']
    ink = [tuple(map(int, p)) for p in np.argwhere(m == oc)]
    blank = [(r, c) for r in range(1, info['ph'] - 1) for c in range(1, info['pw'] - 1)
             if m[r, c] == bg and (r, c) != (info['L'], info['s'] + 1)]
    return ink, blank


def _allocation(rng, info, lesson):
    ink, blank = _editable(info)
    b, cap = info['bands'], info['capacity']
    far = max(u for u, v in info['obstacle'])
    if lesson:
        options = []
        if lesson == 'visible':
            for a in range(2, min(8, 2 * len(ink), 2 * len(blank)) + 1, 2):
                for z in range(1, min(8, len(blank)) + 1):
                    if far + 3 <= a + z <= cap:
                        # If the balanced count equals obstacle area and the
                        # final added segment ends at the frame, an area-count
                        # shortcut can hide all its extra material off-grid.
                        # Keep that order only when its error is visible.
                        if a != len(ink) or a + z < cap:
                            options.append(([a, z], ['exchange', 'add']))
                        options.append(([z, a], ['add', 'exchange']))
        else:
            for a in range(1, min(8, len(ink)) + 1):
                for z in range(1, min(8, len(blank)) + 1):
                    if cap < a + z <= cap + 3:
                        if a < cap:
                            options.append(([a, z], ['remove', 'add']))
                        if z < cap:
                            options.append(([z, a], ['add', 'remove']))
        _require(bool(options), "lesson length cannot fit this obstacle")
        return options[int(rng.integers(len(options)))]
    for _ in range(48):
        total = int(rng.integers(b, min(cap + 2, 8 * b, 18) + 1))
        cuts = sorted(map(int, rng.choice(np.arange(1, total), b - 1, replace=False))) if b > 1 else []
        lengths = [q - p for p, q in zip([0] + cuts, cuts + [total])]
        if max(lengths) > 8 or sum(lengths[:-1]) >= cap:
            continue
        modes = []
        for n in lengths:
            choices = ['scatter']
            if n <= len(blank):
                choices.append('add')
            if n <= len(ink):
                choices.append('remove')
            if n % 2 == 0 and n // 2 <= min(len(ink), len(blank)):
                choices.append('exchange')
            modes.append(str(rng.choice(choices)))
        return lengths, modes
    # The guaranteed fallback template has capacity >=6 for at most four bands.
    _require(b <= cap and len(blank) >= 1, "cannot expose every segment")
    return [1] * b, ['add'] * b


def _take(rng, cells, count):
    _require(0 <= count <= len(cells), "too many edits")
    if not count:
        return []
    if rng.random() < .5:
        indices = rng.choice(len(cells), count, replace=False)
        return [cells[int(i)] for i in indices]
    # Clustered and scattered discrepancies coexist; both still encode count.
    anchor = cells[int(rng.integers(len(cells)))]
    tie_break = rng.random(len(cells))
    order = sorted(range(len(cells)), key=lambda j:
                   (abs(cells[j][0] - anchor[0]) + abs(cells[j][1] - anchor[1]), tie_break[j]))
    return [cells[j] for j in order[:count]]


def _variant(rng, info, n, mode):
    ink, blank = _editable(info)
    if mode == 'add':
        edits = _take(rng, blank, n)
    elif mode == 'remove':
        edits = _take(rng, ink, n)
    elif mode == 'exchange':
        edits = _take(rng, ink, n // 2) + _take(rng, blank, n // 2)
    else:
        edits = _take(rng, ink + blank, n)
    out = info['master'].copy()
    for p in edits:
        out[p] = info['obstacle_color'] if out[p] == info['bg'] else info['bg']
    return out


def _make_pair(rng, geometry, pose, approach, lesson, colors):
    ph, pw, bands, copies = geometry
    for attempt in range(65):
        try:
            info = _scene(rng, ph, pw, approach, colors, fallback=attempt == 64)
            info.update(H=bands * (ph + 1) + 1, W=copies * (pw + 1) + 1,
                        bands=bands, copies=copies)
            info['capacity'] = _capacity(info)
            lengths, modes = _allocation(rng, info, lesson)
        except ValueError:
            if attempt == 64:
                raise
            continue
        break
    tags = list(map(int, rng.choice(colors[4:], bands, replace=False)))
    if not lesson and bands > 1 and rng.random() < .3:
        tags[int(rng.integers(bands))] = tags[int(rng.integers(bands))]
    info.update(tags=tags, lengths=lengths)
    columns = list(map(int, rng.integers(copies, size=bands)))
    if lesson == 'visible' and len(set(columns)) == 1:
        columns[-1] = (columns[0] + int(rng.integers(1, copies))) % copies
    inp = np.full((info['H'], info['W']), info['frame'], dtype=np.int8)
    for b in range(bands):
        r0 = 1 + b * (ph + 1)
        inp[r0, 0] = tags[b]
        variant = _variant(rng, info, lengths[b], modes[b])
        for c in range(copies):
            c0 = 1 + c * (pw + 1)
            inp[r0:r0 + ph, c0:c0 + pw] = variant if c == columns[b] else info['master']
    out = _draw(info, construction=True)
    return {'input': _present(inp, pose), 'output': _present(out, pose)}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' is outside its legal range')
    return int(value)


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    used = sorted(map(int, np.unique(pair['input'])))
    # Each distinct permutation gives a distinct input. At most 13 are excluded.
    for replacement in permutations(range(10), len(used)):
        lut = np.arange(10, dtype=np.int8)
        lut[used] = replacement
        candidate = {k: lut[v] for k, v in pair.items()}
        if _key(candidate['input']) not in seen:
            return candidate
    raise AssertionError('palette exhaustion is impossible')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, nq = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))  # band-count, repetition-count, orientation
    base_pose = int(rng.integers(8))
    visible_slot, clipped_slot = map(int, rng.choice(nt, 2, replace=False))
    visible_approach = str(rng.choice(['parallel', 'diagonal']))
    palette = list(map(int, rng.choice(10, 8, replace=False)))
    shared = rng.random() < .2
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            lesson = ('visible' if index == visible_slot else 'clipped' if index == clipped_slot else None) if split == 'train' else None
            if lesson:
                bands = 2
            elif regime == 0:
                bands = int(rng.choice([1, 2] if split == 'train' else [3, 4]))
            else:
                bands = int(rng.choice([1, 2, 3, 4], p=[.15, .40, .30, .15]))
            copies = (3 if split == 'train' else 4) if regime == 1 else int(rng.choice([3, 4], p=[.75, .25]))
            ph_max, pw_max = min(13, 29 // bands - 1), min(8, 29 // copies - 1)
            ph = int(rng.integers(7 if lesson else 6, ph_max + 1))
            pw = int(rng.integers(5, pw_max + 1))
            pose = (base_pose if split == 'train' else 4 * (base_pose // 4) + (base_pose + 1) % 4) if regime == 2 else int(rng.integers(8))
            approach = (visible_approach if lesson == 'visible' else
                        ('diagonal' if visible_approach == 'parallel' else 'parallel') if lesson == 'clipped' else
                        str(rng.choice(['parallel', 'diagonal'])))
            colors = palette if shared else list(map(int, rng.choice(10, 8, replace=False)))
            pair = _unique(_make_pair(rng, (ph, pw, bands, copies), pose, approach, lesson, colors), seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation_lengths(info, mode):
    if mode == 'net':
        return [abs(a - r) for a, r in zip(info['additions'], info['removals'])]
    if mode == 'add':
        return info['additions']
    if mode == 'remove':
        return info['removals']
    return [int(np.count_nonzero(v == info['obstacle_color'])) for v in info['variants']]


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'bad split counts')
    seen, training, split_infos = set(), [], {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _parse(inp)
            expected = _present(_draw(info), info['pose'])
            _require(np.array_equal(expected, out), 'incorrect output')
            canonical = _unpresent(out, info['pose'])
            m = info['master']
            for color in (info['support'], info['obstacle_color']):
                mask = m == color
                _require(np.all(canonical[1:info['ph'] + 1, 1:info['pw'] + 1][mask] == color), 'damaged preserved material')
            _require(np.array_equal(solve(inp.copy()), out), 'input-only solve mismatch')
            _require(np.all(canonical[[0, -1], :] == info['frame']) and
                     np.all(canonical[:, [0, -1]] == info['frame']), 'damaged frame')
            split_infos[split].append(info)
            if split == 'train':
                training.append((info, canonical))
    infos = [i for i, out in training]
    _require({max(v for u, v in i['obstacle']) == 0 for i in infos} == {True, False}, 'teach both tangent types')
    _require(any(sum(i['lengths']) > i['capacity'] for i in infos), 'teach clipping')
    _require(any(sum(i['lengths']) <= i['capacity'] and
                 sum(i['lengths']) >= max(u for u, v in i['obstacle']) + 3 and
                 len(set(i['tags'])) >= 2 and len(set(i['columns'])) >= 2 and
                 any(a == r > 0 for a, r in zip(i['additions'], i['removals']))
                 for i in infos), 'teach balanced counting, order and wrapping')
    changes = [(a, r) for i in infos for a, r in zip(i['additions'], i['removals'])]
    _require(any(a > 0 and r == 0 for a, r in changes) and
             any(r > 0 and a == 0 for a, r in changes), 'teach both edit signs')
    for mode in ('net', 'add', 'remove', 'area'):
        _require(any(not np.array_equal(_draw(i, _ablation_lengths(i, mode)), out)
                     for i, out in training), 'A ablation survived: ' + mode)
    for straight in ('parallel', 'diagonal'):
        _require(any(not np.array_equal(_draw(i, straight=straight), out) for i, out in training),
                 'B ablation survived: ' + straight)
    train, tests = split_infos['train'], split_infos['test']
    band_shift = max(i['bands'] for i in train) <= 2 and min(i['bands'] for i in tests) >= 3
    copy_shift = {i['copies'] for i in train} == {3} and {i['copies'] for i in tests} == {4}
    pose_shift = len({i['pose'] for i in train}) == 1 and all(
        (i['pose'] % 4 - train[0]['pose'] % 4) % 4 in (1, 3) and
        i['pose'] // 4 == train[0]['pose'] // 4 for i in tests)
    _require(band_shift or copy_shift or pose_shift, 'missing documented test shift')


def _metrics(info):
    obs = info['obstacle']
    u0, v0 = min(u for u, v in obs), min(v for u, v in obs)
    silhouette = tuple(sorted((u - u0, v - v0) for u, v in obs))
    normalized = info['master'] == info['obstacle_color']
    return dict(geometry=(info['ph'], info['pw'], info['bands'], info['copies']),
                pose=info['pose'], palette=(info['frame'], info['bg'], info['support'], info['obstacle_color']),
                lengths=tuple(info['lengths']), silhouette=silhouette,
                motif=(normalized.shape, normalized.tobytes()),
                clipped=sum(info['lengths']) > info['capacity'],
                balanced=sum(a == r > 0 for a, r in zip(info['additions'], info['removals'])),
                repeated_column=len(set(info['columns'])) < info['bands'],
                rectangular=len(obs) == (max(u for u, v in obs) - u0 + 1) * (max(v for u, v in obs) - v0 + 1))


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
    # Source-inspired contour regressions; the new family changes the encoding.
    cases = [({(2, 0)}, 6, [(0, 1), (1, 1), (2, 1), (3, 0), (4, -1), (5, -2)]),
             ({(u, v) for u in (2, 3) for v in (0, 1, 2)}, 6,
              [(0, 1), (1, 2), (2, 3), (3, 3), (4, 2), (5, 1)]),
             ({(1, 1), (2, 2), (3, 1)}, 6,
              [(0, 1), (1, 2), (2, 3), (3, 2), (4, 1), (5, 0)])]
    for obs, n, expected in cases:
        assert _analytic_route(obs, n) == expected == _sweep_route(obs, n)
    cap = {(2, 2), (3, 1), (4, 0)}
    box = {(u, v) for u in range(2, 5) for v in range(3)}
    assert _analytic_route(cap, 6) != _analytic_route(box, 6)
    # Exercise the bounded fallback, including the tightest legal panels.
    fallback_rng = np.random.default_rng(8831)
    for ph, pw, bands in ((6, 5, 1), (6, 5, 4), (7, 5, 2), (13, 8, 2)):
        for approach in ('parallel', 'diagonal'):
            for lesson in ((None, 'visible', 'clipped') if bands == 2 else (None,)):
                i = _scene(fallback_rng, ph, pw, approach, list(range(8)), fallback=True)
                i.update(H=bands * (ph + 1) + 1, bands=bands)
                i['capacity'] = _capacity(i)
                ns, modes = _allocation(fallback_rng, i, lesson)
                i.update(lengths=ns, tags=list(range(4, 4 + bands)))
                assert np.array_equal(_draw(i), _draw(i, construction=True))
                for n, mode in zip(ns, modes):
                    v = _variant(fallback_rng, i, n, mode)
                    assert np.count_nonzero(v != i['master']) == n
    signatures, geometries, poses, palettes, silhouettes, motifs = (set() for _ in range(6))
    lengths, backgrounds, frames, counts = set(), set(), set(), Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_fingerprint(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                i = _parse(pair['input'])
                m = _metrics(i)
                geometries.add(m['geometry'])
                poses.add(m['pose'])
                palettes.add(m['palette'])
                silhouettes.add(m['silhouette'])
                motifs.add(m['motif'])
                lengths.update(m['lengths'])
                backgrounds.add(i['bg'])
                frames.add(i['frame'])
                counts['pairs'] += 1
                counts['clipped'] += m['clipped']
                counts['irregular'] += not m['rectangular']
                counts['balanced'] += m['balanced']
                counts['repeated_column'] += m['repeated_column']
                counts['one_band'] += i['bands'] == 1
                counts['four_bands'] += i['bands'] == 4
                # All presentations of each valid input must have transformed answers.
                if seed < 2:
                    for pose in range(8):
                        assert np.array_equal(solve(_present(pair['input'], pose)), _present(pair['output'], pose))
    assert len(signatures) == 42 and len(geometries) >= 35 and len(motifs) >= 65
    assert poses == set(range(8)) and len(palettes) >= 60
    assert len(silhouettes) >= 15 and backgrounds == frames == set(range(10))
    assert lengths >= set(range(1, 9))
    assert all(counts[k] >= 8 for k in ('clipped', 'irregular', 'balanced', 'repeated_column', 'one_band', 'four_bands'))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(900 + 10 * nt + nq, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    # Regression: an area-count error must remain visible despite clipping.
    generate_puzzle(220, 2, 1)
    assert _fingerprint(generate_puzzle(123, 7, 3)) == _fingerprint(generate_puzzle(123, 7, 3))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Corrupt returned material and verify detection; never modify evidence.
    for defect in ('frame', 'support', 'missing_exception'):
        p = generate_puzzle(314, 2, 1)
        pair = p['train'][0]
        info = _parse(pair['input'])
        if defect == 'missing_exception':
            g = _unpresent(pair['input'], info['pose'])
            c0 = 1 + info['columns'][0] * (info['pw'] + 1)
            g[1:info['ph'] + 1, c0:c0 + info['pw']] = info['master']
            pair['input'] = _present(g, info['pose'])
        else:
            g = _unpresent(pair['output'], info['pose'])
            point = (0, 0) if defect == 'frame' else (2, 1 + info['s'])
            g[point] = info['bg']
            pair['output'] = _present(g, info['pose'])
        try:
            validate_generated_puzzle(p)
        except (ValueError, AssertionError):
            continue
        raise AssertionError('corruption accepted: ' + defect)
    return {'seeds': 42, 'pairs': counts['pairs'], 'geometries': len(geometries),
            'reference_motifs': len(motifs), 'silhouettes': len(silhouettes),
            'poses': len(poses), 'role_palettes': len(palettes),
            'edit_lengths': sorted(lengths), **dict(counts)}
