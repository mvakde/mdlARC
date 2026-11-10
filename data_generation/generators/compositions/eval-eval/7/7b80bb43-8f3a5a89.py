"""Restore only the external shell of the marked room.

The two-cell diagonal deviations describe virtual three-cell wall repairs.
Room topology decides which repairs are actually materialized.  This module
uses no parent imports, files, cached examples, seed lookup, or external state.
"""

from collections import deque
import hashlib
import json

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 116878180, 426596766, 429492675, 972871035, 234377225,
              344218646, 47465808, 509172255, 195853203, 938258847,
              201053060, 572225209, 419797688, 740997493, 287196846,
              926242165, 902398716, 721540333, 331184544, 365154658,
              223497811, 986982693, 165599191, 105561484, 787866549,
              745352330, 310950133, 130962401, 626664875, 227350684,
              584593082, 765553346, 911183561, 406756241, 860162031,
              816548728, 971986158, 643215934, 924984184, 733363683,
              321913919)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g


def _roles(g):
    colors, counts = np.unique(g, return_counts=True)
    if len(colors) != 3 or np.count_nonzero(counts == 1) != 1:
        raise ValueError("input must have background, walls and one unique marker")
    bg = int(colors[np.argmax(counts)])
    marker = int(colors[counts == 1][0])
    wall = int(next(c for c in colors if c not in (bg, marker)))
    if np.count_nonzero(g == bg) <= np.count_nonzero(g == wall):
        raise ValueError("background must be strictly dominant")
    return bg, wall, marker


def _near(mask, diagonal=False):
    h, w = mask.shape
    p = np.pad(mask, 1)
    out = np.zeros_like(mask, dtype=bool)
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if r or c] if diagonal else N4
    for dr, dc in steps:
        out |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _flood(allowed, start):
    reached = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    queue = deque([start])
    h, w = allowed.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in N4:
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w and allowed[rr, cc]
                    and not reached[rr, cc]):
                reached[rr, cc] = True
                queue.append((rr, cc))
    return reached


def _components(mask):
    remaining = mask.copy()
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if remaining[r, c]:
            part = _flood(remaining, (r, c))
            remaining[part] = False
            result.append(part)
    return result


def _at(m, r, c):
    return 0 <= r < m.shape[0] and 0 <= c < m.shape[1] and bool(m[r, c])


def _gaps(mask):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        if m.shape[1] < 5:
            continue
        brackets = m[:, :-4] & m[:, 4:]
        brackets &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(brackets):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _recover(raw):
    """Input-only geometric inference; all matches are simultaneous.

    A diagonal-only cell has no cardinal wall neighbor in the original input.
    This distinction keeps orthogonal stubs and island material intact.
    """
    loose = raw & ~_near(raw)
    added = np.zeros_like(raw)
    anchors = np.zeros_like(raw)
    repairs = []
    for r, c, dr, dc in _gaps(raw):
        if not (_at(raw, r - dr, c - dc)
                or _at(raw, r + 5 * dr, c + 5 * dc)):
            continue
        matches = []
        for end in (0, 4):
            ar, ac = r + end * dr, c + end * dc
            direction = 1 if end == 0 else -1
            for bend in (-1, 1):
                sr, sc = direction * dr + bend * dc, direction * dc + bend * dr
                spur = tuple((ar + k * sr, ac + k * sc) for k in (1, 2))
                if (all(_at(loose, rr, cc) for rr, cc in spur)
                        and not _at(raw, ar + 3 * sr, ac + 3 * sc)):
                    matches.append(spur)
        if len(matches) > 1:
            raise ValueError("ambiguous diagonal evidence at a gap")
        if matches:
            gap = tuple((r + k * dr, c + k * dc) for k in (1, 2, 3))
            for rr, cc in gap:
                added[rr, cc] = True
            anchors[r, c] = anchors[r + 4 * dr, c + 4 * dc] = True
            repairs.append((gap, matches[0], 'h' if dc else 'v'))
    # Remove only diagonal-linked loose cells, protecting recognized brackets.
    weak = loose & ~anchors
    diagonal = np.zeros_like(raw)
    links = weak[:-1, :-1] & weak[1:, 1:]
    diagonal[:-1, :-1] |= links
    diagonal[1:, 1:] |= links
    links = weak[:-1, 1:] & weak[1:, :-1]
    diagonal[:-1, 1:] |= links
    diagonal[1:, :-1] |= links
    stable = raw & ~diagonal
    return stable, stable | added, repairs


def _topology(virtual, marker_pos):
    selected = _flood(~virtual, marker_pos)
    outside = _flood(np.pad(~selected, 1, constant_values=True), (0, 0))
    exterior = outside[1:-1, 1:-1]
    shell = virtual & exterior & _near(selected, True)
    islands = virtual & ~exterior
    rim = selected & _near(outside, True)[1:-1, 1:-1]
    return selected, exterior, shell, islands, rim


def _render(stable, virtual, pos, palette):
    bg, wall, marker = palette
    selected, exterior, shell, islands, rim = _topology(virtual, pos)
    out = np.full(virtual.shape, bg, np.int8)
    out[shell | (stable & islands)] = wall
    out[rim] = marker
    out[pos] = marker
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every output cell from the supplied input, without demonstrations."""
    g = _grid(input_grid)
    palette = _roles(g)
    pos = tuple(map(int, np.argwhere(g == palette[2])[0]))
    stable, virtual, _ = _recover(g == palette[1])
    if virtual[pos]:
        raise ValueError("marker cannot occupy a wall or inferred repair")
    return _render(stable, virtual, pos, palette)


def _count(name, value, lo, hi, default):
    # None is accepted as the documented default for convenience, never as a
    # random or seed-dependent convention. Explicit integer counts are exact.
    if value is None:
        value = default
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _damage_options(clean):
    """Enumerate local, separated corruption footprints, not whole puzzles."""
    options = []
    for dr, dc in ((0, 1), (1, 0)):
        for r, c in np.argwhere(clean):
            r, c = int(r), int(c)
            if not all(_at(clean, r + k * dr, c + k * dc) for k in range(5)):
                continue
            if not (_at(clean, r - dr, c - dc)
                    or _at(clean, r + 5 * dr, c + 5 * dc)):
                continue
            gap = tuple((r + k * dr, c + k * dc) for k in (1, 2, 3))
            # Repairs are on thin straight sections, never junctions or masses.
            if any(_at(clean, rr + dc, cc + dr)
                   or _at(clean, rr - dc, cc - dr) for rr, cc in gap):
                continue
            for end in (0, 4):
                ar, ac = r + end * dr, c + end * dc
                direction = 1 if end == 0 else -1
                for side in (-1, 1):
                    sr, sc = direction * dr + side * dc, direction * dc + side * dr
                    spur = tuple((ar + k * sr, ac + k * sc) for k in (1, 2))
                    if not all(0 <= rr < clean.shape[0] and 0 <= cc < clean.shape[1]
                               and not clean[rr, cc] for rr, cc in spur):
                        continue
                    trial = clean.copy()
                    for rr, cc in gap:
                        trial[rr, cc] = False
                    for rr, cc in spur:
                        trial[rr, cc] = True
                    # These two off-axis cells must remain diagonal-only.
                    if any(_at(trial, rr+nr, cc+nc) for rr, cc in spur for nr, nc in N4):
                        continue
                    options.append((gap, spur))
    return options


def _put_motif(rng, clean, region, kind):
    """Place separated relics, with two free cells around their whole box."""
    if kind == 'ring':
        h, w = int(rng.integers(7, 10)), int(rng.integers(7, 10))
        motif = np.zeros((h, w), bool)
        if rng.random() < .45:
            # A six-corner loop, with a three-cell step cut out of one corner.
            # Both the inside and outside are still orthogonally connected.
            vertices = [(0, 0), (0, w-4), (3, w-4), (3, w-1),
                        (h-1, w-1), (h-1, 0), (0, 0)]
            for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
                motif[min(r0, r1):max(r0, r1)+1, min(c0, c1):max(c0, c1)+1] = True
            if rng.integers(2):
                motif = motif[::-1]
            if rng.integers(2):
                motif = motif[:, ::-1]
        else:
            motif[[0, -1], :] = True
            motif[:, [0, -1]] = True
    elif kind == 'dashed':
        motif = np.ones((1, int(rng.integers(9, 12))), bool)
        motif[0, 3:6] = False
    else:
        h, w = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        motif = np.ones((h, w), bool)
        if kind == 'L':
            motif[1:, 1:] = False
    h, w = motif.shape
    locations = []
    for r in range(2, clean.shape[0] - h - 1):
        for c in range(2, clean.shape[1] - w - 1):
            if region[r-2:r+h+2, c-2:c+w+2].all():
                locations.append((r, c))
    if not locations:
        return None
    r, c = locations[int(rng.integers(len(locations)))]
    clean[r:r+h, c:c+w] |= motif
    footprint = np.zeros_like(clean)
    footprint[r:r+h, c:c+w] = motif
    return footprint


def _split_region(rng, clean, region):
    rr, cc = np.where(region)
    if not len(rr):
        return False
    choices = []
    for axis, lo, hi in ((0, int(rr.min()), int(rr.max())),
                         (1, int(cc.min()), int(cc.max()))):
        for cut in range(lo + 3, hi - 2):
            # Area is a cheap necessary test. Connectivity is checked only for
            # proposals actually tried, rather than for every possible cut.
            before = region[:cut] if axis == 0 else region[:, :cut]
            after = region[cut+1:] if axis == 0 else region[:, cut+1:]
            if min(int(before.sum()), int(after.sum())) >= 15:
                choices.append((axis, cut))
    rng.shuffle(choices)
    for axis, cut in choices:
        line = np.zeros_like(clean)
        if axis == 0:
            line[cut] = region[cut]
        else:
            line[:, cut] = region[:, cut]
        parts = _components(region & ~line)
        if len(parts) == 2 and min(int(p.sum()) for p in parts) >= 15:
            clean |= line
            return True
    return False


def _base_scene(rng, witness, test):
    h = int(rng.integers(22 if test else 20, 31))
    w = int(rng.integers(24, 31))
    clean = np.zeros((h, w), bool)
    # A varying profile creates real bent boundaries, rather than padding.
    center = int(rng.integers(14, w - 5))
    profile = np.full(h, center, int)
    if rng.random() < .65:
        cuts = sorted(set([0, h] + [int(x) for x in rng.integers(5, h-4, size=2)]))
        for lo, hi in zip(cuts, cuts[1:]):
            profile[lo:hi] = int(np.clip(center + rng.integers(-2, 3), 12, w-6))
    for r, c in enumerate(profile):
        clean[r, c] = True
        if r:
            clean[r, min(c, profile[r-1]):max(c, profile[r-1])+1] = True
    # A dead-end wall into the large side supplies concave/corner rim evidence.
    stub_row = int(rng.integers(3, h-3))
    stub_col = int(profile[stub_row])
    clean[stub_row, stub_col-int(rng.integers(2, 5)):stub_col] = True
    large = _flood(~clean, (0, 0))
    ring = _put_motif(rng, clean, large, 'ring') if witness or rng.random() < .75 else None
    if ring is None and witness:
        return None
    # The unsupported break is not a displaced fragment and remains a break.
    large = _flood(~clean, (0, 0))
    dashed = _put_motif(rng, clean, large, 'dashed')
    if dashed is None and witness == 1:
        return None
    # Branching partitions on the other side are connected to the main shell.
    desired = int(rng.integers(3, 5)) if test else int(rng.integers(1, 3))
    for _ in range(desired):
        right = [p for p in _components(~clean) if p[0, -1] or p[-1, -1]
                 or p[:, -1].any()]
        right.sort(key=lambda p: -int(p.sum()))
        if not any(_split_region(rng, clean, part) for part in right):
            break
    if not witness:
        # Cross-linked courtyards and extra divisions can coexist with the tree
        # partitions. These change room adjacency and which wall wings survive.
        if rng.random() < .35:
            top, left = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            bottom = h - 1 - int(rng.integers(1, 5))
            right = w - 1 - int(rng.integers(1, 5))
            clean[top, left:right+1] = clean[bottom, left:right+1] = True
            clean[top:bottom+1, left] = clean[top:bottom+1, right] = True
        for _ in range(int(rng.integers(3))):
            regions = _components(~clean)
            rng.shuffle(regions)
            if not any(_split_region(rng, clean, p) for p in regions):
                break
        if rng.random() < .35:
            regions = sorted(_components(~clean), key=lambda p: -int(p.sum()))
            for part in regions:
                if _put_motif(rng, clean, part, 'ring') is not None:
                    break
    # Several independent shapes may be retained islands or erased relics.
    for _ in range(int(rng.integers(4))):
        regions = _components(~clean)
        part = regions[int(rng.integers(len(regions)))]
        _put_motif(rng, clean, part, str(rng.choice(['block', 'L'])))
    # Select the large outside-of-ring room, the inside of the ring, or a new
    # branching subroom. No sampled identity will be needed by solve().
    if witness == 1:
        selected = _flood(~clean, (0, 0))
    elif witness == 2:
        # A stepped ring's first row need not contain its leftmost corner.
        outside_ring = _flood(~ring, (0, 0))
        interior = ~ring & ~outside_ring
        positions = np.argwhere(interior & ~clean)
        if not len(positions):
            return None
        selected = _flood(~clean, tuple(map(int, positions[len(positions)//2])))
    else:
        regions = _components(~clean)
        candidates = [p for p in regions if int(p.sum()) >= 18]
        if test and rng.random() < .70:
            right = [p for p in candidates if p[:, -1].any()]
            candidates = right or candidates
        selected = candidates[int(rng.integers(len(candidates)))]
    outside = _flood(np.pad(~selected, 1, constant_values=True), (0, 0))
    rim = selected & _near(outside, True)[1:-1, 1:-1]
    positions = np.argwhere(selected & (~rim if witness == 1 else rim))
    if not len(positions):
        positions = np.argwhere(selected)
    pos = tuple(map(int, positions[int(rng.integers(len(positions)))]))
    return clean, pos


def _corrupt_scene(rng, clean, pos, witness):
    selected, exterior, shell, islands, rim = _topology(clean, pos)
    options = _damage_options(clean)
    rng.shuffle(options)
    groups = {'shell': [], 'island': [], 'remote': []}
    touches_selected = _near(selected)
    touches_other_room = _near(~clean & ~selected)
    for gap, spur in options:
        if all(shell[r, c] for r, c in gap):
            # The connected three-cell opening must join two pre-existing free
            # rooms. Merely making its own three cells reachable is not a leak.
            if (any(touches_selected[r, c] for r, c in gap)
                    and any(touches_other_room[r, c] for r, c in gap)):
                groups['shell'].append((gap, spur))
        elif all(islands[r, c] for r, c in gap):
            groups['island'].append((gap, spur))
        elif all(not shell[r, c] and not islands[r, c] for r, c in gap):
            groups['remote'].append((gap, spur))
    raw, used, n = clean.copy(), np.zeros_like(clean), 0
    requirements = ['shell', 'island'] if witness == 1 else ['shell', 'remote']
    requirements += ['any'] * int(rng.integers(3))
    for kind in requirements:
        pool = options if kind == 'any' else groups[kind]
        done = False
        for gap, spur in pool:
            footprint = np.zeros_like(clean)
            for r, c in gap + spur:
                footprint[r, c] = True
            if (footprint & (used | _near(used, True))).any():
                continue
            trial = raw.copy()
            for r, c in gap:
                trial[r, c] = False
            for r, c in spur:
                trial[r, c] = True
            try:
                _, predicted, evidence = _recover(trial)
            except ValueError:
                continue
            if np.array_equal(predicted, clean) and len(evidence) == n + 1:
                raw, n = trial, n + 1
                used |= footprint
                done = True
                break
        if not done and kind != 'any':
            return None
    # Detached diagonal distractions, which never grant a repair by proximity.
    for _ in range(int(rng.integers(1, 4))):
        for _ in range(24):
            r, c = int(rng.integers(clean.shape[0])), int(rng.integers(clean.shape[1]))
            dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
            cells = [(r + k*dr, c + k*dc) for k in range(int(rng.integers(2, 4)))]
            forbidden = raw | _near(raw) | clean | _near(used, True)
            if not all(0 <= rr < clean.shape[0] and 0 <= cc < clean.shape[1]
                       and not forbidden[rr, cc] and (rr, cc) != pos for rr, cc in cells):
                continue
            trial = raw.copy()
            for rr, cc in cells:
                trial[rr, cc] = True
            try:
                stable, predicted, evidence = _recover(trial)
            except ValueError:
                continue
            if np.array_equal(predicted, clean) and len(evidence) == n:
                raw = trial
                break
    if raw[pos]:
        return None
    stable, virtual, evidence = _recover(raw)
    assert np.array_equal(virtual, clean)
    return raw, stable


def _make_pair(rng, palette, witness, test):
    for _ in range(100):
        scene = _base_scene(rng, witness, test)
        if scene is None:
            continue
        clean, pos = scene
        corruption = _corrupt_scene(rng, clean, pos, witness)
        if corruption is None:
            continue
        raw, stable = corruption
        if int(raw.sum()) * 2 >= raw.size - 1:
            continue
        bg, wall, marker = palette
        inp = np.where(raw, wall, bg).astype(np.int8)
        inp[pos] = marker
        out = _render(stable, clean, pos, palette)
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        inp, out = np.rot90(inp, turns), np.rot90(out, turns)
        if flip:
            inp, out = inp[:, ::-1], out[:, ::-1]
        pair = {'input': inp.copy(), 'output': out.copy()}
        f = _features(pair)
        if witness == 1 and not (f['withheld'] and f['pruned_connected']
                                 and f['diagonal_rim'] and f['unsupported']):
            continue
        return pair
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    # Optional None values resolve to the ordinary API defaults.
    n = _count('num_train', num_train, 2, 10, 4)
    t = _count('num_test', num_test, 1, 4, 1)
    if n <= t:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    vary_palette = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for index in range(count):
            witness = index + 1 if split == 'train' and index < 2 else 0
            for _ in range(30):
                roles = tuple(map(int, rng.choice(10, 3, replace=False))) if vary_palette else palette
                pair = _make_pair(rng, roles, witness, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique scene construction exhausted")
        rng.shuffle(puzzle[split])
    return puzzle


def _features(pair):
    g, out = np.asarray(pair['input']), np.asarray(pair['output'])
    bg, wall, marker = _roles(g)
    pos = tuple(map(int, np.argwhere(g == marker)[0]))
    raw = g == wall
    stable, virtual, repairs = _recover(raw)
    selected, exterior, shell, islands, rim = _topology(virtual, pos)
    filled = sum(all(shell[r, c] for r, c in gap) for gap, _, _ in repairs)
    withheld = sum(all(islands[r, c] for r, c in gap) for gap, _, _ in repairs)
    erased = sum(all(not shell[r, c] and not islands[r, c] for r, c in gap)
                 for gap, _, _ in repairs)
    pruned_connected = any((part & shell).any() and (part & ~shell & ~islands).any()
                           for part in _components(virtual))
    outside = _flood(np.pad(~selected, 1, constant_values=True), (0, 0))
    orth_rim = selected & _near(outside)[1:-1, 1:-1]
    return {'shape': g.shape, 'palette': (bg, wall, marker), 'repairs': len(repairs),
            'filled': filled, 'withheld': withheld, 'erased_repairs': erased,
            'islands': len(_components(islands)), 'rooms': len(_components(~virtual)),
            'area': int(selected.sum()), 'selected_enclosed': not bool(
                selected[0].any() or selected[-1].any() or selected[:, 0].any() or selected[:, -1].any()),
            'axes': ''.join(sorted({axis for _, _, axis in repairs})),
            'unsupported': len(_gaps(virtual)), 'pruned_connected': pruned_connected,
            'diagonal_rim': int(np.count_nonzero(rim & ~orth_rim)),
            'marker_on_rim': bool(rim[pos]),
            'deleted_diagonal_cells': int(np.count_nonzero(raw & ~stable)),
            'geometry': hashlib.sha256(virtual.tobytes()).hexdigest()}


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("puzzle must contain exactly train and test")
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, fs = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            before = g.copy()
            assert np.array_equal(solve(g), out), 'full output differs from rule'
            assert np.array_equal(g, before), 'solve mutated the input'
            bg, wall, marker = _roles(g)
            assert set(np.unique(out)) == {bg, wall, marker}
            pos = tuple(map(int, np.argwhere(g == marker)[0]))
            stable, virtual, repairs = _recover(g == wall)
            selected, exterior, shell, islands, rim = _topology(virtual, pos)
            added = virtual & ~stable
            assert not np.any((out == wall) & added & ~shell)
            assert np.all(out[stable & islands] == wall)
            assert np.all(out[stable & ~shell & ~islands] == bg)
            assert np.array_equal(out == marker, rim | (g == marker))
            assert out[pos] == marker
            # A deliberately wrong all-repairs-open flood must select more cells.
            escaped = _flood(~stable, pos)
            assert np.any(escaped & ~selected & ~virtual), 'no real room escape without A'
            f = _features(pair)
            assert f['filled'] >= 1 and f['deleted_diagonal_cells'] >= 4
            if split == 'train':
                fs.append(f)
    assert any(f['withheld'] and f['unsupported'] for f in fs)
    assert any(f['erased_repairs'] for f in fs)
    assert any(f['pruned_connected'] and f['diagonal_rim'] for f in fs)
    assert any(f['selected_enclosed'] for f in fs)
    assert any(not f['selected_enclosed'] for f in fs)
    assert {f['marker_on_rim'] for f in fs} == {False, True}


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def _analytic_checks():
    """Coordinate-written fixtures: the same broken loop is island or shell."""
    clean = np.zeros((19, 24), bool)
    clean[:, 15] = True
    clean[5, 4:11] = clean[11, 4:11] = True
    clean[5:12, 4] = clean[5:12, 10] = True
    clean[3:8, 21] = True
    raw = clean.copy()
    raw[8:11, 15] = raw[5, 6:9] = False
    raw[8, 16] = raw[9, 17] = raw[6, 6] = raw[7, 7] = True
    stable, virtual, repairs = _recover(raw)
    assert np.array_equal(virtual, clean) and len(repairs) == 2
    inp = np.where(raw, 8, 0).astype(np.int8)
    inp[3, 2] = 2
    expected = np.zeros_like(inp)
    expected[:, 15] = 8
    expected[5, 4:11] = expected[11, 4:11] = 8
    expected[5:12, 4] = expected[5:12, 10] = 8
    expected[5, 6:9] = 0  # locally evidenced, but not an external wall
    expected[[0, 18], :15] = 2
    expected[:, [0, 14]] = 2
    expected[3, 2] = 2
    assert np.array_equal(solve(inp), expected)
    moved = inp.copy()
    moved[3, 2], moved[9, 7] = 0, 2
    expected = np.zeros_like(inp)
    expected[5, 4:11] = expected[11, 4:11] = 8
    expected[5:12, 4] = expected[5:12, 10] = 8
    expected[6, 5:10] = expected[10, 5:10] = 2
    expected[6:11, 5] = expected[6:11, 9] = 2
    expected[9, 7] = 2
    assert np.array_equal(solve(moved), expected)


def self_test():
    """Serial 42-seed audit; stores hashes/statistics, never a corpus of grids."""
    _analytic_checks()
    signatures, records = set(), []
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(hashlib.sha256(repr(_signature(p)).encode()).hexdigest())
        local = [_features(pair) for s in ('train', 'test') for pair in p[s]]
        records.extend(local)
        assert len({f['geometry'] for f in local}) == 5
        assert len({f['area'] for f in local}) >= 2
        assert len({f['shape'] for f in local}) >= 2
    assert len(signatures) == 42
    assert len({f['shape'] for f in records}) >= 35
    assert len({f['geometry'] for f in records}) >= 190
    assert len({f['palette'] for f in records}) >= 30
    assert {'h', 'v', 'hv'}.issubset({f['axes'] for f in records})
    assert max(f['rooms'] for f in records) >= 6
    assert len({f['repairs'] for f in records}) >= 3
    assert max(f['islands'] for f in records) >= 2
    assert any(f['withheld'] for f in records)
    assert any(f['erased_repairs'] for f in records)
    p = generate_puzzle(0, 2, 1)
    assert _signature(p) == _signature(generate_puzzle(0, 2, 1))
    assert _signature(p) != _signature(generate_puzzle(1, 2, 1))
    # Explicit parent omissions and alternative boundary definitions.
    rejected_all_gap = rejected_hole_rim = False
    for pair in p['train']:
        inp, out = pair['input'], pair['output']
        palette = _roles(inp)
        pos = tuple(map(int, np.argwhere(inp == palette[2])[0]))
        stable, virtual, repairs = _recover(inp == palette[1])
        assert not np.array_equal(_render(stable, stable, pos, palette), out)
        all_repairs = out.copy()
        all_repairs[virtual & ~stable] = palette[1]
        assert not np.array_equal(all_repairs, out)
        selected, _, _, _, rim = _topology(virtual, pos)
        hole_rim = selected & _near(np.pad(~selected, 1, constant_values=True), True)[1:-1, 1:-1]
        rejected_hole_rim |= bool(np.any(hole_rim & ~rim))
        indiscriminate = virtual.copy()
        for r, c, dr, dc in _gaps(virtual):
            for k in (1, 2, 3):
                indiscriminate[r+k*dr, c+k*dc] = True
        rejected_all_gap |= not np.array_equal(_render(indiscriminate, indiscriminate, pos, palette), out)
    assert rejected_hole_rim and rejected_all_gap
    # Boundary/representative counts here; verify.py enumerates every legal
    # combination. Avoid duplicating that full sweep inside every host call.
    for n, t in ((2, 1), (10, 1), (10, 4), (6, 3)):
        q = generate_puzzle(700 + 10*n + t, n, t)
        assert (len(q['train']), len(q['test'])) == (n, t)
        validate_generated_puzzle(q)
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.5, 1), (4, False), (4, 1.2)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts were accepted')
    # Rotation/reflection covariance and visible color-role reassignment.
    for pair in p['train'] + p['test']:
        for turns in range(4):
            for flip in (False, True):
                g, out = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
                if flip:
                    g, out = g[:, ::-1], out[:, ::-1]
                assert np.array_equal(solve(g.copy()), out)
        permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], np.int8)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # Move the marker within the inferred room: its painted rim is unchanged.
    pair = p['train'][0]
    g = pair['input'].copy()
    bg, wall, marker = _roles(g)
    pos = tuple(map(int, np.argwhere(g == marker)[0]))
    stable, virtual, repairs = _recover(g == wall)
    selected, _, _, _, rim = _topology(virtual, pos)
    positions = np.argwhere(selected & (g == bg))
    new_pos = tuple(map(int, positions[len(positions)//2]))
    moved = g.copy()
    moved[pos], moved[new_pos] = bg, marker
    expected = pair['output'].copy()
    expected[pos] = marker if rim[pos] else bg
    expected[new_pos] = marker
    assert np.array_equal(solve(moved), expected)
    # Corrupt a single withheld repair; full-grid validation must reject it.
    q = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    pair = next(x for x in q['train'] if _features(x)['withheld'])
    bg, wall, marker = _roles(pair['input'])
    pos = tuple(map(int, np.argwhere(pair['input'] == marker)[0]))
    stable, virtual, repairs = _recover(pair['input'] == wall)
    _, _, _, islands, _ = _topology(virtual, pos)
    gap = next(gap for gap, _, _ in repairs if all(islands[r, c] for r, c in gap))
    for r, c in gap:
        pair['output'][r, c] = wall
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect island repair accepted')
    return {'seeds': 42, 'pairs': len(records), 'distinct_puzzles': len(signatures),
            'shapes': len({f['shape'] for f in records}),
            'geometries': len({f['geometry'] for f in records}),
            'palettes': len({f['palette'] for f in records}),
            'room_counts': sorted({f['rooms'] for f in records}),
            'repair_counts': sorted({f['repairs'] for f in records}),
            'withheld_pairs': sum(f['withheld'] > 0 for f in records),
            'pruned_connected_pairs': sum(f['pruned_connected'] for f in records)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
