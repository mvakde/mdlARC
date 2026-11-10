"""Centroid-addressed compass docking. Standard library and NumPy only.

Addresses are read in the SOURCE core's coordinate frame. They are not
recomputed after a rigid motion. All solving and validation use input pixels.
No samples, parent modules, files, or random state are consulted by solve.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NEAR = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 938391583, 918177420, 561060297, 136214151, 180879158,
              37802156, 634761105, 382915095, 338307090, 137946990, 906363293,
              287254850, 850750944, 221452499, 986970084, 255154088, 410362246,
              207972458, 904824313, 272583010, 894992963, 340818539, 115679189,
              769903302, 997292538, 286366026, 526273824, 827516570, 350479774,
              266158423, 100008073, 987999516, 972468, 537863121, 717866558,
              135574029, 596059578, 768447884, 24879391, 208319074, 69223445)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "bad grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(points, diagonal=True):
    left, result = set(points), []
    steps = NEAR if diagonal else DIRS
    while left:
        start = min(left)
        left.remove(start)
        stack, part = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _transform(cells, t):
    moved = []
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        moved.append((r, c, v))
    r0, c0 = min(r for r, c, v in moved), min(c for r, c, v in moved)
    return {(r - r0, c - c0): v for r, c, v in moved}


def _shift(cells, r, c):
    return {(rr + r, cc + c): v for (rr, cc), v in cells.items()}


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _canonical(cells):
    return min(_signature(_transform(cells, t)) for t in range(8))


def _bounds(points):
    return (min(r for r, c in points), max(r for r, c in points),
            min(c for r, c in points), max(c for r, c in points))


def _axis(total, count, size):
    # Cell centers in a rectangle divided into three equal physical strips.
    numerator = 3 * (2 * total + count)
    denominator = 2 * count * size
    _require(numerator not in (denominator, 2 * denominator), "centroid on sector boundary")
    return 0 if numerator < denominator else 1 if numerator < 2 * denominator else 2


def _address(cells, reg, bands, bbox_centroid=False):
    core = {p for p, v in cells.items() if v not in bands}
    r0, r1, c0, c1 = _bounds(core)
    marks = [p for p, v in cells.items() if v == reg]
    _require(bool(marks), "missing registration")
    if bbox_centroid:
        lo, hi, le, ri = _bounds(marks)
        rr, cc, n = lo + hi - 2 * r0, le + ri - 2 * c0, 2
    else:
        rr = sum(r - r0 for r, c in marks)
        cc = sum(c - c0 for r, c in marks)
        n = len(marks)
    return 3 * _axis(rr, n, r1 - r0 + 1) + _axis(cc, n, c1 - c0 + 1)


def _labels(cells, bands):
    core = {p for p, v in cells.items() if v not in bands}
    r0, r1, c0, c1 = _bounds(core)
    groups = {}
    for (r, c), v in cells.items():
        if v not in bands:
            continue
        possible = [i for i, ok in enumerate((r < r0, c > c1, r > r1, c < c0)) if ok]
        _require(len(possible) == 1, "label has no clear outward side")
        d = possible[0]
        _require((r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1), "detached label")
        groups.setdefault(d, []).append(((r, c), v))
    _require(2 <= len(groups) <= 4, "need two or more labeled sides")
    for group in groups.values():
        _require(len(group) == 2 and group[0][1] == group[1][1], "label must be a same-color domino")
        p, q = group[0][0], group[1][0]
        _require(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1, "label must be contiguous")
    return {d: group[0][1] for d, group in groups.items()}


def _layout(a):
    h, w = a.shape
    _require((h - 4) % 3 == 0 and (w - 4) % 3 == 0, "expected 3-by-3 lattice")
    ph, pw = (h - 4) // 3, (w - 4) // 3
    _require(6 <= ph <= 8 and 6 <= pw <= 8, "panel size outside family")
    rows, cols = list(range(0, h, ph + 1)), list(range(0, w, pw + 1))
    mask = np.zeros(a.shape, dtype=bool)
    mask[rows, :] = True
    mask[:, cols] = True
    free_r = [r for r in range(h) if r not in rows]
    free_c = [c for c in range(w) if c not in cols]
    rc, cc = [], []
    for r in rows:
        values = a[r, free_c]
        _require(bool(np.all(values == values[0])), "broken horizontal band")
        rc.append(int(values[0]))
        _require(bool(np.all(a[r, cols] == values[0])), "horizontal bands own intersections")
    for c in cols:
        values = a[free_r, c]
        _require(bool(np.all(values == values[0])), "broken vertical band")
        cc.append(int(values[0]))
    _require(len(set(rc)) == len(set(cc)) == 2 and not set(rc) & set(cc), "four compass identities required")
    _require(all(rc[i] != rc[i + 1] and cc[i] != cc[i + 1] for i in range(3)), "alternating bands required")
    bands = set(rc + cc)
    ids = np.full(a.shape, -1, dtype=np.int8)
    rects, sides = [], []
    for i in range(9):
        r, c = divmod(i, 3)
        r0, c0 = 1 + r * (ph + 1), 1 + c * (pw + 1)
        ids[r0:r0 + ph, c0:c0 + pw] = i
        rects.append((r0, r0 + ph, c0, c0 + pw))
        sides.append((rc[r], cc[c + 1], rc[r + 1], cc[c]))
    return mask, ids, rects, sides, bands, (ph, pw)


def _parse(value):
    a = _grid(value)
    mask, ids, rects, sides, bands, panel_shape = _layout(a)
    values, counts = np.unique(a[~mask], return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(bg not in bands and counts.max() > counts.sum() / 2, "fill must dominate pooled interiors")
    objects = _components(map(tuple, np.argwhere((a != bg) & ~mask)))
    sources, targets = [], []
    for obj in objects:
        cells = {p: int(a[p]) for p in obj}
        if len(set(cells.values())) == 1:
            targets.append(cells)
        else:
            sources.append(cells)
    _require(sources and len(sources) == len(targets), "source/stencil counts differ")
    colors = {v for cells in targets for v in cells.values()}
    _require(len(colors) == 1, "one registration color required")
    reg = next(iter(colors))
    bodies, addresses = set(), []
    for cells in sources:
        bodies.update(set(cells.values()) - bands - {reg})
        _require(len({int(ids[p]) for p in cells}) == 1, "source crosses bands")
        _require(len(_components(cells, diagonal=False)) == 1, "payload must be connected")
        _labels(cells, bands)
        core_size = sum(v not in bands for v in cells.values())
        _require(0 < sum(v == reg for v in cells.values()) < core_size / 2, "registration must be a strict minority of core")
        addresses.append(_address(cells, reg, bands))
    _require(len(bodies) == 1 and not ({bg, reg} & bands) and bg != reg, "unreadable palette roles")
    body = next(iter(bodies))
    _require(body != bg, "body equals fill")
    target_panels = []
    for target in targets:
        dest = {int(ids[p]) for p in target}
        _require(len(dest) == 1, "stencil crosses bands")
        target_panels.append(next(iter(dest)))
    _require(sorted(addresses) == sorted(target_panels), "address and stencil panel multiplicities differ")
    return dict(a=a, mask=mask, ids=ids, rects=rects, sides=sides, bands=bands,
                bg=bg, reg=reg, body=body, sources=sources, targets=targets,
                addresses=addresses, target_panels=target_panels, panel_shape=panel_shape)


def _candidates(cells, address, info, use_address=True, use_labels=True):
    result, seen = [], set()
    for t in range(8):
        moved = _transform(cells, t)
        marks = {p for p, v in moved.items() if v == info['reg']}
        labels = _labels(moved, info['bands'])
        for j, target in enumerate(info['targets']):
            panel = info['target_panels'][j]
            if use_address and panel != address:
                continue
            if use_labels and any(info['sides'][panel][d] != v for d, v in labels.items()):
                continue
            if len(marks) != len(target):
                continue
            start, anchor = min(marks), min(target)
            dr, dc = anchor[0] - start[0], anchor[1] - start[1]
            if {(r + dr, c + dc) for r, c in marks} != set(target):
                continue
            placed = _shift(moved, dr, dc)
            r0, r1, c0, c1 = info['rects'][panel]
            if any(not (r0 <= r < r1 and c0 <= c < c1) for r, c in placed):
                continue
            key = j, tuple(sorted(placed.items()))
            if key not in seen:
                seen.add(key)
                result.append((j, placed, t))
    return result


def _solutions(info, use_address=True, use_labels=True, limit=2):
    choices = [_candidates(c, a, info, use_address, use_labels)
               for c, a in zip(info['sources'], info['addresses'])]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    results = {}

    def search(k, used, occupied, placements):
        if len(results) >= limit:
            return
        if k == len(order):
            key = tuple(sorted(occupied.items()))
            results[key] = (occupied, placements)
            return
        index = order[k]
        for target, cells, t in choices[index]:
            if target in used or occupied.keys() & cells.keys():
                continue
            search(k + 1, used | {target}, {**occupied, **cells},
                   placements + [(index, target, cells, t)])
    search(0, set(), {}, [])
    return list(results.values())


def _paint(info, cells):
    out = np.full_like(info['a'], info['bg'])
    out[info['mask']] = info['a'][info['mask']]
    for p, v in cells.items():
        out[p] = v
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    answers = _solutions(info)
    _require(len(answers) == 1, "docking has zero or multiple pixel outputs")
    return _paint(info, answers[0][0])


def _path(rng, start, end):
    r, c = start
    points = {(r, c)}
    while (r, c) != end:
        if r != end[0] and (c == end[1] or rng.random() < .5):
            r += 1 if end[0] > r else -1
        else:
            c += 1 if end[1] > c else -1
        points.add((r, c))
    return points


def _registration(rng, h, w, address, single=False, skew=False):
    if skew:
        points = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)}
        if address == 2:
            points = {(r, 4 - c) for r, c in points}
        elif address == 6:
            points = {(4 - r, c) for r, c in points}
        return points
    for _ in range(150):
        p = int(rng.integers(h)), int(rng.integers(w))
        points = {p}
        size = 1 if single else int(rng.integers(1, 6))
        for _ in range(size - 1):
            edge = sorted({(r + dr, c + dc) for r, c in points for dr, dc in NEAR
                           if 0 <= r + dr < h and 0 <= c + dc < w} - points)
            if edge:
                points.add(edge[int(rng.integers(len(edge)))])
        try:
            a = 3 * _axis(sum(r for r, c in points), len(points), h)
            a += _axis(sum(c for r, c in points), len(points), w)
        except ValueError:
            continue
        fractions = ((sum(r for r, c in points) / len(points) + .5) / h,
                     (sum(c for r, c in points) / len(points) + .5) / w)
        clear = all(min(abs(x - 1 / 3), abs(x - 2 / 3)) >= .03 for x in fractions)
        if a == address and clear:
            return points
    # Every size 3..5 has a cell center strictly inside each of the three strips.
    rr, cc = divmod(address, 3)
    return {((0, h // 2, h - 1)[rr], (0, w // 2, w - 1)[cc])}


def _piece(rng, address, sides, palette, small=False, skew=False, pose=None):
    bg, body, reg = palette[:3]
    h, w = (3, 3) if small else (5, 5) if skew else (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
    marks = _registration(rng, h, w, address, single=small, skew=skew)
    # Two adjacent labeled sides fix handedness as well as rotation. Richer
    # pieces can have three or four labeled sides, always domino patches.
    first = int(rng.integers(4))
    selected = {first, (first + 1) % 4}
    if not small and rng.random() < .4:
        selected.add((first + 2) % 4)
    if not small and rng.random() < .15:
        selected = set(range(4))
    labels, supports = {}, []
    for d in sorted(selected):
        n = w if d in (0, 2) else h
        k = int(rng.integers(n - 1))
        for v in (k, k + 1):
            p, q = (((-1, v), (0, v)), ((v, w), (v, w - 1)),
                    ((h, v), (h - 1, v)), ((v, -1), (v, 0)))[d]
            labels[p] = 20 + d
            supports.append(q)
    ends = sorted(marks) + supports + [(0, int(rng.integers(w))),
            (h - 1, int(rng.integers(w))), (int(rng.integers(h)), 0),
            (int(rng.integers(h)), w - 1)]
    core = {min(marks)}
    rng.shuffle(ends)
    for endpoint in ends:
        nearest = min(abs(r - endpoint[0]) + abs(c - endpoint[1]) for r, c in core)
        anchors = sorted(p for p in core if abs(p[0] - endpoint[0]) + abs(p[1] - endpoint[1]) == nearest)
        core.update(_path(rng, anchors[int(rng.integers(len(anchors)))], endpoint))
    density = float(rng.uniform(.45, 1.0))
    pool = list((r, c) for r in range(h) for c in range(w))
    rng.shuffle(pool)
    for p in pool:
        if len(core) >= max(h * w * density, 2 * len(marks) + 1):
            break
        if any((p[0] + dr, p[1] + dc) in core for dr, dc in DIRS):
            core.add(p)
    if len(core) <= 2 * len(marks):
        core = set(pool)
    cells = {**{p: body for p in core}, **{p: reg for p in marks}, **labels}
    t = int(rng.integers(8)) if pose is None else pose
    moved = _transform(cells, t)
    label_colors = {v: sides[d] for d, v in _labels(moved, set(labels.values())).items()}
    source = {p: label_colors.get(v, v) for p, v in cells.items()}
    return _transform(source, 0), _transform({p: label_colors.get(v, v) for p, v in moved.items()}, 0)


def _origins(cells, rect):
    h, w = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
    r0, r1, c0, c1 = rect
    return [(r, c) for r in range(r0, r1 - h + 1) for c in range(c0, c1 - w + 1)]


def _halo(points):
    return {(r + dr, c + dc) for r, c in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _construct(rng, addresses, palette, shape, teaching=False):
    ph, pw = shape
    bg, body, reg, north, south, west, east = palette
    base = np.full((3 * ph + 4, 3 * pw + 4), bg, dtype=np.int8)
    for j, c in enumerate(range(0, base.shape[1], pw + 1)):
        base[:, c] = (west, east)[j % 2]
    for i, r in enumerate(range(0, base.shape[0], ph + 1)):
        base[r, :] = (north, south)[i % 2]
    mask, ids, rects, sides, bands, _ = _layout(base)
    sources, finals, targets, final_used = [], [], {}, set()
    for i, address in enumerate(addresses):
        small = teaching and i < 2
        skew = teaching and i == 3
        source, final = _piece(rng, address, sides[address], palette, small=small,
                               skew=skew, pose=(1 if i == 0 else 4 if i == 1 else None) if teaching else None)
        origins = _origins(final, rects[address])
        rng.shuffle(origins)
        if small:
            marks = [p for p, v in final.items() if v == reg]
            r0, r1, c0, c1 = rects[address]
            anchor = (r0 + ph // 2, c0 + pw // 2)
            origins = [(anchor[0] - marks[0][0], anchor[1] - marks[0][1])]
        found = False
        for r, c in origins:
            placed = _shift(final, r, c)
            r0, r1, c0, c1 = rects[address]
            if any(not (r0 <= rr < r1 and c0 <= cc < c1) for rr, cc in placed):
                continue
            if placed.keys() & final_used:
                continue
            marks = {p: reg for p, v in placed.items() if v == reg}
            if set(marks) & _halo(targets):
                continue
            final_used.update(placed)
            targets.update(marks)
            sources.append(source)
            finals.append(placed)
            found = True
            break
        if not found:
            return None
    # Pack larger sources first. Source and final footprints may intersect;
    # detached targets and all input pieces have an eight-neighbor margin.
    blocked, packed = _halo(targets), []
    for i in sorted(range(len(sources)), key=lambda j: -len(sources[j])):
        source = sources[i]
        origins = [(r, c) for rect in rects for r, c in _origins(source, rect)]
        rng.shuffle(origins)
        placed = None
        for r, c in origins:
            trial = _shift(source, r, c)
            if not trial.keys() & blocked:
                placed = trial
                break
        if placed is None:
            return None
        blocked.update(_halo(placed))
        packed.append(placed)
    inp, out = base.copy(), base.copy()
    for p, v in targets.items():
        inp[p] = v
    for cells in packed:
        for p, v in cells.items():
            inp[p] = v
    for cells in finals:
        for p, v in cells.items():
            out[p] = v
    try:
        info = _parse(inp)
        answers = _solutions(info)
        if len(answers) != 1 or not np.array_equal(_paint(info, answers[0][0]), out):
            return None
        if teaching and (len(_solutions(info, use_address=False)) < 2 or len(_solutions(info, use_labels=False)) < 2):
            return None
    except ValueError:
        return None
    return {'input': inp, 'output': out}


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)), name + ' must be an integer')
    _require(low <= int(value) <= high, name + ' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    _require(num_train > num_test, 'num_train must exceed num_test')
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0, 'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    corner = int(rng.choice((2, 6)))
    edge = int(rng.choice((1, 3, 5, 7)))
    first = [0, 8, 4, corner, edge]
    second = [a for a in range(9) if a not in first]
    rng.shuffle(second)
    shared = bool(rng.random() < .3)
    palette = list(map(int, rng.permutation(10)[:7]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            teaching = split == 'train' and index == 0
            if not shared or (split == 'train' and index == 1):
                palette = list(map(int, rng.permutation(10)[:7]))
            if teaching:
                addresses, shape = first, (8, 8)
            elif split == 'train' and index == 1:
                addresses, shape = second, ((7, 8), (8, 7))[int(rng.integers(2))]
            else:
                n = int(rng.integers(2, 6))
                addresses = list(map(int, rng.choice(9, n, replace=False)))
                # Tests recombine addresses at a held-out short panel extent.
                shape = ((6, 8), (8, 6))[int(rng.integers(2))] if split == 'test' else ((7, 7), (7, 8), (8, 7), (8, 8))[int(rng.integers(4))]
            for _ in range(120):
                pair = _construct(rng, addresses, palette, shape, teaching=teaching)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded packing could not construct pair')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def _inspect(inp, out):
    info = _parse(inp)
    answers = _solutions(info)
    _require(len(answers) == 1, 'nonunique output')
    pixels, placements = answers[0]
    _require(np.array_equal(_paint(info, pixels), out), 'incorrect answer')
    _require(np.array_equal(inp[info['mask']], out[info['mask']]), 'bands changed')
    _require(not np.array_equal(inp, out), 'uninformative identity')
    final_reg = {p for p, v in pixels.items() if v == info['reg']}
    target_reg = {p for target in info['targets'] for p in target}
    _require(final_reg == target_reg, 'stencils not exactly covered')
    old_reg = sum(v == info['reg'] for s in info['sources'] for v in s.values())
    _require(np.count_nonzero(inp == info['reg']) - np.count_nonzero(out == info['reg']) == old_reg, 'wrong registration count after overlay')
    moves, shapes, patterns, bbox_diff, post_diff = [], [], [], False, False
    overlap, recolored = 0, 0
    old = set().union(*(set(s) for s in info['sources']))
    for index, target, cells, t in placements:
        source = info['sources'][index]
        _require(_canonical(source) == _canonical(cells), 'whole payload was altered')
        possible = [j for j in range(8) if _signature(_transform(source, j)) == _signature(cells)]
        moves.append('translation' if 0 in possible else 'rotation' if any(j < 4 for j in possible) else 'reflection')
        r0, r1, c0, c1 = _bounds(source)
        shapes.append((r1 - r0 + 1, c1 - c0 + 1))
        patterns.append(_canonical({p: 1 for p, v in source.items() if v == info['reg']}))
        try:
            bbox_diff |= _address(source, info['reg'], info['bands'], True) != info['addresses'][index]
        except ValueError:
            bbox_diff = True
        post_diff |= _address(cells, info['reg'], info['bands']) != info['addresses'][index]
    for p in old & pixels.keys():
        overlap += 1
        recolored += int(inp[p]) != int(out[p])
    return info, dict(shape=inp.shape, panel_shape=info['panel_shape'], count=len(info['sources']),
                      addresses=tuple(sorted(info['addresses'])), moves=moves, shapes=shapes,
                      patterns=patterns, bbox_diff=bbox_diff, post_diff=post_diff,
                      overlap=overlap, recolored=recolored, bg=info['bg'])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal pair counts')
    seen, train, metrics = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'size changed')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info, record = _inspect(inp, out)
            metrics.append(record)
            if split == 'train':
                train.append((info, record))
    _require(set().union(*(set(i['addresses']) for i, m in train)) == set(range(9)), 'training does not expose all nine sectors')
    info = train[0][0]
    _require(len(_solutions(info, use_address=False)) == 2, 'A is not necessary in first demonstration')
    _require(len(_solutions(info, use_labels=False)) == 2, 'B compass is not necessary in first demonstration')
    _require(any(m['bbox_diff'] for i, m in train), 'missing collective-centroid demonstration')
    _require(any('rotation' in m['moves'] for i, m in train) and any('reflection' in m['moves'] for i, m in train), 'D4 motion insufficiently demonstrated')
    _require(any(m['post_diff'] for i, m in train), 'source-before-motion address is not exposed')
    _require(len({i['panel_shape'] for i, m in train}) >= 2, 'missing geometry variation')
    return None


def self_test():
    # Streaming: retain feature sets and hashes, never all puzzles or traces.
    shapes, patterns, extents, counts, digests, colors, address_sets = (set() for _ in range(7))
    moves, overlaps, bbox_cases, post_cases = set(), 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info, m = _inspect(inp, out)
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                shapes.add(m['shape'])
                patterns.update(m['patterns'])
                extents.update(m['shapes'])
                counts.add(m['count'])
                colors.update(map(int, np.unique(inp)))
                address_sets.add(m['addresses'])
                moves.update(m['moves'])
                overlaps += m['overlap'] > 0
                bbox_cases += m['bbox_diff']
                post_cases += m['post_diff']
        digests.add(digest.hexdigest())
    _require(len(digests) == 42 and len(shapes) >= 6, 'insufficient scene variation')
    _require(len(patterns) >= 20 and len(extents) >= 10, 'insufficient payload variation')
    _require(counts == {2, 3, 4, 5} and colors == set(range(10)), 'insufficient counts/palette variation')
    _require(moves == {'translation', 'rotation', 'reflection'}, 'missing motion class')
    _require(len(address_sets) >= 35 and overlaps >= 5 and bbox_cases >= 42 and post_cases >= 42, 'missing interaction coverage')
    x, y = generate_puzzle(987, 2, 1), generate_puzzle(987, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in x for p, q in zip(x[s], y[s]) for k in p), 'nondeterminism')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(123 + a * 10 + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('accepted illegal counts')
    return dict(seeds=42, pairs=210, grid_shapes=len(shapes), registration_shapes=len(patterns),
                payload_extents=len(extents), address_sets=len(address_sets), overlap_pairs=overlaps,
                bbox_centroid_counterexamples=bbox_cases, source_frame_counterexamples=post_cases)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
