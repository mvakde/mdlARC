"""Count-keyed sluices: topological matching decides which bent gates close.

Only NumPy and the standard library are used. Generation constructs labels from
room masks; the decoder separately recovers hypothetical enclosures from input.
No parent module, file, seed, training example, or generation state is used by
solve(). All sweeps stream puzzles in memory and write no artifacts.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, bool)
    steps = FOUR + ((1, 1), (1, -1), (-1, 1), (-1, -1)) if diagonal else FOUR
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _box(cells):
    r, c = np.asarray(cells).T
    return int(r.min()), int(r.max()) + 1, int(c.min()), int(c.max()) + 1


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _strong(mask):
    result = np.zeros_like(mask)
    for a, b in ((mask, result), (mask.T, result.T)):
        triples = a[:, :-2] & a[:, 1:-1] & a[:, 2:]
        for offset in range(3):
            b[:, offset:offset + triples.shape[1]] |= triples
    return result


def _candidates(mask):
    """A's local evidence, before the new count condition decides acceptance."""
    strong = _strong(mask)
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        a = mask if dc else mask.T
        brackets = a[:, :-4] & a[:, 4:]
        brackets &= ~(a[:, 1:-3] | a[:, 2:-2] | a[:, 3:-1])
        for y, x in np.argwhere(brackets):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            if not (_at(mask, r - dr, c - dc) or _at(mask, r + 5 * dr, c + 5 * dc)):
                continue
            matches = []
            for endpoint in (0, 4):
                ar, ac = r + endpoint * dr, c + endpoint * dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    dy, dx = toward * dr + side * dc, toward * dc + side * dr
                    cells = tuple((ar + k * dy, ac + k * dx) for k in (1, 2))
                    if (all(_at(mask, yy, xx) and not strong[yy, xx] for yy, xx in cells)
                            and not _at(mask, ar + 3 * dy, ac + 3 * dx)):
                        matches.append(cells)
            if matches:
                assert len(matches) == 1, "ambiguous diagonal attachment"
                result.append(((r, c, dr, dc), matches[0]))
    return result


def _clean(mask, candidates):
    # Hypothetically extend all candidate runs only to identify supported line
    # cells. This does NOT approve those repairs for the final answer.
    extended = mask.copy()
    anchors = np.zeros_like(mask)
    for (r, c, dr, dc), _ in candidates:
        for k in (1, 2, 3):
            extended[r + k * dr, c + k * dc] = True
        anchors[r, c] = anchors[r + 4 * dr, c + 4 * dc] = True
    weak = mask & ~_strong(extended) & ~anchors
    diagonal = np.zeros_like(mask)
    links = weak[:-1, :-1] & weak[1:, 1:]
    diagonal[:-1, :-1] |= links
    diagonal[1:, 1:] |= links
    links = weak[:-1, 1:] & weak[1:, :-1]
    diagonal[:-1, 1:] |= links
    diagonal[1:, :-1] |= links
    return mask & ~diagonal


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _roles_and_keys(grid):
    colors = [int(c) for c in np.unique(grid) if c]
    wall_options, key_options, dot_options = [], [], []
    for color in colors:
        mask = grid == color
        adjacent = bool((mask[:-1] & mask[1:]).any() or (mask[:, :-1] & mask[:, 1:]).any())
        r0, r1, c0, c1 = _box(np.argwhere(mask))
        if adjacent and max(r1 - r0, c1 - c0) > 7:
            wall_options.append(color)
        elif adjacent:
            key_options.append((color, (r0, r1, c0, c1)))
        else:
            dot_options.append(color)
    assert len(wall_options) == 1 and len(dot_options) <= 1
    assert 2 <= len(key_options) <= 3
    keys = {}
    key_info = []
    for color, (r0, r1, c0, c1) in key_options:
        a = grid[r0:r1, c0:c1]
        assert 5 <= len(a) <= 7 and a.shape[0] == a.shape[1]
        assert set(map(int, np.unique(a))) <= {0, color}
        assert np.all(a[0] == color) and np.all(a[-1] == color)
        assert np.all(a[:, 0] == color) and np.all(a[:, -1] == color)
        assert len(_components(a == color)) == 1
        cavities = _components(a == 0)
        k = len(cavities)
        assert 0 <= k <= 5 and k not in keys
        keys[k] = color
        key_info.append({"count": k, "color": color, "area": int((a == 0).sum()),
                         "sizes": tuple(sorted(map(len, cavities))),
                         "eight_count": len(_components(a == 0, True)),
                         "shape": a.shape, "box": (r0, r1, c0, c1)})
    return wall_options[0], dot_options[0] if dot_options else None, keys, key_info


def _enclosed(mask):
    h, w = mask.shape
    return [cells for cells in _components(~mask)
            if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells)]


def _decode(input_grid, *, count_mode="four", accept_all=False, ignore_cues=False):
    """Recover each would-be enclosure separately, then apply its count key."""
    grid = _grid(input_grid)
    wall, dot, keys, key_info = _roles_and_keys(grid)
    mask = grid == wall
    candidates = _candidates(mask)
    assert candidates
    base_full = _clean(mask, candidates)
    r0, r1, c0, c1 = _box(np.argwhere(base_full))
    base = base_full[r0:r1, c0:c1]
    assert min(base.shape) >= 6 and max(base.shape) > 7
    assert not _enclosed(base), "every initial chamber must leak through its gate"
    dots = np.zeros(base.shape, bool) if dot is None else grid[r0:r1, c0:c1] == dot
    assert not dots[base].any()
    if dot is not None:
        assert int((grid == dot).sum()) == int(dots.sum()), "clues must be inside the network"
    assert not (base[:-1, :-1] & base[1:, :-1] & base[:-1, 1:] & base[1:, 1:]).any()
    for index, info in enumerate(key_info):
        a, b, c, d = info["box"]
        assert b < r0 or r1 < a or d < c0 or c1 < c, "key boxes need a blank gap"
        for other in key_info[index + 1:]:
            e, f, g, h = other["box"]
            assert b < e or f < a or d < g or h < c, "keys need a blank gap"
    output = np.where(base, wall, 0).astype(np.int8)
    gate_records, claimed = [], np.zeros_like(base)
    local = [(r - r0, c - c0, dr, dc) for (r, c, dr, dc), _ in candidates]
    if ignore_cues:
        local = _boundary_gaps(base)
    if count_mode == "area":
        lookup = {info["area"]: info["color"] for info in key_info}
    elif count_mode == "eight":
        # Ambiguous alternatives choose the smaller literal color, a fixed
        # deterministic tie rule. Training witnesses also refute the untied case.
        lookup = {}
        for info in key_info:
            k = info["eight_count"]
            lookup[k] = min(lookup.get(k, 10), info["color"])
    else:
        lookup = keys
    for r, c, dr, dc in local:
        assert (dc and r in (0, base.shape[0] - 1)) or (dr and c in (0, base.shape[1] - 1))
        gap = tuple((r + k * dr, c + k * dc) for k in (1, 2, 3))
        trial = base.copy()
        for yy, xx in gap:
            trial[yy, xx] = True
        cavities = _enclosed(trial)
        assert len(cavities) == 1, "a candidate must seal exactly one chamber"
        region = np.zeros_like(base)
        for yy, xx in cavities[0]:
            region[yy, xx] = True
        assert not (region & claimed).any(), "candidate chambers overlap"
        claimed |= region
        count = int(dots[region].sum())
        assert count <= 6
        color = lookup.get(count)
        accepted = color is not None
        if accepted or accept_all:
            for yy, xx in gap:
                output[yy, xx] = wall
        if accepted:
            output[region] = color
        ys, xs = np.where(region)
        rectangular = len(ys) == (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        gate_records.append({"gap": gap, "count": count, "accepted": accepted,
                             "color": color, "area": int(region.sum()),
                             "rectangular": bool(rectangular), "axis": "h" if dc else "v",
                             "region": region})
    info = {"wall": wall, "dot": dot, "keys": keys, "key_info": key_info,
            "box": (r0, r1, c0, c1), "base": base, "dots": dots,
            "gates": gate_records, "removed": int((mask & ~base_full).sum()),
            "shape": grid.shape}
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only, fixed-family solver; never reads files or generated examples."""
    return _decode(input_grid)[0]


def _boundary_gaps(base):
    result = []
    for r, c, dr, dc, length in ((0, 0, 0, 1, base.shape[1]),
                                 (base.shape[0] - 1, 0, 0, 1, base.shape[1]),
                                 (0, 0, 1, 0, base.shape[0]),
                                 (0, base.shape[1] - 1, 1, 0, base.shape[0])):
        for k in range(length - 4):
            yy, xx = r + k * dr, c + k * dc
            if (base[yy, xx] and base[yy + 4 * dr, xx + 4 * dc]
                    and not any(base[yy + t * dr, xx + t * dc] for t in (1, 2, 3))):
                result.append((yy, xx, dr, dc))
    return result


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _partition(rng, rows, cols, number, want_concave=False):
    """Merge coarse cells into connected rooms, all reaching the exterior."""
    for _ in range(32):
        labels = np.arange(rows * cols).reshape(rows, cols)
        if rows == cols == 3:
            rr, cc = _pick(rng, FOUR)
            labels[1, 1] = labels[1 + rr, 1 + cc]
        while len(np.unique(labels)) > number:
            edges = []
            for r in range(rows):
                for c in range(cols):
                    for dr, dc in ((1, 0), (0, 1)):
                        if r + dr < rows and c + dc < cols and labels[r, c] != labels[r + dr, c + dc]:
                            edges.append((int(labels[r, c]), int(labels[r + dr, c + dc])))
            a, b = _pick(rng, edges)
            labels[labels == b] = a
        values = sorted(map(int, np.unique(labels)))
        labels = np.array([[values.index(int(k)) for k in row] for row in labels])
        concave = False
        for k in range(number):
            cells = np.argwhere(labels == k)
            a, b, c, d = _box(cells)
            concave |= len(cells) < (b - a) * (d - c)
        if not want_concave or concave:
            return labels
    # A guaranteed L-shaped room and three singletons, up to rotation.
    assert number == 4 and (rows, cols) in ((2, 3), (3, 2), (3, 3))
    a = np.array([[0, 0, 1], [0, 2, 3]])
    if (rows, cols) == (3, 2):
        a = a.T
    elif rows == cols == 3:
        a = np.vstack((a, a[-1:]))
    return a


def _room_geometry(rng, labels):
    nr, nc = labels.shape
    ys = np.r_[0, np.cumsum(rng.integers(5, 8, nr))].astype(int)
    xs = np.r_[0, np.cumsum(rng.integers(5, 8, nc))].astype(int)
    regions = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r in range(1, len(regions) - 1):
        rs = [i for i in range(nr) if ys[i] <= r <= ys[i + 1]]
        for c in range(1, regions.shape[1] - 1):
            cs = [i for i in range(nc) if xs[i] <= c <= xs[i + 1]]
            values = {int(labels[i, j]) for i in rs for j in cs}
            if len(values) == 1:
                regions[r, c] = values.pop()
    options = {k: [] for k in map(int, np.unique(labels))}
    for i in range(nr):
        for j in range(nc):
            boundaries = []
            if i == 0:
                boundaries.append((0, int(xs[j]), 0, 1, int(xs[j + 1] - xs[j]), 1))
            if i == nr - 1:
                boundaries.append((int(ys[-1]), int(xs[j]), 0, 1, int(xs[j + 1] - xs[j]), -1))
            if j == 0:
                boundaries.append((int(ys[i]), 0, 1, 0, int(ys[i + 1] - ys[i]), 1))
            if j == nc - 1:
                boundaries.append((int(ys[i]), int(xs[-1]), 1, 0, int(ys[i + 1] - ys[i]), -1))
            for r, c, dr, dc, length, inward in boundaries:
                for offset in range(1, length - 2):
                    for endpoint in (0, 4):
                        # Coordinates store the retained anchor BEFORE the gap.
                        options[int(labels[i, j])].append((r + (offset - 1) * dr,
                            c + (offset - 1) * dc, dr, dc, inward, endpoint))
    assert all(options.values())
    return regions, options


def _motif(rng, size, count, witness=False):
    holes = np.zeros((size, size), bool)
    if witness and count == 1:
        holes[1:-1, 1:-1] = True
    elif witness and count == 2:
        holes[1, 1] = holes[2, 2] = holes[2, 3] = True
    elif count:
        parities = [p for p in (0, 1) if sum((r + c) % 2 == p
                    for r in range(1, size - 1) for c in range(1, size - 1)) >= count]
        parity = _pick(rng, parities)
        cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)
                 if (r + c) % 2 == parity]
        for i in rng.permutation(len(cells))[:count]:
            holes[cells[int(i)]] = True
        assert len(_components(holes)) == count
        target = int(rng.integers(count, (size - 2) ** 2 + 1))
        for i in rng.permutation((size - 2) ** 2):
            if int(holes.sum()) >= target:
                break
            r, c = divmod(int(i), size - 2)
            r, c = r + 1, c + 1
            if holes[r, c] or not any(holes[r + dr, c + dc] for dr, dc in FOUR):
                continue
            trial = holes.copy()
            trial[r, c] = True
            if len(_components(trial)) == count and len(_components(~trial)) == 1:
                holes = trial
    holes = np.rot90(holes, int(rng.integers(4)))
    if rng.integers(2):
        holes = holes[:, ::-1]
    assert len(_components(holes)) == count and len(_components(~holes)) == 1
    return holes.copy()


def _isolated_dots(rng, available, count):
    if count == 0:
        return np.zeros_like(available)
    cells = np.argwhere(available)
    for _ in range(12):
        chosen = np.zeros_like(available)
        for i in rng.permutation(len(cells)):
            r, c = map(int, cells[int(i)])
            if not any(_at(chosen, r + dr, c + dc) for dr, dc in FOUR):
                chosen[r, c] = True
                if int(chosen.sum()) == count:
                    return chosen
    for parity in (0, 1):
        cells = [(int(r), int(c)) for r, c in np.argwhere(available) if (r + c) % 2 == parity]
        if len(cells) >= count:
            chosen = np.zeros_like(available)
            for i in rng.permutation(len(cells))[:count]:
                chosen[cells[int(i)]] = True
            return chosen
    raise AssertionError("insufficient clue space")


def _free_origins(boxes, side):
    yy, xx = np.indices((31 - side, 31 - side))
    free = np.ones_like(yy, bool)
    for r, c, h, w in boxes:
        free &= (yy + side < r) | (r + h < yy) | (xx + side < c) | (c + w < xx)
    return np.argwhere(free)


def _place(rng, drawing, motifs):
    h, w = drawing.shape
    sizes = [len(m) for m in motifs]
    boxes = None
    for _ in range(20):
        r = _pick(rng, [0, 30 - h, int(rng.integers(31 - h))])
        c = _pick(rng, [0, 30 - w, int(rng.integers(31 - w))])
        trial = [(r, c, h, w)]
        for size in sizes:
            options = _free_origins(trial, size)
            if not len(options):
                break
            yy, xx = map(int, options[int(rng.integers(len(options)))])
            trial.append((yy, xx, size, size))
        if len(trial) == len(sizes) + 1:
            boxes = trial
            break
    if boxes is None:
        # A two-dimensional drawing is <=22 wide; three keys need <=23 rows.
        # A longer strip is <=8 high and can instead have the keys below it.
        boxes = [(0, 0, h, w)]
        offset = 0
        beside = w + 1 + max(sizes) <= 30
        for size in sizes:
            boxes.append((offset, w + 1, size, size) if beside
                         else (h + 1, offset, size, size))
            offset += size + 1
    canvas = np.zeros((30, 30), np.int8)
    for (r, c, hh, ww), a in zip(boxes, [drawing] + motifs):
        canvas[r:r + hh, c:c + ww] = a
    return canvas, boxes


def _scene(rng, kind, policy, palette):
    witness = kind == "witness"
    repeat = kind == "repeat"
    if witness:
        coarse = _pick(rng, [(2, 2), (2, 3), (3, 2)])
        number = 4
    elif repeat:
        coarse, number = _pick(rng, [(2, 3), (3, 2)]), 4
    elif kind == "test" and policy == "growth":
        coarse, number = _pick(rng, [(2, 3), (3, 2), (3, 3)]), int(rng.integers(5, 7))
    elif kind == "test" and policy == "shape":
        coarse, number = (3, 3), int(rng.integers(3, 6))
    else:
        coarse = _pick(rng, [(1, 3), (1, 4), (2, 2), (2, 3), (3, 2)])
        number = int(rng.integers(3, min(4, coarse[0] * coarse[1]) + 1))
    labels = _partition(rng, *coarse, number, want_concave=repeat)
    regions, gate_options = _room_geometry(rng, labels)
    wall, dot = palette[:2]
    key_count = 2 if witness else int(rng.integers(2, 4))
    if witness:
        counts = [1, 2]
    else:
        pool = list(range(1, 6))
        if kind == "test" and policy == "zero":
            counts = [0] + list(map(int, rng.choice(pool, key_count - 1, replace=False)))
        else:
            counts = list(map(int, rng.choice(pool, key_count, replace=False)))
    keys = dict(zip(counts, palette[2:2 + key_count]))
    order = list(map(int, rng.permutation(number)))
    if repeat:
        concave = []
        for room in range(number):
            cells = np.argwhere(regions == room)
            a, b, c, d = _box(cells)
            if len(cells) < (b - a) * (d - c):
                concave.append(room)
        assert concave
        chosen = _pick(rng, concave)
        at = order.index(chosen)
        order[0], order[at] = order[at], order[0]
    rejected, unsupported = order[-2:]
    accepted_rooms = order[:-2]
    room_counts = {}
    if witness:
        room_counts[accepted_rooms[0]], room_counts[accepted_rooms[1]] = 1, 2
    else:
        shared = _pick(rng, counts)
        for room in accepted_rooms:
            room_counts[room] = shared if repeat else _pick(rng, counts)
        if kind == "test" and policy == "zero":
            room_counts[accepted_rooms[0]] = 0
    room_counts[rejected] = _pick(rng, [k for k in range(7) if k not in keys])
    room_counts[unsupported] = _pick(rng, counts)
    if witness:
        room_counts[rejected], room_counts[unsupported] = 3, 1
    broken = regions == -1
    raw = broken.copy()
    gates, spurs = {}, []
    for room in range(number):
        r, c, dr, dc, side, endpoint = _pick(rng, gate_options[room])
        gap = tuple((r + k * dr, c + k * dc) for k in (1, 2, 3))
        for yy, xx in gap:
            broken[yy, xx] = raw[yy, xx] = False
        gates[room] = gap
        if room != unsupported:
            toward = 1 if endpoint == 0 else -1
            ar, ac = r + endpoint * dr, c + endpoint * dc
            spur = tuple((ar + k * (toward * dr + side * dc),
                          ac + k * (toward * dc + side * dr)) for k in (1, 2))
            assert all(regions[yy, xx] == room for yy, xx in spur)
            for yy, xx in spur:
                assert not raw[yy, xx]
                raw[yy, xx] = True
            spurs.append((spur, (r, c, dr, dc), side, endpoint))
    evidence = _candidates(raw)
    assert len(evidence) == number - 1
    assert np.array_equal(_clean(raw, evidence), broken)
    drawing = np.where(raw, wall, 0).astype(np.int8)
    expected = np.where(broken, wall, 0).astype(np.int8)
    for room in range(number):
        clues = _isolated_dots(rng, (regions == room) & ~raw, room_counts[room])
        drawing[clues] = dot
        if room in accepted_rooms:
            expected[regions == room] = keys[room_counts[room]]
            for yy, xx in gates[room]:
                expected[yy, xx] = wall
    motifs = []
    for k, color in keys.items():
        size = int(rng.integers(5, 8))
        holes = _motif(rng, size, k, witness)
        motifs.append(np.where(holes, 0, color).astype(np.int8))
    rng.shuffle(motifs)
    canvas, boxes = _place(rng, drawing, motifs)
    by, bx, h, w = boxes[0]
    # Occasionally bend away from the room when free canvas space permits.
    for old, (r, c, dr, dc), side, endpoint in spurs:
        if rng.random() >= .5:
            continue
        toward = 1 if endpoint == 0 else -1
        ar, ac = by + r + endpoint * dr, bx + c + endpoint * dc
        new = tuple((ar + k * (toward * dr - side * dc),
                     ac + k * (toward * dc - side * dr)) for k in (1, 2))
        if not all(0 <= yy < 30 and 0 <= xx < 30 and canvas[yy, xx] == 0 for yy, xx in new):
            continue
        trial = canvas.copy()
        for yy, xx in old:
            trial[by + yy, bx + xx] = 0
        for yy, xx in new:
            trial[yy, xx] = wall
        try:
            recovered, _ = _decode(trial)
            if np.array_equal(recovered, expected):
                canvas = trial
        except (AssertionError, IndexError):
            pass
    # Detached diagonal clutter is unrelated to any chamber or count key.
    for _ in range(int(rng.integers(3))):
        for _ in range(12):
            y, x = map(int, rng.integers(0, 30, 2))
            dy, dx = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
            cells = [(y + k * dy, x + k * dx) for k in range(int(rng.integers(2, 4)))]
            if not all(0 <= yy < 30 and 0 <= xx < 30 and canvas[yy, xx] == 0
                       and not canvas[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2].any()
                       for yy, xx in cells):
                continue
            trial = canvas.copy()
            for yy, xx in cells:
                trial[yy, xx] = wall
            try:
                recovered, _ = _decode(trial)
                if np.array_equal(recovered, expected):
                    canvas = trial
                    break
            except (AssertionError, IndexError):
                pass
    # Crop input margins independently of the output's chamber-network crop.
    y0, y1, x0, x1 = _box(np.argwhere(canvas != 0))
    y0 = max(0, y0 - int(rng.integers(3)))
    x0 = max(0, x0 - int(rng.integers(3)))
    y1 = min(30, y1 + int(rng.integers(3)))
    x1 = min(30, x1 + int(rng.integers(3)))
    inp = canvas[y0:y1, x0:x1].copy()
    turns, flip = int(rng.integers(4)), bool(rng.integers(2))
    inp, expected = np.rot90(inp, turns), np.rot90(expected, turns)
    if flip:
        inp, expected = inp[:, ::-1], expected[:, ::-1]
    predicted, info = _decode(inp)
    assert np.array_equal(predicted, expected)
    assert sum(g["accepted"] for g in info["gates"]) == number - 2
    return {"input": inp.copy(), "output": expected.copy()}


def _counts(n, t):
    for value, low, high in ((n, 2, 10), (t, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    return int(n), int(t)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    policy = _pick(rng, ["growth", "zero", "shape"])
    palette = list(map(int, rng.permutation(np.arange(1, 10))))
    steady_palette = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", n), ("test", t)):
        for i in range(number):
            kind = ("witness" if i == 0 else "repeat" if i == 1 else "train") if split == "train" else "test"
            for _ in range(48):
                roles = palette if steady_palette else list(map(int, rng.permutation(np.arange(1, 10))))
                try:
                    pair = _scene(rng, kind, policy, roles)
                except (AssertionError, IndexError):
                    continue
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_coverage(pairs, infos):
    assert any({1, 2} <= set(a["keys"]) and any(
        x["count"] == 1 and x["area"] > max(y["area"] for y in a["key_info"] if y["count"] == 2)
        for x in a["key_info"]) for a in infos), "missing area-inversion witness"
    assert any(any(x["eight_count"] < x["count"] for x in a["key_info"]) for a in infos)
    assert any(any(not g["rectangular"] and g["accepted"] for g in a["gates"]) for a in infos), "no accepted concave room"
    assert any(len([g["count"] for g in a["gates"] if g["accepted"]]) > len(
        {g["count"] for g in a["gates"] if g["accepted"]}) for a in infos), "no repeated key use"
    for mode in ("area", "eight"):
        assert any(not np.array_equal(_decode(p["input"], count_mode=mode)[0], p["output"])
                   for p in pairs), "training permits alternative cavity statistic"
    assert any(not np.array_equal(_decode(p["input"], accept_all=True)[0], p["output"])
               for p in pairs), "training permits unconditional straightening"
    assert any(not np.array_equal(_decode(p["input"], ignore_cues=True)[0], p["output"])
               for p in pairs), "training permits ignoring diagonal evidence"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, t = len(puzzle["train"]), len(puzzle["test"])
    _counts(n, t)
    seen, infos = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, a = _decode(inp)
            assert np.array_equal(expected, out), "incomplete gate, wrong fill, or wrong crop"
            gates = a["gates"]
            assert 2 <= len(gates) <= 5
            assert any(g["accepted"] for g in gates) and any(not g["accepted"] for g in gates)
            assert a["removed"] >= 2 * len(gates)
            assert set(map(int, np.unique(out))) <= {0, a["wall"]} | set(a["keys"].values())
            # Every unsupported opening remains open, even with an available key.
            all_out, all_info = _decode(inp, ignore_cues=True)
            assert len(all_info["gates"]) == len(gates) + 1
            assert not np.array_equal(all_out, out)
            assert len(_enclosed(out == a["wall"])) == sum(g["accepted"] for g in gates)
            if split == "train":
                infos.append(a)
    _training_coverage(puzzle["train"], infos)


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed sweep, counterexamples, symmetries and corruption checks."""
    fingerprints, shapes, networks, motifs, palettes = set(), set(), set(), set(), set()
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                out, a = _decode(pair["input"])
                assert np.array_equal(out, pair["output"])
                shapes.add(pair["input"].shape)
                networks.add(_key(a["base"]))
                palettes.add(tuple(sorted(map(int, np.unique(pair["input"])))))
                totals["pairs"] += 1
                totals["accepted"] += sum(g["accepted"] for g in a["gates"])
                totals["rejected"] += sum(not g["accepted"] for g in a["gates"])
                totals["concave"] += sum(not g["rectangular"] for g in a["gates"])
                totals["zero_keys"] += 0 in a["keys"]
                for info in a["key_info"]:
                    r, s, c, d = info["box"]
                    motifs.add(_key(pair["input"][r:s, c:d] == info["color"]))
        # No mutation by solve, and equivariance of crop and conditional fills.
        p = puzzle["test"][0]
        untouched = p["input"].copy()
        for turn in range(4):
            for flip in (False, True):
                inp, expected = np.rot90(untouched, turn), np.rot90(p["output"], turn)
                if flip:
                    inp, expected = inp[:, ::-1], expected[:, ::-1]
                assert np.array_equal(solve(inp), expected)
        assert np.array_equal(untouched, p["input"])
    assert len(fingerprints) == 42
    assert len(shapes) >= 25 and len(networks) >= 150 and len(motifs) >= 100 and len(palettes) >= 25
    assert totals["concave"] >= 42 and totals["zero_keys"] > 0
    a = generate_puzzle(90210, 2, 1)
    assert _fingerprint(a) == _fingerprint(generate_puzzle(90210, 2, 1))
    for n in range(2, 11):
        for t in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(19, n, t)
            assert len(p["train"]) == n and len(p["test"]) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A partial otherwise-correct label must be rejected, as must labels from
    # either parent-omitting shortcut. No large sweep payloads are retained.
    for variant in ("pixel", "all_gates", "no_cues"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
        p = bad["test"][0]
        if variant == "pixel":
            p["output"][0, 0] = (int(p["output"][0, 0]) + 1) % 10
        else:
            p["output"] = _decode(p["input"], accept_all=variant == "all_gates",
                                    ignore_cues=variant == "no_cues")[0]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted or shortcut output accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "wall_networks": len(networks), "key_masks": len(motifs), "palettes": len(palettes),
            "totals": dict(totals), "independent_semantic_review": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
