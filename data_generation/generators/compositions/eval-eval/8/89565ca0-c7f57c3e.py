"""Equal-room-count frames exchange scaffold-registered, scalable badges.

Only standard-library modules and NumPy are used. The decoder reads all roles,
counts, templates, partners, anchors and scales from the supplied grid.
"""
from collections import defaultdict
from itertools import permutations
from math import gcd

import numpy as np


BACKGROUND = 0
REPAIR = 8


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _bounds(points):
    ys, xs = zip(*points)
    return min(ys), min(xs), max(ys), max(xs)


def _walls(shape, rooms):
    result = np.zeros(shape, dtype=bool)
    for t, l, b, r in rooms:
        result[t, l:r + 1] = result[b, l:r + 1] = True
        result[t:b + 1, l] = result[t:b + 1, r] = True
    return result


def _rooms(mask):
    """Flood-fill elementary rooms; a padded exterior handles edge contact."""
    free = np.pad(~mask, 1, constant_values=True)
    rooms = []
    for points in _components(free):
        t, l, b, r = _bounds(points)
        if t == 0 or l == 0 or b == free.shape[0] - 1 or r == free.shape[1] - 1:
            continue
        assert len(points) == (b - t + 1) * (r - l + 1), "nonrectangular room"
        rooms.append((t - 1, l - 1, b - 1, r - 1))
    return rooms


def _euler_count(mask):
    # All frame walls form one connected pixel graph. Filled unit faces contain
    # no room interior and must be removed from the cycle rank.
    edges = int((mask[:-1] & mask[1:]).sum())
    edges += int((mask[:, :-1] & mask[:, 1:]).sum())
    units = int((mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).sum())
    return edges - int(mask.sum()) + 1 - units


def _signature(mapping):
    return tuple(sorted((int(y), int(x), int(c)) for (y, x), c in mapping.items()))


def _primitive(crop):
    scale = 0
    for array in (crop, crop.T):
        for row in array:
            cuts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            for length in np.diff(cuts):
                scale = gcd(scale, int(length))
    assert scale >= 1
    reduced = crop[::scale, ::scale]
    assert np.array_equal(crop, np.repeat(np.repeat(reduced, scale, 0), scale, 1))
    return scale, reduced


def _separated(a, b):
    # Inclusive boxes; require at least one background row or column.
    at, al, ab, ar = a
    bt, bl, bb, br = b
    return ab + 1 < bt or bb + 1 < at or ar + 1 < bl or br + 1 < al


def _render(grid, template, anchor, scale):
    for (y, x), color in template.items():
        yy, xx = anchor[0] + y * scale, anchor[1] + x * scale
        grid[yy:yy + scale, xx:xx + scale] = color


def _canonical_read(grid, groups, wall):
    engines = []
    for group in groups:
        visible = [p for p in group if grid[p] == wall]
        assert len(visible) >= 8
        t, l, b, r = _bounds(visible)
        assert b - t >= 2 and r - l >= 2
        crop = grid[t:b + 1, l:r + 1]
        assert np.all(np.isin(crop, [0, wall, REPAIR]))
        mask = (crop == wall) | (crop == REPAIR)
        assert mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all()
        rooms = _rooms(mask)
        assert rooms
        boundaries = [(a - 1, c - 1, d + 1, e + 1) for a, c, d, e in rooms]
        assert np.array_equal(mask, _walls(mask.shape, boundaries)), "unexplained wall pixels"
        assert _euler_count(mask) == len(rooms)
        motif = {p for p in group if not (t <= p[0] <= b and l <= p[1] <= r)}
        assert motif and all(y < t for y, x in motif), "badge must be on a common side"
        assert all(grid[p] not in (wall, REPAIR) for p in motif)
        mt, ml, mb, mr = _bounds(motif)
        assert mb + 1 == t, "badge stem must touch the frame"
        piece = grid[mt:mb + 1, ml:mr + 1]
        assert int(np.count_nonzero(piece)) == len(motif), "interleaved badges"
        s, reduced = _primitive(piece)
        raw = {(int(y), int(x)): int(reduced[y, x]) for y, x in np.argwhere(reduced != 0)}
        engines.append({"box": (t, l, b, r), "mask": mask, "rooms": rooms,
                        "count": len(rooms), "scale": s, "origin": (mt, ml), "raw": raw})
    assert len(engines) >= 2
    common_colors = set.intersection(*(set(e["raw"].values()) for e in engines))
    assert len(common_colors) == 1, "one shared scaffold color is required"
    carrier = common_colors.pop()
    signatures, scaffolds = set(), set()
    for e in engines:
        raw, s = e["raw"], e["scale"]
        cy, cx, _, _ = _bounds([p for p, c in raw.items() if c == carrier])
        template = {(y - cy, x - cx): c for (y, x), c in raw.items()}
        e["template"] = template
        e["anchor"] = (e["origin"][0] + cy * s, e["origin"][1] + cx * s)
        e["signature"] = _signature(template)
        signatures.add(e["signature"])
        scaffolds.add(tuple(sorted(p for p, c in template.items() if c == carrier)))
    assert len(signatures) == 2 and len(scaffolds) == 1, "exactly two registered states"
    templates = [{(y, x): c for y, x, c in sig} for sig in sorted(signatures)]
    scaffold = set(next(iter(scaffolds)))
    assert len(scaffold) >= 5
    # No hidden dictionary: both state designs and all recipient scales are local.
    classes = defaultdict(list)
    for e in engines:
        e["state"] = sorted(signatures).index(e["signature"])
        classes[e["count"]].append(e)
    assert all(len(items) <= 2 for items in classes.values()), "count correspondence is ambiguous"
    union = set(templates[0]) | set(templates[1])
    uy, ux, vy, vx = _bounds(union)
    reservations = []
    expected = grid.copy()
    for items in classes.values():
        for index, e in enumerate(items):
            mate = items[1 - index] if len(items) == 2 else e
            e["received_state"] = mate["state"]
            s, (ay, ax) = e["scale"], e["anchor"]
            top, left, bottom, right = e["box"]
            reserve = (min(top, ay + uy * s), min(left, ax + ux * s),
                       max(bottom, ay + (vy + 1) * s - 1), max(right, ax + (vx + 1) * s - 1))
            assert 0 <= reserve[0] <= reserve[2] < grid.shape[0]
            assert 0 <= reserve[1] <= reserve[3] < grid.shape[1]
            assert ay + (vy + 1) * s <= top, "replacement reaches the frame"
            assert all(_separated(reserve, other) for other in reservations), "replacement collision"
            reservations.append(reserve)
            e["reserve"] = reserve
            for y, x in union:
                yy, xx = ay + y * s, ax + x * s
                expected[yy:yy + s, xx:xx + s] = 0
            _render(expected, mate["template"], (ay, ax), s)
    return {"expected": expected, "engines": engines, "wall": wall,
            "carrier": carrier, "templates": templates, "classes": classes}


def _read(input_grid):
    g = _grid(input_grid)
    groups = _components(g != 0)
    assert 2 <= len(groups) <= 6
    candidates = set.intersection(*({int(g[p]) for p in group} for group in groups)) - {0, REPAIR}
    answers = []
    for rotation in range(4):
        canonical = np.rot90(g, rotation)
        local_groups = _components(canonical != 0)
        for wall in sorted(candidates):
            try:
                info = _canonical_read(canonical, local_groups, wall)
            except (AssertionError, ValueError):
                continue
            info["rotation"] = rotation
            info["canonical"] = canonical
            info["expected"] = np.rot90(info["expected"], -rotation).copy()
            answers.append(info)
    assert len(answers) == 1, "no unique frame/scaffold interpretation"
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive a complete answer from the input alone."""
    return _read(input_grid)["expected"].copy()


def _capacity(room):
    t, l, b, r = room
    return ((b - t) // 2) * ((r - l) // 2)


def _split_rooms(rng, height, width, count, style):
    outer = (0, 0, height - 1, width - 1)
    if _capacity(outer) < count:
        return None
    if style in ("strip", "grid"):
        factors = [(nr, count // nr) for nr in range(1, count + 1)
                   if count % nr == 0 and 2 * nr < height and 2 * (count // nr) < width
                   and (style == "grid" or nr == 1 or nr == count)]
        if not factors:
            return None
        nr, nc = factors[int(rng.integers(len(factors)))]
        def cuts(n, span):
            gaps = [2] * n
            for _ in range(span - 2 * n):
                gaps[int(rng.integers(n))] += 1
            return [0] + list(map(int, np.cumsum(gaps)))
        ys, xs = cuts(nr, height - 1), cuts(nc, width - 1)
        rooms = [(ys[i], xs[j], ys[i + 1], xs[j + 1]) for i in range(nr) for j in range(nc)]
    else:
        rooms, first_axis = [outer], None
        for step in range(count - 1):
            choices = []
            for i, (t, l, b, r) in enumerate(rooms):
                for axis, lo, hi in ((0, t, b), (1, l, r)):
                    if style == "partial" and count >= 3 and step == 1 and axis == first_axis:
                        continue
                    for cut in range(lo + 2, hi - 1):
                        halves = [(t, l, cut, r), (cut, l, b, r)] if axis == 0 else [(t, l, b, cut), (t, cut, b, r)]
                        capacity = sum(_capacity(room) for j, room in enumerate(rooms) if j != i)
                        if capacity + sum(_capacity(room) for room in halves) >= count:
                            choices.append((i, axis, halves))
            if not choices:
                return None
            i, axis, halves = choices[int(rng.integers(len(choices)))]
            if step == 0:
                first_axis = axis
            rooms[i:i + 1] = halves
    mask = _walls((height, width), rooms)
    if style == "partial" and count >= 3:
        hs = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
        vs = np.flatnonzero(np.any(mask[:-1] & mask[1:], axis=0))
        if (len(hs) - 1) * (len(vs) - 1) == count:
            return None
    assert len(_rooms(mask)) == count and _euler_count(mask) == count
    return mask


def _design(rng, change):
    w = int(rng.integers(3, 5))
    cx, row = int(rng.integers(1, w - 1)), int(rng.integers(3))
    core = {(y, cx) for y in range(4)} | {(row, x) for x in range(w)}
    # Grow short asymmetric branches while preserving a one-cell docking stem.
    for _ in range(int(rng.integers(3))):
        frontier = {(y, x) for y in range(3) for x in range(w) if (y, x) not in core
                    and sum((y + dy, x + dx) in core for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))) == 1}
        if frontier:
            choices = sorted(frontier)
            core.add(choices[int(rng.integers(len(choices)))])
    def decoration(side):
        if side == 0:
            root, direction = (0, cx), (-1, 0)
        elif side == 1:
            root, direction = (row, 0), (0, -1)
        else:
            root, direction = (row, w - 1), (0, 1)
        depth = int(rng.integers(1, 3)) if side == 0 else 1
        points = {(root[0] + direction[0] * d, root[1] + direction[1] * d) for d in range(1, depth + 1)}
        if side == 0 and rng.random() < .6:
            points.add((-depth, cx + int(rng.choice([-1, 1]))))
        return points
    # Relocating states use different exterior sides, hence different box origins.
    side_a = int(rng.integers(3))
    side_b = int(rng.choice([s for s in range(3) if s != side_a]))
    da = decoration(side_a)
    db = da.copy() if change == "recolor" else decoration(side_b)
    inner = sorted({(y, x) for y in range(3) for x in range(w) if (y, x) not in core
                    and any((y + dy, x + dx) in core for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))})
    rng.shuffle(inner)
    internal = set(inner[:int(rng.integers(1, min(2, len(inner)) + 1))]) if inner and change == "combined" else set()
    palette = list(map(int, rng.permutation([1, 2, 3, 4, 5, 6, 7, 9])))
    wall, carrier, a1, a2, b1, b2 = palette[:6]
    maps = []
    for deco, c1, c2 in ((da, a1, a2), (db, b1, b2)):
        template = {p: carrier for p in core}
        template.update({p: c1 for p in internal})
        template.update({p: c2 if internal or j % 2 else c1 for j, p in enumerate(sorted(deco))})
        maps.append(template)
    return wall, carrier, maps


def _shelf(sizes, limit, order):
    positions, y, x, shelf = [None] * len(sizes), 0, 0, 0
    for i in order:
        h, w = sizes[i]
        if w > limit:
            return None
        if x and x + w > limit:
            y, x, shelf = y + shelf + 1, 0, 0
        if y + h > 30:
            return None
        positions[i] = (y, x, y + h - 1, x + w - 1)
        x += w + 1
        shelf = max(shelf, h)
    return positions


def _pack(rng, sizes):
    n = len(sizes)
    orders = list(permutations(range(n))) if n <= 4 else [tuple(rng.permutation(n)) for _ in range(24)]
    orders.append(tuple(sorted(range(n), key=lambda i: sizes[i], reverse=True)))
    options = []
    for transpose in (False, True):
        dims = [(w, h) for h, w in sizes] if transpose else sizes
        for limit in range(max(w for h, w in dims), 31):
            for order in orders:
                rects = _shelf(dims, limit, order)
                if rects is not None:
                    if transpose:
                        rects = [(l, t, r, b) for t, l, b, r in rects]
                    options.append(rects)
    if not options:
        return None
    rects = options[int(rng.integers(len(options)))]
    h0, w0 = max(r[2] for r in rects) + 1, max(r[3] for r in rects) + 1
    h, w = int(rng.integers(h0, min(30, h0 + 5) + 1)), int(rng.integers(w0, min(30, w0 + 5) + 1))
    dy, dx = int(rng.integers(h - h0 + 1)), int(rng.integers(w - w0 + 1))
    rects = [(t + dy, l + dx, b + dy, r + dx) for t, l, b, r in rects]
    for _ in range(35):
        i = int(rng.integers(n))
        rh, rw = sizes[i]
        t, l = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
        proposed = (t, l, t + rh - 1, l + rw - 1)
        if all(_separated(proposed, old) for j, old in enumerate(rects) if j != i):
            rects[i] = proposed
    return (h, w), rects


def _network(rng, count, scale, union_shape, style, minimal):
    uh, uw = union_shape
    choices = []
    for h in range(3, 11):
        for w in range(max(3, uw * scale), min(24, max(11, uw * scale + 5, 2 * count + 1)) + 1):
            if ((h - 1) // 2) * ((w - 1) // 2) < count:
                continue
            if style == "partial" and count >= 3 and min(h, w) < 5:
                continue
            if style == "strip" and not (2 * count < h or 2 * count < w):
                continue
            choices.append((h, w))
    if minimal:
        choices.sort(key=lambda hw: ((uh * scale + hw[0]) * hw[1], hw))
    else:
        rng.shuffle(choices)
        # Keep several engines readable and leave room for state substitution.
        choices.sort(key=lambda hw: max(0, (uh * scale + hw[0]) * hw[1] - 180))
    for h, w in choices[:30]:
        for _ in range(8):
            mask = _split_rooms(rng, h, w, count, style)
            if mask is not None:
                return mask
    return None


def _pair(rng, counts, states, scales, styles, change, area_witness=False):
    wall, carrier, templates = _design(rng, change)
    union = set(templates[0]) | set(templates[1])
    uy, ux, vy, vx = _bounds(union)
    uh, uw = vy - uy + 1, vx - ux + 1
    for attempt in range(24):
        masks = [_network(rng, k, s, (uh, uw), style, attempt >= 8)
                 for k, s, style in zip(counts, scales, styles)]
        if any(mask is None for mask in masks):
            continue
        if area_witness and masks[0].size == masks[1].size:
            continue
        sizes = [(uh * s + mask.shape[0], mask.shape[1]) for s, mask in zip(scales, masks)]
        packed = _pack(rng, sizes)
        if packed is None:
            continue
        shape, rects = packed
        inp, out = np.zeros(shape, dtype=int), np.zeros(shape, dtype=int)
        grouped = defaultdict(list)
        for i, k in enumerate(counts):
            grouped[k].append(i)
        for i, (s, state, mask, rect) in enumerate(zip(scales, states, masks, rects)):
            t, l, b, r = rect
            slack = mask.shape[1] - uw * s
            anchor = (t - uy * s, l - ux * s + int(rng.integers(slack + 1)))
            fy = t + uh * s
            for grid in (inp, out):
                region = grid[fy:fy + mask.shape[0], l:r + 1]
                region[mask] = wall
            sites = [(int(y), int(x)) for y, x in np.argwhere(mask)
                     if (y, x) not in ((0, 0), (0, mask.shape[1] - 1), (mask.shape[0] - 1, 0),
                                      (mask.shape[0] - 1, mask.shape[1] - 1))]
            amount = int(rng.integers(0, max(2, len(sites) // 10 + 1)))
            if sites and amount:
                for index in rng.choice(len(sites), size=min(amount, len(sites)), replace=False):
                    y, x = sites[int(index)]
                    inp[fy + y, l + x] = out[fy + y, l + x] = REPAIR
            _render(inp, templates[state], anchor, s)
            members = grouped[counts[i]]
            mate = members[1 - members.index(i)] if len(members) == 2 else i
            _render(out, templates[states[mate]], anchor, s)
        rotation = int(rng.integers(4))
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        if rng.random() < .5:
            inp, out = np.fliplr(inp).copy(), np.fliplr(out).copy()
        info = _read(inp)
        assert np.array_equal(info["expected"], out)
        return {"input": inp, "output": out}
    raise AssertionError("bounded room/badge packing exhausted")


def _counts_check(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _badge_geometry(info):
    """Badge vocabulary without color names, state order, or mirror direction."""
    carrier = info["carrier"]
    scaffold = {p for p, c in info["templates"][0].items() if c == carrier}
    variants = []
    for sign in (1, -1):
        transformed = {(y, sign * x) for y, x in scaffold}
        oy, ox, _, _ = _bounds(transformed)
        core = tuple(sorted((y - oy, x - ox) for y, x in transformed))
        states = [tuple(sorted((y - oy, sign * x - ox) for (y, x), c in t.items() if c != carrier))
                  for t in info["templates"]]
        variants.append((core, tuple(sorted(states))))
    return min(variants)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts_check(num_train, num_test)
    rng = np.random.default_rng(seed)
    ceiling, regime = int(rng.integers(4, 6)), int(rng.integers(3))
    puzzle, seen, train_badges = {"train": [], "test": []}, set(), set()
    for split, number in (("train", nt), ("test", ne)):
        for index in range(number):
            if split == "train" and index == 0:
                a, b = int(rng.integers(3, 5)), int(rng.integers(1, 3))
                counts, states, scales = [a, a, b], [0, 1, int(rng.integers(2))], [1, 2, 1]
                styles, change = ["partial", "strip", "strip"], "combined"
            elif split == "train" and index == 1:
                a, b = int(rng.integers(1, 3)), ceiling
                c = int(rng.integers(2))
                counts, states, scales = [a, a, b, b], [0, 1, c, c], [1, 1, 1, 1]
                styles, change = ["strip", "strip", "partial", "strip"], "relocate"
            elif split == "test" and regime == 1:
                a = int(rng.integers(1, ceiling + 1))
                b = int(rng.choice([k for k in range(1, ceiling + 1) if k != a]))
                counts, states, scales = [a, a, b], [0, 1, int(rng.integers(2))], [3, 1, 1]
                styles, change = ["strip", "partial" if a >= 3 else "strip", "strip"], "combined"
            else:
                high = ceiling + int(rng.integers(1, 3)) if split == "test" and regime == 0 else ceiling
                a = high if high > ceiling else int(rng.integers(1, high + 1))
                b = int(rng.choice([k for k in range(1, high + 1) if k != a]))
                n = int(rng.integers(3, 7)) if high <= ceiling else int(rng.integers(3, 5))
                counts, states = [a, a, b], [0, 1, int(rng.integers(2))]
                if n >= 4:
                    counts.append(b)
                    states.append(states[-1])  # homogeneous count class
                available = [k for k in range(1, high + 1) if k not in (a, b)]
                rng.shuffle(available)
                counts += available[:n - 4] if n > 4 else []
                states += [int(rng.integers(2)) for _ in range(len(counts) - len(states))]
                scales = [1] * len(counts)
                if len(counts) <= 4 and rng.random() < .45:
                    scales[int(rng.integers(2))] = 2
                styles = ["partial" if k >= 3 and rng.random() < .55 else str(rng.choice(["strip", "grid", "mixed"])) for k in counts]
                change = "combined" if split == "test" else str(rng.choice(["combined", "relocate", "recolor"]))
            for _ in range(40):
                pair = _pair(rng, counts, states, scales, styles, change, split == "train" and index == 0)
                key = _key(pair["input"])
                vocabulary = _badge_geometry(_read(pair["input"])) if regime == 2 else None
                if split == "test" and regime == 2 and vocabulary in train_badges:
                    continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and regime == 2:
                        train_badges.add(vocabulary)
                    break
            else:
                raise AssertionError("could not produce a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _lattice_count(e):
    mask = e["mask"]
    hs = np.count_nonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
    vs = np.count_nonzero(np.any(mask[:-1] & mask[1:], axis=0))
    return int((hs - 1) * (vs - 1))


def _alternative(info, mode):
    canonical = info["canonical"]
    result = canonical.copy()
    templates = info["templates"]
    union = set(templates[0]) | set(templates[1])
    classes = defaultdict(list)
    for e in info["engines"]:
        k = _lattice_count(e) if mode == "lattice" else e["mask"].size
        classes[k].append(e)
    for e in info["engines"]:
        if mode == "toggle":
            target = templates[1 - e["state"]]
        else:
            k = _lattice_count(e) if mode == "lattice" else e["mask"].size
            members = classes[k]
            mate = members[1 - members.index(e)] if len(members) == 2 else e
            target = mate["template"]
        s, (ay, ax) = e["scale"], e["anchor"]
        for y, x in union:
            result[ay + y * s:ay + (y + 1) * s, ax + x * s:ax + (x + 1) * s] = 0
        _render(result, target, (ay, ax), s)
    return np.rot90(result, -info["rotation"]).copy()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts_check(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            info = _read(g)
            assert np.array_equal(info["expected"], out), "wrong routed badge exchange"
            assert np.array_equal(g == info["wall"], out == info["wall"]), "frame changed"
            assert np.array_equal(g == REPAIR, out == REPAIR), "repair markers changed"
            assert np.array_equal(g == info["carrier"], out == info["carrier"]), "scaffold moved or rescaled"
            assert not np.array_equal(g, out), "no exchange is exposed"
            for items in info["classes"].values():
                assert sorted(e["state"] for e in items) == sorted(e["received_state"] for e in items)
            assert np.array_equal(solve(out), g), "exchange is not involutive"
            infos[split].append(info)
    train = infos["train"]
    assert any(len(items) == 1 for i in train for items in i["classes"].values()), "missing singleton witness"
    assert any(len(items) == 2 and items[0]["state"] == items[1]["state"] for i in train for items in i["classes"].values()), "missing same-state witness"
    assert any(len(items) == 2 and items[0]["state"] != items[1]["state"] and items[0]["scale"] != items[1]["scale"]
               for i in train for items in i["classes"].values()), "missing scale-aware exchange"
    assert any(_bounds(i["templates"][0])[:2] != _bounds(i["templates"][1])[:2]
               and any(len(items) == 2 and items[0]["state"] != items[1]["state"] for items in i["classes"].values())
               for i in train), "bounding-box registration is unchallenged"
    for shortcut in ("toggle", "lattice", "area"):
        assert any(not np.array_equal(_alternative(i, shortcut), p["output"]) for i, p in zip(train, puzzle["train"])), f"{shortcut} shortcut unchallenged"


def self_test():
    """Bounded structural, semantic, metamorphic and diversity checks on 42 seeds."""
    fingerprints, shapes, designs, palettes, counts, schedules, room_shapes = set(), set(), set(), set(), set(), set(), set()
    scales, walls, carriers, orientations = set(), set(), set(), set()
    partial = damage = singleton = homogeneous = unequal = edge = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = tuple(_key(p[r]) for s in puzzle for p in puzzle[s] for r in ("input", "output"))
        fingerprints.add(fingerprint)
        if seed < 3:
            again = generate_puzzle(seed)
            assert fingerprint == tuple(_key(p[r]) for s in again for p in again[s] for r in ("input", "output"))
        for split in puzzle:
            for p in puzzle[split]:
                g, out = p["input"], p["output"]
                info = _read(g)
                shapes.add(g.shape)
                palettes.add(tuple(map(int, np.unique(g))))
                # Remove literal colors from template measurements.
                role_map = {info["carrier"]: 1}
                normalized = []
                for template in info["templates"]:
                    normalized.append(tuple((y, x, role_map.setdefault(c, len(role_map) + 1)) for y, x, c in _signature(template)))
                designs.add(tuple(normalized))
                engines = info["engines"]
                counts.add(tuple(sorted(e["count"] for e in engines)))
                schedules.add(tuple(sorted(e["scale"] for e in engines)))
                scales.update(e["scale"] for e in engines)
                walls.add(info["wall"])
                carriers.add(info["carrier"])
                orientations.add(info["rotation"])
                partial += sum(_lattice_count(e) != e["count"] for e in engines)
                damage += int(np.any(g == REPAIR))
                singleton += sum(len(items) == 1 for items in info["classes"].values())
                homogeneous += sum(len(items) == 2 and items[0]["state"] == items[1]["state"] for items in info["classes"].values())
                unequal += sum(len({(b - t, r - l) for t, l, b, r in e["rooms"]}) > 1 for e in engines)
                for e in engines:
                    room_shapes.update((b - t + 1, r - l + 1) for t, l, b, r in e["rooms"])
                edge += int(any(np.any(a[0] != 0) or np.any(a[-1] != 0) or np.any(a[:, 0] != 0) or np.any(a[:, -1] != 0) for a in (g, out)))
                if seed < 2:
                    for k in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(g, k)), np.rot90(out, k))
                    assert np.array_equal(solve(np.fliplr(g)), np.fliplr(out))
                    perm = np.arange(10)
                    perm[[1, 2, 3, 4, 5, 6, 7, 9]] = [9, 7, 6, 5, 4, 3, 2, 1]
                    assert np.array_equal(solve(perm[g]), perm[out])
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(designs) >= 50 and len(palettes) >= 30
    assert len(counts) >= 20 and len(schedules) >= 6 and scales == {1, 2, 3}
    assert len(walls) == len(carriers) == 8 and orientations == {0, 1, 2, 3}
    assert partial >= 70 and damage >= 100 and singleton >= 60 and homogeneous >= 50
    assert unequal >= 80 and edge >= 20 and len(room_shapes) >= 20
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(11, 2, 1)
    point = tuple(np.argwhere(bad["test"][0]["input"] != bad["test"][0]["output"])[0])
    bad["test"][0]["output"][point] = (bad["test"][0]["output"][point] + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds and all legal count combinations")
