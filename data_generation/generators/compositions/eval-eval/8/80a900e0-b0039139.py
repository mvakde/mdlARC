"""Ray-selected beadwork. Standard library and NumPy only; no file I/O.

The input is its own program: cap endpoints select and order whole objects,
which supply the colors of diagonal copies of the diamond's interior stencil.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
HOST_SEEDS = (0,682127143,915947450,159455195,852706662,196922839,444569184,
              976660279,994270462,562614817,259135536,178675381,972425192,
              231681471,647322752,305809261,36027037,934270910,303052783,
              979242942,366987499,422929264,632313020,733128611,789911588,
              831492906,226030323,642854300,761930612,388391400,560750055,
              275790137,696729100,891312082,996571153,202568379,88642285,
              397308000,636707564,793078667,858227053,354805027)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_),
             "grid cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(points, diagonal=False):
    remaining = set(points)
    result = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _crop_points(points):
    p = np.asarray(sorted(points), dtype=int)
    lo, hi = p.min(axis=0), p.max(axis=0)
    mask = np.zeros(tuple(hi - lo + 1), dtype=bool)
    mask[tuple((p - lo).T)] = True
    return mask


def _outline(center, radius):
    r, c = center
    return {(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius}


def _cap(center, radius, direction):
    r, c = center
    vr, vc = direction
    return [(r + vr * (k + 1), c + vc * (radius - k + 1))
            for k in range(radius + 1)]


def _contacts(points, center, radius, direction):
    """Closed-form first contact, rail set, and hit-pixel count."""
    vr, vc = direction
    hits = []
    rails = set()
    for r, c in points:
        u, v = (r - center[0]) * vr, (c - center[1]) * vc
        if abs(u - v) == radius and min(u, v) >= 2:
            hits.append(min(u, v) - 1)
            rails.add(1 if u > v else -1)
    return (min(hits) if hits else None), rails, len(hits)


def _decode(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    parity = np.indices(g.shape).sum(axis=0) % 2
    background = []
    for p in (0, 1):
        counts = np.bincount(g[parity == p], minlength=10)
        color = int(counts.argmax())
        _require(counts[color] > np.count_nonzero(parity == p) / 2, "checkerboard ambiguous")
        background.append(color)
    _require(background[0] != background[1], "need two background colors")
    base = np.asarray(background, dtype=np.int8)[parity]
    foreground = set(map(tuple, np.argwhere(g != base)))
    candidates = []
    for radius in (3, 4):
        for r in range(radius + 1, h - radius - 1):
            for c in range(radius + 1, w - radius - 1):
                center = (r, c)
                color = int(g[r - radius, c])
                if color in background:
                    continue
                outline = _outline(center, radius)
                if all(int(g[p]) == color for p in outline):
                    candidates.append((center, radius, color, outline))
    _require(len(candidates) == 1, "need exactly one complete diamond")
    center, radius, structural, outline = candidates[0]
    caps = []
    for direction in DIRECTIONS:
        cells = _cap(center, radius, direction)
        if all(0 <= r < h and 0 <= c < w for r, c in cells):
            colors = {int(g[p]) for p in cells}
            if len(colors) == 1 and not colors.intersection((*background, structural)):
                caps.append((direction, colors.pop(), set(cells)))
    _require(len(caps) == 1, "need exactly one complete cap")
    direction, cap_color, cap_cells = caps[0]
    interior = {p for p in foreground
                if abs(p[0] - center[0]) + abs(p[1] - center[1]) < radius}
    _require(bool(interior) and len(_components(interior)) == 1, "invalid interior stencil")
    motif_colors = {int(g[p]) for p in interior}
    _require(len(motif_colors) == 1, "stencil must be monochrome")
    motif_color = motif_colors.pop()
    _require(motif_color not in (*background, structural, cap_color), "ambiguous stencil role")
    motif = _crop_points(interior)
    counter_points = foreground - outline - cap_cells - interior
    objects = _components(counter_points)
    _require(len(objects) == len(_components(counter_points, True)), "counter objects touch diagonally")
    selected, ignored = [], []
    for points in objects:
        colors = {int(g[p]) for p in points}
        _require(len(colors) == 1, "counter objects must be monochrome")
        color = colors.pop()
        _require(color not in (*background, structural, cap_color, motif_color), "ambiguous counter role")
        distance, rails, pixels = _contacts(points, center, radius, direction)
        record = dict(points=points, color=color, distance=distance, rails=rails, pixels=pixels)
        (ignored if distance is None else selected).append(record)
    _require(bool(selected), "no selected objects")
    selected.sort(key=lambda o: o["distance"])
    _require(len({o["distance"] for o in selected}) == len(selected), "first-contact tie")
    n = len(selected)
    _require(all(n * (d + 1) - 1 <= 30 for d in motif.shape), "assembled output too large")
    return dict(grid=g, background=tuple(background), center=center, radius=radius,
                structural=structural, direction=direction, cap_color=cap_color,
                motif_color=motif_color, motif=motif, selected=selected, ignored=ignored)


def _assemble(motif, colors, cap_color, direction):
    th, tw = motif.shape
    n = len(colors)
    out = np.full((n * (th + 1) - 1, n * (tw + 1) - 1), cap_color, dtype=np.int8)
    for i, color in enumerate(colors):
        ri = i if direction[0] > 0 else n - 1 - i
        ci = i if direction[1] > 0 else n - 1 - i
        tile = out[ri * (th + 1):ri * (th + 1) + th,
                   ci * (tw + 1):ci * (tw + 1) + tw]
        tile[motif] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    return _assemble(d["motif"], [o["color"] for o in d["selected"]],
                     d["cap_color"], d["direction"])


def _grow(allowed, start, size, rng):
    points, frontier = set(), {start}
    for _ in range(size):
        if not frontier:
            return None
        choices = sorted(frontier)
        p = choices[int(rng.integers(len(choices)))]
        frontier.remove(p)
        points.add(p)
        r, c = p
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in allowed and q not in points:
                frontier.add(q)
    return points


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask[:, ::-1] if j >= 4 else mask, j % 4))
               for j in range(1, 8))


def _motif(rng, center, radius, n, witness):
    if witness:
        for _ in range(80):
            allowed = {(r, c) for r in range(3) for c in range(3)}
            points = _grow(allowed, (1, 1), int(rng.integers(4, 9)), rng)
            mask = _crop_points(points)
            if mask.shape == (3, 3) and _asymmetric(mask):
                break
        else:
            mask = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
        r0, c0 = center[0] - 1, center[1] - 1
        return {(r0 + r, c0 + c) for r, c in np.argwhere(mask)}, mask
    allowed = {(r, c) for r in range(center[0] - radius + 1, center[0] + radius)
               for c in range(center[1] - radius + 1, center[1] + radius)
               if abs(r - center[0]) + abs(c - center[1]) < radius}
    sites = sorted(allowed)
    for _ in range(100):
        start = sites[int(rng.integers(len(sites)))]
        points = _grow(allowed, start, int(rng.integers(1, len(allowed) + 1)), rng)
        mask = _crop_points(points)
        if all(n * (d + 1) - 1 <= 30 for d in mask.shape):
            return points, mask
    return {center}, np.ones((1, 1), dtype=bool)


def _flip(a, direction):
    if direction[0] < 0:
        a = a[::-1]
    if direction[1] < 0:
        a = a[:, ::-1]
    return a.copy()


def _sample_pair(rng, n, radius, direction, witness=0):
    """Construct in a SE frame, then reflect the entire scene and answer.

    At most 100 scene attempts, each with at most 120 distractor attempts.
    Only the returned pair is retained, with no scratch files or retry logs.
    """
    for attempt in range(100):
        palette = list(map(int, rng.permutation(10)))
        background, structural, cap_color, motif_color = palette[:2], *palette[2:5]
        bead_bank = palette[5:]
        center = (radius + 1 + int(rng.integers(2)), radius + 1 + int(rng.integers(2)))
        minimum = [max(16, c + radius + 3 + 4 * (n - 1)) for c in center]
        if max(minimum) > 30:
            continue
        h, w = (int(rng.integers(v, 31)) for v in minimum)
        max_t = min(h - center[0] - radius - 3, w - center[1] - radius - 3)
        slack = max_t - 1 - 4 * (n - 1)
        if slack < 0:
            continue
        extras = rng.multinomial(slack, np.full(n + 1, 1 / (n + 1)))
        ts, t = [], 1 + int(extras[0])
        for i in range(n):
            ts.append(t)
            t += 4 + int(extras[i + 1])
        motif_points, motif = _motif(rng, center, radius, n, bool(witness))
        fixed = _outline(center, radius) | set(_cap(center, radius, (1, 1))) | motif_points
        occupied, objects = set(), []

        def free(points):
            return (points is not None and all(0 <= r < h and 0 <= c < w for r, c in points)
                    and not points.intersection(fixed)
                    and all((r + dr, c + dc) not in occupied for r, c in points
                            for dr in (-1, 0, 1) for dc in (-1, 0, 1)))

        colors = [int(rng.choice(bead_bank)) for _ in range(n)]
        if n > 1:
            colors[-1] = int(rng.choice([c for c in bead_bank if c != colors[0]]))
        if witness == 2:
            low, middle, high = sorted(map(int, rng.choice(bead_bank, 3, replace=False)))
            orders = [(low, high, middle), (middle, low, high),
                      (high, low, middle), (middle, high, low)]
            colors = list(orders[int(rng.integers(len(orders)))])
        bridge_index = 0 if witness == 1 else (int(rng.integers(n)) if not witness and rng.random() < .35 else -1)
        if witness == 1:
            sizes = [2 * radius + 1, 5]
        elif witness == 2:
            total = 2 * radius + 6
            sizes = [total // 3 + int(i < total % 3) for i in range(3)]
        else:
            sizes = list(map(int, rng.integers(1, 10, n)))
        good = True
        for i, t in enumerate(ts):
            r, c = center[0] + 1 + t, center[1] + radius + 1 + t
            if i == bridge_index:
                # One connected L joins the two rails at equal distance.
                points = {(r, c - k) for k in range(radius + 1)} | {
                    (r + k, c - radius) for k in range(radius + 1)}
                if rng.random() < .5:
                    # The other L around the same anti-diagonal endpoints.
                    points = {(r + k, c) for k in range(radius + 1)} | {
                        (r + radius, c - k) for k in range(radius + 1)}
            else:
                swap_rail = (i % 2 == 0) if witness == 2 else (rng.random() < .5)
                if swap_rail:
                    r, c = center[0] + radius + 1 + t, center[1] + 1 + t
                allowed = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                points = _grow(allowed, (r, c), sizes[i], rng)
            if not free(points):
                good = False
                break
            occupied.update(points)
            objects.append((points, colors[i]))
        if not good:
            continue
        # A component on an interior cap lane, but on neither endpoint rail.
        inside_candidates = list(range(3, max_t + radius))
        rng.shuffle(inside_candidates)
        for t in inside_candidates:
            r, c = center[0] + t, center[1] + t
            points = {(r, c), (r + 1, c), (r + 1, c + 1)}
            if free(points) and _contacts(points, center, radius, (1, 1))[0] is None:
                occupied.update(points)
                objects.append((points, colors[0]))
                break
        else:
            continue
        # Other unselected objects may lie inside or outside the forward quadrant.
        desired = int(rng.integers(1, 4))
        placed = 0
        for _ in range(120):
            if placed == desired:
                break
            r, c = int(rng.integers(h)), int(rng.integers(w))
            allowed = {(r + dr, c + dc) for dr in range(int(rng.integers(1, 4)))
                       for dc in range(int(rng.integers(1, 4)))}
            points = _grow(allowed, (r, c), int(rng.integers(1, len(allowed) + 1)), rng)
            if (free(points) and _contacts(points, center, radius, (1, 1))[0] is None
                    and all(abs(a - center[0]) + abs(b - center[1]) > radius + 2 for a, b in points)):
                occupied.update(points)
                objects.append((points, int(rng.choice(bead_bank))))
                placed += 1
        if placed != desired:
            continue
        parity = np.indices((h, w)).sum(axis=0) % 2
        inp = np.asarray(background, dtype=np.int8)[parity]
        for p in _outline(center, radius):
            inp[p] = structural
        for p in _cap(center, radius, (1, 1)):
            inp[p] = cap_color
        for p in motif_points:
            inp[p] = motif_color
        for points, color in objects:
            for p in points:
                inp[p] = color
        inp = _flip(inp, direction)
        out = _flip(_assemble(motif, colors, cap_color, (1, 1)), direction)
        try:
            d = _decode(inp)
        except ValueError:
            continue
        if len(d["selected"]) != n or [o["color"] for o in d["selected"]] != colors:
            continue
        if witness and sum(len(o["points"]) for o in d["selected"]) != 2 * radius + 6:
            continue
        if not np.array_equal(solve(inp), out):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, name, low, high):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
             name + " must be an integer")
    value = int(value)
    _require(low <= value <= high, name + " outside legal range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _count(seed, "seed", 0, 2**128 - 1)
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    radius = int(rng.integers(3, 5))
    first = int(rng.integers(4))
    directions = [DIRECTIONS[first], DIRECTIONS[(first + 1) % 4]]
    specs = [(2, directions[0], 1), (3, directions[1], 2)]
    for _ in range(nt - 2):
        direction = DIRECTIONS[int(rng.integers(4))]
        specs.append((int(rng.integers(1, 4)), direction, 0))
    available = sorted({s[1] for s in specs})
    test_specs = [(int(rng.integers(4, 6)), available[int(rng.integers(len(available)))], 0)]
    test_specs += [(int(rng.integers(1, 6)), available[int(rng.integers(len(available)))], 0)
                   for _ in range(nq - 1)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plans in (("train", specs), ("test", test_specs)):
        for n, direction, witness in plans:
            for _ in range(10):
                pair = _sample_pair(rng, n, radius, direction, witness)
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("bounded uniqueness construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _walk_selection(d):
    """Check selection and ordering by explicitly walking the two rays."""
    owner = {p: i for i, o in enumerate(d["selected"] + d["ignored"]) for p in o["points"]}
    first = {}
    vr, vc = d["direction"]
    cells = _cap(d["center"], d["radius"], d["direction"])
    for r, c in (cells[0], cells[-1]):
        t = 0
        while True:
            t += 1
            r, c = r + vr, c + vc
            if not (0 <= r < d["grid"].shape[0] and 0 <= c < d["grid"].shape[1]):
                break
            if (r, c) in owner:
                i = owner[(r, c)]
                first[i] = min(t, first.get(i, t))
    return sorted(first, key=first.get), first


def _interior_only(d, o):
    vr, vc = d["direction"]
    return any(abs((r - d["center"][0]) * vr - (c - d["center"][1]) * vc) < d["radius"]
               and min((r - d["center"][0]) * vr, (c - d["center"][1]) * vc) >= 2
               and ((r - d["center"][0]) * vr - (c - d["center"][1]) * vc - d["radius"]) % 2 == 0
               for r, c in o["points"])


def _shortcut_counts(d):
    """Concrete ablations reconstructed from input objects, not latent plans."""
    objects = d["selected"] + d["ignored"]
    owner = {p: i for i, o in enumerate(objects) for p in o["points"]}
    whole_cap = set()
    vr, vc = d["direction"]
    for r, c in _cap(d["center"], d["radius"], d["direction"]):
        while True:
            r, c = r + vr, c + vc
            if not (0 <= r < d["grid"].shape[0] and 0 <= c < d["grid"].shape[1]):
                break
            if (r, c) in owner:
                whole_cap.add(owner[(r, c)])
    return dict(correct=len(d["selected"]), all_objects=len(objects),
                whole_cap=len(whole_cap),
                rail_minus=sum(-1 in o["rails"] for o in d["selected"]),
                rail_plus=sum(1 in o["rails"] for o in d["selected"]),
                per_rail=sum(len(o["rails"]) for o in d["selected"]),
                hit_pixels=sum(o["pixels"] for o in d["selected"]),
                selected_area=sum(len(o["points"]) for o in d["selected"]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, decoded = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            _require(1 <= len(d["selected"]) <= 5, "selected count outside supported range")
            walked, distances = _walk_selection(d)
            _require(walked == list(range(len(d["selected"]))), "ray walk selection disagrees")
            _require(all(distances[i] == o["distance"] for i, o in enumerate(d["selected"])), "contact distance disagrees")
            _require(np.array_equal(solve(inp), out), "incorrect output")
            th, tw = d["motif"].shape
            n = len(d["selected"])
            # Verify the cells by modular coordinates, without calling assembly.
            for r, c in np.ndindex(out.shape):
                ri, rr = divmod(r, th + 1)
                ci, cc = divmod(c, tw + 1)
                i = ri if d["direction"][0] > 0 else n - 1 - ri
                j = ci if d["direction"][1] > 0 else n - 1 - ci
                expected = d["cap_color"]
                if i == j and rr < th and cc < tw and d["motif"][rr, cc]:
                    expected = d["selected"][i]["color"]
                _require(int(out[r, c]) == expected, "tile/gap/orientation invariant failed")
            _require(bool(d["ignored"]) and any(_interior_only(d, o) for o in d["ignored"]),
                     "missing interior-lane distractor")
            decoded[split].append(d)
    train, tests = decoded["train"], decoded["test"]
    _require(len({len(d["selected"]) for d in train}) >= 2, "need count teaching")
    _require(any(len(o["rails"]) == 2 for d in train for o in d["selected"]), "missing bridge witness")
    _require(any(a is not b and len(a["selected"]) != len(b["selected"])
                 and sum(len(o["points"]) for o in a["selected"]) == sum(len(o["points"]) for o in b["selected"])
                 for a in train for b in train), "selected area/count confound")
    shortcuts = [_shortcut_counts(d) for d in train]
    for model in ("all_objects", "whole_cap", "rail_minus", "rail_plus", "per_rail", "hit_pixels", "selected_area"):
        _require(any(row[model] != row["correct"] for row in shortcuts),
                 "training does not reject shortcut: " + model)
    _require(any([o["color"] for o in d["selected"]] not in
                 (sorted(o["color"] for o in d["selected"]), sorted((o["color"] for o in d["selected"]), reverse=True))
                 for d in train), "color-sort shortcut")
    for slope in (-1, 1):
        _require(any(d["direction"][0] * d["direction"][1] == slope and _asymmetric(d["motif"])
                     and len({o["color"] for o in d["selected"]}) > 1 for d in train),
                 "missing asymmetric, ordered-color witness for a diagonal slope")
    _require({d["direction"] for d in tests} <= {d["direction"] for d in train}, "untrained direction")
    _require(max(len(d["selected"]) for d in tests) > max(len(d["selected"]) for d in train), "missing count shift")
    _require(len({d["radius"] for d in train + tests}) == 1, "puzzle radius must be shared")
    return True


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, payload = _key(pair[role])
                h.update(bytes(shape))
                h.update(payload)
    return h.hexdigest()


def self_test():
    """42 host seeds; streaming measurements, no puzzle archive or disk writes."""
    structures, shapes, motifs, signatures, directions, radii, counts = set(), set(), set(), set(), set(), set(), set()
    bridges = multi_hits = 0
    pair_count = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                d = _decode(pair["input"])
                shapes.add(pair["input"].shape)
                motifs.add(_key(d["motif"]))
                directions.add(d["direction"])
                radii.add(d["radius"])
                counts.add(len(d["selected"]))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat)
                structures.add(hashlib.sha256(bytes(pair["input"].shape) + normalized).hexdigest())
                bridges += sum(len(o["rails"]) == 2 for o in d["selected"])
                multi_hits += sum(o["pixels"] > 1 for o in d["selected"])
    _require(len(signatures) == 42 and len(structures) >= 180, "insufficient structure variation")
    _require(len(shapes) >= 35 and len(motifs) >= 45, "insufficient geometry variation")
    _require(directions == set(DIRECTIONS) and radii == {3, 4} and counts == {1, 2, 3, 4, 5}, "missing regimes")
    _require(bridges >= 42 and multi_hits >= 100, "missing whole-object evidence")
    original = generate_puzzle(0, 2, 1)
    _require(_digest(original) == _digest(generate_puzzle(0, 2, 1)), "nondeterminism")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(12345, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "ignored explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal count")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted damaged output")
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(signatures),
                input_structures=len(structures), input_shapes=len(shapes), motifs=len(motifs),
                directions=len(directions), radii=sorted(radii), selected_counts=sorted(counts),
                bridge_objects=bridges, multi_hit_objects=multi_hits,
                semantic_verification="generator-owned checks only")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
