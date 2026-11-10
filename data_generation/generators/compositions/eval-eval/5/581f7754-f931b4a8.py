"""Registered self-printing stamps. Only NumPy and the standard library are used.

Construction uses local templates; solve segments and interprets the actual input.
No examples, seeds, or parent code participate in solving.
"""
from collections import Counter
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_REQUIRED = frozenset((
    "no_alignment", "ignore_bar_count", "no_native_overlay", "tile_only",
    "top_left_phase", "global_phase", "phase_rows", "phase_columns",
    "center_registration", "keep_sources", "no_mask", "block_pin", "sum_rulers",
))
_WITNESSES = frozenset(("full_motion", "visible_coarse", "visible_overwrite"))
_TRAIN_EVIDENCE = _REQUIRED | _WITNESSES


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(int, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _normalized(a):
    names = {}
    return a.shape, bytes(names.setdefault(int(v), len(names)) for v in a.flat)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        points, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    points.add(q)
                    todo.append(q)
        result.append(points)
    return result


def _ruler(points, grid):
    """A monochrome boundary component must be a straight inward bar."""
    h, w = grid.shape
    if len({int(grid[p]) for p in points}) != 1:
        return None
    ends = [p for p in points if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    if not ends:
        return None
    _require(len(ends) == 1, "a ruler must have one boundary endpoint")
    r, c = ends[0]
    _require((r not in (0, h - 1)) != (c not in (0, w - 1)), "corner ruler")
    k = len(points)
    _require(1 <= k <= 3, "ruler factor outside 1..3")
    if c in (0, w - 1):
        axis, guide, side = 0, r, "left" if c == 0 else "right"
        expected = {(r, j if c == 0 else w - 1 - j) for j in range(k)}
    else:
        axis, guide, side = 1, c, "top" if r == 0 else "bottom"
        expected = {(j if r == 0 else h - 1 - j, c) for j in range(k)}
    _require(points == expected, "ruler is not normal to its edge")
    return dict(axis=axis, guide=guide, factor=k, points=points, side=side)


def _decode(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))),
             "input corners must identify background")
    systems, carriers = {}, []
    for points in _components(g != bg):
        bar = _ruler(points, g)
        if bar is None:
            carriers.append(points)
            continue
        color = int(g[next(iter(points))])
        if color in systems:
            s = systems[color]
            _require((s["axis"], s["guide"], s["factor"]) ==
                     (bar["axis"], bar["guide"], bar["factor"]), "inconsistent rulers")
            _require(bar["side"] not in s["sides"], "duplicate ruler side")
            s["points"] |= points
            s["sides"].add(bar["side"])
        else:
            systems[color] = {**bar, "points": set(points), "sides": {bar["side"]}}
    _require(1 <= len(systems) <= 3, "need 1..3 ruler identities")
    stamps = []
    for points in carriers:
        pins = [p for p in points if int(g[p]) in systems]
        _require(len(pins) == 1, "stamp must have one ruler-colored pin")
        pin = pins[0]
        marker = int(g[pin])
        counts = Counter(int(g[p]) for p in points if p != pin)
        _require(counts, "singleton stamps are unsupported")
        body, frequency = counts.most_common(1)[0]
        _require(frequency * 2 > len(points) - 1, "body needs a strict majority")
        accents = set(counts) - {body}
        _require(accents, "stamp needs visible accent ink")
        r0, c0 = min(r for r, _ in points), min(c for _, c in points)
        r1, c1 = max(r for r, _ in points), max(c for _, c in points)
        template = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in points:
            template[r - r0, c - c0] = g[r, c]
        local_pin = pin[0] - r0, pin[1] - c0
        system = systems[marker]
        q = list(pin)
        q[system["axis"]] = system["guide"]
        stamps.append(dict(points=points, pin=pin, marker=marker, body=body,
                           accents=accents, template=template, local_pin=local_pin,
                           target=tuple(q), factor=system["factor"], axis=system["axis"]))
    _require(1 <= len(stamps) <= 5, "need 1..5 stamps")
    _require(set(s["marker"] for s in stamps) == set(systems), "unmatched ruler")
    return g, bg, systems, stamps


def _predict(input_grid, variant=None, check_overlap=True):
    """Input-only cellwise interpretation, also used for diagnostic mutants."""
    g, bg, systems, stamps = _decode(input_grid)
    h, w = g.shape
    result = g.copy() if variant == "keep_sources" else np.full_like(g, bg)
    claimed = set()
    for color, system in systems.items():
        for p in system["points"]:
            result[p] = color
            claimed.add(p)
    for s in stamps:
        k = 1 if variant == "ignore_bar_count" else s["factor"]
        if variant == "sum_rulers":
            k *= len(systems[s["marker"]]["sides"])
        q = s["pin"] if variant == "no_alignment" else s["target"]
        template, (pr, pc) = s["template"], s["local_pin"]
        th, tw = template.shape
        if variant == "center_registration":
            q = list(q)
            center = ((th - 1) // 2, (tw - 1) // 2)
            q[s["axis"]] += k * (s["local_pin"][s["axis"]] - center[s["axis"]])
            q = tuple(q)
        origin = q[0] - k * pr, q[1] - k * pc
        footprint = set()
        for u in range(th):
            for v in range(tw):
                raw = int(template[u, v])
                if raw == bg and variant != "no_mask":
                    continue
                coarse = (s["marker"] if variant == "block_pin" else s["body"]) \
                    if (u, v) == (pr, pc) else raw
                for a in range(k):
                    for b in range(k):
                        r, c = origin[0] + k * u + a, origin[1] + k * v + b
                        if not (0 <= r < h and 0 <= c < w):
                            continue
                        if variant == "top_left_phase":
                            tr, tc = (r - origin[0]) % th, (c - origin[1]) % tw
                        elif variant == "global_phase":
                            tr, tc = r % th, c % tw
                        else:
                            tr = (r - q[0] + pr + (variant == "phase_rows")) % th
                            tc = (c - q[1] + pc + (variant == "phase_columns")) % tw
                        fine = int(template[tr, tc])
                        value = coarse
                        if variant == "tile_only":
                            value = s["body"] if fine == s["marker"] else fine
                        elif variant != "no_native_overlay" and fine in s["accents"]:
                            value = fine
                        if raw == bg and value == bg:
                            continue
                        if check_overlap:
                            _require((r, c) not in claimed, "destination silhouettes overlap")
                        footprint.add((r, c))
                        result[r, c] = value
        claimed |= footprint
        if 0 <= q[0] < h and 0 <= q[1] < w:
            result[q] = s["marker"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell from this input, with fixed visible conventions."""
    return _predict(input_grid)


def _evidence(pair):
    evidence = {name for name in _REQUIRED if not np.array_equal(
        pair["output"], _predict(pair["input"], name, check_overlap=False))}
    for f in _observations(pair["input"]):
        for name in _WITNESSES:
            if f[name]:
                evidence.add(name)
    return evidence


def _blocked(grid, bg):
    occupied = grid != bg
    blocked = occupied.copy()
    blocked[1:] |= occupied[:-1]
    blocked[:-1] |= occupied[1:]
    blocked[:, 1:] |= occupied[:, :-1]
    blocked[:, :-1] |= occupied[:, 1:]
    return blocked


def _shape(rng, max_side):
    """Procedural geometry, not a catalogue of fixed pictures."""
    kind = int(rng.integers(5))
    h, w = (int(rng.integers(2, max_side + 1)) for _ in range(2))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        h, w = max(3, h), max(3, w)
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 2:
        cells = {(r, c) for r in range(h) for c in range(w)}
        if h >= 3 and w >= 3:
            cut_h, cut_w = int(rng.integers(1, h)), int(rng.integers(1, w))
            cells -= {(r, c) for r in range(cut_h) for c in range(cut_w)}
    elif kind == 3:
        cells = {(h // 2, w // 2)}
        count = int(rng.integers(5, max(6, min(h * w + 1, 26))))
        count = min(count, h * w)
        for _ in range(count - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    else:
        length = int(rng.integers(3, max_side + 2))
        cells = {(r, 0) for r in range(length)}
        for r in rng.choice(length, size=int(rng.integers(1, min(4, length) + 1)), replace=False):
            sign = int(rng.choice([-1, 1]))
            for c in range(1, int(rng.integers(2, 4))):
                cells.add((int(r), sign * c))
    while len(cells) < 5:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _template(rng, marker, payload, bg, max_side, forbidden_periods=()):
    points = _shape(rng, max_side)
    h, w = max(r for r, _ in points) + 1, max(c for _, c in points) + 1
    if (h, w) in forbidden_periods:
        return None
    candidates = [p for p in points if 2 * p[0] != h - 1 or 2 * p[1] != w - 1]
    # Pick eccentric pins frequently without imposing a fixed corner convention.
    choices = candidates if candidates and rng.random() < .8 else points
    pin = choices[int(rng.integers(len(choices)))]
    colors = list(map(int, rng.permutation(payload)))
    body = colors[0]
    remaining = [p for p in points if p != pin]
    accent_count = int(rng.integers(1, (len(remaining) - 1) // 2 + 1))
    ink_count = int(rng.integers(1, min(3, len(colors) - 1, accent_count) + 1))
    accents = colors[1:ink_count + 1]
    # Independent dispersed versus locally grouped accents.
    if rng.random() < .5:
        chosen = [remaining[int(i)] for i in rng.choice(len(remaining), accent_count, replace=False)]
    else:
        center = remaining[int(rng.integers(len(remaining)))]
        chosen = sorted(remaining, key=lambda p: (abs(p[0] - center[0]) + abs(p[1] - center[1]), p))[:accent_count]
    a = np.full((h, w), bg, dtype=int)
    for p in points:
        a[p] = body
    for i, p in enumerate(chosen):
        a[p] = accents[i % ink_count]
    a[pin] = marker
    return a, pin, body


def _construct_print(template, pin, body, marker, k, target, bg):
    """Vectorized construction, separate from the solver's cellwise traversal."""
    coarse = template.copy()
    coarse[pin] = body
    large = np.repeat(np.repeat(coarse, k, axis=0), k, axis=1)
    support = large != bg
    rr, cc = np.indices(large.shape)
    fine = template[(rr - (k - 1) * pin[0]) % template.shape[0],
                    (cc - (k - 1) * pin[1]) % template.shape[1]]
    opaque = (fine != bg) & (fine != body) & (fine != marker)
    printed = np.where(opaque & support, fine, large)
    printed[k * pin[0], k * pin[1]] = marker
    origin = target[0] - k * pin[0], target[1] - k * pin[1]
    return printed, support, origin


def _observations(input_grid):
    """Semantic/geometry measurements reconstructed from input cells only."""
    grid, bg, _, stamps = _decode(input_grid)
    h, w = grid.shape
    observations = []
    for s in stamps:
        t, pin, k, q = s["template"], s["local_pin"], s["factor"], s["target"]
        coarse = t.copy()
        coarse[pin] = s["body"]
        large = np.repeat(np.repeat(coarse, k, axis=0), k, axis=1)
        rr, cc = np.indices(large.shape)
        gr, gc = rr + q[0] - k * pin[0], cc + q[1] - k * pin[1]
        inside = (gr >= 0) & (gr < h) & (gc >= 0) & (gc < w)
        support = large != bg
        fine = t[(gr - q[0] + pin[0]) % t.shape[0], (gc - q[1] + pin[1]) % t.shape[1]]
        opaque = np.isin(fine, list(s["accents"]))
        original_accent = np.isin(large, list(s["accents"]))
        clipped = support & ~inside
        observations.append(dict(axis=s["axis"], factor=k, period=t.shape,
            moving=q != s["pin"], full_motion=k > 1 and q != s["pin"] and not clipped.any(),
            visible_coarse=bool(np.any(inside & support & ~opaque & original_accent)),
            visible_overwrite=bool(np.any(inside & support & opaque & (fine != large))),
            clipped=bool(clipped.any()), clip_sides={name for name, condition in (
                ("top", gr < 0), ("bottom", gr >= h), ("left", gc < 0), ("right", gc >= w))
                if np.any(support & condition)},
            edge_contact=any(r in (0, h - 1) or c in (0, w - 1) for r, c in s["points"]),
            pin_degree=sum((s["pin"][0] + dr, s["pin"][1] + dc) in s["points"] for dr, dc in _DIRS),
            accent_inks=len(s["accents"]), occupied_area=len(s["points"])))
    return observations


def _place(rng, inp, out, bg, system, template_info, must_move=False):
    if template_info is None:
        return False
    template, pin, body = template_info
    h, w = inp.shape
    th, tw = template.shape
    if th > h or tw > w:
        return False
    rr, cc = np.meshgrid(np.arange(pin[0], h - th + pin[0] + 1),
                         np.arange(pin[1], w - tw + pin[1] + 1), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    valid = np.ones(len(rr), dtype=bool)
    if must_move:
        valid &= (rr if system["axis"] == 0 else cc) != system["guide"]
    blocked = _blocked(inp, bg)
    blocked[0, 0] = blocked[0, -1] = blocked[-1, 0] = blocked[-1, -1] = True
    points = np.argwhere(template != bg)
    k = system["factor"]
    tr = np.full_like(rr, system["guide"]) if system["axis"] == 0 else rr
    tc = cc if system["axis"] == 0 else np.full_like(cc, system["guide"])
    for u, v in points:
        dr, dc = int(u) - pin[0], int(v) - pin[1]
        valid &= ~blocked[rr + dr, cc + dc]
        for a in range(k):
            for b in range(k):
                r, c = tr + k * dr + a, tc + k * dc + b
                inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
                valid &= ~inside | (out[np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)] == bg)
    choices = np.flatnonzero(valid)
    if not len(choices):
        return False
    j = int(rng.choice(choices))
    source_pin, target = (int(rr[j]), int(cc[j])), (int(tr[j]), int(tc[j]))
    origin = source_pin[0] - pin[0], source_pin[1] - pin[1]
    for u, v in points:
        inp[origin[0] + u, origin[1] + v] = template[u, v]
    printed, support, origin = _construct_print(template, pin, body, system["marker"], k, target, bg)
    for r, c in np.argwhere(support):
        q = origin[0] + int(r), origin[1] + int(c)
        if 0 <= q[0] < h and 0 <= q[1] < w:
            out[q] = printed[r, c]
    return True


def _sample_pair(rng, params, primary_axis=None, teaching=False, forbidden_periods=(),
                 forbid_all=False, must_print=False):
    """Bounded placement of independent geometry under one visible rule."""
    for _ in range(50):
        palette = params["palette"] if params["stable_palette"] else rng.permutation(10)
        bg = int(palette[0])
        n = int(rng.choice([1, 2, 3], p=[.50, .42, .08]))
        low = 10 if n == 1 else 17
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        markers = list(map(int, palette[1:n + 1]))
        payload = list(map(int, palette[n + 1:]))
        inp, out = np.full((h, w), bg, dtype=int), np.full((h, w), bg, dtype=int)
        systems = []
        for j, marker in enumerate(markers):
            axis = primary_axis if j == 0 and primary_axis is not None else int(rng.integers(2))
            k = int(rng.choice([2, 3])) if teaching and j == 0 else int(rng.choice([1, 2, 3], p=[.15, .50, .35]))
            ends = [0, 1] if rng.random() < .28 else [int(rng.integers(2))]
            guides = rng.permutation(np.arange(1, inp.shape[axis] - 1))
            for guide in guides:
                points = {(int(guide), z if end == 0 else w - 1 - z)
                          if axis == 0 else (z if end == 0 else h - 1 - z, int(guide))
                          for end in ends for z in range(k)}
                if any(_blocked(inp, bg)[p] for p in points):
                    continue
                system = dict(axis=axis, factor=k, guide=int(guide), marker=marker)
                for p in points:
                    inp[p] = out[p] = marker
                systems.append(system)
                break
            else:
                break
        if len(systems) != n:
            continue
        total = 0
        success = True
        for j, system in enumerate(systems):
            for _ in range(35):
                template = _template(rng, system["marker"], payload, bg,
                                     params["max_side"] if n == 1 else min(params["max_side"], 5),
                                     forbidden_periods if j == 0 or forbid_all else ())
                if _place(rng, inp, out, bg, system, template, must_move=j == 0):
                    total += 1
                    break
            else:
                success = False
                break
        if not success:
            continue
        # More than one stamp may use a ruler; source and target packing decide feasibility.
        for _ in range(int(rng.integers(0, min(3, 5 - total) + 1))):
            system = systems[int(rng.integers(n))]
            for _ in range(5):
                template = _template(rng, system["marker"], payload, bg, min(5, params["max_side"]),
                                     forbidden_periods if forbid_all else ())
                if _place(rng, inp, out, bg, system, template):
                    break
        pair = {"input": inp, "output": out}
        _require(np.array_equal(solve(inp), out), "construction/solve disagreement")
        if must_print and not any(f["visible_overwrite"] for f in _observations(inp)):
            continue
        return pair
    raise RuntimeError("bounded placement failed")


def _counts(num_train, num_test):
    for x, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or not low <= int(x) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    params = dict(palette=rng.permutation(10), stable_palette=bool(rng.random() < .25),
                  max_side=int(rng.choice([4, 5, 6])))
    heldout = {(int(rng.integers(3, params["max_side"] + 1)),
                int(rng.integers(3, params["max_side"] + 1)))}
    pool, covers = [], []
    training = None
    for i in range(64):
        pair = _sample_pair(rng, params, primary_axis=i % 2, teaching=True,
                            forbidden_periods=heldout, forbid_all=True)
        cover = _evidence(pair)
        periods = {s["template"].shape for s in _decode(pair["input"])[3]}
        axes = {s["axis"] for s in _decode(pair["input"])[3]
                if s["factor"] > 1 and s["target"] != s["pin"]}
        for j in rng.permutation(len(pool)):
            old = pool[int(j)]
            old_stamps = _decode(old["input"])[3]
            if (_TRAIN_EVIDENCE <= cover | covers[int(j)]
                    and axes | {s["axis"] for s in old_stamps
                                if s["factor"] > 1 and s["target"] != s["pin"]} == {0, 1}
                    and len(periods | {s["template"].shape for s in old_stamps}) >= 2
                    and _key(old["input"]) != _key(pair["input"])):
                training = [old, pair]
                break
        pool.append(pair)
        covers.append(cover)
        if training is not None:
            break
    if training is None:
        raise RuntimeError("bounded teaching pool did not distinguish the rule")
    puzzle, seen = {"train": training, "test": []}, {_key(p["input"]) for p in training}
    for split, count in (("train", a), ("test", b)):
        while len(puzzle[split]) < count:
            periods = {s["template"].shape for p in puzzle["train"] for s in _decode(p["input"])[3]}
            for _ in range(24):
                pair = _sample_pair(rng, params, forbidden_periods=periods if split == "test" else heldout,
                                    forbid_all=split == "train", must_print=split == "test")
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded unique-input sampling failed")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence, train_axes, train_periods = set(), set(), set(), set()
    test_periods = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            _require(np.array_equal(solve(inp), out), "incorrect registered print")
            _, bg, systems, stamps = _decode(inp)
            _require(not np.array_equal(inp, out), "unchanged example")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "invented color")
            for color, s in systems.items():
                _require(all(int(out[p]) == color for p in s["points"]), "ruler changed")
            for s in stamps:
                _require(int(out[s["target"]]) == s["marker"], "pin lost")
                _require(s["target"][1 - s["axis"]] == s["pin"][1 - s["axis"]], "along-guide motion")
                expected_pins = sum(t["marker"] == s["marker"] for t in stamps)
                _require(int(np.sum(out == s["marker"])) == len(systems[s["marker"]]["points"]) + expected_pins,
                         "registration pin was replicated or erased")
            periods = {s["template"].shape for s in stamps}
            if split == "train":
                train_axes |= {s["axis"] for s in stamps if s["factor"] > 1 and s["target"] != s["pin"]}
                train_periods |= periods
                evidence |= _evidence(pair)
            else:
                _require(any(f["visible_overwrite"] for f in _observations(inp)), "test has no fine-print overwrite")
                _require({"no_alignment", "no_native_overlay"} <= _evidence(pair), "test omits a parent mechanism")
                test_periods.append(periods)
    _require(train_axes == {0, 1}, "both guide orientations need demonstrations")
    _require(len(train_periods) >= 2, "training needs distinct repeat periods")
    _require(_TRAIN_EVIDENCE <= evidence, "missing training evidence: " + str(sorted(_TRAIN_EVIDENCE - evidence)))
    _require(all(p - train_periods for p in test_periods), "test must contain a novel repeat period")


def _fingerprint(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded structural, semantic, metamorphic and variation checks over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features, shapes, structures, palettes, periods = [], set(), set(), set(), set()
    stamp_observations = []
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, bg, systems, stamps = _decode(pair["input"])
                shapes.add(g.shape)
                structures.add(_normalized(g))
                palettes.add(bg)
                periods |= {s["template"].shape for s in stamps}
                stamp_observations.extend(_observations(g))
                features.append(dict(seed=seed, split=split, systems=len(systems), stamps=len(stamps),
                    factors={s["factor"] for s in stamps}, axes={s["axis"] for s in stamps},
                    bilateral=any(len(s["sides"]) == 2 for s in systems.values()),
                    holes=any(np.any(s["template"] == bg) for s in stamps),
                    shared=len(stamps) > len(systems),
                    sides=set().union(*(s["sides"] for s in systems.values()))))
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "repeated puzzles")
    _require(len(shapes) >= 60 and len(structures) >= 180, "insufficient geometry variation")
    _require(len(periods) >= 15 and len(palettes) >= 8, "period/palette variation")
    _require(set().union(*(f["clip_sides"] for f in stamp_observations)) ==
             {"left", "right", "top", "bottom"}, "clipping edges")
    _require(any(f["edge_contact"] for f in stamp_observations), "input boundary contact")
    _require(any(not f["clipped"] for f in stamp_observations)
             and any(f["clipped"] for f in stamp_observations), "boundary variation")
    _require({1, 2, 3} <= {f["accent_inks"] for f in stamp_observations}, "accent palette variation")
    _require(len({f["pin_degree"] for f in stamp_observations}) >= 3, "pin location variation")
    for split in ("train", "test"):
        fs = [f for f in features if f["split"] == split]
        _require(set().union(*(f["axes"] for f in fs)) == {0, 1}, "axis variation")
        _require(set().union(*(f["factors"] for f in fs)) == {1, 2, 3}, "factor variation")
        _require(set().union(*(f["sides"] for f in fs)) == {"left", "right", "top", "bottom"}, "side variation")
        for name in ("bilateral", "holes", "shared"):
            _require(sum(f[name] for f in fs) >= 3, split + " lacks " + name)
        _require(len({f["stamps"] for f in fs}) >= 3, "stamp-count variation")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(78000 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0)), "nondeterminism")
    # A hand-calculated one-stamp check fixes phase, gated holes, coarse colors,
    # overwrite, pin-block replacement, source erasure and guide projection.
    hand = np.zeros((12, 15), dtype=int)
    hand[4, :2] = 9
    hand[7:10, 7:10] = [[1, 1, 2], [1, 0, 1], [9, 1, 1]]
    expected = np.zeros_like(hand)
    expected[4, :2] = 9
    expected[:6, 7:13] = 1
    expected[:2, 11:13] = 2
    expected[2:4, 9:11] = 0
    expected[2, 12] = expected[5, 9] = expected[5, 12] = 2
    expected[4, 7] = 9
    _require(np.array_equal(solve(hand), expected), "hand-calculated registration/print mismatch")
    # Palette reassignment is a genuine semantic symmetry, independent per pair.
    rng = np.random.default_rng(991)
    recolored = {split: [{role: rng_colors[np.asarray(pair[role])] for role in ("input", "output")}
                         for pair in puzzles[0][split] for rng_colors in [rng.permutation(10)]]
                 for split in ("train", "test")}
    validate_generated_puzzle(recolored)
    # Transposition preserves the top-left pin-block convention; rotations do not.
    transposed = {split: [{role: np.asarray(pair[role]).T.copy() for role in ("input", "output")}
                         for pair in puzzles[0][split]] for split in ("train", "test")}
    validate_generated_puzzle(transposed)
    damaged = {split: [{role: np.asarray(pair[role]).copy() for role in ("input", "output")}
                      for pair in puzzles[0][split]] for split in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, pairs=len(features), canvas_shapes=len(shapes),
                normalized_input_structures=len(structures), native_periods=len(periods),
                background_colors=len(palettes),
                factor_values=sorted(set().union(*(f["factors"] for f in features))),
                stamps=len(stamp_observations), clipped_stamps=sum(f["clipped"] for f in stamp_observations),
                complete_magnified_movers=sum(f["full_motion"] for f in stamp_observations),
                input_boundary_stamps=sum(f["edge_contact"] for f in stamp_observations))
