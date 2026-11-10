"""Grow a blueprint mosaic and ornament its convex silhouette corners.

Only standard-library modules and NumPy are used. All solver geometry and
color roles are reconstructed from the input; generation has no solver state.
"""
from __future__ import annotations

import numpy as np


_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _connected(mask):
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if not cells:
        return False
    seen = {min(cells)}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _valid_occupancy(mask):
    if not _connected(mask):
        return False
    if any(not edge.any() for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1])):
        return False
    # Two diagonally touching tiles with both intervening tiles empty have no
    # unambiguous inward corner direction. Such pinches are outside the family.
    for r in range(mask.shape[0] - 1):
        for c in range(mask.shape[1] - 1):
            a, b, d, e = map(bool, mask[r:r + 2, c:c + 2].flat)
            if (a and e and not b and not d) or (b and d and not a and not e):
                return False
    return True


def _corners(plan, bg):
    """(tile row, tile column, orientation) at one-occupied-site vertices."""
    occupied = plan != bg
    h, w = plan.shape
    result = []
    for r, c in np.argwhere(occupied):
        r, c = int(r), int(c)
        for q, (dy, dx) in enumerate(_DIRS):
            neighbours = ((r - dy, c), (r, c - dx), (r - dy, c - dx))
            if all(not (0 <= y < h and 0 <= x < w and occupied[y, x])
                   for y, x in neighbours):
                result.append((r, c, q))
    return result


def _points(origin, side, corner, pattern):
    r, c, q = corner
    dy, dx = _DIRS[q]
    y = origin[0] + r * side + (side - 1 if dy < 0 else 0)
    x = origin[1] + c * side + (side - 1 if dx < 0 else 0)
    return [(y + dy * u, x + dx * v) for u, v in sorted(pattern)]


def _render(inp, plan, bg, origin, side, pattern, reference_q=0,
            mode="correct"):
    out = np.asarray(inp, dtype=np.int8).copy()
    for (r, c), color in np.ndenumerate(plan):
        if color != bg:
            y, x = origin[0] + r * side, origin[1] + c * side
            out[y:y + side, x:x + side] = color
    if mode == "no_ornament":
        return out
    corners = (_corners(plan, bg) if mode != "each_tile" else
               [(int(r), int(c), q) for r, c in np.argwhere(plan != bg)
                for q in range(4)])
    for corner in corners:
        local = pattern
        if mode == "rotation" and ((corner[2] ^ reference_q) in (1, 2)):
            local = frozenset((v, u) for u, v in pattern)
        for p in _points(origin, side, corner, local):
            out[p] = plan[corner[0], corner[1]]
    return out


def _separate(a, b, gap=0):
    # Half-open rectangles, with an optional background gap.
    return (a[2] + gap <= b[0] or b[2] + gap <= a[0] or
            a[3] + gap <= b[1] or b[3] + gap <= a[1])


def _blueprint(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > np.partition(counts, -2)[-2], "ambiguous background"
    frames = []
    for color in range(10):
        if color == bg or counts[color] < 24:
            continue
        pts = np.argwhere(inp == color)
        t, l = map(int, pts.min(axis=0))
        b, r = map(int, pts.max(axis=0))
        h, w = b - t + 1, r - l + 1
        if min(h, w) < 7 or h % 2 != 1 or w % 2 != 1:
            continue
        mask = np.zeros(inp.shape, dtype=bool)
        mask[t, l:r + 1] = mask[b, l:r + 1] = True
        mask[t:b + 1, l] = mask[t:b + 1, r] = True
        if np.array_equal(inp == color, mask):
            frames.append((color, t, l, b, r))
    assert len(frames) == 1, "need one complete blueprint frame"
    frame, t, l, b, r = frames[0]
    plan = inp[t + 2:b:2, l + 2:r:2].copy()
    assert 2 <= min(plan.shape) and max(plan.shape) <= 5
    inside = inp[t + 1:b, l + 1:r]
    slots = np.zeros(inside.shape, dtype=bool)
    slots[1::2, 1::2] = True
    assert np.all(inside[~slots] == bg) and frame not in plan
    assert _valid_occupancy(plan != bg), "disconnected or pinched blueprint"
    payload = set(map(int, plan.flat)) - {bg}
    assert 2 <= len(payload) <= 6
    anchors = ((t, l), (t, r), (b, l), (b, r))
    references = []
    for q, (y, x) in enumerate(anchors):
        dy, dx = _DIRS[q]
        coords = []
        for a in range(max(0, y - 2), min(inp.shape[0], y + 3)):
            for z in range(max(0, x - 2), min(inp.shape[1], x + 3)):
                u, v = (a - y) * dy, (z - x) * dx
                if (u < 0 or v < 0) and int(inp[a, z]) not in (bg, frame):
                    coords.append((a, z))
        if coords:
            colors = {int(inp[p]) for p in coords}
            assert len(colors) == 1 and not (colors & payload)
            ink = colors.pop()
            pattern = frozenset(((a - y) * dy, (z - x) * dx) for a, z in coords)
            assert 3 <= len(pattern) <= 10
            assert {tuple(map(int, p)) for p in np.argwhere(inp == ink)} == set(coords)
            references.append((q, ink, pattern))
    assert len(references) == 1, "need one intact exterior corner stencil"
    q, ink, pattern = references[0]
    radius = max(max(abs(u), abs(v)) for u, v in pattern)
    assert radius in (1, 2)
    # Eight-connectivity is allowed for the ornament, unlike the tile shape.
    seen = {min(pattern)}
    todo = list(seen)
    while todo:
        u, v = todo.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                p = (u + du, v + dv)
                if p in pattern and p not in seen:
                    seen.add(p)
                    todo.append(p)
    assert seen == set(pattern), "disconnected reference stencil"
    rect = (t - radius, l - radius, b + radius + 1, r + radius + 1)
    assert 0 <= rect[0] < rect[2] <= inp.shape[0]
    assert 0 <= rect[1] < rect[3] <= inp.shape[1]
    keep = np.zeros(inp.shape, dtype=bool)
    keep[rect[0]:rect[2], rect[1]:rect[3]] = True
    # Everything in the protected halo is explained by frame, swatches, or ink.
    explained = np.full(inp.shape, bg, dtype=np.int8)
    explained[t, l:r + 1] = explained[b, l:r + 1] = frame
    explained[t:b + 1, l] = explained[t:b + 1, r] = frame
    explained[t + 2:b:2, l + 2:r:2] = plan
    y, x = anchors[q]
    dy, dx = _DIRS[q]
    for u, v in pattern:
        explained[y + dy * u, x + dx * v] = ink
    assert np.array_equal(inp[keep], explained[keep])
    assert all(int(v) in payload or int(v) == bg for v in inp[~keep])
    return dict(background=bg, frame=frame, ink=ink, plan=plan,
                pattern=pattern, reference_q=q, radius=radius,
                blueprint_rect=rect, keep=keep, payload=payload)


def _matches(plan, sequence, axis):
    n = len(sequence)
    result = []
    for r in range(plan.shape[0] - (n - 1 if axis == 0 else 0)):
        for c in range(plan.shape[1] - (n - 1 if axis == 1 else 0)):
            seq = tuple(int(plan[r + (axis == 0) * k, c + (axis == 1) * k])
                        for k in range(n))
            if seq == sequence:
                result.append((r, c))
    return result


def _recover(input_grid):
    inp = _grid(input_grid)
    info = _blueprint(inp)
    bg, plan, radius = info["background"], info["plan"], info["radius"]
    target = (inp != bg) & ~info["keep"]
    assert target.any()
    answers = []
    # Summed-area tables locate uniform square cores even when incomplete
    # ornaments touch or extend their components. No output is read here.
    tables = {}
    for color in info["payload"]:
        mask = (inp == color) & ~info["keep"]
        tables[color] = np.pad(mask.astype(int).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    for side in range(2 * radius + 1, 7):
        if side > min(inp.shape):
            continue
        square = np.full((inp.shape[0] - side + 1, inp.shape[1] - side + 1), -1, dtype=int)
        for color, table in tables.items():
            sums = table[side:, side:] - table[:-side, side:] - table[side:, :-side] + table[:-side, :-side]
            square[sums == side * side] = color
        for axis in (0, 1):
            for y, x in np.argwhere(square >= 0):
                y, x = int(y), int(x)
                seq = [int(square[y, x])]
                for k in range(1, min(4, plan.shape[axis])):
                    a, z = y + (axis == 0) * k * side, x + (axis == 1) * k * side
                    if not (a < square.shape[0] and z < square.shape[1]):
                        break
                    color = int(square[a, z])
                    if color < 0 or color == seq[-1]:
                        break
                    seq.append(color)
                    matches = _matches(plan, tuple(seq), axis)
                    if len(matches) != 1:
                        continue
                    match = matches[0]
                    origin = (y - match[0] * side, x - match[1] * side)
                    rect = (origin[0] - radius, origin[1] - radius,
                            origin[0] + plan.shape[0] * side + radius,
                            origin[1] + plan.shape[1] * side + radius)
                    if not (0 <= rect[0] < rect[2] <= inp.shape[0] and
                            0 <= rect[1] < rect[3] <= inp.shape[1] and
                            _separate(rect, info["blueprint_rect"], 1)):
                        continue
                    seed_sites = {(match[0] + (axis == 0) * j,
                                   match[1] + (axis == 1) * j) for j in range(len(seq))}
                    if np.count_nonzero(plan != bg) <= len(seed_sites):
                        continue
                    allowed = np.full(inp.shape, bg, dtype=np.int8)
                    for r, c in seed_sites:
                        a, z = origin[0] + r * side, origin[1] + c * side
                        allowed[a:a + side, z:z + side] = plan[r, c]
                    corners = _corners(plan, bg)
                    fragments = []
                    for corner in corners:
                        if corner[:2] in seed_sites:
                            points = _points(origin, side, corner, info["pattern"])
                            count = sum(inp[p] == plan[corner[:2]] for p in points)
                            if count:
                                fragments.append((corner, count))
                            for p in points:
                                allowed[p] = plan[corner[:2]]
                    if not np.array_equal(inp[target], allowed[target]):
                        continue
                    if any(n == len(info["pattern"]) for _, n in fragments):
                        continue  # supplied target ornaments are genuinely partial
                    pred = _render(inp, plan, bg, origin, side, info["pattern"])
                    meta = dict(info, side=side, axis=axis, match=match,
                                length=len(seq), origin=origin, corners=corners,
                                fragments=fragments, seed_sites=seed_sites)
                    answers.append((pred, meta))
    # Require a unique geometric interpretation, not a seed-dependent tie break.
    assert len(answers) == 1, f"missing or ambiguous seed geometry: {len(answers)}"
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _recover(input_grid)[0]


def _stencil(rng, radius, asymmetric=False):
    domain = [(u, v) for u in range(-radius, radius + 1)
              for v in range(-radius, radius + 1) if u < 0 or v < 0]
    for _ in range(80):
        target = int(rng.integers(3, min(10, len(domain)) + 1))
        pattern = {domain[int(rng.integers(len(domain)))]}
        while len(pattern) < target:
            frontier = [p for p in domain if p not in pattern and
                        any(max(abs(p[0] - u), abs(p[1] - v)) <= 1 for u, v in pattern)]
            pattern.add(frontier[int(rng.integers(len(frontier)))])
        if max(max(abs(u), abs(v)) for u, v in pattern) != radius:
            continue
        if asymmetric and pattern == {(v, u) for u, v in pattern}:
            continue
        return frozenset(pattern)
    return frozenset({(-radius, 0), (-radius, -1), (-radius, 1)})


def _options(radius, side=None, axis=1, interior=False):
    options = []
    for h in range(2, 6):
        for w in range(2, 6):
            if interior and (h, w)[1 - axis] < 3:
                continue
            for s in ([side] if side is not None else range(2 * radius + 1, 7)):
                if s < 2 * radius + 1:
                    continue
                mini = (2 * h + 3 + 2 * radius, 2 * w + 3 + 2 * radius)
                mosaic = (h * s + 2 * radius, w * s + 2 * radius)
                for sep in (0, 1):
                    required = [max(mini[d], mosaic[d]) for d in (0, 1)]
                    required[sep] = mini[sep] + mosaic[sep] + 1
                    if max(required) <= 30:
                        options.append((h, w, s, sep))
    return options


def _place(rng, plan_shape, radius, side, sep):
    mini = tuple(2 * n + 3 + 2 * radius for n in plan_shape)
    large = tuple(n * side + 2 * radius for n in plan_shape)
    gap = int(rng.integers(1, min(3, 30 - mini[sep] - large[sep]) + 1))
    required = [max(mini[d], large[d]) for d in (0, 1)]
    required[sep] = mini[sep] + large[sep] + gap
    shape = [int(rng.integers(n, min(30, n + 4) + 1)) for n in required]
    if rng.random() < .2:
        shape = [max(shape), max(shape)]
    if rng.random() < .1:
        shape = [30, 30]
    order = int(rng.integers(2))
    first, second = (mini, large) if order == 0 else (large, mini)
    extra = shape[sep] - required[sep]
    leading = int(rng.integers(extra + 1))
    extra_gap = int(rng.integers(extra - leading + 1))
    p, q = [0, 0], [0, 0]
    p[sep] = leading
    q[sep] = leading + first[sep] + gap + extra_gap
    other = 1 - sep
    p[other] = int(rng.integers(shape[other] - first[other] + 1))
    q[other] = int(rng.integers(shape[other] - second[other] + 1))
    ref, big = (p, q) if order == 0 else (q, p)
    return tuple(shape), (ref[0] + radius, ref[1] + radius), (big[0] + radius, big[1] + radius)


def _sample_plan(rng, shape, bg, palette, axis, boundary, need_blank,
                 need_fragment):
    for _ in range(120):
        occupancy = rng.random(shape) < float(rng.uniform(.65, 1.0))
        if not _valid_occupancy(occupancy) or (need_blank and occupancy.all()):
            continue
        plan = rng.choice(palette, size=shape).astype(np.int8)
        plan[~occupancy] = bg
        corners = _corners(plan, bg)
        choices = []
        for length in range(2, min(4, shape[axis]) + 1):
            for r in range(shape[0] - (length - 1 if axis == 0 else 0)):
                for c in range(shape[1] - (length - 1 if axis == 1 else 0)):
                    cross = (r, c)[1 - axis]
                    if boundary is None:
                        if not 0 < cross < shape[1 - axis] - 1:
                            continue
                    elif cross != (0 if boundary == 0 else shape[1 - axis] - 1):
                        continue
                    sites = [(r + (axis == 0) * k, c + (axis == 1) * k) for k in range(length)]
                    seq = tuple(int(plan[p]) for p in sites)
                    if bg in seq or any(a == b for a, b in zip(seq, seq[1:])):
                        continue
                    if len(_matches(plan, seq, axis)) != 1 or occupancy.sum() <= length:
                        continue
                    if need_fragment and not any(corner[:2] in sites for corner in corners):
                        continue
                    choices.append(((r, c), length))
        if choices:
            match, length = choices[int(rng.integers(len(choices)))]
            return plan, match, length
    return None


def _draw(rng, plan, bg, frame, ink, pattern, side, axis, match, length,
          sep, need_fragment=False):
    radius = max(max(abs(u), abs(v)) for u, v in pattern)
    shape, ref, origin = _place(rng, plan.shape, radius, side, sep)
    inp = np.full(shape, bg, dtype=np.int8)
    t, l = ref
    b, r = t + 2 * plan.shape[0] + 2, l + 2 * plan.shape[1] + 2
    inp[t, l:r + 1] = inp[b, l:r + 1] = frame
    inp[t:b + 1, l] = inp[t:b + 1, r] = frame
    inp[t + 2:b:2, l + 2:r:2] = plan
    reference_q = int(rng.integers(4))
    y, x = ((t, l), (t, r), (b, l), (b, r))[reference_q]
    dy, dx = _DIRS[reference_q]
    for u, v in pattern:
        inp[y + dy * u, x + dx * v] = ink
    seed = {(match[0] + (axis == 0) * k, match[1] + (axis == 1) * k) for k in range(length)}
    for a, z in seed:
        y, x = origin[0] + a * side, origin[1] + z * side
        inp[y:y + side, x:x + side] = plan[a, z]
    eligible = [c for c in _corners(plan, bg) if c[:2] in seed]
    rng.shuffle(eligible)
    count = int(rng.integers(int(need_fragment), min(4, len(eligible)) + 1))
    for corner in eligible[:count]:
        pts = _points(origin, side, corner, pattern)
        rng.shuffle(pts)
        keep = int(rng.integers(1, len(pts)))
        for p in pts[:keep]:
            inp[p] = plan[corner[:2]]
    out = _render(inp, plan, bg, origin, side, pattern)
    return {"input": inp, "output": out}


def _make_pair(rng, axis, boundary, side=None, asymmetric=False,
               need_blank=False, need_fragment=False, previous=None):
    # Bounded attempts; the final attempts use a constructive L/T blueprint.
    for attempt in range(100):
        colors = rng.permutation(10)
        bg, frame, ink = map(int, colors[:3])
        palette = colors[3:3 + int(rng.integers(2, 7))]
        radius = 1 if asymmetric else int(rng.choice((1, 1, 1, 2)))
        if side is not None and side < 5:
            radius = 1
        pattern = _stencil(rng, radius, asymmetric)
        options = _options(radius, side, axis, boundary is None)
        if previous is not None:
            options = [p for p in options if p[2] != previous["side"]]
            if bg == previous["background"] or pattern == previous["pattern"]:
                continue
        if not options:
            continue
        h, w, s, sep = options[int(rng.integers(len(options)))]
        sampled = _sample_plan(rng, (h, w), bg, palette, axis, boundary,
                               need_blank, need_fragment)
        if attempt >= 80:
            # Extent-witnessing L (boundary clue) or T (interior clue). Literal
            # colors can repeat away from the selected unequal two-tile run.
            a, z, d = map(int, colors[3:6])
            plan = np.array([[a, z], [d, bg]], dtype=np.int8)
            match = (0, 0)
            if boundary is None:
                plan = np.array([[a, bg], [z, d], [a, bg]], dtype=np.int8)
                match = (1, 0)
            elif boundary == 1:
                plan = np.flip(plan, 0).copy()
                match = (1, 0)
            if axis == 0:
                plan = plan.T.copy()
                match = match[::-1]
            feasible = [o for o in _options(radius, s, axis, boundary is None)
                        if o[:2] == plan.shape]
            if not feasible:
                continue
            sep = feasible[int(rng.integers(len(feasible)))][3]
            sampled = plan, match, 2
        if sampled is None:
            continue
        try:
            pair = _draw(rng, sampled[0], bg, frame, ink, pattern, s, axis,
                         sampled[1], sampled[2], sep, need_fragment)
            prediction, meta = _recover(pair["input"])
            assert np.array_equal(prediction, pair["output"])
            if asymmetric:
                rotated = _render(pair["input"], meta["plan"], bg, meta["origin"],
                                  s, pattern, meta["reference_q"], "rotation")
                assert not np.array_equal(rotated, pair["output"])
            return pair, meta
        except AssertionError:
            continue
    raise RuntimeError("bounded construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))
    axes = [mode] if mode < 2 else [0, 1]
    first_boundary = int(rng.integers(2))
    result, seen, train_meta = {"train": [], "test": []}, set(), []
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            axis = axes[index % len(axes)] if split == "train" else int(rng.choice(axes))
            boundary = (first_boundary if index % 2 == 0 else 1 - first_boundary) if split == "train" else None
            requested_side = int(rng.choice([m["side"] for m in train_meta])) if split == "test" else None
            for _ in range(32):
                pair, meta = _make_pair(rng, axis, boundary, requested_side,
                                       asymmetric=split == "train" and index == 0,
                                       need_blank=split == "test" or index == 0,
                                       need_fragment=split == "train" and index == 0,
                                       previous=train_meta[0] if split == "train" and index == 1 else None)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not obtain a distinct input")
            seen.add(key)
            result[split].append(pair)
            if split == "train":
                train_meta.append(meta)
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, metas = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            pred, m = _recover(inp)
            assert np.array_equal(pred, out), "wrong completion or unrelated change"
            assert np.any(inp != out)
            assert np.all(inp[inp != out] == m["background"]), "overwrote visible evidence"
            assert np.array_equal(inp[m["keep"]], out[m["keep"]])
            assert set(map(int, inp.flat)) == set(map(int, out.flat))
            # Omitting A or replacing global corners with per-tile corners must
            # change the result. These are family-specific negative controls.
            for mode in ("no_ornament", "each_tile"):
                alternative = _render(inp, m["plan"], m["background"], m["origin"],
                                      m["side"], m["pattern"], mode=mode)
                assert not np.array_equal(alternative, out), mode
            local_only = inp.copy()
            for corner, _ in m["fragments"]:
                for p in _points(m["origin"], m["side"], corner, m["pattern"]):
                    local_only[p] = m["plan"][corner[:2]]
            assert not np.array_equal(local_only, out), "omitting blueprint growth"
            assert len(m["corners"]) < 4 * np.count_nonzero(m["plan"] != m["background"])
            cross = m["match"][1 - m["axis"]]
            extent = m["plan"].shape[1 - m["axis"]]
            assert (cross in (0, extent - 1)) if split == "train" else (0 < cross < extent - 1)
            metas[split].append(m)
    teaching = metas["train"]
    assert len({m["side"] for m in teaching}) >= 2
    assert len({m["pattern"] for m in teaching}) >= 2
    assert len({m["background"] for m in teaching}) >= 2
    assert any(m["fragments"] for m in teaching)
    assert any(np.any(m["plan"] == m["background"]) for m in teaching)
    assert {m["axis"] for m in metas["test"]} <= {m["axis"] for m in teaching}
    assert {m["side"] for m in metas["test"]} <= {m["side"] for m in teaching}
    assert any(not np.array_equal(
        _render(pair["input"], m["plan"], m["background"], m["origin"], m["side"],
                m["pattern"], m["reference_q"], "rotation"), pair["output"])
        for pair, m in zip(puzzle["train"], teaching)), "reflection is untaught"


def self_test() -> None:
    # Hand-authored pixel fixture: a 2x2 L blueprint, side-three seed, and an
    # asymmetric four-pixel stencil. Expected corner pixels are explicitly
    # listed, independently of _corners, _points, and _render.
    observation = np.zeros((11, 21), dtype=np.int8)
    observation[2, 2:9] = observation[8, 2:9] = 8
    observation[2:9, 2] = observation[2:9, 8] = 8
    observation[4, 4], observation[4, 6], observation[6, 4] = 1, 2, 3
    observation[1, 1:4] = 9
    observation[2, 1] = 9
    observation[2:5, 13:16] = 1
    observation[2:5, 16:19] = 2
    observation[1, 13] = 1  # offset partial clue
    expected = observation.copy()
    expected[5:8, 13:16] = 3
    for color, points in (
        (1, ((1, 12), (1, 13), (1, 14), (2, 12))),
        (2, ((1, 19), (1, 18), (1, 17), (2, 19),
             (5, 19), (5, 18), (5, 17), (4, 19))),
        (3, ((8, 12), (8, 13), (8, 14), (7, 12),
             (8, 16), (8, 15), (7, 16))),
    ):
        for p in points:
            expected[p] = color
    assert np.array_equal(solve(observation), expected), "hand-authored contour fixture"
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = {tuple((_key(p["input"]), _key(p["output"]))
                        for split in ("train", "test") for p in puzzle[split])
                  for puzzle in puzzles}
    assert len(signatures) == 42
    records = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pred, m = _recover(pair["input"].copy())
                assert np.array_equal(pred, pair["output"])
                records.append(dict(m, split=split, canvas=pair["input"].shape))
    for feature, minimum in (("canvas", 30), ("pattern", 30), ("origin", 30),
                             ("match", 6), ("side", 4), ("background", 10)):
        assert len({m[feature] for m in records}) >= minimum, feature
    assert {m["axis"] for m in records} == {0, 1}
    assert {m["reference_q"] for m in records} == {0, 1, 2, 3}
    assert {m["radius"] for m in records} == {1, 2}
    assert len({m["plan"].shape for m in records}) >= 8
    assert len({len(m["corners"]) for m in records}) >= 5
    assert any(m["length"] >= 3 for m in records)
    assert any(len(m["fragments"]) >= 2 for m in records)
    assert any(np.count_nonzero(m["plan"] == m["background"]) >= 3 for m in records)
    assert any(not np.any(m["plan"] == m["background"]) for m in records)
    assert any(30 in m["canvas"] for m in records)
    # Exact seed replay, maximal/minimal counts, malformed outputs, invalid
    # counts, and an explicitly ambiguous color-sequence regression.
    a, b = generate_puzzle(912, 2, 1), generate_puzzle(912, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for split in a
               for x, y in zip(a[split], b[split]) for k in x)
    generate_puzzle(913, 10, 4)
    lists = {split: [{k: v.tolist() for k, v in pair.items()} for pair in a[split]] for split in a}
    validate_generated_puzzle(lists)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    plan = np.array([[2, 3], [2, 3], [4, 0]], dtype=np.int8)
    assert len(_matches(plan, (2, 3), 1)) == 2
    ambiguous = _draw(np.random.default_rng(12), plan, 0, 8, 9,
                      frozenset({(-1, -1), (-1, 0), (0, -1)}),
                      3, 1, (0, 0), 2, 1)
    try:
        solve(ambiguous["input"])
    except AssertionError:
        pass
    else:
        raise AssertionError("two color-sequence matches accepted")
    mask = np.array([[True, False], [False, True]])
    assert not _valid_occupancy(mask)
    # A hole is concave on the material side: only the four outer corners.
    ring = np.ones((3, 3), dtype=np.int8)
    ring[1, 1] = 0
    assert len(_corners(ring, 0)) == 4
    damaged = {split: [{k: v.copy() for k, v in pair.items()} for pair in a[split]] for split in a}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
