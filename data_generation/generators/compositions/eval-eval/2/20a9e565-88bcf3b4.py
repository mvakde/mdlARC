"""Patterned cord under tension: arithmetic dye bands in intrinsic cord coordinates.

Only standard-library modules and NumPy are used. No file I/O, caches, saved
examples, parent imports, seed-dependent solving, or execution traces occur here.
"""
from itertools import combinations
import hashlib
import numpy as np


_D8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger grid"
    assert np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _connected(cells):
    if not cells:
        return False
    seen = {min(cells)}
    todo = list(seen)
    while todo:
        u, v = todo.pop()
        for du, dv in _D8:
            q = u + du, v + dv
            if q in cells and q not in seen:
                seen.add(q)
                todo.append(q)
    return seen == cells


def _corners(bb):
    r0, r1, c0, c1 = bb
    return {(r0, c0), (r0 + 1, c0), (r0, c0 + 1),
            (r1, c1), (r1 - 1, c1), (r1, c1 - 1)}


def _marker(a):
    matches = []
    for color in np.unique(a):
        pos = np.argwhere(a == color)
        if len(pos) != 6:
            continue
        r0, c0 = map(int, pos.min(axis=0))
        r1, c1 = map(int, pos.max(axis=0))
        bb = r0, r1, c0, c1
        if min(r1 - r0, c1 - c0) >= 3 and set(map(tuple, pos)) == _corners(bb):
            matches.append((int(color), bb))
    assert len(matches) == 1, "two canonical corner Ls must identify one window"
    return matches[0]


def _global(p, end, f, lateral):
    u, v = p
    return end[0] + u * f[0] + v * lateral[0], end[1] + u * f[1] + v * lateral[1]


def _local(p, end, f, lateral):
    dr, dc = p[0] - end[0], p[1] - end[1]
    return dr * f[0] + dc * f[1], dr * lateral[0] + dc * lateral[1]


def _route(obstacle, n):
    """Input-only contour envelope; one axial step is one material cell."""
    assert obstacle and min(u for u, v in obstacle) >= 2
    peak = max(v for u, v in obstacle)
    assert peak >= 0
    assert peak == 0 or max(v - u for u, v in obstacle) == 0
    first_peak = min(u for u, v in obstacle if v == peak)
    check_n = max(n, max(u for u, v in obstacle) + 4)
    path = [(u, 1 if peak == 0 and u <= first_peak else
             max(v + 1 - abs(u - t) for t, v in obstacle)) for u in range(check_n)]
    slopes = [b[1] - a[1] for a, b in zip(path, path[1:])]
    assert path[0] == (0, 1)
    assert all(s in (-1, 0, 1) for s in slopes)
    assert slopes == sorted(slopes, reverse=True), "unsupported exposed concavity"
    assert not set(path) & obstacle
    return path[:n]


def _construct_route(obstacle, n):
    """Construction by bidirectional clearance propagation, without _route."""
    size = max(n, max(u for u, v in obstacle) + 3)
    y = [-1000] * size
    for u, v in obstacle:
        y[u] = max(y[u], v + 1)
    for u in range(1, size):
        y[u] = max(y[u], y[u - 1] - 1)
    for u in range(size - 2, -1, -1):
        y[u] = max(y[u], y[u + 1] - 1)
    if max(v for u, v in obstacle) == 0:
        k = min(u for u, v in obstacle if v == 0)
        y[:k + 1] = [1] * (k + 1)
    return list(enumerate(y[:n]))


def _fit_prefix(values):
    runs = []
    for value in values:
        if not runs or runs[-1][0] != value:
            runs.append([int(value), 1])
        else:
            runs[-1][1] += 1
    colors = [x[0] for x in runs]
    widths = [x[1] for x in runs]
    palette = list(dict.fromkeys(colors))
    p = len(palette)
    assert p in (2, 3) and len(runs) >= p + 1
    assert all(c == palette[i % p] for i, c in enumerate(colors))
    delta = widths[1] - widths[0]
    assert delta in (0, 1)
    assert all(w == widths[0] + i * delta for i, w in enumerate(widths))
    return dict(palette=tuple(palette), nruns=len(runs), last=widths[-1],
                delta=delta, first=widths[0], widths=tuple(widths))


def _continuation(model, n):
    """Extend observed whole bands; neither obstacle nor crop resets phase."""
    colors = []
    j, w = model["nruns"], model["last"] + model["delta"]
    while len(colors) < n:
        colors.extend([model["palette"][j % len(model["palette"])]] * min(w, n - len(colors)))
        j += 1
        w += model["delta"]
    return colors


def _paint_crop(a, m, *, straight=False, infinite=False, constant=False, flat=False, restart=False):
    r0, r1, c0, c1 = m["bounds"]
    out = a[r0:r1 + 1, c0:c1 + 1].copy()
    out[out == m["marker"]] = m["background"]
    n = m["length"]
    if infinite:
        n = max(n, max(_local(p, m["end"], m["f"], m["l"])[0]
                       for p in ((r0, c0), (r1, c1), (r0, c1), (r1, c0))) + 1)
    path = [(u, 1) for u in range(n)] if straight else _route(m["obstacle"], n)
    pattern = dict(m["pattern"])
    if constant:
        pattern["delta"] = 0
    colors = _continuation(pattern, n)
    if flat:
        colors = [colors[0]] * n
    first = min((u for u, v in path if r0 <= _global((u, v), m["end"], m["f"], m["l"])[0] <= r1
                 and c0 <= _global((u, v), m["end"], m["f"], m["l"])[1] <= c1), default=0)
    for i, q in enumerate(path):
        r, c = _global(q, m["end"], m["f"], m["l"])
        if r0 <= r <= r1 and c0 <= c <= c1 and out[r - r0, c - c0] == m["background"]:
            out[r - r0, c - c0] = colors[i - first if restart else i]
    return out


def _analyze(input_grid):
    a = _grid(input_grid)
    marker, bounds = _marker(a)
    counts = np.bincount(a.ravel(), minlength=10)
    background = int(counts.argmax())
    assert np.count_nonzero(counts == counts.max()) == 1 and background != marker
    groups = {int(c): set(map(tuple, np.argwhere(a == c)))
              for c in np.unique(a) if c not in (background, marker)}
    assert len(groups) in (4, 5)
    found = []
    for p in (2, 3):
        for palette in combinations(sorted(groups), p):
            support = set().union(*(groups[c] for c in palette))
            if len(support) < 3:
                continue
            if len({r for r, c in support}) == 1:
                f0 = (0, 1)
            elif len({c for r, c in support}) == 1:
                f0 = (1, 0)
            else:
                continue
            ordered = sorted(support)
            if any(abs(b[0] - x[0]) + abs(b[1] - x[1]) != 1 for x, b in zip(ordered, ordered[1:])):
                continue
            rest = sorted(set(groups) - set(palette))
            if len(rest) != 2:
                continue
            for sign in (1, -1):
                f = sign * f0[0], sign * f0[1]
                seq = ordered if sign == 1 else ordered[::-1]
                end = seq[-1]
                try:
                    pattern = _fit_prefix([int(a[q]) for q in seq])
                except AssertionError:
                    continue
                for lateral in ((-f[1], f[0]), (f[1], -f[0])):
                    for sc, oc in (rest, rest[::-1]):
                        slack = {_local(q, end, f, lateral) for q in groups[sc]}
                        obs = {_local(q, end, f, lateral) for q in groups[oc]}
                        if (0, 1) not in slack or any(u > 0 or v < 1 for u, v in slack):
                            continue
                        if not _connected(slack) or not _connected(obs):
                            continue
                        r0, r1, c0, c1 = bounds
                        crop_corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
                        if min(_local(q, end, f, lateral)[0] for q in crop_corners) < 1:
                            continue
                        inside = a[r0:r1 + 1, c0:c1 + 1]
                        if not set(map(int, np.unique(inside))) <= {background, marker, oc}:
                            continue
                        try:
                            path = _route(obs, len(slack))
                        except AssertionError:
                            continue
                        found.append(dict(background=background, marker=marker, bounds=bounds,
                                          support=support, end=end, f=f, l=lateral, slack=slack,
                                          obstacle=obs, length=len(slack), obstacle_color=oc,
                                          slack_color=sc, pattern=pattern, path=path))
    assert len(found) == 1, "input must establish unique support, attachment, and obstacle roles"
    m = found[0]
    out = _paint_crop(a, m)
    return out, m


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the selected continuation crop using only the given input."""
    return _analyze(input_grid)[0]


def _obstacle(rng, kind, approach):
    if kind == 0:
        raw = {(0, 0)}
    elif kind == 1:
        n = int(rng.integers(2, 5))
        raw = {(u, 0) for u in range(n)} if rng.random() < .5 else {(0, v) for v in range(n)}
    elif kind == 2:
        h, w = map(int, rng.integers(2, 5, size=2))
        raw = {(u, v) for u in range(h) for v in range(w)}
    elif kind == 3:
        rad = int(rng.integers(1, 3))
        raw = {(u, v) for u in range(2 * rad + 1) for v in range(-rad, rad + 1)
               if abs(u - rad) + abs(v) <= rad}
    else:
        ramp, plateau = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        profile = [ramp - max(0, u - plateau + 1) for u in range(ramp + plateau)]
        raw = {(u, v) for u, hi in enumerate(profile)
               for v in (range(hi + 1) if kind == 4 else (hi,))}
        if rng.random() < .5:
            end = max(u for u, v in raw)
            raw = {(end - u, v) for u, v in raw}
    near = int(rng.integers(2, 5))
    offset = -max(v for u, v in raw) if approach == 0 else near - max(v - u for u, v in raw)
    return {(u + near, v + offset) for u, v in raw}


def _slack(rng, n, style, diagonal_lesson=False):
    if diagonal_lesson:
        rest = _slack(rng, n - 1, style, False)
        return {(0, 1)} | {(u - 1, v + 1) for u, v in rest}
    cells = {(0, 1)}
    depth, width = int(rng.integers(2, 6)), int(rng.integers(3, 8))
    depth = max(depth, (n + width - 1) // width)
    while len(cells) < n:
        choices = sorted({(u + du, v + dv) for u, v in cells for du, dv in _D8
                          if -depth <= u + du <= 0 and 1 <= v + dv <= width} - cells)
        assert choices
        if style == 0:
            weights = [1.0] * len(choices)
        else:
            neighbors = [sum((u + du, v + dv) in cells for du, dv in _D8) for u, v in choices]
            weights = [(x + 1) ** (2 if style == 1 else -2) for x in neighbors]
        weights = np.asarray(weights, dtype=float)
        cells.add(choices[int(rng.choice(len(choices), p=weights / weights.sum()))])
    return cells


def _build_pair(rng, split, index, regime, axis, roles):
    """Sample explicit geometry; construct labels without invoking solve."""
    for attempt in range(160):
        p = int(rng.integers(2, 4))
        nruns = p + int(rng.integers(1, 3))
        delta = (1 if index == 0 else 0) if split == "train" and index < 2 else int(rng.integers(2))
        first = 1 if delta else int(rng.integers(1 if index != 1 else 2, 4))
        widths = [first + delta * j for j in range(nruns)]
        length_support = sum(widths)
        kind, approach = int(rng.integers(6)), int(rng.integers(2))
        obs = _obstacle(rng, kind, approach)
        near, far = min(u for u, v in obs), max(u for u, v in obs)
        n = far + int(rng.integers(7, 10) if split == "test" and regime == 2 else rng.integers(4, 7))
        if length_support + n > 27:
            continue
        try:
            _route(obs, n)
        except AssertionError:
            continue
        slack = _slack(rng, n, int(rng.integers(3)), split == "train" and index == 0)
        # A distant query moves the window, never the pattern's coordinate origin.
        if split == "test" and regime == 1:
            u0 = far + int(rng.integers(1, 3))
        elif not (split == "train" and index < 2) and rng.random() < .45:
            u0 = int(rng.integers(near, far + 1))
        else:
            u0 = max(1, near - int(rng.integers(1, 3)))
        if not (split == "train" and index < 2) and n - 1 >= u0 + 3 and rng.random() < .55:
            u1 = int(rng.integers(u0 + 3, n))
        else:
            u1 = n + int(rng.integers(0, 3))
        if u1 - u0 < 3:
            u0 = u1 - 3
        extended_path = _construct_route(obs, u1 + 1)
        target_v = [v for u, v in extended_path if u0 <= u <= u1]
        v0 = min(target_v) - int(rng.integers(0, 3))
        v1 = max(target_v) + int(rng.integers(1, 4))
        # Sometimes inspect only a transverse slice rather than the whole bow.
        if not (split == "train" and index < 2) and rng.random() < .35 and v1 - v0 >= 6:
            if rng.random() < .5:
                v0 += int(rng.integers(1, 4))
            else:
                v1 -= int(rng.integers(1, 3))
        while v1 - v0 < 3:
            v0 -= 1
        bb = u0, u1, v0, v1
        support = {(u, 0) for u in range(-length_support + 1, 1)}
        occupied = support | slack | obs | _corners(bb)
        lo_u, hi_u = min(u for u, v in occupied), max(u for u, v in occupied)
        lo_v, hi_v = min(v for u, v in occupied), max(v for u, v in occupied)
        if hi_u - lo_u + 1 > 28 or hi_v - lo_v + 1 > 28 or _corners(bb) & obs:
            continue
        pad = list(map(int, rng.integers(0, 3, size=4)))
        h, w = hi_u - lo_u + 1, hi_v - lo_v + 1
        if h + pad[0] + pad[1] > 30 or w + pad[2] + pad[3] > 30:
            continue
        du, dv = pad[0] - lo_u, pad[2] - lo_v
        bg, mark, reservoir, obstacle_color = roles[:4]
        palette = roles[4:4 + p]
        inp = np.full((h + pad[0] + pad[1], w + pad[2] + pad[3]), bg, dtype=np.int8)
        bar = [color for j, width in enumerate(widths) for color in [palette[j % p]] * width]
        for u, color in zip(range(-length_support + 1, 1), bar):
            inp[u + du, dv] = color
        for cells, color in ((slack, reservoir), (obs, obstacle_color)):
            for u, v in cells:
                inp[u + du, v + dv] = color
        out = inp[u0 + du:u1 + du + 1, v0 + dv:v1 + dv + 1].copy()
        suffix = []
        j = nruns
        while len(suffix) < n:
            suffix += [palette[j % p]] * (first + j * delta)
            j += 1
        path = _construct_route(obs, n)
        for (u, v), color in zip(path, suffix):
            if u0 <= u <= u1 and v0 <= v <= v1:
                assert out[u - u0, v - v0] == bg
                out[u - u0, v - v0] = color
        # Transform the scene/window together, then redraw canonical corner Ls.
        mask = np.zeros(inp.shape, dtype=bool)
        mask[u0 + du:u1 + du + 1, v0 + dv:v1 + dv + 1] = True
        if regime == 0:
            k = (axis + (split == "test")) % 2 + 2 * int(rng.integers(2))
        else:
            k = int(rng.integers(4))
        inp, mask, out = np.rot90(inp, k).copy(), np.rot90(mask, k), np.rot90(out, k).copy()
        if rng.random() < .5:
            inp, mask, out = inp[:, ::-1].copy(), mask[:, ::-1], out[:, ::-1].copy()
        pos = np.argwhere(mask)
        r0, c0 = map(int, pos.min(axis=0))
        r1, c1 = map(int, pos.max(axis=0))
        if any(inp[q] != bg for q in _corners((r0, r1, c0, c1))):
            continue
        for q in _corners((r0, r1, c0, c1)):
            inp[q] = mark
        try:
            predicted, m = _analyze(inp)
            assert np.array_equal(predicted, out)
            colored = sum(int(c) in palette for c in out.flat)
            assert colored >= 2 and np.any(out == bg)
            if split == "train" and index < 2:
                assert not np.array_equal(out, _paint_crop(inp, m, straight=True))
                assert not np.array_equal(out, _paint_crop(inp, m, infinite=True))
            if split == "train" and index == 0:
                assert len(set(map(int, np.unique(out))) & set(palette)) >= 2
                assert not np.array_equal(out, _paint_crop(inp, m, constant=True))
        except AssertionError:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded geometric sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name + " is outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = list(map(int, rng.permutation(10)))
    stable_roles = bool(rng.integers(2))
    regime, axis = int(rng.integers(3)), int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            pair_roles = roles if stable_roles else list(map(int, rng.permutation(10)))
            for retry in range(12):
                pair = _build_pair(rng, split, i, regime, axis, pair_roles)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input sampling exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, lessons = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            expected, m = _analyze(inp)
            assert np.array_equal(expected, out), "wrong patterned contour crop"
            assert not np.any(out == m["marker"])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            r0, r1, c0, c1 = m["bounds"]
            assert out.shape == (r1 - r0 + 1, c1 - c0 + 1)
            palette = set(m["pattern"]["palette"])
            visible = []
            for q in m["path"]:
                r, c = _global(q, m["end"], m["f"], m["l"])
                if r0 <= r <= r1 and c0 <= c <= c1:
                    visible.append((r - r0, c - c0))
            assert len(visible) >= 2
            assert sum(int(v) in palette for v in out.flat) == len(visible)
            assert np.any(out == m["background"])
            for q in m["obstacle"]:
                r, c = _global(q, m["end"], m["f"], m["l"])
                if r0 <= r <= r1 and c0 <= c <= c1:
                    assert out[r - r0, c - c0] == m["obstacle_color"]
            if split == "train":
                lessons.append((m["pattern"]["delta"],
                                not np.array_equal(out, _paint_crop(inp, m, constant=True)),
                                not np.array_equal(out, _paint_crop(inp, m, flat=True)),
                                not np.array_equal(out, _paint_crop(inp, m, straight=True)),
                                not np.array_equal(out, _paint_crop(inp, m, infinite=True))))
    assert {x[0] for x in lessons} == {0, 1}, "show constant and growing bands"
    assert all(any(x[i] for x in lessons) for i in range(1, 5)), "missing counterexample to a shortcut"


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _check_slack_shape_invariance(inp, out, m):
    """Move a slack cell while retaining its count, attachment and connectivity."""
    slack = m["slack"]
    frontier = sorted({(u + du, v + dv) for u, v in slack for du, dv in _D8
                       if u + du <= 0 and v + dv >= 1} - slack)
    for old in sorted(slack - {(0, 1)}):
        for new in frontier:
            r, c = _global(new, m["end"], m["f"], m["l"])
            if not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]):
                continue
            if inp[r, c] != m["background"] or not _connected((slack - {old}) | {new}):
                continue
            changed = inp.copy()
            changed[_global(old, m["end"], m["f"], m["l"])] = m["background"]
            changed[r, c] = m["slack_color"]
            assert np.array_equal(solve(changed), out), "slack shape affected answer"
            return
    raise AssertionError("could not construct slack-shape regression")


def self_test():
    """Bounded: 42 streamed puzzles, all count pairs, and semantic regressions."""
    route_cases = [({(2, 0)}, [(0, 1), (1, 1), (2, 1), (3, 0), (4, -1), (5, -2)]),
                   ({(u, v) for u in (2, 3) for v in (0, 1, 2)},
                    [(0, 1), (1, 2), (2, 3), (3, 3), (4, 2), (5, 1)]),
                   ({(2, 2), (3, 3), (4, 2)},
                    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 3), (5, 2)])]
    for obstacle, expected in route_cases:
        assert _route(obstacle, 6) == expected == _construct_route(obstacle, 6)
    pattern = _fit_prefix([2, 3, 3, 2, 2, 2])
    assert _continuation(pattern, 10) == [3] * 4 + [2] * 5 + [3]
    signatures, shapes, directions, lengths, palettes, silhouettes, widths, crops = (set() for _ in range(8))
    metrics = dict(pairs=0, growing=0, constant=0, parallel=0, diagonal=0,
                   irregular=0, growth_matters=0, route_matters=0, count_matters=0,
                   phase_matters=0, partial_obstacle=0, crop_ends_on_cord=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                expected, m = _analyze(inp)
                assert np.array_equal(inp, before) and np.array_equal(expected, out)
                if seed == 0 and split == "train":
                    _check_slack_shape_invariance(inp, out, m)
                    recolor = np.array([9 - c for c in range(10)], dtype=np.int8)
                    assert np.array_equal(solve(recolor[inp]), recolor[out])
                shapes.add((inp.shape, out.shape))
                directions.add(m["f"])
                lengths.add(m["length"])
                palettes.add(m["pattern"]["palette"])
                widths.add(m["pattern"]["widths"])
                crops.add(out.shape)
                obs = m["obstacle"]
                umin, vmin = min(u for u, v in obs), min(v for u, v in obs)
                umax, vmax = max(u for u, v in obs), max(v for u, v in obs)
                silhouettes.add(tuple(sorted((u - umin, v - vmin) for u, v in obs)))
                metrics["pairs"] += 1
                metrics["growing" if m["pattern"]["delta"] else "constant"] += 1
                metrics["parallel" if vmax == 0 else "diagonal"] += 1
                metrics["irregular"] += len(obs) < (umax - umin + 1) * (vmax - vmin + 1)
                r0, r1, c0, c1 = m["bounds"]
                last_u = max(_local(q, m["end"], m["f"], m["l"])[0]
                             for q in ((r0, c0), (r1, c1)))
                metrics["crop_ends_on_cord"] += any(u == last_u and
                    r0 <= _global((u, v), m["end"], m["f"], m["l"])[0] <= r1 and
                    c0 <= _global((u, v), m["end"], m["f"], m["l"])[1] <= c1 for u, v in m["path"])
                for key, kw in (("growth_matters", {"constant": True}), ("route_matters", {"straight": True}),
                                ("count_matters", {"infinite": True}), ("phase_matters", {"restart": True})):
                    metrics[key] += not np.array_equal(out, _paint_crop(inp, m, **kw))
                metrics["partial_obstacle"] += 0 < np.count_nonzero(out == m["obstacle_color"]) < len(obs)
    assert len(signatures) == 42 and len(shapes) >= 100
    assert len(directions) == 4 and len(lengths) >= 8 and len(palettes) >= 20
    assert len(silhouettes) >= 15 and len(widths) >= 8 and len(crops) >= 35
    assert all(metrics[k] >= 10 for k in metrics if k != "partial_obstacle"), metrics
    assert metrics["partial_obstacle"] >= 1
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(812 + nt * 5 + nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("invalid counts accepted")
    damaged = generate_puzzle(981, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return {**{k: int(v) for k, v in metrics.items()}, "puzzles": len(signatures),
                "shape_pairs": len(shapes), "directions": len(directions),
                "lengths": len(lengths), "band_palettes": len(palettes),
                "silhouettes": len(silhouettes), "width_histories": len(widths),
                "crop_shapes": len(crops)}
    raise AssertionError("corrupted output accepted")
