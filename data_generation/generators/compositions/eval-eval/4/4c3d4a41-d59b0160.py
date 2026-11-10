"""Roof-triggered fuses: query the prefix expelled by an inserted comb.

Only visible geometry and literal within-scene color equality affect the rule.
The constructor and input-only solver use separate output algorithms.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _orient(grid, code):
    return np.rot90(np.fliplr(grid) if code >= 4 else grid, code % 4).copy()


def _unorient(grid, code):
    result = np.rot90(grid, -(code % 4))
    return (np.fliplr(result) if code >= 4 else result).copy()


def _runs(mask):
    points = np.flatnonzero(mask)
    if not len(points):
        return []
    groups = np.split(points, np.flatnonzero(np.diff(points) > 1) + 1)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups]


def _selector(grid):
    """Find the isolated seven-cell L, wherever it is placed or reflected."""
    candidates = []
    for color in np.unique(grid):
        points = np.argwhere(grid == color)
        if len(points) != 7:
            continue
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0) + 1
        if (r1 - r0, c1 - c0) != (4, 4):
            continue
        block = grid[r0:r1, c0:c1]
        for corner in range(4):
            view = block[::-1, :] if corner & 2 else block
            view = view[:, ::-1] if corner & 1 else view
            border = np.zeros((4, 4), dtype=bool)
            border[3, :] = border[:, 3] = True
            if not np.array_equal(view == color, border):
                continue
            values, counts = np.unique(view[:3, :3], return_counts=True)
            bg = values[counts == 6]
            if len(bg) != 1 or len(values) != 4:
                continue
            background = int(bg[0])
            query = frozenset(int(v) for v in values if v != background)
            if int(color) == background or int(color) in query:
                continue
            candidates.append(dict(background=background, frame=int(color), query=query,
                                   selector=(int(r0), int(r1), int(c0), int(c1))))
    if len(candidates) != 1:
        raise ValueError("Expected one isolated three-color L selector")
    return candidates[0]


def _walls(grid, bg, frame):
    runs = {}
    for row, values in enumerate(grid):
        for color in np.unique(values):
            if color in (bg, frame):
                continue
            for left, stop in _runs(values == color):
                if stop - left >= 5:
                    runs.setdefault((int(color), left, stop - 1), []).append(row)
    return [(color, left, right, top, bottom)
            for (color, left, right), rows in runs.items()
            for top in rows for bottom in rows if bottom > top]


def _draw_housing(grid, f):
    t, b, l, r, s = (f[k] for k in ("top", "bottom", "left", "right", "structure"))
    grid[t, l:r + 1] = grid[b, l:r + 1] = s
    grid[t:b + 1, l] = grid[t:b + 1, r] = s
    grid[f["base"] - 1:f["base"] + 1, l] = f["background"]


def _render_mechanism(f):
    grid = np.full(f["shape"], f["background"], dtype=np.int8)
    _draw_housing(grid, f)
    base, kl, kw, s = (f[k] for k in ("base", "key_left", "key_width", "structure"))
    grid[base, kl:kl + kw] = s
    for lane in f["lanes"]:
        off, width, height, seq = (lane[k] for k in ("offset", "width", "height", "sequence"))
        grid[base - height:base, kl + off:kl + off + width] = s
        col = f["first_lane"] + off
        grid[f["top"] + 1:f["top"] + len(seq) + 1, col:col + width] = np.array(seq)[:, None]
    return grid


def _render_input(f):
    grid = _render_mechanism(f)
    r0, r1, c0, c1 = f["selector"]
    grid[r0:r1, c0:c1] = f["selector_pixels"]
    return grid


def _read_canonical(grid, selector, wall):
    s, left, right, top, bottom = wall
    bg, frame = selector["background"], selector["frame"]
    assert s not in selector["query"]
    assert np.all(grid[top:bottom + 1, right] == s)
    edge = grid[top + 1:bottom, left]
    assert np.all((edge == s) | (edge == bg))
    opening = np.flatnonzero(edge == bg) + top + 1
    assert len(opening) == 2 and opening[1] == opening[0] + 1
    base = int(opening[1])
    assert base - top >= 3 and bottom - base >= 2
    key_runs = _runs(grid[base, :left] == s)
    assert len(key_runs) == 1
    kl, stop = key_runs[0]
    assert stop < left and stop - kl >= 3
    teeth = _runs(grid[base - 1, kl:stop] == s)
    assert len(teeth) >= 2 and teeth[0][0] == 0 and teeth[-1][1] == stop - kl
    pin_runs = _runs(grid[top + 1, left + 1:right] != bg)
    pin_runs = [(a + left + 1, b + left + 1) for a, b in pin_runs]
    assert len(pin_runs) == len(teeth)
    first = pin_runs[0][0]
    assert first >= left + 2 and pin_runs[-1][1] <= right - 1
    lanes = []
    for (a, b), (p, q) in zip(teeth, pin_runs):
        assert (p - first, q - first) == (a, b)
        row = base - 1
        while row >= 0 and np.all(grid[row, kl + a:kl + b] == s):
            row -= 1
        height = base - row - 1
        assert 1 <= height <= base - top - 2
        strip = grid[top + 1:base, p:q]
        occupied = np.flatnonzero(np.any(strip != bg, axis=1))
        assert len(occupied) and np.array_equal(occupied, np.arange(len(occupied)))
        content = strip[:len(occupied)]
        assert np.all(content != bg) and np.all(content == content[:, :1])
        seq = tuple(int(v) for v in content[:, 0])
        assert frame not in seq
        capacity = base - top - height - 1
        assert len(seq) >= capacity  # Contact or overlap; no downward motion.
        lanes.append(dict(offset=a, width=b - a, height=height, sequence=seq))
    f = dict(selector, structure=s, shape=grid.shape, top=top, bottom=bottom,
             left=left, right=right, base=base, key_left=kl, key_width=stop - kl,
             first_lane=first, lanes=lanes)
    r0, r1, c0, c1 = f["selector"]
    f["selector_pixels"] = grid[r0:r1, c0:c1].copy()
    assert r1 <= top or r0 > bottom or c1 <= left or c0 > right, "Selector must be outside the housing"
    # The selector's complete box must have a background moat.
    rr0, rr1 = max(0, r0 - 1), min(grid.shape[0], r1 + 1)
    cc0, cc1 = max(0, c0 - 1), min(grid.shape[1], c1 + 1)
    moat = grid[rr0:rr1, cc0:cc1].copy()
    moat[r0 - rr0:r1 - rr0, c0 - cc0:c1 - cc0] = bg
    assert np.all(moat == bg)
    assert np.array_equal(_render_input(f), grid), "Unexpected cells or overlapping objects"
    return f


def _analyze(input_grid):
    a = np.asarray(input_grid)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Input must be an integer 2D grid")
    if not all(1 <= n <= 30 for n in a.shape) or np.any((a < 0) | (a > 9)):
        raise ValueError("Input dimensions/colors outside ARC bounds")
    local = _selector(a)
    candidates = []
    for code in range(8):
        view = _unorient(a, code)
        info = _selector(view)
        for wall in _walls(view, local["background"], local["frame"]):
            try:
                f = _read_canonical(view, info, wall)
            except (AssertionError, ValueError, IndexError):
                continue
            f["orientation"] = code
            candidates.append(f)
    if len(candidates) != 1:
        raise ValueError("Expected one comb/housing interpretation")
    return candidates[0]


def _motion(f, lane):
    capacity = f["base"] - f["top"] - lane["height"] - 1
    seq = lane["sequence"]
    shift = len(seq) - capacity
    expelled = seq[:shift]
    return shift, expelled, f["query"] <= set(expelled)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Push actual input strips one cell at a time, inspecting expelled rows."""
    f = _analyze(input_grid)
    a = _unorient(np.asarray(input_grid), f["orientation"])
    out = a.copy()
    bg, s, roof, base = (f[k] for k in ("background", "structure", "top", "base"))
    kl, kw, first = (f[k] for k in ("key_left", "key_width", "first_lane"))
    key_mask = np.zeros(a.shape, dtype=bool)
    key_mask[base, kl:kl + kw] = True
    for lane in f["lanes"]:
        col = kl + lane["offset"]
        key_mask[base - lane["height"]:base, col:col + lane["width"]] = True
    key = np.argwhere(key_mask & (a == s))
    for row, col in key:
        out[row, col] = bg
    for lane in f["lanes"]:
        col, width = first + lane["offset"], lane["width"]
        out[roof + 1:base, col:col + width] = bg
    for row, col in key:
        out[row, col + first - kl] = s
    for lane in f["lanes"]:
        col, width = first + lane["offset"], lane["width"]
        cells = [(r, int(a[r, col])) for r in range(roof + 1, base) if a[r, col] != bg]
        top_of_tooth = base - lane["height"]
        expelled = []
        while cells[-1][0] >= top_of_tooth:
            shifted = [(r - 1, color) for r, color in cells]
            expelled.extend(color for r, color in shifted if r <= roof)
            cells = [(r, color) for r, color in shifted if r > roof]
            assert cells
        assert cells[-1][0] == top_of_tooth - 1
        if not f["query"] <= set(expelled):
            for row, color in cells:
                out[row, col:col + width] = color
    return _orient(out, f["orientation"])


def _construct_output(f):
    """Truth uses a closed-form prefix cut and direct destination coordinates."""
    out = _render_input(f)
    s, bg, base = (f[k] for k in ("structure", "background", "base"))
    kl, kw, first = (f[k] for k in ("key_left", "key_width", "first_lane"))
    out[base, kl:kl + kw] = bg
    out[base, first:first + kw] = s
    for lane in f["lanes"]:
        off, width, height = (lane[k] for k in ("offset", "width", "height"))
        out[base - height:base, kl + off:kl + off + width] = bg
        col = first + off
        out[f["top"] + 1:base, col:col + width] = bg
        shift, _, erase = _motion(f, lane)
        if not erase:
            suffix = lane["sequence"][shift:]
            out[f["top"] + 1:base - height, col:col + width] = np.array(suffix)[:, None]
        out[base - height:base, col:col + width] = s
    return out


def _geometry(rng, n):
    # Two copies of the comb must fit; independently vary widths and gaps.
    for _ in range(60):
        widths = rng.choice((1, 2), n, p=(.72, .28)).tolist()
        gaps = rng.integers(1, 4, n - 1).tolist()
        kw = sum(widths) + sum(gaps)
        if kw <= 11:
            break
    else:
        widths, gaps, kw = [1] * n, [1] * (n - 1), 2 * n - 1
    budget = 30 - (2 * kw + 5)
    extras = [0] * 5
    for i in rng.permutation(5):
        extras[i] = int(rng.integers(0, min((2, 3, 1, 1, 2)[i], budget) + 1))
        budget -= extras[i]
    kl, extra_gap, il, ir, rm = extras
    left = kl + kw + 1 + extra_gap
    # Leave space for a four-column selector below the external comb.
    if left < 5:
        left = 5
    first = left + 2 + il
    right = first + kw - 1 + 2 + ir
    rm = min(rm, 29 - right)
    assert right < 30
    offsets, cursor = [], 0
    for i, width in enumerate(widths):
        offsets.append(cursor)
        cursor += width + (gaps[i] if i < n - 1 else 0)
    return dict(key_left=kl, key_width=kw, left=left, right=right,
                first_lane=first, widths=widths, offsets=offsets, canvas_width=right + rm + 1)


def _sample_lane(rng, span, q, extras, s, kind):
    # The four teaching roles do not occupy fixed lane positions.
    if kind == "exact" or kind.startswith("miss"):
        height = int(rng.integers(3, span - 1))
        capacity, shift = span - height - 1, 3
        prefix = list(q) if kind == "exact" else [c for i, c in enumerate(q) if i != int(kind[-1])] + [extras[0]]
        suffix = [q[int(kind[-1])]] if kind.startswith("miss") else []
    elif kind == "extras":
        height = int(rng.integers(6, span - 1))
        capacity = span - height - 1
        shift = int(rng.integers(6, height + 1))
        prefix = list(q) + [extras[0], q[0], extras[0]]
        suffix = []
    elif kind == "below":
        height = int(rng.integers(4, span - 1))
        capacity = span - height - 1
        shift = int(rng.integers(4, height + 1))
        prefix = [q[0], q[1], extras[0], extras[1]]
        suffix = [q[2]]
    elif kind == "stationary":
        height = int(rng.integers(1, span - 4))
        capacity, shift = span - height - 1, 0
        prefix, suffix = [], list(q) + [extras[0]]
    elif kind == "smallshift":
        height = int(rng.integers(1, span - 1))
        capacity = span - height - 1
        shift = int(rng.integers(1, min(2, height) + 1))
        prefix, suffix = [q[0]] * shift, [q[1]]
    else:
        height = int(rng.integers(1, span - 1))
        capacity = span - height - 1
        shift = int(rng.integers(0, height + 1))
        labels = [s] + list(q) + list(extras)
        prefix = rng.choice(labels, shift).tolist()
        suffix = rng.choice(labels, capacity).tolist()
        if rng.random() < .16:
            prefix, suffix = [s] * shift, [s] * capacity
        elif shift >= 3 and rng.random() < .35:
            prefix[:3] = list(q)
    while len(prefix) < shift:
        prefix.append(int(rng.choice(prefix)))
    # Teaching negatives may contain MORE marks and identities than positives.
    rng.shuffle(prefix)
    palette = [s] + list(q) + list(extras)
    while len(suffix) < capacity:
        suffix.append(int(rng.choice(palette)))
    rng.shuffle(suffix)
    return dict(height=height, sequence=tuple(int(v) for v in prefix + suffix))


def _place_selector(rng, f, q, frame):
    grid = _render_mechanism(f)
    options = []
    for r in range(grid.shape[0] - 3):
        for c in range(grid.shape[1] - 3):
            if not (r + 4 <= f["top"] or r > f["bottom"] or
                    c + 4 <= f["left"] or c > f["right"]):
                continue
            region = grid[max(0, r - 1):min(grid.shape[0], r + 5),
                          max(0, c - 1):min(grid.shape[1], c + 5)]
            if np.all(region == f["background"]):
                options.append((r, c))
    assert options, "Selector requires a separated empty four-by-four region"
    r, c = options[int(rng.integers(len(options)))]
    block = np.full((4, 4), f["background"], dtype=np.int8)
    block[3, :] = block[:, 3] = frame
    positions = rng.choice(9, 3, replace=False)
    for pos, color in zip(positions, rng.permutation(q)):
        block[int(pos) // 3, int(pos) % 3] = color
    corner = int(rng.integers(4))
    if corner & 1:
        block = block[:, ::-1]
    if corner & 2:
        block = block[::-1, :]
    f["selector"], f["selector_pixels"] = (r, r + 4, c, c + 4), block.copy()


def _make_features(rng, n, span, colors, q, package, code):
    bg, frame, s = colors[:3]
    extras = [c for c in colors[3:] if c not in q]
    g = _geometry(rng, n)
    top = int(rng.integers(0, min(3, 24 - span) + 1))
    base = top + span
    bottom = base + int(rng.integers(2, 5))
    height = min(30, max(base + 6, bottom + 1 + int(rng.integers(0, 4))))
    if package == "identity":
        kinds = ["exact", "miss0", "miss1", "miss2"] + ["free"] * (n - 4)
    elif package == "mechanics":
        kinds = ["extras", "below", "stationary", "smallshift"] + ["free"] * (n - 4)
    else:
        kinds = ["free"] * n
    rng.shuffle(kinds)
    lanes = [_sample_lane(rng, span, q, extras, s, k) for k in kinds]
    if package == "identity":
        # Identical cross-sections, lengths and shifts isolate color identity.
        h = int(rng.integers(3, min(6, span - 4) + 1))
        cap = span - h - 1
        for lane, kind in zip(lanes, kinds):
            if kind != "free":
                prefix = lane["sequence"][:3]
                suffix = list(lane["sequence"][3:])
                if kind.startswith("miss"):
                    suffix = [q[int(kind[-1])], s] + suffix
                suffix = (suffix + [s] * cap)[:cap]
                lane["height"], lane["sequence"] = h, tuple(prefix) + tuple(suffix)
        # Match at least one positive/negative width as well as length/shift.
        pos = kinds.index("exact")
        neg = kinds.index("miss0")
        if g["widths"][pos] != g["widths"][neg]:
            g = _geometry(rng, n)
            # All thin lanes is a deterministic, compact witness fallback.
            if g["widths"][pos] != g["widths"][neg]:
                g["widths"] = [1] * n
                g["offsets"] = [2 * i for i in range(n)]
                g["key_width"] = 2 * n - 1
    for lane, offset, width in zip(lanes, g["offsets"], g["widths"]):
        lane.update(offset=offset, width=width)
    f = dict(background=bg, frame=frame, structure=s, query=frozenset(q),
             top=top, base=base, bottom=bottom, lanes=lanes, orientation=code,
             shape=(height, g.pop("canvas_width")))
    f.update({k: v for k, v in g.items() if k not in ("widths", "offsets")})
    _place_selector(rng, f, q, frame)
    return f


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("depth", "lanes", "query")))
    base_colors = [int(v) for v in rng.permutation(10)]
    base_colors = base_colors[:int(rng.integers(8, 11))]
    change_roles = shift != "query" and rng.random() < .65
    mixed_orientation = bool(rng.random() < .6)
    first_code = int(rng.integers(8))
    other_code = (first_code + int(rng.integers(1, 8))) % 8
    familiar_queries = (tuple(base_colors[3:6]),
                        (base_colors[3], base_colors[4], base_colors[6]))
    new_query = (base_colors[3], base_colors[5], base_colors[6])
    seen, train_codes = set(), []
    first_roles = None
    puzzle = {"train": [], "test": []}
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            package = ("identity" if index == 0 else "mechanics" if index == 1 else "free") if split == "train" else "free"
            for attempt in range(64):
                colors = base_colors
                if change_roles:
                    colors = [int(v) for v in rng.permutation(10)][:int(rng.integers(8, 11))]
                    if split == "train" and index == 1 and tuple(colors[:3]) == first_roles:
                        colors[0], colors[1] = colors[1], colors[0]
                if shift == "query":
                    q = familiar_queries[index % 2] if split == "train" else new_query
                else:
                    q = tuple(int(v) for v in rng.choice(colors[3:], 3, replace=False))
                if split == "train":
                    code = (first_code if index == 0 else other_code if index == 1 else int(rng.integers(8))) if mixed_orientation else first_code
                else:
                    code = int(rng.choice(train_codes))
                if shift == "depth":
                    span = int(rng.integers(11 if package == "mechanics" else 8, 16)) if split == "train" else int(rng.integers(17, 21))
                else:
                    span = int(rng.integers(11 if package == "mechanics" else 8, 18))
                if shift == "lanes":
                    n = 4 if split == "train" else int(rng.choice((2, 3, 5, 6)))
                else:
                    n = int(rng.integers(4, 7)) if package != "free" else int(rng.integers(2, 7))
                f = _make_features(rng, n, span, colors, q, package, code)
                inp = _orient(_render_input(f), code)
                key = _key(inp)
                if key in seen:
                    continue
                # Rare symmetric/malformed geometric coincidences are resampled.
                try:
                    _analyze(inp)
                except ValueError:
                    continue
                seen.add(key)
                puzzle[split].append({"input": inp, "output": _orient(_construct_output(f), code)})
                if split == "train":
                    train_codes.append(code)
                    if index == 0:
                        first_roles = tuple(colors[:3])
                break
            else:
                raise RuntimeError("Distinct, uniquely readable scene sampling exhausted")
        rng.shuffle(puzzle[split])
    return puzzle


def _training_coverage(infos):
    records = [(f, lane, *_motion(f, lane)) for f in infos for lane in f["lanes"]]
    positives = [r for r in records if r[-1]]
    negatives = [r for r in records if not r[-1]]
    assert positives and negatives
    assert any(d == 3 and set(prefix) == f["query"] for f, lane, d, prefix, erase in positives)
    assert any(f["query"] < set(prefix) and len(prefix) > len(set(prefix))
               for f, lane, d, prefix, erase in positives)
    assert any(len(set(prefix)) >= 4 for f, lane, d, prefix, erase in negatives)
    assert any(d == 0 for f, lane, d, prefix, erase in negatives)
    assert any(0 < d < 3 for f, lane, d, prefix, erase in negatives)
    assert any(f["query"] <= set(lane["sequence"]) and 0 < d and not erase
               for f, lane, d, prefix, erase in negatives)
    assert any(0 < d and len(set(lane["sequence"][d:])) >= 2
               for f, lane, d, prefix, erase in negatives), "Demonstrate ordered multicolor motion"
    assert any(all(any(not _motion(f, lane)[2] and set(_motion(f, lane)[1]) & f["query"] == f["query"] - {q}
                      for lane in f["lanes"]) for q in f["query"]) for f in infos)
    assert any(d == nd and lane["width"] == nl["width"] and lane["height"] == nl["height"]
               and len(lane["sequence"]) == len(nl["sequence"]) and len(set(prefix)) == len(set(nprefix))
               for f, lane, d, prefix, erase in positives
               for nf, nl, nd, nprefix, nerase in negatives)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            arrays = {k: np.asarray(v) for k, v in pair.items()}
            for a in arrays.values():
                assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
                assert np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))
            inp, out = arrays["input"], arrays["output"]
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _analyze(inp)
            actual = _unorient(out, f["orientation"])
            assert np.array_equal(solve(inp), out)
            assert not np.array_equal(inp, out)
            r0, r1, c0, c1 = f["selector"]
            assert np.array_equal(actual[r0:r1, c0:c1], f["selector_pixels"])
            assert np.all(actual[f["top"], f["left"]:f["right"] + 1] == f["structure"])
            for lane in f["lanes"]:
                col, width = f["first_lane"] + lane["offset"], lane["width"]
                cut, expelled, erase = _motion(f, lane)
                survivor = actual[f["top"] + 1:f["base"] - lane["height"], col:col + width]
                if erase:
                    assert np.all(survivor == f["background"])
                else:
                    assert np.array_equal(survivor, np.repeat(np.array(lane["sequence"][cut:])[:, None], width, axis=1))
            if split == "train":
                training.append(f)
    _training_coverage(training)


def _counterfactual_checks(puzzle):
    # Query-only change reverses a matched positive and negative, keeping all
    # mechanism pixels fixed. Tooth-only change changes the inspection region.
    for pair in puzzle["train"]:
        f = _analyze(pair["input"])
        exact = [l for l in f["lanes"] if _motion(f, l)[2] and _motion(f, l)[0] == 3]
        hard = [l for l in f["lanes"] if not _motion(f, l)[2] and
                len(set(_motion(f, l)[1])) == 3 and len(set(_motion(f, l)[1]) & f["query"]) == 2]
        if not exact or not hard:
            continue
        a = _unorient(pair["input"], f["orientation"])
        changed = a.copy()
        r0, r1, c0, c1 = f["selector"]
        block = changed[r0:r1, c0:c1]
        points = np.argwhere((block != f["background"]) & (block != f["frame"]))
        for point, color in zip(points, sorted(set(_motion(f, hard[0])[1]))):
            block[tuple(point)] = color
        result = _unorient(solve(_orient(changed, f["orientation"])), f["orientation"])
        for lane, should_erase in ((exact[0], False), (hard[0], True)):
            col = f["first_lane"] + lane["offset"]
            region = result[f["top"] + 1:f["base"] - lane["height"], col]
            assert bool(np.all(region == f["background"])) == should_erase
        # Reduce a deleted lane's tooth by one. Exactly one query identity no
        # longer reaches the roof; the same full sequence now survives.
        lane = exact[0]
        changed = a.copy()
        row = f["base"] - lane["height"]
        col = f["key_left"] + lane["offset"]
        changed[row, col:col + lane["width"]] = f["background"]
        result = _unorient(solve(_orient(changed, f["orientation"])), f["orientation"])
        col = f["first_lane"] + lane["offset"]
        assert np.array_equal(result[f["top"] + 1:f["base"] - lane["height"] + 1, col],
                              np.array(lane["sequence"][2:]))
        # Selector positions have no spatial relationship to lane identities.
        changed = a.copy()
        block = changed[r0:r1, c0:c1]
        colors = [int(block[tuple(p)]) for p in points]
        for p, color in zip(points, colors[1:] + colors[:1]):
            block[tuple(p)] = color
        expected = _unorient(pair["output"], f["orientation"])
        expected[r0:r1, c0:c1] = block
        assert np.array_equal(solve(_orient(changed, f["orientation"])), _orient(expected, f["orientation"]))
        return
    raise AssertionError("Missing matched counterfactual witness")


def self_test():
    """Bounded sweep: 42 seeds, semantic witnesses and structural variation."""
    signatures, shapes, layouts, profiles, palettes = set(), set(), set(), set(), set()
    counts, widths, shifts, spans, codes = set(), set(), set(), set(), set()
    states = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _counterfactual_checks(puzzle)
        signatures.add(tuple(_key(p["input"]) for split in puzzle for p in puzzle[split]))
        for split in puzzle:
            for pair in puzzle[split]:
                f = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                layouts.add((f["shape"], f["top"], f["base"], f["left"], f["selector"],
                             tuple((l["offset"], l["width"]) for l in f["lanes"])))
                profiles.add(tuple(l["height"] for l in f["lanes"]))
                palettes.add(len(np.unique(pair["input"])))
                counts.add(len(f["lanes"]))
                spans.add(f["base"] - f["top"])
                codes.add(f["orientation"])
                for lane in f["lanes"]:
                    d, prefix, erase = _motion(f, lane)
                    widths.add(lane["width"])
                    shifts.add(d)
                    states["erase" if erase else "retain"] += 1
                    states["stationary"] += d == 0
                    states["full_match_retained"] += not erase and f["query"] <= set(lane["sequence"])
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for split in puzzle for p, q in zip(puzzle[split], again[split]) for k in p)
    assert len(signatures) == 42 and len(shapes) >= 35 and len(layouts) >= 150
    assert len(profiles) >= 80 and len(palettes) >= 3
    assert counts == set(range(2, 7)) and widths == {1, 2} and codes == set(range(8))
    assert len(spans) >= 10 and len(shifts) >= 10
    assert all(states[k] >= 42 for k in ("erase", "retain", "stationary", "full_match_retained"))
    for a, b in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(77, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Mutation of an output must be caught by semantic validation.
    bad = generate_puzzle(0, 2, 1)
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "layouts": len(layouts), "profiles": len(profiles), "palette_sizes": sorted(palettes),
            "lane_counts": sorted(counts), "strip_widths": sorted(widths),
            "shifts": sorted(shifts), "spans": sorted(spans), "orientations": sorted(codes),
            "lane_states": dict(states)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
