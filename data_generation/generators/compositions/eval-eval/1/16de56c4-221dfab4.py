"""Cadenced shutters: a measured scan captures its first complete object cover.

No file I/O, parent imports, puzzle cache, or answer-affecting hidden state.
Generation and tests retain only bounded in-memory arrays and small statistics.
"""

import hashlib

import numpy as np


_OMISSIONS = ("fixed_two", "ignore_latch", "off_step", "mixed_cover",
              "last_latch", "strip_only", "keep_gaps", "reference_paint")
_WITNESSES = ("stopped", "unstopped", "false_off_step", "false_mixed",
              "later_other_color", "clear_object", "remote_change",
              "outside_background", "split_component", "past_stop")


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=False)


def _parse(input_grid):
    """Recover geometry solely from the majority and two-bar color footprint."""
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    if counts.max() <= grid.size / 2:
        raise ValueError("a strict majority background is required")
    bg = int(values[np.argmax(counts)])
    candidates = []
    for turn in range(4):
        g = np.rot90(grid, -turn)
        depth, span = g.shape
        for color in values:
            ref = int(color)
            if ref == bg:
                continue
            rr, cc = np.nonzero(g == ref)
            rows = np.unique(rr)
            if len(rows) != 2 or rows[0] != 0:
                continue
            p = int(rows[1])
            if 2 * p >= depth:
                continue
            top = np.flatnonzero(g[0] == ref)
            inner = np.flatnonzero(g[p] == ref)
            if len(top) < 2 or not np.array_equal(top, inner):
                continue
            start, end = int(top[0]), int(top[-1]) + 1
            if not (0 < start < end < span):
                continue
            if not np.array_equal(top, np.arange(start, end)):
                continue
            candidates.append(dict(g=g, turn=turn, bg=bg, ref=ref,
                                   p=p, start=start, end=end))
    if len(candidates) != 1:
        raise ValueError("expected exactly one unambiguous paired edge bar")
    return candidates[0]


def _covers(info, p=None, off_step=False, mixed=False):
    g, bg, ref = info["g"], info["bg"], info["ref"]
    p = info["p"] if p is None else p
    rows = range(info["p"] + 1, len(g)) if off_step else range(2 * p, len(g), p)
    candidates = []
    for d in rows:
        line = g[d, info["start"]:info["end"]]
        if np.all((line != bg) & (line != ref)) and (mixed or np.all(line == line[0])):
            candidates.append((d, int(line[0])))
    return candidates


def _paint(info, variant=None):
    """Operational painter, also used for explicitly named omission checks."""
    g = info["g"]
    p = 2 if variant == "fixed_two" else info["p"]
    covers = [] if variant == "ignore_latch" else _covers(
        info, p=p, off_step=variant == "off_step", mixed=variant == "mixed_cover")
    if covers:
        stop, color = covers[-1] if variant == "last_latch" else covers[0]
    else:
        stop, color = len(g) - 1, info["ref"]
    if variant == "reference_paint":
        color = info["ref"]
    result = g.copy()
    if variant != "keep_gaps":
        result[:stop + 1, info["start"]:info["end"]] = info["bg"]
    objects = (g != info["bg"]) & (g != info["ref"])
    for d in range(0, stop + 1, p):
        result[d, info["start"]:info["end"]] = color
        if variant != "strip_only":
            result[d, objects[d]] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return np.rot90(_paint(info), info["turn"]).copy()


def _pixel_expected(info):
    """Coordinate formulation for checking painter output, using original cells."""
    g = info["g"]
    p, a, b = info["p"], info["start"], info["end"]
    stop, color = len(g) - 1, info["ref"]
    for d in range(2 * p, len(g), p):
        v = g[d, a:b]
        if v[0] not in (info["bg"], info["ref"]) and bool(np.all(v == v[0])):
            stop, color = d, int(v[0])
            break
    expected = g.copy()
    for r in range(len(g)):
        for c in range(g.shape[1]):
            if r > stop:
                continue
            if a <= c < b:
                expected[r, c] = color if r % p == 0 else info["bg"]
            elif r % p == 0 and g[r, c] not in (info["bg"], info["ref"]):
                expected[r, c] = color
    return expected


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _features(info):
    g = info["g"]
    result = _paint(info)
    depth, span = g.shape
    p, a, b = info["p"], info["start"], info["end"]
    covers = _covers(info)
    stop, color = covers[0] if covers else (depth - 1, info["ref"])
    objects = (g != info["bg"]) & (g != info["ref"])
    strip = np.zeros_like(objects)
    strip[:, a:b] = True
    distances = np.arange(depth)[:, None]
    active = distances <= stop
    ticks = distances % p == 0
    changed = objects & ~strip & active & ticks & (g != color)
    preserved = objects & ~strip & active & ~ticks
    split_component = False
    component_count = 0
    for hue in np.unique(g[objects]):
        for cells in _components(g == hue):
            component_count += 1
            rr, cc = np.array(cells).T
            if np.any(changed[rr, cc]) and np.any(preserved[rr, cc]):
                split_component = True
    false_off = any(d < stop and d % p != 0 for d, _ in _covers(info, off_step=True))
    false_mixed = any(d < stop and len(np.unique(g[d, a:b])) > 1
                      for d, _ in _covers(info, mixed=True))
    witnesses = dict(
        stopped=bool(covers), unstopped=not covers,
        false_off_step=false_off, false_mixed=false_mixed,
        later_other_color=any(d > stop and c != color for d, c in covers),
        clear_object=bool(np.any(objects & strip & active & ~ticks)),
        remote_change=bool(np.any(changed & ((np.arange(span)[None, :] < a - 1)
                                            | (np.arange(span)[None, :] > b)))),
        outside_background=bool(np.any((g == info["bg"]) & ~strip & active & ticks)),
        split_component=split_component,
        past_stop=bool(covers and stop + p < depth))
    omissions = {v: not np.array_equal(result, _paint(info, v)) for v in _OMISSIONS}
    return dict(witnesses=witnesses, omissions=omissions, period=p,
                turn=info["turn"], shape=g.shape, bar_width=b - a,
                start=a, stop=stop, latch_count=len(covers),
                latch_color=color, foreground=int(objects.sum()),
                components=component_count, bg=info["bg"], ref=info["ref"],
                palette=tuple(int(c) for c in np.unique(g)))


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _input_key(g):
    g = np.asarray(g)
    return g.shape, g.astype(np.int8).tobytes()


def _puzzle_hash(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(split.encode())
        h.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def _random_object(rng, depth, span):
    """A fresh mask: rectangles, holes, diamonds, profiles, walks, and elbows."""
    h = int(rng.integers(1, min(depth - 1, 12) + 1))
    w = int(rng.integers(1, min(span, 12) + 1))
    r = int(rng.integers(1, depth - h + 1))
    c = int(rng.integers(0, span - w + 1))
    kind = int(rng.integers(6))
    local = np.ones((h, w), dtype=bool)
    if kind == 1 and h >= 3 and w >= 3:
        local[1:-1, 1:-1] = False
        if rng.random() < .5:
            local[0, int(rng.integers(1, w - 1))] = False
    elif kind == 2:
        rr, cc = np.indices((h, w))
        local = (np.abs(rr - (h - 1) / 2) / max(1, h / 2)
                 + np.abs(cc - (w - 1) / 2) / max(1, w / 2)) <= 1.15
    elif kind == 3:
        spine = int(rng.integers(w))
        for row in range(h):
            lo = int(rng.integers(spine + 1))
            hi = int(rng.integers(spine + 1, w + 1))
            local[row, :lo] = local[row, hi:] = False
    elif kind == 4:
        local[:] = False
        rr, cc = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(2 * h * w):
            local[rr, cc] = True
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            rr, cc = int(np.clip(rr + dr, 0, h - 1)), int(np.clip(cc + dc, 0, w - 1))
    elif kind == 5:
        local[:] = False
        local[:, int(rng.integers(w))] = True
        local[int(rng.integers(h)), :] = True
    mask = np.zeros((depth, span), dtype=bool)
    mask[r:r + h, c:c + w] = local
    return mask


def _gate(g, rng, row, a, b, color, p):
    """A varied spanning object with a guaranteed complete central section."""
    up, down = int(rng.integers(min(p, 3))), int(rng.integers(min(p, 3)))
    left = max(0, a - int(rng.integers(4)))
    right = min(g.shape[1], b + int(rng.integers(4)))
    for r in range(max(1, row - up), min(len(g), row + down + 1)):
        lo = int(rng.integers(left, a + 1))
        hi = int(rng.integers(b, right + 1))
        g[r, lo:hi] = color
    g[row, a:b] = color


def _remote_profile(g, rng, a, b, p, color):
    """A variable connected object crossing samples and gaps outside the strip."""
    # The wider margin always has at least three columns in sampled geometry.
    if a >= g.shape[1] - b:
        low, high = 0, a - 1
    else:
        low, high = b + 1, g.shape[1]
    w = int(rng.integers(1, min(4, high - low) + 1))
    c = int(rng.integers(low, high - w + 1))
    top = int(rng.integers(1, p + 1))
    bottom = int(rng.integers(2 * p + 2, len(g) + 1))
    spine = int(rng.integers(w))
    for r in range(top, bottom):
        lo = int(rng.integers(spine + 1))
        hi = int(rng.integers(spine + 1, w + 1))
        g[r, c + lo:c + hi] = color


def _build_pair(rng, p, turn, palette, mode, teaching=False, density=.25):
    stopped = mode == "stop"
    minimum = 4 * p + 1 if stopped and teaching else 3 * p + 2
    depth = int(rng.integers(minimum, 31))
    span = depth if rng.random() < .2 else int(rng.integers(10, 31))
    width = int(rng.integers(2, min(8, span - 5) + 1))
    a = int(rng.integers(1, span - width))
    b = a + width
    bg, ref = palette[:2]
    hues = palette[2:2 + int(rng.integers(2, 5))]
    color = int(hues[int(rng.integers(len(hues)))])
    other = int(next(c for c in hues if c != color))
    g = np.full((depth, span), bg, dtype=np.int8)
    for _ in range(int(rng.integers(2, 12))):
        mask = _random_object(rng, depth, span)
        if np.count_nonzero((g != bg) | mask) < density * depth * span:
            g[mask] = int(hues[int(rng.integers(len(hues)))])
    ticks = list(range(0, depth, p))
    later_row = None
    if stopped:
        first_index = int(rng.integers(3 if teaching else 2,
                                       len(ticks) - 1 if teaching else len(ticks)))
        stop = ticks[first_index]
        _gate(g, rng, stop, a, b, color, p)
        later = ticks[first_index + 1:]
        if later and (teaching or rng.random() < .7):
            later_row = int(later[int(rng.integers(len(later)))])
            _gate(g, rng, later_row, a, b, other, p)
    else:
        first_index, stop = len(ticks), depth - 1
    # An off-step complete crossing competes with the actual latch. Its distance
    # and shape vary; it does not carry an output-specific fixed color code.
    off_options = [r for r in range(p + 1, stop) if r % p]
    off_row = None
    if off_options and (teaching or rng.random() < .6):
        off_row = int(off_options[int(rng.integers(len(off_options)))])
        _gate(g, rng, off_row, a, b, other, p)
    if teaching or rng.random() < .75:
        _remote_profile(g, rng, a, b, p, other)
    # Finish intended nonterminal encounters. These are randomly positioned
    # gaps/windows in objects, not extra marker colors or invisible commands.
    for d in ticks[2:first_index]:
        v = g[d, a:b]
        if np.all(v != bg) and np.all(v == v[0]):
            g[d, int(rng.integers(a, b))] = bg
    if teaching:
        # A fully covered multicolor sample before the latch teaches homogeneity.
        mosaic = ticks[int(rng.integers(2, first_index))]
        split = int(rng.integers(a + 1, b))
        g[mosaic, a:split] = color
        g[mosaic, split:b] = other
    if off_row is not None:
        g[off_row, a:b] = other
    if stopped:
        # Gate fringes from other objects may overlap; the chosen center remains
        # a full monochromatic cover. Subsequent objects are read from this scene.
        g[stop, a:b] = color
        if later_row is not None:
            g[later_row, a:b] = other
    g[0] = bg
    g[0, a:b] = ref
    g[p, a:b] = ref
    if np.count_nonzero(g == bg) <= g.size / 2:
        return None
    inp = np.rot90(g, turn).copy()
    try:
        info = _parse(inp)
    except ValueError:
        return None
    covers = _covers(info)
    if (bool(covers) != stopped or (stopped and covers[0] != (stop, color))):
        return None
    if teaching:
        f = _features(info)
        needed = ("false_off_step", "false_mixed", "clear_object", "remote_change",
                  "outside_background", "split_component")
        if stopped:
            needed += ("later_other_color", "past_stop")
        if not all(f["witnesses"][k] for k in needed):
            return None
    return {"input": inp, "output": np.rot90(_paint(info), turn).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    ns = _count("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    periods = sorted(int(x) for x in rng.choice([2, 3, 4, 5], 2, replace=False))
    rng.shuffle(periods)
    test_periods = [p for p in range(2, 8) if p not in periods]
    regime = int(rng.integers(4))
    base = int(rng.integers(4))
    stable_palette = bool(rng.random() < .3)
    palette = [int(c) for c in rng.permutation(10)[:6]]
    density = float(rng.uniform(.18, .38))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            teaching = split == "train" and index < 2
            p = periods[index % 2] if split == "train" else int(rng.choice(test_periods))
            if regime == 0:
                turn = base
            elif regime in (1, 2):
                turn = base if split == "train" else (base + regime) % 4
            else:
                turn = int(rng.integers(4))
            if not stable_palette:
                palette = [int(c) for c in rng.permutation(10)[:6]]
            mode = ("stop" if index == 0 else "none") if teaching else (
                "stop" if rng.random() < .65 else "none")
            for _ in range(96):
                pair = _build_pair(rng, p, turn, palette, mode, teaching, density)
                if pair is None or _input_key(pair["input"]) in seen:
                    continue
                seen.add(_input_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded scene proposals exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            try:
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                info = _parse(inp)
            except ValueError as exc:
                raise AssertionError(str(exc)) from exc
            assert inp.shape == out.shape
            assert not np.array_equal(inp, out), "identity sample"
            assert _input_key(inp) not in seen, "duplicate input"
            seen.add(_input_key(inp))
            expected = np.rot90(_pixel_expected(info), info["turn"])
            assert np.array_equal(expected, out), "incorrect shutter output"
            if split == "train":
                features.append(_features(info))
    for name in _WITNESSES:
        assert any(f["witnesses"][name] for f in features), f"missing training witness: {name}"
    for name in _OMISSIONS:
        assert any(f["omissions"][name] for f in features), f"unrefuted shortcut: {name}"
    assert len({f["period"] for f in features}) >= 2, "teach measured rather than fixed spacing"


def self_test():
    """Bounded tests; return only compact aggregate measurements, never payloads."""
    hashes, shapes, widths, starts, hues, backgrounds = set(), set(), set(), set(), set(), set()
    periods = {"train": set(), "test": set()}
    turns = {"train": set(), "test": set()}
    latches, foreground, components = set(), [], []
    pairs = 0
    structural = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(_puzzle_hash(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                f = _features(info)
                pairs += 1
                shapes.add(pair["input"].shape)
                widths.add(f["bar_width"])
                starts.add(f["start"])
                backgrounds.add(f["bg"])
                hues.update(f["palette"])
                periods[split].add(f["period"])
                turns[split].add(f["turn"])
                latches.add(f["latch_count"])
                foreground.append(f["foreground"])
                components.append(f["components"])
                structural.add((pair["input"].shape, f["period"], f["turn"],
                                f["bar_width"], f["start"], f["stop"], f["foreground"]))
                assert np.array_equal(solve(pair["input"]), pair["output"])
        train_p = {_parse(p["input"])["p"] for p in puzzle["train"]}
        test_p = {_parse(p["input"])["p"] for p in puzzle["test"]}
        assert train_p.isdisjoint(test_p), "test cadence must transfer"
        if seed < 3:
            assert _puzzle_hash(puzzle) == _puzzle_hash(generate_puzzle(seed))
    assert len(hashes) == 42
    assert len(shapes) >= 60 and len(structural) >= 190
    assert widths == set(range(2, 9)) and len(starts) >= 12
    assert backgrounds == set(range(10)) and hues == set(range(10))
    assert periods["train"] == {2, 3, 4, 5}
    assert periods["test"] >= {2, 3, 4, 5, 6, 7}
    assert turns["train"] == turns["test"] == {0, 1, 2, 3}
    assert {0, 1, 2} <= latches and max(foreground) > 100
    assert max(components) >= 5
    # All thirty legal explicit count combinations, including the minimal case.
    legal_calls = 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(65000 + nt * 10 + ns, nt, ns)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ns
            legal_calls += 1
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (None, 1), (4, None), (4, "1")):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Direct hand-specified canonical regression: off-step and mixed false
    # covers, first latch, untouched later latch, and disconnected foreground.
    g = np.zeros((15, 12), dtype=np.int8)
    g[0, 4:7] = g[3, 4:7] = 1
    g[6, 4:7] = [2, 3, 2]
    g[7, 4:7] = 2
    g[9, 4:7] = 3
    g[12, 4:7] = 2
    g[5:11, 1:3] = 2
    expected = g.copy()
    expected[:10, 4:7] = 0
    expected[[0, 3, 6, 9], 4:7] = 3
    expected[6, 1:3] = expected[9, 1:3] = 3
    assert np.array_equal(solve(g), expected)
    rng = np.random.default_rng(991)
    for turn in range(4):
        palette = rng.permutation(10)
        assert np.array_equal(solve(palette[np.rot90(g, turn)]),
                              palette[np.rot90(expected, turn)])
    # No latch: strip tail beyond the last tick clears, outside tail survives.
    g[9, 5] = 0
    g[12, 5] = 0
    no_latch = solve(g)
    assert np.all(no_latch[[0, 3, 6, 9, 12], 4:7] == 1)
    assert np.all(no_latch[14, 4:7] == 0) and no_latch[7, 1] == 2
    # A latch at the opposite boundary counts.
    edge = np.zeros((13, 10), dtype=np.int8)
    edge[0, 3:5] = edge[3, 3:5] = 1
    edge[-1, 3:5] = 4
    edge_out = solve(edge)
    assert np.all(edge_out[[0, 3, 6, 9, 12], 3:5] == 4)
    # p=1 has adjacent sightings, no cleared gaps, and still supports stopping.
    solid = np.zeros((8, 9), dtype=np.int8)
    solid[:2, 3:6] = 2
    solid[4, 3:6] = 5
    solid[2:7, 0] = 6
    solid_out = solve(solid)
    assert np.all(solid_out[:5, 3:6] == 5)
    assert solid_out[3, 0] == 5 and solid_out[5, 0] == 6
    # Input-only solver does not mutate arrays and handles ordinary nested lists.
    before = solid.copy()
    assert np.array_equal(solve(solid.tolist()), solid_out)
    assert np.array_equal(before, solid)
    broken = generate_puzzle(4001, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(hashes),
                input_shapes=len(shapes), structural_signatures=len(structural),
                train_periods=sorted(periods["train"]), test_periods=sorted(periods["test"]),
                bar_widths=sorted(widths), bar_starts=len(starts),
                background_colors=sorted(backgrounds), latch_counts=sorted(latches),
                foreground_range=[min(foreground), max(foreground)],
                component_range=[min(components), max(components)],
                legal_count_calls=legal_calls, independent_solver=False)
