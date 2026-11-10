"""Growing stitch clocks: colored programs calibrate and extend their own path.

Only standard Python and NumPy are used. Labels use explicit latent cell lists;
the input-only solver parses the observed execution and extrapolates its clocks.
"""

import numpy as np


# Geometry determines direction. Length is learned separately for each glyph.
_MASKS = (
    ((1, 1, 0), (1, 0, 1), (0, 1, 0)),  # right, inherited from A
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)),  # left, inherited from A
    ((1, 0, 1), (0, 1, 0), (0, 1, 0)),  # down, inherited from A
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),  # down-right
    ((0, 0, 1), (0, 1, 0), (1, 0, 0)),  # down-left
)
_DIRECTIONS = ((0, 1), (0, -1), (1, 0), (1, 1), (1, -1))
_MASK_TO_ID = {tuple(np.asarray(m).flat): k for k, m in enumerate(_MASKS)}


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is out of range")
    return value


def _counts(num_train, num_test):
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    return nt, nq


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _marker_cells(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _find_marker(a):
    candidates = []
    for color in np.unique(a):
        p = np.argwhere(a == color)
        if len(p) != 6:
            continue
        r0, c0 = map(int, p.min(axis=0))
        r1, c1 = map(int, p.max(axis=0))
        # This makes the canonical three-cell Ls distinct and four-disconnected.
        if min(r1 - r0, c1 - c0) < 1 or r1 - r0 + c1 - c0 < 4:
            continue
        bounds = r0, r1, c0, c1
        if set(map(tuple, p)) == _marker_cells(bounds):
            candidates.append((int(color), bounds))
    assert len(candidates) == 1, "ambiguous L-marker annotation"
    return candidates[0]


def _runs(present):
    positions = np.flatnonzero(present)
    groups = []
    for raw in positions:
        x = int(raw)
        if groups and x == groups[-1][1]:
            groups[-1][1] += 1
        else:
            groups.append([x, x + 1])
    return groups


def _read_input(input_grid):
    """Recover role colors, tokens, observed strokes, and each local clock."""
    a = _grid(input_grid)
    vals, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(vals[counts.argmax()])
    marker, global_bounds = _find_marker(a)
    assert marker != bg
    dividers = [c for c in range(1, a.shape[1] - 1)
                if a[0, c] not in (bg, marker) and np.all(a[:, c] == a[0, c])]
    assert len(dividers) == 1, "expected one full-height divider"
    divider = dividers[0]
    separator = int(a[0, divider])
    panel, scene = a[:, :divider], a[:, divider + 1:]
    rows = _runs(np.any(panel != bg, axis=1))
    cols = _runs(np.any(panel != bg, axis=0))
    assert 2 <= len(cols) <= 4 and 2 <= len(rows) <= 7
    assert rows[0][0] == cols[0][0] == 0 and cols[-1][1] == divider
    assert all(y - x == 3 for x, y in rows + cols)
    assert all(b[0] - x[1] == 1 for x, b in zip(rows, rows[1:]))
    assert all(b[0] - x[1] == 1 for x, b in zip(cols, cols[1:]))
    tokens = []
    for x0, x1 in cols:
        for y0, y1 in rows:
            block = panel[y0:y1, x0:x1]
            mask = tuple(map(int, (block != bg).flat))
            assert mask in _MASK_TO_ID, "unknown directional glyph"
            ink = np.unique(block[block != bg])
            assert len(ink) == 1 and ink[0] not in (marker, separator)
            tokens.append((_MASK_TO_ID[mask], int(ink[0])))
    assert all(x[1] != y[1] for x, y in zip(tokens, tokens[1:])), "merged calibration colors"
    vocabulary = set(k for k, _ in tokens)
    assert vocabulary in ({0, 1}, {0, 1, 2}, {0, 1, 3}, {0, 1, 4})
    starts = np.flatnonzero(scene[0] != bg)
    assert len(starts) == 1
    start = int(starts[0])
    seed_color = int(scene[0, start])
    assert seed_color not in (bg, marker, separator)
    assert np.count_nonzero(scene == seed_color) == 1
    assert all(color != seed_color for _, color in tokens)
    r0, r1, gc0, gc1 = global_bounds
    bounds = r0, r1, gc0 - divider - 1, gc1 - divider - 1
    assert 0 <= bounds[2] < bounds[3] < scene.shape[1]
    clean = scene.copy()
    clean[clean == marker] = bg
    assert np.all(clean[r0:r1 + 1, bounds[2]:bounds[3] + 1] == bg)
    ink_rows = np.flatnonzero(np.any(clean[1:] != bg, axis=1)) + 1
    assert len(ink_rows) and np.array_equal(ink_rows, np.arange(1, ink_rows[-1] + 1))
    last_row = int(ink_rows[-1])
    assert last_row < r0, "the requested viewport must follow the observed prefix"
    row_colors = []
    for r in ink_rows:
        color = np.unique(clean[r, clean[r] != bg])
        assert len(color) == 1
        row_colors.append(int(color[0]))
    segments = []
    for r, color in zip(ink_rows, row_colors):
        if segments and segments[-1][2] == color:
            segments[-1][1] = int(r) + 1
        else:
            segments.append([int(r), int(r) + 1, color])
    assert len(segments) < len(tokens)
    observed = {k: [] for k in vocabulary}
    r, c = 0, start
    expected = np.full_like(clean, bg)
    expected[0, start] = seed_color
    for i, (top, bottom, color) in enumerate(segments):
        k, token_color = tokens[i]
        assert color == token_color and top == r + 1
        dr, dc = _DIRECTIONS[k]
        if dr == 0:
            assert bottom == top + 1
            xs = np.flatnonzero(clean[top] != bg)
            length = len(xs)
            assert np.array_equal(xs, np.arange(xs[0], xs[-1] + 1))
            assert (int(xs[0]) if dc == 1 else int(xs[-1])) == c
            expected[top, xs] = color
            r, c = top, c + dc * (length - 1)
        else:
            length = bottom - top
            for j in range(length):
                xs = np.flatnonzero(clean[top + j] != bg)
                assert len(xs) == 1 and xs[0] == c + dc * j
                expected[top + j, int(xs[0])] = color
            r, c = bottom - 1, c + dc * (length - 1)
        observed[k].append(length)
    assert np.array_equal(clean, expected), "content outside the calibrated prefix"
    clocks = {}
    for k, sizes in observed.items():
        assert len(sizes) >= 3, "each clock needs three visible observations"
        assert max(sizes) >= 2, "glyph direction has no visible displacement evidence"
        delta = sizes[1] - sizes[0]
        assert delta in (-1, 0, 1)
        assert sizes == [sizes[0] + j * delta for j in range(len(sizes))]
        clocks[k] = sizes[0], delta
    assert any(delta for _, delta in clocks.values()), "no extrapolating clock"
    row_major = [tokens[c * len(rows) + r] for r in range(len(rows)) for c in range(len(cols))]
    assert row_major[:len(segments)] != tokens[:len(segments)], "traversal lacks visible evidence"
    info = dict(background=bg, marker=marker, separator=separator, seed_color=seed_color,
                bounds=bounds, scene_shape=scene.shape, start=start, tokens=tokens,
                visible=len(segments), observed=observed, clocks=clocks,
                endpoint=(r, c), lanes=len(cols), bands=len(rows))
    return info


def _predict(info, *, freeze=False, repeat=False):
    """Render only the requested crop, using input-fitted clocks and coordinates.

    Two explicit counterfactuals support validation: freeze lengths at their last
    observation, or replace the hidden instructions by a repeated visible tape.
    """
    r0, r1, c0, c1 = info["bounds"]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), info["background"], dtype=np.int8)
    counters = {k: len(v) for k, v in info["observed"].items()}
    r, c = info["endpoint"]
    prefix = info["tokens"][:info["visible"]]
    for j, actual in enumerate(info["tokens"][info["visible"]:]):
        k, color = prefix[j % len(prefix)] if repeat else actual
        base, delta = info["clocks"][k]
        n = info["observed"][k][-1] if freeze else base + counters[k] * delta
        assert n >= 1, "a clock ran out of positive lengths"
        dr, dc = _DIRECTIONS[k]
        rr = r + 1 + dr * np.arange(n)
        cc = c + dc * np.arange(n)
        if not (freeze or repeat):
            assert rr[-1] < info["scene_shape"][0]
            assert cc.min() >= 0 and cc.max() < info["scene_shape"][1]
        keep = (rr >= r0) & (rr <= r1) & (cc >= c0) & (cc <= c1)
        out[rr[keep] - r0, cc[keep] - c0] = color
        r, c = int(rr[-1]), int(cc[-1])
        counters[k] += 1
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict from this input alone; never inspect seed, examples, or metadata."""
    return _predict(_read_input(input_grid))


def _latent_strokes(program, clocks):
    """Generation-side ground truth, explicit scalar endpoint updates."""
    r, c = 0, 0
    counters = {k: 0 for k in clocks}
    strokes = []
    for k in program:
        base, delta = clocks[k]
        n = base + delta * counters[k]
        if n < 1:
            return None
        dr, dc = _DIRECTIONS[k]
        cells = [(r + 1 + dr * j, c + dc * j) for j in range(n)]
        strokes.append(cells)
        r, c = cells[-1]
        counters[k] += 1
    return strokes


def _build_pair(rng, vocabulary, test, index, fallback=None):
    m = len(vocabulary)
    length = (12 if test else (8 if index % 2 == 0 else 10)) if m == 2 else (15 if test else 12)
    nvisible = 3 * m
    if fallback is None:
        prefix = list(map(int, rng.permutation(list(vocabulary) * 3)))
        if prefix == prefix[:m] * 3:
            return None
        tail = []
        while len(tail) < length - nvisible:
            tail.extend(map(int, rng.permutation(vocabulary)))
        program = prefix + tail[:length - nvisible]
        clocks = {}
        for k in vocabulary:
            delta = int(rng.choice([-1, 0, 1], p=[0.25, 0.25, 0.5]))
            count = program.count(k)
            base = int(rng.integers(max(1, count if delta == -1 else 1),
                                    max(1, count if delta == -1 else 1) + 3))
            clocks[k] = base, delta
        if not any(d for _, d in clocks.values()):
            return None
        roles = list(map(int, rng.permutation(10)))
        bg, sep, seed_color, marker = roles[:4]
        palette = roles[4:4 + int(rng.integers(2, 7))]
        colors = [int(rng.choice(palette))]
        for _ in range(length - 1):
            colors.append(int(rng.choice([v for v in palette if v != colors[-1]])))
    else:
        extra = vocabulary[-1]
        prefix = [0, 0, 1, 1, 1, 0] if m == 2 else [0, 1, 0, extra, 1, extra, 0, extra, 1]
        tail = ([1, 0, 1, 0] if not test else [0, 0, 1, 1, 0, 1]) if m == 2 else [1, extra, 0, 0, extra, 1]
        program = prefix + tail[:length - nvisible]
        clocks = {k: (2, 1 if k in (0, 1) else 0) for k in vocabulary}
        bg = fallback % 10
        sep, seed_color, marker = ((bg + j) % 10 for j in (1, 2, 3))
        palette = [(bg + j) % 10 for j in range(4, 10)]
        colors = [palette[(i + fallback // 10) % 6] for i in range(length)]
    # Each pair exposes that colors are paint, and that clocks count a glyph's
    # own occurrences rather than global rounds of m instructions.
    if any(len({color for k2, color in zip(program, colors) if k2 == k}) < 2
           for k in vocabulary):
        return None
    if not any(len({k for k, color in zip(program, colors) if color == v}) > 1
               for v in set(colors)):
        return None
    if not any(delta and [i // m for i, k2 in enumerate(prefix) if k2 == k] != [0, 1, 2]
               for k, (base, delta) in clocks.items()):
        return None
    strokes = _latent_strokes(program, clocks)
    if strokes is None:
        return None
    points = [(0, 0)] + [q for s in strokes for q in s]
    cmin, cmax = min(c for r, c in points), max(c for r, c in points)
    path_height = strokes[-1][-1][0] + 1
    path_width = cmax - cmin + 1
    layouts = []
    for lanes in (2, 3, 4):
        if length % lanes:
            continue
        bands = length // lanes
        panel_h, panel_w = 4 * bands - 1, 4 * lanes - 1
        max_width = 30 - panel_w - 1
        if 2 <= bands <= 7 and max(path_height, panel_h) <= 30 and path_width + 4 <= max_width:
            layouts.append((lanes, bands, panel_h, panel_w, max_width))
    if not layouts:
        return None
    choice = int(rng.integers(len(layouts))) if fallback is None else fallback % len(layouts)
    lanes, bands, panel_h, panel_w, max_width = layouts[choice]
    minimum_width = max(7, path_width + 4)
    width = int(rng.integers(minimum_width, max_width + 1)) if fallback is None else minimum_width
    shift = 2 - cmin
    if fallback is None:
        shift += int(rng.integers(width - path_width - 3))
    strokes = [[(r, c + shift) for r, c in cells] for cells in strokes]
    height = max(path_height, panel_h)
    if fallback is None:
        height += int(rng.integers(min(3, 30 - height) + 1))
    full = np.full((height, width), bg, dtype=np.int8)
    full[0, shift] = seed_color
    for cells, color in zip(strokes, colors):
        for r, c in cells:
            assert full[r, c] == bg
            full[r, c] = color
    hidden_count = length - nvisible
    skip = 0 if not test and index == 0 else (1 if not test else
           (int(rng.integers(1, min(3, hidden_count - 2) + 1)) if fallback is None else 1))
    selected = [q for s in strokes[nvisible + skip:] for q in s]
    r0, r1 = min(r for r, c in selected), max(r for r, c in selected)
    c0, c1 = min(c for r, c in selected) - 1, max(c for r, c in selected) + 1
    if fallback is None and (test or index > 0):
        if r1 - r0 >= 3:
            r0 += int(rng.integers(min(3, r1 - r0 - 1)))
        if c1 - c0 >= 4:
            c0 += int(rng.integers(3))
            c1 -= int(rng.integers(min(3, c1 - c0 - 1)))
    if r1 == r0:
        r1 += 1
        if r1 >= height:
            if height == 30:
                return None
            height += 1
            full = np.vstack((full, np.full((1, width), bg, dtype=np.int8)))
    if r1 - r0 + c1 - c0 < 4:
        c0 = max(0, c0 - (4 - (r1 - r0 + c1 - c0)))
    bounds = r0, r1, c0, c1
    if min(r1 - r0, c1 - c0) < 1 or r1 - r0 + c1 - c0 < 4:
        return None
    output = full[r0:r1 + 1, c0:c1 + 1].copy()
    if not (np.any(output == bg) and np.any(output != bg)):
        return None
    scene = np.full_like(full, bg)
    scene[0, shift] = seed_color
    for cells, color in zip(strokes[:nvisible], colors[:nvisible]):
        for r, c in cells:
            scene[r, c] = color
    for r, c in _marker_cells(bounds):
        scene[r, c] = marker
    inp = np.full((height, panel_w + 1 + width), bg, dtype=np.int8)
    inp[:, panel_w] = sep
    inp[:, panel_w + 1:] = scene
    for i, (k, color) in enumerate(zip(program, colors)):
        lane, band = divmod(i, bands)
        block = inp[4 * band:4 * band + 3, 4 * lane:4 * lane + 3]
        block[np.asarray(_MASKS[k], dtype=bool)] = color
    try:
        info = _read_input(inp)
        if not np.array_equal(_predict(info), output):
            return None
        if np.array_equal(_predict(info, freeze=True), output):
            return None
        if np.array_equal(_predict(info, repeat=True), output):
            return None
    except AssertionError:
        return None
    return {"input": inp, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    vocabulary = (0, 1) if rng.random() < 0.35 else (0, 1, int(rng.integers(2, 5)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for attempt in range(160):
                pair = _build_pair(rng, vocabulary, split == "test", index,
                                   None if attempt < 128 else attempt - 128)
                if pair is not None and _key(pair["input"]) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                raise AssertionError("bounded construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in infos:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _read_input(inp)
            infos[split].append(info)
            assert all(len(v) == 3 for v in info["observed"].values())
            assert np.array_equal(out, _predict(info)), "incorrect continuation crop"
            assert np.any(out == info["background"]) and np.any(out != info["background"])
            assert not set(np.unique(out)) & {info["marker"], info["separator"], info["seed_color"]}
            assert set(np.unique(out)) <= set(np.unique(inp))
            assert not np.array_equal(out, _predict(info, freeze=True)), "B growth is irrelevant"
            assert not np.array_equal(out, _predict(info, repeat=True)), "A program is irrelevant"
            m = len(info["clocks"])
            prefix = [k for k, color in info["tokens"][:info["visible"]]]
            assert prefix != prefix[:m] * 3, "local clocks lack interleaving evidence"
            assert any(delta and [j // m for j, k2 in enumerate(prefix) if k2 == k] != [0, 1, 2]
                       for k, (base, delta) in info["clocks"].items()), "global-round clock shortcut"
    all_infos = infos["train"] + infos["test"]
    assert len({tuple(sorted(i["clocks"])) for i in all_infos}) == 1
    assert min(len(i["tokens"]) for i in infos["test"]) > max(len(i["tokens"]) for i in infos["train"])
    first = infos["train"][0]
    assert first["bounds"][0] == first["endpoint"][0] + 1
    for i in infos["train"][1:2] + infos["test"]:
        assert i["bounds"][0] > i["endpoint"][0] + 1, "no hidden-stroke gap"
    # Color is demonstrably an occurrence property, rather than a glyph code.
    colors = {k: set() for k in first["clocks"]}
    masks_by_color = {}
    for i in infos["train"]:
        for k, color in i["tokens"]:
            colors[k].add(color)
            masks_by_color.setdefault(color, set()).add(k)
    assert all(len(v) >= 2 for v in colors.values())
    assert any(len(v) >= 2 for v in masks_by_color.values())


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed structural/semantic coverage and failure regressions."""
    signatures, shapes, crops, programs = set(), set(), set(), set()
    vocabularies, lanes, rates, starts, backgrounds, crop_colors = set(), set(), set(), set(), set(), set()
    growth = shrink = diagonal = partial = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_puzzle_key(p))
        for pair in p["train"] + p["test"]:
            info = _read_input(pair["input"])
            shapes.add(pair["input"].shape)
            crops.add(pair["output"].shape)
            programs.add(tuple(k for k, color in info["tokens"]))
            vocabularies.add(tuple(sorted(info["clocks"])))
            lanes.add(info["lanes"])
            rates.update(d for b, d in info["clocks"].values())
            starts.add(info["start"])
            backgrounds.add(info["background"])
            crop_colors.add(len(set(np.unique(pair["output"])) - {info["background"]}))
            growth += any(d > 0 for b, d in info["clocks"].values())
            shrink += any(d < 0 for b, d in info["clocks"].values())
            diagonal += bool(set(info["clocks"]) & {3, 4})
            # Count actual crop cuts through a reconstructed stroke, not RNG flags.
            strokes = _latent_strokes([k for k, color in info["tokens"]], info["clocks"])
            r0, r1, c0, c1 = info["bounds"]
            for cells in strokes[info["visible"]:]:
                inside = [r0 <= r <= r1 and c0 <= c + info["start"] <= c1 for r, c in cells]
                partial += any(inside) and not all(inside)
            original = pair["input"].copy()
            assert np.array_equal(solve(original), pair["output"])
            assert np.array_equal(original, pair["input"]), "solve mutated its input"
    assert len(signatures) == 42 and len(programs) >= 100
    assert len(shapes) >= 40 and len(crops) >= 30
    assert vocabularies == {(0, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4)}
    assert lanes == {2, 3, 4} and rates == {-1, 0, 1}
    assert len(starts) >= 6 and len(backgrounds) == 10 and max(crop_colors) >= 3
    assert growth >= 100 and shrink >= 20 and diagonal >= 50 and partial >= 30
    # Each fallback pool contains 32 accepted distinct inputs, greater than the
    # maximum 13 previously generated inputs. No random retry is unbounded.
    for vocabulary in ((0, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4)):
        for test, index in ((False, 0), (False, 1), (True, 0)):
            pool = [_build_pair(np.random.default_rng(0), vocabulary, test, index, v)
                    for v in range(32)]
            assert all(p is not None for p in pool), "invalid finite fallback"
            assert len({_key(p["input"]) for p in pool}) == 32
    # Every legal explicit count combination, including the two-demonstration boundary.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(7000 + nt * 10 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    assert _puzzle_key(generate_puzzle(123, 10, 4)) == _puzzle_key(generate_puzzle(123, 10, 4))
    # Reject a wrong label and a damaged calibration observation.
    for mode in ("output", "input"):
        p = generate_puzzle(44, 2, 1)
        pair = p["test"][0]
        if mode == "output":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        else:
            info = _read_input(pair["input"])
            divider = 4 * info["lanes"] - 1
            pair["input"][1, divider + 1 + info["start"]] = info["background"]
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("semantic corruption accepted")
