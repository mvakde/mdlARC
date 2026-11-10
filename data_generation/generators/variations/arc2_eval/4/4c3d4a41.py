"""Insert a comb and bottom-align its ordered pins, clipping at the roof.

Colors, orientation and geometry are visible pair data. Puzzle schedules vary
the demonstrations; they never select a different physical operation.
"""

import numpy as np


_SOURCE_PAIR_ROWS = (
    ((
        "00000000055555555555", "00000000050302050505",
        "05000005050302040705", "05050005050302040705",
        "05050505000002040005", "05555555000000000005",
        "00000000050000000005", "00000000055555555555",
    ), (
        "00000000055555555555", "00000000050302040705",
        "00000000050502040505", "00000000050505040505",
        "00000000000505050505", "00000000000555555505",
        "00000000050000000005", "00000000055555555555",
    )),
    ((
        "00000000055555555555", "00000000050505050505",
        "00000005050405050805", "05000005050406070805",
        "05050505000006070005", "05555555000000000005",
        "00000000050000000005", "00000000055555555555",
    ), (
        "00000000055555555555", "00000000050405050805",
        "00000000050406070505", "00000000050506070505",
        "00000000000505050505", "00000000000555555505",
        "00000000050000000005", "00000000055555555555",
    )),
    ((
        "00000000055555555555", "00000000050505050805",
        "00050500050505050805", "05050500050906030805",
        "05050505000006030005", "05555555000000000005",
        "00000000050000000005", "00000000055555555555",
    ), (
        "00000000055555555555", "00000000050506030805",
        "00000000050905050805", "00000000050505050805",
        "00000000000505050505", "00000000000555555505",
        "00000000050000000005", "00000000055555555555",
    )),
    ((
        "00000000055555555555", "00000000050302050505",
        "00000000050302040705", "00000000050302040705",
        "05050505000002040005", "05555555000000000005",
        "00000000050000000005", "00000000055555555555",
    ), (
        "00000000055555555555", "00000000050302040505",
        "00000000050302040705", "00000000050302040705",
        "00000000000505050505", "00000000000555555505",
        "00000000050000000005", "00000000055555555555",
    )),
)


def _validate_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_count("num_train", num_train, 2, 10)
    num_test = _validate_count("num_test", num_test, 1, 4)
    selector = int(seed)

    if num_train is None and num_test is None:
        num_test = 1 + selector % 4
        low = max(2, num_test + 1)
        num_train = low + (selector // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + selector % (11 - low)
    elif num_test is None:
        high = min(4, num_train - 1)
        num_test = 1 + selector % high

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        grid.shape,
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


def _rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    if (
        input_grid.shape != (8, 20)
        or output_grid.shape != (8, 20)
    ):
        return False

    key = (
        _rows_key(input_grid),
        _rows_key(output_grid),
    )
    return key in _SOURCE_PAIR_ROWS


def _puzzles_equal(first: dict, second: dict) -> bool:
    if first.keys() != second.keys():
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            for name in ("input", "output"):
                if pair_a[name].dtype != pair_b[name].dtype:
                    return False
                if not np.array_equal(
                    pair_a[name],
                    pair_b[name],
                ):
                    return False

    return True


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            tuple(
                _pair_key(pair)
                for pair in puzzle[split]
            ),
        )
        for split in ("train", "test")
    )


def _orient(grid: np.ndarray, code: int) -> np.ndarray:
    return np.rot90(np.fliplr(grid) if code >= 4 else grid, code % 4).copy()


def _unorient(grid: np.ndarray, code: int) -> np.ndarray:
    result = np.rot90(grid, -(code % 4))
    return (np.fliplr(result) if code >= 4 else result).copy()


def _draw_housing(grid: np.ndarray, f: dict) -> None:
    t, b, l, r = f["top"], f["bottom"], f["housing_left"], f["housing_right"]
    s = f["structure"]
    grid[t, l:r + 1] = grid[b, l:r + 1] = s
    grid[t:b + 1, l] = grid[t:b + 1, r] = s
    grid[f["base"] - 1:f["base"] + 1, l] = f["background"]


def _render_input(f: dict) -> np.ndarray:
    grid = np.full(f["shape"], f["background"], dtype=np.int8)
    _draw_housing(grid, f)
    base, left, spacing = f["base"], f["key_left"], f["spacing"]
    grid[base, left:left + spacing * (len(f["heights"]) - 1) + 1] = f["structure"]
    for i, h in enumerate(f["heights"]):
        grid[base - h:base, left + spacing * i] = f["structure"]
        col = f["first_lane"] + spacing * i
        seq = f["stacks"][i]
        grid[f["top"] + 1:f["top"] + 1 + len(seq), col] = seq
    return grid


def _render_output(f: dict) -> np.ndarray:
    """Construction uses a direct destination formula, independent of simulation."""
    grid = np.full(f["shape"], f["background"], dtype=np.int8)
    _draw_housing(grid, f)
    base, start, spacing = f["base"], f["first_lane"], f["spacing"]
    grid[base, start:start + spacing * (len(f["heights"]) - 1) + 1] = f["structure"]
    for i, h in enumerate(f["heights"]):
        col = start + spacing * i
        grid[base - h:base, col] = f["structure"]
        seq = f["stacks"][i]
        first = base - h - len(seq)
        for j, value in enumerate(seq):
            if first + j > f["top"]:
                grid[first + j, col] = value
    return grid


def _wall_candidates(grid: np.ndarray, background: int) -> list:
    # Read long horizontal runs of every color in a single pass. In the
    # canonical view, the roof and floor have matching endpoints.
    runs = {}
    for row, values in enumerate(grid):
        breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
        endpoints = [0, *breaks.tolist(), len(values)]
        for left, stop in zip(endpoints, endpoints[1:]):
            color = int(values[left])
            if stop - left >= 7 and color != background:
                runs.setdefault((color, left, stop - 1), []).append(row)
    return [(s, l, r, top, bottom)
            for (s, l, r), rows in runs.items()
            for top in rows for bottom in rows if bottom > top]


def _read_canonical(grid: np.ndarray, bg: int, wall: tuple) -> dict:
    """Read geometry, contact and pin sequences from pixels, including source scale."""
    s, left, right, top, bottom = wall
    assert np.all(grid[top:bottom + 1, right] == s)
    edge = grid[top + 1:bottom, left]
    assert np.all((edge == s) | (edge == bg))
    opening = np.flatnonzero(edge == bg) + top + 1
    assert len(opening) == 2 and opening[1] == opening[0] + 1
    base = int(opening[-1])
    assert base - top >= 3 and bottom - base >= 2
    # A solid base outside the opening defines the insertion row.
    key_cells = np.flatnonzero(grid[base, :left] != bg)
    assert len(key_cells) >= 3
    kl, kr = int(key_cells[0]), int(key_cells[-1])
    assert kr < left - 1 and len(key_cells) == kr - kl + 1
    assert np.all(grid[base, kl:kr + 1] == s)
    teeth = np.flatnonzero(grid[base - 1, kl:kr + 1] == s) + kl
    assert 2 <= len(teeth) <= 6 and teeth[0] == kl and teeth[-1] == kr
    steps = np.diff(teeth)
    assert np.all(steps == steps[0]) and steps[0] >= 2
    heights = []
    for col in teeth:
        row = base - 1
        while row >= 0 and grid[row, col] == s:
            row -= 1
        heights.append(base - 1 - row)
    assert all(1 <= h <= base - top - 2 for h in heights)
    lanes = np.flatnonzero(grid[top + 1, left + 1:right] != bg) + left + 1
    assert len(lanes) == len(teeth)
    assert np.all(lanes - teeth == lanes[0] - teeth[0])
    assert lanes[0] >= left + 2 and lanes[-1] <= right - 2
    stacks = []
    for col, h in zip(lanes, heights):
        segment = grid[top + 1:base, col]
        occupied = np.flatnonzero(segment != bg)
        assert len(occupied) and np.array_equal(occupied, np.arange(len(occupied)))
        seq = tuple(int(v) for v in segment[:len(occupied)])
        prefix = 0
        while prefix < len(seq) and seq[prefix] == s:
            prefix += 1
        assert prefix < len(seq)
        assert seq[prefix] not in (bg, s) and len(set(seq[prefix:])) == 1
        # Contact or overlap is admissible; a gap has no demonstrated rule.
        assert len(seq) >= base - top - h - 1
        stacks.append(seq)
    f = dict(shape=grid.shape, background=bg, structure=s, top=top, base=base,
             bottom=bottom, housing_left=left, housing_right=right, key_left=kl,
             first_lane=int(lanes[0]), spacing=int(steps[0]),
             heights=tuple(heights), stacks=tuple(stacks))
    # Reject noise, disconnected tooth fragments, extra lanes and malformed walls.
    assert np.array_equal(grid, _render_input(f))
    return f


def _parse_input(grid: np.ndarray) -> dict:
    values, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(values[np.argmax(counts)])
    candidates = []
    for code in range(8):
        canonical = _unorient(grid, code)
        for wall in _wall_candidates(canonical, bg):
            try:
                f = _read_canonical(canonical, bg, wall)
            except AssertionError:
                continue
            f["orientation"] = code
            candidates.append(f)
    assert len(candidates) == 1, "input must have one geometric interpretation"
    return candidates[0]


def _simulate_pixels(grid: np.ndarray, f: dict) -> np.ndarray:
    """Translate actual cells and push each pin one step at a time to contact.

Only an input-derived reading is passed here, never construction metadata or
an output. This deliberately does not call the output renderer.
"""
    a = _unorient(grid, f["orientation"])
    result = a.copy()
    bg, roof, base = f["background"], f["top"], f["base"]
    external = np.argwhere(a[:, :f["housing_left"]] != bg)
    translation = f["first_lane"] - f["key_left"]
    for row, col in external:
        result[row, col] = bg
    for i in range(len(f["heights"])):
        col = f["first_lane"] + f["spacing"] * i
        result[roof + 1:base, col] = bg
    for row, col in external:
        result[row, col + translation] = a[row, col]
    for i in range(len(f["heights"])):
        col = f["first_lane"] + f["spacing"] * i
        # Locate the tooth and original pin directly in the two pixel arrays.
        tooth_top = base
        while result[tooth_top - 1, col] == f["structure"]:
            tooth_top -= 1
        cells = [(row, int(a[row, col])) for row in range(roof + 1, base)
                 if a[row, col] != bg]
        while cells[-1][0] >= tooth_top:
            cells = [(row - 1, color) for row, color in cells if row - 1 > roof]
            assert cells, "at least one payload cell must survive"
        assert cells[-1][0] == tooth_top - 1
        for row, color in cells:
            result[row, col] = color
    return _orient(result, f["orientation"])


def solve(train_pairs: list, test_input: np.ndarray) -> np.ndarray:
    """Input-only family solver; training establishes the invariant operation.

This is an authored regression check, not a protected independent reference.
No seed, answer or generation state is available to the prediction path.
"""
    return _simulate_pixels(test_input, _parse_input(test_input))


_PROFILE_MODES = ("flat", "levels", "rising", "falling", "plateau", "valley", "irregular")


def _sample_profile(rng: np.random.Generator, n: int, span: int, mode: str) -> tuple:
    cap = span - 2
    low = int(rng.integers(1, cap))
    high = int(rng.integers(low + 1, cap + 1))
    if mode == "flat":
        h = [int(rng.integers(1, cap + 1))] * n
    elif mode == "levels":
        h = [low, high] + [int(rng.choice((low, high))) for _ in range(n - 2)]
        rng.shuffle(h)
    elif mode in ("rising", "falling"):
        h = sorted([low, high] + rng.integers(low, high + 1, n - 2).tolist(),
                   reverse=mode == "falling")
    elif mode == "plateau" and n >= 3:
        h = rng.integers(low, high, n).tolist()
        pos = int(rng.integers(0, n - 1))
        h[pos:pos + 2] = [high, high]
    elif mode == "valley" and n >= 3:
        h = [high] + rng.integers(low, high, n - 2).tolist() + [high]
    else:
        h = rng.integers(1, cap + 1, n).tolist()
        if len(set(h)) == 1:
            pos = int(rng.integers(n))
            h[pos] = 1 if h[pos] != 1 else cap
    return tuple(int(v) for v in h)


def _prefix(seq: tuple, structure: int) -> int:
    return next(i for i, value in enumerate(seq) if value != structure)


def _sample_features(rng: np.random.Generator, colors: tuple, orientation: int,
                     profile: str, n: int, spans: tuple, demo: str = "") -> dict:
    source_scale = n == 4 and spans[0] <= 5 <= spans[1] and rng.random() < 0.4
    span = 5 if source_scale else int(rng.integers(spans[0], spans[1] + 1))
    # Sample from all spacings that fit both copies of the key, separated lanes
    # and two clear side insets. Larger counts naturally restrict spacing.
    spacings = [s for s in range(2, 5) if 2 * (s * (n - 1) + 1) + 5 <= 30]
    spacing = 2 if source_scale else int(rng.choice(spacings))
    key_width = spacing * (n - 1) + 1
    # Distribute the remaining width among independent geometric slack terms.
    extras = [0] * 5
    budget = 30 - (2 * key_width + 5)
    for i in rng.permutation(5):
        cap = (3, 4, 2, 2, 3)[i]
        extras[i] = int(rng.integers(0, min(cap, budget) + 1))
        budget -= extras[i]
    kl, gap_extra, inset_l, inset_r, right_margin = extras
    if source_scale:
        kl, gap_extra, inset_l, inset_r, right_margin = 1, 0, 0, 0, 0
    gap, il, ir = gap_extra + 1, inset_l + 2, inset_r + 2
    left = kl + key_width + gap
    first = left + il
    right = first + key_width - 1 + ir
    top = int(rng.integers(0, 4))
    base = top + span
    bottom = base + int(rng.integers(2, 5))
    margin = int(rng.integers(0, min(3, 29 - bottom) + 1))
    if source_scale:
        top, base, bottom, margin = 0, 5, 7, 0
    h = list(_sample_profile(rng, n, span, profile))
    # Any lane with room for two surviving cells can demonstrate a moving
    # structural prefix. Choose the lane independently of profile extrema.
    demonstration_lane = int(rng.integers(n))
    if demo == "prefix" and h[demonstration_lane] == span - 2:
        if profile == "flat":
            h = [span - 3] * n
        else:
            h[demonstration_lane] = int(rng.integers(1, span - 2))
            if len(set(h)) == 1:
                other = int(rng.choice([i for i in range(n) if i != demonstration_lane]))
                h[other] = 1 if h[other] != 1 else span - 2
    bg, structure = colors
    available = [v for v in range(10) if v not in colors]
    palette_size = int(rng.integers(1, min(n, 8) + 1))
    palette = rng.choice(available, palette_size, replace=False).tolist()
    payloads = palette + rng.choice(palette, n - palette_size).tolist()
    rng.shuffle(payloads)
    prefix_mode = str(rng.choice(("all", "none", "mixed")))
    contact_mode = str(rng.choice(("mixed", "random", "moving", "stationary"),
                                  p=(0.35, 0.35, 0.2, 0.1)))
    shifts = [int(rng.integers(1, height + 1)) if rng.random() > 0.3 else 0 for height in h]
    if contact_mode == "stationary":
        shifts = [0] * n
    elif contact_mode == "moving":
        shifts = [int(rng.integers(1, height + 1)) for height in h]
    elif contact_mode == "mixed":
        a, b = rng.choice(n, 2, replace=False)
        shifts[a], shifts[b] = 0, int(rng.integers(1, h[b] + 1))
    if demo:
        shifts[demonstration_lane] = int(rng.integers(1, h[demonstration_lane] + 1))
    if demo == "clip":
        still = int(rng.choice([i for i in range(n) if i != demonstration_lane]))
        shifts[still] = 0
    stacks = []
    for i, height in enumerate(h):
        capacity = span - height - 1
        length = capacity + shifts[i]
        lower = 1 if prefix_mode == "all" and length >= 2 else 0
        p = 0 if prefix_mode == "none" else int(rng.integers(lower, length))
        if i == demonstration_lane and demo == "prefix":
            p = int(rng.integers(shifts[i] + 1, length))
        elif i == demonstration_lane and demo == "clip":
            p = int(rng.integers(0, shifts[i]))
        stacks.append((structure,) * p + (int(payloads[i]),) * (length - p))
    return dict(shape=(bottom + margin + 1, right + right_margin + 1),
                background=bg, structure=structure, orientation=orientation,
                top=top, base=base, bottom=bottom, key_left=kl,
                housing_left=left, housing_right=right, first_lane=first,
                spacing=spacing, heights=tuple(h), stacks=tuple(stacks))


def _comparison(rng: np.random.Generator, template: dict, flat: bool) -> dict | None:
    """Same complete hardware, new feasible key; contact may be exactly zero."""
    f = dict(template)
    span = f["base"] - f["top"]
    lower = [max(1, span - len(seq) - 1) for seq in f["stacks"]]
    cap = span - 2
    if flat:
        h = [int(rng.integers(max(lower), cap + 1))] * len(lower)
    else:
        h = [int(rng.integers(lo, cap + 1)) for lo in lower]
    if tuple(h) == f["heights"]:
        choices = [(i, value) for i, lo in enumerate(lower)
                   for value in range(lo, cap + 1) if value != h[i]]
        if not choices or flat:
            return None
        i, value = choices[int(rng.integers(len(choices)))]
        h[i] = value
    f["heights"] = tuple(h)
    return f


def _arrangement_signature(f: dict) -> tuple:
    return tuple((key, f[key]) for key in sorted(f) if key != "heights")


def _pair_from_features(f: dict) -> dict:
    return {"input": _orient(_render_input(f), f["orientation"]),
            "output": _orient(_render_output(f), f["orientation"])}


def _ensure_new(f: dict, seen: set) -> dict:
    # Deterministic finite fallback. Independent bottom/right padding alone
    # supplies more canvases than the maximum 14 pairs in a puzzle.
    for extra_r in range(31 - f["shape"][0]):
        for extra_c in range(31 - f["shape"][1]):
            candidate = dict(f, shape=(f["shape"][0] + extra_r, f["shape"][1] + extra_c))
            pair = _pair_from_features(candidate)
            if _grid_key(pair["input"]) not in seen and not _is_source_pair(pair["input"], pair["output"]):
                return candidate
    # Even a full canvas has 64 distinct first-two-lane payload assignments,
    # exceeding the 14-pair bound plus four excluded source pairs. This keeps
    # color roles, geometry, contact and demonstration features unchanged.
    payload_colors = [c for c in range(10) if c not in (f["background"], f["structure"])]
    for color_a in payload_colors:
        for color_b in payload_colors:
            stacks = list(f["stacks"])
            for i, color in enumerate((color_a, color_b)):
                stacks[i] = tuple(v if v == f["structure"] else color for v in stacks[i])
            candidate = dict(f, stacks=tuple(stacks))
            pair = _pair_from_features(candidate)
            if _grid_key(pair["input"]) not in seen and not _is_source_pair(pair["input"], pair["output"]):
                return candidate
    raise AssertionError("caller exceeded the puzzle pair bound")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    # These are evidence schedules, not hidden output conventions.
    schedule = str(rng.choice(("mixed", "comparison", "profile_holdout", "size_holdout"),
                               p=(0.4, 0.25, 0.18, 0.17)))
    orientation_mode = str(rng.choice(("fixed", "mixed", "transfer"), p=(0.3, 0.45, 0.25)))
    fixed_orientation = int(rng.integers(8))
    fixed_colors = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    changing_colors = bool(rng.random() < 0.75)
    size_axis = str(rng.choice(("lanes", "span")))
    result = {"train": [], "test": []}
    training = []
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            orientation = fixed_orientation
            if orientation_mode == "mixed":
                orientation = (fixed_orientation + (index if index < 2 else int(rng.integers(8)))) % 8
            elif orientation_mode == "transfer" and split == "test":
                orientation = (fixed_orientation + int(rng.integers(1, 8))) % 8
            colors = fixed_colors
            if changing_colors:
                colors = tuple(int(c) for c in rng.choice(10, 2, replace=False))
                if split == "train" and index == 1 and colors == (training[0]["background"], training[0]["structure"]):
                    colors = (colors[1], colors[0])
            modes = _PROFILE_MODES
            if schedule == "profile_holdout":
                modes = ("flat",) if split == "test" else tuple(m for m in modes if m != "flat")
            profile = str(rng.choice(modes))
            # Simple, moderate and tall cases occur in both splits. A deliberate
            # size transfer is only one of the independently sampled schedules.
            scale = int(rng.choice(3, p=(0.3, 0.45, 0.25)))
            spans = ((4, 7), (5, 11), (9, 18))[scale]
            lane_choices = list(range(2, 7))
            if schedule == "size_holdout":
                if size_axis == "lanes":
                    lane_choices = [2, 3, 4] if split == "train" else [5, 6]
                else:
                    spans = (4, 8) if split == "train" else (10, 18)
            if split == "train" and index == 1:
                lane_choices = [n for n in lane_choices if n != len(training[0]["heights"])]
            n = int(rng.choice(lane_choices))
            f = None
            # Comparisons can occur in either split, with flat or nonflat keys.
            reuse = schedule != "size_holdout" and (
                (split == "test" and rng.random() < (0.65 if schedule == "comparison" else 0.25))
                or (split == "train" and index >= 2 and schedule == "comparison" and rng.random() < 0.6))
            if reuse:
                template = training[int(rng.integers(len(training)))]
                f = _comparison(rng, template, profile == "flat")
                if f is not None and orientation_mode == "transfer" and split == "test":
                    f["orientation"] = orientation
            if f is None:
                demo = ("prefix", "clip")[index] if split == "train" and index < 2 else ""
                f = _sample_features(rng, colors, orientation, profile, n, spans, demo)
            f = _ensure_new(f, seen)
            pair = _pair_from_features(f)
            seen.add(_grid_key(pair["input"]))
            result[split].append(pair)
            if split == "train":
                training.append(f)
        rng.shuffle(result[split])
    return result


def _features(f: dict) -> dict:
    """Review measurements computed from a semantic reading of returned pixels."""
    h, seqs = f["heights"], f["stacks"]
    n = len(h)
    span = f["base"] - f["top"]
    shifts = tuple(len(seq) - (span - height - 1) for seq, height in zip(seqs, h))
    prefixes = tuple(_prefix(seq, f["structure"]) for seq in seqs)
    maxima = [i for i, v in enumerate(h) if v == max(h)]
    flat = len(set(h)) == 1
    palette = {v for seq in seqs for v in seq if v != f["structure"]}
    return dict(lanes=n, spacing=f["spacing"], span=span, heights=h,
                flat=flat, flat_height=h[0] if flat else 0,
                no_height_one=min(h) > 1, two_level=len(set(h)) == 2,
                adjacent_maxima=not flat and any(b == a + 1 for a, b in zip(maxima, maxima[1:])),
                separated_maxima=not flat and any(b > a + 1 for a, b in zip(maxima, maxima[1:])),
                rising=not flat and tuple(sorted(h)) == h,
                falling=not flat and tuple(sorted(h, reverse=True)) == h,
                irregular=not flat and tuple(sorted(h)) != h and tuple(sorted(h, reverse=True)) != h,
                shifts=shifts, prefixes=prefixes,
                stationary=sum(s == 0 for s in shifts), moving=sum(s > 0 for s in shifts),
                mixed_contact=min(shifts) == 0 < max(shifts),
                lane_specific=len(set(shifts)) > 1,
                moving_prefix=any(0 < s < p for s, p in zip(shifts, prefixes)),
                prefix_removed=any(0 < p <= s for s, p in zip(shifts, prefixes)),
                no_retained_prefix=all(p <= s for s, p in zip(shifts, prefixes)),
                payload_clip=any(s > p for s, p in zip(shifts, prefixes)),
                payload_clips=tuple(max(0, s - p) for s, p in zip(shifts, prefixes)),
                prefix_mode="none" if not any(prefixes) else "all" if all(prefixes) else "mixed",
                repeated_payloads=len(palette) < n, palette_size=len(palette) + 2,
                background=f["background"], structure=f["structure"], orientation=f["orientation"],
                gap=f["housing_left"] - f["key_left"] - f["spacing"] * (n - 1) - 1,
                left_inset=f["first_lane"] - f["housing_left"],
                right_inset=f["housing_right"] - f["first_lane"] - f["spacing"] * (n - 1),
                floor_clearance=f["bottom"] - f["base"],
                margins=(f["top"], f["key_left"], f["shape"][0] - f["bottom"] - 1,
                         f["shape"][1] - f["housing_right"] - 1))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    demonstrations = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert pair["input"].shape == pair["output"].shape
            key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(pair["input"], pair["output"])
            assert not np.array_equal(pair["input"], pair["output"])
            f = _parse_input(pair["input"])
            assert np.array_equal(_simulate_pixels(pair["input"], f), pair["output"])
            if split == "train":
                demonstrations.append(_features(f))
    # Invariant semantics are shared, while color and direction roles are
    # locally identifiable. There is no hidden per-puzzle color map or axis.
    assert any(d["moving_prefix"] for d in demonstrations)
    assert any(d["payload_clip"] for d in demonstrations)
    assert any(d["lane_specific"] for d in demonstrations)
    # Profile types are visible quantities, so neither split has a fixed list
    # of allowed shapes. These are evidence requirements, not profile bans.


def _source_regressions() -> None:
    inputs = [np.array([[int(c) for c in row] for row in source[0]], dtype=np.int8)
              for source in _SOURCE_PAIR_ROWS]
    predictions = [solve([], a.copy()) for a in inputs]
    # Labels are accessed only after predictions have been made.
    expected = [np.array([[int(c) for c in row] for row in source[1]], dtype=np.int8)
                for source in _SOURCE_PAIR_ROWS]
    assert all(np.array_equal(a, b) for a, b in zip(predictions, expected))
    parsed = [_parse_input(a) for a in inputs]
    assert parsed[0]["heights"] == (3, 2, 1, 3)
    assert _features(parsed[2])["shifts"] == (1, 3, 3, 0)
    assert _features(parsed[3])["shifts"] == (0, 1, 1, 0)
    for code in range(8):
        mapping = np.array([(i * 3 + code) % 10 for i in range(10)], dtype=np.int8)
        for inp, out in zip(inputs, expected):
            transformed = _orient(mapping[inp], code)
            assert np.array_equal(solve([], transformed), _orient(mapping[out], code))
    # Concrete contact and clipping regression with retained prefix, removed
    # prefix and payload loss in the same scene. Lane 0 stays exactly still.
    f = dict(parsed[0], heights=(1, 2, 3, 2),
             stacks=((5, 3, 3), (5, 5, 5, 2), (5, 4, 4, 4), (7, 7)))
    a = _render_input(f)
    out = solve([], a)
    assert np.array_equal(out[1:4, 11], np.array([5, 3, 3]))
    assert np.array_equal(out[1:3, 13], np.array([5, 2]))
    assert int(out[1, 15]) == 4
    assert np.array_equal(out[1:3, 17], np.array([7, 7]))
    m = _features(_parse_input(a))
    assert m["mixed_contact"] and m["moving_prefix"] and m["payload_clip"] and m["prefix_removed"]
    # Reject an unsupported gap and a broken housing from their input pixels.
    for row, col in ((3, 11), (0, 12)):
        malformed = a.copy()
        malformed[row, col] = 0
        try:
            _parse_input(malformed)
        except AssertionError:
            pass
        else:
            raise AssertionError("malformed scene accepted")
    full = dict(f, shape=(30, 30))
    seen = set()
    for _ in range(14):
        fresh = _ensure_new(full, seen)
        seen.add(_grid_key(_pair_from_features(fresh)["input"]))
    assert len(seen) == 14


def self_test() -> None:
    _source_regressions()
    train_counts, test_counts, signatures, shapes, profiles, layouts = set(), set(), set(), set(), set(), set()
    records = []
    within_colors = within_orientations = comparisons_train = comparisons_test = 0
    fixed_orientations = transfers = fresh_flat = reused_nonflat = source_scale = 0
    single_flat = single_nonflat = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_key(puzzle))
        readings = {split: [_parse_input(p["input"]) for p in puzzle[split]] for split in puzzle}
        for split in puzzle:
            for pair, f in zip(puzzle[split], readings[split]):
                # Prediction takes only demonstrations and the current input.
                prediction = solve(puzzle["train"], pair["input"].copy())
                assert np.array_equal(prediction, pair["output"])
                shapes.add(pair["input"].shape)
                profiles.add(f["heights"])
                layouts.add((f["shape"], f["spacing"], f["top"], f["base"], f["first_lane"]))
                records.append(_features(f))
                source_scale += f["shape"] == (8, 20)
        all_f = readings["train"] + readings["test"]
        within_colors += len({(f["background"], f["structure"]) for f in all_f}) > 1
        orientations = {f["orientation"] for f in all_f}
        within_orientations += len({f["orientation"] for f in readings["train"]}) > 1
        fixed_orientations += len(orientations) == 1
        train_orientation = {f["orientation"] for f in readings["train"]}
        transfers += len(train_orientation) == 1 and any(f["orientation"] not in train_orientation for f in readings["test"])
        arrangements = [_arrangement_signature(f) for f in readings["train"]]
        comparisons_train += len(set(arrangements)) < len(arrangements)
        for f in readings["test"]:
            reused = _arrangement_signature(f) in arrangements
            flat = len(set(f["heights"])) == 1
            comparisons_test += reused
            fresh_flat += flat and not reused
            reused_nonflat += reused and not flat
            if len(puzzle["test"]) == 1:
                single_flat += flat
                single_nonflat += not flat
        assert len({len(f["heights"]) for f in readings["train"]}) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 80 and len(layouts) >= 150 and len(profiles) >= 100
    assert {d["lanes"] for d in records} == set(range(2, 7))
    assert {d["spacing"] for d in records} == {2, 3, 4}
    assert {d["span"] for d in records}.issuperset(range(4, 19))
    assert {d["orientation"] for d in records} == set(range(8))
    assert {d["palette_size"] for d in records}.issuperset(range(3, 9))
    assert {d["background"] for d in records} == set(range(10))
    assert {d["structure"] for d in records} == set(range(10))
    assert {d["prefix_mode"] for d in records} == {"all", "none", "mixed"}
    for flag in ("mixed_contact", "adjacent_maxima", "separated_maxima", "no_height_one",
                 "two_level", "rising", "falling", "irregular", "moving_prefix",
                 "prefix_removed", "payload_clip", "no_retained_prefix", "repeated_payloads"):
        assert sum(d[flag] for d in records) >= 10, flag
    assert len({d["flat_height"] for d in records if d["flat"]}) >= 5
    assert max(max(d["payload_clips"]) for d in records) >= 5
    assert sum(d["stationary"] for d in records) >= 100
    assert sum(d["moving"] for d in records) >= 100
    assert sum(d["flat"] and d["mixed_contact"] for d in records) >= 5
    assert sum(d["adjacent_maxima"] and d["mixed_contact"] and d["repeated_payloads"] and d["orientation"] >= 4 for d in records) >= 3
    assert within_colors >= 15 and within_orientations >= 10
    assert fixed_orientations >= 5 and transfers >= 3
    assert comparisons_train >= 2 and comparisons_test >= 5
    assert fresh_flat >= 10 and reused_nonflat >= 5 and single_flat and single_nonflat
    assert source_scale >= 3
    assert _puzzles_equal(generate_puzzle(123), generate_puzzle(123))
    assert not _puzzles_equal(generate_puzzle(123), generate_puzzle(124))
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(713, *counts)
        validate_generated_puzzle(p)
        assert _puzzles_equal(p, generate_puzzle(713, *counts))
        assert counts[0] is None or len(p["train"]) == counts[0]
        assert counts[1] is None or len(p["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.0, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
