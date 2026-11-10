"""Bottom-anchored vertical tile completion for ARC grids."""

import numpy as np


_SOURCE_PAIRS = (
    (
        ("55555", "55555", "55555", "55555", "55555",
         "52225", "55255", "58855", "55885", "55855"),
        ("52225", "55255", "58855", "55885", "55855",
         "52225", "55255", "58855", "55885", "55855"),
    ),
    (
        ("3333333",) * 6 + ("3339293", "3332923", "3339993", "3333933"),
        ("3339993", "3333933", "3339293", "3332923", "3339993",
         "3333933", "3339293", "3332923", "3339993", "3333933"),
    ),
    (
        ("77777",) * 6 + ("76767", "67276", "72627", "76767"),
        ("72627", "76767", "76767", "67276", "72627",
         "76767", "76767", "67276", "72627", "76767"),
    ),
)

_MOTIFS = (
    "marks", "runs", "objects", "bands", "holes",
    "row_groups", "periodic", "drift", "mirror",
)


def _seed_value(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a nonnegative integer")
    return result


def _count_value(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(seed, num_train, num_test):
    train = _count_value("num_train", num_train, 2, 10)
    test = _count_value("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(test + 1, 2 + (seed // 4) % 9)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _choose_size(rng):
    if rng.random() < 0.10:
        return (int(rng.integers(4, 8)), int(rng.integers(4, 6)))
    width_band = int(rng.choice(4, p=(0.30, 0.30, 0.23, 0.17)))
    width_bounds = ((3, 6), (7, 12), (13, 21), (22, 30))[width_band]
    height_band = int(rng.choice(4, p=(0.18, 0.40, 0.27, 0.15)))
    height_bounds = ((3, 3), (4, 6), (7, 9), (10, 12))[height_band]
    return (
        int(rng.integers(width_bounds[0], width_bounds[1] + 1)),
        int(rng.integers(height_bounds[0], height_bounds[1] + 1)),
    )


def _choose_palette(rng, prior=None):
    """Colors are local to a pair; overlap with another pair is optional."""
    if prior is None:
        return tuple(int(v) for v in rng.choice(10, 3, replace=False))
    style = int(rng.integers(0, 4))
    if style == 0:  # Same background, independently changed foregrounds.
        background = prior[0]
        other = rng.choice([v for v in range(10) if v != background], 2, replace=False)
        return (background, int(other[0]), int(other[1]))
    if style == 1:  # One foreground shared, as in the source task.
        shared = prior[int(rng.integers(1, 3))]
        other = rng.choice([v for v in range(10) if v not in (shared, prior[0])],
                           2, replace=False)
        return (int(other[0]), shared, int(other[1]))
    if style == 2:  # An entirely different literal palette.
        other = rng.choice([v for v in range(10) if v not in prior], 3, replace=False)
        return tuple(int(v) for v in other)
    return tuple(int(v) for v in rng.choice(10, 3, replace=False))


def _prefix_height(rng, tile_height, category=None):
    maximum = 30 - tile_height
    choices = {
        "short": [p for p in range(1, min(tile_height, maximum + 1))],
        "exact": [p for p in range(tile_height, maximum + 1, tile_height)],
        "long": [p for p in range(2 * tile_height + 1, maximum + 1)],
        "partial": [p for p in range(1, maximum + 1) if p % tile_height],
        "edge": [maximum],
    }
    if category is None:
        category = ("short", "exact", "long", "partial", "edge")[
            int(rng.choice(5, p=(0.24, 0.18, 0.22, 0.27, 0.09)))
        ]
    available = choices[category]
    if not available:
        available = choices["partial"]
    return int(available[int(rng.integers(0, len(available)))])


def _draw_row(rng, width, kind="runs"):
    row = np.zeros(width, dtype=np.int8)
    if kind == "marks":
        count = int(rng.integers(1, max(2, width // 3 + 1)))
        cols = rng.choice(width, size=min(width, count), replace=False)
        row[cols] = rng.integers(1, 3, size=len(cols))
    else:
        length = int(rng.integers(1, width + 1))
        start = int(rng.integers(0, width - length + 1))
        row[start:start + length] = int(rng.integers(1, 3))
        if width >= 5 and rng.random() < 0.35:
            row[int(rng.integers(0, width))] = int(rng.integers(1, 3))
    return row


def _draw_roles(rng, height, width, motif, seam_equal):
    roles = np.zeros((height, width), dtype=np.int8)
    if motif in ("marks", "runs"):
        for r in range(height):
            roles[r] = _draw_row(rng, width, motif)
    elif motif == "objects":
        for _ in range(int(rng.integers(2, 6))):
            rh = int(rng.integers(2, height + 1))
            rw = int(rng.integers(1, width + 1))
            y = int(rng.integers(0, height - rh + 1))
            x = int(rng.integers(0, width - rw + 1))
            roles[y:y + rh, x:x + rw] = int(rng.integers(1, 3))
    elif motif == "bands":
        band_count = int(rng.integers(1, max(2, height // 2 + 1)))
        bands = rng.choice(height, size=min(height, band_count), replace=False)
        for r in range(height):
            if r in bands:
                roles[r, :] = int(rng.integers(1, 3))
                if rng.random() < 0.45:
                    hole_count = int(rng.integers(1, max(2, width // 3 + 1)))
                    holes = rng.choice(width, size=hole_count, replace=False)
                    roles[r, holes] = 0
            else:
                roles[r] = _draw_row(rng, width)
    elif motif == "holes":
        roles[:, :] = int(rng.integers(1, 3))
        # A second coherent region can be nested inside the first.
        top = int(rng.integers(0, height - 1))
        left = int(rng.integers(0, width - 1))
        bottom = int(rng.integers(top + 1, height + 1))
        right = int(rng.integers(left + 1, width + 1))
        roles[top:bottom, left:right] = 3 - roles[0, 0]
        for _ in range(int(rng.integers(1, 4))):
            rh = int(rng.integers(1, max(2, height // 2 + 1)))
            rw = int(rng.integers(1, max(2, width // 2 + 1)))
            y = int(rng.integers(0, height - rh + 1))
            x = int(rng.integers(0, width - rw + 1))
            roles[y:y + rh, x:x + rw] = 0
    elif motif == "row_groups":
        group_count = int(rng.integers(2, min(5, height + 1)))
        prototypes = [_draw_row(rng, width) for _ in range(group_count)]
        r = 0
        while r < height:
            prototype = prototypes[int(rng.integers(0, group_count))]
            run = int(rng.integers(1, min(4, height - r + 1)))
            roles[r:r + run] = prototype
            r += run
    elif motif == "periodic":
        divisors = [p for p in (2, 3, 4) if height % p == 0 and (not seam_equal or p >= 3)]
        period = int(divisors[int(rng.integers(0, len(divisors)))]) if divisors else 2
        prototypes = [_draw_row(rng, width) for _ in range(period)]
        prototypes[0][0] = 1
        prototypes[1][0] = 2
        if seam_equal and period >= 3:
            prototypes[-1] = prototypes[0].copy()
        elif not seam_equal and np.array_equal(prototypes[0], prototypes[-1]):
            prototypes[-1] = prototypes[1].copy()
        for r in range(height):
            roles[r] = prototypes[r % period]
    elif motif == "drift":
        center = int(rng.integers(0, width))
        step = int(rng.choice((-1, 1)))
        color = int(rng.integers(1, 3))
        for r in range(height):
            thickness = int(rng.integers(1, min(4, width + 1)))
            x = min(width - thickness, max(0, center))
            roles[r, x:x + thickness] = color
            center = min(width - 1, max(0, center + step + int(rng.integers(-1, 2))))
            if rng.random() < 0.3:
                color = 3 - color
    else:  # mirror
        for r in range(height):
            left = int(rng.integers(0, (width + 1) // 2))
            color = int(rng.integers(1, 3))
            roles[r, left] = color
            roles[r, width - 1 - left] = color
            if rng.random() < 0.5:
                next_left = int(rng.integers(0, (width + 1) // 2))
                roles[r, next_left] = 3 - color
                roles[r, width - 1 - next_left] = 3 - color

    # Each row of the bottom suffix is visibly active. This makes its top
    # boundary observable without knowing the seed or any output.
    for r in range(height):
        if not np.any(roles[r]):
            roles[r] = _draw_row(rng, width, "marks")
    if seam_equal:
        roles[-1] = roles[0]
    elif np.array_equal(roles[0], roles[-1]):
        col = int(rng.integers(0, width))
        roles[-1, col] = 1 if roles[-1, col] != 1 else 2
    if not np.any(roles == 1):
        roles[height // 2, int(rng.integers(0, width))] = 1
    if not np.any(roles == 2):
        roles[height // 2, int(rng.integers(0, width))] = 2
    # Repair may have changed a seam row in a three-row tile.
    if seam_equal:
        roles[-1] = roles[0]
    return roles


def _make_pair(rng, width, tile_height, prefix, colors, motif, seam_equal):
    roles = _draw_roles(rng, tile_height, width, motif, seam_equal)
    background, first, second = colors
    tile = np.full((tile_height, width), background, dtype=np.int8)
    tile[roles == 1] = first
    tile[roles == 2] = second
    total = prefix + tile_height
    inp = np.full((total, width), background, dtype=np.int8)
    inp[prefix:] = tile
    out = tile[(np.arange(total) - prefix) % tile_height].copy()
    return {"input": inp, "output": out}


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _source_copy(pair):
    inp = tuple("".join(str(int(v)) for v in row) for row in pair["input"])
    out = tuple("".join(str(int(v)) for v in row) for row in pair["output"])
    return (inp, out) in _SOURCE_PAIRS


def _pair_description(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    height, width = inp.shape
    background = int(inp[0, 0])
    assert np.all(inp[0] == background)
    active = np.any(inp != background, axis=1)
    starts = np.flatnonzero(active)
    assert starts.size
    prefix = int(starts[0])
    assert prefix >= 1
    assert np.all(~active[:prefix]) and np.all(active[prefix:])
    tile = inp[prefix:]
    tile_height = len(tile)
    assert 3 <= tile_height <= 12 and 3 <= width <= 30
    assert prefix + tile_height <= 30
    colors = set(int(v) for v in np.unique(inp))
    assert len(colors) == 3 and background in colors
    assert set(int(v) for v in np.unique(out)) <= colors
    assert np.array_equal(out, tile[(np.arange(height) - prefix) % tile_height])
    assert np.array_equal(out[prefix:], tile)
    assert not np.array_equal(inp, out)
    return {
        "height": height, "width": width, "prefix": prefix,
        "tile_height": tile_height, "background": background,
        "palette": frozenset(colors), "tile": tile,
        "seam_equal": bool(np.array_equal(tile[0], tile[-1])),
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _seed_value(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)

    # The only puzzle-wide conventions are visible in the demonstrations:
    # keep colors/dimensions and extend the full bottom suffix upward.
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_sizes = []
    first_palette = None
    first_motif = None
    equal_first = bool(rng.integers(0, 2))
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            local = np.random.default_rng(pair_seed)
            if split == "train":
                if index == 1 and train_count >= 3 and local.random() < 0.40:
                    width, tile_height = train_sizes[0]
                else:
                    width, tile_height = _choose_size(local)
                train_sizes.append((width, tile_height))
                category = "exact" if index == 0 else ("partial" if index == 1 else None)
                seam_equal = equal_first if index == 0 else (
                    not equal_first if index == 1 else bool(local.integers(0, 2))
                )
            else:
                if test_count >= 2:
                    familiar = index == 0
                else:
                    familiar = bool(local.integers(0, 2))
                if familiar:
                    width, tile_height = train_sizes[int(local.integers(0, len(train_sizes)))]
                else:
                    for _ in range(20):
                        width, tile_height = _choose_size(local)
                        if (width, tile_height) not in train_sizes:
                            break
                    else:
                        width, tile_height = next(
                            (w, h) for w in range(3, 31) for h in range(3, 13)
                            if (w, h) not in train_sizes
                        )
                category = None
                seam_equal = bool(local.integers(0, 2))

            prefix = _prefix_height(local, tile_height, category)
            colors = _choose_palette(local, first_palette if split == "train" and index == 1 else None)
            motif = _MOTIFS[int(local.choice(
                len(_MOTIFS),
                p=(0.10, 0.10, 0.12, 0.10, 0.10, 0.13, 0.18, 0.085, 0.085),
            ))]
            if split == "train" and index == 1:
                while motif == first_motif:
                    motif = _MOTIFS[int(local.choice(
                        len(_MOTIFS),
                        p=(0.10, 0.10, 0.12, 0.10, 0.10, 0.13, 0.18, 0.085, 0.085),
                    ))]
            if split == "train" and index == 0:
                first_palette, first_motif = colors, motif

            for _ in range(30):
                pair = _make_pair(local, width, tile_height, prefix, colors, motif, seam_equal)
                key = _grid_key(pair["input"])
                if key not in used_inputs and not _source_copy(pair):
                    used_inputs.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                fallback = None
                for background in range(10):
                    if fallback is not None:
                        break
                    for first in range(10):
                        if fallback is not None:
                            break
                        for second in range(10):
                            if len({background, first, second}) < 3:
                                continue
                            candidate = _make_pair(
                                local, width, tile_height, prefix,
                                (background, first, second), motif, seam_equal,
                            )
                            key = _grid_key(candidate["input"])
                            if key not in used_inputs and not _source_copy(candidate):
                                fallback = candidate
                                used_inputs.add(key)
                                puzzle[split].append(candidate)
                                break
                if fallback is None:
                    raise RuntimeError("unable to create a unique pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    keys = set()
    descriptions = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _grid_key(pair["input"])
            assert key not in keys
            keys.add(key)
            assert not _source_copy(pair)
            descriptions[split].append(_pair_description(pair))

    train = descriptions["train"]
    assert any(d["prefix"] >= d["tile_height"] for d in train)
    assert any(d["prefix"] % d["tile_height"] for d in train)
    assert {d["seam_equal"] for d in train} == {False, True}
    # Color roles and geometry are local; no literal palette, width, motif,
    # or tile height must be inferred from another pair.
    for d in descriptions["test"]:
        assert d["prefix"] >= 1 and d["tile_height"] >= 3
    if test_count >= 2:
        train_sizes = {(d["width"], d["tile_height"]) for d in train}
        test_sizes = {(d["width"], d["tile_height"]) for d in descriptions["test"]}
        assert test_sizes & train_sizes
        assert test_sizes - train_sizes


def _puzzles_equal(left, right):
    return all(
        len(left[split]) == len(right[split])
        and all(
            np.array_equal(a[role], b[role])
            for a, b in zip(left[split], right[split])
            for role in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_grid_key(p["train"][0]["input"]) for p in puzzles}) >= 40
    all_descriptions = []
    seen_motifs = set()
    review_counts = {name: 0 for name in (
        "shared_size_new_background", "same_background_new_foregrounds",
        "shared_foreground", "disjoint_palettes", "familiar_test_size",
        "new_test_size",
    )}
    split_counts = {split: {name: 0 for name in (
        "short", "exact", "long", "partial", "seam", "adjacent",
        "internal_period", "full_row", "sparse", "source_like",
    )} for split in ("train", "test")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        per_split = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _pair_description(pair)
                all_descriptions.append((split, d))
                per_split[split].append(d)
                # Independent reconstruction uses only the visible input.
                inp = pair["input"]
                bg = int(inp[0, 0])
                start = next(r for r in range(len(inp)) if np.any(inp[r] != bg))
                suffix = inp[start:]
                predicted = np.stack([suffix[(r - start) % len(suffix)]
                                      for r in range(len(inp))])
                assert np.array_equal(pair["output"], predicted)
                seen_motifs.add(_grid_key(suffix))
                h, prefix, tile, bg = d["tile_height"], d["prefix"], d["tile"], d["background"]
                counts = split_counts[split]
                counts["short"] += prefix < h
                counts["exact"] += prefix % h == 0
                counts["long"] += prefix >= 2 * h
                counts["partial"] += prefix % h != 0
                counts["seam"] += d["seam_equal"]
                counts["adjacent"] += any(np.array_equal(tile[i], tile[i + 1]) for i in range(h - 1))
                counts["internal_period"] += any(
                    h % k == 0 and all(np.array_equal(tile[i], tile[i % k]) for i in range(h))
                    for k in range(1, h)
                )
                counts["full_row"] += bool(np.any(np.all(tile != bg, axis=1)))
                counts["sparse"] += bool(np.mean(tile != bg) <= 0.25)
                counts["source_like"] += d["width"] in (5, 7) and h in (4, 5)
        train, test = per_split["train"], per_split["test"]
        review_counts["shared_size_new_background"] += any(
            a["width"] == b["width"] and a["tile_height"] == b["tile_height"]
            and a["background"] != b["background"]
            for i, a in enumerate(train) for b in train[i + 1:]
        )
        all_rows = train + test
        review_counts["same_background_new_foregrounds"] += any(
            a["background"] == b["background"]
            and a["palette"] - {a["background"]} != b["palette"] - {b["background"]}
            for i, a in enumerate(all_rows) for b in all_rows[i + 1:]
        )
        review_counts["shared_foreground"] += any(
            (a["palette"] - {a["background"]}) & (b["palette"] - {b["background"]})
            for i, a in enumerate(train) for b in train[i + 1:]
        )
        review_counts["disjoint_palettes"] += any(
            not (a["palette"] & b["palette"])
            for i, a in enumerate(train) for b in train[i + 1:]
        )
        train_sizes = {(d["width"], d["tile_height"]) for d in train}
        review_counts["familiar_test_size"] += any(
            (d["width"], d["tile_height"]) in train_sizes for d in test
        )
        review_counts["new_test_size"] += any(
            (d["width"], d["tile_height"]) not in train_sizes for d in test
        )
    assert len(seen_motifs) >= 150
    assert len({d["width"] for _, d in all_descriptions}) >= 16
    assert len({d["tile_height"] for _, d in all_descriptions}) >= 9
    assert len({d["palette"] for _, d in all_descriptions}) >= 50
    assert len({d["background"] for _, d in all_descriptions}) == 10
    assert any(d["prefix"] < d["tile_height"] for _, d in all_descriptions)
    assert any(d["prefix"] >= 2 * d["tile_height"] for _, d in all_descriptions)
    assert any(d["height"] == 30 for _, d in all_descriptions)
    assert any(np.any(np.all(d["tile"] != d["background"], axis=1))
               for _, d in all_descriptions)
    assert {d["seam_equal"] for s, d in all_descriptions if s == "test"} == {False, True}
    assert any(np.array_equal(d["tile"][r], d["tile"][r + 1])
               for _, d in all_descriptions
               for r in range(d["tile_height"] - 1))
    assert any(len({tuple(row) for row in d["tile"]}) <= 2
               for _, d in all_descriptions)
    assert review_counts["shared_size_new_background"] >= 5
    assert all(review_counts[name] >= 15 for name in review_counts if name != "shared_size_new_background")
    for split in ("train", "test"):
        counts = split_counts[split]
        assert all(counts[name] >= 5 for name in ("short", "exact", "long", "partial"))
        assert all(counts[name] >= 5 for name in ("seam", "adjacent", "full_row", "sparse"))
        assert counts["internal_period"] >= 3
        assert counts["source_like"] >= 3
    for seed, train, test in ((1000, 2, 1), (1001, 10, 1),
                              (1002, 10, 4), (1003, 7, 3)):
        puzzle = generate_puzzle(seed, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(2000, train, test))
    assert _puzzles_equal(generate_puzzle(31415, 8, 3),
                          generate_puzzle(31415, 8, 3))
    assert _puzzles_equal(generate_puzzle(2718), generate_puzzle(2718))
    for train, test in ((1, None), (11, None), (None, 0),
                        (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count was accepted")
