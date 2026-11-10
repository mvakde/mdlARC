"""Generate puzzles that colour the five regions of a divided cross."""

from __future__ import annotations

import numpy as np


# For this family geometry and the ordered palette uniquely determine a pair.
# Only these exact source-pair specifications are excluded.
_SOURCE_PALETTE = (0, 8, 2, 4, 6, 3, 1)
_SOURCE_GEOMETRIES = {
    (12, 14, 4, 7, 2, 9),
    (18, 19, 2, 7, 4, 11),
    (17, 15, 6, 13, 3, 8),
}


def _validate_optional_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _validate_optional_count(num_train, "num_train", 2, 10)
    num_test = _validate_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _construct_output(input_grid, geometry, roles):
    _, _, r1, r2, c1, c2 = geometry
    top, left, center, right, bottom = roles
    output = input_grid.copy()
    output[:r1, c1 + 1:c2] = top
    output[r1 + 1:r2, :c1] = left
    output[r1 + 1:r2, c1 + 1:c2] = center
    output[r1 + 1:r2, c2 + 1:] = right
    output[r2 + 1:, c1 + 1:c2] = bottom
    return output


def _axis_options(length, forbidden=None):
    """All positive three-band partitions at a full-grid length."""
    area = length - 2
    return [
        (a, b, area - a - b)
        for a in range(1, area - 1)
        for b in range(1, area - a)
        if forbidden is None or all(
            value != old for value, old in zip((a, b, area - a - b), forbidden)
        )
    ]


def _sample_axis(rng, length, forbidden=None):
    options = _axis_options(length, forbidden)
    if not options:
        raise ValueError("no admissible three-band partition")
    # Independent preferences expose thin, dominant, and symmetric centres.
    for probability, condition in (
        (0.23, lambda x: x[1] == 1),
        (0.23, lambda x: x[1] > max(x[0], x[2])),
        (0.22, lambda x: x[0] == x[2]),
    ):
        if rng.random() < probability:
            subset = [x for x in options if condition(x)]
            if subset:
                options = subset
    return options[int(rng.integers(len(options)))]


def _sample_geometry(rng, limits, forbidden=None, forced_shape=None):
    (hlo, hhi), (wlo, whi) = limits
    row_forbidden = None if forbidden is None else forbidden[:3]
    col_forbidden = None if forbidden is None else forbidden[3:]
    if forced_shape is None:
        heights = [n for n in range(hlo, hhi + 1) if _axis_options(n, row_forbidden)]
        widths = [n for n in range(wlo, whi + 1) if _axis_options(n, col_forbidden)]
        if not heights or not widths:
            raise ValueError("dimension range has no admissible bands")
        height = heights[int(rng.integers(len(heights)))]
        width = widths[int(rng.integers(len(widths)))]
        if hlo <= 9 and hhi >= 22 and wlo <= 9 and whi >= 22 and rng.random() < 0.19:
            short_heights = [n for n in heights if n <= 9]
            tall_heights = [n for n in heights if n >= 22]
            short_widths = [n for n in widths if n <= 9]
            wide_widths = [n for n in widths if n >= 22]
            if rng.random() < 0.5 and short_heights and wide_widths:
                height = int(rng.choice(short_heights))
                width = int(rng.choice(wide_widths))
            elif tall_heights and short_widths:
                height = int(rng.choice(tall_heights))
                width = int(rng.choice(short_widths))
        if forbidden is None and hlo <= 5 and hhi >= 30 and wlo <= 5 and whi >= 30:
            if rng.random() < 0.035:
                height = width = 5 if rng.random() < 0.5 else 30
    else:
        height, width = forced_shape
    rows = _sample_axis(rng, height, row_forbidden)
    cols = _sample_axis(rng, width, col_forbidden)
    return rows + cols


def _geometry_from_bands(bands):
    top, middle_h, bottom, left, middle_w, right = bands
    return (
        top + middle_h + bottom + 2,
        left + middle_w + right + 2,
        top,
        top + middle_h + 1,
        left,
        left + middle_w + 1,
    )


def _limits(regime, shift):
    if regime == "source_train":
        if shift == "grow":
            return ((5, 17), (5, 17))
        if shift == "shrink":
            return ((20, 30), (20, 30))
    elif regime == "target_test":
        if shift == "grow":
            return ((20, 30), (20, 30))
        if shift == "shrink":
            return ((5, 17), (5, 17))
    else:
        raise ValueError(f"unknown regime: {regime}")
    if shift == "overlap":
        return ((8, 24), (8, 24))
    return ((5, 30), (5, 30))


def _recombine_bands(rng, train_bands):
    values = [[bands[i] for bands in train_bands] for i in range(6)]
    for _ in range(32):
        chosen = tuple(int(rng.choice(v)) for v in values)
        if sum(chosen[:3]) + 2 <= 30 and sum(chosen[3:]) + 2 <= 30:
            return chosen
    return None


def _pick_bands(rng, regime, shift, used_geometries, forbidden=None,
                forced_shape=None, train_bands=None, source_palette=False):
    limits = _limits(regime, shift)
    if (regime == "target_test" and shift in ("overlap", "mixed")
            and train_bands and rng.random() < 0.38 and forced_shape is None):
        for _ in range(32):
            bands = _recombine_bands(rng, train_bands)
            if bands is None:
                break
            geometry = _geometry_from_bands(bands)
            if (geometry not in used_geometries
                    and not (source_palette and geometry in _SOURCE_GEOMETRIES)):
                return bands
    for attempt in range(96):
        bands = _sample_geometry(rng, limits, forbidden,
                                 forced_shape if attempt == 0 else None)
        geometry = _geometry_from_bands(bands)
        if (geometry not in used_geometries
                and not (source_palette and geometry in _SOURCE_GEOMETRIES)):
            return bands
    # Bounded deterministic fallback for crowded explicit-count requests.
    (hlo, hhi), (wlo, whi) = limits
    for h in range(hlo, hhi + 1):
        for rows in _axis_options(h, None if forbidden is None else forbidden[:3]):
            for w in range(wlo, whi + 1):
                for cols in _axis_options(w, None if forbidden is None else forbidden[3:]):
                    bands = rows + cols
                    geometry = _geometry_from_bands(bands)
                    if (geometry not in used_geometries
                            and not (source_palette and geometry in _SOURCE_GEOMETRIES)):
                        return bands
    raise RuntimeError("no unused legal geometry")


def _structural_colors(rng, available, first, kind):
    background, divider = first
    if kind == "swap":
        return divider, background
    if kind == "one":
        if rng.random() < 0.5:
            return int(rng.choice([c for c in available if c not in first])), divider
        return background, int(rng.choice([c for c in available if c not in first]))
    if kind == "both":
        choices = [(a, b) for a in available for b in available
                   if a != b and a != background and b != divider]
        return choices[int(rng.integers(len(choices)))]
    raise ValueError("unknown recoloring mode")


def _make_pair(bands, structural, roles):
    geometry = _geometry_from_bands(bands)
    height, width, r1, r2, c1, c2 = geometry
    background, divider = structural
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[[r1, r2], :] = divider
    input_grid[:, [c1, c2]] = divider
    return {"input": input_grid, "output": _construct_output(input_grid, geometry, roles)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
    available = tuple(c for c in range(10) if c not in roles)
    first = tuple(int(x) for x in rng.choice(available, size=2, replace=False))
    varied_palette = bool(rng.random() < 0.7)
    shift = ("grow", "shrink", "overlap", "mixed")[int(rng.integers(4))]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=num_train + num_test, dtype=np.int64)
    structural = [first]
    if varied_palette:
        kind = ("swap", "one", "both")[int(rng.integers(3))]
        structural.append(_structural_colors(rng, available, first, kind))
    else:
        structural.append(first)
    for i in range(2, num_train + num_test):
        if not varied_palette or rng.random() < 0.28:
            structural.append(first)
        else:
            kind = ("swap", "one", "both")[int(rng.integers(3))]
            structural.append(_structural_colors(rng, available, first, kind))

    puzzle = {"train": [], "test": []}
    used_geometries = set()
    train_bands = []
    for i in range(num_train + num_test):
        regime = "source_train" if i < num_train else "target_test"
        split = "train" if i < num_train else "test"
        pair_rng = np.random.default_rng(int(pair_seeds[i]))
        forced_shape = None
        if seed % 9 == 0 and ((shift == "shrink" and i == num_train)
                              or (shift != "shrink" and i == 0)):
            forced_shape = (5, 5)
        if seed % 9 == 1 and ((shift == "grow" and i == num_train)
                              or (shift != "grow" and i == 0)):
            forced_shape = (30, 30)
        forbidden = train_bands[0] if i == 1 else None
        pair_palette = structural[i] + roles
        bands = _pick_bands(pair_rng, regime, shift, used_geometries,
                            forbidden=forbidden, forced_shape=forced_shape,
                            train_bands=train_bands,
                            source_palette=pair_palette == _SOURCE_PALETTE)
        geometry = _geometry_from_bands(bands)
        used_geometries.add(geometry)
        if split == "train":
            train_bands.append(bands)
        puzzle[split].append(_make_pair(bands, structural[i], roles))
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    canonical = grid.astype(np.int8, copy=False)
    return canonical.shape, canonical.tobytes()


def _analyze_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    input_colors = [int(c) for c in np.unique(input_grid)]
    assert len(input_colors) == 2, "input needs exactly two structural colors"
    candidates = []
    for color in input_colors:
        rows = np.flatnonzero(np.all(input_grid == color, axis=1))
        cols = np.flatnonzero(np.all(input_grid == color, axis=0))
        if len(rows) == len(cols) == 2:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "divider lines must be identifiable"
    divider, rows, cols = candidates[0]
    background = next(c for c in input_colors if c != divider)
    r1, r2 = (int(x) for x in rows)
    c1, c2 = (int(x) for x in cols)
    height, width = input_grid.shape
    bands = (r1, r2-r1-1, height-r2-1, c1, c2-c1-1, width-c2-1)
    assert min(bands) >= 1, "all nine rectangles must have positive area"
    expected_input = np.full(input_grid.shape, background, dtype=input_grid.dtype)
    expected_input[[r1, r2], :] = divider
    expected_input[:, [c1, c2]] = divider
    assert np.array_equal(input_grid, expected_input), "malformed divider input"
    assert output_grid.shape == input_grid.shape, "output shape changed"
    roles = (int(output_grid[0, c1+1]), int(output_grid[r1+1, 0]),
             int(output_grid[r1+1, c1+1]), int(output_grid[r1+1, c2+1]),
             int(output_grid[r2+1, c1+1]))
    palette = (background, divider) + roles
    assert len(set(palette)) == 7, "seven semantic roles must have distinct colors"
    geometry = (height, width, r1, r2, c1, c2)
    assert np.array_equal(output_grid, _construct_output(input_grid, geometry, roles)), (
        "cross fill or preserved cells are wrong")
    return {"geometry": geometry, "bands": bands, "palette": palette, "roles": roles}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    analyses = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            inp = _grid_key(pair["input"])
            key = (inp, _grid_key(pair["output"]))
            assert inp not in seen_inputs and key not in seen_pairs, "duplicate pair or input"
            seen_inputs.add(inp)
            seen_pairs.add(key)
            analysis = _analyze_pair(pair)
            assert not (analysis["palette"] == _SOURCE_PALETTE and
                        analysis["geometry"] in _SOURCE_GEOMETRIES), "exact source pair"
            analyses[split].append(analysis)
    learned_roles = analyses["train"][0]["roles"]
    for split in ("train", "test"):
        for pair, analysis in zip(puzzle[split], analyses[split]):
            assert analysis["roles"] == learned_roles, "learned fill roles changed"
            assert np.array_equal(pair["output"], _construct_output(
                pair["input"], analysis["geometry"], learned_roles)), (
                "test not recoverable from training")
    first, second = analyses["train"][:2]
    assert all(a != b for a, b in zip(first["bands"], second["bands"])), (
        "all six band positions must vary within training")


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert any(all(_analyze_pair(pair)["palette"] == _analyze_pair(p["train"][0])["palette"]
                   for split in ("train", "test") for pair in p[split]) for p in puzzles)
    assert any(_analyze_pair(p["train"][0])["palette"][:2] !=
               _analyze_pair(p["train"][1])["palette"][:2] for p in puzzles)
    records = [_analyze_pair(pair) for p in puzzles
               for split in ("train", "test") for pair in p[split]]
    assert any(r["geometry"][:2] == (5, 5) for r in records)
    assert any(r["geometry"][:2] == (30, 30) for r in records)
    assert any(r["bands"][1] == 1 for r in records)
    assert any(r["bands"][4] == 1 for r in records)
    assert any(r["bands"][1] > max(r["bands"][0], r["bands"][2]) for r in records)
    assert any(r["bands"][4] > max(r["bands"][3], r["bands"][5]) for r in records)
    assert any(r["bands"][0] == r["bands"][2] for r in records)
    assert any(r["bands"][3] == r["bands"][5] for r in records)
    assert len({r["geometry"] for r in records}) > 140
    assert len({r["geometry"][:2] for r in records}) > 70
    for counts in ((2, 1), (10, 1), (10, 4), (5, 2), (8, 3)):
        p = generate_puzzle(1000 + counts[0] * 10 + counts[1], *counts)
        assert (len(p["train"]), len(p["test"])) == counts
    for requested in ((None, 1), (None, 4), (2, None), (10, None)):
        generate_puzzle(2000, *requested)
    for requested in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(3000, *requested)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    a = generate_puzzle(8675309, 7, 3)
    b = generate_puzzle(8675309, 7, 3)
    c = generate_puzzle(8675310, 7, 3)
    key = lambda p: tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                          for split in ("train", "test") for pair in p[split])
    assert key(a) == key(b) and key(a) != key(c)
