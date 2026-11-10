"""Cyclic next-band color transfer for separated ARC color bands."""

import numpy as np


def _rows(*lines):
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int8)


_SOURCE_PAIRS = (
    (_rows("42446455", "24446445", "42446455", "24464445", "42464455",
           "24464445", "42464455", "24446445", "42446455", "24446445"),
     _rows("62456455", "26456425", "62456455", "26465425", "62465455",
           "26465425", "62465455", "26456425", "62456455", "26456425")),
    (_rows("00000009", "40003090", "04000009", "00403009", "04000090",
           "40003009", "04000009", "00403090", "04000009", "40003009"),
     _rows("33309049", "43303094", "34309049", "33403049", "34309094",
           "43303049", "34309049", "33403094", "34309049", "43303049")),
    (_rows("65599555", "65595588", "56559585", "56599558", "65599588",
           "65595555", "56559588", "56599585", "65599558", "65595588"),
     _rows("69599566", "69598588", "96589586", "96599568", "69599588",
           "69598566", "96589588", "96599586", "69599568", "69598588")),
)


def _count(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return value


def _counts(rng, seed, num_train, num_test):
    tr = _count("num_train", num_train, 2, 10)
    te = _count("num_test", num_test, 1, 4)
    if tr is not None and te is not None:
        if tr <= te:
            raise ValueError("num_train must exceed num_test")
    elif tr is None and te is None:
        te = 1 + seed % 4
        tr = max(2, te + 1) + (seed // 4 + 3 * seed) % (11 - max(2, te + 1))
    elif tr is None:
        tr = int(rng.integers(max(2, te + 1), 11))
    else:
        te = int(rng.integers(1, min(4, tr - 1) + 1))
    return tr, te


def _sample_span(rng, minimum):
    regime = int(rng.choice(3, p=[0.22, 0.50, 0.28]))
    bounds = ((minimum, 11), (max(minimum, 12), 23), (max(minimum, 24), 30))
    lo, hi = bounds[regime]
    return int(rng.integers(lo, hi + 1))


def _geometry(rng, band_count):
    axis = _sample_span(rng, 2 * band_count - 1)
    cross = _sample_span(rng, 5)
    gaps = [int(rng.integers(1, 4)) for _ in range(band_count - 1)]
    if axis - sum(gaps) < band_count:
        gaps = [1] * (band_count - 1)
    total = axis - sum(gaps)

    if band_count == 3:
        style = int(rng.choice(4, p=[0.23, 0.28, 0.32, 0.17]))
        if style == 0:
            # Move the sampled axis by at most two cells to permit equal bands.
            equal = max(1, min((30 - sum(gaps)) // 3, round(total / 3)))
            widths = [equal] * 3
        elif style == 1:
            candidates = [(a, a, total - 2 * a) for a in range(1, total // 2 + 1)
                          if total - 2 * a >= 1 and total - 2 * a != a]
            if candidates:
                widths = list(candidates[int(rng.integers(len(candidates)))])
                rng.shuffle(widths)
            else:
                widths = None
        elif style == 2:
            candidates = [(a, b, total - a - b) for a in range(1, total - 1)
                          for b in range(1, total - a)
                          if total - a - b >= 1 and len({a, b, total - a - b}) == 3]
            widths = list(candidates[int(rng.integers(len(candidates)))]) if candidates else None
        else:
            widths = None
    else:
        widths = None

    if widths is None:
        concentration = float(rng.uniform(0.45, 2.0))
        proportions = rng.dirichlet(np.full(band_count, concentration))
        widths = list(1 + rng.multinomial(total - band_count, proportions))
    assert sum(widths) + sum(gaps) <= 30
    return cross, tuple(int(x) for x in widths), tuple(gaps)


def _colors(rng, band_count, first_palette, pair_number):
    needed = band_count + 1
    if pair_number == 0:
        return tuple(int(x) for x in rng.choice(10, size=needed, replace=False))
    if pair_number == 1:
        # A visible role exchange in training: a prior seed becomes background.
        fixed = [int(rng.choice(first_palette[1:])), int(first_palette[0])]
        remaining = [x for x in range(10) if x not in fixed]
        others = [int(x) for x in rng.choice(remaining, size=needed - 2, replace=False)]
        rng.shuffle(others)
        return tuple(fixed + others)

    mode = int(rng.choice(4, p=[0.42, 0.27, 0.16, 0.15]))
    old = list(first_palette)
    if mode == 2 and 10 - len(old) >= needed:
        pool = [x for x in range(10) if x not in old]
        colors = rng.choice(pool, size=needed, replace=False)
    elif mode in (1, 3):
        overlap = min(len(old), needed, int(rng.integers(1, min(4, needed) + 1)))
        selected = [int(x) for x in rng.choice(old, size=overlap, replace=False)]
        pool = [x for x in range(10) if x not in selected]
        colors = selected + [int(x) for x in rng.choice(pool, size=needed - overlap, replace=False)]
    else:
        colors = rng.choice(10, size=needed, replace=False)
    colors = list(map(int, colors))
    rng.shuffle(colors)
    return tuple(colors)


def _mask(rng, height, width, teaching):
    mode = int(rng.integers(5))
    density = float(rng.uniform(0.30, 0.70) if teaching else rng.uniform(0.16, 0.90))
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    phase = int(rng.integers(0, 13))
    noise = rng.random((height, width))
    if mode == 0:
        scores = noise + 1.4 * ((rows + cols + phase) % 2 == 0)
    elif mode == 1:
        period = int(rng.integers(2, 6))
        scores = noise + 1.5 * ((rows + 2 * cols + phase) % period == 0)
    elif mode == 2:
        scores = noise.copy()
        for _ in range(int(rng.integers(1, 4))):
            r0 = int(rng.integers(height))
            c0 = int(rng.integers(width))
            run = int(rng.integers(2, max(3, height // 2 + 2)))
            scores[r0:min(height, r0 + run), c0:min(width, c0 + int(rng.integers(1, width + 1)))] += 1.7
    elif mode == 3:
        scores = noise
    else:
        period = int(rng.integers(3, 7))
        scores = noise + 1.3 * ((3 * rows - cols + phase) % period <= 1)

    target = int(round(density * height * width))
    target = min(height * width - 1, max(width, target))
    mask = np.zeros((height, width), dtype=bool)
    for col in range(width):
        mask[int(np.argmax(scores[:, col])), col] = True
    if width >= 2 and height >= 6 and rng.random() < 0.20:
        solid = int(rng.integers(width))
        mask[:, solid] = True
        target = max(target, int(mask.sum()))
    remaining = target - int(mask.sum())
    if remaining:
        eligible = np.argwhere(~mask)
        order = np.argsort(scores[~mask], kind="stable")[::-1]
        chosen = eligible[order[:remaining]]
        mask[chosen[:, 0], chosen[:, 1]] = True
    assert np.all(mask.any(axis=0)) and np.any(~mask)
    return mask


def _axis_analysis(matrix):
    height, width = matrix.shape
    colors = {int(x) for x in np.unique(matrix)}
    candidates = []
    for background in colors:
        empty = np.all(matrix == background, axis=0)
        if empty[0] or empty[-1]:
            continue
        bands = []
        gaps = []
        at = 0
        while at < width:
            start = at
            kind = bool(empty[at])
            while at < width and bool(empty[at]) == kind:
                at += 1
            (gaps if kind else bands).append((start, at))
        if not 3 <= len(bands) <= 5 or len(gaps) != len(bands) - 1:
            continue
        seeds = []
        densities = []
        valid = True
        for start, stop in bands:
            band = matrix[:, start:stop]
            other = {int(x) for x in np.unique(band)} - {background}
            if len(other) != 1 or not np.any(band == background):
                valid = False
                break
            seed = next(iter(other))
            if not np.all(np.any(band == seed, axis=0)):
                valid = False
                break
            seeds.append(seed)
            densities.append(float(np.mean(band == seed)))
        if valid and len(set(seeds)) == len(seeds) and colors == {background, *seeds}:
            candidates.append({"background": background, "bands": tuple(bands),
                               "gaps": tuple(gaps), "seeds": tuple(seeds),
                               "densities": tuple(densities),
                               "widths": tuple(b - a for a, b in bands),
                               "span": (height, width)})
    return candidates


def _analyze_input(grid):
    candidates = []
    for orientation, matrix in (("vertical", grid), ("horizontal", grid.T)):
        for info in _axis_analysis(matrix):
            candidates.append({**info, "orientation": orientation})
    assert len(candidates) == 1, "input bands must have one visible interpretation"
    return candidates[0]


def _apply_rule_from_input(grid):
    info = _analyze_input(grid)
    matrix = grid if info["orientation"] == "vertical" else grid.T
    result = matrix.copy()
    seeds = info["seeds"]
    for index, (start, stop) in enumerate(info["bands"]):
        region = result[:, start:stop]
        region[matrix[:, start:stop] == info["background"]] = seeds[(index + 1) % len(seeds)]
    return result if info["orientation"] == "vertical" else result.T.copy()


def _make_pair(rng, orientation, band_count, first_palette, pair_number, teaching):
    cross, widths, gaps = _geometry(rng, band_count)
    colors = _colors(rng, band_count, first_palette, pair_number)
    background, seeds = colors[0], colors[1:]
    matrix = np.full((cross, sum(widths) + sum(gaps)), background, dtype=np.int8)
    at = 0
    for index, width in enumerate(widths):
        mask = _mask(rng, cross, width, teaching)
        matrix[:, at:at + width][mask] = seeds[index]
        at += width + (gaps[index] if index < len(gaps) else 0)
    grid = matrix if orientation == "vertical" else matrix.T.copy()
    return {"input": grid, "output": _apply_rule_from_input(grid)}


def _fallback_pair(rng, orientation, band_count, first_palette, pair_number, used_shapes):
    # Distinct visible shapes give a guaranteed exit even for unusually unlucky
    # random draws. At most fourteen pairs precede this call, versus 26 spans.
    widths = (2,) * band_count
    axis = 3 * band_count - 1
    colors = _colors(rng, band_count, first_palette, pair_number)
    for cross in range(5, 31):
        shape = (cross, axis) if orientation == "vertical" else (axis, cross)
        if shape in used_shapes:
            continue
        matrix = np.full((cross, axis), colors[0], dtype=np.int8)
        at = 0
        for index in range(band_count):
            row = np.arange(cross)[:, None]
            column = np.arange(2)[None, :]
            mask = (row + column + pair_number + index) % 3 == 0
            matrix[:, at:at + 2][mask] = colors[index + 1]
            at += 3
        grid = matrix if orientation == "vertical" else matrix.T.copy()
        pair = {"input": grid, "output": _apply_rule_from_input(grid)}
        if not _source_copy(pair["input"], pair["output"]):
            return pair
    raise AssertionError("no unused fallback shape")


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _structural_key(grid):
    info = _analyze_input(grid)
    matrix = grid if info["orientation"] == "vertical" else grid.T
    masks = tuple((matrix[:, a:b] == seed).tobytes()
                  for (a, b), seed in zip(info["bands"], info["seeds"]))
    return info["orientation"], info["span"], info["widths"], info["gaps"], masks


def _source_copy(inp, out):
    return any(np.array_equal(inp, a) and np.array_equal(out, b)
               for a, b in _SOURCE_PAIRS)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    orientation = "vertical" if rng.random() < 0.70 else "horizontal"
    maximum_bands = int(rng.choice([3, 4, 5], p=[0.36, 0.30, 0.34]))
    puzzle = {"train": [], "test": []}
    first_palette = None
    first_shape = None
    seen = set()
    pair_number = 0
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            band_count = (3 if pair_number == 0 else maximum_bands if pair_number == 1
                          else int(rng.integers(3, maximum_bands + 1)))
            for _ in range(24):
                child = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
                pair = _make_pair(child, orientation, band_count, first_palette,
                                  pair_number, pair_number == 0)
                key = _grid_key(pair["input"])
                structure = _structural_key(pair["input"])
                if (key in seen or _source_copy(pair["input"], pair["output"])
                        or (pair_number == 1 and pair["input"].shape == first_shape)
                        or any(_structural_key(p["input"]) == structure
                               for p in puzzle["train"] + puzzle["test"])):
                    continue
                break
            else:
                used_shapes = {p["input"].shape for p in puzzle["train"] + puzzle["test"]}
                pair = _fallback_pair(rng, orientation, band_count, first_palette,
                                      pair_number, used_shapes)
                key = _grid_key(pair["input"])
            puzzle[split].append(pair)
            seen.add(key)
            if pair_number == 0:
                info = _analyze_input(pair["input"])
                first_palette = (info["background"], *info["seeds"])
                first_shape = pair["input"].shape
            pair_number += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    structures = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= length <= 30 for length in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            ikey, pkey, skey = _grid_key(inp), (_grid_key(inp), _grid_key(out)), _structural_key(inp)
            assert ikey not in inputs and pkey not in pairs and skey not in structures
            inputs.add(ikey)
            pairs.add(pkey)
            structures.add(skey)
            assert not _source_copy(inp, out)
            info = _analyze_input(inp)
            assert np.array_equal(out, _apply_rule_from_input(inp))
            assert set(map(int, np.unique(inp))) == set(map(int, np.unique(out)))
            infos[split].append(info)
    train = infos["train"]
    test = infos["test"]
    orientations = {info["orientation"] for info in train + test}
    assert len(orientations) == 1
    assert len(train[0]["bands"]) == 3
    assert puzzle["train"][0]["input"].shape != puzzle["train"][1]["input"].shape
    # Each input has holes in every band. The first three-band training pair
    # therefore demonstrates all transfers, including wraparound, and the
    # opposite cyclic direction predicts a different output.
    for info in train:
        assert all(0 < density < 1 for density in info["densities"])
        if len(info["bands"]) >= 3:
            assert all(info["seeds"][(i + 1) % len(info["seeds"])]
                       != info["seeds"][(i - 1) % len(info["seeds"])]
                       for i in range(len(info["seeds"])))
    assert all(info["orientation"] == train[0]["orientation"] for info in test)


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    assert _puzzle_key(generate_puzzle(13579)) == _puzzle_key(generate_puzzle(13579))
    assert _puzzle_key(generate_puzzle(24680, 7, 3)) == _puzzle_key(generate_puzzle(24680, 7, 3))
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(3000 + (tr or 0) * 10 + (te or 0), tr, te)
        validate_generated_puzzle(p)
        if tr is not None:
            assert len(p["train"]) == tr
        if te is not None:
            assert len(p["test"]) == te
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(5000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts, shapes, palettes, structures = set(), set(), set(), set(), set()
    orientations, band_counts, layouts, gaps, densities = set(), set(), set(), set(), []
    small = large = solid = overlap = disjoint = 0
    split_width_styles = {"train": set(), "test": set()}
    split_sparse = {"train": 0, "test": 0}
    split_dense = {"train": 0, "test": 0}
    split_long_gaps = {"train": 0, "test": 0}
    split_combined = {"train": 0, "test": 0}
    test_new_color_counts = set()
    test_role_exchange = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first_colors = set(map(int, np.unique(puzzle["train"][0]["input"])))
        assert len({p["input"].shape for p in puzzle["train"] + puzzle["test"]}) >= 2
        first_info = _analyze_input(puzzle["train"][0]["input"])
        first_roles = {first_info["background"]: -1}
        first_roles.update({color: index for index, color in enumerate(first_info["seeds"])})
        for pair_index, pair in enumerate(puzzle["train"] + puzzle["test"]):
            split = "train" if pair_index < len(puzzle["train"]) else "test"
            inp, out = pair["input"], pair["output"]
            info = _analyze_input(inp)
            matrix = inp if info["orientation"] == "vertical" else inp.T
            assert np.array_equal(out, _apply_rule_from_input(inp))
            assert not _source_copy(inp, out)
            orientations.add(info["orientation"])
            band_counts.add(len(info["bands"]))
            layouts.add(info["widths"])
            gaps.update(b - a for a, b in info["gaps"])
            shapes.add(inp.shape)
            palettes.add(tuple(sorted(map(int, np.unique(inp)))))
            structures.add(_structural_key(inp))
            densities.extend(info["densities"])
            split_sparse[split] += sum(d < 0.35 for d in info["densities"])
            split_dense[split] += sum(d > 0.70 for d in info["densities"])
            long_gap = any(b - a > 1 for a, b in info["gaps"])
            split_long_gaps[split] += long_gap
            if len(info["bands"]) == 3:
                style = len(set(info["widths"]))
                split_width_styles[split].add(style)
                split_combined[split] += (style > 1 and long_gap
                                          and min(info["densities"]) < 0.55
                                          and max(info["densities"]) > 0.65)
            if split == "test":
                test_new_color_counts.add(len(set(map(int, np.unique(inp))) - first_colors))
                test_role_exchange += any(first_roles.get(color) != position
                                          for position, color in enumerate(info["seeds"])
                                          if color in first_roles)
            small += min(inp.shape) <= 7
            large += max(inp.shape) >= 27
            overlap += bool(set(map(int, np.unique(inp))) & first_colors)
            disjoint += not bool(set(map(int, np.unique(inp))) & first_colors)
            for (a, b), seed in zip(info["bands"], info["seeds"]):
                solid += int(np.any(np.all(matrix[:, a:b] == seed, axis=0)))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert orientations == {"vertical", "horizontal"}
    assert band_counts == {3, 4, 5}
    assert len(shapes) >= 35 and len(palettes) >= 35 and len(structures) >= 180
    assert len(layouts) >= 45 and gaps == {1, 2, 3}
    assert min(densities) < 0.30 and max(densities) > 0.80
    assert small >= 8 and large >= 8 and solid >= 10
    assert overlap > 0 and disjoint > 0
    assert split_width_styles == {"train": {1, 2, 3}, "test": {1, 2, 3}}
    assert all(split_sparse[split] > 0 and split_dense[split] > 0
               and split_long_gaps[split] > 0 and split_combined[split] > 0
               for split in ("train", "test"))
    assert 0 in test_new_color_counts and max(test_new_color_counts) >= 3
    assert test_role_exchange > 0
    # The original first training grid has a solid right-edge seed column.
    assert _analyze_input(_SOURCE_PAIRS[0][0])["widths"] == (2, 2, 2)
