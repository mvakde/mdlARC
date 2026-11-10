"""Generate independent panels with a shared positional four-colour key."""

import hashlib

import numpy as np


_AUTO_COUNTS = (
    (2, 1), (3, 1), (4, 1), (4, 2), (5, 2), (6, 1),
    (6, 3), (7, 2), (7, 3), (8, 4), (9, 3), (10, 4),
)
_SOURCE_INPUT_DIGESTS = frozenset((
    "e21fcd471625ab023fe221f8211c383f3fcce9080342ad383c5aa3e994f13916",
    "fff1ae4e3c3028cba4dcf92467dc95a21f8efcc44f42b9679576355c3e6c0ca7",
    "8a6e6a7dbd6c5b92295fcc920543183091e639d4650ce49c728ffb6033ed2acc",
))


def _checked_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = None if num_train is None else _checked_count(num_train, "num_train", 2, 10)
    test = None if num_test is None else _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _sample_shape(rng, regime):
    if rng.random() < 0.15:
        side = int(rng.choice((22, 28)))
        return side, side
    low = 14 if regime == "source_train" else 16
    return int(rng.integers(low, 31)), int(rng.integers(low, 31))


def _capacity(height, width):
    return ((height - 1) // 6) * ((width - 1) // 6)


def _partition_regions(rng, height, width, count):
    """Split independently within branches, permitting staggered stacks."""
    for _ in range(40):
        regions = [(1, 1, height - 2, width - 2)]
        while len(regions) < count:
            options = []
            for index, (_, _, rh, rw) in enumerate(regions):
                if rh >= 11:
                    options.append((index, 0, rh * rw))
                if rw >= 11:
                    options.append((index, 1, rh * rw))
            if not options:
                break
            weights = np.array([option[2] for option in options], dtype=float)
            index, axis, _ = options[int(rng.choice(len(options), p=weights / weights.sum()))]
            r, c, rh, rw = regions.pop(index)
            length = rh if axis == 0 else rw
            first = int(rng.integers(5, length - 5))
            if axis == 0:
                regions.extend(((r, c, first, rw), (r + first + 1, c, rh - first - 1, rw)))
            else:
                regions.extend(((r, c, rh, first), (r, c + first + 1, rh, rw - first - 1)))
        if len(regions) == count:
            return regions
    return None


def _grid_regions(rng, height, width, count):
    factors = [(a, count // a) for a in range(1, count + 1)
               if count % a == 0 and height - 2 >= 6 * a - 1
               and width - 2 >= 6 * (count // a) - 1]
    if not factors:
        return None
    rows, cols = factors[int(rng.integers(len(factors)))]

    def slots(length, parts):
        sizes = np.full(parts, 5, dtype=int)
        extra = length - 2 - (parts - 1) - 5 * parts
        sizes += rng.multinomial(extra, np.full(parts, 1 / parts))
        starts = []
        pos = 1
        for size in sizes:
            starts.append((pos, int(size)))
            pos += int(size) + 1
        return starts

    return [(r, c, rh, rw) for r, rh in slots(height, rows)
            for c, rw in slots(width, cols)]


def _scatter_regions(rng, height, width, count):
    if count > 5:
        return None
    regions = []
    for _ in range(count):
        placed = False
        for _ in range(100):
            rh = int(rng.integers(5, min(15, height - 2) + 1))
            rw = int(rng.integers(5, min(15, width - 2) + 1))
            r = int(rng.integers(1, height - rh))
            c = int(rng.integers(1, width - rw))
            if all(r + rh < er or er + eh < r or c + rw < ec or ec + ew < c
                   for er, ec, eh, ew in regions):
                regions.append((r, c, rh, rw))
                placed = True
                break
        if not placed:
            return None
    return regions


def _panel_size(rng, available, minimum):
    mode = int(rng.integers(5))
    if mode <= 1:
        return max(minimum, available - int(rng.integers(min(3, available - minimum + 1))))
    if mode == 2:
        return int(rng.integers(minimum, min(minimum + 3, available) + 1))
    return int(rng.integers(minimum, available + 1))


def _make_panel_specs(rng, height, width, count, balanced):
    regions = None
    minimum = 7 if balanced else 5
    for mode in rng.permutation(3):
        if mode == 0:
            regions = _partition_regions(rng, height, width, count)
        elif mode == 1:
            regions = _scatter_regions(rng, height, width, count)
        else:
            regions = _grid_regions(rng, height, width, count)
        if regions is not None and all(
            rh >= minimum and rw >= minimum for _, _, rh, rw in regions
        ):
            break
        regions = None
    if regions is None:
        raise ValueError("panel packing failed")
    specs = []
    for row, col, available_h, available_w in regions:
        ph = _panel_size(rng, available_h, minimum)
        pw = _panel_size(rng, available_w, minimum)
        r0 = row + int(rng.integers(available_h - ph + 1))
        c0 = col + int(rng.integers(available_w - pw + 1))
        def split(size):
            interior = size - 3
            low = 2 if interior >= 4 and (balanced or rng.random() < 0.55) else 1
            return int(rng.integers(low, interior - low + 1))

        top = split(ph)
        left = split(pw)
        specs.append((r0, c0, top, ph - 3 - top, left, pw - 3 - left))
    rng.shuffle(specs)
    return specs


def _make_clue_masks(rng, count):
    masks = np.zeros((count, 4), dtype=bool)
    mode = int(rng.integers(7))
    if mode <= 2:
        donor_count = (1 if rng.random() < 0.15 else
                       int(rng.integers(2, min(3, count - 1) + 1))) if count > 2 else 2
        donors = rng.choice(count, size=donor_count, replace=False)
        for role in range(4):
            masks[int(rng.choice(donors)), role] = True
    else:
        for role in range(4):
            masks[int(rng.integers(count)), role] = True
        if mode == 6:
            masks[int(rng.integers(count)), :] = True
        probability = {3: 0.18, 4: 0.32, 5: 0.45, 6: 0.34}[mode]
        masks |= rng.random((count, 4)) < probability
    if np.all(masks):
        masks[int(rng.integers(count)), int(rng.integers(4))] = False
    assert np.all(masks.any(axis=0))
    return masks


def _compartments(spec):
    r, c, top, bottom, left, right = spec
    middle_r, middle_c = r + top + 1, c + left + 1
    end_r, end_c = middle_r + bottom + 1, middle_c + right + 1
    return (
        (slice(r + 1, middle_r), slice(c + 1, middle_c)),
        (slice(r + 1, middle_r), slice(middle_c + 1, end_c)),
        (slice(middle_r + 1, end_r), slice(c + 1, middle_c)),
        (slice(middle_r + 1, end_r), slice(middle_c + 1, end_c)),
    )


def _generate_pair(rng, regime, background):
    balanced = bool(rng.random() < 0.3)
    for _ in range(100):
        height, width = _sample_shape(rng, regime)
        if balanced and (height < 20 or width < 20):
            continue
        maximum = min(4 if balanced else (7 if regime == "target_test" else 6),
                      _capacity(height, width))
        if maximum < 2:
            continue
        count = int(rng.integers(2, maximum + 1))
        try:
            specs = _make_panel_specs(rng, height, width, count, balanced)
        except ValueError:
            continue
        break
    else:
        raise ValueError("could not construct panel layout")

    available = [color for color in range(10) if color != background]
    palette = rng.choice(available, size=5, replace=False)
    structure = int(palette[0])
    payload = tuple(int(color) for color in palette[1:])
    masks = _make_clue_masks(rng, count)
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for index, spec in enumerate(specs):
        r, c, top, bottom, left, right = spec
        rr, cc = r + top + bottom + 3, c + left + right + 3
        inp[r:rr, c:cc] = structure
        out[r:rr, c:cc] = structure
        for role, region in enumerate(_compartments(spec)):
            out[region] = payload[role]
            inp[region] = payload[role] if masks[index, role] else background
    return {"input": inp, "output": out}


def _grid_key(grid):
    return tuple(grid.shape), grid.tobytes()


def _source_digest(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.int8).tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    backgrounds = list(rng.permutation(10))
    if train_count + test_count > 10:
        backgrounds += list(rng.permutation(10))
    puzzle = {"train": [], "test": []}
    seen = set()
    index = 0
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for _ in range(count):
            background = int(backgrounds[index])
            index += 1
            for _attempt in range(100):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _generate_pair(child_rng, regime, background)
                key = _grid_key(pair["input"])
                color_counts = np.bincount(pair["input"].ravel(), minlength=10)
                if (key not in seen
                        and color_counts[background] > max(
                            color_counts[color] for color in range(10) if color != background
                        )
                        and _source_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS):
                    break
            else:
                raise ValueError("could not make a unique pair")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    components = []
    while remaining:
        first = remaining.pop()
        stack = [first]
        cells = [first]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cells.append(neighbor)
                    stack.append(neighbor)
        components.append(cells)
    return components


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert all(1 <= side <= 30 for side in grid.shape)
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape
    background = int(inp[0, 0])
    for grid in (inp, out):
        assert np.all(grid[0] == background) and np.all(grid[-1] == background)
        assert np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)
    points = np.argwhere(inp != background)
    assert len(points)
    structure = int(inp[tuple(points[0])])
    assert structure != background
    components = _components(inp == structure)
    assert 2 <= len(components) <= 7
    covered = np.zeros(inp.shape, dtype=bool)
    expected = inp.copy()
    template = [None] * 4
    clues = np.zeros((len(components), 4), dtype=bool)
    dimensions = []
    boxes = []
    for index, cells in enumerate(components):
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        boxes.append((r0, c0, r1, c1))
        assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5
        assert not covered[r0:r1 + 1, c0:c1 + 1].any()
        covered[r0:r1 + 1, c0:c1 + 1] = True
        a = inp[r0:r1 + 1, c0:c1 + 1]
        b = out[r0:r1 + 1, c0:c1 + 1]
        assert np.all(a[0] == structure) and np.all(a[-1] == structure)
        assert np.all(a[:, 0] == structure) and np.all(a[:, -1] == structure)
        drs = [r for r in range(r0 + 1, r1) if np.all(inp[r, c0:c1 + 1] == structure)]
        dcs = [c for c in range(c0 + 1, c1) if np.all(inp[r0:r1 + 1, c] == structure)]
        assert len(drs) == len(dcs) == 1
        dr, dc = drs[0], dcs[0]
        mask = np.zeros(a.shape, dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        mask[dr - r0, :] = True
        mask[:, dc - c0] = True
        assert np.array_equal(a == structure, mask)
        assert np.array_equal(b == structure, mask)
        dimensions.append((dr - r0 - 1, r1 - dr - 1, dc - c0 - 1, c1 - dc - 1))
        spec = (r0, c0, *dimensions[-1])
        for role, region in enumerate(_compartments(spec)):
            iv, ov = np.unique(inp[region]), np.unique(out[region])
            assert len(iv) == len(ov) == 1
            iv, ov = int(iv[0]), int(ov[0])
            assert ov not in (background, structure) and iv in (background, ov)
            if template[role] is None:
                template[role] = ov
            assert template[role] == ov
            if iv != background:
                clues[index, role] = True
            expected[region] = ov
    assert np.all(inp[~covered] == background) and np.all(out[~covered] == background)
    assert all(value is not None for value in template) and len(set(template)) == 4
    assert np.all(clues.any(axis=0)) and not np.all(clues)
    assert np.array_equal(expected, out) and not np.array_equal(inp, out)
    assert _source_digest(inp) not in _SOURCE_INPUT_DIGESTS
    assert set(np.unique(inp)) == {background, structure, *template}
    return {"background": background, "structure": structure,
            "template": tuple(template), "panel_count": len(components),
            "clues": clues, "dimensions": dimensions, "boxes": boxes}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    reports = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            report = _inspect_pair(pair)
            reports.append(report)
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    assert len({report["background"] for report in reports}) >= 2


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(137), generate_puzzle(137))
    assert _puzzles_equal(generate_puzzle(991, 6, 3), generate_puzzle(991, 6, 3))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(2000 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(3000 + (train or 0) + (test or 0), train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(4000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {args}")
    train_counts, test_counts = set(), set()
    shapes, panels, layouts, palettes = set(), set(), set(), set()
    blank = full = thin = tall = flat = sparse = 0
    source_sized = {22: 0, 28: 0}
    no_both_gutters = sparse_blank_staggered = non_extreme = 0
    backgrounds_vary = panel_counts_vary = shapes_vary = 0
    four_to_six = test_fewer = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        reports = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                report = _inspect_pair(pair)
                reports.append(report)
                shapes.add(tuple(pair["input"].shape))
                for side in source_sized:
                    source_sized[side] += pair["input"].shape == (side, side)
                panels.add(report["panel_count"])
                layouts.add((pair["input"].shape,
                             (pair["input"] == report["structure"]).tobytes()))
                palettes.add((report["background"], report["structure"],
                              report["template"]))
                counts = report["clues"].sum(axis=1)
                blank += int(np.any(counts == 0))
                full += int(np.any(counts == 4))
                sparse += int(report["clues"].sum() == 4)
                thin += int(any(min(d) == 1 for d in report["dimensions"]))
                tall += int(any((d[0] + d[1] + 3) >= 2 * (d[2] + d[3] + 3)
                                for d in report["dimensions"]))
                flat += int(any((d[2] + d[3] + 3) >= 2 * (d[0] + d[1] + 3)
                                for d in report["dimensions"]))
                boxes = report["boxes"]
                top, bottom = min(b[0] for b in boxes), max(b[2] for b in boxes)
                left, right = min(b[1] for b in boxes), max(b[3] for b in boxes)
                background = report["background"]
                whole_row = any(np.all(pair["input"][r] == background)
                                for r in range(top + 1, bottom))
                whole_col = any(np.all(pair["input"][:, c] == background)
                                for c in range(left + 1, right))
                staggered = not (whole_row and whole_col)
                no_both_gutters += staggered
                sparse_blank_staggered += (staggered and counts.sum() == 4
                                           and np.any(counts == 0)
                                           and any(min(d) == 1 for d in report["dimensions"]))
                non_extreme += not any(
                    (min(d[:2]) == 1 and max(d[:2]) >= 3)
                    or (min(d[2:]) == 1 and max(d[2:]) >= 3)
                    for d in report["dimensions"])
        backgrounds_vary += len({r["background"] for r in reports}) >= 2
        panel_counts_vary += len({r["panel_count"] for r in reports}) >= 2
        shapes_vary += len({pair["input"].shape for split in ("train", "test")
                            for pair in puzzle[split]}) >= 2
        four_to_six += (any(_inspect_pair(pair)["panel_count"] == 4
                            for pair in puzzle["train"])
                        and any(_inspect_pair(pair)["panel_count"] == 6
                                for pair in puzzle["test"]))
        test_fewer += any(_inspect_pair(a)["panel_count"] < _inspect_pair(b)["panel_count"]
                          for a in puzzle["test"] for b in puzzle["train"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 45 and len(panels) >= 5
    assert len(layouts) >= 150 and len(palettes) >= 150
    assert blank >= 20 and full >= 15 and sparse >= 20
    assert thin >= 50 and tall >= 5 and flat >= 5
    assert all(count >= 1 for count in source_sized.values())
    assert no_both_gutters >= 40 and sparse_blank_staggered >= 5
    assert non_extreme >= 20 and four_to_six >= 1 and test_fewer >= 1
    assert backgrounds_vary == panel_counts_vary == shapes_vary == 42
