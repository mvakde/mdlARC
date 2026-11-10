import hashlib

import numpy as np


_OFFSETS = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
_RELATIONS = ("contained", "center_inside", "center_outside", "outside")
_SOURCE_PAIR_HASHES = {
    "78dfada9fa16dd7036fe2cf7ee7782a1d971cc9fd0c05d36328fe2128234a8de",
    "31a9d8036d68b5dcf11386040ab4cb2c459fb62a3075f07d5238f20eb287067c",
    "cf8382e819ff1f59bb794bf781f2101e38954d44c69181aaaf4dbfe71f6c94e3",
    "a6c86bdb7af4c49529e28c850df85a63a63c5577211929005eb230d4067aa69e",
}


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _integer(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _plus(center):
    r, c = center
    return tuple((r + dr, c + dc) for dr, dc in _OFFSETS)


def _bbox(squares):
    return (min(r for r, c, s, color in squares),
            max(r + s - 1 for r, c, s, color in squares),
            min(c for r, c, s, color in squares),
            max(c + s - 1 for r, c, s, color in squares))


def _inside(cell, box):
    r, c = cell
    return box[0] <= r <= box[1] and box[2] <= c <= box[3]


def _relation(cells, box):
    flags = [_inside(cell, box) for cell in cells]
    if all(flags):
        return "contained"
    if not any(flags):
        return "outside"
    return "center_inside" if flags[2] else "center_outside"


def _dimensions(rng, mode, regime):
    if isinstance(mode, tuple):
        return mode
    if rng.random() < 0.13:
        return 16, 16
    if mode == "matched":
        base = int(rng.integers(16, 27))
        height = base + int(rng.integers(-3, 4))
        width = base + int(rng.integers(-3, 4))
    elif mode == "expanding":
        lo, hi = (15, 24) if regime == "source_train" else (21, 31)
        height = int(rng.integers(lo, hi))
        width = int(rng.integers(lo, hi))
    else:
        height = int(rng.integers(14, 31))
        width = int(rng.integers(14, 31))
    if rng.random() < 0.25:
        height = int(rng.integers(12, 20))
        width = int(rng.integers(22, 31))
        if rng.random() < 0.5:
            height, width = width, height
    return max(12, min(30, height)), max(12, min(30, width))


def _sizes(rng, count, height, width, pattern, regime):
    # The area budget affects packing only; parity and multiplicity remain
    # sampled separately from dimensions and split.
    weights = np.array([0.15, 0.23, 0.24, 0.17, 0.10, 0.06, 0.03, 0.015, 0.005])
    if regime == "target_test":
        weights = np.array([0.14, 0.20, 0.23, 0.17, 0.11, 0.07, 0.045, 0.025, 0.01])
    weights /= weights.sum()
    options = np.arange(1, 10)
    if pattern == "even":
        options = options[options % 2 == 0]
        weights = weights[[1, 3, 5, 7]]
    elif pattern == "odd":
        options = options[options % 2 == 1]
        weights = weights[[0, 2, 4, 6, 8]]
    weights /= weights.sum()
    sizes = [int(x) for x in rng.choice(options, count, p=weights)]
    if pattern == "mixed":
        if not any(s % 2 for s in sizes):
            sizes[0] = 1 if rng.random() < 0.5 else 3
        if not any(s % 2 == 0 for s in sizes):
            sizes[-1] = 2 if rng.random() < 0.7 else 4
    # Dense draws are reduced one square at a time, preserving the required
    # parity in mixed demonstrations.
    budget = int(0.62 * height * width)
    for _ in range(80):
        if sum((s + 1) ** 2 for s in sizes) <= budget and max(sizes) <= min(height, width) - 2:
            break
        candidates = [i for i, s in enumerate(sizes) if s > (2 if pattern == "even" else 1)]
        if not candidates:
            break
        index = max(candidates, key=lambda i: sizes[i])
        sizes[index] -= 2 if pattern != "mixed" else 1
    if pattern == "mixed":
        if not any(s % 2 for s in sizes):
            sizes[0] = 1
        if not any(s % 2 == 0 for s in sizes):
            sizes[-1] = 2
    return sizes


def _place_squares(rng, height, width, sizes, region, anchor=None):
    r_lo, r_hi, c_lo, c_hi = region
    mask = np.zeros((height, width), dtype=bool)
    squares = []
    order = list(rng.permutation(len(sizes)))
    order.sort(key=lambda i: sizes[i], reverse=True)
    for rank, index in enumerate(order):
        side = sizes[index]
        places = []
        for row in range(r_lo, r_hi - side + 2):
            for col in range(c_lo, c_hi - side + 2):
                if rank == 0 and anchor is not None:
                    if anchor == "top" and row != 0:
                        continue
                    if anchor == "bottom" and row + side != height:
                        continue
                    if anchor == "left" and col != 0:
                        continue
                    if anchor == "right" and col + side != width:
                        continue
                if np.any(mask[max(0, row-1):min(height, row+side+1),
                               max(0, col-1):min(width, col+side+1)]):
                    continue
                places.append((row, col))
        if not places:
            return None
        row, col = places[int(rng.integers(len(places)))]
        mask[row:row+side, col:col+side] = True
        squares.append((row, col, side))
    return mask, squares


def _plus_centers(mask, box, relation, edge=False):
    height, width = mask.shape
    candidates = []
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            cells = _plus((row, col))
            if _relation(cells, box) != relation:
                continue
            if edge and not any(r in (0, height-1) or c in (0, width-1) for r, c in cells):
                continue
            if any(np.any(mask[max(0, r-1):min(height, r+2),
                               max(0, c-1):min(width, c+2)]) for r, c in cells):
                continue
            candidates.append((row, col))
    return candidates


def _region(rng, height, width, relation, edge_plus):
    if edge_plus:
        return (0, height-1, 0, width-1)
    if relation == "outside":
        side = int(rng.integers(4))
        gap = int(rng.integers(3, 6))
    elif relation in ("center_inside", "center_outside"):
        side = int(rng.integers(4))
        gap = int(rng.integers(2, 5))
    else:
        return (0, height-1, 0, width-1)
    if side == 0:
        return (gap, height-1, 0, width-1)
    if side == 1:
        return (0, height-gap-1, 0, width-1)
    if side == 2:
        return (0, height-1, gap, width-1)
    return (0, height-1, 0, width-gap-1)


def _colors(rng, theme, pair_index, multi):
    reuse_palette = pair_index == 0 or (pair_index == 2 and theme["stable_palette"])
    if reuse_palette or (pair_index > 1 and rng.random() < 0.48):
        background = theme["background"]
    else:
        background = int(rng.choice([c for c in range(10) if c != theme["background"]]))
    if background == theme["background"] and (reuse_palette or rng.random() < 0.46):
        marker = theme["marker"]
        base = theme["payload"]
    else:
        marker = int(rng.choice([c for c in range(10) if c != background]))
        base = int(rng.choice([c for c in range(10) if c not in (background, marker)]))
    payloads = [base]
    if multi:
        other = [c for c in range(10) if c not in (background, marker, base)]
        rng.shuffle(other)
        payloads += other[:int(rng.integers(1, 3))]
    return background, marker, payloads


def _construct_pair(rng, regime, theme, pair_index, relation, pattern, multi,
                    edge_plus=False):
    for attempt in range(280):
        height, width = _dimensions(rng, theme["dimension_mode"], regime)
        if pattern == "single":
            count = 1
            relation = "outside"
        elif regime == "source_train":
            count = int(rng.integers(2, 11))
        else:
            count = int(rng.integers(2, 14))
        if pattern == "mixed":
            count = max(2, count)
        side_pattern = "odd" if pattern == "single" else pattern
        sizes = _sizes(rng, count, height, width, side_pattern, regime)
        if sum(s*s for s in sizes) + 5 > 0.48 * height * width:
            continue
        region = _region(rng, height, width, relation, edge_plus)
        anchor = None
        if edge_plus:
            anchor = str(rng.choice(("top", "bottom", "left", "right")))
        placement = _place_squares(rng, height, width, sizes, region, anchor)
        if placement is None:
            continue
        mask, bare_squares = placement
        box = _bbox([(r, c, s, 0) for r, c, s in bare_squares])
        centers = _plus_centers(mask, box, relation, edge_plus)
        if not centers:
            continue
        center = centers[int(rng.integers(len(centers)))]
        background, marker, payloads = _colors(rng, theme, pair_index, multi)
        squares = []
        inp = np.full((height, width), background, dtype=np.int8)
        out = inp.copy()
        for index, (r, c, s) in enumerate(bare_squares):
            color = payloads[index % len(payloads)]
            squares.append((r, c, s, color))
            inp[r:r+s, c:c+s] = color
            out[r:r+s, c:c+s] = color
            if s % 2:
                out[r+s//2, c+s//2] = marker
        cells = _plus(center)
        for r, c in cells:
            inp[r, c] = marker
            out[r, c] = marker if relation == "contained" else background
        return {"input": inp, "output": out}
    raise RuntimeError(f"could not construct {regime} {relation} pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0, 2**63-1)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    palette = list(rng.permutation(10))
    dimension_mode = str(rng.choice(("matched", "mixed", "expanding")))
    if rng.random() < 0.22:
        fixed_shapes = ((16, 16), (18, 18), (20, 20), (24, 24),
                        (18, 26), (26, 18))
        dimension_mode = fixed_shapes[int(rng.integers(len(fixed_shapes)))]
    theme = {
        "background": int(palette[0]),
        "marker": int(palette[1]),
        "payload": int(palette[2]),
        "dimension_mode": dimension_mode,
        "multi": bool(rng.random() < 0.55),
        "stable_palette": bool(rng.random() < 0.32),
    }
    plans = [("contained", "mixed"), ("center_inside", "mixed")]
    if rng.random() < 0.13:
        plans[0] = ("contained", "even")
    for _ in range(2, train_count):
        relation = str(rng.choice(_RELATIONS, p=(0.26, 0.24, 0.18, 0.32)))
        pattern = str(rng.choice(("mixed", "odd", "even", "single"),
                                 p=(0.47, 0.21, 0.25, 0.07)))
        if pattern == "single":
            relation = "outside"
        plans.append((relation, pattern))
    rng.shuffle(plans)
    test_plans = []
    for _ in range(test_count):
        relation = str(rng.choice(_RELATIONS, p=(0.34, 0.23, 0.16, 0.27)))
        pattern = str(rng.choice(("mixed", "odd", "even", "single"),
                                 p=(0.54, 0.19, 0.22, 0.05)))
        if pattern == "single":
            relation = "outside"
        test_plans.append((relation, pattern))
    pair_seeds = rng.integers(0, 2**63-1, size=train_count+test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    multi_demo_index = next(i for i, (_, pattern) in enumerate(plans) if pattern != "single")
    for section, pair_plans in (("train", plans), ("test", test_plans)):
        for local_index, (relation, pattern) in enumerate(pair_plans):
            index = local_index if section == "train" else train_count + local_index
            pair_rng = np.random.default_rng(int(pair_seeds[index]))
            multi = theme["multi"] and (section == "train" and local_index == multi_demo_index
                                        or pair_rng.random() < 0.55)
            edge_plus = relation == "contained" and pattern == "mixed" and pair_rng.random() < 0.10
            pair = _construct_pair(pair_rng, "source_train" if section == "train" else "target_test",
                                   theme, index, relation, pattern, multi, edge_plus)
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    groups = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        group = [start]
        while stack:
            r, c = stack.pop()
            for cell in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if cell in remaining:
                    remaining.remove(cell)
                    stack.append(cell)
                    group.append(cell)
        groups.append(group)
    return groups


def _analyze(grid):
    height, width = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    assert counts.max() > grid.size / 2, "background must be visually dominant"
    marker_candidates = []
    for color in colors:
        if int(color) == background:
            continue
        cells = set(zip(*np.nonzero(grid == color)))
        if len(cells) == 5:
            center = tuple(map(int, np.mean(np.array(list(cells)), axis=0)))
            if cells == set(_plus(center)):
                marker_candidates.append((int(color), center))
    assert len(marker_candidates) == 1, "one unique plus required"
    marker, center = marker_candidates[0]
    plus_cells = _plus(center)
    payload_mask = (grid != background) & (grid != marker)
    squares = []
    for group in _components(payload_mask):
        rows = [int(r) for r, c in group]
        cols = [int(c) for r, c in group]
        r, c = min(rows), min(cols)
        side = max(rows) - r + 1
        assert max(cols) - c + 1 == side and len(group) == side*side
        color = int(grid[r, c])
        assert np.all(grid[r:r+side, c:c+side] == color)
        squares.append((r, c, side, color))
    assert squares, "at least one square required"
    for i, (r, c, s, color) in enumerate(squares):
        for r2, c2, s2, color2 in squares[i+1:]:
            assert (r+s < r2 or r2+s2 < r or c+s < c2 or c2+s2 < c), "squares touch"
        for pr, pc in plus_cells:
            assert not (r-1 <= pr <= r+s and c-1 <= pc <= c+s), "plus touches square"
    box = _bbox(squares)
    return {"background": background, "marker": marker, "squares": squares,
            "plus": plus_cells, "bbox": box, "relation": _relation(plus_cells, box),
            "height": height, "width": width}


def _expected(inp, analysis):
    out = inp.copy()
    marker = analysis["marker"]
    for r, c, s, color in analysis["squares"]:
        if s % 2:
            out[r+s//2, c+s//2] = marker
    if analysis["relation"] != "contained":
        for r, c in analysis["plus"]:
            out[r, c] = analysis["background"]
    return out


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        inputs, pairs = set(), set()
        training = []
        all_records = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                inp, out = pair["input"], pair["output"]
                for grid in (inp, out):
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert all(1 <= n <= 30 for n in grid.shape)
                    assert np.all((grid >= 0) & (grid <= 9))
                assert inp.shape == out.shape
                ik, ok = _key(inp), _key(out)
                assert ik not in inputs and (ik, ok) not in pairs
                source_hash = hashlib.sha256(
                    bytes(inp.shape) + np.asarray(inp, dtype=np.int8).tobytes()
                    + np.asarray(out, dtype=np.int8).tobytes()
                ).hexdigest()
                assert source_hash not in _SOURCE_PAIR_HASHES, "source pair copied"
                inputs.add(ik)
                pairs.add((ik, ok))
                analysis = _analyze(inp)
                assert np.array_equal(out, _expected(inp, analysis))
                if section == "train":
                    training.append(analysis)
                all_records.append(analysis)
        relations = {a["relation"] for a in training}
        assert "contained" in relations and "center_inside" in relations
        assert any(s % 2 for a in training for r, c, s, color in a["squares"])
        assert any(s % 2 == 0 for a in training for r, c, s, color in a["squares"])
        assert len({a["background"] for a in all_records}) >= 2
        if any(len({color for r, c, s, color in a["squares"]}) > 1 for a in all_records):
            assert any(len({color for r, c, s, color in a["squares"]}) > 1 for a in training)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed puzzle") from exc


def self_test() -> None:
    train_counts, test_counts, shapes, layouts = set(), set(), set(), set()
    relations = {"train": set(), "test": set()}
    backgrounds, parity_profiles, edge_squares = set(), set(), 0
    coverage = {name: 0 for name in (
        "multicolor", "single", "even_unchanged", "retained_edge_plus",
        "corner_straddle", "side_above_seven", "sixteen", "compact",
        "same_dimensions_puzzle", "dimension_overlap_puzzle", "repeat_palette_puzzle",
    )}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_shapes = {section: set() for section in ("train", "test")}
        puzzle_palettes = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                a = _analyze(pair["input"])
                assert np.array_equal(pair["output"], _expected(pair["input"], a))
                expected_changes = {(r+s//2, c+s//2) for r, c, s, color in a["squares"]
                                    if s % 2}
                if a["relation"] != "contained":
                    expected_changes.update(a["plus"])
                actual_changes = set(zip(*np.nonzero(pair["input"] != pair["output"])))
                assert actual_changes == expected_changes
                shapes.add(pair["input"].shape)
                puzzle_shapes[section].add(pair["input"].shape)
                layouts.add((pair["input"].shape, tuple(sorted(
                    (r, c, s) for r, c, s, color in a["squares"])), a["plus"]))
                backgrounds.add(a["background"])
                relations[section].add(a["relation"])
                ss = [s for r, c, s, color in a["squares"]]
                colors = {color for r, c, s, color in a["squares"]}
                puzzle_palettes.append((a["background"], tuple(sorted(colors)), a["marker"]))
                coverage["multicolor"] += len(colors) > 1
                coverage["single"] += len(ss) == 1
                coverage["even_unchanged"] += (a["relation"] == "contained"
                                               and all(s % 2 == 0 for s in ss)
                                               and np.array_equal(pair["input"], pair["output"]))
                coverage["retained_edge_plus"] += (a["relation"] == "contained" and any(
                    r in (0, a["height"]-1) or c in (0, a["width"]-1) for r, c in a["plus"]))
                box = a["bbox"]
                coverage["corner_straddle"] += (a["relation"] in
                    ("center_inside", "center_outside") and
                    any(r < box[0] or r > box[1] for r, c in a["plus"]) and
                    any(c < box[2] or c > box[3] for r, c in a["plus"]))
                coverage["side_above_seven"] += max(ss) > 7
                coverage["sixteen"] += pair["input"].shape == (16, 16)
                coverage["compact"] += sum(s*s for s in ss) / pair["input"].size >= 0.30
                parity_profiles.add((sum(s % 2 for r, c, s, color in a["squares"]),
                                     len(a["squares"])))
                edge_squares += any(r == 0 or c == 0 or r+s == a["height"]
                                    or c+s == a["width"] for r, c, s, color in a["squares"])
        coverage["same_dimensions_puzzle"] += len(puzzle_shapes["train"] |
                                                   puzzle_shapes["test"]) == 1
        coverage["dimension_overlap_puzzle"] += bool(puzzle_shapes["train"] &
                                                      puzzle_shapes["test"])
        coverage["repeat_palette_puzzle"] += len(puzzle_palettes) != len(set(puzzle_palettes))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(layouts) >= 100
    assert len(backgrounds) >= 7 and len(parity_profiles) >= 12
    assert edge_squares >= 10
    assert relations["test"] == set(_RELATIONS)
    assert coverage["multicolor"] >= 10 and coverage["single"] >= 5
    assert coverage["even_unchanged"] >= 5 and coverage["retained_edge_plus"] >= 5
    assert coverage["corner_straddle"] >= 3 and coverage["side_above_seven"] >= 5
    assert coverage["sixteen"] >= 10 and coverage["compact"] >= 10
    assert coverage["same_dimensions_puzzle"] >= 2
    assert coverage["dimension_overlap_puzzle"] >= 3
    assert coverage["repeat_palette_puzzle"] >= 3
    for seed, nt, ne in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        puzzle = generate_puzzle(seed, nt, ne)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ne)
    for seed, nt, ne in ((2001, None, 1), (2002, None, 4),
                         (2003, 2, None), (2004, 10, None)):
        generate_puzzle(seed, nt, ne)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(999, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    a = generate_puzzle(3001, 7, 3)
    b = generate_puzzle(3001, 7, 3)
    assert all(np.array_equal(x[k], y[k]) for section in ("train", "test")
               for x, y in zip(a[section], b[section]) for k in ("input", "output"))
    assert _key(a["train"][0]["input"]) != _key(generate_puzzle(3002, 7, 3)["train"][0]["input"])
