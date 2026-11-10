"""Eight-connected exemplar matching in varied color mosaics."""

import hashlib

import numpy as np


DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
PLUS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
T_FIVE = ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1))
H3 = ((0, 0), (0, 1), (0, 2))
H2 = ((0, 0), (0, 1))
H4 = ((0, 0), (0, 1), (0, 2), (0, 3))
H5 = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))
V4 = ((0, 0), (1, 0), (2, 0), (3, 0))
V5 = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
SINGLE = ((0, 0),)
BENT = ((0, 0), (1, 0), (1, 1), (2, 1))
PLUS_EXTENDED = ((0, 0),) + PLUS
L_SHAPES = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)
FAMILIAR = (SINGLE, H2, ((0, 0), (1, 0)), H3, H4, H5, V4, V5,
            PLUS, T_FIVE, DIAMOND, BENT) + L_SHAPES
SOURCE_PAIR_HASHES = frozenset((
    "d88cb5aa445f12255c5c2e4df966f23e7d9d1807bd6e2cd0446b590593f54b0d",
    "c4ebe17f8f023b4a3676b02acfc6f23621b22f64861a30b6ad8573b230174483",
    "29f0c83dada909a74ebf09c54219e90b21999445c7e5d9ed47b0d56808e10cfa",
    "575927e844d555916f5a3995f02f7e8513120b9ed287804fdf5b0024ddc40f84",
))


def _resolve_counts(rng, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _components(grid, color):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or grid[r, c] != color:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < height and 0 <= nc < width and
                                not seen[nr, nc] and grid[nr, nc] == color):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            result.append(cells)
    return result


def _transform(grid, marker, carrier):
    templates = {_normalize(c) for c in _components(grid, marker)}
    assert templates
    output = grid.copy()
    for cells in _components(grid, carrier):
        if _normalize(cells) in templates:
            for r, c in cells:
                output[r, c] = marker
    return output


def _box(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _variants(shape):
    result = set()
    for mirror in (False, True):
        for turns in range(4):
            cells = [(r, -c if mirror else c) for r, c in shape]
            for _ in range(turns):
                cells = [(c, -r) for r, c in cells]
            result.add(_normalize(cells))
    return tuple(sorted(result))


def _random_shape(rng):
    for _ in range(48):
        target = int(rng.integers(3, 9))
        cells = {(0, 0)}
        while len(cells) < target:
            r, c = sorted(cells)[int(rng.integers(len(cells)))]
            dr, dc = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
            if dr or dc:
                cells.add((r + dr, c + dc))
        shape = _normalize(cells)
        if max(_box(shape)) <= 5 and shape not in (DIAMOND, PLUS):
            return shape
    return BENT


def _pick_shape(rng, banned=()):
    banned = set(banned)
    for _ in range(64):
        shape = (FAMILIAR[int(rng.integers(len(FAMILIAR)))]
                 if rng.random() < 0.45 else _random_shape(rng))
        if shape not in banned:
            return shape
    return next(shape for shape in FAMILIAR if shape not in banned)


def _nonmatch(rng, template, selected):
    if template == PLUS and T_FIVE not in selected:
        return T_FIVE
    if template == DIAMOND and SINGLE not in selected:
        return SINGLE
    variants = [s for s in _variants(template) if s not in selected]
    return (variants[int(rng.integers(len(variants)))] if variants
            else _pick_shape(rng, selected))


def _selected_shapes(rng, split, index, training_shapes):
    if split == "train" and index == 0:
        return [DIAMOND] + ([H3] if rng.random() < 0.55 else [])
    if split == "train" and index == 1:
        return [PLUS, BENT] + ([H2] if rng.random() < 0.35 else [])
    if split == "test" and rng.random() < 0.18:
        return list(((L_SHAPES[0], L_SHAPES[3]) if rng.random() < 0.5
                     else (L_SHAPES[1], L_SHAPES[2])))
    count = int(rng.choice((1, 2, 3, 4), p=(0.28, 0.37, 0.26, 0.09)))
    selected = []
    mode = rng.choice(("familiar", "novel", "mixed")) if split == "test" else "mixed"
    for i in range(count):
        known = [s for s in training_shapes if s not in selected]
        if known and (mode == "familiar" or (mode == "mixed" and split == "test" and i == 0)):
            shape = known[int(rng.integers(len(known)))]
        else:
            banned = set(selected) | (set(training_shapes) if mode == "novel" else set())
            shape = _pick_shape(rng, banned)
        selected.append(shape)
    return selected


def _specifications(rng, split, index, selected, carrier, marker, extra):
    specs = []
    for shape in selected:
        marked = int(rng.integers(1, 4)) if split == "test" else int(rng.integers(1, 3))
        matches = int(rng.integers(1, 7)) if split == "test" else int(rng.integers(1, 5))
        if split == "train" and index == 0 and shape == DIAMOND:
            marked, matches = 1, 2
        specs.extend([(marker, shape)] * marked)
        specs.extend([(carrier, shape)] * matches)
    decoys = int(rng.integers(2, 7)) if split == "test" else int(rng.integers(2, 5))
    for i in range(decoys):
        specs.append((carrier, _nonmatch(rng, selected[i % len(selected)], set(selected))))
    if split == "train" and index == 0:
        specs.append((carrier, SINGLE))
    if split == "train" and index == 1:
        specs.extend(((carrier, T_FIVE), (carrier, PLUS_EXTENDED)))
        specs.append((carrier, next(s for s in _variants(BENT) if s != BENT)))
    for _ in range(int(rng.integers(0, 4))):
        specs.append((extra, _pick_shape(rng, selected)))
    return specs


def _scaffold(grid, rng, color, style):
    h, w = grid.shape
    if style == "regular":
        for axis, size in ((0, h), (1, w)):
            count = int(rng.integers(1, 4))
            cuts = rng.choice(np.arange(5, size - 4), size=count, replace=False)
            for cut in cuts:
                if axis == 0:
                    grid[int(cut), :] = color
                else:
                    grid[:, int(cut)] = color
        grid[0, :] = grid[-1, :] = color
        grid[:, 0] = grid[:, -1] = color
    elif style == "partial":
        for _ in range(int(rng.integers(1, 5))):
            if rng.random() < 0.5:
                r = int(rng.integers(1, h - 1))
                length = int(rng.integers(9, w + 1))
                c = int(rng.integers(0, w - length + 1))
                grid[r, c:c + length] = color
            else:
                c = int(rng.integers(1, w - 1))
                length = int(rng.integers(9, h + 1))
                r = int(rng.integers(0, h - length + 1))
                grid[r:r + length, c] = color
        for side in range(4):
            if rng.random() < 0.25:
                length = int(rng.integers(9, (w if side < 2 else h) + 1))
                if side == 0:
                    grid[0, :length] = color
                elif side == 1:
                    grid[-1, :length] = color
                elif side == 2:
                    grid[:length, 0] = color
                else:
                    grid[:length, -1] = color


def _protected_pocket(grid, rng, fill, scaffold, shape):
    bh, bw = _box(shape)
    ph, pw = bh + 2, bw + 2
    h, w = grid.shape
    spots = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)
             if np.all(grid[r:r + ph, c:c + pw] == fill)]
    if not spots:
        return False
    r, c = spots[int(rng.integers(len(spots)))]
    grid[r:r + ph, c:c + pw] = scaffold
    for dr, dc in shape:
        # Reserve the negative example while other objects are packed.
        grid[r + dr + 1, c + dc + 1] = 10
    return True


def _can_place(grid, fill, color, shape, r, c):
    h, w = grid.shape
    cells = {(r + dr, c + dc) for dr, dc in shape}
    for rr, cc in cells:
        if grid[rr, cc] != fill:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if (0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
                        and grid[nr, nc] == color):
                    return False
    return True


def _place_specs(grid, rng, specs, fill):
    ordered = list(specs)
    rng.shuffle(ordered)
    ordered.sort(key=lambda item: -len(item[1]))
    h, w = grid.shape
    for color, shape in ordered:
        bh, bw = _box(shape)
        positions = [(r, c) for r in range(h - bh + 1) for c in range(w - bw + 1)]
        rng.shuffle(positions)
        for r, c in positions:
            if _can_place(grid, fill, color, shape, r, c):
                for dr, dc in shape:
                    grid[r + dr, c + dc] = color
                break
        else:
            return False
    return True


def _build_pair(rng, split, index, params, training_shapes):
    marker, fill = params["marker"], params["fill"]
    carrier, scaffold, extra = params["roles"][(split, index)]
    selected = _selected_shapes(rng, split, index, training_shapes)
    specs = _specifications(rng, split, index, selected, carrier, marker, extra)
    protected = split == "train" and index == 1
    area = sum(len(shape) for _, shape in specs)
    minimum = max(19 if protected else 13, min(26, int(np.sqrt(area * 5)) + 4))
    for attempt in range(8):
        low = min(30, minimum + (2 if attempt else 0))
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        style = ("regular", "partial", "open")[int(rng.choice(3, p=(0.27, 0.43, 0.30)))]
        grid = np.full((h, w), fill, dtype=np.int8)
        _scaffold(grid, rng, scaffold, style)
        if protected and not _protected_pocket(grid, rng, fill, scaffold, PLUS):
            continue
        if _place_specs(grid, rng, specs, fill):
            grid[grid == 10] = fill
            output = _transform(grid, marker, carrier)
            if np.any(grid != output):
                return grid, output
    for side in (28, 29, 30):
        grid = np.full((side, side), fill, dtype=np.int8)
        if protected:
            assert _protected_pocket(grid, rng, fill, scaffold, PLUS)
        if _place_specs(grid, rng, specs, fill):
            grid[grid == 10] = fill
            return grid, _transform(grid, marker, carrier)
    raise AssertionError("could not pack objects")


def _key(grid):
    return grid.shape, grid.tobytes()


def _pair_hash(a, b):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.uint8).tobytes()
                          + b.astype(np.uint8).tobytes()).hexdigest()


def _inspect_pair(pair, marker):
    a, b = pair["input"], pair["output"]
    changed = a != b
    assert np.any(changed) and np.all(b[changed] == marker)
    old_colors = set(int(x) for x in a[changed])
    assert len(old_colors) == 1
    carrier = old_colors.pop()
    templates = {_normalize(c) for c in _components(a, marker)}
    assert templates and carrier != marker
    assert np.array_equal(a[a != carrier], b[a != carrier])
    carrier_components = _components(a, carrier)
    for cells in carrier_components:
        expected_color = marker if _normalize(cells) in templates else carrier
        assert all(b[r, c] == expected_color for r, c in cells)
    candidates = set()
    for raw in np.unique(a):
        color = int(raw)
        if color != marker and any(_normalize(c) in templates for c in _components(a, color)):
            candidates.add(color)
    return {"carrier": carrier, "templates": templates, "candidates": candidates,
            "marker_components": _components(a, marker),
            "carrier_components": carrier_components}


def _infer_fill(train, marker):
    witnesses = set()
    for pair in train:
        a, b = pair["input"], pair["output"]
        templates = {_normalize(c) for c in _components(a, marker)}
        for raw in np.unique(a):
            color = int(raw)
            if color == marker:
                continue
            for cells in _components(a, color):
                if (_normalize(cells) in templates and
                        all(b[r, c] == color for r, c in cells)):
                    witnesses.add(color)
    assert len(witnesses) == 1
    return witnesses.pop()


def _predict_from_training(train, test_input):
    marker_set = {int(p["output"][p["input"] != p["output"]][0]) for p in train}
    assert len(marker_set) == 1
    marker = marker_set.pop()
    fill = _infer_fill(train, marker)
    templates = {_normalize(c) for c in _components(test_input, marker)}
    candidates = [int(raw) for raw in np.unique(test_input)
                  if int(raw) not in (marker, fill) and
                  any(_normalize(c) in templates for c in _components(test_input, int(raw)))]
    assert len(candidates) == 1
    return _transform(test_input, marker, candidates[0])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    keys, markers = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        a, b = pair["input"], pair["output"]
        for grid in (a, b):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.all((0 <= grid) & (grid <= 9))
        assert a.shape == b.shape and _key(a) not in keys
        keys.add(_key(a))
        assert _pair_hash(a, b) not in SOURCE_PAIR_HASHES
        delta = a != b
        assert np.any(delta) and len(np.unique(b[delta])) == 1
        markers.add(int(b[delta][0]))
    assert len(markers) == 1
    marker = markers.pop()
    fill = _infer_fill(puzzle["train"], marker)
    train_info = [_inspect_pair(p, marker) for p in puzzle["train"]]
    test_info = [_inspect_pair(p, marker) for p in puzzle["test"]]
    assert fill != marker
    for info in train_info + test_info:
        assert info["candidates"] <= {fill, info["carrier"]}
        assert info["carrier"] in info["candidates"]
    assert DIAMOND in train_info[0]["templates"]
    assert PLUS in train_info[1]["templates"] and BENT in train_info[1]["templates"]
    assert SINGLE in {_normalize(c) for c in train_info[0]["carrier_components"]}
    assert T_FIVE in {_normalize(c) for c in train_info[1]["carrier_components"]}
    assert PLUS_EXTENDED in {_normalize(c) for c in train_info[1]["carrier_components"]}
    assert len({i["carrier"] for i in train_info[:2]}) == 2
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                              pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    palette = [int(x) for x in rng.permutation(10)]
    params = {"marker": palette[0], "fill": palette[1], "ordinary": palette[2:]}
    roles = {}
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            choices = list(params["ordinary"])
            if split == "train" and index == 1:
                choices.remove(roles[("train", 0)][0])
            carrier = choices[int(rng.integers(len(choices)))]
            others = [color for color in params["ordinary"] if color != carrier]
            picked = rng.choice(others, size=2, replace=False)
            roles[(split, index)] = carrier, int(picked[0]), int(picked[1])
    params["roles"] = roles
    pair_seeds = rng.integers(0, 2**63, size=num_train + num_test, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    training_shapes, seen = [], set()
    for split, count, offset in (("train", num_train, 0), ("test", num_test, num_train)):
        for index in range(count):
            prng = np.random.default_rng(int(pair_seeds[offset + index]))
            for _ in range(10):
                a, b = _build_pair(prng, split, index, params, training_shapes)
                if _key(a) not in seen:
                    break
            else:
                raise AssertionError("duplicate pair construction")
            seen.add(_key(a))
            puzzle[split].append({"input": a, "output": b})
            if split == "train":
                training_shapes.extend(_normalize(c) for c in _components(a, params["marker"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(np.array_equal(a[k], b[k]) for split in ("train", "test")
               for a, b in zip(left[split], right[split]) for k in ("input", "output"))


def self_test() -> None:
    counts, dimensions, palettes, shapes = set(), set(), set(), set()
    template_counts, marker_counts, match_counts = set(), set(), set()
    fill_amounts, normalized = set(), set()
    marker_digits, fill_digits, carrier_digits = set(), set(), set()
    evidence = {name: 0 for name in (
        "diagonal", "equal_box", "rotated", "larger_subpattern", "protected",
        "edge", "unframed", "small", "large", "novel_test", "familiar_test",
        "single_test", "three_test", "redundant_test", "one_match", "five_match")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        first = puzzle["train"][0]
        delta = first["input"] != first["output"]
        marker = int(first["output"][delta][0])
        fill = _infer_fill(puzzle["train"], marker)
        palettes.add((marker, fill))
        marker_digits.add(marker)
        fill_digits.add(fill)
        known = {_normalize(c) for pair in puzzle["train"]
                 for c in _components(pair["input"], marker)}
        train_areas = [pair["input"].size for pair in puzzle["train"]]
        within = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _inspect_pair(pair, marker)
                carrier_digits.add(info["carrier"])
                dimensions.add(a.shape)
                within.add(a.shape)
                shapes.update(info["templates"])
                template_counts.add(len(info["templates"]))
                marker_counts.add(len(info["marker_components"]))
                match_counts.add(sum(_normalize(c) in info["templates"]
                                     for c in info["carrier_components"]))
                fill_amounts.add((a.shape, int(np.count_nonzero(a == fill))))
                canonical = np.zeros_like(a)
                canonical[a == marker] = 1
                canonical[a == info["carrier"]] = 2
                canonical[a == fill] = 3
                normalized.add(_key(canonical))
                templates = info["templates"]
                carrier_shapes = [_normalize(c) for c in info["carrier_components"]]
                decoys = [s for s in carrier_shapes if s not in templates]
                frequencies = {s: carrier_shapes.count(s) for s in templates}
                evidence["diagonal"] += int(DIAMOND in templates)
                evidence["edge"] += int(any(r in (0, a.shape[0] - 1) or
                                             c in (0, a.shape[1] - 1)
                                             for cells in info["marker_components"] +
                                                          info["carrier_components"]
                                             for r, c in cells))
                evidence["unframed"] += int(not (np.all(a[0] == a[0, 0]) and
                                                  np.all(a[-1] == a[0, 0]) and
                                                  np.all(a[:, 0] == a[0, 0]) and
                                                  np.all(a[:, -1] == a[0, 0])))
                evidence["one_match"] += int(1 in frequencies.values())
                evidence["five_match"] += int(any(n >= 5 for n in frequencies.values()))
                if split == "train":
                    evidence["equal_box"] += int(any(len(s) == len(t) and _box(s) == _box(t)
                                                     for s in decoys for t in templates))
                    evidence["rotated"] += int(any(s in _variants(t)
                                                   for s in decoys for t in templates))
                    evidence["larger_subpattern"] += int(PLUS_EXTENDED in decoys and
                                                           PLUS in templates)
                    evidence["protected"] += int(any(_normalize(c) in templates
                                                      for c in _components(a, fill)))
                else:
                    evidence["small"] += int(a.size < min(train_areas))
                    evidence["large"] += int(a.size > max(train_areas))
                    evidence["novel_test"] += int(bool(templates - known))
                    evidence["familiar_test"] += int(templates <= known)
                    evidence["single_test"] += int(len(templates) == 1)
                    evidence["three_test"] += int(len(templates) >= 3)
                    evidence["redundant_test"] += int(len(info["marker_components"]) >
                                                       len(templates))
        assert len(within) >= 2
    assert len({n for n, _ in counts}) >= 4 and len({n for _, n in counts}) >= 3
    assert len(dimensions) >= 30 and len(palettes) >= 20
    assert len(shapes) >= 25 and {1, 2, 3}.issubset(template_counts)
    assert max(marker_counts) >= 4 and max(match_counts) >= 5
    assert len(fill_amounts) >= 80 and len(normalized) >= 100
    assert min(len(marker_digits), len(fill_digits), len(carrier_digits)) >= 8
    assert evidence["diagonal"] >= 42 and evidence["equal_box"] >= 42
    assert evidence["rotated"] >= 42 and evidence["larger_subpattern"] >= 42
    assert evidence["protected"] >= 42
    assert evidence["edge"] >= 10 and evidence["unframed"] >= 10
    assert all(evidence[name] > 0 for name in (
        "small", "large", "novel_test", "familiar_test", "single_test",
        "three_test", "redundant_test", "one_match", "five_match"))
    for nt, ns in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(9000 + nt + ns, nt, ns)
        assert _puzzles_equal(p, generate_puzzle(9000 + nt + ns, nt, ns))
    assert not _puzzles_equal(generate_puzzle(9010, 7, 3), generate_puzzle(9011, 7, 3))
    for i, kwargs in enumerate(({"num_test": 1}, {"num_test": 4},
                                 {"num_train": 2}, {"num_train": 10})):
        validate_generated_puzzle(generate_puzzle(10000 + i, **kwargs))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(11000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
