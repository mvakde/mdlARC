import numpy as np


_SOURCE_RAW = (
    ("7777777777/7666667777/7677767777/7677767777/7677767777/7666667777/7777777777/7557777777/7557777777/7777777777", "7777777777/7666667777/7655567777/7655567777/7655567777/7666667777/7777777777/7777777777/7777777777/7777777777"),
    ("7777777777/7666777777/7676777777/7666777777/7777777777/7777777777/7777777777/7777777727/7777777777", "7777777777/7666777777/7626777777/7666777777/7777777777/7777777777/7777777777/7777777777/7777777777"),
    ("7877777777/7887777777/7777777777/7666666777/7677776777/7677776777/7677776777/7677776777/7666666777/7777777777", "7777777777/7777777777/7777777777/7666666777/7687776777/7687776777/7687776777/7688886777/7666666777/7777777777"),
    ("7777777799/7777777779/6666666677/6777777677/6777777677/6777777677/6777777677/6777777677/6777777677/6666666677", "7777777777/7777777777/6666666677/6999999677/6777779677/6777779677/6777779677/6777779677/6777779677/6666666677"),
)


def _decode_grid(text):
    return np.array([[int(ch) for ch in row] for row in text.split("/")], dtype=np.int8)


def _array_key(array):
    return (tuple(array.shape), tuple(int(x) for x in array.flat))


def _pair_key(pair):
    return _array_key(pair["input"]), _array_key(pair["output"])


_SOURCE_PAIR_KEYS = frozenset((_array_key(_decode_grid(a)), _array_key(_decode_grid(b))) for a, b in _SOURCE_RAW)
_LS = ("tl", "tr", "bl", "br")
_US = ("u_n", "u_s", "u_e", "u_w")
_PLACEMENTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _checked_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _checked_optional_count(num_train, "num_train", 2, 10)
    test = _checked_optional_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _motif_mask(kind, height, width):
    mask = np.zeros((height, width), dtype=bool)
    if kind == "point":
        assert height == width == 1
        mask[0, 0] = True
    elif kind == "solid":
        assert height >= 2 and width >= 2
        mask[:] = True
    elif kind in _LS:
        assert height >= 2 and width >= 2
        mask[0 if kind[0] == "t" else -1, :] = True
        mask[:, 0 if kind[1] == "l" else -1] = True
    elif kind in _US:
        assert height >= 3 and width >= 3
        if kind != "u_n":
            mask[0, :] = True
        if kind != "u_s":
            mask[-1, :] = True
        if kind != "u_w":
            mask[:, 0] = True
        if kind != "u_e":
            mask[:, -1] = True
    else:
        raise ValueError("unknown motif")
    return mask


def _classify_template(crop):
    height, width = crop.shape
    if (height, width) == (1, 1) and crop[0, 0]:
        return "point"
    if height >= 2 and width >= 2 and np.all(crop):
        return "solid"
    choices = _LS if min(height, width) == 2 else _LS + _US
    for kind in choices:
        if np.array_equal(crop, _motif_mask(kind, height, width)):
            return kind
    return None


def _margin_split(rng, minimum):
    spare = 30 - minimum
    assert spare >= 0
    if spare == 0:
        return 0, 0
    mode = int(rng.integers(0, 10))
    if mode < 1:
        extra = spare
    elif mode < 3:
        extra = int(rng.integers(min(5, spare), spare + 1))
    else:
        extra = int(rng.integers(0, min(4, spare) + 1))
    side = int(rng.integers(0, 4))
    before = 0 if side == 0 else extra if side == 1 else int(rng.integers(0, extra + 1))
    return before, extra - before


def _geometry(rng, frame_h, frame_w, glyph_h, glyph_w, placement):
    gap_v = int(rng.integers(1, 5))
    gap_h = int(rng.integers(1, 5))
    if "N" in placement:
        glyph_r = -glyph_h - gap_v
    elif "S" in placement:
        glyph_r = frame_h + gap_v
    else:
        glyph_r = int(rng.integers(-glyph_h + 1, frame_h))
    if "W" in placement:
        glyph_c = -glyph_w - gap_h
    elif "E" in placement:
        glyph_c = frame_w + gap_h
    else:
        glyph_c = int(rng.integers(-glyph_w + 1, frame_w))
    low_r, low_c = min(0, glyph_r), min(0, glyph_c)
    extent_h = max(frame_h, glyph_r + glyph_h) - low_r
    extent_w = max(frame_w, glyph_c + glyph_w) - low_c
    if extent_h > 30 or extent_w > 30:
        return None
    top, bottom = _margin_split(rng, extent_h)
    left, right = _margin_split(rng, extent_w)
    frame_r, frame_c = top - low_r, left - low_c
    glyph_r += top - low_r
    glyph_c += left - low_c
    return extent_h + top + bottom, extent_w + left + right, frame_r, frame_c, glyph_r, glyph_c


def _dimensions(rng, kind, style, section, anchor, held_out):
    if kind == "point":
        return 1, 1, 1, 1
    low = 3 if kind in _US else 2
    maximum = 4 if style == "simple" else 6
    glyph_h = int(rng.integers(low, maximum + 1))
    glyph_w = int(rng.integers(low, maximum + 1))
    if anchor:
        mode = "both"
    else:
        mode = ("both", "horizontal", "vertical", "same", "unequal")[int(rng.integers(0, 5))]
    cap = 6 if style == "simple" else 10 if style == "mixed" else (8 if section == "train" else 12)
    cap = max(cap, glyph_h, glyph_w)
    if mode == "both":
        dh, dw = int(rng.integers(1, 7)), int(rng.integers(1, 7))
    elif mode == "horizontal":
        dh, dw = 0, int(rng.integers(1, 7))
    elif mode == "vertical":
        dh, dw = int(rng.integers(1, 7)), 0
    elif mode == "same":
        dh, dw = 0, 0
    else:
        dh, dw = (1, 5) if rng.integers(0, 2) else (5, 1)
    cavity_h = min(cap, glyph_h + dh)
    cavity_w = min(cap, glyph_w + dw)
    if anchor:
        cavity_h = max(cavity_h, glyph_h + 1)
        cavity_w = max(cavity_w, glyph_w + 1)
    if style == "shift" and section == "test" and mode == "both":
        cavity_h = max(cavity_h, min(cap, 6))
        cavity_w = max(cavity_w, min(cap, 6))
    if held_out:
        cavity_h = int(rng.integers(9, 13))
        cavity_w = int(rng.integers(9, 13))
    return glyph_h, glyph_w, cavity_h, cavity_w


def _make_pair(rng, kind, placement, colors, style, section, anchor, held_out):
    glyph_h, glyph_w, cavity_h, cavity_w = _dimensions(rng, kind, style, section, anchor, held_out)
    frame_h, frame_w = cavity_h + 2, cavity_w + 2
    geometry = _geometry(rng, frame_h, frame_w, glyph_h, glyph_w, placement)
    if geometry is None:
        return None
    height, width, frame_r, frame_c, glyph_r, glyph_c = geometry
    background, frame, accent = colors
    source = np.full((height, width), background, dtype=np.int8)
    source[frame_r, frame_c:frame_c + frame_w] = frame
    source[frame_r + frame_h - 1, frame_c:frame_c + frame_w] = frame
    source[frame_r:frame_r + frame_h, frame_c] = frame
    source[frame_r:frame_r + frame_h, frame_c + frame_w - 1] = frame
    glyph_mask = _motif_mask(kind, glyph_h, glyph_w)
    glyph_view = source[glyph_r:glyph_r + glyph_h, glyph_c:glyph_c + glyph_w]
    glyph_view[glyph_mask] = accent
    target = source.copy()
    target[source == accent] = background
    cavity = target[frame_r + 1:frame_r + frame_h - 1, frame_c + 1:frame_c + frame_w - 1]
    cavity[_motif_mask(kind, cavity_h, cavity_w)] = accent
    return {"input": source, "output": target}


def _placement_name(glyph_box, frame_box):
    gr0, gr1, gc0, gc1 = glyph_box
    fr0, fr1, fc0, fc1 = frame_box
    vertical = "N" if gr1 < fr0 else "S" if gr0 > fr1 else ""
    horizontal = "W" if gc1 < fc0 else "E" if gc0 > fc1 else ""
    return vertical + horizontal


def _analyze_pair(source, target):
    assert source.shape == target.shape
    colors = [int(x) for x in np.unique(source)]
    assert len(colors) == 3
    frame_candidates = []
    for color in colors:
        points = np.argwhere(source == color)
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = np.zeros(source.shape, dtype=bool)
        border[r0, c0:c1 + 1] = True
        border[r1, c0:c1 + 1] = True
        border[r0:r1 + 1, c0] = True
        border[r0:r1 + 1, c1] = True
        if np.array_equal(source == color, border):
            frame_candidates.append((color, int(r0), int(r1), int(c0), int(c1), border))
    assert len(frame_candidates) == 1
    frame, fr0, fr1, fc0, fc1, border = frame_candidates[0]
    candidates = []
    for color in colors:
        if color == frame:
            continue
        points = np.argwhere(source == color)
        gr0, gc0 = points.min(axis=0)
        gr1, gc1 = points.max(axis=0)
        placement = _placement_name((gr0, gr1, gc0, gc1), (fr0, fr1, fc0, fc1))
        if not placement:
            continue
        crop = source[gr0:gr1 + 1, gc0:gc1 + 1] == color
        kind = _classify_template(crop)
        if kind is not None:
            candidates.append((color, kind, int(gr0), int(gr1), int(gc0), int(gc1), placement, crop))
    assert len(candidates) == 1
    accent, kind, gr0, gr1, gc0, gc1, placement, crop = candidates[0]
    background = next(color for color in colors if color not in (frame, accent))
    cavity_h, cavity_w = fr1 - fr0 - 1, fc1 - fc0 - 1
    glyph_h, glyph_w = gr1 - gr0 + 1, gc1 - gc0 + 1
    assert cavity_h >= glyph_h and cavity_w >= glyph_w
    assert placement in _PLACEMENTS
    # Reconstruct both grids, so extra objects, holes, and altered cells fail.
    expected_input = np.full(source.shape, background, dtype=source.dtype)
    expected_input[border] = frame
    view = expected_input[gr0:gr1 + 1, gc0:gc1 + 1]
    view[crop] = accent
    assert np.array_equal(source, expected_input)
    expected_output = np.full(source.shape, background, dtype=source.dtype)
    expected_output[border] = frame
    cavity = expected_output[fr0 + 1:fr1, fc0 + 1:fc1]
    cavity[_motif_mask(kind, cavity_h, cavity_w)] = accent
    assert np.array_equal(target, expected_output)
    # A background cell separates the external glyph from the frame.
    for row, col in np.argwhere(source == accent):
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < source.shape[0] and 0 <= nc < source.shape[1]:
                assert source[nr, nc] != frame
    return {"background": background, "frame": frame, "accent": accent,
            "motif": kind, "placement": placement, "frame_box": (fr0, fr1, fc0, fc1),
            "glyph_box": (gr0, gr1, gc0, gc1), "cavity_h": cavity_h,
            "cavity_w": cavity_w, "glyph_h": glyph_h, "glyph_w": glyph_w}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            key = _array_key(pair["input"])
            full = _pair_key(pair)
            assert key not in seen_inputs and full not in seen_pairs
            assert full not in _SOURCE_PAIR_KEYS
            seen_inputs.add(key)
            seen_pairs.add(full)
            analyses[section].append(_analyze_pair(pair["input"], pair["output"]))
    train = analyses["train"]
    test = analyses["test"]
    assert any(a["motif"] == "solid" and a["cavity_h"] > a["glyph_h"] and a["cavity_w"] > a["glyph_w"] for a in train)
    assert any(a["motif"] in _LS and a["cavity_h"] > a["glyph_h"] and a["cavity_w"] > a["glyph_w"] for a in train)
    if any(a["motif"] in _US for a in test):
        assert any(a["motif"] in _US for a in train)
    if any(a["motif"] == "point" for a in test):
        assert any(a["motif"] == "point" for a in train)
    assert len({a["accent"] for a in train}) >= 2
    assert len({a["background"] for a in train}) >= 2
    assert len({a["frame"] for a in train}) >= 2


def _sample_roles(rng, accent, previous=None):
    for _ in range(20):
        choices = [x for x in range(10) if x != accent]
        selected = rng.choice(choices, size=2, replace=False)
        bg, frame = int(selected[0]), int(selected[1])
        if previous is None or (bg != previous[0] and frame != previous[1]):
            return bg, frame, accent
    raise AssertionError("could not sample color roles")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    style = ("simple", "mixed", "shift")[int(rng.integers(0, 3))]
    accent_pool = [int(x) for x in rng.choice(10, size=4, replace=False)]
    base_l = _LS[int(rng.integers(0, 4))]
    base_u = _US[int(rng.integers(0, 4))]
    positions = [str(x) for x in rng.permutation(_PLACEMENTS)]
    train_specs = [("solid", positions[0], True), (base_l, positions[1], True)]
    if train_count >= 3:
        train_specs.append((base_u, positions[2], True))
    if train_count >= 4:
        train_specs.append(("point", positions[3], False))
    if train_count >= 5:
        train_specs.append((base_l, positions[4], False))
    if train_count >= 6:
        alternate = _LS[(_LS.index(base_l) + int(rng.integers(1, 4))) % 4]
        train_specs.append((alternate, positions[1], False))
    while len(train_specs) < train_count:
        kind = ("solid",) + _LS + _US
        train_specs.append((kind[int(rng.integers(0, len(kind)))], _PLACEMENTS[int(rng.integers(0, 8))], False))
    order = rng.permutation(train_count)
    train_specs = [train_specs[int(i)] for i in order]
    # Training contains both a filled expansion and a thin-stroke expansion.
    # New glyph types appear in tests only after their family is demonstrated.
    allowed_test = ("solid",) + _LS + (_US if train_count >= 3 else ()) + (("point",) if train_count >= 4 else ())
    test_specs = []
    for index in range(test_count):
        kind = allowed_test[int(rng.integers(0, len(allowed_test)))]
        held_out = style == "shift" and index == 0
        if held_out:
            kind = _LS[(_LS.index(base_l) + int(rng.integers(1, 4))) % 4]
        elif index == 0 and train_count >= 3 and rng.integers(0, 3) == 0:
            kind = _US[int(rng.integers(0, 4))]
        test_specs.append((kind, _PLACEMENTS[int(rng.integers(0, 8))], False, held_out))
    train_accents = [accent_pool[0], accent_pool[1]]
    train_accents += [accent_pool[int(rng.integers(0, 4))] for _ in range(train_count - 2)]
    reused = [accent_pool[0]]
    novel = [x for x in range(10) if x not in set(train_accents)]
    rng.shuffle(novel)
    test_accents = []
    for index in range(test_count):
        if test_count >= 2:
            use_new = index % 2 == 1
        else:
            use_new = bool(rng.integers(0, 2))
        test_accents.append(novel[index % len(novel)] if use_new else reused[0])
    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = None
    train_specs = [(kind, placement, anchor, False) for kind, placement, anchor in train_specs]
    for section, specs, accents in (("train", train_specs, train_accents), ("test", test_specs, test_accents)):
        for index, ((kind, placement, anchor, held_out), accent) in enumerate(zip(specs, accents)):
            roles = _sample_roles(rng, accent, first_roles if section == "train" and index == 1 else None)
            if section == "train" and index == 0:
                first_roles = roles
            for _ in range(100):
                pair = _make_pair(rng, kind, placement, roles, style, section, anchor, held_out)
                if pair is not None and _array_key(pair["input"]) not in seen and _pair_key(pair) not in _SOURCE_PAIR_KEYS:
                    break
            else:
                raise AssertionError("could not construct a distinct pair")
            seen.add(_array_key(pair["input"]))
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output")) for s in ("train", "test"))


def _predict_from_visible_grid(grid):
    """Side-based reconstruction for self-test; uses no generation metadata."""
    colors = [int(color) for color in np.unique(grid)]
    frames = []
    for color in colors:
        points = np.argwhere(grid == color)
        top, left = points.min(axis=0)
        bottom, right = points.max(axis=0)
        if bottom - top < 2 or right - left < 2:
            continue
        edge = np.zeros(grid.shape, dtype=bool)
        edge[top, left:right + 1] = True
        edge[bottom, left:right + 1] = True
        edge[top:bottom + 1, left] = True
        edge[top:bottom + 1, right] = True
        if np.array_equal(edge, grid == color):
            frames.append((color, int(top), int(bottom), int(left), int(right), edge))
    assert len(frames) == 1
    frame, top, bottom, left, right, edge = frames[0]
    exterior = []
    for color in colors:
        if color == frame:
            continue
        points = np.argwhere(grid == color)
        if (points[:, 0].max() < top or points[:, 0].min() > bottom
                or points[:, 1].max() < left or points[:, 1].min() > right):
            exterior.append(color)
    assert len(exterior) == 1
    accent = exterior[0]
    background = next(color for color in colors if color not in (frame, accent))
    points = np.argwhere(grid == accent)
    gr0, gc0 = points.min(axis=0)
    gr1, gc1 = points.max(axis=0)
    glyph = grid[gr0:gr1 + 1, gc0:gc1 + 1] == accent
    cavity = np.zeros((bottom - top - 1, right - left - 1), dtype=bool)
    if glyph.shape == (1, 1):
        assert cavity.shape == (1, 1)
        cavity[0, 0] = True
    elif glyph.all():
        cavity[:] = True
    else:
        sides = (bool(glyph[0, :].all()), bool(glyph[-1, :].all()),
                 bool(glyph[:, 0].all()), bool(glyph[:, -1].all()))
        assert sum(sides) in (2, 3)
        observed = np.zeros(glyph.shape, dtype=bool)
        for mask in (observed, cavity):
            if sides[0]:
                mask[0, :] = True
            if sides[1]:
                mask[-1, :] = True
            if sides[2]:
                mask[:, 0] = True
            if sides[3]:
                mask[:, -1] = True
        assert np.array_equal(observed, glyph)
    predicted = np.full(grid.shape, background, dtype=grid.dtype)
    predicted[edge] = frame
    predicted[top + 1:bottom, left + 1:right][cavity] = accent
    return predicted


def self_test() -> None:
    puzzle_keys, shapes, layouts, role_pairs = set(), set(), set(), set()
    counts_train, counts_test, motifs, placements, stretch_modes = set(), set(), set(), set(), set()
    reused, novel, edge, size_30, unequal = 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(tuple(_pair_key(pair) for section in ("train", "test") for pair in puzzle[section]))
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        train_accents = {_analyze_pair(pair["input"], pair["output"])["accent"] for pair in puzzle["train"]}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                assert np.array_equal(
                    _predict_from_visible_grid(pair["input"]), pair["output"]
                )
                a = _analyze_pair(pair["input"], pair["output"])
                motifs.add(a["motif"])
                placements.add(a["placement"])
                shapes.add(pair["input"].shape)
                role_pairs.add((a["background"], a["frame"]))
                layouts.add((pair["input"].shape, a["frame_box"], a["glyph_box"], a["motif"]))
                dh, dw = a["cavity_h"] - a["glyph_h"], a["cavity_w"] - a["glyph_w"]
                stretch_modes.add((dh > 0, dw > 0))
                unequal += abs(dh - dw) >= 3
                fr0, fr1, fc0, fc1 = a["frame_box"]
                edge += fr0 == 0 or fc0 == 0 or fr1 == pair["input"].shape[0] - 1 or fc1 == pair["input"].shape[1] - 1
                size_30 += 30 in pair["input"].shape
                if section == "test":
                    reused += a["accent"] in train_accents
                    novel += a["accent"] not in train_accents
    assert len(puzzle_keys) == 42
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(shapes) >= 50 and len(layouts) >= 150 and len(role_pairs) >= 30
    assert set(_LS + _US + ("solid", "point")) <= motifs
    assert set(_PLACEMENTS) <= placements
    assert {(True, True), (True, False), (False, True), (False, False)} <= stretch_modes
    assert reused > 0 and novel > 0 and edge > 0 and size_30 > 0 and unequal > 0
    assert _puzzles_equal(generate_puzzle(99), generate_puzzle(99))
    assert _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(generate_puzzle(2718, 7, 3), generate_puzzle(2719, 7, 3))
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        puzzle = generate_puzzle(1000 + i, train, test)
        assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for i, kwargs in enumerate(({"num_test": 1}, {"num_test": 4}, {"num_train": 2}, {"num_train": 10})):
        validate_generated_puzzle(generate_puzzle(2000 + i, **kwargs))
    for i, kwargs in enumerate(({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4})):
        try:
            generate_puzzle(3000 + i, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
