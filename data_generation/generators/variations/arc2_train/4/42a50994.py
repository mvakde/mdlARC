import numpy as np


_SOURCE_INPUT_ROWS = {
    (
        "0000000000005005000",
        "0000500005000000000",
        "0000000500500000500",
        "0000000005055000500",
        "0500505000000505500",
        "0050000050000000000",
        "0050000000005055000",
        "0005000000000000055",
        "5005000000050500500",
        "5000000005050000550",
        "0500000000500500005",
    ),
    (
        "000000000006000000",
        "060006006060006000",
        "006000000000006000",
        "060000000606006006",
        "000000060000000000",
        "000000006000006000",
        "006000000000600000",
        "000006000000000000",
        "000000066000000000",
        "006000000006060000",
        "000000060006000000",
        "006000000000000066",
    ),
    (
        "000004040",
        "000040000",
        "040000400",
        "000440000",
        "004000000",
        "004040044",
        "404000000",
        "000000000",
        "004000400",
        "004000000",
        "040000404",
        "404040040",
        "040000040",
        "000004000",
        "000044000",
        "404040044",
        "004000000",
    ),
    (
        "08000000000",
        "80000000000",
        "00008008080",
        "00008000000",
        "00880000000",
        "00080000800",
        "00800080000",
        "00000000000",
        "00000800008",
        "00000880000",
        "08008000000",
        "00000000000",
        "08000000000",
        "00000080080",
        "00008000000",
        "00000000800",
    ),
    (
        "03003000003003",
        "30000003333003",
        "30000030000000",
        "00303000033300",
        "30300000003003",
        "00000300000000",
        "30000000000300",
        "00003300300000",
        "00000000003300",
        "03000000000003",
        "00003030003000",
        "00030300000000",
        "00033303300000",
        "30030030000030",
        "30300000300300",
        "30000300000000",
        "00330030000033",
    ),
}

_NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _check_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_optional_count(num_train, "num_train", 2, 10)
    _check_optional_count(num_test, "num_test", 1, 4)

    selector = int(seed)

    if num_train is None and num_test is None:
        test_count = 1 + selector % 4
        minimum_train = max(2, test_count + 1)
        train_count = minimum_train + (
            (selector // 4) % (11 - minimum_train)
        )
    elif num_train is None:
        test_count = int(num_test)
        minimum_train = max(2, test_count + 1)
        train_count = minimum_train + selector % (11 - minimum_train)
    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)
        test_count = 1 + selector % maximum_test
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _sample_palettes(rng: np.random.Generator, train_count: int, test_count: int) -> list[tuple[int, int]]:
    """Color identity is presentation; the majority/background rule is universal."""
    mode = int(rng.integers(0, 4))  # fixed, changing background, changing signal, both
    background = int(rng.integers(0, 10))
    signal = int(rng.choice([c for c in range(10) if c != background]))
    palettes = [(background, signal)]
    total = train_count + test_count
    for index in range(1, total):
        if index >= train_count and rng.random() < 0.42:
            palettes.append(palettes[int(rng.integers(0, train_count))])
            continue
        if mode == 0:
            pair = (background, signal)
        elif mode == 1:
            choices = [c for c in range(10) if c != signal and (index != 1 or c != background)]
            pair = (int(rng.choice(choices)), signal)
        elif mode == 2:
            choices = [c for c in range(10) if c != background and (index != 1 or c != signal)]
            pair = (background, int(rng.choice(choices)))
        else:
            if rng.random() < 0.26:
                pair = (palettes[-1][1], palettes[-1][0])
            else:
                bg = int(rng.integers(0, 10))
                sig = int(rng.choice([c for c in range(10) if c != bg]))
                pair = (bg, sig)
            if index == 1 and pair == palettes[0]:
                pair = (signal, background)
        palettes.append(pair)
    return palettes


def _edge_kinds(coords) -> tuple[bool, bool]:
    cells = set(coords)
    orthogonal = diagonal = False
    for row, col in cells:
        for dr, dc in _NEIGHBORS_8:
            if (row + dr, col + dc) in cells:
                if dr == 0 or dc == 0:
                    orthogonal = True
                else:
                    diagonal = True
    return orthogonal, diagonal


def _normalize_shape(coords) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _transform_shape(shape, code: int):
    points = []
    for r, c in shape:
        if code >= 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        points.append((r, c))
    return _normalize_shape(points)


def _random_connected_shape(rng: np.random.Generator, size: int):
    cells = {(0, 0)}
    while len(cells) < size:
        candidates = set()
        for r, c in cells:
            for dr, dc in _NEIGHBORS_8:
                point = (r + dr, c + dc)
                if point not in cells:
                    candidates.add(point)
        options = sorted(candidates)
        cells.add(options[int(rng.integers(0, len(options)))])
    return _normalize_shape(cells)


def _make_shape(rng: np.random.Generator, kind: str, size: int):
    if kind == "dimer":
        return ((0, 0), (0, 1)) if rng.random() < 0.5 else ((0, 0), (1, 1))
    if kind == "line":
        direction = _NEIGHBORS_8[int(rng.integers(0, 8))]
        return _normalize_shape([(i * direction[0], i * direction[1]) for i in range(size)])
    if kind == "bend":
        first = _NEIGHBORS_8[int(rng.integers(0, 8))]
        second = _NEIGHBORS_8[int(rng.integers(0, 8))]
        if second == (-first[0], -first[1]):
            second = (first[1], -first[0])
        turn = max(2, size // 2)
        coords = [(i * first[0], i * first[1]) for i in range(turn)]
        r, c = coords[-1]
        coords.extend((r + i * second[0], c + i * second[1]) for i in range(1, size - turn + 1))
        return _normalize_shape(coords)
    if kind == "branch":
        trunk = max(3, size // 2)
        coords = [(i, 0) for i in range(trunk)]
        for i in range(size - trunk):
            origin = 1 + i % (trunk - 1)
            arm = 1 + i // (2 * (trunk - 1))
            side = -1 if i % 2 else 1
            coords.append((origin, side * arm))
        return _normalize_shape(coords)
    if kind == "ribbon":
        thickness = int(rng.integers(2, 5))
        length = max(4, size)
        coords = [(r, c) for r in range(thickness) for c in range(length)
                  if r == 0 or rng.random() > .18 or c in (0, length - 1)]
        return _normalize_shape(coords)
    if kind == "ring":
        side = int(rng.integers(5, 9))
        return _normalize_shape([(r, c) for r in range(side) for c in range(side)
                                 if r in (0, side - 1) or c in (0, side - 1)])
    return _random_connected_shape(rng, size)


def _stamp(rng: np.random.Generator, mask: np.ndarray, shape, separated: bool):
    h, w = mask.shape
    for code in rng.permutation(8):
        trial = _transform_shape(shape, int(code))
        sh = max(r for r, _ in trial) + 1
        sw = max(c for _, c in trial) + 1
        if sh > h or sw > w:
            continue
        for _ in range(55):
            top = int(rng.integers(0, h - sh + 1))
            left = int(rng.integers(0, w - sw + 1))
            if separated:
                blocked = False
                for r, c in trial:
                    rr, cc = top + r, left + c
                    if np.any(mask[max(0, rr - 1):min(h, rr + 2), max(0, cc - 1):min(w, cc + 2)]):
                        blocked = True
                        break
                if blocked:
                    continue
            for r, c in trial:
                mask[top + r, left + c] = True
            return top, left, sh, sw
    return None


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask.astype(np.int8), 1)
    count = np.zeros((h, w), dtype=np.int8)
    for dr, dc in _NEIGHBORS_8:
        count += padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return count


def _dimensions(rng: np.random.Generator, regime: str, anchor, geometry_mode: int, shift: int):
    if regime == "source_train" and anchor is None:
        if rng.random() < 0.22:
            side = int(rng.integers(11, 21))
            return side, side
        return int(rng.integers(11, 21)), int(rng.integers(11, 21))
    if geometry_mode == 1:
        return anchor
    if regime == "target_test" and shift == 1:
        category = int(rng.choice(6, p=[.05, .18, .34, .12, .16, .15]))
    else:
        category = int(rng.choice(6, p=[.20, .29, .19, .14, .11, .07]))
    if category == 0:
        h, w = int(rng.integers(6, 12)), int(rng.integers(6, 12))
    elif category == 1:
        h, w = int(rng.integers(10, 23)), int(rng.integers(10, 23))
    elif category == 2:
        h, w = int(rng.integers(18, 31)), int(rng.integers(18, 31))
    elif category == 3:
        h, w = int(rng.integers(5, 10)), int(rng.integers(16, 31))
    elif category == 4:
        h = w = int(rng.integers(7, 31))
    else:
        h, w = 30, int(rng.integers(8, 31))
    if rng.random() < 0.5:
        h, w = w, h
    if geometry_mode == 2 and anchor is not None:
        if anchor[0] < anchor[1] and h > w or anchor[0] > anchor[1] and h < w:
            h, w = w, h
    return h, w


def _add_singletons(rng: np.random.Generator, mask: np.ndarray, requested: int) -> int:
    h, w = mask.shape
    added = 0
    for flat in rng.permutation(h * w):
        r, c = divmod(int(flat), w)
        if not np.any(mask[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)]):
            mask[r, c] = True
            added += 1
            if added >= requested:
                break
    return added


def _anchor_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    teaching_shapes = (
        ((0, 0), (0, 1), (1, 2), (2, 2)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 1)),
    )
    for _ in range(10):
        mask = np.zeros((h, w), dtype=bool)
        if all(_stamp(rng, mask, shape, True) is not None for shape in teaching_shapes):
            break
    else:
        mask = np.zeros((h, w), dtype=bool)
        for r, c in teaching_shapes[0]:
            mask[1 + r, 1 + c] = True
        for r, c in teaching_shapes[1]:
            mask[1 + r, w - 3 + c] = True
        for r, c in teaching_shapes[2]:
            mask[h - 3 + r, 1 + c] = True
    for _ in range(int(rng.integers(0, 4))):
        size = int(rng.integers(3, min(14, h + w) + 1))
        _stamp(rng, mask, _make_shape(rng, str(rng.choice(["line", "bend", "branch", "ribbon", "irregular"])), size), True)
    _add_singletons(rng, mask, int(rng.integers(2, 7)))
    return mask


def _ordinary_mask(rng: np.random.Generator, h: int, w: int, regime: str, shift: int) -> np.ndarray:
    area = h * w
    styles = ["scattered", "crowded", "shaped", "hybrid", "dimer_rich", "dimer_free", "loop", "all_singletons"]
    if regime == "target_test" and shift == 3:
        probs = [.06, .06, .12, .12, .20, .17, .20, .07]
    else:
        probs = [.19, .13, .17, .20, .10, .10, .08, .03]
    style = str(rng.choice(styles, p=probs))
    mask = np.zeros((h, w), dtype=bool)
    center = None
    if style in ("scattered", "crowded", "hybrid"):
        if style == "crowded" or regime == "target_test" and shift == 2:
            density = float(rng.uniform(.17, .29))
        elif style == "scattered":
            density = float(rng.uniform(.075, .24))
        else:
            density = float(rng.uniform(.025, .12))
        mask = rng.random((h, w)) < density
    if style in ("shaped", "hybrid", "dimer_rich", "dimer_free", "loop"):
        if style == "loop":
            placement = _stamp(rng, mask, _make_shape(rng, "ring", 0), True)
            if placement is not None:
                top, left, sh, sw = placement
                center = (top + sh // 2, left + sw // 2)
        if style == "dimer_rich":
            count = int(rng.integers(3, min(11, max(4, area // 27)) + 1))
        else:
            count = int(rng.integers(1, min(8, max(2, area // 60 + 1)) + 1))
        for _ in range(count):
            if style == "dimer_rich" or style != "dimer_free" and rng.random() < .24:
                kind, size = "dimer", 2
            else:
                kind = str(rng.choice(["line", "bend", "branch", "ribbon", "irregular"],
                                      p=[.25, .19, .14, .13, .29]))
                size = int(rng.integers(3, min(27, max(5, min(h, w) + 7)) + 1))
                if kind == "ribbon":
                    size = int(rng.integers(4, min(16, max(h, w)) + 1))
            shape = _make_shape(rng, kind, size)
            _stamp(rng, mask, shape, style != "hybrid")
    # Remove naturally isolated specks so the noise burden can be drawn separately.
    mask &= _neighbor_count(mask) > 0
    if style == "all_singletons":
        mask[:] = False
    if center is not None and not np.any(mask[max(0, center[0] - 1):center[0] + 2,
                                               max(0, center[1] - 1):center[1] + 2]):
        mask[center] = True
    noise_probabilities = ([.20, .24, .20, .36]
                           if regime == "target_test" and shift == 3
                           else [.10, .27, .36, .27])
    noise_mode = int(rng.choice(4, p=noise_probabilities))
    if noise_mode == 0 and style != "all_singletons":
        requested = 0
    elif noise_mode == 1:
        requested = int(rng.integers(1, 4))
    elif noise_mode == 2:
        requested = int(rng.integers(4, 10))
    else:
        requested = int(rng.integers(10, 23))
    if style == "all_singletons":
        requested = max(2, requested)
    if requested:
        _add_singletons(rng, mask, requested)
    return mask


def _remove_singletons(grid: np.ndarray, background: int) -> np.ndarray:
    signal = grid != background
    result = grid.copy()
    result[signal & (_neighbor_count(signal) == 0)] = background
    return result


def _components(mask: np.ndarray):
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    result = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        component = []
        while stack:
            row, col = stack.pop()
            component.append((row, col))
            for dr, dc in _NEIGHBORS_8:
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(tuple(component))
    return result


def _grid_signature(grid: np.ndarray):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _input_rows(grid: np.ndarray):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _infer_background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2 and counts.max() > counts.min()
    return int(values[int(np.argmax(counts))])


def _enclosed_singleton(mask: np.ndarray, components) -> bool:
    retained = np.zeros_like(mask)
    singles = []
    for component in components:
        if len(component) == 1:
            singles.extend(component)
        else:
            for row, col in component:
                retained[row, col] = True
    height, width = mask.shape
    outside = np.zeros_like(mask)
    stack = [(row, col) for row in range(height) for col in (0, width - 1)]
    stack += [(row, col) for col in range(width) for row in (0, height - 1)]
    while stack:
        row, col = stack.pop()
        if outside[row, col] or retained[row, col]:
            continue
        outside[row, col] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = row + dr, col + dc
            if 0 <= rr < height and 0 <= cc < width and not outside[rr, cc] and not retained[rr, cc]:
                stack.append((rr, cc))
    return any(not outside[row, col] for row, col in singles)


def _pair(rng: np.random.Generator, dimensions, palette, anchor: bool, regime: str, shift: int):
    h, w = dimensions
    background, signal = palette
    for _ in range(12):
        mask = _anchor_mask(rng, h, w) if anchor else _ordinary_mask(rng, h, w, regime, shift)
        if 0 < int(mask.sum()) < h * w * .43:
            grid = np.full((h, w), background, dtype=np.int8)
            grid[mask] = signal
            return grid, _remove_singletons(grid, background)
    # A deterministic, always feasible fallback still teaches the same rule.
    mask = np.zeros((h, w), dtype=bool)
    mask[1, 1:3] = True
    if h >= 6 and w >= 6:
        mask[-2, -2] = True
    grid = np.full((h, w), background, dtype=np.int8)
    grid[mask] = signal
    return grid, _remove_singletons(grid, background)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    signatures = set()
    pair_signatures = set()
    train_singletons = 0
    train_retained = 0
    teach_orthogonal = teach_diagonal = False
    teach_large_mixed = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            x, y = pair["input"], pair["output"]
            for grid in (x, y):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert x.shape == y.shape
            sx = _grid_signature(x)
            sp = (sx, _grid_signature(y))
            assert sx not in signatures and sp not in pair_signatures
            signatures.add(sx)
            pair_signatures.add(sp)
            assert _input_rows(x) not in _SOURCE_INPUT_ROWS
            background = _infer_background(x)
            colors = set(int(v) for v in np.unique(x))
            assert set(int(v) for v in np.unique(y)).issubset(colors)
            mask = x != background
            neighbors = _neighbor_count(mask)
            expected = x.copy()
            expected[mask & (neighbors == 0)] = background
            assert np.array_equal(y, expected)
            components = _components(mask)
            if section == "train":
                train_singletons += sum(len(c) == 1 for c in components)
                train_retained += sum(len(c) >= 2 for c in components)
                for component in components:
                    if len(component) == 2:
                        orth, diag = _edge_kinds(component)
                        teach_orthogonal |= orth
                        teach_diagonal |= diag
                    if len(component) >= 4 and all(_edge_kinds(component)):
                        teach_large_mixed = True
    assert train_singletons > 0 and train_retained > 0
    assert teach_orthogonal and teach_diagonal and teach_large_mixed


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(int(seed))
    palettes = _sample_palettes(rng, train_count, test_count)
    geometry_mode = int(rng.choice(3, p=[.70, .12, .18]))
    shift = int(rng.integers(0, 4))  # matched, larger, denser, altered component/noise mix
    puzzle = {"train": [], "test": []}
    used = set()
    anchor_shape = None
    for index in range(train_count + test_count):
        section = "train" if index < train_count else "test"
        regime = "source_train" if section == "train" else "target_test"
        is_anchor = index == 0
        accepted = False
        for _ in range(16):
            child_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            shape = _dimensions(child_rng, regime, anchor_shape, geometry_mode, shift)
            if is_anchor:
                anchor_shape = shape
            x, y = _pair(child_rng, shape, palettes[index], is_anchor, regime, shift)
            signature = _grid_signature(x)
            if signature not in used and _input_rows(x) not in _SOURCE_INPUT_ROWS:
                used.add(signature)
                puzzle[section].append({"input": x, "output": y})
                accepted = True
                break
        if not accepted:
            # An unused canvas shape guarantees a new input even in the
            # exceptional event that every random placement collides.
            used_shapes = {signature[0] for signature in used}
            for height in range(11, 31):
                for width in range(11, 31):
                    if (height, width) in used_shapes:
                        continue
                    child_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                    x, y = _pair(child_rng, (height, width), palettes[index], is_anchor, regime, shift)
                    if _input_rows(x) in _SOURCE_INPUT_ROWS:
                        continue
                    used.add(_grid_signature(x))
                    puzzle[section].append({"input": x, "output": y})
                    if is_anchor:
                        anchor_shape = (height, width)
                    accepted = True
                    break
                if accepted:
                    break
        if not accepted:
            raise RuntimeError("no legal unique canvas remained")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    for seed, nt, nv in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 6, 2)):
        puzzle = generate_puzzle(seed, nt, nv)
        assert len(puzzle["train"]) == nt and len(puzzle["test"]) == nv
    for seed, nt, nv in ((201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, nt, nv)
        assert nt is None or len(puzzle["train"]) == nt
        assert nv is None or len(puzzle["test"]) == nv
    for nt, nv in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(17, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    records = []
    for puzzle in puzzles:
        for section in ("train", "test"):
            for pair in puzzle[section]:
                x, y = pair["input"], pair["output"]
                bg = _infer_background(x)
                mask = x != bg
                comps = _components(mask)
                singleton_positions = {cell for c in comps if len(c) == 1 for cell in c}
                changed = {tuple(map(int, rc)) for rc in np.argwhere(x != y)}
                assert changed == singleton_positions
                records.append((section, x.shape, bg, int(next(v for v in np.unique(x) if v != bg)),
                                float(mask.mean()), len(singleton_positions),
                                [c for c in comps if len(c) >= 2], mask.tobytes()))
    assert len({r[1] for r in records}) >= 45
    assert len({r[2] for r in records}) == 10
    assert len({(r[1], r[7]) for r in records}) >= .98 * len(records)
    assert any(min(r[1]) <= 8 for r in records)
    assert any(max(r[1]) == 30 for r in records)
    assert any(r[1][0] == r[1][1] for r in records)
    assert any(r[5] == 0 for r in records)
    assert any(r[5] >= 10 for r in records)
    assert any(r[4] > .20 for r in records)
    assert any(r[4] < .08 for r in records)
    assert any(len(r[6]) == 0 for r in records)
    assert any(len(r[6]) >= 11 for r in records)
    assert any(sum(len(c) == 2 for c in r[6]) == 0 for r in records)
    assert any(sum(len(c) == 2 for c in r[6]) >= 4 for r in records)
    assert any(any(max(max(p[0] for p in c)-min(p[0] for p in c)+1,
                           max(p[1] for p in c)-min(p[1] for p in c)+1) >= 7
                   for c in r[6]) for r in records)
    assert any(len(r[6]) <= 3 and r[6] for r in records)
    assert any(len(r[6]) >= 13 for r in records)
    assert any(r[0] == "train" and .13 <= r[4] <= .23 and r[5] >= 8 for r in records)
    assert any(r[5] <= 3 and any(len(c) >= 12 for c in r[6]) for r in records)
    assert any(r[5] >= 8 and r[6] and max(len(c) for c in r[6]) <= 8 for r in records)
    assert any(r[5] >= 8 and any(max(max(p[0] for p in c)-min(p[0] for p in c)+1,
                                     max(p[1] for p in c)-min(p[1] for p in c)+1) >= 7
                                 for c in r[6]) for r in records)
    assert any(any(max(max(p[0] for p in c)-min(p[0] for p in c)+1,
                       max(p[1] for p in c)-min(p[1] for p in c)+1) >= 7
                   and min(max(p[0] for p in c)-min(p[0] for p in c)+1,
                           max(p[1] for p in c)-min(p[1] for p in c)+1) <= 2
                   for c in r[6]) for r in records)
    assert any(any(max(max(p[0] for p in c)-min(p[0] for p in c)+1,
                       max(p[1] for p in c)-min(p[1] for p in c)+1) >= 7
                   and min(max(p[0] for p in c)-min(p[0] for p in c)+1,
                           max(p[1] for p in c)-min(p[1] for p in c)+1) >= 3
                   for c in r[6]) for r in records)
    assert any(r[1][0] == r[1][1] for r in records)
    assert any(max(r[1]) / min(r[1]) >= 2 for r in records)
    assert any(any(row in (0, pair["input"].shape[0] - 1) or
                   col in (0, pair["input"].shape[1] - 1)
                   for component in _components(pair["input"] != _infer_background(pair["input"]))
                   if len(component) == 1 for row, col in component)
               for puzzle in puzzles for section in ("train", "test") for pair in puzzle[section])
    assert any(any(row in (0, pair["input"].shape[0] - 1) or
                   col in (0, pair["input"].shape[1] - 1)
                   for component in _components(pair["input"] != _infer_background(pair["input"]))
                   if len(component) >= 2 for row, col in component)
               for puzzle in puzzles for section in ("train", "test") for pair in puzzle[section])
    assert any(_enclosed_singleton(pair["input"] != _infer_background(pair["input"]),
                                   _components(pair["input"] != _infer_background(pair["input"])))
               for puzzle in puzzles for section in ("train", "test") for pair in puzzle[section])
    backgrounds_changed = signals_changed = fixed_palettes = role_swapped = 0
    familiar_test = novel_test = fixed_dimensions = mixed_dimensions = 0
    mixed_orientation = varied_noise = test_not_always_denser = 0
    for puzzle in puzzles:
        all_pairs = puzzle["train"] + puzzle["test"]
        palettes = []
        dimensions = []
        noise = []
        density = []
        for pair in all_pairs:
            grid = pair["input"]
            background = _infer_background(grid)
            signal = next(int(v) for v in np.unique(grid) if v != background)
            palettes.append((background, signal))
            dimensions.append(grid.shape)
            mask = grid != background
            noise.append(sum(len(c) == 1 for c in _components(mask)))
            density.append(float(mask.mean()))
        backgrounds_changed += len({b for b, _ in palettes}) > 1
        signals_changed += len({s for _, s in palettes}) > 1
        fixed_palettes += len(set(palettes)) == 1
        role_swapped += any(a == (b[1], b[0]) for i, a in enumerate(palettes)
                            for b in palettes[i + 1:])
        training_signals = {s for _, s in palettes[:len(puzzle["train"])]}
        test_signals = {s for _, s in palettes[len(puzzle["train"]):]}
        familiar_test += bool(training_signals & test_signals)
        novel_test += bool(test_signals - training_signals)
        fixed_dimensions += len(set(dimensions)) == 1
        mixed_dimensions += len(set(dimensions)) > 1
        mixed_orientation += len({np.sign(w - h) for h, w in dimensions}) > 1
        varied_noise += len(set(noise)) > 1
        test_not_always_denser += min(density[len(puzzle["train"]):]) <= max(density[:len(puzzle["train"])])
    assert backgrounds_changed >= 10 and signals_changed >= 10
    assert fixed_palettes >= 3 and role_swapped >= 2
    assert familiar_test >= 20 and novel_test >= 5
    assert fixed_dimensions >= 1 and mixed_dimensions >= 25
    assert mixed_orientation >= 20 and varied_noise >= 35
    assert test_not_always_denser >= 25
