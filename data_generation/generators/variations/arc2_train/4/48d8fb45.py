import numpy as np


_SOURCE_PAIR_SIGNATURES = frozenset({
    (
        (
            "0000000000",
            "0000000500",
            "0000004400",
            "0040000040",
            "0404000400",
            "0044000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        ("440", "004", "040"),
    ),
    (
        (
            "0000000000",
            "0000000011",
            "0005000110",
            "0001000010",
            "0011100000",
            "0001100000",
            "0000000000",
            "0000001100",
            "0000011100",
            "0000001100",
        ),
        ("010", "111", "011"),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0022000000",
            "0202000000",
            "0020000500",
            "0000000220",
            "0000002200",
            "0000000200",
            "0000000000",
            "0000000000",
        ),
        ("022", "220", "020"),
    ),
    (
        (
            "0000005000",
            "0000003000",
            "0000033000",
            "0000003300",
            "0030000000",
            "0330000000",
            "0030003000",
            "0000033300",
            "0000003300",
            "0000000000",
        ),
        ("030", "330", "033"),
    ),
})


def _rows_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_signature(pair: dict) -> tuple:
    return (
        _array_signature(pair["input"]),
        _array_signature(pair["output"]),
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    signature = (
        _rows_signature(input_grid),
        _rows_signature(output_grid),
    )
    return signature in _SOURCE_PAIR_SIGNATURES


def _validated_optional_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not low <= result <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validated_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validated_optional_count(
        "num_test", num_test, 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return train_count, test_count

    if train_count is None and test_count is None:
        # This deterministic schedule guarantees broad count diversity
        # over consecutive seeds.
        test_count = 1 + seed % 4
        train_count = (
            test_count
            + 1
            + (seed // 4) % (10 - test_count)
        )
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    else:
        maximum_test = min(4, train_count - 1)
        test_count = int(
            rng.integers(1, maximum_test + 1)
        )

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError(
            "the requested counts cannot form a legal puzzle"
        )

    return train_count, test_count


def _count_components(mask: np.ndarray) -> int:
    remaining = set(zip(*np.nonzero(mask)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, alt) for alt in
               [np.rot90(mask, k) for k in (1, 2, 3)] +
               [np.rot90(np.fliplr(mask), k) for k in range(4)])


def _motif(rng: np.random.Generator, informative: bool) -> np.ndarray:
    # Every occupancy from zero through nine is a legitimate literal crop.
    weights = np.array([3, 6, 10, 12, 14, 16, 14, 12, 9, 4], dtype=float)
    weights /= weights.sum()
    for _ in range(100):
        if not informative and rng.random() < .07:
            mask = np.zeros((3, 3), dtype=bool)
            index = int(rng.integers(3))
            if rng.random() < .5:
                mask[index, :] = True
            else:
                mask[:, index] = True
            return mask
        size = int(rng.choice(10, p=weights))
        if informative and not 3 <= size <= 7:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        if size:
            mask.ravel()[rng.choice(9, size=size, replace=False)] = True
        if informative and not _asymmetric(mask):
            continue
        return mask
    raise RuntimeError("motif sampling failed")


def _size(rng: np.random.Generator) -> tuple[int, int]:
    kind = int(rng.choice(6, p=[.21, .10, .27, .16, .16, .10]))
    if kind == 0:
        return int(rng.integers(4, 10)), int(rng.integers(4, 10))
    if kind == 1:
        return 10, 10
    if kind == 2:
        return int(rng.integers(9, 21)), int(rng.integers(9, 21))
    if kind == 3:
        return int(rng.integers(17, 31)), int(rng.integers(4, 11))
    if kind == 4:
        return int(rng.integers(4, 11)), int(rng.integers(17, 31))
    if rng.random() < .2:
        return 30, 30
    return int(rng.integers(20, 31)), int(rng.integers(20, 31))


def _marker_position(rng: np.random.Generator, height: int,
                     width: int, band: str) -> tuple[int, int]:
    bottom = height - 4
    if band == "top":
        row = 0
    elif band == "bottom":
        row = bottom
    elif band == "interior" and bottom >= 2:
        row = int(rng.integers(1, bottom))
    else:
        row = int(rng.integers(0, bottom + 1))
    side = int(rng.choice(3, p=[.22, .22, .56]))
    if side == 0:
        col = 1
    elif side == 1:
        col = width - 2
    else:
        col = int(rng.integers(1, width - 1))
    return row, col


def _band(rng: np.random.Generator, regime: str, split: str) -> str:
    if regime == "source_shift":
        return "interior" if split == "train" else "top"
    if regime == "reverse_shift":
        return "top" if split == "train" else "interior"
    return str(rng.choice(("top", "bottom", "interior", "any"),
                          p=(.24, .24, .32, .20)))


def _polyomino(rng: np.random.Generator, size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        edge = sorted({(r + dr, c + dc) for r, c in cells
                       for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                       if (r + dr, c + dc) not in cells})
        cells.add(edge[int(rng.integers(len(edge)))])
    return cells


def _shape(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    kind = int(rng.integers(7))
    if kind == 0:
        cells = {(0, 0)}
    elif kind == 1:
        cells = _polyomino(rng, int(rng.integers(2, 12)))
    elif kind == 2:
        length = int(rng.integers(2, 9))
        cells = {(i, 0) for i in range(length)} if rng.random() < .5 else {(0, i) for i in range(length)}
    elif kind == 3:
        length = int(rng.integers(2, 7))
        cells = {(i, i) for i in range(length)}
        if rng.random() < .5:
            cells |= {(i, i + 1) for i in range(0, length, 2)}
    elif kind == 4:
        height, width = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        cells = {(r, c) for r in range(height) for c in range(width)
                 if rng.random() < .65}
        if not cells:
            cells = {(0, 0)}
    elif kind == 5:
        cells = {(r, c) for r in range(3) for c in range(3)
                 if r in (0, 2) or c in (0, 2)}
        if rng.random() < .5:
            cells.remove((0, int(rng.integers(3))))
    else:
        size = int(rng.integers(1, 10))
        chosen = rng.choice(9, size=size, replace=False)
        cells = {(int(i) // 3, int(i) % 3) for i in chosen}
    low_r = min(r for r, _ in cells)
    low_c = min(c for _, c in cells)
    return tuple(sorted((r - low_r, c - low_c) for r, c in cells))


def _put_shape(rng: np.random.Generator, occupied: set[tuple[int, int]],
               forbidden: set[tuple[int, int]], height: int, width: int,
               shape: tuple[tuple[int, int], ...]) -> bool:
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    options = []
    for top in range(height - sh + 1):
        for left in range(width - sw + 1):
            placed = tuple((top + r, left + c) for r, c in shape)
            if all(cell not in forbidden and cell not in occupied for cell in placed):
                options.append(placed)
    if not options:
        return False
    occupied.update(options[int(rng.integers(len(options)))])
    return True


def _continuation(rng: np.random.Generator, mask: np.ndarray,
                  top: int, left: int, height: int, width: int,
                  marker: tuple[int, int]) -> tuple[int, int] | None:
    choices = []
    for rr, cc in zip(*np.nonzero(mask)):
        r, c = top + int(rr), left + int(cc)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) != marker:
                if not (top <= nr < top + 3 and left <= nc < left + 3):
                    choices.append((nr, nc))
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _palette(rng: np.random.Generator, train_count: int) -> dict:
    order = [int(v) for v in rng.permutation(10)]
    style = str(rng.choice(("stable", "varied", "multi"), p=(.25, .43, .32)))
    bg_count = 1 if style == "stable" else int(rng.integers(2, min(3, train_count) + 1))
    backgrounds = tuple(order[1:1 + bg_count])
    contents = tuple(order[1 + bg_count:])
    return {"marker": order[0], "backgrounds": backgrounds,
            "contents": contents, "style": style,
            "regime": str(rng.choice(("source_shift", "reverse_shift", "mixed", "similar"))),
            "boundary_demo": bool(rng.random() < .55)}


def _pair_colors(rng: np.random.Generator, params: dict,
                 split: str, index: int, train_colors: tuple[int, ...]) -> tuple[int, int, int | None]:
    backgrounds = params["backgrounds"]
    background = backgrounds[index % len(backgrounds)] if split == "train" else int(rng.choice(backgrounds))
    available = params["contents"]
    if params["style"] == "stable":
        main = available[0]
        accent = None
    elif split == "train":
        main = available[index % min(3, len(available))]
        accent = available[(index + 1) % min(3, len(available))] if params["style"] == "multi" and (index == 0 or rng.random() < .4) else None
    else:
        # Familiar and unseen test colors are both allowed. Neither changes the crop rule.
        if rng.random() < .55:
            main = int(rng.choice(train_colors))
        else:
            unseen = [v for v in available if v not in train_colors]
            main = int(rng.choice(unseen if unseen else train_colors))
        accent = None
        if params["style"] == "multi" and rng.random() < .5:
            choices = [v for v in train_colors if v != main]
            if choices:
                accent = int(rng.choice(choices))
    return background, main, accent


def _make_pair(rng: np.random.Generator, params: dict, split: str,
               index: int, train_colors: tuple[int, ...],
               used_motifs: set[tuple[int, ...]]) -> tuple[dict, tuple[int, ...]]:
    background, main, accent = _pair_colors(rng, params, split, index, train_colors)
    informative = split == "train" and index == 0
    want_continuation = params["boundary_demo"] and (
        (split == "train" and index == 0) or (split == "test" and rng.random() < .6))
    for _ in range(160):
        height, width = _size(rng)
        if ((params["regime"] == "source_shift" and split == "train")
                or (params["regime"] == "reverse_shift" and split == "test")) and height < 6:
            continue
        marker = _marker_position(rng, height, width, _band(rng, params["regime"], split))
        top, left = marker[0] + 1, marker[1] - 1
        mask = _motif(rng, informative)
        motif_key = tuple(int(v) for v in mask.flat)
        if motif_key in used_motifs:
            continue
        if accent is not None and int(mask.sum()) < 4:
            continue
        crop = {(r, c) for r in range(top, top + 3) for c in range(left, left + 3)}
        occupied = {(top + int(r), left + int(c)) for r, c in zip(*np.nonzero(mask))}
        outside: set[tuple[int, int]] = set()
        extension = _continuation(rng, mask, top, left, height, width, marker) if want_continuation else None
        if want_continuation and extension is None:
            continue
        if extension is not None:
            outside.add(extension)
        free = height * width - 10
        max_arrangements = min(10, free // 18)
        arrangement_count = int(rng.integers(0, max_arrangements + 1))
        if informative and arrangement_count == 0 and not outside and free >= 12:
            arrangement_count = 1
        if split == "test" and height * width >= 100 and rng.random() < .35:
            arrangement_count = max(arrangement_count, 2)
        for _ in range(arrangement_count):
            for _ in range(16):
                if _put_shape(rng, outside, crop | {marker}, height, width, _shape(rng)):
                    break
            else:
                # A single available cell is a deterministic placement fallback.
                candidates = [(r, c) for r in range(height) for c in range(width)
                              if (r, c) not in crop and (r, c) != marker and (r, c) not in outside]
                if candidates:
                    outside.add(candidates[int(rng.integers(len(candidates)))])
        total = len(occupied) + len(outside)
        if total == 1:
            candidates = [(r, c) for r in range(height) for c in range(width)
                          if (r, c) not in crop and (r, c) != marker and (r, c) not in outside]
            if not candidates:
                continue
            outside.add(candidates[int(rng.integers(len(candidates)))])
        grid = np.full((height, width), background, dtype=np.int8)
        grid[marker] = params["marker"]
        for cell in occupied | outside:
            grid[cell] = main
        if accent is not None:
            target_cells = sorted(occupied)
            chosen = rng.choice(len(target_cells), size=2, replace=False)
            for item in chosen:
                grid[target_cells[int(item)]] = accent
            # Main already has at least two cells in a four-cell target.
        if split == "test" and not params["boundary_demo"]:
            adjacent = any(any((rr + dr, cc + dc) in crop
                               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                           for rr, cc in outside)
            if adjacent:
                continue
        values, counts = np.unique(grid, return_counts=True)
        color_counts = {int(v): int(n) for v, n in zip(values, counts)}
        if color_counts[background] <= max(n for v, n in color_counts.items() if v != background):
            continue
        if any(n == 1 for v, n in color_counts.items() if v not in (background, params["marker"])):
            continue
        output = grid[top:top + 3, left:left + 3].copy()
        return {"input": grid, "output": output}, motif_key
    raise RuntimeError("could not construct a valid pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    params = _palette(rng, train_count)
    puzzle = {"train": [], "test": []}
    used_inputs: set[tuple] = set()
    used_motifs: set[tuple[int, ...]] = set()
    train_colors = tuple(params["contents"][:min(3, train_count)])
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(40):
                pair, motif_key = _make_pair(rng, params, split, index, train_colors, used_motifs)
                key = _array_signature(pair["input"])
                if key in used_inputs or _is_source_pair(pair["input"], pair["output"]):
                    continue
                used_inputs.add(key)
                used_motifs.add(motif_key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair: dict, marker: int | None = None) -> dict:
    inp, out = pair["input"], pair["output"]
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) >= 2
    assert int(np.count_nonzero(counts == counts.max())) == 1
    bg = int(values[int(np.argmax(counts))])
    singleton = [int(v) for v, n in zip(values, counts) if int(v) != bg and int(n) == 1]
    assert len(singleton) == 1, "marker must be the only singleton color"
    assert marker is None or singleton[0] == marker
    marker = singleton[0]
    location = np.argwhere(inp == marker)
    assert location.shape == (1, 2)
    row, col = map(int, location[0])
    assert 0 <= row <= inp.shape[0] - 4
    assert 1 <= col <= inp.shape[1] - 2
    assert out.shape == (3, 3)
    assert np.array_equal(out, inp[row + 1:row + 4, col - 1:col + 2])
    assert marker not in out
    return {"background": bg, "marker": marker, "row": row, "col": col,
            "occupancy": int(np.count_nonzero(out != bg)),
            "components": _count_components(out != bg),
            "asymmetric": _asymmetric(out != bg),
            "colors": {int(v) for v in values if int(v) not in (bg, marker)}}


def _boundary_features(pair: dict, info: dict) -> tuple[bool, bool, np.ndarray]:
    grid = pair["input"]
    row, col = info["row"], info["col"]
    crop = {(r, c) for r in range(row + 1, row + 4)
            for c in range(col - 1, col + 2)}
    outside = grid != info["background"]
    outside[row, col] = False
    outside[row + 1:row + 4, col - 1:col + 2] = False
    target = {(r, c) for r, c in crop if grid[r, c] != info["background"]}
    near = False
    extension = False
    for r, c in zip(*np.nonzero(outside)):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (r + dr, c + dc)
            near |= neighbor in crop
            extension |= neighbor in target
    return near, extension, outside


def _component_sizes(mask: np.ndarray) -> list[int]:
    pending = set(zip(*np.nonzero(mask)))
    sizes = []
    while pending:
        stack = [pending.pop()]
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for neighbor in ((r - 1, c), (r + 1, c),
                             (r, c - 1), (r, c + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def _has_hole(mask: np.ndarray) -> bool:
    height, width = mask.shape
    exterior = {(r, c) for r in range(height) for c in (0, width - 1)
                if not mask[r, c]}
    exterior |= {(r, c) for c in range(width) for r in (0, height - 1)
                 if not mask[r, c]}
    stack = list(exterior)
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c),
                       (r, c - 1), (r, c + 1)):
            if (0 <= nr < height and 0 <= nc < width
                    and not mask[nr, nc] and (nr, nc) not in exterior):
                exterior.add((nr, nc))
                stack.append((nr, nc))
    return any(not mask[r, c] and (r, c) not in exterior
               for r in range(height) for c in range(width))


def _inferred_offsets(train: list[dict], marker: int) -> list[tuple[int, int]]:
    candidates = []
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            valid = True
            for pair in train:
                grid = pair["input"]
                r, c = map(int, np.argwhere(grid == marker)[0])
                top, left = r + dr, c + dc
                if (top < 0 or left < 0 or top + 3 > grid.shape[0]
                        or left + 3 > grid.shape[1]
                        or not np.array_equal(grid[top:top + 3, left:left + 3], pair["output"])):
                    valid = False
                    break
            if valid:
                candidates.append((dr, dc))
    return candidates


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    infos = {"train": [], "test": []}
    marker = None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            key = _pair_signature(pair)
            assert key[0] not in inputs and key not in pairs
            assert not _is_source_pair(pair["input"], pair["output"])
            inputs.add(key[0])
            pairs.add(key)
            info = _inspect_pair(pair, marker)
            marker = info["marker"]
            infos[split].append(info)
    # At least one literal, oriented motif in training establishes the crop.
    assert any(3 <= x["occupancy"] <= 7 and x["asymmetric"] for x in infos["train"])
    assert _inferred_offsets(puzzle["train"], marker) == [(1, -1)]
    train_bgs = {x["background"] for x in infos["train"]}
    test_bgs = {x["background"] for x in infos["test"]}
    assert test_bgs <= train_bgs
    if any(len(x["colors"]) >= 2 and x["occupancy"] >= 4 for x in infos["test"]):
        assert any(len(x["colors"]) >= 2 and x["occupancy"] >= 4 for x in infos["train"])
    if any(_boundary_features(pair, info)[0]
           for pair, info in zip(puzzle["test"], infos["test"])):
        assert any(_boundary_features(pair, info)[0]
                   for pair, info in zip(puzzle["train"], infos["train"]))


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, motifs, placements, occupancies = set(), set(), set(), set()
    connected, disconnected = 0, 0
    symmetric, multi, near, empty, solid = 0, 0, 0, 0, 0
    line, no_clutter, many_components, singleton, large_outside = 0, 0, 0, 0, 0
    holes, diagonal, extensions, train_extensions, combined = 0, 0, 0, 0, 0
    thirty, first_test_connected, first_test_disconnected = 0, 0, 0
    first_train_connected, first_train_disconnected = 0, 0
    top_train, top_test, bottom, side, small, large, tall, wide = 0, 0, 0, 0, 0, 0, 0, 0
    backgrounds, markers, contents = set(), set(), set()
    familiar_tests, unseen_tests, varied_bg_puzzles, stable_palette_puzzles = 0, 0, 0, 0
    source_shift, reverse_shift, mixed_train, mixed_test, size_changes = 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {s: [_inspect_pair(p) for p in puzzle[s]] for s in ("train", "test")}
        size_changes += len({pair["input"].shape for s in ("train", "test") for pair in puzzle[s]}) > 1
        first_train_connected += infos["train"][0]["components"] == 1
        first_train_disconnected += infos["train"][0]["components"] > 1
        first_test_connected += infos["test"][0]["components"] == 1
        first_test_disconnected += infos["test"][0]["components"] > 1
        train_content = set().union(*(x["colors"] for x in infos["train"]))
        all_content = set().union(*(x["colors"] for s in infos for x in infos[s]))
        if len({x["background"] for x in infos["train"] + infos["test"]}) > 1:
            varied_bg_puzzles += 1
        if len(all_content) == 1 and len({x["background"] for x in infos["train"] + infos["test"]}) == 1:
            stable_palette_puzzles += 1
        if all(x["row"] > 0 for x in infos["train"]) and all(x["row"] == 0 for x in infos["test"]):
            source_shift += 1
        if all(x["row"] == 0 for x in infos["train"]) and all(x["row"] > 0 for x in infos["test"]):
            reverse_shift += 1
        mixed_train += any(x["row"] == 0 for x in infos["train"]) and any(x["row"] > 0 for x in infos["train"])
        mixed_test += any(x["row"] == 0 for x in infos["test"]) and any(x["row"] > 0 for x in infos["test"])
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                inp, out = pair["input"], pair["output"]
                h, w = inp.shape
                r, c = info["row"], info["col"]
                assert np.array_equal(out, inp[r + 1:r + 4, c - 1:c + 2])
                shapes.add((h, w))
                placements.add((h, w, r, c))
                motifs.add(tuple(int(v != info["background"]) for v in out.flat))
                occupancies.add(info["occupancy"])
                backgrounds.add(info["background"])
                markers.add(info["marker"])
                contents.update(info["colors"])
                if info["components"] == 1:
                    connected += 1
                if info["components"] > 1:
                    disconnected += 1
                symmetric += not info["asymmetric"]
                multi += len(info["colors"]) >= 2 and len(set(out.flat)) >= 3
                empty += info["occupancy"] == 0
                solid += info["occupancy"] == 9
                motif_mask = out != info["background"]
                line += (info["occupancy"] == 3 and
                         (any(motif_mask[j].all() for j in range(3)) or
                          any(motif_mask[:, j].all() for j in range(3))))
                top_train += split == "train" and r == 0
                top_test += split == "test" and r == 0
                bottom += r == h - 4
                side += c in (1, w - 2)
                small += h < 10 or w < 10
                large += h >= 20 and w >= 20
                tall += h >= 17 and w <= 10
                wide += w >= 17 and h <= 10
                thirty += (h, w) == (30, 30)
                is_near, extension, outside = _boundary_features(pair, info)
                near += is_near
                extensions += extension
                train_extensions += split == "train" and extension
                no_clutter += not bool(outside.any())
                sizes = _component_sizes(outside)
                many_components += len(sizes) > 3
                singleton += 1 in sizes
                large_outside += any(size > 6 for size in sizes)
                holes += _has_hole(outside)
                outside_cells = set(zip(*np.nonzero(outside)))
                diagonal += any((rr + 1, cc + 1) in outside_cells or
                                (rr + 1, cc - 1) in outside_cells
                                for rr, cc in outside_cells)
                combined += extension and len(set(out.flat)) >= 3 and (h < 10 or w < 10)
                if split == "test":
                    familiar_tests += bool(info["colors"] & train_content)
                    unseen_tests += bool(info["colors"] - train_content)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 80 and len(placements) >= 120 and len(motifs) >= 100
    assert len(occupancies) >= 9
    assert connected > 0 and disconnected > 0 and symmetric > 0
    assert empty > 0 and solid > 0 and multi > 0 and near > 0
    assert line > 0 and no_clutter > 0 and many_components > 0
    assert singleton > 0 and large_outside > 0 and holes > 0 and diagonal > 0
    assert extensions > 0 and train_extensions > 0 and combined > 0
    assert thirty > 0
    assert first_train_connected > 0 and first_train_disconnected > 0
    assert first_test_connected > 0 and first_test_disconnected > 0
    assert top_train > 0 and top_test > 0 and bottom > 0 and side > 0
    assert small > 0 and large > 0 and tall > 0 and wide > 0
    assert varied_bg_puzzles > 0 and stable_palette_puzzles > 0
    assert familiar_tests > 0 and unseen_tests > 0
    assert source_shift > 0 and reverse_shift > 0
    assert mixed_train > 0 and mixed_test > 0 and size_changes >= 40
    assert len(backgrounds) >= 7 and len(markers) >= 7 and len(contents) >= 7
    assert _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(generate_puzzle(314159), generate_puzzle(314160))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        p = generate_puzzle(5000 + i, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for i, kwargs in enumerate(({"num_test": 1}, {"num_test": 4},
                                {"num_train": 2}, {"num_train": 10})):
        p = generate_puzzle(6000 + i, **kwargs)
        for name, val in kwargs.items():
            assert len(p["train" if name == "num_train" else "test"]) == val
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(987654, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
