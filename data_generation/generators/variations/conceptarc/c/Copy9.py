"""Generate seed-centered motif-copy ARC puzzles."""

import numpy as np


# Exact source inputs, including dimensions. Rejecting an input is stronger than
# rejecting only its input/output pair and does not exclude source dimensions.
_SOURCE_INPUTS = {
    (8, 9, "000000000011100000011100000011100000000000000000000100000000000000000000"),
    (12, 9, "000000000000200000000000000000000000020000303222000030020000303000000000000000000000003000000000000000000000"),
    (12, 11, "000000000000888000006008080000000088800000000000000000000000080000000000000000000000000000000666000000000600000000066600000000000000"),
    (10, 9, "000000000090000990000009990000000900000000000000000000000000000666000000066000060006000000"),
    (10, 10, "0000000000033300444003030040400333004440000000000000000000000000000000000030004000000000000000000000"),
    (12, 10, "002000000000200000002222200000002000000000200000000000000555000005005000000005050020000000000000000000000000000000000000"),
}

_FAMILIAR = {
    3: (
        ("111", "111", "111"),
        ("111", "101", "111"),
        ("010", "111", "010"),
        ("101", "010", "101"),
        ("111", "010", "111"),
        ("111", "011", "001"),
        ("111", "010", "101"),
        ("110", "011", "001"),
    ),
    5: (
        ("00100", "00100", "11111", "00100", "00100"),
        ("11111", "10001", "10001", "10001", "11111"),
        ("10001", "01010", "00100", "01010", "10001"),
        ("01110", "11111", "11111", "11111", "01110"),
        ("11111", "00100", "00100", "00100", "11111"),
    ),
}


def _from_rows(rows):
    return np.array([[int(v) for v in row] for row in rows], dtype=np.int8)


def _components(points):
    """Eight-connected components of the given coordinates."""
    unseen = set(points)
    parts = []
    while unseen:
        first = unseen.pop()
        part = {first}
        stack = [first]
        while stack:
            row, col = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (row + dr, col + dc)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        part.add(neighbor)
                        stack.append(neighbor)
        parts.append(part)
    return parts


def _valid_mask(mask):
    height, width = mask.shape
    points = {tuple(map(int, p)) for p in np.argwhere(mask)}
    return (
        height in (3, 5)
        and width == height
        and 4 <= len(points) < height * width + 1
        and any(row == 0 for row, _ in points)
        and any(row == height - 1 for row, _ in points)
        and any(col == 0 for _, col in points)
        and any(col == width - 1 for _, col in points)
        and len(_components(points)) == 1
    )


def _mask(rng, size, center=None, asymmetric=None):
    """Mix recognizable examples with many connected, bounded variations."""
    bases = _FAMILIAR[size]
    for _ in range(160):
        mode = int(rng.integers(3))
        if mode == 0:
            candidate = _from_rows(bases[int(rng.integers(len(bases)))])
        elif mode == 1:
            density = float(rng.uniform(0.37, 0.78))
            candidate = (rng.random((size, size)) < density).astype(np.int8)
        else:
            candidate = _from_rows(bases[int(rng.integers(len(bases)))])
            for _ in range(int(rng.integers(1, size + 3))):
                row = int(rng.integers(size))
                col = int(rng.integers(size))
                candidate[row, col] ^= 1
        if not _valid_mask(candidate):
            continue
        if center is not None and bool(candidate[size // 2, size // 2]) != center:
            continue
        if asymmetric is not None and (
            not np.array_equal(candidate, candidate[::-1, ::-1])
        ) != asymmetric:
            continue
        return candidate
    # The exact shapes are a bounded fallback for unusual requested conditions.
    for rows in bases:
        candidate = _from_rows(rows)
        if (center is None or bool(candidate[size // 2, size // 2]) == center) and (
            asymmetric is None or (
                not np.array_equal(candidate, candidate[::-1, ::-1])
            ) == asymmetric
        ):
            return candidate
    raise AssertionError("No motif satisfies the requested geometry")


def _relative_points(mask):
    radius = mask.shape[0] // 2
    return {(int(r) - radius, int(c) - radius) for r, c in np.argwhere(mask)}


def _shift_options(mask):
    points = _relative_points(mask)
    options = {"touch": [], "near": [], "apart": []}
    radius = mask.shape[0] // 2
    for dr in range(-9, 10):
        for dc in range(-9, 10):
            if (dr, dc) == (0, 0):
                continue
            if min(max(abs(dr - r), abs(dc - c)) for r, c in points) <= 1:
                continue  # The input marker must be an isolated same-color component.
            moved = {(r + dr, c + dc) for r, c in points}
            if moved & points:
                continue
            gap = min(
                abs(r - s) + abs(c - t)
                for r, c in points for s, t in moved
            )
            category = "touch" if gap == 1 else "near" if gap == 2 else "apart"
            if category == "apart" and max(abs(dr), abs(dc)) > mask.shape[0] + 5:
                continue
            options[category].append((dr, dc))
    assert options["touch"] and options["near"] and options["apart"]
    return options


def _canvas(rng, minimum):
    band = int(rng.choice(3, p=(0.48, 0.34, 0.18)))
    ranges = ((8, 12), (12, 16), (16, 22))
    low, high = ranges[band]
    low = max(low, minimum)
    high = max(high, low)
    return int(rng.integers(low, high + 1))


def _layout(rng, background, specs):
    """Place templates and isolated centers without clipped/conflicting copies."""
    count = len(specs)
    minimum = 8 if count == 1 else 9 if count == 2 else 12
    if any(mask.shape[0] == 5 for _, mask, _ in specs):
        minimum += 1 if count < 3 else 2
    options = [_shift_options(mask) for _, mask, _ in specs]
    for attempt in range(240):
        # More room after difficult compact attempts, while retaining both bands.
        extra = 0 if attempt < 120 else 2
        height = _canvas(rng, minimum + extra)
        width = _canvas(rng, minimum + extra)
        occupied = set()
        placed = []
        for (color, mask, gap), shifts in zip(specs, options):
            radius = mask.shape[0] // 2
            points = _relative_points(mask)
            choices = shifts[gap]
            chosen = None
            for _ in range(180):
                center = (
                    int(rng.integers(radius, height - radius)),
                    int(rng.integers(radius, width - radius)),
                )
                dr, dc = choices[int(rng.integers(len(choices)))]
                seed = (center[0] + dr, center[1] + dc)
                if not (
                    radius <= seed[0] < height - radius
                    and radius <= seed[1] < width - radius
                ):
                    continue
                original = {(center[0] + r, center[1] + c) for r, c in points}
                copied = {(seed[0] + r, seed[1] + c) for r, c in points}
                footprint = original | copied | {seed}
                if footprint & occupied:
                    continue
                chosen = (color, original, copied, seed, footprint)
                break
            if chosen is None:
                break
            placed.append(chosen)
            occupied |= chosen[4]
        if len(placed) != count or len(occupied) * 2 >= height * width:
            continue
        inp = np.full((height, width), background, dtype=np.int8)
        out = inp.copy()
        for (color, original, copied, seed, _), (_, mask, _) in zip(placed, specs):
            for row, col in original | {seed}:
                inp[row, col] = color
            for row, col in original | copied:
                out[row, col] = color
            if mask[mask.shape[0] // 2, mask.shape[1] // 2]:
                out[seed] = color
        return inp, out
    raise RuntimeError("Could not place distinct motifs")


def _source_key(grid):
    return (grid.shape[0], grid.shape[1], "".join(str(int(v)) for v in grid.flat))


def _inspect_pair(inp, out):
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == out.ndim == 2 and inp.shape == out.shape
    assert all(1 <= size <= 30 for size in inp.shape)
    assert np.issubdtype(inp.dtype, np.integer)
    assert np.issubdtype(out.dtype, np.integer)
    assert np.all((0 <= inp) & (inp <= 9))
    assert np.all((0 <= out) & (out <= 9))
    assert _source_key(inp) not in _SOURCE_INPUTS
    colors, amounts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(amounts))])
    assert np.count_nonzero(out == background) * 2 > out.size
    motif_colors = tuple(int(c) for c in colors if c != background)
    assert 1 <= len(motif_colors) <= 3
    assert {int(c) for c in np.unique(out) if c != background} == set(motif_colors)
    expected = inp.copy()
    details = []
    for color in motif_colors:
        points = {tuple(map(int, p)) for p in np.argwhere(inp == color)}
        parts = _components(points)
        singles = [part for part in parts if len(part) == 1]
        motifs = [part for part in parts if len(part) > 1]
        assert len(parts) == 2 and len(singles) == len(motifs) == 1
        seed = next(iter(singles[0]))
        motif = motifs[0]
        top = min(r for r, _ in motif)
        bottom = max(r for r, _ in motif)
        left = min(c for _, c in motif)
        right = max(c for _, c in motif)
        size = bottom - top + 1
        assert size == right - left + 1 and size in (3, 5)
        center = ((top + bottom) // 2, (left + right) // 2)
        mask = np.zeros((size, size), dtype=np.int8)
        for row, col in motif:
            mask[row - top, col - left] = 1
        assert _valid_mask(mask)
        assert min(max(abs(seed[0] - r), abs(seed[1] - c)) for r, c in motif) > 1
        copy = {(seed[0] + r - center[0], seed[1] + c - center[1]) for r, c in motif}
        assert copy.isdisjoint(motif)
        assert all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] for r, c in copy)
        # A blank template center removes only its own marker; other blank
        # template offsets leave existing colors untouched.
        if not mask[size // 2, size // 2]:
            expected[seed] = background
        for row, col in copy:
            assert expected[row, col] in (background, color)
            expected[row, col] = color
        gap = min(abs(r - s) + abs(c - t) for r, c in motif for s, t in copy)
        details.append({
            "color": color,
            "size": size,
            "center_filled": bool(mask[size // 2, size // 2]),
            "asymmetric": not np.array_equal(mask, mask[::-1, ::-1]),
            "gap": gap,
            "mask": mask.tobytes(),
            "center": center,
            "seed": seed,
        })
    assert np.array_equal(expected, out)
    assert not np.array_equal(inp, out)
    return background, details


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen_inputs = set()
    seen_pairs = set()
    train_details = []
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            background, details = _inspect_pair(inp, out)
            assert background not in {d["color"] for d in details}
            input_key = (inp.shape, tuple(int(v) for v in inp.flat))
            pair_key = (input_key, out.shape, tuple(int(v) for v in out.flat))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if regime == "train":
                train_details.append(details)
    # All conventions are visible: marker and template share a color; copy
    # preserves offsets, orientation, original motifs, and the canvas.
    flat = [item for pair in train_details for item in pair]
    assert any(len(pair) == 1 for pair in train_details)
    assert any(len(pair) >= 2 for pair in train_details)
    assert any(item["size"] == 3 for item in flat)
    assert any(item["center_filled"] for item in flat)
    assert any(not item["center_filled"] for item in flat)
    assert any(item["asymmetric"] for item in flat)


def _counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (type(value) is not int or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        return legal[int(rng.integers(len(legal)))]
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    # The only shared convention is the one shown in every training output:
    # copy colored offsets to the same-color seed, replacing that marker
    # with the template's center value.
    train_counts = [int(rng.choice((1, 2, 3), p=(0.42, 0.48, 0.10))) for _ in range(num_train)]
    single_index, multi_index = rng.choice(num_train, size=2, replace=False)
    train_counts[int(single_index)] = 1
    train_counts[int(multi_index)] = int(rng.choice((2, 3), p=(0.82, 0.18)))
    total_specs = sum(train_counts)
    empty_slot, filled_slot, asym_slot = rng.choice(total_specs, size=3, replace=False)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    slot = 0
    for regime, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(24):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                motif_count = train_counts[index] if regime == "train" else int(
                    pair_rng.choice((1, 2, 3), p=(0.22, 0.65, 0.13))
                )
                background = int(pair_rng.choice(10, p=(0.48,) + (0.52 / 9,) * 9))
                available = [v for v in range(10) if v != background]
                colors = [int(v) for v in pair_rng.choice(available, size=motif_count, replace=False)]
                coverage_slots = (empty_slot, filled_slot, asym_slot)
                same_silhouette = (
                    motif_count > 1
                    and pair_rng.random() < 0.17
                    and (regime == "test" or all(
                        slot + j not in coverage_slots for j in range(motif_count)
                    ))
                )
                specs = []
                for j in range(motif_count):
                    absolute_slot = slot + j if regime == "train" else -1
                    size = 3 if absolute_slot in (empty_slot, filled_slot, asym_slot) else (
                        5 if pair_rng.random() < (0.22 if regime == "train" else 0.46) else 3
                    )
                    if same_silhouette and j > 0:
                        mask = specs[0][1].copy()
                    else:
                        mask = _mask(
                            pair_rng, size,
                            center=False if absolute_slot == empty_slot else True if absolute_slot == filled_slot else None,
                            asymmetric=True if absolute_slot == asym_slot else None,
                        )
                    gap = str(pair_rng.choice(("touch", "near", "apart"), p=(0.27, 0.23, 0.50)))
                    specs.append((colors[j], mask, gap))
                try:
                    inp, out = _layout(pair_rng, background, specs)
                except RuntimeError:
                    continue
                key = (inp.shape, inp.tobytes())
                if key in seen_inputs or _source_key(inp) in _SOURCE_INPUTS:
                    continue
                seen_inputs.add(key)
                puzzle[regime].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("Could not generate a unique pair")
            if regime == "train":
                slot += motif_count
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple(
        (regime, pair["input"].shape, pair["input"].tobytes(), pair["output"].tobytes())
        for regime in ("train", "test") for pair in puzzle[regime]
    )


def self_test() -> None:
    train_counts = set()
    test_counts = set()
    dimensions = set()
    backgrounds = set()
    colors = set()
    masks = set()
    layouts = set()
    gap_by_size = set()
    center_states = set()
    silhouettes = set()
    signatures = set()
    compact = 0
    large = 0
    crowded = 0
    source_dimensions = 0
    same_shape_pairs = 0
    sparse_five = 0
    new_test_colors = 0
    varied_backgrounds = 0
    five_empty_touch = 0
    source_shapes = {(8, 9), (12, 9), (12, 11), (10, 9), (10, 10), (12, 10)}
    sparse_plus = _from_rows(_FAMILIAR[5][0]).tobytes()
    size_by_regime = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _signature(puzzle)
        assert signature == _signature(generate_puzzle(seed))
        signatures.add(signature)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = set()
        test_colors = set()
        puzzle_backgrounds = set()
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                background, details = _inspect_pair(inp, out)
                dimensions.add(inp.shape)
                backgrounds.add(background)
                colors.update(d["color"] for d in details)
                puzzle_backgrounds.add(background)
                (train_colors if regime == "train" else test_colors).update(
                    d["color"] for d in details
                )
                compact += max(inp.shape) <= 12
                large += max(inp.shape) >= 16
                crowded += np.count_nonzero(pair["output"] != background) / inp.size >= 0.25
                source_dimensions += inp.shape in source_shapes
                same_shape_pairs += len({(d["size"], d["mask"]) for d in details}) < len(details)
                silhouettes.add((regime, len(details)))
                for d in details:
                    masks.add((d["size"], d["mask"]))
                    layouts.add((inp.shape, d["center"], d["seed"]))
                    gap_by_size.add((d["size"], min(d["gap"], 3)))
                    center_states.add((regime, d["center_filled"]))
                    size_by_regime[regime].add(d["size"])
                    sparse_five += d["size"] == 5 and d["mask"] == sparse_plus
                    five_empty_touch += d["size"] == 5 and not d["center_filled"] and d["gap"] == 1
                    assert pair["output"][d["seed"]] == (
                        d["color"] if d["center_filled"] else background
                    )
        new_test_colors += bool(test_colors - train_colors)
        varied_backgrounds += len(puzzle_backgrounds) > 1
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 30 and len(backgrounds) >= 6
    assert len(colors) >= 8 and len(masks) >= 35 and len(layouts) >= 100
    assert {("train", 1), ("train", 2), ("test", 1), ("test", 2)} <= silhouettes
    assert size_by_regime == {"train": {3, 5}, "test": {3, 5}}
    assert {("test", False), ("test", True)} <= center_states
    assert {(3, 1), (3, 2), (3, 3), (5, 1), (5, 3)} <= gap_by_size
    assert compact >= 50 and large >= 75 and crowded >= 20
    assert source_dimensions >= 5 and same_shape_pairs >= 5
    assert sparse_five >= 3 and five_empty_touch >= 3
    assert new_test_colors >= 15 and varied_backgrounds >= 30
    for request in (
        {"num_train": 2, "num_test": 1},
        {"num_train": 10, "num_test": 1},
        {"num_train": 10, "num_test": 4},
        {"num_train": None, "num_test": 1},
        {"num_train": None, "num_test": 4},
        {"num_train": 2, "num_test": None},
        {"num_train": 10, "num_test": None},
    ):
        puzzle = generate_puzzle(1234, **request)
        validate_generated_puzzle(puzzle)
        if request["num_train"] is not None:
            assert len(puzzle["train"]) == request["num_train"]
        if request["num_test"] is not None:
            assert len(puzzle["test"]) == request["num_test"]
        assert _signature(puzzle) == _signature(generate_puzzle(1234, **request))
    for request in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(0, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted illegal counts: {request}")
