import hashlib

import numpy as np


# Hashes of the four supplied clean grids, including their dimensions.
_SOURCE_OUTPUTS = {
    "744087743aab5cf2be6d4d5f2704e1ac57419a888710314f4348dd4ee9dca747",
    "e4e7aea73e0637be4c3ae340562730d4fa3c018d266cc78dc3756fd994380408",
    "d72dc10487d73dc4a2d355a0cca83a7aec6ccbd967b1b77c0eb724ff63607e46",
    "dbe1fc12e54a5f801333cdc6868e4cc6d184910834dd120fb337d02f8f4a1ab6",
}
_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _counts(seed, num_train, num_test):
    if num_train is not None:
        if not _integer(num_train) or not 2 <= num_train <= 10:
            raise ValueError("num_train must be an integer from 2 to 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer(num_test) or not 1 <= num_test <= 4:
            raise ValueError("num_test must be an integer from 1 to 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if num_train is None:
        choices = range(max(2, num_test + 1), 11)
        num_train = choices[seed % len(choices)]
    if num_test is None:
        choices = range(1, min(4, num_train - 1) + 1)
        num_test = choices[(seed // 3) % len(choices)]
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _key(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def _output_hash(grid):
    return hashlib.sha256(
        bytes(grid.shape) + grid.astype(np.uint8, copy=False).tobytes()
    ).hexdigest()


def _axis(rng, first):
    copies = int(rng.integers(3 if first else 2, 5))
    feasible = [(size, gap) for size in range(3, 8)
                for gap in range(1, 5)
                if copies * size + (copies - 1) * gap <= 30]
    size, gap = feasible[int(rng.integers(len(feasible)))]
    return copies, size, gap


def _signature(rng, first=False):
    rows, height, v_gap = _axis(rng, first)
    columns, width, h_gap = _axis(rng, first)
    return rows, columns, height, width, v_gap, h_gap


def _canvas_axis(rng, lattice):
    available = 30 - lattice
    if rng.random() < 0.09:
        extra = available
    else:
        extra = int(rng.integers(0, min(9, available) + 1))
    before = int(rng.integers(0, extra + 1))
    return lattice + extra, before


def _colors(rng, previous, stable):
    if previous and rng.random() < (0.88 if stable else 0.22):
        background = previous[int(rng.integers(len(previous)))][0]
    else:
        background = int(rng.integers(10))
    if previous and rng.random() < (0.86 if stable else 0.16):
        matching = [palette for bg, palette in previous if bg == background]
        if matching:
            return background, matching[int(rng.integers(len(matching)))]
    count = int(rng.choice(np.array([2, 3, 4, 5, 6]),
                           p=np.array([0.18, 0.24, 0.26, 0.20, 0.12])))
    available = [color for color in range(10) if color != background]
    palette = tuple(int(x) for x in rng.choice(available, count, replace=False))
    return background, palette


def _motif(rng, height, width, palette):
    count = len(palette)
    style = int(rng.integers(6))
    rr, cc = np.indices((height, width))
    if style == 0:  # mosaic
        codes = rng.integers(count, size=(height, width))
    elif style == 1:  # intact horizontal or vertical bands
        axis = rr if rng.random() < 0.5 else cc
        run = int(rng.integers(1, 4))
        codes = (axis // run + int(rng.integers(count))) % count
    elif style == 2:  # rectangular blocks
        run_r = int(rng.integers(1, 4))
        run_c = int(rng.integers(1, 4))
        block_codes = rng.integers(
            count, size=((height + run_r - 1) // run_r,
                         (width + run_c - 1) // run_c)
        )
        codes = block_codes[rr // run_r, cc // run_c]
    elif style == 3:  # left-right reflection
        half = rng.integers(count, size=(height, (width + 1) // 2))
        codes = np.concatenate((half, half[:, :width // 2][:, ::-1]), axis=1)
    elif style == 4:  # repeated rows
        period = int(rng.integers(1, min(3, height) + 1))
        proto = rng.integers(count, size=(period, width))
        codes = proto[rr % period, cc]
    else:  # solid edge strips with a varied interior
        codes = rng.integers(count, size=(height, width))
        edge = int(rng.integers(count))
        codes[0, :] = edge
        if rng.random() < 0.6:
            codes[-1, :] = edge
        if rng.random() < 0.5:
            codes[:, 0] = edge
    if len(np.unique(codes)) == 1:
        # A whole band maintains the sampled structure, including reflection.
        codes[-1, :] = (int(codes[0, 0]) + 1) % count
    return np.asarray(palette, dtype=np.int8)[codes]


def _runs(flags):
    indices = np.flatnonzero(flags)
    if not len(indices):
        return []
    starts = [int(indices[0])]
    ends = []
    for old, new in zip(indices[:-1], indices[1:]):
        if new != old + 1:
            ends.append(int(old) + 1)
            starts.append(int(new))
    ends.append(int(indices[-1]) + 1)
    return list(zip(starts, ends))


def _geometry_from_runs(rows, columns):
    assert 2 <= len(rows) <= 4 and 2 <= len(columns) <= 4
    height = rows[0][1] - rows[0][0]
    width = columns[0][1] - columns[0][0]
    assert 3 <= height <= 7 and 3 <= width <= 7
    assert all(end - start == height for start, end in rows)
    assert all(end - start == width for start, end in columns)
    v_gaps = [rows[i + 1][0] - rows[i][1] for i in range(len(rows) - 1)]
    h_gaps = [columns[i + 1][0] - columns[i][1]
              for i in range(len(columns) - 1)]
    assert len(set(v_gaps)) == len(set(h_gaps)) == 1
    assert 1 <= v_gaps[0] <= 4 and 1 <= h_gaps[0] <= 4
    return (len(rows), len(columns), height, width, v_gaps[0], h_gaps[0])


def _analyze_output(grid):
    uniform_rows = {int(row[0]) for row in grid
                    if np.all(row == row[0])}
    uniform_columns = {int(column[0]) for column in grid.T
                       if np.all(column == column[0])}
    candidates = uniform_rows & uniform_columns
    assert len(candidates) == 1
    background = candidates.pop()
    mask = grid != background
    rows = _runs(mask.any(axis=1))
    columns = _runs(mask.any(axis=0))
    signature = _geometry_from_runs(rows, columns)
    _, _, height, width, _, _ = signature
    motif = grid[rows[0][0]:rows[0][1],
                 columns[0][0]:columns[0][1]].copy()
    assert np.all(motif != background)
    expected = np.full_like(grid, background)
    for top, _ in rows:
        for left, _ in columns:
            expected[top:top + height, left:left + width] = motif
    assert np.array_equal(expected, grid)
    return {
        "background": background, "signature": signature,
        "motif": motif, "rows": rows, "columns": columns,
        "footprint": mask,
    }


def _decode_for_background(grid, background):
    # Full motif rows and columns have many foreground cells. Sparse gap noise
    # cannot reach the band threshold enforced at construction and validation.
    mask = grid != background
    row_density = mask.sum(axis=1)
    col_density = mask.sum(axis=0)
    rows = _runs(row_density > 0.45 * row_density.max())
    columns = _runs(col_density > 0.45 * col_density.max())
    signature = _geometry_from_runs(rows, columns)
    nr, nc, height, width, _, _ = signature
    stack = np.stack([
        grid[top:top + height, left:left + width]
        for top, _ in rows for left, _ in columns
    ])
    motif = np.empty((height, width), dtype=np.int8)
    for r in range(height):
        for c in range(width):
            frequencies = np.bincount(stack[:, r, c].astype(int), minlength=10)
            color = int(np.argmax(frequencies))
            assert color != background
            assert frequencies[color] > nr * nc // 2
            motif[r, c] = color
    clean = np.full_like(grid, background)
    for top, _ in rows:
        for left, _ in columns:
            clean[top:top + height, left:left + width] = motif
    assert _analyze_output(clean)["background"] == background
    return clean


def _decode_input(grid):
    # Compare all visible background candidates. Separator rows and columns,
    # a strict aligned-cell majority, and few edits select one clean grid.
    solutions = []
    for background in np.unique(grid):
        try:
            clean = _decode_for_background(grid, int(background))
        except (AssertionError, ValueError):
            continue
        solutions.append((int(np.count_nonzero(clean != grid)), clean))
    assert solutions
    best = min(score for score, _ in solutions)
    winners = {_key(clean) for score, clean in solutions if score == best}
    assert len(winners) == 1
    return next(clean for score, clean in solutions if score == best)


def _damage_inside(rng, noisy, motif, rows, columns, palette, level):
    height, width = motif.shape
    nr, nc = len(rows), len(columns)
    copies = nr * nc
    max_wrong = (copies - 1) // 2
    eligible = [(r, c) for r in range(height) for c in range(width)
                if any(color != int(motif[r, c]) for color in palette)]
    if not eligible:
        return
    rate = {"light": (0.025, 0.065), "medium": (0.07, 0.14),
            "hard": (0.13, 0.23)}[level]
    target = max(1, int(rng.uniform(*rate) * copies * height * width))
    target = min(target, len(eligible) * max_wrong)
    damaged = set()
    usage = {position: 0 for position in eligible}

    def put(copy, position):
        if (copy, position) in damaged or usage[position] >= max_wrong:
            return False
        r, c = position
        alternatives = [color for color in palette if color != int(motif[r, c])]
        if not alternatives:
            return False
        top = rows[copy // nc][0]
        left = columns[copy % nc][0]
        noisy[top + r, left + c] = alternatives[int(rng.integers(len(alternatives)))]
        damaged.add((copy, position))
        usage[position] += 1
        return True

    if level == "hard" and copies >= 7 and target >= max_wrong:
        position = eligible[int(rng.integers(len(eligible)))]
        for copy in rng.choice(copies, max_wrong, replace=False):
            put(int(copy), position)
    if level == "hard" and target >= copies and rng.random() < 0.5:
        for copy in rng.permutation(copies):
            for position in rng.permutation(len(eligible)):
                if put(int(copy), eligible[int(position)]):
                    break
    attempts = 0
    while len(damaged) < target and attempts < target * 40 + 100:
        attempts += 1
        position = eligible[int(rng.integers(len(eligible)))]
        copy = int(rng.integers(copies))
        put(copy, position)
    assert len(damaged) == target


def _damage_outside(rng, noisy, clean, background, palette, level):
    positions = np.argwhere(clean == background)
    if not len(positions):
        return
    rate = {"light": (0.012, 0.027), "medium": (0.035, 0.068),
            "hard": (0.065, 0.105)}[level]
    target = max(1, int(rng.uniform(*rate) * clean.size))
    target = min(target, int(len(positions) * 0.30))
    info = _analyze_output(clean)
    nr, nc, height, width, _, _ = info["signature"]
    motif_row = np.zeros(clean.shape[0], dtype=bool)
    motif_col = np.zeros(clean.shape[1], dtype=bool)
    for top, end in info["rows"]:
        motif_row[top:end] = True
    for left, end in info["columns"]:
        motif_col[left:end] = True
    row_count = np.count_nonzero(noisy != background, axis=1)
    col_count = np.count_nonzero(noisy != background, axis=0)
    row_base, col_base = nc * width, nr * height
    blank_row_cap = max(1, int(0.27 * row_base))
    blank_col_cap = max(1, int(0.27 * col_base))
    active_row_cap = int(0.33 * row_base)
    active_col_cap = int(0.33 * col_base)
    placed = []
    attempts = 0
    while len(placed) < target and attempts < target * 120 + 300:
        attempts += 1
        if placed and rng.random() < 0.28:
            old_r, old_c = placed[int(rng.integers(len(placed)))]
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[
                int(rng.integers(4))]
            r, c = old_r + dr, old_c + dc
            if not (0 <= r < clean.shape[0] and 0 <= c < clean.shape[1]):
                continue
        else:
            r, c = positions[int(rng.integers(len(positions)))]
            r, c = int(r), int(c)
        if clean[r, c] != background or noisy[r, c] != background:
            continue
        r_cap = row_base + active_row_cap if motif_row[r] else blank_row_cap
        c_cap = col_base + active_col_cap if motif_col[c] else blank_col_cap
        if row_count[r] >= r_cap or col_count[c] >= c_cap:
            continue
        noisy[r, c] = palette[int(rng.integers(len(palette)))]
        row_count[r] += 1
        col_count[c] += 1
        placed.append((r, c))
    assert len(placed) >= max(1, min(target, 3))


def _pair(rng, signature, previous_colors, stable, level, mode):
    nr, nc, height, width, v_gap, h_gap = signature
    lattice_h = nr * height + (nr - 1) * v_gap
    lattice_w = nc * width + (nc - 1) * h_gap
    canvas_h, top = _canvas_axis(rng, lattice_h)
    canvas_w, left = _canvas_axis(rng, lattice_w)
    background, palette = _colors(rng, previous_colors, stable)
    motif = _motif(rng, height, width, palette)
    actual_palette = tuple(int(x) for x in np.unique(motif))
    rows = [(top + i * (height + v_gap), top + i * (height + v_gap) + height)
            for i in range(nr)]
    columns = [(left + i * (width + h_gap), left + i * (width + h_gap) + width)
               for i in range(nc)]
    clean = np.full((canvas_h, canvas_w), background, dtype=np.int8)
    for r, _ in rows:
        for c, _ in columns:
            clean[r:r + height, c:c + width] = motif
    # Noise uses a per-pair subset; sometimes one legitimate motif color is
    # the sole corruption color, as in the original examples.
    subset_size = int(rng.integers(1, min(3, len(actual_palette)) + 1))
    noise_palette = tuple(int(x) for x in rng.choice(
        actual_palette, subset_size, replace=False))
    noisy = clean.copy()
    if mode != "outside":
        _damage_inside(rng, noisy, motif, rows, columns, noise_palette, level)
    if mode != "inside":
        _damage_outside(rng, noisy, clean, background, noise_palette, level)
    pair = {"input": noisy, "output": clean}
    _validate_pair(pair)
    return pair, (background, palette)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Difficulty is a puzzle-level sampling convention; all color meanings
    # remain local to each pair and are visible through its dense lattice.
    tier = int(rng.integers(3))
    stable = tier == 0
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_colors = []
    train_signatures = []
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for attempt in range(100):
                first = split == "train" and index == 0
                if not first and train_signatures and rng.random() < (
                    0.78 if stable else 0.12
                ):
                    signature = train_signatures[int(rng.integers(
                        len(train_signatures)))]
                else:
                    signature = _signature(rng, first=first)
                if first:
                    level, mode = ("hard" if tier == 2 else "medium"), "both"
                else:
                    levels = ("light", "medium", "hard")
                    probabilities = ((0.45, 0.40, 0.15) if split == "train"
                                     else (0.18, 0.43, 0.39))
                    level = str(rng.choice(levels, p=probabilities))
                    mode = str(rng.choice(("both", "inside", "outside"),
                                          p=(0.70, 0.15, 0.15)))
                try:
                    pair, colors = _pair(
                        rng, signature, previous_colors, stable, level, mode
                    )
                except AssertionError:
                    continue
                key = _key(pair["input"])
                if key in seen:
                    continue
                seen.add(key)
                previous_colors.append(colors)
                puzzle[split].append(pair)
                if split == "train":
                    train_signatures.append(signature)
                break
            else:
                raise RuntimeError("could not construct a recoverable pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= size <= 30 for size in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert 0 <= int(grid.min()) and int(grid.max()) <= 9


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    noisy, clean = pair["input"], pair["output"]
    _validate_grid(noisy)
    _validate_grid(clean)
    assert noisy.shape == clean.shape
    assert not np.array_equal(noisy, clean)
    assert _output_hash(clean) not in _SOURCE_OUTPUTS
    info = _analyze_output(clean)
    bg = info["background"]
    motif = info["motif"]
    assert 2 <= len(np.unique(motif)) <= 6
    assert np.all(noisy[info["footprint"]] != bg)
    assert set(np.unique(noisy)).issubset(set(np.unique(clean)))
    difference = noisy != clean
    assert np.all(noisy[difference] != bg)
    rows, columns = info["rows"], info["columns"]
    height, width = motif.shape
    stack = np.stack([
        noisy[top:top + height, left:left + width]
        for top, _ in rows for left, _ in columns
    ])
    n = len(stack)
    wrong = np.count_nonzero(stack != motif, axis=0)
    assert np.all(wrong <= (n - 1) // 2)
    predicted = _decode_input(noisy)
    assert np.array_equal(predicted, clean)
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_repair = False
    train_cleanup = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            if split == "train":
                difference = pair["input"] != pair["output"]
                train_repair |= bool(np.any(difference & info["footprint"]))
                train_cleanup |= bool(np.any(difference & ~info["footprint"]))
    assert train_repair and train_cleanup


def _same_puzzle(a, b):
    for split in ("train", "test"):
        if len(a[split]) != len(b[split]):
            return False
        for x, y in zip(a[split], b[split]):
            for role in ("input", "output"):
                if not np.array_equal(x[role], y[role]):
                    return False
    return True


def self_test() -> None:
    train_counts, test_counts = set(), set()
    signatures, shapes, palettes, backgrounds = set(), set(), set(), set()
    arrangements, gaps, motif_shapes = set(), set(), set()
    strong, source_noise, symmetric, edge, thirty, small = (
        False, False, False, False, False, False
    )
    background_changes = 0
    all_keys = set()
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _validate_pair(pair)
                signature = info["signature"]
                signatures.add(signature)
                arrangements.add(signature[:2])
                gaps.add(signature[4:])
                motif_shapes.add(signature[2:4])
                shapes.add(pair["input"].shape)
                palette_size = len(np.unique(info["motif"]))
                palettes.add(palette_size)
                backgrounds.add(info["background"])
                puzzle_backgrounds.add(info["background"])
                diff = pair["input"] != pair["output"]
                source_noise |= int(diff.sum()) >= 26
                motif = info["motif"]
                symmetric |= np.array_equal(motif, motif[:, ::-1])
                edge |= any(v == 0 for v in (
                    info["rows"][0][0], info["columns"][0][0],
                    pair["input"].shape[0] - info["rows"][-1][1],
                    pair["input"].shape[1] - info["columns"][-1][1],
                ))
                thirty |= 30 in pair["input"].shape
                small |= min(pair["input"].shape) <= 14
                height, width = motif.shape
                stack = np.stack([
                    pair["input"][r:r + height, c:c + width]
                    for r, _ in info["rows"] for c, _ in info["columns"]
                ])
                strong |= bool(np.any(np.count_nonzero(
                    stack != motif, axis=0) >= 4))
                key = _key(pair["input"])
                assert key not in all_keys
                all_keys.add(key)
                total += 1
        background_changes += len(puzzle_backgrounds) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 45 and len(shapes) >= 35
    assert len(arrangements) >= 7 and len(gaps) >= 8
    assert len(motif_shapes) >= 12
    assert {2, 3, 4}.issubset(palettes) and max(palettes) >= 5
    assert len(backgrounds) >= 8 and background_changes >= 20
    assert strong and source_noise and symmetric and edge and thirty and small
    assert len(all_keys) == total
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                     (None, 1), (None, 4),
                                     (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + index, *counts)
        validate_generated_puzzle(puzzle)
        for actual, requested in zip((len(puzzle["train"]),
                                      len(puzzle["test"])), counts):
            assert requested is None or actual == requested
    first = generate_puzzle(314159, 6, 3)
    assert _same_puzzle(first, generate_puzzle(314159, 6, 3))
    assert not _same_puzzle(first, generate_puzzle(314160, 6, 3))
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(1, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
