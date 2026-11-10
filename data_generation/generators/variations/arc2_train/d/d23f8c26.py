"""Synthetic puzzles that retain only the vertical middle column."""

import numpy as np


_SOURCE_PAIR_ROWS = frozenset({
    (("80300", "86560", "36300", "00059", "50900"),
     ("00300", "00500", "00300", "00000", "00900")),
    (("640", "039", "100"), ("040", "030", "000")),
    (("30400", "30470", "06007", "00800", "08022"),
     ("00400", "00400", "00000", "00800", "00000")),
    (("0030007", "8108000", "0030803", "0701070", "0000000",
      "1086000", "0806010"),
     ("0000000", "0008000", "0000000", "0001000", "0000000",
      "0006000", "0006000")),
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _integer(value, name: str, low: int, high: int | None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < low or (high is not None and result > high):
        raise ValueError(f"{name} is outside the permitted range")
    return result


def _counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    test = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(v) for v in grid.flat)


def _background(grid: np.ndarray) -> int:
    counts = np.bincount(grid.reshape(-1).astype(np.int64), minlength=10)
    order = np.argsort(counts)
    _require(counts[order[-1]] > counts[order[-2]], "background is not uniquely dominant")
    return int(order[-1])


def _output(grid: np.ndarray, background: int) -> np.ndarray:
    result = np.full(grid.shape, background, dtype=np.int8)
    result[:, grid.shape[1] // 2] = grid[:, grid.shape[1] // 2]
    return result


def _size(rng: np.random.Generator, source_like: bool, test: bool) -> tuple[int, int]:
    if source_like:
        width = int(rng.choice([7, 9])) if test else int(rng.choice([3, 5]))
        return width, width
    width = int(rng.choice([3, 5, 7, 9, 11, 13]))
    if rng.random() < 0.28:
        return width, width
    return int(rng.integers(3, 16)), width


def _center_rows(rng: np.random.Generator, height: int, profile: str) -> list[int]:
    if profile == "single":
        return [int(rng.integers(height))]
    if profile == "full":
        return list(range(height))
    if profile == "sparse":
        count = int(rng.integers(2, max(3, (height + 1) // 2 + 1)))
        count = min(count, height)
        return sorted(int(x) for x in rng.choice(height, size=count, replace=False))
    if profile == "long":
        length = int(rng.integers(2, height + 1))
        start = int(rng.integers(height - length + 1))
        return list(range(start, start + length))
    if profile == "runs":
        count = min(height - 1, max(2, height // 2))
        rows = {0, height - 1}
        candidates = list(range(1, height - 1))
        rng.shuffle(candidates)
        rows.update(candidates[:count - 2])
        return sorted(rows)
    count = int(rng.integers(1, height + 1))
    return sorted(int(x) for x in rng.choice(height, size=count, replace=False))


def _motif_positions(rng: np.random.Generator, height: int, width: int,
                     center_rows: list[int], motif: str) -> set[tuple[int, int]]:
    mid = width // 2
    sides = [c for c in range(width) if c != mid]
    positions: set[tuple[int, int]] = set()
    if motif == "side":
        col = int(rng.choice(sides))
        length = min(height, max(len(center_rows), int(rng.integers(2, height + 1))))
        start = int(rng.integers(height - length + 1))
        positions.update((r, col) for r in range(start, start + length))
    elif motif == "bar":
        row = int(rng.integers(height))
        left = int(rng.integers(0, mid + 1))
        right = int(rng.integers(mid, width))
        positions.update((row, c) for c in range(left, right + 1) if c != mid)
    elif motif == "symmetric":
        for _ in range(int(rng.integers(1, height + 1))):
            row = int(rng.integers(height))
            col = int(rng.integers(mid))
            positions.add((row, col))
            positions.add((row, width - 1 - col))
    elif motif == "cluster":
        row = int(rng.integers(height))
        col = int(rng.choice(sides))
        for r in range(max(0, row - 1), min(height, row + 2)):
            for c in range(max(0, col - 1), min(width, col + 2)):
                if c != mid and rng.random() < 0.78:
                    positions.add((r, c))
    return positions


def _pair(rng: np.random.Generator, size: tuple[int, int], background: int,
          allowed: list[int], shared: int | None, off_only: int | None,
          center_required: int | None, first_train: bool) -> dict:
    height, width = size
    mid = width // 2
    profiles = ["single", "sparse", "runs", "long", "full", "mixed"]
    profile = "sparse" if first_train else str(rng.choice(profiles))
    center_rows = _center_rows(rng, height, profile)
    if first_train and len(center_rows) < 2:
        center_rows = [0, height - 1]
    palette_size = int(rng.integers(min(2, len(allowed)), len(allowed) + 1))
    palette = [int(v) for v in rng.choice(allowed, size=palette_size, replace=False)]
    required_palette = [v for v in (shared, off_only, center_required) if v is not None]
    for color in required_palette:
        if color not in palette:
            palette[-1] = color
    palette = list(dict.fromkeys(palette))
    if first_train and len(palette) < 2:
        palette.append(next(v for v in allowed if v != palette[0]))

    grid = np.full((height, width), background, dtype=np.int8)
    center_choices = [v for v in palette if v != off_only]
    if not center_choices:
        center_choices = [v for v in allowed if v != off_only]
    color_style = str(rng.choice(["iid", "repeat", "cycle"]))
    repeated = int(rng.choice(center_choices))
    for i, row in enumerate(center_rows):
        if color_style == "repeat" and rng.random() < 0.72:
            color = repeated
        elif color_style == "cycle":
            color = center_choices[i % len(center_choices)]
        else:
            color = int(rng.choice(center_choices))
        grid[row, mid] = color
    if shared is not None:
        grid[center_rows[0], mid] = shared
    if center_required is not None:
        grid[center_rows[-1], mid] = center_required

    all_off = [(r, c) for r in range(height) for c in range(width) if c != mid]
    motif_names = ["scatter", "side", "bar", "symmetric", "cluster"]
    motif = "side" if first_train else str(rng.choice(motif_names))
    positions = _motif_positions(rng, height, width, center_rows, motif)
    if rng.random() < 0.38:
        other = str(rng.choice(motif_names[1:]))
        positions.update(_motif_positions(rng, height, width, center_rows, other))
    density = float(rng.uniform(0.04, 0.72))
    positions.update(p for p in all_off if rng.random() < density)
    if not positions:
        positions.add(all_off[int(rng.integers(len(all_off)))])
    if first_train:
        side = int(rng.choice([c for c in range(width) if c != mid]))
        target = min(height, len(center_rows) + int(rng.integers(0, 2)))
        positions.update((r, side) for r in rng.choice(height, size=target, replace=False))

    # Keep enough blank cells for a unique dominant background even at 3x3.
    max_off = 0
    for count in range(1, len(all_off) + 1):
        foreground = len(center_rows) + count
        blank = height * width - foreground
        if blank > (foreground + len(palette) - 1) // len(palette):
            max_off = count
    max_off = max(1, max_off)
    protected = set()
    if first_train:
        for c in range(width):
            if c != mid:
                column = {(r, c) for r in range(height) if (r, c) in positions}
                if len(column) >= len(center_rows):
                    protected = column
                    break
    if len(positions) > max_off:
        kept = list(protected)
        others = list(positions - protected)
        rng.shuffle(others)
        positions = set(kept[:max_off] + others[:max(0, max_off - len(kept))])
    if not positions:
        positions = {all_off[0]}

    off_positions = list(positions)
    rng.shuffle(off_positions)
    forced = []
    if shared is not None:
        forced.append(shared)
    if off_only is not None and off_only not in forced:
        forced.append(off_only)
    for pos, color in zip(off_positions, forced):
        grid[pos] = color
    for pos in off_positions[len(forced):]:
        counts = {color: int(np.count_nonzero(grid == color)) for color in palette}
        least = min(counts.values())
        options = [color for color in palette if counts[color] <= least + int(rng.random() < 0.30)]
        grid[pos] = int(rng.choice(options))
    if motif == "cluster":
        eligible = off_positions[len(forced):]
        adjacent = [(a, b) for a in eligible for b in eligible
                    if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
        if adjacent:
            a, b = adjacent[int(rng.integers(len(adjacent)))]
            grid[b] = grid[a]

    # Remove dispensable clutter if a forced repeated color is too frequent.
    removable = [p for p in off_positions if p not in off_positions[:len(forced)] and p not in protected]
    rng.shuffle(removable)
    while removable:
        counts = np.bincount(grid.reshape(-1).astype(np.int64), minlength=10)
        if counts[background] > max(counts[c] for c in range(10) if c != background):
            break
        grid[removable.pop()] = background
    counts = np.bincount(grid.reshape(-1).astype(np.int64), minlength=10)
    if counts[background] <= max(counts[c] for c in range(10) if c != background):
        for pos in off_positions[len(forced):]:
            if grid[pos] != background:
                grid[pos] = background
                counts = np.bincount(grid.reshape(-1).astype(np.int64), minlength=10)
                if counts[background] > max(counts[c] for c in range(10) if c != background):
                    break
    return {"input": grid, "output": _output(grid, background)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0, None)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(["zero", "zero", "zero", "fixed", "vary", "vary"]))
    if seed % 13 == 0:
        mode = "zero"
    if mode == "zero":
        backgrounds = [0]
    elif mode == "fixed":
        backgrounds = [int(rng.integers(1, 10))]
    else:
        number = int(rng.integers(2, min(4, train_count + 1)))
        backgrounds = [int(v) for v in rng.choice(10, size=number, replace=False)]
    available = [v for v in range(10) if v not in backgrounds]
    if mode == "zero":
        available = list(range(1, 10))
    palette_count = (9 if seed % 13 == 0
                     else int(rng.integers(2, len(available) + 1)))
    palette = [int(v) for v in rng.choice(available, size=palette_count, replace=False)]
    transfer = str(rng.choice(["ordinary", "ordinary", "off_only", "off_only", "novel"]))
    if len(palette) < 3:
        transfer = "ordinary"
    special = int(rng.choice(palette)) if transfer != "ordinary" else None
    train_colors = [c for c in palette if c != special] if special is not None else palette[:]
    shared = int(rng.choice(train_colors))
    source_like = bool(rng.random() < 0.23)
    train_sizes = [_size(rng, source_like, False) for _ in range(train_count)]
    test_sizes = [_size(rng, source_like, True) for _ in range(test_count)]
    if source_like:
        first_two = [3, 5]
        rng.shuffle(first_two)
        train_sizes[:2] = [(width, width) for width in first_two]
    train_backgrounds = ([backgrounds[i % len(backgrounds)] for i in range(train_count)]
                         if len(backgrounds) > 1 else backgrounds * train_count)
    rng.shuffle(train_backgrounds)
    puzzle = {"train": [], "test": []}
    seen: set[tuple] = set()
    for section, sizes in (("train", train_sizes), ("test", test_sizes)):
        for i, size in enumerate(sizes):
            background = (train_backgrounds[i] if section == "train"
                          else int(rng.choice(train_backgrounds)))
            allowed = [c for c in (train_colors if section == "train" else palette)
                       if c != background]
            for _ in range(80):
                pair = _pair(rng, size, background, allowed,
                             shared if section == "train" and i == 0 else None,
                             special if section == "train" and i == 0 and transfer == "off_only" else None,
                             special if section == "test" and i == 0 and special is not None else None,
                             section == "train" and i == 0)
                key = _key(pair["input"])
                source_key = (_rows(pair["input"]), _rows(pair["output"]))
                if key not in seen and source_key not in _SOURCE_PAIR_ROWS:
                    break
            else:
                raise RuntimeError("could not construct distinct pair")
            seen.add(key)
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "sections must be lists")
    _require(2 <= len(puzzle["train"]) <= 10, "invalid train count")
    _require(1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"]), "invalid test count")
    seen_inputs = set()
    seen_pairs = set()
    training_backgrounds = set()
    training_shared = False
    training_rival = False
    training_multicell = False
    training_horizontal_counterexample = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid is not a 2D array")
                _require(grid.dtype.kind in "iu" and grid.dtype.kind != "b", "grid is not integer")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid dimensions outside ARC range")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "illegal ARC color")
            _require(inp.shape == out.shape and inp.shape[1] >= 3 and inp.shape[1] % 2 == 1, "invalid dimensions")
            background = _background(inp)
            mid = inp.shape[1] // 2
            _require(np.array_equal(out, _output(inp, background)), "wrong middle-column extraction")
            _require(np.any(inp[:, mid] != background), "center has no signal")
            off = inp.copy()
            off[:, mid] = background
            _require(np.any(off != background), "no erased distractor")
            input_key = _key(inp)
            pair_key = input_key, _key(out)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate generated pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require((_rows(inp), _rows(out)) not in _SOURCE_PAIR_ROWS, "source pair copied")
            if section == "train":
                training_backgrounds.add(background)
                center_colors = set(int(v) for v in inp[:, mid] if v != background)
                off_colors = set(int(v) for v in off.flat if v != background)
                training_shared |= bool(center_colors & off_colors)
                signal_count = int(np.count_nonzero(inp[:, mid] != background))
                training_multicell |= signal_count >= 2
                training_rival |= any(int(np.count_nonzero(inp[:, c] != background)) >= signal_count
                                      for c in range(inp.shape[1]) if c != mid)
                horizontal = np.full(inp.shape, background, dtype=np.int8)
                horizontal[inp.shape[0] // 2, :] = inp[inp.shape[0] // 2, :]
                training_horizontal_counterexample |= not np.array_equal(horizontal, out)
            else:
                _require(background in training_backgrounds, "test background unseen in training")
    _require(training_shared, "training lacks a color on and off the axis")
    _require(training_multicell, "training lacks a multi-cell center signal")
    _require(training_rival, "training lacks a side-column rival")
    _require(training_horizontal_counterexample, "training does not distinguish horizontal selection")


def _same(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (201, None, 1), (202, None, 4),
                              (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "train count not honored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "test count not honored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(17, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_same(generate_puzzle(17), generate_puzzle(17)), "not deterministic")
    _require(not _same(generate_puzzle(17), generate_puzzle(18)), "seeds identical")
    train_counts, test_counts = set(), set()
    shapes, layouts, palettes = set(), set(), set()
    tall = wide = even_height = larger_train = smaller_test = 0
    nonzero_bg = varying_bg = off_transfer = novel_transfer = 0
    single = full = dense = sparse = side_rival = 0
    bars = symmetric = clusters = adjacent_different = adjacent_repeat = 0
    rich = all_nine = overlap = train_denser = 0
    within_signal = within_density = within_palette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = set()
        train_survivors = set()
        train_bgs = set()
        test_bgs = set()
        all_colors = set()
        pair_signals = set()
        pair_densities = []
        pair_palette_sizes = set()
        train_density = []
        test_density = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                bg = _background(inp)
                mid = inp.shape[1] // 2
                expected = np.full(inp.shape, bg, dtype=np.int8)
                expected[:, mid] = inp[:, mid]
                _require(np.array_equal(out, expected), "independent self-test transformation failed")
                _require((_rows(inp), _rows(out)) not in _SOURCE_PAIR_ROWS, "self-test source copy")
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple(bool(v != bg) for v in inp.flat)))
                tall += inp.shape[0] > inp.shape[1]
                wide += inp.shape[0] < inp.shape[1]
                even_height += inp.shape[0] % 2 == 0
                nonzero_bg += bg != 0
                signal = int(np.count_nonzero(inp[:, mid] != bg))
                off = int(np.count_nonzero(inp != bg)) - signal
                pair_signals.add(signal)
                pair_densities.append((signal + off) / inp.size)
                pair_palette_sizes.add(len({int(v) for v in inp.flat if v != bg}))
                if section == "train":
                    train_density.append((signal + off) / inp.size)
                else:
                    test_density.append((signal + off) / inp.size)
                single += signal == 1
                full += signal == inp.shape[0]
                dense += (signal + off) > inp.size / 2
                sparse += (signal + off) / inp.size < 0.22
                bars += any(inp[r, mid] != bg and inp[r, mid - 1] != bg
                            and inp[r, mid + 1] != bg for r in range(inp.shape[0]))
                symmetric += any(inp[r, c] != bg and inp[r, inp.shape[1] - 1 - c] != bg
                                 for r in range(inp.shape[0]) for c in range(mid))
                clusters += any(inp[r, c] != bg and inp[r, c] == inp[r + 1, c]
                                for r in range(inp.shape[0] - 1) for c in range(inp.shape[1]) if c != mid)
                adjacent_different += any(inp[r, mid] != bg and inp[r + 1, mid] != bg
                                          and inp[r, mid] != inp[r + 1, mid]
                                          for r in range(inp.shape[0] - 1))
                adjacent_repeat += any(inp[r, mid] != bg and inp[r, mid] == inp[r + 1, mid]
                                       for r in range(inp.shape[0] - 1))
                if section == "train":
                    train_bgs.add(bg)
                    train_colors.update(int(v) for v in inp.flat if v != bg)
                    train_survivors.update(int(v) for v in inp[:, mid] if v != bg)
                    side_rival += any(np.count_nonzero(inp[:, c] != bg) >= signal
                                      for c in range(inp.shape[1]) if c != mid)
                else:
                    test_bgs.add(bg)
                all_colors.update(int(v) for v in inp.flat if v != bg)
        palettes.add(tuple(sorted(all_colors)))
        rich += len(all_colors) >= 7
        all_nine += all_colors == set(range(1, 10))
        overlap += bool({p["input"].shape[1] for p in puzzle["train"]} &
                        {p["input"].shape[1] for p in puzzle["test"]})
        train_denser += max(train_density) > min(test_density)
        within_signal += len(pair_signals) > 1
        within_density += max(pair_densities) - min(pair_densities) > 0.2
        within_palette += len(pair_palette_sizes) > 1
        varying_bg += len(train_bgs) > 1
        off_transfer += any(int(v) in train_colors - train_survivors
                            for p in puzzle["test"] for v in p["output"].flat
                            if v != _background(p["input"]))
        novel_transfer += any(int(v) not in train_colors
                              for p in puzzle["test"] for v in p["output"].flat
                              if v != _background(p["input"]))
        larger_train += any(p["input"].shape[1] >= 7 for p in puzzle["train"])
        smaller_test += min(p["input"].shape[1] for p in puzzle["test"]) < max(
            p["input"].shape[1] for p in puzzle["train"])
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity too low")
    _require(len(shapes) >= 25 and len(layouts) >= 150, "geometry diversity too low")
    _require(len(palettes) >= 20, "palette diversity too low")
    _require(min(tall, wide, even_height, larger_train, smaller_test) > 0, "geometry regimes missing")
    _require(min(nonzero_bg, varying_bg, off_transfer, novel_transfer) > 0, "color regime missing")
    _require(min(single, full, dense, sparse, side_rival) > 0, "signal/clutter regime missing")
    _require(min(bars, symmetric, clusters, adjacent_different, adjacent_repeat) > 0,
             "geometric or color-run motif missing")
    _require(min(rich, all_nine, overlap, train_denser) > 0,
             "palette or train/test regime missing")
    _require(min(within_signal, within_density, within_palette) > 0,
             "within-puzzle variation missing")
