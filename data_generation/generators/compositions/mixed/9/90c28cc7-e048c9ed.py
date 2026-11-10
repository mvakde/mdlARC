"""Unstretch a registered mosaic, then label each bar's logical adjacencies.

Collapse whole-canvas full-vector runs before measuring bars. One or two
preserved multicolor registration rows expose all logical column seams;
the unique singleton row selects the column where each bar writes its
squared adjacency count's decimal units digit. Lengths are 2..5. The saved
proposal explicitly requires 5->6 despite parent B's source disagreement.

Only NumPy and the standard library are used. Generation constructs expected
outputs from compact source objects; solve reads only the supplied input.
No generation, validation, or self-test function writes files. At most 14
pairs and one candidate are retained; each grid has at most 900 int8 cells.
"""
from __future__ import annotations

import hashlib

import numpy as np


LENGTHS = (2, 3, 4, 5)
DIGITS = (1, 4, 9, 6)
BACKGROUNDS = (0, 2, 3, 5, 7, 8)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return tuple(grid.shape), grid.astype(np.uint8, copy=False).tobytes()


def _collapse(grid):
    """Maximal consecutive full-vector runs, independently on both axes."""
    rows = np.r_[0, 1 + np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=1))]
    cols = np.r_[0, 1 + np.flatnonzero(np.any(grid[:, 1:] != grid[:, :-1], axis=0))]
    compact = grid[np.ix_(rows, cols)].copy()
    heights = np.diff(np.r_[rows, grid.shape[0]])
    widths = np.diff(np.r_[cols, grid.shape[1]])
    return compact, heights, widths


def _read_compact(compact):
    """Recognize the background, singleton, registration rows, and bar rows."""
    uniform = [int(row[0]) for row in compact if np.all(row == row[0])]
    if not uniform or len(set(uniform)) != 1:
        raise ValueError("Background must be identified by uniform empty rows")
    background = uniform[0]
    if background not in BACKGROUNDS:
        raise ValueError("Background would hide a square digit")
    singles, bars, registers = [], [], []
    for r, row in enumerate(compact):
        occupied = np.flatnonzero(row != background)
        if len(occupied) == 0:
            continue
        if len(occupied) == 1:
            singles.append((r, int(occupied[0]), int(row[occupied[0]])))
        elif len(np.unique(row[occupied])) >= 2:
            if len(occupied) != compact.shape[1]:
                raise ValueError("A registration row must span the mosaic")
            registers.append(r)
        else:
            start, end = int(occupied[0]), int(occupied[-1])
            length = len(occupied)
            if length not in LENGTHS or end - start + 1 != length:
                raise ValueError("A bar must be a single monochrome run of length 2..5")
            bars.append((r, start, length, int(row[start])))
    if len(singles) != 1 or not bars or not 1 <= len(registers) <= 2:
        raise ValueError("Need one reference cell, bars, and one or two registration rows")
    reference_row, column, marker = singles[0]
    for r in (reference_row - 1, reference_row + 1):
        if 0 <= r < compact.shape[0] and np.any(compact[r] != background):
            raise ValueError("Reference cell must have empty neighboring rows")
    if any(b[0] - a[0] < 2 for a, b in zip(bars, bars[1:])):
        raise ValueError("Bar rows cannot be adjacent")
    enriched = []
    for r, start, length, color in bars:
        if not (start + length <= column or start > column):
            raise ValueError("The reference column must be empty on bar rows")
        side = -1 if start < column else 1
        enriched.append((r, start, length, color, side))
    register = compact[registers, :]
    if np.any(np.all(register[:, 1:] == register[:, :-1], axis=0)):
        raise ValueError("Registration rows must jointly expose every column seam")
    return {"background": background, "marker_row": reference_row,
            "column": column, "marker": marker, "bars": enriched,
            "registers": tuple(registers)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    compact, _, _ = _collapse(_grid(input_grid))
    info = _read_compact(compact)
    output = compact.copy()
    for row, _, length, _, _ in info["bars"]:
        output[row, info["column"]] = (length - 1) ** 2 % 10
    return output


def _registration(rng, width, count, background):
    """A joint column ruler; with two rows neither row alone need suffice."""
    available = [c for c in range(10) if c != background]
    palette = [int(c) for c in rng.choice(available, int(rng.integers(2, 6)), replace=False)]
    rows = np.empty((count, width), dtype=np.int8)
    rows[:, 0] = rng.choice(palette, count)
    if count == 1:
        changes = [3] * (width - 1)
    else:
        # There is a seam witnessed by each row alone, at random positions.
        changes = [1, 2] + [int(rng.integers(1, 4)) for _ in range(width - 3)]
        rng.shuffle(changes)
    for j, change in enumerate(changes, 1):
        rows[:, j] = rows[:, j - 1]
        for r in range(count):
            if count == 1 or change & (1 << r):
                rows[r, j] = rng.choice([c for c in palette if c != rows[r, j - 1]])
    return rows


def _axis(rng, size, mode):
    if mode == "uniform" and size <= 15:
        return np.full(size, int(rng.integers(2, min(4, 30 // size) + 1)), dtype=int)
    bands = np.ones(size, dtype=int)
    if mode == "peak":
        peak = int(rng.integers(size))
        bands[peak] += int(rng.integers(2, min(10, 30 - size) + 1))
        extra = int(rng.integers(0, min(5, 30 - bands.sum()) + 1))
    else:
        extra = int(rng.integers(1, min(30 - size, 2 * size + 4) + 1))
    if extra:
        bands += rng.multinomial(extra, np.full(size, 1 / size))
    if len(set(bands)) == 1:
        if bands[0] > 1:
            bands[0] -= 1
            bands[-1] += 1
        else:
            bands[int(rng.integers(size))] += 1
    return bands


def _make_pair(rng, specs, background, marker, register_count, *, uneven=False,
               reversal=False):
    """Build a compact scene first; expand only after the answer is constructed."""
    needs = {side: max((length for length, _, s in specs if s == side), default=0)
             for side in (-1, 1)}
    capacity = {side: needs[side] + int(rng.integers(0, 4)) for side in (-1, 1)}
    column = capacity[-1]
    width = column + 1 + capacity[1]
    registers = _registration(rng, width, register_count, background)
    events = [("bar", i) for i in range(len(specs))]
    events += [("register", i) for i in range(register_count)] + [("reference", 0)]
    rng.shuffle(events)
    blank = np.full(width, background, dtype=np.int8)
    rows, bar_records = [], []
    if rng.random() < .5:
        rows.append(blank.copy())
    previous = None
    for kind, index in events:
        if previous is not None:
            gap = (kind == "reference" or previous == "reference"
                   or kind == previous == "bar" or rng.random() < .55)
            if gap:
                rows.append(blank.copy())
        row = blank.copy()
        if kind == "register":
            row = registers[index].copy()
        elif kind == "reference":
            row[column] = marker
        else:
            length, color, side = specs[index]
            offset = int(rng.integers(0, capacity[side] - length + 1))
            start = offset if side == -1 else column + 1 + offset
            row[start:start + length] = color
            bar_records.append((len(rows), start, length))
        rows.append(row)
        previous = kind
    if rng.random() < .5:
        rows.append(blank.copy())
    compact = np.stack(rows)
    # This expected output is built from source-object records, not solve().
    output = compact.copy()
    for row, _, length in bar_records:
        output[row, column] = (length - 1) ** 2 % 10
    profile = "uneven" if uneven else str(rng.choice(
        ("uniform", "row_uniform", "column_uniform", "uneven", "peak"),
        p=(.08, .16, .16, .40, .20)))
    heights = _axis(rng, compact.shape[0],
                    "uniform" if profile in ("uniform", "row_uniform") else
                    "peak" if profile == "peak" else "uneven")
    widths = _axis(rng, width,
                   "uniform" if profile in ("uniform", "column_uniform") else
                   "peak" if profile == "peak" else "uneven")
    if reversal:
        # The first two specs are opposite-side lengths 2 and 5. Give a short
        # bar's band a large width; the long bar's five bands remain one each.
        # Other randomly sampled bars of the same lengths can overlap either
        # side. Select the opposite-side mandatory records explicitly.
        candidates = [(a, b) for a in bar_records for b in bar_records
                      if a[2] == 2 and b[2] == 5
                      and (a[1] < column) != (b[1] < column)]
        short, long = candidates[int(rng.integers(len(candidates)))]
        widths = np.ones(width, dtype=int)
        widths[short[1] + int(rng.integers(2))] = int(rng.integers(6, 11))
        protected = set(range(long[1], long[1] + 5))
        for j in rng.permutation(width):
            j = int(j)
            if j not in protected and widths[j] == 1 and widths.sum() < 30:
                widths[j] += int(rng.integers(0, min(2, 30 - widths.sum()) + 1))
    inp = np.repeat(np.repeat(compact, heights, axis=0), widths, axis=1)
    return {"input": inp.astype(np.int8), "output": output.astype(np.int8)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    background_count = 1 if rng.random() < .6 else 2
    backgrounds = [int(c) for c in rng.choice(BACKGROUNDS, background_count, replace=False)]
    available = [c for c in range(10) if c not in backgrounds]
    palette = [int(c) for c in rng.choice(available, int(rng.integers(2, 6)), replace=False)]
    a, b = palette[:2]
    permutation = [int(x) for x in rng.permutation(LENGTHS)]
    counts = [int(x) for x in rng.integers(1, 7, size=num_train)]
    while sum(counts) < 6 or max(counts) < 2:
        eligible = [i for i, n in enumerate(counts) if n < 6]
        counts[int(rng.choice(eligible))] += 1
    # Guarantee all four digits and both directions of color/length independence.
    entries = [(permutation[0], a), (permutation[1], a),
               (permutation[2], b), (permutation[3], b),
               (permutation[0], b), (permutation[1], a)]
    entries += [(int(rng.choice(LENGTHS)), int(rng.choice(palette)))
                for _ in range(sum(counts) - 6)]
    rng.shuffle(entries)
    train_specs, offset = [], 0
    mixed_index = int(rng.choice([i for i, n in enumerate(counts) if n >= 2]))
    for i, count in enumerate(counts):
        sides = [int(rng.choice((-1, 1))) for _ in range(count)]
        if i == mixed_index:
            sides[:2] = [-1, 1]
            rng.shuffle(sides)
        train_specs.append([(length, color, side) for (length, color), side
                            in zip(entries[offset:offset + count], sides)])
        offset += count
    variable_marker = bool(rng.random() < .5)
    fixed_marker = a
    first_marker = train_specs[0][0][1]
    register_counts = [1, 2] if rng.random() < .5 else [2, 1]
    register_counts += [int(rng.integers(1, 3)) for _ in range(num_train - 2)]
    special_index = int(rng.integers(num_train))
    puzzle = {"train": [], "test": []}
    used, train_columns = set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                specs = train_specs[index]
                n_registers = register_counts[index]
            else:
                n_bars = int(rng.integers(2 if index == 0 else 1, 7))
                specs = [(int(rng.choice(LENGTHS)), int(rng.choice(palette)),
                          int(rng.choice((-1, 1)))) for _ in range(n_bars)]
                if index == 0:
                    side = int(rng.choice((-1, 1)))
                    specs[:2] = [(2, int(rng.choice(palette)), side),
                                 (5, int(rng.choice(palette)), -side)]
                n_registers = int(rng.integers(1, 3))
            bg = (backgrounds[index] if split == "train" and index < background_count
                  else int(rng.choice(backgrounds)))
            if not variable_marker:
                marker = fixed_marker
            elif split == "train" and index == 0:
                marker = first_marker
            elif split == "train" and index == 1:
                marker = int(rng.choice([c for c in available if c != first_marker]))
            else:
                marker = int(rng.choice(available))
            for _ in range(48):
                pair = _make_pair(rng, specs, bg, marker, n_registers,
                                  uneven=split == "train" and index == special_index,
                                  reversal=split == "test" and index == 0)
                compact, _, widths = _collapse(pair["input"])
                meta = _read_compact(compact)
                # Distinguish a fixed output column and the physical-width rule
                # even with only two training examples.
                if split == "train" and index == 1 and meta["column"] in train_columns:
                    continue
                if split == "train" and index == special_index:
                    raw_differs = any((int(widths[s:s + length].sum()) - 1) ** 2 % 10
                                      != (length - 1) ** 2 % 10
                                      for _, s, length, _, _ in meta["bars"])
                    if not raw_differs:
                        continue
                key = _key(pair["input"])
                if key in used:
                    continue
                used.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_columns.add(meta["column"])
                break
            else:
                raise RuntimeError("Could not construct a distinct, informative example")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    compact, heights, widths = _collapse(inp)
    meta = _read_compact(compact)
    _require(1 <= len(meta["bars"]) <= 6, "Expected one through six bar rows")
    if len(meta["registers"]) == 2:
        register = compact[list(meta["registers"]), :]
        changes = register[:, 1:] != register[:, :-1]
        _require(np.any(changes[0] & ~changes[1])
                 and np.any(changes[1] & ~changes[0]),
                 "Two-row registration must demonstrate joint seam evidence")
    _require(np.array_equal(inp, np.repeat(np.repeat(compact, heights, axis=0),
                                         widths, axis=1)), "Replication reconstruction failed")
    _require(out.shape == compact.shape, "Output must retain the compact canvas dimensions")
    _require(max(heights) > 1 and max(widths) > 1, "Both axes must actually expand")
    expected = compact.copy()
    for row, _, length, _, _ in meta["bars"]:
        _require(compact[row, meta["column"]] == meta["background"], "Occupied label target")
        expected[row, meta["column"]] = (length - 1) ** 2 % 10
    _require(np.array_equal(expected, out), "Wrong digit or changed mosaic content")
    _require(np.count_nonzero(compact != out) == len(meta["bars"]), "One added cell per bar required")
    _require(np.array_equal(solve(inp), out), "Input-only solve disagrees")
    meta.update({"compact": compact, "heights": heights, "widths": widths})
    return meta


def _physical_width(meta, bar):
    return int(meta["widths"][bar[1]:bar[1] + bar[2]].sum())


def _reversal(meta):
    return any(a[2] < b[2] and _physical_width(meta, a) > _physical_width(meta, b)
               for a in meta["bars"] for b in meta["bars"])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Wrong puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, metas = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Wrong pair schema")
            inp = _grid(pair["input"])
            key = _key(inp)
            _require(key not in seen, "Inputs must be distinct")
            seen.add(key)
            metas[split].append(_inspect_pair(pair))
    training = metas["train"]
    taught = {b[2] for m in training for b in m["bars"]}
    _require(taught == set(LENGTHS), "Training must teach all four codes")
    length_colors, color_lengths = {}, {}
    lookup = {}
    for pair, meta in zip(puzzle["train"], training):
        out = np.asarray(pair["output"])
        for row, _, length, color, _ in meta["bars"]:
            length_colors.setdefault(length, set()).add(color)
            color_lengths.setdefault(color, set()).add(length)
            digit = int(out[row, meta["column"]])
            _require(length not in lookup or lookup[length] == digit, "Inconsistent taught digit")
            lookup[length] = digit
    _require(any(len(v) >= 2 for v in length_colors.values()), "Need a length in different colors")
    _require(any(len(v) >= 2 for v in color_lengths.values()), "Need a color at different lengths")
    _require(any(b[3] == m["marker"] for m in training for b in m["bars"]), "Teach geometric marker role")
    _require(len({m["column"] for m in training}) >= 2, "Reference column must vary in training")
    _require({len(m["registers"]) for m in training} == {1, 2}, "Teach single and joint registration")
    _require({b[4] for m in training for b in m["bars"]} == {-1, 1}, "Teach both sides")
    _require(any(len(set(m["heights"])) > 1 and len(set(m["widths"])) > 1
                 for m in training), "Teach independent nonuniform expansion")
    _require(any((_physical_width(m, b) - 1) ** 2 % 10 != (b[2] - 1) ** 2 % 10
                 for m in training for b in m["bars"]), "Raw-pixel arithmetic must fail in training")
    # Predict test labels from the demonstrated lookup, without needing the
    # arithmetic formula. This establishes coverage, not an independent solver.
    for pair, meta in zip(puzzle["test"], metas["test"]):
        predicted = meta["compact"].copy()
        for row, _, length, _, _ in meta["bars"]:
            _require(length in lookup, "Untaught test length")
            predicted[row, meta["column"]] = lookup[length]
        _require(np.array_equal(predicted, pair["output"]), "Test not recoverable from taught codes")
    _require(_reversal(metas["test"][0]), "First test must reverse physical and logical width order")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming sweep: 42 seeds, all legal counts, and mutation checks."""
    shapes, compact_shapes, palettes, layouts, profiles, signatures = (set() for _ in range(6))
    register_modes, bar_counts, backgrounds = set(), set(), set()
    split_profiles = {"train": set(), "test": set()}
    stats = {name: 0 for name in ("internal_marker", "zero_bar", "multiple_marker_colors",
                                  "multiple_backgrounds", "repeated_bar_row", "large_band")}
    for seed in range(42):
        nt = 2 + seed % 9
        ns = 1 + (seed // 3) % min(4, nt - 1)
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        if seed in (0, 17, 41):
            _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed, nt, ns)), "Nondeterministic seed")
        local_markers, local_backgrounds = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                meta = _inspect_pair(pair)
                compact = meta["compact"]
                shapes.add(pair["input"].shape)
                compact_shapes.add(compact.shape)
                palettes.add(tuple(int(c) for c in np.unique(pair["input"])))
                layouts.add((compact.shape, meta["marker_row"], meta["column"],
                             tuple((r, s, length, side) for r, s, length, _, side in meta["bars"]),
                             meta["registers"]))
                profiles.add((tuple(meta["heights"]), tuple(meta["widths"])))
                split_profiles[split].add((len(set(meta["heights"])) == 1,
                                           len(set(meta["widths"])) == 1))
                register_modes.add(len(meta["registers"]))
                bar_counts.add(len(meta["bars"]))
                backgrounds.add(meta["background"])
                local_markers.add(meta["marker"])
                local_backgrounds.add(meta["background"])
                stats["internal_marker"] += min(b[0] for b in meta["bars"]) < meta["marker_row"] < max(b[0] for b in meta["bars"])
                stats["zero_bar"] += any(b[3] == 0 for b in meta["bars"])
                bar_vectors = [compact[b[0]].tobytes() for b in meta["bars"]]
                stats["repeated_bar_row"] += len(set(bar_vectors)) < len(bar_vectors)
                stats["large_band"] += max(max(meta["heights"]), max(meta["widths"])) >= 8
                # Both parent ablations differ on every pair. Omitting either
                # row or column collapse alone also has the wrong dimensions.
                _require(pair["input"].shape[0] > compact.shape[0], "Row-collapse ablation invisible")
                _require(pair["input"].shape[1] > compact.shape[1], "Column-collapse ablation invisible")
                _require(not np.array_equal(compact, pair["output"]), "Arithmetic ablation invisible")
        stats["multiple_marker_colors"] += len(local_markers) > 1
        stats["multiple_backgrounds"] += len(local_backgrounds) > 1
    _require(len(signatures) == 42, "Repeated puzzles across seeds")
    _require(len(shapes) >= 60 and len(compact_shapes) >= 35, "Insufficient shape variation")
    _require(len(layouts) >= 150 and len(profiles) >= 150 and len(palettes) >= 35,
             "Insufficient structural, scale, or palette variation")
    _require(backgrounds == set(BACKGROUNDS) and register_modes == {1, 2}, "Missing role variation")
    _require(bar_counts == set(range(1, 7)), "Missing bar counts")
    _require(all(len(v) == 4 for v in split_profiles.values()), "Missing axis-scale regimes")
    _require(all(v > 0 for v in stats.values()), f"Missing compatible variation: {stats}")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns), "Counts ignored")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Reject changes to a digit or a preserved registration cell, and duplicate
    # input evidence. Only three small copies of one pair are needed.
    puzzle = generate_puzzle(12345, 2, 1)
    meta = _inspect_pair(puzzle["train"][0])
    for r, c in ((meta["bars"][0][0], meta["column"]), (meta["registers"][0], 0)):
        bad = {s: list(puzzle[s]) for s in ("train", "test")}
        pair = {k: v.copy() for k, v in bad["train"][0].items()}
        pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
        bad["train"][0] = pair
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupted output accepted")
    bad = {s: list(puzzle[s]) for s in ("train", "test")}
    bad["test"][0] = bad["train"][0]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Duplicate input accepted")
    # solve is input-only: lists work, it does not mutate the argument, and a
    # new nonuniform expansion of the same compact scene has the same answer.
    inp = puzzle["train"][0]["input"]
    saved = inp.copy()
    _require(np.array_equal(solve(inp.tolist()), puzzle["train"][0]["output"]), "List input failed")
    _require(np.array_equal(inp, saved), "solve mutated its input")
    compact = meta["compact"]
    _require(np.array_equal(solve(compact), puzzle["train"][0]["output"]),
             "Unexpanded compact input changed the answer")
    rh = np.ones(compact.shape[0], dtype=int)
    cw = np.ones(compact.shape[1], dtype=int)
    rh[-1] = min(7, 31 - len(rh))
    cw[0] = min(8, 31 - len(cw))
    expanded = np.repeat(np.repeat(compact, rh, axis=0), cw, axis=1)
    _require(np.array_equal(solve(expanded), puzzle["train"][0]["output"]), "Rescaling changed answer")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds; all legal counts; family invariants)")
