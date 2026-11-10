import numpy as np


_SOURCE_TEXT_PAIRS = (
    ((
        "055550000000", "050050000000", "050050000000", "055550000000",
        "000000555550", "000000500050", "000000500050", "055500500050",
        "050500555550", "055500000000", "000000000000", "000000000000",
    ), (
        "055550000000", "057750000000", "057750000000", "055550000000",
        "000000555550", "000000588850", "000000588850", "055500588850",
        "056500555550", "055500000000", "000000000000", "000000000000",
    )),
    ((
        "000000055555", "000000050005", "005550050005", "005050050005",
        "005550055555", "000000000000", "000000555500", "000000500500",
        "000000500500", "000000555500", "000000000000", "000000000000",
    ), (
        "000000055555", "000000058885", "005550058885", "005650058885",
        "005550055555", "000000000000", "000000555500", "000000577500",
        "000000577500", "000000555500", "000000000000", "000000000000",
    )),
    ((
        "000000000000", "055555000000", "050005000000", "050005000000",
        "050005000000", "055555000000", "000000000000", "000055550000",
        "000050050000", "000050050000", "000055550000", "000000000000",
    ), (
        "000000000000", "055555000000", "058885000000", "058885000000",
        "058885000000", "055555000000", "000000000000", "000055550000",
        "000057750000", "000057750000", "000055550000", "000000000000",
    )),
    ((
        "000000000000", "055555000000", "050005000000", "050005000000",
        "050005005550", "055555005050", "000000005550", "000000000000",
        "000055550000", "000050050000", "000050050000", "000055550000",
    ), (
        "000000000000", "055555000000", "058885000000", "058885000000",
        "058885005550", "055555005650", "000000005550", "000000000000",
        "000055550000", "000057750000", "000057750000", "000055550000",
    )),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_requested_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_requested_count("num_train", num_train, 2, 10)
    _validate_requested_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(test_count + 1, 11))
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(test_count + 1, 11))
    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum_test + 1))
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    palette = [int(c) for c in rng.permutation(10)]
    mode = int(rng.choice(4, p=[0.23, 0.18, 0.18, 0.41]))
    return {
        "fill_by_size": dict(zip((3, 4, 5), palette[:3])),
        "backgrounds": palette[3:3 + (2 if mode in (1, 3) else 1)],
        "borders": palette[5:5 + (2 if mode in (2, 3) else 1)],
    }


def _separated(a: tuple, b: tuple) -> bool:
    ar, ac, size_a = a[:3]
    br, bc, size_b = b[:3]
    return (ar + size_a < br or br + size_b < ar
            or ac + size_a < bc or bc + size_b < ac)


def _sample_sizes(rng: np.random.Generator, split: str,
                  index: int, bridge: bool) -> tuple[int, ...]:
    if split == "source_train":
        if index == 0:
            return (3, 4) if bridge else (3, 4, 5, int(rng.choice((3, 4, 5))))
        if index == 1:
            return (4, 5)
        if index == 2:
            size = int(rng.choice((3, 4, 5)))
            return (size,) * int(rng.integers(1, 4))
    elif split == "target_test" and index == 0:
        return (3, 5) if bridge else (3, 3, 5)

    category = int(rng.choice(5, p=[0.22, 0.24, 0.25, 0.20, 0.09]))
    if category == 0:
        return (int(rng.choice((3, 4, 5))),)
    if category == 1:
        choices = ((3, 4), (3, 5), (4, 5))
        return choices[int(rng.integers(3))]
    if category == 2:
        size = int(rng.choice((3, 4, 5)))
        return (size,) * int(rng.integers(2, 5))
    if category == 3:
        return tuple(int(x) for x in rng.choice((3, 4, 5),
                                                size=int(rng.integers(3, 7))))
    return (3, 4, 5)


def _place_frames(rng: np.random.Generator,
                  sizes: tuple[int, ...]) -> tuple[int, int, list[tuple[int, int, int]]]:
    # Grow a spatial arrangement from randomly selected existing frames. The
    # direction and secondary offset are sampled separately for each object.
    for _ in range(90):
        style = int(rng.choice(3, p=[0.40, 0.35, 0.25]))
        order = [int(sizes[int(i)]) for i in rng.permutation(len(sizes))]
        frames = [(0, 0, order[0])]
        for size in order[1:]:
            for _ in range(90):
                row, col, old_size = frames[int(rng.integers(len(frames)))]
                direction = int(rng.integers(4))
                gap = int(rng.integers(1, (3, 6, 9)[style]))
                offset = int(rng.integers(-size + 1, old_size))
                if direction == 0:
                    candidate = (row + offset, col + old_size + gap, size)
                elif direction == 1:
                    candidate = (row + offset, col - size - gap, size)
                elif direction == 2:
                    candidate = (row + old_size + gap, col + offset, size)
                else:
                    candidate = (row - size - gap, col + offset, size)
                if all(_separated(candidate, prior) for prior in frames):
                    frames.append(candidate)
                    break
            else:
                break
        if len(frames) != len(sizes):
            continue
        top = min(r for r, _, _ in frames)
        left = min(c for _, c, _ in frames)
        frames = [(r - top, c - left, s) for r, c, s in frames]
        box_h = max(r + s for r, _, s in frames)
        box_w = max(c + s for _, c, s in frames)
        if box_h > 28 or box_w > 28:
            continue

        max_margin = (2, 5, 7)[style]
        margins = [int(rng.integers(max_margin + 1)) for _ in range(4)]
        # Half the layouts touch a boundary. Other layouts have a complete
        # exterior background margin, including around a single frame.
        if rng.random() < 0.5:
            margins[int(rng.integers(4))] = 0
        else:
            margins = [max(1, value) for value in margins]
        if margins[0] + margins[1] == 0:
            margins[int(rng.integers(2))] = 1
        if margins[2] + margins[3] == 0:
            margins[2 + int(rng.integers(2))] = 1
        height = box_h + margins[0] + margins[1]
        width = box_w + margins[2] + margins[3]
        if height > 30 or width > 30:
            continue
        frames = sorted((r + margins[0], c + margins[2], s)
                        for r, c, s in frames)
        return height, width, frames
    raise ValueError("could not pack separated frames")


def _draw_pair(height: int, width: int,
               frames: list[tuple[int, int, int, int]],
               background: int, fill_by_size: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    input_grid = np.full((height, width), background, dtype=np.int8)
    for row, col, size, color in frames:
        input_grid[row, col:col + size] = color
        input_grid[row + size - 1, col:col + size] = color
        input_grid[row:row + size, col] = color
        input_grid[row:row + size, col + size - 1] = color
    output_grid = input_grid.copy()
    for row, col, size, _ in frames:
        output_grid[row + 1:row + size - 1,
                    col + 1:col + size - 1] = fill_by_size[size]
    return input_grid, output_grid


def generate_pair(rng: np.random.Generator, regime: str,
                  puzzle_parameters: dict,
                  sizes: tuple[int, ...] | None = None,
                  index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test") or not sizes:
        raise ValueError("regime and sizes are required")
    height, width, positions = _place_frames(rng, sizes)
    backgrounds = puzzle_parameters["backgrounds"]
    borders = puzzle_parameters["borders"]
    if regime == "source_train" and index < 2:
        background = backgrounds[index % len(backgrounds)]
        primary_border = borders[index % len(borders)]
    else:
        background = int(rng.choice(backgrounds))
        primary_border = int(rng.choice(borders))
    colors = [primary_border] * len(positions)
    if len(borders) > 1 and len(positions) > 1:
        colors = [int(rng.choice(borders)) for _ in positions]
        if regime == "source_train" and index == 0:
            # If this first lesson repeats a size, explicitly show that its
            # border color does not determine its cavity fill.
            for size in (3, 4, 5):
                matching = [i for i, frame in enumerate(positions)
                            if frame[2] == size]
                if len(matching) > 1:
                    colors[matching[0]] = borders[0]
                    colors[matching[1]] = borders[1]
                    break
            else:
                colors[0] = borders[0]
                colors[1] = borders[1]
    colored = [(r, c, s, color) for (r, c, s), color
               in zip(positions, colors)]
    return _draw_pair(height, width, colored, background,
                      puzzle_parameters["fill_by_size"])


def _array_signature(array: np.ndarray) -> tuple:
    return (array.shape, tuple(int(value) for value in array.flat))


def _pair_signature(pair: dict) -> tuple:
    return (_array_signature(pair["input"]), _array_signature(pair["output"]))


def _grid_as_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _is_exact_source_pair(input_grid: np.ndarray,
                          output_grid: np.ndarray) -> bool:
    return (_grid_as_rows(input_grid), _grid_as_rows(output_grid)) in _SOURCE_TEXT_PAIRS


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    bridge = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    used_inputs: set[tuple] = set()
    for split, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            sizes = _sample_sizes(rng, regime, index, bridge)
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(20):
                input_grid, output_grid = generate_pair(
                    pair_rng, regime, parameters, sizes, index)
                signature = _array_signature(input_grid)
                if (signature not in used_inputs and
                        not _is_exact_source_pair(input_grid, output_grid)):
                    break
            else:
                raise ValueError("could not construct a unique pair")
            used_inputs.add(signature)
            puzzle[split].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            cells = [(row, col)]
            seen[row, col] = True
            cursor = 0
            while cursor < len(cells):
                r, c = cells[cursor]
                cursor += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < height and 0 <= cc < width and
                            mask[rr, cc] and not seen[rr, cc]):
                        cells.append((rr, cc))
                        seen[rr, cc] = True
            result.append(cells)
    return result


def _try_parse_frames(grid: np.ndarray, background: int) -> list[tuple] | None:
    frames = []
    for cells in _components(grid != background):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        top, left = min(rows), min(cols)
        size = max(rows) - top + 1
        if size not in (3, 4, 5) or max(cols) - left + 1 != size:
            return None
        border = int(grid[top, left])
        expected = {(r, c) for r in range(top, top + size)
                    for c in range(left, left + size)
                    if r in (top, top + size - 1) or c in (left, left + size - 1)}
        if set(cells) != expected:
            return None
        if any(int(grid[r, c]) != border for r, c in cells):
            return None
        if not np.all(grid[top + 1:top + size - 1,
                           left + 1:left + size - 1] == background):
            return None
        frames.append((top, left, size, border))
    if not frames:
        return None
    frames.sort()
    if any(not _separated(a, b) for i, a in enumerate(frames)
           for b in frames[i + 1:]):
        return None
    return frames


def _parse_input_grid(grid: np.ndarray) -> tuple[int, list[tuple]]:
    parses = []
    for candidate in np.unique(grid):
        background = int(candidate)
        frames = _try_parse_frames(grid, background)
        if frames is not None:
            parses.append((background, frames))
    _require(len(parses) == 1, "input has no unique closed-frame interpretation")
    return parses[0]


def _validate_grid(grid: np.ndarray, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    height, width = grid.shape
    _require(1 <= height <= 30 and 1 <= width <= 30,
             f"{label} dimensions out of range")
    _require(np.issubdtype(grid.dtype, np.integer),
             f"{label} must have integer dtype")
    _require(bool(np.all((0 <= grid) & (grid <= 9))),
             f"{label} has an invalid color")


def _analyze_puzzle(puzzle: dict) -> dict:
    records = {"train": [], "test": []}
    inputs = set()
    pairs = set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{split}[{index}] schema is invalid")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp, f"{split}[{index}].input")
            _validate_grid(out, f"{split}[{index}].output")
            _require(inp.shape == out.shape, "input/output shape differs")
            signature = _array_signature(inp)
            pair_signature = _pair_signature(pair)
            _require(signature not in inputs and pair_signature not in pairs,
                     "duplicate input or pair")
            inputs.add(signature)
            pairs.add(pair_signature)
            _require(not _is_exact_source_pair(inp, out), "source pair copied")
            background, frames = _parse_input_grid(inp)
            records[split].append({"input": inp, "output": out,
                                   "background": background, "frames": frames})

    fill_by_size: dict[int, int] = {}
    train_backgrounds = set()
    train_borders = set()
    for record in records["train"]:
        inp, out = record["input"], record["output"]
        train_backgrounds.add(record["background"])
        train_borders.update(frame[3] for frame in record["frames"])
        changed = np.zeros(inp.shape, dtype=bool)
        for row, col, size, _ in record["frames"]:
            cavity = out[row + 1:row + size - 1,
                         col + 1:col + size - 1]
            fill = int(cavity[0, 0])
            _require(bool(np.all(cavity == fill)), "cavity is not uniform")
            _require(fill not in set(int(v) for v in np.unique(inp)),
                     "fill color coincides with input role")
            _require(size not in fill_by_size or fill_by_size[size] == fill,
                     "size-to-fill mapping changes")
            fill_by_size[size] = fill
            changed[row + 1:row + size - 1,
                    col + 1:col + size - 1] = True
        _require(np.array_equal(inp[~changed], out[~changed]),
                 "cells outside cavities changed")
    _require(set(fill_by_size) == {3, 4, 5},
             "training fails to teach every size")
    _require(len(set(fill_by_size.values())) == 3,
             "size fills must be distinct")
    _require(not (set(fill_by_size.values()) &
                  (train_backgrounds | train_borders)),
             "color roles overlap across training")

    for split in ("train", "test"):
        for record in records[split]:
            inp, out = record["input"], record["output"]
            _require(record["background"] in train_backgrounds,
                     "test background was not demonstrated")
            _require(all(frame[3] in train_borders for frame in record["frames"]),
                     "test border color was not demonstrated")
            _require(not (set(int(v) for v in np.unique(inp)) &
                          set(fill_by_size.values())),
                     "input color collides with fill role")
            expected = inp.copy()
            for row, col, size, _ in record["frames"]:
                expected[row + 1:row + size - 1,
                         col + 1:col + size - 1] = fill_by_size[size]
            _require(np.array_equal(out, expected),
                     "output violates learned absolute-size fill rule")
    _require(len({tuple(sorted(f[2] for f in r["frames"]))
                  for r in records["train"]}) >= 2,
             "training size multisets do not vary")
    return {"records": records, "fill_by_size": fill_by_size,
            "train_backgrounds": train_backgrounds,
            "train_borders": train_borders}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle schema is invalid")
    _require(isinstance(puzzle["train"], list) and
             isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and
             train_count > test_count, "invalid pair counts")
    _analyze_puzzle(puzzle)


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k])
                   for a, b in zip(first[s], second[s])
                   for k in ("input", "output"))
               for s in ("train", "test"))


def _expect_value_error(call) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    train_counts, test_counts = set(), set()
    palettes, shapes, multisets, layouts, fingerprints = set(), set(), set(), set(), set()
    test_repeated = test_omitted = edge = interior = 0
    varying_background = varying_border = mixed_border = one_size_train = 0
    layout_kinds = {"row": 0, "column": 0, "cluster": 0}
    direction_counts = {"left": 0, "right": 0, "above": 0, "below": 0}
    tight_gaps = wide_gaps = small_single = many_large = bridge_cases = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        analysis = _analyze_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_pair_signature(p) for split in ("train", "test")
                               for p in puzzle[split]))
        palettes.add(tuple(analysis["fill_by_size"][s] for s in (3, 4, 5)))
        varying_background += len(analysis["train_backgrounds"]) > 1
        varying_border += len(analysis["train_borders"]) > 1
        train_sizes = [tuple(sorted(f[2] for f in r["frames"]))
                       for r in analysis["records"]["train"]]
        first_test_sizes = tuple(sorted(f[2] for f in
                                        analysis["records"]["test"][0]["frames"]))
        bridge_cases += (train_sizes[:2] == [(3, 4), (4, 5)]
                         and first_test_sizes == (3, 5))
        for split in ("train", "test"):
            for record in analysis["records"][split]:
                inp = record["input"]
                frames = record["frames"]
                sizes = tuple(sorted(f[2] for f in frames))
                small_single += len(frames) == 1 and max(inp.shape) <= 9
                many_large += len(frames) >= 5 and max(inp.shape) >= 20
                shapes.add(inp.shape)
                multisets.add((split, sizes))
                layouts.add((split, inp.shape, tuple((r, c, s) for r, c, s, _ in frames)))
                if len(frames) >= 3:
                    vertical = [r + s / 2 for r, _, s, _ in frames]
                    horizontal = [c + s / 2 for _, c, s, _ in frames]
                    vertical_span = max(vertical) - min(vertical)
                    horizontal_span = max(horizontal) - min(horizontal)
                    if vertical_span <= 3 and horizontal_span >= 7:
                        layout_kinds["row"] += 1
                    elif horizontal_span <= 3 and vertical_span >= 7:
                        layout_kinds["column"] += 1
                    elif vertical_span >= 7 and horizontal_span >= 7:
                        layout_kinds["cluster"] += 1
                small = [f for f in frames if f[2] == 3]
                large = [f for f in frames if f[2] == 5]
                if len(small) == len(large) == 1:
                    if small[0][1] < large[0][1]:
                        direction_counts["left"] += 1
                    elif small[0][1] > large[0][1]:
                        direction_counts["right"] += 1
                    if small[0][0] < large[0][0]:
                        direction_counts["above"] += 1
                    elif small[0][0] > large[0][0]:
                        direction_counts["below"] += 1
                gaps = []
                for i, first in enumerate(frames):
                    for second in frames[i + 1:]:
                        row_gap = max(second[0] - first[0] - first[2],
                                      first[0] - second[0] - second[2], 0)
                        col_gap = max(second[1] - first[1] - first[2],
                                      first[1] - second[1] - second[2], 0)
                        gaps.append(max(row_gap, col_gap))
                if gaps:
                    tight_gaps += min(gaps) == 1
                    wide_gaps += min(gaps) >= 4
                touches = any(r == 0 or c == 0 or r + s == inp.shape[0]
                              or c + s == inp.shape[1] for r, c, s, _ in frames)
                edge += touches
                interior += not touches
                if split == "train":
                    one_size_train += len(set(sizes)) == 1
                    by_size = {}
                    for _, _, size, color in frames:
                        by_size.setdefault(size, set()).add(color)
                    mixed_border += any(len(colors) > 1 for colors in by_size.values())
                else:
                    test_repeated += len(sizes) > len(set(sizes))
                    test_omitted += len(set(sizes)) < 3
        # Recover the mapping from copies of training evidence alone.
        learned = analysis["fill_by_size"]
        for pair in puzzle["test"]:
            background, frames = _parse_input_grid(pair["input"].copy())
            _require(background in analysis["train_backgrounds"],
                     "test background is unseen")
            guess = pair["input"].copy()
            for row, col, size, _ in frames:
                guess[row + 1:row + size - 1,
                      col + 1:col + size - 1] = learned[size]
            _require(np.array_equal(guess, pair["output"]),
                     "test prediction from training failed")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(fingerprints) >= 40 and len(palettes) >= 25,
             "puzzles or fill palettes lack diversity")
    _require(len(shapes) >= 35 and len(layouts) >= 150,
             "dimensions or placements lack diversity")
    _require(len(multisets) >= 15 and test_repeated >= 10 and
             test_omitted >= 20 and one_size_train >= 10,
             "size multiset variation is too narrow")
    _require(edge >= 20 and interior >= 20,
             "boundary and interior layouts lack coverage")
    _require(varying_background >= 8 and varying_border >= 8 and
             mixed_border >= 5,
             "pair and within-pair color variation lacks coverage")
    _require(all(count >= 5 for count in layout_kinds.values()) and
             all(count >= 10 for count in direction_counts.values()),
             "rows, columns, clusters, or size directions lack coverage")
    _require(tight_gaps >= 30 and wide_gaps >= 15 and
             small_single >= 10 and many_large >= 5 and bridge_cases >= 5,
             "packing scale, gap range, or two-pair bridge lacks coverage")

    for seed, train_count, test_count in ((901, 2, 1), (902, 10, 1),
                                           (903, 10, 4), (904, None, 1),
                                           (905, None, 4), (906, 2, None),
                                           (907, 10, None), (908, 7, 3),
                                           (909, None, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        _require(_puzzles_equal(puzzle,
                 generate_puzzle(seed, train_count, test_count)),
                 "generation is not deterministic")
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count,
                     "train count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count,
                     "test count ignored")
    for call in (lambda: generate_puzzle(1, 1),
                 lambda: generate_puzzle(1, 11),
                 lambda: generate_puzzle(1, num_test=0),
                 lambda: generate_puzzle(1, num_test=5),
                 lambda: generate_puzzle(1, 2, 2),
                 lambda: generate_puzzle(1, 3, 4),
                 lambda: generate_puzzle(1, 2.0, 1)):
        _expect_value_error(call)
