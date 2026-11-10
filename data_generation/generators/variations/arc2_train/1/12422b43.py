from __future__ import annotations

import numpy as np


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(x) for x in row] for row in rows], dtype=np.int8)


# Exact source pairs are kept as rule fixtures and excluded from generation.
_SOURCE_RAW = (
    (("500330", "500320", "500230", "500880", "000880", "000000",
      "000000", "000000", "000000", "000000", "000000", "000000", "000000"),
     ("500330", "500320", "500230", "500880", "000880", "000330",
      "000320", "000230", "000880", "000330", "000320", "000230", "000880")),
    (("5088000", "5007000", "5004400", "0033000", "0011000", "0000000",
      "0000000", "0000000"),
     ("5088000", "5007000", "5004400", "0033000", "0011000", "0088000",
      "0007000", "0004400")),
    (("5004400", "5088800", "0002000", "0003300", "0044000", "0000000",
      "0000000", "0000000", "0000000"),
     ("5004400", "5088800", "0002000", "0003300", "0044000", "0004400",
      "0088800", "0004400", "0088800")),
    (("506800", "008300", "000000", "000000", "000000", "000000", "000000"),
     ("506800", "008300", "006800", "006800", "006800", "006800", "006800")),
    (("50600", "54440", "00600", "00000", "00000"),
     ("50600", "54440", "00600", "00600", "04440")),
    (("5044400", "5008000", "5006000", "0022000", "0000000",
      "0000000", "0000000", "0000000", "0000000", "0000000"),
     ("5044400", "5008000", "5006000", "0022000", "0044400",
      "0008000", "0006000", "0044400", "0008000", "0006000")),
)
_SOURCE_SIGNATURES = frozenset(
    (_signature(_grid(a)), _signature(_grid(b))) for a, b in _SOURCE_RAW
)


def _validate_count(value: int | None, name: str, low: int, high: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _canonical_info(grid: np.ndarray) -> dict:
    """Recognize a downward, left-marker instance using its input alone."""
    h, w = grid.shape
    _require(h >= 2 and w >= 4, "Canvas is too small")
    bg = int(grid[-1, 0])
    _require(bool(np.all(grid[-1] == bg)), "No blank destination edge")
    suffix = 0
    while suffix < h and bool(np.all(grid[h - suffix - 1] == bg)):
        suffix += 1
    start = h - suffix
    _require(start >= 1, "No occupied prefix")
    selector = int(grid[0, 0])
    _require(selector != bg, "Selector equals background")
    count = 0
    while count < start and int(grid[count, 0]) == selector:
        count += 1
    _require(count >= 1, "No selected rows")
    _require(bool(np.all(grid[count:start, 0] == bg)), "Unmarked strip is occupied")
    positions = np.argwhere(grid == selector)
    _require(positions.shape == (count, 2), "Selector is not exclusive")
    _require(bool(np.array_equal(positions[:, 0], np.arange(count))), "Selectors are not consecutive")
    _require(bool(np.all(positions[:, 1] == 0)), "Selectors are not aligned")
    _require(bool(np.all(np.any(grid[:start, 1:] != bg, axis=1))), "Empty content row")
    return {
        "background": bg, "selector": selector, "templates": count,
        "unmarked": start - count, "suffix": suffix, "width": w, "height": h,
        "start": start,
    }


def _recognize(grid: np.ndarray) -> dict:
    candidates = []
    for turns in range(4):
        rotated = np.rot90(grid, turns)
        try:
            info = _canonical_info(rotated)
        except AssertionError:
            continue
        info["turns_to_canonical"] = turns
        candidates.append(info)
    _require(len(candidates) == 1, "Input orientation or selector is ambiguous")
    return candidates[0]


def _predict(input_grid: np.ndarray) -> np.ndarray:
    info = _recognize(input_grid)
    turns = info["turns_to_canonical"]
    source = np.rot90(input_grid, turns)
    result = source.copy()
    templates = source[:info["templates"]].copy()
    templates[:, 0] = info["background"]
    for offset in range(info["suffix"]):
        result[info["start"] + offset] = templates[offset % info["templates"]]
    return np.rot90(result, -turns).copy()


def _choose(rng: np.random.Generator, values: list[int] | tuple[int, ...]) -> int:
    return int(values[int(rng.integers(len(values)))])


def _payload_row(
    rng: np.random.Generator, width: int, bg: int, palette: tuple[int, ...],
    style: str | None = None,
) -> np.ndarray:
    row = np.full(width, bg, dtype=np.int8)
    if style is None:
        styles = ("solid", "solid", "multi", "gapped", "scatter")
        style = styles[int(rng.integers(len(styles)))]
    if style == "multi" and len(palette) < 2:
        style = "gapped"
    if style in ("solid", "multi"):
        length = int(rng.integers(1 if style == "solid" else 2, width))
        left = int(rng.integers(1, width - length + 1))
        row[left:left + length] = _choose(rng, palette)
        if style == "multi":
            first, second = rng.choice(palette, size=2, replace=False)
            row[left] = int(first)
            row[left + length - 1] = int(second)
            for pos in range(left + 1, left + length - 1):
                row[pos] = _choose(rng, palette)
    elif style == "gapped":
        a, b = rng.choice(np.arange(1, width), size=2, replace=False)
        row[int(a)] = _choose(rng, palette)
        row[int(b)] = _choose(rng, palette)
        if abs(int(a) - int(b)) == 1 and width >= 4:
            row[int(a)] = bg
            row[1 if int(b) != 1 else width - 1] = _choose(rng, palette)
    else:
        occupied = rng.random(width - 1) < float(rng.uniform(0.3, 0.8))
        if not np.any(occupied):
            occupied[int(rng.integers(width - 1))] = True
        for pos in np.flatnonzero(occupied) + 1:
            row[pos] = _choose(rng, palette)
    return row


def _variant(
    rng: np.random.Generator, source: np.ndarray, bg: int,
    palette: tuple[int, ...],
) -> np.ndarray:
    row = source.copy()
    mode = int(rng.integers(3))
    if mode == 0:
        row[1:] = np.roll(row[1:], int(rng.choice((-1, 1))))
    elif mode == 1:
        occupied = np.flatnonzero(row[1:] != bg) + 1
        row[_choose(rng, occupied.tolist())] = _choose(rng, palette)
    else:
        pos = int(rng.integers(1, row.size))
        row[pos] = bg if row[pos] != bg else _choose(rng, palette)
    if not np.any(row[1:] != bg):
        row[int(rng.integers(1, row.size))] = _choose(rng, palette)
    return row


def _sample_length(
    rng: np.random.Generator, templates: int, unmarked: int, regime: str,
    anchor: bool,
) -> int:
    maximum = 30 - templates - unmarked
    if anchor:
        return int(rng.integers(templates + 1, min(maximum, 3 * templates + 5) + 1))
    mode = int(rng.integers(5))
    if mode == 0 and templates > 1:
        return int(rng.integers(1, templates))
    if mode == 1:
        return templates
    if mode == 2:
        return min(maximum, int(rng.integers(templates + 1, 2 * templates + 2)))
    if mode == 3:
        return int(rng.integers(1, maximum + 1))
    if regime == "target_test":
        return int(rng.integers(max(1, maximum // 2), maximum + 1))
    return maximum


def _make_pair(
    rng: np.random.Generator, role: tuple[int, int], turns: int,
    regime: str, anchor: bool, shift: bool, novel: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    bg, selector = role
    templates = 3 if shift else int(rng.integers(2 if anchor else 1, 8))
    unmarked = int(rng.integers(1 if anchor or shift else 0, 6))
    width = int(rng.integers(4, 31))
    if shift:
        suffix = 3 if regime == "source_train" else 6
    else:
        suffix = _sample_length(rng, templates, unmarked, regime, anchor)
    available = [c for c in range(10) if c not in (bg, selector) and (novel is None or c != novel)]
    if novel is not None and regime == "source_train":
        available = [c for c in available if c != novel]
    if novel is not None and regime == "target_test":
        base_size = int(rng.integers(0, 4))
        palette = tuple([novel] + [int(c) for c in rng.choice(available, size=base_size, replace=False)])
    else:
        palette = tuple(int(c) for c in rng.choice(available, size=int(rng.integers(1, 6)), replace=False))
    rows = []
    motif = bool(rng.integers(2))
    for idx in range(templates):
        if idx == 0 and anchor and len(palette) >= 2:
            row = _payload_row(rng, width, bg, palette, "multi")
        elif idx and motif and rng.random() < 0.65:
            row = _variant(rng, rows[0], bg, palette)
        else:
            row = _payload_row(rng, width, bg, palette)
        rows.append(row)
    if templates >= 3 and rng.random() < 0.30:
        rows[-1] = rows[0].copy()
    if templates >= 2 and all(np.array_equal(row, rows[0]) for row in rows[1:]):
        rows[-1] = _variant(rng, rows[-1], bg, palette)
        if np.array_equal(rows[-1], rows[0]):
            rows[-1] = _payload_row(rng, width, bg, palette, "solid")
    if novel is not None and regime == "target_test":
        occupied = np.flatnonzero(rows[0][1:] != bg) + 1
        rows[0][int(occupied[0])] = novel
    extras = []
    for _ in range(unmarked):
        chance = rng.random()
        if chance < 0.30:
            extra = rows[int(rng.integers(templates))].copy()
        elif chance < 0.65:
            extra = _variant(rng, rows[int(rng.integers(templates))], bg, palette)
        else:
            extra = _payload_row(rng, width, bg, palette)
        extras.append(extra)
    h = templates + unmarked + suffix
    source = np.full((h, width), bg, dtype=np.int8)
    for idx, row in enumerate(rows):
        source[idx] = row
        source[idx, 0] = selector
    for idx, row in enumerate(extras):
        source[templates + idx] = row
    result = source.copy()
    for idx in range(suffix):
        result[templates + unmarked + idx] = rows[idx % templates]
    return np.rot90(source, -turns).copy(), np.rot90(result, -turns).copy()


def _roles(
    rng: np.random.Generator, colors: list[int], variable_bg: bool,
    variable_selector: bool, split: str, index: int,
) -> tuple[int, int]:
    if variable_bg and split == "train" and index < 2:
        bg = colors[0 if index == 0 else 3]
    elif variable_bg:
        bg = _choose(rng, [c for c in colors if c != colors[2] and c != colors[1]])
    else:
        bg = colors[0]
    if variable_selector and split == "train" and index < 2:
        selector = colors[1 if index == 0 else 4]
    elif variable_selector:
        selector = _choose(rng, [c for c in colors if c not in (bg, colors[2])])
    else:
        selector = colors[1]
    if bg == selector:
        bg = _choose(rng, [c for c in colors if c not in (selector, colors[2])])
    return bg, selector


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    _validate_count(num_train, "num_train", 2, 10)
    _validate_count(num_test, "num_test", 1, 4)
    rng = np.random.default_rng(int(seed))
    if num_train is None and num_test is None:
        test_count = 1 + int(seed) % 4
        train_count = max(2, test_count + 1) + (int(seed) // 4) % (11 - max(2, test_count + 1))
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    colors = [int(c) for c in rng.permutation(10)]
    variable_bg = bool(rng.random() < 0.45)
    variable_selector = bool(rng.random() < 0.45)
    novel = colors[2] if rng.random() < 0.40 else None
    orientation_count = 1 if rng.random() < 0.38 else int(rng.integers(2, min(4, train_count) + 1))
    orientations = [int(x) for x in rng.choice(4, size=orientation_count, replace=False)]
    train_orientations = orientations + [
        _choose(rng, orientations) for _ in range(train_count - orientation_count)
    ]
    rng.shuffle(train_orientations)
    anchor_index = int(rng.integers(train_count))
    shift = bool(rng.random() < 0.27)
    shift_train = int((anchor_index + 1 + rng.integers(train_count - 1)) % train_count) if shift else -1
    puzzle = {"train": [], "test": []}
    used_inputs: set[tuple] = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            role = _roles(rng, colors, variable_bg, variable_selector, split, index)
            turns = train_orientations[index] if split == "train" else _choose(rng, orientations)
            is_anchor = split == "train" and index == anchor_index
            is_shift = shift and ((split == "train" and index == shift_train)
                                  or (split == "test" and index == 0))
            accepted = False
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                regime = "source_train" if split == "train" else "target_test"
                inp, out = _make_pair(pair_rng, role, turns, regime, is_anchor, is_shift, novel)
                signature = _signature(inp)
                if signature in used_inputs or (signature, _signature(out)) in _SOURCE_SIGNATURES:
                    continue
                try:
                    if not np.array_equal(_predict(inp), out):
                        continue
                except AssertionError:
                    continue
                puzzle[split].append({"input": inp, "output": out})
                used_inputs.add(signature)
                accepted = True
                break
            if not accepted:
                raise RuntimeError("Could not construct an unambiguous unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Bad puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "Bad split lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "Bad pair counts")
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Bad pair schema")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "Grid must be 2D NumPy")
                _require(np.issubdtype(grid.dtype, np.integer), "Grid must use integers")
                _require(all(1 <= side <= 30 for side in grid.shape), "Grid exceeds ARC bounds")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "Grid has illegal colors")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape, "Dimensions changed")
            insig, outsig = _signature(inp), _signature(out)
            _require(insig not in seen_inputs and (insig, outsig) not in seen_pairs, "Duplicate pair")
            _require((insig, outsig) not in _SOURCE_SIGNATURES, "Exact source copy")
            seen_inputs.add(insig)
            seen_pairs.add((insig, outsig))
            info = _recognize(inp)
            predicted = _predict(inp)
            _require(bool(np.array_equal(predicted, out)), "Wrong ordered marker-free cyclic copy")
            _require(not bool(np.array_equal(inp, out)), "No change")
            _require(set(np.unique(inp)) == set(np.unique(out)), "Color set changed")
            infos[split].append(info)
    train = infos["train"]
    tests = infos["test"]
    _require(any(x["templates"] >= 2 and x["unmarked"] >= 1 and x["suffix"] > x["templates"] for x in train),
             "Training lacks wraparound with an unmarked row")
    train_orientations = {x["turns_to_canonical"] for x in train}
    _require({x["turns_to_canonical"] for x in tests} <= train_orientations,
             "Test orientation is not demonstrated")
    for role in ("background", "selector"):
        values = {x[role] for x in train}
        if len(values) == 1:
            _require({x[role] for x in tests} <= values, "Undemonstrated role variation")
    for pair in puzzle["train"] + puzzle["test"]:
        info = _recognize(pair["input"])
        _require(info["templates"] + info["unmarked"] + info["suffix"] == info["height"],
                 "Invalid region lengths")


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, tuple((_signature(p["input"]), _signature(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(9100 + (a or 0) * 10 + (b or 0), a, b))
    for a, b in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    for a, b in _SOURCE_RAW:
        inp, out = _grid(a), _grid(b)
        _require(bool(np.array_equal(_predict(inp), out)), "Source fixture was rejected")
    counterfactual = _grid(_SOURCE_RAW[-1][0])
    counterfactual[2, 0] = 0
    changed = _predict(counterfactual)
    first_two = counterfactual[:2].copy()
    first_two[:, 0] = 0
    _require(bool(np.array_equal(changed[:4], counterfactual[:4])) and
             all(np.array_equal(changed[4 + i], first_two[i % 2]) for i in range(6)),
             "Counterfactual selector removal did not produce ABABAB")
    puzzles = [generate_puzzle(12000 + i) for i in range(42)]
    _require(_puzzle_signature(generate_puzzle(8123, 6, 2)) ==
             _puzzle_signature(generate_puzzle(8123, 6, 2)), "Explicit generation varies")
    _require(_puzzle_signature(generate_puzzle(8124)) ==
             _puzzle_signature(generate_puzzle(8124)), "Automatic generation varies")
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, "Repeated puzzles")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "Too few train counts")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "Too few test counts")
    infos = [_recognize(pair["input"]) for p in puzzles for pair in p["train"] + p["test"]]
    _require({x["turns_to_canonical"] for x in infos} == {0, 1, 2, 3}, "Missing orientation")
    _require(max(x["templates"] for x in infos) >= 6, "Missing large template block")
    _require(max(x["unmarked"] for x in infos) >= 4, "Missing many unmarked rows")
    _require(max(x["suffix"] for x in infos) >= 15, "Missing long suffix")
    _require(any(x["unmarked"] == 0 for x in infos), "Missing zero unmarked rows")
    _require(any(x["suffix"] < x["templates"] for x in infos), "Missing partial cycle")
    _require(len({(x["height"], x["width"]) for x in infos}) >= 60, "Too few layouts")
    coverage = dict.fromkeys(("multicolor", "gapped", "long", "solid", "selected_repeat",
                              "cross_repeat", "near_copy", "off_phase", "narrow_test"), 0)
    novel_seeds = mixed_seeds = variable_bg_seeds = variable_selector_seeds = shift_seeds = 0
    for p in puzzles:
        validate_generated_puzzle(p)
        train_infos, test_infos = [], []
        train_colors = set()
        for split in ("train", "test"):
            for pair in p[split]:
                inp = pair["input"]
                info = _recognize(inp)
                normalized = np.rot90(inp, info["turns_to_canonical"])
                bg = info["background"]
                templates = normalized[:info["templates"]].copy()
                templates[:, 0] = bg
                extras = normalized[info["templates"]:info["start"]]
                rows = list(templates) + list(extras)
                colors = set(int(c) for c in np.unique(inp)) - {bg, info["selector"]}
                if split == "train":
                    train_infos.append(info)
                    train_colors.update(colors)
                else:
                    test_infos.append((info, colors))
                    coverage["narrow_test"] += len(colors) <= 2
                coverage["multicolor"] += any(len(set(int(c) for c in row if c != bg)) > 1
                                               for row in rows)
                coverage["gapped"] += any(len(pos := np.flatnonzero(row != bg)) >= 2 and
                                          pos[-1] - pos[0] + 1 > len(pos) for row in rows)
                coverage["long"] += any(any(np.all(row[j:j + 5] != bg)
                                             for j in range(1, len(row) - 4)) for row in rows)
                coverage["solid"] += any(len(set(int(c) for c in row if c != bg)) == 1 and
                                           len(pos := np.flatnonzero(row != bg)) == pos[-1] - pos[0] + 1
                                           for row in rows)
                coverage["selected_repeat"] += len({tuple(row) for row in templates}) < len(templates)
                coverage["cross_repeat"] += any(np.array_equal(a, b) for a in templates for b in extras)
                coverage["near_copy"] += any(1 <= np.count_nonzero(a != b) <= 2
                                             for a in templates for b in extras)
                coverage["off_phase"] += info["unmarked"] > 0 and info["unmarked"] % info["templates"] != 0
        novel_seeds += any(bool(colors - train_colors) for _, colors in test_infos)
        mixed_seeds += len({x["turns_to_canonical"] for x in train_infos}) > 1
        variable_bg_seeds += len({x["background"] for x in train_infos}) > 1
        variable_selector_seeds += len({x["selector"] for x in train_infos}) > 1
        shift_seeds += (any(x["templates"] == 3 and x["suffix"] == 3 for x in train_infos)
                        and any(x["templates"] == 3 and x["suffix"] == 6 for x, _ in test_infos))
    _require(all(coverage[key] >= 10 for key in coverage), "Missing row, palette, or phase variety")
    _require(all(x >= 3 for x in (novel_seeds, mixed_seeds, variable_bg_seeds,
                                 variable_selector_seeds, shift_seeds)),
             "Missing puzzle-level regime variety")
