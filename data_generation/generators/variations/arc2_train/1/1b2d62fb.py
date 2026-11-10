"""Generators for aligned-panel shared-empty-space ARC puzzles."""

import numpy as np


_SOURCE_ROWS = (
    (("0991909", "9001900", "9991999", "0901000", "9001900"),
     ("000", "088", "000", "808", "088")),
    (("0001900", "9091999", "0991999", "0001999", "0991999"),
     ("088", "000", "000", "000", "000")),
    (("9001909", "9001090", "9001900", "0991099", "0091090"),
     ("080", "008", "088", "800", "800")),
    (("0991909", "9091999", "9991009", "9001900", "9991009"),
     ("000", "000", "000", "088", "000")),
    (("0991999", "0091990", "9091990", "0001900", "0991999"),
     ("000", "000", "000", "088", "000")),
    (("9901090", "0991000", "9901090", "9991909", "0991099"),
     ("008", "800", "008", "000", "800")),
)
_SOURCE_PAIRS = tuple(
    (np.array([[int(c) for c in row] for row in inp], dtype=np.int8),
     np.array([[int(c) for c in row] for row in out], dtype=np.int8))
    for inp, out in _SOURCE_ROWS
)
_CASES = {0, 1, 2, 3}


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _pair_counts(rng: np.random.Generator, seed: int,
                 num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _shape(rng: np.random.Generator, orientation: str,
           both_orientations: bool = False) -> tuple[int, int]:
    max_h = 14 if orientation == "stack" or both_orientations else 30
    max_w = 14 if orientation == "side" or both_orientations else 30
    mode = int(rng.choice(5, p=[.19, .28, .18, .18, .17]))
    if mode == 0:  # smaller than the source panels
        h = int(rng.integers(3, 7))
        w = int(rng.integers(2, 6))
    elif mode == 1:
        h = int(rng.integers(5, min(12, max_h) + 1))
        w = int(rng.integers(4, min(10, max_w) + 1))
    elif mode == 2:  # tall rectangle
        h = int(rng.integers(min(10, max_h), max_h + 1))
        w = int(rng.integers(2, min(6, max_w) + 1))
    elif mode == 3:  # wide rectangle
        h = int(rng.integers(3, min(7, max_h) + 1))
        w = int(rng.integers(min(9, max_w), max_w + 1))
    else:
        h = int(rng.integers(min(9, max_h), max_h + 1))
        w = int(rng.integers(min(8, max_w), max_w + 1))
    return h, w


def _sample_mask(rng: np.random.Generator, shape: tuple[int, int],
                 regime: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if rng.random() < (.035 if regime == "source_train" else .09):
        mask.flat[int(rng.integers(mask.size))] = True
        return mask
    if rng.random() < (.07 if regime == "source_train" else .10):
        count = int(rng.integers(2, min(4, 2 * h) + 1))
        left_row = int(rng.integers(h))
        right_row = int(rng.integers(h))
        mask[left_row, 0] = True
        mask[right_row, -1] = True
        candidates = [(r, c) for r in range(h) for c in {0, w - 1}
                      if not mask[r, c]]
        for i in np.atleast_1d(rng.choice(len(candidates), size=count - 2,
                                             replace=False)):
            mask[candidates[int(i)]] = True
        return mask
    modes = ("scattered", "sparse", "dense", "edge", "interior",
             "band", "block", "line", "patch", "empty", "full")
    weights = ([.22, .13, .10, .08, .08, .09, .09, .08, .07, .03, .03]
               if regime == "source_train" else
               [.16, .11, .15, .08, .10, .09, .09, .08, .09, .03, .02])
    mode = str(rng.choice(modes, p=weights))
    n = h * w
    if mode in ("scattered", "sparse", "dense", "edge", "interior"):
        if mode == "edge":
            candidates = [(r, c) for r in range(h) for c in range(w)
                          if r in (0, h - 1) or c in (0, w - 1)]
            count = int(rng.integers(1, min(len(candidates), max(2, n // 5)) + 1))
        elif mode == "interior" and h > 2 and w > 2:
            candidates = [(r, c) for r in range(1, h - 1)
                          for c in range(1, w - 1)]
            count = int(rng.integers(1, len(candidates) + 1))
        else:
            candidates = [(r, c) for r in range(h) for c in range(w)]
            low, high = ((.12, .55) if mode == "scattered" else
                         (.01, .13) if mode == "sparse" else (.5, .88))
            count = max(1, min(n, int(round(n * rng.uniform(low, high)))))
        for i in np.atleast_1d(rng.choice(len(candidates), size=count,
                                             replace=False)):
            mask[candidates[int(i)]] = True
    elif mode == "band":
        if rng.random() < .5:
            thick = int(rng.integers(1, max(2, h // 3) + 1))
            start = int(rng.integers(0, h - thick + 1))
            mask[start:start + thick, :] = True
        else:
            thick = int(rng.integers(1, max(2, w // 3) + 1))
            start = int(rng.integers(0, w - thick + 1))
            mask[:, start:start + thick] = True
    elif mode == "block":
        bh = int(rng.integers(2, h + 1))
        bw = int(rng.integers(2, w + 1))
        r = int(rng.integers(0, h - bh + 1))
        c = int(rng.integers(0, w - bw + 1))
        mask[r:r + bh, c:c + bw] = True
    elif mode == "line":
        if rng.random() < .5:
            mask[int(rng.integers(h)), :] = True
        else:
            mask[:, int(rng.integers(w))] = True
    elif mode == "patch":
        target = int(rng.integers(2, max(3, int(.65 * n)) + 1))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        while len(cells) < target:
            frontier = {(r + dr, c + dc) for r, c in cells
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        if 0 <= r + dr < h and 0 <= c + dc < w
                        and (r + dr, c + dc) not in cells}
            if not frontier:
                break
            cells.add(sorted(frontier)[int(rng.integers(len(frontier)))])
        for cell in cells:
            mask[cell] = True
    elif mode == "full":
        mask[:] = True
    return mask


def _codes(rng: np.random.Generator, mask: np.ndarray,
           relation: str, anchor: bool) -> np.ndarray:
    # 0 empty/empty, 1 empty/occupied, 2 occupied/empty, 3 occupied/occupied.
    codes = np.zeros(mask.shape, dtype=np.int8)
    if relation == "complement":
        codes[:] = rng.choice(np.array([1, 2], dtype=np.int8), size=mask.shape)
    elif relation == "identical":
        codes[~mask] = 3
    elif relation in ("left_empty", "right_empty"):
        codes[~mask] = 1 if relation == "left_empty" else 2
    elif relation in ("left_occupied", "right_occupied"):
        codes[:] = rng.choice(np.array([2, 3] if relation == "left_occupied"
                                     else [1, 3], dtype=np.int8), size=mask.shape)
    else:
        if rng.random() < .32:
            p_left, p_right = (rng.uniform(.08, .32), rng.uniform(.68, .94))
            if rng.random() < .5:
                p_left, p_right = p_right, p_left
        else:
            p_left, p_right = rng.uniform(.12, .91, size=2)
        weights = np.array([(1 - p_left) * p_right,
                            p_left * (1 - p_right), p_left * p_right])
        weights[2] *= rng.uniform(.4, 2.5)
        weights /= weights.sum()
        count = int(np.count_nonzero(~mask))
        codes[~mask] = rng.choice(np.array([1, 2, 3], dtype=np.int8),
                                  size=count, p=weights)
    if anchor:
        flat = codes.ravel()
        zero = np.flatnonzero(flat == 0)
        if zero.size == 0:
            flat[int(rng.integers(flat.size))] = 0
        false = np.flatnonzero(flat != 0)
        if false.size < 3:
            for i in np.atleast_1d(rng.choice(np.flatnonzero(flat == 0),
                                               size=3 - false.size,
                                               replace=False)):
                flat[int(i)] = 3
            false = np.flatnonzero(flat != 0)
        for code, index in enumerate(rng.choice(false, size=3,
                                                replace=False), start=1):
            flat[int(index)] = code
    return codes


def _assemble(codes: np.ndarray, orientation: str, empty: int,
              occupied: int, divider: int, highlight: int) -> dict:
    left = np.where((codes == 2) | (codes == 3), occupied, empty)
    right = np.where((codes == 1) | (codes == 3), occupied, empty)
    h, w = codes.shape
    if orientation == "side":
        separator = np.full((h, 1), divider)
        inp = np.concatenate((left, separator, right), axis=1)
    else:
        separator = np.full((1, w), divider)
        inp = np.concatenate((left, separator, right), axis=0)
    out = np.where(codes == 0, highlight, empty)
    return {"input": inp.astype(np.int8), "output": out.astype(np.int8)}


def _key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.ravel()))


def _source_copy(inp: np.ndarray, out: np.ndarray) -> bool:
    return any(np.array_equal(inp, source_in) or
               (np.array_equal(inp, source_in) and np.array_equal(out, source_out))
               for source_in, source_out in _SOURCE_PAIRS)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    n_train, n_test = _pair_counts(rng, seed, num_train, num_test)
    palette = [int(x) for x in rng.permutation(10)]
    empty, highlight = palette[:2]
    color_counts = np.array([1, 2, 3][:min(3, n_train)])
    color_weights = np.array([.20, .40, .40][:len(color_counts)])
    color_weights /= color_weights.sum()
    n_occupied = int(rng.choice(color_counts, p=color_weights))
    n_divider = int(rng.choice(color_counts, p=color_weights))
    occupied_colors = palette[2:2 + n_occupied]
    divider_colors = palette[2 + n_occupied:2 + n_occupied + n_divider]
    orientation_mode = str(rng.choice(["side", "stack", "mixed"],
                                      p=[.38, .23, .39]))
    if orientation_mode == "mixed":
        first = "side" if rng.random() < .5 else "stack"
        orientations = [first, "stack" if first == "side" else "side"]
        orientations += [str(rng.choice(["side", "stack"]))
                         for _ in range(n_train + n_test - 2)]
    else:
        orientations = [orientation_mode] * (n_train + n_test)
    fixed_size = bool(rng.random() < .27)
    fixed_shape = (_shape(rng, orientations[0], orientation_mode == "mixed")
                   if fixed_size else None)
    anchor_index = int(rng.integers(n_train))
    repeat_index = (-1 if rng.random() >= .38 else
                    int(rng.integers(anchor_index + 1, n_train + n_test)))
    pairs: list[dict] = []
    seen_inputs: set[tuple] = set()
    anchor_mask = None
    anchor_shape = None
    for index in range(n_train + n_test):
        regime = "source_train" if index < n_train else "target_test"
        occ = (occupied_colors[index] if index < n_occupied else
               int(rng.choice(occupied_colors)))
        div = (divider_colors[index] if index < n_divider else
               int(rng.choice(divider_colors)))
        repeated = (index == repeat_index and anchor_mask is not None and
                    (anchor_shape[1] <= 14 if orientations[index] == "side"
                     else anchor_shape[0] <= 14))
        shape = (anchor_shape if repeated else fixed_shape if fixed_shape is not None
                 else _shape(rng, orientations[index]))
        for _ in range(64):
            mask = (anchor_mask.copy() if repeated else
                    _sample_mask(rng, shape, regime))
            if index == anchor_index or index < n_occupied:
                relation = "general"
                if np.all(mask):
                    mask.flat[int(rng.integers(mask.size))] = False
            elif repeated:
                relation = "general"
            else:
                relation = str(rng.choice(
                    ["general", "identical", "complement", "left_empty",
                     "right_empty", "left_occupied", "right_occupied"],
                    p=[.60, .10, .07, .06, .06, .055, .055]))
            codes = _codes(rng, mask, relation, index == anchor_index)
            pair = _assemble(codes, orientations[index], empty, occ, div, highlight)
            if _key(pair["input"]) in seen_inputs:
                continue
            if _source_copy(pair["input"], pair["output"]):
                continue
            seen_inputs.add(_key(pair["input"]))
            pairs.append(pair)
            if index == anchor_index:
                anchor_mask = codes == 0
                anchor_shape = shape
            break
        else:
            raise RuntimeError("could not construct a distinct pair")
    puzzle = {"train": pairs[:n_train], "test": pairs[n_train:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _split(inp: np.ndarray) -> tuple[str, int, np.ndarray, np.ndarray]:
    h, w = inp.shape
    candidates = []
    if w % 2 == 1 and w >= 5:
        middle = w // 2
        color = int(inp[0, middle])
        left, right = inp[:, :middle], inp[:, middle + 1:]
        if np.all(inp[:, middle] == color) and not np.any(left == color) and not np.any(right == color):
            candidates.append(("side", color, left, right))
    if h % 2 == 1 and h >= 5:
        middle = h // 2
        color = int(inp[middle, 0])
        top, bottom = inp[:middle, :], inp[middle + 1:, :]
        if np.all(inp[middle, :] == color) and not np.any(top == color) and not np.any(bottom == color):
            candidates.append(("stack", color, top, bottom))
    _require(len(candidates) == 1, "input needs one unambiguous center divider")
    return candidates[0]


def _inferred_roles(train: list[dict]) -> tuple[int, int, set[int], set[int]]:
    panel_colors: set[int] = set()
    output_colors: set[int] = set()
    occupied_colors: set[int] = set()
    divider_colors: set[int] = set()
    for pair in train:
        _, divider, first, second = _split(pair["input"])
        divider_colors.add(divider)
        panel_colors.update(int(x) for x in np.unique(first))
        panel_colors.update(int(x) for x in np.unique(second))
        output_colors.update(int(x) for x in np.unique(pair["output"]))
    shared = panel_colors & output_colors
    _require(len(output_colors) == 2 and len(shared) == 1,
             "training does not identify empty and highlight colors")
    empty = next(iter(shared))
    highlight = next(iter(output_colors - {empty}))
    occupied_colors = panel_colors - {empty}
    _require(bool(occupied_colors), "training has no occupied color")
    _require(not (divider_colors & (panel_colors | output_colors)),
             "divider colors collide with panel or output roles")
    return empty, highlight, occupied_colors, divider_colors


def _predict(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    empty, highlight, _, _ = _inferred_roles(train)
    _, _, first, second = _split(test_input)
    return np.where((first == empty) & (second == empty),
                    highlight, empty).astype(np.int8)


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
                 "puzzle must have exactly train and test")
        _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
                 "train and test must be lists")
        nt, nv = len(puzzle["train"]), len(puzzle["test"])
        _require(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv,
                 "illegal pair counts")
        seen: set[tuple] = set()
        for pair in puzzle["train"] + puzzle["test"]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{name} must be a 2D NumPy array")
                _require(all(1 <= n <= 30 for n in grid.shape),
                         f"{name} has illegal dimensions")
                _require(np.issubdtype(grid.dtype, np.integer) and
                         np.all((0 <= grid) & (grid <= 9)),
                         f"{name} has illegal cells")
            inp, out = pair["input"], pair["output"]
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            _require(not _source_copy(inp, out), "source input copied")
            _, _, first, second = _split(inp)
            _require(first.shape == second.shape == out.shape,
                     "output must match each congruent panel")
        empty, highlight, occupied_colors, divider_colors = _inferred_roles(puzzle["train"])
        training_cases: set[int] = set()
        training_orientations: set[str] = set()
        observed_occupied: set[int] = set()
        observed_dividers: set[int] = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                orientation, divider, first, second = _split(pair["input"])
                panel_values = {int(x) for x in np.unique(np.concatenate(
                    (first.ravel(), second.ravel())))}
                nonempty = panel_values - {empty}
                _require(len(nonempty) <= 1 and not (nonempty & {highlight, divider}),
                         "one occupied color per pair, distinct from output and divider")
                _require(set(int(x) for x in np.unique(pair["output"])) <= {empty, highlight},
                         "output has an unlearned color")
                expected = np.where((first == empty) & (second == empty),
                                    highlight, empty)
                _require(np.array_equal(pair["output"], expected),
                         "output is not the shared-empty mask")
                codes = (first != empty).astype(np.int8) * 2 + (second != empty).astype(np.int8)
                if section == "train":
                    training_cases.update(int(x) for x in np.unique(codes))
                    training_orientations.add(orientation)
                    observed_occupied.update(nonempty)
                    observed_dividers.add(divider)
                else:
                    _require(orientation in training_orientations,
                             "test orientation absent from training")
                    _require(nonempty <= occupied_colors,
                             "test occupied color absent from training")
                    _require(divider in divider_colors,
                             "test divider color absent from training")
                    _require(np.array_equal(_predict(puzzle["train"], pair["input"]),
                                            pair["output"]),
                             "training-only prediction failed")
        _require(training_cases == _CASES,
                 "training must collectively demonstrate all four cell cases")
        _require(observed_occupied == occupied_colors and observed_dividers == divider_colors,
                 "training role palette is inconsistent")
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"malformed puzzle: {exc}") from exc


def _direct_task_check(puzzle: dict) -> None:
    """Reconstruct the rule from grids without generation-time parameters."""
    empty, highlight, _, _ = _inferred_roles(puzzle["train"])
    cases: set[int] = set()
    train_orientations: set[str] = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            inp, out = pair["input"], pair["output"]
            h, w = out.shape
            if inp.shape == (h, 2 * w + 1):
                first, second = inp[:, :w], inp[:, w + 1:]
                divider = inp[:, w]
                orientation = "side"
            else:
                _require(inp.shape == (2 * h + 1, w), "direct shape check failed")
                first, second = inp[:h, :], inp[h + 1:, :]
                divider = inp[h, :]
                orientation = "stack"
            _require(bool(np.all(divider == divider[0])),
                     "direct divider check failed")
            expected = np.full((h, w), empty, dtype=out.dtype)
            expected[(first == empty) & (second == empty)] = highlight
            _require(np.array_equal(out, expected), "direct NOR check failed")
            if section == "train":
                train_orientations.add(orientation)
                cases.update(int(x) for x in np.unique(
                    (first != empty).astype(np.int8) * 2 +
                    (second != empty).astype(np.int8)))
            else:
                _require(orientation in train_orientations,
                         "direct orientation coverage failed")
    _require(cases == _CASES, "direct truth-table coverage failed")


def _component_count(mask: np.ndarray) -> int:
    remaining = set(zip(*np.nonzero(mask)))
    count = 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            row, column = pending.pop()
            for neighbor in ((row - 1, column), (row + 1, column),
                             (row, column - 1), (row, column + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
    return count


def _equal_puzzles(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        _direct_task_check(puzzle)
        for pair in puzzle["test"]:
            _require(np.array_equal(_predict(puzzle["train"], pair["input"]),
                                    pair["output"]), "prediction check failed")
    _require(len({len(p["train"]) for p in puzzles}) >= 4,
             "train count diversity is insufficient")
    _require(len({len(p["test"]) for p in puzzles}) >= 3,
             "test count diversity is insufficient")
    shapes = {pair["output"].shape for p in puzzles for pair in p["train"] + p["test"]}
    orientations = {_split(pair["input"])[0] for p in puzzles
                    for pair in p["train"] + p["test"]}
    _require(len(shapes) >= 35 and orientations == {"side", "stack"},
             "shape or orientation diversity is insufficient")
    _require(any(len({pair["output"].shape for pair in p["train"] + p["test"]}) > 1
                 for p in puzzles), "within-puzzle size variation is absent")
    review_counts = {name: 0 for name in (
        "test_zero", "test_singleton", "test_dense", "test_interior",
        "test_one_edge", "test_edge_only", "test_sparse_edge_both",
        "test_adjacent", "test_full_line", "test_scattered",
        "test_rectangle", "test_irregular_connected",
        "test_identical", "test_complement", "test_uniform_panel",
        "test_asymmetric", "pair_missing_case", "repeated_output",
        "mixed_orientation", "fixed_shape", "varied_shape", "new_test_shape",
        "occupied_color_variation", "divider_color_variation",
        "combined_variation", "large_panel", "small_panel")}
    for puzzle in puzzles:
        empty, highlight, _, _ = _inferred_roles(puzzle["train"])
        all_pairs = puzzle["train"] + puzzle["test"]
        pair_shapes = {pair["output"].shape for pair in all_pairs}
        orientations_seen = set()
        occupied_seen = set()
        divider_seen = set()
        output_keys = [_key(pair["output"]) for pair in all_pairs]
        if len(output_keys) != len(set(output_keys)):
            review_counts["repeated_output"] += 1
        if len(pair_shapes) == 1:
            review_counts["fixed_shape"] += 1
        else:
            review_counts["varied_shape"] += 1
        if any(pair["output"].shape not in
               {x["output"].shape for x in puzzle["train"]}
               for pair in puzzle["test"]):
            review_counts["new_test_shape"] += 1
        for section in ("train", "test"):
            for pair in puzzle[section]:
                orientation, divider, first, second = _split(pair["input"])
                mask = pair["output"] == highlight
                h, w = mask.shape
                n, k = mask.size, int(np.count_nonzero(mask))
                orientations_seen.add(orientation)
                divider_seen.add(divider)
                occupied_seen.update(set(int(x) for x in np.unique(first)) - {empty})
                occupied_seen.update(set(int(x) for x in np.unique(second)) - {empty})
                review_counts["large_panel"] += max(h, w) >= 18
                review_counts["small_panel"] += n <= 12
                binary_cases = set(int(x) for x in np.unique(
                    (first != empty).astype(np.int8) * 2 +
                    (second != empty).astype(np.int8)))
                review_counts["pair_missing_case"] += len(binary_cases) < 4
                if section == "test":
                    row_indices, column_indices = np.nonzero(mask)
                    rectangle = (k >= 4 and len(set(row_indices)) >= 2 and
                                 len(set(column_indices)) >= 2 and
                                 k == (int(row_indices.max()) - int(row_indices.min()) + 1) *
                                 (int(column_indices.max()) - int(column_indices.min()) + 1))
                    component_count = _component_count(mask)
                    review_counts["test_zero"] += k == 0
                    review_counts["test_singleton"] += k == 1
                    review_counts["test_dense"] += k / n >= .4
                    review_counts["test_interior"] += (
                        bool(np.any(mask[1:-1, 1:-1])) and
                        not bool(np.any(mask[0, :]) or np.any(mask[-1, :]) or
                                 np.any(mask[:, 0]) or np.any(mask[:, -1])))
                    review_counts["test_one_edge"] += bool(np.any(mask[:, 0])) != bool(np.any(mask[:, -1]))
                    review_counts["test_edge_only"] += k > 0 and not bool(np.any(mask[:, 1:-1]))
                    review_counts["test_sparse_edge_both"] += (
                        2 <= k <= 4 and not bool(np.any(mask[:, 1:-1])) and
                        bool(np.any(mask[:, 0])) and bool(np.any(mask[:, -1])))
                    review_counts["test_adjacent"] += bool(
                        np.any(mask[:, :-1] & mask[:, 1:]) or
                        np.any(mask[:-1, :] & mask[1:, :]))
                    review_counts["test_full_line"] += bool(
                        np.any(np.all(mask, axis=0)) or np.any(np.all(mask, axis=1)))
                    review_counts["test_scattered"] += component_count >= 2 and k >= 3
                    review_counts["test_rectangle"] += rectangle
                    review_counts["test_irregular_connected"] += (
                        component_count == 1 and k >= 4 and not rectangle)
                    review_counts["test_identical"] += bool(np.array_equal(first, second))
                    review_counts["test_complement"] += bool(np.all(first != second))
                    review_counts["test_uniform_panel"] += (
                        len(np.unique(first)) == 1 or len(np.unique(second)) == 1)
                    review_counts["test_asymmetric"] += (
                        abs(float(np.mean(first != empty)) -
                            float(np.mean(second != empty))) >= .30)
        review_counts["mixed_orientation"] += len(orientations_seen) > 1
        review_counts["occupied_color_variation"] += len(occupied_seen) > 1
        review_counts["divider_color_variation"] += len(divider_seen) > 1
        review_counts["combined_variation"] += (
            len(orientations_seen) > 1 and len(pair_shapes) > 1 and
            len(occupied_seen) > 1 and len(divider_seen) > 1)
    _require(all(count > 0 for count in review_counts.values()),
             f"review feature absent from 42 seeds: {review_counts}")
    _require(_equal_puzzles(puzzles[17], generate_puzzle(17)),
             "automatic generation is not deterministic")
    _require(not _equal_puzzles(puzzles[17], puzzles[18]),
             "different seeds produced the same puzzle")
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        p = generate_puzzle(1000 + i, train, test)
        validate_generated_puzzle(p)
        _require((len(p["train"]), len(p["test"])) == (train, test),
                 "explicit counts were ignored")
        _require(_equal_puzzles(p, generate_puzzle(1000 + i, train, test)),
                 "explicit generation is not deterministic")
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000 + i, train, test)
        validate_generated_puzzle(p)
        _require((train is None or len(p["train"]) == train) and
                 (test is None or len(p["test"]) == test),
                 "partial count was ignored")
    for i, (train, test) in enumerate(((1, None), (11, None), (None, 0),
                                      (None, 5), (2, 2), (3, 4), (4, 4), (2, 3))):
        try:
            generate_puzzle(3000 + i, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal count request did not raise ValueError")
