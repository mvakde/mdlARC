"""Framed rectangular panels with a pair-local, edge-anchored accent stencil."""

import hashlib

import numpy as np


_SOURCE_INPUT_HASHES = {
    "099c645e5532eeb2190d48f0edada46c8c8f27840e902ea7fc4673d9e9fe6596",
    "fa21ac99c32cca008d7294b959c412f7a0aa7a84442f191b4d91362bb4949d95",
    "7982abf6357f7be0493adaed76e511ae11b1b6bd61eb93264a4a77f6d584a887",
    "640fad09cecaf604dbe84dc2777554560eb567a40cd13a07cfe496d7088ac77c",
}

_CORNER_SETS = (
    "abcd", "ab", "cd", "ac", "bd", "ad", "bc",
)
_MOTIFS = tuple(
    [("dot", subset, 1) for subset in _CORNER_SETS]
    + [(kind, subset, length)
       for kind in ("hbar", "vbar", "ell")
       for subset in _CORNER_SETS
       for length in (2, 3)]
    + [("run", mask, 0) for mask in range(1, 16)]
    + [("mix", variant, length)
       for variant in range(4) for length in (2, 3)]
)
_CELL_CACHE: dict[tuple, frozenset[tuple[int, int]]] = {}


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _checked_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(rng: np.random.Generator, seed: int,
                    num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        choices = list(range(max(2, test + 1), 11))
        train = choices[(seed // 4) % len(choices)]
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return int(train), int(test)


def _segments(total: int, count: int, minimum: int,
              spaced: bool, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Partition a one-dimensional interior, optionally leaving wide gutters."""
    extra = total - count * minimum - (count - 1)
    if extra < 0:
        raise ValueError("layout does not fit")
    gaps = [1] * (count - 1)
    before = after = 0
    if spaced:
        for index in range(count - 1):
            addition = int(rng.integers(0, min(4, extra) + 1))
            gaps[index] += addition
            extra -= addition
        before = int(rng.integers(0, min(3, extra) + 1))
        extra -= before
        after = int(rng.integers(0, min(3, extra) + 1))
        extra -= after
    additions = rng.multinomial(extra, np.full(count, 1.0 / count))
    lengths = [minimum + int(value) for value in additions]
    intervals = []
    cursor = before
    for index, length in enumerate(lengths):
        intervals.append((cursor, cursor + length))
        cursor += length
        if index < count - 1:
            cursor += gaps[index]
    _require(cursor + after == total, "invalid segment construction")
    return intervals


def _row_layout(rng: np.random.Generator, height: int, width: int,
                count: int, spaced: bool) -> list[tuple[int, int, int, int]]:
    options = {
        3: ((1, 2), (2, 1)),
        4: ((2, 2), (1, 3), (3, 1)),
        5: ((2, 3), (3, 2), (1, 2, 2), (2, 1, 2)),
        6: ((3, 3), (2, 2, 2), (1, 2, 3), (3, 2, 1)),
    }
    groups = options[count][int(rng.integers(len(options[count])))]
    bands = _segments(height - 2, len(groups), 5, spaced, rng)
    rectangles = []
    for (top, bottom), group_count in zip(bands, groups):
        columns = _segments(width - 2, group_count, 5, spaced, rng)
        for left, right in columns:
            row0, row1 = top + 1, bottom + 1
            if spaced and row1 - row0 > 5:
                cut = int(rng.integers(0, min(3, row1 - row0 - 5) + 1))
                row0 += cut
                cut = int(rng.integers(0, min(3, row1 - row0 - 5) + 1))
                row1 -= cut
            rectangles.append((row0, row1, left + 1, right + 1))
    return rectangles


def _hub_layout(rng: np.random.Generator, height: int, width: int,
                count: int) -> list[tuple[int, int, int, int]]:
    narrow = int(rng.integers(5, min(9, width - 10)))
    maximum_gap = min(4, width - 2 - narrow - 8)
    gap = int(rng.integers(1, maximum_gap + 1))
    small_count = count - 1
    if small_count == 2 and rng.random() < 0.45:
        small_height = min(8, (height - 3) // 2)
        rows = [(1, 1 + small_height),
                (height - 1 - small_height, height - 1)]
    else:
        rows = [(a + 1, b + 1) for a, b in
                _segments(height - 2, small_count, 5, True, rng)]
    rectangles = [(a, b, 1, 1 + narrow) for a, b in rows]
    rectangles.append((1, height - 1, 1 + narrow + gap, width - 1))
    return rectangles


def _mini_layout(rng: np.random.Generator, height: int,
                 width: int) -> list[tuple[int, int, int, int]]:
    """One small target beside two references larger in both dimensions."""
    target_size = int(rng.integers(5, 7))
    gap = int(rng.integers(1, min(4, width - target_size - 10) + 1))
    top = int(rng.integers(1, height - target_size))
    columns = (1 + target_size + gap, width - 1)
    reference_rows = _segments(height - 2, 2, 7, True, rng)
    rectangles = [(top, top + target_size, 1, 1 + target_size)]
    rectangles.extend((a + 1, b + 1, *columns) for a, b in reference_rows)
    return rectangles


def _layout(rng: np.random.Generator, regime: str) -> tuple[
        int, int, list[tuple[int, int, int, int]], int, str]:
    if regime == "source_train":
        count = int(rng.choice((3, 4, 5), p=(0.43, 0.43, 0.14)))
        height = int(rng.integers(20, 29))
        width = int(rng.integers(20, 29))
    else:
        count = int(rng.choice((3, 4, 5, 6), p=(0.21, 0.24, 0.31, 0.24)))
        height = int(rng.integers(20, 31))
        width = int(rng.integers(20, 31))
    mode_draw = rng.random()
    mode = ("mini" if count == 3 and mode_draw < 0.18
            else "hub" if count <= 4 and mode_draw < 0.45
            else "spaced" if rng.random() < 0.65 else "tiled")
    if mode == "mini":
        rectangles = _mini_layout(rng, height, width)
    elif mode == "hub":
        rectangles = _hub_layout(rng, height, width, count)
    else:
        rectangles = _row_layout(rng, height, width, count, mode == "spaced")
    if rng.random() < 0.5:
        rectangles = [(c0, c1, r0, r1) for r0, r1, c0, c1 in rectangles]
        height, width = width, height
    if mode == "mini":
        target = 0
    elif mode == "hub" and rng.random() < 0.64:
        target = count - 1
    else:
        target = int(rng.integers(count))
    return height, width, rectangles, target, mode


def _motif_cells(motif: tuple, height: int, width: int) -> frozenset[tuple[int, int]]:
    key = (motif, height, width)
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    kind, variant, length = motif
    anchors = {
        "a": (1, 1, 1, 1),
        "b": (1, width - 2, 1, -1),
        "c": (height - 2, 1, -1, 1),
        "d": (height - 2, width - 2, -1, -1),
    }
    cells: set[tuple[int, int]] = set()
    if kind in ("dot", "hbar", "vbar", "ell"):
        for anchor in variant:
            row, column, dr, dc = anchors[anchor]
            if kind in ("dot", "hbar", "ell"):
                for step in range(1 if kind == "dot" else length):
                    cells.add((row, column + dc * step))
            if kind in ("vbar", "ell"):
                for step in range(length):
                    cells.add((row + dr * step, column))
    elif kind == "run":
        if variant & 1:
            cells.update((1, col) for col in range(1, width - 1))
        if variant & 2:
            cells.update((height - 2, col) for col in range(1, width - 1))
        if variant & 4:
            cells.update((row, 1) for row in range(1, height - 1))
        if variant & 8:
            cells.update((row, width - 2) for row in range(1, height - 1))
    elif kind == "mix":
        if variant < 2:
            run_row = 1 if variant == 0 else height - 2
            cells.update((run_row, col) for col in range(1, width - 1))
            anchors_to_use = "cd" if variant == 0 else "ab"
            for anchor in anchors_to_use:
                row, column, dr, _ = anchors[anchor]
                cells.update((row + dr * step, column) for step in range(length))
        else:
            run_col = 1 if variant == 2 else width - 2
            cells.update((row, run_col) for row in range(1, height - 1))
            anchors_to_use = "bd" if variant == 2 else "ac"
            for anchor in anchors_to_use:
                row, column, _, dc = anchors[anchor]
                cells.update((row, column + dc * step) for step in range(length))
    else:
        raise ValueError("unknown stencil")
    result = frozenset(cells)
    _CELL_CACHE[key] = result
    return result


def _sample_motif(rng: np.random.Generator, prefer_simple: bool) -> tuple:
    if prefer_simple and rng.random() < 0.58:
        return ("dot", "abcd", 1)
    family = int(rng.choice((0, 1, 2), p=(0.57, 0.29, 0.14)))
    if family == 0:
        kind = ("dot", "hbar", "vbar", "ell")[int(rng.integers(4))]
        subset = _CORNER_SETS[int(rng.integers(len(_CORNER_SETS)))]
        length = 1 if kind == "dot" else int(rng.integers(2, 4))
        return kind, subset, length
    if family == 1:
        return "run", int(rng.integers(1, 16)), 0
    return "mix", int(rng.integers(4)), int(rng.integers(2, 4))


def _rotate_motif(motif: tuple) -> tuple:
    kind, variant, length = motif
    if kind in ("dot", "hbar", "vbar", "ell"):
        rotated = {"a": "b", "b": "d", "d": "c", "c": "a"}
        subset = "".join(anchor for anchor in "abcd"
                         if anchor in {rotated[value] for value in variant})
        rotated_kind = {"hbar": "vbar", "vbar": "hbar"}.get(kind, kind)
        return rotated_kind, subset, length
    if kind == "run":
        converted = 0
        for old, new in ((1, 8), (2, 4), (4, 1), (8, 2)):
            if variant & old:
                converted |= new
        return kind, converted, length
    return kind, (3, 2, 0, 1)[variant], length


def _matching_predictions(crops: list[np.ndarray], target: int,
                          accent: int) -> dict[frozenset[tuple[int, int]], tuple]:
    observations = [frozenset((int(row), int(col)) for row, col in
                   np.argwhere(crop == accent)) for crop in crops]
    predictions = {}
    for motif in _MOTIFS:
        if all(_motif_cells(motif, *crop.shape) == observations[index]
               for index, crop in enumerate(crops) if index != target):
            predicted = _motif_cells(motif, *crops[target].shape)
            predictions[predicted] = motif
    return predictions


def _generic_cells(shape: tuple[int, int], corner_mask: int,
                   horizontal: int, vertical: int,
                   edge_mask: int) -> frozenset[tuple[int, int]]:
    """Broader competing explanation using any inset corners and full edges."""
    height, width = shape
    anchors = ((1, 1, 1, 1), (1, width - 2, 1, -1),
               (height - 2, 1, -1, 1),
               (height - 2, width - 2, -1, -1))
    cells = set()
    for index, (row, col, dr, dc) in enumerate(anchors):
        if corner_mask & (1 << index):
            cells.update((row, col + dc * step) for step in range(horizontal))
            cells.update((row + dr * step, col) for step in range(vertical))
    if edge_mask & 1:
        cells.update((1, col) for col in range(1, width - 1))
    if edge_mask & 2:
        cells.update((height - 2, col) for col in range(1, width - 1))
    if edge_mask & 4:
        cells.update((row, 1) for row in range(1, height - 1))
    if edge_mask & 8:
        cells.update((row, width - 2) for row in range(1, height - 1))
    return frozenset(cells)


def _generic_predictions(crops: list[np.ndarray], target: int,
                         accent: int) -> set[frozenset[tuple[int, int]]]:
    references = [(crop.shape,
                   frozenset((int(row), int(col)) for row, col in
                             np.argwhere(crop == accent)))
                  for index, crop in enumerate(crops) if index != target]
    target_shape = crops[target].shape
    eligible_edges = 15
    for (height, width), observed in references:
        edges = (
            {(1, col) for col in range(1, width - 1)},
            {(height - 2, col) for col in range(1, width - 1)},
            {(row, 1) for row in range(1, height - 1)},
            {(row, width - 2) for row in range(1, height - 1)},
        )
        for bit, edge in enumerate(edges):
            if not edge.issubset(observed):
                eligible_edges &= ~(1 << bit)
    edge_masks = [mask for mask in range(16) if not mask & ~eligible_edges]
    answers: set[frozenset[tuple[int, int]]] = set()
    for corner_mask in range(16):
        for horizontal in range(4):
            for vertical in range(4):
                for edge_mask in edge_masks:
                    if not corner_mask and not edge_mask:
                        continue
                    model = (corner_mask, horizontal, vertical, edge_mask)
                    if all(_generic_cells(shape, *model) == observed
                           for shape, observed in references):
                        answers.add(_generic_cells(target_shape, *model))
                        if len(answers) > 1:
                            return answers
    return answers


def _paint(height: int, width: int, base: int, accent: int,
           cells: frozenset[tuple[int, int]]) -> np.ndarray:
    result = np.full((height, width), base, dtype=np.int64)
    for row, column in cells:
        result[row, column] = accent
    return result


def _make_pair(rng: np.random.Generator, regime: str,
               preferred_structure: int | None,
               requested_motif: tuple | None = None) -> tuple[dict, dict]:
    for _ in range(160):
        height, width, rectangles, target, mode = _layout(rng, regime)
        shapes = [(r1 - r0, c1 - c0) for r0, r1, c0, c1 in rectangles]
        prefer_simple = (mode == "hub" and target == len(rectangles) - 1)
        if requested_motif is not None:
            motif = requested_motif
        elif mode == "mini" and rng.random() < 0.8:
            motif = (("hbar", "vbar", "ell")[int(rng.integers(3))],
                     "abcd", 3)
        else:
            motif = _sample_motif(rng, prefer_simple)
        observed = [_motif_cells(motif, *shape) for shape in shapes]
        if not all(observed[index] for index in range(len(shapes)) if index != target):
            continue
        # The panel bases are distinct enough to identify the common accent,
        # but some panels may share a base when at least four panels are shown.
        structure = (preferred_structure if preferred_structure is not None
                     else int(rng.integers(10)))
        accent = int(rng.choice([value for value in range(10)
                                 if value != structure]))
        available = [value for value in range(10)
                     if value not in (structure, accent)]
        order = rng.permutation(available)
        bases = [int(value) for value in order[:len(rectangles)]]
        if len(rectangles) >= 4 and rng.random() < 0.47:
            second = int(rng.integers(1, len(rectangles)))
            bases[second] = bases[0]
        grid = np.full((height, width), structure, dtype=np.int64)
        crops = []
        for index, ((r0, r1, c0, c1), (panel_height, panel_width)) in enumerate(
                zip(rectangles, shapes)):
            cells = frozenset() if index == target else observed[index]
            crop = _paint(panel_height, panel_width, bases[index], accent, cells)
            grid[r0:r1, c0:c1] = crop
            crops.append(crop)
        predictions = _matching_predictions(crops, target, accent)
        if len(predictions) != 1 or observed[target] not in predictions:
            continue
        if _generic_predictions(crops, target, accent) != {observed[target]}:
            continue
        output = _paint(*shapes[target], bases[target], accent, observed[target])
        if hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest() in _SOURCE_INPUT_HASHES:
            continue
        details = {"motif": motif, "mode": mode, "target": target,
                   "panel_count": len(rectangles), "structure": structure}
        return {"input": grid, "output": output}, details
    raise ValueError("unable to construct an inferable pair")


def _extract_panels(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    height, width = grid.shape
    structure = int(grid[0, 0])
    _require(np.all(grid[0] == structure) and np.all(grid[-1] == structure)
             and np.all(grid[:, 0] == structure) and np.all(grid[:, -1] == structure),
             "incomplete structural frame")
    occupied = grid != structure
    visited = np.zeros(grid.shape, dtype=bool)
    rectangles = []
    for start_row in range(1, height - 1):
        for start_col in range(1, width - 1):
            if not occupied[start_row, start_col] or visited[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            cells = []
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for rr, cc in ((row - 1, col), (row + 1, col),
                               (row, col - 1), (row, col + 1)):
                    if 0 <= rr < height and 0 <= cc < width and occupied[rr, cc] and not visited[rr, cc]:
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            rows, cols = zip(*cells)
            r0, r1 = min(rows), max(rows) + 1
            c0, c1 = min(cols), max(cols) + 1
            _require((r1 - r0) * (c1 - c0) == len(cells)
                     and np.all(occupied[r0:r1, c0:c1]), "nonrectangular panel")
            _require(r1 - r0 >= 5 and c1 - c0 >= 5, "panel is too small")
            rectangles.append((r0, r1, c0, c1))
    rectangles.sort()
    _require(3 <= len(rectangles) <= 6, "wrong number of panels")
    return structure, rectangles


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    structure, rectangles = _extract_panels(input_grid)
    crops = [input_grid[r0:r1, c0:c1] for r0, r1, c0, c1 in rectangles]
    count = len(crops)
    presence: dict[int, set[int]] = {}
    for index, crop in enumerate(crops):
        for color in np.unique(crop):
            presence.setdefault(int(color), set()).add(index)
    candidates = [color for color, panels in presence.items()
                  if len(panels) == count - 1]
    _require(len(candidates) == 1, "shared accent is not identifiable")
    accent = candidates[0]
    targets = [index for index, crop in enumerate(crops)
               if accent not in crop]
    _require(len(targets) == 1, "target is not unique")
    target = targets[0]
    bases = []
    for index, crop in enumerate(crops):
        colors = set(int(value) for value in np.unique(crop))
        remainder = colors - {accent}
        _require(len(remainder) == 1, "panel has unrelated color")
        base = next(iter(remainder))
        bases.append(base)
        _require(colors == ({base} if index == target else {base, accent}),
                 "missing or extra accent")
    _require(all(bases.count(color) <= 2 for color in set(bases)),
             "base color overwhelms accent identification")
    _require(structure not in presence and accent != structure,
             "structural color appears inside a panel")
    predictions = _matching_predictions(crops, target, accent)
    _require(len(predictions) == 1, "references do not determine one target rendering")
    cells = next(iter(predictions))
    _require(_generic_predictions(crops, target, accent) == {cells},
             "broader corner-and-edge explanation changes the target")
    expected = _paint(*crops[target].shape, bases[target], accent, cells)
    _require(np.array_equal(expected, output_grid), "incorrect stencil transfer or crop")
    _require(not np.array_equal(input_grid, output_grid), "unchanged pair")
    return {"structure": structure, "accent": accent, "bases": bases,
            "rectangles": rectangles, "target": target,
            "target_shape": crops[target].shape,
            "reference_shapes": [crop.shape for index, crop in enumerate(crops)
                                 if index != target],
            "motif": next(iter(predictions.values()))}


def _signature(array: np.ndarray) -> tuple:
    return array.shape, array.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    # The geometric frame and shared-accent meanings are the puzzle convention.
    # Some puzzles also keep the structural literal fixed; others vary it by pair.
    fixed_structure = bool(rng.integers(2))
    preferred_structure = int(rng.integers(10)) if fixed_structure else None
    puzzle: dict[str, list] = {"train": [], "test": []}
    used = set()
    train_motifs = []
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            requested_motif = None
            if split == "test":
                relation = int(rng.choice((0, 1, 2), p=(0.29, 0.25, 0.46)))
                if relation == 0:
                    requested_motif = train_motifs[int(rng.integers(len(train_motifs)))]
                elif relation == 1:
                    rotatable = [motif for motif in train_motifs
                                 if _rotate_motif(motif) != motif]
                    if rotatable:
                        requested_motif = _rotate_motif(
                            rotatable[int(rng.integers(len(rotatable)))])
            for _ in range(20):
                child_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair, details = _make_pair(child_rng,
                                           "source_train" if split == "train" else "target_test",
                                           preferred_structure, requested_motif)
                signature = _signature(pair["input"])
                if signature not in used and not (split == "train" and
                                                 index == 1 and
                                                 details["motif"] == train_motifs[0]):
                    used.add(signature)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_motifs.append(details["motif"])
                    break
            else:
                raise ValueError("unable to make unique pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "invalid pair schema")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2
                         and np.issubdtype(grid.dtype, np.integer), "invalid grid type")
                _require(all(1 <= size <= 30 for size in grid.shape)
                         and np.all((grid >= 0) & (grid <= 9)), "invalid grid values or size")
            input_signature = _signature(pair["input"])
            output_signature = _signature(pair["output"])
            _require(input_signature not in seen_inputs, "duplicate input")
            _require((input_signature, output_signature) not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_signature)
            seen_pairs.add((input_signature, output_signature))
            digest = hashlib.sha256(pair["input"].astype(np.uint8).tobytes()).hexdigest()
            _require(digest not in _SOURCE_INPUT_HASHES, "source input copied")
            analyses.append(_analyze_pair(pair["input"], pair["output"]))
    # Every training pair demonstrates the full geometric convention. The
    # stencil itself is recovered independently within each pair, including tests.
    _require(all(len(item["reference_shapes"]) >= 2 for item in analyses),
             "insufficient reference panels")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[name], b[name])
                   for a, b in zip(first[split], second[split])
                   for name in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    seeds = range(42)
    counts = set()
    input_signatures = set()
    structures = set()
    motifs = set()
    panel_counts = set()
    structural_blocks = 0
    large_targets = 0
    compact_tests = 0
    reused_bases = 0
    variable_structure_puzzles = 0
    for seed in seeds:
        puzzle = generate_puzzle(seed)
        _require(_puzzles_equal(puzzle, generate_puzzle(seed)), "nondeterministic puzzle")
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        structures_in_puzzle = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                input_signatures.add(_signature(grid))
                item = _analyze_pair(grid, pair["output"])
                structures.add(item["structure"])
                structures_in_puzzle.add(item["structure"])
                motifs.add(item["motif"])
                panel_counts.add(len(item["rectangles"]))
                mask = grid == item["structure"]
                structural_blocks += int(np.any(mask[:-1, :-1] & mask[1:, :-1]
                                                & mask[:-1, 1:] & mask[1:, 1:]))
                large_targets += int(all(item["target_shape"][0] > h and
                                         item["target_shape"][1] > w
                                         for h, w in item["reference_shapes"]))
                reused_bases += int(len(set(item["bases"])) < len(item["bases"]))
                if split == "test":
                    compact_tests += int(max(grid.shape) <= 24)
        variable_structure_puzzles += int(len(structures_in_puzzle) > 1)
    _require(len({a for a, _ in counts}) >= 4 and len({b for _, b in counts}) >= 3,
             "automatic count diversity is insufficient")
    _require(len(input_signatures) > 160 and len(motifs) >= 12
             and panel_counts == {3, 4, 5, 6}, "structural diversity is insufficient")
    _require(4 in structures and len(structures) >= 8 and variable_structure_puzzles >= 4,
             "structural color diversity is insufficient")
    _require(structural_blocks >= 20 and large_targets >= 4
             and compact_tests >= 4 and reused_bases >= 10,
             "reviewed geometry and palette are underrepresented")
    for train, test in ((2, 1), (10, 1), (10, 4), (5, 2)):
        puzzle = generate_puzzle(1000 + train * 10 + test, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts ignored")
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(1357, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
