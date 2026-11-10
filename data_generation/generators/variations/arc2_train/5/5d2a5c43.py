import numpy as np


_SOURCE_PAIR_SIGNATURES = {
    (
        (
            "004410044",
            "044410000",
            "040014040",
            "044414404",
            "044414044",
            "040414000",
        ),
        ("0088", "0888", "8880", "8888", "8888", "8808"),
    ),
    (
        (
            "000414400",
            "004410400",
            "404410440",
            "444014400",
            "404414004",
            "000014444",
        ),
        ("8808", "0888", "8888", "8880", "8088", "8888"),
    ),
    (
        (
            "400410404",
            "004410400",
            "404414040",
            "040414044",
            "440410440",
            "044410404",
        ),
        ("8808", "0888", "8088", "8888", "8888", "0888"),
    ),
    (
        (
            "404014044",
            "404014440",
            "440414040",
            "040014004",
            "004414440",
            "440414000",
        ),
        ("8088", "8880", "8888", "8808", "8888", "8808"),
    ),
    (
        (
            "444410000",
            "040414000",
            "400010400",
            "044010000",
            "404014444",
            "044410000",
        ),
        ("8888", "8808", "8800", "0880", "8888", "0888"),
    ),
    (
        (
            "444010044",
            "440010004",
            "400410400",
            "044410444",
            "040410040",
            "004010440",
        ),
        ("8888", "8808", "8808", "0888", "0888", "0880"),
    ),
    (
        (
            "004010040",
            "404414404",
            "000414404",
            "404014440",
            "040014444",
            "440410404",
        ),
        ("0080", "8888", "8808", "8880", "8888", "8808"),
    ),
}

_TEST_STYLES = ("overlap", "complement", "asymmetry")


def _rows_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _is_original_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return (
        _rows_signature(input_grid),
        _rows_signature(output_grid),
    ) in _SOURCE_PAIR_SIGNATURES


def _checked_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}"
        )
    return value


def _resolve_counts(
    seed_token: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_count(
        num_train,
        "num_train",
        2,
        10,
    )
    num_test = _checked_count(
        num_test,
        "num_test",
        1,
        4,
    )

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return num_train, num_test

    if num_train is None and num_test is None:
        legal = [
            (train_count, test_count)
            for train_count in range(2, 11)
            for test_count in range(
                1,
                min(4, train_count - 1) + 1,
            )
        ]
        return legal[seed_token % len(legal)]

    if num_train is None:
        legal_train = np.arange(
            max(2, num_test + 1),
            11,
            dtype=int,
        )
        if legal_train.size == 0:
            raise ValueError(
                "no legal num_train exists for num_test"
            )
        index = int(rng.integers(legal_train.size))
        num_train = int(legal_train[index])
    else:
        largest_test = min(4, num_train - 1)
        if largest_test < 1:
            raise ValueError(
                "no legal num_test exists for num_train"
            )
        num_test = int(
            rng.integers(1, largest_test + 1)
        )

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = rng.choice(
        10,
        size=4,
        replace=False,
    )

    return {
        "height": int(rng.integers(4, 10)),
        "panel_width": int(rng.integers(3, 9)),
        "background": int(colors[0]),
        "foreground": int(colors[1]),
        "separator": int(colors[2]),
        "highlight": int(colors[3]),
    }


def _mask_counts(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[int, int, int, int]:
    empty = int(
        np.count_nonzero(~left & ~right)
    )
    left_only = int(
        np.count_nonzero(left & ~right)
    )
    right_only = int(
        np.count_nonzero(~left & right)
    )
    overlap = int(
        np.count_nonzero(left & right)
    )

    return (
        empty,
        left_only,
        right_only,
        overlap,
    )


# All panel arrangements below share the same literal cellwise union rule.
_RELATIONS = (
    "ordinary", "blank", "identical", "contained", "complementary",
    "sparse", "full", "disjoint", "overlap", "asymmetry",
    "shifted", "ordinary",
)


def _shape(rng, height, width, style=None):
    """Compose one occupancy mask from cells, connected pieces, or motifs."""
    if style is None:
        style = int(rng.integers(8))
    mask = np.zeros((height, width), dtype=bool)
    if style == 0:  # scattered cells at independently sampled density
        mask = rng.random((height, width)) < float(rng.uniform(.18, .68))
    elif style == 1:  # one or several solid blocks
        for _ in range(int(rng.integers(1, 4))):
            y0 = int(rng.integers(height))
            x0 = int(rng.integers(width))
            y1 = int(rng.integers(y0 + 1, height + 1))
            x1 = int(rng.integers(x0 + 1, width + 1))
            mask[y0:y1, x0:x1] = True
    elif style == 2:  # thin horizontal and vertical strokes
        for _ in range(int(rng.integers(1, 5))):
            if int(rng.integers(2)):
                y = int(rng.integers(height))
                x = int(rng.integers(width))
                mask[y, x:int(rng.integers(x + 1, width + 1))] = True
            else:
                x = int(rng.integers(width))
                y = int(rng.integers(height))
                mask[y:int(rng.integers(y + 1, height + 1)), x] = True
    elif style == 3:  # separated islands
        for _ in range(int(rng.integers(2, 7))):
            y = int(rng.integers(height))
            x = int(rng.integers(width))
            mask[y:min(height, y + int(rng.integers(1, 3))),
                 x:min(width, x + int(rng.integers(1, 3)))] = True
    elif style == 4:  # a hollow frame, possibly with another mark
        y0 = int(rng.integers(max(1, height - 2)))
        x0 = int(rng.integers(max(1, width - 2)))
        y1 = int(rng.integers(y0 + 3, height + 1))
        x1 = int(rng.integers(x0 + 3, width + 1))
        mask[y0:y1, x0:x1] = True
        mask[y0 + 1:y1 - 1, x0 + 1:x1 - 1] = False
        if int(rng.integers(2)):
            mask[int(rng.integers(height)), int(rng.integers(width))] = True
    elif style == 5:  # repeated motifs with adjustable spacing
        py = int(rng.integers(2, min(height, 5) + 1))
        px = int(rng.integers(2, min(width, 5) + 1))
        oy = int(rng.integers(py))
        ox = int(rng.integers(px))
        for y in range(height):
            for x in range(width):
                mask[y, x] = (y - oy) % py == 0 and (x - ox) % px == 0
        if int(rng.integers(2)):
            mask |= np.roll(mask, 1, axis=int(rng.integers(2)))
    elif style == 6:  # complete or empty rows and columns
        if int(rng.integers(2)):
            mask[rng.choice(height, size=int(rng.integers(1, height)), replace=False), :] = True
        else:
            mask[:, rng.choice(width, size=int(rng.integers(1, width)), replace=False)] = True
    else:  # connected random walk
        y = int(rng.integers(height))
        x = int(rng.integers(width))
        for _ in range(int(rng.integers(3, max(5, height * width // 2)))):
            mask[y, x] = True
            dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            y = min(height - 1, max(0, y + dy))
            x = min(width - 1, max(0, x + dx))
    if not np.any(mask):
        mask[int(rng.integers(height)), int(rng.integers(width))] = True
    if np.all(mask):
        mask[int(rng.integers(height)), int(rng.integers(width))] = False
    return mask


def _informative(rng, left, right):
    """Guarantee four local truth cases in a teaching example."""
    left = left.copy()
    right = right.copy()
    cells = rng.choice(left.size, size=4, replace=False)
    for flat, (a, b) in zip(cells, ((False, False), (True, False),
                                     (False, True), (True, True))):
        left.flat[int(flat)] = a
        right.flat[int(flat)] = b
    return left, right


def _shift(mask, dy, dx):
    h, w = mask.shape
    shifted = np.zeros_like(mask)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    shifted[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = mask[sy0:sy1, sx0:sx1]
    return shifted


def _masks(rng, height, width, relation):
    left = _shape(rng, height, width)
    if relation == "ordinary":
        right = _shape(rng, height, width)
        left, right = _informative(rng, left, right)
    elif relation == "blank":
        right = np.zeros_like(left)
        if int(rng.integers(2)):
            left, right = right, left
    elif relation == "identical":
        right = left.copy()
    elif relation == "contained":
        right = left & _shape(rng, height, width)
        if np.array_equal(left, right):
            right.flat[int(np.flatnonzero(left)[0])] = False
        if not np.any(right):
            right.flat[int(np.flatnonzero(left)[0])] = True
        if int(rng.integers(2)):
            left, right = right, left
    elif relation == "complementary":
        right = ~left
    elif relation == "sparse":
        left[:] = False
        right = np.zeros_like(left)
        for flat in rng.choice(left.size, size=int(rng.integers(1, min(4, left.size) + 1)), replace=False):
            if int(rng.integers(2)):
                left.flat[int(flat)] = True
            else:
                right.flat[int(flat)] = True
    elif relation == "full":
        right = ~left
        right |= left & (rng.random(left.shape) < float(rng.uniform(.1, .6)))
    elif relation == "disjoint":
        right = ~left & _shape(rng, height, width)
        if not np.any(right):
            right.flat[int(np.flatnonzero(~left)[0])] = True
    elif relation == "overlap":
        right = left.copy()
        for flat in rng.choice(left.size, size=int(rng.integers(1, min(4, left.size) + 1)), replace=False):
            right.flat[int(flat)] = ~right.flat[int(flat)]
    elif relation == "asymmetry":
        right = left & (rng.random(left.shape) < float(rng.uniform(.05, .25)))
        if not np.any(right):
            right.flat[int(np.flatnonzero(left)[0])] = True
    elif relation == "shifted":
        dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
        right = _shift(left, dy, dx)
        if int(rng.integers(2)):
            right |= _shape(rng, height, width) & (rng.random(left.shape) < .25)
    else:
        raise ValueError("unknown relation")
    if not np.any(left | right):
        left.flat[0] = True
    if np.all(left & right):
        right.flat[0] = False
    return left, right


def _size(rng, mode, index):
    if mode == "fixed":
        raise ValueError("fixed size supplied separately")
    category = (index if index < 2 else int(rng.integers(4))) % 4
    if category == 0:  # tall
        return int(rng.integers(7, 14)), int(rng.integers(3, 7))
    if category == 1:  # wide
        return int(rng.integers(3, 7)), int(rng.integers(7, 14))
    if category == 2:  # compact
        return int(rng.integers(3, 7)), int(rng.integers(3, 7))
    return int(rng.integers(7, 14)), int(rng.integers(7, 14))


def _make_pair(rng, height, width, orientation, background, mark, separator, highlight, relation):
    left, right = _masks(rng, height, width, relation)
    return _pack_pair(left, right, orientation, background, mark, separator, highlight)


def _pack_pair(left, right, orientation, background, mark, separator, highlight):
    height, width = left.shape
    first = np.where(left, mark, background).astype(np.int8)
    second = np.where(right, mark, background).astype(np.int8)
    if orientation == "vertical":
        stripe = np.full((height, 1), separator, dtype=np.int8)
        grid = np.concatenate((first, stripe, second), axis=1)
    else:
        stripe = np.full((1, width), separator, dtype=np.int8)
        grid = np.concatenate((first, stripe, second), axis=0)
    output = np.where(left | right, highlight, background).astype(np.int8)
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_token = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_token)
    train_count, test_count = _resolve_counts(seed_token, rng, num_train, num_test)

    # These are learned conventions. Geometry and local occupancy are pair-level.
    palette_mode = ("fixed", "varying")[int(rng.integers(2))]
    geometry_mode = ("fixed", "varying")[int(rng.integers(2))]
    orientation_mode = ("vertical", "horizontal", "both")[int(rng.integers(3))]
    role_count = 1 if palette_mode == "fixed" else min(4, train_count)
    colors = [int(x) for x in rng.choice(10, size=2 + 2 * role_count, replace=False)]
    background, highlight = colors[:2]
    marks = colors[2:2 + role_count]
    separators = colors[2 + role_count:]
    fixed_size = (int(rng.integers(3, 14)), int(rng.integers(3, 14)))
    relation_offset = int(rng.integers(len(_RELATIONS)))
    puzzle = {"train": [], "test": []}
    used = set()

    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            global_index = index if split == "train" else train_count + index
            if geometry_mode == "fixed":
                height, width = fixed_size
            else:
                height, width = _size(rng, geometry_mode, global_index)
            if orientation_mode == "both":
                orientation = ("vertical", "horizontal")[index] if split == "train" and index < 2 else ("vertical", "horizontal")[int(rng.integers(2))]
            else:
                orientation = orientation_mode
            if split == "train" and index < role_count:
                role_index = index
                sep_index = index
            else:
                role_index = int(rng.integers(role_count))
                sep_index = int(rng.integers(role_count))
            relation = "ordinary" if split == "train" and index == 0 else _RELATIONS[(relation_offset + global_index * 5) % len(_RELATIONS)]
            # Unique inputs are overwhelmingly likely; a bounded redraw handles tiny grids.
            for attempt in range(32):
                pair = _make_pair(rng, height, width, orientation, background,
                                  marks[role_index], separators[sep_index], highlight, relation)
                signature = _array_signature(pair["input"])
                if signature not in used and not _is_original_pair(pair["input"], pair["output"]):
                    break
                relation = _RELATIONS[(relation_offset + global_index * 5 + attempt + 1) % len(_RELATIONS)]
            else:
                # A deterministic finite fallback is guaranteed: there are
                # more candidate codes than prior inputs and source pairs.
                for code in range(len(used) + len(_SOURCE_PAIR_SIGNATURES) + 1):
                    left = np.zeros((height, width), dtype=bool)
                    right = np.zeros_like(left)
                    for flat, (a, b) in enumerate(((False, False), (True, False),
                                                   (False, True), (True, True))):
                        left.flat[flat], right.flat[flat] = a, b
                    remainder = code
                    for flat in range(4, min(left.size, 9)):
                        category = remainder % 4
                        left.flat[flat] = bool(category & 1)
                        right.flat[flat] = bool(category & 2)
                        remainder //= 4
                    pair = _pack_pair(left, right, orientation, background,
                                      marks[role_index], separators[sep_index], highlight)
                    signature = _array_signature(pair["input"])
                    if signature not in used and not _is_original_pair(pair["input"], pair["output"]):
                        break
                else:
                    raise AssertionError("finite unique-pair fallback exhausted")
            used.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= side <= 30 for side in grid.shape)
    assert np.all((0 <= grid) & (grid <= 9))


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, out = pair["input"], pair["output"]
    _validate_grid(grid)
    _validate_grid(out)
    h, w = out.shape
    if grid.shape == (h, 2 * w + 1):
        orientation = "vertical"
        first, second, stripe = grid[:, :w], grid[:, w + 1:], grid[:, w]
    elif grid.shape == (2 * h + 1, w):
        orientation = "horizontal"
        first, second, stripe = grid[:h], grid[h + 1:], grid[h]
    else:
        raise AssertionError("panels and output do not align")
    assert h >= 3 and w >= 3
    assert np.unique(stripe).size == 1
    separator = int(stripe.flat[0])
    values = set(int(x) for x in np.unique(np.concatenate((first.ravel(), second.ravel()))))
    assert 1 <= len(values) <= 2 and separator not in values
    return {"orientation": orientation, "shape": (h, w), "first": first,
            "second": second, "separator": separator, "panel_values": values,
            "output_values": set(int(x) for x in np.unique(out)), "output": out}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    pairs = puzzle["train"] + puzzle["test"]
    analyses = [_analyze_pair(pair) for pair in pairs]
    seen = set()
    for pair in pairs:
        sig = _array_signature(pair["input"])
        assert sig not in seen
        seen.add(sig)
        assert not _is_original_pair(pair["input"], pair["output"])

    # The first training example contains all four cases, so its output and
    # panel colors uniquely identify the shared background and highlight.
    anchor = analyses[0]
    assert len(anchor["panel_values"]) == 2
    assert len(anchor["output_values"]) == 2
    common = anchor["panel_values"] & anchor["output_values"]
    assert len(common) == 1
    background = next(iter(common))
    highlight = next(iter(anchor["output_values"] - {background}))
    assert background != highlight
    train_cases = set()
    train_marks, train_separators, train_orientations = set(), set(), set()
    informative = 0
    for i, a in enumerate(analyses):
        assert background in a["panel_values"]
        marks = a["panel_values"] - {background}
        assert len(marks) == 1
        mark = next(iter(marks))
        assert len({background, mark, a["separator"], highlight}) == 4
        assert a["output_values"] <= {background, highlight}
        left = a["first"] != background
        right = a["second"] != background
        assert np.array_equal(a["output"], np.where(left | right, highlight, background))
        cases = {(bool(l), bool(r)) for l, r in zip(left.ravel(), right.ravel())}
        if i < nt:
            train_cases.update(cases)
            train_marks.add(mark)
            train_separators.add(a["separator"])
            train_orientations.add(a["orientation"])
            informative += len(cases) == 4
        else:
            assert mark in train_marks and a["separator"] in train_separators
            assert a["orientation"] in train_orientations
    assert train_cases == {(False, False), (True, False), (False, True), (True, True)}
    assert informative >= 1
    assert train_marks.isdisjoint(train_separators)


def _predict_from_demonstrations(train, test_input):
    """Infer the output palette and separator roles from training grids only."""
    first = _analyze_pair(train[0])
    common = first["panel_values"] & first["output_values"]
    assert len(common) == 1
    background = next(iter(common))
    highlight = next(iter(first["output_values"] - {background}))
    separator_colors = {_analyze_pair(pair)["separator"] for pair in train}
    h, w = test_input.shape
    vertical = w % 2 == 1 and int(test_input[0, w // 2]) in separator_colors and np.all(test_input[:, w // 2] == test_input[0, w // 2])
    horizontal = h % 2 == 1 and int(test_input[h // 2, 0]) in separator_colors and np.all(test_input[h // 2] == test_input[h // 2, 0])
    assert vertical != horizontal
    if vertical:
        width = w // 2
        first_panel, second_panel = test_input[:, :width], test_input[:, width + 1:]
    else:
        height = h // 2
        first_panel, second_panel = test_input[:height], test_input[height + 1:]
    return np.where((first_panel != background) | (second_panel != background),
                    highlight, background).astype(np.int8)


def _puzzle_signature(puzzle):
    return tuple((_array_signature(p["input"]), _array_signature(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _observed_features(pair, background):
    a = _analyze_pair(pair)
    left = a["first"] != background
    right = a["second"] != background
    empty, lo, ro, both = _mask_counts(left, right)
    total = left.size
    union = lo + ro + both
    return a, {"blank": min(int(left.sum()), int(right.sum())) == 0,
               "identical": np.array_equal(left, right),
               "contained": np.all(left <= right) or np.all(right <= left),
               "proper_containment": (np.all(left <= right) or np.all(right <= left))
                   and not np.array_equal(left, right) and left.any() and right.any(),
               "complementary": empty == 0 and both == 0,
               "full": empty == 0,
               "sparse": union <= max(4, total // 8),
               "disjoint": both == 0,
               "overlap_heavy": union > 0 and both / union >= .65,
               "ordinary": min(empty, lo, ro, both) > 0 and .1 <= both / union <= .55,
               "partial_occlusion": both > 0 and lo > 0 and ro > 0,
               "asymmetric": abs(int(left.sum()) - int(right.sum())) / total >= .35,
               "edge": bool(np.any(left[0]) or np.any(right[0]) or np.any(left[-1]) or np.any(right[-1]) or np.any(left[:, 0]) or np.any(right[:, -1])),
               "empty_line": bool(np.any(~left.any(axis=0)) or np.any(~right.any(axis=1))),
               "full_line": bool(np.any(left.all(axis=0)) or np.any(right.all(axis=1)))}


def _component_count(mask):
    height, width = mask.shape
    visited = set()
    components = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or (y, x) in visited:
                continue
            components += 1
            frontier = [(y, x)]
            visited.add((y, x))
            for row, col in frontier:
                for a, b in ((row + 1, col), (row - 1, col),
                             (row, col + 1), (row, col - 1)):
                    if (0 <= a < height and 0 <= b < width and mask[a, b]
                            and (a, b) not in visited):
                        visited.add((a, b))
                        frontier.append((a, b))
    return components


def _has_hole(mask):
    height, width = mask.shape
    visited = set()
    for y in range(height):
        for x in range(width):
            if mask[y, x] or (y, x) in visited:
                continue
            frontier = [(y, x)]
            visited.add((y, x))
            touches_edge = False
            for row, col in frontier:
                touches_edge |= row in (0, height - 1) or col in (0, width - 1)
                for a, b in ((row + 1, col), (row - 1, col),
                             (row, col + 1), (row, col - 1)):
                    if (0 <= a < height and 0 <= b < width and not mask[a, b]
                            and (a, b) not in visited):
                        visited.add((a, b))
                        frontier.append((a, b))
            if not touches_edge:
                return True
    return False


def _has_stroke(mask):
    height, width = mask.shape
    for y in range(height):
        for x in range(width - 2):
            if (np.all(mask[y, x:x + 3]) and
                    (y == 0 or not np.any(mask[y - 1, x:x + 3])) and
                    (y == height - 1 or not np.any(mask[y + 1, x:x + 3]))):
                return True
    for x in range(width):
        for y in range(height - 2):
            if (np.all(mask[y:y + 3, x]) and
                    (x == 0 or not np.any(mask[y:y + 3, x - 1])) and
                    (x == width - 1 or not np.any(mask[y:y + 3, x + 1]))):
                return True
    return False


def _has_period(mask):
    height, width = mask.shape
    if not .03 < mask.mean() < .8:
        return False
    return any(np.array_equal(mask[:, :-d], mask[:, d:]) and mask[:, :-d].any()
               for d in range(2, min(5, width))) or any(
                   np.array_equal(mask[:-d], mask[d:]) and mask[:-d].any()
                   for d in range(2, min(5, height)))


def self_test() -> None:
    observed = {name: 0 for name in ("blank", "identical", "contained", "proper_containment",
                                     "complementary", "full", "sparse", "disjoint", "overlap_heavy",
                                     "ordinary", "partial_occlusion", "asymmetric", "edge",
                                     "empty_line", "full_line")}
    train_counts, test_counts, shapes, orientations, palettes, structures = set(), set(), set(), set(), set(), set()
    varied_size = varied_palette = both_orientations = 0
    train_ordinary = test_ordinary = 0
    geometry = {name: 0 for name in ("connected", "islands", "holes", "block",
                                     "stroke", "period", "full_line", "empty_line",
                                     "edge", "separator_touch", "shifted")}
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            demonstrations = [{"input": p["input"].copy(), "output": p["output"].copy()}
                              for p in puzzle["train"]]
            predicted = _predict_from_demonstrations(demonstrations, pair["input"].copy())
            assert np.array_equal(predicted, pair["output"])
        puzzles.append(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        anchor = _analyze_pair(puzzle["train"][0])
        background = next(iter(anchor["panel_values"] & anchor["output_values"]))
        local_shapes, local_orientations, local_marks, local_seps = set(), set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, features = _observed_features(pair, background)
                local_shapes.add(a["shape"])
                local_orientations.add(a["orientation"])
                local_marks.update(a["panel_values"] - {background})
                local_seps.add(a["separator"])
                shapes.add(a["shape"])
                orientations.add(a["orientation"])
                palettes.add((background, next(iter(a["panel_values"] - {background})), a["separator"], next(iter(a["output_values"] - {background})) if len(a["output_values"]) == 2 else -1))
                structures.add((a["orientation"], a["shape"], tuple(int(x != background) for x in a["first"].ravel()), tuple(int(x != background) for x in a["second"].ravel())))
                for key, value in features.items():
                    observed[key] += int(value)
                left = a["first"] != background
                right = a["second"] != background
                for mask in (left, right):
                    geometry["connected"] += _component_count(mask) == 1 and mask.sum() >= 4
                    geometry["islands"] += _component_count(mask) >= 2
                    geometry["holes"] += _has_hole(mask)
                    geometry["block"] += bool(np.any(mask[:-1, :-1] & mask[1:, :-1] &
                                                     mask[:-1, 1:] & mask[1:, 1:]))
                    geometry["stroke"] += _has_stroke(mask)
                    geometry["period"] += _has_period(mask)
                    geometry["full_line"] += bool(np.any(mask.all(axis=0)) or np.any(mask.all(axis=1)))
                    geometry["empty_line"] += bool(np.any(~mask.any(axis=0)) or np.any(~mask.any(axis=1)))
                    geometry["edge"] += bool(np.any(mask[0]) or np.any(mask[-1]) or
                                             np.any(mask[:, 0]) or np.any(mask[:, -1]))
                if a["orientation"] == "vertical":
                    geometry["separator_touch"] += bool(np.any(left[:, -1]) or np.any(right[:, 0]))
                else:
                    geometry["separator_touch"] += bool(np.any(left[-1]) or np.any(right[0]))
                geometry["shifted"] += bool(not np.array_equal(left, right) and
                                            left.any() and right.any() and
                                            any(np.array_equal(_shift(left, dy, dx), right)
                                                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))))
                if split == "train":
                    train_ordinary += int(features["ordinary"])
                else:
                    test_ordinary += int(features["ordinary"])
        varied_size += len(local_shapes) > 1
        varied_palette += len(local_marks) > 1 and len(local_seps) > 1
        both_orientations += len(local_orientations) > 1
    assert len(set(puzzles)) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and orientations == {"vertical", "horizontal"}
    assert len(palettes) >= 50 and len(structures) >= 150
    assert varied_size >= 10 and varied_palette >= 10 and both_orientations >= 5
    assert train_ordinary > 0 and test_ordinary > 0
    assert all(count > 0 for count in observed.values())
    assert all(count > 0 for count in geometry.values())
    assert _puzzle_signature(generate_puzzle(17)) == puzzles[17]
    assert _puzzle_signature(generate_puzzle(7001, 8, 3)) == _puzzle_signature(generate_puzzle(7001, 8, 3))
    for seed, nt, ns in ((8001, 2, 1), (8002, 10, 1), (8003, 10, 4)):
        p = generate_puzzle(seed, nt, ns)
        assert len(p["train"]) == nt and len(p["test"]) == ns
        validate_generated_puzzle(p)
    for seed, nt, ns in ((9001, None, 1), (9002, None, 4), (9003, 2, None), (9004, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        assert (nt is None or len(p["train"]) == nt) and (ns is None or len(p["test"]) == ns)
        validate_generated_puzzle(p)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(12345, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
