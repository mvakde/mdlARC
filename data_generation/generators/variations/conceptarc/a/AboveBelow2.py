"""Generate solid-rectangle ARC scenes with a learned horizontal cutoff."""

import hashlib

import numpy as np


_AUTO_COUNTS = (
    (2, 1), (3, 1), (4, 2), (5, 3), (6, 1),
    (7, 4), (8, 2), (9, 3), (10, 4),
)
_SOURCE_DIGESTS = {
    "b792d7766cb87cee2e87f8cf0dde4f960a31ebe882d00f46cf65bef56e43356a",
    "05cae1624930e245accaabe5b9ff1f33acefeb8e90d6a35916c0e888126b8ed7",
    "01a607f9b6677839d12a1b7cabbebb2c2f2114d14a36b73469cd5b07fc1d8628",
    "c5310724bb7f7f65712b17a0e0ced53d4350cc5cb67a319f7f02bade9863583b",
    "2817bdd2d2208aad4a87944b31b42b3ddb2923eeb8d945e0c36351239830f3c6",
    "c29102f5543cf84130e1eb3046bb1b2442fac520f550d7382a1562b5d8b2cb2a",
}
_SOURCE_DIMENSIONS = ((15, 7), (10, 10), (10, 11), (13, 14), (18, 20), (20, 16))


def _counts(rng, seed, num_train, num_test):
    for value in (num_train, num_test):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError("Pair counts must be integers or None")
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _fits(grid, bg, color, row, col, height, width):
    rows, cols = grid.shape
    if np.any(grid[row:row + height, col:col + width] != bg):
        return False
    # Only equal colors need a gap. Different rectangles may touch the reference.
    neighborhood = grid[
        max(0, row - 1):min(rows, row + height + 1),
        max(0, col - 1):min(cols, col + width + 1),
    ]
    return not np.any(neighborhood == color)


def _place_rectangle(grid, rng, bg, color, start, stop, reference, edge):
    rows, cols = grid.shape
    max_height = min(7, stop - start)
    max_width = min(7, cols)
    if max_height < 2 or max_width < 2:
        return False

    def positions(height, width):
        if edge == "above":
            row_choices = (stop - height,)
        elif edge == "below":
            row_choices = (start,)
        else:
            row_choices = range(start, stop - height + 1)
        for row in row_choices:
            for col in range(cols - width + 1):
                if edge and not (col < reference[1] and col + width > reference[0]):
                    continue
                yield row, col

    # Choose dimensions first so large rectangles do not disappear in a pool of
    # many small placements. The exhaustive fallback makes required placements safe.
    for _ in range(80):
        height = int(rng.integers(2, max_height + 1))
        width = int(rng.integers(2, max_width + 1))
        if edge == "above":
            row = stop - height
        elif edge == "below":
            row = start
        else:
            row = int(rng.integers(start, stop - height + 1))
        if edge:
            leftmost = max(0, reference[0] - width + 1)
            rightmost = min(cols - width, reference[1] - 1)
            if leftmost > rightmost:
                continue
            col = int(rng.integers(leftmost, rightmost + 1))
        else:
            col = int(rng.integers(0, cols - width + 1))
        if _fits(grid, bg, color, row, col, height, width):
            grid[row:row + height, col:col + width] = color
            return True

    choices = []
    for height in range(2, max_height + 1):
        for width in range(2, max_width + 1):
            for row, col in positions(height, width):
                if _fits(grid, bg, color, row, col, height, width):
                    choices.append((row, col, height, width))
    if not choices:
        return False
    row, col, height, width = choices[int(rng.integers(len(choices)))]
    grid[row:row + height, col:col + width] = color
    return True


def _dimensions(rng, regime):
    if rng.random() < 0.20:
        return _SOURCE_DIMENSIONS[int(rng.integers(len(_SOURCE_DIMENSIONS)))]
    if regime == "target_test" and rng.random() < 0.55:
        return int(rng.integers(9, 19)), int(rng.integers(7, 19))
    return int(rng.integers(10, 30)), int(rng.integers(7, 30))


def _reference_size(rng, regime, height, width, with_below):
    max_height = min(6, height - 2 - (2 if with_below else 0))
    max_width = min(6, width)
    if regime == "target_test" and rng.random() < 0.55:
        ref_h = int(rng.integers(2, min(3, max_height) + 1))
        ref_w = int(rng.integers(2, min(3, max_width) + 1))
    else:
        ref_h = int(rng.integers(2, max_height + 1))
        ref_w = int(rng.integers(2, max_width + 1))
    return ref_h, ref_w


def _generate_pair(rng, regime, bg, ref, colors, with_below, shared, novel_color=None):
    height, width = _dimensions(rng, regime)
    ref_h, ref_w = _reference_size(rng, regime, height, width, with_below)
    last_top = height - ref_h - (2 if with_below else 0)
    placement = rng.random()
    if placement < 0.17:
        top = 2
    elif placement < 0.37:
        top = last_top
    else:
        top = int(rng.integers(2, last_top + 1))
    left = int(rng.integers(0, width - ref_w + 1))
    grid = np.full((height, width), bg, dtype=np.int8)
    grid[top:top + ref_h, left:left + ref_w] = ref
    reference = (left, left + ref_w)
    end = top + ref_h

    shared_color = int(novel_color if novel_color is not None else colors[int(rng.integers(len(colors)))])
    above_color = (int(novel_color) if novel_color is not None else
                   shared_color if shared else int(colors[int(rng.integers(len(colors)))]))
    below_color = shared_color if shared else int(colors[int(rng.integers(len(colors)))])
    adjacent_above = rng.random() < (0.55 if regime == "target_test" else 0.40)
    adjacent_below = rng.random() < (0.55 if regime == "target_test" else 0.40)
    assert _place_rectangle(
        grid, rng, bg, above_color, 0, top, reference,
        "above" if adjacent_above else None,
    )
    if with_below:
        assert _place_rectangle(
            grid, rng, bg, below_color, end, height, reference,
            "below" if adjacent_below else None,
        )

    # Count targets are sampled independently of dimensions and reference shape.
    # If a compact band is full, optional placements simply stop at capacity.
    crowded = rng.random() < 0.26
    above_target = int(rng.integers(4, 8) if crowded else rng.integers(1, 5))
    below_target = int(rng.integers(3, 6) if crowded else rng.integers(1, 5)) if with_below else 0
    for _ in range(above_target - 1):
        color = int(colors[int(rng.integers(len(colors)))])
        _place_rectangle(grid, rng, bg, color, 0, top, reference, None)
    for _ in range(below_target - int(with_below)):
        color = int(colors[int(rng.integers(len(colors)))])
        _place_rectangle(grid, rng, bg, color, end, height, reference, None)

    output = grid.copy()
    output[:top] = bg
    return grid, output


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    shuffled = tuple(int(x) for x in rng.permutation(10))
    bg, ref = shuffled[:2]
    familiar_palette = shuffled[2:2 + int(rng.integers(2, 5))]
    novel_palette = tuple(x for x in shuffled[2:] if x not in familiar_palette)

    # Two demonstrations have distinct jobs: one identifies the reference as
    # the sole survivor, and one keeps an ordinary hue below while erasing it above.
    order = rng.permutation(train_count)
    blank_below_index = int(order[0])
    shared_index = int(order[1])
    train_with_below = [bool(rng.integers(2)) for _ in range(train_count)]
    train_with_below[blank_below_index] = False
    train_with_below[shared_index] = True
    puzzle = {"train": [], "test": []}
    seen = set()

    def add_pair(key, regime, colors, with_below, shared, novel_color=None):
        for _ in range(30):
            inp, out = _generate_pair(
                rng, regime, bg, ref, colors, with_below, shared, novel_color,
            )
            fingerprint = (inp.shape, inp.tobytes())
            if fingerprint not in seen and _pair_digest(inp, out) not in _SOURCE_DIGESTS:
                seen.add(fingerprint)
                puzzle[key].append({"input": inp, "output": out})
                return
        raise RuntimeError("Could not produce distinct inputs")

    for index in range(train_count):
        add_pair(
            "train", "source_train", familiar_palette,
            train_with_below[index], index == shared_index,
        )

    observed = {
        int(value)
        for pair in puzzle["train"]
        for value in np.unique(pair["input"])
        if value not in (bg, ref)
    }
    familiar = tuple(sorted(observed))
    novel_index = int(rng.integers(test_count)) if test_count > 1 else -1
    for index in range(test_count):
        use_novel = index == novel_index or (test_count == 1 and rng.random() < 0.5)
        if use_novel:
            novel = int(novel_palette[int(rng.integers(len(novel_palette)))])
            colors = familiar + (novel,)
        else:
            colors = familiar
        with_below = bool(rng.integers(2))
        add_pair("test", "target_test", colors, with_below,
                 bool(rng.integers(2)), novel if use_novel else None)
    return puzzle


def _components(grid, bg):
    rows, cols = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    components = []
    for row in range(rows):
        for col in range(cols):
            color = int(grid[row, col])
            if color == bg or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= rr < rows and 0 <= cc < cols and not visited[rr, cc]
                            and grid[rr, cc] == color):
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            assert len(cells) == (bottom - top + 1) * (right - left + 1)
            assert np.all(grid[top:bottom + 1, left:right + 1] == color)
            components.append((color, top, bottom, left, right))
    return components


def _pair_digest(inp, out):
    a = np.asarray(inp, dtype=np.uint8)
    b = np.asarray(out, dtype=np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _analyze_pair(pair, bg, ref):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape
    assert _pair_digest(inp, out) not in _SOURCE_DIGESTS
    components = _components(inp, bg)
    references = [part for part in components if part[0] == ref]
    assert len(references) == 1
    _, top, bottom, left, right = references[0]
    assert 2 <= bottom - top + 1 <= 6 and 2 <= right - left + 1 <= 6
    assert np.all(out[:top] == bg)
    assert np.array_equal(out[top:], inp[top:])
    assert not np.array_equal(inp, out)
    above, below = [], []
    for part in components:
        color, r0, r1, c0, c1 = part
        if color == ref:
            continue
        assert 2 <= r1 - r0 + 1 <= 7 and 2 <= c1 - c0 + 1 <= 7
        if r1 < top:
            above.append(part)
        else:
            assert r0 > bottom
            below.append(part)
    assert above
    return {
        "shape": inp.shape, "reference": references[0],
        "above": above, "below": below,
        "adjacent_above": any(x[2] == top - 1 and x[3] <= right and x[4] >= left for x in above),
        "adjacent_below": any(x[1] == bottom + 1 and x[3] <= right and x[4] >= left for x in below),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    pairs = puzzle["train"] + puzzle["test"]
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        assert isinstance(pair["input"], np.ndarray)
        assert isinstance(pair["output"], np.ndarray)
        assert pair["input"].shape == pair["output"].shape
        assert np.issubdtype(pair["input"].dtype, np.integer)
        assert np.issubdtype(pair["output"].dtype, np.integer)
    changed = [pair["output"][pair["input"] != pair["output"]] for pair in pairs]
    assert all(len(values) > 0 and len(np.unique(values)) == 1 for values in changed)
    bg = int(changed[0][0])
    assert all(int(values[0]) == bg for values in changed)

    # The training example with no lower ordinary blocks identifies the
    # persistent reference hue without knowing generation parameters.
    candidates = []
    for pair in puzzle["train"]:
        surviving = _components(pair["output"], bg)
        if len(surviving) == 1:
            candidates.append(surviving[0][0])
    assert candidates and len(set(candidates)) == 1
    ref = candidates[0]
    assert bg != ref
    infos = {key: [_analyze_pair(pair, bg, ref) for pair in puzzle[key]]
             for key in ("train", "test")}
    assert any(not x["below"] for x in infos["train"])
    assert any(
        {part[0] for part in x["above"]} & {part[0] for part in x["below"]}
        for x in infos["train"]
    )
    assert len({(x["shape"], x["reference"]) for x in infos["train"]}) > 1
    seen = set()
    for pair in pairs:
        key = (pair["input"].shape, pair["input"].tobytes())
        assert key not in seen
        seen.add(key)


def _same_puzzle(a, b):
    return all(
        len(a[key]) == len(b[key]) and all(
            np.array_equal(x[grid], y[grid])
            for x, y in zip(a[key], b[key])
            for grid in ("input", "output")
        ) for key in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(x["train"]) for x in puzzles}) >= 4
    assert len({len(x["test"]) for x in puzzles}) >= 3
    assert _same_puzzle(generate_puzzle(5), generate_puzzle(5))
    assert not _same_puzzle(generate_puzzle(5), generate_puzzle(6))
    for train, test in ((2, 1), (10, 1), (10, 4),
                        (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(54321, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        assert _same_puzzle(puzzle, generate_puzzle(54321, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(1, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")

    backgrounds, references, shapes, layouts = set(), set(), set(), set()
    small_train = large_test = familiar_test = novel_test = False
    adjacent_above = adjacent_below = both_adjacent = False
    bottom_edge = upper_reference = crowded_above = crowded_below = False
    shared_sides = compact = tall_block = False
    source_dimension = one_row_below = False
    separated_above = separated_below = ordinary_at_least_reference = False
    train_first_below = set()
    test_second_below = set()
    within_shape_varied = within_ref_varied = 0
    train_ref_shapes, test_ref_shapes = set(), set()
    no_below_orders = set()
    for puzzle in puzzles:
        bg = int(puzzle["train"][0]["output"][
            puzzle["train"][0]["input"] != puzzle["train"][0]["output"]
        ][0])
        ref = next(
            parts[0][0]
            for pair in puzzle["train"]
            if len(parts := _components(pair["output"], bg)) == 1
        )
        backgrounds.add(bg)
        references.add(ref)
        train_colors = {int(v) for p in puzzle["train"] for v in np.unique(p["input"])} - {bg, ref}
        pair_shapes, pair_ref_shapes = set(), set()
        no_below_orders.add(tuple(
            i for i, p in enumerate(puzzle["train"])
            if not _analyze_pair(p, bg, ref)["below"]
        ))
        for key in ("train", "test"):
            for pair in puzzle[key]:
                info = _analyze_pair(pair, bg, ref)
                shape = info["shape"]
                _, top, bottom, left, right = info["reference"]
                rh, rw = bottom - top + 1, right - left + 1
                pair_shapes.add(shape)
                pair_ref_shapes.add((rh, rw))
                shapes.add(shape)
                layouts.add((shape, top, left, rh, rw, len(info["above"]), len(info["below"])))
                adjacent_above |= info["adjacent_above"]
                adjacent_below |= info["adjacent_below"]
                both_adjacent |= info["adjacent_above"] and info["adjacent_below"]
                bottom_edge |= bottom == shape[0] - 1
                upper_reference |= top == 2
                crowded_above |= len(info["above"]) >= 5
                crowded_below |= len(info["below"]) >= 4
                compact |= shape[0] <= 15 and shape[1] <= 10
                source_dimension |= shape in _SOURCE_DIMENSIONS
                one_row_below |= bottom == shape[0] - 2
                separated_above |= np.all(pair["input"][top - 1] == bg)
                separated_below |= bottom + 1 < shape[0] and np.all(pair["input"][bottom + 1] == bg)
                tall_block |= any(x[2] - x[1] + 1 >= 6 for x in info["above"] + info["below"])
                ordinary_at_least_reference |= any(
                    x[2] - x[1] + 1 >= rh or x[4] - x[3] + 1 >= rw
                    for x in info["above"] + info["below"]
                )
                shared_sides |= bool({x[0] for x in info["above"]} & {x[0] for x in info["below"]})
                if key == "train":
                    train_ref_shapes.add((rh, rw))
                    small_train |= rh == 2 or rw == 2
                else:
                    test_ref_shapes.add((rh, rw))
                    large_test |= rh >= 3 and rw >= 3
                    colors = {x[0] for x in info["above"] + info["below"]}
                    familiar_test |= colors <= train_colors
                    novel_test |= bool(colors - train_colors)
        within_shape_varied += len(pair_shapes) > 1
        within_ref_varied += len(pair_ref_shapes) > 1
        train_first_below.add(bool(_analyze_pair(puzzle["train"][0], bg, ref)["below"]))
        if len(puzzle["test"]) > 1:
            test_second_below.add(bool(_analyze_pair(puzzle["test"][1], bg, ref)["below"]))
    assert len(backgrounds) >= 4 and len(references) >= 5
    assert len(shapes) >= 60 and len(layouts) >= 150
    assert len(no_below_orders) >= 3
    assert within_shape_varied >= 38 and within_ref_varied >= 38
    assert train_first_below == {False, True}
    assert test_second_below == {False, True}
    assert len(train_ref_shapes & test_ref_shapes) >= 8
    assert all((small_train, large_test, familiar_test, novel_test,
                adjacent_above, adjacent_below, both_adjacent,
                bottom_edge, upper_reference, crowded_above, crowded_below,
                shared_sides, compact, tall_block, source_dimension,
                one_row_below, separated_above, separated_below,
                ordinary_at_least_reference))
