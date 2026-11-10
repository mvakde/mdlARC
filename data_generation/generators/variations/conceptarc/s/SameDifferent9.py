import hashlib
import numpy as np


SOURCE_INPUT_HASHES = frozenset({
    "90175b68260ecd032bc7998384718952af7be007ff50ebf106627e98c16d8d43",
    "72ba1820e637d507552cfdcf63b45e72054fcf500915c88372b2c86a11de84f7",
    "1fb9a0875a6a865c8b42575cc47c549c3167b4611f2c93e6cc9de0be2b1712bc",
    "7ba391d1e5fa333cb250d2558a34f5fd9a2683b0c69c92d519beb23a9a71c407",
    "c2177f2a898e71acd28c5cdf6f40a79c805de500fad31f2d9d73612116168e92",
    "fbc2757831f92acecce4187d107ee36f06860321b717689ef7adc124249db137",
    "f762f18d9af8c069826467b958b977000a204a808f5aa2b0d2b9637c42334dda",
})
DIMENSIONS = tuple((h, w) for h in range(3, 11) for w in range(3, 11))
RECTANGLES = tuple(d for d in DIMENSIONS if d[0] != d[1])
EQUAL_AREA = tuple(
    (a, b) for a in RECTANGLES for b in RECTANGLES
    if a != b and a[0] * a[1] == b[0] * b[1]
    and sum(a) != sum(b)
)
EQUAL_PERIMETER = tuple(
    (a, b) for a in RECTANGLES for b in RECTANGLES
    if a != b and sum(a) == sum(b)
    and a[0] * a[1] != b[0] * b[1]
)
MODES = ("separated", "edge_full", "edge_partial", "corner", "overlap", "nested")


def _check_count(value, lower, upper, name):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not lower <= value <= upper
    ):
        raise ValueError(f"{name} must be an integer from {lower} through {upper}")


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _source_hash(grid):
    return hashlib.sha256(str(grid.shape).encode() + grid.astype(np.int8).tobytes()).hexdigest()


def _bbox(grid, color):
    rows, cols = np.where(grid == color)
    assert len(rows)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _dimensions(box):
    return box[2] - box[0] + 1, box[3] - box[1] + 1


def _frame_cells(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}


def _paint_frame(grid, box, color):
    r0, c0, r1, c1 = box
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color


def _choose_dims(rng, category):
    pick = lambda choices: choices[int(rng.integers(len(choices)))]
    if category == "equal_rect":
        d = pick(RECTANGLES)
        return d, d
    if category == "equal_square":
        side = int(rng.integers(3, 11))
        return (side, side), (side, side)
    if category == "transpose":
        a = pick(RECTANGLES)
        return a, (a[1], a[0])
    if category == "equal_area":
        return pick(EQUAL_AREA)
    if category == "equal_perimeter":
        return pick(EQUAL_PERIMETER)
    if category == "square_rect":
        side = int(rng.integers(3, 11))
        a, b = (side, side), pick(RECTANGLES)
        return (b, a) if rng.integers(2) else (a, b)
    if category == "unequal_rect":
        a = pick(RECTANGLES)
        choices = [b for b in RECTANGLES if b != a and b[0] != a[0] and b[1] != a[1]]
        return a, pick(choices)
    if category == "nested_rect":
        outer = pick(tuple(d for d in RECTANGLES if min(d) >= 7))
        inner = (int(rng.integers(3, outer[0] - 3)),
                 int(rng.integers(3, outer[1] - 3)))
        return (inner, outer) if rng.integers(2) else (outer, inner)
    a = pick(DIMENSIONS)
    return a, pick(DIMENSIONS)


def _relative_boxes(rng, a, b, mode):
    ha, wa = a
    hb, wb = b
    if mode == "nested":
        if ha >= hb + 4 and wa >= wb + 4:
            dr = int(rng.integers(2, ha - hb - 1))
            dc = int(rng.integers(2, wa - wb - 1))
            return (0, 0, ha - 1, wa - 1), (dr, dc, dr + hb - 1, dc + wb - 1)
        if hb >= ha + 4 and wb >= wa + 4:
            dr = -int(rng.integers(2, hb - ha - 1))
            dc = -int(rng.integers(2, wb - wa - 1))
            return (0, 0, ha - 1, wa - 1), (dr, dc, dr + hb - 1, dc + wb - 1)
        return None
    if mode == "separated":
        gap = int(rng.integers(1, 6))
        dr, dc = int(rng.integers(-hb + 1, ha)), wa + gap
    elif mode == "corner":
        dr, dc = ha, wa
    elif mode in ("edge_full", "edge_partial"):
        dc = wa
        offsets = [d for d in range(-hb + 1, ha)
                   if (min(ha, d + hb) - max(0, d) == min(ha, hb)) == (mode == "edge_full")
                   and min(ha, d + hb) - max(0, d) > 0]
        if not offsets:
            return None
        dr = int(rng.choice(offsets))
    else:
        return None
    return (0, 0, ha - 1, wa - 1), (dr, dc, dr + hb - 1, dc + wb - 1)


def _normalize_boxes(a, b):
    top = min(a[0], b[0])
    left = min(a[1], b[1])
    a = (a[0] - top, a[1] - left, a[2] - top, a[3] - left)
    b = (b[0] - top, b[1] - left, b[2] - top, b[3] - left)
    return a, b


def _valid_overlap(a, b, first_color, second_color, background, order):
    a, b = _normalize_boxes(a, b)
    h = max(a[2], b[2]) + 1
    w = max(a[3], b[3]) + 1
    grid = np.full((h, w), background, dtype=np.int8)
    if order:
        _paint_frame(grid, a, first_color)
        _paint_frame(grid, b, second_color)
    else:
        _paint_frame(grid, b, second_color)
        _paint_frame(grid, a, first_color)
    return (first_color in grid and second_color in grid
            and _bbox(grid, first_color) == a and _bbox(grid, second_color) == b
            and np.count_nonzero(grid == first_color) >= .65 * len(_frame_cells(a))
            and np.count_nonzero(grid == second_color) >= .65 * len(_frame_cells(b)))


def _overlap_boxes(rng, a, b, colors, background, order):
    ha, wa = a
    hb, wb = b
    candidates = []
    first = (0, 0, ha - 1, wa - 1)
    for dr in range(-hb + 1, ha):
        for dc in range(-wb + 1, wa):
            second = (dr, dc, dr + hb - 1, dc + wb - 1)
            if _frame_cells(first) & _frame_cells(second) and _valid_overlap(
                first, second, colors[0], colors[1], background, order
            ):
                candidates.append((first, second))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _canvas(rng, boxes, background):
    a, b = _normalize_boxes(*boxes)
    extent_h = max(a[2], b[2]) + 1
    extent_w = max(a[3], b[3]) + 1
    extra_h = int(rng.choice((4, 4, 4, 8, 8, 12, 16)))
    extra_w = int(rng.choice((4, 4, 4, 8, 8, 12, 16)))
    height = int(rng.integers(max(7, extent_h + 2), min(30, extent_h + extra_h) + 1))
    width = int(rng.integers(max(7, extent_w + 2), min(30, extent_w + extra_w) + 1))
    top = int(rng.integers(height - extent_h + 1))
    left = int(rng.integers(width - extent_w + 1))
    shifted = tuple((box[0] + top, box[1] + left, box[2] + top, box[3] + left) for box in (a, b))
    return np.full((height, width), background, dtype=np.int8), shifted


def _palette(rng, background, previous=None):
    pool = [c for c in range(10) if c not in (background, 5, 8)]
    if previous is None:
        return tuple(int(x) for x in rng.choice(pool, size=2, replace=False))
    mode = int(rng.integers(3))
    if mode == 0:
        colors = list(previous)
    elif mode == 1:
        colors = [int(rng.choice(previous)), int(rng.choice([c for c in pool if c not in previous]))]
    else:
        colors = list(rng.choice([c for c in pool if c not in previous], size=2, replace=False))
    if rng.integers(2):
        colors.reverse()
    return tuple(colors)


def _construct_pair(rng, category, mode, background, palette):
    for _ in range(80):
        dims = _choose_dims(rng, category)
        order = bool(rng.integers(2))
        if mode == "overlap":
            boxes = _overlap_boxes(rng, *dims, palette, background, order)
        else:
            boxes = _relative_boxes(rng, *dims, mode)
        if boxes is None:
            continue
        grid, (a, b) = _canvas(rng, boxes, background)
        if order:
            _paint_frame(grid, a, palette[0])
            _paint_frame(grid, b, palette[1])
        else:
            _paint_frame(grid, b, palette[1])
            _paint_frame(grid, a, palette[0])
        if _bbox(grid, palette[0]) != a or _bbox(grid, palette[1]) != b:
            continue
        output_color = 8 if dims[0] == dims[1] else 5
        output = np.where(grid == background, background, output_color).astype(np.int8)
        return {"input": grid, "output": output}
    if mode == "nested":
        return _construct_pair(rng, category, "overlap", background, palette)
    raise RuntimeError("Could not construct recoverable frames")


def _pair_plan(rng, train_count, test_count):
    train_categories = ["equal_rect", "transpose"]
    if train_count >= 3:
        train_categories.append("square_rect")
    if train_count >= 4:
        train_categories.append("equal_area")
    if train_count >= 5:
        train_categories.append("equal_perimeter")
    if train_count >= 6:
        train_categories.append("equal_square")
    if train_count >= 7:
        train_categories.append("nested_rect")
    extras = ("unequal_rect", "equal_rect", "square_rect", "equal_area", "equal_perimeter", "equal_square", "nested_rect")
    while len(train_categories) < train_count:
        train_categories.append(extras[int(rng.integers(len(extras)))])
    train_modes = ["separated", "overlap" if rng.integers(2) else "edge_partial"]
    train_modes += [MODES[int(rng.integers(len(MODES)))] for _ in range(train_count - 2)]
    rng.shuffle(train_categories)
    rng.shuffle(train_modes)
    test_categories = [("equal_rect", "equal_square", "square_rect", "transpose")
                       [int(rng.integers(4))]]
    test_categories += [extras[int(rng.integers(len(extras)))] for _ in range(test_count - 1)]
    test_modes = ["overlap" if rng.integers(2) else "edge_full"]
    test_modes += [MODES[int(rng.integers(len(MODES)))] for _ in range(test_count - 1)]
    rng.shuffle(test_categories)
    rng.shuffle(test_modes)
    train_plan = [(category, "nested" if category == "nested_rect" else mode)
                  for category, mode in zip(train_categories, train_modes)]
    test_plan = [(category, "nested" if category == "nested_rect" else mode)
                 for category, mode in zip(test_categories, test_modes)]
    return train_plan, test_plan


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    _check_count(num_train, 2, 10, "num_train")
    _check_count(num_test, 1, 4, "num_test")
    legal = [(a, b) for a in range(2, 11) for b in range(1, 5)
             if a > b and (num_train is None or a == num_train)
             and (num_test is None or b == num_test)]
    if not legal:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    train_count, test_count = legal[int(rng.integers(len(legal)))]
    background = int(rng.choice((0, 0, 0, 1, 2, 3, 4, 6, 7, 9)))
    plans = _pair_plan(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    previous = None
    for split, plan in zip(("train", "test"), plans):
        for category, mode in plan:
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(100):
                palette = _palette(pair_rng, background, previous)
                pair = _construct_pair(pair_rng, category, mode, background, palette)
                key = _grid_key(pair["input"])
                if key not in seen and _source_hash(pair["input"]) not in SOURCE_INPUT_HASHES:
                    seen.add(key)
                    previous = palette
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, output = pair["input"], pair["output"]
    for arr in (grid, output):
        assert isinstance(arr, np.ndarray) and arr.ndim == 2
        assert all(1 <= n <= 30 for n in arr.shape)
        assert np.issubdtype(arr.dtype, np.integer)
        assert bool(np.all((0 <= arr) & (arr <= 9)))
    assert grid.shape == output.shape and not np.array_equal(grid, output)
    input_colors = set(int(c) for c in np.unique(grid))
    output_colors = set(int(c) for c in np.unique(output))
    assert len(input_colors) == 3 and len(output_colors) == 2
    common = input_colors & output_colors
    assert len(common) == 1
    background = common.pop()
    colors = sorted(input_colors - {background})
    target = (output_colors - {background}).pop()
    assert target in (5, 8) and all(c not in (5, 8) for c in colors)
    boxes = tuple(_bbox(grid, c) for c in colors)
    dims = tuple(_dimensions(b) for b in boxes)
    assert all(min(d) >= 3 for d in dims)
    assert bool(np.any(grid == background))
    forward = np.full_like(grid, background)
    reverse = forward.copy()
    _paint_frame(forward, boxes[0], colors[0])
    _paint_frame(forward, boxes[1], colors[1])
    _paint_frame(reverse, boxes[1], colors[1])
    _paint_frame(reverse, boxes[0], colors[0])
    assert np.array_equal(grid, forward) or np.array_equal(grid, reverse)
    equal = dims[0] == dims[1]
    assert target == (8 if equal else 5)
    assert np.array_equal(output, np.where(grid == background, background, target))
    assert _source_hash(grid) not in SOURCE_INPUT_HASHES
    fa, fb = (_frame_cells(b) for b in boxes)
    overlap = bool(fa & fb)
    adjacent = any((r + dr, c + dc) in fb for r, c in fa for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    corner = not overlap and not adjacent and any((r + dr, c + dc) in fb for r, c in fa for dr in (-1, 1) for dc in (-1, 1))
    nested = (not overlap and not adjacent and
              (all(boxes[0][i] < boxes[1][i] for i in (0, 1)) and all(boxes[0][i] > boxes[1][i] for i in (2, 3)) or
               all(boxes[1][i] < boxes[0][i] for i in (0, 1)) and all(boxes[1][i] > boxes[0][i] for i in (2, 3))))
    return {"background": background, "target": target, "equal": equal,
            "dims": dims, "colors": frozenset(colors), "overlap": overlap,
            "adjacent": adjacent, "corner": corner, "nested": nested}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs = set()
    seen_pairs = set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            details = _inspect_pair(pair)
            ikey = _grid_key(pair["input"])
            pkey = ikey, _grid_key(pair["output"])
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            info[split].append(details)
    assert len({x["background"] for split in info.values() for x in split}) == 1
    assert {x["equal"] for x in info["train"]} == {True, False}
    assert any(x["equal"] and x["dims"][0][0] != x["dims"][0][1] for x in info["train"])
    assert any(not x["equal"] and x["dims"][0][0] * x["dims"][0][1] == x["dims"][1][0] * x["dims"][1][1]
               and sum(x["dims"][0]) == sum(x["dims"][1]) for x in info["train"])
    if nt >= 3:
        assert any((d[0] == d[1]) != (e[0] == e[1]) for x in info["train"] for d, e in (x["dims"],))
    if nt >= 4:
        assert any(not x["equal"] and x["dims"][0][0] * x["dims"][0][1] == x["dims"][1][0] * x["dims"][1][1]
                   and sum(x["dims"][0]) != sum(x["dims"][1]) for x in info["train"])
    if nt >= 5:
        assert any(not x["equal"] and sum(x["dims"][0]) == sum(x["dims"][1])
                   and x["dims"][0][0] * x["dims"][0][1] != x["dims"][1][0] * x["dims"][1][1]
                   for x in info["train"])
    if nt >= 7:
        assert any(x["nested"] for x in info["train"])
    assert any(x["overlap"] or x["adjacent"] or x["corner"] or x["nested"] for x in info["test"])
    learned = {x["equal"]: x["target"] for x in info["train"]}
    assert learned == {True: 8, False: 5}
    for x in info["test"]:
        assert x["target"] == learned[x["equal"]]


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    shapes, geometries, backgrounds, palettes = set(), set(), set(), set()
    equal_overlap = unequal_overlap = area = perimeter = 0
    train_contacts = 0
    edge_kinds = set()
    palette_changes = set()
    regime_examples = set()
    compact = roomy = source_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_grid_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        if any(_inspect_pair(p)["adjacent"] or _inspect_pair(p)["overlap"] for p in puzzle["train"]):
            train_contacts += 1
        previous_palette = None
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = _inspect_pair(pair)
                shapes.add(pair["input"].shape)
                geometries.add(x["dims"])
                backgrounds.add(x["background"])
                palettes.add(x["colors"])
                if previous_palette is not None:
                    palette_changes.add(len(previous_palette & x["colors"]))
                previous_palette = x["colors"]
                if x["colors"] in (frozenset((4, 7)), frozenset((1, 6))):
                    source_palettes += 1
                compact += max(pair["input"].shape) <= 12
                roomy += max(pair["input"].shape) >= 24
                if x["adjacent"] and not x["overlap"]:
                    colors = sorted(x["colors"])
                    a, b = (_bbox(pair["input"], color) for color in colors)
                    if a[3] + 1 == b[1] or b[3] + 1 == a[1]:
                        length = min(a[2], b[2]) - max(a[0], b[0]) + 1
                        edge_kinds.add("full" if length == min(x["dims"][0][0], x["dims"][1][0]) else "partial")
                    else:
                        length = min(a[3], b[3]) - max(a[1], b[1]) + 1
                        edge_kinds.add("full" if length == min(x["dims"][0][1], x["dims"][1][1]) else "partial")
                if x["nested"]:
                    regime_examples.add((split, "nested"))
                if x["corner"]:
                    regime_examples.add((split, "corner"))
                if (x["dims"][0][0] == x["dims"][0][1]) != (x["dims"][1][0] == x["dims"][1][1]):
                    regime_examples.add((split, "square_rect"))
                if x["overlap"]:
                    equal_overlap += x["equal"]
                    unequal_overlap += not x["equal"]
                d, e = x["dims"]
                if split == "train" and not x["equal"]:
                    area += d[0] * d[1] == e[0] * e[1] and sum(d) != sum(e)
                    perimeter += sum(d) == sum(e) and d[0] * d[1] != e[0] * e[1]
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 60 and len(geometries) >= 80
    assert len(backgrounds) >= 4 and len(palettes) >= 12
    assert equal_overlap >= 5 and unequal_overlap >= 5
    assert area >= 5 and perimeter >= 5 and train_contacts >= 30
    assert edge_kinds == {"full", "partial"} and palette_changes == {0, 1, 2}
    assert all((split, kind) in regime_examples for split in ("train", "test")
               for kind in ("nested", "corner", "square_rect"))
    assert compact >= 10 and roomy >= 10 and source_palettes >= 5
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + i, *counts)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(1000 + i, *counts))
        for actual, expected in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            assert expected is None or actual == expected
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
