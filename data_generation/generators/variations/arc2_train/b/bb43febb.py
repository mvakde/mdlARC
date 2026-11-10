"""Generate solid-rectangle puzzles whose strict interiors are recolored."""

import numpy as np


_SOURCE_PAIRS = frozenset({
    (("0000000000", "0555555000", "0555555000", "0555555000",
      "0555555000", "0555555000", "0000000000", "0000555550",
      "0000555550", "0000555550"),
     ("0000000000", "0555555000", "0522225000", "0522225000",
      "0522225000", "0555555000", "0000000000", "0000555550",
      "0000522250", "0000555550")),
    (("0000000000", "0000000000", "5555500000", "5555505550",
      "5555505550", "5555505550", "5555505550", "0000000000",
      "0000000000", "0000000000"),
     ("0000000000", "0000000000", "5555500000", "5222505550",
      "5222505250", "5222505250", "5555505550", "0000000000",
      "0000000000", "0000000000")),
    (("5555550000", "5555550000", "5555550000", "0000000000",
      "0005555555", "0005555555", "0005555555", "0005555555",
      "0005555555", "0005555555"),
     ("5555550000", "5222250000", "5555550000", "0000000000",
      "0005555555", "0005222225", "0005222225", "0005222225",
      "0005222225", "0005555555")),
})


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validate_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(seed, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_test = int(num_test)
        low = max(2, num_test + 1)
        num_train = low + seed % (11 - low)
    elif num_test is None:
        num_train = int(num_train)
        num_test = 1 + seed % min(4, num_train - 1)
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _key(grid):
    return (grid.shape, grid.tobytes())


def _source_copy(inp, out):
    def rows(grid):
        return tuple("".join(str(int(v)) for v in row) for row in grid)
    return (rows(inp), rows(out)) in _SOURCE_PAIRS


def _components(grid, background):
    """Find each non-background four-connected object from the observed grid."""
    mask = grid != background
    seen = np.zeros(grid.shape, dtype=bool)
    found = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
                            and mask[rr, cc] and not seen[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            top = min(r for r, _ in cells)
            left = min(c for _, c in cells)
            bottom = max(r for r, _ in cells)
            right = max(c for _, c in cells)
            height, width = bottom-top+1, right-left+1
            assert height >= 3 and width >= 3
            assert len(cells) == height*width
            color = int(grid[top, left])
            assert bool(np.all(grid[top:bottom+1, left:right+1] == color))
            found.append((top, left, height, width, color))
    found.sort()
    return found


def _separated(a, b):
    ar, ac, ah, aw = a[:4]
    br, bc, bh, bw = b[:4]
    return (ar+ah+1 <= br or br+bh+1 <= ar
            or ac+aw+1 <= bc or bc+bw+1 <= ac)


def _interior_output(inp, boxes, highlight):
    out = inp.copy()
    for top, left, height, width, _ in boxes:
        out[top+1:top+height-1, left+1:left+width-1] = highlight
    return out


def _canvas(rng, count):
    style = int(rng.integers(0, 5))
    if style == 0:
        low = (7, 9, 12, 16, 17)[count-1]
        return (int(rng.integers(low, min(low+7, 26))),
                int(rng.integers(low, min(low+7, 26))))
    if style == 1:
        return int(rng.integers(14, 23)), int(rng.integers(14, 23))
    if style == 2:
        return int(rng.integers(22, 31)), int(rng.integers(22, 31))
    if style == 3:
        return int(rng.integers(10, 20)), int(rng.integers(20, 31))
    return int(rng.integers(20, 31)), int(rng.integers(10, 20))


def _dimensions(rng, canvas_height, canvas_width, count, strong):
    max_h = min(canvas_height-1, 18 if count <= 2 else 12)
    max_w = min(canvas_width-1, 18 if count <= 2 else 12)
    boxes = []
    equal = count >= 2 and int(rng.integers(0, 7)) == 0
    for index in range(count):
        if equal and index and boxes:
            boxes.append(boxes[0])
            continue
        kind = int(rng.integers(0, 6))
        if strong and index == 0:
            kind = int(rng.choice((3, 4, 5)))
        if kind == 0:
            h = w = 3
        elif kind == 1:
            h = w = int(rng.integers(3, min(max_h, max_w, 11)+1))
        elif kind == 2:
            h = int(rng.integers(3, min(max_h, 8)+1))
            w = int(rng.integers(3, min(max_w, 8)+1))
        elif kind == 3:
            h = int(rng.integers(3, min(max_h, 6)+1))
            w = int(rng.integers(min(7, max_w), max_w+1))
        elif kind == 4:
            h = int(rng.integers(min(7, max_h), max_h+1))
            w = int(rng.integers(3, min(max_w, 6)+1))
        else:
            h = int(rng.integers(3, max_h+1))
            w = int(rng.integers(3, max_w+1))
        boxes.append((h, w))
    return boxes


def _positions(rng, canvas, dimensions, layout, anchor):
    height, width = canvas
    count = len(dimensions)
    corners = (("top", "left"), ("top", "right"),
               ("bottom", "left"), ("bottom", "right"))
    forced = {}
    if anchor == "corner":
        forced[0] = corners[int(rng.integers(0, 4))]
    elif anchor == "opposite":
        first = int(rng.integers(0, 4))
        forced[0], forced[1] = corners[first], corners[3-first]
    elif anchor == "edge":
        forced[0] = (str(rng.choice(("top", "bottom", "left", "right"))),)
    placed = []
    for index, (bh, bw) in enumerate(dimensions):
        candidates = []
        for top in range(height-bh+1):
            for left in range(width-bw+1):
                if anchor == "internal" and (top == 0 or left == 0
                                              or top+bh == height or left+bw == width):
                    continue
                edges = set()
                if top == 0:
                    edges.add("top")
                if left == 0:
                    edges.add("left")
                if top+bh == height:
                    edges.add("bottom")
                if left+bw == width:
                    edges.add("right")
                required = forced.get(index)
                if required and not set(required).issubset(edges):
                    continue
                if index and index not in forced:
                    if layout == "row" and top != placed[0][0]:
                        continue
                    if layout == "column" and left != placed[0][1]:
                        continue
                candidate = (top, left, bh, bw)
                if all(_separated(candidate, previous) for previous in placed):
                    candidates.append(candidate)
        if not candidates:
            return None
        placed.append(candidates[int(rng.integers(0, len(candidates)))])
    return placed


def _colors(rng, params, split, index, count):
    palette = params["palette"]
    mode = params["mode"]
    if mode == "fixed":
        background, foregrounds = palette[0], [palette[1]]*count
    elif split == "train" and index == 0:
        background = palette[0]
        foregrounds = [palette[1]]*count
        if mode == "mixed":
            foregrounds[1] = palette[2]
    elif split == "train" and index == 1:
        background, foregrounds = palette[1], [palette[0]]*count
    else:
        background = palette[int(rng.integers(0, 5))]
        choices = [color for color in palette if color != background]
        foregrounds = [choices[int(rng.integers(0, len(choices)))] for _ in range(count)]
        if mode == "pairwise":
            foregrounds = [foregrounds[0]]*count
        elif count >= 2 and int(rng.integers(0, 3)) != 0:
            foregrounds[1] = next(c for c in choices if c != foregrounds[0])
    return background, foregrounds


def _make_pair(rng, params, split, index, count):
    for _ in range(160):
        canvas = _canvas(rng, count)
        strong = int(rng.integers(0, 100)) < (22 if split == "train" else 38)
        dimensions = _dimensions(rng, *canvas, count, strong)
        if sum(h*w for h, w in dimensions)*2 >= canvas[0]*canvas[1]:
            continue
        layout = str(rng.choice(("scatter", "scatter", "row", "column")))
        anchors = ("internal", "corner", "edge", "free", "opposite")
        weights = (20, 23, 18, 28, 11) if split == "train" else (18, 20, 10, 24, 28)
        anchor = str(rng.choice(anchors, p=np.array(weights)/100))
        if count == 1 and anchor == "opposite":
            anchor = "corner"
        if anchor == "opposite":
            layout = "scatter"
        placements = _positions(rng, canvas, dimensions, layout, anchor)
        if placements is None:
            continue
        background, colors = _colors(rng, params, split, index, count)
        inp = np.full(canvas, background, dtype=np.int8)
        boxes = []
        for (top, left, height, width), color in zip(placements, colors):
            inp[top:top+height, left:left+width] = color
            boxes.append((top, left, height, width, color))
        out = _interior_output(inp, boxes, params["highlight"])
        if not _source_copy(inp, out):
            return {"input": inp, "output": out}
    # All legal count requests have a deterministic construction if crowded
    # random layouts failed: five small, well-separated boxes fit in 20x20.
    canvas = (20, 20)
    slots = [(0, 0), (0, 8), (8, 0), (8, 8), (16, 16)]
    order = rng.permutation(len(slots))
    background, colors = _colors(rng, params, split, index, count)
    inp = np.full(canvas, background, dtype=np.int8)
    boxes = []
    for j, color in enumerate(colors):
        top, left = slots[int(order[j])]
        inp[top:top+3, left:left+3] = color
        boxes.append((top, left, 3, 3, color))
    return {"input": inp, "output": _interior_output(inp, boxes, params["highlight"])}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    colors = [int(c) for c in rng.permutation(10)]
    params = {"highlight": colors[0], "palette": colors[1:],
              "mode": str(rng.choice(("fixed", "pairwise", "mixed"), p=(.25, .35, .40)))}
    puzzle = {"train": [], "test": []}
    used = set()
    first_count = int(rng.integers(2, 5))
    for split, total in (("train", train_count), ("test", test_count)):
        for index in range(total):
            if split == "train" and index == 0:
                count = first_count
            elif split == "train" and index == 1:
                options = [n for n in range(1, 6) if n != first_count]
                count = options[int(rng.integers(0, len(options)))]
            else:
                count = int(rng.choice((1, 2, 3, 4, 5),
                                       p=(.12, .31, .28, .20, .09) if split == "test"
                                       else (.22, .30, .25, .16, .07)))
            for _ in range(20):
                pair = _make_pair(rng, params, split, index, count)
                signature = _key(pair["input"])
                if signature not in used:
                    used.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not generate unique inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _background(inp):
    colors, counts = np.unique(inp, return_counts=True)
    winner = int(np.argmax(counts))
    assert counts[winner]*2 > inp.size
    return int(colors[winner])


def _learn_highlight(train):
    observed = set()
    for pair in train:
        inp, out = pair["input"], pair["output"]
        changed = inp != out
        assert bool(np.any(changed))
        observed.update(int(x) for x in np.unique(out[changed]))
    assert len(observed) == 1
    return next(iter(observed))


def _predict_from_train(train, test_input):
    highlight = _learn_highlight(train)
    background = _background(test_input)
    boxes = _components(test_input, background)
    return _interior_output(test_input, boxes, highlight)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_counts = set()
    train_mixed = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape
            signature = _key(inp)
            assert signature not in seen
            seen.add(signature)
            assert not _source_copy(inp, out)
            background = _background(inp)
            boxes = _components(inp, background)
            assert 1 <= len(boxes) <= 5
            assert all(_separated(a, b) for i, a in enumerate(boxes)
                       for b in boxes[i+1:])
            if split == "train":
                train_counts.add(len(boxes))
                train_mixed |= len({box[4] for box in boxes}) > 1
    assert max(train_counts) >= 2
    assert len(train_counts) >= 2
    highlight = _learn_highlight(puzzle["train"])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            assert not bool(np.any(inp == highlight))
            boxes = _components(inp, _background(inp))
            assert not (split == "test" and len({box[4] for box in boxes}) > 1
                        and not train_mixed)
            expected = _interior_output(inp, boxes, highlight)
            assert np.array_equal(out, expected)
            assert not np.array_equal(inp, out)
            if split == "test":
                assert np.array_equal(_predict_from_train(puzzle["train"], inp), out)


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and
        np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, dimensions = set(), set(), set(), set()
    component_counts, palettes, layouts, structures, highlights = set(), set(), set(), set(), set()
    corners = set()
    single_train = mixed_train = mixed_test = internal_test = 0
    equal_pair = compact = large = tall = wide = 0
    tiny = palette_varied = count_varied = train_corner = 0
    test_within = test_beyond = combined_variation = 0
    background_varied = aligned_rows = aligned_columns = scattered = 0
    edge_categories = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        highlights.add(_learn_highlight(puzzle["train"]))
        train_palettes = set()
        train_backgrounds = set()
        train_object_counts = set()
        puzzle_dimensions = set()
        train_dimensions = set()
        test_dimensions = []
        has_train_corner = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                boxes = _components(inp, _background(inp))
                component_counts.add(len(boxes))
                shapes.add(inp.shape)
                palettes.add(tuple(int(c) for c in np.unique(inp)))
                structures.add((inp.shape, tuple(box[:4] for box in boxes)))
                layouts.add(tuple((box[0], box[1]) for box in boxes))
                if split == "train":
                    train_palettes.add(tuple(int(c) for c in np.unique(inp)))
                    train_backgrounds.add(_background(inp))
                    train_object_counts.add(len(boxes))
                    single_train += len(boxes) == 1
                    mixed_train += len({box[4] for box in boxes}) > 1
                else:
                    mixed_test += len({box[4] for box in boxes}) > 1
                    internal_test += all(box[0] > 0 and box[1] > 0 and
                                         box[0]+box[2] < inp.shape[0] and
                                         box[1]+box[3] < inp.shape[1] for box in boxes)
                compact += max(inp.shape) <= 15
                large += max(inp.shape) >= 24
                equal_pair += len(boxes) > 1 and len({(b[2], b[3]) for b in boxes}) == 1
                if len(boxes) >= 2:
                    aligned_rows += len({b[0] for b in boxes}) == 1
                    aligned_columns += len({b[1] for b in boxes}) == 1
                    scattered += (len({b[0] for b in boxes}) > 1 and
                                  len({b[1] for b in boxes}) > 1)
                for top, left, h, w, _ in boxes:
                    dimensions.add((h, w))
                    puzzle_dimensions.add((h, w))
                    if split == "train":
                        train_dimensions.add((h, w))
                    else:
                        test_dimensions.append((h, w))
                    tiny += (h, w) == (3, 3)
                    tall += h >= 2*w
                    wide += w >= 2*h
                    edge = (top == 0, left+w == inp.shape[1],
                            top+h == inp.shape[0], left == 0)
                    edge_categories[split].add(sum(edge))
                    if split == "train" and sum(edge) == 2:
                        has_train_corner = True
                    if edge[0] and edge[3]:
                        corners.add("top_left")
                    if edge[0] and edge[1]:
                        corners.add("top_right")
                    if edge[2] and edge[3]:
                        corners.add("bottom_left")
                    if edge[2] and edge[1]:
                        corners.add("bottom_right")
                # Independent pixel-mask check, separate from construction.
                changed = inp != out
                mask = np.zeros(inp.shape, dtype=bool)
                for top, left, h, w, _ in boxes:
                    mask[top+1:top+h-1, left+1:left+w-1] = True
                assert np.array_equal(changed, mask)
                assert np.all(out[changed] == _learn_highlight(puzzle["train"]))
        palette_varied += len(train_palettes) > 1
        background_varied += len(train_backgrounds) > 1
        count_varied += len(train_object_counts) > 1
        train_corner += has_train_corner
        max_train_height = max(h for h, _ in train_dimensions)
        max_train_width = max(w for _, w in train_dimensions)
        test_within += all(h <= max_train_height and w <= max_train_width
                           for h, w in test_dimensions)
        test_beyond += any(h > max_train_height or w > max_train_width
                           for h, w in test_dimensions)
        combined_variation += (len(train_palettes) > 1 and
                               len(train_object_counts) > 1 and
                               len(puzzle_dimensions) >= 4)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert component_counts == {1, 2, 3, 4, 5}
    assert len(shapes) >= 70 and len(dimensions) >= 45
    assert len(palettes) >= 45 and len(structures) >= 120
    assert len(layouts) >= 100 and len(highlights) >= 8
    assert corners == {"top_left", "top_right", "bottom_left", "bottom_right"}
    assert single_train >= 4 and mixed_train >= 4 and mixed_test >= 2
    assert internal_test >= 2 and equal_pair >= 5
    assert compact >= 5 and large >= 20 and tall >= 5 and wide >= 5
    assert tiny >= 5 and edge_categories == {"train": {0, 1, 2}, "test": {0, 1, 2}}
    assert palette_varied >= 10 and count_varied == 42 and train_corner >= 5
    assert background_varied >= 10
    assert aligned_rows >= 5 and aligned_columns >= 5 and scattered >= 50
    assert test_within >= 10 and test_beyond >= 3 and combined_variation >= 8
    assert _same_puzzle(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    for seed, a, b in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        p = generate_puzzle(seed, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    for seed, a, b in ((2001, None, 1), (2002, None, 4),
                       (2003, 2, None), (2004, 10, None)):
        p = generate_puzzle(seed, a, b)
        assert (a is None or len(p["train"]) == a)
        assert (b is None or len(p["test"]) == b)
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 3), (4, 4), (True, 1), (3, False)):
        try:
            generate_puzzle(3000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts were accepted")
    assert not _same_puzzle(generate_puzzle(500), generate_puzzle(501))
