"""Generate solved puzzles about the strict interiors of rectangular frames."""

import numpy as np


# Source inputs are represented by their canvas and frame bounding boxes.  A
# source-palette pair with one of these layouts would copy an original pair.
_SOURCE_LAYOUTS = {
    (15, 15, ((1, 1, 5, 5), (7, 10, 8, 11))),
    (10, 10, ((1, 4, 4, 6),)),
    (10, 10, ((1, 1, 3, 3), (5, 5, 9, 8))),
    (25, 25, ((1, 1, 3, 7), (7, 4, 8, 5), (9, 15, 12, 23), (18, 1, 23, 5))),
    (10, 10, ((0, 0, 2, 2), (4, 1, 9, 8))),
}


def _counts(seed, rng, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= int(value) <= hi):
            raise ValueError(name + " is outside its legal range")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _rect_size(rect):
    return rect[2] - rect[0] + 1, rect[3] - rect[1] + 1


def _separated(a, b):
    return (a[2] + 1 < b[0] or b[2] + 1 < a[0]
            or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _boundary(rect, height, width):
    return rect[0] == 0 or rect[1] == 0 or rect[2] == height - 1 or rect[3] == width - 1


def _child_rng(rng):
    return np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))


def _palette(rng, fill, base, variable, index):
    if not variable or index == 0:
        return base
    choices = [c for c in range(10) if c != fill]
    if index == 1:
        choices = [c for c in choices if c != base[0]]
    background = int(rng.choice(choices))
    frame = int(rng.choice([c for c in range(10) if c not in (fill, background)]))
    return background, frame, fill


def _canvas(rng, split, index):
    # Training begins with an easy source-sized canvas. Later pairs and tests
    # use the full range, with tests weighted towards roomy examples.
    if split == "train" and index == 0:
        height = int(rng.integers(9, 16))
        width = int(rng.integers(9, 17))
    elif split == "train" and index == 1:
        height = int(rng.integers(12, 27))
        width = int(rng.integers(12, 27))
    else:
        bands = ((8, 14), (14, 22), (22, 31))
        weights = (0.25, 0.35, 0.40) if split == "train" else (0.28, 0.27, 0.45)
        first = int(rng.choice(3, p=weights))
        second = first if rng.random() < 0.63 else int(rng.choice(3, p=weights))
        height = int(rng.integers(*bands[first]))
        width = int(rng.integers(*bands[second]))
    if rng.random() < 0.12:
        width = height
    return height, width


def _shape(rng, height, width, kind):
    max_h, max_w = min(height, 19), min(width, 19)
    if kind == "square_cell":
        return 3, 3
    if kind == "thin_h":
        return 3, int(rng.integers(4, max(5, max_w + 1)))
    if kind == "thin_v":
        return int(rng.integers(4, max(5, max_h + 1))), 3
    if kind == "wide":
        h = int(rng.integers(3, min(8, max_h) + 1))
        return h, int(rng.integers(max(4, min(7, max_w)), max_w + 1))
    if kind == "tall":
        w = int(rng.integers(3, min(8, max_w) + 1))
        return int(rng.integers(max(4, min(7, max_h)), max_h + 1)), w
    if kind == "large":
        return (int(rng.integers(max(3, min(7, max_h)), max_h + 1)),
                int(rng.integers(max(3, min(7, max_w)), max_w + 1)))
    if kind == "deg_h":
        return 2, int(rng.integers(3, min(max_w, 12) + 1))
    if kind == "deg_v":
        return int(rng.integers(3, min(max_h, 12) + 1)), 2
    if kind == "deg_square":
        return 2, 2
    return (int(rng.integers(3, min(max_h, 11) + 1)),
            int(rng.integers(3, min(max_w, 11) + 1)))


def _place(rng, height, width, shapes, require_boundary=False,
           full_canvas=False, margin=0):
    if full_canvas:
        return [(0, 0, height - 1, width - 1)]
    ordered = sorted(shapes, key=lambda s: s[0] * s[1], reverse=True)
    if require_boundary:
        # Force an independently chosen side or corner for one object.
        forced = int(rng.integers(0, len(ordered)))
        ordered.insert(0, ordered.pop(forced))
    placed = []
    for index, (h, w) in enumerate(ordered):
        if h > height - 2 * margin or w > width - 2 * margin:
            return None
        positions = [(r, c) for r in range(margin, height - h - margin + 1)
                     for c in range(margin, width - w - margin + 1)]
        if require_boundary and index == 0:
            positions = [(r, c) for r, c in positions
                         if r == 0 or c == 0 or r + h == height or c + w == width]
        # Try every legal location in random order. This is bounded even at 30x30.
        success = False
        for j in rng.permutation(len(positions)):
            r, c = positions[int(j)]
            candidate = (r, c, r + h - 1, c + w - 1)
            if all(_separated(candidate, other) for other in placed):
                placed.append(candidate)
                success = True
                break
        if not success:
            return None
    return sorted(placed)


def _sample_layout(rng, height, width, split, index, allow_boundary,
                   need_degenerate, degenerate_only=False, allow_degenerate=True,
                   simple_inset=False):
    area = height * width
    capacity = 2 if area < 145 else 3 if area < 240 else 5 if area < 425 else 7
    if split == "train" and index == 0:
        productive = 1
        degenerate = int(need_degenerate)
    elif degenerate_only:
        productive = 0
        degenerate = int(rng.integers(1, min(4, capacity) + 1))
    else:
        upper = min(capacity, 4 if split == "train" else 7)
        productive = int(rng.integers(1, upper + 1))
        degenerate = int(rng.integers(0, min(3, capacity - productive) + 1))
        if need_degenerate and degenerate == 0:
            degenerate = 1
    if not allow_degenerate:
        degenerate = 0
    if simple_inset:
        productive, degenerate = 1, 0
    if productive + degenerate > capacity:
        productive = max(0, capacity - degenerate)
    require_boundary = allow_boundary and (index == 1 and split == "train"
                                           or rng.random() < (0.47 if split == "test" else 0.30))
    if simple_inset:
        require_boundary = False
    full_canvas = (allow_boundary and not simple_inset and not need_degenerate and not degenerate_only
                   and productive == 1 and rng.random() < (0.22 if split == "test" else 0.11))
    if full_canvas:
        return [(0, 0, height - 1, width - 1)]
    kinds = ["square_cell", "thin_h", "thin_v", "moderate", "wide", "tall", "large"]
    if split == "train" and index == 0:
        kinds = ["square_cell", "thin_h", "thin_v", "moderate"]
    for attempt in range(28):
        shapes = []
        for _ in range(productive):
            kind = str(rng.choice(kinds))
            shapes.append(_shape(rng, height, width, kind))
        for d in range(degenerate):
            kind = "deg_square" if (need_degenerate and d == 0
                                    and split == "train" and rng.random() < 0.52) else str(rng.choice(
                                        ["deg_square", "deg_h", "deg_v"]))
            shapes.append(_shape(rng, height, width, kind))
        rectangles = _place(rng, height, width, shapes, require_boundary,
                            margin=1 if simple_inset or not allow_boundary else 0)
        if rectangles is not None:
            return rectangles
    # Deterministic, bounded fallback keeps all requested objects and a forced
    # boundary where requested. The small shapes remain complete and separate.
    while productive + degenerate:
        shapes = [(3, 3)] * productive + [(2, 2)] * degenerate
        rectangles = _place(rng, height, width, shapes, require_boundary,
                            margin=1 if simple_inset or not allow_boundary else 0)
        if rectangles is not None:
            return rectangles
        if productive > 1:
            productive -= 1
        elif degenerate > 1:
            degenerate -= 1
        elif productive and degenerate and not (split == "train" and index == 0):
            if need_degenerate:
                productive = 0
            else:
                degenerate = 0
        else:
            break
    raise RuntimeError("could not place rectangles")


def _draw(height, width, rectangles, colors):
    background, frame, fill = colors
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = np.full((height, width), background, dtype=np.int8)
    for top, left, bottom, right in rectangles:
        input_grid[top, left:right + 1] = frame
        input_grid[bottom, left:right + 1] = frame
        input_grid[top:bottom + 1, left] = frame
        input_grid[top:bottom + 1, right] = frame
        output_grid[top + 1:bottom, left + 1:right] = fill
    return {"input": input_grid, "output": output_grid}


def _input_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    n_train, n_test = _counts(seed, rng, num_train, num_test)
    # Fill is the only color convention shared by all pairs. A common-palette
    # mode retains source-like examples; variable mode teaches color roles.
    if rng.random() < 0.22:
        base = (0, 2, 3)
    else:
        base = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
    fill = base[2]
    variable = rng.random() < 0.58
    allow_boundary = rng.random() < 0.82
    degenerate_training = rng.random() < 0.78
    degenerate_anchor = int(rng.integers(0, n_train)) if degenerate_training else -1
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", n_train), ("test", n_test)):
        for index in range(count):
            for attempt in range(24):
                pair_rng = _child_rng(rng)
                height, width = _canvas(pair_rng, split, index)
                deg_only = (index > 0 and pair_rng.random() <
                            (0.11 if split == "train" else 0.07)
                            and degenerate_training)
                # Degenerate examples are placed throughout both splits. Every
                # puzzle still has a productive first training demonstration.
                need_deg = degenerate_training and (split == "train" and index == degenerate_anchor
                        or pair_rng.random() < (0.30 if split == "train" else 0.26))
                if split == "test" and not degenerate_training:
                    need_deg = False
                simple_inset = split == "test" and pair_rng.random() < 0.19
                rectangles = _sample_layout(pair_rng, height, width, split, index,
                                            allow_boundary, need_deg, deg_only,
                                            split == "train" or degenerate_training,
                                            simple_inset)
                colors = _palette(pair_rng, fill, base, variable,
                                  index if split == "train" else index + n_train)
                pair = _draw(height, width, rectangles, colors)
                source_layout = (height, width, tuple(rectangles))
                key = _input_key(pair["input"])
                if key not in seen and (colors != (0, 2, 3)
                                        or source_layout not in _SOURCE_LAYOUTS):
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not create a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _rectangles(mask):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    result = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            top = min(y for y, _ in cells)
            bottom = max(y for y, _ in cells)
            left = min(x for _, x in cells)
            right = max(x for _, x in cells)
            assert bottom > top and right > left
            expected = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
            expected[0, :] = expected[-1, :] = True
            expected[:, 0] = expected[:, -1] = True
            assert np.array_equal(mask[top:bottom + 1, left:right + 1], expected)
            result.append((top, left, bottom, right))
    assert result
    result.sort()
    for i, rect in enumerate(result):
        assert all(_separated(rect, other) for other in result[:i])
    return result


def _inspect(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    in_colors = set(map(int, np.unique(inp)))
    out_colors = set(map(int, np.unique(out)))
    assert len(in_colors) == 2 and 1 <= len(out_colors) <= 2
    common = in_colors & out_colors
    assert len(common) == 1
    background = next(iter(common))
    frame = next(iter(in_colors - {background}))
    fill_candidates = out_colors - {background}
    fill = next(iter(fill_candidates)) if fill_candidates else None
    rectangles = _rectangles(inp == frame)
    expected = np.full(inp.shape, background, dtype=np.int8)
    productive = 0
    for top, left, bottom, right in rectangles:
        if bottom - top >= 2 and right - left >= 2:
            productive += 1
            assert fill is not None
            expected[top + 1:bottom, left + 1:right] = fill
    assert (fill is None) == (productive == 0)
    assert np.array_equal(out, expected)
    return (background, frame, fill), rectangles


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen = set()
        train_fills = set()
        all_analyses = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert all(1 <= dim <= 30 for dim in grid.shape)
                    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
                key = _input_key(pair["input"])
                assert key not in seen
                seen.add(key)
                roles, rectangles = _inspect(pair)
                all_analyses[split].append((roles, rectangles))
                h, w = pair["input"].shape
                if roles[2] is not None:
                    assert len(set(roles)) == 3
                    if split == "train":
                        train_fills.add(roles[2])
                if roles[:2] == (0, 2) and (roles[2] == 3 or roles[2] is None):
                    assert (h, w, tuple(rectangles)) not in _SOURCE_LAYOUTS
        assert len(train_fills) == 1
        fill = next(iter(train_fills))
        for split in ("train", "test"):
            for (background, frame, local_fill), rectangles in all_analyses[split]:
                assert background != fill and frame != fill
                assert local_fill is None or local_fill == fill
        # A productive demonstration is always first. The other examples may
        # have empty outputs, including the two-pair minimum.
        assert all_analyses["train"][0][0][2] == fill
        train_boundary = any(_boundary(rect, *pair["input"].shape)
                             for pair, (_, rects) in zip(puzzle["train"], all_analyses["train"])
                             for rect in rects)
        test_boundary = any(_boundary(rect, *pair["input"].shape)
                            for pair, (_, rects) in zip(puzzle["test"], all_analyses["test"])
                            for rect in rects)
        assert not test_boundary or train_boundary
        train_degenerate = any(2 in _rect_size(rect)
                               for _, rects in all_analyses["train"] for rect in rects)
        test_degenerate = any(2 in _rect_size(rect)
                              for _, rects in all_analyses["test"] for rect in rects)
        assert not test_degenerate or train_degenerate
        train_palettes = {(r[0], r[1]) for r, _ in all_analyses["train"]}
        test_palettes = {(r[0], r[1]) for r, _ in all_analyses["test"]}
        if len(train_palettes) == 1:
            assert test_palettes == train_palettes
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def _signature(puzzle):
    return tuple((split, tuple((pair["input"].shape, pair["input"].tobytes(),
                                pair["output"].tobytes()) for pair in puzzle[split]))
                 for split in ("train", "test"))


def self_test() -> None:
    assert _signature(generate_puzzle(314159, 7, 3)) == _signature(generate_puzzle(314159, 7, 3))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (6, 2),
                                  (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + i, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
        validate_generated_puzzle(puzzle)
    for i, (nt, ns) in enumerate(((1, None), (11, None), (None, 0),
                                  (None, 5), (2, 2), (3, 3), (3, 4))):
        try:
            generate_puzzle(2000 + i, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts = set(), set()
    shapes, rect_sizes, counts, palettes, signatures = set(), set(), set(), set(), set()
    small = large = nonsquare = boundary_train = boundary_test = degenerate = 0
    deg_only = long_deg = thin_h = thin_v = many = 0
    review = {name: 0 for name in (
        "train_medium", "test_small", "test_medium", "test_large",
        "test_single_inset", "train_blank", "test_blank", "multi_deg",
        "train_full_canvas", "test_full_canvas", "edge_inset_mix",
        "source_palette", "variable_palette_puzzles", "varying_train_shapes",
        "mixed_shape_sizes", "test_many", "train_many")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_pair_shapes = set()
        train_pair_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                roles, rects = _inspect(pair)
                h, w = pair["input"].shape
                if split == "train":
                    train_pair_shapes.add((h, w))
                    train_pair_palettes.add(roles[:2])
                shapes.add((h, w))
                palettes.add(roles[:2])
                counts.add(len(rects))
                small += h <= 13 and w <= 13
                large += h >= 22 or w >= 22
                nonsquare += h != w
                boundary_train += split == "train" and any(_boundary(r, h, w) for r in rects)
                boundary_test += split == "test" and any(_boundary(r, h, w) for r in rects)
                degenerate += any(2 in _rect_size(r) for r in rects)
                deg_only += all(2 in _rect_size(r) for r in rects)
                many += len(rects) >= 5
                if split == "train" and 14 <= h <= 21 and 14 <= w <= 21:
                    review["train_medium"] += 1
                if split == "test":
                    if h <= 13 and w <= 13:
                        review["test_small"] += 1
                    if 14 <= h <= 21 and 14 <= w <= 21:
                        review["test_medium"] += 1
                    if h >= 22 or w >= 22:
                        review["test_large"] += 1
                    if len(rects) == 1 and not _boundary(rects[0], h, w):
                        review["test_single_inset"] += 1
                if all(2 in _rect_size(r) for r in rects):
                    review["train_blank" if split == "train" else "test_blank"] += 1
                if sum(2 in _rect_size(r) for r in rects) >= 2:
                    review["multi_deg"] += 1
                if (0, 0, h - 1, w - 1) in rects:
                    review["train_full_canvas" if split == "train" else "test_full_canvas"] += 1
                if any(_boundary(r, h, w) for r in rects) and any(
                        not _boundary(r, h, w) for r in rects):
                    review["edge_inset_mix"] += 1
                if roles[:2] == (0, 2) and (roles[2] in (3, None)):
                    review["source_palette"] += 1
                if len({_rect_size(r) for r in rects}) >= 2:
                    review["mixed_shape_sizes"] += 1
                if len(rects) >= 5:
                    review["train_many" if split == "train" else "test_many"] += 1
                for rect in rects:
                    rh, rw = _rect_size(rect)
                    rect_sizes.add((rh, rw))
                    long_deg += (rh == 2 and rw > 2) or (rw == 2 and rh > 2)
                    thin_h += rh == 3 and rw >= 7
                    thin_v += rw == 3 and rh >= 7
        review["variable_palette_puzzles"] += len(train_pair_palettes) > 1
        review["varying_train_shapes"] += len(train_pair_shapes) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(rect_sizes) >= 30 and len(counts) >= 5 and len(palettes) >= 20
    assert min(small, large, nonsquare, boundary_train, boundary_test,
               degenerate, deg_only, long_deg, thin_h, thin_v, many) > 0
    assert review["variable_palette_puzzles"] >= 10
    assert review["varying_train_shapes"] == 42
    assert all(value > 0 for value in review.values())
