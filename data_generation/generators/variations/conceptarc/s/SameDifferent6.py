"""Generate ARC puzzles that remove both objects with a repeated defining color."""

from collections import Counter

import numpy as np


_SOURCE_INPUTS = {
    ("000000000", "022200222", "021200212", "022200222", "000000000",
     "044400888", "041400818", "044400888", "000000000"),
    ("00000000000", "00000033333", "06660037773", "06760037773",
     "06660037773", "00000033333", "55555000000", "55555333300",
     "55755377300", "55555377300", "55555333300"),
    ("000000000000", "000000000000", "044444000111", "040004000101",
     "044444000101", "000004440101", "000004040111", "777004040000",
     "707004440000", "707000000000", "707000000000", "777000000000"),
    ("000000000000", "003330000000", "003830000000", "003330000000",
     "000000000000", "000002220000", "000002820000", "011102220000",
     "018100000222", "011100000282", "000000000222", "000000000000"),
    ("000000000000", "003333000000", "003223004440", "003223004240",
     "003333004440", "000000000000", "000000000000", "000000000000",
     "004440000000", "004240000000", "004240000000", "004440000000"),
    ("00000000888880", "00000000844480", "00666660888880",
     "00644460000000", "00644460003330", "00644460003430", "00666660003330",
     "00000000000000", "00000000000600", "00007000006460", "00074700064446",
     "00744470006460", "00074700000600", "00007000000000"),
}


def _grid_rows(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _pick_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError("Invalid train or test pair count")
    legal = [(t, s) for t in range(2, 11) for s in range(1, 5)
             if t > s and (num_train is None or num_train == t)
             and (num_test is None or num_test == s)]
    if not legal:
        raise ValueError("num_train must be greater than num_test")
    return legal[int(rng.integers(len(legal)))]


def _boundary(footprint):
    padded = np.pad(footprint, 1)
    h, w = footprint.shape
    return footprint & (~padded[:h, 1:w+1] | ~padded[2:h+2, 1:w+1] |
                        ~padded[1:h+1, :w] | ~padded[1:h+1, 2:w+2])


def _mask(kind, h, w, orientation=0, split=3):
    if kind == "rect":
        mask = np.ones((h, w), dtype=bool)
    elif kind == "diamond":
        assert h == w and h in (5, 7)
        r = h // 2
        y, x = np.indices((h, w))
        mask = abs(y-r) + abs(x-r) <= r
    elif kind == "chamfer":
        assert 4 <= h <= 7 and 4 <= w <= 7
        mask = np.ones((h, w), dtype=bool)
        mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
    elif kind == "bent":
        assert 5 <= h <= 7 and 5 <= w <= 7
        mask = np.zeros((h, w), dtype=bool)
        mask[:3, :] = True
        mask[:, -3:] = True
    elif kind == "step":
        assert 5 <= h <= 7 and 5 <= w <= 7 and 3 <= split <= h-2
        mask = np.zeros((h, w), dtype=bool)
        mask[:split, :w-2] = True
        mask[split-1:, 2:] = True
    else:
        raise AssertionError("Unknown shape")
    return np.rot90(mask, orientation)


def _new_shape(rng, kind):
    if kind == "rect":
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
    elif kind == "diamond":
        h = w = int(rng.choice([5, 7]))
    elif kind == "chamfer":
        h, w = int(rng.integers(4, 8)), int(rng.integers(4, 8))
    else:
        h, w = int(rng.integers(5, 8)), int(rng.integers(5, 8))
    orientation = int(rng.integers(4))
    split = int(rng.integers(3, h-1)) if kind == "step" else 3
    footprint = _mask(kind, h, w, orientation, split)
    return {"kind": kind, "footprint": footprint, "boundary": _boundary(footprint)}


def _sample_colors(rng, count, previous_repeated=None):
    background = 0 if rng.random() < 0.72 else int(rng.integers(1, 10))
    available = [c for c in range(10) if c != background]
    repeated_pool = [c for c in available if c != previous_repeated]
    repeated = int(rng.choice(repeated_pool))
    singletons = [int(c) for c in rng.choice(
        [c for c in available if c != repeated], size=count-2, replace=False)]
    used = {background, repeated, *singletons}
    pigment = int(rng.choice([c for c in range(10) if c not in used]))
    return background, [repeated, repeated] + singletons, pigment


def _shape_plan(rng, count, anchor, rectangle_only, touching):
    families = ["rect", "diamond", "chamfer", "bent", "step"]
    probs = [0.39, 0.14, 0.16, 0.17, 0.14]
    kinds = ["rect" if rectangle_only else str(rng.choice(families, p=probs))
             for _ in range(count)]
    if anchor:
        family = str(rng.choice(families[1:]))
        kinds[0], kinds[1], kinds[2] = family, "rect", family
    if touching:
        kinds[0], kinds[2] = "rect", "rect"
    return kinds


def _styles(rng, count, anchor, all_hollow):
    if all_hollow:
        return ["hollow"] * count
    choices = ["hollow", "full", "spot", "body"]
    styles = [str(rng.choice(choices, p=[0.29, 0.34, 0.23, 0.14]))
              for _ in range(count)]
    if anchor:
        styles[0] = str(rng.choice(["full", "spot"]))
        styles[2] = str(rng.choice(["full", "spot"]))
    return styles


def _dilate(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    expanded = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            expanded |= padded[dr:dr+h, dc:dc+w]
    return expanded


def _place_objects(rng, specs, height, width, touching, separation):
    count = len(specs)
    for _ in range(24):
        occupied = np.zeros((height, width), dtype=bool)
        placed = [None] * count
        order = list(range(count))
        if touching:
            first, second = specs[0], specs[2]
            h0, w0 = first["footprint"].shape
            h2, w2 = second["footprint"].shape
            if h0+h2 > height or w0+w2 > width:
                return None
            top = int(rng.integers(height-h0-h2+1))
            left = int(rng.integers(width-w0-w2+1))
            vertical_flip = bool(rng.integers(2))
            horizontal_flip = bool(rng.integers(2))
            y0 = top+h2 if vertical_flip else top
            y2 = top if vertical_flip else top+h0
            x0 = left+w2 if horizontal_flip else left
            x2 = left if horizontal_flip else left+w0
            for idx, y, x in ((0, y0, x0), (2, y2, x2)):
                fp = specs[idx]["footprint"]
                occupied[y:y+fp.shape[0], x:x+fp.shape[1]] |= fp
                placed[idx] = (y, x)
            order = [i for i in order if i not in (0, 2)]
        rng.shuffle(order)
        succeeded = True
        for idx in order:
            spec = specs[idx]
            fp = spec["footprint"]
            h, w = fp.shape
            if h > height or w > width:
                succeeded = False
                break
            accepted = False
            for _ in range(190):
                top = int(rng.integers(height-h+1))
                left = int(rng.integers(width-w+1))
                candidate = np.zeros((height, width), dtype=bool)
                candidate[top:top+h, left:left+w] = fp
                if np.any(candidate & occupied):
                    continue
                if separation and np.any(candidate & _dilate(occupied)):
                    continue
                if any(spec["color"] == specs[j]["color"] and
                       np.any(candidate & _dilate(_global_footprint(
                           specs[j]["footprint"], placed[j], height, width)))
                       for j in range(count) if placed[j] is not None):
                    continue
                occupied |= candidate
                placed[idx] = (top, left)
                accepted = True
                break
            if not accepted:
                succeeded = False
                break
        if succeeded:
            return placed
    return None


def _global_footprint(footprint, position, height, width):
    result = np.zeros((height, width), dtype=bool)
    y, x = position
    h, w = footprint.shape
    result[y:y+h, x:x+w] = footprint
    return result


def _draw_pair(rng, regime, count, previous_repeated, anchor=False,
               all_hollow=False):
    rectangle_only = not anchor and rng.random() < (0.22 if regime == "target_test" else 0.20)
    touching = not anchor and rng.random() < (0.25 if regime == "target_test" else 0.20)
    separation = 0 if not touching and rng.random() < 0.12 else 1
    kinds = _shape_plan(rng, count, anchor, rectangle_only, touching)
    styles = _styles(rng, count, anchor, all_hollow)
    background, colors, pigment = _sample_colors(rng, count, previous_repeated)
    specs = [_new_shape(rng, kind) for kind in kinds]
    for spec, color, style in zip(specs, colors, styles):
        spec["color"], spec["style"] = color, style

    if count <= 4:
        low, high = (10, 20) if rectangle_only else (12, 22)
    elif count == 5:
        low, high = 16, 24
    else:
        low, high = 18, 27
    height = int(rng.integers(low, high+1))
    width = int(rng.integers(low, high+1))
    if touching:
        height = max(height, specs[0]["footprint"].shape[0] + specs[2]["footprint"].shape[0])
        width = max(width, specs[0]["footprint"].shape[1] + specs[2]["footprint"].shape[1])
    for growth in range(7):
        h, w = min(30, height+growth*2), min(30, width+growth*2)
        positions = _place_objects(rng, specs, h, w, touching, separation)
        if positions is not None:
            break
    else:
        raise RuntimeError("Could not place distinct objects")

    input_grid = np.full((h, w), background, dtype=np.int8)
    erased = np.zeros((h, w), dtype=bool)
    for idx, (spec, position) in enumerate(zip(specs, positions)):
        top, left = position
        fp, boundary = spec["footprint"], spec["boundary"]
        sh, sw = fp.shape
        view = input_grid[top:top+sh, left:left+sw]
        view[boundary] = spec["color"]
        inside = fp & ~boundary
        if spec["style"] == "full":
            view[inside] = pigment
        elif spec["style"] == "body":
            view[inside] = spec["color"]
        elif spec["style"] == "spot":
            view[inside] = spec["color"]
            cells = np.argwhere(inside)
            if len(cells):
                y, x = cells[int(rng.integers(len(cells)))]
                view[y, x] = pigment
        if idx < 2:
            erased[top:top+sh, left:left+sw] |= fp
    output_grid = input_grid.copy()
    output_grid[erased] = background
    # Keep the background inferable even in a compact scene with two large
    # same-colored bodies. Padding is sampled only when that color dominates.
    while True:
        values, counts = np.unique(input_grid, return_counts=True)
        background_count = int(counts[values == background][0])
        if background_count > max(int(n) for v, n in zip(values, counts)
                                  if v != background):
            break
        assert max(input_grid.shape) < 30
        input_grid = np.pad(input_grid, 1, constant_values=background)
        output_grid = np.pad(output_grid, 1, constant_values=background)
    return input_grid, output_grid, colors[0]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _pick_counts(rng, num_train, num_test)
    # The repeated defining color is the only puzzle-level convention.
    # All color identities and geometric properties belong to each input.
    anchor_index = int(rng.integers(train_count))
    hollow_index = int(rng.integers(train_count)) if train_count > 2 and rng.random() < 0.5 else -1
    if hollow_index == anchor_index:
        hollow_index = (hollow_index+1) % train_count
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_repeated = None
    anchor_count = 4
    for regime, n, split in (("source_train", train_count, "train"),
                             ("target_test", test_count, "test")):
        for pair_index in range(n):
            anchor = split == "train" and pair_index == anchor_index
            if anchor:
                count = anchor_count
            elif split == "train" and pair_index == (anchor_index+1) % train_count:
                count = int(rng.choice([3, 5, 6]))
            else:
                probabilities = ([0.25, 0.20, 0.35, 0.20] if split == "test"
                                 else [0.26, 0.30, 0.28, 0.16])
                count = int(rng.choice([3, 4, 5, 6], p=probabilities))
            all_hollow = pair_index == hollow_index and split == "train"
            if not anchor and not all_hollow:
                all_hollow = rng.random() < 0.17
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                inp, out, repeated = _draw_pair(
                    pair_rng, regime, count, previous_repeated if split == "train" else None,
                    anchor, all_hollow)
                key = _grid_key(inp)
                if key not in seen and _grid_rows(inp) not in _SOURCE_INPUTS:
                    seen.add(key)
                    puzzle[split].append({"input": inp, "output": out})
                    if split == "train":
                        previous_repeated = repeated
                    break
            else:
                raise RuntimeError("Could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components_of_color(grid, color):
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros_like(mask)
    components = []
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        cells = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = row+dy, col+dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        components.append(cells)
    return components


def _matching_shapes(component, color, grid):
    rows, cols = zip(*component)
    top, left, bottom, right = min(rows), min(cols), max(rows)+1, max(cols)+1
    h, w = bottom-top, right-left
    colored = np.zeros((h, w), dtype=bool)
    for y, x in component:
        colored[y-top, x-left] = True
    candidates = []
    shapes = [("rect", 0, 3)]
    if h == w and h in (5, 7):
        shapes.append(("diamond", 0, 3))
    if 4 <= h <= 7 and 4 <= w <= 7:
        shapes.append(("chamfer", 0, 3))
    if 5 <= h <= 7 and 5 <= w <= 7:
        shapes += [("bent", o, 3) for o in range(4)]
        shapes += [("step", o, s) for o in range(4)
                   for s in range(3, ((w if o % 2 else h)-1))]
    for kind, orientation, split in shapes:
        if kind == "rect" and not (3 <= h <= 7 and 3 <= w <= 7):
            continue
        if kind == "rect":
            fp = _mask(kind, h, w)
        elif kind == "diamond":
            fp = _mask(kind, h, w)
        elif kind == "chamfer":
            fp = _mask(kind, h, w)
        else:
            # Rotation may interchange the unrotated height and width.
            raw_h, raw_w = (w, h) if orientation % 2 else (h, w)
            if kind == "step" and not 3 <= split <= raw_h-2:
                continue
            fp = _mask(kind, raw_h, raw_w, orientation, split)
        if fp.shape != (h, w):
            continue
        boundary = _boundary(fp)
        if np.all(colored[boundary]) and np.all(~colored | fp):
            candidates.append((kind, fp))
    return top, left, candidates


def _parse_objects(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    assert np.count_nonzero(counts == counts.max()) == 1
    candidates = []
    h, w = grid.shape
    for color in values:
        color = int(color)
        if color == background:
            continue
        for component in _components_of_color(grid, color):
            top, left, shapes = _matching_shapes(component, color, grid)
            for kind, fp in shapes:
                whole = _global_footprint(fp, (top, left), h, w)
                candidates.append((int(fp.sum()), color, kind, whole, top, left))
    candidates.sort(key=lambda c: -c[0])
    claimed = np.zeros((h, w), dtype=bool)
    records = []
    for area, color, kind, footprint, top, left in candidates:
        if np.any(footprint & claimed):
            continue
        boundary = _boundary(footprint)
        if not np.all(grid[boundary] == color):
            continue
        records.append({"outline": color, "kind": kind, "footprint": footprint,
                        "boundary": boundary, "box": (top, left)})
        claimed |= footprint
    assert np.all((grid == background) | claimed)
    assert 3 <= len(records) <= 6
    outlines = [record["outline"] for record in records]
    pigment_colors = set()
    for record in records:
        inside = record["footprint"] & ~record["boundary"]
        pigment_colors.update(int(c) for c in np.unique(grid[inside])
                              if c not in (background, record["outline"]))
    assert not pigment_colors.intersection(outlines)
    assert len(pigment_colors) <= 1
    return background, records, pigment_colors


def _check_pair(pair, regime):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    for grid in pair.values():
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape and _grid_rows(inp) not in _SOURCE_INPUTS
    bg, records, pigment = _parse_objects(inp)
    counts = Counter(r["outline"] for r in records)
    repeated = [c for c, n in counts.items() if n == 2]
    assert len(repeated) == 1 and all(n <= 2 for n in counts.values())
    expected = inp.copy()
    for record in records:
        if record["outline"] == repeated[0]:
            expected[record["footprint"]] = bg
    assert np.array_equal(expected, out) and not np.array_equal(inp, out)
    return bg, records, pigment, repeated[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _check_pair(pair, "source_train" if split == "train" else "target_test")
            key = _grid_key(pair["input"])
            pair_key = (key, _grid_key(pair["output"]))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            analyses[split].append(analysis)
    assert len({a[3] for a in analyses["train"]}) >= 2
    # An erased and a preserved object both carry interior pigment in a
    # demonstration, ruling out an interior-color deletion rule.
    assert any(a[2] and any(np.any(p["input"][r["footprint"] &
                                                ~r["boundary"]] == next(iter(a[2])))
                                 for r in a[1] if r["outline"] == a[3]) and
               any(np.any(p["input"][r["footprint"] &
                                    ~r["boundary"]] == next(iter(a[2])))
                   for r in a[1] if r["outline"] != a[3])
               for p, a in zip(puzzle["train"], analyses["train"]))
    assert any(any(r["kind"] != "rect" and r["outline"] == a[3] for r in a[1])
               and any(r["kind"] != "rect" and r["outline"] != a[3] for r in a[1])
               for a in analyses["train"])


def _same_puzzle(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    observed = {"families": set(), "counts": set(), "shapes": set(),
                "backgrounds": set(), "rect_test": 0, "hollow": 0,
                "touch": 0, "retained_irregular": 0, "erased_irregular": 0,
                "rect_irregular_erasure": 0, "mixed_pair": 0,
                "non_square": 0, "distinct_train_counts": 0,
                "distinct_train_sizes": 0, "different_colors": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        analyses = {s: [_check_pair(p, "source_train" if s == "train" else "target_test")
                        for p in puzzle[s]] for s in ("train", "test")}
        if len({len(a[1]) for a in analyses["train"]}) > 1:
            observed["distinct_train_counts"] += 1
        if len({p["input"].shape for p in puzzle["train"]}) > 1:
            observed["distinct_train_sizes"] += 1
        if len({tuple(sorted(int(x) for x in np.unique(p["input"])))
                for p in puzzle["train"]}) > 1:
            observed["different_colors"] += 1
        for split in ("train", "test"):
            for pair, (bg, records, pigment, repeated) in zip(puzzle[split], analyses[split]):
                observed["counts"].add(len(records))
                observed["shapes"].add(pair["input"].shape)
                observed["backgrounds"].add(bg)
                observed["families"].update(r["kind"] for r in records)
                if pair["input"].shape[0] != pair["input"].shape[1]:
                    observed["non_square"] += 1
                if not pigment:
                    observed["hollow"] += 1
                if split == "test" and all(r["kind"] == "rect" for r in records):
                    observed["rect_test"] += 1
                if any(r["kind"] != "rect" and r["outline"] != repeated for r in records):
                    observed["retained_irregular"] += 1
                if any(r["kind"] != "rect" and r["outline"] == repeated for r in records):
                    observed["erased_irregular"] += 1
                if any(r["kind"] != "rect" and r["outline"] == repeated for r in records) and any(
                        r["kind"] == "rect" and r["outline"] == repeated for r in records):
                    observed["rect_irregular_erasure"] += 1
                if any(r["kind"] != "rect" and r["outline"] == repeated for r in records) and any(
                        r["kind"] != "rect" and r["outline"] != repeated for r in records):
                    observed["mixed_pair"] += 1
                for i, a in enumerate(records):
                    for b in records[i+1:]:
                        if a["outline"] == b["outline"]:
                            continue
                        if np.any(a["footprint"] & _dilate(b["footprint"])):
                            observed["touch"] += 1
                            break
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert observed["counts"] == {3, 4, 5, 6}
    assert observed["families"] == {"rect", "diamond", "chamfer", "bent", "step"}
    assert len(observed["shapes"]) >= 50 and len(observed["backgrounds"]) >= 4
    for key in ("rect_test", "hollow", "touch", "retained_irregular",
                "erased_irregular", "rect_irregular_erasure", "mixed_pair",
                "non_square", "distinct_train_counts", "distinct_train_sizes",
                "different_colors"):
        assert observed[key] > 0, key
    for t, s in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                 (2, None), (10, None), (5, 3)):
        puzzle = generate_puzzle(93, t, s)
        validate_generated_puzzle(puzzle)
        assert (t is None or len(puzzle["train"]) == t)
        assert (s is None or len(puzzle["test"]) == s)
        assert _same_puzzle(puzzle, generate_puzzle(93, t, s))
    for t, s in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(7, t, s)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
    assert not _same_puzzle(generate_puzzle(0, 4, 2), generate_puzzle(1, 4, 2))
    altered = generate_puzzle(8, 2, 1)
    altered["test"][0]["output"] = altered["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(altered)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a wrong output")
