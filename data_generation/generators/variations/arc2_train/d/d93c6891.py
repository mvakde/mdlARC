"""ARC count-transfer generator: contiguous exterior bars fill target layers."""

import hashlib
import numpy as np

ORIENTATIONS = ("top", "bottom", "left", "right")
SOURCE_PAIR_HASHES = {
    "1fc0d9d441272001fef89f715586fcbc76e4be7a40cee6e4c078574ed217e29f",
    "acbf46f3703abdc7035599eef665e28237364407442097aed68046c5a1dd3271",
    "6c45c6d2bb61cf4343e3c2668cec68468188abf9d220e267c88fa45f8cff0812",
    "57fc2954a8a780b52f7d34c2bf919e70e77e54d947a1ae7ff17d42b27d7a0250",
}


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_hash(inp, out):
    shape = bytes((inp.shape[0], inp.shape[1]))
    return hashlib.sha256(shape + inp.astype(np.uint8).tobytes() + out.astype(np.uint8).tobytes()).hexdigest()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, group = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            group.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(group)
    return groups


def _canvas(rng, count, regime):
    low = 9 if count <= 2 else 14
    wh = int(rng.integers(low, 27 if regime == "source_train" else 28))
    ww = int(rng.integers(low, 28 if regime == "source_train" else 29))
    if regime == "target_test" and rng.random() < 0.38 and max(ww, wh) < 30:
        ww = int(rng.integers(max(ww, wh), 30))
    eh = int(rng.integers(1, min(7, 31 - wh)))
    ew = int(rng.integers(1, min(7, 31 - ww)))
    top = int(rng.integers(0, eh + 1))
    left = int(rng.integers(0, ew + 1))
    return (wh + eh, ww + ew), (top, left, wh, ww)


def _separated(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah < br or br + bh < ar or ac + aw < bc or bc + bw < ac


def _rectangles(rng, work, count, inert, full_index):
    wr, wc, wh, ww = work
    rectangles = []
    for index in range(count):
        if index == full_index and rng.random() < 0.75:
            h = int(rng.integers(2, min(4, wh) + 1))
            w = int(rng.integers(2, min(4, ww) + 1))
        else:
            hi_h = min(wh, 7 if count <= 2 else 6)
            hi_w = min(ww, 7 if count <= 2 else 6)
            h = 1 if index in inert and rng.random() < 0.24 else int(rng.integers(2, hi_h + 1))
            w = 1 if index in inert and rng.random() < 0.24 else int(rng.integers(2, hi_w + 1))
        if h == w == 1:
            w = 2
        choices = []
        for r in range(wr, wr + wh - h + 1):
            for c in range(wc, wc + ww - w + 1):
                rect = (r, c, h, w)
                if all(_separated(rect, other) for other in rectangles):
                    choices.append(rect)
        if not choices:
            return None
        edges = [x for x in choices if x[0] == wr or x[1] == wc or x[0] + h == wr + wh or x[1] + w == wc + ww]
        if edges and rng.random() < 0.33:
            choices = edges
        rectangles.append(choices[int(rng.integers(len(choices)))])
    return rectangles


def _ray(rect, orientation, side, work):
    r, c, h, w = rect
    wr, wc, wh, ww = work
    if orientation in ("top", "bottom"):
        y = r if orientation == "top" else r + h - 1
        if side == 0:
            return [(y, x) for x in range(c - 1, wc - 1, -1)]
        return [(y, x) for x in range(c + w, wc + ww)]
    x = c if orientation == "left" else c + w - 1
    if side == 0:
        return [(y, x) for y in range(r - 1, wr - 1, -1)]
    return [(y, x) for y in range(r + h, wr + wh)]


def _capacity(ray, occupied, targets):
    result = []
    for cell in ray:
        if cell in occupied or cell in targets:
            break
        result.append(cell)
    if len(result) < len(ray) and result:
        result.pop()  # a field cell separates the run from a distant block or cue
    return result


def _fill(out, rect, orientation, layers, payload):
    r, c, h, w = rect
    if orientation == "top":
        out[r:r + layers, c:c + w] = payload
    elif orientation == "bottom":
        out[r + h - layers:r + h, c:c + w] = payload
    elif orientation == "left":
        out[r:r + h, c:c + layers] = payload
    else:
        out[r:r + h, c + w - layers:c + w] = payload


def _annotations(rng, rectangles, work, annotated, forced, depth_mode, split_mode):
    targets = {(y, x) for r, c, h, w in rectangles for y in range(r, r + h) for x in range(c, c + w)}
    occupied, records = set(), []
    for index in annotated:
        rect = rectangles[index]
        orientations = [forced[index]] if index in forced else list(ORIENTATIONS)
        if index not in forced:
            rng.shuffle(orientations)
        plans = []
        for orientation in orientations:
            before = _capacity(_ray(rect, orientation, 0, work), occupied, targets)
            after = _capacity(_ray(rect, orientation, 1, work), occupied, targets)
            edge = rect[3] if orientation in ("top", "bottom") else rect[2]
            depth = rect[2] if orientation in ("top", "bottom") else rect[3]
            for layers in range(1, depth + 1):
                needed = edge * layers
                for split in (False, True):
                    feasible = (len(before) > 0 and len(after) > 0 and len(before) + len(after) >= needed and needed >= 2) if split else max(len(before), len(after)) >= needed
                    if feasible:
                        plans.append((orientation, layers, split, before, after))
        if depth_mode.get(index) == "full":
            plans = [p for p in plans if p[1] == (rect[2] if p[0] in ("top", "bottom") else rect[3])]
        elif depth_mode.get(index) == "partial":
            plans = [p for p in plans if p[1] < (rect[2] if p[0] in ("top", "bottom") else rect[3])]
        if split_mode.get(index) is not None:
            plans = [p for p in plans if p[2] == split_mode[index]]
        if not plans:
            return None
        weights = np.array([1 / (p[1] ** 0.65) for p in plans], dtype=float)
        weights /= weights.sum()
        orientation, layers, split, before, after = plans[int(rng.choice(len(plans), p=weights))]
        needed = (rect[3] if orientation in ("top", "bottom") else rect[2]) * layers
        if split:
            lo, hi = max(1, needed - len(after)), min(needed - 1, len(before))
            take_before = int(rng.integers(lo, hi + 1))
            cells = before[:take_before] + after[:needed - take_before]
        else:
            sides = [side for side in (before, after) if len(side) >= needed]
            cells = sides[int(rng.integers(len(sides)))][:needed]
        occupied.update(cells)
        records.append((index, orientation, layers, cells))
    return occupied, records


def _pair(rng, regime, index, colors):
    for _ in range(550):
        if regime == "source_train":
            count = int(rng.integers(1, 4))
            annotated_count = int(rng.integers(1, min(2, count) + 1))
        else:
            count = int(rng.integers(1, 5))
            annotated_count = int(rng.integers(1, min(3, count) + 1))
        annotated = [int(x) for x in rng.permutation(count)[:annotated_count]]
        inert = set(range(count)) - set(annotated)
        forced, depth_mode, split_mode = {}, {}, {}
        main = annotated[0]
        if regime == "source_train" and index < 2:
            forced[main] = ("top", "bottom")[int(rng.integers(2))] if index == 0 else ("left", "right")[int(rng.integers(2))]
            depth_mode[main] = "partial" if index == 0 else "full"
            split_mode[main] = bool(rng.integers(2))
        elif rng.random() < 0.42:
            forced[main] = ORIENTATIONS[int(rng.integers(4))]
        shape, work = _canvas(rng, count, regime)
        rectangles = _rectangles(rng, work, count, inert, main if depth_mode.get(main) == "full" else -1)
        if rectangles is None:
            continue
        result = _annotations(rng, rectangles, work, annotated, forced, depth_mode, split_mode)
        if result is None:
            continue
        cells, records = result
        background, field, target, payload = colors
        inp = np.full(shape, background, dtype=np.int8)
        wr, wc, wh, ww = work
        inp[wr:wr + wh, wc:wc + ww] = field
        for r, c, h, w in rectangles:
            inp[r:r + h, c:c + w] = target
        for r, c in cells:
            inp[r, c] = payload
        out = inp.copy()
        for r, c in cells:
            out[r, c] = field
        for owner, orientation, layers, _ in records:
            _fill(out, rectangles[owner], orientation, layers, payload)
        try:
            _analyze_pair(inp, out)
        except AssertionError:
            continue
        return {"input": inp, "output": out}
    raise ValueError("could not construct a clear pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count = _count("num_train", num_train, 2, 10)
    test_count = _count("num_test", num_test, 1, 4)
    if train_count is not None and test_count is not None and train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    if test_count is None and train_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(test_count + 1, 11))
    elif train_count is None:
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    base = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    mixed = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            for _ in range(12):
                colors = base if not mixed or rng.random() < 0.28 else tuple(int(x) for x in rng.choice(10, size=4, replace=False))
                pair = _pair(rng, regime, index, colors)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not produce unique pair inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(inp, out):
    assert inp.shape == out.shape
    changed = inp != out
    assert np.any(changed)
    transitions = {(int(a), int(b)) for a, b in zip(inp[changed], out[changed])}
    sources = {a for a, _ in transitions}
    destinations = {b for _, b in transitions}
    common = sources & destinations
    assert len(transitions) == 2 and len(common) == 1
    payload = next(iter(common))
    assert len(sources - {payload}) == len(destinations - {payload}) == 1
    target = next(iter(sources - {payload}))
    field = next(iter(destinations - {payload}))
    assert transitions == {(payload, field), (target, payload)}
    used = {int(x) for x in np.unique(inp)}
    assert len(used) == 4 and {field, target, payload} <= used
    background = next(iter(used - {field, target, payload}))
    return background, field, target, payload


def _analyze_pair(inp, out):
    background, field, target, payload = _roles(inp, out)
    rows, cols = np.nonzero(inp != background)
    assert len(rows) > 0
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
    assert np.all(inp[r0:r1 + 1, c0:c1 + 1] != background)
    outside = np.ones(inp.shape, dtype=bool)
    outside[r0:r1 + 1, c0:c1 + 1] = False
    assert np.all(inp[outside] == background) and np.all(out[outside] == background)
    work = (r0, c0, r1 - r0 + 1, c1 - c0 + 1)
    rectangles = []
    for component in _components(inp == target):
        ys, xs = zip(*component)
        r, c, h, w = min(ys), min(xs), max(ys) - min(ys) + 1, max(xs) - min(xs) + 1
        assert len(component) == h * w and np.all(inp[r:r + h, c:c + w] == target)
        rectangles.append((r, c, h, w))
    assert rectangles
    assert all(_separated(a, b) for i, a in enumerate(rectangles) for b in rectangles[i + 1:])
    memberships = {}
    annotations = []
    expected = inp.copy()
    expected[inp == payload] = field
    for index, rect in enumerate(rectangles):
        choices = []
        for orientation in ORIENTATIONS:
            sides = []
            for side in (0, 1):
                run = []
                for cell in _ray(rect, orientation, side, work):
                    if inp[cell] != payload:
                        break
                    run.append(cell)
                sides.append(run)
            if sides[0] or sides[1]:
                choices.append((orientation, sides))
        assert len(choices) <= 1
        if not choices:
            continue
        orientation, sides = choices[0]
        cells = sides[0] + sides[1]
        for cell in cells:
            memberships[cell] = memberships.get(cell, 0) + 1
        edge = rect[3] if orientation in ("top", "bottom") else rect[2]
        depth = rect[2] if orientation in ("top", "bottom") else rect[3]
        assert len(cells) > 0 and len(cells) % edge == 0
        layers = len(cells) // edge
        assert 1 <= layers <= depth
        _fill(expected, rect, orientation, layers, payload)
        annotations.append({"rectangle": index, "orientation": orientation, "layers": layers,
                            "depth": depth, "edge_length": edge, "payload_count": len(cells),
                            "side_lengths": tuple(map(len, sides)), "split": bool(sides[0] and sides[1])})
    all_payload = {tuple(map(int, cell)) for cell in np.argwhere(inp == payload)}
    assert set(memberships) == all_payload and all(v == 1 for v in memberships.values())
    assert np.array_equal(expected, out)
    assert int(np.count_nonzero((inp == payload) & (out == field))) == int(np.count_nonzero((inp == target) & (out == payload))) > 0
    return {"shape": inp.shape, "work": work, "rectangles": rectangles, "annotations": annotations, "roles": (background, field, target, payload)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    infos = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert _source_hash(pair["input"], pair["output"]) not in SOURCE_PAIR_HASHES
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info = _analyze_pair(pair["input"], pair["output"])
            assert 1 <= len(info["rectangles"]) <= (3 if split == "train" else 4)
            assert 1 <= len(info["annotations"]) <= (2 if split == "train" else 3)
            infos[split].append(info)
    train_annotations = [a for info in infos["train"] for a in info["annotations"]]
    assert any(a["orientation"] in ("top", "bottom") for a in train_annotations)
    assert any(a["orientation"] in ("left", "right") for a in train_annotations)
    assert any(a["layers"] < a["depth"] for a in train_annotations)
    assert any(a["layers"] == a["depth"] for a in train_annotations)


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) for s in ("train", "test")) and all(np.array_equal(a[k], b[k]) for split in ("train", "test") for a, b in zip(first[split], second[split]) for k in ("input", "output"))


def self_test() -> None:
    counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    orientations, split_modes, depths, boundary, inert = set(), set(), set(), set(), set()
    dimensions, flush_sides, train_objects, test_objects = set(), set(), set(), set()
    shared_lines = crossings = unequal_splits = combined = 0
    stable_palettes = mixed_palettes = varied_shapes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_roles, puzzle_shapes = set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_pair(pair["input"], pair["output"])
                shapes.add(info["shape"])
                puzzle_shapes.add(info["shape"])
                palettes.add(info["roles"])
                puzzle_roles.add(info["roles"])
                layouts.add((info["shape"], info["work"], tuple(info["rectangles"])))
                inert.add(len(info["rectangles"]) - len(info["annotations"]))
                (train_objects if section == "train" else test_objects).add(len(info["rectangles"]))
                wr, wc, wh, ww = info["work"]
                H, W = info["shape"]
                for side, margin in zip(ORIENTATIONS, (wr, H - wr - wh, wc, W - wc - ww)):
                    if margin == 0:
                        flush_sides.add(side)
                has_boundary = any(r == wr or c == wc or r + h == wr + wh or c + w == wc + ww for r, c, h, w in info["rectangles"])
                boundary.add(has_boundary)
                for _, _, h, w in info["rectangles"]:
                    dimensions.add((h, w))
                for i, a in enumerate(info["rectangles"]):
                    for b in info["rectangles"][i + 1:]:
                        if ({a[0], a[0] + a[2] - 1} & {b[0], b[0] + b[2] - 1}) or ({a[1], a[1] + a[3] - 1} & {b[1], b[1] + b[3] - 1}):
                            shared_lines += 1
                for a in info["annotations"]:
                    orientations.add(a["orientation"])
                    split_modes.add(a["split"])
                    depths.add((a["layers"], a["depth"]))
                    unequal_splits += a["split"] and a["side_lengths"][0] != a["side_lengths"][1]
                    owner = info["rectangles"][a["rectangle"]]
                    for side, length in enumerate(a["side_lengths"]):
                        for y, x in _ray(owner, a["orientation"], side, info["work"])[:length]:
                            for j, (r, c, h, w) in enumerate(info["rectangles"]):
                                if j != a["rectangle"] and ((y in (r, r + h - 1) and (x < c or x >= c + w)) or (x in (c, c + w - 1) and (y < r or y >= r + h))):
                                    crossings += 1
                    assert a["payload_count"] == a["edge_length"] * a["layers"]
                combined += has_boundary and len(info["rectangles"]) > len(info["annotations"]) and any(a["split"] for a in info["annotations"]) and len({a["orientation"] for a in info["annotations"]}) > 1
        stable_palettes += len(puzzle_roles) == 1
        mixed_palettes += len(puzzle_roles) > 1
        varied_shapes += len(puzzle_shapes) > 1
    assert len(counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(layouts) >= 80 and len(palettes) >= 20
    assert orientations == set(ORIENTATIONS) and split_modes == {False, True}
    assert {False, True} <= boundary and {0, 1} <= inert
    assert any(x < y for x, y in depths) and any(x == y for x, y in depths)
    assert train_objects == {1, 2, 3} and test_objects == {1, 2, 3, 4}
    assert flush_sides == set(ORIENTATIONS)
    assert any(1 in size for size in dimensions) and any(max(size) >= 6 for size in dimensions)
    assert shared_lines > 0 and crossings > 0 and unequal_splits > 0 and combined > 0
    assert stable_palettes > 0 and mixed_palettes > 0 and varied_shapes >= 40
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        seed = 1000 + (tr or 0) * 10 + (te or 0)
        puzzle = generate_puzzle(seed, tr, te)
        validate_generated_puzzle(puzzle)
        assert _equal(puzzle, generate_puzzle(seed, tr, te))
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(1, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
