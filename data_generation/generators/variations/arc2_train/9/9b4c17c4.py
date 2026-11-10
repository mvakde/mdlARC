"""Generate two-carrier ARC puzzles with carrier-relative horizontal motion."""
import hashlib
import numpy as np

_ROLES = ("left", "right")
_SOURCE_PAIRS = {
    ((12, 11), "4a509c78b036aeee0de073bbe58fffae1a12d331321eef4527e6e1e117d06dc2"),
    ((11, 13), "2c8aed6070d411aba60d2de96a80174c2e9df137210acc9d48902f89d03f8355"),
    ((12, 14), "cef189f13e45b41ac6b16b15d9df53511e3e384c2c52d057bdbc7e57b2a706b0"),
    ((15, 15), "8696f2cc720d90ac55fefa50dbcc96248fc61329b9e5303449f142be91c1ddb6"),
    ((9, 9), "c27c3483fe6098a4bc1be42baf0e44b51f1a959bc923b4d15f1ba7337c280ec4"),
    ((10, 11), "c844ff3f203f8b25c3ecae57e469622516a44bcff7e99b12fb786930d5d6f1a5"),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _key(grid):
    return tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes()


def _source_key(inp, out):
    return tuple(inp.shape), hashlib.sha256(
        inp.astype(np.int8, copy=False).tobytes()
        + out.astype(np.int8, copy=False).tobytes()).hexdigest()


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            result.append(cells)
    return result


def _choose_x(rng, width, block_width, direction, previous=None):
    positions = list(range(width - block_width + 1))
    if previous is not None:
        px, pw = previous
        positions = [x for x in positions if x + block_width < px or px + pw < x]
    if not positions:
        return None
    distances = {x: x if direction == "left" else width - x - block_width for x in positions}
    mode = int(rng.choice(4, p=[0.25, 0.28, 0.32, 0.15]))
    if mode == 2:
        return int(rng.choice(positions))
    target = 1 if mode == 0 else max(distances.values()) if mode == 1 else 0
    best = min(abs(d - target) for d in distances.values())
    return int(rng.choice([x for x in positions if abs(distances[x] - target) == best]))


def _make_field(rng, height, width, carrier, payload, direction, count,
                width_cap, forced_width=None):
    for _ in range(160):
        widths = [int(rng.integers(1, min(width_cap, width - 1) + 1)) for _ in range(count)]
        if forced_width is not None:
            if forced_width >= width:
                continue
            widths[int(rng.integers(count))] = forced_width
        xs, gaps = [], []
        for i, block_width in enumerate(widths):
            adjacent = i > 0 and bool(rng.random() < 0.48)
            previous = (xs[-1], widths[i-1]) if adjacent else None
            x = _choose_x(rng, width, block_width, direction, previous)
            if x is None:
                adjacent = False
                x = _choose_x(rng, width, block_width, direction)
            xs.append(x)
            if i:
                gaps.append(0 if adjacent else 1)
        if count + sum(gaps) > height:
            continue
        heights = [1] * count
        free = height - count - sum(gaps)
        for i in rng.permutation(count):
            if free == 0:
                break
            addition = int(rng.integers(0, min(3, free) + 1))
            heights[int(i)] += addition
            free -= addition
        extras = rng.multinomial(free, np.full(count+1, 1/(count+1)))
        starts = []
        cursor = int(extras[0])
        for i in range(count):
            starts.append(cursor)
            cursor += heights[i]
            if i < count-1:
                cursor += gaps[i] + int(extras[i+1])
        distances = [x if direction == "left" else width-x-widths[i] for i, x in enumerate(xs)]
        if not any(distances):
            continue
        inp = np.full((height, width), carrier, dtype=np.int8)
        out = np.full((height, width), carrier, dtype=np.int8)
        for y, h, x, w in zip(starts, heights, xs, widths):
            inp[y:y+h, x:x+w] = payload
            target = 0 if direction == "left" else width-w
            out[y:y+h, target:target+w] = payload
        if len(_components(inp == payload)) == count:
            return inp, out
    raise RuntimeError("could not place distinct rectangles")


def _sample_count(rng, field_height, regime):
    maximum = min(5 if regime == "target_test" else 4, (field_height+1)//2)
    choices = np.arange(1, maximum+1)
    weights = np.array([1.0, 2.3, 2.0, 1.3, 0.7][:maximum])
    return int(rng.choice(choices, p=weights/weights.sum()))


def _generate_pair(rng, regime, mapping, orientation, first_role, width_cap,
                   forced_width=None):
    if regime == "source_train":
        height, width = int(rng.integers(9, 22)), int(rng.integers(10, 23))
    else:
        height, width = int(rng.integers(8, 19)), int(rng.integers(9, 21))
    roles = (first_role, "right" if first_role == "left" else "left")
    if orientation == "horizontal":
        split = int(rng.integers(3, height-2))
        shapes = ((split, width), (height-split, width))
    else:
        split = int(rng.integers(3, width-2))
        shapes = ((height, split), (height, width-split))
    counts = [_sample_count(rng, shape[0], regime) for shape in shapes]
    wide_field = None
    if forced_width is not None:
        capable = [i for i, (_, w) in enumerate(shapes) if w > forced_width]
        if not capable:
            return None
        wide_field = int(rng.choice(capable))
    fields = []
    for i, (role, shape, count) in enumerate(zip(roles, shapes, counts)):
        carrier = mapping[0 if role == "left" else 1]
        fields.append(_make_field(rng, shape[0], shape[1], carrier, mapping[2],
                                  role, count, width_cap,
                                  forced_width if i == wide_field else None))
    if orientation == "horizontal":
        return np.vstack([f[0] for f in fields]), np.vstack([f[1] for f in fields])
    return np.hstack([f[0] for f in fields]), np.hstack([f[1] for f in fields])


def _transform_region(region, carrier, payload, direction):
    if not set(int(v) for v in np.unique(region)).issubset({carrier, payload}):
        return None
    components = _components(region == payload)
    if not components:
        return None
    expected = np.full(region.shape, carrier, dtype=np.int8)
    occupied = np.zeros(region.shape, dtype=bool)
    objects = []
    for component in components:
        ys, xs = zip(*component)
        y0, y1, x0, x1 = min(ys), max(ys)+1, min(xs), max(xs)+1
        h, w = y1-y0, x1-x0
        if len(component) != h*w or not np.all(region[y0:y1, x0:x1] == payload):
            return None
        target = 0 if direction == "left" else region.shape[1]-w
        if np.any(occupied[y0:y1, target:target+w]):
            return None
        occupied[y0:y1, target:target+w] = True
        expected[y0:y1, target:target+w] = payload
        objects.append((y0, x0, h, w, abs(target-x0)))
    return expected, tuple(objects)


def _input_interpretations(inp, mapping):
    left, right, payload = mapping
    if set(int(v) for v in np.unique(inp)) != {left, right, payload}:
        return []
    height, width = inp.shape
    found = []
    for orientation in ("horizontal", "vertical"):
        extent = height if orientation == "horizontal" else width
        for split in range(1, extent):
            if orientation == "horizontal":
                slices = ((slice(0, split), slice(None)), (slice(split, height), slice(None)))
            else:
                slices = ((slice(None), slice(0, split)), (slice(None), slice(split, width)))
            for first_role in _ROLES:
                roles = (first_role, "right" if first_role == "left" else "left")
                results = []
                for slc, role in zip(slices, roles):
                    carrier = left if role == "left" else right
                    result = _transform_region(inp[slc], carrier, payload, role)
                    if result is None:
                        break
                    results.append(result)
                if len(results) != 2:
                    continue
                expected = np.empty(inp.shape, dtype=np.int8)
                objects = []
                for i, (slc, role, result) in enumerate(zip(slices, roles, results)):
                    expected[slc] = result[0]
                    oy = split if orientation == "horizontal" and i else 0
                    ox = split if orientation == "vertical" and i else 0
                    objects.extend((role, oy+y, ox+x, h, w, travel)
                                   for y, x, h, w, travel in result[1])
                found.append(dict(orientation=orientation, split=split,
                                  first_role=first_role, objects=tuple(objects),
                                  expected=expected))
    return found


def _infer_mappings(train, palette):
    colors = sorted(palette)
    valid = []
    for payload in colors:
        carriers = [c for c in colors if c != payload]
        for left, right in (carriers, carriers[::-1]):
            mapping = (left, right, payload)
            if all(any(np.array_equal(i["expected"], p["output"])
                       for i in _input_interpretations(p["input"], mapping))
                   for p in train):
                valid.append(mapping)
    return valid


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    mapping = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
    extension = bool(rng.random() < 0.38)
    train_cap = 3 if extension else int(rng.choice([3, 4], p=[0.28, 0.72]))
    test_cap = 4 if extension else int(rng.choice([3, 4, 5], p=[0.34, 0.57, 0.09]))
    extension_index = int(rng.integers(test_count)) if extension else None
    layout_mode = int(rng.choice(3, p=[0.22, 0.22, 0.56]))
    anchor_orientation = "horizontal" if rng.random() < 0.5 else "vertical"
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count, regime, cap in (("train", train_count, "source_train", train_cap),
                                        ("test", test_count, "target_test", test_cap)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(100):
                if layout_mode == 0:
                    orientation = anchor_orientation
                elif layout_mode == 1:
                    orientation = (anchor_orientation if section == "train" else
                                   ("vertical" if anchor_orientation == "horizontal" else "horizontal"))
                else:
                    orientation = "horizontal" if pair_rng.random() < 0.5 else "vertical"
                first_role = _ROLES[int(pair_rng.integers(2))]
                force = 4 if section == "test" and index == extension_index else None
                pair = _generate_pair(pair_rng, regime, mapping, orientation, first_role, cap, force)
                if pair is None:
                    continue
                inp, out = pair
                if _key(inp) in seen or _source_key(inp, out) in _SOURCE_PAIRS:
                    continue
                interpretations = _input_interpretations(inp, mapping)
                if len(interpretations) != 1 or not np.array_equal(interpretations[0]["expected"], out):
                    continue
                break
            else:
                raise RuntimeError("could not construct a distinct, recoverable pair")
            seen.add(_key(inp))
            puzzle[section].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "sections must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "invalid counts")
    seen_inputs, seen_pairs, palette = set(), set(), None
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be 2D array")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "invalid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer) and bool(np.all((grid >= 0) & (grid <= 9))), "invalid colors")
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "invalid transformation")
            colors = frozenset(int(c) for c in np.unique(inp))
            _require(len(colors) == 3 and colors == frozenset(int(c) for c in np.unique(out)), "invalid palette")
            if palette is None:
                palette = colors
            _require(colors == palette, "puzzle palette changed")
            ik, pk = _key(inp), (_key(inp), _key(out))
            _require(ik not in seen_inputs and pk not in seen_pairs, "duplicate pair")
            _require(_source_key(inp, out) not in _SOURCE_PAIRS, "source pair copy")
            seen_inputs.add(ik)
            seen_pairs.add(pk)
    mappings = _infer_mappings(train, palette)
    _require(len(mappings) == 1, "training does not reveal unique color directions")
    mapping = mappings[0]
    for section in ("train", "test"):
        for pair in puzzle[section]:
            interpretations = _input_interpretations(pair["input"], mapping)
            _require(len(interpretations) == 1, "ambiguous carrier boundary or objects")
            item = interpretations[0]
            _require(np.array_equal(item["expected"], pair["output"]), "incorrect slide output")
            _require(all(h <= 4 and w <= (4 if section == "train" else 5)
                         for _, _, _, h, w, _ in item["objects"]), "unsupported rectangle size")
            _require(all(any(role == r for r, *_ in item["objects"]) for role in _ROLES), "both carriers need objects")
            _require(any(travel > 0 for *_, travel in item["objects"]), "no motion")


def _puzzle_key(puzzle):
    return tuple((section, tuple((_key(p["input"]), _key(p["output"]))
                                 for p in puzzle[section])) for section in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    shapes, layouts, palettes, orientations, orders = set(), set(), set(), set(), set()
    widths, heights, totals, travels = set(), set(), set(), set()
    adjacent = flush = sparse = dense = 0
    consistent = transfer = mixed = wide_later = 0
    overlap = narrow = asymmetric = adjacent_fixed = mixed_travel = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        palette = frozenset(int(x) for x in np.unique(puzzle["train"][0]["input"]))
        palettes.add(palette)
        mapping = _infer_mappings(puzzle["train"], palette)[0]
        orientations_by_section = {}
        for section in ("train", "test"):
            section_orientations = []
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                item = _input_interpretations(inp, mapping)[0]
                section_orientations.append(item["orientation"])
                _require(np.array_equal(item["expected"], out), "self-test prediction failed")
                shapes.add(inp.shape)
                layouts.add((section, inp.shape, item["orientation"], item["split"],
                             item["first_role"], item["objects"]))
                orientations.add((section, item["orientation"]))
                orders.add((item["orientation"], item["first_role"]))
                totals.add(len(item["objects"]))
                mixed_travel += (any(o[5] == 1 for o in item["objects"])
                                 and any(o[5] >= 5 for o in item["objects"]))
                if item["orientation"] == "vertical":
                    smaller = min(item["split"], inp.shape[1] - item["split"])
                    narrow += smaller <= 4
                    asymmetric += max(item["split"], inp.shape[1] - item["split"]) >= 2 * smaller
                pair_adjacent = False
                for role in _ROLES:
                    objects = sorted((y, x, h, w, t) for r, y, x, h, w, t in item["objects"] if r == role)
                    sparse += len(objects) == 1
                    dense += len(objects) >= 4
                    flush += sum(t == 0 for *_, t in objects)
                    touching_rows = sum(a[0]+a[2] == b[0] for a, b in zip(objects, objects[1:]))
                    adjacent += touching_rows
                    pair_adjacent |= bool(touching_rows)
                adjacent_fixed += pair_adjacent and any(o[5] == 0 for o in item["objects"])
                widths.update(o[4] for o in item["objects"])
                heights.update(o[3] for o in item["objects"])
                travels.update(o[5] for o in item["objects"])
            orientations_by_section[section] = section_orientations
        train_orientations = set(orientations_by_section["train"])
        test_orientations = set(orientations_by_section["test"])
        consistent += len(train_orientations) == 1 and train_orientations == test_orientations
        transfer += bool(test_orientations - train_orientations)
        mixed += len(train_orientations) == 2
        wide_later += any(4 in [o[4] for o in _input_interpretations(pair["input"], mapping)[0]["objects"]]
                          for pair in puzzle["test"][1:])
        overlap += any(t["input"].shape[0] >= min(q["input"].shape[0] for q in puzzle["train"])
                       or t["input"].shape[1] >= min(q["input"].shape[1] for q in puzzle["train"])
                       for t in puzzle["test"])
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len(puzzle_keys) == 42 and len(shapes) >= 35 and len(layouts) >= 150, "layout diversity")
    _require(len(palettes) >= 15 and len(orders) == 4 and len(orientations) == 4, "palette/orientation diversity")
    _require(widths.issuperset({1, 2, 3, 4}) and heights.issuperset({1, 2, 3, 4}), "rectangle diversity")
    _require(adjacent >= 10 and flush >= 10 and sparse >= 10 and dense >= 5, "occupancy diversity")
    _require(1 in travels and max(travels) >= 5 and len(totals) >= 5, "motion/count diversity")
    _require(consistent >= 4 and transfer >= 4 and mixed >= 8 and wide_later >= 5,
             "orientation and width-extension diversity")
    _require(overlap >= 15 and narrow >= 20 and asymmetric >= 20,
             "canvas and field-proportion diversity")
    _require(adjacent_fixed >= 10 and mixed_travel >= 20,
             "combined object and travel variation")
    source_palette_puzzle = generate_puzzle(91)
    _require(frozenset(int(c) for c in np.unique(source_palette_puzzle["train"][0]["input"]))
             == {1, 2, 8}, "original palette was improperly excluded")
    a = generate_puzzle(777, 7, 3)
    _require(_puzzle_key(a) == _puzzle_key(generate_puzzle(777, 7, 3)), "nondeterministic")
    _require(_puzzle_key(a) != _puzzle_key(generate_puzzle(778, 7, 3)), "seed collision")
    for seed, nt, nv in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (1101, None, 1), (1102, None, 4), (1103, 2, None), (1104, 10, None)):
        p = generate_puzzle(seed, nt, nv)
        _require((nt is None or len(p["train"]) == nt) and
                 (nv is None or len(p["test"]) == nv), "count request ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 4, "num_test": 4}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
