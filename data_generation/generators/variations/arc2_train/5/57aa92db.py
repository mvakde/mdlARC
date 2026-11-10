import hashlib

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, num_train, num_test):
    if num_train is not None:
        if not _integer(num_train) or not 2 <= num_train <= 10:
            raise ValueError("num_train must be an integer from 2 to 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer(num_test) or not 1 <= num_test <= 4:
            raise ValueError("num_test must be an integer from 1 to 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _key(grid):
    return (tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes())


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for cell in ((row - 1, col), (row + 1, col),
                         (row, col - 1), (row, col + 1)):
                if cell in remaining:
                    remaining.remove(cell)
                    component.add(cell)
                    stack.append(cell)
        result.append(component)
    return result


def _connected(mask):
    return len(_components(mask)) == 1


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def _source_template(rng):
    # M is the shared registration cell; X is reference foreground.
    forms = (
        ("XXX", "XMX", "X.X"),
        ("...X", "MXXX", "...X"),
        ("..X", "XXM", "..X"),
        (".X.", "XXM", ".X."),
        (".XX", "XMX", "XX."),
    )
    form = forms[int(rng.integers(len(forms)))]
    code = np.asarray([[{"X": 1, "M": 2, ".": 0}[c] for c in row]
                       for row in form], dtype=np.int8)
    code = np.rot90(code, int(rng.integers(4)))
    if rng.integers(2):
        code = np.fliplr(code)
    return code.copy()


def _template(rng):
    if rng.random() < 0.28:
        return _source_template(rng)
    sizes = ((2, 3), (3, 2), (3, 3), (3, 4),
             (4, 3), (2, 4), (4, 2), (4, 4))
    for _ in range(300):
        height, width = sizes[int(rng.integers(len(sizes)))]
        density = float(rng.uniform(0.38, 0.88))
        occupied = rng.random((height, width)) < density
        if not (occupied[0].any() and occupied[-1].any()
                and occupied[:, 0].any() and occupied[:, -1].any()):
            continue
        if occupied.sum() < 4 or occupied.sum() == occupied.size:
            continue
        if not _connected(occupied):
            continue
        cells = np.argwhere(occupied)
        marker = cells[int(rng.integers(len(cells)))]
        code = occupied.astype(np.int8)
        code[tuple(marker)] = 2
        return code
    return _source_template(rng)


def _object_cells(code, top, left, scale):
    cells = set()
    for row, col in np.argwhere(code != 0):
        row, col = int(row), int(col)
        for dr in range(scale):
            for dc in range(scale):
                cells.add((top + row * scale + dr,
                           left + col * scale + dc))
    return cells


def _marker_cells(code, top, left, scale):
    row, col = map(int, np.argwhere(code == 2)[0])
    return {(top + row * scale + dr, left + col * scale + dc)
            for dr in range(scale) for dc in range(scale)}


def _boxes_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _touch(a, b):
    return any((r - 1, c) in b or (r + 1, c) in b
               or (r, c - 1) in b or (r, c + 1) in b for r, c in a)


def _layout(rng, code, scales):
    th, tw = code.shape
    largest_h = th * max(scales)
    largest_w = tw * max(scales)
    if largest_h + 2 > 30 or largest_w + 2 > 30:
        return None
    height = int(rng.integers(max(10, largest_h + 2), 31))
    width = int(rng.integers(max(10, largest_w + 2), 31))
    if (int(np.count_nonzero(code)) *
            (1 + sum(scale * scale for scale in scales))
            > height * width * 0.49):
        return None
    ref_top = int(rng.integers(height - th + 1))
    ref_left = int(rng.integers(width - tw + 1))
    reference = (ref_top, ref_left)
    ref_box = (ref_top - 1, ref_left - 1,
               ref_top + th + 1, ref_left + tw + 1)
    occupied = _object_cells(code, *reference, 1)
    markers = _marker_cells(code, *reference, 1)
    positions = []
    boxes = []
    targets = []
    for index, scale in enumerate(scales):
        positions_all = [(r, c)
                         for r in range(height - th * scale + 1)
                         for c in range(width - tw * scale + 1)]
        rng.shuffle(positions_all)
        free = []
        interlocking = []
        adjacent = []
        for top, left in positions_all:
            box = (top, left, top + th * scale, left + tw * scale)
            if _boxes_overlap(box, ref_box):
                continue
            cells = _object_cells(code, top, left, scale)
            if cells & occupied:
                continue
            marker_cells = _marker_cells(code, top, left, scale)
            if _touch(marker_cells, markers):
                continue
            candidate = (top, left, cells, marker_cells, box)
            free.append(candidate)
            if any(_boxes_overlap(box, other) for other in boxes):
                interlocking.append(candidate)
            if targets and _touch(cells, targets[-1]):
                adjacent.append(candidate)
        if not free:
            return None
        mode = int(rng.choice(3, p=[0.54, 0.25, 0.21]))
        pool = (interlocking if mode == 1 else
                adjacent if mode == 2 else free)
        if not pool:
            pool = free
        top, left, cells, marker_cells, box = pool[int(rng.integers(len(pool)))]
        positions.append((top, left))
        boxes.append(box)
        targets.append(cells)
        occupied |= cells
        markers |= marker_cells
    return (height, width), reference, positions


def _colors(rng, preferred_background, count):
    background = (preferred_background if rng.random() < 0.48
                  else int(rng.integers(10)))
    palette = [int(c) for c in rng.permutation(10) if c != background]
    marker = palette[0]
    reference = palette[1]
    payloads = palette[2:2 + count]
    return background, marker, reference, payloads


def _paint(grid, code, top, left, scale, marker, payload, foreground=None):
    for row, col in np.argwhere(code != 0):
        row, col = int(row), int(col)
        if code[row, col] == 1 and foreground is not None:
            if (row, col) not in foreground:
                continue
        color = marker if code[row, col] == 2 else payload
        grid[top + row * scale:top + (row + 1) * scale,
             left + col * scale:left + (col + 1) * scale] = color


def _make_pair(rng, regime, preferred_background):
    for _ in range(220):
        code = _template(rng)
        count_prob = ([0.55, 0.32, 0.13] if regime == "source_train"
                      else [0.20, 0.35, 0.45])
        count = int(rng.choice((1, 2, 3), p=count_prob))
        scale_prob = ([0.40, 0.35, 0.20, 0.05] if regime == "source_train"
                      else [0.18, 0.32, 0.32, 0.18])
        scales = [int(x) for x in rng.choice((1, 2, 3, 4),
                                             size=count, p=scale_prob)]
        layout = _layout(rng, code, scales)
        if layout is None:
            continue
        shape, reference_pos, positions = layout
        background, marker, ref_color, payloads = _colors(
            rng, preferred_background, count)
        source = np.full(shape, background, dtype=np.int8)
        answer = source.copy()
        _paint(source, code, *reference_pos, 1, marker, ref_color)
        _paint(answer, code, *reference_pos, 1, marker, ref_color)
        fg = [tuple(map(int, p)) for p in np.argwhere(code == 1)]
        for (top, left), scale, payload in zip(positions, scales, payloads):
            _paint(answer, code, top, left, scale, marker, payload)
            supplied_count = int(rng.integers(1, len(fg)))
            chosen = set(rng.choice(len(fg), size=supplied_count,
                                    replace=False).tolist())
            supplied = {fg[i] for i in chosen}
            _paint(source, code, top, left, scale, marker, payload, supplied)
        if np.array_equal(source, answer):
            continue
        try:
            models = _infer_models(source)
        except (AssertionError, ValueError):
            continue
        if len(models) == 1 and np.array_equal(models[0]["output"], answer):
            return {"input": source, "output": answer}
    raise RuntimeError("unable to build an unambiguous pair")


def _marker_components(grid, background, color):
    mask = grid == color
    result = []
    for cells in _components(mask):
        top, left, bottom, right = _bbox(cells)
        side = bottom - top
        if side != right - left or side > 4 or len(cells) != side * side:
            return []
        result.append((top, left, side))
    return result if len(result) >= 2 else []


def _candidate_from_reference(grid, background, marker, markers, reference_index):
    height, width = grid.shape
    ref_top, ref_left, _ = markers[reference_index]
    component = next((c for c in _components(grid != background)
                      if (ref_top, ref_left) in c), None)
    if component is None:
        return None
    top, left, bottom, right = _bbox(component)
    th, tw = bottom - top, right - left
    if not (2 <= th <= 4 and 2 <= tw <= 4):
        return None
    code = np.zeros((th, tw), dtype=np.int8)
    colors = set()
    for row, col in component:
        color = int(grid[row, col])
        code[row - top, col - left] = 2 if color == marker else 1
        if color != marker:
            colors.add(color)
    if len(colors) != 1 or int(np.count_nonzero(code == 2)) != 1:
        return None
    if int(np.count_nonzero(code == 1)) < 3:
        return None
    if not (np.any(code[0]) and np.any(code[-1])
            and np.any(code[:, 0]) and np.any(code[:, -1])):
        return None
    ref_color = colors.pop()
    expected_input = np.full(grid.shape, background, dtype=np.int8)
    expected_output = expected_input.copy()
    _paint(expected_input, code, top, left, 1, marker, ref_color)
    _paint(expected_output, code, top, left, 1, marker, ref_color)
    marker_row, marker_col = map(int, np.argwhere(code == 2)[0])
    targets = []
    occupied = _object_cells(code, top, left, 1)
    payloads = set()
    for index, (mr, mc, scale) in enumerate(markers):
        if index == reference_index:
            continue
        target_top = mr - marker_row * scale
        target_left = mc - marker_col * scale
        if (target_top < 0 or target_left < 0
                or target_top + th * scale > height
                or target_left + tw * scale > width):
            return None
        cells = _object_cells(code, target_top, target_left, scale)
        if cells & occupied:
            return None
        occupied |= cells
        supplied = []
        missing = 0
        for row, col in np.argwhere(code == 1):
            row, col = int(row), int(col)
            block = grid[target_top + row * scale:target_top + (row + 1) * scale,
                         target_left + col * scale:target_left + (col + 1) * scale]
            values = np.unique(block)
            if len(values) != 1:
                return None
            color = int(values[0])
            if color == background:
                missing += 1
            elif color in (marker, ref_color):
                return None
            else:
                supplied.append((row, col, color))
        if not supplied or not missing:
            return None
        payload = supplied[0][2]
        if any(item[2] != payload for item in supplied) or payload in payloads:
            return None
        payloads.add(payload)
        _paint(expected_input, code, target_top, target_left, scale,
               marker, payload, {(r, c) for r, c, _ in supplied})
        _paint(expected_output, code, target_top, target_left, scale,
               marker, payload)
        targets.append({"scale": scale, "top": target_top,
                        "left": target_left, "payload": payload,
                        "supplied": len(supplied), "missing": missing,
                        "box": (target_top, target_left,
                                target_top + th * scale,
                                target_left + tw * scale)})
    if not np.array_equal(expected_input, grid):
        return None
    return {"output": expected_output, "background": background,
            "marker": marker, "reference_color": ref_color,
            "template": code, "reference_box": (top, left, bottom, right),
            "targets": targets}


def _infer_models(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= grid.size // 2:
        return []
    models = []
    for marker in np.unique(grid):
        marker = int(marker)
        if marker == background:
            continue
        components = _marker_components(grid, background, marker)
        for i, (_, _, scale) in enumerate(components):
            if scale == 1:
                model = _candidate_from_reference(
                    grid, background, marker, components, i)
                if model is not None:
                    models.append(model)
    return models


def _source_keys():
    # Digests of complete input/output grids, including their shape.
    return {
        "51312fbe254ff3d5c1581e02208dd2d630c9362892ddeb54f5b1495aae8a3998",
        "adf57664ecd9aa769388537a13c8bbc1d6fbe45f166ab534f448cbda1c4f5ba3",
        "5f5f40eb73f14169149d4a71be91681dc93ba8ac7087284f381e0e0ec2fb68fe",
        "f26ad345ce147241c3992cf7adc0f394ef94ebe26eceae5fa1bf861050269234",
        "004c5c88a6d6c41f417af5c2e911a37a3a63e5b91b53359e4ca3bfd15ce3a3d8",
    }


def _pair_digest(source, answer):
    content = (source.shape[0], source.shape[1],
               source.tolist(), answer.tolist())
    return hashlib.sha256(repr(content).encode()).hexdigest()


def _assert_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4
    assert train_count > test_count
    inputs = set()
    pairs = set()
    source_keys = _source_keys()
    train_models = []
    test_models = []
    for section, models in (("train", train_models), ("test", test_models)):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, answer = pair["input"], pair["output"]
            _assert_grid(source)
            _assert_grid(answer)
            assert source.shape == answer.shape
            ikey, okey = _key(source), _key(answer)
            assert ikey not in inputs and (ikey, okey) not in pairs
            inputs.add(ikey)
            pairs.add((ikey, okey))
            assert _pair_digest(source, answer) not in source_keys
            assert not np.array_equal(source, answer)
            found = _infer_models(source)
            assert len(found) == 1, "input must have one stencil interpretation"
            model = found[0]
            assert np.array_equal(model["output"], answer)
            assert len(model["targets"]) in (1, 2, 3)
            assert all(1 <= t["scale"] <= 4 and t["supplied"] >= 1
                       and t["missing"] >= 1 for t in model["targets"])
            models.append(model)
    assert any(any(t["scale"] > 1 for t in m["targets"])
               for m in train_models)
    # The registration color and the unrotated square scaling are demonstrated
    # in every training pair; no literal palette role is a hidden convention.


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, num_train, num_test)
    preferred_background = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    used = set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(50):
                pair = _make_pair(rng, "source_train" if section == "train"
                                  else "target_test", preferred_background)
                key = _key(pair["input"])
                if key in used:
                    continue
                if section == "train" and index == train_count - 1:
                    prior = puzzle["train"] + [pair]
                    if not any(any(t["scale"] > 1 for t in _infer_models(
                            p["input"])[0]["targets"]) for p in prior):
                        continue
                puzzle[section].append(pair)
                used.add(key)
                break
            else:
                raise RuntimeError("unable to sample a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same(first, second):
    return all(len(first[s]) == len(second[s])
               and all(np.array_equal(a[k], b[k])
                       for a, b in zip(first[s], second[s])
                       for k in ("input", "output"))
               for s in ("train", "test"))


def _copy_puzzle(puzzle):
    return {
        section: [{name: grid.copy() for name, grid in pair.items()}
                  for pair in puzzle[section]]
        for section in ("train", "test")
    }


def _must_reject(puzzle):
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return
    raise AssertionError("invalid stencil pair was accepted")


def self_test() -> None:
    one = generate_puzzle(1234)
    assert _same(one, generate_puzzle(1234))
    assert not _same(one, generate_puzzle(1235))
    model = _infer_models(one["train"][0]["input"])[0]
    damaged = _copy_puzzle(one)
    target = model["targets"][0]
    damaged_grid = damaged["train"][0]["input"]
    damaged_grid[damaged_grid == target["payload"]] = model["background"]
    _must_reject(damaged)
    damaged = _copy_puzzle(one)
    changed = np.argwhere(one["train"][0]["input"] != one["train"][0]["output"])
    row, col = map(int, changed[0])
    damaged["train"][0]["output"][row, col] = model["background"]
    _must_reject(damaged)
    for seed, train, test in (
        (2000, 2, 1), (2001, 10, 1), (2002, 10, 4),
        (3000, None, 1), (3001, None, 4),
        (3002, 2, None), (3003, 10, None),
    ):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0),
                        (None, 5), (2, 2), (3, 4), (4, 4),
                        (True, 1)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts accepted")
    counts = set()
    train_counts = set()
    test_counts = set()
    dimensions = set()
    shapes = set()
    scales_train = set()
    scales_test = set()
    markers = set()
    backgrounds = set()
    supplied = set()
    target_counts = set()
    overlap = 0
    adjacency = 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                model = _infer_models(pair["input"])[0]
                dimensions.add(pair["input"].shape)
                shapes.add(_key(model["template"]))
                markers.add(model["marker"])
                backgrounds.add(model["background"])
                target_counts.add(len(model["targets"]))
                for target in model["targets"]:
                    supplied.add((target["supplied"], target["missing"]))
                    (scales_train if section == "train" else scales_test).add(
                        target["scale"])
                boxes = [t["box"] for t in model["targets"]]
                overlap += any(_boxes_overlap(a, b)
                               for i, a in enumerate(boxes)
                               for b in boxes[i + 1:])
                adjacency += any(_touch(
                    _object_cells(model["template"], a["top"], a["left"], a["scale"]),
                    _object_cells(model["template"], b["top"], b["left"], b["scale"]))
                    for i, a in enumerate(model["targets"])
                    for b in model["targets"][i + 1:])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(counts) >= 8 and len(dimensions) >= 20
    assert len(shapes) >= 30 and len(supplied) >= 8
    assert scales_train == {1, 2, 3, 4} and scales_test == {1, 2, 3, 4}
    assert target_counts == {1, 2, 3}
    assert len(markers) >= 5 and len(backgrounds) >= 5
    assert overlap > 0 and adjacency > 0
