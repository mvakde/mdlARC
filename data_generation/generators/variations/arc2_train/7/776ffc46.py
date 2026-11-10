import hashlib

import numpy as np


_CROSS = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.int8)
_ASYMMETRIC = (
    ((1, 0, 0), (1, 1, 1), (0, 1, 0)),
    ((1, 1, 0), (0, 1, 0), (0, 1, 1)),
    ((1, 0, 0, 0), (1, 1, 1, 0), (0, 0, 1, 1)),
    ((0, 1, 1), (1, 1, 0), (0, 1, 0)),
    ((1, 1, 1), (1, 0, 0), (1, 1, 0)),
    ((0, 1, 0, 0), (1, 1, 1, 1), (0, 0, 1, 0)),
)
_SOURCE_PAIR_DIGESTS = frozenset((
    "6696a1e648283b8660db21f41ea4eca3b4caf07cfcb5ad37e734e543bd083c52",
    "5aaa197c690756089833fb2d66d1e556de13e350314b9b24f014f69a33cb81a8",
    "6620f8372c68c83d92bdb5811a67f690bbe5cb5672e6e7fc24ececc86fa4329e",
    "6a83095e80efbcf0de561eb7dc630bf8bbed33d69a586edf2fed6db9a43071b5",
    "68c64cc2a7217a4e6c1d736c1b659cdd220b6fbec392dfe14a8dc5906fd4f96f",
))


def _trim(mask):
    cells = np.argwhere(mask)
    assert len(cells)
    low = cells.min(axis=0)
    high = cells.max(axis=0)
    return (mask[low[0]:high[0] + 1, low[1]:high[1] + 1] != 0).astype(np.int8)


def _key(mask):
    mask = _trim(mask)
    return (mask.shape, mask.tobytes())


def _bbox(cells):
    a = np.asarray(cells)
    low, high = a.min(axis=0), a.max(axis=0)
    return int(low[0]), int(low[1]), int(high[0]), int(high[1])


def _cell_mask(cells):
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.int8)
    for r, c in cells:
        mask[r - r0, c - c0] = 1
    return mask


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    answer = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            seen[r, c] = True
            todo = [(r, c)]
            cells = []
            while todo:
                x, y = todo.pop()
                cells.append((x, y))
                for u, v in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= u < h and 0 <= v < w and not seen[u, v] and grid[u, v] == color:
                        seen[u, v] = True
                        todo.append((u, v))
            answer.append(tuple(cells))
    return answer


def _connected(mask):
    mask = _trim(mask)
    return len(_components(mask, 1)) == 1


def _perimeter(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}


def _frame_kind(cells):
    box = _bbox(cells)
    r0, c0, r1, c1 = box
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None
    actual, edge = set(cells), _perimeter(box)
    if actual == edge:
        return "complete"
    if _fragment_style(cells) is not None:
        return "fragment"
    return None


def _fragment_style(cells):
    r0, c0, r1, c1 = _bbox(cells)
    actual = set(cells)
    edge = _perimeter((r0, c0, r1, c1))
    if not actual < edge:
        return None
    for row in (r0, r1):
        for col in (c0, c1):
            corner = {(row, c) for c in range(c0, c1 + 1)} | {
                (r, col) for r in range(r0, r1 + 1)}
            if actual == corner:
                return "corner"
    missing = edge - actual
    if len(missing) == 1:
        return "gap"
    sides = ({(r0, c) for c in range(c0 + 1, c1)},
             {(r1, c) for c in range(c0 + 1, c1)},
             {(r, c0) for r in range(r0 + 1, r1)},
             {(r, c1) for r in range(r0 + 1, r1)})
    if any(missing == side for side in sides):
        return "three"
    return None


def _background(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    return int(np.argmax(counts))


def _inside_color(grid, box, color):
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)
            if grid[r, c] == color]


def _margin(cells, box):
    assert cells
    r0, c0, r1, c1 = box
    a, b, d, e = _bbox(cells)
    assert a >= r0 + 2 and b >= c0 + 2 and d <= r1 - 2 and e <= c1 - 2


def _analyze_pair(inp, out):
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == out.ndim == 2 and inp.shape == out.shape
    assert np.issubdtype(inp.dtype, np.integer) and np.issubdtype(out.dtype, np.integer)
    assert all(1 <= n <= 30 for n in inp.shape)
    assert 0 <= inp.min() <= inp.max() <= 9 and 0 <= out.min() <= out.max() <= 9
    bg = _background(inp)
    changed = inp != out
    assert changed.any()
    old, new = np.unique(inp[changed]), np.unique(out[changed])
    assert len(old) == len(new) == 1
    candidate, accent = int(old[0]), int(new[0])
    colors = {int(x) for x in np.unique(inp)}
    assert len(colors) == 4 and colors == {int(x) for x in np.unique(out)}
    rest = colors - {bg, candidate, accent}
    assert len(rest) == 1
    frame = rest.pop()
    frames = _components(inp, frame)
    kinds = [_frame_kind(c) for c in frames]
    assert kinds.count("complete") == 1 and all(k is not None for k in kinds)
    selected = frames[kinds.index("complete")]
    selected_box = _bbox(selected)
    assert set(np.unique(inp[selected_box[0] + 1:selected_box[2],
                                  selected_box[1] + 1:selected_box[3]])) <= {bg, accent}
    proto_cells = _inside_color(inp, selected_box, accent)
    _margin(proto_cells, selected_box)
    proto_mask = _cell_mask(proto_cells)
    assert _connected(proto_mask)
    proto_key = _key(proto_mask)
    boxes = [_bbox(c) for c in frames]
    accent_covered = set(proto_cells)
    decoy_keys = []
    fragment_kinds = []
    fragment_styles = []
    for comp, kind in zip(frames, kinds):
        if kind != "fragment":
            continue
        box = _bbox(comp)
        assert set(np.unique(inp[box[0] + 1:box[2], box[1] + 1:box[3]])) <= {bg, accent}
        cells = _inside_color(inp, box, accent)
        _margin(cells, box)
        mask = _cell_mask(cells)
        assert _connected(mask) and _key(mask) != proto_key
        accent_covered.update(cells)
        decoy_keys.append(_key(mask))
        fragment_kinds.append(len(_perimeter(box) - set(comp)))
        fragment_styles.append(_fragment_style(comp))
    assert accent_covered == {tuple(x) for x in np.argwhere(inp == accent)}
    expected = inp.copy()
    matches = 0
    nonmatches = []
    candidate_components = _components(inp, candidate)
    assert candidate_components
    for comp in candidate_components:
        assert all(not (a <= r <= d and b <= c <= e)
                   for r, c in comp for a, b, d, e in boxes)
        key = _key(_cell_mask(comp))
        if key == proto_key:
            matches += 1
            for r, c in comp:
                expected[r, c] = accent
        else:
            nonmatches.append(key)
    assert matches >= 1 and nonmatches
    assert np.array_equal(expected, out)
    return {"shape": inp.shape, "roles": (bg, frame, candidate, accent),
            "prototype": proto_key, "prototype_area": int(proto_mask.sum()),
            "prototype_mask": proto_mask, "frame_box": selected_box,
            "decoys": len(decoy_keys), "decoy_keys": decoy_keys,
            "fragment_missing": fragment_kinds, "fragment_styles": fragment_styles,
            "candidates": len(candidate_components), "matches": matches,
            "nonmatches": nonmatches,
            "max_candidate_area": max(len(c) for c in candidate_components)}


def _sample_shape(rng, category):
    if category == "block":
        h, w = ((2, 3), (3, 3), (2, 4), (3, 4), (4, 3), (2, 5), (4, 4))[
            int(rng.integers(0, 7))]
        return np.ones((h, w), dtype=np.int8)
    if category == "bar":
        length = int(rng.integers(3, 7))
        return np.ones((1, length) if rng.integers(0, 2) else (length, 1), dtype=np.int8)
    if category == "cross":
        return _CROSS.copy()
    if rng.integers(0, 3) == 0:
        size = int(rng.integers(5, 11))
        cells = {(0, 0)}
        for _ in range(size - 1):
            options = {(r + dr, c + dc) for r, c in cells
                       for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                       if (r + dr, c + dc) not in cells}
            options = sorted(p for p in options if
                             max(x for x, y in cells | {p}) - min(x for x, y in cells | {p}) < 5 and
                             max(y for x, y in cells | {p}) - min(y for x, y in cells | {p}) < 5)
            cells.add(options[int(rng.integers(0, len(options)))])
        mask = _cell_mask(tuple(cells))
        if _key(mask) != _key(np.rot90(mask)) and _key(mask) != _key(np.fliplr(mask)):
            return mask
    return np.array(_ASYMMETRIC[int(rng.integers(0, len(_ASYMMETRIC)))], dtype=np.int8)


def _near_options(mask):
    seen = {_key(mask)}
    result = []
    variants = [np.rot90(mask, k) for k in (1, 2, 3)]
    variants += [np.fliplr(mask), np.flipud(mask)]
    for r, c in np.argwhere(mask):
        variant = mask.copy()
        variant[r, c] = 0
        if variant.any() and _connected(variant):
            variants.append(variant)
    padded = np.pad(mask, 1)
    for r, c in np.argwhere(padded == 0):
        if any(0 <= r + dr < padded.shape[0] and 0 <= c + dc < padded.shape[1]
               and padded[r + dr, c + dc] for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            variant = padded.copy()
            variant[r, c] = 1
            variants.append(variant)
    for variant in variants:
        variant = _trim(variant)
        key = _key(variant)
        if key not in seen and _connected(variant):
            seen.add(key)
            result.append(variant)
    return result


class _PlacementError(Exception):
    pass


def _separate(box, others):
    r0, c0, r1, c1 = box
    return all(r1 + 1 < a or d + 1 < r0 or c1 + 1 < b or e + 1 < c0
               for a, b, d, e in others)


def _place_box(rng, h, w, bh, bw, others):
    if bh > h or bw > w:
        raise _PlacementError
    positions = [(r, c) for r in range(h - bh + 1) for c in range(w - bw + 1)]
    for index in rng.permutation(len(positions)):
        r, c = positions[int(index)]
        box = (r, c, r + bh - 1, c + bw - 1)
        if _separate(box, others):
            others.append(box)
            return box
    raise _PlacementError


def _draw_mask(grid, mask, r, c, color):
    for x, y in np.argwhere(mask):
        grid[r + int(x), c + int(y)] = color


def _draw_frame(grid, box, color, style, rng):
    r0, c0, r1, c1 = box
    cells = _perimeter(box)
    if style == "corner":
        row = r0 if rng.integers(0, 2) else r1
        col = c0 if rng.integers(0, 2) else c1
        cells = {(row, c) for c in range(c0, c1 + 1)} | {(r, col) for r in range(r0, r1 + 1)}
    elif style == "three":
        side = int(rng.integers(0, 4))
        cells -= ({(r0, c) for c in range(c0 + 1, c1)} if side == 0 else
                  {(r1, c) for c in range(c0 + 1, c1)} if side == 1 else
                  {(r, c0) for r in range(r0 + 1, r1)} if side == 2 else
                  {(r, c1) for r in range(r0 + 1, r1)})
    elif style == "gap":
        side = int(rng.integers(0, 4))
        if side < 2:
            row = r0 if side == 0 else r1
            col = int(rng.integers(c0 + 1, c1))
            cells.remove((row, col))
        else:
            col = c0 if side == 2 else c1
            row = int(rng.integers(r0 + 1, r1))
            cells.remove((row, col))
    for r, c in cells:
        grid[r, c] = color


def _draw_inside(grid, box, mask, color, rng):
    r0, c0, r1, c1 = box
    h, w = mask.shape
    row = int(rng.integers(r0 + 2, r1 - h))
    col = int(rng.integers(c0 + 2, c1 - w))
    _draw_mask(grid, mask, row, col, color)


def _construct(rng, shape, roles, decoy_count, candidate_count, match_count,
               forced_rotation, size):
    bg, frame, candidate, accent = roles
    h, w = size
    grid = np.full((h, w), bg, dtype=np.int8)
    boxes = []
    fh = shape.shape[0] + 4 + int(rng.integers(0, 3))
    fw = shape.shape[1] + 4 + int(rng.integers(0, 3))
    selector = _place_box(rng, h, w, fh, fw, boxes)
    _draw_frame(grid, selector, frame, "complete", rng)
    _draw_inside(grid, selector, shape, accent, rng)
    decoys = []
    for _ in range(decoy_count):
        for _ in range(20):
            decoy = _sample_shape(rng, ("block", "asym", "cross", "bar")[int(rng.integers(0, 4))])
            if _key(decoy) != _key(shape) and max(decoy.shape) <= 4:
                break
        else:
            raise _PlacementError
        dh, dw = decoy.shape
        box = _place_box(rng, h, w, dh + 4 + int(rng.integers(0, 2)),
                         dw + 4 + int(rng.integers(0, 2)), boxes)
        style = ("corner", "three", "gap")[int(rng.integers(0, 3))]
        _draw_frame(grid, box, frame, style, rng)
        _draw_inside(grid, box, decoy, accent, rng)
        decoys.append(decoy)
    options = _near_options(shape)
    masks = [shape.copy() for _ in range(match_count)]
    if forced_rotation:
        rotated = _trim(np.rot90(shape))
        if _key(rotated) == _key(shape):
            rotated = _trim(np.fliplr(shape))
        assert _key(rotated) != _key(shape)
        masks.append(rotated)
    for index, decoy in enumerate(decoys):
        if len(masks) < candidate_count and (forced_rotation and index == 0 or rng.integers(0, 3) != 0):
            masks.append(decoy.copy())
    while len(masks) < candidate_count:
        if rng.integers(0, 5) == 0:
            extra = _sample_shape(rng, "block")
            if _key(extra) != _key(shape):
                masks.append(extra)
                continue
        masks.append(options[int(rng.integers(0, len(options)))].copy())
    masks = [masks[int(i)] for i in rng.permutation(len(masks))]
    for mask in sorted(masks, key=lambda m: -int(m.sum())):
        bh, bw = mask.shape
        box = _place_box(rng, h, w, bh, bw, boxes)
        _draw_mask(grid, mask, box[0], box[1], candidate)
    output = grid.copy()
    target = _key(shape)
    for comp in _components(grid, candidate):
        if _key(_cell_mask(comp)) == target:
            for r, c in comp:
                output[r, c] = accent
    return grid, output


def _counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} is out of range")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _pair_roles(rng, base, variable, previous_accent):
    if variable:
        colors = [int(x) for x in rng.permutation(10)[:4]]
        if colors[3] == previous_accent:
            colors[3] = next(c for c in range(10) if c not in colors and c != previous_accent)
        return tuple(colors)
    choices = [c for c in range(10) if c not in base[:3] and c != previous_accent]
    return base[:3] + (int(choices[int(rng.integers(0, len(choices)))]),)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    base = tuple(int(x) for x in rng.permutation(10)[:4])
    variable_colors = bool(rng.integers(0, 2))
    train_categories = ["block", "asym"]
    rng.shuffle(train_categories)
    train_categories += [("block", "asym", "cross", "bar")[int(rng.integers(0, 4))]
                         for _ in range(train_count - 2)]
    puzzle = {"train": [], "test": []}
    train_masks = []
    previous_accent = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train":
                category = train_categories[index]
                shape = _sample_shape(rng, category)
                if index < 2:
                    decoys = 1 if category == "asym" else 0
                    matches = int(rng.integers(1, 3))
                    candidates = int(rng.integers(max(4, matches + 2), 6))
                    forced_rotation = category == "asym"
                else:
                    decoys = int(rng.integers(0, 4))
                    matches = int(rng.integers(1, 4))
                    candidates = int(rng.integers(max(3, matches + 2), 9))
                    forced_rotation = category == "asym" and bool(rng.integers(0, 2))
                train_masks.append(shape.copy())
            else:
                if rng.integers(0, 5) < 2:
                    shape = train_masks[int(rng.integers(0, len(train_masks)))].copy()
                else:
                    category = ("block", "asym", "cross", "bar")[int(rng.integers(0, 4))]
                    shape = _sample_shape(rng, category)
                decoys = int(rng.integers(0, 4))
                matches = int(rng.integers(1, 4))
                candidates = int(rng.integers(max(3, matches + 2), 9))
                forced_rotation = (not np.all(shape) and _key(shape) != _key(_CROSS)
                                   and _key(np.rot90(shape)) != _key(shape)
                                   and bool(rng.integers(0, 2)))
            roles = _pair_roles(rng, base, variable_colors, previous_accent)
            previous_accent = roles[3]
            made = False
            for attempt in range(70):
                if attempt < 32:
                    size = (int(rng.integers(20, 31)), int(rng.integers(20, 31)))
                    use_decoys, use_candidates = decoys, candidates
                elif attempt < 55:
                    size = (int(rng.integers(26, 31)), int(rng.integers(26, 31)))
                    use_decoys, use_candidates = min(decoys, 2), min(candidates, 6)
                else:
                    size = (30, 30)
                    use_decoys, use_candidates = min(decoys, 2), max(matches + 2, 5)
                try:
                    inp, out = _construct(rng, shape, roles, use_decoys, use_candidates,
                                          matches, forced_rotation, size)
                    _analyze_pair(inp, out)
                    puzzle[split].append({"input": inp, "output": out})
                    made = True
                    break
                except (_PlacementError, AssertionError, ValueError):
                    continue
            if not made:
                raise ValueError("could not construct a separated layout")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen = set()
        infos = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                inp, out = pair["input"], pair["output"]
                info = _analyze_pair(inp, out)
                if inp.shape == (20, 20):
                    digest = hashlib.sha256(inp.astype(np.int8).tobytes() +
                                            out.astype(np.int8).tobytes()).hexdigest()
                    assert digest not in _SOURCE_PAIR_DIGESTS
                key = (inp.shape, inp.tobytes())
                assert key not in seen
                seen.add(key)
                infos[split].append(info)
        train = infos["train"]
        assert any(i["decoys"] == 0 for i in train)
        assert any(i["decoys"] >= 1 and set(i["decoy_keys"]) & set(i["nonmatches"])
                   for i in train)
        assert any(np.all(i["prototype_mask"]) and i["prototype_mask"].shape[0] >= 2
                   and i["prototype_mask"].shape[1] >= 2 for i in train)
        assert any(_key(np.rot90(i["prototype_mask"])) != i["prototype"] and
                   _key(np.rot90(i["prototype_mask"])) in i["nonmatches"]
                   for i in train)
        assert all(i["matches"] >= 1 for i in infos["test"])
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError("malformed generated puzzle") from error


def _same_puzzle(a, b):
    return all(np.array_equal(x[role], y[role])
               for split in ("train", "test") for x, y in zip(a[split], b[split])
               for role in ("input", "output")) and all(
                   len(a[split]) == len(b[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(12000 + i) for i in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _same_puzzle(generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3))
    assert not _same_puzzle(generate_puzzle(778, 7, 3), generate_puzzle(779, 7, 3))
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(20000 + i, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    pairs = [(split, _analyze_pair(pair["input"], pair["output"]))
             for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    assert len({i["shape"] for _, i in pairs}) >= 30
    assert len({i["prototype"] for _, i in pairs}) >= 20
    assert len({i["frame_box"] for _, i in pairs}) >= 80
    assert {i["decoys"] for _, i in pairs} >= {0, 1, 2}
    assert len({i["roles"][0] for _, i in pairs}) >= 8
    assert any(i["prototype_area"] >= 9 for _, i in pairs)
    assert any(i["max_candidate_area"] >= 12 for _, i in pairs)
    assert any(i["fragment_missing"] and max(i["fragment_missing"]) == 1 for _, i in pairs)
    assert any(i["candidates"] >= 7 for split, i in pairs if split == "train")
    assert any(i["candidates"] <= 5 for split, i in pairs if split == "test")
    categories = {"train": set(), "test": set()}
    styles = set()
    quadrants = set()
    block_sizes = set()
    decoy_without_copy = 0
    combined = 0
    rotated_tests = 0
    for split, info in pairs:
        mask = info["prototype_mask"]
        category = ("cross" if info["prototype"] == _key(_CROSS) else
                    "bar" if 1 in mask.shape else "block" if np.all(mask) else "asym")
        categories[split].add(category)
        styles.update(info["fragment_styles"])
        if category == "block":
            block_sizes.add(mask.shape)
        if info["decoys"] and not set(info["decoy_keys"]) & set(info["nonmatches"]):
            decoy_without_copy += 1
        combined += category == "asym" and info["decoys"] >= 1 and info["candidates"] >= 6
        rotated_tests += (split == "test" and category == "asym" and
                          _key(np.rot90(mask)) in info["nonmatches"])
        h, w = info["shape"]
        r0, c0, r1, c1 = info["frame_box"]
        quadrants.add(((r0 + r1) >= h, (c0 + c1) >= w))
    assert categories["train"] == categories["test"] == {"block", "bar", "cross", "asym"}
    assert styles == {"corner", "three", "gap"}
    assert {(2, 3), (3, 3)} <= block_sizes
    assert len(quadrants) == 4 and decoy_without_copy >= 5
    assert combined >= 10 and rotated_tests >= 5
    assert all(len({_analyze_pair(pair["input"], pair["output"])["prototype"]
                    for split in ("train", "test") for pair in puzzle[split]}) >= 2
               for puzzle in puzzles)
    assert all(len({_analyze_pair(pair["input"], pair["output"])["roles"][3]
                    for split in ("train", "test") for pair in puzzle[split]}) >= 2
               for puzzle in puzzles)
