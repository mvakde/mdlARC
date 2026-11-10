"""Compositional instances of the contour/interior toggle ARC task."""

import numpy as np


_SOURCE_INPUTS = (
    ("00000000555550", "04444000555550", "04004000555550",
     "04004000555550", "04444000555550", "00000000000000",
     "00000044440000", "03330044440000", "03030044440000",
     "03330044440000", "00000000000000"),
    ("000000000000", "022222220000", "020000020000", "020000020000",
     "022222220000", "000000000000", "000000077770", "006660077770",
     "006660077770", "006660077770", "000000000000", "000000000000"),
    ("0000880000", "0008008000", "0080000800", "0800000080",
     "8888888888", "3333333333", "3333333333", "0333333330",
     "0033333300", "0003333000"),
    ("00000000000000", "00000111111000", "00001100001000",
     "00110000001100", "00100000000100", "00110000000100",
     "00001000000100", "00011000000100", "00100000000100",
     "00111111111100", "00000000000000", "00000000000000",
     "00007777777770", "00007777777770", "00007777777770",
     "00007777777770", "00007777777770", "00007777777770",
     "00000000000000"),
)


def _boundary(mask):
    padded = np.pad(mask, 1, constant_values=False)
    return mask & (~padded[:-2, 1:-1] | ~padded[2:, 1:-1]
                   | ~padded[1:-1, :-2] | ~padded[1:-1, 2:])


def _components(mask, diagonal=True):
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        seen[row, col] = True
        stack = [(row, col)]
        points = []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                        and mask[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(points)
    return result


def _holes(occupied):
    """Four-neighbor background regions enclosed by one colored component."""
    padded = np.pad(occupied, 1, constant_values=False)
    exterior = np.zeros_like(padded)
    exterior[0, 0] = True
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if (0 <= rr < padded.shape[0] and 0 <= cc < padded.shape[1]
                    and not padded[rr, cc] and not exterior[rr, cc]):
                exterior[rr, cc] = True
                stack.append((rr, cc))
    return (~padded & ~exterior)[1:-1, 1:-1]


def _trim(mask):
    rows, cols = np.where(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _shape(rng, kind):
    for _ in range(60):
        if kind == "square":
            size = int(rng.integers(3, 9))
            mask = np.ones((size, size), dtype=bool)
        elif kind == "rect":
            h = int(rng.integers(3, 11))
            w = int(rng.integers(3, 12))
            if abs(h - w) < 2:
                w = min(12, h + int(rng.integers(2, 5)))
            mask = np.ones((h, w), dtype=bool)
        elif kind == "taper":
            h = int(rng.integers(4, 9))
            w = int(rng.integers(6, 13))
            tip = int(rng.integers(2, min(w - 1, 6)))
            plateau = int(rng.integers(1, 3))
            mask = _taper(h, w, tip, plateau)
            mask = np.rot90(mask, int(rng.integers(0, 4)))
            if rng.random() < .5:
                mask = np.fliplr(mask)
        else:
            h = int(rng.integers(5, 11))
            w = int(rng.integers(6, 12))
            left, right = [int(rng.integers(0, 3))], []
            right.append(left[0] + w - 1)
            for _row in range(1, h):
                l = int(np.clip(left[-1] + rng.integers(-3, 4), 0, 5))
                r = int(np.clip(right[-1] + rng.integers(-3, 4),
                                max(l + 3, w - 4), w + 4))
                left.append(l)
                right.append(r)
            mask = np.zeros((h, max(right) - min(left) + 1), dtype=bool)
            for row, (l, r) in enumerate(zip(left, right)):
                mask[row, l - min(left):r - min(left) + 1] = True
            if rng.random() < .5:
                mask = np.rot90(mask, int(rng.integers(0, 4)))
            mask = _trim(mask)
        outline = _boundary(mask)
        if (np.any(mask & ~outline) and len(_components(mask)) == 1
                and len(_components(outline)) == 1
                and len(_components(_holes(outline), False)) == 1
                and np.array_equal(_holes(outline), mask & ~outline)):
            return mask
    raise RuntimeError("Could not construct an enclosed silhouette")


def _taper(height, width, tip, plateau):
    mask = np.zeros((height, width), dtype=bool)
    inset = (width - tip) // 2
    slope_rows = max(2, height - plateau)
    for row in range(height):
        amount = int(round(inset * max(0, slope_rows - 1 - row) /
                           (slope_rows - 1)))
        mask[row, amount:width - amount] = True
    return mask


def _outline(rng, mask, thick):
    edge = _boundary(mask)
    if not thick:
        return edge
    candidates = np.argwhere(mask & ~edge)
    rng.shuffle(candidates)
    # Keep one four-connected enclosed interior while adding local contour cells.
    target = int(rng.integers(1, min(5, len(candidates)) + 1))
    added = 0
    for row, col in candidates:
        if added == target:
            break
        row, col = int(row), int(col)
        trial = edge.copy()
        trial[row, col] = True
        hole = _holes(trial)
        if (len(_components(hole, False)) == 1
                and np.array_equal(hole, mask & ~trial)):
            edge = trial
            added += 1
    return edge


def _item(mask, hollow, outline=None, kind=""):
    if outline is None:
        outline = _boundary(mask)
    return {"mask": mask, "outline": outline, "hollow": bool(hollow),
            "kind": kind, "color": None, "group": None, "row": 0, "col": 0}


def _contact_group(rng, flavor, solid_first=False):
    if flavor == "taper":
        width = int(rng.integers(7, 13))
        upper = _taper(int(rng.integers(4, 8)), width,
                       int(rng.integers(2, min(width - 1, 6))),
                       int(rng.integers(1, 3)))
        lower = np.flipud(_taper(int(rng.integers(4, 8)), width,
                                 int(rng.integers(2, min(width - 1, 6))),
                                 int(rng.integers(1, 3))))
        shift = 0
    else:
        upper = np.ones((int(rng.integers(3, 7)),
                         int(rng.integers(4, 9))), dtype=bool)
        lower = np.ones((int(rng.integers(3, 7)),
                         int(rng.integers(4, 9))), dtype=bool)
        shift = int(rng.integers(-min(3, lower.shape[1] - 2),
                                 min(4, upper.shape[1] - 1)))
    left = min(0, shift)
    total_w = max(upper.shape[1], shift + lower.shape[1]) - left
    total_h = upper.shape[0] + lower.shape[0]
    masks = []
    for arr, row, col in ((upper, 0, -left),
                          (lower, upper.shape[0], shift - left)):
        canvas = np.zeros((total_h, total_w), dtype=bool)
        canvas[row:row + arr.shape[0], col:col + arr.shape[1]] = arr
        masks.append(canvas)
    rotation = int(rng.integers(0, 4))
    reflected = rng.random() < .5
    transformed = []
    for canvas in masks:
        canvas = np.rot90(canvas, rotation)
        if reflected:
            canvas = np.fliplr(canvas)
        transformed.append(canvas)
    if np.any(transformed[0] & transformed[1]):
        raise AssertionError("Contact silhouettes overlap")
    states = (False, bool(rng.integers(0, 2))) if solid_first else (
        bool(rng.integers(0, 2)), bool(rng.integers(0, 2)))
    result = []
    for canvas, state in zip(transformed, states):
        rows, cols = np.where(canvas)
        local = canvas[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        obj = _item(local, state, kind=flavor)
        obj["row"], obj["col"] = int(rows.min()), int(cols.min())
        result.append(obj)
    return result


def _groups_for_pair(rng, regime, index):
    groups = []
    if regime == "train" and index == 0:
        n = int(rng.choice((2, 4)))
        for i in range(n):
            kind = "square" if i < 2 or rng.random() < .75 else "rect"
            groups.append([_item(_shape(rng, kind), i == 0 if i < 2
                                 else bool(rng.integers(0, 2)), kind=kind)])
        return groups
    if regime == "train" and index == 1:
        irregular = _shape(rng, "irregular")
        groups.append([_item(irregular, True,
                             _outline(rng, irregular, True), "irregular")])
        groups.append(_contact_group(rng, "taper" if rng.random() < .6
                                     else "rect", solid_first=True))
        return groups
    n = int(rng.integers(2, 6))
    square_only = regime == "test" and rng.random() < .18
    contact = not square_only and rng.random() < (.33 if regime == "train" else .43)
    if contact:
        groups.append(_contact_group(rng, "taper" if rng.random() < .55
                                     else "rect"))
    while sum(map(len, groups)) < n:
        if square_only:
            kind = "square"
        else:
            choices = ("square", "rect", "taper", "irregular")
            probabilities = ((.28, .29, .20, .23) if regime == "train"
                             else (.17, .29, .25, .29))
            kind = str(rng.choice(choices, p=probabilities))
        mask = _shape(rng, kind)
        thick = kind == "irregular" and rng.random() < .38
        hollow = True if thick else bool(rng.integers(0, 2))
        groups.append([_item(mask, hollow, _outline(rng, mask, thick), kind)])
    return groups


def _assign_colors(rng, groups, background, reserved, regime):
    allowed = [c for c in range(10) if c != background
               and (regime == "test" or c != reserved)]
    for group in groups:
        used = set()
        for obj in group:
            choices = [c for c in allowed if c not in used]
            color = int(rng.choice(choices))
            obj["color"] = color
            used.add(color)
    items = [obj for group in groups for obj in group]
    if regime == "test" and reserved in allowed and rng.random() < .75:
        target = int(rng.integers(0, len(items)))
        group = next(g for g in groups if any(o is items[target] for o in g))
        if all(other is items[target] or other["color"] != reserved
               for other in group):
            items[target]["color"] = reserved
    if regime == "test" and len(groups) >= 2 and rng.random() < .45:
        first_group, second_group = rng.choice(len(groups), size=2, replace=False)
        first = groups[int(first_group)][int(rng.integers(0, len(groups[int(first_group)])))]
        second = groups[int(second_group)][int(rng.integers(0, len(groups[int(second_group)])))]
        first["hollow"], second["hollow"] = True, False
        forbidden = {background}
        forbidden.update(o["color"] for o in groups[int(first_group)] if o is not first)
        forbidden.update(o["color"] for o in groups[int(second_group)] if o is not second)
        options = [c for c in allowed if c not in forbidden]
        if options:
            shared = int(rng.choice(options))
            first["color"] = second["color"] = shared


def _expand(mask):
    padded = np.pad(mask, 1, constant_values=False)
    expanded = np.zeros_like(mask)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            expanded |= padded[dr:dr + mask.shape[0],
                               dc:dc + mask.shape[1]]
    return expanded


def _place(rng, groups, regime):
    complex_pair = any(len(group) == 2 for group in groups)
    lower = 21 if complex_pair else (16 if len(groups) <= 3 else 20)
    canvas_h = int(rng.integers(lower, 31))
    canvas_w = int(rng.integers(lower, 31))
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    rng.shuffle(groups)
    for group in groups:
        gh = max(obj["row"] + obj["mask"].shape[0] for obj in group)
        gw = max(obj["col"] + obj["mask"].shape[1] for obj in group)
        if gh > canvas_h or gw > canvas_w:
            return None
        group_mask = np.zeros((gh, gw), dtype=bool)
        for obj in group:
            r, c = obj["row"], obj["col"]
            h, w = obj["mask"].shape
            group_mask[r:r + h, c:c + w] |= obj["mask"]
        clearance = _expand(occupied)
        positions = [(r, c) for r in range(canvas_h - gh + 1)
                     for c in range(canvas_w - gw + 1)]
        rng.shuffle(positions)
        found = False
        for r, c in positions:
            if not np.any(clearance[r:r + gh, c:c + gw] & group_mask):
                for obj in group:
                    obj["row"] += r
                    obj["col"] += c
                occupied[r:r + gh, c:c + gw] |= group_mask
                found = True
                break
        if not found:
            return None
    return [obj for group in groups for obj in group]


def _draw(rng, items, background):
    min_r = min(obj["row"] for obj in items)
    min_c = min(obj["col"] for obj in items)
    max_r = max(obj["row"] + obj["mask"].shape[0] for obj in items)
    max_c = max(obj["col"] + obj["mask"].shape[1] for obj in items)
    margins = [int(rng.integers(0, 4)) for _ in range(4)]
    for axis, span in ((0, max_r - min_r), (2, max_c - min_c)):
        if span + margins[axis] + margins[axis + 1] > 30:
            excess = span + margins[axis] + margins[axis + 1] - 30
            take = min(excess, margins[axis + 1])
            margins[axis + 1] -= take
            margins[axis] -= excess - take
    height = max_r - min_r + margins[0] + margins[1]
    width = max_c - min_c + margins[2] + margins[3]
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    silhouettes = np.zeros((height, width), dtype=bool)
    for obj in items:
        r = obj["row"] - min_r + margins[0]
        c = obj["col"] - min_c + margins[2]
        mask, edge = obj["mask"], obj["outline"]
        h, w = mask.shape
        assert not np.any(silhouettes[r:r + h, c:c + w] & mask)
        silhouettes[r:r + h, c:c + w] |= mask
        inp[r:r + h, c:c + w][edge if obj["hollow"] else mask] = obj["color"]
        out[r:r + h, c:c + w][mask if obj["hollow"] else edge] = obj["color"]
    return inp, out


def _make_pair(rng, regime, index, reserved):
    for _ in range(120):
        groups = _groups_for_pair(rng, regime, index)
        background = int(rng.choice([c for c in range(10)
                                     if c != reserved or (regime == "test" and index > 0)]))
        _assign_colors(rng, groups, background, reserved, regime)
        if regime == "train" and index == 0:
            # One visible color demonstrates both directions, as in the source.
            groups[1][0]["color"] = groups[0][0]["color"]
        if regime == "test" and index == 0:
            # This color is absent even as a training background.
            if not any(o["color"] == reserved for g in groups for o in g):
                target = groups[0][0]
                target["color"] = reserved
        items = _place(rng, groups, regime)
        if items is None:
            continue
        inp, out = _draw(rng, items, background)
        if np.array_equal(inp, out):
            continue
        try:
            _analyze_pair(inp, out)
        except AssertionError:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("Could not place a valid pair")


def _check_count(value, lower, upper, name):
    if value is not None and (isinstance(value, (bool, np.bool_))
                              or not isinstance(value, (int, np.integer))
                              or not lower <= value <= upper):
        raise ValueError(f"{name} must be an integer from {lower} to {upper}")


def _signature(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, 2, 10, "num_train")
    _check_count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # The toggle has no hidden color meaning. Reserve one label for possible
    # unseen test colors, while all displayed backgrounds and colors vary by pair.
    reserved = int(rng.integers(0, 10))
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(40):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(pair_rng, regime, index, reserved)
                if (regime == "train" and index == 1
                        and _background(pair["input"]) ==
                        _background(puzzle["train"][0]["input"])):
                    continue
                sig = _signature(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[regime].append(pair)
                    break
            else:
                raise RuntimeError("Could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _background(grid):
    perimeter = np.concatenate((grid[0], grid[-1], grid[1:-1, 0],
                                grid[1:-1, -1]))
    counts = np.bincount(perimeter.astype(int), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > len(perimeter) // 2
    return background


def _analyze_pair(inp, out):
    background = _background(inp)
    assert _background(out) == background
    expected = np.full_like(inp, background)
    used = np.zeros_like(inp, dtype=bool)
    infos = []
    for color in range(10):
        if color == background:
            continue
        for component in _components(inp == color):
            rows, cols = zip(*component)
            r0, r1 = min(rows), max(rows) + 1
            c0, c1 = min(cols), max(cols) + 1
            occupied = np.zeros((r1 - r0, c1 - c0), dtype=bool)
            for r, c in component:
                occupied[r - r0, c - c0] = True
            holes = _holes(occupied)
            assert len(_components(holes, False)) <= 1
            hollow = bool(np.any(holes))
            silhouette = occupied | holes
            edge = _boundary(silhouette)
            assert np.any(silhouette & ~edge)
            if hollow:
                assert np.all(occupied[edge])
            else:
                assert np.array_equal(occupied, silhouette)
            assert not np.any(used[r0:r1, c0:c1] & silhouette)
            used[r0:r1, c0:c1] |= silhouette
            expected[r0:r1, c0:c1][silhouette if hollow else edge] = color
            infos.append({"color": color, "hollow": hollow,
                          "shape": silhouette.shape,
                          "square": bool(np.all(silhouette)
                                         and silhouette.shape[0] == silhouette.shape[1]),
                          "rect": bool(np.all(silhouette)),
                          "thick": bool(hollow and np.any(occupied & ~edge)),
                          "narrow_tip": bool(not np.all(silhouette) and any(
                              min(int(np.sum(line)) for line in axis if np.any(line)) <= 2
                              and max(int(np.sum(line)) for line in axis) >= 5
                              for axis in (silhouette, silhouette.T))),
                          "sharp_step": bool(not np.all(silhouette) and any(
                              any(abs(a - b) >= 3 for a, b in zip(widths, widths[1:]))
                              for widths in ([int(np.sum(line)) for line in axis]
                                             for axis in (silhouette, silhouette.T)))),
                          "touches_edge": r0 == 0 or c0 == 0 or r1 == inp.shape[0]
                          or c1 == inp.shape[1]})
    assert infos and np.array_equal(expected, out)
    assert not np.array_equal(inp, out)
    return infos


def _touching_colors(grid, background):
    return bool(np.any((grid[:-1] != background) & (grid[1:] != background)
                       & (grid[:-1] != grid[1:]))
                or np.any((grid[:, :-1] != background)
                          & (grid[:, 1:] != background)
                          & (grid[:, :-1] != grid[:, 1:])))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    train_hollows = train_solids = train_thick = train_contact = False
    train_colors = set()
    test_object_colors = set()
    for regime, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            for rows in _SOURCE_INPUTS:
                if inp.shape == (len(rows), len(rows[0])):
                    assert not np.array_equal(inp, np.array(
                        [[int(c) for c in row] for row in rows]))
            infos = _analyze_pair(inp, out)
            assert len(infos) >= 2
            background = _background(inp)
            assert {int(c) for c in np.unique(inp)} == {
                int(c) for c in np.unique(out)}
            if regime == "train":
                train_colors.update(int(c) for c in np.unique(inp))
                train_hollows |= any(x["hollow"] for x in infos)
                train_solids |= any(not x["hollow"] for x in infos)
                train_thick |= any(x["thick"] for x in infos)
                train_contact |= _touching_colors(inp, background)
            else:
                test_object_colors.update(x["color"] for x in infos)
    assert train_hollows and train_solids and train_thick and train_contact
    assert test_object_colors - train_colors
    first = _analyze_pair(train[0]["input"], train[0]["output"])
    assert any(x["hollow"] for x in first) and any(not x["hollow"] for x in first)
    assert any(x["square"] for x in first)
    assert any({x["hollow"] for x in first if x["color"] == color}
               == {False, True} for color in {x["color"] for x in first})
    assert _background(train[0]["input"]) != _background(train[1]["input"])
    assert len(first) != len(_analyze_pair(train[1]["input"], train[1]["output"]))


def _equal_puzzles(left, right):
    return all(len(left[k]) == len(right[k]) and all(
        np.array_equal(a[t], b[t]) for a, b in zip(left[k], right[k])
        for t in ("input", "output")) for k in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, dimensions, structures = set(), set(), set(), set()
    hollow_counts, thick_count, contact_count, edge_count = set(), 0, 0, 0
    backgrounds, pair_counts = set(), set()
    square_tests = mixed_pairs = tiny = tall = wide = narrow = sharp = 0
    same_color = unseen_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = {int(c) for pair in puzzle["train"]
                        for c in np.unique(pair["input"])}
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                infos = _analyze_pair(inp, out)
                dimensions.add(inp.shape)
                backgrounds.add(_background(inp))
                pair_counts.add(len(infos))
                hollow_counts.add(sum(x["hollow"] for x in infos))
                thick_count += any(x["thick"] for x in infos)
                contact_count += _touching_colors(inp, _background(inp))
                edge_count += any(x["touches_edge"] for x in infos)
                square_tests += regime == "test" and all(x["square"] for x in infos)
                mixed_pairs += (any(x["rect"] for x in infos)
                                and any(not x["rect"] for x in infos))
                tiny += any(x["square"] and x["shape"] == (3, 3) for x in infos)
                tall += any(x["rect"] and x["shape"][0] >= x["shape"][1] + 2
                            for x in infos)
                wide += any(x["rect"] and x["shape"][1] >= x["shape"][0] + 2
                            for x in infos)
                narrow += any(x["narrow_tip"] for x in infos)
                sharp += any(x["sharp_step"] for x in infos)
                same_color += any({x["hollow"] for x in infos if x["color"] == c}
                                  == {False, True} for c in {x["color"] for x in infos})
                unseen_tests += (regime == "test" and any(
                    x["color"] not in train_colors for x in infos))
                structures.add((inp.shape, tuple(sorted(x["shape"] for x in infos)),
                                tuple(sorted((x["rect"], x["hollow"])
                                             for x in infos))))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 25 and len(structures) >= 50
    assert len(backgrounds) >= 7 and len(pair_counts) >= 3
    assert len(hollow_counts) >= 4 and thick_count >= 20
    assert contact_count >= 20 and edge_count >= 20
    assert square_tests >= 5 and mixed_pairs >= 20
    assert tiny >= 10 and tall >= 10 and wide >= 10
    assert narrow >= 10 and sharp >= 10 and same_color >= 42
    assert unseen_tests >= 42
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                 (2, None), (10, None)):
        puzzle = generate_puzzle(913, a, b)
        assert (a is None or len(puzzle["train"]) == a)
        assert (b is None or len(puzzle["test"]) == b)
        assert _equal_puzzles(puzzle, generate_puzzle(913, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(12, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
