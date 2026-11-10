import numpy as np


_SOURCE_INPUTS = {
    "7777111777777777/7777711778777777/7777771777777777/"
    "7777777777777777/7770777777777777/7770777777777777/"
    "7700077777377777/7700077777777777/7700077777777777/"
    "7700077777555555/7777777777755555/7777777777775555/"
    "7777777777777555/7777700007777777/7777700000077777/"
    "7777700007777777",

    "7777777777777777/9999777000077777/9997777000000777/"
    "9977777000077777/7777777777777777/7777777777777777/"
    "5555555577777770/5555555770000777/5555557770000007/"
    "5555577770000777/7777777777777777/7777777777777777/"
    "7777777777777777/7777777777777777/7977777777777777/"
    "7777777777777777",

    "7777777777777777/5555577777770077/5555777000770077/"
    "5557777000777777/5577777000777777/7777777000722222/"
    "7777777707772222/7777777707777222/7777777777777722/"
    "7777770000777772/7777770000007777/7777770000777777/"
    "7777777777777777/7777777777787777/7777777777777777/"
    "7777777777777777",
}


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")
    if num_test is not None:
        num_test = _integer(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    key = seed % (2**63 - 1)
    if num_train is None and num_test is None:
        num_test = 1 + key % 4
        num_train = max(num_test + 1, 2 + (key // 4 + key) % 9)
    elif num_train is None:
        low = num_test + 1
        num_train = low + key % (11 - low)
    else:
        num_test = 1 + key % min(4, num_train - 1)
    return num_train, num_test


def _components(grid, color):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
    found = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        found.append(component)
    return found


def _bbox(cells):
    ys = [p[0] for p in cells]
    xs = [p[1] for p in cells]
    return min(ys), max(ys), min(xs), max(xs)


def _rows(cells):
    rows = {}
    for y, x in cells:
        rows.setdefault(y, []).append(x)
    for xs in rows.values():
        xs.sort()
        if xs != list(range(xs[0], xs[-1]+1)):
            return None
    ys = sorted(rows)
    if ys != list(range(ys[0], ys[-1]+1)):
        return None
    return rows


def _repair(cells):
    """Recover one aligned rectangle and exactly two surplus cells."""
    rows = _rows(cells)
    if rows is None:
        return None
    ys = sorted(rows)
    possibilities = []
    if len(ys) >= 5:
        for direction in ("top", "bottom"):
            body_ys = ys[2:] if direction == "top" else ys[:-2]
            stem_ys = ys[:2] if direction == "top" else ys[-2:]
            body = [rows[y] for y in body_ys]
            if (len(body) >= 3 and len(body[0]) >= 3
                    and all(xs == body[0] for xs in body)
                    and len(rows[stem_ys[0]]) == len(rows[stem_ys[1]]) == 1
                    and rows[stem_ys[0]] == rows[stem_ys[1]]
                    and rows[stem_ys[0]][0] in body[0]):
                core = {(y, x) for y in body_ys for x in body[0]}
                possibilities.append((core, cells-core, direction))
    if len(ys) >= 3:
        lengths = [len(rows[y]) for y in ys]
        for index in range(len(ys)):
            ordinary = [rows[y] for y in ys if y != ys[index]]
            if (not ordinary or len(ordinary[0]) < 3
                    or any(xs != ordinary[0] for xs in ordinary)
                    or lengths[index] != len(ordinary[0]) + 2):
                continue
            long = rows[ys[index]]
            if long[2:] == ordinary[0]:
                direction = "left"
            elif long[:-2] == ordinary[0]:
                direction = "right"
            else:
                continue
            core = {(y, x) for y in ys for x in ordinary[0]}
            possibilities.append((core, cells-core, direction))
    return possibilities[0] if len(possibilities) == 1 else None


def _stair_shape(height, width, side):
    return {(y, x) for y in range(height)
            for x in (range(width-y) if side == "left" else range(y, width))}


def _staircases(grid, background):
    result = []
    for value in np.unique(grid):
        color = int(value)
        if color == background:
            continue
        for cells in _components(grid, color):
            rows = _rows(cells)
            if rows is None or len(rows) < 3:
                continue
            ys = sorted(rows)
            widths = [len(rows[y]) for y in ys]
            if widths != list(range(widths[0], widths[0]-len(ys), -1)):
                continue
            if widths[-1] < 1:
                continue
            lefts = [rows[y][0] for y in ys]
            rights = [rows[y][-1] for y in ys]
            if len(set(lefts)) == 1:
                side = "left"
            elif len(set(rights)) == 1:
                side = "right"
            else:
                continue
            result.append({"color": color, "cells": cells, "side": side,
                           "height": len(ys), "width": widths[0],
                           "rows": set(ys), "bbox": _bbox(cells)})
    return result


def _associated(guide, core):
    _, _, gleft, gright = guide["bbox"]
    _, _, cleft, cright = _bbox(core)
    core_rows = {y for y, _ in core}
    return (bool(guide["rows"] & core_rows)
            and (gright < cleft if guide["side"] == "left" else gleft > cright))


def _body_shape(height, width, defect, row=0, stem=0):
    core = {(y, x) for y in range(height) for x in range(width)}
    if defect == "top":
        extra = {(-2, stem), (-1, stem)}
    elif defect == "bottom":
        extra = {(height, stem), (height+1, stem)}
    elif defect == "left":
        extra = {(row, -2), (row, -1)}
    else:
        extra = {(row, width), (row, width+1)}
    return core, extra


def _translate(cells, y, x):
    return {(y+dy, x+dx) for dy, dx in cells}


def _in_bounds(cells, height, width):
    return all(0 <= y < height and 0 <= x < width for y, x in cells)


def _touching(cells, other):
    return bool(cells & other) or any(
        (y+dy, x+dx) in other for y, x in cells
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))


def _place(grid, local, color, rng, background, guide=None,
           reject_eligible=False, require_overlap=False, must_associate=None):
    miny, maxy, minx, maxx = _bbox(local)
    height, width = grid.shape
    anchors = [(y, x) for y in range(-miny, height-maxy)
               for x in range(-minx, width-maxx)]
    rng.shuffle(anchors)
    occupied_same = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
    for ay, ax in anchors:
        cells = _translate(local, ay, ax)
        if any(int(grid[y, x]) != background for y, x in cells):
            continue
        if _touching(cells, occupied_same):
            continue
        if require_overlap and guide is not None:
            if not ({y for y, _ in cells} & guide["rows"]):
                continue
        if must_associate is not None:
            side, core = must_associate
            staircase = {"bbox": _bbox(cells),
                         "rows": {y for y, _ in cells}, "side": side}
            if not _associated(staircase, core):
                continue
        if reject_eligible and guide is not None:
            repair = _repair(cells)
            if repair is not None and _associated(guide, repair[0]):
                continue
        for y, x in cells:
            grid[y, x] = color
        return cells
    return None


def _dimension(rng):
    band = int(rng.choice(3, p=(0.27, 0.38, 0.35)))
    bounds = ((16, 18), (19, 24), (25, 30))[band]
    return int(rng.integers(bounds[0], bounds[1]+1))


def _construct_pair(rng, roles, spec, rich):
    height, width = _dimension(rng), _dimension(rng)
    background = int(rng.choice(roles["backgrounds"]))
    carrier, accent, guide_color = roles["carrier"], roles["accent"], roles["guide"]
    extras = roles["extras"]
    grid = np.full((height, width), background, dtype=np.int8)

    body_h = int(rng.integers(3, 6))
    body_w = int(rng.integers(3, 7))
    defect = spec["defect"]
    side = spec["side"]
    row = int(rng.integers(body_h))
    stem = int(rng.integers(body_w))
    core_local, surplus_local = _body_shape(body_h, body_w, defect, row, stem)
    target_local = core_local | surplus_local
    guide_width = int(rng.integers(4, 9))
    gap = int(rng.integers(0, 5))

    layouts = []
    miny, maxy, minx, maxx = _bbox(target_local)
    for cy in range(-miny, height-maxy):
        for cx in range(-minx, width-maxx):
            target = _translate(target_local, cy, cx)
            _, _, tleft, tright = _bbox(target)
            gx = tleft-gap-guide_width if side == "left" else tright+gap+1
            if gx < 0 or gx+guide_width > width:
                continue
            for gy in range(height-3):
                if set(range(gy, gy+4)) & set(range(cy, cy+body_h)):
                    layouts.append((cy, cx, gy, gx))
    if not layouts:
        raise RuntimeError("no fitting guide and body")
    edge_layouts = [p for p in layouts if
                    min(p[0]+miny, p[1]+minx, p[2], p[3]) == 0 or
                    p[0]+maxy == height-1 or p[1]+maxx == width-1 or
                    p[2]+3 == height-1 or p[3]+guide_width-1 == width-1]
    candidates = edge_layouts if edge_layouts and rng.random() < 0.38 else layouts
    cy, cx, gy, gx = candidates[int(rng.integers(len(candidates)))]
    core = _translate(core_local, cy, cx)
    surplus = _translate(surplus_local, cy, cx)
    target = core | surplus
    guide_cells = _translate(_stair_shape(4, guide_width, side), gy, gx)
    if target & guide_cells:
        raise RuntimeError("overlapping primary shapes")
    for y, x in target:
        grid[y, x] = carrier
    for y, x in guide_cells:
        grid[y, x] = guide_color
    guide = {"bbox": _bbox(guide_cells), "rows": {y for y, _ in guide_cells},
             "side": side}

    area = height * width
    count = 1 if area < 370 else (int(rng.integers(1, 3)) if area < 620
                                     else int(rng.integers(1, 4)))
    for index in range(count):
        if index == 0 and rng.random() < 0.55:
            local = target_local
        else:
            dh = int(rng.integers(3, 6))
            dw = int(rng.integers(3, 7))
            direction = ("top", "bottom", "left", "right")[int(rng.integers(4))]
            a, b = _body_shape(dh, dw, direction,
                               int(rng.integers(dh)), int(rng.integers(dw)))
            local = a | b
        if _place(grid, local, carrier, rng, background, guide,
                  reject_eligible=True) is None:
            raise RuntimeError("no room for carrier distractor")

    if rng.random() < 0.7:
        blobs = ({(0, 0)}, {(0, 0), (0, 1)},
                 {(0, 0), (1, 0), (1, 1)},
                 {(0, 0), (0, 1), (1, 0), (1, 1)})
        local = blobs[int(rng.integers(len(blobs)))]
        _place(grid, local, carrier, rng, background, guide,
               reject_eligible=True, require_overlap=rng.random() < 0.6)

    stair_count = int(rng.integers(1, 3)) if rich else 0
    for index in range(stair_count):
        stair_h = int(rng.choice((3, 4, 5)))
        stair_w = int(rng.integers(stair_h, min(stair_h+5, 10)))
        stair_side = ("right" if side == "left" else "left") if index == 0 and rich else ("left", "right")[int(rng.integers(2))]
        color = extras[int(rng.integers(len(extras)))]
        local = _stair_shape(stair_h, stair_w, stair_side)
        _place(grid, local, color, rng, background,
               guide={"rows": {y for y, _ in core}},
               require_overlap=rich and index == 0)

    if rich and rng.random() < 0.48:
        extra_width = int(rng.integers(4, 9))
        color = extras[int(rng.integers(len(extras)))]
        _place(grid, _stair_shape(4, extra_width, side), color, rng,
               background, must_associate=(side, core))

    for _ in range(int(rng.integers(0, 4))):
        color = extras[int(rng.integers(len(extras)))]
        shapes = ({(0, 0)}, {(0, 0), (0, 1)},
                  {(0, 0), (1, 0)}, {(0, 0), (0, 1), (1, 0)})
        _place(grid, shapes[int(rng.integers(len(shapes)))], color, rng, background)

    output = grid.copy()
    for y, x in surplus:
        output[y, x] = background
    edge_x = min(x for _, x in core) if side == "left" else max(x for _, x in core)
    for y in range(cy, cy+body_h):
        output[y, edge_x] = accent
    return grid, output


def _compact(grid):
    return "/".join("".join(map(str, map(int, row))) for row in grid)


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _pair_info(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    values, counts = np.unique(input_grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    diff = {tuple(map(int, p)) for p in np.argwhere(input_grid != output_grid)}
    assert len(diff) >= 5
    pre = {int(input_grid[y, x]) for y, x in diff}
    assert len(pre) == 1
    carrier = pre.pop()
    assert carrier != background
    erased = {(y, x) for y, x in diff if int(output_grid[y, x]) == background}
    marked = diff - erased
    assert len(erased) == 2 and len(marked) >= 3
    accents = {int(output_grid[y, x]) for y, x in marked}
    assert len(accents) == 1
    accent = accents.pop()
    assert accent not in (background, carrier)
    assert not np.any(input_grid == accent)
    assert {tuple(map(int, p)) for p in np.argwhere(output_grid == accent)} == marked
    carrier_parts = _components(input_grid, carrier)
    owners = [part for part in carrier_parts if diff <= part]
    assert len(owners) == 1
    target = owners[0]
    repair = _repair(target)
    assert repair is not None
    core, surplus, defect = repair
    assert surplus == erased
    assert len({y for y, _ in core}) == len(marked)
    left = {(y, min(x for yy, x in core if yy == y)) for y, _ in core}
    right = {(y, max(x for yy, x in core if yy == y)) for y, _ in core}
    if marked == left:
        side = "left"
    else:
        assert marked == right
        side = "right"
    expected = input_grid.copy()
    for y, x in surplus:
        expected[y, x] = background
    for y, x in marked:
        expected[y, x] = accent
    assert np.array_equal(expected, output_grid)
    staircases = _staircases(input_grid, background)
    candidates = {stair["color"] for stair in staircases
                  if stair["height"] == 4 and stair["side"] == side
                  and _associated(stair, core) and stair["color"] != carrier}
    assert candidates
    return {"background": background, "carrier": carrier, "accent": accent,
            "side": side, "defect": defect, "core": core, "target": target,
            "carrier_parts": carrier_parts, "stairs": staircases,
            "guide_candidates": candidates, "shape": input_grid.shape}


def _learn(train):
    infos = [_pair_info(p["input"], p["output"]) for p in train]
    assert len({p["carrier"] for p in infos}) == 1
    assert len({p["accent"] for p in infos}) == 1
    guides = set.intersection(*(p["guide_candidates"] for p in infos))
    assert len(guides) == 1
    return {"carrier": infos[0]["carrier"], "accent": infos[0]["accent"],
            "guide": guides.pop()}, infos


def _predict(roles, input_grid):
    values, counts = np.unique(input_grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    carrier = roles["carrier"]
    stairs = [s for s in _staircases(input_grid, background)
              if s["color"] == roles["guide"] and s["height"] == 4]
    assert len(stairs) == 1
    guide = stairs[0]
    eligible = []
    for part in _components(input_grid, carrier):
        repair = _repair(part)
        if repair is not None and _associated(guide, repair[0]):
            eligible.append(repair)
    assert len(eligible) == 1
    core, surplus, _ = eligible[0]
    output = input_grid.copy()
    for y, x in surplus:
        output[y, x] = background
    for y in {y for y, _ in core}:
        xs = [x for yy, x in core if yy == y]
        x = min(xs) if guide["side"] == "left" else max(xs)
        output[y, x] = roles["accent"]
    return output


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed")
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed % (2**63-1))
    colors = [int(c) for c in rng.permutation(10)]
    roles = {"carrier": colors[0], "accent": colors[1], "guide": colors[2],
             "backgrounds": colors[3:6], "extras": colors[6:]}
    first_side = ("left", "right")[int(rng.integers(2))]
    consistent_side = rng.random() < 0.18
    sides = [first_side, first_side if consistent_side else
             ("right" if first_side == "left" else "left")]
    defects = ["top", "bottom", "left", "right"]
    first_defect = defects[int(rng.integers(4))]
    second_defect = defects[int(rng.integers(4))]
    if second_defect == first_defect:
        second_defect = defects[(defects.index(first_defect)+1) % 4]
    specs = [{"side": sides[0], "defect": first_defect},
             {"side": sides[1], "defect": second_defect}]
    specs += [{"side": sides[int(rng.integers(2))],
               "defect": defects[int(rng.integers(4))]}
              for _ in range(train_count-2)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train":
                spec = specs[index]
                rich = index > 0 and rng.random() < 0.48
            else:
                combinations = [(s, d) for s in sides for d in defects]
                unseen = [p for p in combinations if
                          p not in {(s["side"], s["defect"]) for s in specs}]
                pool = unseen if unseen and rng.random() < 0.6 else combinations
                side, defect = pool[int(rng.integers(len(pool)))]
                spec = {"side": side, "defect": defect}
                rich = rng.random() < 0.75
            for attempt in range(180):
                try:
                    inp, out = _construct_pair(rng, roles, spec, rich)
                    if _grid_key(inp) in seen or _compact(inp) in _SOURCE_INPUTS:
                        continue
                    info = _pair_info(inp, out)
                    assert info["side"] == spec["side"]
                    assert info["defect"] == spec["defect"]
                    assert info["carrier"] == roles["carrier"]
                    assert info["accent"] == roles["accent"]
                    assert len([p for p in info["carrier_parts"] if _repair(p)]) >= 2
                    guide_stairs = [s for s in info["stairs"] if s["color"] == roles["guide"]]
                    assert len(guide_stairs) == 1
                    assert np.array_equal(_predict(roles, inp), out)
                    seen.add(_grid_key(inp))
                    puzzle[split].append({"input": inp, "output": out})
                    break
                except (AssertionError, RuntimeError):
                    continue
            else:
                raise RuntimeError(f"failed to construct {split} pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    infos = []
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
        inp, out = pair["input"], pair["output"]
        assert inp.shape == out.shape and not np.array_equal(inp, out)
        key = _grid_key(inp)
        assert key not in seen
        seen.add(key)
        assert _compact(inp) not in _SOURCE_INPUTS
        info = _pair_info(inp, out)
        assert len([part for part in info["carrier_parts"] if _repair(part)]) >= 2
        infos.append(info)
    roles, train_infos = _learn(train)
    assert {info["side"] for info in infos[len(train):]} <= {
        info["side"] for info in train_infos}
    assert len({info["defect"] for info in train_infos}) >= 2
    for pair, info in zip(train + test, infos):
        assert info["carrier"] == roles["carrier"]
        assert info["accent"] == roles["accent"]
        guides = [s for s in info["stairs"] if s["color"] == roles["guide"]]
        assert len(guides) == 1 and guides[0]["height"] == 4
        assert guides[0]["side"] == info["side"]
        eligible = [part for part in info["carrier_parts"] if
                    (repair := _repair(part)) is not None
                    and _associated(guides[0], repair[0])]
        assert len(eligible) == 1 and eligible[0] == info["target"]
        assert np.array_equal(_predict(roles, pair["input"]), pair["output"])


def _equal(a, b):
    return (len(a["train"]) == len(b["train"])
            and len(a["test"]) == len(b["test"])
            and all(np.array_equal(p[k], q[k]) for split in ("train", "test")
                    for p, q in zip(a[split], b[split]) for k in ("input", "output")))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_grid_key(p["train"][0]["input"]) for p in puzzles}) >= 40
    infos = [_pair_info(pair["input"], pair["output"])
             for puzzle in puzzles for pair in puzzle["train"] + puzzle["test"]]
    assert len({info["shape"] for info in infos}) >= 35
    assert {info["defect"] for info in infos} == {"top", "bottom", "left", "right"}
    assert {info["side"] for info in infos} == {"left", "right"}
    assert len({(info["side"], info["defect"]) for info in infos}) == 8
    assert len({info["background"] for info in infos}) >= 8
    assert any(info["shape"][0] <= 18 and info["shape"][1] <= 18 for info in infos)
    assert any(info["shape"][0] >= 25 and info["shape"][1] >= 25 for info in infos)
    guide_widths = set()
    other_heights = set()
    repairable_counts = set()
    body_sizes = set()
    identical_competitors = 0
    smaller_targets = 0
    edge_pairs = 0
    opposite_stairs_on_target_rows = 0
    associated_color_decoys = 0
    varying_backgrounds = 0
    varying_bodies = 0
    varying_counts = 0
    held_out_combinations = 0
    side_modes = set()
    for puzzle in puzzles:
        roles, train_infos = _learn(puzzle["train"])
        all_infos = [_pair_info(pair["input"], pair["output"])
                     for pair in puzzle["train"] + puzzle["test"]]
        side_modes.add(len({info["side"] for info in train_infos}))
        varying_backgrounds += len({info["background"] for info in all_infos}) > 1
        body_dimensions = {(len({y for y, _ in info["core"]}),
                            len({x for _, x in info["core"]})) for info in all_infos}
        varying_bodies += len(body_dimensions) > 1
        varying_counts += len({sum(_repair(p) is not None
                                   for p in info["carrier_parts"])
                               for info in all_infos}) > 1
        train_combos = {(i["side"], i["defect"]) for i in train_infos}
        held_out_combinations += any((i["side"], i["defect"]) not in train_combos
                                     for i in all_infos[len(train_infos):])
        for pair, info in zip(puzzle["train"] + puzzle["test"], all_infos):
            guide = next(s for s in info["stairs"] if s["color"] == roles["guide"])
            guide_widths.add(guide["width"])
            others = [s for s in info["stairs"] if s["color"] != roles["guide"]]
            other_heights.update(s["height"] for s in others)
            opposite_stairs_on_target_rows += any(
                s["side"] != guide["side"]
                and bool(s["rows"] & {y for y, _ in info["target"]})
                for s in others)
            associated_color_decoys += any(
                s["height"] == 4 and s["side"] == guide["side"]
                and _associated(s, info["core"]) for s in others)
            repairables = [p for p in info["carrier_parts"] if _repair(p)]
            repairable_counts.add(len(repairables))
            body_sizes.add((len({y for y, _ in info["core"]}),
                            len({x for _, x in info["core"]})))
            target = info["target"]
            target_min_y, _, target_min_x, _ = _bbox(target)
            normalized_target = {(y-target_min_y, x-target_min_x) for y, x in target}
            competitors = [p for p in repairables if p != target]
            identical_competitors += any(
                {(y-_bbox(p)[0], x-_bbox(p)[2]) for y, x in p}
                == normalized_target for p in competitors)
            smaller_targets += any(len(p) > len(target) for p in competitors)
            grid = pair["input"]
            border = set((0, x) for x in range(grid.shape[1])) | set(
                (grid.shape[0]-1, x) for x in range(grid.shape[1])) | set(
                (y, 0) for y in range(grid.shape[0])) | set(
                (y, grid.shape[1]-1) for y in range(grid.shape[0]))
            edge_pairs += any(int(grid[y, x]) != info["background"] for y, x in border)
    assert side_modes == {1, 2}
    assert guide_widths == {4, 5, 6, 7, 8}
    assert {3, 4, 5} <= other_heights
    assert repairable_counts == {2, 3, 4}
    assert len(body_sizes) >= 10
    assert identical_competitors >= 25 and smaller_targets >= 20
    assert edge_pairs >= 80 and opposite_stairs_on_target_rows >= 25
    assert associated_color_decoys >= 10
    assert varying_backgrounds >= 25 and varying_bodies >= 30
    assert varying_counts >= 20 and held_out_combinations >= 20
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(5000+i, nt, ne)
        assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for i, (nt, ne) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(6000+i, nt, ne)
        assert nt is None or len(p["train"]) == nt
        assert ne is None or len(p["test"]) == ne
    fixed = generate_puzzle(918273, 7, 3)
    assert _equal(fixed, generate_puzzle(918273, 7, 3))
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(7000, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal count was accepted")
