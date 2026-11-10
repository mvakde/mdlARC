"""Compositional open-contour containment puzzles."""

import numpy as np


_SOURCE = (
    (("101", "171", "111"), ("101", "101", "111")),
    (("00000000000", "00022222000", "02205500000", "02205500000", "00222220000", "00000000000", "00000066600", "00006680000", "00006008000", "00006666600", "00000000000"),
     ("00000000000", "00022222000", "02200000000", "02200000000", "00222220000", "00000000000", "00000066600", "00006600000", "00006000000", "00006666600", "00000000000")),
    (("000006660000000", "000300600000000", "003000000000000", "030600000000000", "030660300000000", "003003000000000", "000330000770000", "000000007807000", "000000070880700", "000000000007000", "000008000070000", "000000800700000", "000000080000000", "000000000000000", "000000000000000"),
     ("000006660000000", "000300600000000", "003000000000000", "030000000000000", "030000300000000", "003003000000000", "000330000770000", "000000007007000", "000000070000700", "000000000007000", "000008000070000", "000000800700000", "000000080000000", "000000000000000", "000000000000000")),
    (("00000000000", "00022000000", "02200000000", "02205000000", "00222000000", "00000000000", "00000066600", "00006600009", "00006000000", "00006666600", "00000000000"),
     ("00000000000", "00022000000", "02200000000", "02200000000", "00222000000", "00000000000", "00000066600", "00006600009", "00006000000", "00006666600", "00000000000")),
    (("700000000000", "000033333330", "700330000030", "000300770033", "000380077033", "000388000033", "000300000030", "090000000000", "000660007000", "906096000000", "006990607007", "900600000000", "090060000007"),
     ("700000000000", "000033333330", "700330000030", "000300000033", "000300000033", "000300000033", "000300000030", "090000000000", "000660007000", "906006000000", "006000607007", "900600000000", "090060000007")),
)


def _counts(seed, num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
            raise ValueError(name)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = int(num_test) + 1 + (seed // 4) % (10 - int(num_test))
    elif num_train is None:
        num_train = int(num_test) + 1 + seed % (10 - int(num_test))
    elif num_test is None:
        num_test = 1 + seed % min(4, int(num_train) - 1)
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _source_copy(pair):
    return (_key(pair["input"]), _key(pair["output"])) in _SOURCE


def _shape(rng, compact=False):
    """A two-lipped, one-bottom contour, expressed before rotation."""
    h = 3 if compact else int(rng.integers(3, 10))
    mouth = int(rng.integers(1, 3 if compact else 8))
    straight = compact or rng.random() < .22
    left = [h + 2]
    right = [left[0] + mouth + 1]
    for _ in range(1, h - 1):
        dl = 0 if straight else int(rng.choice((-1, 0, 0, 1)))
        dr = 0 if straight else int(rng.choice((-1, 0, 0, 1)))
        a, b = left[-1] + dl, right[-1] + dr
        if b - a < 2 or b - a > 11:
            a, b = left[-1], right[-1]
        left.append(a)
        right.append(b)
    width = max(right) + h + 4
    wall = np.zeros((h, width), dtype=bool)
    for r, (a, b) in enumerate(zip(left, right)):
        gap = b - a - 1
        ta = 1 + (not compact and gap >= 3 and rng.random() < .27)
        tb = 1 + (not compact and gap >= 4 and rng.random() < .27)
        wall[r, a:a + ta] = True
        wall[r, b - tb + 1:b + 1] = True
    wall[-1, left[-1]:right[-1] + 1] = True
    cols = np.flatnonzero(wall.any(axis=0))
    wall = wall[:, cols[0]:cols[-1] + 1]
    if not compact and not straight and rng.random() < .27:
        extension = int(rng.integers(1, 3))
        side = int(rng.integers(2))
        lip_columns = np.flatnonzero(wall[0])
        lip = int(lip_columns[0] if side == 0 else lip_columns[-1])
        extra = np.zeros((extension, wall.shape[1]), dtype=bool)
        extra[:, lip] = True
        wall = np.vstack((extra, wall))
    return wall


def _cavity(wall):
    cavity = np.zeros_like(wall)
    for r in range(wall.shape[0] - 1):
        cols = np.flatnonzero(wall[r])
        openings = np.flatnonzero(~wall[r, cols[0]:cols[-1] + 1])
        if len(openings):
            cavity[r, cols[0] + openings] = True
    return cavity


def _recognize(component):
    """Infer a hollow from visible wall cells, independent of generation records."""
    found = []
    for turns in range(4):
        a = np.rot90(component, turns)
        h, w = a.shape
        if h < 3 or w < 3 or not a[0].any() or not a[-1].any():
            continue
        sides = []
        prefix = []
        started = False
        okay = True
        for r in range(h - 1):
            cols = np.flatnonzero(a[r])
            if not len(cols):
                okay = False
                break
            lo, hi = int(cols[0]), int(cols[-1])
            middle = a[r, lo:hi + 1]
            gaps = np.flatnonzero(~middle)
            if not len(gaps) and not started and r < 2 and len(cols) <= 2:
                prefix.append((lo, hi))
                continue
            if not len(gaps) or np.any(middle[gaps[0]:gaps[-1] + 1]):
                okay = False
                break
            started = True
            if gaps[0] > 2 or len(middle) - 1 - gaps[-1] > 2:
                okay = False
                break
            sides.append((lo, hi))
        if not okay or not sides or (prefix and min(abs(prefix[-1][0] - sides[0][0]), abs(prefix[-1][1] - sides[0][1])) > 1):
            continue
        bottom = np.flatnonzero(a[-1])
        if len(bottom) < 3 or len(bottom) != bottom[-1] - bottom[0] + 1:
            continue
        if bottom[0] != sides[-1][0] or bottom[-1] != sides[-1][1]:
            continue
        if any(abs(x[0] - y[0]) > 1 or abs(x[1] - y[1]) > 1 for x, y in zip(sides, sides[1:])):
            continue
        cavity = np.rot90(_cavity(a), -turns)
        if cavity.any():
            found.append((turns, cavity))
    return found[0] if len(found) == 1 else None


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        group = {remaining.pop()}
        todo = list(group)
        while todo:
            r, c = todo.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    point = (r + dr, c + dc)
                    if point in remaining:
                        remaining.remove(point)
                        group.add(point)
                        todo.append(point)
        result.append(group)
    return result


def _analyze(inp, out):
    changed = inp != out
    assert changed.any() and inp.shape == out.shape
    backgrounds = np.unique(out[changed])
    assert len(backgrounds) == 1
    bg = int(backgrounds[0])
    assert np.all(inp[changed] != bg)
    walls = np.zeros(inp.shape, dtype=bool)
    hollows = np.zeros(inp.shape, dtype=bool)
    containers = []
    for color in np.unique(out):
        if int(color) == bg:
            continue
        for comp in _components(out == color):
            rr = [p[0] for p in comp]
            cc = [p[1] for p in comp]
            top, left, bottom, right = min(rr), min(cc), max(rr) + 1, max(cc) + 1
            local = np.zeros((bottom - top, right - left), dtype=bool)
            for r, c in comp:
                local[r - top, c - left] = True
            recognized = _recognize(local)
            if recognized is None:
                continue
            direction, cavity = recognized
            assert not np.any(walls[top:bottom, left:right] & local)
            walls[top:bottom, left:right] |= local
            hollows[top:bottom, left:right] |= cavity
            containers.append((int(color), direction, (top, left, bottom, right), int(cavity.sum()), int(local.sum())))
    assert containers and not np.any(walls & hollows)
    assert np.array_equal(changed, hollows & (inp != bg))
    assert np.all(out[~changed] == inp[~changed])
    exterior = (out != bg) & ~walls
    return {"background": bg, "walls": walls, "hollows": hollows, "containers": containers, "exterior": exterior, "changed": changed}


def _review_features(pair, analysis):
    inp, out = pair["input"], pair["output"]
    walls = analysis["walls"]
    exterior = analysis["exterior"]
    wall_points = [tuple(p) for p in np.argwhere(walls)]
    exterior_points = [tuple(p) for p in np.argwhere(exterior)]
    edited_colors = set(int(v) for v in np.unique(inp[analysis["changed"]]))
    exterior_colors = set(int(v) for v in np.unique(out[exterior]))
    features = {
        "square": inp.shape[0] == inp.shape[1],
        "compact": max(inp.shape) <= 5,
        "large": max(inp.shape) >= 25,
        "one": len(analysis["containers"]) == 1,
        "many": len(analysis["containers"]) >= 3,
        "exterior_cluster": any(len(group) >= 2 for group in _components(exterior)),
        "near_exterior": any(_near_wall(p, wall_points) == 2 for p in exterior_points),
        "witness": bool(edited_colors & exterior_colors),
        "empty": False,
        "multicolor": False,
        "block": False,
        "dense": False,
        "sparse": False,
        "diagonal_payload": False,
        "separated_groups": False,
        "bent_payload": False,
        "edge": False,
        "irregular": False,
        "stepped": False,
        "thick": False,
        "unequal_arm": False,
        "shared_wall": len({v[0] for v in analysis["containers"]}) < len(analysis["containers"]),
        "wall_as_mark": any(v[0] in edited_colors | exterior_colors for v in analysis["containers"]),
        "close_boxes": False,
    }
    boxes = []
    for color, direction, box, cavity_size, wall_size in analysis["containers"]:
        top, left, bottom, right = box
        boxes.append(box)
        features["edge"] |= top == 0 or left == 0 or bottom == inp.shape[0] or right == inp.shape[1]
        local_wall = walls[top:bottom, left:right]
        local_cavity = _recognize(local_wall)
        assert local_cavity is not None
        local_cavity = local_cavity[1]
        edited = (inp[top:bottom, left:right] != out[top:bottom, left:right]) & local_cavity
        payload_colors = set(int(v) for v in np.unique(inp[top:bottom, left:right][edited]))
        features["empty"] |= not edited.any()
        features["multicolor"] |= len(payload_colors) >= 2
        features["dense"] |= int(edited.sum()) >= 7
        features["sparse"] |= int(edited.sum()) == 1
        edited_points = set(map(tuple, np.argwhere(edited)))
        features["diagonal_payload"] |= any((r + 1, c + 1) in edited_points or
                                            (r + 1, c - 1) in edited_points
                                            for r, c in edited_points)
        features["separated_groups"] |= len(_components(edited)) >= 2
        features["bent_payload"] |= any(
            ((r + 1, c) in edited_points or (r - 1, c) in edited_points) and
            ((r, c + 1) in edited_points or (r, c - 1) in edited_points)
            for r, c in edited_points)
        features["block"] |= any(edited[r:r + 2, c:c + 2].all()
                                 for r in range(edited.shape[0] - 1)
                                 for c in range(edited.shape[1] - 1))
        upright = np.rot90(local_wall, direction)
        lefts, rights, gaps = [], [], []
        for row in upright[:-1]:
            columns = np.flatnonzero(row)
            lefts.append(int(columns[0]))
            rights.append(int(columns[-1]))
            segment = row[columns[0]:columns[-1] + 1]
            opening = np.flatnonzero(~segment)
            if len(opening):
                gaps.append(len(opening))
                features["thick"] |= opening[0] > 1 or len(segment) - 1 - opening[-1] > 1
            else:
                features["unequal_arm"] = True
        features["irregular"] |= features["unequal_arm"] or len(set(lefts)) > 1 or len(set(rights)) > 1 or len(set(gaps)) > 1
        features["stepped"] |= any(x != y for x, y in zip(lefts, lefts[1:])) or any(
            x != y for x, y in zip(rights, rights[1:]))
        assert not np.any(edited[0] & local_cavity[0]) or direction != 0
    for index, a in enumerate(boxes):
        for b in boxes[index + 1:]:
            row_gap = max(0, b[0] - a[2], a[0] - b[2])
            col_gap = max(0, b[1] - a[3], a[1] - b[3])
            features["close_boxes"] |= (row_gap == 1 and col_gap == 0) or (col_gap == 1 and row_gap == 0)
    features["combined"] = features["irregular"] and features["multicolor"] and features["near_exterior"]
    return features


def _layout(rng, shapes):
    n = len(shapes)
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        rows, cols = (1, 2) if rng.random() < .5 else (2, 1)
    else:
        cols = 2 if rng.random() < .75 else n
        rows = (n + cols - 1) // cols
    cells = [(i // cols, i % cols) for i in range(n)]
    row_heights = [max(shapes[i].shape[0] for i in range(n) if cells[i][0] == r) for r in range(rows)]
    col_widths = [max(shapes[i].shape[1] for i in range(n) if cells[i][1] == c) for c in range(cols)]
    row_gaps = [int(rng.integers(1, 4)) for _ in range(rows - 1)]
    col_gaps = [int(rng.integers(1, 4)) for _ in range(cols - 1)]
    raw_h = sum(row_heights) + sum(row_gaps)
    raw_w = sum(col_widths) + sum(col_gaps)
    return cells, row_heights, col_widths, row_gaps, col_gaps, raw_h, raw_w


def _payload(rng, cavity, colors):
    pts = [tuple(int(x) for x in p) for p in np.argwhere(cavity)]
    if not pts:
        return {}
    target = int(rng.integers(1, min(len(pts), max(2, int(len(pts) * .8)), 12) + 1))
    mode = int(rng.integers(0, 5))
    selected = []
    if mode in (0, 1, 2) and len(pts) >= 4:
        starts = pts.copy()
        rng.shuffle(starts)
        for start in starts:
            r, c = start
            block = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            if mode == 0 and all(p in pts for p in block):
                selected = block
                break
            if mode == 1:
                selected = [start]
                break
            if mode == 2:
                selected = [start]
                break
    if not selected:
        selected = [pts[int(rng.integers(len(pts)))]]
    for _ in range(12 * target):
        if len(selected) >= target:
            break
        if mode == 3 or rng.random() < .2:
            p = pts[int(rng.integers(len(pts)))]
        else:
            r, c = selected[int(rng.integers(len(selected)))]
            if mode == 2:
                delta = ((-1, -1), (-1, 1), (1, -1), (1, 1))[int(rng.integers(4))]
            else:
                delta = ((-1, 0), (0, -1), (1, 0), (0, 1))[int(rng.integers(4))]
            p = (r + delta[0], c + delta[1])
        if p in pts and p not in selected:
            selected.append(p)
        elif len(selected) < target and rng.random() < .15:
            unused = [q for q in pts if q not in selected]
            if unused:
                selected.append(unused[int(rng.integers(len(unused)))])
    if len(selected) < target:
        unused = [q for q in pts if q not in selected]
        rng.shuffle(unused)
        selected.extend(unused[:target - len(selected)])
    assignments = [colors[i % len(colors)] for i in range(len(selected))]
    rng.shuffle(assignments)
    return dict(zip(selected, assignments))


def _near_wall(point, wall_positions):
    r, c = point
    return min(max(abs(r - wr), abs(c - wc)) for wr, wc in wall_positions)


def _make_pair(rng, regime, plan, stable_bg, background):
    n = plan["count"]
    shapes = []
    for i in range(n):
        compact = bool(plan["compact"] and i == 0)
        wall = _shape(rng, compact)
        shapes.append(np.rot90(wall, int(plan["directions"][i])))
    cells, heights, widths, row_gaps, col_gaps, raw_h, raw_w = _layout(rng, shapes)
    top_pad = int(rng.integers(0, 4))
    left_pad = int(rng.integers(0, 4))
    bottom_pad = int(rng.integers(0, 4))
    right_pad = int(rng.integers(0, 4))
    if plan["compact"] and n == 1:
        top_pad = left_pad = bottom_pad = right_pad = 0
    height = raw_h + top_pad + bottom_pad
    width = raw_w + left_pad + right_pad
    if not plan["compact"] and rng.random() < .28:
        side = max(height, width)
        if height < side:
            extra = int(rng.integers(side - height + 1))
            top_pad += extra
            bottom_pad += side - height - extra
        elif width < side:
            extra = int(rng.integers(side - width + 1))
            left_pad += extra
            right_pad += side - width - extra
        height = width = side
    if height > 30 or width > 30:
        return None
    bg = background if stable_bg else int(rng.integers(10))
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    all_walls = np.zeros_like(inp, dtype=bool)
    all_cavities = np.zeros_like(inp, dtype=bool)
    placements = []
    for i, wall in enumerate(shapes):
        row, col = cells[i]
        top = top_pad + sum(heights[:row]) + sum(row_gaps[:row])
        left = left_pad + sum(widths[:col]) + sum(col_gaps[:col])
        top += int(rng.integers(heights[row] - wall.shape[0] + 1))
        left += int(rng.integers(widths[col] - wall.shape[1] + 1))
        local_cavity = _cavity(np.rot90(wall, -int(plan["directions"][i])))
        local_cavity = np.rot90(local_cavity, int(plan["directions"][i]))
        sl = np.s_[top:top + wall.shape[0], left:left + wall.shape[1]]
        all_walls[sl] |= wall
        all_cavities[sl] |= local_cavity
        placements.append((sl, wall, local_cavity))
    if np.any(all_walls & all_cavities):
        return None
    wall_colors = []
    for i, (sl, wall, cavity) in enumerate(placements):
        if plan["fixed_walls"] is not None:
            color = plan["fixed_walls"][i % len(plan["fixed_walls"])]
        elif i and rng.random() < .2:
            color = wall_colors[int(rng.integers(len(wall_colors)))]
        else:
            color = int(rng.choice([c for c in range(10) if c != bg]))
        wall_colors.append(color)
        inp[sl][wall] = color
        out[sl][wall] = color
    erased = []
    for i, (sl, wall, cavity) in enumerate(placements):
        if i in plan["empty"]:
            continue
        color_count = int(rng.choice((1, 1, 1, 2, 2, 3))) if plan["multi"] else 1
        palette = [c for c in range(10) if c not in (bg, wall_colors[i])]
        rng.shuffle(palette)
        eligible = np.rot90(cavity, -int(plan["directions"][i])).copy()
        eligible[0] = False
        eligible = np.rot90(eligible, int(plan["directions"][i]))
        assignments = _payload(rng, eligible, palette[:color_count])
        top, left = sl[0].start, sl[1].start
        for (r, c), value in assignments.items():
            inp[top + r, left + c] = value
            erased.append(value)
    if not erased:
        return None
    wall_positions = [tuple(int(x) for x in p) for p in np.argwhere(all_walls)]
    available = []
    near = []
    for r in range(height):
        for c in range(width):
            if all_walls[r, c] or all_cavities[r, c]:
                continue
            distance = _near_wall((r, c), wall_positions)
            if distance >= 2:
                available.append((r, c))
                if distance == 2:
                    near.append((r, c))
    exterior_count = 0 if plan["compact"] and n == 1 else int(rng.integers(0, min(12, max(2, len(available) // 7)) + 1))
    if plan["witness"] and available:
        exterior_count = max(1, exterior_count)
    selected = []
    if exterior_count and available:
        candidates = near if near and rng.random() < .75 else available
        selected.append(candidates[int(rng.integers(len(candidates)))])
    while len(selected) < exterior_count and available:
        if selected and rng.random() < .55:
            r, c = selected[int(rng.integers(len(selected)))]
            neighbors = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))]
            candidates = [p for p in neighbors if p in available and p not in selected]
        else:
            candidates = [p for p in available if p not in selected]
        if not candidates:
            break
        selected.append(candidates[int(rng.integers(len(candidates)))])
    distractor_palette = [c for c in range(10) if c != bg]
    for j, (r, c) in enumerate(selected):
        if j == 0 and plan["witness"]:
            color = erased[int(rng.integers(len(erased)))]
        elif j and rng.random() < .5:
            color = int(inp[selected[j - 1]])
        else:
            color = int(rng.choice(distractor_palette))
        inp[r, c] = color
        out[r, c] = color
    pair = {"input": inp, "output": out}
    try:
        _analyze(inp, out)
    except AssertionError:
        return None
    return pair


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt, nv = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    stable_bg = bool(rng.random() < .4)
    background = int(rng.integers(10))
    fixed_walls = None
    if stable_bg and rng.random() < .5:
        wall_options = [c for c in range(10) if c != background]
        rng.shuffle(wall_options)
        fixed_walls = tuple(wall_options[:int(rng.integers(1, 3))])
    puzzle = {"train": [], "test": []}
    seen = set()
    train_directions = [int(x) for x in rng.permutation(4)]
    for section, count in (("train", nt), ("test", nv)):
        for index in range(count):
            if section == "train" and index == 0:
                n = 1 if rng.random() < .65 else 2
            elif section == "train" and index == 1:
                n = 2 if rng.random() < .7 else 3
            else:
                n = int(rng.choice((1, 1, 2, 2, 3, 3, 4)))
            if section == "train" and index < 2:
                directions = [train_directions[(index + j) % 4] for j in range(n)]
            else:
                directions = [int(rng.choice(train_directions)) for _ in range(n)]
            empty = set()
            if n > 1 and ((section == "train" and index == 1) or rng.random() < .23):
                empty.add(int(rng.integers(n)))
            compact = n == 1 and ((section == "train" and index == 0 and rng.random() < .55)
                                  or (section == "test" and rng.random() < .45))
            plan = {"count": n, "directions": directions, "empty": empty,
                    "compact": compact, "multi": section == "test" or index >= 1,
                    "witness": section == "train" and index == 1 or rng.random() < .45,
                    "fixed_walls": fixed_walls}
            for attempt in range(100):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(child, "source_train" if section == "train" else "target_test", plan, stable_bg, background)
                if pair is None:
                    continue
                key = _key(pair["input"])
                if key in seen or _source_copy(pair):
                    continue
                seen.add(key)
                puzzle[section].append(pair)
                break
            else:
                raise ValueError("could not construct a distinct valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
    inputs, pairs = set(), set()
    train_occupied = 0
    train_witness = False
    train_empty = False
    train_directions = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            ik = _key(pair["input"])
            pk = (ik, _key(pair["output"]))
            assert ik not in inputs and pk not in pairs and not _source_copy(pair)
            inputs.add(ik)
            pairs.add(pk)
            analysis = _analyze(pair["input"], pair["output"])
            assert analysis["containers"]
            features = _review_features(pair, analysis)
            if section == "train":
                train_occupied += bool(analysis["changed"].any())
                train_witness |= features["witness"]
                train_empty |= features["empty"]
                train_directions.update(item[1] for item in analysis["containers"])
    assert train_occupied >= 2 and train_witness and train_empty
    assert len(train_directions) >= 2


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, geometries, backgrounds, counts = set(), set(), set(), set()
    review_counts = {}
    split_counts = {"train": 0, "test": 0}
    witness_puzzles = 0
    empty_training_puzzles = 0
    stable_wall_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_witness = False
        puzzle_empty = False
        pair_wall_palettes = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                info = _analyze(inp, out)
                features = _review_features(pair, info)
                split_counts[section] += 1
                for name, present in features.items():
                    if present:
                        key = section + "_" + name
                        review_counts[key] = review_counts.get(key, 0) + 1
                if section == "train":
                    puzzle_witness |= features["witness"]
                    puzzle_empty |= features["empty"]
                pair_wall_palettes.add(frozenset(item[0] for item in info["containers"]))
                shapes.add(inp.shape)
                backgrounds.add(info["background"])
                counts.add(len(info["containers"]))
                for color, direction, box, cavity_size, wall_size in info["containers"]:
                    geometries.add((direction, box[2] - box[0], box[3] - box[1], cavity_size, wall_size))
        witness_puzzles += puzzle_witness
        empty_training_puzzles += puzzle_empty
        stable_wall_puzzles += len(pair_wall_palettes) == 1
    assert len(shapes) >= 15 and len(geometries) >= 30 and len(backgrounds) >= 6
    assert {1, 2, 3} <= counts
    assert witness_puzzles == empty_training_puzzles == 42
    assert 0 < stable_wall_puzzles < 42
    for section in ("train", "test"):
        assert 0 < review_counts.get(section + "_square", 0) < split_counts[section]
        for name in ("compact", "large", "one", "many", "exterior_cluster", "near_exterior",
                     "witness", "empty", "multicolor", "block", "dense", "sparse", "diagonal_payload",
                     "separated_groups", "bent_payload", "edge", "irregular", "stepped",
                     "thick", "unequal_arm", "shared_wall", "wall_as_mark", "close_boxes", "combined"):
            assert review_counts.get(section + "_" + name, 0) > 0, (section, name)
    a = generate_puzzle(9173, 6, 3)
    b = generate_puzzle(9173, 6, 3)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test") for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    assert not np.array_equal(a["train"][0]["input"], generate_puzzle(9174, 6, 3)["train"][0]["input"])
    for nt, nv in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(2000 + (nt or 0) * 10 + (nv or 0), nt, nv)
        assert nt is None or len(p["train"]) == nt
        assert nv is None or len(p["test"]) == nv
    for kw in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(30, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
