"""Synthetic ARC puzzles: group shapes by color and repack them by left edge."""

import numpy as np
from collections import Counter


_SOURCE_ROWS = (
    ("0000000000", "8800004000", "8000444400", "8880004000", "0000000000", "0000044440", "0008800000", "0088880000", "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0333300333", "0330000000", "0330444000", "0000044000", "0000040040", "0000000040", "0000000440", "0000000000"),
    ("0000000000", "0770009990", "7777009990", "0770000000", "0000000000", "0099999000", "0000000000", "0000007770", "0000007770", "0000000000"),
    ("0001000000", "0011100000", "0000006000", "6600006600", "6600000000", "0011100000", "0111100000", "0000000000", "0000000000", "0000000000"),
    ("0000000000", "2222003330", "0002000000", "0022000000", "0000000000", "0003300000", "0333302220", "0003002220", "0000000000", "0000000000"),
)
_SOURCE_INPUTS = tuple(np.array([[int(v) for v in row] for row in rows], dtype=np.int8) for rows in _SOURCE_ROWS)


def _check(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _key(grid):
    return (grid.shape, grid.tobytes())


def _crop(mask):
    rows, cols = np.nonzero(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for start_r, start_c in np.argwhere(grid == color):
        start_r, start_c = int(start_r), int(start_c)
        if seen[start_r, start_c]:
            continue
        todo = [(start_r, start_c)]
        seen[start_r, start_c] = True
        cells = []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and grid[rr, cc] == color:
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells)
        c1 = max(c for _, c in cells)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        result.append((r0, c0, c1, mask))
    return result


def _description(inp):
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    foreground = [int(c) for c in colors if int(c) != background]
    if len(foreground) < 2 or len(foreground) > 4:
        raise ValueError("Need two to four foreground groups")
    groups = {}
    for color in foreground:
        objects = _components(inp, color)
        if not 1 <= len(objects) <= 3:
            raise ValueError("Need one to three separated objects per group")
        lefts = [obj[1] for obj in objects]
        if len(set(lefts)) != len(lefts):
            raise ValueError("Object left edges must be unique within a group")
        groups[color] = sorted(objects, key=lambda obj: obj[1])
    group_lefts = [groups[c][0][1] for c in foreground]
    if len(set(group_lefts)) != len(group_lefts):
        raise ValueError("Group left edges must be unique")
    order = sorted(foreground, key=lambda c: groups[c][0][1])
    packed_width = sum(max(obj[3].shape[1] for obj in groups[c]) for c in order) + len(order) - 1
    if packed_width > inp.shape[1]:
        raise ValueError("Packed groups do not fit horizontally")
    for color in order:
        packed_height = sum(obj[3].shape[0] for obj in groups[color]) + len(groups[color]) - 1
        if packed_height > inp.shape[0]:
            raise ValueError("Packed group does not fit vertically")
    return background, groups, order


def _solve(inp):
    background, groups, order = _description(inp)
    out = np.full(inp.shape, background, dtype=np.int8)
    col = 0
    for color in order:
        row = inp.shape[0]
        for _, _, _, mask in groups[color]:
            mh, mw = mask.shape
            row -= mh
            patch = out[row:row + mh, col:col + mw]
            if np.any(patch[mask] != background):
                raise ValueError("Packed objects overlap")
            patch[mask] = color
            row -= 1
        col += max(obj[3].shape[1] for obj in groups[color]) + 1
    return out


def _category(mask):
    if 1 in mask.shape:
        return "bar"
    if np.all(mask):
        return "rectangle"
    return "irregular"


def _witness(groups):
    for objects in groups.values():
        for earlier, later in zip(objects, objects[1:]):
            if earlier[1] < later[1] <= earlier[2] and earlier[2] > later[2]:
                if not (earlier[3].shape == later[3].shape and np.array_equal(earlier[3], later[3])):
                    return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Puzzle keys")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "Pair lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "Pair counts")
    seen = set()
    train_info = []
    test_info = []
    for split, target in (("train", train_info), ("test", test_info)):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair keys")
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "Grid array")
                _check(np.issubdtype(grid.dtype, np.integer), "Grid dtype")
                _check(all(1 <= size <= 30 for size in grid.shape), "Grid size")
                _check(np.all((grid >= 0) & (grid <= 9)), "Grid colors")
            inp, out = pair["input"], pair["output"]
            _check(inp.shape == out.shape and not np.array_equal(inp, out), "Changed same-size grid")
            _check(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            _check(not any(inp.shape == s.shape and np.array_equal(inp, s) for s in _SOURCE_INPUTS), "Source copy")
            background, groups, order = _description(inp)
            _check(int(np.count_nonzero(inp == background)) > max(int(np.count_nonzero(inp == c)) for c in groups), "Ambiguous background")
            _check(np.array_equal(_solve(inp), out), "Incorrect packing")
            target.append((background, groups, order))
    _check(any(_witness(groups) for _, groups, _ in train_info), "Missing left-edge witness in training")
    _check(train_info[0][2][0] in train_info[1][2][1:], "Training must reassign a familiar color's stack role")
    train_max_groups = max(len(groups) for _, groups, _ in train_info)
    train_max_objects = max(len(objects) for _, groups, _ in train_info for objects in groups.values())
    for _, groups, _ in test_info:
        _check(len(groups) <= train_max_groups, "Test group count undemonstrated")
        _check(max(map(len, groups.values())) <= train_max_objects, "Test object count undemonstrated")


def _count(value, lower, upper, name):
    if value is None:
        return None
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Invalid " + name)
    return value


def _counts(rng, num_train, num_test):
    nt = _count(num_train, 2, 10, "num_train")
    ns = _count(num_test, 1, 4, "num_test")
    if nt is not None and ns is not None and nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if nt is None and ns is None:
        nt = int(rng.integers(2, 11))
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    elif nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    return nt, ns


def _shape(rng, category, style, extended):
    cap = (5, 7, 10)[style]
    if extended:
        cap = min(cap, 6)
    if category == "bar":
        length = int(rng.integers(2, cap + 1))
        mask = np.ones((1, length), dtype=bool)
    elif category == "rectangle":
        h = int(rng.integers(2, min(cap, 7) + 1))
        w = int(rng.integers(2, min(cap, 8) + 1))
        mask = np.ones((h, w), dtype=bool)
    elif rng.random() < 0.16 and cap >= 4:
        h = int(rng.integers(3, min(cap, 6) + 1))
        w = int(rng.integers(3, min(cap, 7) + 1))
        mask = np.ones((h, w), dtype=bool)
        mask[1:h - 1, 1:w - 1] = False
    else:
        h = int(rng.integers(2, cap + 1))
        w = int(rng.integers(2, cap + 1))
        target = int(rng.integers(max(4, h + w - 2), min(h * w, 2 * (h + w)) + 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
        for _ in range(target - 1):
            frontier = set()
            for r, c in np.argwhere(mask):
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        frontier.add((rr, cc))
            if not frontier:
                break
            options = sorted(frontier)
            rr, cc = options[int(rng.integers(len(options)))]
            mask[rr, cc] = True
        mask = _crop(mask)
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return np.ascontiguousarray(mask)


def _shapes(rng, counts, style, witness):
    extended = len(counts) > 2 or max(counts) > 2
    if rng.random() < 0.22:
        one_category = ("bar", "rectangle", "irregular")[int(rng.integers(3))]
    else:
        one_category = None
    groups = []
    for count in counts:
        group = []
        for _ in range(count):
            category = one_category or ("bar", "rectangle", "irregular")[int(rng.choice(3, p=(0.22, 0.31, 0.47)))]
            group.append(_shape(rng, category, style, extended))
        groups.append(group)
    if witness:
        if rng.random() < 0.2:
            wide = np.ones((6, 6), dtype=bool)
            wide[1:5, 1:5] = False
            narrow = np.ones((2, 1), dtype=bool)
        else:
            width = int(rng.integers(5, min(9, (5, 7, 10)[style] + 3)))
            variant = int(rng.integers(3))
            if variant == 0:
                wide = np.ones((1, width), dtype=bool)
            elif variant == 1:
                wide = np.ones((int(rng.integers(2, 5)), width), dtype=bool)
            else:
                wide = np.zeros((3, width), dtype=bool)
                wide[1, :] = True
                wide[0, :int(rng.integers(2, 4))] = True
                wide[2, width - int(rng.integers(2, 4)):] = True
            if wide.shape[0] >= 2 and rng.random() < 0.4:
                wide[0, int(rng.integers(1, width - 1))] = False
            narrow = np.ones((int(rng.integers(2, 5)), int(rng.integers(1, 3))), dtype=bool)
        groups[0] = [wide, narrow]
    for group_id, group in enumerate(groups):
        if len(group) >= 2 and rng.random() < 0.09 and (not witness or group_id != 0):
            group[-1] = group[0].copy()
    return groups


def _dimensions(rng, groups, style):
    min_h = max(sum(shape.shape[0] for shape in group) + len(group) - 1 for group in groups)
    min_w = sum(max(shape.shape[1] for shape in group) for group in groups) + len(groups) - 1
    if min_h > 30 or min_w > 30:
        return None
    dims = []
    for minimum in (min_h, min_w):
        probabilities = ((0.48, 0.52, 0.0), (0.32, 0.48, 0.20), (0.20, 0.42, 0.38))[style]
        regime = int(rng.choice(3, p=probabilities))
        slack = (int(rng.integers(0, 3)), int(rng.integers(2, 7 if style == 0 else 9)), int(rng.integers(9, 22)))[regime]
        size = min(30, minimum + slack)
        if style > 0 and rng.random() < 0.055:
            size = 30
        dims.append(size)
    if style > 0 and rng.random() < 0.035:
        dims = [30, 30]
    area = sum(int(mask.sum()) for group in groups for mask in group)
    while dims[0] * dims[1] < area * 1.7:
        grow = int(rng.integers(2))
        if dims[grow] == 30:
            grow = 1 - grow
        if dims[grow] == 30:
            return None
        dims[grow] += 1
    return tuple(dims)


def _can_put(board, group_id, mask, row, col):
    rr, cc = np.nonzero(mask)
    rr = rr + row
    cc = cc + col
    if np.any(board[rr, cc] != -1):
        return False
    h, w = board.shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = rr + dr, cc + dc
        valid = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        if np.any(board[nr[valid], nc[valid]] == group_id):
            return False
    return True


def _place(rng, groups, dims, witness):
    h, w = dims
    for _ in range(20):
        board = np.full((h, w), -1, dtype=np.int8)
        lefts = [[] for _ in groups]
        tops = [[] for _ in groups]
        objects = [(g, mask) for g, group in enumerate(groups) for mask in group]
        if witness:
            wide, narrow = groups[0]
            base = int(rng.integers(0, min(3, w - wide.shape[1]) + 1))
            if wide.shape == (6, 6):
                narrow_col = base + 2
                a = int(rng.integers(h - wide.shape[0] + 1))
                b = a + 2
            else:
                narrow_col = base + int(rng.integers(1, min(3, wide.shape[1] - narrow.shape[1] - 1) + 1))
                row_options = [(a, b) for a in range(h - wide.shape[0] + 1) for b in range(h - narrow.shape[0] + 1)
                               if a + wide.shape[0] < b or b + narrow.shape[0] < a]
                if not row_options:
                    return None
                a, b = row_options[int(rng.integers(len(row_options)))]
            for mask, row, col in ((wide, a, base), (narrow, b, narrow_col)):
                rr, cc = np.nonzero(mask)
                board[rr + row, cc + col] = 0
                lefts[0].append(col)
                tops[0].append(row)
            objects = objects[2:]
        else:
            rng.shuffle(objects)
        failed = False
        for group_id, mask in objects:
            mh, mw = mask.shape
            nc = w - mw + 1
            total = (h - mh + 1) * nc
            candidates = rng.permutation(total)
            if tops[group_id] and rng.random() < 0.22:
                same_row = candidates // nc == tops[group_id][0]
                candidates = np.concatenate((candidates[same_row], candidates[~same_row]))
            placed = False
            for index in candidates[:min(total, 320)]:
                row, col = divmod(int(index), nc)
                if col in lefts[group_id]:
                    continue
                if _can_put(board, group_id, mask, row, col):
                    rr, cc = np.nonzero(mask)
                    board[rr + row, cc + col] = group_id
                    lefts[group_id].append(col)
                    tops[group_id].append(row)
                    placed = True
                    break
            if not placed:
                failed = True
                break
        if failed or len({min(group) for group in lefts}) != len(groups):
            continue
        return board
    return None


def _pair(rng, counts, style, colors, background, witness=False, forced_dims=None):
    for _ in range(80):
        shapes = _shapes(rng, counts, style, witness)
        if forced_dims is None:
            dims = _dimensions(rng, shapes, style)
        else:
            min_h = max(sum(shape.shape[0] for shape in group) + len(group) - 1 for group in shapes)
            min_w = sum(max(shape.shape[1] for shape in group) for group in shapes) + len(shapes) - 1
            area = sum(int(shape.sum()) for group in shapes for shape in group)
            dims = forced_dims if min_h <= forced_dims[0] and min_w <= forced_dims[1] and forced_dims[0] * forced_dims[1] >= area * 1.7 else None
        if dims is None:
            continue
        board = _place(rng, shapes, dims, witness)
        if board is None:
            continue
        minima = [int(np.nonzero(board == g)[1].min()) for g in range(len(shapes))]
        order = sorted(range(len(shapes)), key=lambda g: minima[g])
        if len(set(minima)) != len(minima):
            continue
        actual_colors = [None] * len(shapes)
        for g, color in zip(order, colors):
            actual_colors[g] = color
        inp = np.full(dims, background, dtype=np.int8)
        for g, color in enumerate(actual_colors):
            inp[board == g] = color
        if np.count_nonzero(inp == background) <= max(np.count_nonzero(inp == c) for c in colors):
            continue
        try:
            out = _solve(inp)
        except ValueError:
            continue
        if np.array_equal(inp, out):
            continue
        if any(inp.shape == s.shape and np.array_equal(inp, s) for s in _SOURCE_INPUTS):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("Could not construct a valid pair")


def _counts_for_pair(rng, extended, max_groups, split, index):
    if split == "train" and index == 0:
        return [2, 2]
    if extended and split == "train" and index == 1:
        result = [1, 2, 3] if max_groups == 3 else [1, 1, 2, 3]
        rng.shuffle(result)
        return result
    if not extended:
        return [2, 2]
    groups = int(rng.integers(2, max_groups + 1))
    if groups == 2:
        return [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
    result = [int(rng.integers(1, 4)) for _ in range(groups)]
    if split == "train" and max(result) == min(result):
        result[0] = 1 if result[0] != 1 else 3
    return result


def _choose_colors(rng, amount, background, familiar, mode, palette):
    possible = [c for c in palette if c != background]
    known = [c for c in possible if c in familiar]
    new = [c for c in possible if c not in familiar]
    chosen = []
    if mode == "familiar" and len(known) >= amount:
        chosen = list(rng.choice(known, size=amount, replace=False))
    elif mode == "mixed" and known and new and amount >= 2:
        chosen = [int(rng.choice(new)), int(rng.choice(known))]
    remaining = [c for c in possible if c not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[:amount - len(chosen)])
    rng.shuffle(chosen)
    return [int(c) for c in chosen]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    style = int(rng.choice(3, p=(0.32, 0.42, 0.26)))
    extended = bool(rng.random() < 0.48)
    max_groups = (4 if rng.random() < 0.16 else 3) if extended else 2
    palette_size = max(max_groups + 2, (4, 7, 10)[style])
    palette = [int(c) for c in rng.choice(10, size=palette_size, replace=False)]
    train_palette = palette[:-(1 if style == 0 else 2)]
    size_mode = ("same", "mixed", "free")[int(rng.choice(3, p=(0.25, 0.30, 0.45)))]
    if size_mode == "same":
        low = 10 + (0 if max_groups == 2 else 7 if max_groups == 3 else 11) + 2 * style
        shared_dims = (int(rng.integers(low, min(30, low + 5))), int(rng.integers(low, min(30, low + 5))))
    else:
        shared_dims = None
    base_background = int(rng.choice(train_palette))
    base_colors = _choose_colors(rng, 2, base_background, set(), "free", train_palette)
    a, b = base_colors
    puzzle = {"train": [], "test": []}
    familiar = set()
    used_inputs = set()
    for split, total in (("train", nt), ("test", ns)):
        for index in range(total):
            counts = _counts_for_pair(rng, extended, max_groups, split, index)
            forced_dims = shared_dims
            if size_mode == "mixed" and split == "test" and index % 2 == 0:
                forced_dims = puzzle["train"][1]["input"].shape
            for _ in range(25):
                if split == "train" and index == 0:
                    colors = [a, b]
                    background = base_background
                elif split == "train" and index == 1:
                    background = int(rng.choice([c for c in train_palette if c != base_background and c != a]))
                    first = int(rng.choice([c for c in train_palette if c not in (background, a)]))
                    colors = [first, a]
                    extra = _choose_colors(rng, len(counts), background, familiar | {first, a}, "free", train_palette)
                    colors.extend(c for c in extra if c not in colors)
                    colors = colors[:len(counts)]
                else:
                    background = int(rng.choice(train_palette))
                    mode = "free" if split == "train" else ("familiar", "mixed", "free")[int(rng.choice(3, p=(0.38, 0.38, 0.24)))]
                    colors = _choose_colors(rng, len(counts), background, familiar, mode, train_palette if split == "train" else palette)
                try:
                    pair = _pair(rng, counts, style, colors, background, witness=(split == "train" and index == 0), forced_dims=forced_dims)
                except RuntimeError:
                    continue
                key = _key(pair["input"])
                if key not in used_inputs:
                    used_inputs.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        familiar.update(colors)
                    break
            else:
                raise RuntimeError("Could not make a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _independent_prediction(inp):
    height, width = inp.shape
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    groups = {}
    for color in colors:
        color = int(color)
        if color == background:
            continue
        remaining = {(int(r), int(c)) for r, c in np.argwhere(inp == color)}
        objects = []
        while remaining:
            todo = [remaining.pop()]
            cells = []
            while todo:
                r, c = todo.pop()
                cells.append((r, c))
                for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if point in remaining:
                        remaining.remove(point)
                        todo.append(point)
            top = min(r for r, _ in cells)
            left = min(c for _, c in cells)
            object_width = max(c for _, c in cells) - left + 1
            object_height = max(r for r, _ in cells) - top + 1
            points = [(r - top, c - left) for r, c in cells]
            objects.append((left, object_width, object_height, points))
        groups[color] = objects
    out = np.full((height, width), background, dtype=np.int8)
    column = 0
    for color in sorted(groups, key=lambda value: min(obj[0] for obj in groups[value])):
        row = height
        for left, object_width, object_height, points in sorted(groups[color], key=lambda obj: obj[0]):
            row -= object_height
            for dr, dc in points:
                out[row + dr, column + dc] = color
            row -= 1
        column += max(obj[1] for obj in groups[color]) + 1
    return out


def _has_hole(mask):
    height, width = mask.shape
    remaining = {(r, c) for r in range(height) for c in range(width) if not mask[r, c]}
    todo = [(r, c) for r, c in remaining if r in (0, height - 1) or c in (0, width - 1)]
    while todo:
        point = todo.pop()
        if point not in remaining:
            continue
        remaining.remove(point)
        r, c = point
        todo.extend(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
    return bool(remaining)


def _review_self_check(puzzles):
    measured = Counter()
    masks = set()
    for puzzle in puzzles:
        train_info = [_description(pair["input"]) for pair in puzzle["train"]]
        train_colors = set().union(*(set(groups) for _, groups, _ in train_info))
        train_sizes = {pair["input"].shape for pair in puzzle["train"]}
        puzzle_backgrounds = set()
        all_sizes = set()
        test_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                background, groups, order = _description(inp)
                puzzle_backgrounds.add(background)
                all_sizes.add(inp.shape)
                measured["pairs"] += 1
                _check(np.array_equal(_independent_prediction(inp), pair["output"]), "Independent reconstruction")
                heights = [sum(obj[3].shape[0] for obj in groups[color]) + len(groups[color]) - 1 for color in order]
                widths = [max(obj[3].shape[1] for obj in groups[color]) for color in order]
                packed_width = sum(widths) + len(widths) - 1
                measured["tight"] += packed_width == inp.shape[1] or max(heights) == inp.shape[0]
                measured["spacious"] += inp.shape[1] - packed_width >= 8 or inp.shape[0] - max(heights) >= 8
                measured["small"] += min(inp.shape) <= 8
                measured["large"] += max(inp.shape) >= 25
                measured["thirty"] += inp.shape == (30, 30)
                measured["zero_foreground"] += 0 in groups
                measured["three_groups"] += len(groups) >= 3
                measured["four_groups"] += len(groups) == 4
                measured["original_counts"] += len(groups) == 2 and all(len(v) == 2 for v in groups.values())
                measured["unequal_counts"] += len({len(v) for v in groups.values()}) > 1
                measured["upper_width"] += any(len(groups[c]) > 1 and max(obj[3].shape[1] for obj in groups[c][1:]) > groups[c][0][3].shape[1] for c in order)
                measured["left_taller"] += heights[0] > heights[-1]
                measured["right_taller"] += heights[-1] > heights[0]
                categories = []
                group_categories = []
                large_irregular = False
                for color in order:
                    local = []
                    objects = groups[color]
                    for index, obj in enumerate(objects):
                        mask = obj[3]
                        category = _category(mask)
                        local.append(category)
                        categories.append(category)
                        masks.add((mask.shape, mask.tobytes()))
                        measured["long_bar"] += category == "bar" and max(mask.shape) >= 5
                        measured["large_rectangle"] += category == "rectangle" and max(mask.shape) >= 5
                        is_large_irregular = category == "irregular" and (max(mask.shape) >= 5 or int(mask.sum()) > 9)
                        measured["large_irregular"] += is_large_irregular
                        large_irregular |= is_large_irregular
                        measured["hole"] += _has_hole(mask)
                        for other in objects[index + 1:]:
                            measured["identical"] += mask.shape == other[3].shape and np.array_equal(mask, other[3])
                            measured["bbox_overlap"] += max(obj[0], other[0]) <= min(obj[0] + mask.shape[0] - 1, other[0] + other[3].shape[0] - 1) and max(obj[1], other[1]) <= min(obj[2], other[2])
                    group_categories.append(tuple(sorted(local)))
                measured["uneven_categories"] += len(set(group_categories)) > 1
                measured[f"{split}_all_irregular"] += set(categories) == {"irregular"}
                measured[f"{split}_all_rectangle"] += set(categories) == {"rectangle"}
                measured["combined_axes"] += background != train_info[0][0] and len(groups) >= 3 and large_irregular
                if len(order) >= 2 and all(len(groups[c]) >= 2 for c in order[:2]):
                    a = np.sign(groups[order[0]][0][0] - groups[order[0]][1][0])
                    b = np.sign(groups[order[1]][0][0] - groups[order[1]][1][0])
                    measured[f"{split}_equal_rows"] += a == 0 or b == 0
                    measured[f"{split}_cross_up_down"] += a < 0 and b > 0
                    measured[f"{split}_cross_down_up"] += a > 0 and b < 0
                    measured[f"{split}_uncrossed"] += a * b > 0
                if split == "test":
                    test_palettes.add(tuple(sorted(groups)))
                    new = [color for color in order if color not in train_colors]
                    measured["test_familiar"] += len(new) == 0
                    measured["test_new_left"] += bool(new) and order[0] in new
                    measured["test_new_right"] += bool(new) and order[-1] in new
                    measured["heldout_size"] += inp.shape not in train_sizes
                    measured["reused_size"] += inp.shape in train_sizes
        measured["backgrounds_within"] += len(puzzle_backgrounds) > 1
        measured["shared_size_puzzle"] += len(all_sizes) == 1
        measured["varied_size_puzzle"] += len(all_sizes) > 1
        measured["test_palettes_within"] += len(test_palettes) > 1
    _check(measured["pairs"] >= 100 and len(masks) >= 150, "Mask diversity")
    for name in ("tight", "spacious", "small", "large", "thirty", "zero_foreground", "three_groups", "four_groups", "original_counts", "unequal_counts", "upper_width", "left_taller", "right_taller", "long_bar", "large_rectangle", "large_irregular", "hole", "identical", "bbox_overlap", "uneven_categories", "combined_axes", "train_all_irregular", "test_all_irregular", "train_all_rectangle", "test_all_rectangle", "train_equal_rows", "test_equal_rows", "train_cross_up_down", "train_cross_down_up", "test_cross_up_down", "test_cross_down_up", "train_uncrossed", "test_uncrossed", "test_familiar", "test_new_left", "test_new_right", "heldout_size", "reused_size", "shared_size_puzzle", "varied_size_puzzle", "test_palettes_within"):
        _check(measured[name] > 0, "Missing review feature: " + name)
    _check(measured["backgrounds_within"] == len(puzzles), "Backgrounds must vary within puzzles")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(all(_equal(p, generate_puzzle(seed)) for seed, p in enumerate(puzzles[:4])), "Determinism")
    _check(len({_key(p["train"][0]["input"]) for p in puzzles}) >= 38, "Cross-seed diversity")
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "Train count diversity")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "Test count diversity")
    features = {"backgrounds": set(), "dimensions": set(), "categories": set(), "counts": set(), "large": False,
                "three_groups": False, "zero_foreground": False, "all_irregular": False, "all_rectangle": False}
    for p in puzzles:
        validate_generated_puzzle(p)
        for pair in p["train"] + p["test"]:
            inp = pair["input"]
            background, groups, order = _description(inp)
            features["backgrounds"].add(background)
            features["dimensions"].add(inp.shape)
            cats = [_category(obj[3]) for objects in groups.values() for obj in objects]
            features["categories"].update(cats)
            features["counts"].add(tuple(sorted(map(len, groups.values()))))
            features["large"] |= max(inp.shape) >= 25
            features["three_groups"] |= len(groups) >= 3
            features["zero_foreground"] |= 0 in groups
            features["all_irregular"] |= set(cats) == {"irregular"}
            features["all_rectangle"] |= set(cats) == {"rectangle"}
            _check(np.array_equal(_solve(inp), pair["output"]), "Transformation")
    _check(len(features["backgrounds"]) >= 8 and len(features["dimensions"]) >= 35, "Canvas diversity")
    _check(features["categories"] == {"bar", "rectangle", "irregular"}, "Shape categories")
    _check(len(features["counts"]) >= 4 and all(features[k] for k in ("large", "three_groups", "zero_foreground", "all_irregular", "all_rectangle")), "Review diversity")
    _review_self_check(puzzles)
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(777 + (nt or 0) + (ns or 0), nt, ns)
        _check((nt is None or len(p["train"]) == nt) and (ns is None or len(p["test"]) == ns), "Requested counts")
    for nt, ns in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(7, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
