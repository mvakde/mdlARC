"""Generate puzzles that erase the upper and lower appendages of a figure."""

import hashlib
import json

import numpy as np


_DIRECTIONS = ("up", "down", "left", "right")
_KINDS = ("filled", "hollow", "diamond", "hourglass", "cross", "composed")
_ARM_STYLES = ("stem", "block", "flare", "neck", "diagonal", "fork", "hollow")
_SOURCE_DIGESTS = {
    "30331678cb154c4194251ac5848b630906982de53140ca19fb044e8446f0cef1",
    "e4752e0190aee469357017ddd61a12b82ed2fd503e934c93ae711a5a4888a851",
    "c7e1eaeb0668873e76b08f8ac91cd5c28a2959838d2ddae354f512df1afc4ba2",
    "3ae84aa09ad84e4c1fcf37c52426291a55853203fc528fecbccf0a6013b0a83a",
    "a2574164f197cd1180a3d473cdc6badc560af4549e56ad254781c67b19e7cc52",
    "5b1dcd40c245dfbe70f1dfd6db863489386ec127eacabbaa6c414399e62a74d5",
}


def _resolve_counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} is outside its legal range")
    if num_train is None and num_test is None:
        choices = ((2, 1), (3, 1), (3, 2), (4, 1), (4, 2),
                   (4, 3), (5, 2), (5, 3), (6, 2), (6, 3),
                   (7, 3), (8, 4), (9, 3), (10, 4))
        weights = np.array((10, 14, 14, 8, 12, 10, 8, 7, 5, 4,
                            3, 2, 2, 1), dtype=float)
        index = int(rng.choice(len(choices), p=weights / weights.sum()))
        num_train, num_test = choices[index]
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _connected8(cells):
    if not cells:
        return False
    visited = {next(iter(cells))}
    stack = list(visited)
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                point = (r + dr, c + dc)
                if point in cells and point not in visited:
                    visited.add(point)
                    stack.append(point)
    return visited == cells


def _touches8(first, second):
    return any((r + dr, c + dc) in second
               for r, c in first for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _make_figure(rng, kind):
    if kind == "filled":
        h, w = int(rng.integers(4, 9)), int(rng.integers(4, 11))
        widths = [w] * h
    elif kind == "hollow":
        h, w = int(rng.integers(7, 12)), int(rng.integers(7, 12))
        if rng.random() < 0.7:
            widths = [w - 2 * max(0, 2 - min(r, h - 1 - r)) for r in range(h)]
        else:
            widths = [w] * h
    elif kind == "diamond":
        radius = int(rng.integers(2, 6))
        h = w = 2 * radius + 1
        widths = [1 + 2 * min(r, h - 1 - r) for r in range(h)]
    elif kind == "hourglass":
        w = int(rng.integers(7, 12))
        waist = int(rng.choice((1, 2, 3)))
        cap_top = int(rng.integers(1, 4))
        cap_bottom = int(rng.integers(1, 4))
        neck = int(rng.integers(2, 5))
        widths = [w] * cap_top + [waist] * neck + [w] * cap_bottom
        h = len(widths)
    elif kind == "cross":
        h, w = int(rng.integers(5, 11)), int(rng.integers(5, 12))
        stem = int(rng.choice((1, 1, 2, 3)))
        bar = int(rng.choice((1, 1, 2)))
        start = (h - bar) // 2
        widths = [w if start <= r < start + bar else stem for r in range(h)]
    elif kind == "composed":
        w = int(rng.integers(7, 12))
        neck = int(rng.choice((1, 2, 3)))
        top = [w] * int(rng.integers(1, 3))
        bottom = [w] * int(rng.integers(1, 3))
        widths = top + [max(neck, w - 3), neck, neck,
                        max(neck, w - 2)] + bottom
        h = len(widths)
    else:
        raise ValueError("unknown figure kind")
    shifts = [0] * h
    if kind not in ("diamond", "cross", "hollow") and rng.random() < 0.38:
        shifts = [int(v) for v in rng.choice((-1, 0, 1), size=h,
                                              p=(0.25, 0.5, 0.25))]
        shifts = [0 if span <= 2 else shift
                  for span, shift in zip(widths, shifts)]
    cells = set()
    for row, span in enumerate(widths):
        left = (w - span) // 2 + shifts[row]
        right = left + span - 1
        if kind == "hollow" and row not in (0, h - 1):
            cells.update(((row, left), (row, right)))
        else:
            cells.update((row, col) for col in range(left, right + 1))
    min_col = min(c for _, c in cells)
    cells = {(r, c - min_col) for r, c in cells}
    w = max(c for _, c in cells) + 1
    return cells, h, w


def _arm_offsets(style, distance, depth, width, slope):
    if style == "stem":
        return range(0, width)
    if style == "block":
        return range(-(width // 2), (width + 1) // 2)
    if style == "flare":
        size = min(width, 1 + distance)
        return range(-(size // 2), (size + 1) // 2)
    if style == "neck":
        size = max(1, width - distance + 1)
        return range(-(size // 2), (size + 1) // 2)
    if style == "diagonal":
        return (slope * (distance - 1),)
    if style == "fork":
        offset = min(2, distance - 1)
        return (0,) if distance == 1 else (-offset, offset)
    if style == "hollow":
        radius = max(1, width // 2)
        return range(-radius, radius + 1) if distance in (1, depth) else (-radius, radius)
    raise ValueError("unknown arm style")


def _make_arm(rng, direction, center, height, width, simple):
    if direction == "up":
        boundary = [(0, c) for r, c in center if r == 0]
    elif direction == "down":
        boundary = [(height - 1, c) for r, c in center if r == height - 1]
    elif direction == "left":
        boundary = [(r, min(c for rr, c in center if rr == r))
                    for r in range(height // 3, (2 * height + 2) // 3)]
    else:
        boundary = [(r, max(c for rr, c in center if rr == r))
                    for r in range(height // 3, (2 * height + 2) // 3)]
    if not boundary:
        raise AssertionError("figure has no attachment boundary")
    for _ in range(60):
        if direction in ("left", "right") and rng.random() < 0.6:
            spans = [sum(rr == r for rr, _ in center) for r, _ in boundary]
            narrow = [p for p, span in zip(boundary, spans) if span == min(spans)]
            anchor = narrow[int(rng.integers(len(narrow)))]
        else:
            anchor = boundary[int(rng.integers(len(boundary)))]
        style = str(rng.choice(("stem", "block", "flare", "neck")
                               if simple else _ARM_STYLES))
        depth = int(rng.integers(1, 4 if simple else 6))
        if style in ("fork", "hollow"):
            depth = max(depth, 3)
        thickness = int(rng.integers(1, 4 if simple else 6))
        slope = int(rng.choice((-1, 1)))
        cells = set()
        for distance in range(1, depth + 1):
            for offset in _arm_offsets(style, distance, depth, thickness, slope):
                if direction == "up":
                    point = (-distance, anchor[1] + offset)
                elif direction == "down":
                    point = (height - 1 + distance, anchor[1] + offset)
                elif direction == "left":
                    point = (anchor[0] + offset, anchor[1] - distance)
                else:
                    point = (anchor[0] + offset, anchor[1] + distance)
                cells.add(point)
        if direction in ("left", "right") and any(
            not 0 <= r < height for r, _ in cells
        ):
            continue
        if direction == "left" and any(
            c >= min(x for rr, x in center if rr == r) for r, c in cells
        ):
            continue
        if direction == "right" and any(
            c <= max(x for rr, x in center if rr == r) for r, c in cells
        ):
            continue
        if _connected8(cells) and _touches8(cells, center):
            return cells
    anchor = boundary[len(boundary) // 2]
    if direction == "up":
        return {(-1, anchor[1])}
    if direction == "down":
        return {(height, anchor[1])}
    if direction == "left":
        return {(anchor[0], anchor[1] - 1)}
    return {(anchor[0], anchor[1] + 1)}


def _make_pair(rng, kind, background, center_color, role_colors, simple):
    center, height, width = _make_figure(rng, kind)
    assert _connected8(center)
    regions = {"center": center}
    for direction in _DIRECTIONS:
        regions[direction] = _make_arm(rng, direction, center, height, width, simple)
    all_cells = set().union(*regions.values())
    assert len(all_cells) == sum(map(len, regions.values()))
    min_row = min(r for r, _ in all_cells)
    max_row = max(r for r, _ in all_cells)
    min_col = min(c for _, c in all_cells)
    max_col = max(c for _, c in all_cells)
    margins = [int(rng.integers(0, 6)) for _ in range(4)]
    margins[0] = min(margins[0], 30 - (max_row - min_row + 1))
    margins[2] = min(margins[2], 30 - (max_col - min_col + 1))
    margins[1] = min(margins[1], 30 - (max_row - min_row + 1) - margins[0])
    margins[3] = min(margins[3], 30 - (max_col - min_col + 1) - margins[2])
    if min(margins) < 0:
        return None
    top, bottom, left, right = margins
    shape = (max_row - min_row + 1 + top + bottom,
             max_col - min_col + 1 + left + right)
    input_grid = np.full(shape, background, dtype=np.int8)
    output_grid = np.full(shape, background, dtype=np.int8)
    for role, cells in regions.items():
        color = center_color if role == "center" else role_colors[role]
        for row, col in cells:
            rr, cc = row - min_row + top, col - min_col + left
            input_grid[rr, cc] = color
            if role not in ("up", "down"):
                output_grid[rr, cc] = color
    if len({int(input_grid[r, c]) for r in (0, shape[0] - 1)
            for c in (0, shape[1] - 1)}) != 1:
        return None
    return {"input": input_grid, "output": output_grid}


def _new_colors(rng, background, center, count=4):
    return [int(x) for x in rng.choice(
        [v for v in range(10) if v not in (background, center)],
        size=count, replace=False)]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    simple = bool(rng.random() < 0.2)
    base_background = 0 if rng.random() < 0.68 else int(rng.integers(1, 10))
    base_center = int(rng.choice([c for c in range(10) if c != base_background]))
    first_palette = _new_colors(rng, base_background, base_center)
    rng.shuffle(first_palette)
    first_roles = dict(zip(_DIRECTIONS, first_palette))
    if simple or rng.random() < 0.35:
        second_roles = dict(zip(
            _DIRECTIONS, (first_palette[2], first_palette[3],
                          first_palette[0], first_palette[1])))
    else:
        available = [c for c in range(10) if c not in
                     (base_background, base_center, *first_palette)]
        extras = [int(x) for x in rng.choice(available, size=2, replace=False)]
        removed, kept = [first_palette[2], extras[0]], [first_palette[0], extras[1]]
        rng.shuffle(removed)
        rng.shuffle(kept)
        second_roles = dict(zip(_DIRECTIONS, removed + kept))

    puzzle = {"train": [], "test": []}
    seen = set()
    train_kinds = [str(k) for k in rng.permutation(_KINDS)]
    if simple:
        train_kinds = train_kinds[:2]
    test_weights = np.array((1, 1, 2, 3, 3, 2), dtype=float)
    for index in range(num_train + num_test):
        training = index < num_train
        if training and index < 2:
            background, center_color = base_background, base_center
            colors = first_roles if index == 0 else second_roles
        else:
            background = (int(rng.integers(0, 10)) if not simple and rng.random() < 0.19
                          else base_background)
            center_color = (int(rng.choice([c for c in range(10) if c != background]))
                            if (not simple and rng.random() < 0.25) or background == base_center
                            else base_center)
            if simple:
                colors = dict(zip(_DIRECTIONS,
                                  [int(c) for c in rng.permutation(first_palette)]))
            elif rng.random() < 0.45:
                pool = [c for c in first_palette if c not in (background, center_color)]
                pool += [c for c in range(10) if c not in
                         (background, center_color, *pool)]
                colors = dict(zip(_DIRECTIONS, [int(x) for x in
                    rng.choice(pool[:max(4, len(pool) - 2)], size=4, replace=False)]))
            else:
                colors = dict(zip(_DIRECTIONS,
                                  _new_colors(rng, background, center_color)))
        kind = (train_kinds[index % len(train_kinds)] if training else
                str(rng.choice(_KINDS, p=test_weights / test_weights.sum())))
        section = "train" if training else "test"
        for _ in range(80):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _make_pair(pair_rng, kind, background, center_color, colors, simple)
            if pair is None:
                continue
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                seen.add(key)
                puzzle[section].append(pair)
                break
        else:
            raise RuntimeError("could not build a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles_from_grid(grid):
    height, width = grid.shape
    corners = {int(grid[r, c]) for r in (0, height - 1) for c in (0, width - 1)}
    assert len(corners) == 1
    background = next(iter(corners))
    colors = [int(c) for c in np.unique(grid) if c != background]
    assert len(colors) == 5
    regions = {c: {tuple(map(int, p)) for p in np.argwhere(grid == c)}
               for c in colors}
    candidates = []
    for center_color in colors:
        center = regions[center_color]
        cr, cc = [r for r, _ in center], [c for _, c in center]
        top, bottom, left, right = min(cr), max(cr), min(cc), max(cc)
        roles = {}
        for color in colors:
            if color == center_color:
                continue
            cells = regions[color]
            rows, cols = [r for r, _ in cells], [c for _, c in cells]
            if max(rows) < top:
                role = "up"
            elif min(rows) > bottom:
                role = "down"
            elif (min(rows) >= top and max(rows) <= bottom and
                  all(c < min(x for rr, x in center if rr == r)
                      for r, c in cells)):
                role = "left"
            elif (min(rows) >= top and max(rows) <= bottom and
                  all(c > max(x for rr, x in center if rr == r)
                      for r, c in cells)):
                role = "right"
            else:
                break
            if role in roles or not _connected8(cells) or not _touches8(cells, center):
                break
            roles[role] = color
        if len(roles) == 4 and _connected8(center):
            candidates.append((center_color, roles))
    assert len(candidates) == 1
    return background, candidates[0][0], candidates[0][1]


def _pair_digest(pair):
    serial = {key: pair[key].tolist() for key in ("input", "output")}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":"),
                                      sort_keys=True).encode()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen = set()
    train_roles = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            assert _pair_digest(pair) not in _SOURCE_DIGESTS
            bg, center, roles = _roles_from_grid(inp)
            expected = inp.copy()
            expected[np.isin(inp, (roles["up"], roles["down"]))] = bg
            assert np.array_equal(expected, out) and not np.array_equal(inp, out)
            assert np.all(out[inp == center] == center)
            if section == "train":
                train_roles.append((bg, center, roles))
    bg0, center0, roles0 = train_roles[0]
    bg1, center1, roles1 = train_roles[1]
    assert bg0 == bg1 and center0 == center1
    assert {roles0["up"], roles0["down"]} & {roles1["left"], roles1["right"]}
    assert {roles0["left"], roles0["right"]} & {roles1["up"], roles1["down"]}


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def _components4(cells):
    unseen = set(cells)
    count = 0
    while unseen:
        stack = [unseen.pop()]
        count += 1
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return count


def _review_features(pair):
    inp = pair["input"]
    background, center_color, roles = _roles_from_grid(inp)
    center = {tuple(map(int, p)) for p in np.argwhere(inp == center_color)}
    rows = [r for r, _ in center]
    cols = [c for _, c in center]
    top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
    crop = inp[top:bottom + 1, left:right + 1] == center_color
    widths = [int(row.sum()) for row in crop]
    middle = len(widths) // 2
    arms = {d: {tuple(map(int, p)) for p in np.argwhere(inp == color)}
            for d, color in roles.items()}
    side_contacts = []
    for direction in ("left", "right"):
        contact_rows = []
        for r, c in arms[direction]:
            same_row = [x for rr, x in center if rr == r]
            if direction == "left" and c == min(same_row) - 1:
                contact_rows.append(r)
            if direction == "right" and c == max(same_row) + 1:
                contact_rows.append(r)
        side_contacts.append(np.mean(contact_rows))
    top_contact = [c for r, c in arms["up"] if r == top - 1]
    bottom_contact = [c for r, c in arms["down"] if r == bottom + 1]
    horizontal_middle = (left + right) / 2
    occupied = np.argwhere(inp != background)
    margins = (int(occupied[:, 0].min()),
               int(inp.shape[0] - 1 - occupied[:, 0].max()),
               int(occupied[:, 1].min()),
               int(inp.shape[1] - 1 - occupied[:, 1].max()))
    return {
        "even_center_width": crop.shape[1] % 2 == 0,
        "hollow": any(np.any(row) and not np.all(
            row[np.where(row)[0].min():np.where(row)[0].max() + 1]) for row in crop),
        "diamond": (len(widths) % 2 == 1 and widths == widths[::-1]
                    and all(widths[i] < widths[i + 1] for i in range(middle))),
        "thin_cross": (widths[0] == widths[-1] == 1
                       and sum(w > 1 for w in widths) <= 2 and max(widths) >= 5),
        "hourglass": min(widths) <= 3 and widths[0] >= 5 and widths[-1] >= 5,
        "asymmetric": not np.array_equal(crop, crop[:, ::-1]),
        "diagonal_arm": any(_components4(cells) > 1 for cells in arms.values()),
        "side_span4": any(max(r for r, _ in arms[d]) - min(r for r, _ in arms[d]) + 1 >= 4
                          for d in ("left", "right")),
        "even_arm_thickness": any(
            (max(c for _, c in arms[d]) - min(c for _, c in arms[d]) + 1) % 2 == 0
            for d in ("up", "down")) or any(
            (max(r for r, _ in arms[d]) - min(r for r, _ in arms[d]) + 1) % 2 == 0
            for d in ("left", "right")),
        "offmid_topbottom": any(abs(float(np.mean(contact)) - horizontal_middle) >= 0.75
                                for contact in (top_contact, bottom_contact)),
        "unequal_side_heights": abs(float(side_contacts[0] - side_contacts[1])) >= 1,
        "offcenter": (abs((left + right) / 2 - (inp.shape[1] - 1) / 2) >= 1 or
                      abs((top + bottom) / 2 - (inp.shape[0] - 1) / 2) >= 1),
        "zero_margin": 0 in margins,
        "roomy_margin": max(margins) >= 4,
    }


def self_test() -> None:
    counts_train, counts_test, shapes, backgrounds, centers = set(), set(), set(), set(), set()
    dimensions, palettes, kinds_seen, palette_changes = set(), set(), set(), 0
    review_counts = {key: 0 for key in (
        "even_center_width", "hollow", "diamond", "thin_cross", "hourglass",
        "asymmetric", "diagonal_arm", "side_span4", "even_arm_thickness",
        "offmid_topbottom", "unequal_side_heights", "offcenter",
        "zero_margin", "roomy_margin", "combined")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        pair_palettes = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp = pair["input"]
                bg, center, roles = _roles_from_grid(inp)
                backgrounds.add(bg)
                centers.add(center)
                dimensions.add(inp.shape)
                palettes.add(tuple(sorted(np.unique(inp).tolist())))
                pair_palettes.add(tuple(sorted(roles.values())))
                coords = np.argwhere(inp == center)
                bbox = inp[coords[:, 0].min():coords[:, 0].max() + 1,
                           coords[:, 1].min():coords[:, 1].max() + 1] == center
                shapes.add((bbox.shape, bbox.tobytes()))
                kinds_seen.add((bbox.shape[0] % 2, bbox.shape[1] % 2,
                                bool(np.any(~bbox)), section))
                observed = _review_features(pair)
                for key, value in observed.items():
                    review_counts[key] += bool(value)
                review_counts["combined"] += all(observed[key] for key in (
                    "diagonal_arm", "asymmetric", "offmid_topbottom",
                    "unequal_side_heights"))
        palette_changes += len(pair_palettes) > 1
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(dimensions) >= 35 and len(shapes) >= 70
    assert len(backgrounds) >= 4 and len(centers) >= 5
    assert len(palettes) >= 35 and palette_changes >= 30
    assert len(kinds_seen) >= 8
    assert all(review_counts[key] >= limit for key, limit in (
        ("even_center_width", 25), ("hollow", 10), ("diamond", 10),
        ("thin_cross", 5), ("hourglass", 20), ("asymmetric", 25),
        ("diagonal_arm", 25), ("side_span4", 15),
        ("even_arm_thickness", 25), ("offmid_topbottom", 25),
        ("unequal_side_heights", 10), ("offcenter", 25),
        ("zero_margin", 20), ("roomy_margin", 20), ("combined", 3)))
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(707, tr, te)
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
        assert _same_puzzle(puzzle, generate_puzzle(707, tr, te))
    assert not _same_puzzle(generate_puzzle(101, 5, 2), generate_puzzle(102, 5, 2))
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (4, 4), (3, 4)):
        try:
            generate_puzzle(9, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
