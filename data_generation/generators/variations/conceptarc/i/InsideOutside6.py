import numpy as np


_SOURCE_INPUTS = frozenset(
    """000000000000/600333333300/603300003006/003006003000/003306063060/000300003000/600300600300/003300000300/000333000300/000003300000/000000000000
000000000000000/000000100000000/000000001004400/004400100040400/004040000000400/040040000400040/401044000400040/440004444000440/044000000000400/004401000004400/000440010004000/000040000044000/000040000400010/010000444401000/000010000000000
000008000008000/077077770000000/070000000000000/770080007000080/700000007770000/070000000070000/077000000007000/000770000007000/000007000007000/000007000000000/080007000007000/000007700070008/000000777700000/000000000000000/008000000000800
000000000000000/030000003000300/002202000000000/002230020000000/002000002222000/022003000002203/000000030300200/002003000000200/002200003000200/000200000000200/030220302022200/000020022000000/003002220303003/000000000000000/000000000000000
0000000002000000/0000099900000000/0009900009000000/0009002209999000/0090000000000900/0009002000209000/0200000000009000/0000992020090000/0020090000090000/0000900000000000/0000902000920000/0000900000900000/0200999099000002/0000000000000000/0000000000000000""".splitlines()
)


def _text(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _boundary(region):
    padded = np.pad(region, 1)
    h, w = region.shape
    return region & (
        ~padded[:h, 1:w + 1]
        | ~padded[2:h + 2, 1:w + 1]
        | ~padded[1:h + 1, :w]
        | ~padded[1:h + 1, 2:w + 2]
    )


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = remaining.pop()
        group = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    point = (row + dr, col + dc)
                    if (dr or dc) and point in remaining:
                        remaining.remove(point)
                        group.add(point)
                        stack.append(point)
        groups.append(group)
    return groups


def _make_region(rng, height, width, complex_shape, force_edge):
    box_h = int(rng.integers(max(7, (height + 1) // 2), min(height - 2, 17) + 1))
    box_w = int(rng.integers(max(7, (width + 1) // 2), min(width - 2, 18) + 1))
    left = np.empty(box_h, dtype=int)
    right = np.empty(box_h, dtype=int)
    l = int(rng.integers(1, min(4, box_w - 4)))
    r = box_w - int(rng.integers(2, min(5, box_w - 2)))
    for row in range(box_h):
        if row:
            steps = (0.28, 0.44, 0.28) if complex_shape else (0.12, 0.76, 0.12)
            l += int(rng.choice((-1, 0, 1), p=steps))
            r += int(rng.choice((-1, 0, 1), p=steps))
        l = int(np.clip(l, 0, min(4, box_w - 4)))
        r = int(np.clip(r, max(l + 5, box_w - 5), box_w - 1))
        left[row], right[row] = l, r

    if complex_shape and rng.random() < 0.78:
        center = int(rng.integers(box_w // 3, 2 * box_w // 3 + 1))
        taper = int(rng.integers(2, min(5, box_h - 1)))
        for row in range(taper):
            radius = 1 + row
            left[row] = max(left[row], center - radius)
            right[row] = min(right[row], center + radius)
        if rng.random() < 0.35:
            for j in range(taper):
                row = box_h - 1 - j
                radius = 1 + j
                left[row] = max(left[row], center - radius)
                right[row] = min(right[row], center + radius)

    if complex_shape and rng.random() < 0.75:
        start = int(rng.integers(2, box_h - 2))
        length = int(rng.integers(1, min(4, box_h - start) + 1))
        depth = int(rng.integers(2, min(5, box_w // 2) + 1))
        if rng.random() < 0.5:
            left[start:start + length] += depth
        else:
            right[start:start + length] -= depth

    if complex_shape and rng.random() < 0.58:
        start = int(rng.integers(1, box_h - 2))
        length = int(rng.integers(1, min(4, box_h - start) + 1))
        depth = int(rng.integers(1, 4))
        if rng.random() < 0.5:
            left[start:start + length] -= depth
        else:
            right[start:start + length] += depth

    if complex_shape and rng.random() < 0.47:
        start = int(rng.integers(2, box_h - 2))
        length = int(rng.integers(1, min(4, box_h - start) + 1))
        left[start:start + length] += int(rng.integers(1, 3))
        right[start:start + length] -= int(rng.integers(1, 3))

    left = np.clip(left, 0, box_w - 5)
    right = np.clip(right, 4, box_w - 1)
    for row in range(box_h):
        if right[row] - left[row] < 4:
            middle = (right[row] + left[row]) // 2
            left[row] = max(0, middle - 2)
            right[row] = min(box_w - 1, left[row] + 4)
        if row and min(right[row], right[row - 1]) - max(left[row], left[row - 1]) < 3:
            left[row] = max(0, min(left[row], right[row - 1] - 3))
            right[row] = min(box_w - 1, max(right[row], left[row - 1] + 3))

    local = np.zeros((box_h, box_w), dtype=bool)
    for row in range(box_h):
        local[row, left[row]:right[row] + 1] = True
    rows, cols = np.nonzero(local)
    local = local[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    local = np.rot90(local, int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        local = np.fliplr(local)
    rh, rw = local.shape
    if rh > height or rw > width:
        return None
    row = int(rng.integers(1, height - rh)) if height - rh >= 2 else 0
    col = int(rng.integers(1, width - rw)) if width - rw >= 2 else 0
    if force_edge:
        if rng.random() < 0.5:
            row = 0 if rng.random() < 0.5 else height - rh
        else:
            col = 0 if rng.random() < 0.5 else width - rw
    region = np.zeros((height, width), dtype=bool)
    region[row:row + rh, col:col + rw] = local
    return region


def _gap_candidates(wall):
    h, w = wall.shape
    straight, turn = [], []
    for row, col in np.argwhere(wall):
        row, col = int(row), int(col)
        if not (1 <= row < h - 1 and 1 <= col < w - 1):
            continue
        n, s = wall[row - 1, col], wall[row + 1, col]
        e, west = wall[row, col + 1], wall[row, col - 1]
        if (n and s) or (e and west):
            straight.append((row, col))
        elif int(n) + int(s) + int(e) + int(west) >= 2:
            turn.append((row, col))
        elif sum(
            bool(wall[row + dr, col + dc])
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if dr or dc
        ) >= 3:
            turn.append((row, col))
    return straight, turn


def _choose_gaps(rng, wall, count, first_kind):
    straight, turn = _gap_candidates(wall)
    if first_kind == "turn" and not turn:
        return None
    if first_kind == "straight" and not straight:
        return None
    gaps = []
    categories = [turn] if first_kind == "turn" else ([straight] if first_kind == "straight" else [])
    categories += [turn + straight] * (count - len(categories))
    for candidates in categories:
        order = rng.permutation(len(candidates))
        chosen = None
        for index in order:
            point = candidates[int(index)]
            if all(max(abs(point[0] - r), abs(point[1] - c)) > 2 for r, c in gaps):
                chosen = point
                break
        if chosen is None:
            return None
        gaps.append(chosen)
    if rng.random() < 0.23:
        candidates = [point for point in gaps if point in straight]
        if candidates:
            row, col = candidates[int(rng.integers(len(candidates)))]
            directions = ((1, 0), (0, 1))
            for dr, dc in directions:
                rr, cc = row + dr, col + dc
                r2, c2 = row + 2 * dr, col + 2 * dc
                r0, c0 = row - dr, col - dc
                if (0 <= r0 < wall.shape[0] and 0 <= c0 < wall.shape[1]
                    and 0 <= r2 < wall.shape[0] and 0 <= c2 < wall.shape[1]
                    and wall[r0, c0] and wall[rr, cc] and wall[r2, c2]
                    and all(max(abs(rr - r), abs(cc - c)) > 2
                            for r, c in gaps if (r, c) != (row, col))):
                    gaps.append((rr, cc))
                    break
    shown = wall.copy()
    for row, col in gaps:
        shown[row, col] = False
    if int(shown.sum()) < 18:
        return None
    groups = _components(shown)
    if len(groups) > 4 or min(map(len, groups)) < 3 or max(map(len, groups)) < 8:
        return None
    return gaps, shown


def _scan_region(wall, transposed):
    viewed = wall.T if transposed else wall
    rows = np.flatnonzero(viewed.any(axis=1))
    left = np.full(viewed.shape[0], -1, dtype=int)
    right = np.full(viewed.shape[0], -1, dtype=int)
    for row in rows:
        columns = np.flatnonzero(viewed[row])
        if columns[-1] - columns[0] >= 3:
            left[row], right[row] = columns[0], columns[-1]
    reliable = np.flatnonzero(left >= 0)
    if len(reliable) < 2:
        return None, 10**9
    for row in rows:
        if left[row] >= 0:
            continue
        before = reliable[reliable < row]
        after = reliable[reliable > row]
        p = before[-1] if len(before) else after[0]
        q = after[0] if len(after) else before[-1]
        left[row] = round((left[p] + left[q]) / 2)
        right[row] = round((right[p] + right[q]) / 2)
    region = np.zeros_like(viewed)
    for row in rows:
        region[row, left[row]:right[row] + 1] = True
    predicted_wall = _boundary(region)
    score = int(np.count_nonzero(predicted_wall & ~viewed)
                + 3 * np.count_nonzero(viewed & ~predicted_wall))
    return (region.T if transposed else region), score


def _recover_region(wall):
    candidates = (_scan_region(wall, False), _scan_region(wall, True))
    usable = [item for item in candidates if item[0] is not None]
    assert usable, "Contour has no recoverable span"
    return min(usable, key=lambda item: item[1])[0]


def _pick_positions(rng, pool, count, chosen, grouped):
    available = set(map(tuple, np.argwhere(pool))) - chosen
    while len(chosen) < count and available:
        neighbors = [
            point for point in available
            if any(max(abs(point[0] - r), abs(point[1] - c)) == 1 for r, c in chosen)
        ]
        separated = [
            point for point in available
            if all(max(abs(point[0] - r), abs(point[1] - c)) > 1 for r, c in chosen)
        ]
        if grouped and neighbors and rng.random() < 0.55:
            point = neighbors[int(rng.integers(len(neighbors)))]
        else:
            points = tuple(separated if separated else available)
            point = points[int(rng.integers(len(points)))]
        chosen.add(point)
        available.remove(point)
    return chosen


def _generate_pair(rng, section, index, colors):
    background, outline, marker = colors
    if section == "train" and index == 0:
        complex_shape, break_count = False, 1
    elif section == "train" and index == 1:
        complex_shape, break_count = True, 2
    else:
        complex_shape = bool(rng.random() < (0.65 if section == "test" else 0.60))
        break_count = int(rng.choice((1, 2, 3), p=(0.40, 0.42, 0.18)))
    height = int(rng.integers(11, 23))
    width = int(rng.integers(11, 24))
    force_edge = bool(rng.random() < 0.32)

    for _ in range(120):
        region = _make_region(rng, height, width, complex_shape, force_edge)
        if region is None:
            continue
        wall = _boundary(region)
        interior = region & ~wall
        if int(interior.sum()) < 10 or len(_components(wall)) != 1:
            continue
        chosen_gaps = _choose_gaps(
            rng, wall, break_count,
            "straight" if section == "train" and index == 0 else
            "turn" if section == "train" and index == 1 else None,
        )
        if chosen_gaps is None:
            continue
        gaps, shown_wall = chosen_gaps
        inside_pool = interior.copy()
        outside_pool = ~region
        if section == "train" and index < 2:
            inside_count = int(rng.integers(1, 3 if index == 0 else 6))
            outside_count = int(rng.integers(3, 7)) if index == 0 else int(rng.integers(1, 4))
        else:
            inside_count = int(rng.integers(0, 9))
            if section == "test" and rng.random() < 0.22:
                inside_count = 1
            outside_count = int(rng.integers(0, 9))
            if inside_count == outside_count == 0:
                inside_count = 1
        if inside_count > int(inside_pool.sum()) or outside_count > int(outside_pool.sum()):
            continue

        inside, outside = set(), set()
        rows, cols = np.nonzero(region)
        bbox = np.zeros(region.shape, dtype=bool)
        bbox[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = True
        bay = outside_pool & bbox
        if outside_count and np.any(bay) and rng.random() < 0.70:
            points = np.argwhere(bay)
            outside.add(tuple(points[int(rng.integers(len(points)))]))

        near_wall = np.zeros(region.shape, dtype=bool)
        for row, col in np.argwhere(wall):
            near_wall[max(0, row - 1):min(height, row + 2),
                      max(0, col - 1):min(width, col + 2)] = True
        adjacent_inner = inside_pool & near_wall
        if inside_count and np.any(adjacent_inner) and rng.random() < 0.75:
            points = np.argwhere(adjacent_inner)
            inside.add(tuple(points[int(rng.integers(len(points)))]))

        if rng.random() < 0.48:
            near_gap = np.zeros(region.shape, dtype=bool)
            for row, col in gaps:
                near_gap[max(0, row - 1):min(height, row + 2),
                         max(0, col - 1):min(width, col + 2)] = True
            sides = []
            if len(inside) < inside_count and np.any(inside_pool & near_gap):
                sides.append((inside_pool & near_gap, inside))
            if len(outside) < outside_count and np.any(outside_pool & near_gap):
                sides.append((outside_pool & near_gap, outside))
            if sides:
                pool, selected = sides[int(rng.integers(len(sides)))]
                points = np.argwhere(pool)
                selected.add(tuple(points[int(rng.integers(len(points)))]))

        grouped = bool(rng.random() < 0.48)
        inside = _pick_positions(rng, inside_pool, inside_count, inside, grouped)
        outside = _pick_positions(rng, outside_pool, outside_count, outside, grouped)
        if len(inside) != inside_count or len(outside) != outside_count:
            continue
        marker_mask = np.zeros(region.shape, dtype=bool)
        for row, col in inside | outside:
            marker_mask[row, col] = True
        if max(map(len, _components(marker_mask))) > 3:
            continue
        recovered = _recover_region(shown_wall)
        if any(not recovered[row, col] for row, col in inside):
            continue
        if any(recovered[row, col] for row, col in outside):
            continue
        inp = np.full(region.shape, background, dtype=np.int8)
        inp[shown_wall] = outline
        for row, col in inside | outside:
            inp[row, col] = marker
        out = np.full(region.shape, background, dtype=np.int8)
        for row, col in outside:
            out[row, col] = marker
        pair = {"input": inp, "output": out}
        try:
            if _roles(pair) != colors:
                continue
        except AssertionError:
            continue
        return pair
    raise RuntimeError("Could not construct a clear contour")


def _roles(pair):
    inp = pair["input"]
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 3
    background = int(colors[np.argmax(counts)])
    others = [int(color) for color in colors if int(color) != background]
    component_sizes = [max(map(len, _components(inp == color))) for color in others]
    assert component_sizes[0] != component_sizes[1]
    outline = others[int(np.argmax(component_sizes))]
    marker = others[1 - int(np.argmax(component_sizes))]
    return background, outline, marker


def _signature(puzzle):
    return tuple(
        (pair["input"].shape, pair["input"].tobytes(), pair["output"].tobytes())
        for section in ("train", "test") for pair in puzzle[section]
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    for value, lower, upper in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or not lower <= value <= upper
        ):
            raise ValueError("Invalid pair count")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("Training count must exceed test count")

    puzzle = {"train": [], "test": []}
    seen = set()
    training_colors = set()
    first_roles = None
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if section == "train" and index == 1:
                outline, marker = first_roles
                outline, marker = marker, outline
                bg_options = [c for c in range(10) if c not in (outline, marker, first_background)]
                background = int(rng.choice(bg_options))
            else:
                background = int(rng.choice(range(10))) if rng.random() < 0.38 else 0
                usable = [c for c in range(10) if c != background]
                if section == "test" and rng.random() < 0.65:
                    novel = [c for c in usable if c not in training_colors]
                    outline = int(rng.choice(novel)) if novel else int(rng.choice(usable))
                else:
                    outline = int(rng.choice(usable))
                marker = int(rng.choice([c for c in usable if c != outline]))
            if section == "train" and index == 0:
                first_roles = (outline, marker)
                first_background = background
            colors = (background, outline, marker)
            for _ in range(12):
                pair = _generate_pair(rng, section, index, colors)
                key = (pair["input"].shape, _text(pair["input"]))
                if key not in seen and key[1] not in _SOURCE_INPUTS:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("Could not make a unique pair")
            if section == "train":
                training_colors.update(colors)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs, pairs = set(), set()
    evidence = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            key = (inp.shape, _text(inp))
            pair_key = (key, _text(out))
            assert key not in inputs and pair_key not in pairs
            assert key[1] not in _SOURCE_INPUTS
            inputs.add(key)
            pairs.add(pair_key)
            background, outline, marker = _roles(pair)
            wall = inp == outline
            marker_cells = inp == marker
            assert int(wall.sum()) >= 18 and int(marker_cells.sum()) >= 1
            assert max(map(len, _components(marker_cells))) <= 3
            assert int((inp == background).sum()) > int(wall.sum())
            assert np.all(out[wall] == background)
            assert np.all(out[~(wall | marker_cells)] == background)
            assert np.all((out[marker_cells] == background) | (out[marker_cells] == marker))
            removed = marker_cells & (out == background)
            kept = marker_cells & (out == marker)
            recovered = _recover_region(wall)
            assert np.array_equal(removed, marker_cells & recovered)
            assert np.array_equal(kept, marker_cells & ~recovered)
            wr, wc = np.nonzero(wall)
            if np.any(removed):
                rr, cc = np.nonzero(removed)
                assert np.all((rr >= wr.min()) & (rr <= wr.max()))
                assert np.all((cc >= wc.min()) & (cc <= wc.max()))
            assert len(_components(wall)) <= 4
            evidence.append((section, background, outline, marker,
                             int(removed.sum()), int(kept.sum())))
    assert all(row[4] >= 1 and row[5] >= 1 for row in evidence[:2])
    assert evidence[0][2] == evidence[1][3]
    assert evidence[0][3] == evidence[1][2]
    assert evidence[0][1] != evidence[1][1]


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, palettes, structures, balances = set(), set(), set(), set()
    varied_shapes = varied_colors = varied_balance = 0
    edge_pairs = single_inside = zero_side = bay_survivors = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_shapes, local_colors, local_balances = set(), set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                bg, outline, marker = _roles(pair)
                wall = inp == outline
                markers = inp == marker
                removed = int(np.count_nonzero(markers & (out == bg)))
                kept = int(np.count_nonzero(markers & (out == marker)))
                shapes.add(inp.shape)
                palettes.add((bg, outline, marker))
                structures.add((inp.shape, tuple(wall.flat)))
                balances.add((removed, kept))
                local_shapes.add(inp.shape)
                local_colors.add((bg, outline, marker))
                local_balances.add((removed, kept))
                edge_pairs += int(np.any(wall[0]) or np.any(wall[-1])
                                  or np.any(wall[:, 0]) or np.any(wall[:, -1]))
                single_inside += int(removed == 1)
                zero_side += int(removed == 0 or kept == 0)
                wr, wc = np.nonzero(wall)
                surviving = markers & (out == marker)
                bay_survivors += int(np.any(surviving[wr.min():wr.max() + 1,
                                                      wc.min():wc.max() + 1]))
        varied_shapes += len(local_shapes) >= 2
        varied_colors += len(local_colors) >= 2
        varied_balance += len(local_balances) >= 2
    assert len(shapes) >= 45 and len(palettes) >= 45
    assert len(structures) >= 150 and len(balances) >= 25
    assert min(varied_shapes, varied_colors, varied_balance) >= 35
    assert edge_pairs >= 20 and single_inside >= 15 and zero_side >= 10
    assert bay_survivors >= 10
    for a, b in ((2, 1), (10, 1), (10, 4), (4, 2),
                 (None, 1), (None, 4), (2, None), (10, None)):
        first = generate_puzzle(301 + (a or 0) * 10 + (b or 0), a, b)
        second = generate_puzzle(301 + (a or 0) * 10 + (b or 0), a, b)
        assert _signature(first) == _signature(second)
        validate_generated_puzzle(first)
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(8, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count was accepted")
