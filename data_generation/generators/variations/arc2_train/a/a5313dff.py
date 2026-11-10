import numpy as np


_AUTO_COUNTS = (
    (2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
    (7, 1), (8, 2), (9, 3), (10, 4),
)

_SOURCE_TEXT = (
    ("00020000/02020000/00020000/22222220/00200020/00202020/00200020/00222220",
     "00020000/02020000/00020000/22222220/00211120/00212120/00211120/00222220"),
    ("00000000/02222200/02000200/02020200/02000200/02222200/00000000/00000000",
     "00000000/02222200/02111200/02121200/02111200/02222200/00000000/00000000"),
    ("000222220000/000200020000/000202022220/000200020020/000222220020/000200020020/000202022220/000200020000/000222220000/000000022222/000000020002/000000020202",
     "000222220000/000211120000/000212122220/000211121120/000222221120/000211121120/000212122220/000211120000/000222220000/000000022222/000000020002/000000020202"),
    ("002222200/002000200/002000200/222222200/200020000/202020000/200022222/222220000/000020000",
     "002222200/002111200/002111200/222222200/211120000/212120000/211122222/222220000/000020000"),
)


def _text_grid(text):
    return np.array([[int(x) for x in row] for row in text.split("/")], dtype=np.int8)


def _matches_source_pair(inp, out):
    return any(
        inp.shape == _text_grid(a).shape
        and np.array_equal(inp, _text_grid(a))
        and np.array_equal(out, _text_grid(b))
        for a, b in _SOURCE_TEXT
    )


def _draw_h(wall, r, c0, c1):
    wall[r, min(c0, c1):max(c0, c1) + 1] = True


def _draw_v(wall, c, r0, r1):
    wall[min(r0, r1):max(r0, r1) + 1, c] = True


def _rect(wall, box):
    t, l, b, r = box
    _draw_h(wall, t, l, r)
    _draw_h(wall, b, l, r)
    _draw_v(wall, l, t, b)
    _draw_v(wall, r, t, b)


def _exterior(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _solve(inp, background, fill):
    blank = inp == background
    out = inp.copy()
    out[blank & ~_exterior(blank)] = fill
    return out


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            component = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            result.append(component)
    return result


def _filled(wall):
    blank = ~wall
    return blank & ~_exterior(blank)


def _bounds(wall):
    rows, cols = np.where(wall)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop(wall, rng):
    t, l, b, r = _bounds(wall)
    pads = [int(rng.integers(0, 5)) for _ in range(4)]
    if rng.random() < 0.3:
        pads[int(rng.integers(0, 4))] = 0
    top, left, bottom, right = pads
    h = b - t + 1 + top + bottom
    w = r - l + 1 + left + right
    if h < 8:
        bottom += 8 - h
    if w < 8:
        right += 8 - w
    if b - t + 1 + top + bottom > 30 or r - l + 1 + left + right > 30:
        return None
    result = np.zeros((b - t + 1 + top + bottom, r - l + 1 + left + right), dtype=bool)
    result[top:top + b - t + 1, left:left + r - l + 1] = wall[t:b + 1, l:r + 1]
    return result


def _stepped_frame(wall, box, rng):
    t, l, b, r = box
    cut_r = int(rng.integers(t + 2, b - 1))
    cut_c = int(rng.integers(l + 2, r - 1))
    # A closed L-shaped loop; rotations vary the missing corner.
    corner = int(rng.integers(0, 4))
    if corner == 0:
        points = ((t, l), (t, cut_c), (cut_r, cut_c), (cut_r, r), (b, r), (b, l))
    elif corner == 1:
        points = ((t, l), (t, r), (b, r), (b, cut_c), (cut_r, cut_c), (cut_r, l))
    elif corner == 2:
        points = ((t, l), (t, r), (cut_r, r), (cut_r, cut_c), (b, cut_c), (b, l))
    else:
        points = ((t, cut_c), (t, r), (b, r), (b, l), (cut_r, l), (cut_r, cut_c))
    for a, z in zip(points, points[1:] + points[:1]):
        if a[0] == z[0]:
            _draw_h(wall, a[0], a[1], z[1])
        else:
            _draw_v(wall, a[1], a[0], z[0])


def _shared_box(anchor, rng):
    t, l, b, r = anchor
    height = int(rng.integers(4, 9))
    width = int(rng.integers(4, 9))
    side = int(rng.integers(0, 4))
    if side in (0, 2):
        new_l = int(rng.integers(l - width + 3, r - 1))
        new_r = new_l + width - 1
        if side == 0:
            return (t - height + 1, new_l, t, new_r)
        return (b, new_l, b + height - 1, new_r)
    new_t = int(rng.integers(t - height + 3, b - 1))
    new_b = new_t + height - 1
    if side == 1:
        return (new_t, r, new_b, r + width - 1)
    return (new_t, l - width + 1, new_b, l)


def _add_open_outline(wall, rng):
    t, l, b, r = _bounds(wall)
    height = int(rng.integers(4, 8))
    width = int(rng.integers(4, 8))
    side = int(rng.integers(0, 4))
    offset = int(rng.integers(-2, 3))
    if side == 0:
        box = (t - height - 2, l + offset, t - 3, l + offset + width - 1)
    elif side == 1:
        box = (t + offset, r + 3, t + offset + height - 1, r + width + 2)
    elif side == 2:
        box = (b + 3, l + offset, b + height + 2, l + offset + width - 1)
    else:
        box = (t + offset, l - width - 2, t + offset + height - 1, l - 3)
    ot, ol, ob, oright = box
    if min(ot, ol) < 2 or max(ob, oright) >= 62:
        return False
    candidate = wall.copy()
    _rect(candidate, box)
    gap_side = int(rng.integers(0, 4))
    gap_len = int(rng.integers(1, min(3, (width if gap_side in (0, 2) else height) - 2) + 1))
    if gap_side in (0, 2):
        start = int(rng.integers(ol + 1, oright - gap_len + 1))
        candidate[ot if gap_side == 0 else ob, start:start + gap_len] = False
    else:
        start = int(rng.integers(ot + 1, ob - gap_len + 1))
        candidate[start:start + gap_len, oright if gap_side == 1 else ol] = False
    if rng.random() < 0.35:
        # One connection makes the open frame part of the wall network.
        if side == 0:
            _draw_v(candidate, ol, t, ob)
        elif side == 1:
            _draw_h(candidate, ot, r, ol)
        elif side == 2:
            _draw_v(candidate, ol, b, ot)
        else:
            _draw_h(candidate, ot, oright, l)
    before = _filled(wall)
    after = _filled(candidate)
    if np.any(after & ~before) or after[ot + 1, ol + 1]:
        return False
    wall[:] = candidate
    return True


def _add_branches(wall, rng):
    for _ in range(int(rng.integers(0, 4))):
        t, l, b, r = _bounds(wall)
        side = int(rng.integers(0, 4))
        if side in (0, 2):
            choices = [c for c in range(l, r + 1) if wall[t if side == 0 else b, c]]
            if not choices:
                continue
            c = choices[int(rng.integers(len(choices)))]
            end = (t - 1 if side == 0 else b + 1)
            length = int(rng.integers(1, 6))
            far = end - length + 1 if side == 0 else end + length - 1
            if 1 <= far <= 62:
                _draw_v(wall, c, end, far)
        else:
            choices = [y for y in range(t, b + 1) if wall[y, r if side == 1 else l]]
            if not choices:
                continue
            y = choices[int(rng.integers(len(choices)))]
            end = (r + 1 if side == 1 else l - 1)
            length = int(rng.integers(1, 6))
            far = end + length - 1 if side == 1 else end - length + 1
            if 1 <= far <= 62:
                _draw_h(wall, y, end, far)


def _try_mark(wall, cells):
    trial = wall.copy()
    for r, c in cells:
        trial[r, c] = True
    if np.any(trial[:-1, :-1] & trial[1:, :-1] & trial[:-1, 1:] & trial[1:, 1:]):
        return False
    if np.any(_filled(trial) & ~_filled(wall)):
        return False
    wall[:] = trial
    return True


def _add_marks(wall, rng, required_exterior=False):
    added_inside = added_outside = 0
    plan = [False, True] if required_exterior else []
    plan += [bool(rng.integers(0, 2)) for _ in range(int(rng.integers(0, 5)))]
    rng.shuffle(plan)
    for inside in plan:
        filled = _filled(wall)
        region = filled if inside else (~wall & ~filled)
        candidates = list(zip(*np.where(region)))
        rng.shuffle(candidates)
        for r, c in candidates[:70]:
            h, w = wall.shape
            if inside and (r == 0 or c == 0 or r == h - 1 or c == w - 1):
                continue
            if any(0 <= nr < h and 0 <= nc < w and wall[nr, nc]
                   for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))):
                continue
            cells = [(r, c)]
            if rng.random() < 0.35:
                choices = [(nr, nc) for nr, nc in ((r + 1, c), (r, c + 1), (r - 1, c), (r, c - 1))
                           if 0 <= nr < h and 0 <= nc < w and region[nr, nc]]
                if choices:
                    cells.append(choices[int(rng.integers(len(choices)))])
            if _try_mark(wall, cells):
                added_inside += int(inside)
                added_outside += int(not inside)
                break
    return added_inside, added_outside


def _make_wall(rng, style, require_open=False):
    for _ in range(120):
        wall = np.zeros((64, 64), dtype=bool)
        stepped = style == "step" or (style != "simple" and rng.random() < 0.25)
        height = int(rng.integers(6 if stepped else 4, 10))
        width = int(rng.integers(6 if stepped else 4, 10))
        base = (26, 26, 26 + height - 1, 26 + width - 1)
        if stepped:
            _stepped_frame(wall, base, rng)
        else:
            _rect(wall, base)
        boxes = [base]
        count = 1 if style in ("simple", "step") else int(rng.integers(2, 5 if style == "complex" else 4))
        for _ in range(count - 1):
            for _ in range(25):
                anchor = boxes[int(rng.integers(len(boxes)))]
                box = _shared_box(anchor, rng)
                if min(box[:2]) < 6 or max(box[2:]) >= 57:
                    continue
                trial = wall.copy()
                _rect(trial, box)
                if len(_components(_filled(trial))) > len(_components(_filled(wall))):
                    wall = trial
                    boxes.append(box)
                    break
            else:
                break
        if len(boxes) < count:
            continue
        if style != "simple" and rng.random() < 0.35:
            box = boxes[int(rng.integers(len(boxes)))]
            t, l, b, r = box
            if r - l >= 5 and rng.random() < 0.5:
                _draw_v(wall, int(rng.integers(l + 2, r - 1)), t, b)
            elif b - t >= 5:
                _draw_h(wall, int(rng.integers(t + 2, b - 1)), l, r)
        if require_open or rng.random() < (0.75 if style == "complex" else 0.45):
            if not _add_open_outline(wall, rng) and require_open:
                continue
        before_branches = _filled(wall)
        _add_branches(wall, rng)
        if np.any(_filled(wall) & ~before_branches):
            continue
        wall = _crop(wall, rng)
        if wall is None:
            continue
        if np.any(wall[:-1, :-1] & wall[1:, :-1] & wall[:-1, 1:] & wall[1:, 1:]):
            continue
        if not np.any(_filled(wall)):
            continue
        _add_marks(wall, rng, required_exterior=style == "simple")
        if np.mean(wall) >= 0.48 or not np.any(~wall & ~_filled(wall)):
            continue
        if require_open and not _open_outline_sides(wall, ~wall & ~_filled(wall)):
            continue
        if style == "simple" and len(_components(_filled(wall))) != 1:
            continue
        return wall
    raise ValueError("unable to construct a valid wall network")


def _valid_count(value, name, low, high):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def _choose_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _valid_count(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _valid_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def sample_puzzle_parameters(rng, train_count):
    colors = [int(x) for x in rng.permutation(10)]
    fill = colors[0]
    schemes = [(colors[1], colors[2])]
    if rng.random() < 0.65:
        schemes.append((colors[2], colors[1]))
        if train_count >= 4 and rng.random() < 0.55:
            schemes.append((colors[3], colors[4]))
    return {"fill": fill, "schemes": schemes}


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(seed, rng, num_train, num_test)
    params = sample_puzzle_parameters(rng, train_count)
    schemes = params["schemes"]
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count in (("train", train_count), ("test", test_count)):
        for slot in range(count):
            if regime == "train":
                style = "simple" if slot == 0 else "complex" if slot == 1 else ("simple", "network", "complex", "step")[int(rng.integers(4))]
                scheme = schemes[slot] if slot < len(schemes) else schemes[int(rng.integers(len(schemes)))]
                require_open = slot == 1
            else:
                style = ("simple", "network", "complex", "step")[int(rng.choice(4, p=[0.12, 0.28, 0.45, 0.15]))]
                scheme = schemes[int(rng.integers(len(schemes)))]
                require_open = False
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                wall = _make_wall(pair_rng, style, require_open)
                background, barrier = scheme
                inp = np.full(wall.shape, background, dtype=np.int8)
                inp[wall] = barrier
                out = _solve(inp, background, params["fill"])
                key = _grid_key(inp)
                if key not in seen and not _matches_source_pair(inp, out):
                    seen.add(key)
                    puzzle[regime].append({"input": inp, "output": out})
                    break
            else:
                raise ValueError("unable to construct distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(inp, out):
    assert inp.shape == out.shape
    assert len(np.unique(inp)) == 2
    changed = inp != out
    assert np.any(changed)
    backgrounds = np.unique(inp[changed])
    fills = np.unique(out[changed])
    assert len(backgrounds) == 1 and len(fills) == 1
    background, fill = int(backgrounds[0]), int(fills[0])
    barrier = int(next(x for x in np.unique(inp) if x != background))
    assert fill not in np.unique(inp)
    assert np.count_nonzero(inp == background) > inp.size / 2
    return background, barrier, fill


def _open_outline_sides(wall, exterior):
    h, w = wall.shape
    found = set()
    for height in range(4, 8):
        for width in range(4, 8):
            for top in range(h - height + 1):
                for left in range(w - width + 1):
                    bottom, right = top + height - 1, left + width - 1
                    if not exterior[top + 1, left + 1]:
                        continue
                    sides = (
                        wall[top, left:right + 1],
                        wall[top:bottom + 1, right],
                        wall[bottom, left:right + 1],
                        wall[top:bottom + 1, left],
                    )
                    complete = [bool(np.all(side)) for side in sides]
                    if sum(complete) != 3:
                        continue
                    missing_side = complete.index(False)
                    holes = np.flatnonzero(~sides[missing_side])
                    if (1 <= len(holes) <= 2 and holes[0] > 0
                            and holes[-1] < len(sides[missing_side]) - 1
                            and np.all(np.diff(holes) == 1)):
                        found.add(missing_side)
    return found


def _review_features(inp, out):
    background, barrier, fill = _roles(inp, out)
    wall = inp == barrier
    filled = out == fill
    outside = (inp == background) & (out == background)
    h, w = inp.shape
    enclosed = _components(filled)
    labels = np.full(inp.shape, -1, dtype=np.int16)
    for index, component in enumerate(enclosed):
        for r, c in component:
            labels[r, c] = index
    shared = False
    junction = False
    for r, c in zip(*np.where(wall)):
        adjacent = [(nr, nc) for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    if 0 <= nr < h and 0 <= nc < w and wall[nr, nc]]
        junction |= len(adjacent) >= 3
        if 0 < r < h - 1 and 0 < c < w - 1:
            shared |= (labels[r - 1, c] >= 0 and labels[r + 1, c] >= 0
                       and labels[r - 1, c] != labels[r + 1, c])
            shared |= (labels[r, c - 1] >= 0 and labels[r, c + 1] >= 0
                       and labels[r, c - 1] != labels[r, c + 1])
    dots_inside = dots_outside = strokes_inside = strokes_outside = edge_marks = 0
    for component in _components(wall):
        if len(component) > 2:
            continue
        members = set(component)
        neighbors = set()
        for r, c in component:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in members:
                    neighbors.add((nr, nc))
        inside = bool(neighbors) and all(filled[point] for point in neighbors)
        exterior = bool(neighbors) and all(outside[point] for point in neighbors)
        if len(component) == 1:
            dots_inside += inside
            dots_outside += exterior
        else:
            strokes_inside += inside
            strokes_outside += exterior
        edge_marks += any(r in (0, h - 1) or c in (0, w - 1) for r, c in component)
    narrow = False
    stepped = False
    for component in enclosed:
        rows = [r for r, _ in component]
        cols = [c for _, c in component]
        top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
        narrow |= min(bottom - top + 1, right - left + 1) <= 2
        if bottom - top >= 3 and right - left >= 3:
            stepped |= any(
                np.all(outside[rr:rr + 2, cc:cc + 2])
                for rr in (top, bottom - 1)
                for cc in (left, right - 1)
            )
    return {
        "enclosed": len(enclosed),
        "dots_inside": dots_inside,
        "dots_outside": dots_outside,
        "strokes_inside": strokes_inside,
        "strokes_outside": strokes_outside,
        "edge_marks": edge_marks,
        "shared": shared,
        "junction": junction,
        "open_sides": _open_outline_sides(wall, outside),
        "narrow": narrow,
        "stepped": stepped,
        "edge_wall": bool(np.any(wall[0]) or np.any(wall[-1])
                          or np.any(wall[:, 0]) or np.any(wall[:, -1])),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_schemes = set()
    fills = set()
    train_enclosure_counts = set()
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            background, barrier, fill = _roles(inp, out)
            fills.add(fill)
            if regime == "train":
                train_schemes.add((background, barrier))
            else:
                assert (background, barrier) in train_schemes
            wall = inp == barrier
            assert not np.any(wall[:-1, :-1] & wall[1:, :-1] & wall[:-1, 1:] & wall[1:, 1:])
            assert len(_components(_filled(wall))) >= 1
            if regime == "train":
                train_enclosure_counts.add(len(_components(_filled(wall))))
            assert np.any(~wall & ~_filled(wall))
            assert np.array_equal(out, _solve(inp, background, fill))
            assert np.all(out[wall] == barrier)
            assert not _matches_source_pair(inp, out)
    assert len(fills) == 1
    assert len(train_schemes) >= 1
    assert 1 in train_enclosure_counts
    assert any(count >= 2 for count in train_enclosure_counts)
    train_open = False
    for pair in puzzle["train"]:
        inp, out = pair["input"], pair["output"]
        background, barrier, _ = _roles(inp, out)
        exterior = (inp == background) & (out == background)
        if _open_outline_sides(inp == barrier, exterior):
            train_open = True
            break
    assert train_open


def _puzzles_equal(a, b):
    return all(len(a[k]) == len(b[k]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[k], b[k])) for k in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    shapes, structures, palettes, pair_counts = set(), set(), set(), set()
    varied_puzzles = role_swaps = consistent_puzzles = 0
    combined = 0
    totals = {split: {name: 0 for name in (
        "dots_inside", "dots_outside", "strokes_inside", "strokes_outside",
        "edge_marks", "shared", "junction", "open", "narrow", "stepped",
        "edge_wall", "single", "multi",
    )} for split in ("train", "test")}
    opening_sides = set()
    overlap_shared = overlap_open = 0
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        schemes = set()
        seen_by_split = {"train": {"shared": False, "open": False},
                         "test": {"shared": False, "open": False}}
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                bg, barrier, fill = _roles(inp, out)
                wall = inp == barrier
                shapes.add(inp.shape)
                structures.add((inp.shape, wall.tobytes()))
                palettes.add((bg, barrier, fill))
                schemes.add((bg, barrier))
                review = _review_features(inp, out)
                count = review["enclosed"]
                pair_counts.add(count)
                current = totals[regime]
                current["single"] += count == 1
                current["multi"] += count >= 2
                for name in ("dots_inside", "dots_outside", "strokes_inside",
                             "strokes_outside", "edge_marks"):
                    current[name] += review[name]
                for name in ("shared", "junction", "narrow", "stepped", "edge_wall"):
                    current[name] += review[name]
                current["open"] += bool(review["open_sides"])
                opening_sides.update(review["open_sides"])
                seen_by_split[regime]["shared"] |= review["shared"]
                seen_by_split[regime]["open"] |= bool(review["open_sides"])
                combined += bool(review["shared"] and review["open_sides"]
                                 and (review["dots_inside"] or review["dots_outside"]
                                      or review["strokes_inside"] or review["strokes_outside"]))
                assert np.array_equal(out, _solve(inp, bg, fill))
        varied_puzzles += int(len(schemes) > 1)
        consistent_puzzles += int(len(schemes) == 1)
        role_swaps += any((b, a) in schemes for a, b in schemes)
        overlap_shared += seen_by_split["train"]["shared"] and seen_by_split["test"]["shared"]
        overlap_open += seen_by_split["train"]["open"] and seen_by_split["test"]["open"]
    assert len(shapes) >= 25 and len(structures) >= 120 and len(palettes) >= 25
    assert len(pair_counts) >= 3 and varied_puzzles >= 15 and role_swaps >= 15
    assert consistent_puzzles >= 5
    assert totals["train"]["single"] >= 40 and totals["train"]["multi"] >= 35
    assert totals["test"]["single"] >= 3 and totals["test"]["multi"] >= 20
    for split in ("train", "test"):
        measured = totals[split]
        assert measured["dots_inside"] >= 10 and measured["dots_outside"] >= 10
        assert measured["strokes_inside"] >= 3 and measured["strokes_outside"] >= 3
        assert measured["edge_marks"] >= 5
        assert measured["shared"] >= 10 and measured["junction"] >= 10
        assert measured["open"] >= 5 and measured["narrow"] >= 10
        assert measured["stepped"] >= 2 and measured["edge_wall"] >= 10
    assert len(opening_sides) == 4
    assert overlap_shared >= 10 and overlap_open >= 5 and combined >= 20
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 6, 2)):
        p = generate_puzzle(seed, train, test)
        assert len(p["train"]) == train and len(p["test"]) == test
    for seed, train, test in ((1101, None, 1), (1102, None, 4), (1103, 2, None), (1104, 10, None)):
        p = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(p["train"]) == train
        if test is not None:
            assert len(p["test"]) == test
    assert _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    assert not _puzzles_equal(generate_puzzle(12345), generate_puzzle(12346))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(99, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"illegal count accepted: {kwargs}")
