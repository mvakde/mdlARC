import numpy as np


_SOURCE_INPUTS = (
    "0000000000000/0000000000000/0000111000010/0000101000111/0200111000010/2220000080000/0200008888000/0000000080000/0055550000000/0050050000222/0055550000202/0000000000222/0000000000000",
    "0000000000000000/0000000000200000/0111000002220000/0101000000200000/0111000000000100/0000088800000100/0000080800001111/0000080800000100/0000088800000000/0000000000000000/0000000000000000/0000000000000000/0000000000000000",
    "01000000000000000/11111000000000000/01000000000000800/00222200000088880/00200200033300800/00200200030300000/00222200033300000/00000000000000000/00000001000000000/03000111100000000/33300001000088880/03000001000080080/00000000000088880/00000000000000000/00000000000000000/00000000000000000",
    "000000000000000/000000000000200/000333000022222/080303000000200/888333000000200/080000000000000/080011110011111/000010010010001/000011110010001/000000000010001/020000000000000/222000000000000/020000000000000/000000000000000/000000000000000/000000000000000",
)


def _counts(rng, seed, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, (int, np.integer)) or
                                  not lo <= int(value) <= hi):
            raise ValueError(name + " is outside its legal range")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lower = max(2, num_test + 1)
        num_train = lower + (seed // 4) % (11 - lower)
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _trim(mask):
    yy, xx = np.nonzero(mask)
    return mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1].copy()


def _hole_count(mask):
    """Count four-connected background components enclosed by the mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            exterior = False
            while stack:
                cy, cx = stack.pop()
                exterior |= cy in (0, h - 1) or cx in (0, w - 1)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < h and 0 <= nx < w and
                            not mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            count += not exterior
    return count


def _connected(mask):
    cells = np.argwhere(mask)
    if not len(cells):
        return False
    start = tuple(int(v) for v in cells[0])
    seen = {start}
    stack = [start]
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            key = (y + dy, x + dx)
            if (0 <= key[0] < mask.shape[0] and
                    0 <= key[1] < mask.shape[1] and
                    mask[key] and key not in seen):
                seen.add(key)
                stack.append(key)
    return len(seen) == len(cells)


def _hollow_shape(rng, scale):
    hi = 5 if scale == "small" else 7
    h = int(rng.integers(3, hi + 1))
    w = int(rng.integers(3, hi + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    kind = int(rng.integers(0, 5))
    if kind == 1 and w >= 5:
        mask[1:-1, int(rng.integers(2, w - 2))] = True
    elif kind == 2 and h >= 5:
        mask[int(rng.integers(2, h - 2)), 1:-1] = True
    elif kind == 3 and h >= 4 and w >= 4:
        side = int(rng.integers(0, 4))
        padded = np.pad(mask, 1)
        if side == 0:
            padded[0, 2:2 + int(rng.integers(1, min(3, w)))] = True
        elif side == 1:
            padded[-1, 2:2 + int(rng.integers(1, min(3, w)))] = True
        elif side == 2:
            padded[2:2 + int(rng.integers(1, min(3, h))), 0] = True
        else:
            padded[2:2 + int(rng.integers(1, min(3, h))), -1] = True
        mask = padded
    elif kind == 4 and h >= 5 and w >= 5:
        # An interior peninsula gives a nonrectangular enclosed region.
        mask[1:int(rng.integers(2, h - 1)), int(rng.integers(1, w - 1))] = True
    mask = _trim(mask)
    assert _connected(mask) and _hole_count(mask) >= 1
    return mask


def _holeless_shape(rng, scale, make_open=False):
    limit = 5 if scale == "small" else 7
    for _ in range(30):
        kind = 4 if make_open else int(rng.integers(0, 6))
        h = int(rng.integers(3, limit + 1))
        w = int(rng.integers(3, limit + 1))
        mask = np.zeros((h, w), dtype=bool)
        if kind == 0:  # cross or off-centre T
            cy, cx = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
            mask[cy, :] = True
            mask[:, cx] = True
            if rng.random() < .5:
                mask[cy + 1:, cx] = False
        elif kind == 1:  # L
            mask[:, 0] = True
            mask[-1, :] = True
        elif kind == 2:  # straight or thick bar
            mask[:, int(rng.integers(0, w))] = True
            if rng.random() < .5:
                mask[int(rng.integers(0, h)), :] = True
        elif kind == 3:  # stair or connected branch
            for y in range(h):
                x = min(w - 1, (y * w) // h)
                mask[y, x:min(w, x + 2)] = True
        else:  # almost closed loop, with one or two openings
            mask[0, :] = mask[-1, :] = True
            mask[:, 0] = mask[:, -1] = True
            gap = int(rng.integers(1, w - 1))
            mask[0, gap] = False
            if kind == 4 and w >= 5 and rng.random() < .5:
                neighbor = gap + (1 if gap < w - 2 else -1)
                mask[0, neighbor] = False
            if kind == 5 and h >= 4:
                mask[-1, int(rng.integers(1, w - 1))] = False
                mask[int(rng.integers(1, h - 1)), :] = True
        mask = _trim(mask)
        mask = np.rot90(mask, int(rng.integers(0, 4))).copy()
        if rng.random() < .5:
            mask = np.fliplr(mask).copy()
        if _connected(mask) and _hole_count(mask) == 0:
            return mask
    raise AssertionError("could not construct a holeless shape")


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    assert counts[index] > grid.size // 2
    return int(colors[index])


def _objects(grid, background):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for y in range(h):
        for x in range(w):
            color = int(grid[y, x])
            if color == background or seen[y, x]:
                continue
            cells = []
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < h and 0 <= nx < w and not seen[ny, nx]
                            and int(grid[ny, nx]) == color):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            a = np.array(cells, dtype=np.int16)
            top, left = a.min(axis=0)
            bottom, right = a.max(axis=0)
            mask = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
            mask[a[:, 0] - top, a[:, 1] - left] = True
            objects.append({"id": len(objects), "color": color, "cells": a,
                            "mask": mask, "top": int(top), "bottom": int(bottom),
                            "left": int(left), "right": int(right),
                            "hollow": _hole_count(mask) > 0})
    return objects


def _pack(objects, height, direction):
    if direction == -1:
        ordered = sorted(objects, key=lambda o: (o["top"], o["left"], o["id"]))
    else:
        ordered = sorted(objects, key=lambda o: (-o["bottom"], o["left"], o["id"]))
    occupied = set()
    placed = []
    for obj in ordered:
        dy = 0
        while True:
            candidate = obj["cells"] + np.array([dy + direction, 0])
            if (candidate[:, 0].min() < 0 or candidate[:, 0].max() >= height or
                    any((int(y), int(x)) in occupied for y, x in candidate)):
                break
            dy += direction
        cells = obj["cells"] + np.array([dy, 0])
        for y, x in cells:
            key = int(y), int(x)
            assert key not in occupied
            occupied.add(key)
        placed.append({"id": obj["id"], "color": obj["color"],
                       "cells": cells, "dy": dy})
    return placed


def _transform(grid):
    bg = _background(grid)
    objects = _objects(grid, bg)
    hollow = [o for o in objects if o["hollow"]]
    solid = [o for o in objects if not o["hollow"]]
    up = _pack(hollow, grid.shape[0], -1)
    down = _pack(solid, grid.shape[0], 1)
    output = np.full(grid.shape, bg, dtype=np.int8)
    for obj in up + down:
        for y, x in obj["cells"]:
            assert output[y, x] == bg
            output[y, x] = obj["color"]
    return output, objects, up + down


def _stats(grid):
    output, objects, placed = _transform(grid)
    final = {o["id"]: o for o in placed}
    hollow = [o for o in objects if o["hollow"]]
    solid = [o for o in objects if not o["hollow"]]
    top = any(final[o["id"]]["cells"][:, 0].min() > 0 for o in hollow)
    bottom = any(final[o["id"]]["cells"][:, 0].max() < grid.shape[0] - 1
                 for o in solid)
    moved_up = any(final[o["id"]]["dy"] < 0 for o in hollow)
    moved_down = any(final[o["id"]]["dy"] > 0 for o in solid)
    return {"output": output, "objects": objects, "hollow": hollow, "solid": solid,
            "placed": placed, "top": top, "bottom": bottom,
            "moved_up": moved_up, "moved_down": moved_down,
            "stationary": any(final[o["id"]]["dy"] == 0 for o in objects)}


def _groups(count, congested, rng):
    indices = list(map(int, rng.permutation(count)))
    if not congested:
        return [[i] for i in indices]
    groups = []
    while indices:
        if len(indices) >= 4 and rng.random() < .4:
            size = 2
        elif len(indices) >= 3 and rng.random() < .4:
            size = 3
        else:
            size = 2 if len(indices) >= 2 else 1
        groups.append(indices[:size])
        indices = indices[size:]
    return groups


def _x_positions(masks, groups, width, rng):
    group_data = []
    for group in groups:
        offsets = [0]
        for _ in group[1:]:
            offsets.append(int(rng.integers(0, 3)))
        span = max(offset + masks[i].shape[1] for i, offset in zip(group, offsets))
        group_data.append((group, offsets, span))
    rng.shuffle(group_data)
    required = sum(item[2] for item in group_data) + len(group_data) - 1
    if required > width:
        return None
    gaps = np.zeros(len(group_data) + 1, dtype=int)
    if len(group_data) > 1:
        gaps[1:-1] = 1
    for _ in range(width - required):
        gaps[int(rng.integers(0, len(gaps)))] += 1
    positions = [0] * len(masks)
    x = int(gaps[0])
    for j, (group, offsets, span) in enumerate(group_data):
        for i, offset in zip(group, offsets):
            positions[i] = x + offset
        x += span + int(gaps[j + 1])
    return positions


def _can_place(occupied, cells, y, x):
    h, w = occupied.shape
    for cy, cx in cells:
        yy, xx = y + int(cy), x + int(cx)
        if not (0 <= yy < h and 0 <= xx < w) or occupied[yy, xx]:
            return False
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = yy + dy, xx + dx
            if 0 <= ny < h and 0 <= nx < w and occupied[ny, nx]:
                return False
    return True


def _input_layout(specs, height, width, background, rng, edge_start):
    occupied = np.zeros((height, width), dtype=bool)
    positions = [None] * len(specs)
    order = list(map(int, rng.permutation(len(specs))))
    if edge_start is not None:
        preferred = [i for i, obj in enumerate(specs) if obj["hollow"] == edge_start]
        if preferred:
            first = preferred[int(rng.integers(0, len(preferred)))]
            order.remove(first)
            order.insert(0, first)
    for k, i in enumerate(order):
        obj = specs[i]
        max_y = height - obj["mask"].shape[0]
        if max_y < 0:
            return None
        candidates = list(map(int, rng.permutation(max_y + 1)))
        if k == 0 and edge_start is not None:
            edge_y = 0 if obj["hollow"] else max_y
            candidates.remove(edge_y)
            candidates.insert(0, edge_y)
        elif rng.random() < .65:
            # Mixed initial order provides visible crossing in some pairs.
            candidates.sort(key=lambda y: (-y if obj["hollow"] else y,
                                            abs(y - height // 2)))
        placed = False
        for y in candidates:
            if _can_place(occupied, obj["cells"], y, obj["x"]):
                rows = y + obj["cells"][:, 0]
                cols = obj["x"] + obj["cells"][:, 1]
                occupied[rows, cols] = True
                positions[i] = y
                placed = True
                break
        if not placed:
            return None
    grid = np.full((height, width), background, dtype=np.int8)
    for obj, y in zip(specs, positions):
        rows = y + obj["cells"][:, 0]
        cols = obj["x"] + obj["cells"][:, 1]
        grid[rows, cols] = obj["color"]
    return grid


def _source_copy(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS


def _near_open_frame(mask):
    h, w = mask.shape
    return h >= 3 and w >= 3 and _hole_count(mask) == 0 and int(mask.sum()) >= 2 * h + 2 * w - 6


def _pair(rng, regime, palette, desired_top, desired_bottom, color_phase):
    target = regime == "target_test"
    for attempt in range(180):
        count = int(rng.integers(4, 8 if not target else 9))
        nh = int(rng.integers(1, count))
        ns = count - nh
        if desired_top and nh < 2 or desired_bottom and ns < 2:
            continue
        height = int(rng.integers(12, 24 if not target else 27))
        width = int(rng.integers(11, 27 if not target else 29))
        if count >= 7 and width < 17:
            continue
        scale = "small" if height <= 15 or rng.random() < (.68 if not target else .48) else "large"
        hollow_masks = [_hollow_shape(rng, scale) for _ in range(nh)]
        solid_masks = [_holeless_shape(rng, scale, make_open=(i == 0 and
                       (color_phase == 0 or rng.random() < .25)))
                       for i in range(ns)]
        hx = _x_positions(hollow_masks, _groups(nh, desired_top, rng), width, rng)
        sx = _x_positions(solid_masks, _groups(ns, desired_bottom, rng), width, rng)
        if hx is None or sx is None:
            continue
        background_options = [0] + [v for v in range(1, 10) if v not in palette]
        background = int(background_options[int(rng.integers(0, len(background_options)))])
        specs = []
        for hollow, masks, xs in ((True, hollow_masks, hx), (False, solid_masks, sx)):
            for i, (mask, x) in enumerate(zip(masks, xs)):
                if i == 0 and color_phase in (0, 1):
                    color = palette[(color_phase + (not hollow)) % 2]
                else:
                    color = palette[int(rng.integers(0, len(palette)))]
                specs.append({"mask": mask, "cells": np.argwhere(mask).astype(np.int16),
                              "x": x, "color": color, "hollow": hollow})
        edge_start = None
        if rng.random() < .23:
            edge_start = bool(rng.integers(0, 2))
        grid = _input_layout(specs, height, width, background, rng, edge_start)
        if grid is None or _source_copy(grid):
            continue
        colors, counts = np.unique(grid, return_counts=True)
        if int(counts[np.where(colors == background)[0][0]]) <= grid.size // 2:
            continue
        try:
            stats = _stats(grid)
        except AssertionError:
            continue
        if (len(stats["objects"]) != count or len(stats["hollow"]) != nh or
                len(stats["solid"]) != ns or stats["top"] != desired_top or
                stats["bottom"] != desired_bottom or
                not stats["moved_up"] or not stats["moved_down"] or
                np.array_equal(grid, stats["output"])):
            continue
        return {"input": grid, "output": stats["output"]}
    raise ValueError("could not construct a valid pair")


def _signature(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    palette = tuple(int(v) for v in rng.choice(np.arange(1, 10),
                                               size=int(rng.integers(3, 7)),
                                               replace=False))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for i in range(count):
            if split == "train" and i == 0:
                top, bottom = True, False
            elif split == "train" and i == 1:
                top, bottom = False, True
            else:
                modes = ((False, False), (True, False), (False, True), (True, True))
                weights = ([.24, .23, .23, .30] if split == "train"
                           else [.13, .18, .18, .51])
                top, bottom = modes[int(rng.choice(4, p=weights))]
            for _ in range(8):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                try:
                    pair = _pair(pair_rng, "source_train" if split == "train"
                                 else "target_test", palette, top, bottom,
                                 i if split == "train" and i < 2 else -1)
                except ValueError:
                    continue
                signature = _signature(pair["input"])
                if signature not in used:
                    used.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.all((0 <= grid) & (grid <= 9))


def _separated(grid, background):
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            if grid[y, x] == background:
                continue
            for dy, dx in ((0, 1), (1, 0)):
                ny, nx = y + dy, x + dx
                if (ny < h and nx < w and grid[ny, nx] != background and
                        grid[ny, nx] != grid[y, x]):
                    return False
    return True


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    inputs, pairs = set(), set()
    train_stats, test_stats = [], []
    train_colors = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _signature(inp) not in inputs
            signature = (_signature(inp), _signature(out))
            assert signature not in pairs and not _source_copy(inp)
            inputs.add(_signature(inp))
            pairs.add(signature)
            bg = _background(inp)
            assert _separated(inp, bg)
            stats = _stats(inp)
            assert np.array_equal(out, stats["output"])
            assert len(stats["objects"]) >= 4
            assert stats["hollow"] and stats["solid"]
            assert stats["moved_up"] and stats["moved_down"]
            assert all(_connected(o["mask"]) for o in stats["objects"])
            assert all(_hole_count(o["mask"]) >= 1 for o in stats["hollow"])
            assert all(_hole_count(o["mask"]) == 0 for o in stats["solid"])
            if split == "train":
                train_stats.append(stats)
                for o in stats["objects"]:
                    train_colors.add(o["color"])
            else:
                test_stats.append(stats)
    # Both collision directions and both topological directions are demonstrated.
    assert any(s["top"] for s in train_stats)
    assert any(s["bottom"] for s in train_stats)
    train_hollow_colors = {o["color"] for s in train_stats for o in s["hollow"]}
    train_solid_colors = {o["color"] for s in train_stats for o in s["solid"]}
    assert train_hollow_colors & train_solid_colors
    assert all(s["moved_up"] and s["moved_down"] for s in test_stats)
    assert any(_near_open_frame(o["mask"]) for s in train_stats for o in s["solid"])


def _collision_features(stats):
    originals = {o["id"]: o for o in stats["objects"]}
    placed = {o["id"]: o for o in stats["placed"]}
    links = []
    class_sizes = []
    for group, direction in ((stats["hollow"], -1), (stats["solid"], 1)):
        cells_to_id = {(int(y), int(x)): o["id"] for o in group
                       for y, x in placed[o["id"]]["cells"]}
        parent = {o["id"]: o["id"] for o in group}

        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for o in group:
            for y, x in placed[o["id"]]["cells"]:
                other = cells_to_id.get((int(y) + direction, int(x)))
                if other is not None and other != o["id"]:
                    links.append((o["id"], other))
                    parent[root(o["id"])] = root(other)
        sizes = {}
        for i in parent:
            r = root(i)
            sizes[r] = sizes.get(r, 0) + 1
        class_sizes.append(list(sizes.values()))
    return {
        "offset": any(originals[a]["left"] != originals[b]["left"] for a, b in links),
        "unequal": any(originals[a]["mask"].shape[1] !=
                       originals[b]["mask"].shape[1] for a, b in links),
        "chain": any(n >= 3 for sizes in class_sizes for n in sizes),
        "two_stacks": any(sum(n > 1 for n in sizes) >= 2 for sizes in class_sizes),
        "stack_and_disjoint": any(any(n > 1 for n in sizes) and 1 in sizes
                                  for sizes in class_sizes),
    }


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and
        np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(123456, 6, 3)
    assert _equal(a, generate_puzzle(123456, 6, 3))
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4),
                                  (None, 1), (None, 4),
                                  (2, None), (10, None))):
        p = generate_puzzle(20000 + i, nt, ne)
        validate_generated_puzzle(p)
        if nt is not None:
            assert len(p["train"]) == nt
        if ne is not None:
            assert len(p["test"]) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 4)):
        try:
            generate_puzzle(31, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzles = [generate_puzzle(i) for i in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    dimensions, layouts, shapes, backgrounds, counts, modes = (
        set(), set(), set(), set(), set(), set())
    seen_inputs = set()
    measured = {}
    vertical_gaps = set()

    def record(name, condition):
        if condition:
            measured[name] = measured.get(name, 0) + 1

    for p in puzzles:
        validate_generated_puzzle(p)
        record("backgrounds_within", len({_background(q["input"])
                                          for q in p["train"] + p["test"]}) > 1)
        record("counts_within", len({len(_stats(q["input"])["objects"])
                                      for q in p["train"] + p["test"]}) > 1)
        for split in ("train", "test"):
            for pair in p[split]:
                inp = pair["input"]
                stats = _stats(inp)
                assert np.array_equal(stats["output"], pair["output"])
                collision = _collision_features(stats)
                final = {o["id"]: o for o in stats["placed"]}
                up_bottom = max(final[o["id"]]["cells"][:, 0].max()
                                for o in stats["hollow"])
                down_top = min(final[o["id"]]["cells"][:, 0].min()
                               for o in stats["solid"])
                vertical_gaps.add(int(down_top - up_bottom - 1))
                crossing = any(
                    h["top"] > s["bottom"] and
                    bool(set(h["cells"][:, 1]) & set(s["cells"][:, 1])) and
                    final[h["id"]]["cells"][:, 0].max() <
                    final[s["id"]]["cells"][:, 0].min()
                    for h in stats["hollow"] for s in stats["solid"])
                record("crossing", crossing)
                record("no_crossing", not crossing)
                record(split + "_mode_" + str(int(stats["top"])) +
                       str(int(stats["bottom"])), True)
                record("four_objects", len(stats["objects"]) == 4)
                record("stationary", stats["stationary"])
                record("shared_color", bool({o["color"] for o in stats["hollow"]} &
                                            {o["color"] for o in stats["solid"]}))
                record("disjoint_color", not ({o["color"] for o in stats["hollow"]} &
                                              {o["color"] for o in stats["solid"]}))
                record(split + "_large_hollow", any(max(o["mask"].shape) >= 5
                      for o in stats["hollow"]))
                record(split + "_has_5x5", any(o["mask"].shape == (5, 5)
                      for o in stats["hollow"]))
                record(split + "_no_5x5", not any(o["mask"].shape == (5, 5)
                      for o in stats["hollow"]))
                record("multiple_holes", any(_hole_count(o["mask"]) >= 2
                      for o in stats["hollow"]))
                record("irregular_hollow", any(not (
                    o["mask"][0, :].all() and o["mask"][-1, :].all() and
                    o["mask"][:, 0].all() and o["mask"][:, -1].all())
                    for o in stats["hollow"]))
                for obj in stats["solid"]:
                    mask = obj["mask"]
                    if not _near_open_frame(mask):
                        continue
                    h, w = mask.shape
                    border = ([(0, x) for x in range(w)] +
                              [(y, w - 1) for y in range(1, h)] +
                              [(h - 1, x) for x in range(w - 2, -1, -1)] +
                              [(y, 0) for y in range(h - 2, 0, -1)])
                    missing = [j for j, (y, x) in enumerate(border) if not mask[y, x]]
                    record("one_gap_cell", len(missing) == 1)
                    record("two_separate_gaps", len(missing) == 2 and
                           (missing[1] - missing[0]) > 1)
                    record("two_adjacent_gap_cells", len(missing) == 2 and
                           (missing[1] - missing[0]) == 1)
                    for j in missing:
                        y, x = border[j]
                        side = "top" if y == 0 else ("bottom" if y == h - 1 else
                               ("left" if x == 0 else "right"))
                        record("gap_" + side, True)
                for name, present in collision.items():
                    record(name, present)
                u = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=bool)
                record("no_exact_u", not any(any(
                    o["mask"].shape == np.rot90(u, turns).shape and
                    np.array_equal(o["mask"], np.rot90(u, turns))
                    for turns in range(4)) for o in stats["solid"]))
                seen_inputs.add(_signature(inp))
                dimensions.add(inp.shape)
                backgrounds.add(_background(inp))
                counts.add(len(stats["objects"]))
                modes.add((split, stats["top"], stats["bottom"]))
                layouts.add((inp.shape, len(stats["hollow"]), len(stats["solid"]),
                             stats["top"], stats["bottom"]))
                for obj in stats["objects"]:
                    shapes.add((obj["hollow"], obj["mask"].shape,
                                obj["mask"].tobytes()))
    assert len(seen_inputs) >= 170
    assert len(dimensions) >= 35 and len(layouts) >= 60
    assert len(shapes) >= 40 and len(backgrounds) >= 3
    assert len(counts) >= 4 and len(modes) >= 7
    assert len(vertical_gaps) >= 10 and min(vertical_gaps) <= 0 and max(vertical_gaps) >= 8
    assert all(measured.get(split + "_mode_" + mode, 0) >= 5
               for split in ("train", "test") for mode in ("00", "01", "10", "11"))
    for name, minimum in (("four_objects", 30), ("stationary", 30),
                          ("crossing", 30), ("no_crossing", 10),
                          ("shared_color", 40), ("disjoint_color", 20),
                          ("train_large_hollow", 40), ("test_large_hollow", 10),
                          ("train_has_5x5", 10), ("test_no_5x5", 10),
                          ("multiple_holes", 20), ("irregular_hollow", 15),
                          ("offset", 40), ("unequal", 40), ("chain", 10),
                          ("two_stacks", 10), ("stack_and_disjoint", 10),
                          ("no_exact_u", 30), ("backgrounds_within", 25),
                          ("counts_within", 25), ("one_gap_cell", 40),
                          ("two_separate_gaps", 10),
                          ("two_adjacent_gap_cells", 10),
                          ("gap_top", 10), ("gap_bottom", 10),
                          ("gap_left", 10), ("gap_right", 10)):
        assert measured.get(name, 0) >= minimum, (name, measured.get(name, 0))
    train_total = sum(len(p["train"]) for p in puzzles)
    test_total = sum(len(p["test"]) for p in puzzles)
    assert measured["test_mode_11"] / test_total > measured["train_mode_11"] / train_total
