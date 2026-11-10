"""Compositional four-neighbor enclosure puzzles."""

import hashlib

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

# An exact input match is enough to exclude a copied source pair.
# Hashes include the two dimensions followed by the int8 grid bytes.
SOURCE_INPUTS = {
    "88ad7f5b41baf5da74a9847436b4b8f8aeb8242af71496123ce6620f5d03f020",
    "b37f69d6678ca3a968ce59547458e8857a97373b9f85ef0b20a5eb399aa6f6fe",
    "4aafc188c7f689baea60f40fcb2ede5f81f2c02e7fb8f1b1f71478b02e848a9b",
    "7c47fcf0d734d00e4087028cc7c3ae0e1c7e7f4c29ec1e5e9673b7769e2fa0a1",
    "48951719fc19fe074f61cc39f7b8e0a30085fdae13edfb5bda798ac50fc347a2",
    "20ec9fb597be6d7e88021ab6db5654a19c6991b3439e29dda654d44b64a36cd4",
}


def _source_key(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.int8).tobytes()).hexdigest()


def _components(mask, neighbors=N4):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells = set()
            while stack:
                y, x = stack.pop()
                cells.add((y, x))
                for dy, dx in neighbors:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            result.append(cells)
    return result


def _enclosed_mask(grid, background, neighbors=N4):
    h, w = grid.shape
    free = grid == background
    reached = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if free[r, c] and not reached[r, c]:
                reached[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if free[r, c] and not reached[r, c]:
                reached[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and not reached[rr, cc]:
                reached[rr, cc] = True
                stack.append((rr, cc))
    return free & ~reached


def _fill_enclosed(grid, background, fill):
    out = grid.copy()
    out[_enclosed_mask(grid, background)] = fill
    return out


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    bounds = (2, 10) if name == "num_train" else (1, 4)
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} outside legal range")
    return value


def _resolve_counts(rng, num_train, num_test):
    nt, ns = _count(num_train, "num_train"), _count(num_test, "num_test")
    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(ns + 1, 11))
    elif nt is None:
        nt = int(rng.integers(ns + 1, 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _room_shape(rng, size_class, limit):
    """Return an orthogonally connected interior, with independently sampled extent."""
    if size_class == "dot":
        return {(0, 0)}
    if size_class == "line":
        length = int(rng.integers(2, min(10, limit) + 1))
        shape = {(0, c) for c in range(length)}
    else:
        upper = min(9, limit)
        if size_class == "large":
            low = min(4, upper)
            a = int(rng.integers(low, upper + 1))
            b = int(rng.integers(min(3, upper), upper + 1))
        else:
            a = int(rng.integers(2, min(7, upper) + 1))
            b = int(rng.integers(2, min(7, upper) + 1))
        motif = int(rng.integers(4))
        if motif == 0:
            shape = {(r, c) for r in range(a) for c in range(b)}
        elif motif == 1:  # bent, occasionally thick corridor
            thick = int(rng.integers(1, min(3, a, b) + 1))
            shape = {(r, c) for r in range(a) for c in range(thick)}
            shape |= {(r, c) for r in range(a - thick, a) for c in range(b)}
        elif motif == 2:  # concave notch
            shape = {(r, c) for r in range(a) for c in range(b)}
            notch_h = int(rng.integers(1, max(2, a // 2 + 1)))
            notch_w = int(rng.integers(1, max(2, b // 2 + 1)))
            shape -= {(r, c) for r in range(notch_h) for c in range(b - notch_w, b)}
        else:  # a solid barrier island within a larger chamber
            shape = {(r, c) for r in range(a) for c in range(b)}
            if a >= 4 and b >= 4:
                ir = int(rng.integers(1, a - 1))
                ic = int(rng.integers(1, b - 1))
                shape.remove((ir, ic))
    if int(rng.integers(2)):
        shape = {(c, r) for r, c in shape}
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    if int(rng.integers(2)):
        shape = {(h - 1 - r, c) for r, c in shape}
    if int(rng.integers(2)):
        shape = {(r, w - 1 - c) for r, c in shape}
    return shape


def _room_geometry(shape, top, left, full_corners):
    inside = {(top + r, left + c) for r, c in shape}
    wall = set()
    for r, c in inside:
        for dr, dc in N8 if full_corners else N4:
            p = (r + dr, c + dc)
            if p not in inside:
                wall.add(p)
    return inside, wall


def _candidate_positions(rng, h, w, shape, rooms, shared):
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    positions = []
    if shared and rooms:
        order = rng.permutation(len(rooms))
        for index in order:
            old = rooms[int(index)]
            lo_r = min(r for r, _ in old)
            hi_r = max(r for r, _ in old)
            lo_c = min(c for _, c in old)
            hi_c = max(c for _, c in old)
            for _ in range(25):
                direction = int(rng.integers(4))
                if direction == 0:
                    positions.append((lo_r + int(rng.integers(-sh + 1, hi_r - lo_r + 1)), hi_c + 2))
                elif direction == 1:
                    positions.append((lo_r + int(rng.integers(-sh + 1, hi_r - lo_r + 1)), lo_c - sw - 1))
                elif direction == 2:
                    positions.append((hi_r + 2, lo_c + int(rng.integers(-sw + 1, hi_c - lo_c + 1))))
                else:
                    positions.append((lo_r - sh - 1, lo_c + int(rng.integers(-sw + 1, hi_c - lo_c + 1))))
    for _ in range(90):
        positions.append((int(rng.integers(1, max(2, h - sh))), int(rng.integers(1, max(2, w - sw)))))
    return positions


def _place_room(rng, grid, barrier, shape, rooms, interior, full_corners, shared):
    h, w = grid.shape
    old_wall = set(zip(*np.where(grid == barrier)))
    for top, left in _candidate_positions(rng, h, w, shape, rooms, shared):
        new_inside, wall = _room_geometry(shape, top, left, full_corners)
        if any(r < 1 or r >= h - 1 or c < 1 or c >= w - 1 for r, c in new_inside):
            continue
        if any(r < 0 or r >= h or c < 0 or c >= w for r, c in wall):
            continue
        if new_inside & (old_wall | interior) or wall & interior:
            continue
        if shared and rooms and not (wall & old_wall):
            continue
        for r, c in wall:
            grid[r, c] = barrier
        rooms.append(new_inside)
        interior.update(new_inside)
        return True
    return False


def _open_u(rng, grid, background, barrier, protected):
    h, w = grid.shape
    for _ in range(30):
        rh = int(rng.integers(3, min(8, h - 1) + 1))
        rw = int(rng.integers(3, min(8, w - 1) + 1))
        top = int(rng.integers(0, h - rh + 1))
        left = int(rng.integers(0, w - rw + 1))
        # A rectangle with exactly one orthogonal gate; random rotation follows.
        cells = {(top, c) for c in range(left, left + rw)}
        cells |= {(top + rh - 1, c) for c in range(left, left + rw)}
        cells |= {(r, left) for r in range(top, top + rh)}
        cells |= {(r, left + rw - 1) for r in range(top, top + rh)}
        gate = (top, left + int(rng.integers(1, rw - 1)))
        cells.remove(gate)
        if all(grid[r, c] == background and (r, c) not in protected for r, c in cells):
            if all(grid[r, c] == background for r in range(top + 1, top + rh - 1)
                   for c in range(left + 1, left + rw - 1)):
                for r, c in cells:
                    grid[r, c] = barrier
                return True
    return False


def _branches(rng, grid, background, barrier, protected):
    h, w = grid.shape
    walls = list(zip(*np.where(grid == barrier)))
    for _ in range(int(rng.integers(1, 5))):
        if walls and rng.random() < 0.65:
            r, c = walls[int(rng.integers(len(walls)))]
        else:
            r, c = int(rng.integers(h)), int(rng.integers(w))
        direction = int(rng.integers(4))
        for _ in range(int(rng.integers(3, min(16, max(h, w)) + 1))):
            if rng.random() < 0.3:
                direction = (direction + (1 if int(rng.integers(2)) else -1)) % 4
            dr, dc = N4[direction]
            r, c = r + dr, c + dc
            if not (0 <= r < h and 0 <= c < w) or (r, c) in protected:
                break
            if grid[r, c] == background:
                grid[r, c] = barrier


def _thicken_wall(rng, grid, background, barrier, protected):
    walls = list(zip(*np.where(grid == barrier)))
    h, w = grid.shape
    for _ in range(30):
        r, c = walls[int(rng.integers(len(walls)))]
        dr, dc = N4[int(rng.integers(4))]
        if not (0 <= r + dr < h and 0 <= c + dc < w):
            continue
        if grid[r + dr, c + dc] != barrier:
            continue
        for side in (-1, 1):
            sr, sc = -dc * side, dr * side
            thick = ((r + sr, c + sc), (r + dr + sr, c + dc + sc))
            if all(0 <= y < h and 0 <= x < w and grid[y, x] == background
                   and (y, x) not in protected for y, x in thick):
                for y, x in thick:
                    grid[y, x] = barrier
                return


def _edge_divider(rng, grid, background, barrier, protected):
    h, w = grid.shape
    choices = []
    for r in range(h):
        if all((r, c) not in protected for c in range(w)):
            choices.append(("row", r))
    for c in range(w):
        if all((r, c) not in protected for r in range(h)):
            choices.append(("col", c))
    if choices:
        axis, k = choices[int(rng.integers(len(choices)))]
        if axis == "row":
            grid[k, :] = barrier
        else:
            grid[:, k] = barrier


def _dimensions(rng, regime):
    tier = int(rng.choice(3, p=(0.25, 0.45, 0.30) if regime == "source_train"
                          else (0.12, 0.40, 0.48)))
    ranges = ((6, 12), (11, 21), (18, 30))
    low, high = ranges[tier]
    h = int(rng.integers(low, high + 1))
    w = int(rng.integers(low, high + 1))
    if rng.random() < 0.28:
        if rng.random() < 0.5:
            h = int(rng.integers(6, 31))
        else:
            w = int(rng.integers(6, 31))
    return h, w


def _generate_pair(rng, regime, fill, palette, mode):
    background, barrier = palette
    for _ in range(12):
        h, w = _dimensions(rng, regime)
        grid = np.full((h, w), background, dtype=np.int8)
        rooms, interior = [], set()
        capacity = max(1, min(8, (h * w) // 52))
        if regime == "target_test" and rng.random() < 0.65:
            target = int(rng.integers(max(1, capacity // 2), capacity + 1))
        else:
            target = int(rng.integers(1, capacity + 1))
        target = max(1, target)
        size_classes = [mode]
        size_classes += [("dot", "line", "medium", "large")[int(rng.integers(4))]
                         for _ in range(target - 1)]
        rng.shuffle(size_classes)
        for size_class in size_classes:
            if size_class == "large" and min(h, w) < 12:
                size_class = "medium"
            if size_class == "medium" and min(h, w) < 8:
                size_class = "line"
            placed = False
            for attempt in range(5):
                shape = _room_shape(rng, size_class if attempt < 3 else "dot",
                                    max(2, min(h, w) - 2))
                shared = bool(rooms) and (rng.random() < 0.65)
                full = size_class != "dot" and rng.random() < 0.55
                if _place_room(rng, grid, barrier, shape, rooms, interior, full, shared):
                    placed = True
                    break
                if shared and _place_room(rng, grid, barrier, shape, rooms, interior, full, False):
                    placed = True
                    break
            if not placed and not rooms:
                break
        if not rooms:
            continue
        if rng.random() < 0.55:
            _open_u(rng, grid, background, barrier, interior)
        if rng.random() < 0.35:
            _thicken_wall(rng, grid, background, barrier, interior)
        if rng.random() < 0.86:
            _branches(rng, grid, background, barrier, interior)
        if rng.random() < 0.27:
            _edge_divider(rng, grid, background, barrier, interior)
        mask = _enclosed_mask(grid, background)
        if not mask.any():
            continue
        if np.count_nonzero(grid == background) <= np.count_nonzero(grid == barrier):
            continue
        if _source_key(grid) in SOURCE_INPUTS:
            continue
        output = grid.copy()
        output[mask] = fill
        return grid, output
    raise RuntimeError("could not construct a pair")


def _generate_open_pair(rng, palette):
    background, barrier = palette
    for _ in range(12):
        h, w = _dimensions(rng, "source_train")
        grid = np.full((h, w), background, dtype=np.int8)
        if not _open_u(rng, grid, background, barrier, set()):
            continue
        if rng.random() < 0.5:
            _branches(rng, grid, background, barrier, set())
        if not _enclosed_mask(grid, background).any() and np.count_nonzero(grid == background) > np.count_nonzero(grid == barrier):
            return grid, grid.copy()
    raise RuntimeError("could not construct an open scene")


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _independent_fill(grid, background, fill):
    """Component-based oracle used separately from the border flood."""
    result = grid.copy()
    h, w = grid.shape
    for component in _components(grid == background):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in component):
            for r, c in component:
                result[r, c] = fill
    return result


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    colors = [int(x) for x in rng.permutation(10)]
    fill = colors[0]
    base_palette = (colors[1], colors[2])
    variable_palette = bool(rng.random() < 0.65)
    puzzle = {"train": [], "test": []}
    seen = set()
    # The order of essential demonstrations changes with the seed.
    modes = ["dot", "large"] + [
        ("line", "medium", "large", "dot")[int(rng.integers(4))]
        for _ in range(num_train - 2)
    ]
    if num_train >= 3 and rng.random() < 0.35:
        modes[int(rng.integers(2, num_train))] = "open"
    rng.shuffle(modes)
    all_modes = modes + [
        ("dot", "line", "medium", "large")[int(rng.integers(4))]
        for _ in range(num_test)
    ]
    for index, mode in enumerate(all_modes):
        split = "train" if index < num_train else "test"
        if variable_palette and index > 0:
            options = [x for x in range(10) if x != fill]
            chosen = rng.choice(options, size=2, replace=False)
            palette = int(chosen[0]), int(chosen[1])
        else:
            palette = base_palette
        for _ in range(16):
            local_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            local_rng = np.random.default_rng(local_seed)
            try:
                if mode == "open":
                    pair = _generate_open_pair(local_rng, palette)
                else:
                    pair = _generate_pair(local_rng, "source_train" if split == "train"
                                          else "target_test", fill, palette, mode)
            except RuntimeError:
                continue
            key = _grid_key(pair[0])
            if key in seen:
                continue
            seen.add(key)
            puzzle[split].append({"input": pair[0], "output": pair[1]})
            break
        else:
            raise ValueError("could not generate enough distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, fills = set(), set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
        assert inp.ndim == out.ndim == 2 and inp.shape == out.shape
        assert all(1 <= x <= 30 for x in inp.shape)
        assert np.issubdtype(inp.dtype, np.integer) and np.issubdtype(out.dtype, np.integer)
        assert np.all((0 <= inp) & (inp <= 9)) and np.all((0 <= out) & (out <= 9))
        key = _grid_key(inp)
        assert key not in seen
        seen.add(key)
        assert _source_key(inp) not in SOURCE_INPUTS
        values, counts = np.unique(inp, return_counts=True)
        assert len(values) == 2
        background = int(values[np.argmax(counts)])
        barrier = int(values[np.argmin(counts)])
        assert background != barrier
        changed = inp != out
        if changed.any():
            assert np.all(inp[changed] == background)
            new = np.unique(out[changed])
            assert len(new) == 1
            fill = int(new[0])
            assert fill not in (background, barrier)
            fills.add(fill)
            assert np.array_equal(out, _fill_enclosed(inp, background, fill))
        else:
            assert not _enclosed_mask(inp, background).any()
        assert np.array_equal(out[inp == barrier], inp[inp == barrier])
    assert len(fills) == 1
    shared_fill = next(iter(fills))
    assert all(shared_fill not in {int(x) for x in np.unique(p["input"])}
               for p in train + test)
    train_fill = {int(np.unique(p["output"][p["input"] != p["output"]])[0])
                  for p in train if np.any(p["input"] != p["output"])}
    assert train_fill == fills
    assert all(np.any(p["input"] != p["output"]) for p in test)


def self_test() -> None:
    ring = np.zeros((7, 8), dtype=np.int8)
    ring[1, 1:7] = ring[5, 1:7] = 3
    ring[1:6, 1] = ring[1:6, 6] = 3
    assert _enclosed_mask(ring, 0).sum() == 12
    ring[1, 3] = 0
    assert not _enclosed_mask(ring, 0).any()
    almost_ring = np.zeros((7, 8), dtype=np.int8)
    almost_ring[1, 1:7] = almost_ring[5, 1:7] = 3
    almost_ring[1:6, 1] = almost_ring[1:6, 6] = 3
    almost_ring[1, 1] = 0
    assert _enclosed_mask(almost_ring, 0).sum() == 12  # diagonal corner gap
    diamond = np.zeros((5, 5), dtype=np.int8)
    for r, c in ((1, 2), (2, 1), (2, 3), (3, 2)):
        diamond[r, c] = 3
    assert _enclosed_mask(diamond, 0).sum() == 1
    assert not _enclosed_mask(diamond, 0, N8).any()

    train_counts, test_counts, shapes, signatures = set(), set(), set(), set()
    areas, edge_walls, full_seals, palettes, profiles = set(), 0, 0, set(), set()
    open_pairs = 0
    variable_palette_puzzles = 0
    variable_shape_puzzles = 0
    variable_hole_puzzles = 0
    shared_dividers = 0
    multiple_exteriors = 0
    nonrectangular_holes = 0
    large_holes = 0
    test_simpler = 0
    test_harder = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_grid_key(p["input"]) for p in puzzle["train"] + puzzle["test"]))
        pair_palettes = set()
        pair_shapes = set()
        pair_hole_counts = set()
        training_hole_counts = []
        testing_hole_counts = []
        fill = next(int(np.unique(p["output"][p["input"] != p["output"]])[0])
                    for p in puzzle["train"] if np.any(p["input"] != p["output"]))
        for index, pair in enumerate(puzzle["train"] + puzzle["test"]):
            inp, out = pair["input"], pair["output"]
            shapes.add(inp.shape)
            pair_shapes.add(inp.shape)
            vals, counts = np.unique(inp, return_counts=True)
            bg = int(vals[np.argmax(counts)])
            barrier = int(vals[np.argmin(counts)])
            pair_palettes.add((bg, barrier))
            palettes.add((bg, barrier, fill))
            comps = _components(inp != out)
            pair_hole_counts.add(len(comps))
            if index < len(puzzle["train"]):
                training_hole_counts.append(len(comps))
            else:
                testing_hole_counts.append(len(comps))
            open_pairs += not comps
            areas.update(len(c) for c in comps)
            profiles.add((inp.shape, tuple(sorted(len(c) for c in comps))))
            if np.any(inp[0] == barrier) or np.any(inp[-1] == barrier) or np.any(inp[:, 0] == barrier) or np.any(inp[:, -1] == barrier):
                edge_walls += 1
            if _enclosed_mask(inp, bg, N8).any():
                full_seals += 1
            free_components = _components(inp == bg)
            if sum(any(r in (0, inp.shape[0] - 1) or c in (0, inp.shape[1] - 1)
                       for r, c in comp) for comp in free_components) > 1:
                multiple_exteriors += 1
            hole_id = {cell: i for i, comp in enumerate(comps) for cell in comp}
            for r, c in np.argwhere(inp == barrier):
                neighbors = {hole_id[(r + dr, c + dc)] for dr, dc in N4
                             if (r + dr, c + dc) in hole_id}
                if len(neighbors) >= 2:
                    shared_dividers += 1
                    break
            for comp in comps:
                rows = [r for r, _ in comp]
                cols = [c for _, c in comp]
                bbox = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
                nonrectangular_holes += len(comp) < bbox
                large_holes += len(comp) >= 24
            assert np.array_equal(out, _independent_fill(inp, bg, fill))
        variable_palette_puzzles += len(pair_palettes) > 1
        variable_shape_puzzles += len(pair_shapes) > 1
        variable_hole_puzzles += len(pair_hole_counts) > 1
        test_simpler += any(n < max(training_hole_counts) for n in testing_hole_counts)
        test_harder += any(n > max(training_hole_counts) for n in testing_hole_counts)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 40 and len(profiles) >= 80
    assert 1 in areas and any(a >= 10 for a in areas) and any(2 <= a <= 5 for a in areas)
    assert edge_walls >= 10 and full_seals >= 10 and variable_palette_puzzles >= 10
    assert open_pairs >= 2
    assert shared_dividers >= 10 and multiple_exteriors >= 10
    assert nonrectangular_holes >= 10 and large_holes >= 10
    assert variable_shape_puzzles >= 35 and variable_hole_puzzles >= 30
    assert test_simpler >= 10 and test_harder >= 5
    assert len(palettes) >= 25

    a = generate_puzzle(987654321, 7, 3)
    b = generate_puzzle(987654321, 7, 3)
    assert all(np.array_equal(x[k], y[k]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output"))
    for nt, ns in ((2, 1), (10, 1), (10, 4), (5, 2),
                   (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1000 + (nt or 0) * 10 + (ns or 0), nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
        validate_generated_puzzle(puzzle)
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(3001, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
