"""Self-describing rectangle constellations. Standard library + NumPy only.

The rule is fixed: three equally spaced core centers, with 1/2/3 distinct
core colors, describe a horizontal / descending / ascending line respectively.
"""
from itertools import combinations

import numpy as np


# Directions are in row, column coordinates. Their indexes are color count - 1.
_DIRECTIONS = ((0, 1), (1, 1), (-1, 1))
_SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5))
_TEST_SHAPES = _SHAPES + ((3, 7), (7, 3))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _array(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or any(not 1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return grid.astype(np.int8, copy=False)


def _background(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    if np.count_nonzero(counts == counts[background]) != 1:
        raise ValueError("background must be the unique most frequent color")
    return background


def _cores(grid, background):
    """All inclusion-maximal solid nonbackground boxes, both sides >= 3.

    A core can sit inside an irregular same-color component. Component bounding
    boxes are deliberately not used. Coordinates are half-open (r0,r1,c0,c1,c).
    """
    height, width = grid.shape
    candidates = set()
    for color in range(10):
        if color == background or np.count_nonzero(grid == color) < 9:
            continue
        mask = grid == color
        for r0 in range(height - 2):
            common = mask[r0].copy()
            for last in range(r0 + 1, height):
                common &= mask[last]
                if np.count_nonzero(common) < 3:
                    break
                if last - r0 < 2:
                    continue
                starts = np.flatnonzero(common & ~np.r_[False, common[:-1]])
                ends = np.flatnonzero(common & ~np.r_[common[1:], False]) + 1
                for c0, c1 in zip(starts, ends):
                    if c1 - c0 >= 3:
                        candidates.add((r0, last + 1, int(c0), int(c1), color))
    maximal = []
    for box in sorted(candidates, key=lambda b: (-(b[1]-b[0])*(b[3]-b[2]), b)):
        r0, r1, c0, c1, color = box
        if not any(a[4] == color and a[0] <= r0 and r1 <= a[1]
                   and a[2] <= c0 and c1 <= a[3] for a in maximal):
            maximal.append(box)
    maximal.sort()
    for a, b in combinations(maximal, 2):
        if max(a[0], b[0]) < min(a[1], b[1]) and max(a[2], b[2]) < min(a[3], b[3]):
            raise ValueError("overlapping noncontained rectangular cores are ambiguous")
    return maximal


def _center(box):
    # Doubled coordinates support both even and odd dimensions, without rounding.
    return box[0] + box[1] - 1, box[2] + box[3] - 1


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr == 0 and dc != 0:
        return 0
    if dr != 0 and abs(dr) == abs(dc):
        return 1 if dr * dc > 0 else 2
    return None


def _relationships(cores):
    """Return geometric triples and their count-valid subset, using input only."""
    centers = [_center(box) for box in cores]
    by_center = {point: i for i, point in enumerate(centers)}
    if len(by_center) != len(cores):
        raise ValueError("core centers must be distinct")
    geometric, valid = [], []
    for i, j in combinations(range(len(cores)), 2):
        a, b = centers[i], centers[j]
        direction = _direction(a, b)
        if direction is None or (a[0]+b[0]) % 2 or (a[1]+b[1]) % 2:
            continue
        midpoint = ((a[0]+b[0])//2, (a[1]+b[1])//2)
        middle = by_center.get(midpoint)
        if middle is None or middle in (i, j):
            continue
        triple = tuple(sorted((i, middle, j)))
        count = len({cores[k][4] for k in triple})
        geometric.append((triple, direction, count))
        if count == direction + 1:
            valid.append((triple, direction, count))
    return geometric, valid


def _participants(valid):
    return {i for triple, _, _ in valid for i in triple}


def _paint(shape, background, cores, selected):
    output = np.full(shape, background, dtype=np.int8)
    for i in selected:
        r0, r1, c0, c1, color = cores[i]
        output[r0:r1, c0:c1] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _array(input_grid)
    background = _background(grid)
    cores = _cores(grid, background)
    _, valid = _relationships(cores)
    return _paint(grid.shape, background, cores, _participants(valid))


def _colors(rng, count, anchor_color, anchor_index, palette):
    others = [c for c in palette if c != anchor_color]
    rng.shuffle(others)
    result = [anchor_color] * 3
    slots = [i for i in range(3) if i != anchor_index]
    rng.shuffle(slots)
    if count == 2:
        result[slots[0]] = others[0]
        if rng.integers(2):
            result[slots[1]] = others[0]
    elif count == 3:
        result[slots[0]], result[slots[1]] = others[:2]
    return result


def _line_group(rng, directions, palette, shapes, shared, endpoint=False,
                false_kind=None, large_false=False):
    """A group is a list of (center_row,center_col,height,width,color)."""
    objects = []
    anchor_color = int(rng.choice(palette))
    shared_shape = shapes[int(rng.integers(len(shapes)))]
    for line_number, direction in enumerate(directions):
        dr, dc = _DIRECTIONS[direction]
        step = int(rng.integers(6, 9))
        anchor_index = (int(rng.choice((0, 2))) if endpoint and line_number == 0 else 1)
        offsets = [0, step, 2*step]
        if false_kind == "spacing":
            offsets[2] += int(rng.integers(1, 4))
        offsets = [x - offsets[anchor_index] for x in offsets]
        count = direction + 1
        if false_kind == "count":
            count = int(rng.choice([n for n in (1, 2, 3) if n != count]))
        colors = _colors(rng, count, anchor_color, anchor_index, palette)
        for j, offset in enumerate(offsets):
            if shared and line_number and j == anchor_index:
                continue
            h, w = (shared_shape if shared and j == anchor_index
                    else shapes[int(rng.integers(len(shapes)))])
            if large_false and j == 0:
                h = w = 5
            objects.append((dr*offset, dc*offset, h, w, colors[j]))
    return objects


def _boxes(objects):
    return [(r-h//2, r+h//2+1, c-w//2, c+w//2+1, color)
            for r, c, h, w, color in objects]


def _apart(a, b):
    # Leave at least a one-cell gap between boxes in at least one axis.
    return (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def _place_group(rng, shape, group, existing):
    relative = _boxes(group)
    if any(not _apart(a, b) for a, b in combinations(relative, 2)):
        return None
    low_r = -min(b[0] for b in relative)
    high_r = shape[0] - max(b[1] for b in relative)
    low_c = -min(b[2] for b in relative)
    high_c = shape[1] - max(b[3] for b in relative)
    if low_r > high_r or low_c > high_c:
        return None
    for _ in range(110):
        rr = int(rng.integers(low_r, high_r + 1))
        cc = int(rng.integers(low_c, high_c + 1))
        placed = [(r0+rr, r1+rr, c0+cc, c1+cc, color)
                  for r0, r1, c0, c1, color in relative]
        if all(_apart(a, b) for a in placed for b in existing):
            return placed
    return None


def _add_noise(rng, grid, cores, density, force_contact, contact_cores=None):
    background = _background(grid)
    colors = [c for c in range(10) if c != background]
    rng.shuffle(colors)
    colors = colors[:int(rng.integers(3, 10))]
    protected = np.zeros(grid.shape, dtype=bool)
    for r0, r1, c0, c1, color in cores:
        protected[r0:r1, c0:c1] = True
    outside = np.argwhere(~protected)
    target = max(14, int(len(outside) * density))
    # Short runs and 2-by-2 blocks are noise, not eligible cores.
    if rng.random() < 0.65:
        for _ in range(30):
            rr = int(rng.integers(grid.shape[0]-1))
            cc = int(rng.integers(grid.shape[1]-1))
            if not np.any(protected[rr:rr+2, cc:cc+2]):
                grid[rr:rr+2, cc:cc+2] = int(rng.choice(colors))
                break
    if force_contact or rng.random() < 0.65:
        contact_cores = cores if contact_cores is None else contact_cores
        box = contact_cores[int(rng.integers(len(contact_cores)))]
        r0, r1, c0, c1, color = box
        border = ([(r0-1, c) for c in range(c0, c1)]
                  + [(r1, c) for c in range(c0, c1)]
                  + [(r, c0-1) for r in range(r0, r1)]
                  + [(r, c1) for r in range(r0, r1)])
        border = [(r, c) for r, c in border
                  if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                  and not protected[r, c]]
        if border:
            grid[border[int(rng.integers(len(border)))]] = color
    already = np.count_nonzero((grid != background) & ~protected)
    available = np.argwhere(~protected & (grid == background))
    for index in rng.choice(len(available), size=max(0, target-already), replace=False):
        grid[tuple(available[int(index)])] = int(rng.choice(colors))


def _make_pair(rng, background, palette, directions, shared, endpoint,
               false_kind, test=False, introductory=False):
    for _ in range(250):
        shape = (int(rng.integers(25, 31)), int(rng.integers(25, 31)))
        shapes = _TEST_SHAPES if test else (_SHAPES[:3] if introductory else _SHAPES)
        groups = []
        if shared:
            groups.append((_line_group(rng, directions, palette, shapes, True, endpoint), True))
        else:
            for direction in directions:
                groups.append((_line_group(rng, [direction], palette, shapes, False), True))
        false_direction = int(rng.choice(directions))
        groups.append((_line_group(rng, [false_direction], palette, shapes, False,
                                   false_kind=false_kind, large_false=introductory), False))
        cores, selected = [], set()
        for group, truthful in groups:
            placed = _place_group(rng, shape, group, cores)
            if placed is None:
                break
            if truthful:
                selected.update(range(len(cores), len(cores)+len(placed)))
            cores.extend(placed)
        else:
            _, valid = _relationships(cores)
            if _participants(valid) != selected or len(valid) != len(directions):
                continue
            if test and not any(7 in (cores[i][1]-cores[i][0], cores[i][3]-cores[i][2])
                                for i in selected):
                continue
            grid = _paint(shape, background, cores, range(len(cores)))
            density = float(rng.uniform(0.07, 0.13) if introductory else rng.uniform(0.09, 0.21))
            _add_noise(rng, grid, cores, density, force_contact=True,
                       contact_cores=[cores[i] for i in sorted(selected)])
            try:
                detected = _cores(grid, background)
            except ValueError:
                continue
            if _background(grid) != background or detected != sorted(cores):
                continue
            # Labels come from the planted true groups; detection is an acceptance check.
            output = _paint(shape, background, cores, selected)
            if not np.array_equal(solve(grid), output):
                continue
            return {"input": grid, "output": output}
    raise RuntimeError("could not pack an unambiguous constellation scene")


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    num_train = _integer(num_train, "num_train")
    num_test = _integer(num_test, "num_test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    background = int(rng.integers(10))
    palette = [c for c in range(10) if c != background]
    rng.shuffle(palette)
    palette = palette[:int(rng.integers(4, 10))]
    order = [int(x) for x in rng.permutation(3)]
    puzzle = {"train": [], "test": []}
    used = set()
    for index in range(num_train + num_test):
        test = index >= num_train
        if index == 0:
            directions, shared, endpoint, false_kind = order[:2], False, False, "count"
        elif index == 1:
            directions = [order[2], int(rng.choice(order[:2]))]
            shared, endpoint, false_kind = True, False, "spacing"
        elif index == 2 and not test:
            directions, shared, endpoint, false_kind = order, True, False, "count"
        else:
            nlines = (int(rng.integers(2, 4)) if test and num_train >= 3
                      else 2 if test else int(rng.integers(1, 4)))
            directions = [int(x) for x in rng.choice(3, nlines, replace=False)]
            shared = test or nlines == 3 or bool(rng.integers(2))
            endpoint = test
            false_kind = "count" if rng.integers(2) else "spacing"
        for _ in range(20):
            pair = _make_pair(rng, background, palette, directions, shared, endpoint,
                              false_kind, test, introductory=index == 0)
            key = _key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle["test" if test else "train"].append(pair)
                break
        else:
            raise RuntimeError("could not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _uneven_controls(cores, selected):
    """Correctly colored but unequally spaced collinear discarded triples."""
    found = []
    for triple in combinations(range(len(cores)), 3):
        if any(i in selected for i in triple):
            continue
        points = sorted(_center(cores[i]) for i in triple)
        direction = _direction(points[0], points[2])
        if direction is None or _direction(points[0], points[1]) != direction:
            continue
        if ((points[1][0]-points[0][0], points[1][1]-points[0][1])
                == (points[2][0]-points[1][0], points[2][1]-points[1][1])):
            continue
        if len({cores[i][4] for i in triple}) == direction + 1:
            found.append(triple)
    return found


def _analyze(pair):
    grid, output = _array(pair["input"]), _array(pair["output"])
    assert grid.shape == output.shape
    background = _background(grid)
    cores = _cores(grid, background)
    geometric, valid = _relationships(cores)
    selected = _participants(valid)
    assert valid and len(selected) < len(cores)
    assert np.array_equal(output, _paint(grid.shape, background, cores, selected))
    assert not np.array_equal(grid, output)
    assert all(np.all(grid[b[0]:b[1], b[2]:b[3]] == b[4]) for b in cores)
    assert all(_apart(a, b) for a, b in combinations(cores, 2))
    # Actual generated families have 3..7 odd-sized cores; the solver is broader.
    assert all(3 <= b[1]-b[0] <= 7 and 3 <= b[3]-b[2] <= 7 for b in cores)
    assert 6 <= len(cores) <= 10
    mask = np.zeros(grid.shape, dtype=bool)
    contact = False
    for r0, r1, c0, c1, color in cores:
        mask[r0:r1, c0:c1] = True
        border = ([(r0-1, c) for c in range(c0, c1)] + [(r1, c) for c in range(c0, c1)]
                  + [(r, c0-1) for r in range(r0, r1)] + [(r, c1) for r in range(r0, r1)])
        contact |= any(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                       and grid[r, c] == color for r, c in border)
    noise = int(np.count_nonzero((grid != background) & ~mask))
    assert 0.06 <= noise / np.count_nonzero(~mask) <= 0.22
    # Omitting A (keep every core) and B (keep entire selected color components)
    # are observable ablations, not private generation settings.
    without_a = _paint(grid.shape, background, cores, range(len(cores)))
    assert not np.array_equal(without_a, output)
    without_b = np.full_like(grid, background)
    seen = set()
    for index in selected:
        b = cores[index]
        stack = [(b[0], b[2])]
        while stack:
            r, c = stack.pop()
            if (r, c) in seen or grid[r, c] != b[4]:
                continue
            seen.add((r, c))
            without_b[r, c] = b[4]
            stack.extend((rr, cc) for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                         if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1])
    wrong_count = [t for t, direction, count in geometric
                   if count != direction + 1 and not any(i in selected for i in t)]
    degrees = [sum(i in triple for triple, _, _ in valid) for i in selected]
    endpoint_shared = False
    for i in selected:
        memberships = [t for t, _, _ in valid if i in t]
        if len(memberships) >= 2:
            for triple in memberships:
                centers = [_center(cores[k]) for k in triple]
                mean = tuple(sum(p[axis] for p in centers)//3 for axis in (0, 1))
                endpoint_shared |= _center(cores[i]) != mean
    return {"background": background, "cores": cores, "selected": selected,
            "directions": {direction for _, direction, _ in valid},
            "valid": valid, "wrong_count": bool(wrong_count),
            "uneven": bool(_uneven_controls(cores, selected)),
            "shared": max(degrees, default=0) >= 2, "endpoint_shared": endpoint_shared,
            "contact": contact, "without_b_changes": not np.array_equal(without_b, output),
            "noise_density": noise / np.count_nonzero(~mask)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, analyses = set(), {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_array(pair["input"]))
            assert key not in seen
            seen.add(key)
            analyses[section].append(_analyze(pair))
    all_infos = analyses["train"] + analyses["test"]
    assert len({x["background"] for x in all_infos}) == 1
    training = analyses["train"]
    assert set.union(*(x["directions"] for x in training)) == {0, 1, 2}
    assert training[0]["wrong_count"] and training[1]["uneven"]
    assert training[1]["shared"]
    assert not any(x["endpoint_shared"] for x in training)
    assert all(x["without_b_changes"] for x in all_infos)
    assert all(x["shared"] and x["endpoint_shared"] for x in analyses["test"])
    assert all(any(7 in (b[1]-b[0], b[3]-b[2]) for i, b in enumerate(x["cores"])
                   if i in x["selected"]) for x in analyses["test"])
    return None


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed structural, semantic-ablation, and diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    backgrounds, canvases, shapes, positions, counts, layouts = set(), set(), set(), set(), set(), set()
    directions, densities, endpoint_count, contact_count = set(), [], 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze(pair)
                backgrounds.add(info["background"])
                canvases.add(pair["input"].shape)
                counts.add(len(info["cores"]))
                directions.update(info["directions"])
                densities.append(info["noise_density"])
                endpoint_count += info["endpoint_shared"]
                contact_count += info["contact"]
                for r0, r1, c0, c1, color in info["cores"]:
                    shapes.add((r1-r0, c1-c0))
                    positions.add((r0, c0))
                layouts.add(tuple((b[0],b[1],b[2],b[3]) for b in info["cores"]))
                assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(backgrounds) >= 8 and len(canvases) >= 20
    assert len(shapes) >= 6 and len(positions) >= 100
    assert len(counts) >= 4 and len(layouts) >= 180
    assert directions == {0, 1, 2} and endpoint_count >= 42 and contact_count >= 170
    assert min(densities) < 0.09 and max(densities) > 0.18
    assert _signature(generate_puzzle(123, 3, 2)) == _signature(generate_puzzle(123, 3, 2))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1,1), (11,1), (4,0), (4,5), (2,2), (3,4), (True,1), (4,1.5), (None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicit input-only boundary/even-size check outside the sampling subset.
    grid = np.zeros((12, 24), dtype=np.int8)
    for c, color in ((0, 4), (8, 4), (16, 4)):
        grid[0:4, c:c+4] = color
    grid[4, 1] = 4  # attached noise must not alter a core's center
    output = solve(grid)
    expected = grid.copy()
    expected[4, 1] = 0
    assert np.array_equal(output, expected)
    return {"seeds": 42, "distinct_puzzles": 42, "canvas_shapes": len(canvases),
            "core_shapes": sorted(shapes), "core_counts": sorted(counts),
            "backgrounds": sorted(backgrounds), "distinct_layouts": len(layouts)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
