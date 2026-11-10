"""Companion chambers: rotational companionship selects the contour origin.

Only standard-library modules and NumPy are used. No parent code is imported.
"""
from collections import Counter, deque
from itertools import product

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple(p for p in product((-1, 0, 1), repeat=2) if p != (0, 0))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= x <= 30 for x in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected an integer 1..30 by 1..30 grid in colors 0..9")
    return grid.astype(np.int8, copy=True)


def _components(mask, directions=DIR4):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        cells, queue = [], [(r, c)]
        for y, x in queue:
            cells.append((y, x))
            for dy, dx in directions:
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and mask[yy, xx]
                        and not seen[yy, xx]):
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        result.append(cells)
    return result


def _trim(mask):
    yy, xx = np.nonzero(mask)
    return mask[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()


def _mask_key(mask):
    mask = _trim(mask)
    return mask.shape, mask.tobytes()


def _canonical(mask, reflections=False):
    versions = [mask, np.fliplr(mask)] if reflections else [mask]
    return min(_mask_key(np.rot90(m, k)) for m in versions for k in range(4))


def _erosion(mask, directions=DIR8):
    """Construction/validation depths, calculated by stripping complete layers."""
    current = mask.copy()
    depth = np.full(mask.shape, -1, dtype=int)
    h, w = mask.shape
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        inner = current.copy()
        for dy, dx in directions:
            inner &= padded[1+dy:1+dy+h, 1+dx:1+dx+w]
        depth[current & ~inner] = level
        current = inner
        level += 1
    return depth


def _distances(mask):
    """Solver depths, computed by an eight-neighbor boundary BFS."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=int)
    queue = deque()
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if any(not (0 <= y+dy < h and 0 <= x+dx < w
                    and mask[y+dy, x+dx]) for dy, dx in DIR8):
            depth[y, x] = 0
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in DIR8:
            yy, xx = y+dy, x+dx
            if (0 <= yy < h and 0 <= xx < w and mask[yy, xx]
                    and depth[yy, xx] < 0):
                depth[yy, xx] = depth[y, x]+1
                queue.append((yy, xx))
    return depth


def _inside(mask, y, x):
    return (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
            and bool(mask[y, x]))


def _paths(mask, depth):
    paths = []
    for y, x in np.argwhere(mask & (depth == 0)):
        y, x = int(y), int(x)
        for dy, dx in product((-1, 1), repeat=2):
            if (_inside(mask, y-dy, x) or _inside(mask, y, x-dx)
                    or not _inside(mask, y+dy, x)
                    or not _inside(mask, y, x+dx)):
                continue
            length = 0
            while (_inside(mask, y+length*dy, x+length*dx)
                   and depth[y+length*dy, x+length*dx] == length):
                length += 1
            paths.append((y, x, dy, dx, length))
    return paths


def _decode(local, mask, depth, neutral):
    cues = [tuple(map(int, p)) for p in np.argwhere(mask & (local != neutral))]
    if not cues:
        return (neutral,), ()
    length = max(int(depth[p]) for p in cues)+1
    if (len(cues) != length
            or sorted(int(depth[p]) for p in cues) != list(range(length))):
        raise ValueError("A key must have one visible cue at every consecutive depth")
    periods, directions = set(), set()
    for y, x, dy, dx, capacity in _paths(mask, depth):
        positions = [(y+d*dy, x+d*dx) for d in range(length)]
        if capacity >= length and set(positions) == set(cues):
            periods.add(tuple(int(local[p]) for p in positions))
            directions.add((dy, dx))
    if len(periods) != 1:
        raise ValueError("Key does not uniquely specify an inward corner palette")
    return periods.pop(), tuple(sorted(directions))


def _analyze(value):
    grid = _grid(value)
    h, w = grid.shape
    neutral = int(grid[0, 0])
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    if np.any(border != neutral):
        raise ValueError("All edges must have the exterior color")
    counts = np.bincount(grid.ravel(), minlength=10)
    counts[neutral] = 0
    order = np.argsort(counts)
    wall = int(order[-1])
    if counts[wall] == 0 or counts[wall] <= counts[order[-2]]:
        raise ValueError("Wall must be the strict most frequent non-neutral color")
    wall_mask = grid == wall
    exterior = np.zeros(grid.shape, dtype=bool)
    rooms = []
    exterior_count = 0
    for cells in _components(~wall_mask):
        if any(y in (0, h-1) or x in (0, w-1) for y, x in cells):
            exterior_count += 1
            for p in cells:
                exterior[p] = True
            continue
        y0, x0 = min(y for y, x in cells), min(x for y, x in cells)
        y1, x1 = max(y for y, x in cells), max(x for y, x in cells)
        mask = np.zeros((y1-y0+1, x1-x0+1), dtype=bool)
        for y, x in cells:
            mask[y-y0, x-x0] = True
        local = grid[y0:y1+1, x0:x1+1]
        depth = _distances(mask)
        period, directions = _decode(local, mask, depth, neutral)
        rooms.append({"top": y0, "left": x0, "mask": mask,
                      "depth": depth, "D": int(depth.max()),
                      "period": period, "directions": directions,
                      "canonical": _canonical(mask),
                      "marked": bool(np.any(local[mask] != neutral))})
    if exterior_count != 1 or np.any(grid[exterior] != neutral) or not rooms:
        raise ValueError("Expected one neutral exterior and enclosed keyed rooms")
    return {"grid": grid, "neutral": neutral, "wall": wall,
            "wall_mask": wall_mask, "exterior": exterior, "rooms": rooms}


def _render(analysis, matching="rotation", contours="eight", direction="normal"):
    rooms = analysis["rooms"]
    keys = []
    for room in rooms:
        if matching == "rotation":
            key = room["canonical"]
        elif matching == "reflection":
            key = _canonical(room["mask"], True)
        elif matching == "identity":
            key = _mask_key(room["mask"])
        elif matching == "area":
            key = int(room["mask"].sum())
        elif matching == "palette":
            key = room["period"]
        elif matching in ("exactly_two", "marked_only", "mirror_only"):
            key = room["canonical"]
        else:
            raise ValueError("Unknown matching mode")
        keys.append(key)
    frequencies = Counter(key for key, room in zip(keys, rooms)
                          if matching != "marked_only" or room["marked"])
    result = analysis["grid"].copy()
    for key, room in zip(keys, rooms):
        mask = room["mask"]
        if contours == "eight":
            depth = room["depth"]
        elif contours == "four":
            depth = _erosion(mask, DIR4)
        elif contours == "box":
            yy, xx = np.indices(mask.shape)
            depth = np.minimum.reduce((yy, xx, mask.shape[0]-1-yy,
                                       mask.shape[1]-1-xx))
        else:
            raise ValueError("Unknown contour mode")
        partnered = (frequencies[key] == 2 if matching == "exactly_two"
                     else frequencies[key] >= 2)
        if matching == "mirror_only":
            reflected = _canonical(np.fliplr(mask))
            partnered = not (frequencies[key] == 1 and reflected != key
                              and frequencies[reflected] > 0)
        D = int(depth[mask].max())
        coordinate = depth if partnered else D-depth
        if direction == "rim":
            coordinate = depth
        elif direction == "core":
            coordinate = D-depth
        elif direction == "reverse_zero" and not partnered:
            coordinate = -depth
        period = np.asarray(room["period"], dtype=np.int8)
        y, x = room["top"], room["left"]
        view = result[y:y+mask.shape[0], x:x+mask.shape[1]]
        view[mask] = period[coordinate[mask] % len(period)]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, enclosures, masks, palettes and phases entirely from input."""
    return _render(_analyze(input_grid))


def _sample_mask(rng, low, high, min_depth=2, max_depth=3,
                 forbidden=(), chiral=False, concave=False):
    for _ in range(1200):
        h, w = map(int, rng.integers(low, high+1, size=2))
        mask = np.ones((h, w), dtype=bool)
        n = int(rng.choice([0, 1, 2, 3], p=[.12, .52, .28, .08]))
        if chiral or concave:
            n = max(1, n)
        corners = rng.permutation(4)[:n]
        for corner in corners:
            a = int(rng.integers(1, min(3, h-3)+1))
            b = int(rng.integers(1, min(3, w-3)+1))
            ys = slice(0, a) if corner < 2 else slice(h-a, h)
            xs = slice(0, b) if corner % 2 == 0 else slice(w-b, w)
            mask[ys, xs] = False
        if rng.random() < .22 and h >= 8 and w >= 8:
            # An edge dent is distinct from a corner cut and changes the core.
            a = int(rng.integers(1, 3))
            b = int(rng.integers(1, 3))
            start = int(rng.integers(2, w-b-1))
            mask[:a, start:start+b] = False
        mask = _trim(mask)
        if len(_components(mask)) != 1:
            continue
        # No enclosed empty holes in the silhouette.
        complement = _components(~np.pad(mask, 1))
        if len(complement) != 1:
            continue
        depth = _erosion(mask)
        if not min_depth <= int(depth.max()) <= max_depth:
            continue
        if not any(p[4] >= 3 for p in _paths(mask, depth)):
            continue
        key = _canonical(mask)
        if key in forbidden or (chiral and key == _canonical(np.fliplr(mask))):
            continue
        return mask
    raise RuntimeError("Bounded chamber sampling exhausted")


def _compositions(total):
    if total == 0:
        return [()]
    return [(n,)+tail for n in range(1, min(3, total)+1)
            for tail in _compositions(total-n)]


def _pack(rng, masks):
    """Constructively pack separated wall boxes; vary row groups and offsets."""
    order = [int(i) for i in rng.permutation(len(masks))]
    boxes = [(m.shape[0]+2, m.shape[1]+2) for m in masks]
    layouts = []
    for profile in _compositions(len(masks)):
        groups, cursor = [], 0
        for count in profile:
            groups.append(order[cursor:cursor+count])
            cursor += count
        widths = [sum(boxes[i][1] for i in group)+len(group)-1 for group in groups]
        heights = [max(boxes[i][0] for i in group) for group in groups]
        min_h = sum(heights)+len(groups)-1+2
        min_w = max(widths)+2
        if min_h <= 30 and min_w <= 30:
            layouts.append((groups, widths, heights, min_h, min_w))
    if not layouts:
        raise RuntimeError("Rooms cannot fit on a 30 by 30 canvas")
    groups, widths, heights, min_h, min_w = layouts[int(rng.integers(len(layouts)))]
    h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    row_slack = rng.multinomial(h-min_h, np.ones(len(groups)+1)/(len(groups)+1))
    y = 1+int(row_slack[0])
    positions = [None]*len(masks)
    for j, group in enumerate(groups):
        slack = rng.multinomial(w-2-widths[j], np.ones(len(group)+1)/(len(group)+1))
        x = 1+int(slack[0])
        for k, i in enumerate(group):
            yy = y+int(rng.integers(heights[j]-boxes[i][0]+1))
            positions[i] = yy+1, x+1  # mask lies one cell inside its wall box
            x += boxes[i][1]+1+int(slack[k+1])
        y += heights[j]+1+int(row_slack[j+1])
    return (h, w), positions


def _period(rng, palette, length):
    return tuple(int(c) for c in rng.choice(palette, length, replace=False))


def _build_pair(rng, kind):
    roles = [int(c) for c in rng.permutation(10)]
    neutral, wall, palette = roles[0], roles[1], roles[2:]
    masks, classes, periods = [], [], []
    if kind == "mirror_teaching":
        base = _sample_mask(rng, 8, 9, 3, 3, chiral=True)
        turn = int(rng.integers(4))
        masks = [np.rot90(base, turn).copy(), np.rot90(base, turn+1).copy(),
                 np.rot90(np.fliplr(base), int(rng.integers(4))).copy()]
        classes = [0, 0, 1]
        p = _period(rng, palette, 3)
        periods = [p, (neutral,), p]
    elif kind == "triplet_teaching":
        base = _sample_mask(rng, 6, 9, 2, 3)
        forbidden = {_canonical(base), _canonical(np.fliplr(base))}
        unique = _sample_mask(rng, 7, 8, 2, 2, forbidden=forbidden, concave=True)
        turn = int(rng.integers(4))
        masks = [np.rot90(base, turn).copy(), np.rot90(base, turn).copy(),
                 np.rot90(base, turn+2).copy(),
                 np.rot90(unique, int(rng.integers(4))).copy()]
        classes = [0, 0, 0, 1]
        p = _period(rng, palette, 3)
        q = _period(rng, palette, 3)
        while q == p:
            q = _period(rng, palette, 3)
        periods = [p, q, _period(rng, palette, 2), p]
    else:
        if kind == "deep_test":
            sizes, low, high, dmin, dmax = [2, 1, 1], 9, 11, 4, 4
        elif kind == "crowded_test":
            profiles = [[2, 2, 1], [3, 2, 1], [2, 2, 1, 1]]
            sizes = profiles[int(rng.integers(3))]
            low = high = 6
            dmin = dmax = 2
        else:
            profiles = [[2], [1, 1], [2, 1], [3, 1], [2, 2], [1, 1, 1]]
            sizes = profiles[int(rng.integers(len(profiles)))]
            low, high, dmin, dmax = 6, 9, 2, 3
        forbidden = set()
        for class_id, size in enumerate(sizes):
            base = _sample_mask(rng, low, high, dmin, dmax, forbidden=forbidden)
            forbidden.add(_canonical(base))
            for _ in range(size):
                masks.append(np.rot90(base, int(rng.integers(4))).copy())
                classes.append(class_id)
                if rng.random() < .12:
                    periods.append((neutral,))
                else:
                    length = (3 if size == 1 and kind.endswith("test") else
                              int(rng.choice([1, 2, 3], p=[.08, .27, .65])))
                    periods.append(_period(rng, palette, length))
        # Deep tests must visibly use the core extrapolation.
        if kind == "deep_test":
            periods[-1] = _period(rng, palette, 3)
    shape, positions = _pack(rng, masks)
    inp = np.full(shape, neutral, dtype=np.int8)
    out = inp.copy()
    frequencies = Counter(classes)
    for mask, (y, x), class_id, period in zip(masks, positions, classes, periods):
        padded = np.pad(mask, 1)
        halo = padded.copy()
        for dy, dx in DIR8:
            # Dilation of the local mask into its one-cell exterior halo.
            halo[1+dy:1+dy+mask.shape[0], 1+dx:1+dx+mask.shape[1]] |= mask
        wall_local = halo & ~padded
        for grid in (inp, out):
            view = grid[y-1:y+mask.shape[0]+1, x-1:x+mask.shape[1]+1]
            assert np.all(view[halo] == neutral)
            view[wall_local] = wall
        depth = _erosion(mask)
        if period != (neutral,):
            choices = [p for p in _paths(mask, depth) if p[4] >= len(period)]
            yy, xx, dy, dx, _ = choices[int(rng.integers(len(choices)))]
            for d, color in enumerate(period):
                inp[y+yy+d*dy, x+xx+d*dx] = color
        coordinate = depth if frequencies[class_id] >= 2 else int(depth.max())-depth
        view = out[y:y+mask.shape[0], x:x+mask.shape[1]]
        view[mask] = np.asarray(period, dtype=np.int8)[coordinate[mask] % len(period)]
    turn = int(rng.integers(4))
    return {"input": np.rot90(inp, turn).copy(), "output": np.rot90(out, turn).copy()}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fingerprint(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    kinds = ["mirror_teaching", "triplet_teaching"]+["random_train"]*(nt-2)
    rng.shuffle(kinds)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, kind in [("train", k) for k in kinds]+[
            ("test", "crowded_test" if rng.random() < .6 else "deep_test")
            for _ in range(ne)]:
        for _ in range(40):
            pair = _build_pair(rng, kind)
            if np.array_equal(pair["input"], pair["output"]):
                continue
            key = _fingerprint(pair["input"])
            if key not in seen:
                analysis = _analyze(pair["input"])
                if split == "train" and len(puzzle["train"]) == 1:
                    previous = _analyze(puzzle["train"][0]["input"])
                    if (analysis["neutral"] == previous["neutral"]
                            or analysis["wall"] == previous["wall"]):
                        continue
                # The constructor uses class identities; decoding must agree.
                assert np.array_equal(_render(analysis), pair["output"])
                if kind == "mirror_teaching":
                    if any(np.array_equal(_render(analysis, contours=c), pair["output"])
                           for c in ("box", "four")):
                        continue
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("Distinct, informative example sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(analysis):
    rooms = analysis["rooms"]
    groups = Counter(r["canonical"] for r in rooms)
    return {
        "count": len(rooms), "sizes": sorted(groups.values()),
        "max_depth": max(r["D"] for r in rooms),
        "ordinary_unique": any(groups[r["canonical"]] == 1 and
                               (_canonical(np.fliplr(r["mask"])) == r["canonical"]
                                or _canonical(np.fliplr(r["mask"])) not in groups)
                               for r in rooms),
        "mirror_negative": any(groups[r["canonical"]] == 1 and
                               _canonical(np.fliplr(r["mask"])) != r["canonical"]
                               and _canonical(np.fliplr(r["mask"])) in groups
                               for r in rooms),
        "blank_partner": any(not r["marked"] and groups[r["canonical"]] >= 2
                             for r in rooms),
        "quarter_turn": any(a["canonical"] == b["canonical"]
                            and _mask_key(a["mask"]) != _mask_key(b["mask"])
                            and _mask_key(b["mask"]) in
                            (_mask_key(np.rot90(a["mask"])),
                             _mask_key(np.rot90(a["mask"], 3)))
                            for i, a in enumerate(rooms) for b in rooms[i+1:]),
        "cycle_repetition": any(len(r["period"]) >= 2 and r["D"] >= len(r["period"])
                                for r in rooms),
    }


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    analyses, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "Inputs must be distinct"
            seen.add(key)
            analysis = _analyze(inp)
            assert np.array_equal(out, _render(analysis)), "Incorrect contour phase or class"
            assert np.array_equal(out[analysis["wall_mask"]], inp[analysis["wall_mask"]])
            assert np.all(out[analysis["exterior"]] == analysis["neutral"])
            assert np.array_equal(out == analysis["wall"], analysis["wall_mask"])
            assert not np.array_equal(inp, out)
            assert set(np.unique(out)) <= set(np.unique(inp))
            for room in analysis["rooms"]:
                assert np.array_equal(room["depth"], _erosion(room["mask"]))
                assert len(room["period"]) == len(set(room["period"]))
                assert len(_components(room["mask"])) == 1
                assert len(_components(~np.pad(room["mask"], 1))) == 1
                assert any(p[4] >= 3 for p in _paths(room["mask"], room["depth"]))
                # Exterior halo is wall, not a second open region or a key.
                y, x, mask = room["top"], room["left"], room["mask"]
                for yy, xx in np.argwhere(mask):
                    for dy, dx in DIR8:
                        if not _inside(mask, int(yy)+dy, int(xx)+dx):
                            assert inp[y+int(yy)+dy, x+int(xx)+dx] == analysis["wall"]
            f = _features(analysis)
            if split == "train":
                assert 2 <= f["count"] <= 4 and f["max_depth"] <= 3
            else:
                assert f["count"] > 4 or f["max_depth"] >= 4
            analyses[split].append(analysis)
    train = analyses["train"]
    fs = [_features(a) for a in train]
    for feature in ("ordinary_unique", "mirror_negative", "blank_partner",
                    "quarter_turn", "cycle_repetition"):
        assert any(f[feature] for f in fs), f"Missing teaching evidence: {feature}"
    assert any(3 in f["sizes"] for f in fs)
    assert any(2 in f["sizes"] for f in fs)
    assert {2, 3} <= {len(r["period"]) for a in train for r in a["rooms"]}
    # Reject simpler rules by actual full predictions on training data.
    alternatives = [dict(matching=m) for m in
                    ("reflection", "identity", "area", "palette", "exactly_two",
                     "marked_only", "mirror_only")]
    alternatives += [dict(contours=c) for c in ("box", "four")]
    alternatives += [dict(direction=d) for d in ("rim", "core", "reverse_zero")]
    for alternative in alternatives:
        assert any(not np.array_equal(_render(a, **alternative), _render(a)) for a in train), alternative
    # These are visible pair-level assignments, not a puzzle-level hidden palette.
    assert len({(a["neutral"], a["wall"]) for s in analyses for a in analyses[s]}) >= 2


def _puzzle_key(puzzle):
    return tuple((_fingerprint(_grid(p["input"])), _fingerprint(_grid(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded regression and measured structural variation across 42 seeds."""
    signatures, shapes, roles, masks, counts, profiles, depths, directions = (set() for _ in range(8))
    samples = []
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_puzzle_key(p))
        samples.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze(pair["input"])
                f = _features(a)
                shapes.add(a["grid"].shape)
                roles.add((a["neutral"], a["wall"]))
                counts.add(f["count"])
                profiles.add(tuple(f["sizes"]))
                for r in a["rooms"]:
                    masks.add(r["canonical"])
                    depths.add(r["D"])
                    directions.update(r["directions"])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(roles) >= 50 and len(masks) >= 80
    assert {2, 3, 4, 5, 6} <= counts
    assert len(profiles) >= 7 and {2, 3, 4} <= depths
    assert directions == set(product((-1, 1), repeat=2))
    assert _puzzle_key(samples[0]) == _puzzle_key(generate_puzzle(0))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(8100+nt*10+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (4, False), (None, 1), (4, None)]:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in samples[0][s]]
              for s in samples[0]}
    a = _analyze(broken["test"][0]["input"])
    y, x = map(int, np.argwhere(a["wall_mask"])[0])
    broken["test"][0]["output"][y, x] = a["neutral"]
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Changed wall was not rejected")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "canvas_shapes": len(shapes), "role_assignments": len(roles),
            "rotational_shape_classes": len(masks), "room_counts": sorted(counts),
            "class_profiles": [list(x) for x in sorted(profiles)],
            "maximum_depths": sorted(depths), "key_directions": sorted(directions)}
