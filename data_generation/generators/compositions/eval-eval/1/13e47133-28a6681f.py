"""Contour aqueducts: conserved liquid settles inside erosion-depth gutters.

Only standard library and NumPy are used. No parent module, file, seed, or
stored example participates in solve(). Diagnostic storage is caller-owned;
self_test keeps one puzzle at a time and writes no files.
"""
from collections import deque
import hashlib
import json
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
LIQUID = 9


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b) or not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 train, 1..4 test, train > test")
    return int(a), int(b)


def _components(mask):
    h, w = mask.shape
    unseen = mask.copy()
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if not unseen[r, c]:
            continue
        part = np.zeros_like(mask)
        unseen[r, c] = False
        queue = deque([(r, c)])
        while queue:
            y, x = queue.popleft()
            part[y, x] = True
            for dy, dx in D4:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and unseen[yy, xx]:
                    unseen[yy, xx] = False
                    queue.append((yy, xx))
        result.append(part)
    return result


def _flood(mask, start):
    h, w = mask.shape
    part = np.zeros_like(mask)
    queue = deque([start])
    part[start] = True
    while queue:
        y, x = queue.popleft()
        for dy, dx in D4:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not part[yy, xx]:
                part[yy, xx] = True
                queue.append((yy, xx))
    return part


def _depth(mask):
    """Parallel eight-neighbor peeling; key pegs do not change the geometry."""
    h, w = mask.shape
    current = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int16)
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        inner = current.copy()
        for dy, dx in D8:
            inner &= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        depth[current & ~inner] = level
        current = inner
        level += 1
    return depth


def _distance(mask, neighbors=D8):
    """Boundary-distance BFS, used as a separate depth invariant."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if any(not (0 <= y + dy < h and 0 <= x + dx < w) or
               not mask[y + dy, x + dx] for dy, dx in neighbors):
            depth[y, x] = 0
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbors:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and depth[yy, xx] < 0:
                depth[yy, xx] = depth[y, x] + 1
                queue.append((yy, xx))
    return depth


def _key_paths(region, depth, length):
    h, w = region.shape
    paths = []
    def inside(y, x):
        return 0 <= y < h and 0 <= x < w and region[y, x]
    for y, x in np.argwhere(region & (depth == 0)):
        y, x = int(y), int(x)
        for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            if inside(y - dy, x) or inside(y, x - dx):
                continue
            if not inside(y + dy, x) or not inside(y, x + dx):
                continue
            points = [(y + t * dy, x + t * dx) for t in range(length)]
            if all(inside(yy, xx) and depth[yy, xx] == t for t, (yy, xx) in enumerate(points)):
                paths.append((points, (dy, dx)))
    return paths


def _read(input_grid):
    inp = np.asarray(input_grid)
    if inp.ndim != 2 or not np.issubdtype(inp.dtype, np.integer) or not all(1 <= n <= 30 for n in inp.shape):
        raise ValueError("expected a 1..30 integer grid")
    if np.any((inp < 0) | (inp > 9)) or min(inp.shape) < 7:
        raise ValueError("invalid colors or no room for a framed chamber")
    wall = int(inp[0, 0])
    if wall == LIQUID or not (np.all(inp[0] == wall) and np.all(inp[-1] == wall) and
                             np.all(inp[:, 0] == wall) and np.all(inp[:, -1] == wall)):
        raise ValueError("uniform solid exterior frame required")
    counts = np.bincount(inp.astype(np.int64, copy=False).ravel(), minlength=10)
    available = [c for c in range(10) if c not in (wall, LIQUID)]
    neutral = max(available, key=lambda c: int(counts[c]))
    if sum(counts[c] == counts[neutral] for c in available) != 1:
        raise ValueError("neutral role must have a unique plurality among non-wall, non-liquid cells")
    regions = []
    for region in _components(inp != wall):
        depth = _depth(region)
        pegs = region & (inp != neutral) & (inp != LIQUID)
        points = list(map(tuple, np.argwhere(pegs)))
        length = len(points)
        if not 2 <= length <= 4:
            raise ValueError("each chamber requires one explicit key of length 2..4")
        if sorted(int(depth[p]) for p in points) != list(range(length)):
            raise ValueError("key must visit consecutive erosion depths")
        points.sort(key=lambda p: int(depth[p]))
        period = tuple(int(inp[p]) for p in points)
        if len(set(period)) != length:
            raise ValueError("key entries must be distinct")
        paths = [(p, direction) for p, direction in _key_paths(region, depth, length)
                 if p == points]
        if len(paths) != 1:
            raise ValueError("key must be an inward convex-corner diagonal")
        regions.append({"region": region, "depth": depth, "pegs": pegs,
                        "period": period, "direction": paths[0][1]})
    return inp, neutral, wall, regions


def _pour(channel, source, reverse=False, top_first=False):
    """Unit pour, including hydraulic equalization through existing liquid.

    Ties are irrelevant on supported family inputs: all four adversarial orders
    agree and every final surface is a complete level. Partial-level inputs are
    outside the family, rather than assigned a hidden left/right convention.
    """
    h, w = channel.shape
    water = np.zeros_like(channel)
    rows = np.arange(h)[:, None]

    def add_to_body(point, incoming_col):
        body = _flood(water, point)
        surface = int(np.nonzero(body)[0].min())
        # Complete the current level first. Only when it is full may its
        # surface rise one row and cross a newly reached spillway. A filled
        # water cell transmits pressure; it is not a solid plug in a pipe.
        for level in range(surface, -1, -1):
            closure = _flood(channel & (rows >= level), point)
            choices = np.argwhere(closure & ~water)
            if len(choices):
                bottom = int(choices[:, 0].max())
                choices = choices[choices[:, 0] == bottom]
                y, x = min(map(tuple, choices), key=lambda p: (
                    abs(int(p[1]) - incoming_col), -int(p[1]) if reverse else int(p[1])))
                water[int(y), int(x)] = True
                return
        raise ValueError("liquid exceeds its connected gutter capacity")

    points = sorted(map(tuple, np.argwhere(source)),
                    key=lambda p: (p[0] if top_first else -p[0], -p[1] if reverse else p[1]))
    for row, col in points:
        row, col = int(row), int(col)
        if not channel[row, col]:
            raise ValueError("source is outside its channel")
        if water[row, col]:
            add_to_body((row, col), col)
            continue
        for _ in range(h + 1):
            if row + 1 < h and channel[row + 1, col] and not water[row + 1, col]:
                row += 1
                continue
            if row + 1 < h and water[row + 1, col]:
                add_to_body((row + 1, col), col)
                break
            left = right = col
            while left > 0 and channel[row, left - 1] and not water[row, left - 1]:
                left -= 1
            while right + 1 < w and channel[row, right + 1] and not water[row, right + 1]:
                right += 1
            outlets = [] if row + 1 == h else [c for c in range(left, right + 1)
                if channel[row + 1, c]]
            if not outlets:
                water[row, col] = True
                break
            col = min(outlets, key=lambda c: (abs(c - col), -c if reverse else c))
            row += 1
            if water[row, col]:
                add_to_body((row, col), col)
                break
        else:
            raise AssertionError("pour did not terminate")
    return water


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; no training pairs or generation state."""
    inp, neutral, wall, regions = _read(input_grid)
    out = inp.copy()
    out[inp == LIQUID] = neutral
    for info in regions:
        for d in range(int(info["depth"].max()) + 1):
            channel = (info["depth"] == d) & ~info["pegs"]
            source = channel & (inp == LIQUID)
            if source.any():
                target = _pour(channel, source)
                out[target] = info["period"][d % len(info["period"])]
    return out


def _capture(channel, target):
    """Dry reachability: bit 1 is target, bit 2 is any other resting place."""
    h, w = channel.shape
    labels = np.zeros(channel.shape, dtype=np.int8)
    for row in range(h - 1, -1, -1):
        col = 0
        while col < w:
            if not channel[row, col]:
                col += 1
                continue
            end = col
            while end + 1 < w and channel[row, end + 1]:
                end += 1
            exits = 0
            for c in range(col, end + 1):
                if target[row, c]:
                    exits |= 1
                if row + 1 < h and channel[row + 1, c]:
                    exits |= int(labels[row + 1, c])
            for c in range(col, end + 1):
                if target[row, c]:
                    labels[row, c] = 1
                elif row + 1 < h and channel[row + 1, c]:
                    labels[row, c] = labels[row + 1, c]
                else:
                    labels[row, c] = exits or 2
            col = end + 1
    return labels


def _pool_source(rng, channel, style):
    """Choose a complete stable level and supply in its unambiguous catchment."""
    rows = np.flatnonzero(channel.any(axis=1))
    if len(rows) < 3:
        return None
    tops = rows[1:]
    tops = tops[rng.permutation(len(tops))]
    yy = np.arange(channel.shape[0])[:, None]
    for top in tops:
        for target in _components(channel & (yy >= top)):
            volume = int(target.sum())
            if not 2 <= volume <= 65 or not target[int(top)].any():
                continue
            eligible = _capture(channel, target) == 1
            outside = np.argwhere(eligible & ~target)
            if not len(outside) or int(eligible.sum()) < volume:
                continue
            anchor = outside[int(rng.integers(len(outside)))]
            points = np.argwhere(eligible)
            dr, dc = abs(points[:, 0] - anchor[0]), abs(points[:, 1] - anchor[1])
            noise = rng.random(len(points))
            if style == 0:
                score = dr + dc + noise
            elif style == 1:
                score = noise
            elif style == 2:
                score = 3 * dc + dr * .1 + noise
            else:
                score = (~target[points[:, 0], points[:, 1]]) * 3 + noise
            selected = points[np.argsort(score)[:volume]].copy()
            if not np.any(np.all(selected == anchor, axis=1)):
                selected[-1] = anchor
            source = np.zeros_like(channel)
            source[selected[:, 0], selected[:, 1]] = True
            try:
                consistent = all(np.array_equal(_pour(channel, source, rev, order), target)
                                 for rev in (False, True) for order in (False, True))
            except ValueError:
                continue
            if not consistent:
                continue
            if np.array_equal(source.sum(axis=0), target.sum(axis=0)):
                continue
            return source, target
    return None


def _layout(rng, shape, split_budget, obstacles, plain=False):
    h, w = shape
    solid = np.zeros(shape, dtype=bool)
    solid[[0, -1], :] = True
    solid[:, [0, -1]] = True
    rooms = [(1, h - 2, 1, w - 2)]
    for _ in range(split_budget):
        options = [(i, axis) for i, (a, b, c, d) in enumerate(rooms)
                   for axis, extent in ((0, b - a + 1), (1, d - c + 1)) if extent >= 15]
        if not options:
            break
        i, axis = options[int(rng.integers(len(options)))]
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(a + 7, b - 6))
            solid[cut, c - 1:d + 2] = True
            rooms.extend(((a, cut - 1, c, d), (cut + 1, b, c, d)))
        else:
            cut = int(rng.integers(c + 7, d - 6))
            solid[a - 1:b + 2, cut] = True
            rooms.extend(((a, b, c, cut - 1), (a, b, cut + 1, d)))
    if not plain:
        for _ in range(obstacles):
            a, b, c, d = rooms[int(rng.integers(len(rooms)))]
            # Attached notches or interior islands alter the contour topology.
            bh = int(rng.integers(1, max(2, min(5, (b - a) // 3) + 1)))
            bw = int(rng.integers(1, max(2, min(5, (d - c) // 3) + 1)))
            y = int(rng.integers(a + 2, max(a + 3, b - bh)))
            x = int(rng.integers(c + 2, max(c + 3, d - bw)))
            side = int(rng.integers(5))
            if side == 0:
                y = a
            elif side == 1:
                y = b - bh + 1
            elif side == 2:
                x = c
            elif side == 3:
                x = d - bw + 1
            solid[y:y + bh, x:x + bw] = True
    return solid


def _scaffold(rng, solid, neutral, wall, key_length, lower_keys=False):
    inp = np.full(solid.shape, neutral, dtype=np.int8)
    inp[solid] = wall
    infos = []
    for region in _components(~solid):
        depth = _depth(region)
        length = min(key_length, int(depth.max()) + 1)
        if length < 2:
            return None
        paths = _key_paths(region, depth, length)
        if lower_keys:
            paths = [p for p in paths if p[1][0] == -1] or paths
        else:
            paths = [p for p in paths if p[1][0] == 1] or paths
        if not paths:
            return None
        points, direction = paths[int(rng.integers(len(paths)))]
        palette = [c for c in range(9) if c not in (neutral, wall)]
        period = tuple(map(int, rng.choice(palette, length, replace=False)))
        pegs = np.zeros_like(solid)
        for p, color in zip(points, period):
            inp[p] = color
            pegs[p] = True
        infos.append({"region": region, "depth": depth, "pegs": pegs,
                      "period": period, "direction": direction})
    return inp, infos


def _geometry_witness(inp, infos, target):
    """Actual wet cells distinguish L-infinity contours from box and L1 bands."""
    for info in infos:
        region, depth = info["region"], info["depth"]
        rr, cc = np.nonzero(region)
        box = np.full(inp.shape, -1, dtype=np.int16)
        box[rr, cc] = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
        l1 = _distance(region, D4)
        palette = np.asarray(info["period"])
        wet = region & target
        if wet.any() and np.any(palette[depth[wet] % len(palette)] != palette[box[wet] % len(palette)]) and np.any(
                palette[depth[wet] % len(palette)] != palette[l1[wet] % len(palette)]):
            return True
    return False


def _make_pair(rng, shape, neutral, wall, index, test):
    # Fixed finite search; no retained candidate grids, logs or retry artifacts.
    for attempt in range(80):
        first = not test and index == 0
        key_length = 3 if first else 2 if not test and index == 1 else int(rng.integers(2, 5))
        budget = 0 if first else int(rng.integers(0, 3 if test else 2))
        solid = _layout(rng, shape, budget, int(rng.integers(1, 5 if test else 4)))
        scaffold = _scaffold(rng, solid, neutral, wall, key_length, test or index % 3 == 2)
        if scaffold is None:
            continue
        inp, infos = scaffold
        channels = [(info, d, (info["depth"] == d) & ~info["pegs"])
                    for info in infos for d in range(int(info["depth"].max()) + 1)]
        rng.shuffle(channels)
        if first:
            channels.sort(key=lambda item: item[1] < len(item[0]["period"]))
        wanted = int(rng.integers(3, 6)) if test else int(rng.integers(2, 5))
        source = np.zeros_like(solid)
        target = np.zeros_like(solid)
        selected = []
        for info, d, channel in channels:
            candidate = _pool_source(rng, channel, (index + len(selected)) % 4)
            if candidate is None:
                continue
            supply, pool = candidate
            if (source | supply).sum() > ((~solid).sum() - sum(len(i["period"]) for i in infos)) // 3:
                continue
            source |= supply
            target |= pool
            selected.append((info, d, pool))
            if len(selected) >= wanted:
                break
        if len(selected) < 2 or len({d for _, d, _ in selected}) < 2:
            continue
        if first and (not any(len(info["period"]) == 3 and d >= 3 and
                              d % 3 != 2 and d % 3 != d % 2 for info, d, _ in selected) or
                      not _geometry_witness(inp, infos, target)):
            continue
        if not test and index == 1 and not any(
                a is b and da != db and da % len(a["period"]) == db % len(b["period"])
                for a, da, _ in selected for b, db, _ in selected):
            continue
        if test and len(selected) < 3:
            continue
        inp[source] = LIQUID
        out = inp.copy()
        out[source] = neutral
        for info, d, pool in selected:
            out[pool] = info["period"][d % len(info["period"])]
        # A real interaction witness: removing depth barriers changes placement.
        unguttered = np.zeros_like(solid)
        try:
            for info in infos:
                unguttered |= _pour(info["region"] & ~info["pegs"], source & info["region"])
        except ValueError:
            continue
        if np.array_equal(target, unguttered):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded contour-gutter construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # Rotated permutations vary roles within every puzzle without hidden rules.
    colors = list(map(int, rng.permutation(9)))
    offset = int(rng.integers(1, 9))
    shapes = set()
    puzzle = {"train": [], "test": []}
    for serial in range(nt + ne):
        test = serial >= nt
        index = serial - nt if test else serial
        low = 21 if serial == 0 else 18 if test else 15
        choices = [(h, w) for h in range(low, 31) for w in range(low, 31) if (h, w) not in shapes]
        shape = choices[int(rng.integers(len(choices)))]
        shapes.add(shape)
        neutral, wall = colors[serial % 9], colors[(serial + offset) % 9]
        pair = _make_pair(rng, shape, neutral, wall, index, test)
        puzzle["test" if test else "train"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, neutral, wall, infos = _read(pair["input"])
    out = np.asarray(pair["output"])
    assert inp.shape == out.shape and np.any(inp != out)
    assert not np.any(out == LIQUID)
    assert np.array_equal(inp == wall, out == wall)
    source = inp == LIQUID
    wet = np.zeros_like(source)
    all_pegs = np.zeros_like(source)
    active, repeats, lateral, levels, geometries = [], False, False, set(), []
    cycle_disambiguation = same_material = False
    no_gutter = np.zeros_like(source)
    for info in infos:
        region, depth, pegs, period = (info[k] for k in ("region", "depth", "pegs", "period"))
        assert np.array_equal(_distance(region), depth)
        assert np.array_equal(inp[pegs], out[pegs])
        all_pegs |= pegs
        no_gutter |= _pour(region & ~pegs, source & region)
        local_depths = []
        for d in range(int(depth.max()) + 1):
            channel = (depth == d) & ~pegs
            supply = channel & source
            pool = channel & (out != neutral)
            assert np.all(out[pool] == period[d % len(period)])
            assert int(supply.sum()) == int(pool.sum()), "volume must be conserved within each depth"
            if not supply.any():
                assert not pool.any()
                continue
            assert np.all(_capture(channel, pool)[supply] == 1), "source has another dry resting place"
            for reverse in (False, True):
                for order in (False, True):
                    assert np.array_equal(_pour(channel, supply, reverse, order), pool), "pour order changes result"
            for body in _components(pool):
                top = int(np.nonzero(body)[0].min())
                lower = channel & (np.arange(inp.shape[0])[:, None] >= top)
                closure = next(p for p in _components(lower) if np.any(p & body))
                assert np.array_equal(closure, body), "incomplete level or leaking gutter"
                levels.add(top)
            wet |= pool
            active.append(d)
            local_depths.append(d)
            repeats |= d >= len(period)
            cycle_disambiguation |= len(period) == 3 and d >= 3 and d % 3 != 2 and d % 3 != d % 2
            lateral |= not np.array_equal(supply.sum(axis=0), pool.sum(axis=0))
        same_material |= any(a != b and a % len(period) == b % len(period)
                             for a in local_depths for b in local_depths)
        geometries.append(hashlib.sha256(region.tobytes()).hexdigest())
    assert int(source.sum()) == int(wet.sum()) > 0
    assert np.array_equal(out[~(wet | all_pegs | (inp == wall))],
                          np.full(int((~(wet | all_pegs | (inp == wall))).sum()), neutral))
    assert not np.array_equal(wet, no_gutter), "ignoring contour gutters must change placement"
    assert len(active) >= 2 and len(set(active)) >= 2 and lateral
    assert not np.array_equal(wet, source)
    assert int(np.nonzero(wet)[0].sum()) > int(np.nonzero(source)[0].sum())
    assert np.array_equal(solve(inp), out)
    return {"shape": inp.shape, "neutral": neutral, "wall": wall, "regions": len(infos),
            "active_depths": tuple(active), "repeats": bool(repeats), "lateral": bool(lateral),
            "cycle_disambiguation": bool(cycle_disambiguation), "same_material": bool(same_material),
            "levels": tuple(sorted(levels)), "volume": int(source.sum()),
            "geometry_witness": _geometry_witness(inp, infos, wet),
            "key_lengths": tuple(len(i["period"]) for i in infos),
            "directions": tuple(i["direction"] for i in infos),
            "geometry": tuple(geometries)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, stats = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for value in pair.values():
                grid = np.asarray(value)
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            inp = np.asarray(pair["input"])
            key = (inp.shape, inp.astype(np.int8).tobytes())
            assert key not in seen
            seen.add(key)
            info = _analyze(pair)
            stats.append(info)
            if split == "test":
                assert len(info["active_depths"]) >= 3
    assert stats[0]["cycle_disambiguation"] and stats[0]["geometry_witness"]
    assert stats[1]["same_material"]
    assert 3 in stats[0]["key_lengths"] and 2 in stats[1]["key_lengths"]
    assert len({s["neutral"] for s in stats}) >= min(3, len(stats))
    assert len({s["wall"] for s in stats}) >= min(3, len(stats))
    assert len({s["shape"] for s in stats}) == len(stats)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                arr = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(arr.shape))
                digest.update(arr.tobytes())
    return digest.hexdigest()


def _rectangular_fixture():
    """An explicit expected answer, including pressure through a narrow bend."""
    inp = np.zeros((13, 13), dtype=np.int8)
    inp[[0, -1], :] = 8
    inp[:, [0, -1]] = 8
    for k in range(3):
        inp[k + 1, k + 1] = k + 1
    expected = inp.copy()
    for d in range(4):
        start, end = d + 1, 11 - d
        first = start + 1 if d < 3 else start
        inp[start, first:end + 1] = LIQUID
        if d < 3:
            inp[start + 1, end] = LIQUID
        expected[end, start:end + 1] = d % 3 + 1
    assert int((inp == LIQUID).sum()) == 32
    assert np.array_equal(solve(inp), expected)
    for info in _read(inp)[3]:
        for d in range(4):
            channel = (info["depth"] == d) & ~info["pegs"]
            for reverse in (False, True):
                for order in (False, True):
                    assert np.array_equal(_pour(channel, channel & (inp == LIQUID), reverse, order),
                                          channel & (expected != 0))


def self_test():
    """42 seeds, O(one puzzle) live grids, no disk writes or unbounded search."""
    fingerprints, shapes, geometry, roles, volumes, region_counts, lengths, directions = (set() for _ in range(8))
    total = repeats = irregular = lateral = 0
    _rectangular_fixture()
    recolor = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0, 9])
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for pair in (puzzle["train"][0], puzzle["test"][0]):
            inp, out = pair["input"], pair["output"]
            original = inp.copy()
            assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
            assert np.array_equal(solve(recolor[inp]), recolor[out])
            solve(inp)
            assert np.array_equal(inp, original), "solve changed its input"
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair)
                total += 1
                repeats += info["repeats"]
                irregular += info["geometry_witness"]
                lateral += info["lateral"]
                shapes.add(info["shape"])
                geometry.add(info["geometry"])
                roles.add((info["neutral"], info["wall"]))
                volumes.add(info["volume"])
                region_counts.add(info["regions"])
                lengths.update(info["key_lengths"])
                directions.update(info["directions"])
    assert len(fingerprints) == 42
    assert len(shapes) >= 50 and len(geometry) >= 150 and len(roles) >= 35
    assert len(volumes) >= 20 and len(region_counts) >= 3
    assert lengths == {2, 3, 4} and directions == {(1, 1), (1, -1), (-1, 1), (-1, -1)}
    assert repeats >= 42 and irregular >= 42 and lateral == total
    a = generate_puzzle(924, 2, 1)
    assert _fingerprint(a) == _fingerprint(generate_puzzle(924, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(710 + nt * 5 + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Reject an extra painted cell: catches volume/complete-level corruption.
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in a[s]] for s in a}
    pair = broken["train"][0]
    inp, neutral, _, infos = _read(pair["input"])
    free = (pair["output"] == neutral) & (inp == neutral)
    y, x = map(int, np.argwhere(free)[0])
    info = next(i for i in infos if i["region"][y, x])
    pair["output"][y, x] = info["period"][int(info["depth"][y, x]) % len(info["period"])]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("extra painted volume accepted")
    return {"seeds": 42, "pairs": total, "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "geometries": len(geometry), "role_assignments": len(roles),
            "volumes": len(volumes), "region_counts": sorted(region_counts),
            "key_lengths": sorted(lengths), "cyclic_witness_pairs": repeats,
            "irregular_contour_witness_pairs": irregular, "lateral_pairs": lateral,
            "explicit_pressure_fixture": True, "equivariance_checks": 168}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
