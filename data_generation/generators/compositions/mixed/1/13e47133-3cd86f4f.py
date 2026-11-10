"""Keyed contour regions, then a bottom-anchored shear of complete rows.

Only standard Python and NumPy are used. Generation writes no files and keeps
one bounded candidate at a time; solve reads only the supplied input.
"""
from collections import deque
import hashlib

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return (a.shape, a.astype(np.int8, copy=False).tobytes())


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    result = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        component = np.zeros_like(mask)
        queue = deque([(r, c)])
        seen[r, c] = component[r, c] = True
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = component[nr, nc] = True
                    queue.append((nr, nc))
        result.append(component)
    return result


def _erosion_depth(component, diagonal=True):
    """Repeated erosion; off-grid positions are always outside the region."""
    h, w = component.shape
    depth = np.full((h, w), -1, dtype=np.int8)
    remaining = component.copy()
    level = 0
    while remaining.any():
        padded = np.pad(remaining, 1, constant_values=False)
        interior = remaining.copy()
        for dr, dc in DIR8 if diagonal else DIR4:
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[remaining & ~interior] = level
        remaining = interior
        level += 1
    return depth


def _distance_depth(component):
    """Generation-side BFS; checked against erosion, rather than reused by solve."""
    h, w = component.shape
    depth = np.full((h, w), -1, dtype=np.int8)
    queue = deque()
    for rr, cc in np.argwhere(component):
        r, c = int(rr), int(cc)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w and component[r + dr, c + dc])
               for dr, dc in DIR8):
            depth[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and component[nr, nc] and depth[nr, nc] < 0:
                depth[nr, nc] = depth[r, c] + 1
                queue.append((nr, nc))
    return depth


def _inside(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _paths(mask, depth):
    """Convex corner diagonals whose kth cell has contour depth k."""
    result = []
    for rr, cc in np.argwhere(mask & (depth == 0)):
        r, c = int(rr), int(cc)
        for dr in (-1, 1):
            for dc in (-1, 1):
                if (_inside(mask, r - dr, c) or _inside(mask, r, c - dc)
                        or not all(_inside(mask, r + a, c + b)
                                   for a, b in ((dr, 0), (0, dc), (dr, dc)))):
                    continue
                length = 0
                while (_inside(mask, r + length * dr, c + length * dc)
                       and depth[r + length * dr, c + length * dc] == length):
                    length += 1
                result.append((r, c, dr, dc, length))
    return result


def _roles(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    order = np.argsort(-counts, kind="stable")
    neutral, divider, third = map(int, order[:3])
    if not counts[neutral] > counts[divider] > counts[third]:
        raise ValueError("neutral and divider must be uniquely first and second in frequency")
    return neutral, divider


def _decode(inp, mask, depth, neutral):
    cues = [(int(r), int(c)) for r, c in np.argwhere(mask & (inp != neutral))]
    if not cues:
        return (neutral,), ()
    length = max(int(depth[r, c]) for r, c in cues) + 1
    candidates = []
    for r, c, dr, dc, capacity in _paths(mask, depth):
        if capacity >= length and all(
                (rr, cc) == (r + int(depth[rr, cc]) * dr, c + int(depth[rr, cc]) * dc)
                for rr, cc in cues):
            period = tuple(int(inp[r + d * dr, c + d * dc]) for d in range(length))
            candidates.append((period, (dr, dc)))
    if not candidates or len({p for p, _ in candidates}) != 1:
        raise ValueError("region key must determine one period")
    return candidates[0][0], tuple(sorted({d for _, d in candidates}))


def _concave(mask):
    rr, cc = np.nonzero(mask)
    return mask.sum() < (int(rr.max() - rr.min()) + 1) * (int(cc.max() - cc.min()) + 1)


def _complete(inp):
    """Input-only reconstruction of A's completed, unsheared rectangle."""
    neutral, divider = _roles(inp)
    wall = inp == divider
    if len(_components(wall)) != 1:
        raise ValueError("divider must be connected")
    if np.any(wall[:-1, :-1] & wall[1:, :-1] & wall[:-1, 1:] & wall[1:, 1:]):
        raise ValueError("divider cannot contain a solid 2 by 2 block")
    components = _components(~wall)
    if len(components) < 2:
        raise ValueError("divider must separate at least two regions")
    filled = inp.copy()
    regions = []
    for mask in components:
        depth = _erosion_depth(mask)
        period, directions = _decode(inp, mask, depth, neutral)
        filled[mask] = np.asarray(period, dtype=np.int8)[depth[mask] % len(period)]
        regions.append({"mask": mask, "depth": depth, "period": period,
                        "directions": directions, "concave": bool(_concave(mask))})
    return filled, {"neutral": neutral, "divider": divider, "wall": wall, "regions": regions}


def _shear(filled, neutral):
    h, w = filled.shape
    out = np.full((h, h + w - 1), neutral, dtype=np.int8)
    for r in range(h):
        start = h - 1 - r
        out[r, start:start + w] = filled[r]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    if sum(inp.shape) - 1 > 30:
        raise ValueError("sheared output would exceed 30 columns")
    filled, info = _complete(inp)
    return _shear(filled, info["neutral"])


def _contour_witness(info):
    """A colored concavity refutes both bounding-box rings and four-neighbor depth."""
    for region in info["regions"]:
        if not region["concave"] or len(region["period"]) < 2:
            continue
        mask, depth = region["mask"], region["depth"]
        rr, cc = np.nonzero(mask)
        box = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
        period = np.asarray(region["period"])
        actual = period[depth[mask] % len(period)]
        if (np.any(actual != period[box % len(period)])
                and np.any(actual != period[_erosion_depth(mask, False)[mask] % len(period)])):
            return True
    return False


def _jog(rng, wall, budget):
    """Move isolated straight intervals sideways, retaining orthogonal connectivity."""
    for _ in range(budget):
        for _ in range(30):
            work = wall.T if rng.integers(2) else wall
            points = np.argwhere(work)
            r, c = map(int, points[int(rng.integers(len(points)))])
            end = r + int(rng.integers(3, 8))
            shift = int(rng.choice((-2, -1, 1, 2)))
            lo, hi = sorted((c, c + shift))
            if r < 1 or end + 1 >= work.shape[0] or lo < 2 or hi + 2 >= work.shape[1]:
                continue
            block = work[r - 1:end + 2, lo - 1:hi + 2]
            expected = np.zeros_like(block)
            expected[:, c - lo + 1] = True
            if not np.array_equal(block, expected):
                continue
            work[r + 1:end, c] = False
            work[r, lo:hi + 1] = True
            work[end, lo:hi + 1] = True
            work[r:end + 1, c + shift] = True
            break


def _layout(rng, shape, target, anchor=False):
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    if anchor:
        # Reserve a deep room, two smaller rooms, and a contour-changing jog.
        cut = int(rng.integers(7, min(9, w - 5)))
        a = int(rng.integers(3, 5))
        b = h - int(rng.integers(3, 5))
        shift = int(rng.integers(1, min(3, w - cut - 3)))
        wall[:, cut] = True
        wall[a + 1:b, cut] = False
        wall[a, cut:cut + shift + 1] = True
        wall[b, cut:cut + shift + 1] = True
        wall[a:b + 1, cut + shift] = True
        branch = (a + b) // 2
        wall[branch, cut + shift:] = True
    else:
        # Successive guillotine cuts join existing walls or the canvas edge.
        rooms = [(0, h, 0, w)]
        while len(rooms) < target:
            choices = [(i, axis) for i, (a, b, c, d) in enumerate(rooms)
                       for axis, size in ((0, b - a), (1, d - c)) if size >= 7]
            if not choices:
                break
            i, axis = choices[int(rng.integers(len(choices)))]
            a, b, c, d = rooms.pop(i)
            if axis == 0:
                q = int(rng.integers(a + 3, b - 3))
                wall[q, max(0, c - 1):min(w, d + 1)] = True
                rooms += [(a, q, c, d), (q + 1, b, c, d)]
            else:
                q = int(rng.integers(c + 3, d - 3))
                wall[max(0, a - 1):min(h, b + 1), q] = True
                rooms += [(a, b, c, q), (a, b, q + 1, d)]
        # Optional U-shaped attachment: a small enclosure and surrounding concavity.
        for idx in rng.permutation(len(rooms)):
            if rng.random() > 0.28:
                continue
            a, b, c, d = rooms[int(idx)]
            work = wall
            if rng.integers(2):
                work = wall.T
                a, b, c, d = c, d, a, b
            if b - a < 8 or d - c < 5 or c == 0 or not work[a:b, c - 1].all():
                continue
            top = int(rng.integers(a + 1, b - 5))
            bottom = int(rng.integers(top + 4, b - 1))
            far = int(rng.integers(c + 3, d - 1))
            work[top, c - 1:far + 1] = True
            work[bottom, c - 1:far + 1] = True
            work[top:bottom + 1, far] = True
            break
        _jog(rng, wall, int(rng.integers(0, 4)))
    if rng.integers(2):
        wall = wall[::-1].copy()
    if rng.integers(2):
        wall = wall[:, ::-1].copy()
    return wall


def _construct(rng, wall, neutral, divider, kind):
    components = _components(~wall)
    depths = [_distance_depth(m) for m in components]
    # Training teaches both horizontal key directions, then tests reverse vertically.
    paths = [[p for p in _paths(m, d) if kind == "test" or p[2] == 1]
             for m, d in zip(components, depths)]
    assert all(paths)
    capacity = [max(p[4] for p in ps) for ps in paths]
    assignments = {}
    if kind == "anchor":
        assert len(components) == 3
        big = int(np.argmax(capacity))
        assert capacity[big] >= 4
        assignments[big] = (3, 1, None)
        rest = [i for i in range(3) if i != big]
        two = max(rest, key=lambda i: capacity[i])
        assert capacity[two] >= 2
        assignments[two] = (2, None, None)
        assignments[next(i for i in rest if i != two)] = (1, None, None)
    elif kind == "second":
        lead = int(np.argmax(capacity))
        assert capacity[lead] >= 2
        assignments[lead] = (2, 0, None)
        assignments[next(i for i in range(len(components)) if i != lead)] = (0, None, None)
    elif kind == "test":
        options = [(i, p) for i, ps in enumerate(paths) for p in ps if p[2] == -1 and p[4] >= 2]
        assert options
        i, path = options[int(rng.integers(len(options)))]
        assignments[i] = (int(rng.integers(2, min(4, path[4]) + 1)), None, path)
    inp = np.full(wall.shape, neutral, dtype=np.int8)
    filled = inp.copy()
    inp[wall] = filled[wall] = divider
    palette = [c for c in range(10) if c not in (neutral, divider)]
    for i, (mask, depth, options) in enumerate(zip(components, depths, paths)):
        if i in assignments:
            length, gap, path = assignments[i]
        else:
            length = 0 if rng.random() < 0.18 else int(rng.integers(1, min(4, capacity[i]) + 1))
            gap, path = None, None
        if length == 0:
            continue
        period = [int(c) for c in rng.choice(palette, size=length, replace=False)]
        if gap is not None:
            period[gap] = neutral
        elif length >= 2 and rng.random() < 0.22 and kind != "test":
            period[int(rng.integers(length - 1))] = neutral
        available = [p for p in options if p[4] >= length]
        path = path or available[int(rng.integers(len(available)))]
        r, c, dr, dc, _ = path
        for k, color in enumerate(period):
            inp[r + k * dr, c + k * dc] = color
        filled[mask] = np.asarray(period, dtype=np.int8)[depth[mask] % length]
    return {"input": inp, "output": _shear(filled, neutral)}


def _assert_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    h, w = inp.shape
    assert h + w - 1 <= 30 and h > 1
    assert out.shape == (h, h + w - 1)
    filled, info = _complete(inp)
    assert not np.array_equal(filled, inp), "A must change some payload cells"
    # Check individual coordinate copies and padding, not merely solve equality.
    for r in range(h):
        left = h - 1 - r
        assert np.array_equal(out[r, left:left + w], filled[r])
        assert np.all(out[r, :left] == info["neutral"])
        assert np.all(out[r, left + w:] == info["neutral"])
    # All divider cells move with their row, and no divider is invented as a band.
    assert np.count_nonzero(out == info["divider"]) == np.count_nonzero(info["wall"])
    assert set(np.unique(out)) <= set(np.unique(inp))
    assert not np.array_equal(out, _shear(inp, info["neutral"])), "omitting A must fail"
    return info


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    role_order = [int(c) for c in rng.permutation(10)]
    role_offset = int(rng.integers(1, 10))
    puzzle = {"train": [], "test": []}
    seen, used_shapes = set(), set()
    for i in range(num_train + num_test):
        kind = "anchor" if i == 0 else "second" if i == 1 else "train" if i < num_train else "test"
        neutral = role_order[i % 10]
        divider = role_order[(i + role_offset) % 10]
        for attempt in range(64):
            if kind == "anchor":
                shapes = [(h, w) for h in range(12, 18) for w in range(14, 20) if 28 <= h + w <= 31]
            else:
                low, high = (8, 24) if kind == "test" else (7, 23)
                shapes = [(h, w) for h in range(low, high) for w in range(low, high) if h + w <= 31]
                if kind == "test" and rng.random() < 0.45:
                    shapes = [s for s in shapes if sum(s) == 31]
            shapes = [s for s in shapes if s not in used_shapes]
            if kind == "second":
                first_height = puzzle["train"][0]["input"].shape[0]
                shapes = [s for s in shapes if s[0] != first_height]
            shape = shapes[int(rng.integers(len(shapes)))]
            target = int(rng.integers(2, 8 if kind == "test" else 6))
            wall = _layout(rng, shape, target, kind == "anchor")
            try:
                pair = _construct(rng, wall, neutral, divider, kind)
                info = _assert_pair(pair)
                if kind == "anchor":
                    assert _contour_witness(info)
                if kind == "second":
                    assert len(info["regions"]) != 3
                key = _signature(pair["input"])
                assert key not in seen
            except (AssertionError, ValueError):
                continue
            puzzle["train" if i < num_train else "test"].append(pair)
            seen.add(key)
            used_shapes.add(shape)
            break
        else:
            raise RuntimeError(f"bounded generation exhausted for example {i}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _assert_pair(pair)
            key = _signature(np.asarray(pair["input"]))
            assert key not in seen
            seen.add(key)
            analyses[split].append(info)
    train = [(a["neutral"], r) for a in analyses["train"] for r in a["regions"]]
    assert {1, 2, 3} <= {len(r["period"]) for n, r in train if r["period"] != (n,)}
    assert any(r["period"] == (n,) for n, r in train)
    assert any(len(r["period"]) > 1 and r["period"][0] == n for n, r in train)
    assert any(n in r["period"][1:-1] for n, r in train)
    assert any(len(r["period"]) >= 3 and int(r["depth"].max()) >= len(r["period"]) for _, r in train)
    assert any(_contour_witness(a) for a in analyses["train"])
    # Every test shifts key orientation; the depth convention stays unchanged.
    for a in analyses["test"]:
        assert any(len(r["period"]) >= 2 and r["directions"]
                   and all(dr == -1 for dr, _ in r["directions"]) for r in a["regions"])
    combined = analyses["train"] + analyses["test"]
    assert len({a["neutral"] for a in combined}) >= 3
    assert len({a["divider"] for a in combined}) >= 3
    assert len({len(a["regions"]) for a in combined}) >= 2
    assert len({np.asarray(p["input"]).shape[0] for p in puzzle["train"]}) >= 2


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, bounded in memory; return descriptive metrics, save no payloads."""
    shapes, geometries, roles, lengths, directions, region_counts, digests = (set() for _ in range(7))
    pairs = concavities = cycles = boundary = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        if seed == 0:
            assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                assert np.array_equal(solve(inp), pair["output"])
                _, a = _complete(inp)
                pairs += 1
                shapes.add(inp.shape)
                roles.add((a["neutral"], a["divider"]))
                wall = a["wall"]
                geometries.add(_signature(wall.astype(np.int8)))
                region_counts.add(len(a["regions"]))
                edges = int((wall[:-1] & wall[1:]).sum() + (wall[:, :-1] & wall[:, 1:]).sum())
                cycles += edges >= int(wall.sum())
                boundary += sum(inp.shape) - 1 == 30
                for r in a["regions"]:
                    assert np.array_equal(_distance_depth(r["mask"]), r["depth"])
                    lengths.add(len(r["period"]))
                    directions.update(r["directions"])
                    concavities += r["concave"]
    assert len(digests) == 42 and len(shapes) >= 50 and len(geometries) >= 100
    assert len(roles) >= 35 and len(region_counts) >= 4
    assert lengths == {1, 2, 3, 4}
    assert directions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert concavities >= 60 and boundary >= 15 and cycles >= 1
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(6000 + 10 * nt + ne, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject an incorrect divider placement and an incorrect padding cell.
    original = generate_puzzle(42, 2, 1)
    for damage in ("divider", "padding"):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        pair = broken["test"][0]
        inp, out = pair["input"], pair["output"]
        _, a = _complete(inp)
        if damage == "divider":
            r, c = map(int, np.argwhere(a["wall"])[0])
            out[r, inp.shape[0] - 1 - r + c] = a["neutral"]
        else:
            out[0, 0] = a["divider"]
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{damage} corruption accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(digests),
            "input_shapes": len(shapes), "divider_geometries": len(geometries),
            "role_pairs": len(roles), "region_counts": sorted(region_counts),
            "period_lengths": sorted(lengths), "concave_regions": int(concavities),
            "wall_cycles": int(cycles), "width_30_outputs": int(boundary)}
