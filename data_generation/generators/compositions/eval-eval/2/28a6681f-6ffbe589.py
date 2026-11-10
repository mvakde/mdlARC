"""Flood-contact rotors: a liquid probe winds independent terrain layers.

Family conventions: background 0, liquid 1, neutral chamber frame 9.
Colors 2..8 are both impermeable terrain and the layers of the final imprint.
No parent module or example is consulted by generation or solving.
"""
from collections import Counter
import numpy as np


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, points = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    points.append(p)
        part = np.zeros(mask.shape, dtype=bool)
        rr, cc = zip(*points)
        part[rr, cc] = True
        result.append(part)
    return result


def _adjacent(mask):
    near = np.zeros_like(mask, dtype=bool)
    near[1:] |= mask[:-1]
    near[:-1] |= mask[1:]
    near[:, 1:] |= mask[:, :-1]
    near[:, :-1] |= mask[:, 1:]
    return near


def _pour(board, reverse=False, top_first=False):
    """Unit pouring with lateral motion only after support is reached.

    Generated members have complete contained levels and agree for all four
    pour orders. The switches are verification probes, not hidden conventions.
    The cropped board's four boundaries are impermeable.
    """
    n = len(board)
    points = sorted(map(tuple, np.argwhere(board == 1).tolist()),
                    key=lambda p: (p[0] if top_first else -p[0],
                                   -p[1] if reverse else p[1]))
    result = board.copy()
    result[result == 1] = 0
    for r, c in points:
        while r > 0 and result[r, c] == 1:
            r -= 1
        if result[r, c] != 0:
            raise ValueError("liquid trapped beneath an already filled ceiling")
        for _ in range(n + 1):
            if r + 1 < n and result[r+1, c] == 0:
                r += 1
                continue
            left = right = c
            while left > 0 and result[r, left-1] == 0:
                left -= 1
            while right+1 < n and result[r, right+1] == 0:
                right += 1
            outlets = [] if r == n-1 else [x for x in range(left, right+1)
                                          if result[r+1, x] == 0]
            if not outlets:
                result[r, c] = 1
                break
            c = min(outlets, key=lambda x: (abs(x-c), -x if reverse else x))
            r += 1
        else:
            raise AssertionError("pour did not terminate")
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or a.dtype.kind not in "iu" or
            not all(1 <= d <= 30 for d in a.shape) or
            np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _locate(input_grid):
    a = _grid(input_grid)
    coords = np.argwhere(a == 9)
    if len(coords) == 0:
        raise ValueError("missing neutral frame")
    r, c = map(int, coords.min(axis=0))
    bottom, right = map(int, coords.max(axis=0))
    side = bottom-r+1
    if side != right-c+1 or side < 7:
        raise ValueError("frame must be square")
    frame = np.zeros(a.shape, dtype=bool)
    frame[r, c:right+1] = frame[bottom, c:right+1] = True
    frame[r:bottom+1, c] = frame[r:bottom+1, right] = True
    if not np.array_equal(a == 9, frame):
        raise ValueError("9 must be exactly the frame perimeter")
    exterior = np.ones(a.shape, dtype=bool)
    exterior[r:bottom+1, c:right+1] = False
    if np.any(a[exterior] != 0):
        raise ValueError("only background may lie outside the chamber")
    board = a[r+1:bottom, c+1:right].copy()
    if not np.any(board == 1):
        raise ValueError("missing liquid probe")
    return board, (r, c)


def _contacts(board, settled):
    wet = _adjacent(settled == 1) & (board >= 2) & (board <= 8)
    return {int(color): int(np.sum(wet & (board == color)))
            for color in np.unique(board) if 2 <= color <= 8}


def _imprint(board, counts, coordinate=False):
    n = len(board)
    result = np.zeros_like(board)
    for color, count in counts.items():
        if coordinate:
            for r, c in np.argwhere(board == color):
                r, c = int(r), int(c)
                for _ in range(count % 4):
                    r, c = c, n-1-r
                if result[r, c] != 0:
                    raise ValueError("rotated layers overlap")
                result[r, c] = color
        else:
            mask = np.rot90(board == color, -(count % 4))
            if np.any(mask & (result != 0)):
                raise ValueError("rotated layers overlap")
            result[mask] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the visible chamber, volume, geometry, and wet contacts."""
    board, _ = _locate(input_grid)
    settled = _pour(board)
    return _imprint(board, _contacts(board, settled))


def _branches(pool):
    top = int(np.argwhere(pool)[:, 0].min())
    return max([1] + [len(_components(pool &
                (np.arange(len(pool))[:, None] >= level)))
                for level in range(top+1, len(pool))])


def _capture(solid, pools):
    """Bit sets of all possible dry downward destinations, including spills."""
    n = len(solid)
    labels = np.zeros(solid.shape, dtype=np.int64)
    fixed = np.zeros_like(labels)
    unknown = 1 << len(pools)
    for i, pool in enumerate(pools):
        fixed[pool] = 1 << i
    for r in range(n-1, -1, -1):
        c = 0
        while c < n:
            if solid[r, c]:
                c += 1
                continue
            end = c
            while end+1 < n and not solid[r, end+1]:
                end += 1
            destinations = 0
            for x in range(c, end+1):
                destinations |= int(fixed[r, x])
                if r+1 < n and not solid[r+1, x]:
                    destinations |= int(labels[r+1, x])
            for x in range(c, end+1):
                if fixed[r, x]:
                    labels[r, x] = fixed[r, x]
                elif r+1 < n and not solid[r+1, x]:
                    labels[r, x] = labels[r+1, x]
                else:
                    labels[r, x] = destinations or unknown
            c = end+1
    return labels


def _terrain(rng, n):
    """Grounded stepped/sloping masses, suspended cups, and dry deflectors."""
    solid = np.zeros((n, n), dtype=bool)
    # A varying height field creates low pockets with joining spill levels.
    edges = sorted(set([0, n] + rng.integers(1, n, size=int(rng.integers(3, 7))).tolist()))
    for left, right in zip(edges, edges[1:]):
        floor = int(rng.integers(n-6, n))
        slope = int(rng.choice([-1, 0, 0, 1]))
        for x in range(left, right):
            y = int(np.clip(floor+slope*(x-left), n-6, n-1))
            solid[y:, x] = True
    # Elevated containers allow independent pools and inaccessible low gaps.
    for _ in range(int(rng.integers(1, 4))):
        width = int(rng.integers(3, min(8, n-1)))
        left = int(rng.integers(0, n-width+1))
        bottom = int(rng.integers(3, max(4, n-4)))
        depth = int(rng.integers(2, min(5, bottom)+1))
        solid[bottom, left:left+width] = True
        solid[bottom-depth:bottom+1, left] = True
        solid[bottom-depth:bottom+1, left+width-1] = True
    # Unsubmerged islands have a meaningful zero-contact (unmarked) role.
    for _ in range(int(rng.integers(1, 4))):
        r = int(rng.integers(0, max(1, n//3)))
        c = int(rng.integers(0, n-2))
        solid[r, c:c+int(rng.integers(1, 4))] = True
    if rng.integers(2):
        solid = solid[:, ::-1].copy()
    return solid


def _choose_pools(rng, solid, mode):
    n = len(solid)
    candidates = []
    for level in range(2, n):
        for pool in _components(~solid & (np.arange(n)[:, None] >= level)):
            volume = int(pool.sum())
            if 2 <= volume <= min(100, n*n//3) and pool[level].any():
                if np.any(pool[level] & ~solid[level-1]):
                    candidates.append(pool)
    rng.shuffle(candidates)
    required = [p for p in candidates if (_branches(p) == 1 if mode == 0 else _branches(p) >= 2)]
    if not required:
        return []
    chosen = [required[0]]
    wanted = 1 if mode == 0 else (int(rng.integers(2, 4)) if mode == 2 else int(rng.integers(1, 3)))
    occupied = chosen[0].copy()
    for pool in candidates:
        if len(chosen) >= wanted:
            break
        if not np.any(pool & (occupied | _adjacent(occupied))):
            chosen.append(pool)
            occupied |= pool
    return chosen if mode != 2 or len(chosen) >= 2 else []


def _supply(rng, labels, pools, style, separate=False):
    source = np.zeros(labels.shape, dtype=bool)
    for i, pool in enumerate(pools):
        eligible = np.argwhere(labels == 1 << i)
        outside = np.argwhere((labels == 1 << i) & ~pool)
        volume = int(pool.sum())
        if len(eligible) < volume or len(outside) == 0:
            return None
        if separate:
            if len(outside) < volume:
                return None
            eligible = outside
        anchor = outside[int(rng.integers(len(outside)))]
        dr, dc = np.abs(eligible-anchor).T
        jitter = rng.random(len(eligible))
        if style == 0:
            score = dr + dc + jitter
        elif style == 1:
            score = jitter
        elif style == 2:
            score = dr*3 + dc*.1 + jitter
        elif style == 3:
            score = dc*3 + dr*.1 + jitter
        else:
            score = (~pool[eligible[:, 0], eligible[:, 1]])*3 + jitter
        points = eligible[np.argsort(score)[:volume]].copy()
        if not np.any(np.all(points == anchor, axis=1)):
            points[-1] = anchor
        source[points[:, 0], points[:, 1]] = True
    return source


def _orbits(n):
    remaining = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while remaining:
        r, c = min(remaining)
        orbit = []
        for _ in range(4):
            if (r, c) not in orbit:
                orbit.append((r, c))
            r, c = c, n-1-r
        remaining.difference_update(orbit)
        result.append(orbit)
    return result


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _paint(rng, solid, pools, witness):
    """Colors own rotation orbits; wet orbits encode teaching witnesses.

    Ownership is a generation constraint only, not visible metadata. It avoids
    overlap for ALL rotations, not merely the sampled wet-contact counts.
    """
    n = len(solid)
    wet = solid & _adjacent(np.logical_or.reduce(pools))
    active = []
    for orbit in _orbits(n):
        cells = [p for p in orbit if solid[p]]
        if cells:
            active.append((cells, sum(bool(wet[p]) for p in cells)))
    count = 5 if witness == 0 else 4 if witness == 1 else int(rng.integers(2, 8))
    if len(active) < count+3:
        return None
    colors = list(map(int, rng.choice(np.arange(2, 9), count, replace=False)))
    owners = {}
    centers = rng.uniform(0, n, (count, 2))
    style = int(rng.integers(2))
    def spatial_owner(cells):
        r, c = cells[0]
        if style == 0:
            return min(count-1, min(r, c, n-1-r, n-1-c)*count//max(1, n//2))
        scores = [min((r-y)**2+(c-x)**2 for r, c in cells) for y, x in centers]
        return int(np.argmin(scores))
    wet_ids = [i for i, (_, weight) in enumerate(active) if weight]
    dry_ids = [i for i, (_, weight) in enumerate(active) if not weight]
    rng.shuffle(wet_ids)
    rng.shuffle(dry_ids)
    if witness >= 0:
        goals = [1, 2, 4] if witness == 0 else [3, 5]
        units = [i for i in wet_ids if active[i][1] == 1]
        if len(units) < sum(goals)+1 or len(dry_ids) < count:
            return None
        cursor = 0
        for k, goal in enumerate(goals):
            for i in units[cursor:cursor+goal]:
                owners[i] = k
            cursor += goal
        for i in wet_ids:
            if i not in owners:
                owners[i] = len(goals)
        # Last layer is wholly dry; every layer receives a dry orbit too.
        for k, i in enumerate(dry_ids[:count]):
            owners[i] = k
        for i in dry_ids[count:]:
            owners[i] = spatial_owner(active[i][0])
    else:
        # Spatial orbit territories vary between shells, patches and fragments.
        for i, (cells, _) in enumerate(active):
            owners[i] = spatial_owner(cells)
        # Keep every sampled layer present without an overlap priority.
        for k, i in enumerate(rng.choice(len(active), count, replace=False)):
            owners[int(i)] = k
    board = np.zeros(solid.shape, dtype=np.int8)
    for i, (cells, _) in enumerate(active):
        for p in cells:
            board[p] = colors[owners[i]]
    if witness >= 0 and not all(_asymmetric(board == c) for c in colors):
        return None
    return board


def _embed(rng, board):
    n = len(board)
    h = n+2+int(rng.integers(0, min(6, 29-n)))
    w = n+2+int(rng.integers(0, min(6, 29-n)))
    r = int(rng.integers(h-n-1))
    c = int(rng.integers(w-n-1))
    inp = np.zeros((h, w), dtype=np.int8)
    inp[r:r+n+2, c:c+n+2] = 9
    inp[r+1:r+n+1, c+1:c+n+1] = board
    return inp


def _make_pair(rng, mode, witness, test):
    for _ in range(300):
        n = int(rng.integers(11 if test else 9, 22))
        solid = _terrain(rng, n)
        pools = _choose_pools(rng, solid, mode)
        if not pools:
            continue
        board = _paint(rng, solid, pools, witness)
        if board is None:
            continue
        labels = _capture(solid, pools)
        source = _supply(rng, labels, pools, int(rng.integers(5)),
                         separate=witness >= 0)
        if source is None:
            continue
        board[source] = 1
        target = board.copy()
        target[target == 1] = 0
        target[np.logical_or.reduce(pools)] = 1
        try:
            agreement = all(np.array_equal(_pour(board, rev, order), target)
                            for rev in (False, True) for order in (False, True))
        except ValueError:
            continue
        if not agreement:
            continue
        counts = _contacts(board, target)
        if test and not any(k >= 5 and k % 4 for k in counts.values()):
            continue
        out = _imprint(board, counts)
        # Each pair is informative; either inherited mechanism matters here.
        if np.array_equal(out, _imprint(board, {c: 0 for c in counts})):
            continue
        if np.array_equal(out, _imprint(board, _contacts(board, board))):
            continue
        return {"input": _embed(rng, board), "output": out}
    raise RuntimeError("bounded family sampling exhausted")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test) or
            not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or
            num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = index if split == "train" and index < 2 else -1
            mode = (0 if index == 0 else 1 if index == 1 else int(rng.integers(3))) if split == "train" else 2
            for _ in range(10):
                pair = _make_pair(rng, mode, witness, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unable to produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _analyze(pair):
    board, position = _locate(pair["input"])
    out = _grid(pair["output"])
    n = len(board)
    assert out.shape == (n, n)
    solid = board >= 2
    settled = _pour(board)
    water = settled == 1
    assert int(water.sum()) == int((board == 1).sum()) > 0
    assert np.array_equal(settled[solid], board[solid])
    assert np.array_equal(settled >= 2, solid)
    assert np.argwhere(water)[:, 0].sum() > np.argwhere(board == 1)[:, 0].sum()
    pools = _components(water)
    for pool in pools:
        top = int(np.argwhere(pool)[:, 0].min())
        available = ~solid & (np.arange(n)[:, None] >= top)
        containing = [p for p in _components(available) if np.any(p & pool)]
        assert len(containing) == 1 and np.array_equal(containing[0], pool), "partial or leaking pool"
    labels = _capture(solid, pools)
    accounted = np.zeros_like(water)
    for i, pool in enumerate(pools):
        supply = (board == 1) & (labels == 1 << i)
        assert int(supply.sum()) == int(pool.sum()), "ambiguous catchment allocation"
        accounted |= supply
    assert np.array_equal(accounted, board == 1)
    for reverse in (False, True):
        for top_first in (False, True):
            assert np.array_equal(_pour(board, reverse, top_first), settled), "order-dependent liquid"
    counts = _contacts(board, settled)
    # Check distinct terrain-cell contacts directly, independently of dilation.
    explicit = Counter()
    for r, c in np.argwhere(solid):
        if any(0 <= y < n and 0 <= x < n and water[y, x]
               for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))):
            explicit[int(board[r, c])] += 1
    assert counts == {c: explicit[c] for c in counts}
    assert np.array_equal(out, _imprint(board, counts, coordinate=True))
    assert np.array_equal(out, solve(pair["input"]))
    # Orbit ownership is checked from actual cells, not sampled assignments.
    for orbit in _orbits(n):
        assert len({int(board[p]) for p in orbit if board[p] >= 2}) <= 1
    for c in counts:
        assert int((board == c).sum()) == int((out == c).sum())
    dry = _imprint(board, {c: 0 for c in counts})
    initial = _imprint(board, _contacts(board, board))
    assert not np.array_equal(out, dry), "B ablation did not change output"
    assert not np.array_equal(out, initial), "A ablation did not change output"
    return {"size": n, "shape": tuple(np.asarray(pair["input"]).shape),
            "position": position, "layers": len(counts), "volume": int(water.sum()),
            "counts": counts, "asymmetric": {c for c in counts if _asymmetric(board == c)},
            "pools": len(pools), "branches": max(_branches(p) for p in pools),
            "levels": tuple(sorted({int(np.argwhere(p)[:, 0].min()) for p in pools})),
            "source_components": len(_components(board == 1)),
            "retained": int(((board == 1) & water).sum()),
            "lateral": bool(np.any((board == 1).sum(axis=0) != water.sum(axis=0))),
            "terrain": _key(solid.astype(np.int8))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            analyses[split].append(_analyze(pair))
    first, second = analyses["train"][:2]
    assert first["pools"] == 1 and first["branches"] == 1
    assert second["branches"] >= 2
    assert first["retained"] == second["retained"] == 0
    for stats, required in ((first, {0, 1, 2, 4}), (second, {0, 3, 5})):
        observed = {stats["counts"][c] for c in stats["asymmetric"]}
        assert required <= observed, "insufficient contact/count/direction evidence"
    for stats in analyses["test"]:
        assert stats["pools"] >= 2 and stats["branches"] >= 2
        assert any(k >= 5 and k % 4 for k in stats["counts"].values())
    assert len({s["terrain"] for split in analyses.values() for s in split}) >= 2


def self_test():
    """Bounded authored checks over 42 seeds; no independent semantic oracle."""
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            records.extend(_analyze(p) for p in puzzle[split])
    assert len(signatures) == 42
    assert len({s["shape"] for s in records}) >= 50
    assert len({s["terrain"] for s in records}) >= 180
    assert len({s["volume"] for s in records}) >= 35
    assert len({s["size"] for s in records}) >= 10
    assert {s["layers"] for s in records} >= {2, 3, 4, 5, 6, 7}
    assert len({s["position"] for s in records}) >= 15
    assert sum(s["pools"] > 1 for s in records) >= 50
    assert sum(len(s["levels"]) > 1 for s in records) >= 40
    assert sum(s["source_components"] > 1 for s in records) >= 80
    assert sum(s["lateral"] for s in records) >= 80
    assert sum(s["retained"] > 0 for s in records) >= 80
    # A multiply exposed corner is a single unary token, not several faces.
    board = np.zeros((5, 5), dtype=np.int8)
    board[3, 2] = 2
    settled = board.copy()
    settled[3, 1] = settled[3, 3] = settled[2, 2] = 1
    assert _contacts(board, settled) == {2: 1}
    # Equal volume, different basin widths => different contact counts.
    # Two six-cell pools: width 3/depth 2 has seven colored wet cells;
    # width 2/depth 3 has eight. Both have a neutral chamber boundary.
    totals = []
    for width, depth in ((3, 2), (2, 3)):
        probe = np.zeros((9, 9), dtype=np.int8)
        probe[7, 2:width+4] = 2
        probe[3:8, 2] = probe[3:8, width+3] = 2
        probe[0:6, 3] = 1
        settled = _pour(probe)
        assert int((settled == 1).sum()) == 6
        assert int((settled == 1).sum(axis=1).max()) == width
        assert int(np.any(settled == 1, axis=1).sum()) == depth
        totals.append(_contacts(probe, settled)[2])
    assert totals == [7, 8]
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    p = generate_puzzle(2026, 2, 1)
    q = generate_puzzle(2026, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in p
               for x, y in zip(p[s], q[s]) for k in x)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = 1
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt readout accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
