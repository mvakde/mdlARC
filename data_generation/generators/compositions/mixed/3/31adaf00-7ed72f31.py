"""Marker-selected symmetry copies into locally maximal blank squares.

The solver reads only its array. Generation keeps an independent painted label,
then checks the complete assembled scene against the public rule. No files,
traces, caches, network calls, or parent generators are used at runtime.
"""
from collections import Counter, deque
import hashlib
import numpy as np


BLANK = 0
MARKER = 5


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid of dimensions 1..30")
    return a


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in steps:
                q = (p[0] + dr, p[1] + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(sorted(cells))
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows), max(cols)


def _wall(a):
    colors, counts = np.unique(a, return_counts=True)
    keep = (colors != BLANK) & (colors != MARKER)
    colors, counts = colors[keep], counts[keep]
    if not len(colors) or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("Wall must be the unique most frequent nonzero, nonmarker color")
    return int(colors[counts.argmax()])


def _squares(a, maximal=True):
    """Enumerate square windows; a partial exterior strip is not an extension."""
    blank = a == BLANK
    integral = np.pad((~blank).astype(np.int32), ((1, 0), (1, 0)))
    integral = integral.cumsum(0).cumsum(1)
    h, w = a.shape
    result = []
    for side in range(2, min(h, w) + 1):
        sums = (integral[side:, side:] - integral[:-side, side:]
                - integral[side:, :-side] + integral[:-side, :-side])
        for r, c in np.argwhere(sums == 0):
            r, c = int(r), int(c)
            if maximal and (
                (r > 0 and blank[r - 1, c:c + side].all())
                or (r + side < h and blank[r + side, c:c + side].all())
                or (c > 0 and blank[r:r + side, c - 1].all())
                or (c + side < w and blank[r:r + side, c + side].all())
            ):
                continue
            result.append((r, c, side))
    return result


def _kind(marker):
    r, c, b, d = _bbox(marker)
    if len(marker) == 1:
        return "P"
    if r == b and d - c + 1 == len(marker):
        return "H"
    if c == d and b - r + 1 == len(marker):
        return "V"
    raise ValueError("Each four-connected marker component must be straight")


def _contact(x, y):
    return any(max(abs(r - u), abs(c - v)) == 1
               for r, c in x for u, v in y)


def _mapping(marker, kind, payload):
    if not _contact(marker, payload):
        return None
    ar, ac = marker[0]
    r, c, b, d = _bbox(payload)
    mr, mc, mb, md = _bbox(marker)
    if kind == "H":
        if not (b < ar or r > ar) or d < mc or c > md:
            return None
        return [(2 * ar - u, v) for u, v in payload]
    if kind == "V":
        if not (d < ac or c > ac) or b < mr or r > mb:
            return None
        return [(u, 2 * ac - v) for u, v in payload]
    if not ((b <= ar or r >= ar) and (d <= ac or c >= ac)):
        return None
    return [(2 * ar - u, 2 * ac - v) for u, v in payload]


def _adjacent(marker, kind, square):
    r, c, n = square
    b, d = r + n - 1, c + n - 1
    mr, mc, mb, md = _bbox(marker)
    if kind == "H":
        return mr in (r - 1, b + 1) and max(mc, c) <= min(md, d)
    if kind == "V":
        return mc in (c - 1, d + 1) and max(mr, r) <= min(mb, b)
    # An exterior singleton can touch a corner or a side, including alignment.
    distance = max(max(r - mr, 0, mr - b), max(c - mc, 0, mc - d))
    return distance == 1


def _inside(cells, square):
    r, c, n = square
    return all(r <= u < r + n and c <= v < c + n for u, v in cells)


def _analyze(a, maximal=True):
    wall = _wall(a)
    markers = _components(a == MARKER)
    kinds = [_kind(m) for m in markers]
    payloads = []
    for color in sorted(set(map(int, np.unique(a))) - {BLANK, MARKER, wall}):
        payloads.extend(_components(a == color, diagonal=True))
    squares = _squares(a, maximal)
    matches = []
    for mi, (marker, kind) in enumerate(zip(markers, kinds)):
        for pi, payload in enumerate(payloads):
            dest = _mapping(marker, kind, payload)
            if dest is None:
                continue
            for si, square in enumerate(squares):
                if _adjacent(marker, kind, square) and _inside(dest, square):
                    matches.append({"si": si, "mi": mi, "pi": pi,
                                    "kind": kind, "source": payload,
                                    "dest": dest, "square": square,
                                    "marker": marker})
    degrees = {key: Counter(m[key] for m in matches) for key in ("si", "mi", "pi")}
    eligible = [m for m in matches if all(degrees[k][m[k]] == 1 for k in degrees)]
    # All decisions use the original input. Reject every colliding claim at once.
    counts = Counter(q for m in eligible for q in m["dest"])
    accepted = [m for m in eligible if all(counts[q] == 1 for q in m["dest"])]
    return {"wall": wall, "markers": markers, "payloads": payloads,
            "squares": squares, "matches": matches, "accepted": accepted}


def _paint(a, analysis, translate=False):
    out = a.copy()
    for m in analysis["accepted"]:
        dest = m["dest"]
        if translate:
            sr, sc, _, _ = _bbox(m["source"])
            dr, dc, _, _ = _bbox(dest)
            dest = [(r + dr - sr, c + dc - sc) for r, c in m["source"]]
        for source, target in zip(m["source"], dest):
            out[target] = a[source]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the fixed family rule without training pairs or generation state."""
    a = _grid(input_grid)
    return _paint(a, _analyze(a))


def _orientation_witness(m):
    sr, sc, _, _ = _bbox(m["source"])
    dr, dc, _, _ = _bbox(m["dest"])
    shifted = {(r + dr - sr, c + dc - sc) for r, c in m["source"]}
    return shifted != set(m["dest"])


def _shape(rng, h, w, kind, witness):
    """Eight-connected silhouettes, with holes, diagonal bridges and concavities."""
    family = int(rng.integers(6))
    mask = np.zeros((h, w), dtype=bool)
    if family == 0:
        mask[:] = True
    elif family == 1:
        mask[0, :] = mask[:, 0] = True
    elif family == 2:
        mask[:] = True
        mask[1:, 1:] = False
        if h > 2:
            mask[-1, :max(1, w - 1)] = True
    elif family == 3 and h >= 3 and w >= 3:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        mask[0, -1] = False
    else:
        # A diagonal spine reaches both extents; local growth changes its topology.
        for t in range(max(h, w)):
            r = round(t * (h - 1) / max(1, max(h, w) - 1))
            c = round(t * (w - 1) / max(1, max(h, w) - 1))
            mask[r, c] = True
        for _ in range(int(rng.integers(0, h * w + 1))):
            r, c = map(int, rng.choice(np.argwhere(mask)))
            rr, cc = r + int(rng.integers(-1, 2)), c + int(rng.integers(-1, 2))
            if 0 <= rr < h and 0 <= cc < w:
                mask[rr, cc] = True
    transformed = np.flipud(mask) if kind == "H" else (
        np.fliplr(mask) if kind == "V" else mask[::-1, ::-1])
    if witness and np.array_equal(mask, transformed):
        mask[:] = False
        mask[0, :] = mask[:, 0] = True
    # This also guarantees contact to a corner-centered point marker.
    mask[0, 0] = True
    return mask


def _tile(rng, status, kind, n, wall, palette, full=True, branch=False,
          lock_axis=False):
    cells = {(r, c): BLANK for r in range(n) for c in range(n)}
    additions = {}
    if status == "nonmax":
        extension = int(rng.integers(1, 4))
        cells.update({(r, c): BLANK for r in range(n)
                      for c in range(n, n + extension)})
    if status != "unmarked":
        h = n if full else int(rng.integers(2, n))
        w = n if full else int(rng.integers(2, n))
        if status == "oversize":
            h = n + 1
        # Canonical construction is H or P; V is its transpose below.
        shape_kind = "P" if kind == "P" else "H"
        mask = _shape(rng, h, w, shape_kind, witness=status == "active")
        offset = 0 if kind == "P" else int(rng.integers(0, n - w + 1))
        dest = [(int(r), int(c) + offset) for r, c in np.argwhere(mask)]
        if kind == "P":
            center = (-1, int(rng.choice([-1, 0])))
            marker = [center]
            source = [(2 * center[0] - r, 2 * center[1] - c) for r, c in dest]
        else:
            # Finite axes may be shorter or longer than the payload span.
            start = int(rng.integers(-1, offset + 1))
            end = int(rng.integers(max(start + 1, offset), n + 1))
            marker = [(-1, c) for c in range(start, end + 1)]
            source = [(-2 - r, c) for r, c in dest]
        color = int(rng.choice(palette))
        cells.update({q: MARKER for q in marker})
        cells.update({q: color for q in source})
        if status == "active":
            additions.update({q: color for q in dest})
        if status == "ambiguous":
            # A second independent axis/payload fits the same square.
            # Restrict the first axis endpoint to avoid orthogonal marker joining.
            for q in marker:
                cells.pop(q)
            cells.update({(-1, c): MARKER for c in range(n)})
            cells.update({(r, n): MARKER for r in range(n)})
            second_color = int(rng.choice(palette))
            cells.update({(r, 2 * n - c): second_color for r, c in dest})
    if branch:
        # A narrow partial-side extension preserves the full-side maximality test.
        side = int(rng.choice([0, 1]))
        pos = int(rng.integers(n))
        length = int(rng.integers(1, 4))
        for step in range(length):
            q = (n + step, pos) if side == 0 else (pos, n + step)
            if q not in cells:
                cells[q] = BLANK
    if kind == "V":
        cells = {(c, r): color for (r, c), color in cells.items()}
        additions = {(c, r): color for (r, c), color in additions.items()}
    r0, c0, r1, c1 = _bbox(cells)
    inp = np.full((r1 - r0 + 1, c1 - c0 + 1), wall, dtype=np.int8)
    for (r, c), color in cells.items():
        inp[r - r0, c - c0] = color
    out = inp.copy()
    for (r, c), color in additions.items():
        out[r - r0, c - c0] = color
    rotation = 2 * int(rng.integers(2)) if lock_axis else int(rng.integers(4))
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    if rng.random() < .5:
        inp, out = inp[:, ::-1], out[:, ::-1]
    return inp.copy(), out.copy()


def _pack(rng, tiles, wall):
    area = sum(x.size for x, _ in tiles)
    for attempt in range(32):
        if attempt >= 20:
            height = width = 30
        else:
            base = max(16, int(np.sqrt(area * 1.8)))
            height = min(30, max(max(x.shape[0] for x, _ in tiles),
                                base + int(rng.integers(-3, 7))))
            width = min(30, max(max(x.shape[1] for x, _ in tiles),
                               base + int(rng.integers(-3, 7))))
        inp = np.full((height, width), wall, dtype=np.int8)
        out = inp.copy()
        blocked = np.zeros(inp.shape, dtype=bool)
        ordered = sorted(range(len(tiles)), key=lambda i: -tiles[i][0].size)
        placed = True
        for index in ordered:
            x, y = tiles[index]
            h, w = x.shape
            integral = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0)))
            integral = integral.cumsum(0).cumsum(1)
            sums = (integral[h:, w:] - integral[:-h, w:]
                    - integral[h:, :-w] + integral[:-h, :-w])
            choices = np.argwhere(sums == 0)
            if not len(choices):
                placed = False
                break
            r, c = map(int, choices[int(rng.integers(len(choices)))])
            inp[r:r + h, c:c + w] = x
            out[r:r + h, c:c + w] = y
            blocked[max(0, r - 1):min(height, r + h + 1),
                    max(0, c - 1):min(width, c + w + 1)] = True
        if placed:
            return inp, out
    return None


def _connect(rng, inp, out, wall):
    """Try a narrow blank corridor while preserving windows and teaching evidence."""
    regions = _components(inp == BLANK)
    if len(regions) < 2:
        return inp, out, False
    order = rng.permutation(len(regions)).tolist()
    first, second = regions[order[0]], set(regions[order[1]])
    queue = deque(first)
    previous = {q: None for q in first}
    target = None
    while queue:
        q = queue.popleft()
        if q in second:
            target = q
            break
        r, c = q
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (r + dr, c + dc)
            if (0 <= p[0] < inp.shape[0] and 0 <= p[1] < inp.shape[1]
                    and p not in previous and inp[p] in (BLANK, wall)):
                previous[p] = q
                queue.append(p)
    if target is None:
        return inp, out, False
    path = []
    while target is not None:
        path.append(target)
        target = previous[target]
    if sum(inp[q] == wall for q in path) > 24:
        return inp, out, False
    a, b = inp.copy(), out.copy()
    for q in path:
        if a[q] == wall:
            a[q] = b[q] = BLANK
    # Label equality alone is insufficient: erasing an ambiguous/unmarked square
    # also leaves its output blank, but destroys negative teaching evidence.
    preserve_windows = set(_squares(inp)) <= set(_squares(a))
    old_relaxed = _paint(inp, _analyze(inp, maximal=False))
    had_extra = bool(np.any((old_relaxed != inp) & (out == inp)))
    new_relaxed = _paint(a, _analyze(a, maximal=False))
    keep_extra = not had_extra or bool(np.any((new_relaxed != a) & (b == a)))
    if (preserve_windows and keep_extra and np.count_nonzero(a == wall) > a.size // 2
            and np.array_equal(solve(a), b)):
        return a, b, True
    return inp, out, False


def _make_pair(rng, plan, wall, palette, connect):
    for _ in range(24):
        tiles = [_tile(rng, wall=wall, palette=palette, **spec) for spec in plan]
        packed = _pack(rng, tiles, wall)
        if packed is None:
            continue
        inp, out = packed
        if np.count_nonzero(inp == wall) <= inp.size // 2:
            continue
        if not np.array_equal(solve(inp), out):
            continue
        if connect:
            for _ in range(4):
                inp, out, joined = _connect(rng, inp, out, wall)
                if joined:
                    break
        return {"input": inp, "output": out}
    raise RuntimeError("Bounded scene construction exhausted")


def _spec(status, kind, n, **kwargs):
    return dict(status=status, kind=kind, n=int(n), **kwargs)


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"),
                                (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    colors = rng.permutation([1, 2, 3, 4, 6, 7, 8, 9]).tolist()
    wall = int(colors[0])
    palette = colors[1:1 + int(rng.integers(1, 6))]
    maximum = int(rng.integers(3, 7))
    global_rotation = int(rng.integers(4))
    plans = [
        [_spec("active", "H", 2, lock_axis=True),
         _spec("active", "P", 3),
         _spec("nonmax", "V", 3),
         _spec("unmarked", "H", int(rng.integers(2, 5))),
         _spec("oversize", "H", 2)],
        [_spec("active", "V", maximum, full=False, branch=True, lock_axis=True),
         _spec("active", "P", 2),
         _spec("ambiguous", "H", int(rng.integers(2, 4))),
         _spec("unmarked", "H", int(rng.integers(2, 5)), branch=True)]
    ]
    for _ in range(num_train - 2):
        kinds = list(rng.choice(["H", "V", "P"], size=int(rng.integers(1, 4))))
        plan = [_spec("active", str(k), int(rng.choice([2, 3] if k == "P"
                                                       else [2, 3, maximum])), branch=True)
                for k in kinds]
        plan += [_spec("nonmax", str(rng.choice(["H", "V"])), 2),
                 _spec("unmarked", "H", int(rng.integers(2, 5)))]
        if rng.random() < .6:
            plan.append(_spec("ambiguous", "H", 2))
        plans.append(plan)
    for _ in range(num_test):
        plan = [_spec("active", "H", int(rng.choice([2, 3, maximum])), branch=True),
                _spec("active", "V", int(rng.choice([2, 3, maximum])), branch=True),
                _spec("active", "P", int(rng.choice([2, 3]))),
                _spec("nonmax", str(rng.choice(["H", "V"])), int(rng.integers(2, 4))),
                _spec("ambiguous", "H", int(rng.integers(2, 4))),
                _spec("unmarked", "H", int(rng.integers(2, 5)), branch=True)]
        if rng.random() < .35:
            plan.append(_spec("oversize", str(rng.choice(["H", "V"])), 2))
        plans.append(plan)
    puzzle, used = {"train": [], "test": []}, set()
    # The two teaching scenes may appear in either order.
    if rng.random() < .5:
        plans[0], plans[1] = plans[1], plans[0]
    for index, plan in enumerate(plans):
        split = "train" if index < num_train else "test"
        for _ in range(8):
            pair = _make_pair(rng, plan, wall, palette, connect=index != 0)
            pair = {k: np.rot90(v, global_rotation).copy() for k, v in pair.items()}
            key = _key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("Unable to make distinct inputs in bounded attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, walls = set(), set()
    trained, witnessed, trained_sizes = set(), set(), set()
    has_nonmax_counterexample = has_ambiguity = has_unmarked = has_containment = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            key = _key(a)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            analysis = _analyze(a)
            walls.add(analysis["wall"])
            assert np.count_nonzero(a == analysis["wall"]) > a.size // 2
            assert np.array_equal(_paint(a, analysis), b), "Incorrect symmetric copy"
            assert np.array_equal(a[a != BLANK], b[a != BLANK]), "Source or marker changed"
            assert np.any(a != b), "Generated examples must show a positive copy"
            assert np.count_nonzero(a != b) == sum(len(m["dest"]) for m in analysis["accepted"])
            assert all(_inside(m["dest"], m["square"]) for m in analysis["accepted"])
            if split == "train":
                trained.update(m["kind"] for m in analysis["accepted"])
                witnessed.update(m["kind"] for m in analysis["accepted"] if _orientation_witness(m))
                trained_sizes.update(m["square"][2] for m in analysis["accepted"])
                relaxed = _paint(a, _analyze(a, maximal=False))
                has_nonmax_counterexample |= bool(np.any((relaxed != a) & (b == a)))
                counts = Counter(m["si"] for m in analysis["matches"])
                has_ambiguity |= any(n > 1 for n in counts.values())
                has_unmarked |= any(
                    not any(_adjacent(marker, _kind(marker), square)
                            for marker in analysis["markers"])
                    for square in analysis["squares"])
                for m in analysis["accepted"]:
                    r, c, d, e = _bbox(m["dest"])
                    has_containment |= d - r + 1 < m["square"][2] or e - c + 1 < m["square"][2]
            else:
                assert {m["kind"] for m in analysis["accepted"]} <= trained
                assert {m["square"][2] for m in analysis["accepted"]} <= trained_sizes
    assert len(walls) == 1, "Wall role changes within the puzzle"
    assert trained == witnessed == {"H", "V", "P"}, "Training must expose all three orientations"
    assert has_nonmax_counterexample, "Training does not refute omitting A"
    assert has_ambiguity and has_unmarked and has_containment, "Missing discriminating demonstration"


def self_test():
    """Bounded sweep: keep only hashes and aggregate diversity, never grid logs."""
    _geometry_regressions()
    shapes, normalized, silhouettes, sizes, walls = set(), set(), set(), set(), set()
    signatures, statuses = set(), Counter()
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                total_pairs += 1
                shapes.add(a.shape)
                digest.update(bytes(a.shape) + a.tobytes() + b.tobytes())
                roles = {}
                normalized.add(hashlib.sha256(bytes(
                    roles.setdefault(int(x), len(roles)) for x in a.flat)).digest())
                info = _analyze(a)
                walls.add(info["wall"])
                counts = Counter(m["si"] for m in info["matches"])
                statuses["ambiguous"] += sum(v > 1 for v in counts.values())
                statuses["unmatched"] += len(info["squares"]) - len(counts)
                assert np.array_equal(solve(a.copy()), b)
                for m in info["accepted"]:
                    sizes.add(m["square"][2])
                    r, c, d, e = _bbox(m["source"])
                    local = np.zeros((d - r + 1, e - c + 1), dtype=bool)
                    for u, v in m["source"]:
                        local[u - r, v - c] = True
                    silhouettes.add((local.shape, local.tobytes()))
                    expected = (np.flipud(local) if m["kind"] == "H" else
                                np.fliplr(local) if m["kind"] == "V" else np.rot90(local, 2))
                    rr, cc, dd, ee = _bbox(m["dest"])
                    actual = np.zeros((dd - rr + 1, ee - cc + 1), dtype=bool)
                    for u, v in m["dest"]:
                        actual[u - rr, v - cc] = True
                    assert np.array_equal(actual, expected), "Coordinate mapping disagrees with array symmetry"
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 20
    assert len(normalized) >= 180 and len(silhouettes) >= 25
    assert sizes == {2, 3, 4, 5, 6} and len(walls) >= 6
    assert statuses["ambiguous"] >= 42 and statuses["unmatched"] >= 42
    a, b = generate_puzzle(987, 2, 1), generate_puzzle(987, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s])
               for k in ("input", "output"))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)]:
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Mutated labels exercise the two causal requirements, not just schemas.
    puzzle = generate_puzzle(0, 2, 1)
    for mode in ("omit_maximality", "omit_orientation"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        for pair in bad["train"]:
            a = pair["input"]
            mutant = (_paint(a, _analyze(a, maximal=False)) if mode == "omit_maximality"
                      else _paint(a, _analyze(a), translate=True))
            if not np.array_equal(mutant, pair["output"]):
                pair["output"] = mutant
                break
        else:
            raise AssertionError("Missing parent-ablation witness")
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Accepted a parent-ablation label")
    return {"seeds": 42, "pairs": total_pairs, "distinct_puzzles": len(signatures),
            "grid_shapes": len(shapes), "normalized_inputs": len(normalized),
            "payload_silhouettes": len(silhouettes), "square_sides": sorted(sizes),
            "wall_colors": sorted(walls), "negative_windows": dict(statuses)}


def _geometry_regressions():
    # An exhaustive small-grid predicate checks the integral-window implementation.
    rng = np.random.default_rng(250023)
    for _ in range(16):
        a = np.where(rng.random((6, 7)) < .65, 0, 8).astype(np.int8)
        brute = set()
        for r in range(6):
            for c in range(7):
                for n in range(2, min(6 - r, 7 - c) + 1):
                    cells = {(u, v) for u in range(r, r + n) for v in range(c, c + n)}
                    if any(a[q] != BLANK for q in cells):
                        continue
                    strips = [[(r - 1, v) for v in range(c, c + n)],
                              [(r + n, v) for v in range(c, c + n)],
                              [(u, c - 1) for u in range(r, r + n)],
                              [(u, c + n) for u in range(r, r + n)]]
                    if not any(all(0 <= u < 6 and 0 <= v < 7 and a[u, v] == BLANK
                                   for u, v in strip) for strip in strips):
                        brute.add((r, c, n))
        assert set(_squares(a)) == brute
    # A source overhangs a finite line endpoint; a partial blank branch is harmless.
    a = np.full((9, 12), 8, dtype=np.int8)
    a[4:7, 2:5] = BLANK
    a[6, 5] = BLANK
    a[3, 2:4] = MARKER
    a[2, 2:5] = 2
    a[1, 2] = 2
    expected = a.copy()
    expected[4, 2:5] = 2
    expected[5, 2] = 2
    assert np.array_equal(solve(a), expected)
    # A complete exterior strip cancels admission; ignoring A paints a new copy.
    a[4:7, 5] = BLANK
    assert np.array_equal(solve(a), a)
    assert np.any(_paint(a, _analyze(a, maximal=False)) != a)
    # A maximal square can meet the canvas edge.
    a = np.full((7, 7), 8, dtype=np.int8)
    a[:2, :2] = BLANK
    a[2, :2] = MARKER
    a[3, :2] = 2
    a[4, 0] = 2
    expected = a.copy()
    expected[1, :2] = 2
    expected[0, 0] = 2
    assert np.array_equal(solve(a), expected)
    # A singleton may share the payload's bounding column.
    a = np.full((10, 10), 8, dtype=np.int8)
    a[4:7, 4:7] = BLANK
    a[3, 4] = MARKER
    for q in ((2, 4), (2, 3), (1, 4)):
        a[q] = 2
    expected = a.copy()
    for q in ((4, 4), (4, 5), (5, 4)):
        expected[q] = 2
    assert np.array_equal(solve(a), expected)
    # Overlapping maximal 3x3/2x2 windows share a possible singleton copy.
    # Neither gets priority; one marker/source fitting both is ambiguous.
    a = np.full((11, 12), 8, dtype=np.int8)
    a[4:7, 4:7] = BLANK
    a[6:8, 3:5] = BLANK
    a[7, 5] = MARKER
    a[8, 6] = 2
    info = _analyze(a)
    assert len(info["matches"]) == 2 and not info["accepted"]
    assert np.array_equal(solve(a), a)
    a[7, 3] = 8  # Remove the smaller square: the same point copy now has one home.
    expected = a.copy()
    expected[6, 4] = 2
    assert np.array_equal(solve(a), expected)
    # Different, individually unique claims can collide inside overlapping windows.
    a = np.full((11, 12), 8, dtype=np.int8)
    a[4:7, 4:7] = BLANK
    a[6:8, 3:5] = BLANK
    a[3, 4:6] = MARKER
    a[:3, 4] = 2
    a[6:8, 2] = MARKER
    a[6, :2] = 3
    info = _analyze(a)
    assert len(info["matches"]) == 2 and not info["accepted"]
    assert np.array_equal(solve(a), a)
    a[6, 0] = 8  # Shorten the second copy so the two painted sets are disjoint.
    expected = a.copy()
    expected[4:7, 4] = 2
    expected[6, 3] = 3
    assert np.array_equal(solve(a), expected)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
