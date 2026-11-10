"""Register overlapping sheet views, then inset-fill its largest neutral window.

Only standard-library modules and NumPy are used. The solver reads pixels only.
Zero is the carrier outside views; eight is the fixed output ink. All other
colors are exchangeable. Rectangular views include observed neutral material.
"""
import hashlib

import numpy as np


CARRIER = 0
INK = 8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _panels(grid):
    """Separate the solid material views by the fixed zero carrier."""
    _require(not np.any(grid == INK), "output ink must be absent from inputs")
    unseen = set(zip(*np.nonzero(grid != CARRIER)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
                    cells.append(p)
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        r1, c1 = max(r for r, c in cells)+1, max(c for r, c in cells)+1
        _require(len(cells) == (r1-r0)*(c1-c0), "each view must be a solid rectangular component")
        result.append(grid[r0:r1, c0:c1].copy())
    _require(2 <= len(result) <= 6, "expected two to six material views")
    return result


def _neutral(panels):
    """The neutral is the sole material color with a solid 3 by 3 occurrence."""
    colors = set()
    for part in panels:
        h, w = part.shape
        for r in range(h-2):
            for c in range(w-2):
                patch = part[r:r+3, c:c+3]
                if np.all(patch == patch[0, 0]):
                    colors.add(int(patch[0, 0]))
    _require(len(colors) == 1, "neutral material is not identifiable")
    return next(iter(colors))


def _signatures(part, neutral):
    """Same-orientation, 2 by 3 or 3 by 2 multicolor texture signatures.

    These are observed patches, not a glyph bank. Neutral runs cannot serve as
    registration evidence. Full intersection context must also agree later.
    """
    found = {}
    for h, w in ((2, 3), (3, 2)):
        for r in range(part.shape[0]-h+1):
            for c in range(part.shape[1]-w+1):
                patch = part[r:r+h, c:c+w]
                if not np.any(patch == neutral) and len(np.unique(patch)) >= 3:
                    found.setdefault((h, w, patch.tobytes()), []).append((r, c))
    return found


def _intersection(a, b, delta):
    """Slices for two observed rectangles when b has origin delta in a."""
    dr, dc = delta
    r0, c0 = max(0, dr), max(0, dc)
    r1, c1 = min(a.shape[0], dr+b.shape[0]), min(a.shape[1], dc+b.shape[1])
    if r0 >= r1 or c0 >= c1:
        return None
    return (slice(r0, r1), slice(c0, c1)), (slice(r0-dr, r1-dr), slice(c0-dc, c1-dc))


def _translations(a, b, sa, sb):
    candidates = set()
    for pattern in sa.keys() & sb.keys():
        for ra, ca in sa[pattern]:
            for rb, cb in sb[pattern]:
                candidates.add((ra-rb, ca-cb))
    return [d for d in sorted(candidates)
            if np.array_equal(a[_intersection(a, b, d)[0]], b[_intersection(a, b, d)[1]])]


def _assembled(panels, offsets):
    r0, c0 = min(p[0] for p in offsets), min(p[1] for p in offsets)
    r1 = max(p[0]+a.shape[0] for p, a in zip(offsets, panels))
    c1 = max(p[1]+a.shape[1] for p, a in zip(offsets, panels))
    _require(r1-r0 <= 30 and c1-c0 <= 30, "assembled sheet exceeds ARC bounds")
    field = np.full((r1-r0, c1-c0), -1, dtype=np.int8)
    coverage = np.zeros(field.shape, dtype=np.int8)
    normalized = []
    for (r, c), part in zip(offsets, panels):
        r, c = r-r0, c-c0
        normalized.append((r, c))
        region = field[r:r+part.shape[0], c:c+part.shape[1]]
        _require(not np.any((region != -1) & (region != part)), "conflicting observed material")
        region[:] = part
        coverage[r:r+part.shape[0], c:c+part.shape[1]] += 1
    _require(bool(np.all(coverage)), "observations do not cover the complete sheet")
    return field, normalized, coverage


def _register(panels, neutral):
    """Unique solution of the connected signature-translation constraints.

    All context-supported matching edges must be consistent. No overlap-size,
    area, compactness, or input-position score selects a placement.
    """
    signatures = [_signatures(p, neutral) for p in panels]
    n = len(panels)
    adjacency = [[] for _ in panels]
    for i in range(n):
        for j in range(i+1, n):
            shifts = _translations(panels[i], panels[j], signatures[i], signatures[j])
            if shifts:
                adjacency[i].append((j, shifts))
                adjacency[j].append((i, [(-r, -c) for r, c in shifts]))
    offsets, solutions = {0: (0, 0)}, []
    visits = 0

    def search():
        nonlocal visits
        visits += 1
        _require(visits <= 2000, "unresolved registration search")
        if len(solutions) > 1:
            return
        if len(offsets) == n:
            try:
                solutions.append(_assembled(panels, [offsets[i] for i in range(n)]))
            except ValueError:
                pass
            return
        options = []
        for j in range(n):
            if j in offsets:
                continue
            domain = None
            for i, shifts in adjacency[j]:
                if i in offsets:
                    proposed = {(offsets[i][0]-r, offsets[i][1]-c) for r, c in shifts}
                    domain = proposed if domain is None else domain & proposed
            if domain is not None:
                if not domain:
                    return
                options.append((len(domain), j, domain))
        if not options:
            return
        _, j, domain = min(options, key=lambda item: (item[0], item[1]))
        for p in sorted(domain):
            consistent = True
            for i, q in offsets.items():
                intersection = _intersection(panels[i], panels[j], (p[0]-q[0], p[1]-q[1]))
                if intersection is not None and not np.array_equal(
                        panels[i][intersection[0]], panels[j][intersection[1]]):
                    consistent = False
                    break
            if consistent:
                offsets[j] = p
                search()
                del offsets[j]

    search()
    _require(len(solutions) == 1, "registration is disconnected, inconsistent, or ambiguous")
    return solutions[0]


def _largest_rectangles(field, color):
    """All global-area winners, including skinny windows; exclusive bounds."""
    best, winners = 0, []
    h, w = field.shape
    for top in range(h):
        columns = np.ones(w, dtype=bool)
        for bottom in range(top+1, h+1):
            columns &= field[bottom-1] == color
            c = 0
            while c < w:
                if not columns[c]:
                    c += 1
                    continue
                left = c
                while c < w and columns[c]:
                    c += 1
                area = (bottom-top)*(c-left)
                if area > best:
                    best, winners = area, [(top, bottom, left, c)]
                elif area == best:
                    winners.append((top, bottom, left, c))
    return best, winners


def _window(field, neutral):
    area, winners = _largest_rectangles(field, neutral)
    _require(len(winners) == 1, "largest neutral window is tied or missing")
    t, b, l, r = winners[0]
    _require(b-t >= 3 and r-l >= 3, "window needs a nonempty strict interior")
    return (t, b, l, r), area


def _paint(field, rectangle):
    t, b, l, r = rectangle
    out = field.copy()
    out[t+1:b-1, l+1:r-1] = INK
    return out


def _local_first_error(parts, neutral, offsets, shape, rectangle):
    """Falsify painting each view before the full vacancy field is available.

    This diagnostic uses solved offsets only to compare masks. It does not
    register painted fragments or choose the correct answer. Ambiguous local
    windows do not count as a demonstrated counterexample to this shortcut.
    """
    local = np.zeros(shape, dtype=bool)
    for part, (dr, dc) in zip(parts, offsets):
        _, winners = _largest_rectangles(part, neutral)
        if len(winners) != 1:
            return 0
        t, b, l, r = winners[0]
        if b-t < 3 or r-l < 3:
            return 0
        local[dr+t+1:dr+b-1, dc+l+1:dc+r-1] = True
    expected = np.zeros(shape, dtype=bool)
    t, b, l, r = rectangle
    expected[t+1:b-1, l+1:r-1] = True
    return int(np.count_nonzero(local != expected))


def _decode(input_grid):
    inp = _grid(input_grid)
    parts = _panels(inp)
    neutral = _neutral(parts)
    field, offsets, coverage = _register(parts, neutral)
    rectangle, area = _window(field, neutral)
    return parts, neutral, field, offsets, coverage, rectangle, area


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the displayed input, with no puzzle, seed, or generated state."""
    _, _, field, _, _, rectangle, _ = _decode(input_grid)
    return _paint(field, rectangle)


def _contains(box, rect):
    a, b, c, d = box
    t, u, l, r = rect
    return a <= t and u <= b and c <= l and r <= d


def _boxes(shape, rectangle, count, rng):
    """Two strips, a T subdivision, or a staggered four-view subdivision."""
    h, w = shape
    t, b, l, r = rectangle
    band = int(rng.integers(2, min(3, r-l-4)+1))
    cut = int(rng.integers(l+2, r-band-1))
    left, right = (0, h, 0, cut+band), (0, h, cut, w)
    boxes = [left, right]
    if count >= 3:
        which = int(rng.integers(2))
        band_y = int(rng.integers(2, min(3, b-t-4)+1))
        cut_y = int(rng.integers(t+2, b-band_y-1))
        a, z, c, d = boxes[which]
        boxes[which:which+1] = [(a, cut_y+band_y, c, d), (cut_y, z, c, d)]
        if count == 4:
            other = 2 if which == 0 else 0
            band_y = int(rng.integers(2, min(3, b-t-4)+1))
            cut_y = int(rng.integers(t+2, b-band_y-1))
            a, z, c, d = boxes[other]
            boxes[other:other+1] = [(a, cut_y+band_y, c, d), (cut_y, z, c, d)]
    return boxes


def _disjoint(a, b):
    return a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2]


def _make_field(rng, neutral, textures, count, first):
    # A target's entire strict interior crosses the first overlap band.
    rh = int(rng.integers(4 if count == 2 else 6, 10))
    rw = int(rng.integers(6, 10))
    if first and rh == rw:
        rh = 4 if rw != 4 else 5
    h, w = rh+int(rng.integers(5, 8)), rw+int(rng.integers(5, 8))
    t, l = int(rng.integers(1, h-rh)), int(rng.integers(1, w-rw))
    target = (t, t+rh, l, l+rw)
    field = rng.choice(textures, size=(h, w)).astype(np.int8)
    density = float(rng.uniform(.16, .43))
    field[rng.random((h, w)) < density] = neutral
    field[t:t+rh, l:l+rw] = neutral

    # A smaller, genuinely separate blank window is present in every pair.
    candidates = []
    ch, cw = 3, int(rng.integers(3, 5))
    for a in range(h-ch+1):
        for c in range(w-cw+1):
            box = (a, a+ch, c, c+cw)
            if _disjoint(box, target):
                candidates.append(box)
    _require(candidates, "no competitor space")
    a, b, c, d = candidates[int(rng.integers(len(candidates)))]
    field[a:b, c:d] = neutral
    competitor = (a, b, c, d)

    # An incidental blank abuts a side: there is no all-colored halo cue.
    side = int(rng.integers(4))
    p = ((t-1, int(rng.integers(l, l+rw))),
         (t+rh, int(rng.integers(l, l+rw))),
         (int(rng.integers(t, t+rh)), l-1),
         (int(rng.integers(t, t+rh)), l+rw))[side]
    field[p] = neutral
    boxes = _boxes(field.shape, target, count, rng)

    # Natural-looking noise patches in each subdivision overlap supply exact
    # signatures. No fixed motif, special key color, or patch location is used.
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            x, y = boxes[i], boxes[j]
            overlap = (max(x[0], y[0]), min(x[1], y[1]), max(x[2], y[2]), min(x[3], y[3]))
            patches = []
            for ph, pw in ((2, 3), (3, 2)):
                for pr in range(overlap[0], overlap[1]-ph+1):
                    for pc in range(overlap[2], overlap[3]-pw+1):
                        box = (pr, pr+ph, pc, pc+pw)
                        if _disjoint(box, target) and _disjoint(box, competitor):
                            patches.append(box)
            if patches:
                a, b, c, d = patches[int(rng.integers(len(patches)))]
                patch = rng.choice(textures, size=(b-a, d-c)).astype(np.int8)
                patch.flat[:3] = rng.choice(textures, size=3, replace=False)
                field[a:b, c:d] = patch
    return field, boxes, target


def _orient(field, boxes, rng):
    # Transform the complete sheet and every observation together. Individual
    # panels are never independently rotated or reflected during registration.
    if rng.random() < .5:
        h = field.shape[0]
        field = field[::-1].copy()
        boxes = [(h-b, h-a, c, d) for a, b, c, d in boxes]
    if rng.random() < .5:
        w = field.shape[1]
        field = field[:, ::-1].copy()
        boxes = [(a, b, w-d, w-c) for a, b, c, d in boxes]
    if rng.random() < .5:
        field = field.T.copy()
        boxes = [(c, d, a, b) for a, b, c, d in boxes]
    return field, boxes


def _pack(parts, rng):
    """Bounded random rectangle packing, then remove unused carrier margins."""
    # The sampler supplies at most four panels, each at most 16 by 16.
    # A 30-square work canvas
    # costs 900 bytes; only one packing attempt is retained at any time.
    for _ in range(80):
        grid = np.zeros((30, 30), dtype=np.int8)
        order = list(rng.permutation(len(parts)))
        # Large views first, with random ties, keeps placement robust.
        order.sort(key=lambda i: -parts[i].size)
        valid = True
        for i in order:
            part = parts[i]
            h, w = part.shape
            positions = []
            for r in range(31-h):
                for c in range(31-w):
                    if not np.any(grid[max(0, r-1):min(30, r+h+1), max(0, c-1):min(30, c+w+1)]):
                        positions.append((r, c))
            if not positions:
                valid = False
                break
            # Prefer a random corner for some attempts; other attempts are
            # unrestricted. Placement does not encode solved adjacency.
            if rng.random() < .55:
                corner = int(rng.integers(4))
                scores = [((r if corner < 2 else 30-r-h) +
                           (c if corner % 2 == 0 else 30-c-w)) for r, c in positions]
                best = min(scores)
                positions = [p for p, s in zip(positions, scores) if s <= best+2]
            r, c = positions[int(rng.integers(len(positions)))]
            grid[r:r+h, c:c+w] = part
        if valid:
            rows, cols = np.nonzero(grid)
            t, b, l, r = min(rows), max(rows)+1, min(cols), max(cols)+1
            t = max(0, t-int(rng.integers(3)))
            b = min(30, b+int(rng.integers(3)))
            l = max(0, l-int(rng.integers(3)))
            r = min(30, r+int(rng.integers(3)))
            return grid[t:b, l:r].copy()
    raise ValueError("views do not fit the carrier")


def _sample_pair(rng, neutral, count, first, need_counterexample=False,
                 avoid_shape=None, avoid_target=None):
    colors = [c for c in range(1, 10) if c not in (neutral, INK)]
    textures = rng.choice(colors, size=int(rng.integers(3, 7)), replace=False)
    for _ in range(320 if need_counterexample else 120):
        try:
            field, boxes, target = _make_field(rng, neutral, textures, count, first)
            rectangle, _ = _window(field, neutral)
            if rectangle != target:
                continue
            field, boxes = _orient(field, boxes, rng)
            rectangle, _ = _window(field, neutral)
            t, b, l, r = rectangle
            if field.shape == avoid_shape or (b-t, r-l) == avoid_target:
                continue
            parts = [field[a:b, c:d].copy() for a, b, c, d in boxes]
            if _neutral(parts) != neutral:
                continue
            reconstructed, offsets, coverage = _register(parts, neutral)
            if not np.array_equal(reconstructed, field):
                continue
            _family_invariants(parts, neutral, field, offsets, coverage, rectangle)
            if need_counterexample and not _local_first_error(
                    parts, neutral, offsets, field.shape, rectangle):
                continue
            return {"input": _pack(parts, rng), "output": _paint(field, rectangle)}
        except ValueError:
            continue
    raise ValueError("bounded generation exhausted; no ambiguous fallback is used")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        _require(_integer(value) and low <= value <= high, name+" is out of range")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    neutral = int(rng.choice([c for c in range(1, 10) if c != INK]))
    result, seen = {"train": [], "test": []}, set()
    first_shape = first_target = None
    has_counterexample = False
    for split, number in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(number):
            count = 4 if split == "test" else (2 if i == 0 else 3 if i == 1 else int(rng.integers(2, 4)))
            for _ in range(8):
                second = split == "train" and i == 1
                pair = _sample_pair(
                    rng, neutral, count, split == "train" and i == 0,
                    need_counterexample=second and not has_counterexample,
                    avoid_shape=first_shape if second else None,
                    avoid_target=first_target if second else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    if split == "train" and i == 0:
                        parts, _, field, offsets, _, rect, _ = _decode(pair["input"])
                        first_shape = field.shape
                        t, b, l, r = rect
                        first_target = (b-t, r-l)
                        has_counterexample = bool(_local_first_error(
                            parts, neutral, offsets, field.shape, rect))
                    break
            else:
                raise ValueError("could not make distinct observations")
    return result


def _family_invariants(parts, neutral, field, offsets, coverage, rectangle):
    """Semantic checks beyond agreement between generation and solve."""
    t, b, l, r = rectangle
    interior = (t+1, b-1, l+1, r-1)
    _require(not np.any(field == CARRIER) and not np.any(field == INK), "invalid material color")
    _require(bool(np.all(field[t:b, l:r] == neutral)), "window is not solid")
    for (a, c), part in zip(offsets, parts):
        box = (a, a+part.shape[0], c, c+part.shape[1])
        _require(not _contains(box, interior), "one view already exposes the entire answer interior")
        exclusive = coverage[box[0]:box[1], box[2]:box[3]] == 1
        _require(np.any(exclusive & (part != neutral)), "a view contributes no exclusive texture")
    # The winner is inclusion-maximal on each side, not a handpicked subwindow.
    _require(t == 0 or not np.all(field[t-1, l:r] == neutral), "window extends upward")
    _require(b == field.shape[0] or not np.all(field[b, l:r] == neutral), "window extends downward")
    _require(l == 0 or not np.all(field[t:b, l-1] == neutral), "window extends left")
    _require(r == field.shape[1] or not np.all(field[t:b, r] == neutral), "window extends right")
    outside = field.copy()
    outside[t:b, l:r] = -1
    competitor_area, competitors = _largest_rectangles(outside, neutral)
    _require(any(y-x >= 3 and v-u >= 3 for x, y, u, v in competitors), "missing substantial competitor")
    _require(competitor_area < (b-t)*(r-l), "competitor is not smaller")
    ring = np.zeros(field.shape, dtype=bool)
    ring[max(0, t-1):min(field.shape[0], b+1), max(0, l-1):min(field.shape[1], r+1)] = True
    ring[t:b, l:r] = False
    _require(np.any((field == neutral) & ring), "artificial clean halo")
    return competitor_area


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid split sizes")
    seen, neutrals = set(), set()
    train_shapes, train_targets, train_counterexamples = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            parts, neutral, field, offsets, coverage, rectangle, _ = _decode(inp)
            neutrals.add(neutral)
            _family_invariants(parts, neutral, field, offsets, coverage, rectangle)
            t, b, l, r = rectangle
            if split == "train":
                train_shapes.add(field.shape)
                train_targets.add((b-t, r-l))
                train_counterexamples.append(_local_first_error(
                    parts, neutral, offsets, field.shape, rectangle))
            _require(out.shape == field.shape, "crop lost sheet material")
            expected_changes = np.zeros(field.shape, dtype=bool)
            expected_changes[t+1:b-1, l+1:r-1] = True
            _require(np.array_equal(out != field, expected_changes), "border, noise, or smaller window changed")
            _require(np.all(out[expected_changes] == INK), "wrong strict-interior ink")
            _require(np.array_equal(out[~expected_changes], field[~expected_changes]), "noise was not preserved")
    _require(len(neutrals) == 1, "neutral convention varies within the puzzle")
    _require(len(train_shapes) >= 2 and len(train_targets) >= 2,
             "demonstrations must vary sheet and target geometry")
    _require(any(train_counterexamples), "no demonstration refutes painting before registration")
    return True


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain metrics and digests, never all puzzle payloads."""
    metrics = {name: set() for name in ("inputs", "outputs", "targets", "neutral", "palettes", "patches", "origins",
                                       "parities", "positions", "view_counts", "competitor_areas")}
    digests = set()
    total = counterexamples = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                parts, neutral, field, offsets, coverage, rect, area = _decode(pair["input"])
                t, b, l, r = rect
                metrics["inputs"].add(pair["input"].shape)
                metrics["outputs"].add(field.shape)
                metrics["targets"].add((b-t, r-l))
                metrics["neutral"].add(neutral)
                metrics["palettes"].add(tuple(int(c) for c in np.unique(field)))
                metrics["origins"].add(tuple(sorted(offsets)))
                metrics["parities"].add(((b-t) % 2, (r-l) % 2))
                metrics["positions"].add((t < (field.shape[0]-(b-t))/2,
                                          l < (field.shape[1]-(r-l))/2))
                metrics["view_counts"].add(len(parts))
                metrics["competitor_areas"].add(_family_invariants(
                    parts, neutral, field, offsets, coverage, rect))
                if split == "train":
                    counterexamples += bool(_local_first_error(
                        parts, neutral, offsets, field.shape, rect))
                for part in parts:
                    metrics["patches"].update(_signatures(part, neutral))
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "input-only solve failed")
                _require(len(parts) == 4 if split == "test" else len(parts) in (2, 3), "missing structural split shift")
                total += 1
    _require(len(digests) == 42, "seed diversity failed")
    for name, minimum in (("inputs", 30), ("outputs", 25), ("targets", 20),
                          ("neutral", 6), ("palettes", 25), ("patches", 100), ("origins", 80),
                          ("parities", 4), ("positions", 4), ("view_counts", 3), ("competitor_areas", 4)):
        _require(len(metrics[name]) >= minimum, "insufficient "+name+" variation")
    _require(counterexamples >= 42, "causal-order counterexamples are missing")
    a, b = generate_puzzle(901, 2, 1), generate_puzzle(901, 2, 1)
    _require(_digest(a) == _digest(b), "seed is not deterministic")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(700+nt*10+nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts not honored")
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt an unchanged texture cell: the validator must detect lost noise.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in ("train", "test")}
    output = bad["train"][0]["output"]
    p = tuple(np.argwhere(output != INK)[0])
    output[p] = INK
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("lost noise accepted")
    # A tied pair of equally large neutral windows is explicitly invalid.
    tied = np.ones((5, 9), dtype=np.int8)
    tied[1:4, 0:3] = 2
    tied[1:4, 6:9] = 2
    try:
        _window(tied, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("area tie accepted")

    # Repacking, all-plane orientation changes, and material-color relabeling
    # must not leak input layout or a hidden glyph vocabulary into solving.
    pair = a["train"][0]
    for transform in (lambda x: x.T.copy(), lambda x: x[::-1, :].copy(),
                      lambda x: x[:, ::-1].copy()):
        _require(np.array_equal(solve(transform(pair["input"])),
                                transform(pair["output"])), "orientation dependence")
    material_map = np.array([0, 9, 7, 6, 5, 4, 3, 2, 8, 1], dtype=np.int8)
    _require(np.array_equal(solve(material_map[pair["input"]]),
                            material_map[pair["output"]]), "material-color dependence")
    parts = _panels(pair["input"])
    repacked = _pack(list(reversed(parts)), np.random.default_rng(902))
    _require(np.array_equal(solve(repacked), pair["output"]), "carrier-position dependence")

    # The complete rectangle may meet the sheet edge. Partial neutral side
    # contacts do not extend its rectangular bounds or get painted.
    edge = np.ones((7, 12), dtype=np.int8)
    edge[0:3, 0:5] = 2
    edge[4:7, 8:11] = 2
    edge[3, 2] = 2
    rectangle, area = _window(edge, 2)
    _require(rectangle == (0, 3, 0, 5) and area == 15, "edge/global-area selection failed")
    painted = _paint(edge, rectangle)
    expected = edge.copy()
    expected[1, 1:4] = INK
    _require(np.array_equal(painted, expected), "strict border or competitor changed")

    # Registration ambiguity is rejected, rather than settled by an overlap
    # score, carrier order, or the rectangle that would be convenient to paint.
    tile = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    periodic = [np.tile(tile, (2, 3)), np.tile(tile, (2, 2))]
    try:
        _register(periodic, 9)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous signature registration accepted")
    try:
        _register([tile, np.full((3, 3), 7, dtype=np.int8)], 9)
    except ValueError:
        pass
    else:
        raise AssertionError("disconnected signature graph accepted")
    return {"valid": True, "seeds": 42, "pairs": total,
            "training_local_first_counterexamples": counterexamples,
            "diversity": {k: len(v) for k, v in metrics.items()}}
