"""Infer a mirror-overlap painting relation from a local worked example.

Only Python's standard library and NumPy are used. No evidence files are read.
Background is 0; each card visibly identifies its marker color by its outline.
"""
from collections import Counter
import hashlib
from itertools import permutations

import numpy as np


_KINDS = ("H", "V", "P")
# Class colors use roles: 0=empty, 1=specimen ink, 2=shared extra dye.
_POLICIES = tuple(permutations((0, 1, 2), 2))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _mate(p, kind, origin):
    r, c = p
    y, x = origin
    return (2*y-r if kind != "V" else r,
            2*x-c if kind != "H" else c)


def _marker(mask):
    cells = list(map(tuple, np.argwhere(mask).tolist()))
    _require(bool(cells) and len(_components(mask)) == 1, "one connected marker required")
    rows, cols = zip(*cells)
    if len(cells) == 1:
        return "P", cells[0]
    if min(rows) == max(rows) and len(cells) == max(cols)-min(cols)+1:
        return "H", (rows[0], min(cols))
    if min(cols) == max(cols) and len(cells) == max(rows)-min(rows)+1:
        return "V", (min(rows), cols[0])
    raise ValueError("marker must be a straight segment or singleton")


def _echo(mask, kind, origin):
    out = np.zeros(mask.shape, dtype=bool)
    for p in map(tuple, np.argwhere(mask).tolist()):
        q = _mate(p, kind, origin)
        _require(0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1],
                 "occupied reflection leaves its card")
        out[q] = True
    return out


def _card_data(interior, frame):
    mark = interior == frame
    kind, origin = _marker(mark)
    payload = (interior != 0) & ~mark
    # Fixed marker cells cannot be payload or receive reflected payload.
    echo = _echo(payload, kind, origin)
    _require(not np.any(echo & mark), "payload reflects onto marker")
    _require(not np.any(echo[0]) and not np.any(echo[-1])
             and not np.any(echo[:, 0]) and not np.any(echo[:, -1]),
             "reflected payload must respect the neutral rim")
    coverage = payload.astype(np.int8) + echo.astype(np.int8)
    return dict(grid=interior, frame=frame, marker=mark, kind=kind,
                origin=origin, payload=payload, coverage=coverage)


def _contains(outer, inner):
    r, c, h, w, _ = outer
    y, x, hh, ww, _ = inner
    return r < y and c < x and y+hh < r+h and x+ww < c+w


def _frames(grid):
    frames = []
    for color in map(int, np.unique(grid)):
        if color == 0:
            continue
        for component in _components(grid == color):
            coords = np.asarray(component)
            r, c = map(int, coords.min(axis=0))
            y, x = map(int, coords.max(axis=0))
            h, w = y-r+1, x-c+1
            if h < 7 or w < 7 or len(component) != 2*(h+w)-4:
                continue
            if all(a in (r, y) or b in (c, x) for a, b in component):
                frames.append((r, c, h, w, color))
    # A specimen can itself contain an outline. Only outermost cards count.
    return [f for f in frames if not any(_contains(g, f) for g in frames)]


def _paint(data, colors):
    out = np.zeros(data["grid"].shape, dtype=np.int8)
    for count, color in enumerate(colors, 1):
        out[data["coverage"] == count] = color
    out[data["marker"]] = data["frame"]
    return out


def _read_scene(grid):
    frames = _frames(grid)
    _require(len(frames) == 3, "exactly three outer framed cards required")
    frequency = Counter(f[4] for f in frames)
    _require(sorted(frequency.values()) == [1, 2], "two reference outlines and one target outline required")
    reference_color = next(c for c, n in frequency.items() if n == 2)
    reference_boxes = sorted(f for f in frames if f[4] == reference_color)
    target_box = next(f for f in frames if f[4] != reference_color)
    # Cards have a one-cell exterior moat, including diagonally.
    for i, f in enumerate(frames):
        for g in frames[i+1:]:
            r, c, h, w, _ = f
            y, x, hh, ww, _ = g
            _require(r+h < y or y+hh < r or c+w < x or x+ww < c,
                     "cards overlap or lack a separating background cell")
    occupied = np.zeros(grid.shape, dtype=bool)
    cards = {}
    for f in frames:
        r, c, h, w, color = f
        occupied[r:r+h, c:c+w] = True
        interior = grid[r+1:r+h-1, c+1:c+w-1].copy()
        _require(not np.any(interior[0]) and not np.any(interior[-1])
                 and not np.any(interior[:, 0]) and not np.any(interior[:, -1]),
                 "card interior needs a one-cell neutral rim")
        cards[f] = _card_data(interior, color)
    _require(not np.any(grid[~occupied]), "exterior must be neutral")
    a, b = (cards[f] for f in reference_boxes)
    _require(a["grid"].shape == b["grid"].shape
             and np.array_equal(a["marker"], b["marker"]), "reference geometry must correspond")
    asymmetric = [d for d in (a, b) if not np.array_equal(
        d["payload"], _echo(d["payload"], d["kind"], d["origin"]))]
    _require(len(asymmetric) == 1, "reference must distinguish asymmetric before from symmetric after")
    source = asymmetric[0]
    result = b if source is a else a
    source_box = reference_boxes[0] if source is a else reference_boxes[1]
    result_box = reference_boxes[1] if source is a else reference_boxes[0]
    target = cards[target_box]
    for d in (source, target):
        inks = np.unique(d["grid"][d["payload"]])
        _require(len(inks) == 1, "unprocessed specimen must be monochrome")
        d["ink"] = int(inks[0])
        _require(all(np.any(d["coverage"] == n) for n in (1, 2)),
                 "both single and double coverage must be demonstrated")
    _require(not np.array_equal(target["payload"], _echo(target["payload"], target["kind"], target["origin"])),
             "target must contain a nontrivial one-sided orbit")
    classes = []
    for n in (1, 2):
        values = np.unique(result["grid"][source["coverage"] == n])
        _require(len(values) == 1, "reference does not give a uniform color per coverage class")
        classes.append(int(values[0]))
    _require(classes[0] != classes[1], "reference must distinguish coverage classes")
    extras = set(classes) - {0, source["ink"]}
    _require(len(extras) <= 1, "at most one shared extra dye")
    _require(not (extras & {source["frame"], target["frame"], target["ink"]}),
             "extra dye must be distinct from structural and ink roles")
    _require(source["ink"] not in frequency and target["ink"] not in frequency,
             "ink cannot be a frame color")
    _require(np.array_equal(_paint(source, classes), result["grid"]),
             "reference changes markers or paints outside the echo union")
    target_colors = [target["ink"] if c == source["ink"] else c for c in classes]
    policy = tuple(0 if c == 0 else 1 if c == source["ink"] else 2 for c in classes)
    return dict(source=source, result=result, target=target, frames=frames,
                source_box=source_box, result_box=result_box, target_box=target_box,
                colors=classes, target_colors=target_colors, policy=policy)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all conventions and the two class colors from this input alone."""
    scene = _read_scene(_array(input_grid))
    return _paint(scene["target"], scene["target_colors"])


def _sample_marker(rng, shape, kind, off_center):
    h, w = shape
    rr, cc = (h-1)//2, (w-1)//2
    if off_center:
        # The ink sampler checks that enough valid orbits remain.
        if kind != "V":
            rr = int(np.clip(rr + int(rng.choice([-1, 1])), 2, h-3))
        if kind != "H":
            cc = int(np.clip(cc + int(rng.choice([-1, 1])), 2, w-3))
    mark = np.zeros(shape, dtype=bool)
    if kind == "P":
        mark[rr, cc] = True
    elif kind == "H":
        length = int(rng.integers(2, min(6, w-2)+1))
        start = int(rng.integers(1, w-length))
        mark[rr, start:start+length] = True
    else:
        length = int(rng.integers(2, min(6, h-2)+1))
        start = int(rng.integers(1, h-length))
        mark[start:start+length, cc] = True
    return mark


def _orbits(shape, mark):
    kind, origin = _marker(mark)
    h, w = shape
    result = []
    for r in range(1, h-1):
        for c in range(1, w-1):
            p = (r, c)
            q = _mate(p, kind, origin)
            if (1 <= q[0] < h-1 and 1 <= q[1] < w-1
                    and not mark[p] and not mark[q] and p <= q):
                result.append((p,) if p == q else (p, q))
    return result


def _motif(rng, shape, family):
    """Geometric starting masks, subsequently supplied with orbit witnesses."""
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if family == 0:
        mask = rng.random(shape) < rng.uniform(.20, .65)
    elif family == 1:
        r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        for _ in range(int(rng.integers(h+w, 3*(h+w)))):
            mask[r, c] = True
            dr, dc = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1))[int(rng.integers(6))]
            r, c = int(np.clip(r+dr, 1, h-2)), int(np.clip(c+dc, 1, w-2))
    elif family == 2:
        for _ in range(int(rng.integers(1, 4))):
            r0, r1 = sorted(map(int, rng.choice(np.arange(1, h), 2, replace=False)))
            c0, c1 = sorted(map(int, rng.choice(np.arange(1, w), 2, replace=False)))
            mask[r0:r1, c0:c1] = True
    elif family == 3:
        rr, cc = np.indices(shape)
        period = int(rng.integers(2, 5))
        mask = (rr + int(rng.choice([-1, 1]))*cc + int(rng.integers(period))) % period == 0
    elif family == 4:
        r0, r1 = 1, h-2
        c0, c1 = 1, w-2
        if h >= 9 and rng.integers(2):
            r0 += 1
        if w >= 9 and rng.integers(2):
            c1 -= 1
        mask[r0, c0:c1+1] = mask[r1, c0:c1+1] = True
        mask[r0:r1+1, c0] = mask[r0:r1+1, c1] = True
    else:
        rr, cc = np.indices(shape)
        y, x = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        radius = int(rng.integers(2, max(h, w)//2+1))
        mask = abs(rr-y)+abs(cc-x) <= radius
    mask[0] = mask[-1] = False
    mask[:, 0] = mask[:, -1] = False
    return mask


def _specimen(rng, shape, kind, frame, ink, family, off_center):
    mark = None
    for _ in range(12):
        candidate = _sample_marker(rng, shape, kind, off_center)
        orbits = _orbits(shape, candidate)
        if sum(len(o) == 2 for o in orbits) >= 5:
            mark = candidate
            break
    if mark is None:
        mark = _sample_marker(rng, shape, kind, False)
        orbits = _orbits(shape, mark)
    valid = np.zeros(shape, dtype=bool)
    for orbit in orbits:
        for p in orbit:
            valid[p] = True
    mask = _motif(rng, shape, family) & valid
    doubles = [o for o in orbits if len(o) == 2]
    rng.shuffle(doubles)
    # Two visibly different lone mates, two agreements, and a genuinely empty
    # orbit make all three multiplicities observable without relying on chance.
    for i, orbit in enumerate(doubles[:5]):
        for p in orbit:
            mask[p] = False
        if i < 2:
            mask[orbit[int(rng.integers(2))]] = True
        elif i < 4:
            for p in orbit:
                mask[p] = True
    grid = np.zeros(shape, dtype=np.int8)
    grid[mask] = ink
    grid[mark] = frame
    return grid


def _pack(rng, sizes):
    """Free placement of three rectangles; bounded constructive fallback."""
    order = list(rng.permutation(3))
    boxes = None
    for _ in range(24):
        placed = {}
        for i in order:
            h, w = sizes[i]
            for _ in range(64):
                r, c = int(rng.integers(31-h)), int(rng.integers(31-w))
                if all(r+h < y or y+hh < r or c+w < x or x+ww < c
                       for y, x, hh, ww in placed.values()):
                    placed[i] = (r, c, h, w)
                    break
            else:
                break
        if len(placed) == 3:
            boxes = [placed[i] for i in range(3)]
            break
    if boxes is None:
        # Every generated outer dimension is <=14, so this always fits 29x29.
        cells = [(0, 0), (0, 15), (15, 0), (15, 15)]
        rng.shuffle(cells)
        boxes = [(r, c, *size) for (r, c), size in zip(cells, sizes)]
    top = min(b[0] for b in boxes)
    left = min(b[1] for b in boxes)
    bottom = max(b[0]+b[2] for b in boxes)
    right = max(b[1]+b[3] for b in boxes)
    hh, ww = bottom-top, right-left
    extra_h = int(rng.integers(31-hh))
    extra_w = int(rng.integers(31-ww))
    before_h = int(rng.integers(extra_h+1))
    before_w = int(rng.integers(extra_w+1))
    return (hh+extra_h, ww+extra_w), [(r-top+before_h, c-left+before_w, h, w)
                                                 for r, c, h, w in boxes]


def _put_card(canvas, box, interior, frame):
    r, c, h, w = box
    canvas[r, c:c+w] = canvas[r+h-1, c:c+w] = frame
    canvas[r:r+h, c] = canvas[r:r+h, c+w-1] = frame
    canvas[r+1:r+h-1, c+1:c+w-1] = interior


def _make_pair(rng, ref_kind, target_kind, policy, test, index):
    colors = list(map(int, rng.permutation(np.arange(1, 10))))
    ref_frame, target_frame, ref_ink, target_ink, dye = colors[:5]
    if rng.random() < .3 and (test or index != 0):
        target_ink = ref_ink
    upper = 13 if test else 11
    ref_shape = tuple(map(int, rng.integers(7, upper, size=2)))
    target_shape = tuple(map(int, rng.integers(7, upper, size=2)))
    if index < 2 and target_shape == ref_shape:
        target_shape = (7 if ref_shape[0] != 7 else 8, ref_shape[1])
    a = _specimen(rng, ref_shape, ref_kind, ref_frame, ref_ink,
                  int(rng.integers(6)), off_center=(index % 2 == 0 or test))
    t = _specimen(rng, target_shape, target_kind, target_frame, target_ink,
                  int(rng.integers(6)), off_center=(index % 2 == 1 or test))
    ref_colors = [(0, ref_ink, dye)[n] for n in policy]
    target_colors = [(0, target_ink, dye)[n] for n in policy]
    b = _paint(_card_data(a, ref_frame), ref_colors)
    out = _paint(_card_data(t, target_frame), target_colors)
    interiors = [a, b, t]
    frames = [ref_frame, ref_frame, target_frame]
    sizes = [(h+2, w+2) for h, w in (ref_shape, ref_shape, target_shape)]
    shape, boxes = _pack(rng, sizes)
    grid = np.zeros(shape, dtype=np.int8)
    for box, content, frame in zip(boxes, interiors, frames):
        _put_card(grid, box, content, frame)
    # A half-turn changes physical card order and occupied sides without
    # changing the H/V identities used by the puzzle's held-out pairing.
    turns = 2*int(rng.integers(2))
    return {"input": np.rot90(grid, turns).copy(), "output": np.rot90(out, turns).copy()}


def _integer(value, low, high, name):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer)) and low <= value <= high,
             f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    a = _integer(num_train, 2, 10, "num_train")
    b = _integer(num_test, 1, 4, "num_test")
    _require(a > b, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Independent half-turns preserve the intended held-out directed pairing
    # of actual marker kinds while varying sides and physical card order.
    kinds = [str(x) for x in rng.permutation(_KINDS)]
    train_pairs = [(kinds[0], kinds[1]), (kinds[1], kinds[2]),
                   (kinds[0], kinds[2]), (kinds[1], kinds[0])]
    # The first class-separation example exposes both ink-role remapping and
    # retention of the shared extra dye. A different second policy establishes
    # that the complete reference, rather than a global fixed policy, matters.
    first = int(rng.choice([i for i, p in enumerate(_POLICIES) if set(p) == {1, 2}]))
    second = int(rng.choice([i for i in range(len(_POLICIES)) if i != first]))
    first_policies = [first, second]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            if split == "train":
                pair_kinds = train_pairs[i] if i < 2 else train_pairs[int(rng.integers(4))]
                policy = _POLICIES[first_policies[i]] if i < 2 else _POLICIES[int(rng.integers(6))]
            else:
                pair_kinds = (kinds[2], kinds[int(rng.integers(2))])
                policy = _POLICIES[int(rng.integers(6))]
            for _ in range(16):
                pair = _make_pair(rng, *pair_kinds, policy, split == "test", i)
                key = _fingerprint(pair["input"])
                # The rule is read from actual pixels before accepting a pair.
                try:
                    answer = solve(pair["input"])
                except ValueError:
                    continue
                if key not in seen and np.array_equal(answer, pair["output"]):
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded readable-card construction exhausted")
    return puzzle


def _fingerprint(grid):
    return grid.shape, hashlib.sha256(grid.astype(np.uint8).tobytes()).digest()


def _oracle(data, colors):
    """Coordinate-wise check, independently of the vector coverage painter."""
    grid = data["grid"]
    out = np.zeros_like(grid)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if data["marker"][r, c]:
                out[r, c] = data["frame"]
                continue
            y, x = _mate((r, c), data["kind"], data["origin"])
            hit = int(data["payload"][r, c])
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                hit += int(data["payload"][y, x])
            if hit:
                out[r, c] = colors[hit-1]
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _integer(len(puzzle["train"]), 2, 10, "training length")
    _integer(len(puzzle["test"]), 1, 4, "test length")
    _require(len(puzzle["train"]) > len(puzzle["test"]), "training must be longer")
    seen, train_policies, train_kinds = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _array(pair["input"]), _array(pair["output"])
            key = _fingerprint(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            scene = _read_scene(inp)
            target = scene["target"]
            _require(np.array_equal(_oracle(scene["source"], scene["colors"]), scene["result"]["grid"]),
                     "reference multiplicity mismatch")
            _require(np.array_equal(_oracle(target, scene["target_colors"]), out), "incorrect target result")
            _require(out.shape == target["grid"].shape, "output must crop the target interior")
            # Neither bare copying nor B's ordinary additive completion suffices.
            union = _paint(target, (target["ink"], target["ink"]))
            _require(not np.array_equal(union, out), "reference can be bypassed by additive completion")
            _require(not np.array_equal(target["grid"], out), "identity hides the transformation")
            _require(np.array_equal(out[target["marker"]], target["grid"][target["marker"]]), "marker changed")
            for n in (1, 2):
                _require(np.count_nonzero(target["coverage"] == n) >= 2, "insufficient class witnesses")
            kinds = {scene["source"]["kind"], target["kind"]}
            if split == "train":
                train_policies.add(scene["policy"])
                train_kinds.update(kinds)
            else:
                _require(kinds <= train_kinds, "test marker geometry lacks a training demonstration")
    _require(len(train_policies) >= 2, "training must demonstrate reference-dependent policies")
    _require(train_kinds == set(_KINDS), "training must expose all three marker kinds")


def _must_fail(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("invalid case was accepted")


def _copy_puzzle(puzzle):
    return {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}


def _diagnostics():
    p = generate_puzzle(731, 2, 1)
    original = p["train"][0]["input"]
    scene = _read_scene(original)
    # Exact A-dependence twins: all cells except the demonstrated AFTER stay fixed.
    outputs = []
    dye = next(c for c in range(1, 10) if c not in set(map(int, np.unique(original))))
    for policy in ((1, 0), (0, 1), (1, 2)):
        twin = original.copy()
        colors = [(0, scene["source"]["ink"], dye)[n] for n in policy]
        r, c, h, w, _ = scene["result_box"]
        twin[r+1:r+h-1, c+1:c+w-1] = _paint(scene["source"], colors)
        parsed = _read_scene(twin)
        assert np.array_equal(parsed["target"]["grid"], scene["target"]["grid"])
        outputs.append(_fingerprint(solve(twin)))
    assert len(set(outputs)) == 3
    # Exact B-dependence twins: same ink, different marker geometry. Check against
    # explicit point/line formulas, including an asymmetric L and paired evidence.
    payload = [(1, 1), (1, 2), (2, 1), (2, 6), (6, 2), (6, 6)]
    results = []
    for kind, markers in (("H", [(4, 4), (4, 5)]), ("V", [(4, 4), (5, 4)]), ("P", [(4, 4)])):
        grid = np.zeros((9, 9), dtype=np.int8)
        for q in payload:
            grid[q] = 1
        for q in markers:
            grid[q] = 5
        data = _card_data(grid, 5)
        assert data["kind"] == kind
        answer = _paint(data, (1, 2))
        assert np.array_equal(answer, _oracle(data, (1, 2)))
        # Independent array transforms check the geometric formulas themselves.
        occupied = grid == 1
        mirror = (np.flipud(occupied) if kind == "H" else
                  np.fliplr(occupied) if kind == "V" else np.rot90(occupied, 2))
        counts = occupied.astype(np.int8) + mirror.astype(np.int8)
        expected = counts.copy()
        expected[grid == 5] = 5
        assert np.array_equal(answer, expected)
        answer[answer == 5] = 0
        results.append(_fingerprint(answer))
    assert len(set(results)) == 3
    # A fixed point outside a finite line segment agrees with its own echo.
    grid = np.zeros((9, 9), dtype=np.int8)
    grid[4, 4:6] = 5
    grid[4, 1] = 1
    data = _card_data(grid, 5)
    assert data["coverage"][4, 1] == 2
    # Physical source/result order carries no meaning.
    swapped = original.copy()
    for box, interior in ((scene["source_box"], scene["result"]["grid"]),
                          (scene["result_box"], scene["source"]["grid"])):
        r, c, h, w, _ = box
        swapped[r+1:r+h-1, c+1:c+w-1] = interior
    assert np.array_equal(solve(swapped), solve(original))
    # Uniform class labels are required, rather than guessed from a sample cell.
    damaged = original.copy()
    r, c, _, _, _ = scene["result_box"]
    y, x = np.argwhere(scene["source"]["coverage"] == 1)[0]
    damaged[r+1+y, c+1+x] = (int(damaged[r+1+y, c+1+x])+1) % 10
    _must_fail(lambda: solve(damaged))
    bad = _copy_puzzle(p)
    bad["test"][0]["output"][0, 0] = 9
    _must_fail(lambda: validate_generated_puzzle(bad))
    return {"reference_twins": 3, "marker_twins": 3, "reference_order_invariant": True,
            "fixed_point_agreement": True, "damaged_cases_rejected": 2}


def self_test():
    """Bounded streaming checks: 42 seeds, no disk writes or retained corpus."""
    shapes, outputs, policies, structures, masks, marker_pairs = set(), set(), set(), set(), set(), set()
    input_colors, ink_counts, orbit_counts = set(), set(), set()
    off_center = overhang = fixed_hits = same_ink = changed_ink = 0
    total_pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        training_combinations = set()
        for split in ("train", "test"):
            for pair in p[split]:
                total_pairs += 1
                scene = _read_scene(pair["input"])
                target = scene["target"]
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                policies.add(scene["policy"])
                input_colors.update(map(int, np.unique(pair["input"])))
                marker_pairs.add((scene["source"]["kind"], target["kind"]))
                combination = (scene["source"]["kind"], target["kind"])
                if split == "train":
                    training_combinations.add(combination)
                else:
                    assert combination not in training_combinations
                mask = target["payload"]
                masks.add(_fingerprint(mask))
                ink_counts.add(int(mask.sum()))
                orbit_counts.add(tuple(int(np.count_nonzero(target["coverage"] == n)) for n in (1, 2)))
                roles = {0: 0}
                normalized = np.array([roles.setdefault(int(c), len(roles)) for c in pair["input"].flat],
                                      dtype=np.int8).reshape(pair["input"].shape)
                structures.add(_fingerprint(normalized))
                same_ink += scene["source"]["ink"] == target["ink"]
                changed_ink += scene["source"]["ink"] != target["ink"]
                for d in (scene["source"], target):
                    h, w = d["grid"].shape
                    r, c = d["origin"]
                    if d["kind"] == "H":
                        off_center += 2*r != h-1
                        axis = 1
                    elif d["kind"] == "V":
                        off_center += 2*c != w-1
                        axis = 0
                    else:
                        off_center += (2*r, 2*c) != (h-1, w-1)
                        continue
                    m = np.argwhere(d["marker"])[:, axis]
                    x = np.argwhere(d["payload"])[:, axis]
                    overhang += x.min() < m.min() or x.max() > m.max()
                    fixed_hits += sum(_mate(tuple(q), d["kind"], d["origin"]) == tuple(q)
                                      for q in np.argwhere(d["payload"]).tolist())
        # Covariance to rotation and role-color permutations is checked on one
        # entire input per seed, with the output transformed independently.
        pair = p["test"][0]
        assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
        palette = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int8)
        assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
    assert len(shapes) >= 20 and len(outputs) >= 20
    assert len(policies) == 6 and len(marker_pairs) >= 6
    assert len(structures) >= 200 and len(masks) >= 180
    assert len(ink_counts) >= 20 and len(orbit_counts) >= 80, (len(ink_counts), len(orbit_counts))
    assert input_colors == set(range(10))
    assert min(off_center, overhang, fixed_hits, same_ink, changed_ink) > 0
    a, b = generate_puzzle(991, 2, 1), generate_puzzle(991, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    legal_counts = 0
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(10000+10*n+m, n, m)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (n, m)
            legal_counts += 1
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4., 1), (4, 1.), (None, 1), (4, None)):
        _must_fail(lambda n=n, m=m: generate_puzzle(0, n, m))
    for seed in (-1, True, 1.5, None):
        _must_fail(lambda seed=seed: generate_puzzle(seed))
    return dict(seeds=42, pairs=total_pairs, input_shapes=len(shapes), output_shapes=len(outputs),
                policies=len(policies), marker_pairs=len(marker_pairs), structures=len(structures),
                target_masks=len(masks), ink_counts=len(ink_counts), orbit_count_pairs=len(orbit_counts),
                off_center_cards=int(off_center), overhanging_cards=int(overhang),
                fixed_point_ink_cells=int(fixed_hits), same_ink_pairs=int(same_ink),
                remapped_ink_pairs=int(changed_ink), legal_counts=legal_counts, diagnostics=_diagnostics())
