"""Corner receivers: square geometry routes eight exterior taps to four OR bits.

Only NumPy and the standard library are used. There is no learned palette, seed
lookup, or generator state in solve(). See task_spec.md for the full contract.
"""

from collections import Counter
import numpy as np


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, size 1..30, colors 0..9")
    return a


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    i = int(np.argmax(counts))
    if int(counts[i]) * 2 <= a.size:
        raise ValueError("the background must occupy strictly more than half the input")
    return int(colors[i])


def _components(a, background):
    """Maximal FOUR-connected MONOCHROME components, irrespective of role."""
    seen = np.zeros(a.shape, dtype=bool)
    answer = []
    for rr, cc in np.argwhere(a != background):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        color = int(a[r, c])
        seen[r, c] = True
        stack, cells = [(r, c)], set()
        while stack:
            p = stack.pop()
            cells.add(p)
            for dr, dc in NEIGHBORS:
                nr, nc = p[0] + dr, p[1] + dc
                if (0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]
                        and not seen[nr, nc] and int(a[nr, nc]) == color):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        answer.append(_record(cells, color))
    return answer


def _record(cells, color):
    top = min(r for r, c in cells)
    bottom = max(r for r, c in cells)
    left = min(c for r, c in cells)
    right = max(c for r, c in cells)
    h, w = bottom - top + 1, right - left + 1
    full = len(cells) == h * w
    frame = (h >= 2 and w >= 2 and len(cells) == 2 * h + 2 * w - 4
             and all(r in (top, bottom) or c in (left, right) for r, c in cells))
    kind = ("solid" if full else "frame") if h == w and h >= 2 and (full or frame) else None
    return {"cells": cells, "color": int(color), "box": (top, bottom, left, right),
            "kind": kind, "height": h, "width": w}


def _wiring(record):
    """Clockwise corner order; each pair is (horizontal tap, vertical tap)."""
    t, b, l, r = record["box"]
    return (((t, l), ((t, l - 1), (t - 1, l))),
            ((t, r), ((t, r + 1), (t - 1, r))),
            ((b, r), ((b, r + 1), (b + 1, r))),
            ((b, l), ((b, l - 1), (b + 1, l))))


def _occupied(a, background, p):
    r, c = p
    return 0 <= r < a.shape[0] and 0 <= c < a.shape[1] and int(a[r, c]) != background


def _receiver_output(a, background, targets, operation="or"):
    out = np.full(a.shape, background, dtype=np.int8)
    for q in targets:
        for corner, (hp, vp) in _wiring(q):
            h, v = _occupied(a, background, hp), _occupied(a, background, vp)
            active = {"or": h or v, "and": h and v, "xor": h != v,
                      "horizontal": h, "vertical": v, "all": True}[operation]
            if active:
                out[corner] = q["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the colored corner constellation, using only this input."""
    a = _array(input_grid)
    bg = _background(a)
    targets = [q for q in _components(a, bg) if q["kind"] is not None]
    return _receiver_output(a, bg, targets)


def _square(side, kind):
    return {(r, c) for r in range(side) for c in range(side)
            if kind == "solid" or r in (0, side - 1) or c in (0, side - 1)}


def _transform(cells, turns, reflect=False):
    cells = {(r, -c if reflect else c) for r, c in cells}
    for _ in range(turns % 4):
        cells = {(c, -r) for r, c in cells}
    return cells


def _normalize_group(group, turns, reflect):
    group = [(_transform(cells, turns, reflect), color) for cells, color in group]
    top = min(r for cells, _ in group for r, c in cells)
    left = min(c for cells, _ in group for r, c in cells)
    return [({(r - top, c - left) for r, c in cells}, color) for cells, color in group]


def _place_group(rng, scene, group, avoid=()):
    """Keep drawing components separate; later signal strokes may touch them."""
    group = _normalize_group(group, int(rng.integers(4)), bool(rng.integers(2)))
    h, w = scene["shape"]
    gh = max(r for cells, _ in group for r, c in cells) + 1
    gw = max(c for cells, _ in group for r, c in cells) + 1
    if gh + 2 > h or gw + 2 > w:
        return None
    positions = [(r, c) for r in range(1, h - gh) for c in range(1, w - gw)]
    blocked = scene["body"] | set(avoid)
    halo = scene["body"] | {(r + dr, c + dc) for r, c in scene["body"] for dr, dc in NEIGHBORS}
    for raw in rng.permutation(len(positions)):
        t, l = positions[int(raw)]
        placed = [({(r + t, c + l) for r, c in cells}, color) for cells, color in group]
        all_cells = set().union(*(cells for cells, color in placed))
        if all_cells & (blocked | halo):
            continue
        scene["objects"].extend(placed)
        scene["body"].update(all_cells)
        return [_record(cells, color) for cells, color in placed]
    return None


def _target_group(rng, motif, body_colors, core_kind=None, test=False):
    def color():
        return int(rng.choice(body_colors))
    # Each shape is independently colored; target eligibility ignores its color.
    a = int(rng.integers(3, 7 if test else 6))
    kind = core_kind or str(rng.choice(("solid", "frame")))
    if motif == "single":
        if core_kind is None:
            a = int(rng.integers(2, 8 if test else 7))
        return [(_square(a, kind), color())]
    if motif == "shared":
        b = int(rng.integers(2, 6))
        return [(_square(a, kind), color()),
                ({(r + a, c + a) for r, c in _square(b, str(rng.choice(("solid", "frame"))))}, color())]
    if motif == "nested":
        outer = a + int(rng.integers(4, 9))
        off_r = int(rng.integers(2, outer - a - 1))
        off_c = int(rng.integers(2, outer - a - 1))
        # Core first, so demonstration truth-table bits can be set on it.
        return [({(r + off_r, c + off_c) for r, c in _square(a, kind)}, color()),
                (_square(outer, "frame"), color())]
    if motif == "nested_shared":
        a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        outer = a + b + int(rng.integers(4, 8))
        x = int(rng.integers(2, outer - a - b - 1))
        y = int(rng.integers(2, outer - a - b - 1))
        return [(_square(outer, "frame"), color()),
                ({(r + x, c + y) for r, c in _square(a, "solid")}, color()),
                ({(r + x + a, c + y + a) for r, c in _square(b, "frame")}, color())]
    raise ValueError(motif)


def _negative(rng, category):
    if category == "rectangle":
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 7))
        if h == w:
            w += 1
        hollow = bool(rng.integers(2))
        return {(r, c) for r in range(h) for c in range(w)
                if not hollow or r in (0, h - 1) or c in (0, w - 1)}
    if category == "broken":
        n = int(rng.integers(3, 7))
        cells = _square(n, "frame")
        cells.remove((0, int(rng.integers(1, n - 1))))
        return cells
    if category == "elbow":
        n = int(rng.integers(3, 7))
        return {(r, 0) for r in range(n)} | {(n - 1, c) for c in range(n)}
    if category == "bar":
        return {(0, c) for c in range(int(rng.integers(2, 8)))}
    return {(0, 0)}


def _signature(q):
    return (q["box"], q["kind"], q["color"])


def _build_pair(rng, palette, motif, core_kind, teaching, test):
    bg, body_colors, signal_colors = palette
    group = _target_group(rng, motif, body_colors, core_kind, test)
    extent = max(max(max(r, c) for r, c in cells) for cells, _ in group) + 1
    low = max(12 if teaching else 9, extent + 4)
    high = min(30, max(low + 3, 23 if teaching else 29))
    h, w = int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))
    scene = {"shape": (h, w), "body": set(), "objects": []}
    targets = _place_group(rng, scene, group)
    if targets is None:
        return None
    core = targets[0]
    core_ports = {p for _, ports in _wiring(core) for p in ports}
    t, b, l, r = core["box"]
    diagonals = {(t - 1, l - 1), (t - 1, r + 1), (b + 1, r + 1), (b + 1, l - 1)}
    avoid = core_ports | diagonals if teaching else set()
    extra = int(rng.integers(0, 2 if teaching else 4))
    for _ in range(extra):
        q = _place_group(rng, scene, _target_group(rng, "single", body_colors, test=test), avoid)
        if q:
            targets.extend(q)

    negatives = []
    categories = (["rectangle" if teaching == 1 else "broken"] if teaching else [])
    categories += [str(rng.choice(("rectangle", "broken", "elbow", "bar", "pixel")))
                   for _ in range(int(rng.integers(0, 3)))]
    for category in categories:
        negative = _place_group(rng, scene, [(_negative(rng, category), int(rng.choice(body_colors)))], avoid)
        if negative:
            negatives.extend(negative)
        elif teaching and not negatives:
            return None

    ports = {p for q in targets for _, taps in _wiring(q) for p in taps}
    on, off = set(), set()
    if teaching:
        # The four receiver positions demonstrate the entire OR truth table.
        states = list(rng.permutation(4))
        for state, (corner, (hp, vp)) in zip(states, _wiring(core)):
            (on if int(state) & 1 else off).add(hp)
            (on if int(state) & 2 else off).add(vp)
            if int(state) == 0:
                cr, cc = corner
                dh = hp[1] - cc
                dv = vp[0] - cr
                diagonal = (cr + dv, cc + dh)
                if diagonal not in scene["body"]:
                    on.add(diagonal)
    density = float(rng.uniform(.18, .77))
    on.update(p for p in sorted(ports) if p not in off and rng.random() < density)

    if motif in ("shared", "nested_shared"):
        uses = Counter(p for q in targets for _, taps in _wiring(q) for p in taps)
        shared_taps = sorted(p for p, count in uses.items() if count > 1 and p not in off)
        if not shared_taps:
            return None
        on.add(shared_taps[int(rng.integers(len(shared_taps)))])

    # A powered nonsquare deliberately defeats 'use every component's corners'.
    if teaching:
        candidates = [p for q in negatives[:1] if q["height"] >= 2 and q["width"] >= 2
                      for corner, taps in _wiring(q) for p in taps
                      if corner in q["cells"] and p not in off and p not in scene["body"]
                      and 0 <= p[0] < h and 0 <= p[1] < w]
        if not candidates:
            return None
        on.add(candidates[int(rng.integers(len(candidates)))])

    # Correlated source strokes, not just independently colored port dots.
    for _ in range(int(rng.integers(0, 5))):
        if on and rng.random() < .65:
            start = sorted(on)[int(rng.integers(len(on)))]
        else:
            start = (int(rng.integers(h)), int(rng.integers(w)))
        dr, dc = NEIGHBORS[int(rng.integers(4))]
        for k in range(int(rng.integers(1, 5))):
            p = (start[0] + k * dr, start[1] + k * dc)
            if 0 <= p[0] < h and 0 <= p[1] < w and p not in off and p not in scene["body"]:
                on.add(p)
    on.difference_update(scene["body"] | off)
    a = np.full((h, w), bg, dtype=np.int8)
    for cells, color in scene["objects"]:
        for p in cells:
            a[p] = color
    for p in sorted(on):
        a[p] = int(rng.choice(signal_colors))
    if np.count_nonzero(a == bg) * 2 <= a.size:
        return None
    decoded = [q for q in _components(a, bg) if q["kind"]]
    if sorted(map(_signature, decoded)) != sorted(map(_signature, targets)):
        return None  # A signal cluster must not accidentally become a new square.
    out = _receiver_output(a, bg, targets)
    if not np.any(out != bg):
        return None
    if teaching:
        states = {int(_occupied(a, bg, hp)) + 2 * int(_occupied(a, bg, vp))
                  for corner, (hp, vp) in _wiring(core)}
        if states != {0, 1, 2, 3}:
            return None
    return {"input": a, "output": out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return (a.shape, a.astype(np.int8).tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    mode = str(rng.choice(("separate", "nested", "shared", "mixed")))
    palette_mode = int(rng.integers(3))
    base = [int(x) for x in rng.choice(10, size=6, replace=False)]
    puzzle, seen = {"train": [], "test": []}, set()
    taught_nested = taught_shared = False
    for split, total in (("train", ntrain), ("test", ntest)):
        for i in range(total):
            teaching = i + 1 if split == "train" and i < 2 else 0
            if teaching == 1:
                motif, core_kind = ("nested" if mode == "mixed" else "single"), "frame"
            elif teaching == 2:
                motif = "shared" if mode in ("mixed", "shared") else "nested" if mode == "nested" else "single"
                core_kind = "solid"
            else:
                choices = {"separate": ("single",), "nested": ("single", "nested"),
                           "shared": ("single", "shared"),
                           "mixed": ("single", "nested", "shared", "nested_shared")}[mode]
                motif = "nested_shared" if mode == "mixed" and split == "test" and i == 0 else str(rng.choice(choices))
                if split == "test" and mode in ("nested", "shared") and i == 0:
                    motif = mode
                core_kind = None
            colors = base[:]
            if palette_mode == 1:
                available = [c for c in range(10) if c != base[0]]
                colors = [base[0]] + [int(c) for c in rng.choice(available, 5, replace=False)]
            elif palette_mode == 2:
                colors = [int(c) for c in rng.choice(10, 6, replace=False)]
            palette = (colors[0], colors[1:4], colors[4:6])
            for _ in range(120):
                pair = _build_pair(rng, palette, motif, core_kind, teaching, split == "test")
                if pair is not None and _key(pair["input"]) not in seen:
                    *_, nested, shared = _features(pair["input"])
                    if split == "test" and ((nested and not taught_nested) or (shared and not taught_shared)):
                        continue
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    if split == "train":
                        taught_nested |= nested
                        taught_shared |= shared
                    break
            else:
                raise RuntimeError(f"construction failed for {split} {i}, {motif}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    bg = _background(a)
    records = _components(a, bg)
    targets = [q for q in records if q["kind"]]
    states, powered = set(), 0
    tap_uses = Counter()
    for q in targets:
        for corner, (hp, vp) in _wiring(q):
            hv = (int(_occupied(a, bg, hp)), int(_occupied(a, bg, vp)))
            states.add(hv[0] + 2 * hv[1])
            powered += bool(hv[0] or hv[1])
            for p in (hp, vp):
                tap_uses[p] += 1
    nested = any(x is not y and x["kind"] == "frame"
                 and x["box"][0] < y["box"][0] <= y["box"][1] < x["box"][1]
                 and x["box"][2] < y["box"][2] <= y["box"][3] < x["box"][3]
                 for x in targets for y in targets)
    shared = any(n > 1 and _occupied(a, bg, p) for p, n in tap_uses.items())
    return bg, records, targets, states, powered, nested, shared


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_states, kinds = set(), set(), set()
    taught_nested = taught_shared = False
    diagonal_witness = False
    distinguished = {name: False for name in ("and", "xor", "horizontal", "vertical", "all", "no_square_filter")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _array(pair["input"]), _array(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            bg, records, targets, states, powered, nested, shared = _features(a)
            # All eight taps exist; edge-reaching taps are allowed.
            assert all(0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]
                       for q in targets for _, taps in _wiring(q) for p in taps)
            expected = np.full(a.shape, bg, dtype=np.int8)
            corner_set = set()
            for q in targets:
                for corner, taps in _wiring(q):
                    corner_set.add(corner)
                    if any(_occupied(a, bg, p) for p in taps):
                        expected[corner] = a[corner]
            assert np.array_equal(out, expected)
            assert np.array_equal(solve(a), expected)
            assert all(tuple(p) in corner_set for p in np.argwhere(out != bg))
            assert np.all(out[out != bg] == a[out != bg])
            if split == "train":
                taught_nested |= nested
                taught_shared |= shared
                train_states.update(states)
                kinds.update(q["kind"] for q in targets)
                for q in targets:
                    for corner, (hp, vp) in _wiring(q):
                        diagonal = (vp[0], hp[1])
                        diagonal_witness |= (not _occupied(a, bg, hp) and not _occupied(a, bg, vp)
                                             and _occupied(a, bg, diagonal))
                for op in distinguished:
                    if op == "no_square_filter":
                        candidates = [q for q in records if q["height"] >= 2 and q["width"] >= 2]
                        alt = _receiver_output(a, bg, candidates)
                    else:
                        alt = _receiver_output(a, bg, targets, op)
                    distinguished[op] |= not np.array_equal(alt, out)
            else:
                assert not nested or taught_nested
                assert not shared or taught_shared
    assert train_states == {0, 1, 2, 3}
    assert kinds == {"frame", "solid"}
    assert diagonal_witness, "training must reject a diagonal false friend"
    assert all(distinguished.values()), distinguished


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded semantic, metamorphic, count, corruption, and 42-seed checks."""
    # Hand-constructed receiver truth table, including a diagonal false friend.
    a = np.zeros((9, 10), dtype=np.int8)
    a[2:7, 2:7] = 4
    a[3:6, 3:6] = 0
    a[1, 1] = 8                    # TL: diagonal only, off
    a[2, 7] = 3                    # TR: horizontal only
    a[7, 6] = 8                    # BR: vertical only
    a[6, 1] = a[7, 2] = 3         # BL: both
    expected = np.zeros_like(a)
    expected[2, 6] = expected[6, 6] = expected[6, 2] = 4
    assert np.array_equal(solve(a), expected)
    # Adding a same-colored tail invalidates the WHOLE receiver component.
    tail = a.copy()
    tail[4, 1] = 4
    assert not np.any(solve(tail))
    # A powered nonsquare loop and a square-bounding L remain invisible.
    rect = np.zeros((10, 12), dtype=np.int8)
    rect[2, 2:8] = rect[6, 2:8] = 5
    rect[2:7, 2] = rect[2:7, 7] = 5
    rect[1, 2] = 1
    assert not np.any(solve(rect))

    # One source at a shared tap fans out to two different colored receivers.
    shared = np.zeros((10, 10), dtype=np.int8)
    for r, c in _square(3, "frame"):
        shared[r + 2, c + 2] = 4
    shared[5:7, 5:7] = 6
    shared[4, 5] = 8
    expected = np.zeros_like(shared)
    expected[4, 4], expected[5, 5] = 4, 6
    assert np.array_equal(solve(shared), expected)

    # An inner body does not disqualify a disconnected outer frame.
    nested = np.zeros((12, 12), dtype=np.int8)
    for r, c in _square(9, "frame"):
        nested[r + 1, c + 1] = 4
    nested[4:7, 4:7] = 3
    nested[3, 4] = nested[1, 10] = 8
    expected = np.zeros_like(nested)
    expected[4, 4], expected[1, 9] = 3, 4
    assert np.array_equal(solve(nested), expected)

    dimensions, structures, palettes, target_counts, sides, metrics = set(), set(), set(), set(), set(), Counter()
    fingerprints = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(_fingerprint(p))
        for split in ("train", "test"):
            for pair in p[split]:
                a, out = pair["input"], pair["output"]
                bg, records, targets, states, powered, nested, shared = _features(a)
                dimensions.add(a.shape)
                roles = {}
                structures.add((a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.ravel())))
                palettes.add((bg, tuple(sorted(set(int(c) for c in a.ravel())))))
                target_counts.add(len(targets))
                sides.update(q["height"] for q in targets)
                metrics[split + "_nested"] += nested
                metrics[split + "_shared"] += shared
                metrics["mixed"] += nested and shared
                metrics["multicolor_targets"] += len({q["color"] for q in targets}) > 1
                metrics["large_solid"] += any(q["kind"] == "solid" and q["height"] > 2 for q in targets)
                metrics["port_on_edge"] += any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
                                                  for q in targets for _, taps in _wiring(q) for r, c in taps)
                # Geometry, not literal colors, determines the operation.
                permutation = np.random.default_rng(seed + 1000).permutation(10)
                assert np.array_equal(solve(permutation[a]), permutation[out])
                # Rotation/reflection preserve the wiring, including H/V exchange.
                assert np.array_equal(solve(np.rot90(a)), np.rot90(out))
                assert np.array_equal(solve(np.fliplr(a)), np.fliplr(out))
    assert len(fingerprints) == 42
    assert len(dimensions) >= 60 and len(structures) >= 195 and len(palettes) >= 35
    assert len(target_counts) >= 5 and len(sides) >= 8
    assert all(metrics[k] > 0 for k in ("train_nested", "test_nested", "train_shared", "test_shared",
                                        "mixed", "multicolor_targets", "large_solid", "port_on_edge"))
    assert _fingerprint(generate_puzzle(12345, 7, 3)) == _fingerprint(generate_puzzle(12345, 7, 3))
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(900 + ntrain * 10 + ntest, ntrain, ntest)
            assert len(p["train"]) == ntrain and len(p["test"]) == ntest
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4., 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {counts}")
    p = generate_puzzle(0)
    p["test"][0]["output"] = p["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an unprocessed input as output")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(dimensions),
            "structures": len(structures), "palettes": len(palettes),
            "target_counts": sorted(target_counts), "square_sides": sorted(sides),
            "features": dict(metrics)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
