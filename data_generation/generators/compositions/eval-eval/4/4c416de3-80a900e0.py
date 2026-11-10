"""Twin-ray capture restores locally reflected corner stencils.

All interpretation is recovered from input pixels. Sampling geometry is used
only to make inputs and construction labels, never by solve().
"""

from collections import Counter
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
INWARD = ((1, 1), (1, -1), (-1, -1), (-1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _background(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _infer_background(a):
    parity = np.indices(a.shape).sum(axis=0) % 2
    colors = []
    for phase in (0, 1):
        counts = np.bincount(a[parity == phase], minlength=10)
        c = int(counts.argmax())
        assert counts[c] * 2 > np.count_nonzero(parity == phase), "Background is not identifiable"
        colors.append(c)
    assert colors[0] != colors[1]
    return tuple(colors)


def _diamond(center, radius):
    y, x = center
    return {(y + u, x + v) for u in range(-radius, radius + 1)
            for v in range(-radius, radius + 1) if abs(u) + abs(v) == radius}


def _cap(center, radius, direction):
    y, x = center
    dy, dx = direction
    return [(y + dy * (k + 1), x + dx * (radius - k + 1))
            for k in range(radius + 1)]


def _ray(endpoint, direction, shape):
    y, x = endpoint
    dy, dx = direction
    path = []
    y, x = y + dy, x + dx
    while _inside((y, x), shape):
        path.append((y, x))
        y, x = y + dy, x + dx
    return tuple(path)


def _signals(emitters, shape):
    signals = []
    used = set()
    for e in emitters:
        for d, color in sorted(e["caps"].items()):
            cap = _cap(e["center"], e["radius"], d)
            paths = (_ray(cap[0], d, shape), _ray(cap[-1], d, shape))
            pts = set(paths[0]) | set(paths[1])
            assert not used & pts, "Different rays overlap"
            used |= pts
            signals.append(dict(center=e["center"], radius=e["radius"],
                                direction=d, color=color, paths=paths))
    return signals


def _body(b):
    y, x = b["corner"]
    dy, dx = b["inward"]
    length = b["length"]
    return {(y + k * dy, x) for k in range(length + 1)} | {
        (y, x + k * dx) for k in range(length + 1)}


def _ports(b):
    y, x = b["corner"]
    dy, dx = b["inward"]
    return ((y + b["length"] * dy, x), (y, x + b["length"] * dx))


def _positions(b, pattern):
    y, x = b["corner"]
    dy, dx = b["inward"]
    return {(y + u * dy, x + v * dx) for u, v in pattern}


def _contacts(b, signals):
    """Two distinct rails of ONE appropriately facing cap must reach the tips."""
    p, q = _ports(b)
    matches, single_hits = [], 0
    for i, s in enumerate(signals):
        if b["inward"] != tuple(-v for v in s["direction"]):
            continue
        a, z = map(set, s["paths"])
        paired = (p in a and q in z) or (p in z and q in a)
        if paired:
            matches.append(i)
        single_hits = max(single_hits, int(p in a or p in z) + int(q in a or q in z))
    assert len(matches) <= 1, "Two different signals capture one bracket"
    return tuple(matches), single_hits


def _components(points):
    pending = set(points)
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        seen, todo = {start}, [start]
        while todo:
            y, x = todo.pop()
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in pending:
                    pending.remove(p)
                    seen.add(p)
                    todo.append(p)
        result.append(seen)
    return result


def _read_brackets(points):
    brackets = []
    for component in _components(points):
        choices = []
        for y in (min(p[0] for p in component), max(p[0] for p in component)):
            for x in (min(p[1] for p in component), max(p[1] for p in component)):
                for inward in INWARD:
                    for length in (2, 3, 4):
                        b = dict(corner=(y, x), inward=inward, length=length)
                        if _body(b) == component:
                            choices.append(b)
        assert len(choices) == 1, "Structural component is not a unique bracket"
        brackets.append(choices[0])
    assert 3 <= len(brackets) <= 12
    return sorted(brackets, key=lambda b: b["corner"])


def _recover(input_grid, rotate=False):
    a = _grid(input_grid)
    colors = _infer_background(a)
    base = _background(a.shape, colors)
    foreground = set(map(tuple, np.argwhere(a != base)))
    emitters = []
    for radius in (2, 3, 4):
        h, w = a.shape
        if min(h, w) < 2 * radius + 1:
            continue
        shape = (h - 2 * radius, w - 2 * radius)
        offsets = sorted(_diamond((0, 0), radius))
        u, v = offsets[0]
        vals = a[radius + u:radius + u + shape[0], radius + v:radius + v + shape[1]]
        mask = ~np.isin(vals, colors)
        for u, v in offsets[1:]:
            mask &= a[radius + u:radius + u + shape[0], radius + v:radius + v + shape[1]] == vals
        for y, x in np.argwhere(mask):
            emitters.append(dict(center=(int(y) + radius, int(x) + radius), radius=radius,
                                 structural=int(vals[y, x]), caps={}))
    assert 1 <= len(emitters) <= 3, "Missing or ambiguous diamond outlines"
    occupied = set()
    structural_colors = {e["structural"] for e in emitters}
    for e in emitters:
        cells = _diamond(e["center"], e["radius"])
        assert not occupied & cells
        occupied |= cells
    for e in emitters:
        for d in DIRECTIONS:
            cells = _cap(e["center"], e["radius"], d)
            # This family requires complete visible caps; absent sides are background.
            if not all(_inside(p, a.shape) for p in cells):
                continue
            values = {int(a[p]) for p in cells}
            if len(values) == 1 and not values & (set(colors) | structural_colors):
                assert not occupied & set(cells)
                e["caps"][d] = values.pop()
                occupied |= set(cells)
        assert e["caps"]
    emitters.sort(key=lambda e: e["center"])
    signals = _signals(emitters, a.shape)
    rays = {p for s in signals for path in s["paths"] for p in path}
    assert not rays & occupied, "A ray hits another emitter"
    remaining = foreground - occupied
    cap_colors = {s["color"] for s in signals}
    candidates = []
    for frame in sorted({int(a[p]) for p in remaining} - structural_colors - cap_colors):
        try:
            points = {p for p in remaining if int(a[p]) == frame}
            brackets = _read_brackets(points)
            clues = {b["corner"]: [] for b in brackets}
            corners = [b["corner"] for b in brackets]
            for p in sorted(remaining - points):
                distances = [(p[0] - y) ** 2 + (p[1] - x) ** 2 for y, x in corners]
                assert distances.count(min(distances)) == 1, "Ambiguous clue association"
                i = distances.index(min(distances))
                b = brackets[i]
                assert max(abs(p[k] - b["corner"][k]) for k in (0, 1)) <= 2
                clues[b["corner"]].append(p)
            patterns = []
            for b in brackets:
                coords = clues[b["corner"]]
                b["color"] = int(a[coords[0]]) if coords else None
                assert all(int(a[p]) == b["color"] for p in coords)
                if coords:
                    assert b["color"] not in structural_colors | {frame}
                y, x = b["corner"]
                dy, dx = b["inward"]
                pattern = frozenset(((p[0] - y) * dy, (p[1] - x) * dx) for p in coords)
                if rotate and dy != dx:
                    pattern = frozenset((v, u) for u, v in pattern)
                b["clue"] = pattern
                if pattern:
                    patterns.append(pattern)
            maximal = {p for p in patterns if all(q <= p for q in patterns)}
            assert len(maximal) == 1, "No intact compatible template"
            pattern = maximal.pop()
            assert 3 <= len(pattern) <= 14
            assert any(u > 0 and v > 0 for u, v in pattern)
            assert any(u < 0 or v < 0 for u, v in pattern)
            assert all((u, v) != (0, 0) and not (u == 0 and v > 0)
                       and not (v == 0 and u > 0) for u, v in pattern)
            for b in brackets:
                local = frozenset((v, u) for u, v in pattern) if rotate and b["inward"][0] != b["inward"][1] else pattern
                full = _positions(b, local)
                assert all(_inside(p, a.shape) for p in full)
                assert all(abs(u - v) < b["length"] for u, v in local)
                assert not full & (points | occupied | rays), "Template intersects structure or ray"
                for p in full:
                    dist = [(p[0] - y) ** 2 + (p[1] - x) ** 2 for y, x in corners]
                    idx = corners.index(b["corner"])
                    assert dist.count(min(dist)) == 1 and dist[idx] == min(dist)
                    if b["color"] is not None:
                        assert int(a[p]) in (*colors, b["color"])
                    else:
                        assert a[p] == base[p]
                b["matches"], b["hits"] = _contacts(b, signals)
            candidates.append(dict(background=colors, base=base, foreground=foreground,
                                   emitters=emitters, signals=signals, frame=frame,
                                   brackets=brackets, pattern=pattern, rotated=rotate))
        except AssertionError:
            continue
    assert len(candidates) == 1, "Input has no unique geometric interpretation"
    return candidates[0]


def _predict(a, info, restoration="paired", whole_cap=False):
    out = a.copy()
    for s in info["signals"]:
        paths = s["paths"]
        if whole_cap:
            paths = [_ray(p, s["direction"], a.shape)
                     for p in _cap(s["center"], s["radius"], s["direction"])]
        for path in paths:
            for p in path:
                if a[p] == info["base"][p]:
                    out[p] = s["color"]
    for b in info["brackets"]:
        if b["color"] is None:
            continue
        restore = (bool(b["matches"]) if restoration == "paired" else
                   b["hits"] > 0 if restoration == "any" else
                   restoration == "all")
        if restore:
            pattern = info["pattern"]
            if info["rotated"] and b["inward"][0] != b["inward"][1]:
                pattern = frozenset((v, u) for u, v in pattern)
            for p in _positions(b, pattern):
                out[p] = b["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use checkerboard geometry, cap endpoints and the input's intact stencil."""
    a = _grid(input_grid)
    return _predict(a, _recover(a))


def _sample_pattern(rng, radius, aperture, asymmetric=False):
    domain = {(u, v) for u in range(-radius, radius + 1)
              for v in range(-radius, radius + 1)
              if abs(u - v) < aperture and (u, v) != (0, 0)
              and not (u == 0 and v > 0) and not (v == 0 and u > 0)}
    symmetric = not asymmetric and rng.random() < .40
    for _ in range(100):
        size = int(rng.integers(3, min(14, len(domain)) + 1))
        cells = sorted(domain)
        rng.shuffle(cells)
        p = set(cells[:size])
        if symmetric:
            p |= {(v, u) for u, v in p}
        if (len(p) <= 14 and any(u > 0 and v > 0 for u, v in p)
                and any(u < 0 or v < 0 for u, v in p)
                and (not asymmetric or p != {(v, u) for u, v in p})):
            return frozenset(p)
    return frozenset({(-1, -1), (-1, 0), (1, 1)})


def _emitter_cells(e):
    return _diamond(e["center"], e["radius"]) | {
        p for d in e["caps"] for p in _cap(e["center"], e["radius"], d)}


def _can_place(b, brackets, occupied, rays, pattern, shape):
    points = _body(b) | _positions(b, pattern)
    if not all(_inside(p, shape) for p in points):
        return False
    if points & occupied or _positions(b, pattern) & rays:
        return False
    for old in brackets:
        if max(abs(b["corner"][i] - old["corner"][i]) for i in (0, 1)) < 6:
            return False
        oldpoints = _body(old) | _positions(old, pattern)
        if points & oldpoints:
            return False
        if any((y + dy, x + dx) in _body(old) for y, x in _body(b)
               for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            return False
    return True


def _make_scene(rng, max_radius, masks=None, test=False, roles=None):
    witness = masks is not None
    shape = tuple(int(rng.integers(25 if witness or test else 19, 31)) for _ in range(2))
    if rng.random() < .25:
        shape = (shape[0], shape[0])
    colors = tuple(map(int, rng.choice(10, 2, replace=False))) if roles is None else roles[:2]
    available = [c for c in range(10) if c not in colors]
    rng.shuffle(available)
    frame = available[0] if roles is None else roles[2]
    structural = next(c for c in available if c != frame)
    payload = [c for c in available if c not in (frame, structural)]
    radius = max_radius if witness else int(rng.integers(2, max_radius + 1))
    if masks is None:
        mask = int(rng.integers(1, 16))
        directions = [d for i, d in enumerate(DIRECTIONS) if mask & (1 << i)]
    else:
        directions = list(masks)
    repeated = directions[int(rng.integers(len(directions)))] if test else None
    center = tuple(int(rng.integers(radius + 2, s - radius - 2)) for s in shape)
    if test:
        # Room for two separate capture stations along the repeated direction.
        center = tuple(int(rng.integers(max(14, radius + 3), s - radius - 1)) if v < 0
                       else int(rng.integers(radius + 1, s - 14))
                       for s, v in zip(shape, repeated))
    bank = list(map(int, rng.choice(payload, int(rng.integers(1, min(4, len(directions)) + 1)), replace=False)))
    caps = {d: int(rng.choice(bank)) for d in directions}
    emitters = [dict(center=center, radius=radius, structural=structural, caps=caps)]
    occupied = _emitter_cells(emitters[0])
    assert all(_inside(p, shape) for p in occupied)
    signals = _signals(emitters, shape)
    rays = {p for s in signals for path in s["paths"] for p in path}
    # Additional emitters are free placed, with separately sampled masks/radii.
    if not witness and rng.random() < .45:
        for _ in range(40):
            rr = int(rng.integers(2, max_radius + 1))
            cc = tuple(int(rng.integers(rr + 2, s - rr - 2)) for s in shape)
            dd = [d for i, d in enumerate(DIRECTIONS) if int(rng.integers(1, 16)) & (1 << i)]
            if not dd:
                continue
            e = dict(center=cc, radius=rr, structural=structural,
                     caps={d: int(rng.choice(payload)) for d in dd})
            pp = _emitter_cells(e)
            ss = _signals([e], shape)
            qq = {p for s in ss for path in s["paths"] for p in path}
            if pp & (occupied | rays) or qq & (occupied | rays):
                continue
            emitters.append(e)
            occupied |= pp
            rays |= qq
            signals = _signals(emitters, shape)
            break
    aperture = min(e["radius"] for e in emitters)
    stencil_radius = 1 if rng.random() < .28 else 2
    pattern = _sample_pattern(rng, stencil_radius, aperture, asymmetric=witness)
    brackets = []
    primary_signals = [s for s in signals if s["center"] == center]
    rng.shuffle(primary_signals)
    for s in primary_signals:
        dy, dx = s["direction"]
        candidates = []
        for distance in range(radius + 2, 30):
            b = dict(corner=(center[0] + dy * distance, center[1] + dx * distance),
                     inward=(-dy, -dx), length=radius)
            if _can_place(b, [], occupied, rays, pattern, shape) and _contacts(b, signals)[0]:
                candidates.append(b)
        rng.shuffle(candidates)
        needed = 2 if test and s["direction"] == repeated else 1
        chosen = []
        for b in candidates:
            if _can_place(b, brackets + chosen, occupied, rays, pattern, shape):
                chosen.append(b)
                if len(chosen) == needed:
                    break
        assert len(chosen) == needed, "No room for capture stations"
        for b in chosen:
            b["kind"] = "active"
            b["signal_color"] = s["color"]
            brackets.append(b)
    # A one-rail decoy differs in diagonal alignment by twice the radius.
    options = []
    for s in signals:
        y, x = s["center"]
        dy, dx = s["direction"]
        r = s["radius"]
        for t in range(r + 2, 30):
            for offset in (-2 * r, 2 * r):
                for axis in (0, 1):
                    c = [y + dy * t, x + dx * t]
                    c[axis] += offset
                    b = dict(corner=tuple(c), inward=(-dy, -dx), length=r, kind="single")
                    if (_can_place(b, brackets, occupied, rays, pattern, shape)
                            and _contacts(b, signals) == ((), 1)
                            and sum(p in rays for p in _ports(b)) == 1):
                        options.append(b)
    assert options, "No one-rail control fits"
    brackets.append(options[int(rng.integers(len(options)))])
    # Separate intact reference; optional extra controls can be unmarked.
    width = max(abs(u - v) for u, v in pattern)
    extras = ["reference"] + (["inactive"] if rng.random() < .55 else [])
    if rng.random() < .28:
        extras.append("unmarked")
    for kind in extras:
        for _ in range(220):
            b = dict(corner=tuple(int(rng.integers(2, s - 2)) for s in shape),
                     inward=INWARD[int(rng.integers(4))],
                     length=int(rng.integers(max(2, width + 1), 5)), kind=kind)
            if (_can_place(b, brackets, occupied, rays, pattern, shape)
                    and not _contacts(b, signals)[0] and _contacts(b, signals)[1] == 0):
                brackets.append(b)
                break
        else:
            assert kind != "reference", "No reference fits"
    assert any(b["kind"] == "reference" for b in brackets)
    base = _background(shape, colors)
    inp = base.copy()
    for e in emitters:
        for p in _diamond(e["center"], e["radius"]):
            inp[p] = e["structural"]
        for d, color in e["caps"].items():
            for p in _cap(e["center"], e["radius"], d):
                inp[p] = color
    palette_size = int(rng.integers(1, min(len(payload), len(brackets)) + 1))
    accent_bank = list(map(int, rng.choice(payload, palette_size, replace=False)))
    expected = inp.copy()
    for b in brackets:
        for p in _body(b):
            inp[p] = frame
            expected[p] = frame
        if b["kind"] == "unmarked":
            continue
        color = int(rng.choice(accent_bank))
        if witness and b["kind"] == "active":
            color = int(rng.choice([c for c in payload if c != b["signal_color"]]))
        pts = sorted(_positions(b, pattern))
        rng.shuffle(pts)
        if b["kind"] == "reference" or (b["kind"] == "active" and not witness and rng.random() < .16):
            retained = pts
        else:
            retained = pts[:int(rng.integers(1, len(pts)))]
        for p in retained:
            inp[p] = color
            expected[p] = color
        if b["kind"] == "active":
            for p in pts:
                expected[p] = color
    for s in signals:
        for path in s["paths"]:
            for p in path:
                if inp[p] == base[p]:
                    expected[p] = s["color"]
    info = _recover(inp)
    assert len(info["emitters"]) == len(emitters) and len(info["brackets"]) == len(brackets)
    assert info["pattern"] == pattern
    assert np.array_equal(_predict(inp, info), expected)
    assert not np.array_equal(_predict(inp, info, "none"), expected), "No active partial"
    assert not np.array_equal(_predict(inp, info, "all"), expected), "No inactive partial"
    if witness:
        try:
            rotated = _predict(inp, _recover(inp, rotate=True))
        except AssertionError:
            rotated = None
        assert rotated is None or not np.array_equal(rotated, expected), "Reflection witness failed"
        assert set(_teaching_directions(inp, expected, info)) >= set(masks)
    return {"input": inp, "output": expected}


def _teaching_directions(inp, out, info):
    result = {}
    for s in info["signals"]:
        d = s["direction"]
        if min(map(len, s["paths"])) < 2:
            continue
        if not all(sum(inp[p] == info["base"][p] and out[p] == s["color"] for p in path) >= 2
                   for path in s["paths"]):
            continue
        cap = _cap(s["center"], s["radius"], d)
        interior = [(p[0] + k * d[0], p[1] + k * d[1])
                    for p in cap[1:-1] for k in (1, 2)]
        if (all(_inside(p, inp.shape) and inp[p] == out[p] == info["base"][p] for p in interior)
                and any(b["matches"] and b["clue"] and b["clue"] != info["pattern"]
                        and s is info["signals"][b["matches"][0]] for b in info["brackets"])):
            result[d] = max(result.get(d, 0), s["radius"])
    return result


def _count(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(a):
    a = _grid(a)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    maximum = int(rng.integers(2, 5))
    roles = tuple(map(int, rng.choice(10, 3, replace=False))) if rng.random() < .25 else None
    offset = int(rng.integers(4))
    teaching = (tuple(DIRECTIONS[(offset + k) % 4] for k in (0, 1)),
                tuple(DIRECTIONS[(offset + k) % 4] for k in (2, 3)))
    slots = list(map(int, rng.choice(nt, 2, replace=False)))
    witness = dict(zip(slots, teaching))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            for attempt in range(300):
                try:
                    pair = _make_scene(rng, maximum, masks=witness.get(i) if split == "train" else None,
                                       test=split == "test", roles=roles)
                    key = _key(pair["input"])
                    assert key not in seen
                    if split == "train":
                        info = _recover(pair["input"])
                        captures = Counter(b["matches"][0] for b in info["brackets"] if b["matches"])
                        assert max(captures.values(), default=0) == 1
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError(f"Bounded scene search exhausted at {seed}/{split}/{i}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _closed_form_prediction(inp, info):
    """Second ray and port calculation, using diagonal invariants, not walks."""
    out = inp.copy()
    rows, cols = np.indices(inp.shape)
    fields = []
    for s in info["signals"]:
        dy, dx = s["direction"]
        u = (rows - s["center"][0]) * dy
        v = (cols - s["center"][1]) * dx
        lane_a = (u - v == -s["radius"]) & (np.minimum(u, v) >= 2)
        lane_b = (u - v == s["radius"]) & (np.minimum(u, v) >= 2)
        fields.append((lane_a, lane_b))
        mask = (lane_a | lane_b) & (inp == info["base"])
        out[mask] = s["color"]
    for b in info["brackets"]:
        if b["color"] is None:
            continue
        p, q = _ports(b)
        active = any(b["inward"] == tuple(-v for v in s["direction"])
                     and ((a[p] and z[q]) or (z[p] and a[q]))
                     for s, (a, z) in zip(info["signals"], fields))
        if active:
            for p in _positions(b, info["pattern"]):
                out[p] = b["color"]
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, directions, reflection_ruled_out = set(), {}, False
    training_max = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _recover(inp)
            assert np.array_equal(out, _predict(inp, info)), "Output violates capture/restoration rule"
            assert np.array_equal(out, _closed_form_prediction(inp, info)), "Geometric cross-check failed"
            for p in info["foreground"]:
                assert out[p] == inp[p], "Input foreground changed"
            assert set(np.unique(inp)) == set(np.unique(out))
            assert not np.array_equal(out, _predict(inp, info, "none")), "Omitting A does not change restoration"
            assert not np.array_equal(out, _predict(inp, info, "all")), "Omitting B's gate does not change restoration"
            assert not np.array_equal(out, _predict(inp, info, "any")), "One-ray shortcut is not distinguished"
            assert not np.array_equal(out, _predict(inp, info, whole_cap=True)), "Whole-cap shortcut is not distinguished"
            rays = {p for s in info["signals"] for path in s["paths"] for p in path}
            assert any(b["hits"] == 1 and b["clue"] and b["clue"] != info["pattern"]
                       and sum(p in rays for p in _ports(b)) == 1 for b in info["brackets"]), "No literal one-ray control"
            counts = Counter(b["matches"][0] for b in info["brackets"] if b["matches"])
            if split == "train":
                assert max(counts.values(), default=0) == 1
                for d, witnessed_radius in _teaching_directions(inp, out, info).items():
                    directions[d] = max(directions.get(d, 0), witnessed_radius)
                training_max = max(training_max, max(e["radius"] for e in info["emitters"]))
                try:
                    alt = _predict(inp, _recover(inp, rotate=True))
                except AssertionError:
                    reflection_ruled_out = True
                else:
                    reflection_ruled_out |= not np.array_equal(alt, out)
            else:
                assert max(counts.values(), default=0) >= 2, "Test does not recombine multiple captures on one beam"
                assert max(e["radius"] for e in info["emitters"]) <= training_max
                assert all(directions.get(d, 0) >= max(e["radius"] for e in info["emitters"])
                           for d in DIRECTIONS), "A test radius lacks directional teaching evidence"
    assert set(directions) == set(DIRECTIONS), "Training misses an endpoint/interior/capture direction witness"
    assert reflection_ruled_out, "Training leaves quarter-turn convention ambiguous"


def _measure(pair):
    info = _recover(pair["input"])
    brackets = info["brackets"]
    return dict(shape=tuple(np.asarray(pair["input"]).shape), background=info["background"],
                emitter_count=len(info["emitters"]), brackets=len(brackets),
                radii=tuple(e["radius"] for e in info["emitters"]),
                masks=tuple(sum(1 << DIRECTIONS.index(d) for d in e["caps"]) for e in info["emitters"]),
                size=len(info["pattern"]), pattern=tuple(sorted(info["pattern"])),
                symmetric=info["pattern"] == frozenset((v, u) for u, v in info["pattern"]),
                active=sum(bool(b["matches"]) for b in brackets),
                single=sum(b["hits"] == 1 for b in brackets),
                intact=sum(b["clue"] == info["pattern"] for b in brackets),
                unmarked=sum(not b["clue"] for b in brackets),
                lengths=tuple(len(path) for s in info["signals"] for path in s["paths"]),
                clue_counts=tuple(len(b["clue"]) for b in brackets if b["clue"]),
                accent_reuse=len({b["color"] for b in brackets if b["color"] is not None})
                             < sum(b["color"] is not None for b in brackets),
                local_color_diff=any(b["matches"] and b["color"] is not None and
                                     b["color"] != info["signals"][b["matches"][0]]["color"] for b in brackets),
                layout=tuple((e["center"], e["radius"], tuple(e["caps"])) for e in info["emitters"]) +
                       tuple((b["corner"], b["inward"], b["length"]) for b in brackets))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = [_measure(pair) for p in puzzles for split in ("train", "test") for pair in p[split]]
    assert len({_key(p["train"][0]["input"]) for p in puzzles}) == 42
    assert len({f["shape"] for f in features}) >= 25
    assert len({f["layout"] for f in features}) >= 100
    assert len({f["pattern"] for f in features}) >= 60
    assert {r for f in features for r in f["radii"]} == {2, 3, 4}
    assert {m for f in features for m in f["masks"]} == set(range(1, 16))
    assert {f["emitter_count"] for f in features} >= {1, 2}
    assert any(len(set(f["radii"])) > 1 for f in features)
    assert len({f["brackets"] for f in features}) >= 4
    assert len({f["size"] for f in features}) >= 6
    assert any(f["symmetric"] for f in features) and any(not f["symmetric"] for f in features)
    assert any(f["unmarked"] for f in features) and any(f["intact"] > 1 for f in features)
    assert any(f["accent_reuse"] for f in features)
    assert any(f["local_color_diff"] for f in features)
    assert max(v for f in features for v in f["lengths"]) >= 18
    assert len({f["background"] for f in features}) >= 20
    assert {f["shape"][0] == f["shape"][1] for f in features} == {False, True}
    assert any(30 in f["shape"] for f in features)
    assert max(v for f in features for v in f["clue_counts"]) >= 10
    assert min(v for f in features for v in f["clue_counts"]) == 1
    for p in puzzles:
        validate_generated_puzzle(p)
    repeated = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in ("train", "test")
               for a, b in zip(puzzles[0][s], repeated[s]) for k in ("input", "output"))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(42 + a * 10 + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    # Exact contacts, including a port at a cap (not yet on its ray), and
    # two single rails from different signals: neither is a paired capture.
    b = dict(corner=(3, 3), inward=(1, 1), length=2)
    e = dict(center=(9, 9), radius=2, caps={(-1, -1): 5})
    s = _signals([e], (16, 16))
    assert _contacts(b, s)[0]
    wrong = dict(b, inward=(-1, -1))
    assert not _contacts(wrong, s)[0]
    p, q = _ports(b)
    split = [dict(direction=(-1, -1), paths=((p,), ())),
             dict(direction=(-1, -1), paths=((), (q,)))]
    assert _contacts(b, split) == ((), 1)
    near = dict(corner=(6, 6), inward=(1, 1), length=2)
    assert not _contacts(near, s)[0]
    edge = _cap((3, 3), 2, (-1, -1))
    assert not _ray(edge[0], (-1, -1), (9, 9)) and not _ray(edge[-1], (-1, -1), (9, 9))
    # Mutation checks reject both a missing restoration and a missing ray cell.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    pair = bad["test"][0]
    info = _recover(pair["input"])
    active = next(b for b in info["brackets"] if b["matches"] and b["clue"] != info["pattern"])
    point = next(p for p in _positions(active, info["pattern"]) if pair["input"][p] != pair["output"][p])
    pair["output"][point] = pair["input"][point]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Damaged restoration label accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    pair = bad["test"][0]
    info = _recover(pair["input"])
    s = info["signals"][0]
    ray_cell = next(p for path in s["paths"] for p in path if pair["input"][p] == info["base"][p])
    pair["output"][ray_cell] = pair["input"][ray_cell]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Damaged ray label accepted")
    # Input interventions demonstrate causal interaction, beyond label agreement.
    witness = next(pair for pair in puzzles[0]["train"] if len(_recover(pair["input"])["emitters"][0]["caps"]) == 2)
    info = _recover(witness["input"])
    s = next(s for s in info["signals"] if any(
        b["matches"] and b["clue"] != info["pattern"] and
        info["signals"][b["matches"][0]] is s for b in info["brackets"]))
    intervention = witness["input"].copy()
    for p in _cap(s["center"], s["radius"], s["direction"]):
        intervention[p] = info["base"][p]
    answer = solve(intervention)
    target = next(b for b in info["brackets"] if b["matches"] and
                  info["signals"][b["matches"][0]] is s and b["clue"] != info["pattern"])
    absent = _positions(target, info["pattern"]) - _positions(target, target["clue"])
    assert all(answer[p] == intervention[p] == info["base"][p] for p in absent)
    # Changing the observable reference changes a received completion, without
    # changing the cap, rays or bracket geometry.
    removable = [(b, q) for b in info["brackets"] for q in sorted(info["pattern"])
                 if b["matches"] and b["clue"] and len(info["pattern"]) > 3
                 and any(u > 0 and v > 0 for u, v in info["pattern"] - {q})
                 and any(u < 0 or v < 0 for u, v in info["pattern"] - {q})
                 and q not in b["clue"]]
    assert removable
    target, q = removable[0]
    intervention = witness["input"].copy()
    for b in info["brackets"]:
        point = next(iter(_positions(b, {q})))
        if q in b["clue"]:
            intervention[point] = info["base"][point]
    answer = solve(intervention)
    point = next(iter(_positions(target, {q})))
    assert answer[point] == info["base"][point] and witness["output"][point] == target["color"]
    return dict(seeds=42, pairs=len(features), shapes=len({f["shape"] for f in features}),
                templates=len({f["pattern"] for f in features}),
                layouts=len({f["layout"] for f in features}),
                emitter_counts=sorted({f["emitter_count"] for f in features}),
                stencil_sizes=sorted({f["size"] for f in features}))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
