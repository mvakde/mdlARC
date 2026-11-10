"""Clocked scan-order readout of payloads in directional hollow vessels.

Only NumPy and the standard library are used.  No parent modules or stored
examples are imported.  Construction and input recognition are separate.
"""
from __future__ import annotations

import numpy as np


_DIRS = ("U", "R", "D", "L")
_STEP = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, component = [root], {root}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in remaining:
                        remaining.remove(p)
                        component.add(p)
                        stack.append(p)
        result.append(component)
    return result


def _local(p, direction):
    r, c = p
    return {"U": (r, c), "R": (-c, r), "D": (-r, -c), "L": (c, -r)}[direction]


def _world(p, direction):
    y, x = p
    return {"U": (y, x), "R": (x, -y), "D": (-y, -x), "L": (-x, y)}[direction]


def _clock(grid):
    """Return a top-clock view and its rotation, with visible color roles."""
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    if counts.max() * 2 <= grid.size:
        raise ValueError("background must occupy more than half the input")
    candidates = []
    for k in range(4):
        g = np.rot90(grid, k)
        for color in values:
            if color == background:
                continue
            points = np.argwhere(g == color)
            if len(points) < 2 or not np.all(points[:, 0] == 0):
                continue
            cols = sorted(int(c) for c in points[:, 1])
            start, length = cols[0], len(cols)
            if cols != list(range(start, start + length)):
                continue
            for swatch in (start - 1, start + length):
                if not 0 <= swatch < g.shape[1]:
                    continue
                accent = int(g[0, swatch])
                if accent not in (background, int(color)) and np.sum(g == accent) == 1:
                    candidates.append((g.copy(), k, dict(background=background,
                        seed=int(color), accent=accent, start=start,
                        length=length, swatch=swatch)))
    if len(candidates) != 1:
        raise ValueError("expected one edge bar with one adjacent accent swatch")
    return candidates[0]


def _envelope(frame, direction):
    """Recognize thin monotone arms from cross-sections, not a shape bank."""
    points = {_local(p, direction) for p in frame}
    lo, hi = min(y for y, _ in points), max(y for y, _ in points)
    if hi - lo < 2:
        return None
    rows = [sorted(x for y, x in points if y == d) for d in range(lo, hi + 1)]
    if len(rows[0]) != 1:
        return None
    tip_x = rows[0][0]
    split = False
    previous = (tip_x, tip_x)
    cavity = set()
    for i, row in enumerate(rows[1:], 1):
        if len(row) == 1 and not split and row[0] == tip_x:
            continue
        if len(row) != 2 or not row[0] < tip_x < row[1]:
            return None
        a, b = row
        if not (0 <= previous[0] - a <= 1 and 0 <= b - previous[1] <= 1):
            return None
        split = True
        previous = (a, b)
        cavity.update(_world((lo + i, x), direction) for x in range(a + 1, b))
    if not cavity:
        return None
    return _world((lo, tip_x), direction), cavity


def _ray(shape, tip, direction, quantity):
    dr, dc = _STEP[direction]
    r, c = tip
    cells = []
    for _ in range(quantity):
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            break
        cells.append((r, c))
    return cells


def _objects(g, clock):
    bg = clock["background"]
    reserved = {bg, clock["seed"], clock["accent"]}
    candidates = []
    for color in sorted(set(map(int, np.unique(g))) - reserved):
        for frame in _components(map(tuple, np.argwhere(g == color).tolist())):
            if len(frame) < 5:
                continue
            for direction in _DIRS:
                envelope = _envelope(frame, direction)
                if envelope is None:
                    continue
                tip, cavity = envelope
                colors = {int(g[p]) for p in cavity} - {bg}
                if len(colors) != 1 or colors & (reserved | {color}):
                    continue
                payload_color = next(iter(colors))
                payload = {p for p in cavity if g[p] == payload_color}
                if not 1 <= len(payload) < len(cavity):
                    continue
                rr, cc = zip(*frame)
                box = {(r, c) for r in range(min(rr), max(rr) + 1)
                       for c in range(min(cc), max(cc) + 1)}
                candidates.append(dict(frame=frame, cavity=cavity, tip=tip,
                    direction=direction, frame_color=color, payload_color=payload_color,
                    payload=payload, box=box))
    # A bead cluster that happens to have arrow geometry belongs to its vessel.
    objects = [o for o in candidates if not any(o is not b and o["frame"] <= b["cavity"]
                                               for b in candidates)]
    objects.sort(key=lambda o: min(o["frame"]))
    if not objects or len({frozenset(o["frame"]) for o in objects}) != len(objects):
        raise ValueError("missing or directionally ambiguous vessels")
    rail = {(r, c) for r in range(g.shape[0])
            for c in range(clock["start"], clock["start"] + clock["length"])}
    rail.add((0, clock["swatch"]))
    boxes, rays = set(), set()
    for o in objects:
        q = len(o["payload"])
        o["ray"] = _ray(g.shape, o["tip"], o["direction"], q)
        o["available"] = len(_ray(g.shape, o["tip"], o["direction"], 30))
        if boxes & o["box"] or rail & o["box"]:
            raise ValueError("vessel regions must be separate from each other and the rail")
        if rays & set(o["ray"]) or rail & set(o["ray"]):
            raise ValueError("ray collision")
        boxes.update(o["box"])
        rays.update(o["ray"])
    if boxes & rays or any(g[p] != bg for p in rays):
        raise ValueError("rays require empty input corridors")
    return objects


def _word(obj, accent):
    # Same-distance ties are identical colors, so lateral tie order has no effect.
    return [accent if r % 6 == 4 else obj["payload_color"]
            for r, _ in sorted(obj["payload"])]


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    g, k, clock = _clock(grid)
    objects = _objects(g, clock)
    out = g.copy()
    s, e = clock["start"], clock["start"] + clock["length"]
    for r in range(len(g)):
        out[r, s:e] = (clock["background"] if r % 2 else
                       clock["accent"] if r % 6 == 4 else clock["seed"])
    for o in objects:
        for p in o["payload"]:
            out[p] = clock["background"]
        for p, color in zip(o["ray"], _word(o, clock["accent"])):
            out[p] = color
    return np.rot90(out, -k).copy()


def _profile(rng, kind="ordinary"):
    if kind == "anchor":
        a, b = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        stem, tail = int(rng.integers(2)), int(rng.integers(3, 5))
    elif kind == "small":
        a, b = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        stem, tail = 0, int(rng.integers(2, 4))
    elif kind == "large":
        a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        stem, tail = int(rng.integers(2)), int(rng.integers(4, 7))
    else:
        a, b = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        stem, tail = int(rng.integers(3)), int(rng.integers(1, 6))
    arms = [(0, 0)] * (stem + 1) + [(-1, 1)]
    left = right = 1
    while left < a or right < b:
        dl = left < a and (right == b or rng.random() < .8)
        dr = right < b and (left == a or rng.random() < .8)
        if not dl and not dr:
            dl = left < a
            dr = not dl
        left, right = left + int(dl), right + int(dr)
        arms.append((-left, right))
    arms.extend([(-a, b)] * tail)
    frame, cavity = set(), set()
    for y, (l, r) in enumerate(arms):
        frame.update(((y, l), (y, r)))
        cavity.update((y, x) for x in range(l + 1, r))
    return dict(frame=frame, cavity=cavity, tip=(0, 0))


def _orient_profile(profile, direction):
    transformed = {p: _world(p, direction) for p in profile["frame"] | profile["cavity"]}
    r0 = min(r for r, _ in transformed.values())
    c0 = min(c for _, c in transformed.values())
    def f(p):
        r, c = transformed[p]
        return r - r0, c - c0
    frame = {f(p) for p in profile["frame"]}
    return dict(frame=frame, cavity={f(p) for p in profile["cavity"]},
        tip=f(profile["tip"]), direction=direction,
        height=max(r for r, _ in frame) + 1, width=max(c for _, c in frame) + 1)


def _subset(rng, cells, n):
    cells = sorted(cells)
    return {cells[int(i)] for i in rng.choice(len(cells), n, replace=False)}


def _payload(rng, cavity, quantity, pattern):
    if quantity >= len(cavity):
        return None
    if pattern in ("low_a", "low_b", "high"):
        before, special = {"low_a": (1, 2), "low_b": (2, 1), "high": (2, 2)}[pattern]
        before_options = range(1, quantity - special) if pattern == "high" else (before,)
        options = []
        for d in sorted({r for r, _ in cavity if r % 6 == 4}):
            groups = ({p for p in cavity if p[0] < d and p[0] % 6 != 4},
                      {p for p in cavity if p[0] == d},
                      {p for p in cavity if p[0] > d and p[0] % 6 != 4})
            for before in before_options:
                after = quantity - before - special
                amounts = (before, special, after)
                if all(len(group) >= n for group, n in zip(groups, amounts)):
                    options.append((groups, amounts))
        if not options:
            return None
        groups, amounts = options[int(rng.integers(len(options)))]
        selected = [_subset(rng, group, n) for group, n in zip(groups, amounts)]
        if pattern == "low_a":
            # Two connected beads still encode two units, ruling out component count.
            pairs = [(p, (p[0], p[1] + 1)) for p in sorted(groups[1])
                     if (p[0], p[1] + 1) in groups[1]]
            if not pairs:
                return None
            first = min(selected[1])
            pair = min(pairs, key=lambda ab: abs(ab[0][1] - first[1]))
            selected[1] = set(pair)
        return set().union(*selected)
    highlighted = {p for p in cavity if p[0] % 6 == 4}
    normal = cavity - highlighted
    if pattern == "mixed":
        if not highlighted or not normal or quantity < 2:
            return None
        chosen = _subset(rng, highlighted, 1) | _subset(rng, normal, 1)
        return chosen | _subset(rng, cavity - chosen, quantity - 2)
    if pattern == "compact":
        root = sorted(cavity)[int(rng.integers(len(cavity)))]
        chosen = {root}
        while len(chosen) < quantity:
            frontier = {p for p in cavity - chosen if any(abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1 for q in chosen)}
            chosen |= _subset(rng, frontier or cavity - chosen, 1)
        return chosen
    if pattern == "scattered":
        chosen = _subset(rng, cavity, 1)
        while len(chosen) < quantity:
            distance = {p: min(max(abs(p[0]-q[0]), abs(p[1]-q[1])) for q in chosen) for p in cavity - chosen}
            best = max(distance.values())
            chosen |= _subset(rng, {p for p in distance if distance[p] == best}, 1)
        return chosen
    return _subset(rng, cavity, quantity)


def _spec(rng, palette, *, profile=None, direction=None, quantity=None,
          colors=None, pattern=None, mode=None, kind="ordinary"):
    shape = _orient_profile(profile or _profile(rng, kind), direction or str(rng.choice(_DIRS)))
    cap = len(shape["cavity"])
    quantity = quantity if quantity is not None else int(rng.integers(1, min(9, cap - 1) + 1))
    if quantity >= cap:
        return None
    f, p = colors or tuple(map(int, rng.choice(palette, 2, replace=False)))
    shape.update(quantity=quantity, frame_color=f, payload_color=p,
                 pattern=pattern or str(rng.choice(["random", "compact", "scattered"])),
                 mode=mode or str(rng.choice(["free", "free", "any", "exact", "clip", "zero"])))
    return shape


def _pack(rng, shape, rail, specs, required):
    placed, boxes, guards, rays = [], set(), set(), set()
    h, w = shape
    for index, spec in enumerate(specs):
        if spec is None:
            if index < required:
                return None
            continue
        positions = [(r, c) for r in range(h - spec["height"] + 1)
                     for c in range(w - spec["width"] + 1)]
        accepted = None
        for pos_index in rng.permutation(len(positions)):
            r0, c0 = positions[int(pos_index)]
            box = {(r0 + r, c0 + c) for r in range(spec["height"]) for c in range(spec["width"])}
            if box & (rail | guards | rays):
                continue
            tip = (r0 + spec["tip"][0], c0 + spec["tip"][1])
            q = spec["quantity"]
            ray = _ray(shape, tip, spec["direction"], q)
            room = len(_ray(shape, tip, spec["direction"], 30))
            mode = spec["mode"]
            if (mode == "free" and room <= q or mode == "exact" and room != q
                    or mode == "clip" and not 0 < room < q or mode == "zero" and room != 0):
                continue
            ray_guard = set(_ray(shape, tip, spec["direction"], q + 1))
            if ray_guard & (rail | boxes | rays):
                continue
            cavity = {(r0 + r, c0 + c) for r, c in spec["cavity"]}
            payload = _payload(rng, cavity, q, spec["pattern"])
            if payload is None:
                continue
            frame = {(r0 + r, c0 + c) for r, c in spec["frame"]}
            accepted = dict(spec, frame=frame, cavity=cavity, payload=payload,
                            box=box, tip=tip, ray=ray, available=room)
            break
        if accepted is None:
            if index < required:
                return None
            continue
        placed.append(accepted)
        boxes.update(accepted["box"])
        guards.update((r+dr, c+dc) for r, c in accepted["box"]
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        rays.update(_ray(shape, accepted["tip"], accepted["direction"], spec["quantity"] + 1))
    return placed


def _scene(rng, plans, rotation, *, teaching=False, first_test=False):
    for attempt in range(96):
        palette = list(map(int, rng.permutation(10)))
        bg, seed_color, accent = palette[:3]
        # Plans may pin vessel colors across the two count demonstrations.
        pinned = {c for plan in plans for c in plan.get("colors", ())}
        roles = [c for c in palette if c not in pinned]
        bg, seed_color, accent = roles[:3]
        colors = [c for c in palette if c not in (bg, seed_color, accent)]
        h = int(rng.integers(24, 31)) if teaching or first_test else int(rng.integers(14, 31))
        w = int(rng.integers(24, 31)) if teaching or first_test else int(rng.integers(14, 31))
        if attempt >= 48:
            h = w = 30
        length = int(rng.integers(2, 6 if teaching else 7))
        if teaching or first_test:
            start = int(rng.choice([0, 1, w - length - 1, w - length]))
        else:
            start = int(rng.integers(0, w - length + 1))
        swatches = [c for c in (start - 1, start + length) if 0 <= c < w]
        swatch = int(rng.choice(swatches))
        rail = {(r, c) for r in range(h) for c in range(start, start + length)} | {(0, swatch)}
        specs = [_spec(rng, colors, **plan) for plan in plans]
        extras = int(rng.integers(0, 3 if teaching else 4))
        for _ in range(extras):
            specs.append(_spec(rng, colors))
        placed = _pack(rng, (h, w), rail, specs, len(plans))
        if placed is None:
            continue
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[0, start:start + length] = seed_color
        inp[0, swatch] = accent
        out = inp.copy()
        for d in range(h):
            out[d, start:start + length] = bg if d % 2 else accent if d % 6 == 4 else seed_color
        protected = set(rail)
        for obj in placed:
            for p in obj["frame"]:
                inp[p] = out[p] = obj["frame_color"]
            for p in obj["payload"]:
                inp[p] = obj["payload_color"]
            # Direct construction uses ordered source cells, not solve().
            source = sorted(obj["payload"], key=lambda p: (p[0], p[1]))
            for destination, original in zip(obj["ray"], source):
                out[destination] = accent if original[0] % 6 == 4 else obj["payload_color"]
            protected.update((r+dr, c+dc) for r, c in obj["box"]
                             for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            protected.update(_ray((h, w), obj["tip"], obj["direction"], 30))
        # An inert bead on a special cross-section teaches hollow membership.
        special = [(r, c) for r in range(4, h - 1, 6) for c in range(1, w - 1)
                   if (r, c) not in protected]
        if teaching and special:
            p = special[int(rng.integers(len(special)))]
            inp[p] = out[p] = placed[0]["payload_color"]
            protected.update((p[0]+dr, p[1]+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        ordinary = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if (r, c) not in protected]
        for _ in range(int(rng.integers(3))):
            if not ordinary:
                break
            p = ordinary[int(rng.integers(len(ordinary)))]
            inp[p] = out[p] = int(rng.choice(colors))
            ordinary = [q for q in ordinary if max(abs(q[0]-p[0]), abs(q[1]-p[1])) > 1]
        if np.sum(inp == bg) * 2 <= inp.size:
            continue
        clock = dict(background=bg, seed=seed_color, accent=accent,
                     start=start, length=length, swatch=swatch)
        try:
            observed = _objects(inp, clock)
        except ValueError:
            continue
        expected = {(frozenset(o["frame"]), frozenset(o["payload"]), o["direction"]) for o in placed}
        actual = {(frozenset(o["frame"]), frozenset(o["payload"]), o["direction"]) for o in observed}
        if actual != expected:
            continue
        return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ns = _count(num_test, 1, 4, "num_test")
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    anchor_shape = _profile(rng, "anchor")
    anchor_colors = tuple(map(int, rng.choice(10, 2, replace=False)))
    anchor_direction = str(rng.choice(["R", "L"]))
    low, high = int(rng.integers(4, 6)), int(rng.integers(7, 10))
    base_rotation = int(rng.integers(4))
    common = dict(profile=anchor_shape, direction=anchor_direction, colors=anchor_colors, mode="free")
    plans = [
        [dict(common, quantity=low, pattern="low_a"),
         dict(kind="small", direction=str(rng.choice(["U", "D"])), quantity=2, mode="free")],
        [dict(common, quantity=low, pattern="low_b"), dict(common, quantity=high, pattern="high")],
    ]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            first_test = split == "test" and i == 0
            if split == "train" and i < 2:
                plan = plans[i]
                rotation = (base_rotation + i) % 4
            elif first_test:
                plan = [dict(kind="large", direction=str(rng.choice(["U", "D"])),
                             quantity=int(rng.integers(10, 15)), pattern="mixed", mode="free")]
                rotation = (base_rotation + 2) % 4
            else:
                plan = [dict()]
                rotation = int(rng.integers(4))
            for _ in range(16):
                pair = _scene(rng, plan, rotation, teaching=split == "train" and i < 2, first_test=first_test)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _geometry_signature(obj):
    cells = [_local(p, obj["direction"]) for p in obj["frame"]]
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return (tuple(sorted((r-r0, c-c0) for r, c in cells)), obj["direction"],
            obj["frame_color"], obj["payload_color"])


def _evidence(pairs):
    full_counts, matched, repeated = set(), {}, {}
    mixed_order = duplicate_special = inert_special = cell_counterexample = False
    max_q = 0
    for pair in pairs:
        g, k, clock = _clock(_grid(pair["input"]))
        out = np.rot90(_grid(pair["output"]), k)
        objects = _objects(g, clock)
        occupied = set().union(*(o["box"] for o in objects))
        for o in objects:
            q, word = len(o["payload"]), tuple(_word(o, clock["accent"]))
            max_q = max(max_q, q)
            if o["available"] > q:
                assert tuple(int(out[p]) for p in o["ray"]) == word
                full_counts.add(q)
                cell_counterexample |= len(_components(o["payload"])) < q
                signature = _geometry_signature(o)
                matched.setdefault(signature, set()).add(q)
                repeated.setdefault((signature, q), set()).add(word)
                bits = [c == clock["accent"] for c in word]
                mixed_order |= any(not bits[i] and bits[j] and not bits[t]
                                   for i in range(q) for j in range(i+1, q) for t in range(j+1, q))
                duplicate_special |= any(sum(p[0] == d for p in o["payload"]) >= 2
                                         for d in range(4, len(g), 6))
            inert_special |= any(g[r, c] == o["payload_color"] and out[r, c] == g[r, c]
                                 and (r, c) not in occupied
                                 for r in range(4, len(g), 6) for c in range(g.shape[1]))
    return dict(counts=full_counts, matched=any(len(v) >= 2 for v in matched.values()),
                rearranged=any(len(v) >= 2 for v in repeated.values()),
                mixed=mixed_order, duplicate=duplicate_special, inert=inert_special,
                cells=cell_counterexample, max_q=max_q)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            g, k, clock = _clock(inp)
            target = np.rot90(out, k)
            objects = _objects(g, clock)
            assert 1 <= len(objects) <= 4
            changed = set()
            for r in range(len(g)):
                color = clock["background"] if r % 2 else clock["accent"] if r % 6 == 4 else clock["seed"]
                for c in range(clock["start"], clock["start"] + clock["length"]):
                    assert target[r, c] == color, "clock phase or spacing changed"
                    changed.add((r, c))
            for o in objects:
                assert all(target[p] == g[p] for p in o["frame"]), "frame altered"
                assert all(target[p] == clock["background"] for p in o["payload"]), "payload not cleared"
                changed.update(o["payload"])
                word = _word(o, clock["accent"])
                assert len(o["ray"]) == min(len(word), o["available"])
                assert [int(target[p]) for p in o["ray"]] == word[:len(o["ray"])], "incorrect scan-order ray prefix"
                changed.update(o["ray"])
            assert all(target[r, c] == g[r, c] for r in range(len(g)) for c in range(g.shape[1])
                       if (r, c) not in changed), "unrelated cell changed"
            assert np.array_equal(solve(inp), out)
    evidence = _evidence(puzzle["train"])
    assert len(evidence["counts"]) >= 2 and evidence["matched"], "count evidence insufficient"
    assert all(evidence[k] for k in ("rearranged", "mixed", "duplicate", "inert", "cells")), "scan/ownership evidence insufficient"
    # At least one count-dependent test ray goes beyond every demonstrated count.
    test_full = []
    for pair in puzzle["test"]:
        g, _, clock = _clock(_grid(pair["input"]))
        test_full.extend(len(o["payload"]) for o in _objects(g, clock)
                         if o["available"] > len(o["payload"]))
    assert test_full and max(test_full) > evidence["max_q"], "missing visible quantity shift"


def self_test():
    """Bounded 42-seed checks, including semantic variation and equivariance."""
    shapes, profiles, quantities, counts, sides, directions, palettes = (set() for _ in range(7))
    signatures, normalized = set(), set()
    clearances, special_ranks, payload_components = set(), set(), set()
    same_row_probes = phase_probes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = _grid(pair["input"])
                shapes.add(inp.shape)
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                normalized.add((inp.shape, norm))
                g, k, clock = _clock(inp)
                sides.add(k)
                palettes.add((clock["background"], clock["seed"], clock["accent"]))
                objects = _objects(g, clock)
                counts.add(len(objects))
                for o in objects:
                    q, room = len(o["payload"]), o["available"]
                    profiles.add(_geometry_signature(o)[0])
                    quantities.add(q)
                    directions.add(o["direction"])
                    clearances.add("zero" if room == 0 else "clip" if room < q else "exact" if room == q else "free")
                    special_ranks.update(i for i, c in enumerate(_word(o, clock["accent"])) if c == clock["accent"])
                    payload_components.add(len(_components(o["payload"])))
        # Whole-scene rotation preserves both coordinate frames and the output.
        p = puzzle["test"][0]
        for k in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(p["input"], k)), np.rot90(p["output"], k))
        for flip in (np.fliplr, np.flipud):
            assert np.array_equal(solve(flip(p["input"])), flip(p["output"]))
        # Reassign every color, including background and visible accent.
        perm = np.random.default_rng(seed + 800).permutation(10)
        assert np.array_equal(solve(perm[p["input"]]), perm[p["output"]])
        # Move a bead sideways within one scan level: the serialized word stays.
        view, undo, clock = _clock(p["input"])
        for o in _objects(view, clock):
            moves = [(old, new) for old in sorted(o["payload"])
                     for new in sorted(o["cavity"] - o["payload"]) if old[0] == new[0]]
            if moves:
                old, new = moves[0]
                modified = view.copy()
                modified[old], modified[new] = clock["background"], o["payload_color"]
                assert np.array_equal(solve(np.rot90(modified, -undo)), p["output"])
                same_row_probes += 1
            # Move between ordinary and special source levels, holding count fixed.
            moves = [(old, new) for old in sorted(o["payload"]) if old[0] % 6 != 4
                     for new in sorted(o["cavity"] - o["payload"]) if new[0] % 6 == 4]
            if moves and o["available"] >= len(o["payload"]):
                old, new = moves[0]
                modified = view.copy()
                modified[old], modified[new] = clock["background"], o["payload_color"]
                assert not np.array_equal(solve(np.rot90(modified, -undo)), p["output"])
                phase_probes += 1
    assert len(signatures) == 42 and len(normalized) >= 190
    assert len(shapes) >= 45 and len(profiles) >= 50 and len(palettes) >= 35
    assert sides == set(range(4)) and directions == set(_DIRS)
    assert {1, 2, 3, 4} <= counts and {"zero", "clip", "exact", "free"} == clearances
    assert min(quantities) == 1 and max(quantities) >= 14
    assert len(special_ranks) >= 10 and max(payload_components) >= 4
    assert same_row_probes >= 20 and phase_probes >= 20
    # Every legal explicit count combination and representative malformed counts.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + nt * 5 + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(90393752, 2, 1), generate_puzzle(90393752, 2, 1)
    assert all(np.array_equal(x[r], y[r]) for s in a for x, y in zip(a[s], b[s]) for r in x)
    broken = {s: [{r: g.copy() for r, g in p.items()} for p in a[s]] for s in a}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts)")
