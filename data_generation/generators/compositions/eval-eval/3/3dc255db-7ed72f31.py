"""Symmetry purification: retain partnered tokens; export the surplus count.

The reader and solve use only the supplied grid. Construction paints its labels
from sampled token orbits, then compares them with the separate input reader.
Only Python's standard library and NumPy are used. Marker color 5 is fixed.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


MARKER = 5
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
KINDS = ("H", "V", "P")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells):
    rest = set(cells)
    result = []
    while rest:
        start = min(rest)
        rest.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in rest:
                        rest.remove(p)
                        part.add(p)
                        stack.append(p)
        result.append(part)
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    i = int(np.argmax(counts))
    if counts[i] <= grid.size // 2 or int(colors[i]) == MARKER:
        raise ValueError("The non-gray background must occupy more than half the input")
    return int(colors[i])


def _local(p, direction):
    r, c = p
    return ((r, c), (-c, r), (-r, -c), (c, -r))[direction]


def _world(p, direction):
    y, x = p
    return ((y, x), (x, -y), (-y, -x), (-x, y))[direction]


def _geometry(frame, direction):
    """Read a thin open envelope from cross-sections, not sampled parameters."""
    local = {_local(p, direction) for p in frame}
    y0, y1, _, _ = _bbox(local)
    rows = [sorted(x for y, x in local if y == level)
            for level in range(y0, y1 + 1)]
    if len(rows) < 4 or len(rows[0]) != 1 or any(not row for row in rows):
        return None
    tip_x = rows[0][0]
    left = right = tip_x
    split, cavity = False, set()
    parallel = 0
    for offset, row in enumerate(rows[1:], 1):
        if not split and row == [tip_x]:
            continue
        if len(row) != 2 or row[1] - row[0] < 2:
            return None
        lo, hi = row
        if not (0 <= left - lo <= 1 and 0 <= hi - right <= 1):
            return None
        if not split and (lo != tip_x - 1 or hi != tip_x + 1):
            return None
        parallel = parallel + 1 if (lo, hi) == (left, right) else 0
        left, right, split = lo, hi, True
        cavity.update(_world((y0 + offset, x), direction) for x in range(lo + 1, hi))
    if not split or parallel < 1 or not left < tip_x < right:
        return None
    return _world((y0, tip_x), direction), cavity


def _kind(marker):
    r0, r1, c0, c1 = _bbox(marker)
    if len(marker) == 1:
        return "P"
    if r0 == r1 and len(marker) == c1 - c0 + 1:
        return "H"
    if c0 == c1 and len(marker) == r1 - r0 + 1:
        return "V"
    raise ValueError("A marker must be a singleton or a contiguous cardinal line")


def _transform(p, kind, marker):
    r0, _, c0, _ = _bbox(marker)
    r, c = p
    if kind == "H":
        return 2 * r0 - r, c
    if kind == "V":
        return r, 2 * c0 - c
    return 2 * r0 - r, 2 * c0 - c


def _kernel(payload, kind, marker):
    return {p for p in payload if _transform(p, kind, marker) in payload}


def _ray(shape, tip, direction, quantity):
    dr, dc = DIRECTIONS[direction]
    result = []
    for step in range(1, quantity + 1):
        p = tip[0] + dr * step, tip[1] + dc * step
        if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
            break
        result.append(p)
    return tuple(result)


def _room(shape, tip, direction):
    return (tip[0], shape[1] - 1 - tip[1], shape[0] - 1 - tip[0], tip[1])[direction]


def _read(grid):
    bg = _background(grid)
    objects = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, MARKER}):
        for frame in _components(map(tuple, np.argwhere(grid == color).tolist())):
            if len(frame) < 7:
                continue
            for direction in range(4):
                geometry = _geometry(frame, direction)
                if geometry is None:
                    continue
                tip, cavity = geometry
                marker = {p for p in cavity if grid[p] == MARKER}
                if not marker or len(_components(marker)) != 1:
                    continue
                try:
                    kind = _kind(marker)
                except ValueError:
                    continue
                colors = {int(grid[p]) for p in cavity} - {bg, MARKER}
                if len(colors) != 1 or color in colors:
                    continue
                payload_color = next(iter(colors))
                payload = {p for p in cavity if grid[p] == payload_color}
                kept = _kernel(payload, kind, marker)
                surplus = payload - kept
                ray = _ray(grid.shape, tip, direction, len(surplus))
                if any(grid[p] != bg for p in ray):
                    continue
                r0, r1, c0, c1 = _bbox(frame)
                box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                objects.append(dict(frame=frame, cavity=cavity, marker=marker, kind=kind,
                    tip=tip, direction=direction, frame_color=color, color=payload_color,
                    payload=payload, kept=kept, surplus=surplus, q=len(surplus),
                    ray=ray, room=_room(grid.shape, tip, direction), box=box))
    # Incidental arrow-like token clusters cannot become nested vessels.
    objects = [o for o in objects if not any(o is not x and o["frame"] <= x["cavity"]
                                            for x in objects)]
    if not objects or len({frozenset(o["frame"]) for o in objects}) != len(objects):
        raise ValueError("No uniquely directed vessels")
    boxes, rays, covered = set(), set(), set()
    for o in objects:
        if boxes & o["box"] or rays & set(o["ray"]):
            raise ValueError("Vessels or exhausts overlap")
        boxes.update(o["box"])
        rays.update(o["ray"])
        covered.update(o["frame"] | o["marker"] | o["payload"])
    if boxes & rays or covered != set(map(tuple, np.argwhere(grid != bg).tolist())):
        raise ValueError("Unowned cells or exhaust entering a vessel")
    return bg, sorted(objects, key=lambda o: min(o["frame"]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    bg, objects = _read(inp)
    out = inp.copy()
    for o in objects:
        for p in o["surplus"]:
            out[p] = bg
        for p in o["ray"]:
            out[p] = o["color"]
    return out


def _shape(rng, large=False, test=False, direction=None):
    a = int(rng.integers(2, 4 if large else 5))
    b = a if rng.random() < .45 else int(rng.integers(2, 4 if large else 5))
    if test:
        a = b = int(rng.integers(3, 5))
    tail = int(rng.integers(7, 10) if test else rng.integers(5, 8) if large else rng.integers(2, 7))
    stem = int(rng.integers(0, 3))
    rows = [(0, 0)] * (stem + 1)
    left = right = 0
    # The first split moves both sides; later sides can expand independently.
    while left < a or right < b:
        dl = left < a and (left == 0 or right == b or rng.random() < .8)
        dr = right < b and (right == 0 or left == a or rng.random() < .8)
        if not dl and not dr:
            dl = left < a
            dr = not dl
        left, right = left + int(dl), right + int(dr)
        rows.append((-left, right))
    rows.extend([(-a, b)] * tail)
    frame, cavity = set(), set()
    for y, (lo, hi) in enumerate(rows):
        frame.update({(y, lo), (y, hi)})
        cavity.update((y, x) for x in range(lo + 1, hi))
    direction = int(rng.integers(4)) if direction is None else direction
    transformed = {_world(p, direction) for p in frame | cavity}
    r0, _, c0, _ = _bbox(transformed)
    def move(p):
        r, c = _world(p, direction)
        return r - r0, c - c0
    f = {move(p) for p in frame}
    _, r1, _, c1 = _bbox(f)
    return dict(frame=f, cavity={move(p) for p in cavity}, tip=move((0, 0)),
                direction=direction, h=r1 + 1, w=c1 + 1)


def _marker_candidates(cavity, kind):
    if kind == "P":
        return [{p} for p in sorted(cavity)]
    dr, dc = (0, 1) if kind == "H" else (1, 0)
    result = []
    for p in sorted(cavity):
        for length in range(2, 5):
            marker = {(p[0] + i * dr, p[1] + i * dc) for i in range(length)}
            if marker <= cavity:
                result.append(marker)
    return result


def _orbits(domain, kind, marker):
    pairs, outside, fixed = [], [], []
    for p in sorted(domain):
        image = _transform(p, kind, marker)
        if image == p:
            fixed.append(p)
        elif image not in domain:
            outside.append((p,))
        elif p < image:
            pairs.append((p, image))
    return pairs, outside, fixed


def _pattern(rng, shape, q, core_pairs=1, style="random", include_fixed=True, internal_only=False):
    pairs, outside, fixed = _orbits(shape["cavity"] - shape["marker"], shape["kind"], shape["marker"])
    if len(pairs) < core_pairs or len(pairs) + len(outside) < q + core_pairs:
        return None
    order = list(map(int, rng.permutation(len(pairs))))
    kept = {p for i in order[:core_pairs] for p in pairs[i]}
    if fixed and include_fixed:
        count = int(rng.integers(1, min(3, len(fixed)) + 1))
        kept.update(fixed[int(i)] for i in rng.choice(len(fixed), count, replace=False))
    options = [pairs[i] for i in order[core_pairs:]] + ([] if internal_only else outside)
    if len(options) < q:
        return None
    if style in ("adjacent", "scattered"):
        choices = []
        for i, orbit in enumerate(options):
            for p in orbit:
                for j in range(i + 1, len(options)):
                    for t in options[j]:
                        adjacent = max(abs(p[0] - t[0]), abs(p[1] - t[1])) == 1
                        if adjacent == (style == "adjacent"):
                            choices.append((p, t))
        if not choices or q != 2:
            return None
        surplus = set(choices[int(rng.integers(len(choices)))])
    else:
        selected = rng.choice(len(options), q, replace=False)
        surplus = {options[int(i)][int(rng.integers(len(options[int(i)])))] for i in selected}
    return kept, surplus


def _alternate_kernels(o):
    s, kind, marker = o["payload"], o["kind"], o["marker"]
    r0, r1, c0, c1 = _bbox(marker)
    center = {((r0 + r1) // 2, (c0 + c1) // 2)}
    alternatives = {k: _kernel(s, k, center) for k in KINDS if k != kind}
    transformed = {_transform(p, kind, marker) for p in s}
    sr0, _, sc0, _ = _bbox(s)
    tr0, _, tc0, _ = _bbox(transformed)
    translated = {(r + tr0 - sr0, c + tc0 - sc0) for r, c in s}
    alternatives["translate"] = s & translated
    alternatives["drop_fixed"] = {p for p in o["kept"] if _transform(p, kind, marker) != p}
    return alternatives


def _imbalance(o):
    marker, kind = o["marker"], o["kind"]
    r0, _, c0, _ = _bbox(marker)
    if kind == "H":
        return abs(sum(r < r0 for r, c in o["payload"]) - sum(r > r0 for r, c in o["payload"]))
    if kind == "V":
        return abs(sum(c < c0 for r, c in o["payload"]) - sum(c > c0 for r, c in o["payload"]))
    bins = Counter((int(np.sign(r - r0)), int(np.sign(c - c0))) for r, c in o["payload"])
    categories = set(bins) | {(-r, -c) for r, c in bins}
    return sum(abs(bins[p] - bins[(-p[0], -p[1])]) for p in categories if p < (-p[0], -p[1]))


def _discriminating(o):
    if not o["kept"] or not o["surplus"]:
        return False
    if not any(_transform(p, o["kind"], o["marker"]) in o["cavity"] - o["marker"]
               for p in o["surplus"]):
        return False
    alternatives = _alternate_kernels(o)
    required = [k for k in alternatives if k != "drop_fixed" or o["kind"] != "P"]
    return all(alternatives[k] != o["kept"] for k in required) and _imbalance(o) != o["q"]


def _prototype(rng, palette, kind, q, *, base=None, colors=None, style="random",
               witness=False, test=False, clearance=None, core_pairs=None):
    for attempt in range(160):
        ncore = (int(rng.integers(0, 4)) if core_pairs is None else core_pairs)
        if base is None:
            shape = _shape(rng, test=test)
            candidates = _marker_candidates(shape["cavity"], kind)
            if witness:
                eligible = []
                for candidate in candidates:
                    pairs, _, fixed = _orbits(shape["cavity"] - candidate, kind, candidate)
                    if len(pairs) >= q + ncore and (kind == "P" or fixed):
                        eligible.append(candidate)
                candidates = eligible
            if not candidates:
                continue
            marker = candidates[int(rng.integers(len(candidates)))]
            shape.update(marker=marker, kind=kind)
        else:
            shape = dict(base)
        pattern = _pattern(rng, shape, q, ncore, style, internal_only=witness)
        if pattern is None:
            continue
        kept, surplus = pattern
        if not kept and not surplus:
            continue
        o = dict(shape, payload=kept | surplus, kept=kept, surplus=surplus, q=q)
        if witness and not _discriminating(o):
            continue
        f, p = colors if colors is not None else map(int, rng.choice(palette, 2, replace=False))
        o.update(frame_color=f, color=p)
        o["clearance"] = clearance or ("any" if q == 0 else str(rng.choice(
            ["free", "free", "free", "exact", "clip", "zero"])))
        if q == 1 and o["clearance"] == "clip":
            o["clearance"] = "zero"
        return o
    return None


def _anchor(rng, palette, kind):
    colors = tuple(map(int, rng.choice(palette, 2, replace=False)))
    high = int(rng.integers(4, 6))
    for _ in range(100):
        base = _shape(rng, large=True)
        markers = _marker_candidates(base["cavity"], kind)
        rng.shuffle(markers)
        for marker in markers[:30]:
            base.update(marker=marker, kind=kind)
            pairs, outside, fixed = _orbits(base["cavity"] - marker, kind, marker)
            if len(pairs) < high + 2 or (kind != "P" and not fixed):
                continue
            a = _prototype(rng, palette, kind, 2, base=base, colors=colors,
                           style="adjacent", clearance="free", core_pairs=1)
            b = _prototype(rng, palette, kind, high, base=base, colors=colors,
                           witness=True, clearance="free", core_pairs=1)
            c = _prototype(rng, palette, kind, 2, base=base, colors=colors,
                           style="scattered", clearance="free", core_pairs=1)
            if a is not None and b is not None and c is not None:
                return a, b, c
    raise RuntimeError("Cannot create matched counting witnesses")


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _placements(o, height, width, rng, ordered=False):
    positions = []
    for r in range(height - o["h"] + 1):
        for c in range(width - o["w"] + 1):
            tip = o["tip"][0] + r, o["tip"][1] + c
            room = _room((height, width), tip, o["direction"])
            mode, q = o["clearance"], o["q"]
            if ((mode == "free" and room <= q) or (mode == "exact" and room != q)
                    or (mode == "clip" and not 0 < room < q) or (mode == "zero" and room != 0)):
                continue
            positions.append((r, c))
    if not ordered:
        rng.shuffle(positions)
    return positions


def _pack(rng, specs, height, width, required, ordered=False):
    blocked, placed = set(), []
    # Mandatory items precede optional extras; larger mandatory items go first.
    order = sorted(range(required), key=lambda i: -specs[i]["h"] * specs[i]["w"])
    order += list(range(required, len(specs)))
    for index in order:
        o = specs[index]
        accepted = None
        for r, c in _placements(o, height, width, rng, ordered):
            box = {(r + y, c + x) for y in range(o["h"]) for x in range(o["w"])}
            tip = o["tip"][0] + r, o["tip"][1] + c
            guard_quantity = o["q"] + (o["clearance"] == "free" and o["q"] > 0)
            ray_guard = set(_ray((height, width), tip, o["direction"], guard_quantity))
            if not (box | ray_guard) & blocked:
                accepted = r, c, tip, box | ray_guard
                break
        if accepted is None:
            if index < required:
                return None
            continue
        r, c, tip, footprint = accepted
        blocked.update(_halo(footprint))
        placed.append((o, r, c, tip))
    return placed


def _scene(rng, bg, palette, kinds, mandatory, test=False):
    specs = list(mandatory)
    required = len(specs)
    desired = max(required, int(rng.choice([1, 1, 2, 2, 3, 4])))
    for _ in range(desired - required):
        q = int(rng.integers(0, 10 if test else 7))
        spec = _prototype(rng, palette, str(rng.choice(kinds)), q)
        if spec is not None:
            specs.append(spec)
    if not specs:
        for _ in range(10):
            spec = _prototype(rng, palette, str(rng.choice(kinds)), int(rng.integers(1, 7)))
            if spec is not None:
                specs.append(spec)
                break
        if not specs:
            raise RuntimeError("Could not create a vessel")
    required = max(1, required)
    low = 12 if required <= 1 else 21
    height, width = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    placed = None
    for attempt in range(14):
        if attempt >= 5:
            height = width = 30
        placed = _pack(rng, specs, height, width, required, ordered=attempt == 13)
        if placed is not None:
            break
    if placed is None:
        return None
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    for o, r, c, tip in placed:
        for y, x in o["frame"]:
            inp[r + y, c + x] = out[r + y, c + x] = o["frame_color"]
        for y, x in o["marker"]:
            inp[r + y, c + x] = out[r + y, c + x] = MARKER
        for y, x in o["payload"]:
            inp[r + y, c + x] = o["color"]
        for y, x in o["kept"]:
            out[r + y, c + x] = o["color"]
        for p in _ray(inp.shape, tip, o["direction"], o["q"]):
            out[p] = o["color"]
    try:
        predicted = solve(inp)
    except ValueError:
        return None
    assert np.array_equal(out, predicted), "Orbit construction disagrees with input reader"
    return {"input": inp, "output": out}


def _count(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= value <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    bg = 0 if rng.random() < .55 else int(rng.choice([c for c in range(10) if c != MARKER]))
    pool = [c for c in range(10) if c not in (bg, MARKER)]
    palette = tuple(map(int, rng.choice(pool, int(rng.integers(2, 9)), replace=False)))
    subsets = [("H",), ("V",), ("P",), ("H", "V"), ("H", "P"), ("V", "P"), KINDS]
    kinds = subsets[int(rng.integers(len(subsets)))]
    anchor_kind = str(rng.choice(kinds))
    low, high, repeat = _anchor(rng, palette, anchor_kind)
    plans = [[] for _ in range(num_train)]
    i, j = map(int, rng.choice(num_train, 2, replace=False))
    plans[i].append(low)
    plans[j].extend([high, repeat])
    balanced = _prototype(rng, palette, anchor_kind, 0, core_pairs=2, clearance="any")
    if balanced is None:
        raise RuntimeError("Could not make a balanced vessel")
    plans[i].append(balanced)
    for kind in kinds:
        if kind != anchor_kind:
            witness = _prototype(rng, palette, kind, 2, witness=True, clearance="free", core_pairs=1)
            if witness is None:
                raise RuntimeError("Could not demonstrate a symmetry kind")
            destination = min(range(num_train), key=lambda k: len(plans[k]))
            plans[destination].append(witness)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            mandatory = plans[index] if split == "train" else []
            if split == "test" and index == 0:
                primary = _prototype(rng, palette, str(rng.choice(kinds)), int(rng.integers(7, 10)),
                                     witness=True, test=True, clearance="free", core_pairs=1)
                if primary is None:
                    raise RuntimeError("Could not construct a count-generalization test")
                mandatory = [primary]
            for _ in range(32):
                pair = _scene(rng, bg, palette, kinds, mandatory, test=split == "test")
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("Could not pack distinct readable examples")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(o):
    r0, _, c0, _ = _bbox(o["frame"])
    return (tuple(sorted((r-r0, c-c0) for r, c in o["frame"])),
            tuple(sorted((r-r0, c-c0) for r, c in o["marker"])),
            o["direction"], o["frame_color"], o["color"])


def _counterfactual(inp, objects, mode, bg):
    out = inp.copy()
    for o in objects:
        kept = set(o["kept"])
        if mode == "all_tokens":
            kept = set()
        for p in o["payload"] - kept:
            out[p] = bg
        if mode != "no_ray":
            q = len(o["payload"] - kept)
            for p in _ray(inp.shape, o["tip"], o["direction"], q):
                out[p] = o["color"]
    return out


def _training_evidence(puzzle):
    matched, repeated, discriminating = {}, {}, set()
    balanced, component_witness = False, False
    kinds, max_q = set(), 0
    for pair in puzzle["train"]:
        inp, out = _grid(pair["input"]), _grid(pair["output"])
        bg, objects = _read(inp)
        for o in objects:
            kinds.add(o["kind"])
            max_q = max(max_q, o["q"])
            if o["q"] == 0 and len(o["kept"]) >= 2:
                balanced = True
            if _discriminating(o) and o["room"] > o["q"]:
                discriminating.add(o["kind"])
            if o["q"] > 0 and o["room"] > o["q"]:
                matched.setdefault(_signature(o), set()).add(o["q"])
                count = len(_components(o["surplus"]))
                repeated.setdefault((_signature(o), o["q"]), set()).add(count)
                component_witness |= count < o["q"]
                assert out[_ray(inp.shape, o["tip"], o["direction"], o["q"] + 1)[-1]] == bg
    assert any(len(v) >= 2 for v in matched.values()), "No isolated counting demonstration"
    assert any(len(v) >= 2 for v in repeated.values()), "No matched count with different connectivity"
    assert component_witness and balanced, "Missing component-count or zero-surplus witness"
    assert discriminating == kinds, "Training does not distinguish all symmetry actions and side counts"
    for mode in ("all_tokens", "no_ray"):
        assert any(not np.array_equal(_counterfactual(_grid(p["input"]), _read(_grid(p["input"]))[1],
                                                    mode, _background(_grid(p["input"]))), p["output"])
                   for p in puzzle["train"]), f"Training ignores a parent: {mode}"
    return kinds, max_q


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, backgrounds = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            bg, objects = _read(inp)
            backgrounds.add(bg)
            assert np.array_equal(solve(inp), out), "Incorrect purification or exhaust"
            removals, additions = 0, 0
            for o in objects:
                assert all(out[p] == inp[p] for p in o["frame"] | o["marker"])
                retained = {p for p in o["cavity"] if out[p] == o["color"]}
                assert retained == o["kept"]
                assert {_transform(p, o["kind"], o["marker"]) for p in retained} == retained
                assert all(out[p] == bg for p in o["surplus"])
                assert all(out[p] == o["color"] for p in o["ray"])
                assert len(o["ray"]) == min(o["q"], o["room"])
                removals += o["q"]
                additions += len(o["ray"])
                if split == "train":
                    assert o["q"] <= 6
            assert np.count_nonzero((inp != bg) & (out == bg)) == removals
            assert np.count_nonzero((inp == bg) & (out != bg)) == additions
            assert np.count_nonzero(inp != out) == removals + additions
    assert len(backgrounds) == 1
    kinds, largest = _training_evidence(puzzle)
    for p in puzzle["test"]:
        assert {o["kind"] for o in _read(_grid(p["input"]))[1]} <= kinds
    first = puzzle["test"][0]
    inp = _grid(first["input"])
    bg, objects = _read(inp)
    assert any(o["q"] > largest and o["room"] > o["q"] and o["kept"] for o in objects)
    for mode in ("all_tokens", "no_ray"):
        assert not np.array_equal(_counterfactual(inp, objects, mode, bg), first["output"])


def _fingerprint(puzzle):
    return tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    shapes, profiles, markers, quantities, directions, counts, colors = [set() for _ in range(7)]
    fingerprints, subsets, modes, fixed, external, offset = set(), set(), set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        observed_kinds = set()
        for s in ("train", "test"):
            for pair in puzzle[s]:
                inp = _grid(pair["input"])
                before = inp.copy()
                assert np.array_equal(solve(inp), pair["output"]) and np.array_equal(inp, before)
                # Rotation and recoloring retain the rule; gray keeps its fixed role.
                turned = np.rot90(inp, seed % 4)
                assert np.array_equal(solve(turned), np.rot90(pair["output"], seed % 4))
                permutation = np.arange(10, dtype=np.int8)
                non_gray = [c for c in range(10) if c != MARKER]
                permutation[non_gray] = np.roll(non_gray, seed % 9)
                assert np.array_equal(solve(permutation[turned]),
                                      permutation[np.rot90(pair["output"], seed % 4)])
                bg, objects = _read(inp)
                shapes.add(inp.shape)
                colors.add(bg)
                counts.add(len(objects))
                for o in objects:
                    observed_kinds.add(o["kind"])
                    profiles.add(_signature(o)[0])
                    markers.add((o["kind"], len(o["marker"])))
                    quantities.add(o["q"])
                    directions.add(o["direction"])
                    modes.add("zero" if o["q"] and o["room"] == 0 else "clip" if o["q"] > o["room"]
                              else "exact" if o["q"] and o["q"] == o["room"] else "free")
                    fixed += any(_transform(p, o["kind"], o["marker"]) == p for p in o["kept"])
                    external += any(_transform(p, o["kind"], o["marker"]) not in o["cavity"]
                                    for p in o["surplus"])
                    r0, r1, c0, c1 = _bbox(o["frame"])
                    offset += ((o["tip"][1] * 2 != c0 + c1) if o["direction"] % 2 == 0
                               else (o["tip"][0] * 2 != r0 + r1))
        subsets.add(tuple(sorted(observed_kinds)))
    assert len(fingerprints) == 42 and len(shapes) >= 35 and len(profiles) >= 60
    assert directions == set(range(4)) and quantities >= set(range(10))
    assert counts >= {1, 2, 3} and len(colors) >= 5 and len(subsets) == 7
    assert modes == {"free", "exact", "clip", "zero"} and fixed and external and offset
    assert {k for k, length in markers} == set(KINDS) and max(length for k, length in markers) == 4
    assert _fingerprint(generate_puzzle(91)) == _fingerprint(generate_puzzle(91))
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(42000 + n * 10 + m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Hand-written coordinate regressions for all three involutions.
    s = {(3, 3), (3, 5), (4, 2), (5, 3)}
    assert _kernel(s, "V", {(2, 4), (3, 4)}) == {(3, 3), (3, 5)}
    s = {(3, 3), (5, 5), (3, 5), (4, 3)}
    assert _kernel(s, "P", {(4, 4)}) == {(3, 3), (5, 5)}
    s = {(3, 3), (3, 5), (4, 2), (6, 6)}
    o = dict(payload=s, kind="V", marker={(2, 4), (3, 4)})
    assert _imbalance(o) == 0 and len(s - _kernel(s, "V", o["marker"])) == 2
    assert _kernel({(2, 1), (4, 1), (3, 5)}, "H", {(3, 2), (3, 3)}) == {(2, 1), (4, 1), (3, 5)}
    assert _ray((7, 8), (0, 4), 0, 9) == ()
    assert _ray((7, 8), (2, 4), 0, 9) == ((1, 4), (0, 4))
    assert _ray((7, 8), (2, 4), 0, 0) == ()
    # Complete hand-painted vessels, including an axis-fixed token beyond an
    # endpoint, a point-centered pair, exact clipping, and a boundary tip.
    frame = {(2, 3), (3, 2), (3, 4)} | {(r, c) for r in range(4, 8) for c in (1, 5)}
    fixtures = [
        ({(5, 3)}, {(4, 2), (6, 4), (4, 4), (3, 3)}, {(4, 2), (6, 4)}, 2),
        ({(5, 2), (5, 3)}, {(4, 2), (6, 2), (5, 4), (4, 4)}, {(4, 2), (6, 2), (5, 4)}, 1),
        ({(4, 3), (5, 3)}, {(4, 2), (4, 4), (6, 3), (5, 2)}, {(4, 2), (4, 4), (6, 3)}, 1),
    ]
    for marker, tokens, retained, q in fixtures:
        inp, target = np.zeros((9, 7), np.int8), np.zeros((9, 7), np.int8)
        for p in frame:
            inp[p] = target[p] = 2
        for p in marker:
            inp[p] = target[p] = MARKER
        for p in tokens:
            inp[p] = 8
        for p in retained:
            target[p] = 8
        for r in range(2 - q, 2):
            target[r, 3] = 8
        assert np.array_equal(solve(inp), target)
        # Crop the two rows before the tip. Erasure still occurs with no exhaust.
        boundary_target = target[2:].copy()
        assert np.array_equal(solve(inp[2:]), boundary_target)
    # Mutated interior labels cannot slip through a shape/schema-only check.
    p = generate_puzzle(101, 2, 1)
    bad = {s: [{k: _grid(v) for k, v in pair.items()} for pair in p[s]] for s in p}
    pair = bad["train"][0]
    bg, objects = _read(pair["input"])
    cell = next(iter(objects[0]["kept"]))
    pair["output"][cell] = bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Accepted destruction of the symmetric core")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
