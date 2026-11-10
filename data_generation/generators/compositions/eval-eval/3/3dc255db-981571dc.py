"""Pointed canisters export the number of damaged occupied mirror orbits.

Only standard-library modules and NumPy are used. Generation paints its labels
from sampled clean patterns; solve recognizes frames and reads input alone.
No function writes files, caches puzzles, imports parents, or reads evidence.
"""
from __future__ import annotations

from hashlib import sha256
import numpy as np


_EMPTY = 0
_ERASE = 5
_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_DIRECTIONS = ((-1, 0), (0, -1), (1, 0), (0, 1))


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in left:
                        left.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(component)
    return result


def _up(point, direction):
    r, c = point
    return ((r, c), (c, -r), (-r, -c), (-c, r))[direction]


def _world(point, direction):
    y, x = point
    return ((y, x), (-x, y), (-y, -x), (x, -y))[direction]


def _rotate(point, height, width, turns):
    r, c = point
    for _ in range(turns):
        r, c, height, width = width - 1 - c, r, width, height
    return r, c


def _ray(shape, tip, direction, quantity):
    dr, dc = _DIRECTIONS[direction]
    r, c = tip
    result = []
    for _ in range(quantity):
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            break
        result.append((r, c))
    return tuple(result)


def _orbits(height, width):
    return [tuple(sorted({(r, c), (height - 1 - r, c),
                          (r, width - 1 - c), (height - 1 - r, width - 1 - c)}))
            for r in range((height + 1) // 2)
            for c in range((width + 1) // 2)]


def _read_frame(frame, direction):
    """Read full chamber bars and variable monotone prow boundary walks."""
    local = {_up(p, direction) for p in frame}
    ymin, ymax = min(y for y, x in local), max(y for y, x in local)
    xmin, xmax = min(x for y, x in local), max(x for y, x in local)
    width = xmax - xmin + 1
    if not 5 <= width <= 11:
        return None
    rows = [sorted(x - xmin for y, x in local if y == level)
            for level in range(ymin, ymax + 1)]
    full = list(range(width))
    bars = [i for i, row in enumerate(rows) if row == full]
    if len(bars) != 2 or bars[-1] != len(rows) - 1:
        return None
    top, bottom = bars
    if top < 2 or not 3 <= bottom - top - 1 <= 9 or len(rows[0]) != 1:
        return None
    tip_x = rows[0][0]
    if not 0 < tip_x < width - 1:
        return None
    if any(row != [0, width - 1] for row in rows[top + 1:bottom]):
        return None
    lo = hi = tip_x
    split = False
    for row in rows[:top]:
        if not row or len(row) > 2:
            return None
        left, right = row[0], row[-1]
        if not (0 <= lo - left <= 1 and 0 <= right - hi <= 1):
            return None
        if len(row) == 1 and (split or left != tip_x):
            return None
        split |= len(row) == 2
        lo, hi = left, right
    if (lo, hi) != (0, width - 1):
        return None
    inner = [[_world((ymin + y, xmin + x), direction)
              for x in range(1, width - 1)] for y in range(top + 1, bottom)]
    tip = _world((ymin, xmin + tip_x), direction)
    cavity = {p for row in inner for p in row}
    rs, cs = zip(*frame)
    box = {(r, c) for r in range(min(rs), max(rs) + 1)
           for c in range(min(cs), max(cs) + 1)}
    signature = tuple(tuple(row) for row in rows)
    return dict(tip=tip, inner=inner, cavity=cavity, box=box,
                profile=signature, direction=direction, frame=frame)


def _read_tank(tank):
    payloads = set(int(c) for c in np.unique(tank)) - {_EMPTY, _ERASE}
    if len(payloads) != 1:
        raise ValueError("a chamber needs one surviving payload color")
    color = next(iter(payloads))
    jobs = damaged_empty = occupied = repaired_active = diagonal_only = 0
    triple = diagonal_pair = axis_damage = False
    orbit_rows = []
    for orbit in _orbits(*tank.shape):
        visible = {int(tank[p]) for p in orbit if tank[p] != _ERASE}
        if len(visible) != 1:
            raise ValueError("empty or conflicting witness set")
        value = next(iter(visible))
        holes = {p for p in orbit if tank[p] == _ERASE}
        occupied += value == color
        if value == color and holes:
            jobs += 1
            repaired_active += len(holes)
            triple |= len(orbit) == 4 and len(holes) == 3
            diagonal_pair |= (len(orbit) == 4 and len(holes) == 2
                              and len({p[0] for p in holes}) == 2
                              and len({p[1] for p in holes}) == 2)
            axis_damage |= len(orbit) == 2
            h, w = tank.shape
            for r, c in holes:
                diagonal_only += (tank[h - 1 - r, c] == _ERASE
                                  and tank[r, w - 1 - c] == _ERASE
                                  and tank[h - 1 - r, w - 1 - c] != _ERASE)
        damaged_empty += value == _EMPTY and bool(holes)
        orbit_rows.append((orbit, value, len(holes)))
    return dict(quantity=jobs, payload_color=color, occupied_orbits=occupied,
                damaged_empty=damaged_empty, repaired_active=repaired_active,
                triple=triple, diagonal_pair=diagonal_pair, axis_damage=axis_damage,
                diagonal_only=int(diagonal_only), holes=int(np.sum(tank == _ERASE)),
                visible=int(np.sum(tank == color)),
                components=len(_components(map(tuple, np.argwhere(tank == color)))),
                orbit_rows=orbit_rows, tank_shape=tank.shape)


def _objects(grid):
    found = []
    for color in _COLORS:
        for component in _components(map(tuple, np.argwhere(grid == color))):
            if len(component) < 16:
                continue
            for direction in range(4):
                obj = _read_frame(component, direction)
                if obj is not None:
                    found.append(dict(obj, frame_color=color))
    # A payload can accidentally form a small canister-like component.
    found = [o for o in found if not any(o is not p and o["frame"] <= p["cavity"] for p in found)]
    if not found or len({frozenset(o["frame"]) for o in found}) != len(found):
        raise ValueError("missing or ambiguous directional canister")
    found.sort(key=lambda o: min(o["frame"]))
    occupied, rays = set(), set()
    for obj in found:
        tank = np.array([[grid[p] for p in row] for row in obj["inner"]], dtype=int)
        if obj["frame_color"] in tank:
            raise ValueError("frame color inside its own chamber")
        obj.update(_read_tank(tank))
        obj["tank"] = tank
        obj["ray"] = _ray(grid.shape, obj["tip"], obj["direction"], obj["quantity"])
        obj["clearance"] = len(_ray(grid.shape, obj["tip"], obj["direction"], 30))
        if not occupied.isdisjoint(obj["box"]) or not rays.isdisjoint(obj["ray"]):
            raise ValueError("overlapping objects or output rays")
        occupied.update(obj["box"])
        rays.update(obj["ray"])
    if not occupied.isdisjoint(rays) or any(grid[p] != 0 for p in rays):
        raise ValueError("output corridor is obstructed")
    explained = {p for o in found for p in o["frame"] | o["cavity"]}
    if any(tuple(p) not in explained for p in np.argwhere(grid != 0)):
        raise ValueError("unexplained exterior mark")
    return found


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only chamber borders, pointed geometry, and visible orbits."""
    grid = _grid(input_grid)
    output = grid.copy()
    for obj in _objects(grid):
        for p in obj["cavity"]:
            output[p] = _EMPTY
        for p in obj["ray"]:
            output[p] = obj["payload_color"]
    return output


def _profile(rng, height, width):
    outer = width + 2
    center = (outer - 1) // 2
    tip = int(np.clip(center + rng.integers(-1, 2), 1, outer - 2))
    stem = int(rng.integers(3))
    rows = [{tip} for _ in range(stem + 1)]
    left = right = tip
    budget = max(tip, outer - 1 - tip) + 2
    for step in range(budget):
        if left == 0 and right == outer - 1:
            break
        remaining = budget - step
        dl = left > 0 and (left == remaining or rng.random() < .85)
        dr = right < outer - 1 and (outer - 1 - right == remaining or rng.random() < .85)
        if not dl and not dr:
            dl = left > 0
            dr = not dl
        left, right = left - int(dl), right + int(dr)
        rows.append({left, right})
    assert left == 0 and right == outer - 1
    body_top = len(rows)
    frame = {(r, c) for r, row in enumerate(rows) for c in row}
    frame.update((body_top, c) for c in range(outer))
    frame.update((body_top + height + 1, c) for c in range(outer))
    frame.update((r, c) for r in range(body_top + 1, body_top + height + 1)
                 for c in (0, outer - 1))
    cavity = {(body_top + 1 + r, 1 + c) for r in range(height) for c in range(width)}
    return dict(frame=frame, cavity=cavity, body_top=body_top,
                tip=(0, tip), shape=(body_top + height + 2, outer), inner=(height, width))


def _pattern_once(rng, height, width, quantity, require_axis, hard):
    orbits = _orbits(height, width)
    four = [i for i, o in enumerate(orbits) if len(o) == 4]
    four = [int(i) for i in rng.permutation(four)]
    damaged = four[:2]
    blank = four[2]
    two = [i for i, o in enumerate(orbits) if len(o) == 2]
    if require_axis:
        damaged.append(int(rng.choice(two)))
    candidates = [i for i, o in enumerate(orbits) if len(o) > 1 and i not in damaged + [blank]]
    style = int(rng.integers(4))
    anchor_r, anchor_c = int(rng.integers(height)), int(rng.integers(width))
    priorities = {}
    for i in candidates:
        r, c = orbits[i][0]
        priorities[i] = (rng.random() if style == 0 else
                         abs(r - anchor_r) + abs(c - anchor_c) + rng.random() if style == 1 else
                         r + rng.random() if style == 2 else c + rng.random())
    candidates.sort(key=lambda i: priorities[i])
    damaged += candidates[:quantity - len(damaged)]
    remaining = [i for i in range(len(orbits)) if i not in damaged + [blank]]
    intact_count = int(rng.integers(1, min(4, len(remaining)) + 1))
    intact = [int(i) for i in rng.choice(remaining, intact_count, replace=False)]
    clean = np.zeros((height, width), dtype=np.int8)
    mask = np.zeros_like(clean, dtype=bool)
    for i in damaged + intact:
        for p in orbits[i]:
            clean[p] = 1
    for index, i in enumerate(damaged):
        orbit = orbits[i]
        if index == 0:
            survivor = int(rng.integers(4))
            holes = [p for j, p in enumerate(orbit) if j != survivor]
        elif index == 1:
            holes = [orbit[0], orbit[3]] if rng.random() < .5 else [orbit[1], orbit[2]]
        else:
            count = len(orbit) - 1 if hard and rng.random() < .65 else int(rng.integers(1, len(orbit)))
            holes = [orbit[int(j)] for j in rng.choice(len(orbit), count, replace=False)]
        for p in holes:
            mask[p] = True
    empty = [i for i, o in enumerate(orbits) if len(o) > 1 and i not in damaged + intact + [blank]]
    blank_orbits = [blank] + [int(i) for i in rng.choice(empty, min(len(empty), int(rng.integers(3))), replace=False)]
    for i in blank_orbits:
        orbit = orbits[i]
        for j in rng.choice(len(orbit), int(rng.integers(1, len(orbit))), replace=False):
            mask[orbit[int(j)]] = True
    # Direct label arithmetic from the independently sampled clean field/mask.
    expected = sum(bool(any(mask[p] for p in o)) and clean[o[0]] == 1 for o in orbits)
    assert expected == quantity
    return clean, mask


def _pattern(rng, height, width, quantity, require_axis=False, hard=False, component_witness=False):
    for _ in range(64):
        clean, mask = _pattern_once(rng, height, width, quantity, require_axis, hard)
        if (not component_witness or
                len(_components(map(tuple, np.argwhere((clean == 1) & ~mask)))) != quantity):
            return clean, mask
    raise RuntimeError("could not construct the component-count counterexample")


def _dimensions(rng, quantity, lo=3, hi=7):
    choices = []
    for height in range(lo, hi + 1):
        for width in range(lo, hi + 1):
            orbits = _orbits(height, width)
            if (sum(len(o) == 4 for o in orbits) >= 3
                    and sum(len(o) > 1 for o in orbits) >= quantity + 1
                    and len(orbits) >= quantity + 2):
                choices.append((height, width))
    return choices[int(rng.integers(len(choices)))]


def _spec(rng, profile, direction, quantity, colors, mode="free", require_axis=False,
          hard=False, component_witness=False):
    frame_color, payload_color = colors
    clean, mask = _pattern(rng, *profile["inner"], quantity, require_axis, hard, component_witness)
    inp = np.zeros(profile["shape"], dtype=np.int8)
    for p in profile["frame"]:
        inp[p] = frame_color
    top = profile["body_top"] + 1
    tank = np.where(clean, payload_color, 0).astype(np.int8)
    tank[mask] = _ERASE
    inp[top:top + tank.shape[0], 1:1 + tank.shape[1]] = tank
    out = np.zeros_like(inp)
    for p in profile["frame"]:
        out[p] = frame_color
    tip = _rotate(profile["tip"], *inp.shape, direction)
    return dict(input=np.rot90(inp, direction).copy(), output=np.rot90(out, direction).copy(),
                tip=tip, direction=direction, quantity=quantity, payload_color=payload_color, mode=mode)


def _positions(spec, shape, rng):
    h, w = spec["input"].shape
    candidates = []
    for r in range(shape[0] - h + 1):
        for c in range(shape[1] - w + 1):
            tip = (spec["tip"][0] + r, spec["tip"][1] + c)
            room = (tip[0], tip[1], shape[0] - 1 - tip[0], shape[1] - 1 - tip[1])[spec["direction"]]
            q, mode = spec["quantity"], spec["mode"]
            if ((mode == "free" and room > q) or (mode == "exact" and room == q)
                    or (mode == "clip" and 0 < room < q) or (mode == "zero" and room == 0)):
                candidates.append((r, c))
    return [candidates[int(i)] for i in rng.permutation(len(candidates))]


def _scene(rng, specs):
    primary = specs[0]
    h, w = primary["input"].shape
    vertical = primary["direction"] % 2 == 0
    min_h = h + (primary["quantity"] + 1 if vertical else 0)
    min_w = w + (0 if vertical else primary["quantity"] + 1)
    assert max(min_h, min_w) <= 30
    shape = (int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31)))
    inp = np.zeros(shape, dtype=np.int8)
    out = np.zeros_like(inp)
    boxes, guards, ray_guards = set(), set(), set()
    for index, spec in enumerate(specs):
        accepted = None
        for r, c in _positions(spec, shape, rng):
            sh, sw = spec["input"].shape
            box = {(r + y, c + x) for y in range(sh) for x in range(sw)}
            tip = (r + spec["tip"][0], c + spec["tip"][1])
            ray_guard = set(_ray(shape, tip, spec["direction"], spec["quantity"] + 1))
            if box.isdisjoint(guards | ray_guards) and ray_guard.isdisjoint(boxes | ray_guards):
                accepted = r, c, tip, box, ray_guard
                break
        if accepted is None:
            assert index > 0, "a primary free ray must fit by construction"
            continue
        r, c, tip, box, ray_guard = accepted
        sh, sw = spec["input"].shape
        inp[r:r + sh, c:c + sw] = spec["input"]
        out[r:r + sh, c:c + sw] = spec["output"]
        for p in _ray(shape, tip, spec["direction"], spec["quantity"]):
            out[p] = spec["payload_color"]
        boxes.update(box)
        guards.update((y + dy, x + dx) for y, x in box for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        ray_guards.update(ray_guard)
    assert np.array_equal(solve(inp), out), "construction and input-only reading disagree"
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    directions = [int(k) for k in rng.permutation(4)]
    train_dirs = directions[:1] if regime == 2 else directions[:2]
    test_dirs = directions[1:2] if regime == 2 else train_dirs
    anchor_shape = _profile(rng, int(rng.choice([5, 7])), int(rng.choice([5, 7])))
    anchor_colors = tuple(int(c) for c in rng.choice(_COLORS, 2, replace=False))
    high_quantity = int(rng.integers(3, 5))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            anchor = split == "train" and index < 2
            for _ in range(32):
                quantity = (2 if index == 0 else high_quantity) if anchor else int(rng.integers(5, 8) if split == "test" and regime == 0 else rng.integers(2, 5))
                if anchor:
                    profile, direction, colors = anchor_shape, train_dirs[0], anchor_colors
                else:
                    lo, hi = (8, 9) if split == "test" and regime == 1 else ((3, 9) if split == "test" and regime == 0 else (3, 7))
                    profile = _profile(rng, *_dimensions(rng, quantity, lo, hi))
                    direction = int(rng.choice(test_dirs if split == "test" else train_dirs))
                    colors = tuple(int(c) for c in rng.choice(_COLORS, 2, replace=False))
                specs = [_spec(rng, profile, direction, quantity, colors,
                               require_axis=anchor and index == 1,
                               hard=split == "test", component_witness=anchor and index == 0)]
                for _extra in range(int(rng.choice([0, 0, 1, 1, 2]))):
                    q = int(rng.integers(5, 8) if split == "test" and regime == 0 else rng.integers(2, 5))
                    lo, hi = (8, 9) if split == "test" and regime == 1 else ((3, 9) if split == "test" and regime == 0 else (3, 7))
                    p = _profile(rng, *_dimensions(rng, q, lo, hi))
                    d = int(rng.choice(test_dirs if split == "test" else train_dirs))
                    cs = tuple(int(c) for c in rng.choice(_COLORS, 2, replace=False))
                    mode = str(rng.choice(["free", "free", "exact", "clip", "zero"]))
                    specs.append(_spec(rng, p, d, q, cs, mode, hard=split == "test"))
                pair = _scene(rng, specs)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(objects):
    matched, counts = {}, set()
    component_witness = axis_witness = False
    for obj in objects:
        if obj["clearance"] > obj["quantity"]:
            counts.add(obj["quantity"])
            key = (obj["profile"], obj["direction"], obj["frame_color"], obj["payload_color"])
            matched.setdefault(key, set()).add(obj["quantity"])
            component_witness |= obj["components"] != obj["quantity"]
        axis_witness |= obj["axis_damage"]
    assert len(counts) >= 2, "training permits a constant ray length"
    assert any(len(qs) >= 2 for qs in matched.values()), "no matched-geometry count contrast"
    assert component_witness and axis_witness, "missing component/axis counterexample"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = (inp.shape, inp.astype(np.int8).tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            objects = _objects(inp)
            assert 1 <= len(objects) <= 3
            assert any(o["clearance"] > o["quantity"] for o in objects), "all counts clipped away"
            edited = set()
            for obj in objects:
                q, pc = obj["quantity"], obj["payload_color"]
                assert 2 <= q <= 7 and obj["damaged_empty"] > 0
                assert obj["occupied_orbits"] > q and obj["visible"] > q
                assert obj["repaired_active"] > q and obj["holes"] > q
                assert obj["triple"] and obj["diagonal_pair"] and obj["diagonal_only"] > 0
                assert all(out[p] == 0 for p in obj["cavity"])
                assert all(out[p] == pc for p in obj["ray"])
                assert len(obj["ray"]) == min(q, obj["clearance"])
                assert all(out[p] == inp[p] for p in obj["frame"])
                if obj["clearance"] > q:
                    assert out[_ray(inp.shape, obj["tip"], obj["direction"], q + 1)[-1]] == 0
                edited.update(obj["cavity"] | set(obj["ray"]))
            assert all(out[r, c] == inp[r, c] for r in range(inp.shape[0])
                       for c in range(inp.shape[1]) if (r, c) not in edited)
            assert np.array_equal(solve(inp), out)
            if split == "train":
                training.extend(objects)
    _training_evidence(training)


def _fingerprint(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain hashes and small feature sets, never grid corpora."""
    hashes, shapes, profiles, tanks, palettes = set(), set(), set(), set(), set()
    directions, quantities, parities, counts, clearances, damaged_sizes = set(), set(), set(), set(), set(), set()
    sample = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        hashes.add(_fingerprint(puzzle))
        if seed == 0:
            sample = puzzle
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                objects = _objects(inp)
                shapes.add(inp.shape)
                counts.add(len(objects))
                assert np.array_equal(solve(inp.tolist()), out)
                for obj in objects:
                    profiles.add(obj["profile"])
                    tanks.add(obj["tank_shape"])
                    directions.add(obj["direction"])
                    quantities.add(obj["quantity"])
                    palettes.add((obj["frame_color"], obj["payload_color"]))
                    parities.add(tuple(d % 2 for d in obj["tank_shape"]))
                    room, q = obj["clearance"], obj["quantity"]
                    clearances.add("free" if room > q else "exact" if room == q else "zero" if room == 0 else "clip")
                    damaged_sizes.update((len(o), n) for o, v, n in obj["orbit_rows"] if n)
                for k in range(1, 4):
                    assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                # Change only local non-marker color roles, with a bijection.
                table = np.arange(10)
                for a, b in zip(_COLORS, _COLORS[1:] + _COLORS[:1]):
                    table[a] = b
                assert np.array_equal(solve(table[inp]), table[out])
    assert len(hashes) == 42 and len(shapes) >= 45 and len(profiles) >= 40
    assert len(tanks) >= 15 and len(palettes) >= 25
    assert directions == set(range(4)) and quantities == set(range(2, 8))
    assert parities == {(0, 0), (0, 1), (1, 0), (1, 1)} and {1, 2} <= counts
    assert clearances == {"free", "exact", "clip", "zero"}
    assert {(4, 1), (4, 2), (4, 3), (2, 1)} <= damaged_sizes
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for which in ("chamber", "frame", "ray"):
        bad = {s: [{k: a.copy() for k, a in p.items()} for p in sample[s]] for s in sample}
        pair = bad["test"][0]
        obj = next(o for o in _objects(pair["input"]) if o["clearance"] > o["quantity"])
        cell = min(obj["cavity"]) if which == "chamber" else min(obj["frame"]) if which == "frame" else obj["ray"][0]
        pair["output"][cell] = (int(pair["output"][cell]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, count boundaries, rotations, colors, and negative labels")
