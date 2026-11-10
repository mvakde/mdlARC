"""Counted material becomes a taut, intrinsically patterned ribbon.

Only standard-library modules and NumPy are used. The solver reads the input
alone; curriculum parameters affect sampling, never interpretation.
"""
from __future__ import annotations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)
REQUIRED_EVIDENCE = frozenset({
    "straight_rail", "uniform_ink", "raw_area_length", "slack_bbox",
    "four_connected_count", "carrier_tile_u", "carrier_tile_j",
    "stencil_stretch_u", "stencil_stretch_j", "overlay_before_stretch",
    "opaque_background", "ignore_stencil", "world_phase", "turn_reset",
    "reverse_lanes", "squeeze_visible", "obstacle_bbox",
    "visible_carrier", "visible_overwrite",
})


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _array(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int64, copy=False)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def _connected(cells, directions=DIR8, start=None):
    if not cells:
        return set()
    start = min(cells) if start is None else start
    reached, todo = {start}, [start]
    while todo:
        u, v = todo.pop()
        for du, dv in directions:
            z = u + du, v + dv
            if z in cells and z not in reached:
                reached.add(z)
                todo.append(z)
    return reached


def _local(point, tip, forward, lateral):
    r, c = point[0] - tip[0], point[1] - tip[1]
    return r * forward[0] + c * forward[1], r * lateral[0] + c * lateral[1]


def _global(point, tip, forward, lateral):
    u, v = point
    return (tip[0] + u * forward[0] + v * lateral[0],
            tip[1] + u * forward[1] + v * lateral[1])


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _support(s, p, q, width):
    end, divider = -s + 2 * p + 2, -s + p + 1
    result = {(u, 0) for u in range(-s, 1)}
    result |= {(-1, v) for v in range(1, width + 1)}
    result |= {(u, v) for u in range(-s, end + 1)
               for v in range(-q - 2, 0)
               if u in (-s, divider, end) or v in (-q - 2, -1)}
    return result


def _swatch_cells(s, p, q, nearer=False):
    start = -s + (p + 2 if nearer else 1)
    return [[(start + i, -2 - j) for j in range(q)] for i in range(p)]


def _rail(obstacle, length, check=True):
    peak = max(v for u, v in obstacle)
    first_peak = min(u for u, v in obstacle if v == peak)
    values = [max(v + 1 - abs(u - t) for t, v in obstacle)
              for u in range(max(length, max(t for t, v in obstacle) + 4))]
    if peak == 0:
        values[:first_peak + 1] = [1] * (first_peak + 1)
    if check:
        slopes = np.diff(values).tolist()
        if (min(u for u, v in obstacle) < 1 or peak < 0
                or (peak != 0 and max(v - u for u, v in obstacle) != 0)
                or values[0] != 1 or any(d not in (-1, 0, 1) for d in slopes)
                or slopes != sorted(slopes, reverse=True)):
            raise ValueError("unsupported obstacle tangent or exposed contour")
        if any((u, v) in obstacle for u, v in enumerate(values)):
            raise ValueError("rail enters obstacle")
    return values[:length]


def _decode_candidate(a, bg, sc, points, tip, f, lateral):
    local_support = {_local(z, tip, f, lateral) for z in points}
    s = -min(u for u, v in local_support)
    width = max(v for u, v in local_support)
    q = -min(v for u, v in local_support) - 2
    negative = [(u, v) for u, v in local_support if v < 0]
    if not negative:
        raise ValueError("missing cassette")
    span = max(u for u, v in negative) + s - 2
    p = span // 2
    if (span % 2 or not 2 <= p <= 4 or not 1 <= q <= 3
            or not 2 * p + 4 <= s <= 2 * p + 6
            or not 1 <= width <= 6 or width % q
            or local_support != _support(s, p, q, width)):
        raise ValueError("wrong support grammar")
    sample_points = set()
    swatches = []
    for near in (False, True):
        cells = _swatch_cells(s, p, q, near)
        global_cells = [[_global(z, tip, f, lateral) for z in row] for row in cells]
        if not all(_inside(z, a.shape) for row in global_cells for z in row):
            raise ValueError("incomplete cassette")
        swatches.append(np.array([[a[z] for z in row] for row in global_cells]))
        sample_points.update(z for row in global_cells for z in row)
    carrier, stencil = swatches
    if np.any(carrier == bg) or any(np.any(swatch == sc) for swatch in swatches):
        raise ValueError("illegal swatch ink")
    other = {tuple(map(int, z)) for z in np.argwhere(a != bg)} - points - sample_points
    slack, obstacle = set(), set()
    for z in other:
        u, v = _local(z, tip, f, lateral)
        if u <= 0 and v >= 1:
            slack.add(z)
        elif u >= 1:
            obstacle.add(z)
        else:
            raise ValueError("unassigned input ink")
    if not slack or not obstacle:
        raise ValueError("missing material or obstacle")
    tc, oc = int(a[min(slack)]), int(a[min(obstacle)])
    if len({bg, sc, tc, oc}) != 4:
        raise ValueError("scene colors must distinguish roles")
    if not all(a[z] == tc for z in slack) or not all(a[z] == oc for z in obstacle):
        raise ValueError("multicolored material or obstacle")
    if _connected(slack) != slack or _connected(obstacle) != obstacle:
        raise ValueError("disconnected material or obstacle")
    if len(slack) % width:
        raise ValueError("material area is not divisible by width")
    length = len(slack) // width
    if not 2 <= length <= 12 or length % p:
        raise ValueError("illegal ribbon length or replication factor")
    anchors = {_global((0, j + 1), tip, f, lateral) for j in range(width)}
    if not anchors <= slack:
        raise ValueError("incomplete attachment row")
    obs = {_local(z, tip, f, lateral) for z in obstacle}
    rail = _rail(obs, length)
    return dict(bg=bg, support_color=sc, slack_color=tc, obstacle_color=oc,
                support=points, samples=sample_points, slack=slack, obstacle=obstacle,
                local_obstacle=obs, tip=tip, f=f, lateral=lateral, s=s, p=p, q=q,
                width=width, length=length, carrier=carrier, stencil=stencil, rail=rail)


def _decode(grid):
    a = _array(grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("input perimeter must visibly encode background")
    candidates = []
    for sc in np.unique(a):
        sc = int(sc)
        if sc == bg:
            continue
        points = {tuple(map(int, z)) for z in np.argwhere(a == sc)}
        # A legal scaffold has exactly two terminal tips and is 4-connected.
        if len(points) < 27:
            continue
        tips = [(z, [(z[0] + dr, z[1] + dc) for dr, dc in DIR4
                     if (z[0] + dr, z[1] + dc) in points]) for z in sorted(points)]
        tips = [(z, neighbors[0]) for z, neighbors in tips if len(neighbors) == 1]
        if len(tips) != 2 or _connected(points, DIR4) != points:
            continue
        for tip, previous in tips:
            f = tip[0] - previous[0], tip[1] - previous[1]
            for lateral in ((f[1], -f[0]), (-f[1], f[0])):
                try:
                    candidates.append(_decode_candidate(a, bg, sc, points, tip, f, lateral))
                except ValueError:
                    pass
    if len(candidates) != 1:
        raise ValueError(f"expected a unique input-only assembly; found {len(candidates)}")
    return a, candidates[0]


def _texture(d, length=None, variant=None):
    length = d["length"] if length is None else length
    width, p, q = d["width"], d["p"], d["q"]
    u, j = np.arange(length)[:, None], np.arange(width)[None, :]
    cu, cj = u * p // length, j * q // width
    if variant == "carrier_tile_u":
        cu = u % p
    if variant == "carrier_tile_j":
        cj = j % q
    base = d["carrier"][cu, cj]
    su, sj = u % p, j % q
    if variant == "stencil_stretch_u":
        su = u * p // length
    if variant == "stencil_stretch_j":
        sj = j * q // width
    if variant == "turn_reset":
        values = _rail(d["local_obstacle"], length)
        slopes = np.diff(values)
        resets = {k for k in range(1, len(slopes)) if slopes[k] != slopes[k - 1]}
        last, indices = 0, []
        for k in range(length):
            if k in resets:
                last = k
            indices.append((k - last) % p)
        su = np.array(indices)[:, None]
    if variant == "world_phase":
        rail = _rail(d["local_obstacle"], length)
        world = [[_global((k, rail[k] + lane), d["tip"], d["f"], d["lateral"])
                  for lane in range(width)] for k in range(length)]
        coords = np.array(world)
        su, sj = coords[:, :, 0] % p, coords[:, :, 1] % q
    tiled = d["stencil"][su, sj]
    if variant == "ignore_stencil":
        result = base
    elif variant == "opaque_background":
        result = tiled
    elif variant == "overlay_before_stretch":
        fused = np.where(d["stencil"] == d["bg"], d["carrier"], d["stencil"])
        result = fused[u * p // length, j * q // width]
    elif variant == "uniform_ink":
        result = np.full((length, width), d["slack_color"])
    else:
        result = np.where(tiled == d["bg"], base, tiled)
    if variant == "reverse_lanes":
        result = result[:, ::-1]
    return result.astype(np.int64)


def _draw(a, d, variant=None):
    length, width = d["length"], d["width"]
    if variant == "raw_area_length":
        length = len(d["slack"])
    elif variant == "slack_bbox":
        cells = [_local(z, d["tip"], d["f"], d["lateral"]) for z in d["slack"]]
        area = ((max(u for u, v in cells) - min(u for u, v in cells) + 1)
                * (max(v for u, v in cells) - min(v for u, v in cells) + 1))
        length = max(1, area // width)
    elif variant == "four_connected_count":
        attachment = _global((0, 1), d["tip"], d["f"], d["lateral"])
        length = max(1, len(_connected(d["slack"], DIR4, attachment)) // width)
    elif variant == "squeeze_visible":
        visible_steps = [u for u, v in enumerate(d["rail"])
                         if any(_inside(_global((u, v + j), d["tip"], d["f"], d["lateral"]), a.shape)
                                for j in range(width))]
        length = max(1, max(visible_steps, default=0) + 1)
    obstacle = d["local_obstacle"]
    if variant == "obstacle_bbox":
        obstacle = {(u, v) for u in range(min(u for u, v in obstacle), max(u for u, v in obstacle) + 1)
                    for v in range(min(v for u, v in obstacle), max(v for u, v in obstacle) + 1)}
    rail = [1] * length if variant == "straight_rail" else _rail(obstacle, length, check=variant != "obstacle_bbox")
    if variant == "obstacle_bbox":
        rail[0] = 1  # Keep the anchor even in the deliberately wrong route.
    texture = _texture(d, length, variant)
    out = a.copy()
    for z in d["slack"]:
        out[z] = d["bg"]
    preserved = d["support"] | d["samples"] | d["obstacle"]
    for u, v in enumerate(rail):
        for j in range(width):
            z = _global((u, v + j), d["tip"], d["f"], d["lateral"])
            if _inside(z, a.shape):
                if z in preserved:
                    if variant is None:
                        raise ValueError("ribbon collides with preserved material")
                    continue
                out[z] = texture[u, j]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the scaffold, counted material, obstacle, and swatches from input."""
    a, d = _decode(input_grid)
    return _draw(a, d)


def _evidence(a, d, out):
    result = set()
    for name in REQUIRED_EVIDENCE - {"visible_carrier", "visible_overwrite"}:
        if not np.array_equal(out, _draw(a, d, name)):
            result.add(name)
    p, q, length, width = d["p"], d["q"], d["length"], d["width"]
    for u, v in enumerate(d["rail"]):
        for j in range(width):
            z = _global((u, v + j), d["tip"], d["f"], d["lateral"])
            if not _inside(z, a.shape):
                continue
            stencil = d["stencil"][u % p, j % q]
            carrier = d["carrier"][u * p // length, j * q // width]
            if stencil == d["bg"]:
                result.add("visible_carrier")
            elif stencil != carrier:
                result.add("visible_overwrite")
    return result


def _sample_obstacle(rng, approach, length, core=False):
    # Parameterized exposed convex profiles, with varied material below them.
    # Geometry is sampled directly; no parent generator is invoked.
    for _ in range(80):
        kind = int(rng.integers(5))
        near = int(rng.integers(1, min(5, length + 1)))
        if kind == 0:
            raw = {(0, 0)}
        elif kind == 1:
            h, w = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            raw = {(u, v) for u in range(h) for v in range(w)}
        else:
            rise, flat, fall = (int(rng.integers(1, 4)), int(rng.integers(1, 3)), int(rng.integers(1, 4)))
            profile = list(range(rise)) + [rise] * flat + [rise - k for k in range(1, fall + 1)]
            if kind == 2:  # thin cap, including asymmetric descending steps
                raw = {(u, v) for u, v in enumerate(profile)}
            elif kind == 3:  # filled cap, with lower extent independent of top
                bottom = min(profile) - int(rng.integers(0, 3))
                raw = {(u, v) for u, top in enumerate(profile) for v in range(bottom, top + 1)}
            else:  # band beneath the same convex silhouette
                thickness = int(rng.integers(1, 3))
                raw = {(u, top - k) for u, top in enumerate(profile) for k in range(thickness)}
        offset = (-max(v for u, v in raw) if approach == "parallel"
                  else near - max(v - u for u, v in raw))
        obstacle = {(u + near, v + offset) for u, v in raw}
        if core and max(u for u, v in obstacle) > length - 3:
            continue
        try:
            _rail(obstacle, length)
        except ValueError:
            continue
        return obstacle
    # Same rule, conservative point geometry; pattern/scene selection can retry.
    return {(1, 0 if approach == "parallel" else 1)}


def _slack(rng, width, length, depth_cap, lateral_cap, diagonal):
    area = width * length
    anchors = {(0, j) for j in range(1, width + 1)}
    count = area - width
    min_extra = (count + depth_cap - 1) // depth_cap
    extra = int(rng.integers(max(2, min_extra), lateral_cap + 1))
    min_depth = (count + extra - 1) // extra
    depth = int(rng.integers(max(1, min_depth), depth_cap + 1))
    # Reserving u=0 outside the anchor allows controlled diagonal-only contact.
    cells = {(-1, width + 1)}
    style = int(rng.integers(3))
    if style == 0:
        for k in range(count):
            cells.add((-1 - k // extra, width + 1 + k % extra))
    else:
        directions = DIR8 if style == 2 else DIR4
        frontier = set()
        def extend(z):
            for du, dv in directions:
                q = z[0] + du, z[1] + dv
                if -depth <= q[0] <= -1 and width + 1 <= q[1] <= width + extra and q not in cells:
                    frontier.add(q)
        extend((-1, width + 1))
        while len(cells) < count:
            choices = sorted(frontier)
            z = choices[int(rng.integers(len(choices)))]
            frontier.remove(z)
            cells.add(z)
            extend(z)
    # Optionally bridge orthogonally without altering the material area.
    if not diagonal and count >= 2:
        distant = max(cells, key=lambda z: (abs(z[0] + 1) + abs(z[1] - width - 1), z))
        trial = (cells - {distant}) | {(0, width + 1)}
        if len(trial) == count and _connected(trial | anchors) == trial | anchors:
            cells = trial
    return anchors | cells


def _patterns(rng, p, q, bg, inks, core):
    k = int(rng.integers(2 if core else 1, min(5, len(inks)) + 1))
    palette = rng.choice(inks, k, replace=False)
    rr, cc = np.indices((p, q))
    kind = int(rng.integers(6))
    if kind == 0:
        labels = (rr + cc) % k
    elif kind == 1:
        labels = (rr * int(rng.integers(1, k + 1)) + cc) % k
    elif kind == 2:
        labels = np.broadcast_to(rng.integers(k, size=(p, 1)), (p, q)).copy()
    elif kind == 3:
        labels = np.broadcast_to(rng.integers(k, size=(1, q)), (p, q)).copy()
    elif kind == 4:
        rcut, ccut = int(rng.integers(1, p)), int(rng.integers(1, max(2, q)))
        labels = rng.integers(k, size=(2, 2))[(rr >= rcut).astype(int), (cc >= ccut).astype(int)]
    else:
        labels = rng.integers(k, size=(p, q))
    carrier = palette[labels].astype(np.int64)
    stencil = np.full((p, q), bg, dtype=np.int64)
    if not core and rng.random() < .18:
        return carrier, stencil
    density = float(rng.uniform(.15, .75))
    mask = rng.random((p, q)) < density
    style = int(rng.integers(4))
    if style == 1:
        mask = (rr + cc + int(rng.integers(2))) % 2 == 0
    elif style == 2:
        mask = rr == int(rng.integers(p))
    elif style == 3 and q > 1:
        mask = cc == int(rng.integers(q))
    if not mask.any():
        mask.ravel()[int(rng.integers(p * q))] = True
    if mask.all():
        mask.ravel()[int(rng.integers(p * q))] = False
    ink_count = int(rng.integers(1, min(4, len(inks), int(mask.sum())) + 1))
    stencil[mask] = rng.choice(rng.choice(inks, ink_count, replace=False), int(mask.sum()))
    return carrier, stencil


def _sample_pair(rng, regime, axis, split, approach=None, clipped=None, core=False):
    for attempt in range(180):
        p = int(rng.integers(3 if core else 2, 5))
        q = int(rng.integers(2 if core else 1, (3 if regime == 1 else 4)))
        k = 3 if regime == 0 and split == "test" else 2
        t = 3 if regime == 1 and split == "test" else 2
        if not core and regime != 0 and rng.random() < .3:
            k = int(rng.integers(1, 4))
        if not core and regime != 1 and rng.random() < .35:
            t = 1
        length, width = p * k, q * t
        if width > 6:
            continue
        s = 2 * p + int(rng.integers(4, 7))
        background, support_color, slack_color, obstacle_color = map(int, rng.choice(10, 4, replace=False))
        inks = [c for c in range(10) if c not in (background, support_color)]
        carrier, stencil = _patterns(rng, p, q, background, inks, core)
        this_approach = approach or ("parallel" if rng.random() < .4 else "diagonal")
        obstacle = _sample_obstacle(rng, this_approach, length, core)
        rail = _rail(obstacle, length)
        support = _support(s, p, q, width)
        slack = _slack(rng, width, length, s, 25 - width - q,
                       diagonal=core or rng.random() < .5)
        local_input = support | slack | obstacle
        ribbon = {(u, v + j) for u, v in enumerate(rail) for j in range(width)}
        want_clip = bool(rng.random() < .35) if clipped is None else clipped
        # Each input has a full background border. Crop only output material.
        cells_for_bounds = local_input if want_clip else local_input | ribbon
        u0, u1 = min(u for u, v in cells_for_bounds), max(u for u, v in cells_for_bounds)
        v0, v1 = min(v for u, v in cells_for_bounds), max(v for u, v in cells_for_bounds)
        pads = list(map(int, rng.integers(1, 4, size=4)))
        if want_clip:
            pads[1] = 1
            if rng.random() < .35:
                pads[3] = 1
        height, grid_width = u1 - u0 + 1 + pads[0] + pads[1], v1 - v0 + 1 + pads[2] + pads[3]
        if max(height, grid_width) > 30:
            continue
        tip = -u0 + pads[0], -v0 + pads[2]
        f, lateral = (1, 0), (0, 1)
        a = np.full((height, grid_width), background, dtype=np.int64)
        for cells, color in ((support, support_color), (slack, slack_color), (obstacle, obstacle_color)):
            for z in cells:
                a[_global(z, tip, f, lateral)] = color
        for near, data in ((False, carrier), (True, stencil)):
            for i, row in enumerate(_swatch_cells(s, p, q, near)):
                for j, z in enumerate(row):
                    a[_global(z, tip, f, lateral)] = data[i, j]
        # Orient the complete assembly. Swatches and lanes turn together.
        if regime == 2:
            actual_axis = axis if split == "train" else 1 - axis
        else:
            actual_axis = int(rng.integers(2))
        rotation = (0 if actual_axis == 0 else 1) + 2 * int(rng.integers(2))
        a = np.rot90(a, rotation)
        if rng.random() < .5:
            a = a[:, ::-1]
        a = a.copy()
        try:
            a, d = _decode(a)
            out = _draw(a, d)
        except ValueError:
            continue
        visible = _visible(d, a.shape)
        if (clipped is not None and ((len(visible) < width * length) != clipped)):
            continue
        if core and not any(u > max(u for u, v in obstacle) for u, j in visible):
            continue
        if np.array_equal(a, out):
            continue
        return {"input": a, "output": out}
    raise RuntimeError("bounded compatible-scene sampling exhausted")


def _visible(d, shape):
    return {(u, j) for u, v in enumerate(d["rail"]) for j in range(d["width"])
            if _inside(_global((u, v + j), d["tip"], d["f"], d["lateral"]), shape)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt, nq = _integer(num_train, "num_train", 2, 10), _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime, axis = int(rng.integers(3)), int(rng.integers(2))
    train = None
    # Two varied teaching scenes cover the rule even at the smallest count.
    # They are selected for observable distinctions, not fixed output pictures.
    for _ in range(120):
        clipped_first = bool(rng.integers(2))
        first = _sample_pair(rng, regime, axis, "train", "parallel", clipped_first, True)
        second = _sample_pair(rng, regime, axis, "train", "diagonal", not clipped_first, True)
        evidence = set()
        for pair in (first, second):
            a, d = _decode(pair["input"])
            evidence |= _evidence(a, d, pair["output"])
        if REQUIRED_EVIDENCE <= evidence and _key(first["input"]) != _key(second["input"]):
            train = [first, second]
            break
    if train is None:
        raise RuntimeError("bounded demonstration selection exhausted")
    puzzle, seen = {"train": train, "test": []}, {_key(pair["input"]) for pair in train}
    for split, count in (("train", nt), ("test", nq)):
        while len(puzzle[split]) < count:
            for _ in range(50):
                pair = _sample_pair(rng, regime, axis, split)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded unique-input sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence, training, tests = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, d = _decode(pair["input"])
            out = _array(pair["output"])
            assert a.shape == out.shape and not np.array_equal(a, out)
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            assert np.array_equal(_draw(a, d), out), "incorrect intrinsic ribbon answer"
            assert len(d["slack"]) == d["length"] * d["width"]
            visible = _visible(d, a.shape)
            preserved = d["support"] | d["samples"] | d["obstacle"]
            painted = {_global((u, d["rail"][u] + j), d["tip"], d["f"], d["lateral"])
                       for u, j in visible}
            assert not painted & preserved
            assert all(out[z] == a[z] for z in preserved), "changed preserved object"
            assert len(painted) == len(visible), "self-overlapping strip"
            assert all(out[z] != d["bg"] for z in painted), "missing ribbon material"
            actual_material = {tuple(map(int, z)) for z in np.argwhere(out != d["bg"])} - preserved
            assert actual_material == painted, "material left outside the ribbon"
            off_grid = sum(not _inside(_global((u, v + j), d["tip"], d["f"], d["lateral"]), a.shape)
                           for u, v in enumerate(d["rail"]) for j in range(d["width"]))
            assert len(actual_material) == len(d["slack"]) - off_grid, "area conservation or clipping failed"
            allowed_changes = d["slack"] | painted
            assert all(tuple(map(int, z)) in allowed_changes for z in np.argwhere(a != out))
            record = (d, a.shape, len(visible) < d["length"] * d["width"])
            if split == "train":
                training.append(record)
                evidence |= _evidence(a, d, out)
            else:
                tests.append(record)
    assert REQUIRED_EVIDENCE <= evidence, f"missing teaching evidence: {REQUIRED_EVIDENCE - evidence}"
    assert {max(v for u, v in d["local_obstacle"]) == 0 for d, shape, clip in training} == {False, True}
    assert {clip for d, shape, clip in training} == {False, True}
    assert any(not clip and len(_connected(d["slack"], DIR4)) < len(d["slack"])
               for d, shape, clip in training)
    # Recover the shift from arrays alone, without generation metadata.
    length_shift = ({d["length"] // d["p"] for d, _, _ in training} == {2}
                    and {d["length"] // d["p"] for d, _, _ in tests} == {3})
    width_shift = ({d["width"] // d["q"] for d, _, _ in training} == {2}
                   and {d["width"] // d["q"] for d, _, _ in tests} == {3})
    train_axes = {bool(d["f"][0]) for d, _, _ in training}
    test_axes = {bool(d["f"][0]) for d, _, _ in tests}
    orientation_shift = len(train_axes) == len(test_axes) == 1 and train_axes != test_axes
    assert length_shift or width_shift or orientation_shift, "missing meaningful test shift"


def _normalized(grid):
    roles = {}
    return np.asarray(grid).shape, bytes(roles.setdefault(int(c), len(roles)) for c in np.asarray(grid).flat)


def _measure(pair):
    a, d = _decode(pair["input"])
    obs = d["local_obstacle"]
    low_u, low_v = min(u for u, v in obs), min(v for u, v in obs)
    slack = {_local(z, d["tip"], d["f"], d["lateral"]) for z in d["slack"]}
    low_su, low_sv = min(u for u, v in slack), min(v for u, v in slack)
    return dict(shape=a.shape, bg=d["bg"], swatch=(d["p"], d["q"]), width=d["width"], length=d["length"],
                orientation=(*d["f"], *d["lateral"]),
                silhouette=tuple(sorted((u - low_u, v - low_v) for u, v in obs)),
                offset=(low_u, low_v),
                slack=tuple(sorted((u - low_su, v - low_sv) for u, v in slack)),
                carrier=_normalized(d["carrier"]),
                stencil=_normalized(d["stencil"]), mask=_key(d["stencil"] != d["bg"]),
                clipped=len(_visible(d, a.shape)) < len(d["slack"]),
                diagonal=len(_connected(d["slack"], DIR4)) < len(d["slack"]),
                empty=bool(np.all(d["stencil"] == d["bg"])),
                output=_normalized(pair["output"]))


def self_test():
    """Bounded regressions, 42-seed diversity, counts and negative validation."""
    assert _rail({(2, 0)}, 6) == [1, 1, 1, 0, -1, -2]
    assert _rail({(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)}, 6) == [1, 2, 3, 3, 2, 1]
    assert _rail({(2, 2), (3, 3), (4, 2)}, 7) == [1, 2, 3, 4, 3, 2, 1]
    toy = dict(length=4, width=4, p=2, q=2, bg=0,
               carrier=np.array([[4, 5], [6, 7]]), stencil=np.array([[0, 8], [9, 0]]))
    expected = np.array([[4, 8, 5, 8], [9, 4, 9, 5], [6, 8, 7, 8], [9, 6, 9, 7]])
    assert np.array_equal(_texture(toy), expected)
    # The smallest legal scaffold has 27 cells (p=2,q=1,s=8,W=1).
    small = np.zeros((12, 8), dtype=np.int64)
    st = (9, 4)
    for cells, color in ((_support(8, 2, 1, 1), 1), ({(0, 1), (-1, 2)}, 2), ({(1, 1)}, 3)):
        for z in cells:
            small[_global(z, st, (1, 0), (0, 1))] = color
    for i, row in enumerate(_swatch_cells(8, 2, 1)):
        small[_global(row[0], st, (1, 0), (0, 1))] = 4 + i
    small_answer = solve(small)
    assert small_answer[9, 5] == 4 and small_answer[10, 6] == 5
    assert small_answer[8, 6] == 0
    # Explicit lateral-edge regression: one outer-lane cell leaves the canvas
    # at the peak, then that lane re-enters with its original intrinsic phase.
    # Random cohorts need not happen to contain this relatively rare boundary.
    edge = np.zeros((24, 17), dtype=np.int64)
    edge_tip, f, lateral = (15, 6), (1, 0), (0, 1)
    edge_slack = {(0, j) for j in range(1, 7)} | {
        (u, v) for u in range(-14, 0) for v in range(7, 10)}
    for cells, color in ((_support(14, 4, 3, 6), 1), (edge_slack, 2), ({(5, 5)}, 3)):
        for z in cells:
            edge[_global(z, edge_tip, f, lateral)] = color
    for near, data in ((False, np.array([[4, 5, 6], [6, 4, 5], [5, 6, 4], [4, 6, 5]])),
                       (True, np.array([[0, 7, 0], [8, 0, 0], [0, 0, 7], [0, 8, 0]]))):
        for i, row in enumerate(_swatch_cells(14, 4, 3, near)):
            for j, z in enumerate(row):
                edge[_global(z, edge_tip, f, lateral)] = data[i, j]
    _, ed = _decode(edge)
    visibility = _visible(ed, edge.shape)
    assert len(visibility) == 47 and (5, 5) not in visibility and (6, 5) in visibility
    edge_answer = solve(edge)
    reentry = _global((6, ed["rail"][6] + 5), ed["tip"], ed["f"], ed["lateral"])
    assert edge_answer[reentry] == 7  # stencil[6 % 4, 5 % 3], without a reset
    assert all(edge_answer[z] == edge[z] for z in ed["support"] | ed["samples"] | ed["obstacle"])
    metrics, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(pair["input"]) for split in ("train", "test") for pair in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(np.asarray(pair["input"]).copy()), pair["output"])
                metrics.append(_measure(pair))
    assert len(signatures) == 42
    assert len({m["shape"] for m in metrics}) >= 35
    assert len({m["bg"] for m in metrics}) == 10
    assert len({m["orientation"] for m in metrics}) == 8
    assert len({m["swatch"] for m in metrics}) >= 6
    assert len({m["width"] for m in metrics}) >= 5
    assert len({m["length"] for m in metrics}) >= 6
    assert len({m["silhouette"] for m in metrics}) >= 20
    assert len({m["slack"] for m in metrics}) >= 80
    assert len({m["carrier"] for m in metrics}) >= 50
    assert len({m["mask"] for m in metrics}) >= 40
    assert len({m["output"] for m in metrics}) >= 180
    assert {m["clipped"] for m in metrics} == {False, True}
    assert {m["diagonal"] for m in metrics} == {False, True}
    assert any(m["empty"] for m in metrics)
    first, repeat = generate_puzzle(97, 2, 1), generate_puzzle(97, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for split in first for p, q in zip(first[split], repeat[split])
               for k in ("input", "output"))
    # All D4 transforms preserve the interpreted local data and answer.
    example = first["test"][0]
    for k in range(4):
        for reflect in (False, True):
            a, out = np.rot90(example["input"], k), np.rot90(example["output"], k)
            if reflect:
                a, out = a[:, ::-1], out[:, ::-1]
            assert np.array_equal(solve(a), out)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(400 + 10 * nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")
