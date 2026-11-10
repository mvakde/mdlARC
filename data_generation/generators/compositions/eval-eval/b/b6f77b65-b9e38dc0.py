"""Light-released scaffolds: selective dissolution, gravity, and renewed sight.

All inference comes from the input. No parent module, file, seed, or example is
read by solve(). Rods are disjoint, fully visible straight monochrome pieces.
No artifacts or execution traces are written by this module.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < low or (high is not None and int(value) > high):
        raise ValueError(f"{name} is out of range")
    return int(value)


# A rod is (H/V, top row, left column, length, color). No hidden cells/joints.
def _cells(p):
    o, r, c, n, _ = p
    return tuple((r + (j if o == "V" else 0),
                  c + (j if o == "H" else 0)) for j in range(n))


def _shift(p, d):
    return p[0], p[1] + d, p[2], p[3], p[4]


def _settle(state, height):
    """Synchronous, greatest movable set: supported pieces can descend together."""
    state = dict(state)
    for _ in range(height * max(1, len(state)) + 1):
        occupied = {i: set(_cells(p)) for i, p in state.items()}
        targets = {i: {(r + 1, c) for r, c in s} for i, s in occupied.items()}
        moving = {i for i, s in targets.items() if all(r < height for r, c in s)}
        while True:
            blocked = {i for i in moving if any(targets[i] & occupied[j]
                       for j in state if j not in moving)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return state
        state = {i: _shift(p, 1) if i in moving else p for i, p in state.items()}
    raise AssertionError("gravity did not converge")


def _settle_events(state, height):
    """Construction/check path: lower rods jump to their next collision."""
    state = dict(state)
    for _ in range(height * max(1, len(state)) + 1):
        changed = False
        order = sorted(state, key=lambda i: max(r for r, c in _cells(state[i])), reverse=True)
        for i in order:
            p = state[i]
            others = set().union(*(set(_cells(q)) for j, q in state.items() if j != i))
            distance = 0
            for d in range(1, height):
                candidate = _cells(_shift(p, d))
                if any(r >= height or (r, c) in others for r, c in candidate):
                    break
                distance = d
            if distance:
                state[i] = _shift(p, distance)
                changed = True
        if not changed:
            return state
    raise AssertionError("event gravity did not converge")


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in unseen:
                    unseen.remove(q)
                    group.append(q)
                    stack.append(q)
        result.append(group)
    return result


def _fit(values, sign):
    """Longest periodic terminal suffix; period and phase are visible, not latent."""
    ys = np.array(values, dtype=int)
    xs = np.arange(len(ys)) - len(ys) + 1
    candidates = []
    for period in (1, 2, 3):
        for phase in range(period):
            prediction = ys[-1] + sign * ((xs + phase) // period)
            n = 0
            for same in (prediction == ys)[::-1]:
                if not same:
                    break
                n += 1
            if n >= 2 * period + 1:
                candidates.append((n, -period, phase))
    assert candidates, "each flank must expose two complete outward steps"
    n, negative_period, phase = max(candidates)
    return -negative_period, phase, int(ys[-1]), n


def _decode(input_grid):
    g = np.asarray(input_grid)
    assert g.ndim == 2 and np.issubdtype(g.dtype, np.integer)
    assert all(1 <= n <= 30 for n in g.shape) and np.all((0 <= g) & (g <= 9))
    bg = int(g[0, 0])
    assert np.all(g[0] == bg), "input has a blank top margin"
    assert np.count_nonzero(g == bg) > g.size // 2
    candidates = []
    for mirrored in (False, True):
        a = g[:, ::-1] if mirrored else g
        for wall in map(int, np.unique(a)):
            if wall == bg:
                continue
            points = np.argwhere(a == wall)
            rear, mouth = int(points[:, 1].min()), int(points[:, 1].max())
            if mouth - rear < 3:
                continue
            cap = np.flatnonzero(a[:, rear] == wall)
            if len(cap) < 3 or np.any(np.diff(cap) != 1):
                continue
            upper, lower = [], []
            for c in range(rear + 1, mouth + 1):
                rows = np.flatnonzero(a[:, c] == wall)
                if len(rows) != 2 or rows[1] - rows[0] < 2:
                    break
                upper.append(int(rows[0]))
                lower.append(int(rows[1]))
            else:
                try:
                    fits = (_fit(upper, -1), _fit(lower, 1))
                except AssertionError:
                    continue
                footprint = np.zeros(a.shape, dtype=bool)
                for c in range(rear + 1, a.shape[1]):
                    if c <= mouth:
                        lo, hi = upper[c-rear-1], lower[c-rear-1]
                    else:
                        limits = [value + sign * ((c-mouth+phase) // period)
                                  for (period, phase, value, _), sign in zip(fits, (-1, 1))]
                        lo, hi = limits
                    footprint[max(0, lo+1):min(a.shape[0], hi), c] = True
                chamber = footprint.copy()
                chamber[:, mouth+1:] = False
                patch = chamber & (a != bg) & (a != wall)
                colors = np.unique(a[patch])
                if len(colors) != 1 or len(_components(patch)) != 1:
                    continue
                # Nothing else is present upstream of, or alongside, the emitter.
                static_mask = (a == wall) | patch
                if np.any((a[:, :mouth+1] != bg) & ~static_mask[:, :mouth+1]):
                    continue
                rods = []
                good = True
                for color in map(int, np.unique(a[:, mouth+1:])):
                    if color == bg:
                        continue
                    mask = a == color
                    mask[:, :mouth+1] = False
                    for group in _components(mask):
                        rs, cs = zip(*group)
                        if len(group) < 2 or (len(set(rs)) > 1 and len(set(cs)) > 1):
                            good = False
                            break
                        o = "H" if len(set(rs)) == 1 else "V"
                        rods.append((o, min(rs), min(cs), len(group), color))
                if not good or not rods:
                    continue
                rods.sort(key=lambda p: (p[1], p[2], p[0], p[4]))
                static = np.full(a.shape, bg, dtype=np.int8)
                static[static_mask] = a[static_mask]
                state = dict(enumerate(rods))
                if not np.array_equal(_render(static, state), a):
                    continue
                if _settle(state, a.shape[0]) != state:
                    continue
                candidates.append(dict(grid=a.copy(), bg=bg, wall=wall, payload=int(colors[0]),
                                       mirrored=mirrored, rear=rear, mouth=mouth, fits=fits,
                                       footprint=footprint, chamber=chamber, static=static, state=state))
    assert len(candidates) == 1, "one unambiguous fixed funnel and rod decomposition required"
    return candidates[0]


def _render(static, state):
    result = static.copy()
    occupied = set()
    for p in state.values():
        for r, c in _cells(p):
            assert (r, c) not in occupied and result[r, c] == static[0, 0]
            occupied.add((r, c))
            result[r, c] = p[4]
    return result


def _hits_scan(state, footprint, payload):
    owners = {cell: i for i, p in state.items() for cell in _cells(p)}
    hits = set()
    for r in range(footprint.shape[0]):
        for c in np.flatnonzero(footprint[r]):
            i = owners.get((r, int(c)))
            if i is not None:
                if state[i][4] == payload:
                    hits.add(i)
                break
    return hits


def _hits_fronts(state, footprint, payload):
    # Separate construction path: union front surfaces, not a cell-by-cell ray scan.
    fronts = {}
    for i, p in state.items():
        for r, c in _cells(p):
            if footprint[r, c] and (r not in fronts or c < fronts[r][0]):
                fronts[r] = (c, i)
    return {i for c, i in fronts.values() if state[i][4] == payload}


def _evolve(state, footprint, payload, *, construction=False, gravity=True, one_pass=False):
    state = dict(state)
    removed = []
    hit_function = _hits_fronts if construction else _hits_scan
    settle_function = _settle_events if construction else _settle
    for _ in range(len(state) + 1):
        hit = hit_function(state, footprint, payload)
        if not hit:
            return state, removed
        removed.append(tuple(sorted(hit)))
        state = {i: p for i, p in state.items() if i not in hit}
        if gravity:
            state = settle_function(state, footprint.shape[0])
        if one_pass:
            return state, removed
    raise AssertionError("each optical round must remove at least one rod")


def _illuminate(static, state, footprint, bg, payload, *, construction=False):
    result = _render(static, state)
    if construction:
        shadow = np.zeros(footprint.shape, dtype=bool)
        for p in state.values():
            for r, c in _cells(p):
                if footprint[r, c]:
                    shadow[r, c:] = True
        result[footprint & ~shadow & (result == bg)] = payload
    else:
        occupied = {cell for p in state.values() for cell in _cells(p)}
        for r in range(result.shape[0]):
            blocked = False
            for c in np.flatnonzero(footprint[r]):
                if (r, int(c)) in occupied:
                    blocked = True
                if not blocked and result[r, c] == bg:
                    result[r, c] = payload
    return result


def _answer(a, **options):
    state, rounds = _evolve(a["state"], a["footprint"], a["payload"], **options)
    out = _illuminate(a["static"], state, a["footprint"], a["bg"], a["payload"],
                      construction=options.get("construction", False))
    return out, state, rounds


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _decode(input_grid)
    out, _, _ = _answer(a)
    return (out[:, ::-1] if a["mirrored"] else out).copy()


def _geometry(rng, stages):
    periods = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
    phases = [int(rng.integers(p)) for p in periods]
    rear = int(rng.integers(0, 3))
    rear_length = int(rng.integers(1, 3))
    flare_length = 2 * max(periods) + 1
    profiles = []
    for period, phase in zip(periods, phases):
        neck = int(rng.integers(1, 3))
        bulb = neck + int(rng.integers(0, 2))
        profile = [bulb] * (rear_length - 1) + [neck]
        profile += [neck + (t + phase) // period for t in range(1, flare_length + 1)]
        profiles.append(profile)
    mouth = rear + len(profiles[0])
    # Leave room for all interacting bays, their gaps, and a visible beam tail.
    minimum_width = mouth + 3 * stages + 7
    assert minimum_width <= 30
    width = int(rng.integers(minimum_width, 31))
    height = int(rng.integers(max(18, max(profiles[0])+max(profiles[1])+4), 31))
    low_center = max(5, max(profiles[0]) + 1)
    high_center = min(height-max(profiles[1])-2, height-8)
    if high_center < low_center:
        height = 30
        high_center = min(height-max(profiles[1])-2, height-8)
    center = int(rng.integers(low_center, high_center+1))
    wall = np.zeros((height, width), dtype=bool)
    footprint = np.zeros_like(wall)
    wall[center-profiles[0][0]:center+profiles[1][0]+1, rear] = True
    for j, (up, down) in enumerate(zip(*profiles), rear+1):
        wall[center-up, j] = wall[center+down, j] = True
        footprint[center-up+1:center+down, j] = True
    for c in range(mouth+1, width):
        t = flare_length + c-mouth
        up, down = [profile[rear_length-1] + (t+phase)//period
                    for profile, phase, period in zip(profiles, phases, periods)]
        footprint[max(0, center-up+1):min(height, center+down), c] = True
    return wall, footprint, rear, mouth, center, periods


def _construct(rng, stages, curtain, redundant, palette):
    bg, wall_color, payload, *opaque = map(int, palette[:6])
    wall, footprint, rear, mouth, y, periods = _geometry(rng, stages)
    h, w = wall.shape
    static = np.full((h, w), bg, dtype=np.int8)
    static[wall] = wall_color
    patch_cells = [(r, c) for r, c in map(tuple, np.argwhere(footprint)) if c <= mouth]
    anchor = patch_cells[int(rng.integers(len(patch_cells)))]
    patch = {anchor}
    for _ in range(int(rng.integers(0, 4))):
        frontier = sorted({q for r, c in patch for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                           if q in patch_cells and q not in patch})
        if frontier:
            patch.add(frontier[int(rng.integers(len(frontier)))])
    for cell in patch:
        static[cell] = payload
    widths = [2] * (stages + 1)
    if redundant:
        widths[0] = 3  # The teaching examples must actually show two supports.
    gaps = [1] * stages
    left_gap, tail = 1, 2
    spare = w - (mouth+1 + left_gap + sum(widths) + sum(gaps) + tail)
    # Distribute space among unequal overhangs, gaps, placement and projection.
    for _ in range(spare):
        choices = [("width", i) for i, n in enumerate(widths) if n < 5]
        choices += [("gap", i) for i, n in enumerate(gaps) if n < 3]
        choices += [("tail", 0)] + ([("left", 0)] if left_gap < 3 else [])
        kind, i = choices[int(rng.integers(len(choices)))]
        if kind == "width":
            widths[i] += 1
        elif kind == "gap":
            gaps[i] += 1
        elif kind == "left":
            left_gap += 1
        else:
            tail += 1
    rods = []
    x = mouth+1+left_gap
    for stage in range(stages):
        n = widths[stage]
        qcolor = int(rng.choice(opaque))
        other = [c for c in opaque if c != qcolor]
        # Both critical and redundant supports are explicit straight pieces.
        primary = x if redundant and stage == 0 else x + int(rng.integers(n))
        posts = [primary]
        if redundant and n >= 3 and (stage == 0 or rng.random() < .4):
            options = [c for c in range(x, x+n) if abs(c-primary) >= 2]
            if options:
                posts.append(int(rng.choice(options)))
        has_pedestal = False
        for post_index, c in enumerate(posts):
            # Only the first post is tall. Later two-cell target joints lie
            # wholly in its shadow, rather than leaking into newly widened
            # lanes. Their opaque pedestals keep every input gravity-stable.
            if stage == 0 and post_index == 0:
                rods.append(("V", y+1, c, h-y-1, payload))
            else:
                rods.append(("V", y+1, c, 2, payload))
                rods.append(("V", y+3, c, h-y-3, int(rng.choice(other))))
                has_pedestal = True
        rods.append(("H", y, x, n, qcolor))
        closing = curtain and stage == stages-1
        landing_options = [c for c in range(x, x+n) if c not in posts]
        add_landing = bool(landing_options) and (closing or rng.random() < .7)
        if has_pedestal:
            drop = 2
        elif closing:
            drop = int(rng.integers(2, min(4, y-3)+1))
        else:
            drop = int(rng.integers(3, h-y-2)) if add_landing else h-y-1
        if add_landing:
            c = int(rng.choice(landing_options))
            rods.append(("V", y+drop+1, c, h-y-drop-1, int(rng.choice(other))))
        if closing:
            length = min(y-1, drop+int(rng.integers(1, 3)))
            rods.append(("V", y-length, int(rng.integers(x, x+n)), length, int(rng.choice(other))))
        elif drop >= 4 and rng.random() < .7:
            # This load also falls, but leaves the next optical lane open.
            length = int(rng.integers(2, min(5, drop-2, y-2)+1))
            c = int(rng.integers(x, x+n))
            load_color = int(rng.choice(other))
            rods.append(("V", y-length, c, length, load_color))
            if drop >= length+3 and rng.random() < .55:
                rods.append(("H", y-length-1, x, n, qcolor))
        x += n + gaps[stage]
    # Matching terminal rod, initially hidden by the raised beams. Its ordinary
    # opaque foot stays fixed. It also leaves at least two downstream cells.
    rods.append(("H", y, x, widths[-1], payload))
    rods.append(("V", y+1, int(rng.integers(x, x+widths[-1])), h-y-1, int(rng.choice(opaque))))
    state = dict(enumerate(rods))
    inp = _render(static, state)
    # Same-color touching would erase a piece boundary. Never guess that boundary.
    for color in np.unique(inp[:, mouth+1:]):
        if color == bg:
            continue
        mask = inp == color
        mask[:, :mouth+1] = False
        if len(_components(mask)) != sum(p[4] == color for p in rods):
            return None
    final, rounds = _evolve(state, footprint, payload, construction=True)
    out = _illuminate(static, final, footprint, bg, payload, construction=True)
    mirror = bool(rng.integers(2))
    pair = {"input": inp[:, ::-1].copy() if mirror else inp,
            "output": out[:, ::-1].copy() if mirror else out}
    # Decode checks are a contract, not a way of obtaining the constructed label.
    a = _decode(pair["input"])
    assert a["mirrored"] == mirror and a["mouth"] == mouth
    assert np.array_equal(a["footprint"], footprint)
    assert set(a["state"].values()) == set(state.values())
    assert _settle(state, h) == state
    assert np.array_equal(solve(pair["input"]), pair["output"])
    f = _measure(a)
    if not (f["removed"] and f["moved"] and f["outside_erased"]):
        return None
    if curtain and not (f["shielded"] and f["gravity_matters"]):
        return None
    if not curtain and not (f["feedback_matters"] and f["gravity_matters"]):
        return None
    return pair


def _measure(a):
    out, final, rounds = _answer(a)
    original = a["state"]
    removed = set(original) - set(final)
    no_gravity, _, _ = _answer(a, gravity=False)
    one_pass, _, _ = _answer(a, one_pass=True)
    unconditional = {i: p for i, p in original.items() if p[4] != a["payload"]}
    unconditional = _settle(unconditional, out.shape[0])
    without_beam_selection = _illuminate(a["static"], unconditional, a["footprint"], a["bg"], a["payload"])
    return dict(shape=tuple(out.shape), mirrored=a["mirrored"],
                periods=tuple(f[0] for f in a["fits"]), rods=len(original),
                horizontal_levels=len({p[1] for p in original.values() if p[0] == "H"}),
                lengths=tuple(sorted({p[3] for p in original.values()})),
                rounds=len(rounds), removed=len(removed),
                late_removed=sum(len(ids) for ids in rounds[1:]),
                moved=sum(final[i][1] > original[i][1] for i in final),
                distances=tuple(sorted({final[i][1]-original[i][1] for i in final} - {0})),
                shielded=sum(p[4] == a["payload"] for p in final.values()),
                outside_erased=sum(not a["footprint"][cell] for i in removed for cell in _cells(original[i])),
                gravity_matters=not np.array_equal(out, no_gravity),
                feedback_matters=not np.array_equal(out, one_pass),
                beam_selection_matters=not np.array_equal(out, without_beam_selection))


def _free_construct(rng, palette, expanded=False):
    """Compose a support network by insertion, with no bay or common-row grammar.

    Rejection asks for visible causal effects, not a particular answer. These
    networks use exactly the mechanics already taught by the two core examples.
    """
    bg, wall_color, payload, *opaque = map(int, palette[:6])
    wall, footprint, rear, mouth, y, periods = _geometry(rng, 2)
    h, w = wall.shape
    geometry = {}
    target = int(rng.integers(10, 18) if expanded else rng.integers(7, 14))
    for _ in range(80):
        if len(geometry) >= target:
            break
        horizontal = bool(rng.integers(2))
        n = int(rng.integers(2, min(9, w-mouth-1 if horizontal else h-2)+1))
        c = int(rng.integers(mouth+1, w-n+1 if horizontal else w))
        p = ("H" if horizontal else "V", 1, c, n, 0)
        occupied = set().union(*(set(_cells(q)) for q in geometry.values()))
        if set(_cells(p)) & occupied:
            continue
        candidate = dict(geometry)
        candidate[len(candidate)] = p
        geometry = _settle_events(candidate, h)
    if len(geometry) < 7:
        return None
    state = {}
    for i, p in geometry.items():
        neighbors = {(r+dr, c+dc) for r, c in _cells(p)
                     for dr, dc in ((0,0),(0,1),(0,-1),(1,0),(-1,0))}
        forbidden = {q[4] for q in state.values() if neighbors & set(_cells(q))}
        choices = [c for c in opaque if c not in forbidden]
        if payload not in forbidden and (rng.random() < .4 or not choices):
            color = payload
        elif choices:
            color = int(rng.choice(choices))
        else:
            return None
        state[i] = (*p[:4], color)
    static = np.full(wall.shape, bg, dtype=np.int8)
    static[wall] = wall_color
    chamber_cells = [(r, c) for r, c in map(tuple, np.argwhere(footprint)) if c <= mouth]
    static[chamber_cells[int(rng.integers(len(chamber_cells)))]] = payload
    inp = _render(static, state)
    if np.any(inp[0] != bg) or np.count_nonzero(inp == bg) <= inp.size//2:
        return None
    final, rounds = _evolve(state, footprint, payload, construction=True)
    out = _illuminate(static, final, footprint, bg, payload, construction=True)
    mirror = bool(rng.integers(2))
    pair = {"input": inp[:, ::-1].copy() if mirror else inp,
            "output": out[:, ::-1].copy() if mirror else out}
    a = _decode(pair["input"])
    assert np.array_equal(a["footprint"], footprint)
    assert set(a["state"].values()) == set(state.values())
    f = _measure(a)
    if (expanded and f["horizontal_levels"] < 3) or (not expanded and f["horizontal_levels"] > 2):
        return None
    if not (f["removed"] and f["moved"] and f["outside_erased"] and f["gravity_matters"]):
        return None
    if not (f["feedback_matters"] or f["beam_selection_matters"]):
        return None
    assert np.array_equal(solve(pair["input"]), pair["output"])
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 0, None, "seed")
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    shared_palette = bool(rng.random() < .25)
    redundant = bool(rng.random() < .5)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for j in range(count):
            curtain = bool(j == 1) if split == "train" and j < 2 else bool(rng.integers(2))
            stages = 1 if split == "train" else int(rng.integers(2, 4))
            free_network = (split == "test" or j >= 2) and rng.random() < .65
            for attempt in range(64):
                colors = palette if shared_palette else rng.permutation(10)
                # A bounded fallback preserves the rule and training coverage.
                pair = (_free_construct(rng, colors, expanded=split == "test") if free_network and attempt < 40
                        else _construct(rng, stages, curtain, redundant, colors))
                if pair is None:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            a = _decode(inp)
            expected, final, rounds = _answer(a)
            canonical_out = out[:, ::-1] if a["mirrored"] else out
            assert np.array_equal(expected, canonical_out), "incorrect output"
            direct, direct_rounds = _evolve(a["state"], a["footprint"], a["payload"], construction=True)
            assert direct == final and direct_rounds == rounds
            assert _settle(final, inp.shape[0]) == final
            assert not _hits_scan(final, a["footprint"], a["payload"])
            assert all(final[i][1] >= p[1] and final[i][0] == p[0] and final[i][2:] == p[2:]
                       for i, p in a["state"].items() if i in final)
            assert all(p[4] == a["payload"] for i, p in a["state"].items() if i not in final)
            preserved = a["static"] != a["bg"]
            assert np.array_equal(canonical_out[preserved], a["static"][preserved])
            for color in {p[4] for p in a["state"].values()} - {a["payload"]}:
                assert np.sum(a["grid"] == color) == np.sum(canonical_out == color)
            f = _measure(a)
            assert f["moved"] and f["removed"] and f["outside_erased"]
            if split == "train":
                training.append(f)
    assert any(f["feedback_matters"] and f["late_removed"] for f in training)
    assert any(f["shielded"] and f["beam_selection_matters"] for f in training)
    assert all(f["gravity_matters"] for f in training)


def _mechanics_test():
    # Explicit small mechanics fixtures, independent of random scene placement.
    footprint = np.ones((10, 12), dtype=bool)
    rods = {0: ("H", 3, 1, 3, 4), 1: ("V", 4, 1, 6, 2),
            2: ("H", 3, 7, 2, 2), 3: ("V", 4, 8, 6, 5)}
    end, rounds = _evolve(rods, footprint, 2)
    assert rounds == [(1,), (2,)] and end[0][1] == 9 and end[3] == rods[3]
    # A falling upper upright makes a new shadow where the beam used to sit.
    rods[4] = ("V", 0, 2, 3, 6)
    rods[5] = ("V", 6, 3, 4, 5)  # short landing: beam drops two rows
    end, rounds = _evolve(rods, footprint, 2)
    assert rounds == [(1,)] and 2 in end and end[4][1] == 2
    # Redundant supports are screened in sequence; one deletion does not release.
    rods = {0: ("H", 3, 1, 4, 4), 1: ("V", 4, 1, 6, 2),
            2: ("V", 4, 4, 6, 2)}
    first, _ = _evolve(rods, footprint, 2, one_pass=True)
    assert first[0] == rods[0]
    end, rounds = _evolve(rods, footprint, 2)
    assert rounds == [(1,), (2,)] and end[0][1] == 9
    # Thin boundary phase is extrapolated without adding thick corner pixels.
    assert _fit([7, 7, 6, 6, 5, 5, 4], -1)[:3] == (2, 0, 4)


def self_test():
    """Bounded: stream 42 puzzles, then all 30 legal explicit count pairs.

    No files/logs are produced; only one puzzle and small aggregate sets live
    at once. No independent semantic certification is claimed.
    """
    _mechanics_test()
    signatures, shapes, layouts, periods, rod_counts, rounds = set(), set(), set(), set(), set(), set()
    distances, backgrounds, directions = set(), set(), set()
    shielded = late = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            a = _decode(pair["input"])
            f = _measure(a)
            shapes.add(f["shape"])
            rod_counts.add(f["rods"])
            rounds.add(f["rounds"])
            periods.update(f["periods"])
            distances.update(f["distances"])
            backgrounds.add(a["bg"])
            directions.add(a["mirrored"])
            shielded += bool(f["shielded"])
            late += bool(f["late_removed"])
            normalized = tuple((p[0], p[1], p[2]-a["mouth"], p[3]) for p in a["state"].values())
            layouts.add(hashlib.sha256(repr(normalized).encode()).digest())
            assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
        if seed in (0, 17, 41):
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 35 and len(rod_counts) >= 7 and len(rounds) >= 4
    assert periods == {1, 2, 3} and directions == {False, True}
    assert len(backgrounds) == 10 and len(distances) >= 10
    assert shielded >= 42 and late >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(100+a*10+b, a, b))
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(123, 2, 1)
    p["test"][0]["output"] = p["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
