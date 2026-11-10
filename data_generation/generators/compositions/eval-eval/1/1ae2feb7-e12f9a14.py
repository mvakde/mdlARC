"""Framed run-length clocks select which perpendicular rays bend together.

The solver parses the input alone. Geometry is traced through dark pulse gaps;
the pulse counter is distance along that geometry, starting at zero at a port.
No parent generator, seed, saved pair, or construction state is used by solve.
"""
from itertools import combinations
import json
from pathlib import Path

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _inside(point, shape):
    return all(0 <= x < n for x, n in zip(point, shape))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _as_grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(np.all((a >= 0) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _parse(value):
    a = _as_grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background must be unique")
    bg = int(counts.argmax())
    remaining = set(zip(*np.where(a != bg)))
    sources = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    pending.append(point)
        r0 = int(min(r for r, _ in component))
        r1 = int(max(r for r, _ in component))
        c0 = int(min(c for _, c in component))
        c1 = int(max(c for _, c in component))
        h, w = r1 - r0 + 1, c1 - c0 + 1
        _require(h >= 3 and w >= 3 and (h == 3 or w == 3), "core must be a thin bar")
        core = a[r0 + 1:r1, c0 + 1:c1]
        color = int(core.flat[0])
        _require(color != bg and np.all(core == color), "core must be a solid run")
        period = int(core.size)
        _require(1 <= period <= 8, "bar length must be 1..8")
        perimeter = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                     if r in (r0, r1) or c in (c0, c1)]
        gaps = [p for p in perimeter if int(a[p]) == bg]
        ring_colors = {int(a[p]) for p in perimeter if int(a[p]) != bg}
        _require(len(gaps) == 1 and len(ring_colors) == 1, "frame must have exactly one opening")
        ring = ring_colors.pop()
        _require(ring != color, "frame and bar colors must differ")
        port = gaps[0]
        r, c = port
        _require(not (r in (r0, r1) and c in (c0, c1)), "corner openings are outside this family")
        if r == r0:
            direction = (-1, 0)
        elif r == r1:
            direction = (1, 0)
        elif c == c0:
            direction = (0, -1)
        else:
            direction = (0, 1)
        expected_component = {(r, c) for r in range(r0, r1 + 1)
                              for c in range(c0, c1 + 1)} - {port}
        _require(component == expected_component, "unaccounted object cells")
        sources.append({"box": (r0, c0, r1, c1), "port": port, "direction": direction,
                        "period": period, "color": color, "ring": ring,
                        "axis": "H" if h == 3 else "V"})
    sources.sort(key=lambda s: s["box"])
    _require(1 <= len(sources) <= 6, "need 1..6 emitters")
    _require(len({s["ring"] for s in sources}) == 1, "frames must share a structural color")
    for x, y in combinations(sources, 2):
        r0, c0, r1, c1 = x["box"]
        s0, d0, s1, d1 = y["box"]
        _require(r1 + 1 < s0 or s1 + 1 < r0 or c1 + 1 < d0 or d1 + 1 < c0,
                 "frames require a background moat")
    return {"grid": a, "background": bg, "ring": sources[0]["ring"], "sources": sources}


def _walk(point, direction, shape):
    result = []
    while _inside(point, shape):
        result.append(point)
        point = tuple(x + d for x, d in zip(point, direction))
    return result


def _crosses_between(a, b):
    # Only diagonal segments can cross between cells.
    diagonals = {((p[0] + q[0], p[1] + q[1]), (q[0] - p[0]) * (q[1] - p[1]))
                 for p, q in zip(a, a[1:]) if p[0] != q[0] and p[1] != q[1]}
    return any(((p[0] + q[0], p[1] + q[1]), -(q[0] - p[0]) * (q[1] - p[1])) in diagonals
               for p, q in zip(b, b[1:]) if p[0] != q[0] and p[1] != q[1])


def _trace(info):
    """Trace spatial carriers. A member stores its own conserved path age.

    Mutual earliest resonances are processed, so unequal approach distances
    never imply a synchronized global clock. Joined diagonal bundles are
    terminal in the specified domain; a later geometric contact is rejected.
    """
    a, sources = info["grid"], info["sources"]
    groups = [{"direction": s["direction"], "members": [(i, s["port"], 0)]}
              for i, s in enumerate(sources)]
    routes = [[] for _ in sources]
    turns, events = {}, []

    def append(owner, path):
        if routes[owner] and path and routes[owner][-1] == path[0]:
            path = path[1:]
        routes[owner].extend(path)

    for _ in range(len(sources) + 1):
        forecasts = [[_walk(p, g["direction"], a.shape) for _, p, _ in g["members"]]
                     for g in groups]
        candidates = []
        for i, j in combinations(range(len(groups)), 2):
            g, h = groups[i], groups[j]
            for mi, pa in enumerate(forecasts[i]):
                indices = {point: t for t, point in enumerate(pa)}
                for mj, pb in enumerate(forecasts[j]):
                    common = [(indices[p], t, p) for t, p in enumerate(pb) if p in indices]
                    _require(not _crosses_between(pa, pb), "diagonal bundles must not cross other carriers")
                    if not common:
                        continue
                    _require(len(g["members"]) == len(h["members"]) == 1,
                             "joined bundles must have no subsequent contact")
                    u, v = g["direction"], h["direction"]
                    _require(sum(abs(d) for d in u) == sum(abs(d) for d in v) == 1
                             and u[0] * v[0] + u[1] * v[1] == 0,
                             "only perpendicular cardinal crossings are supported")
                    _require(len(common) == 1, "overlapping carriers")
                    ta, tb, point = common[0]
                    _require(min(ta, tb) > 0, "crossing at a port")
                    ia, _, agea = g["members"][mi]
                    ib, _, ageb = h["members"][mj]
                    brighta = (agea + ta) % sources[ia]["period"] == 0
                    brightb = (ageb + tb) % sources[ib]["period"] == 0
                    if brighta and brightb:
                        candidates.append((i, j, ta, tb, point))
        if not candidates:
            for g, paths in zip(groups, forecasts):
                for (owner, _, _), path in zip(g["members"], paths):
                    append(owner, path)
            break
        nearest = {}
        for i, j, ta, tb, _ in candidates:
            nearest[i] = min(nearest.get(i, 1000), ta)
            nearest[j] = min(nearest.get(j, 1000), tb)
        ready = [e for e in candidates if e[2] == nearest[e[0]] and e[3] == nearest[e[1]]]
        _require(bool(ready), "cyclic encounter order")
        i, j, ta, tb, point = min(ready, key=lambda e: (max(e[2:4]), sum(e[2:4]), e[0], e[1]))
        for k, l, x, y, _ in candidates:
            if (k, l) != (i, j):
                _require(not ((k in (i, j) and x == nearest[k]) or
                              (l in (i, j) and y == nearest[l])), "ambiguous multiway resonance")
        u, v = groups[i]["direction"], groups[j]["direction"]
        heading = (u[0] + v[0], u[1] + v[1])
        members, incoming, ages = [], [], []
        for index, t in ((i, ta), (j, tb)):
            owner, _, age = groups[index]["members"][0]
            path = forecasts[index][0]
            append(owner, path[:t])  # omit the crossing; stop at last separate cell
            turns[owner] = len(routes[owner]) - 1
            members.append((owner, path[t - 1], age + t - 1))
            incoming.append(owner)
            ages.append(age + t)
        events.append({"owners": tuple(incoming), "crossing": point, "distances": tuple(ages),
                       "outgoing": heading})
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]
        groups.append({"direction": heading, "members": members})
    else:
        raise AssertionError("network did not terminate")

    # Inspect the completed routes, including contacts with already traced prefixes.
    occupied = {}
    for owner, route in enumerate(routes):
        _require(len(route) == len(set(route)), "self-intersecting carrier")
        for t, point in enumerate(route):
            _require(int(a[point]) == info["background"], "carrier hits a source frame or core")
            occupied.setdefault(point, []).append((owner, t))
    passes = []
    for point, contacts in occupied.items():
        if len(contacts) < 2:
            continue
        _require(len(contacts) == 2, "multiway carrier crossing")
        (i, ti), (j, tj) = contacts
        _require((i not in turns or ti <= turns[i]) and (j not in turns or tj <= turns[j]),
                 "a diagonal bundle touches a previous carrier")
        u, v = sources[i]["direction"], sources[j]["direction"]
        _require(u[0] * v[0] + u[1] * v[1] == 0, "parallel or head-on carrier overlap")
        bright = (ti % sources[i]["period"] == 0, tj % sources[j]["period"] == 0)
        _require(not all(bright), "unresolved coincident pulses")
        passes.append({"owners": (i, j), "crossing": point, "distances": (ti, tj), "bright": bright})
    for i, j in combinations(range(len(routes)), 2):
        _require(not _crosses_between(routes[i], routes[j]), "unresolved between-cell crossing")
    return {"routes": routes, "turns": turns, "events": events, "passes": passes}


def _paint(info, network, reset=False, continuous=False):
    out = info["grid"].copy()
    for owner, (s, route) in enumerate(zip(info["sources"], network["routes"])):
        turn = network["turns"].get(owner)
        for t, point in enumerate(route):
            age = t - turn if reset and turn is not None and t >= turn else t
            if continuous or age % s["period"] == 0:
                out[point] = s["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _paint(info, _trace(info))


def _straight_answer(info):
    # A defined rival: straight periodic paths, lower literal color wins ties.
    out = info["grid"].copy()
    assigned = {}
    for s in info["sources"]:
        for t, point in enumerate(_walk(s["port"], s["direction"], out.shape)):
            if t % s["period"] == 0:
                assigned[point] = min(assigned.get(point, 10), s["color"])
    for point, color in assigned.items():
        out[point] = color
    return out


def _emitter(port, heading, period, axis, slot, color):
    r, c = port
    if heading == "E":
        core = (r, c - period, 1, period) if axis == "H" else (r - slot, c - 1, period, 1)
    elif heading == "N":
        core = (r + 1, c, period, 1) if axis == "V" else (r + 1, c - slot, 1, period)
    else:
        raise ValueError("construction uses E/N before whole-grid transformation")
    return {"core": core, "port": port, "color": color}


def _build(shape, specs):
    a = np.zeros(shape, dtype=np.int8)
    for s in specs:
        r, c, h, w = s["core"]
        _require(r >= 1 and c >= 1 and r + h < shape[0] and c + w < shape[1], "clipped frame")
        _require(np.all(a[r - 1:r + h + 1, c - 1:c + w + 1] == 0), "overlapping frames")
        a[r - 1:r + h + 1, c - 1:c + w + 1] = 8
        a[r:r + h, c:c + w] = s["color"]
        a[s["port"]] = 0
    return a


def _make_crossing(rng, p, q, mode, candidates=1):
    """Sample coordinates from arithmetic constraints, not saved puzzle layouts."""
    for _ in range(100):
        row = int(rng.integers(max(6, q + 1), 12))
        eaxis = str(rng.choice(["H", "V"]))
        port_col = (p + 1 if eaxis == "H" else 2) + int(rng.integers(0, 4))
        room = 30 - port_col - max(p, q) - 2
        if mode == "resonant":
            east_bright = [True]
        elif mode == "pass" and candidates == 2:
            east_bright = [True, False] if rng.integers(2) else [False, True]
        elif mode == "choice" and candidates == 3:
            east_bright = ([True, False] if rng.integers(2) else [False, True]) + [True]
        elif mode == "choice":
            east_bright = [bool(rng.integers(2)), True]
        else:
            east_bright = [bool(rng.integers(2))]
        choices = [ds for ds in combinations(range(3, room + 1), candidates)
                   if all(b - a >= 4 for a, b in zip(ds, ds[1:]))
                   and all((d % p == 0) == bright for d, bright in zip(ds, east_bright))]
        if not choices:
            continue
        ds = choices[int(rng.integers(len(choices)))]
        specs = [_emitter((row, port_col), "E", p, eaxis,
                          int(rng.integers(p)), 1)]
        last_col, max_bottom = port_col + ds[-1], row + 2
        for index, d in enumerate(ds):
            # Multi-candidate scenes use narrow vertical bars to keep the network readable.
            axis = "V" if candidates > 1 else str(rng.choice(["H", "V"]))
            nperiod = q
            resonant = mode == "resonant" or (mode == "choice" and index == candidates - 1)
            north_bright = True if resonant else not east_bright[index]
            max_b = 28 - row - (nperiod if axis == "V" else 1)
            distances = [b for b in range(3, max_b + 1)
                         if (b % nperiod == 0) == north_bright and
                         (not resonant or b != d)]
            if not distances:
                break
            b = int(rng.choice(distances))
            source = _emitter((row + b, port_col + d), "N", nperiod, axis,
                              int(rng.integers(nperiod)), index + 2)
            specs.append(source)
            cr, cc, ch, cw = source["core"]
            max_bottom = max(max_bottom, cr + ch + 1)
        if len(specs) != candidates + 1:
            continue
        # Bounds cover complete frames and enough post-crossing recurrence.
        min_r = min(s["core"][0] - 1 for s in specs)
        min_c = min(s["core"][1] - 1 for s in specs)
        if min_r < 0 or min_c < 0:
            continue
        hmin = max(s["core"][0] + s["core"][2] + 1 for s in specs)
        wmin = max(last_col + max(p, q) + 2,
                   max(s["core"][1] + s["core"][3] + 1 for s in specs))
        if max(hmin, wmin) > 30:
            continue
        shape = (int(rng.integers(hmin, 31)), int(rng.integers(wmin, 31)))
        try:
            a = _build(shape, specs)
            info = _parse(a)
            network = _trace(info)
            if mode == "pass":
                _require(not network["events"] and len(network["passes"]) == candidates, "missing pass")
            else:
                _require(len(network["events"]) == 1, "missing resonance")
            if mode == "choice":
                _require(len(network["passes"]) == candidates - 1, "missing candidate crossings")
            _require(all(sum(e["bright"]) == 1 for e in network["passes"]), "pass needs one bright ray")
            return a
        except AssertionError:
            continue
    raise RuntimeError("bounded crossing construction exhausted")


def _make_single(rng, period):
    # A single source can be near any edge after rotation/reflection.
    axis = str(rng.choice(["H", "V"]))
    row = int(rng.integers(period + 1, period + 7))
    col = (period + 1 if axis == "H" else 2) + int(rng.integers(0, 5))
    s = _emitter((row, col), "E", period, axis, int(rng.integers(period)), 1)
    cr, cc, ch, cw = s["core"]
    hmin = max(cr + ch + 1, row + 2)
    wmin = max(cc + cw + 1, col + 2 * period + 1)
    return _build((int(rng.integers(hmin, 31)), int(rng.integers(wmin, 31))), [s])


def _counts(rng, num_train, num_test):
    for value, lower, upper in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                 not isinstance(value, (int, np.integer)) or
                                 not lower <= int(value) <= upper):
            raise ValueError("illegal example count")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    nt, ne = _counts(rng, num_train, num_test)
    p, q = [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 6)][int(rng.integers(6))]
    if rng.integers(2):
        p, q = q, p
    preferred = rng.permutation(10).astype(np.int8)
    # Canonical foreground uses 1..4; 0/8 are only construction role names.
    rotation = int(rng.integers(4))
    stable_palette = bool(rng.integers(2))
    modes = ["resonant", "pass"]
    modes += [str(rng.choice(["single", "pass", "resonant"], p=[.45, .3, .25]))
              for _ in range(nt - 2)]
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            for attempt in range(40):
                if split == "test":
                    count_crossings = int(rng.choice([2, 3]))
                    a = _make_crossing(rng, p, q, "choice", count_crossings)
                elif modes[index] == "single":
                    a = _make_single(rng, int(rng.choice([1, p, q])))
                else:
                    # Swap the length-to-color association across demonstrations.
                    x, y = (p, q) if index % 2 == 0 else (q, p)
                    a = _make_crossing(rng, x, y, modes[index], 2 if modes[index] == "pass" else 1)
                palette = preferred if stable_palette or rng.random() < .35 else rng.permutation(10)
                a = palette[a]
                turn = (rotation + (index % 2) * 2) % 4 if split == "train" else int(rng.integers(4))
                a = np.rot90(a, turn)
                if rng.integers(2):
                    a = np.fliplr(a)
                a = np.ascontiguousarray(a, dtype=np.int8)
                if _key(a) in seen:
                    continue
                info = _parse(a)
                network = _trace(info)
                pair = {"input": a, "output": _paint(info, network)}
                puzzle[split].append(pair)
                seen.add(_key(a))
                break
            else:
                raise RuntimeError("could not make a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pass_witness(info, network, e):
    if sum(e["bright"]) != 1:
        return False
    # Both rays have an observable later pulse, excluding stop-at-any-contact.
    return all(any(t > distance and t % info["sources"][owner]["period"] == 0
                   for t in range(len(network["routes"][owner])))
               for owner, distance in zip(e["owners"], e["distances"]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal counts")
    seen, taught, needed = set(), set(), set()
    resonant_lesson, pass_lesson, reset_rejected, unconditional_rejected = False, False, False, False
    two_sided_gate = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _as_grid(pair["input"]), _as_grid(pair["output"])
            _require(a.shape == b.shape, "shape changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _parse(a)
            n = _trace(info)
            _require(np.array_equal(_paint(info, n), b), "incorrect pulses, phase, bend, or preservation")
            fixed = a != info["background"]
            _require(np.array_equal(a[fixed], b[fixed]), "input foreground changed")
            _require(np.any(a != b), "identity example")
            for s, route in zip(info["sources"], n["routes"]):
                if split == "train":
                    _require(len(route) > s["period"], "training needs recurrence for every source")
                    _require(b[route[0]] == b[route[s["period"]]] == s["color"], "missing recurrence")
                    if s["period"] > 1:
                        _require(b[route[1]] == info["background"], "pulse gap obscured")
                    taught.add(s["period"])
                else:
                    needed.add(s["period"])
            for e in n["events"]:
                _require(e["crossing"] not in set().union(*(set(r) for r in n["routes"])),
                         "resonant crossing must be bypassed")
                _require(b[e["crossing"]] == info["background"], "crossing cell should remain blank")
                _require(all(t % info["sources"][i]["period"] == 0
                             for i, t in zip(e["owners"], e["distances"])), "nonresonant bend")
            if split == "train":
                _require(len(info["sources"]) <= 3, "training candidate domain changed")
                _require(not (n["events"] and n["passes"]), "training must teach separate encounter outcomes")
                resonant_lesson |= any(e["distances"][0] != e["distances"][1] and
                                      len({info["sources"][i]["period"] for i in e["owners"]}) == 2
                                      for e in n["events"])
                pass_lesson |= any(_pass_witness(info, n, e) for e in n["passes"])
                if len(n["passes"]) == 2:
                    common = set(n["passes"][0]["owners"]) & set(n["passes"][1]["owners"])
                    if len(common) == 1:
                        owner = common.pop()
                        two_sided_gate |= {e["bright"][e["owners"].index(owner)]
                                           for e in n["passes"]} == {False, True}
                reset_rejected |= not np.array_equal(_paint(info, n, reset=True), b)
                # Any unconditional geometric bend must bypass this crossing.
                # A visible pulse there is a concrete contradictory output cell;
                # no extra collision convention is needed to complete a rival.
                unconditional_rejected |= any(sum(e["bright"]) == 1 and
                                               b[e["crossing"]] != info["background"]
                                               for e in n["passes"])
            else:
                _require(3 <= len(info["sources"]) <= 4 and len(n["events"]) == 1, "test needs candidate network")
                _require(len(n["passes"]) == len(info["sources"]) - 2, "test must pass earlier candidates")
                _require(all(_pass_witness(info, n, e) for e in n["passes"]), "unobservable failed candidate")
                event = n["events"][0]
                # Every failed crossing shares the eventual turning ray and lies earlier on it.
                for e in n["passes"]:
                    common = set(e["owners"]) & set(event["owners"])
                    _require(len(common) == 1, "disconnected candidate network")
                    owner = common.pop()
                    _require(e["distances"][e["owners"].index(owner)] <
                             event["distances"][event["owners"].index(owner)], "candidate must occur before turn")
                _require(not np.array_equal(_straight_answer(info), b), "ignoring B must change the test")
    _require(len(taught - {1}) >= 2 and needed <= taught, "numerical recurrence teaching gap")
    _require(resonant_lesson and pass_lesson, "missing resonance or pass-through teaching")
    _require(two_sided_gate, "must show either incoming ray alone is insufficient")
    _require(reset_rejected, "phase conservation is not visibly taught")
    _require(unconditional_rejected, "ignoring A's resonance gate must change training")


def _features(pair):
    info = _parse(pair["input"])
    n = _trace(info)
    s = info["sources"]
    return {"shape": tuple(info["grid"].shape), "count": len(s),
            "periods": tuple(sorted({x["period"] for x in s})),
            "axes": tuple(x["axis"] for x in s), "ports": tuple(x["direction"] for x in s),
            "roles": (info["background"], info["ring"]), "events": len(n["events"]),
            "passes": len(n["passes"]), "distances": tuple(e["distances"] for e in n["events"]),
            "layout": (tuple(info["grid"].shape), tuple((x["box"], x["port"], x["period"]) for x in s))}


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _fixed_checks():
    # Explicit small arithmetic fixtures. Expected pixels are enumerated here,
    # not created by solve or by the network tracer.
    specs = [_emitter((6, 3), "E", 2, "H", 0, 1),
             _emitter((12, 7), "N", 3, "V", 0, 2)]
    a = _build((17, 14), specs)
    expected = a.copy()
    one = [(6, 3), (6, 5), (5, 7), (3, 9), (1, 11)]
    two = [(12, 7), (9, 7), (6, 8), (3, 11)]
    for point in one:
        expected[point] = 1
    for point in two:
        expected[point] = 2
    _require(np.array_equal(solve(a), expected), "fixed unequal-distance resonance fixture")
    for turn in range(4):
        for reflection in (False, True):
            before, after = np.rot90(a, turn), np.rot90(expected, turn)
            if reflection:
                before, after = np.fliplr(before), np.fliplr(after)
            _require(np.array_equal(solve(before), after), "resonant directional equivariance")
    # Shift only the vertical port by one: 5 mod 3 != 0, so both rays remain straight.
    a = _build((17, 14), [specs[0], _emitter((11, 7), "N", 3, "V", 0, 2)])
    expected = a.copy()
    for c in (3, 5, 7, 9, 11, 13):
        expected[6, c] = 1
    for r in (11, 8, 5, 2):
        expected[r, 7] = 2
    _require(np.array_equal(solve(a), expected), "fixed nonresonant fixture")
    for turn in range(4):
        for reflection in (False, True):
            before, after = np.rot90(a, turn), np.rot90(expected, turn)
            if reflection:
                before, after = np.fliplr(before), np.fliplr(after)
            _require(np.array_equal(solve(before), after), "directional equivariance")
    # A complete frame flush against the left edge remains a normal source.
    a = _build((5, 15), [_emitter((2, 5), "E", 4, "H", 0, 1)])
    expected = a.copy()
    expected[2, [5, 9, 13]] = 1
    _require(np.array_equal(solve(a), expected), "edge and clipping fixture")
    for period in (7, 8):
        a = _build((7, 30), [_emitter((3, period + 2), "E", period, "H", 0, 1)])
        expected = a.copy()
        for c in range(period + 2, 30, period):
            expected[3, c] = 1
        _require(np.array_equal(solve(a), expected), "long-period fixture")
    # Neither pulse is bright at (6,8): no output pixel marks that crossing.
    a = _build((17, 15), [_emitter((6, 3), "E", 2, "H", 0, 1),
                         _emitter((11, 8), "N", 3, "V", 0, 1)])
    expected = a.copy()
    expected[6, [3, 5, 7, 9, 11, 13]] = 1
    expected[[11, 8, 5, 2], 8] = 1
    _require(np.array_equal(solve(a), expected), "neither-bright and repeated-color fixture")
    # Two disjoint resonances plus two isolated sources exercise the full count
    # domain. Fixed module pixels are composed with its reflected counterpart.
    base = _build((14, 14), [_emitter((2, 3), "E", 2, "H", 0, 1),
                           _emitter((8, 7), "N", 3, "V", 0, 2)])
    base_expected = base.copy()
    for point in ((2, 3), (2, 5), (1, 7)):
        base_expected[point] = 1
    for point in ((8, 7), (5, 7), (2, 8)):
        base_expected[point] = 2
    a = _build((28, 30), [_emitter((4, 25), "E", 1, "H", 0, 3),
                         _emitter((20, 25), "E", 8, "V", 7, 4)])
    a[:, :14] = np.vstack((base, np.rot90(base, 2)))
    expected = a.copy()
    expected[:, :14] = np.vstack((base_expected, np.rot90(base_expected, 2)))
    expected[4, 25:30] = 3
    expected[20, 25] = 4
    _require(len(_trace(_parse(a))["events"]) == 2, "independent resonance count")
    _require(np.array_equal(solve(a), expected), "six-source independent-network fixture")


def self_test():
    _fixed_checks()
    puzzles, features = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        features += [_features(pair) for split in puzzle.values() for pair in split]
    _require(_equal(puzzles[0], generate_puzzle(0)), "nondeterministic default puzzle")
    _require(len({tuple(_key(p["input"]) for split in x.values() for p in split)
                  for x in puzzles}) == 42, "seed diversity collapsed")
    _require(len({f["shape"] for f in features}) >= 35, "shape variation collapsed")
    _require(len({f["layout"] for f in features}) >= 150, "structural variation collapsed")
    _require({1, 2, 3, 4} <= {f["count"] for f in features}, "source-count variation collapsed")
    _require({1, 2, 3, 4, 5, 6} <= set().union(*(set(f["periods"]) for f in features)), "period range collapsed")
    _require({(1, 0), (-1, 0), (0, 1), (0, -1)} <=
             set().union(*(set(f["ports"]) for f in features)), "port directions collapsed")
    _require({"H", "V"} <= set().union(*(set(f["axes"]) for f in features)), "bar orientations collapsed")
    _require(len({f["roles"] for f in features}) >= 30, "palette variation collapsed")
    _require({0, 1, 2} <= {f["passes"] for f in features}, "candidate-count variation collapsed")
    _require(len({d for f in features for d in f["distances"]}) >= 25, "approach geometry collapsed")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(900 + nt * 10 + ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "count mismatch")
    for counts in ((None, 1), (None, 4), (2, None), (10, None), (None, None)):
        p = generate_puzzle(707, *counts)
        _require(_equal(p, generate_puzzle(707, *counts)), "partial-count determinism")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ("train", "test")}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    # A compact artifact records structural measurements, not semantic approval.
    report = {"seeds": 42, "pairs": len(features), "shapes": len({f["shape"] for f in features}),
              "layouts": len({f["layout"] for f in features}),
              "source_counts": sorted({f["count"] for f in features}),
              "periods": sorted(set().union(*(set(f["periods"]) for f in features))),
              "role_palettes": len({f["roles"] for f in features}),
              "resonance_distances": len({d for f in features for d in f["distances"]}),
              "verification": "generator-owned invariants and fixed fixtures only"}
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    self_test()
    print(Path("scratch/self_test.json").read_text().strip())
