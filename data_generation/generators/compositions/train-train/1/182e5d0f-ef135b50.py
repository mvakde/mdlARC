"""Signal islands and retracting tethers. Standard library + NumPy only.

The input-only decoder reconstructs geometry; generation does not supply it state.
See task_spec.md for the complete family and the teaching/transfer constraints.
"""
from collections import deque
import json

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= g) & (g <= 9)), "invalid grid colors")
    return g.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p, shape, diagonal=False):
    for dy, dx in D8 if diagonal else D4:
        q = (p[0] + dy, p[1] + dx)
        if _inside(q, shape):
            yield q


def _dilate(mask, diagonal=False):
    result = mask.copy()
    h, w = mask.shape
    for dy, dx in D8 if diagonal else D4:
        ya, yb = max(0, dy), min(h, h + dy)
        xa, xb = max(0, dx), min(w, w + dx)
        result[ya:yb, xa:xb] |= mask[ya - dy:yb - dy, xa - dx:xb - dx]
    return result


def _components(mask):
    labels = np.zeros(mask.shape, dtype=np.int16)
    groups = []
    for row, col in np.argwhere(mask):
        p = (int(row), int(col))
        if labels[p]:
            continue
        groups.append([])
        number = len(groups)
        labels[p] = number
        stack = [p]
        while stack:
            q = stack.pop()
            groups[-1].append(q)
            for n in _neighbors(q, mask.shape):
                if mask[n] and not labels[n]:
                    labels[n] = number
                    stack.append(n)
    return labels, groups


def _network(rectangles, shape, rowwise=False):
    islands = np.zeros(shape, dtype=bool)
    for a, b, c, d in rectangles:
        islands[a:b + 1, c:d + 1] = True
    bridges = np.zeros(shape, dtype=bool)
    edges = []
    for i, first in enumerate(rectangles):
        for j in range(i + 1, len(rectangles)):
            second = rectangles[j]
            if first[3] < second[2]:
                left, right = first, second
            elif second[3] < first[2]:
                left, right = second, first
            else:
                continue
            top, bottom = max(left[0], right[0]), min(left[1], right[1])
            x0, x1 = left[3] + 1, right[2]
            if top > bottom or x0 >= x1:
                continue
            strip = islands[top:bottom + 1, x0:x1]
            if rowwise:
                clear = ~np.any(strip, axis=1)
                bridges[top:bottom + 1, x0:x1] |= clear[:, None]
            elif not np.any(strip):
                bridges[top:bottom + 1, x0:x1] = True
                edges.append((i, j))
    labels, groups = _components(islands | bridges)
    return {"islands": islands, "bridges": bridges, "field": islands | bridges,
            "labels": labels, "groups": groups, "edges": edges}


def _socket_candidates(g):
    h, w = g.shape
    found = []
    for side in range(4):
        length = w if side < 2 else h
        for t in range(1, length - 1):
            if side == 0:
                center, caps, first = (0, t), ((0, t - 1), (0, t + 1)), (1, t)
            elif side == 1:
                center, caps, first = (h - 1, t), ((h - 1, t - 1), (h - 1, t + 1)), (h - 2, t)
            elif side == 2:
                center, caps, first = (t, 0), ((t - 1, 0), (t + 1, 0)), (t, 1)
            else:
                center, caps, first = (t, w - 1), ((t - 1, w - 1), (t + 1, w - 1)), (t, w - 2)
            if not _inside(first, g.shape):
                continue
            cap, wire = int(g[caps[0]]), int(g[center])
            if cap and wire and cap != wire and g[caps[1]] == cap and g[first] == wire:
                found.append({"side": side, "center": center, "caps": caps,
                              "first": first, "roles": (cap, wire)})
    return found


def _decode(input_grid):
    g = _grid(input_grid)
    sockets = _socket_candidates(g)
    _require(sockets and len({s["roles"] for s in sockets}) == 1, "ambiguous socket palette")
    neutral, wire = sockets[0]["roles"]
    cap_cells = {p for s in sockets for p in s["caps"]}
    _require(len(cap_cells) == 2 * len(sockets), "overlapping socket caps")
    wire_labels, wire_groups = _components(g == wire)
    _require(len(wire_groups) == len(sockets), "unrooted or shared tether")
    tethers, used = [], set()
    for socket in sockets:
        center = socket["center"]
        number = int(wire_labels[center])
        _require(number not in used, "two sockets share a tether")
        used.add(number)
        cells = set(wire_groups[number - 1])
        degrees = {p: [q for q in _neighbors(p, g.shape) if q in cells] for p in cells}
        _require(len(degrees[center]) == 1 and all(1 <= len(v) <= 2 for v in degrees.values()),
                 "tether must be an unbranched socket-rooted path")
        previous, current, path = None, center, []
        while True:
            following = [q for q in degrees[current] if q != previous]
            if not following:
                break
            _require(len(following) == 1 and following[0] not in path, "tether cycle")
            previous, current = current, following[0]
            path.append(current)
        _require(len(path) == len(cells) - 1 and path[0] == socket["first"], "invalid tether traversal")
        steps = []
        previous = center
        for p in path:
            d = (p[0] - previous[0], p[1] - previous[1])
            if not steps or steps[-1][0] != d:
                steps.append([d, 1])
            else:
                steps[-1][1] += 1
            previous = p
        _require(len(steps) == 2 and sum(a * b for a, b in zip(steps[0][0], steps[1][0])) == 0,
                 "tether must turn exactly once")
        tethers.append(dict(socket, path=path, endpoint=path[-1], legs=tuple(s[1] for s in steps)))
    mask = (g != 0) & (g != wire)
    for p in cap_cells:
        mask[p] = False
    _, groups = _components(mask)
    rectangles, beacons = [], []
    for cells in groups:
        top, bottom = min(p[0] for p in cells), max(p[0] for p in cells)
        left, right = min(p[1] for p in cells), max(p[1] for p in cells)
        _require(bottom > top and right > left and len(cells) == (bottom - top + 1) * (right - left + 1),
                 "island must be a filled rectangle of at least 2 by 2")
        _require(0 < top <= bottom < g.shape[0] - 1 and 0 < left <= right < g.shape[1] - 1,
                 "islands must be interior")
        colored = [p for p in cells if g[p] != neutral]
        _require(len(colored) <= 1, "multiple beacons on an island")
        rectangles.append((top, bottom, left, right))
        if colored:
            beacons.append((colored[0], int(g[colored[0]]), len(rectangles) - 1))
    _require(len(rectangles) >= 2 and beacons, "need islands and a visible beacon")
    net = _network(rectangles, g.shape)
    sources, source_islands = {}, {}
    for p, color, index in beacons:
        label = int(net["labels"][p])
        _require(label not in sources, "a signal network may have only one beacon")
        sources[label] = color
        source_islands[label] = index
    for tether in tethers:
        all_cells = [tether["center"]] + tether["path"]
        _require(not any(net["field"][p] for p in all_cells), "tether overlaps latent network")
        contacts = {int(net["labels"][q]) for q in _neighbors(tether["endpoint"], g.shape)
                    if net["labels"][q]}
        _require(len(contacts) <= 1, "endpoint touches different networks")
        tether["contact"] = next(iter(contacts), 0)
        tether["signal"] = sources.get(tether["contact"], 0)
    boundary = set()
    for y, x in np.argwhere(g != 0):
        if y in (0, g.shape[0] - 1) or x in (0, g.shape[1] - 1):
            boundary.add((int(y), int(x)))
    _require(boundary <= cap_cells | {s["center"] for s in sockets}, "unmarked boundary material")
    return dict(net, grid=g, neutral=neutral, wire=wire, rectangles=rectangles,
                beacons=beacons, sources=sources, source_islands=source_islands, tethers=tethers)


def _render(data, retract=True):
    out = data["grid"].copy()
    for label, color in data["sources"].items():
        out[data["bridges"] & (data["labels"] == label)] = color
    if retract:
        for tether in data["tethers"]:
            if tether["signal"]:
                for p in tether["path"]:
                    out[p] = 0
                out[tether["path"][0]] = tether["signal"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct bridges, broadcast input beacons, and retract exact contacts."""
    return _render(_decode(input_grid))


def _without_bridging(data):
    """Causal ablation: only a beacon's own island can activate a tip."""
    out = data["grid"].copy()
    for tether in data["tethers"]:
        neighbors = list(_neighbors(tether["endpoint"], out.shape))
        colors = []
        for _, color, index in data["beacons"]:
            a, b, c, d = data["rectangles"][index]
            if any(a <= y <= b and c <= x <= d for y, x in neighbors):
                colors.append(color)
        if colors:
            _require(len(set(colors)) == 1, "ambiguous ablated contact")
            for p in tether["path"]:
                out[p] = 0
            out[tether["path"][0]] = colors[0]
    return out


def _distances(n, edges, source):
    adjacency = [set() for _ in range(n)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    result, queue = {source: 0}, deque([source])
    while queue:
        a = queue.popleft()
        for b in sorted(adjacency[a]):
            if b not in result:
                result[b] = result[a] + 1
                queue.append(b)
    return result


def _rect_groups(rects, net):
    result = {}
    for i, (a, b, c, d) in enumerate(rects):
        result.setdefault(int(net["labels"][a, c]), []).append(i)
    return result


def _separated(a, b):
    return a[1] + 1 < b[0] or b[1] + 1 < a[0] or a[3] + 1 < b[2] or b[3] + 1 < a[2]


def _sample_layout(rng, kind):
    if kind == "blocked":
        h, w = int(rng.integers(22, 31)), int(rng.integers(22, 31))
        widths = [int(rng.integers(2, 5)) for _ in range(3)]
        gaps = [int(rng.integers(2, 5)) for _ in range(2)]
        total = sum(widths) + sum(gaps)
        if total > w - 6:
            return None
        x = int(rng.integers(3, w - total - 2))
        y = int(rng.integers(9, h - 9))
        ha, hb = int(rng.integers(5, 8)), int(rng.integers(5, 8))
        yb = y + int(rng.integers(-1, 2))
        lo, hi = max(y, yb), min(y + ha - 1, yb + hb - 1)
        mid_top = int(rng.integers(lo, hi - 1))
        mid_bottom = int(rng.integers(mid_top + 1, hi))
        xm = x + widths[0] + gaps[0]
        xr = xm + widths[1] + gaps[1]
        rects = [(y, y + ha - 1, x, x + widths[0] - 1),
                 (mid_top, mid_bottom, xm, xm + widths[1] - 1),
                 (yb, yb + hb - 1, xr, xr + widths[2] - 1)]
        ch = int(rng.integers(2, 4))
        cw, cw2, cg = int(rng.integers(2, 5)), int(rng.integers(2, 5)), int(rng.integers(2, 6))
        cy = int(rng.integers(3, y - ch - 2))
        cx = int(rng.integers(3, w - cw - cw2 - cg - 2))
        rects.append((cy, cy + ch - 1, cx, cx + cw - 1))
        cx2 = cx + cw + cg
        rects.append((cy, cy + ch - 1, cx2, cx2 + cw2 - 1))
        if rng.random() < .5:
            rects = [(h - 1 - b, h - 1 - a, c, d) for a, b, c, d in rects]
        if rng.random() < .5:
            rects = [(a, b, w - 1 - d, w - 1 - c) for a, b, c, d in rects]
        return (h, w), rects
    h, w = int(rng.integers(18, 31)), int(rng.integers(18, 31))
    count = int(rng.integers(4 if kind == "dual" else 5, 8))
    rects = []
    for _ in range(count):
        for attempt in range(70):
            rh, rw = int(rng.integers(2, 7)), int(rng.integers(2, 6))
            if h - rh - 5 <= 3 or w - rw - 5 <= 3:
                continue
            y, x = int(rng.integers(3, h - rh - 2)), int(rng.integers(3, w - rw - 2))
            r = (y, y + rh - 1, x, x + rw - 1)
            if all(_separated(r, old) for old in rects):
                rects.append(r)
                break
        else:
            return None
    return (h, w), rects


def _choose_sources(rng, rects, net, kind):
    groups = _rect_groups(rects, net)
    large = [label for label, members in groups.items() if len(members) >= 2]
    if not large:
        return None
    if kind == "dual" and len(large) < 2:
        return None
    if kind == "blocked":
        label = int(net["labels"][rects[0][0], rects[0][2]])
        if len(groups[label]) != 3 or len(groups) != 2:
            return None
        return {label: int(rng.choice([0, 2]))}
    rng.shuffle(large)
    chosen = large[:2] if kind == "dual" or (len(large) > 1 and rng.random() < .55) else large[:1]
    sources = {}
    for label in chosen:
        options = list(groups[label])
        rng.shuffle(options)
        if kind == "extended" and label == chosen[0]:
            options = [i for i in options if max(_distances(len(rects), net["edges"], i).values()) >= 2]
            if not options:
                return None
        sources[label] = options[0]
    return sources


def _endpoint_options(shape, rects, net, label, style, allowed):
    rowwise = None
    if style == "blocked":
        rowwise = _network(rects, shape, rowwise=True)
        target = rowwise["bridges"] & ~net["field"]
    elif style == "bridge":
        target = net["bridges"] & (net["labels"] == label)
    elif style == "diagonal":
        target = net["field"] & (net["labels"] == label)
    else:
        target = np.zeros(shape, dtype=bool)
        for i in allowed:
            a, b, c, d = rects[i]
            target[a:b + 1, c:d + 1] = True
    choices = _dilate(target, style == "diagonal") & ~net["field"]
    if style == "diagonal":
        choices &= ~_dilate(net["field"])
    elif style == "blocked":
        choices &= ~rowwise["field"] & ~_dilate(net["field"])
    elif style == "bridge":
        choices &= ~_dilate(net["islands"])
    options = []
    for y, x in np.argwhere(choices):
        p = (int(y), int(x))
        labels = {int(net["labels"][q]) for q in _neighbors(p, shape) if net["labels"][q]}
        if style in ("diagonal", "blocked") or labels == {label}:
            options.append(p)
    return options


def _place_tether(rng, shape, rects, net, occupied, job):
    label, style, allowed, required_side = job
    endpoints = _endpoint_options(shape, rects, net, label, style, allowed)
    if not endpoints:
        return None
    endpoints = [endpoints[int(i)] for i in rng.permutation(len(endpoints))]
    unavailable = net["field"] | _dilate(occupied, True)
    h, w = shape
    for endpoint in endpoints:
        if unavailable[endpoint]:
            continue
        sides = [required_side] if required_side is not None else [int(s) for s in rng.permutation(4)]
        for side in sides:
            length = w if side < 2 else h
            for t in rng.permutation(np.arange(1, length - 1)):
                t = int(t)
                y, x = endpoint
                if side < 2:
                    if t == x or y in (0, h - 1):
                        continue
                    cy, step = (0, 1) if side == 0 else (h - 1, -1)
                    center, caps = (cy, t), ((cy, t - 1), (cy, t + 1))
                    path = [(r, t) for r in range(cy + step, y + step, step)]
                    dx = 1 if x > t else -1
                    path += [(y, c) for c in range(t + dx, x + dx, dx)]
                else:
                    if t == y or x in (0, w - 1):
                        continue
                    cx, step = (0, 1) if side == 2 else (w - 1, -1)
                    center, caps = (t, cx), ((t - 1, cx), (t + 1, cx))
                    path = [(t, c) for c in range(cx + step, x + step, step)]
                    dy = 1 if y > t else -1
                    path += [(r, x) for r in range(t + dy, y + dy, dy)]
                cells = [center, *caps, *path]
                if any(unavailable[p] for p in cells):
                    continue
                for p in cells:
                    occupied[p] = True
                return {"side": side, "center": center, "caps": caps, "path": path}
    return None


def _make_pair(rng, palette, kind, teaching_sides=None):
    for attempt in range(700):
        layout = _sample_layout(rng, kind)
        if layout is None:
            continue
        shape, rects = layout
        net = _network(rects, shape)
        sources = _choose_sources(rng, rects, net, kind)
        if sources is None:
            continue
        groups = _rect_groups(rects, net)
        labels = list(sources)
        label = labels[0]
        source = sources[label]
        distances = _distances(len(rects), net["edges"], source)
        remote = [i for i in groups[label] if distances.get(i, -1) >= (2 if kind == "extended" else 1)]
        if not remote:
            continue
        sides = teaching_sides or (None, None)
        if kind == "blocked":
            jobs = [(label, "bridge", [], sides[0]), (label, "diagonal", [], sides[1])]
            jobs.append((label, "blocked", [], None))
            cold = next(k for k in groups if k not in sources)
            jobs.append((cold, "island", groups[cold], None))
        elif kind == "dual":
            second = labels[1]
            remote2 = [i for i in groups[second] if i != sources[second]]
            jobs = [(label, "island", remote, sides[0]), (second, "island", remote2, sides[1]),
                    (label, "diagonal", [], None)]
        else:
            jobs = [(label, "island", remote, None)]
            if len(labels) > 1:
                second = labels[1]
                jobs.append((second, "island", [i for i in groups[second] if i != sources[second]], None))
            else:
                jobs.append((label, "bridge", [], None))
            if rng.random() < .5 and any(k not in sources for k in groups):
                cold = next(k for k in groups if k not in sources)
                jobs.append((cold, "island", groups[cold], None))
            else:
                jobs.append((label, "diagonal", [], None))
            if rng.random() < .25:
                jobs.append((label, "island", groups[label], None))
        occupied = np.zeros(shape, dtype=bool)
        tethers = []
        for job in jobs:
            tether = _place_tether(rng, shape, rects, net, occupied, job)
            if tether is None:
                break
            tethers.append(tether)
        if len(tethers) != len(jobs):
            continue
        neutral, wire, *signal_colors = palette
        g = np.zeros(shape, dtype=np.int8)
        for a, b, c, d in rects:
            g[a:b + 1, c:d + 1] = neutral
        for n, (source_label, index) in enumerate(sources.items()):
            a, b, c, d = rects[index]
            g[int(rng.integers(a, b + 1)), int(rng.integers(c, d + 1))] = signal_colors[n]
        for tether in tethers:
            for p in tether["caps"]:
                g[p] = neutral
            for p in [tether["center"], *tether["path"]]:
                g[p] = wire
        # Decoding is also a structural check; a rejected geometry is resampled.
        try:
            data = _decode(g)
        except ValueError:
            continue
        f = _features(data)
        if kind == "blocked" and not all(f[k] for k in ("blocked", "bridge_tap", "diagonal", "cold", "nonterminal_only", "blocked_tip")):
            continue
        if kind == "dual" and len({t["signal"] for t in data["tethers"] if t["signal"]}) < 2:
            continue
        if kind == "extended" and not f["remote_two"]:
            continue
        return {"input": g, "output": _render(data)}
    raise RuntimeError(f"could not construct {kind} signal network")


def _features(data):
    g, labels = data["grid"], data["labels"]
    rowwise_net = _network(data["rectangles"], g.shape, rowwise=True)
    rowwise = rowwise_net["bridges"]
    rowwise_powered = {int(rowwise_net["labels"][p]) for p, _, _ in data["beacons"]}
    features = {"blocked": bool(np.any(rowwise & ~data["bridges"])),
                "bridge_tap": False, "diagonal": False, "cold": False,
                "indirect": False, "remote_two": False, "nonterminal_only": False,
                "blocked_tip": False,
                "cold_bridge": bool(np.any(data["bridges"] & ~np.isin(labels, list(data["sources"]))))}
    rect_labels = np.zeros(g.shape, dtype=np.int16)
    for i, (a, b, c, d) in enumerate(data["rectangles"]):
        rect_labels[a:b + 1, c:d + 1] = i + 1
    for t in data["tethers"]:
        neighbors = list(_neighbors(t["endpoint"], g.shape))
        if t["signal"]:
            features["bridge_tap"] |= any(data["bridges"][p] for p in neighbors) and not any(data["islands"][p] for p in neighbors)
            source = data["source_islands"][t["contact"]]
            touched = {int(rect_labels[p]) - 1 for p in neighbors if rect_labels[p]}
            features["indirect"] |= source not in touched
            distances = _distances(len(data["rectangles"]), data["edges"], source)
            features["remote_two"] |= bool(touched) and all(distances.get(i, -1) >= 2 for i in touched)
        elif t["contact"]:
            features["cold"] = True
        else:
            features["diagonal"] |= any(int(labels[p]) in data["sources"] for p in _neighbors(t["endpoint"], g.shape, True))
        if not t["signal"]:
            features["nonterminal_only"] |= any(int(labels[q]) in data["sources"]
                for p in t["path"][:-1] for q in _neighbors(p, g.shape))
            features["blocked_tip"] |= not rowwise_net["field"][t["endpoint"]] and any(
                rowwise[p] and not data["bridges"][p] and int(rowwise_net["labels"][p]) in rowwise_powered
                for p in neighbors)
    return features


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and int(seed) >= 0, "seed must be a nonnegative integer")
    _require(_integer(num_train) and 2 <= num_train <= 10, "num_train must be 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4, "num_test must be 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.choice(np.arange(1, 10), size=5, replace=False)]
    recolor = bool(rng.integers(0, 2))
    first_roles = None
    sides = [int(x) for x in rng.permutation(4)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            kind = ("blocked" if i == 0 else "dual") if split == "train" and i < 2 else (
                "extended" if split == "test" else str(rng.choice(["blocked", "dual", "extended"])))
            pair_palette = [int(c) for c in rng.choice(np.arange(1, 10), size=5, replace=False)] if recolor else palette
            if split == "train" and i == 0:
                first_roles = tuple(pair_palette[:2])
            elif split == "train" and i == 1 and recolor:
                while tuple(pair_palette[:2]) == first_roles:
                    pair_palette = [int(c) for c in rng.choice(np.arange(1, 10), size=5, replace=False)]
            required_sides = tuple(sides[2 * i:2 * i + 2]) if split == "train" and i < 2 else None
            for _ in range(20):
                pair = _make_pair(rng, pair_palette, kind, required_sides)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    # The two teaching examples have different jobs, not a required presentation order.
    order = rng.permutation(len(puzzle["train"]))
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid pair counts")
    seen, teaching, b_causal = set(), [], False
    train_sides, returned_colors = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(g.shape == out.shape, "shape changed")
            key = (g.shape, g.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            data = _decode(g)
            _require(np.array_equal(out, _render(data)), "incorrect signal/retraction output")
            # Preservation and edit support, in addition to exact rendering.
            preserved = data["islands"].copy()
            for t in data["tethers"]:
                preserved[t["center"]] = True
                for p in t["caps"]:
                    preserved[p] = True
                if not t["signal"]:
                    for p in t["path"]:
                        preserved[p] = True
            _require(np.array_equal(out[preserved], g[preserved]), "stationary material changed")
            allowed_edits = data["bridges"].copy()
            for t in data["tethers"]:
                if t["signal"]:
                    for p in t["path"]:
                        allowed_edits[p] = True
                    _require(out[t["path"][0]] == t["signal"], "wrong returned color")
                    _require(all(out[p] == 0 for p in t["path"][1:]), "incomplete retraction")
            _require(not np.any((g != out) & ~allowed_edits), "edit outside bridges and active tethers")
            _require(not np.array_equal(out, _render(data, retract=False)), "A has no effect in pair")
            f = _features(data)
            if split == "train":
                teaching.append(f)
                ablated = _without_bridging(data)
                b_causal |= any(out[t["path"][0]] != ablated[t["path"][0]] for t in data["tethers"])
                train_sides.update(t["side"] for t in data["tethers"])
                returned_colors.update(t["signal"] for t in data["tethers"] if t["signal"])
            else:
                _require(f["remote_two"], "test lacks a recombined two-link route")
    for feature in ("blocked", "bridge_tap", "diagonal", "cold", "indirect", "nonterminal_only", "cold_bridge", "blocked_tip"):
        _require(any(f[feature] for f in teaching), f"training lacks {feature} evidence")
    _require(b_causal, "omitting B does not change a demonstrated returned signal")
    _require(train_sides == {0, 1, 2, 3}, "training lacks a socket orientation")
    _require(len(returned_colors) >= 2, "training lacks multiple signal colors")


def _normalized(g):
    mapping = {0: 0}
    return (g.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in g.flat))


def self_test():
    """Bounded 42-seed audit, role equivariance, count checks, and ablations."""
    shapes, structures, palettes, counts, sides, legs, network_sizes = (set() for _ in range(7))
    totals = {k: 0 for k in ("blocked", "bridge_tap", "diagonal", "cold", "indirect", "remote_two", "nonterminal_only", "cold_bridge", "blocked_tip")}
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple((p["input"].shape, p["input"].tobytes()) for s in puzzle.values() for p in s))
        for split in puzzle.values():
            for p in split:
                g = p["input"]
                data = _decode(g)
                shapes.add(g.shape)
                structures.add(_normalized(g))
                palettes.add((data["neutral"], data["wire"], tuple(sorted(data["sources"].values()))))
                counts.add(len(data["rectangles"]))
                sides.update(t["side"] for t in data["tethers"])
                legs.update(n for t in data["tethers"] for n in t["legs"])
                network_sizes.update(len(v) for v in _rect_groups(data["rectangles"], data).values())
                for k, value in _features(data).items():
                    totals[k] += int(value)
                # Nonzero-color permutations must commute with the input-only solver.
                perm = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int8)
                _require(np.array_equal(solve(perm[g]), perm[p["output"]]), "color-role equivariance failed")
    _require(len(fingerprints) == 42 and len(shapes) >= 35 and len(structures) >= 180, "insufficient structural variation")
    _require(len(palettes) >= 30 and len(counts) >= 3 and len(network_sizes) >= 4, "insufficient palette/count/network variation")
    _require(sides == {0, 1, 2, 3} and 1 in legs and max(legs) >= 10, "insufficient tether geometry variation")
    _require(all(value >= 42 for value in totals.values()), "missing semantic coverage")
    first, second = generate_puzzle(947, 5, 2), generate_puzzle(947, 5, 2)
    _require(all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a), "nondeterminism")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + train * 10 + test, train, test))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "shapes": len(shapes), "structures": len(structures),
            "palettes": len(palettes), "island_counts": sorted(counts),
            "network_sizes": sorted(network_sizes), "semantic_pairs": totals}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
