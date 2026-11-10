"""Fill-and-flip reservoirs: conserved water operates scaffold-registered cups.

Only NumPy and the standard library are used. Literal roles are fixed: 0 air,
5 scaffold, 6 spill pipes, 7 final collector, 8 liquid. Other colors are local
state accents. No examples, random state, or generation metadata enter solve.
"""
from collections import deque
import numpy as np

AIR, FRAME, PIPE, COLLECTOR, WATER = 0, 5, 6, 7, 8
ACCENTS = (1, 2, 3, 4, 9)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, part = [start], {start}
        while todo:
            y, x = todo.pop()
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    todo.append(p)
        result.append(part)
    return result


def _bounds(points):
    return (min(y for y, x in points), max(y for y, x in points),
            min(x for y, x in points), max(x for y, x in points))


def _signature(mapping):
    return tuple(sorted((y, x, int(c)) for (y, x), c in mapping.items()))


def _template(width, depth, profile, accent, state):
    """A cup has fixed side walls/base and a state-dependent terraced floor.

    The lip flag lies above the left wall in state 0 and below the right wall
    in state 1. It makes the full motif boxes differ while anchors stay fixed.
    """
    result = {(r, c): FRAME for r in range(depth + 1) for c in (0, width + 1)}
    result.update({(depth, c): FRAME for c in range(width + 2)})
    for j, floor in enumerate(profile):
        for r in range(int(floor), depth):
            result[r, j + 1] = accent
    result[(-1, 0) if state == 0 else (depth + 1, width + 1)] = accent
    return result


def _draw(grid, mapping, y, x, scale):
    for (r, c), color in mapping.items():
        yy, xx = y + r * scale, x + c * scale
        _require(0 <= yy and yy + scale <= grid.shape[0] and
                 0 <= xx and xx + scale <= grid.shape[1], "clipped motif")
        grid[yy:yy + scale, xx:xx + scale] = color


def _free_cells(mapping, width, depth, y, x, scale):
    cells = []
    for r in range(depth * scale):
        for c in range(scale, (width + 1) * scale):
            if (r // scale, c // scale) not in mapping:
                cells.append((y + r, x + c))
    return sorted(cells, key=lambda p: (-p[0], p[1]))


def _read_input(grid):
    """Infer complete templates, anchors, supply ownership and directed graph."""
    a = _grid(grid)
    h, w = a.shape
    solid = ~np.isin(a, (AIR, PIPE, COLLECTOR, WATER))
    nodes, template_maps = [], {}
    geometry = None
    for group in _components(solid):
        carrier = {p for p in group if a[p] == FRAME}
        _require(bool(carrier), "unregistered foreground")
        y, y1, x, x1 = _bounds(carrier)
        crop = a[y:y1 + 1, x:x1 + 1] == FRAME
        scale = 0
        for r in range(crop.shape[0] - 1, -1, -1):
            if bool(crop[r].all()):
                scale += 1
            else:
                break
        _require(scale > 0 and crop.shape[0] % scale == 0 and
                 crop.shape[1] % scale == 0, "unreadable scale")
        depth, width = crop.shape[0] // scale - 1, crop.shape[1] // scale - 2
        _require(depth >= 2 and width >= 1, "invalid cup")
        expected = np.zeros_like(crop)
        expected[:, :scale] = True
        expected[:, -scale:] = True
        expected[-scale:, :] = True
        _require(np.array_equal(crop, expected), "carrier is not a scaled U")
        if geometry is None:
            geometry = width, depth
        _require(geometry == (width, depth), "pair-local scaffolds disagree")
        blocks = {}
        for yy, xx in group:
            r, c = (yy - y) // scale, (xx - x) // scale
            blocks.setdefault((r, c), []).append(int(a[yy, xx]))
        _require(all(len(v) == scale * scale and len(set(v)) == 1
                     for v in blocks.values()), "nonuniform state blocks")
        mapping = {p: v[0] for p, v in blocks.items()}
        accents = set(mapping.values()) - {FRAME}
        _require(len(accents) == 1 and accents <= set(ACCENTS), "invalid state accent")
        accent = next(iter(accents))
        flags = [p for p in mapping if p[0] < 0 or p[0] > depth]
        _require(flags in ([(-1, 0)], [(depth + 1, width + 1)]), "invalid lip flag")
        state = int(flags[0][0] > depth)
        profile = []
        for c in range(1, width + 1):
            levels = [r for r in range(depth) if (r, c) in mapping]
            floor = min(levels) if levels else depth
            _require(1 <= floor <= depth and levels == list(range(floor, depth)),
                     "floor is not a supported terrace")
            profile.append(floor)
        _require(mapping == _template(width, depth, profile, accent, state), "extra motif cells")
        sig = _signature(mapping)
        template_maps[sig] = mapping
        top, bottom = y - 2 * scale, y + (depth + 2) * scale
        _require(0 <= top and bottom <= h, "source/replacement region clipped")
        nodes.append({"y": y, "x": x, "scale": scale, "signature": sig,
                      "box": (top, bottom, x, x + (width + 2) * scale),
                      "inlet": (y, x - 1), "outlet": (y, x + (width + 2) * scale)})
    _require(len(nodes) >= 2 and len(template_maps) == 2, "exactly two visible cup states required")
    width, depth = geometry
    signatures = sorted(template_maps)
    templates = [template_maps[sig] for sig in signatures]
    capacities = [len(_free_cells(t, width, depth, 0, 0, 1)) for t in templates]
    _require(capacities[0] != capacities[1], "different state capacities required")
    # Neither literal accent ordering nor flag position defines the exchanged
    # design: the entire other locally visible template is used.
    min_scale = min(n["scale"] for n in nodes)
    _require({n["signature"] for n in nodes if n["scale"] == min_scale} == set(signatures),
             "both states need equal-scale references")
    for i, n in enumerate(nodes):
        n["state"] = signatures.index(n["signature"])
        n["cells"] = [_free_cells(t, width, depth, n["y"], n["x"], n["scale"])
                      for t in templates]
        for m in nodes[:i]:
            t, b, l, r = n["box"]
            mt, mb, ml, mr = m["box"]
            _require(b <= mt or mb <= t or r <= ml or mr <= l, "overlapping cup domains")
        t, b, l, r = n["box"]
        _require(not bool(np.any(a[t:b, l:r] == PIPE)), "pipe enters a cup/source domain")
        _require(0 <= n["inlet"][1] and n["outlet"][1] < w, "port outside canvas")
    collector_points = set(map(tuple, np.argwhere(a == COLLECTOR)))
    _require(bool(collector_points), "missing collector")
    cy = min(y for y, x in collector_points)
    expected_collector = {(r, c) for r in range(cy, h) for c in (0, w - 1)}
    expected_collector |= {(h - 1, c) for c in range(w)}
    _require(collector_points == expected_collector and cy < h - 1 and w > 2, "invalid collector")
    _require(all(n["box"][1] < cy for n in nodes), "cup touches collector")
    collector_cells = sorted(((r, c) for r in range(cy, h - 1) for c in range(1, w - 1)),
                             key=lambda p: (-p[0], p[1]))
    _require(all(a[p] == AIR for p in collector_cells), "collector starts empty")
    assigned = set()
    for n in nodes:
        s, y, x = n["scale"], n["y"], n["x"]
        domain = {(r, c) for r in range(y - 2 * s, y + depth * s)
                  for c in range(x + s, x + (width + 1) * s)}
        supply = {p for p in domain if a[p] == WATER}
        _require(not (assigned & supply), "ambiguous rain ownership")
        assigned |= supply
        n["supply"] = len(supply)
    _require(assigned == set(map(tuple, np.argwhere(a == WATER))), "unassigned liquid")
    _require(bool(assigned), "missing liquid")
    pipe_groups = _components(a == PIPE)
    destinations = {}
    for group in pipe_groups:
        sources = [i for i, n in enumerate(nodes) if n["outlet"] in group]
        targets = [i for i, n in enumerate(nodes) if n["inlet"] in group]
        sink = any(r == cy - 1 and 1 <= c < w - 1 for r, c in group)
        _require(bool(sources) and len(targets) + int(sink) == 1, "ambiguous spill pipe")
        target = targets[0] if targets else -1
        endpoints = {nodes[target]["inlet"]} if target >= 0 else {
            p for p in group if p[0] == cy - 1 and 1 <= p[1] < w}
        for i in sources:
            _require(i not in destinations, "multiple outgoing pipes")
            _require(target < 0 or nodes[target]["y"] > nodes[i]["y"], "uphill/cyclic pipe")
            start = nodes[i]["outlet"]
            todo, seen = [start], {start}
            while todo:
                y, x = todo.pop()
                for p in ((y + 1, x), (y, x - 1), (y, x + 1)):
                    if p in group and p not in seen:
                        seen.add(p)
                        todo.append(p)
            _require(bool(seen & endpoints), "no descending route through pipe")
            destinations[i] = target
    _require(len(destinations) == len(nodes), "cup has no spill destination")
    for i, n in enumerate(nodes):
        n["target"] = destinations[i]
    return {"grid": a, "nodes": nodes, "templates": templates, "capacities": capacities,
            "width": width, "depth": depth, "collector_cells": collector_cells}


def _transform(info, flip=True):
    """Each receiving cup operates once, after all upstream deliveries arrive."""
    a = info["grid"]
    out = a.copy()
    out[out == WATER] = AIR
    nodes, templates = info["nodes"], info["templates"]
    incoming = [n["supply"] for n in nodes]
    sink, events = 0, []
    for i in sorted(range(len(nodes)), key=lambda i: (nodes[i]["y"], nodes[i]["x"])):
        n, amount = nodes[i], incoming[i]
        old = n["state"]
        triggered = flip and amount >= len(n["cells"][old])
        new = 1 - old if triggered else old
        retained = min(amount, len(n["cells"][new]))
        overflow = amount - retained
        # Restore the complete union, including a flag beyond either box.
        for r, c in set(templates[0]) | set(templates[1]):
            yy, xx, s = n["y"] + r * n["scale"], n["x"] + c * n["scale"], n["scale"]
            out[yy:yy + s, xx:xx + s] = AIR
        _draw(out, templates[new], n["y"], n["x"], n["scale"])
        for p in n["cells"][new][:retained]:
            out[p] = WATER
        if n["target"] < 0:
            sink += overflow
        else:
            incoming[n["target"]] += overflow
        events.append({"node": i, "amount": amount, "old_capacity": len(n["cells"][old]),
                       "new_capacity": len(n["cells"][new]), "triggered": triggered,
                       "retained": retained, "overflow": overflow,
                       "scale": n["scale"], "target": n["target"]})
    _require(sink <= len(info["collector_cells"]), "collector overflow")
    for p in info["collector_cells"][:sink]:
        out[p] = WATER
    return out, events


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input pixels only, including fresh templates and wiring."""
    return _transform(_read_input(input_grid))[0]


def _route(rng, shape, reserved, ports, starts, target, painted):
    """Construct a separate descending pipe component; merges share one pipe."""
    h, w = shape
    blocked = reserved.copy()
    for p in ports:
        if 0 <= p[0] < h and 0 <= p[1] < w:
            blocked[p] = True
    for p in starts + [target]:
        blocked[p] = False
    for group in painted:
        for y, x in group:
            for yy, xx in ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w:
                    blocked[yy, xx] = True
    component = set()
    steps = [(0, 1), (0, -1), (1, 0)]
    rng.shuffle(steps)
    for start in starts:
        if blocked[start] or blocked[target]:
            return None
        todo, previous = deque([start]), {start: None}
        while todo and target not in previous:
            y, x = todo.popleft()
            for dy, dx in steps:
                p = y + dy, x + dx
                if (0 <= p[0] <= target[0] and 0 <= p[1] < w and
                        not blocked[p] and p not in previous):
                    previous[p] = (y, x)
                    todo.append(p)
        if target not in previous:
            return None
        p = target
        while p is not None:
            component.add(p)
            p = previous[p]
    return component


def _positions(rng, count, span, canvas_width):
    needed = count * span + (count - 1) * 3
    _require(needed + 2 <= canvas_width, "row too wide")
    slack = canvas_width - 2 - needed
    left = 1 + int(rng.integers(slack + 1))
    return [left + i * (span + 3) for i in range(count)]


def _make_pair(rng, regime):
    # Bounded retries only choose geometry/wiring; there is no parent-generator
    # call, hidden template dictionary, or whole-puzzle composition pipeline.
    for attempt in range(160):
        if regime == "test":
            width, depth, lower_scale = int(rng.integers(1, 3)), 2, 3
        else:
            width = int(rng.integers(1, 3 if regime == "second" else 4))
            depth = int(rng.integers(2, 5))
            lower_scale = 2 if regime in ("first", "second") else int(rng.choice([1, 1, 2]))
        span = width + 2
        top_count = 2 if regime == "first" else 4 if regime == "second" else 3 if regime == "test" else int(rng.integers(2, 5))
        if top_count * span + (top_count - 1) * 3 + 2 > 30:
            continue
        lower_count = 1 if regime != "mixed" or lower_scale == 2 else int(rng.integers(1, 3))
        lower_scales = [lower_scale] + [1] * (lower_count - 1)
        lower_needed = sum(span * s for s in lower_scales) + 3 * (lower_count - 1)
        min_width = max(top_count * span + (top_count - 1) * 3, lower_needed) + 2
        canvas_width = int(rng.integers(min_width, 31))
        top_size, lower_size = depth + 4, (depth + 4) * lower_scale
        room = 30 - (top_size + lower_size + 5)
        if room < 0:
            continue
        margin = int(rng.integers(min(room, 2) + 1))
        gap = int(rng.integers(1, min(3, room - margin + 1) + 1))
        top_zone = margin
        lower_zone = margin + top_size + gap
        collector_row = lower_zone + lower_size + 1
        height = collector_row + 3
        if height > 30:
            continue
        shallow = [int(rng.integers(1, depth)) for _ in range(width)]
        deep = None
        # Net capacity grows, but individual terraces can move either way.
        # Thus knowing only the capacity or reflecting a floor is insufficient
        # to reconstruct the other visible template.
        for _ in range(24):
            proposal = [int(np.clip(v + int(rng.integers(-1, 2)), 1, depth))
                        for v in shallow]
            if 0 < sum(proposal) - sum(shallow) <= width:
                deep = proposal
                break
        if deep is None:
            deep = shallow.copy()
            deep[int(rng.integers(width))] += 1
        colors = list(map(int, rng.choice(ACCENTS, size=2, replace=False)))
        if regime == "mixed" and rng.random() < .3:
            colors[1] = colors[0]
        templates = [_template(width, depth, p, c, state)
                     for state, (p, c) in enumerate(zip((shallow, deep), colors))]
        capacities = [sum(shallow), sum(deep)]
        top_xs = _positions(rng, top_count, span, canvas_width)
        lower_left = 1 + int(rng.integers(canvas_width - lower_needed - 1))
        lower_xs = []
        xx = lower_left
        for s in lower_scales:
            lower_xs.append(xx)
            xx += span * s + 3
        states = [0, 1] + [1 if regime in ("second", "test") else int(rng.integers(2))
                            for _ in range(top_count - 2)]
        if regime == "second":
            states = [0, 1, 1, 0]
        if regime == "test":
            states = [1, 1, 0]
        nodes = [{"y": top_zone + 2, "x": x, "s": 1, "state": state}
                 for x, state in zip(top_xs, states)]
        nodes += [{"y": lower_zone + 2 * s, "x": x, "s": s,
                   "state": 1 if regime in ("second", "test") else int(rng.integers(2))}
                  for x, s in zip(lower_xs, lower_scales)]
        reserved = np.zeros((height, canvas_width), dtype=bool)
        for n in nodes:
            y, x, s = n["y"], n["x"], n["s"]
            reserved[y - 2 * s:y + (depth + 2) * s, x:x + span * s] = True
        reserved[collector_row:] = True
        ports = [(n["y"], n["x"] - 1) for n in nodes]
        outlets = [(n["y"], n["x"] + span * n["s"]) for n in nodes]
        if any(not (0 <= x < canvas_width) for y, x in outlets):
            continue
        groups = []
        sink_target = (collector_row - 1, int(rng.integers(1, canvas_width - 1)))
        group = _route(rng, reserved.shape, reserved, ports + outlets,
                       outlets[top_count:], sink_target, groups)
        if group is None:
            continue
        groups.append(group)
        targets = [top_count + min(lower_count - 1, i * lower_count // top_count)
                   for i in range(top_count)]
        okay = True
        for dest in sorted(set(targets), reverse=bool(rng.integers(2))):
            starts = [outlets[i] for i, t in enumerate(targets) if t == dest]
            group = _route(rng, reserved.shape, reserved, ports + outlets, starts, ports[dest], groups)
            if group is None:
                okay = False
                break
            groups.append(group)
        if not okay:
            continue
        # Choose supplies with contrasting threshold and causal cascade cases.
        if regime == "first":
            amounts = [max(0, capacities[0] - 1), capacities[1]]
        elif regime == "second":
            amounts = [capacities[1] + width, capacities[1], capacities[1],
                       max(0, capacities[0] - 1)]
        elif regime == "test":
            amounts = [capacities[1] + int(rng.integers(width + 1)), capacities[1],
                       max(0, capacities[0] - 1)]
        else:
            amounts = [max(0, capacities[0] - 1), capacities[1]]
            amounts += [int(rng.integers(capacities[st] + 2 * width + 1)) for st in states[2:]]
        delivered = [0] * lower_count
        for i, (n, amount) in enumerate(zip(nodes[:top_count], amounts)):
            old = n["state"]
            new = 1 - old if amount >= capacities[old] else old
            delivered[targets[i] - top_count] += max(0, amount - capacities[new])
        for j, n in enumerate(nodes[top_count:]):
            s, old = n["s"], n["state"]
            if regime in ("second", "test"):
                local = max(0, capacities[old] * s * s - delivered[j])
            elif regime == "first":
                local = 0
            else:
                max_local = (capacities[old] + 2 * width) * s * s
                local = int(rng.integers(max_local + 1))
            amounts.append(local)
        inp = np.zeros(reserved.shape, dtype=np.int8)
        inp[collector_row:, 0] = COLLECTOR
        inp[collector_row:, -1] = COLLECTOR
        inp[-1] = COLLECTOR
        for group in groups:
            for p in group:
                inp[p] = PIPE
        for n, amount in zip(nodes, amounts):
            y, x, s, state = n["y"], n["x"], n["s"], n["state"]
            _draw(inp, templates[state], y, x, s)
            eligible = [(r, c) for r in range(y - 2 * s, y + depth * s)
                        for c in range(x + s, x + (width + 1) * s) if inp[r, c] == AIR]
            if amount > len(eligible):
                okay = False
                break
            style = int(rng.integers(4))
            anchor = eligible[int(rng.integers(len(eligible)))]
            jitter = rng.random(len(eligible))
            scores = []
            for k, (r, c) in enumerate(eligible):
                if style == 0:
                    score = abs(r - anchor[0]) + abs(c - anchor[1]) + jitter[k]
                elif style == 1:
                    score = jitter[k]
                elif style == 2:
                    score = -r + .05 * c + .02 * jitter[k]
                else:
                    score = 3 * abs(c - anchor[1]) + .1 * r + jitter[k]
                scores.append(score)
            for k in np.argsort(scores)[:amount]:
                inp[eligible[int(k)]] = WATER
        if not okay:
            continue
        try:
            info = _read_input(inp)
            out, events = _transform(info)
            # In every pair, shape-independent redistribution and switching
            # both affect the visible answer, including downstream allocation.
            if not any(e["triggered"] for e in events):
                continue
            if not any(not e["triggered"] for e in events):
                continue
            if np.array_equal(_transform(info, flip=False)[0], out):
                continue
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded reservoir construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(value) or not lo <= int(value) <= hi:
            raise ValueError("illegal example counts")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            regime = "test" if split == "test" else "first" if i == 0 else "second" if i == 1 else "mixed"
            for _ in range(40):
                pair = _make_pair(rng, regime)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "changed dimensions")
    info = _read_input(inp)
    expected, events = _transform(info)
    _require(np.array_equal(expected, out), "incorrect fill-and-flip answer")
    _require(int((inp == WATER).sum()) == int((out == WATER).sum()), "water is not conserved")
    for color in (FRAME, PIPE, COLLECTOR):
        _require(np.array_equal(inp == color, out == color), "fixed infrastructure changed")
    _require(not np.array_equal(inp == WATER, out == WATER), "no liquid redistribution")
    _require(any(e["triggered"] for e in events), "missing switch event")
    _require(any(not e["triggered"] for e in events), "missing non-switch")
    no_flip, baseline = _transform(info, flip=False)
    _require(not np.array_equal(no_flip, out), "B ablation did not matter")
    _require(not np.array_equal(no_flip == WATER, out == WATER),
             "template switching must change liquid allocation")
    # With A omitted (no volume redistribution/capacity threshold), retaining
    # the supply arrangement fails; unconditional toggling fails as well.
    _require(any(e["overflow"] > 0 for e in events), "no routed overflow")
    changed_liquid = int(np.count_nonzero((inp == WATER) != (out == WATER)))
    caps = info["capacities"]
    kinds = {"expansion" if e["new_capacity"] > e["old_capacity"] else "contraction"
             for e in events if e["triggered"]}
    donor_counts = {}
    for e in events:
        if e["target"] >= 0 and e["overflow"]:
            donor_counts[e["target"]] = donor_counts.get(e["target"], 0) + 1
    affected_downstream = any(a["amount"] != b["amount"] for a, b in zip(events, baseline))
    cascade = any(a["triggered"] and b["amount"] < a["old_capacity"] and
                  a["amount"] >= a["old_capacity"] for a, b in zip(events, baseline))
    profiles = tuple(tuple(next((r for r in range(info["depth"]) if (r, c) in t), info["depth"])
                           for c in range(1, info["width"] + 1)) for t in info["templates"])
    differences = [b - a for a, b in zip(*profiles)]
    shared_accent = (set(info["templates"][0].values()) - {FRAME} ==
                     set(info["templates"][1].values()) - {FRAME})
    return {"info": info, "events": events, "shape": inp.shape, "width": info["width"],
            "depth": info["depth"], "capacities": tuple(sorted(caps)),
            "scales": tuple(sorted(n["scale"] for n in info["nodes"])),
            "nodes": len(events), "kinds": kinds, "merge": max(donor_counts.values(), default=0),
            "cascade": cascade, "affected_downstream": affected_downstream,
            "profiles": profiles, "nonordered": min(differences) < 0 < max(differences),
            "shared_accent": shared_accent,
            "water": int((inp == WATER).sum()), "changed_liquid": changed_liquid,
            "templates": tuple(_signature(t) for t in info["templates"])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal example counts")
    seen, stats = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            stats[split].append(_inspect_pair(pair))
    _require(set.union(*(f["kinds"] for f in stats["train"])) == {"expansion", "contraction"},
             "training must teach both swap directions")
    _require(any(max(f["scales"]) >= 2 for f in stats["train"]), "training omits scaling")
    _require(any(f["merge"] >= 2 for f in stats["train"]), "training omits merged overflow")
    _require(any(f["cascade"] for f in stats["train"]), "training omits causal threshold cascade")
    _require(any(e["amount"] > 0 and not e["triggered"] for f in stats["train"] for e in f["events"]),
             "training omits wet non-switching cups")
    _require(all(3 in f["scales"] for f in stats["test"]), "test shift omits scale three")
    return None


def _hand_fixture():
    """A three-tier witness with a coordinate-authored answer, not solve output."""
    rng = np.random.default_rng(732)
    a = np.zeros((26, 20), dtype=np.int8)
    templates = [_template(1, 2, [1], 2, 0), _template(1, 2, [2], 3, 1)]
    placements = [(2, 1, 0), (2, 8, 1), (10, 4, 0), (18, 10, 1)]
    reserved = np.zeros_like(a, dtype=bool)
    for y, x, state in placements:
        reserved[y - 2:y + 4, x:x + 3] = True
        _draw(a, templates[state], y, x, 1)
    reserved[23:] = True
    ports = [(y, x - 1) for y, x, state in placements]
    outlets = [(y, x + 3) for y, x, state in placements]
    groups = []
    for starts, target in (([outlets[3]], (22, 18)),
                           ([outlets[2]], ports[3]),
                           (outlets[:2], ports[2])):
        group = _route(rng, a.shape, reserved, ports + outlets, starts, target, groups)
        _require(group is not None, "hand fixture routing failed")
        groups.append(group)
        for p in group:
            a[p] = PIPE
    a[23:, 0] = a[23:, -1] = COLLECTOR
    a[-1] = COLLECTOR
    for p in ((0, 9), (1, 9), (2, 9), (3, 9), (16, 11)):
        a[p] = WATER
    expected = a.copy()
    expected[expected == WATER] = AIR
    # Manually: deep top receives 4, shrinks to capacity 1, sends 3;
    # middle receives 3, expands to 2, sends 1; deep bottom receives its
    # own 1 plus that 1, shrinks to 1, sends 1 to the collector.
    for index in (1, 2, 3):
        y, x, state = placements[index]
        for r, c in set(templates[0]) | set(templates[1]):
            expected[y + r, x + c] = AIR
        _draw(expected, templates[1 - state], y, x, 1)
    for p in ((2, 9), (10, 5), (11, 5), (18, 11), (24, 1)):
        expected[p] = WATER
    return a, expected


def self_test():
    """Bounded, grid-derived semantic checks and diversity over 42 seeds."""
    fixture, expected = _hand_fixture()
    _require(np.array_equal(solve(fixture), expected), "three-tier hand answer failed")
    relocated = fixture.copy()
    relocated[16, 11], relocated[19, 11] = AIR, WATER
    _require(np.array_equal(solve(relocated), expected), "source-shape invariance failed")
    signatures, shapes, profiles, schedules, quantities, templates, node_counts = (set() for _ in range(7))
    floor_geometries = set()
    cascades = merges = nonordered = shared_accents = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        fingerprint = tuple(_key(p[k]) for split in ("train", "test")
                            for p in puzzle[split] for k in ("input", "output"))
        _require(fingerprint == tuple(_key(p[k]) for split in ("train", "test")
                                     for p in again[split] for k in ("input", "output")), "nondeterminism")
        signatures.add(fingerprint)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _inspect_pair(pair)
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "solve mismatch")
                shapes.add(f["shape"])
                profiles.add((f["width"], f["depth"], f["capacities"]))
                floor_geometries.add((f["depth"], f["profiles"]))
                schedules.add(f["scales"])
                quantities.add(f["water"])
                templates.add(f["templates"])
                node_counts.add(f["nodes"])
                cascades += int(f["cascade"])
                merges += int(f["merge"] >= 2)
                nonordered += int(f["nonordered"])
                shared_accents += int(f["shared_accent"])
    _require(len(signatures) == 42, "seed diversity missing")
    _require(len(shapes) >= 20 and len(profiles) >= 15 and len(templates) >= 60,
             "insufficient structural/template diversity")
    _require(len(schedules) >= 4 and len(quantities) >= 25 and len(node_counts) >= 3,
             "insufficient count/scale/quantity diversity")
    _require(cascades >= 42 and merges >= 42, "causal variation missing")
    _require(len(floor_geometries) >= 35 and nonordered >= 1 and shared_accents >= 10,
             "floor/appearance variation missing")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(81 + nt * 5 + ns, nt, ns)
            _require(len(p["train"]) == nt and len(p["test"]) == ns, "count mismatch")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (3.5, 1),
                   (True, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Test labels cannot affect input parsing; mutations must be caught.
    p = generate_puzzle(91, 2, 1)
    for color in (WATER, FRAME, PIPE, COLLECTOR):
        bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in p[s]] for s in p}
        point = tuple(np.argwhere(bad["test"][0]["output"] == color)[0])
        bad["test"][0]["output"][point] = AIR
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "capacity_geometries": len(profiles), "local_template_pairs": len(templates),
            "floor_geometries": len(floor_geometries), "nonordered_floor_pairs": nonordered,
            "shared_accent_pairs": shared_accents,
            "scale_schedules": len(schedules), "node_counts": sorted(node_counts),
            "water_quantities": len(quantities), "cascades": cascades, "merges": merges}
