"""Chromatic turn filters: diagonal rays whose color controls mirror eligibility.

Only the input supplies state and the local turning table. Generation grows
geometric scenes; the public solver parses and traces arbitrary valid scenes.
"""
from __future__ import annotations

import json

import numpy as np


DIAGONALS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
CARDINALS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 558472599, 436009025, 755562698, 358323063, 191988868,
              31373999, 419286307, 675269697, 835147611, 629477949,
              504171441, 672195411, 236424522, 277309064, 147833487,
              437150862, 294538600, 867358423, 614529975, 346509494,
              621528894, 296874482, 295744795, 327259628, 149598556,
              377311623, 7707833, 830290011, 275003310, 91320748,
              600129701, 227630034, 353676927, 462598506, 900706185,
              283012242, 449377672, 932739440, 145152686, 933904061,
              34506395)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= grid) & (grid <= 9)), "colors must be 0..9")
    return grid.astype(np.int8, copy=False)


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _left(direction):
    dr, dc = direction
    return -dc, dr


def _reflection(direction, axis):
    dr, dc = direction
    result = (-dr, dc) if axis == 0 else (dr, -dc)
    return result, "L" if result == _left(direction) else "R"


def _header_decode(grid):
    frame = int(grid[0, 0])
    _require(np.all(grid[0] == frame), "missing top frame")
    bottoms = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(grid) - 1, "invalid band height")
    band = grid[1:bottom]
    _require(np.all(band == band[0]), "inconsistent panel rows")
    row = band[0]
    _require(row[0] == row[-1] == frame, "missing side walls")
    interiors = []
    c = 0
    while c < len(row):
        if row[c] == frame:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != frame:
            c += 1
        _require(c - start >= 3, "panel needs a clear central interior")
        interiors.append((start, c))
    _require(bool(interiors), "no key panels")
    background = int(row[interiors[0][0] + 1])
    _require(background != frame, "background and frame must differ")
    table = {}
    for a, b in interiors:
        _require(np.all(row[a+1:b-1] == background), "inconsistent interiors")
        left, right = int(row[a]), int(row[b-1])
        _require((left == background) != (right == background), "ambiguous key")
        color, side = (right, "R") if left == background else (left, "L")
        _require(color not in table and color not in (frame, background),
                 "repeated or structural key color")
        table[color] = side
    return bottom + 1, background, frame, table, tuple(b-a+2 for a, b in interiors)


def _objects(field, background):
    h, w = field.shape
    seen = set()
    sources, mirrors = [], []
    for rr, cc in np.argwhere(field != background):
        start = int(rr), int(cc)
        if start in seen:
            continue
        color = int(field[start])
        stack, cells = [start], set()
        seen.add(start)
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in CARDINALS:
                p = r + dr, c + dc
                if (0 <= p[0] < h and 0 <= p[1] < w and p not in seen
                        and field[p] == color):
                    seen.add(p)
                    stack.append(p)
        r0, c0 = map(int, np.min(list(cells), axis=0))
        r1, c1 = map(int, np.max(list(cells), axis=0))
        if len(cells) == 3 and r1-r0 == c1-c0 == 1:
            missing = ({(r, c) for r in (r0, r1) for c in (c0, c1)} - cells).pop()
            elbow = r0+r1-missing[0], c0+c1-missing[1]
            direction = elbow[0]-missing[0], elbow[1]-missing[1]
            sources.append(dict(cells=cells, color=color, elbow=elbow, direction=direction))
        else:
            _require(len(cells) % 2 == 1 and (r0 == r1 or c0 == c1)
                     and len(cells) == max(r1-r0, c1-c0)+1,
                     "object must be an L or a solid odd bar")
            axis = None if len(cells) == 1 else (0 if r0 == r1 else 1)
            mirrors.append(dict(cells=cells, color=color,
                                midpoint=((r0+r1)//2, (c0+c1)//2), axis=axis))
    _require(bool(sources), "no L source")
    return sources, mirrors


def _decode(input_grid):
    grid = _grid(input_grid)
    crop, background, frame, table, widths = _header_decode(grid)
    field = grid[crop:].copy()
    _require(frame not in field, "frame color cannot occur in the field")
    _require(set(map(int, np.unique(field))) <= {background} | set(table),
             "every foreground color requires a key")
    _require(np.count_nonzero(field == background) > field.size/2,
             "field background must occupy a majority")
    sources, mirrors = _objects(field, background)
    return dict(field=field, background=background, frame=frame, table=table,
                crop=crop, widths=widths, sources=sources, mirrors=mirrors)


def _trace(field, background, table, *, mode="normal"):
    """Cell tracer. The alternative modes are used only as shortcut probes."""
    sources, mirrors = _objects(field, background)
    h, w = field.shape
    source_cells = {p for source in sources for p in source["cells"]}
    contacts = {}
    for i, mirror in enumerate(mirrors):
        for axis in ((0, 1) if mirror["axis"] is None else (mirror["axis"],)):
            for direction in DIAGONALS:
                p = list(mirror["midpoint"])
                p[axis] -= direction[axis]
                contacts.setdefault((tuple(p), direction), []).append((i, axis))
    out = field.copy()
    painted, routes = {}, []
    for source in sources:
        direction = source["direction"]
        color = source["color"]
        anchor = source["elbow"]
        p = anchor[0]+direction[0], anchor[1]+direction[1]
        seen, steps, events = set(), [], []
        while 0 <= p[0] < h and 0 <= p[1] < w:
            state = p, direction, color
            _require(state not in seen, "cyclic ray")
            seen.add(state)
            _require(p not in source_cells, "ray strikes an L source")
            incoming_direction, incoming_color = direction, color
            hits = contacts.get((p, direction), [])
            _require(len(hits) <= 1, "competing midpoint contacts")
            event = None
            if hits:
                _require(field[p] == background, "contact cell is occupied")
                i, axis = hits[0]
                reflected, side = _reflection(direction, axis)
                lookup_color = (mirrors[i]["color"] if mode == "mirror_key"
                                else source["color"] if mode == "frozen_key" else color)
                accepted = mode == "all_reflect" or table[lookup_color] == side
                if accepted:
                    direction = reflected
                    if mode != "no_recolor":
                        color = mirrors[i]["color"]
                event = dict(mirror=i, axis=axis, position=p, incoming=incoming_color,
                             heading=incoming_direction, side=side, accepted=accepted,
                             outgoing=color, step=len(steps))
                events.append(event)
                if mode == "stop_rejected" and not accepted:
                    steps.append((p, incoming_direction, incoming_color, color, event))
                    break
            # Transparent mirror bodies are preserved; only background is painted.
            if field[p] == background:
                _require(p not in painted or painted[p] == color, "conflicting path colors")
                painted[p] = color
                out[p] = color
            steps.append((p, incoming_direction, incoming_color, color, event))
            p = p[0]+direction[0], p[1]+direction[1]
        routes.append(dict(source=source, steps=steps, events=events,
                           exit=p, final_direction=direction, final_color=color))
    return out, dict(sources=sources, mirrors=mirrors, routes=routes, painted=painted)


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    return _trace(info["field"], info["background"], info["table"])[0]


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r-dr, c), (r, c-dc)}


def _mirror_cells(midpoint, axis, length):
    r, c = midpoint
    radius = length // 2
    return {(r, c+x) if axis == 0 else (r+x, c)
            for x in range(-radius, radius+1)}


def _place(field, cells, color, background):
    h, w = field.shape
    for p in cells:
        r, c = p
        if not (0 <= r < h and 0 <= c < w) or field[p] != background:
            return None
        for dr, dc in CARDINALS:
            q = r+dr, c+dc
            if q not in cells and 0 <= q[0] < h and 0 <= q[1] < w and field[q] == color:
                return None
    result = field.copy()
    for p in cells:
        result[p] = color
    if np.count_nonzero(result == background) <= result.size/2:
        return None
    return result


def _add_source(field, background, table, rng, direction=None, color=None):
    h, w = field.shape
    for _ in range(100):
        d = direction if direction is not None else DIAGONALS[int(rng.integers(4))]
        elbow = int(rng.integers(h)), int(rng.integers(w))
        shade = int(rng.choice(list(table))) if color is None else color
        candidate = _place(field, _source_cells(elbow, d), shade, background)
        if candidate is None:
            continue
        try:
            _, info = _trace(candidate, background, table)
        except AssertionError:
            continue
        route = next(q for q in info["routes"] if q["source"]["elbow"] == elbow)
        if len(route["steps"]) >= 3:
            return candidate, elbow
    return None


def _event_key(info, event):
    m = info["mirrors"][event["mirror"]]
    return m["midpoint"], m["axis"], m["color"], event["position"], event["accepted"]


def _extend(field, background, table, rng, elbow, accepted,
            color_policy="any", length=None):
    _, before = _trace(field, background, table)
    route = next(q for q in before["routes"] if q["source"]["elbow"] == elbow)
    last = route["events"][-1]["step"] if route["events"] else -1
    options = list(range(last+2, len(route["steps"])))
    rng.shuffle(options)
    prefix = [_event_key(before, e) for e in route["events"]]
    for step in options[:28]:
        p, direction, incoming, _, _ = route["steps"][step]
        if field[p] != background:
            continue
        axis = next(a for a in (0, 1)
                    if (_reflection(direction, a)[1] == table[incoming]) == accepted)
        midpoint = list(p)
        midpoint[axis] += direction[axis]
        if color_policy == "unchanged":
            colors = [incoming]
        elif color_policy == "switch":
            colors = [c for c in table if table[c] != table[incoming]]
        elif color_policy == "same_side":
            colors = [c for c in table if table[c] == table[incoming]]
        else:
            colors = list(table)
        rng.shuffle(colors)
        sizes = [length] if length is not None else [1, 3, 5, 7]
        rng.shuffle(sizes)
        for n in sizes:
            for shade in colors[:3]:
                candidate = _place(field, _mirror_cells(tuple(midpoint), axis, n), shade, background)
                if candidate is None:
                    continue
                try:
                    _, after = _trace(candidate, background, table)
                except AssertionError:
                    continue
                new_route = next(q for q in after["routes"] if q["source"]["elbow"] == elbow)
                sequence = [_event_key(after, e) for e in new_route["events"]]
                target = tuple(midpoint), (None if n == 1 else axis), shade, p, accepted
                if (sequence[:len(prefix)] == prefix and len(sequence) > len(prefix)
                        and sequence[len(prefix)] == target
                        and len(new_route["steps"]) - new_route["events"][len(prefix)]["step"] >= 3):
                    return candidate
    return None


def _make_header(rng, width, background, frame, table):
    colors = list(table)
    rng.shuffle(colors)
    widths = np.full(len(colors), 5, dtype=int)
    widths += rng.multinomial(width-5*len(colors), np.full(len(colors), 1/len(colors)))
    height = int(rng.integers(3, 6))
    header = np.full((height, width), background, dtype=np.int8)
    header[[0, -1]] = frame
    c = 0
    for color, n in zip(colors, widths):
        header[:, c] = header[:, c+n-1] = frame
        header[1:-1, c+1 if table[color] == "L" else c+n-2] = color
        c += int(n)
    return header


def _pair(rng, *, teaching=None, test=False, direction=None,
          anchor_color=None, anchor_side=None):
    # Visible key tables and palettes are pair-local, never learned by seed.
    for attempt in range(100):
        nkeys = int(rng.integers(2, 7))
        palette = list(map(int, rng.permutation([c for c in range(10) if c != anchor_color])))
        background, frame = palette[:2]
        if anchor_color is None:
            colors = palette[2:2+nkeys]
            sides = ["L", "R"] + ["L" if rng.integers(2) else "R" for _ in range(nkeys-2)]
            rng.shuffle(sides)
        else:
            colors = [anchor_color] + palette[2:1+nkeys]
            sides = [anchor_side, "R" if anchor_side == "L" else "L"]
            sides += ["L" if rng.integers(2) else "R" for _ in range(nkeys-2)]
        table = dict(zip(colors, sides))
        h = int(rng.integers(15 if teaching is not None or test else 9, 26))
        w = int(rng.integers(max(18 if teaching is not None or test else 12, 5*nkeys), 31))
        field = np.full((h, w), background, dtype=np.int8)
        result = _add_source(field, background, table, rng, direction=direction, color=anchor_color)
        if result is None:
            continue
        field, elbow = result
        if teaching == 0:
            plan = [(False, "switch", 5), (True, "switch", 3),
                    (True, "switch", None), (True, "unchanged", None)]
        elif teaching == 1:
            plan = [(True, "unchanged", 5), (False, "switch", 3),
                    (True, "switch", None), (True, "same_side", None)]
        elif test:
            plan = [(False, "switch", int(rng.choice([3, 5]))),
                    (True, "switch", int(rng.choice([7, 9]))),
                    (True, "any", None)]
            plan += [(bool(rng.integers(2)), "any", None)
                     for _ in range(int(rng.integers(0, 3)))]
        else:
            plan = [(bool(rng.integers(2)), "any", int(rng.choice([1, 3, 5])))
                    for _ in range(int(rng.integers(0, 5)))]
        success = True
        for accepted, policy, size in plan:
            if size is None and not test:
                size = int(rng.choice([1, 3, 5]))
            candidate = _extend(field, background, table, rng, elbow, accepted, policy, size)
            if candidate is None:
                success = False
                break
            field = candidate
        if not success and (teaching is not None or test):
            continue
        desired_sources = (int(rng.integers(2, 5)) if test else
                           1 if teaching == 0 else
                           int(rng.integers(2, 4)) if teaching == 1 else int(rng.integers(1, 4)))
        for _ in range(desired_sources-1):
            result = _add_source(field, background, table, rng)
            if result is None:
                break
            field, extra = result
            if teaching is None and rng.random() < .4:
                candidate = _extend(field, background, table, rng, extra,
                                    bool(rng.integers(2)), length=int(rng.choice([1, 3, 5])))
                if candidate is not None:
                    field = candidate
        if teaching == 1 and len(_objects(field, background)[0]) < 2:
            continue
        # Optional genuine off-route mirror; not every key/object must be used.
        if rng.random() < .35:
            for _ in range(12):
                mid = int(rng.integers(h)), int(rng.integers(w))
                axis, size = int(rng.integers(2)), int(rng.choice([1, 3, 5]))
                candidate = _place(field, _mirror_cells(mid, axis, size),
                                   int(rng.choice(colors)), background)
                if candidate is None:
                    continue
                try:
                    _, before = _trace(field, background, table)
                    _, after = _trace(candidate, background, table)
                except AssertionError:
                    continue
                used = {e["mirror"] for q in after["routes"] for e in q["events"]}
                if len(after["mirrors"])-len(used) > len(before["mirrors"])-len({
                        e["mirror"] for q in before["routes"] for e in q["events"]}):
                    field = candidate
                    break
        out, _ = _trace(field, background, table)
        header = _make_header(rng, w, background, frame, table)
        inp = np.vstack((header, field))
        pair = dict(input=inp, output=out)
        if teaching is not None and not _ablation_difference(inp, "no_recolor"):
            continue
        return pair
    raise RuntimeError("bounded construction could not find a clear scene")


def _ablation_difference(inp, mode):
    return _ablation_status(inp, mode) != "equal"


def _ablation_status(inp, mode):
    info = _decode(inp)
    expected = _trace(info["field"], info["background"], info["table"])[0]
    try:
        alternative = _trace(info["field"], info["background"], info["table"], mode=mode)[0]
    except AssertionError:
        return "invalid"
    return "equal" if np.array_equal(expected, alternative) else "different"


def _features(inp):
    info = _decode(inp)
    _, traced = _trace(info["field"], info["background"], info["table"])
    events = [e for q in traced["routes"] for e in q["events"]]
    accepted = [e for e in events if e["accepted"]]
    lengths = tuple(len(m["cells"]) for m in traced["mirrors"])
    return dict(info=info, traced=traced, events=events, accepted=accepted,
                source_count=len(traced["sources"]), mirror_count=len(lengths),
                lengths=lengths, rejected=sum(not e["accepted"] for e in events),
                max_events=max(len(q["events"]) for q in traced["routes"]))


def _teaching_coverage(pairs):
    features = [_features(p["input"]) for p in pairs]
    accepted = [e for f in features for e in f["accepted"]]
    headings = {s: {e["heading"] for e in accepted if e["side"] == s} for s in ("L", "R")}
    lookup_sides = {}
    for f in features:
        for e in f["accepted"]:
            lookup_sides.setdefault(e["incoming"], set()).add(f["info"]["table"][e["incoming"]])
    accepted_bar_axes = {e["axis"] for f in features for e in f["accepted"]
                         if len(f["traced"]["mirrors"][e["mirror"]]["cells"]) > 1}
    accepted_point = any(len(f["traced"]["mirrors"][e["mirror"]]["cells"]) == 1
                         for f in features for e in f["accepted"])
    return (all(len(v) >= 2 for v in headings.values())
            and accepted_bar_axes == {0, 1} and accepted_point
            and len({q["source"]["direction"] for f in features for q in f["traced"]["routes"]}) >= 2
            and {3, 5} <= {n for f in features for n in f["lengths"]}
            and any(e["incoming"] == e["outgoing"] for e in accepted)
            and any(e["incoming"] != e["outgoing"] for e in accepted)
            and any(f["rejected"] for f in features)
            and any(f["source_count"] >= 2 for f in features)
            and any(len(sides) == 2 for sides in lookup_sides.values())
            and all(any(_ablation_status(p["input"], mode) == "different" for p in pairs)
                    for mode in ("all_reflect", "no_recolor"))
            and all(any(_ablation_difference(p["input"], mode) for p in pairs)
                    for mode in ("all_reflect", "no_recolor", "frozen_key", "mirror_key", "stop_rejected")))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, count, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or not low <= int(count) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    anchor_color = int(rng.integers(10))
    anchor_side = "L" if rng.integers(2) else "R"
    for _ in range(20):
        d = int(rng.integers(4))
        train = [_pair(rng, teaching=i, direction=DIAGONALS[(d+i)%4],
                       anchor_color=anchor_color,
                       anchor_side=anchor_side if i == 0 else ("R" if anchor_side == "L" else "L"))
                 for i in range(2)]
        if _teaching_coverage(train):
            break
    else:
        raise RuntimeError("bounded teaching selection exhausted")
    seen = {_signature(p["input"]) for p in train}
    puzzle = dict(train=train, test=[])
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        while len(puzzle[split]) < count:
            for _ in range(30):
                pair = _pair(rng, test=split == "test")
                key = _signature(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded unique input selection exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen, training_lengths = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _decode(inp)
            predicted, traced = _trace(info["field"], info["background"], info["table"])
            _require(out.shape == info["field"].shape, "incorrect crop")
            _require(np.array_equal(out, predicted), "incorrect completed rays")
            occupied = info["field"] != info["background"]
            _require(np.array_equal(out[occupied], info["field"][occupied]), "object changed")
            _require(np.any(out != info["field"]), "no visible completion")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "new color")
            for route in traced["routes"]:
                r, c = route["exit"]
                _require(not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]), "route must exit")
                for e in route["events"]:
                    _require(e["accepted"] == (info["table"][e["incoming"]] == e["side"]),
                             "filter uses wrong color")
                    expected = traced["mirrors"][e["mirror"]]["color"] if e["accepted"] else e["incoming"]
                    _require(e["outgoing"] == expected and out[e["position"]] == expected,
                             "incorrect contact recoloring")
            if split == "train":
                training_lengths.update(len(m["cells"]) for m in traced["mirrors"])
            else:
                _require(any(e["accepted"] and len(traced["mirrors"][e["mirror"]]["cells"]) >= 7
                             for q in traced["routes"] for e in q["events"]),
                         "test must use an accepted longer mirror")
    _require({3, 5} <= training_lengths, "midpoint length evidence missing")
    _require(_teaching_coverage(puzzle["train"]), "training lacks rule/shortcut evidence")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        all(np.array_equal(p[n], q[n]) for n in ("input", "output"))
        for p, q in zip(a[s], b[s])) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed diversity and meaningful positive/negative checks."""
    records, signatures, normalized = [], set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                records.append(_features(pair["input"]))
                mapping = {}
                normalized.add((pair["input"].shape, bytes(
                    mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)))
    _require(len(signatures) == 42, "seed diversity")
    _require(len(normalized) >= 170, "structure diversity")
    _require(len({f["info"]["field"].shape for f in records}) >= 40, "shape diversity")
    _require({f["info"]["background"] for f in records} == set(range(10)), "background coverage")
    _require({f["source_count"] for f in records} >= {1, 2, 3}, "source count diversity")
    _require({n for f in records for n in f["lengths"]} >= {1, 3, 5, 7, 9}, "bar length coverage")
    _require({len(f["info"]["table"]) for f in records} == set(range(2, 7)), "key count diversity")
    _require(any(f["max_events"] >= 5 for f in records), "longer routes missing")
    _require(any(f["max_events"] == 0 for f in records), "straight scenes missing")
    _require(any(len(f["info"]["table"]) > len({m["color"] for m in f["info"]["mirrors"]}
                 | {s["color"] for s in f["info"]["sources"]}) for f in records), "unused keys missing")
    _require(any(f["mirror_count"] > len({e["mirror"] for e in f["events"]}) for f in records),
             "off-route mirrors missing")
    _require(_equal(generate_puzzle(0), generate_puzzle(0)), "determinism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1200+nt*10+nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (None, 1), (True, 1), (4, 1.0)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _regressions()


def _regressions():
    # Enumerated field expectations, independent of random scene construction.
    # SE source: reject a right-turn vertical bar, then accept a left-turn
    # horizontal bar, changing 1 -> 2 and therefore L -> R.
    f = np.zeros((15, 17), dtype=np.int8)
    f[0, 1] = f[1, 0] = f[1, 1] = 1
    f[3:6, 5] = 2
    f[8, 5:10] = 2
    f[2, 9:14] = 1
    expected = f.copy()
    for r, c in ((2, 2), (3, 3), (4, 4), (5, 5), (6, 6)):
        if f[r, c] == 0:
            expected[r, c] = 1
    for p in ((7, 7), (6, 8), (5, 9), (4, 10)):
        expected[p] = 2
    for p in ((3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16)):
        expected[p] = 1
    out, info = _trace(f, 0, {1: "L", 2: "R"})
    _require(np.array_equal(out, expected), "explicit two-state trace")
    _require([e["accepted"] for e in info["routes"][0]["events"]] == [False, True, True],
             "rejected/accepted event sequence")
    header = _make_header(np.random.default_rng(7), 17, 0, 9, {1: "L", 2: "R"})
    inp = np.vstack((header, f))
    _require(np.array_equal(solve(inp), expected), "input-only decoding of explicit field")
    permutation = np.array([7, 4, 8, 0, 1, 2, 3, 5, 6, 9], dtype=np.int8)
    _require(np.array_equal(solve(permutation[inp]), permutation[expected]), "palette equivariance")
    # Longer odd bars use the same geometric midpoint.
    long = f.copy()
    long[8, 2:13] = 2
    expected_long = expected.copy()
    expected_long[8, 2:13] = 2
    _require(np.array_equal(solve(np.vstack((header, long))), expected_long),
             "eleven-cell midpoint mirror")
    even = f.copy()
    even[10, 1:5] = 2
    competing = f.copy()
    competing[5, 4] = 3
    for bad, label in ((even, "even bar"), (competing, "competing midpoint contacts")):
        try:
            _trace(bad, 0, {1: "L", 2: "R", 3: "L"})
        except AssertionError:
            pass
        else:
            raise AssertionError(label + " accepted")
    p = generate_puzzle(0)
    bad_puzzle = {s: [dict(q) for q in p[s]] for s in p}
    pair = bad_puzzle["test"][0]
    pair["output"] = pair["output"].copy()
    info = _decode(pair["input"])
    changed = tuple(np.argwhere(pair["output"] != info["field"])[0])
    pair["output"][changed] = info["field"][changed]
    try:
        validate_generated_puzzle(bad_puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing path cell accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
