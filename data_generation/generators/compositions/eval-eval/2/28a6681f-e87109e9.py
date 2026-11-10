"""Finite liquid in a circuit of keyed, directionally oriented reservoirs.

Only standard library and NumPy are used. No parent generator is imported.
See task_spec.md for the input language and the complete transformation.
"""
from __future__ import annotations

import math
import numpy as np


# Clockwise screen headings: north, east, south, west.
_DIR = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SEEDS = (0, 251621184, 690865354, 799824403, 546745900, 913331152,
          88080360, 107428952, 375307402, 121037194, 184130575, 411080839,
          844065725, 405773429, 612670860, 992650158, 944546506, 79005034,
          801469531, 381299075, 219692161, 299312752, 302449974, 969017696,
          496023864, 756956816, 399084230, 452012773, 709389536, 578969971,
          688881688, 75006073, 645550419, 426272014, 173681816, 497168158,
          552895344, 847905689, 501777996, 998011253, 81377138, 123623477)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid color")
    return a.astype(np.int8, copy=True)


def _add(point, direction, distance=1):
    dr, dc = _DIR[direction]
    return point[0] + distance * dr, point[1] + distance * dc


def _turn(direction, side):
    return (direction + (-1 if side == "L" else 1)) % 4


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], {root}
        while stack:
            point = stack.pop()
            for d in range(4):
                other = _add(point, d)
                if other in remaining:
                    remaining.remove(other)
                    cells.add(other)
                    stack.append(other)
        groups.append(cells)
    return groups


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _cup_cells(origin, direction, profile):
    """Local rows run with flow; local columns run to its relative left."""
    span, depth = len(profile), max(profile)
    walls, cavity, levels = set(), set(), {}
    for u in range(depth + 1):
        for v in range(span + 2):
            p = _add(_add(origin, direction, u), (direction - 1) % 4, v)
            if v in (0, span + 1) or u >= profile[v - 1]:
                walls.add(p)
            else:
                cavity.add(p)
                levels.setdefault(u, set()).add(p)
    return dict(origin=origin, direction=direction, profile=tuple(profile),
                walls=walls, cavity=cavity, levels=levels,
                box=_bounds(walls), capacity=len(cavity))


def _parse_cup(cells, color):
    a, b, c, e = _bounds(cells)
    # Exactly one edge is an open mouth. The other three are solid.
    edges = ({(a, x) for x in range(c, e + 1)},
             {(x, e) for x in range(a, b + 1)},
             {(b, x) for x in range(c, e + 1)},
             {(x, c) for x in range(a, b + 1)})
    missing = [d for d, edge in enumerate(edges) if not edge <= cells]
    _require(len(missing) == 1, "cup must have exactly one open face")
    mouth = missing[0]
    direction = (mouth + 2) % 4
    origin = ((b, e), (b, c), (a, c), (a, e))[direction]
    span = (e - c - 1) if direction % 2 == 0 else (b - a - 1)
    depth = (b - a) if direction % 2 == 0 else (e - c)
    _require(span >= 1 and depth >= 1, "cup has no interior")
    profile = []
    for v in range(1, span + 1):
        line = [_add(_add(origin, direction, u), (direction - 1) % 4, v)
                for u in range(depth + 1)]
        hits = [u for u, p in enumerate(line) if p in cells]
        _require(hits and hits[0] >= 1, "closed or ambiguous mouth")
        _require(hits == list(range(hits[0], depth + 1)), "floor contains a hole")
        profile.append(hits[0])
    cup = _cup_cells(origin, direction, profile)
    _require(cup["walls"] == cells, "cup walls are not one cell thick")
    for u in range(depth):
        free = [v for v, f in enumerate(profile) if f > u]
        _require(free and free == list(range(min(free), max(free) + 1)),
                 "floor must define nested, connected levels")
    cup["color"] = color
    return cup


def _decode(input_grid):
    a = _grid(input_grid)
    frame = int(a[0, 0])
    _require(np.all(a[0] == frame), "missing key top frame")
    bottoms = np.flatnonzero(np.all(a[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing key bottom frame")
    end = int(bottoms[0])
    _require(2 <= end < len(a) - 2, "invalid key band height")
    band = a[1:end]
    _require(np.all(band == band[0]), "inconsistent key rows")
    row, width = band[0], a.shape[1]
    walls = np.flatnonzero(row == frame).tolist()
    _require(walls[0] == 0 and walls[-1] == width - 1, "missing panel side walls")
    runs = [(x + 1, y) for x, y in zip(walls, walls[1:]) if y > x + 1]
    _require(runs and all(y - x >= 3 for x, y in runs), "key panels too narrow")
    background = int(row[runs[0][0] + 1])
    turns = {}
    for x, y in runs:
        _require(np.all(row[x + 1:y - 1] == background), "inconsistent panel background")
        left, right = int(row[x]), int(row[y - 1])
        _require((left == background) != (right == background), "ambiguous side bar")
        color, side = (right, "R") if left == background else (left, "L")
        _require(color not in (background, frame) and color not in turns, "conflicting key")
        turns[color] = side
    field = a[end + 1:].copy()
    other = set(map(int, np.unique(field))) - {frame, background} - set(turns)
    _require(len(other) == 1, "supply color is not unique")
    liquid = other.pop()
    source_groups = _components(field == frame)
    _require(len(source_groups) == 1, "need one supply chamber")
    shell = source_groups[0]
    sr, er, sc, ec = _bounds(shell)
    _require(er - sr >= 2 and ec - sc >= 2, "supply chamber too small")
    border = {(r, c) for r in range(sr, er + 1) for c in range(sc, ec + 1)
              if r in (sr, er) or c in (sc, ec)}
    holes = border - shell
    _require(shell <= border and len(holes) == 1, "supply must be a rectangular ring with one spout")
    gap = next(iter(holes))
    _require(gap not in ((sr, sc), (sr, ec), (er, sc), (er, ec)), "corner spout")
    direction = 0 if gap[0] == sr else 2 if gap[0] == er else 3 if gap[1] == sc else 1
    supply = set(map(tuple, np.argwhere(field == liquid).tolist()))
    _require(supply and all(sr < r < er and sc < c < ec for r, c in supply),
             "liquid must start inside the supply chamber")
    _require(int(field[gap]) == background, "spout must be empty")
    _require(all(int(field[r, c]) in (background, liquid)
                 for r in range(sr + 1, er) for c in range(sc + 1, ec)), "object in supply chamber")
    cups, lookup = [], {}
    for color in turns:
        for cells in _components(field == color):
            cup = _parse_cup(cells, color)
            _require(all(int(field[p]) == background for p in cup["cavity"]), "cups must start empty")
            index = len(cups)
            cups.append(cup)
            lookup.update({p: index for p in cells})
    _require(cups, "no receivers")
    return dict(field=field, frame=frame, background=background, liquid=liquid,
                turns=turns, cups=cups, lookup=lookup, shell=shell, supply=supply,
                start=_add(gap, direction), direction=direction, crop=end + 1,
                panel_widths=tuple(y - x + 2 for x, y in runs))


def _next_cup(info, start, direction):
    """Dry one-cell probe: transit is used for routing but never painted."""
    h, w = info["field"].shape
    point = start
    while 0 <= point[0] < h and 0 <= point[1] < w:
        value = int(info["field"][point])
        if point in info["lookup"]:
            index = info["lookup"][point]
            cup = info["cups"][index]
            _require(cup["direction"] == direction, "stream struck a cup's closed face")
            delta = (point[0] - cup["origin"][0], point[1] - cup["origin"][1])
            dr, dc = _DIR[direction]
            lr, lc = _DIR[(direction - 1) % 4]
            u, v = delta[0] * dr + delta[1] * dc, delta[0] * lr + delta[1] * lc
            _require(1 <= v <= len(cup["profile"]) and u == cup["profile"][v - 1],
                     "stream did not enter the open mouth")
            _require(cup["profile"][v - 1] == max(cup["profile"]), "entry must reach the deepest level")
            entry = _add(cup["origin"], (direction - 1) % 4, v)
            return index, _add(entry, direction, -1)
        _require(value == info["background"], "stream struck supply terrain")
        point = _add(point, direction)
    return None


def _run(info, turns=None, unlimited=False, visit_limit=None):
    turns = info["turns"] if turns is None else turns
    output = info["field"].copy()
    output[output == info["liquid"]] = info["background"]
    remaining = len(info["supply"])
    start, direction = info["start"], info["direction"]
    seen, visits, events = set(), [], []
    while remaining > 0 or unlimited:
        hit = _next_cup(info, start, direction)
        if hit is None:
            _require(unlimited, "surplus has no receiver before the boundary")
            break
        index, pivot = hit
        _require(index not in seen, "cyclic reservoir route")
        seen.add(index)
        cup = info["cups"][index]
        amount = cup["capacity"] if unlimited else min(remaining, cup["capacity"])
        to_place = amount
        for u in sorted(cup["levels"], reverse=True):
            layer = cup["levels"][u]
            if to_place == 0:
                break
            _require(to_place >= len(layer), "supply would partially fill a level")
            for p in layer:
                output[p] = info["liquid"]
            to_place -= len(layer)
        _require(to_place == 0, "bad reservoir capacity")
        visits.append((index, amount))
        if visit_limit is not None and len(visits) >= visit_limit:
            break
        if not unlimited:
            remaining -= amount
            if remaining == 0:
                break
        side = turns[cup["color"]]
        events.append((index, direction, side))
        start, direction = pivot, _turn(direction, side)
    return output, visits, events


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode all conventions from this input and conserve its finite supply."""
    return _run(_decode(input_grid))[0]


def _profile(rng, depth=None, span=None):
    span = int(rng.integers(2, 5)) if span is None else span
    depth = int(rng.integers(1, 4)) if depth is None else depth
    f = [1] * span
    left, right = 0, span
    flat = rng.random() < .3
    for u in range(1, depth):
        if not flat and right - left > 1:
            remove = int(rng.integers(3))
            if remove == 1:
                left += 1
            elif remove == 2:
                right -= 1
        for v in range(left, right):
            f[v] = u + 1
    return tuple(f)


def _separate(box, old):
    a, b, c, e = box
    x, y, z, t = old
    return b + 1 < x or y + 1 < a or e + 1 < z or t + 1 < c


def _fits(boxes, maxh, maxw):
    return (max(b[1] for b in boxes) - min(b[0] for b in boxes) + 1 <= maxh and
            max(b[3] for b in boxes) - min(b[2] for b in boxes) + 1 <= maxw)


def _layout(rng, profiles, sides, inner_h, inner_w, gap_col, maxh, maxw):
    sh, sw = inner_h + 2, inner_w + 2
    shell = {(r, c) for r in range(sh) for c in range(sw)
             if r in (0, sh - 1) or c in (0, sw - 1)}
    shell.remove((sh - 1, gap_col))
    source_box = (0, sh - 1, 0, sw - 1)
    nodes = [0]

    def grow(index, start, direction, placed, occupied, protected):
        if index == len(profiles):
            return placed
        nodes[0] += 1
        if nodes[0] > 700:
            return None
        f = profiles[index]
        entries = [v + 1 for v, value in enumerate(f) if value == max(f)]
        candidates = [(t, v) for t in range(2, 13) for v in entries]
        rng.shuffle(candidates)
        for distance, entry in candidates:
            mouth = _add(start, direction, distance)
            origin = _add(mouth, (direction - 1) % 4, -entry)
            cup = _cup_cells(origin, direction, f)
            boxes = [source_box] + [q["box"] for q in placed]
            if not all(_separate(cup["box"], box) for box in boxes):
                continue
            if not _fits(boxes + [cup["box"]], maxh, maxw):
                continue
            segment = {_add(start, direction, t) for t in range(distance + max(f))}
            if segment & occupied or cup["walls"] & protected:
                continue
            cup["pivot"] = _add(mouth, direction, -1)
            cup["side"] = sides[index]
            result = grow(index + 1, cup["pivot"], _turn(direction, sides[index]),
                          placed + [cup], occupied | cup["walls"], protected | segment)
            if result is not None:
                return result
        return None

    cups = grow(0, (sh, gap_col), 2, [], shell, set())
    return None if cups is None else (shell, source_box, cups)


def _add_branch(rng, shell, source_box, cups, remainder, maxh, maxw):
    first = cups[0]
    direction = _turn(first["direction"], "R" if first["side"] == "L" else "L")
    # A counterfactual receiver holds exactly the entire first-cup surplus.
    spans = list(range(2, min(5, remainder) + 1))
    rng.shuffle(spans)
    for span in spans:
        depth = math.ceil(remainder / span)
        if depth > 7:
            continue
        extra = remainder - (depth - 1) * span
        left = int(rng.integers(span - extra + 1))
        f = [depth - 1] * span
        for v in range(left, left + extra):
            f[v] = depth
        if min(f) < 1:
            continue
        entries = [v + 1 for v, value in enumerate(f) if value == depth]
        candidates = [(t, v) for t in range(2, 13) for v in entries]
        rng.shuffle(candidates)
        for distance, entry in candidates:
            mouth = _add(first["pivot"], direction, distance)
            origin = _add(mouth, (direction - 1) % 4, -entry)
            cup = _cup_cells(origin, direction, f)
            boxes = [source_box] + [q["box"] for q in cups]
            if not all(_separate(cup["box"], b) for b in boxes) or not _fits(boxes + [cup["box"]], maxh, maxw):
                continue
            occupied = shell | set().union(*(q["walls"] for q in cups))
            segment = {_add(first["pivot"], direction, t) for t in range(distance + depth)}
            if segment & occupied:
                continue
            cup["side"] = str(rng.choice(["L", "R"]))
            return cup
    return None


def _header(rng, width, height, frame, background, turns):
    colors = list(turns)
    rng.shuffle(colors)
    # Shared panel walls, at least three interior columns per panel.
    interiors = np.full(len(colors), 3, dtype=int)
    interiors += rng.multinomial(width - (4 * len(colors) + 1), np.full(len(colors), 1 / len(colors)))
    header = np.full((height, width), background, dtype=np.int8)
    header[[0, -1]] = frame
    c = 0
    for color, size in zip(colors, interiors):
        header[1:-1, c] = frame
        header[1:-1, c + size + 1] = frame
        bar = c + 1 if turns[color] == "L" else c + size
        header[1:-1, bar] = color
        c += int(size) + 1
    return header


def _make_pair(rng, parameters, visited, mode, rotation, branch_required=False):
    bg, liquid, frame = parameters["roles"]
    hh = int(rng.integers(3, 6))
    maxh, maxw = ((30 - hh, 30) if rotation % 2 == 0 else (30, 30 - hh))
    # One extra cup is reachable geometrically but must stay dry.
    for attempt in range(80):
        if mode == "LL":
            sides = ["L", "L", "R", str(rng.choice(["L", "R"]))]
        elif mode == "RR":
            sides = ["R", "R", "L", str(rng.choice(["L", "R"]))]
        else:
            pattern = str(rng.choice(["LRLR", "RLRL", "LLRL", "RRLR"]))
            sides = list(pattern[:visited]) + [str(rng.choice(["L", "R"]))]
        # Late bounded retries use small capacities and shorter source chambers,
        # while retaining the requested number of activations and the same rule.
        small = attempt >= 60
        profiles = [((2, 2) if small else _profile(rng)) for _ in sides]
        profiles[visited - 1] = ((2, 2) if small else _profile(rng, depth=int(rng.integers(2, 5))))
        terminal = profiles[visited - 1]
        levels = [sum(value > u for value in terminal) for u in range(max(terminal) - 1, -1, -1)]
        exact = mode == "RR" or (mode not in ("LL",) and rng.random() < .25)
        count = len(levels) if exact else int(rng.integers(1, len(levels)))
        volume = sum(sum(f) for f in profiles[:visited - 1]) + sum(levels[:count])
        iw = 4 if small else int(rng.integers(max(3, math.ceil(volume / 7)), 9))
        ih = math.ceil(volume / iw) + (0 if small else int(rng.integers(0, 2)))
        gap_col = int(rng.integers(1, iw + 1))
        layout = _layout(rng, profiles, sides, ih, iw, gap_col, maxh - 1, maxw - 1)
        if layout is None:
            continue
        shell, source_box, cups = layout
        if branch_required or rng.random() < .45:
            branch = _add_branch(rng, shell, source_box, cups, volume - cups[0]["capacity"], maxh - 1, maxw - 1) if visited > 1 else None
            if branch_required and branch is None:
                continue
            if branch is not None:
                cups.append(branch)
        nkeys = int(rng.integers(2, 6))
        colors = list(parameters["key_colors"][:nkeys])
        # The first color reverses its meaning in the two mandatory lessons.
        turns = {colors[0]: sides[0], colors[1]: "R" if sides[0] == "L" else "L"}
        for color in colors[2:]:
            turns[color] = str(rng.choice(["L", "R"]))
        for i, cup in enumerate(cups):
            eligible = [c for c in colors if turns[c] == cup["side"]]
            cup["color"] = colors[0] if i == 0 else int(rng.choice(eligible))
        all_cells = shell | set().union(*(q["walls"] for q in cups))
        a, b, c, e = _bounds(all_cells)
        ch, cw = b - a + 1, e - c + 1
        # Width after rotation must fit the visible instruction panels.
        minimum = 4 * nkeys + 1
        lowh, loww = max(ch, minimum if rotation % 2 else 1), max(cw, minimum if not rotation % 2 else 1)
        if lowh > maxh or loww > maxw:
            continue
        h = int(rng.integers(lowh, min(maxh, lowh + parameters["padding"]) + 1))
        w = int(rng.integers(loww, min(maxw, loww + parameters["padding"]) + 1))
        ro = int(rng.integers(h - ch + 1)) - a
        co = int(rng.integers(w - cw + 1)) - c
        field = np.full((h, w), bg, dtype=np.int8)
        for r, col in shell:
            field[r + ro, col + co] = frame
        for cup in cups:
            for r, col in cup["walls"]:
                field[r + ro, col + co] = cup["color"]
        points = [(r, col) for r in range(1, ih + 1) for col in range(1, iw + 1)]
        style = int(rng.integers(4))
        jitter = rng.random(len(points))
        anchor = points[int(rng.integers(len(points)))]
        score = [jitter[i] if style == 0 else
                 abs(r - anchor[0]) + abs(col - anchor[1]) + jitter[i] if style == 1 else
                 5 * r + col + jitter[i] if style == 2 else
                 5 * col + r + jitter[i] for i, (r, col) in enumerate(points)]
        for i in np.argsort(score)[:volume]:
            r, col = points[int(i)]
            field[r + ro, col + co] = liquid
        field = np.rot90(field, rotation).copy()
        inp = np.vstack((_header(rng, field.shape[1], hh, frame, bg, turns), field))
        try:
            info = _decode(inp)
            out, actual_visits, _ = _run(info)
            if len(actual_visits) != visited:
                continue
            # Validate the planned spare cup really follows the terminal one.
            if branch_required:
                changed = dict(info["turns"])
                first = info["cups"][actual_visits[0][0]]["color"]
                changed[first] = "R" if changed[first] == "L" else "L"
                alt = _run(info, changed)[0]
                if np.array_equal(alt, out):
                    continue
            # Unlike a beam, this circuit never colors its transit segments.
            pair = {"input": inp.astype(np.int8), "output": out.astype(np.int8)}
            _inspect(pair)
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded circuit construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and int(seed) >= 0, "seed must be a nonnegative integer")
    _require(_integer(num_train) and 2 <= int(num_train) <= 10, "num_train must be 2..10")
    _require(_integer(num_test) and 1 <= int(num_test) <= 4, "num_test must be 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    parameters = dict(roles=tuple(palette[:3]), key_colors=tuple(palette[3:]),
                      padding=int(rng.integers(2, 7)))
    first_rotation = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            if split == "train" and index < 2:
                mode, visited = ("LL" if index == 0 else "RR"), 3
                rotation = (first_rotation + index) % 4
            else:
                mode = "mixed"
                visited = 4 if split == "test" else int(rng.integers(1, 4))
                rotation = int(rng.integers(4))
            for _ in range(12):
                pair = _make_pair(rng, parameters, visited, mode, rotation,
                                  branch_required=split == "train" and index == 0)
                key = _signature(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("distinct circuit construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _inspect(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _decode(inp)
    expected, visits, events = _run(info)
    _require(np.array_equal(expected, out), "incorrect circuit output")
    field = info["field"]
    _require(out.shape == field.shape, "incorrect field crop")
    bg, liquid = info["background"], info["liquid"]
    solid = (field != bg) & (field != liquid)
    _require(np.array_equal(field[solid], out[solid]), "terrain changed")
    _require(int((out == liquid).sum()) == len(info["supply"]), "liquid volume changed")
    _require(all(int(out[p]) == bg for p in info["supply"]), "supply was not drained")
    filled = set(map(tuple, np.argwhere(out == liquid).tolist()))
    all_cavities = set().union(*(q["cavity"] for q in info["cups"]))
    _require(filled <= all_cavities, "painted a transit cell")
    for cup in info["cups"]:
        used = filled & cup["cavity"]
        started = False
        for u in sorted(cup["levels"]):
            layer = cup["levels"][u]
            occupied = used & layer
            _require(not occupied or occupied == layer, "broken meniscus")
            if occupied:
                started = True
            elif started:
                raise ValueError("unfilled deeper level")
    _require(len(visits) < len(info["cups"]), "no dry spare receiver")
    # Reconstruct the outgoing probe even though finite supply stops here.
    last_index, last_amount = visits[-1]
    start, direction = info["start"], info["direction"]
    for index, _ in visits:
        hit = _next_cup(info, start, direction)
        _require(hit is not None and hit[0] == index, "inconsistent route")
        cup = info["cups"][index]
        start, direction = hit[1], _turn(direction, info["turns"][cup["color"]])
    spare = _next_cup(info, start, direction)
    _require(spare is not None and spare[0] not in {i for i, _ in visits}, "missing downstream dry cup")
    _require(not (filled & info["cups"][spare[0]]["cavity"]), "surplus invented")
    info.update(visits=visits, events=events,
                terminal_full=last_amount == info["cups"][last_index]["capacity"])
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, decoded = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            key = _signature(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            decoded[split].append(_inspect(pair))
    all_infos = decoded["train"] + decoded["test"]
    _require(len({(q["background"], q["liquid"], q["frame"]) for q in all_infos}) == 1, "puzzle roles changed")
    headings = {side: {d for q in decoded["train"] for _, d, s in q["events"] if s == side}
                for side in ("L", "R")}
    _require(all(len(ds) >= 2 for ds in headings.values()), "both relative turns need multiple headings")
    _require({q["terminal_full"] for q in decoded["train"]} == {False, True}, "need partial and exact-exhaustion lessons")
    _require({cup["direction"] for q in decoded["train"] for cup in q["cups"]} == set(range(4)), "missing cup orientations")
    _require(all(len(q["visits"]) <= 3 for q in decoded["train"]) and
             all(len(q["visits"]) == 4 for q in decoded["test"]), "missing activation-depth shift")
    tables = [q["turns"] for q in decoded["train"]]
    _require(any(len({t[c] for t in tables if c in t}) == 2 for c in range(10)), "key meaning never changes")
    key_witness = False
    for q in decoded["train"]:
        altered = dict(q["turns"])
        first_color = q["cups"][q["visits"][0][0]]["color"]
        altered[first_color] = "R" if altered[first_color] == "L" else "L"
        try:
            alternate = _run(q, altered)[0]
        except ValueError:
            continue
        key_witness |= not np.array_equal(alternate, _run(q)[0])
    _require(key_witness, "no valid answer-changing key ablation")


def _manual_fill_checks():
    """Literal geometry and hand-calculated targets, separate from route assembly."""
    field = np.zeros((16, 22), dtype=np.int8)
    field[1, 1:7] = field[4, 1:7] = 5
    field[1:5, 1] = field[1:5, 6] = 5
    field[4, 4] = 0
    # South-facing cup: four free cells in two levels.
    field[7:10, 3] = field[7:10, 6] = 2
    field[9, 3:7] = 2
    # East-facing cup: two free vertical levels, each two cells high.
    field[4, 12:15] = field[7, 12:15] = 3
    field[4:8, 14] = 3
    # A third cup is geometrically next, but finite supply never reaches it.
    field[12:15, 10] = field[12:15, 13] = 2
    field[14, 10:14] = 2
    points = ((2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 3), (3, 4), (3, 5))
    for volume in (6, 8):
        inp = field.copy()
        for p in points[:volume]:
            inp[p] = 8
        expected = field.copy()
        expected[7:9, 4:6] = 8
        expected[5:7, 13] = 8
        if volume == 8:
            expected[5:7, 12] = 8
        for rotation in range(4):
            f = np.rot90(inp, rotation)
            header = np.zeros((3, f.shape[1]), dtype=np.int8)
            header[[0, 2]] = 5
            header[1, [0, f.shape[1] // 2, f.shape[1] - 1]] = 5
            header[1, 1] = 2
            header[1, -2] = 3
            combined = np.vstack((header, f))
            _require(np.array_equal(solve(combined), np.rot90(expected, rotation)),
                     "manual rotated capacity fixture failed")
        if volume == 6:
            bad = inp.copy()
            bad[points[6]] = 8  # Seven cannot occupy whole two-cell levels.
            header = np.zeros((3, 22), dtype=np.int8)
            header[[0, 2]] = 5
            header[1, [0, 11, 21]] = 5
            header[1, 1], header[1, 20] = 2, 3
            try:
                solve(np.vstack((header, bad)))
            except ValueError:
                pass
            else:
                raise AssertionError("partial-width level accepted")


def self_test():
    """Bounded authored checks on the 42 supplied seeds, plus API boundaries."""
    _manual_fill_checks()
    shapes, normalized, roles, volumes, capacities = set(), set(), [set(), set(), set()], set(), set()
    visits, profiles, key_counts, cup_counts, source_components = set(), set(), set(), set(), set()
    signatures = set()
    partial = exact = branches = 0
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                q = _inspect(pair)
                _require(np.array_equal(solve(np.asarray(pair["input"]).copy()), pair["output"]), "input-only solve failed")
                shapes.add(q["field"].shape)
                for values, role in zip(roles, (q["background"], q["liquid"], q["frame"])):
                    values.add(role)
                volumes.add(len(q["supply"]))
                visits.add(len(q["visits"]))
                key_counts.add(len(q["turns"]))
                cup_counts.add(len(q["cups"]))
                source_components.add(len(_components(q["field"] == q["liquid"])))
                for cup in q["cups"]:
                    capacities.add(cup["capacity"])
                    profiles.add(cup["profile"])
                canonical = np.where(q["field"] == q["background"], 0,
                             np.where(q["field"] == q["liquid"], 1,
                             np.where(q["field"] == q["frame"], 2, 3)))
                normalized.add(_signature(canonical))
                partial += not q["terminal_full"]
                exact += q["terminal_full"]
                branches += len(q["cups"]) > len(q["visits"]) + 1
        train_infos = [_decode(p["input"]) for p in puzzle["train"]]
        # A dry successor would become filled by a capacity-free propagation.
        q = train_infos[0]
        finite = _run(q)[0]
        unlimited = _run(q, unlimited=True, visit_limit=len(_run(q)[1]) + 1)[0]
        _require(not np.array_equal(unlimited, finite), "capacity ablation made no difference")
    _require(len(signatures) == 42 and len(normalized) >= 180, "insufficient structural variation")
    _require(len(shapes) >= 35 and len(volumes) >= 20 and len(capacities) >= 12, "insufficient geometric variation")
    _require(all(len(r) >= 8 for r in roles), "insufficient palette-role variation")
    _require(visits == {1, 2, 3, 4} and len(profiles) >= 30, "missing circuit/floor variation")
    _require(key_counts == {2, 3, 4, 5} and len(cup_counts) >= 4 and branches >= 42, "missing topology/key variation")
    _require(partial >= 80 and exact >= 42 and max(source_components) >= 3, "missing supply/level variation")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(0, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "requested counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.5, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, again = generate_puzzle(0), generate_puzzle(0)
    _require(all(np.array_equal(a[n], b[n]) for s in p for a, b in zip(p[s], again[s])
                 for n in ("input", "output")), "seed determinism failed")
    # Input-only behavior must also survive a global bijective palette change.
    perm = np.random.default_rng(123).permutation(10)
    for pair in p["train"] + p["test"]:
        _require(np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]]), "literal color dependency")
    return dict(seeds=42, pairs=210, field_shapes=len(shapes), normalized_structures=len(normalized),
                volumes=len(volumes), capacities=len(capacities), floor_profiles=len(profiles),
                role_colors=[len(r) for r in roles], visited_counts=sorted(visits),
                key_counts=sorted(key_counts), cup_counts=sorted(cup_counts),
                partial_terminals=partial, exact_terminals=exact, alternate_branches=branches,
                maximum_source_components=max(source_components))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
