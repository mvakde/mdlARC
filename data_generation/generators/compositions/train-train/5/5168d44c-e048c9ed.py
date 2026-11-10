"""Length-addressed shuttle on a directed dotted orthogonal route.

Only standard-library modules and NumPy are used. Generation draws the answer
from its construction; solve decodes only the supplied input cells.
"""

from collections import deque
from numbers import Integral

import numpy as np


LENGTH_CODE = {2: 1, 3: 4, 4: 9}
ROLE_COLORS = (0, 2, 3, 5, 6, 7, 8)
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(int, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _ring(grid, center, color, guide):
    r, c = center
    grid[r - 1:r + 2, c - 1:c + 2] = color
    grid[r, c] = guide


def _components(a, background):
    """Monochrome four-connected foreground components."""
    seen = set()
    result = []
    for r, c in zip(*np.where(a != background)):
        p = (int(r), int(c))
        if p in seen:
            continue
        color = int(a[p])
        seen.add(p)
        queue = deque([p])
        cells = []
        while queue:
            q = queue.popleft()
            cells.append(q)
            for dr, dc in DIRECTIONS:
                v = (q[0] + dr, q[1] + dc)
                if (0 <= v[0] < a.shape[0] and 0 <= v[1] < a.shape[1]
                        and v not in seen and int(a[v]) == color):
                    seen.add(v)
                    queue.append(v)
        result.append((color, tuple(cells)))
    return result


def _decode(input_grid):
    """Recover the unique route, shuttle, and station associations from cells."""
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    background = int(values[np.argmax(counts)])
    _require(int(counts.max()) * 2 > a.size, "background must be a strict majority")
    _require(background in ROLE_COLORS, "background is a reserved address color")
    comps = _components(a, background)
    rings = []
    for color, cells in comps:
        if color not in LENGTH_CODE.values() or len(cells) != 8:
            continue
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        center = (r0 + 1, c0 + 1)
        wanted = {(r, c) for r in range(r0, r0 + 3)
                  for c in range(c0, c0 + 3) if (r, c) != center}
        if set(cells) == wanted:
            rings.append((color, cells, center))
    _require(len(rings) == 1, "exactly one 3x3 shuttle is needed")
    query, ring_cells, center = rings[0]
    guide = int(a[center])
    _require(guide in ROLE_COLORS and guide != background, "invalid guide role")
    singleton_comps = [(color, cells[0]) for color, cells in comps if len(cells) == 1]
    nodes = {p for _, p in singleton_comps}
    _require(center in nodes, "shuttle center is not a route dot")
    starts = [(color, p) for color, p in singleton_comps if color != guide]
    _require(len(starts) == 1, "one differently colored route endpoint is needed")
    start_color, start = starts[0]
    _require(start_color in ROLE_COLORS and start_color not in (background, guide),
             "invalid start role")
    neighbors = {p: [q for dr, dc in DIRECTIONS
                     if (q := (p[0] + 2 * dr, p[1] + 2 * dc)) in nodes]
                 for p in nodes}
    _require(len(neighbors[start]) == 1, "start is not an endpoint")
    _require(all(1 <= len(v) <= 2 for v in neighbors.values()), "branched route")
    path, visited = [], set()
    previous, current = None, start
    while current is not None:
        _require(current not in visited, "cyclic route")
        path.append(current)
        visited.add(current)
        options = [q for q in neighbors[current] if q != previous]
        _require(len(options) <= 1, "ambiguous route order")
        previous, current = current, options[0] if options else None
    _require(visited == nodes, "disconnected route")
    _require(all((r - start[0]) % 2 == 0 and (c - start[1]) % 2 == 0
                 for r, c in nodes), "inconsistent dot lattice")
    for p, q in zip(path, path[1:]):
        gap = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
        _require(int(a[gap]) == (query if gap in ring_cells else background),
                 "unexpected cell between dots")
    index = {p: i for i, p in enumerate(path)}
    stations = {}
    bars = []
    all_bar_cells = set()
    for color, cells in comps:
        if len(cells) == 1 or cells == ring_cells:
            continue
        length = len(cells)
        _require(length in LENGTH_CODE, "station bar length must be 2..4")
        rows, cols = {r for r, _ in cells}, {c for _, c in cells}
        horizontal = len(rows) == 1
        vertical = len(cols) == 1
        _require(horizontal != vertical, "station bar must be straight")
        ordered = sorted(cells)
        _require(all(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1
                     for p, q in zip(ordered, ordered[1:])), "broken bar")
        aligned = [(min(abs(r - x) + abs(c - y) for r, c in cells), index[(x, y)])
                   for x, y in nodes
                   if (x == cells[0][0] if horizontal else y == cells[0][1])]
        aligned.sort()
        _require(aligned and (len(aligned) == 1 or aligned[0][0] < aligned[1][0]),
                 "bar projection has no unique nearest dot")
        distance, station = aligned[0]
        _require(distance >= 3 and station not in stations, "invalid station association")
        _require(0 < station < len(path) - 1, "station must be internal")
        before, at, after = path[station - 1:station + 2]
        _require((at[0] - before[0], at[1] - before[1]) ==
                 (after[0] - at[0], after[1] - at[1]), "station is at a turn")
        _require((before[1] == at[1]) == horizontal, "bar must be normal to the route")
        _require(all(max(abs(r - x), abs(c - y)) >= 3
                     for r, c in cells for x, y in nodes), "bar too close to route")
        _require(all(max(abs(r - x), abs(c - y)) >= 2
                     for r, c in cells for x, y in all_bar_cells), "station bars touch")
        all_bar_cells.update(cells)
        stations[station] = length
        bars.append({"index": station, "length": length, "color": color,
                     "cells": cells, "horizontal": horizontal, "gap": distance})
    _require(len(stations) >= 1, "no stations")
    current_index = index[center]
    eligible = sorted(i for i, length in stations.items()
                      if i > current_index and LENGTH_CODE[length] == query)
    _require(eligible, "no matching station ahead")
    target_index = eligible[0]
    target = path[target_index]
    _require(all(1 <= p[0] < a.shape[0] - 1 and 1 <= p[1] < a.shape[1] - 1
                 for p in (center, target)), "shuttle would be clipped")
    return {"grid": a, "background": background, "guide": guide,
            "start_color": start_color, "query": query, "center": center,
            "path": path, "stations": stations, "bars": bars,
            "current_index": current_index, "target_index": target_index,
            "ring_cells": ring_cells}


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    output = d["grid"].copy()
    for p in d["ring_cells"]:
        output[p] = d["background"]
    _ring(output, d["path"][d["target_index"]], d["query"], d["guide"])
    return output


def _random_route(rng, long):
    nr, nc = map(int, rng.integers(8 if not long else 10, 14, size=2))
    desired = int(rng.integers(22 if not long else 30, 40 if not long else 50))
    path = [(int(rng.integers(nr)), int(rng.integers(nc)))]
    used = set(path)
    for _ in range(desired - 1):
        options, weights = [], []
        for dr, dc in DIRECTIONS:
            q = path[-1][0] + dr, path[-1][1] + dc
            if not (0 <= q[0] < nr and 0 <= q[1] < nc) or q in used:
                continue
            # An induced path has no accidental shortcut edges between dots.
            if sum((q[0] + x, q[1] + y) in used for x, y in DIRECTIONS) != 1:
                continue
            straight = len(path) > 1 and (dr, dc) == (
                path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
            options.append(q)
            weights.append(5 if straight else 1)
        if not options:
            break
        weights = np.array(weights, dtype=float)
        q = options[int(rng.choice(len(options), p=weights / weights.sum()))]
        path.append(q)
        used.add(q)
    return [(2 + 2 * r, 2 + 2 * c) for r, c in path], (2 * nr + 3, 2 * nc + 3)


def _station_options(path, shape):
    options = {}
    for i in range(1, len(path) - 1):
        a, b, c = path[i - 1:i + 2]
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            continue
        normal = (0, 1) if a[1] == b[1] else (1, 0)
        for sign in (-1, 1):
            for gap in (3, 4, 5):
                ray = tuple((b[0] + sign * normal[0] * t,
                             b[1] + sign * normal[1] * t)
                            for t in range(gap, gap + 4))
                if not all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in ray):
                    continue
                if any(max(abs(r - x), abs(c - y)) < 3
                       for r, c in ray for x, y in path):
                    continue
                good = True
                for length in LENGTH_CODE:
                    candidates = sorted(
                        (min(abs(r - x) + abs(c - y) for r, c in ray[:length]), j)
                        for j, (x, y) in enumerate(path)
                        if (x == b[0] if normal[1] else y == b[1]))
                    if (candidates[0][1] != i or
                            len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
                        good = False
                        break
                if good:
                    options.setdefault(i, []).append(ray)
    return options


def _turns(path, first, last):
    return sum((path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) !=
               (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
               for i in range(max(1, first), min(last, len(path) - 2) + 1))


def _geometry(rng, kind, count):
    """Return route, selected rays, and ordered decision witnesses."""
    for _ in range(800):
        path, shape = _random_route(rng, long=kind != "near")
        options = _station_options(path, shape)
        if len(options) < count:
            continue
        chosen, occupied = {}, []
        for i in rng.permutation(list(options)):
            rays = options[int(i)]
            for k in rng.permutation(len(rays)):
                ray = rays[int(k)]
                if all(max(abs(r - x), abs(c - y)) >= 2
                       for r, c in ray for x, y in occupied):
                    chosen[int(i)] = ray
                    occupied.extend(ray)
                    break
            if len(chosen) == count:
                break
        if len(chosen) != count:
            continue
        s = sorted(chosen)
        if kind == "near":
            choices = [t for t in range(1, count) if s[t - 1] < s[t] - 1]
            if not choices:
                continue
            t = int(rng.choice(choices))
            target, current, behind = s[t], s[t] - 1, s[t - 1]
            later = None
        else:
            b = 0 if kind == "skip" else int(rng.integers(0, count - 4))
            t = 2 if kind == "skip" else int(rng.integers(b + 3, count - 1))
            behind, current, target, later = s[b], s[b] + 1, s[t], s[-1]
            if s[b + 1] <= current or not _turns(path, current, target):
                continue
            p, q = path[current], path[target]
            if abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 2:
                continue
        return path, chosen, current, target, behind, later
    raise RuntimeError("bounded route sampler could not place station witnesses")


def _make_pair(rng, kind, query_length, palette):
    minimum, maximum = ((4, 7) if kind == "near" else
                        (5, 8) if kind == "skip" else (6, 9))
    count = int(rng.integers(minimum, maximum + 1))
    path, rays, current, target, behind, later = _geometry(rng, kind, count)
    background, guide, start_color = palette
    query = LENGTH_CODE[query_length]
    other_lengths = [x for x in LENGTH_CODE if x != query_length]
    lengths = {i: int(rng.choice(other_lengths)) for i in rays}
    lengths[target] = query_length
    if later is not None:
        lengths[behind] = query_length
        lengths[later] = query_length
        for i in rays:
            if i < behind and rng.random() < .65:
                lengths[i] = query_length
    # All three bar lengths are visible, including two kinds of rejected sign.
    wrong = sorted(i for i in rays if lengths[i] != query_length)
    for i, length in zip(wrong, other_lengths):
        lengths[i] = length
    available = [c for c in range(10) if c != background]
    colors = {i: int(rng.choice(available)) for i in rays}
    if later is not None:
        colors[behind] = query if colors[target] != query else guide
    for i in wrong:
        colors[i] = query if rng.random() < .6 else int(rng.choice(available))
    # A matching hue on a wrong-length sign must not cause a stop.
    first_wrong = next((i for i in wrong if current < i < target), wrong[0])
    colors[wrong[-1]] = guide
    colors[first_wrong] = query
    cells = {i: rays[i][:lengths[i]] for i in rays}
    active = list(path) + [p for bar in cells.values() for p in bar]
    for i in (current, target):
        r, c = path[i]
        active.extend(((r - 1, c - 1), (r + 1, c + 1)))
    lo_r, hi_r = min(r for r, _ in active), max(r for r, _ in active)
    lo_c, hi_c = min(c for _, c in active), max(c for _, c in active)
    span_r, span_c = hi_r - lo_r + 1, hi_c - lo_c + 1
    pads = []
    for span in (span_r, span_c):
        before = int(rng.integers(0, min(3, 30 - span) + 1))
        after = int(rng.integers(0, min(3, 30 - span - before) + 1))
        pads.append((before, after))
    def shift(p):
        return p[0] - lo_r + pads[0][0], p[1] - lo_c + pads[1][0]
    latent = np.full((span_r + sum(pads[0]), span_c + sum(pads[1])),
                     background, dtype=np.int8)
    for p in path:
        latent[shift(p)] = guide
    latent[shift(path[0])] = start_color
    for i, bar in cells.items():
        for p in bar:
            latent[shift(p)] = colors[i]
    inp, out = latent.copy(), latent.copy()
    _ring(inp, shift(path[current]), query, guide)
    _ring(out, shift(path[target]), query, guide)
    return {"input": inp, "output": out}


def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base = tuple(int(c) for c in rng.choice(ROLE_COLORS, 3, replace=False))
    varying_roles = bool(rng.integers(2))
    query_order = [int(x) for x in rng.permutation([2, 3, 4])]
    taught = query_order[:min(3, num_train)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for j in range(count):
            kind = ("near" if j == 0 else "skip") if split == "train" else "multi"
            query_length = (taught[j] if split == "train" and j < len(taught)
                            else int(rng.choice(taught)))
            swap = varying_roles and (j == 1 if split == "train" and j < 2
                                      else bool(rng.integers(2)))
            palette = (base[0], base[2], base[1]) if swap else base
            for _ in range(30):
                pair = _make_pair(rng, kind, query_length, palette)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, metas = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input/output keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "canvas changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            d = _decode(a)
            _require(np.array_equal(solve(a), b), "wrong output")
            _require(not np.array_equal(a, b), "identity pair")
            # Check shape preservation and exact restoration, not just target color.
            old = set(d["ring_cells"])
            p = d["path"][d["target_index"]]
            new = {(r, c) for r in range(p[0] - 1, p[0] + 2)
                   for c in range(p[1] - 1, p[1] + 2) if (r, c) != p}
            changed = set(zip(*np.where(a != b)))
            _require(changed == old.symmetric_difference(new), "unexpected edits")
            _require(all(int(b[q]) == d["background"] for q in old - new),
                     "vacated cells not restored")
            _require(all(int(b[q]) == d["query"] for q in new), "shuttle damaged")
            _require(all(int(a[q]) == int(b[q]) for q in d["path"]), "route changed")
            _require(all(int(a[q]) == int(b[q]) for bar in d["bars"] for q in bar["cells"]),
                     "station signs changed")
            _require({bar["length"] for bar in d["bars"]} == {2, 3, 4},
                     "all three sign lengths must be visible")
            i, t = d["current_index"], d["target_index"]
            matches_behind = [j for j, length in d["stations"].items()
                              if j < i and LENGTH_CODE[length] == d["query"]]
            same_length = [bar for bar in d["bars"]
                           if LENGTH_CODE[bar["length"]] == d["query"]]
            if t != i + 1:
                _require(matches_behind, "missing behind-shuttle matching distractor")
                _require(len({bar["color"] for bar in same_length}) >= 2,
                         "length/color independence not demonstrated")
                color_lengths = {}
                for bar in d["bars"]:
                    color_lengths.setdefault(bar["color"], set()).add(bar["length"])
                _require(any(len(ls) > 1 for ls in color_lengths.values()),
                         "same-color different-length signs are needed")
            d["rejected"] = sum(i < j < t for j in d["stations"])
            d["turns_moved"] = _turns(d["path"], i, t)
            metas[split].append(d)
    train, test = metas["train"], metas["test"]
    _require(train[0]["target_index"] - train[0]["current_index"] == 1,
             "first training pair must show adjacent-dot movement")
    for d in train[1:] + test:
        i, t = d["current_index"], d["target_index"]
        _require(d["rejected"] >= (2 if any(d is x for x in test) else 1),
                 "selective movement lacks rejected stations")
        _require(d["turns_moved"] >= 1, "selective move must follow a bend")
        _require(any(j > t and LENGTH_CODE[length] == d["query"]
                     for j, length in d["stations"].items()), "missing later matching distractor")
        behind = [d["path"][j] for j, length in d["stations"].items()
                  if j < i and LENGTH_CODE[length] == d["query"]]
        target = d["path"][t]
        at = d["path"][i]
        _require(min(sum(abs(a - b) for a, b in zip(at, q)) for q in behind) <
                 sum(abs(a - b) for a, b in zip(at, target)),
                 "nearest-match shortcut not distinguished")
        first = min((bar for bar in d["bars"] if bar["index"] > i),
                    key=lambda bar: bar["index"])
        _require(first["color"] == d["query"] and
                 LENGTH_CODE[first["length"]] != d["query"],
                 "color-matching shortcut not distinguished")
    _require({d["query"] for d in test} <= {d["query"] for d in train},
             "test query code not taught")
    _require(len({d["query"] for d in train}) >= 2, "need two query demonstrations")
    # Different destination ranks disprove a global nth-matching-station rule.
    ranks = {sum(j <= d["target_index"] and LENGTH_CODE[length] == d["query"]
                 for j, length in d["stations"].items()) for d in train}
    _require(len(ranks) >= 2, "current position must affect the matching-station rank")
    _require(len({d["background"] for d in train + test}) == 1,
             "background must be fixed within puzzle")
    roles = {(d["guide"], d["start_color"]) for d in train + test}
    if len(roles) > 1:
        _require(len({(d["guide"], d["start_color"]) for d in train}) > 1,
                 "role swapping must be demonstrated")


def self_test():
    """Bounded 42-seed checks, count boundaries, and family-specific diversity."""
    puzzles, shapes, structures, palettes = set(), set(), set(), set()
    directions, queries, gaps, lengths, station_counts = set(), set(), set(), set(), set()
    backgrounds, moved_turns, role_modes = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        signature = tuple((_key(pair["input"]), _key(pair["output"]))
                          for s in ("train", "test") for pair in puzzle[s])
        _require(signature == tuple((_key(p["input"]), _key(p["output"]))
                                    for s in ("train", "test") for p in again[s]),
                 "seed is not deterministic")
        puzzles.add(signature)
        role_modes.add(len({_decode(p["input"])["guide"]
                            for s in ("train", "test") for p in puzzle[s]}) > 1)
        witness = puzzle["train"][1]
        # The same rule survives whole-grid rotation and arbitrary sign ink.
        quarter_turns = seed % 3 + 1
        _require(np.array_equal(solve(np.rot90(witness["input"], quarter_turns)),
                                np.rot90(witness["output"], quarter_turns)),
                 "route rule depends on absolute orientation")
        recolored_in = witness["input"].copy()
        recolored_out = witness["output"].copy()
        d = _decode(recolored_in)
        for bar in d["bars"]:
            for p in bar["cells"]:
                recolored_in[p] = recolored_out[p] = d["guide"]
        _require(np.array_equal(solve(recolored_in), recolored_out),
                 "sign ink affects the address")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                path = d["path"]
                shapes.add(pair["input"].shape)
                structures.add((tuple(path), tuple(sorted(d["stations"].items())),
                                d["current_index"], d["target_index"]))
                palettes.add((d["background"], d["guide"], d["start_color"]))
                backgrounds.add(d["background"])
                directions.update((q[0] - p[0], q[1] - p[1]) for p, q in zip(path, path[1:]))
                queries.add(d["query"])
                lengths.add(len(path))
                station_counts.add(len(d["stations"]))
                gaps.update(bar["gap"] for bar in d["bars"])
                moved_turns.add(_turns(path, d["current_index"], d["target_index"]))
                # NumPy/list parity, input immutability, and palette-role geometry.
                before = pair["input"].copy()
                _require(np.array_equal(solve(before.tolist()), pair["output"]),
                         "nested-list solve failed")
                _require(np.array_equal(before, pair["input"]), "solver mutated input")
    _require(len(puzzles) == 42 and len(shapes) >= 35 and len(structures) >= 180,
             "insufficient structural variation across 42 seeds")
    _require(len(palettes) >= 25 and len(backgrounds) >= 6, "few role palettes")
    _require(directions == {(2, 0), (-2, 0), (0, 2), (0, -2)}, "missing route directions")
    _require(queries == {1, 4, 9} and gaps == {3, 4, 5}, "few codes or bar offsets")
    _require(len(lengths) >= 15 and station_counts == set(range(4, 10)), "few counts")
    _require({0, 1, 2} <= moved_turns and role_modes == {False, True}, "few movement modes")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(42000 + a * 10 + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus all legal count combinations)")
