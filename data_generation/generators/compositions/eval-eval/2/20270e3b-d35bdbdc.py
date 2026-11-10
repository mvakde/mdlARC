"""Register two circuit fragments; compose only the assembled exposed terminals.

No parent implementation, file, seed, or training label is used by solve().
The extra connecting concept is graph topology: a wire union can bury a terminal.
"""
from __future__ import annotations

import heapq
import json
from numbers import Integral

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= g) & (g <= 9))), "colors must be 0..9")
    return g.astype(np.int8, copy=False)


def _components(cells, directions=D4):
    remaining = set(cells)
    parts = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _bbox(cells):
    return min(r for r, c in cells), max(r for r, c in cells), min(c for r, c in cells), max(c for r, c in cells)


def _shift(cells, offset):
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _support(center, kind):
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if kind == "square" or dr == 0 or dc == 0}


def _crop_pixels(pixels, background):
    _require(bool(pixels), "empty assembly")
    t, b, l, r = _bbox(pixels)
    _require(b - t < 30 and r - l < 30, "assembly exceeds 30")
    g = np.full((b - t + 1, r - l + 1), background, dtype=np.int8)
    for (y, x), color in pixels.items():
        g[y - t, x - l] = color
    return g


def _parse_with_roles(g, background, wire, marker, parts):
    """Recover ownership, registration, and centered tokens from input pixels."""
    marker_cells = {tuple(map(int, p)) for p in np.argwhere(g == marker)}
    bars = _components(marker_cells)
    _require(len(bars) == 2 and len(bars[0]) == len(bars[1]), "need two congruent bars")
    length = len(bars[0])
    _require(2 <= length <= 5, "bar length must be 2..5")
    axes = [axis for axis in (0, 1)
            if all(len({p[axis] for p in bar}) == 1 and
                   max(p[1 - axis] for p in bar) - min(p[1 - axis] for p in bar) + 1 == length
                   for bar in bars)]
    _require(len(axes) == 1, "bars must have a visible common axis")
    axis = axes[0]  # 0 = horizontal bars; canonical coordinates are (normal,tangent).
    orient = lambda p: (p[axis], p[1 - axis])
    records, tokens = [], []
    for part in parts:
        owned = [bar for bar in bars if bar <= part]
        _require(len(owned) == 1, "each circuit must own one bar")
        bar = {orient(p) for p in owned[0]}
        r0, _, c0, _ = _bbox(bar)
        origin = r0, c0
        lines = {orient(p) for p in part if int(g[p]) == wire}
        _require(len(lines) >= 4 and len(_components(lines)) == 1, "each wire fragment must be connected")
        negative = any((r0 - 1, c0 + c) in lines for c in range(length))
        positive = any((r0 + 1, c0 + c) in lines for c in range(length))
        _require(negative != positive, "backing must identify exactly one mating side")
        local_tokens = []
        identity_cells = {p for p in part if int(g[p]) not in (wire, marker)}
        for support in _components(identity_cells):
            candidates = []
            for center in sorted(support):
                for kind in ("square", "cross"):
                    if _support(center, kind) != support:
                        continue
                    body = {int(g[p]) for p in support if p != center}
                    if len(body) == 1:
                        candidates.append((center, kind, body.pop(), int(g[center])))
            _require(len(candidates) == 1, "token must have a unique complete centered support")
            center, kind, body, value = candidates[0]
            _require(body not in (background, wire, marker) and value not in (background, wire, marker),
                     "identities must be distinct from structural roles")
            oriented_center = orient(center)
            terminal_sites = {(oriented_center[0] + 2 * dr, oriented_center[1] + 2 * dc) for dr, dc in D4}
            _require(bool(terminal_sites & lines), "each token must have a center-aligned wire contact")
            local_tokens.append(dict(center=oriented_center, cells={orient(p) for p in support},
                                     kind=kind, body=body, value=value))
        _require(bool(local_tokens), "each fragment must contain tokens")
        records.append(dict(negative=negative, origin=origin, wire=lines, tokens=local_tokens,
                            bar=bar, original_cells={orient(p) for p in part}))
    _require(records[0]["negative"] != records[1]["negative"], "backing sides must oppose")
    records.sort(key=lambda rec: not rec["negative"])
    # Set negative-backed guide at (0,0), positive-backed guide at (-1,0).
    assembled_wires = set()
    original_wires = []
    for layer, rec in enumerate(records):
        dr = -rec["origin"][0] - layer
        dc = -rec["origin"][1]
        lines = _shift(rec["wire"], (dr, dc))
        original_wires.append(lines)
        assembled_wires |= lines
        for token in rec["tokens"]:
            token = dict(token)
            token["cells"] = _shift(token["cells"], (dr, dc))
            r, c = token["center"]
            token["center"] = r + dr, c + dc
            token["layer"] = layer
            tokens.append(token)
    occupied = set()
    mapping = {}
    for token in tokens:
        _require(token["cells"].isdisjoint(assembled_wires | occupied), "tokens may not collide with wires or other tokens")
        _require(not any((r + dr, c + dc) in occupied for r, c in token["cells"] for dr, dc in D4),
                 "tokens must remain separately segmentable after assembly")
        occupied |= token["cells"]
        body, value = token["body"], token["value"]
        _require(body not in mapping or mapping[body] == value, "conflicting repeated-body donors")
        mapping[body] = value
    _require(4 <= len(tokens) <= 8, "need 4..8 tokens")
    return dict(background=background, wire=wire, marker=marker, axis=axis, length=length,
                layers=records, wires=assembled_wires, original_wires=original_wires, tokens=tokens)


def _interpret(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    _require(np.sum(counts == counts.max()) == 1, "background must be uniquely most frequent")
    bg = int(colors[np.argmax(counts)])
    parts = _components({tuple(map(int, p)) for p in np.argwhere(g != bg)}, D8)
    _require(len(parts) == 2, "input must contain two separate eight-connected circuits")
    candidates = []
    for marker in map(int, colors):
        if marker == bg:
            continue
        bars = _components({tuple(map(int, p)) for p in np.argwhere(g == marker)})
        if len(bars) != 2 or len(bars[0]) != len(bars[1]) or not 2 <= len(bars[0]) <= 5:
            continue
        for wire in map(int, colors):
            if wire in (bg, marker):
                continue
            try:
                candidates.append(_parse_with_roles(g, bg, wire, marker, parts))
            except ValueError:
                pass
    _require(len(candidates) == 1, "input must uniquely identify structural roles and assembly")
    return candidates[0]


def _ends(wires):
    return {p for p in wires if sum((p[0] + dr, p[1] + dc) in wires for dr, dc in D4) == 1}


def _eligible(token, ends):
    r, c = token["center"]
    return any((r + 2 * dr, c + 2 * dc) in ends for dr, dc in D4)


def _render(parsed, selection="assembled", update="simultaneous", blend="union", guides="transparent"):
    """Ordinary prediction plus finite diagnostic alternatives for teaching checks."""
    tokens, wires = parsed["tokens"], set(parsed["wires"])
    if blend == "xor":
        wires = parsed["original_wires"][0] ^ parsed["original_wires"][1]
    assembled_ends = _ends(parsed["wires"])
    original_ends = [_ends(w) for w in parsed["original_wires"]]
    eligible, donors = [], []
    for i, t in enumerate(tokens):
        selected = (_eligible(t, assembled_ends) if selection == "assembled" else
                    _eligible(t, original_ends[t["layer"]]) if selection == "before" else True)
        eligible.append(selected)
        donors.append([j for j, d in enumerate(tokens) if j != i and d["body"] == t["value"]])
    retained = [eligible[i] and bool(donors[i]) for i in range(len(tokens))]
    pixels = {p: parsed["wire"] for p in wires}
    if guides == "paint":
        pixels.update({(r, c): parsed["wire"] for r in (0, -1) for c in range(parsed["length"])})
    for i, t in enumerate(tokens):
        if not retained[i]:
            continue
        value = tokens[donors[i][0]]["value"]
        if update == "unchanged":
            value = t["value"]
        elif update == "recursive":
            # One extra substitution already distinguishes input reads from propagation.
            extra = [j for j, d in enumerate(tokens) if j != i and d["body"] == value]
            if extra:
                value = tokens[extra[0]]["value"]
        elif update == "surviving_donors":
            live = [j for j in donors[i] if retained[j]]
            if not live:
                continue
            value = tokens[live[0]]["value"]
        pixels.update({p: t["body"] for p in t["cells"]})
        pixels[t["center"]] = value
    if guides == "erase":
        for r in (0, -1):
            for c in range(parsed["length"]):
                pixels.pop((r, c), None)
    g = _crop_pixels(pixels, parsed["background"])
    return g.T.copy() if parsed["axis"] else g


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate by visible bars, union wires, then read original token relations."""
    return _render(_interpret(input_grid))


def _route(rng, shape, start, tree, blocked):
    """Random-cost four-neighbor route from a terminal to a connected scaffold."""
    h, w = shape
    if start in blocked or not (0 <= start[0] < h and 0 <= start[1] < w):
        return None
    costs = rng.uniform(1.0, 3.0, size=shape)
    distance, previous = {start: 0.0}, {}
    queue = [(0.0, start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance[p]:
            continue
        if p in tree:
            path = {p}
            while p != start:
                p = previous[p]
                path.add(p)
            return path
        for dr, dc in D4:
            q = p[0] + dr, p[1] + dc
            if not (0 <= q[0] < h and 0 <= q[1] < w) or q in blocked:
                continue
            new = cost + float(costs[q])
            if new < distance.get(q, float("inf")):
                distance[q], previous[q] = new, p
                heapq.heappush(queue, (new, q))
    return None


def _relations(rng, n, mode, solid_allowed):
    if mode == "chain":
        rel = [(0, 1), (1, 2), (2, 3), (3, 1)]
        mapping = {0: 1, 1: 2, 2: 3, 3: 1}
    elif mode == "mono":
        rel = [(0, 1), (2, 3), (1, 1), (3, 4), (4, 1)]
        mapping = {0: 1, 1: 1, 2: 3, 3: 4, 4: 1}
    elif mode == "cycle":
        rel = [(0, 1), (1, 0), (2, 1), (3, 4), (4, 4)]
        mapping = {0: 1, 1: 0, 2: 1, 3: 4, 4: 4}
    else:
        for _ in range(100):
            bodies = list(map(int, rng.integers(0, 7, size=n)))
            mapping = {b: int(rng.integers(7)) for b in sorted(set(bodies))}
            rel = [(b, mapping[b]) for b in bodies]
            if any(v in mapping and any(j != i and a == v for j, (a, b) in enumerate(rel))
                   for i, (a, v) in enumerate(rel[:2])) and (solid_allowed or not any(
                       a != v and mapping.get(v) == a for a, v in rel)):
                return rel
        return _relations(rng, n, "chain", solid_allowed)
    while len(rel) < n:
        b = int(rng.choice(list(mapping)))
        rel.append((b, mapping[b]))
    return rel


def _circuit_geometry(rng, n, active, kinds, length, teach_uncovered=False):
    """Routes are new geometry, not a parent generator's output."""
    rows = (n + 1) // 2
    width = int(rng.integers(12, 15))
    starts = [5 + int(rng.integers(2)), 5 + int(rng.integers(2))]
    row_positions = []
    for col in range(2):
        positions = [starts[col]]
        for _ in range(rows - 1):
            positions.append(positions[-1] + int(rng.integers(5, 7)))
        row_positions.append(positions)
    slots = [(row_positions[k % 2][k // 2],
              int(rng.integers(3, 5)) if k % 2 == 0 else int(rng.integers(9, width - 1)))
             for k in range(2 * rows)]
    rng.shuffle(slots)
    centers = slots[:n]
    shape = max(r for r, c in centers) + int(rng.integers(3, 5)), width
    if shape[0] > 30:
        return None
    supports = [_support(p, kind) for p, kind in zip(centers, kinds)]
    occupied = set().union(*supports)
    layers = list(map(int, rng.integers(0, 2, size=n)))
    layers[0], layers[1] = 0, 1  # The two teaching endpoints start on different fragments.
    tips, inward, side = [], [], []
    for i, (r, c) in enumerate(centers):
        options = []
        for dr, dc in D4:
            tip, q = (r + 2 * dr, c + 2 * dc), (r + 3 * dr, c + 3 * dc)
            sides = [(tip[0] + dc, tip[1] - dr), (tip[0] - dc, tip[1] + dr)]
            if all(0 <= p[0] < shape[0] and 0 <= p[1] < shape[1] and p not in occupied for p in (tip, q)):
                viable = [p for p in sides if 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1] and p not in occupied]
                if active[i] or viable:
                    options.append((tip, q, viable))
        if not options:
            return None
        tip, q, sides = options[int(rng.integers(len(options)))]
        tips.append(tip)
        inward.append(q)
        side.append(None if active[i] else sides[int(rng.integers(len(sides)))])
    if len(set(tips)) != n:
        return None
    guards = set(occupied)
    for (r, c), tip, q in zip(centers, tips, inward):
        guards.update((r + 2 * dr, c + 2 * dc) for dr, dc in D4)
        guards.update((tip[0] + dr, tip[1] + dc) for dr, dc in D4 if (tip[0] + dr, tip[1] + dc) != q)
    # All routes avoid token pixels and all unintended centered contact sites.
    bars = [{(2 - layer, c) for c in range(2, length + 2)} for layer in range(2)]
    wires = []
    blocked_layers = []
    for layer in range(2):
        back_row = 1 + layer
        backing_end = 2 if teach_uncovered else int(rng.integers(2, length + 2))
        root = {(back_row, c) for c in range(backing_end + 1)} | {(r, 0) for r in range(back_row, 5)}
        facing = {(3 if layer == 0 else 0, c) for c in range(2, length + 2)}
        unused_backing = {(back_row, c) for c in range(backing_end + 1, length + 2)}
        blocked = guards | bars[layer] | facing | unused_backing
        if root & blocked:
            return None
        wires.append(root)
        blocked_layers.append(blocked)
    for i in rng.permutation(n):
        i = int(i)
        layer = layers[i]
        path = _route(rng, shape, inward[i], wires[layer], blocked_layers[layer])
        if path is None:
            return None
        wires[layer] |= path | {tips[i]}
    # Join an inactive terminal sideways from the OTHER layer. It was an exposed
    # terminal in its own fragment, but its degree becomes >=2 after assembly.
    for i in rng.permutation(n):
        i = int(i)
        if active[i]:
            continue
        layer = 1 - layers[i]
        p = side[i]
        # Another tip's guard may not be opened by this join.
        other_guards = set(occupied)
        for j, ((r, c), tip, q) in enumerate(zip(centers, tips, inward)):
            if j != i:
                other_guards.update((r + 2 * dr, c + 2 * dc) for dr, dc in D4)
                other_guards.update((tip[0] + dr, tip[1] + dc) for dr, dc in D4
                                    if (tip[0] + dr, tip[1] + dc) != q)
        if p in other_guards or p in bars[layer]:
            return None
        blocked = blocked_layers[layer] - {p}
        path = _route(rng, shape, p, wires[layer], blocked)
        if path is None:
            return None
        wires[layer] |= path
    union = wires[0] | wires[1]
    ends = _ends(union)
    for i, center in enumerate(centers):
        t = dict(center=center)
        if _eligible(t, ends) != active[i] or not _eligible(t, _ends(wires[layers[i]])):
            return None
    return centers, supports, layers, wires, bars


def _stage(rng, fragments, background, rotation, preferred):
    groups = []
    for pixels in fragments:
        t, b, l, r = _bbox(pixels)
        groups.append(({(y - t, x - l): color for (y, x), color in pixels.items()}, b - t + 1, r - l + 1))
    if rng.random() < .5:
        groups.reverse()
    a, ah, aw = groups[0]
    b, bh, bw = groups[1]
    avoid = {(r + dr, c + dc) for r, c in a for dr, dc in D8 + ((0, 0),)}
    candidates = []
    gap = int(rng.integers(1, 4))
    if aw + bw + gap <= 30:
        candidates.append((int(rng.integers(-min(4, bh - 1), min(4, ah - 1) + 1)), aw + gap, "horizontal"))
    if ah + bh + gap <= 30:
        candidates.append((ah + gap, int(rng.integers(-min(4, bw - 1), min(4, aw - 1) + 1)), "vertical"))
    if aw + bw + gap <= 30 and ah + bh + gap <= 30:
        candidates.append((ah + gap, aw + gap, "diagonal"))
    if preferred == "interleaved":
        offsets = [(dr, dc) for dr in range(-bh + 1, ah) for dc in range(-bw + 1, aw)]
        for j in rng.permutation(len(offsets)):
            dr, dc = offsets[int(j)]
            if _shift(b, (dr, dc)).isdisjoint(avoid):
                candidates.insert(0, (dr, dc, "interleaved"))
                break
    if not candidates:
        return None
    chosen = [q for q in candidates if q[2] == preferred]
    dr, dc, _ = chosen[0] if chosen else candidates[int(rng.integers(len(candidates)))]
    combined = dict(a)
    combined.update({(r + dr, c + dc): color for (r, c), color in b.items()})
    t, bottom, l, right = _bbox(combined)
    h, w = bottom - t + 1, right - l + 1
    if h > 30 or w > 30:
        return None
    pads = [int(rng.integers(3)) for _ in range(4)]
    pads[0], pads[2] = min(pads[0], 30 - h), min(pads[2], 30 - w)
    pads[1], pads[3] = min(pads[1], 30 - h - pads[0]), min(pads[3], 30 - w - pads[2])
    if rng.random() < .15:
        pads[3] = 30 - w - pads[2]
    g = np.full((h + pads[0] + pads[1], w + pads[2] + pads[3]), background, dtype=np.int8)
    for (r, c), color in combined.items():
        g[r - t + pads[0], c - l + pads[2]] = color
    return np.rot90(g, rotation).copy()


def _key(grid):
    g = np.asarray(grid, dtype=np.int8)
    return g.shape, g.tobytes()


def _make_pair(rng, palette, mode, rotation, preferred, length, n, solid_allowed, teach_uncovered=False):
    bg, wire, marker, *identities = palette
    for attempt in range(240):
        rel = _relations(rng, n, mode, solid_allowed)
        active = [i < 2 for i in range(n)]
        if mode == "random":
            active = [bool(rng.random() < .55) for _ in range(n)]
            active[0] = True
            active[-1] = False
        kinds = list(map(str, rng.choice(["square", "cross"], size=n)))
        geometry = _circuit_geometry(rng, n, active, kinds, length, teach_uncovered)
        if geometry is None:
            continue
        centers, supports, layers, wires, bars = geometry
        fragments = [{p: wire for p in w} for w in wires]
        for layer in range(2):
            fragments[layer].update({p: marker for p in bars[layer]})
        for i, ((body, value), center, support) in enumerate(zip(rel, centers, supports)):
            fragments[layers[i]].update({p: identities[body] for p in support})
            fragments[layers[i]][center] = identities[value]
        inp = _stage(rng, fragments, bg, rotation, preferred)
        if inp is None:
            continue
        # Output is constructed from the sampled geometry and original relations;
        # the input parser independently rediscovers all these facts in validation.
        pixels = {p: wire for p in wires[0] | wires[1]}
        survivors = 0
        for i, ((body, value), support, center) in enumerate(zip(rel, supports, centers)):
            donors = [v for j, (b, v) in enumerate(rel) if j != i and b == value]
            if active[i] and donors:
                survivors += 1
                pixels.update({p: identities[body] for p in support})
                pixels[center] = identities[donors[0]]
        if not survivors:
            continue
        out = np.rot90(_crop_pixels(pixels, bg), rotation).copy()
        pair = dict(input=inp, output=out)
        try:
            p = _interpret(inp)
        except ValueError:
            continue
        if not np.array_equal(_render(p), out):
            continue
        if mode in ("chain", "mono", "cycle") and np.array_equal(_render(p, selection="before"), out):
            continue
        return pair
    raise RuntimeError("bounded circuit routing exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable_palette = bool(rng.random() < .45)
    palette = list(map(int, rng.permutation(10)))
    orientation = int(rng.integers(4))
    mixed_axis = bool(rng.random() < .45)
    solid_allowed = nt >= 3 and bool(rng.random() < .4)
    # A structural shift: tests interpolate an unseen guide length. Size, route,
    # token form, palette convention, and relational rule stay in the same ranges.
    train_lengths, test_length = ((2, 4), 3) if rng.random() < .6 else ((3, 5), 4)
    modes = ["chain", "mono"] + (["cycle"] if solid_allowed else [])
    modes += ["random"] * (nt - len(modes))
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            mode = modes[i] if split == "train" else "random"
            length = train_lengths[i % 2] if split == "train" else test_length
            rotation = (orientation + (i % 2 if mixed_axis else 0)) % 4
            colors = palette if stable_palette else list(map(int, rng.permutation(10)))
            minimum = 4 if mode in ("chain", "random") else 5
            n = int(rng.integers(minimum, 9))
            preferred = str(rng.choice(["horizontal", "vertical", "diagonal", "interleaved"]))
            for _ in range(12):
                pair = _make_pair(rng, colors, mode, rotation, preferred, length, n, solid_allowed,
                                  teach_uncovered=split == "train" and i == 0)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(p):
    tokens, wires = p["tokens"], p["wires"]
    ends = _ends(wires)
    retained, buried, compatible_deleted, remote, unchanged, solid, nonrecursive = [], 0, 0, 0, 0, 0, 0
    for i, t in enumerate(tokens):
        donors = [j for j, d in enumerate(tokens) if j != i and d["body"] == t["value"]]
        eligible = _eligible(t, ends)
        before = _eligible(t, _ends(p["original_wires"][t["layer"]]))
        buried += int(before and not eligible)
        compatible_deleted += int(bool(donors) and not eligible)
        retained.append(eligible and bool(donors))
        if retained[-1]:
            value = tokens[donors[0]]["value"]
            unchanged += int(value == t["value"])
            solid += int(value == t["body"] and t["value"] != t["body"])
            remote += int(any(tokens[j]["layer"] != t["layer"] for j in donors))
    for i, t in enumerate(tokens):
        if retained[i]:
            for j, d in enumerate(tokens):
                if j != i and d["body"] == t["value"] and retained[j]:
                    other = [v["value"] for k, v in enumerate(tokens) if k != j and v["body"] == d["value"]]
                    nonrecursive += int(bool(other) and other[0] != d["value"])
    guides = {(r, c) for r in (0, -1) for c in range(p["length"])}
    return dict(tokens=len(tokens), survivors=sum(retained), buried=buried,
                compatible_deleted=compatible_deleted, cross_layer_donors=remote,
                unchanged=unchanged, solid=solid, nonrecursive=nonrecursive,
                overlap=len(p["original_wires"][0] & p["original_wires"][1]),
                covered_guides=len(wires & guides), uncovered_guides=len(guides - wires),
                branches=sum(sum((r + dr, c + dc) in wires for dr, dc in D4) >= 3 for r, c in wires),
                cycle_rank=(sum(sum((r + dr, c + dc) in wires for dr, dc in D4) for r, c in wires) // 2
                            - len(wires) + len(_components(wires))),
                repeated_bodies=len(tokens) - len({t["body"] for t in tokens}),
                monochrome=sum(t["body"] == t["value"] for t in tokens),
                kinds=tuple(sorted({t["kind"] for t in tokens})), axis=p["axis"], length=p["length"],
                roles=(p["background"], p["wire"], p["marker"]))


ALTERNATIVES = (("before", "simultaneous", "union", "transparent"),
                ("all", "simultaneous", "union", "transparent"),
                ("assembled", "unchanged", "union", "transparent"),
                ("assembled", "recursive", "union", "transparent"),
                ("assembled", "surviving_donors", "union", "transparent"),
                ("assembled", "simultaneous", "xor", "transparent"),
                ("assembled", "simultaneous", "union", "paint"),
                ("assembled", "simultaneous", "union", "erase"))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys must be train/test")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count("num_train", nt, 2, 10)
    _count("num_test", ne, 1, 4)
    _require(nt > ne, "num_train must exceed num_test")
    seen, parsed = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys must be input/output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            p = _interpret(inp)
            _require(np.array_equal(_render(p), out), "incorrect assembly, selection, lookup, or crop")
            _require(p["marker"] not in set(map(int, np.unique(out))), "markers must disappear")
            facts = _facts(p)
            _require(0 < facts["survivors"] < facts["tokens"], "need both retained and erased tokens")
            _require(facts["buried"] > 0, "assembly must change terminal eligibility")
            # Pixel accounting: every wire survives exactly once; token support
            # geometry survives intact except the original center's color.
            expected_area = len(p["wires"])
            ends = _ends(p["wires"])
            for i, t in enumerate(p["tokens"]):
                if _eligible(t, ends) and any(j != i and d["body"] == t["value"] for j, d in enumerate(p["tokens"])):
                    expected_area += len(t["cells"])
            _require(int(np.sum(out != p["background"])) == expected_area, "foreground conservation failed")
            _require(all(np.any(edge != p["background"]) for edge in (out[0], out[-1], out[:, 0], out[:, -1])),
                     "output crop is not tight")
            parsed[split].append(p)
    train_facts = [_facts(p) for p in parsed["train"]]
    lengths = {p["length"] for p in parsed["train"]}
    _require(lengths in ({2, 4}, {3, 5}), "training must bracket one held-out guide length")
    middle = sum(lengths) // 2
    _require({p["length"] for p in parsed["test"]} == {middle}, "tests must interpolate guide length")
    _require(any(f["nonrecursive"] for f in train_facts), "training must teach simultaneous retained donors")
    _require(any(f["unchanged"] and f["monochrome"] for f in train_facts), "training must teach monochrome lookup")
    _require(any(f["compatible_deleted"] for f in train_facts), "training must teach topological selection")
    _require({p["axis"] for p in parsed["test"]} <= {p["axis"] for p in parsed["train"]}, "untaught orientation")
    if any(_facts(p)["solid"] for p in parsed["test"]):
        _require(any(f["solid"] for f in train_facts), "solid extension requires a training witness")
    # Require witnesses only for alternatives that actually disagree on a test.
    for alternative in ALTERNATIVES:
        if any(not np.array_equal(_render(p, *alternative), _render(p)) for p in parsed["test"]):
            _require(any(not np.array_equal(_render(p, *alternative), _render(p)) for p in parsed["train"]),
                     f"training fails to distinguish {alternative}")


def self_test():
    """Bounded 42-seed semantic/diversity survey, count checks, and corruptions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    facts, input_shapes, output_shapes, structures, roles, fingerprints = [], set(), set(), set(), set(), set()
    for puzzle in puzzles:
        fingerprints.add(tuple(_key(pair[k]) for s in ("train", "test") for pair in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _interpret(pair["input"])
                f = _facts(p)
                facts.append(f)
                roles.add(f["roles"])
                input_shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                normalized = np.full(pair["input"].shape, 0, dtype=np.int8)
                normalized[pair["input"] == p["wire"]] = 1
                normalized[pair["input"] == p["marker"]] = 2
                normalized[(pair["input"] != p["background"]) & (normalized == 0)] = 3
                structures.add(_key(normalized))
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
    assert len(fingerprints) == 42 and len(structures) >= 180
    assert len(input_shapes) >= 30 and len(output_shapes) >= 25 and len(roles) >= 30
    assert {f["tokens"] for f in facts} >= {4, 5, 6, 7, 8}
    assert {f["length"] for f in facts} == {2, 3, 4, 5}
    assert {f["axis"] for f in facts} == {0, 1}
    for feature in ("buried", "compatible_deleted", "cross_layer_donors", "nonrecursive", "unchanged",
                    "monochrome", "solid", "repeated_bodies", "overlap", "branches", "cycle_rank", "uncovered_guides"):
        assert sum(f[feature] > 0 for f in facts) >= 5, feature
    assert len({f["survivors"] for f in facts}) >= 3
    assert any(30 in shape for shape in input_shapes)
    assert all(len({_key(pair["input"]) for s in puzzle for pair in puzzle[s]}) == 5 for puzzle in puzzles)
    again = generate_puzzle(0)
    assert all(np.array_equal(puzzles[0][s][i][k], again[s][i][k])
               for s in puzzles[0] for i in range(len(again[s])) for k in ("input", "output"))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(100 + a * 10 + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Metamorphic checks use expected transformed outputs, without regeneration.
    pair = puzzles[0]["train"][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
    assert np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"]))
    recolor = np.arange(9, -1, -1, dtype=np.int8)
    assert np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]])
    compact = next(pair for puzzle in puzzles for s in puzzle for pair in puzzle[s]
                   if max(pair["input"].shape) <= 28)
    bg = _interpret(compact["input"])["background"]
    assert np.array_equal(solve(np.pad(compact["input"], 1, constant_values=bg)), compact["output"])
    # Output corruption and conflicting donor values must be rejected.
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    q = corrupted["train"][0]["output"]
    q[0, 0] = (int(q[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    conflicting_input = None
    for puzzle in puzzles:
        for pair in puzzle["train"]:
            p = _interpret(pair["input"])
            originals = [t for layer in p["layers"] for t in layer["tokens"]]
            repeated = next((t for i, t in enumerate(originals)
                             if any(d["body"] == t["body"] for d in originals[:i])), None)
            if repeated is not None:
                conflicting_input = pair["input"].copy()
                center = repeated["center"] if p["axis"] == 0 else repeated["center"][::-1]
                new_value = next(c for c in range(10) if c not in
                                 (p["background"], p["wire"], p["marker"], repeated["value"]))
                conflicting_input[center] = new_value
                break
        if conflicting_input is not None:
            break
    assert conflicting_input is not None
    try:
        solve(conflicting_input)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting donors accepted")
    return {"seeds": 42, "pairs": len(facts), "structures": len(structures),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes), "role_triples": len(roles),
            "token_counts": sorted({f["tokens"] for f in facts}),
            "survivor_counts": sorted({f["survivors"] for f in facts}),
            "features": {feature: sum(f[feature] > 0 for f in facts) for feature in
                         ("buried", "cross_layer_donors", "nonrecursive", "unchanged", "solid", "cycle_rank", "uncovered_guides")}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
