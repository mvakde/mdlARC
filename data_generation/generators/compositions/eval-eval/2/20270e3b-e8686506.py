"""Seal-and-fit: registration is selected by an exact cavity inventory.

The solver reads only pixels. Color 9 is a temporary registration guide; all
other roles are geometric. No parent generator or stored puzzle is used.
"""
from collections import Counter
from functools import lru_cache
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be 2D with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((0 <= a) & (a <= 9))), "invalid grid cells")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _shift(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _normalise(cells):
    t, _, l, _ = _bbox(cells)
    return tuple(sorted(_shift(cells, -t, -l)))


def _cavities(walls):
    """Four-connected blank regions with no route to the unbounded exterior."""
    t, b, l, r = _bbox(walls)
    blank = {(y, x) for y in range(t - 1, b + 2)
             for x in range(l - 1, r + 2)} - walls
    return [part for part in _components(blank)
            if not any(y in (t - 1, b + 1) or x in (l - 1, r + 1)
                       for y, x in part)]


class _SearchLimit(ValueError):
    pass


def _tile(holes, pieces):
    """Return at most two distinct colored tilings, using every piece once.

    A piece is (color, normalised oriented cells). Identical pieces are a
    multiplicity, not distinct identities. Different decompositions producing
    the same colored grid are one answer. There is no symmetry assumption.
    """
    if not holes or sum(len(s) for _, s in pieces) != len(holes):
        return [], {"nodes": 0, "placement_counts": ()}
    ordered = sorted(holes)
    indexes = {cell: i for i, cell in enumerate(ordered)}
    stock = Counter((int(color), _normalise(shape)) for color, shape in pieces)
    types = sorted(stock)
    by_cell = [[] for _ in ordered]
    placement_counts = []
    for index, (color, shape) in enumerate(types):
        placements = set()
        # Align one fixed shape pixel to each cavity pixel: all translations.
        ar, ac = shape[0]
        for r, c in ordered:
            translated = [(r + dy - ar, c + dx - ac) for dy, dx in shape]
            if not all(q in indexes for q in translated):
                continue
            ids = [indexes[q] for q in translated]
            mask = sum(1 << i for i in ids)
            if mask in placements:
                continue
            placements.add(mask)
            paint = sum((color + 1) << (4 * i) for i in ids)
            move = index, mask, paint
            for i in ids:
                by_cell[i].append(move)
        placement_counts.append(len(placements))
    nodes = 0

    @lru_cache(maxsize=None)
    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        if nodes > 30000:
            raise _SearchLimit("tiling uniqueness search exhausted")
        if not remaining:
            return (0,) if not any(counts) else ()
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            cell = bit.bit_length() - 1
            options = [move for move in by_cell[cell]
                       if counts[move[0]] and move[1] & remaining == move[1]]
            if not options:
                return ()
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for kind, mask, paint in best:
            rest = list(counts)
            rest[kind] -= 1
            for suffix in search(remaining ^ mask, tuple(rest)):
                answers.add(paint | suffix)
                if len(answers) == 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search((1 << len(holes)) - 1, tuple(stock[t] for t in types))
    answers = [{cell: ((code >> (4 * i)) & 15) - 1
                for i, cell in enumerate(ordered)} for code in codes]
    return answers, {"nodes": nodes, "placement_counts": tuple(placement_counts)}


def _parse(input_grid):
    inp = _grid(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(np.max(counts) > inp.size // 2 and bg != 9,
             "background must occupy more than half the canvas")
    guides = {tuple(map(int, p)) for p in np.argwhere(inp == 9)}
    bars = _components(guides)
    _require(len(bars) >= 4, "expected several scaffold ports and one cap guide")
    axes = []
    for axis in (0, 1):
        if all(len(bar) >= 2 and len({p[axis] for p in bar}) == 1 and
               max(p[1 - axis] for p in bar) - min(p[1 - axis] for p in bar) + 1 == len(bar)
               for bar in bars):
            axes.append(axis)
    _require(len(axes) == 1, "guides must be parallel straight bars")
    axis = axes[0]  # horizontal bars: row is normal, column is tangent.
    assignments = []
    for color in map(int, colors):
        if color in (bg, 9):
            continue
        walls = {tuple(map(int, p)) for p in np.argwhere(inp == color)}
        parts = _components(walls | guides)
        if len(parts) != 2:
            continue
        records = []
        for part in parts:
            own = [bar for bar in bars if bar <= part]
            fg = part & walls
            if not own or not fg:
                break
            signs = []
            for bar in own:
                backing = []
                for sign in (-1, 1):
                    n = (sign, 0) if axis == 0 else (0, sign)
                    backing.append(any((r + n[0], c + n[1]) in fg for r, c in bar))
                if backing[0] == backing[1]:
                    break
                signs.append(-1 if backing[0] else 1)
            if len(signs) != len(own) or len(set(signs)) != 1:
                break
            records.append({"walls": fg, "bars": own, "sign": signs[0]})
        if len(records) != 2:
            continue
        records.sort(key=lambda x: len(x["bars"]))
        cap, rack = records
        if len(cap["bars"]) != 1 or len(rack["bars"]) < 3 or cap["sign"] == rack["sign"]:
            continue
        # Payloads are monochrome 4-connected components, separated from all
        # guide-bearing objects by at least a Chebyshev distance of two.
        other = {tuple(map(int, p)) for p in np.argwhere((inp != bg) & (inp != 9) & (inp != color))}
        pieces = []
        for part in _components(other):
            values = {int(inp[q]) for q in part}
            if len(values) != 1:
                break
            pieces.append((next(iter(values)), _normalise(part)))
        else:
            if len(pieces) >= 2 and len({c for c, _ in pieces}) >= 2:
                assignments.append({"bg": bg, "wall_color": color, "axis": axis,
                                    "cap": cap, "rack": rack, "pieces": pieces})
    _require(len(assignments) == 1, "wall and payload roles must be unique")
    parsed = assignments[0]
    guide_objects = [record["walls"] | set().union(*record["bars"])
                     for record in (parsed["rack"], parsed["cap"])]
    payload_objects = _components({tuple(map(int, p)) for p in np.argwhere(
        (inp != bg) & (inp != 9) & (inp != parsed["wall_color"]))})
    all_objects = guide_objects + payload_objects
    for i, obj in enumerate(all_objects):
        neighbours = {(r + dy, c + dx) for r, c in obj
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        _require(all(neighbours.isdisjoint(other) for other in all_objects[i + 1:]),
                 "loose objects must be visibly separated")
    _require(not _cavities(parsed["rack"]["walls"]) and
             not _cavities(parsed["cap"]["walls"]), "fragments must start open")
    return parsed


def _candidate(parsed, port, tangent_offset=0):
    cap, rack, axis = parsed["cap"], parsed["rack"], parsed["axis"]
    cap_origin, port_origin = min(cap["bars"][0]), min(port)
    normal = (rack["sign"], 0) if axis == 0 else (0, rack["sign"])
    dr = port_origin[0] + normal[0] - cap_origin[0]
    dc = port_origin[1] + normal[1] - cap_origin[1]
    if axis == 0:
        dc += tangent_offset
    else:
        dr += tangent_offset
    moved = _shift(cap["walls"], dr, dc)
    walls = rack["walls"] | moved
    regions = _cavities(walls)
    holes = set().union(*regions) if regions else set()
    answers, stats = _tile(holes, parsed["pieces"])
    return {"walls": walls, "moved_cap": moved, "holes": holes, "regions": regions,
            "answers": answers, "stats": stats, "translation": (dr, dc), "port": port}


def _render(parsed, candidate, filling):
    t, b, l, r = _bbox(candidate["walls"])
    _require(b - t < 30 and r - l < 30, "assembled crop exceeds 30")
    out = np.full((b - t + 1, r - l + 1), parsed["bg"], dtype=np.int8)
    for y, x in candidate["walls"]:
        out[y - t, x - l] = parsed["wall_color"]
    for (y, x), color in filling.items():
        out[y - t, x - l] = color
    return out


def _infer(input_grid):
    parsed = _parse(input_grid)
    length = len(parsed["cap"]["bars"][0])
    candidates = [_candidate(parsed, port) for port in parsed["rack"]["bars"]
                  if len(port) == length]
    feasible = [c for c in candidates if c["answers"]]
    _require(len(candidates) >= 2, "cap needs at least two compatible ports")
    _require(len(feasible) == 1 and len(feasible[0]["answers"]) == 1,
             "registration and colored tiling must be unique")
    chosen = feasible[0]
    _require(len(chosen["regions"]) == 1, "the cap must seal exactly one cavity")
    return _render(parsed, chosen, chosen["answers"][0]), parsed, chosen, candidates


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Register at the unique inventory-compatible port, fill closed space, crop."""
    return _infer(input_grid)[0]


def _partition(rng, cells, count):
    """Procedural connected regions produced by competing growth fronts."""
    ordered = sorted(cells)
    seeds = rng.choice(len(ordered), size=count, replace=False)
    regions = [{ordered[int(i)]} for i in seeds]
    todo = set(cells) - set().union(*regions)
    while todo:
        frontier = [(i, q) for i, part in enumerate(regions) for r, c in sorted(part)
                    for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if q in todo]
        _require(frontier, "partition became disconnected")
        i, q = frontier[int(rng.integers(len(frontier)))]
        regions[i].add(q)
        todo.remove(q)
    return regions


def _inventory(rng, hole, payload_colors, witness):
    for _ in range(70):
        count = int(rng.integers(2, min(5, len(hole) // 3) + 1))
        if witness or rng.random() < .3:
            # Small pieces have many possible placements before the remainder
            # is fitted. Sometimes two equal units explicitly teach multiplicity.
            remaining = set(hole)
            small = []
            nsmall = 2 if rng.random() < .35 else 1
            for _ in range(nsmall):
                choices = [q for q in sorted(remaining)
                           if len(_components(remaining - {q})) == 1]
                q = choices[int(rng.integers(len(choices)))]
                small.append({q})
                remaining.remove(q)
            count = max(count, nsmall + 1)
            regions = small + _partition(rng, remaining, count - nsmall)
        else:
            regions = _partition(rng, hole, count)
        signatures = sorted({_normalise(p) for p in regions})
        if len(signatures) < 2:
            continue
        ncolors = int(rng.integers(2, min(4, len(signatures)) + 1))
        labels = [int(payload_colors[i % ncolors]) for i in range(len(signatures))]
        rng.shuffle(labels)
        color_for = dict(zip(signatures, labels))
        pieces = [(color_for[_normalise(p)], _normalise(p)) for p in regions]
        expected = {q: color_for[_normalise(p)] for p in regions for q in p}
        answers, stats = _tile(hole, pieces)
        if len(answers) != 1 or answers[0] != expected:
            continue
        if witness and max(stats["placement_counts"], default=0) <= 1:
            continue
        return pieces, expected
    return None


def _hole_from_heights(heights, left):
    return {(r, left + c) for c, depth in enumerate(heights) for r in range(int(depth))}


def _design(rng, profile, palette):
    count = profile["bays"]
    max_width = min(8, 30 // count)
    width = int(rng.integers(5, max_width + 1))
    left = int(rng.choice([1, 1, 2])) if width >= 6 else 1
    right = int(rng.choice([1, 1, 2])) if width - left >= 5 else 1
    interior = width - left - right
    limit = profile["depth"]
    heights = rng.integers(1, limit + 1, size=interior)
    # Rectangular and irregular cavities both occur; every example contains a
    # different-shape, equal-area distractor regardless of the target silhouette.
    if rng.random() < .18:
        heights[:] = int(rng.integers(2, limit + 1))
    if limit > 4:
        heights[int(rng.integers(interior))] = int(rng.integers(5, limit + 1))
    hole = _hole_from_heights(heights, left)
    if len(hole) < 7:
        return None
    inventory = _inventory(rng, hole, palette[2:6], profile["witness"])
    if inventory is None:
        return None
    pieces, target_filling = inventory
    decoy = None
    for _ in range(80):
        alternate = heights.copy()
        rng.shuffle(alternate)
        for _ in range(int(rng.integers(1, 5))):
            a, b = map(int, rng.choice(interior, 2, replace=False))
            if alternate[a] > 1 and alternate[b] < 7:
                alternate[a] -= 1
                alternate[b] += 1
        candidate = _hole_from_heights(alternate, left)
        if candidate == hole or _normalise(candidate) == _normalise(hole):
            continue
        answers, _ = _tile(candidate, pieces)
        if not answers:
            decoy = candidate
            break
    if decoy is None:
        return None
    q = int(rng.integers(left))
    # The cap guide spans both a side-wall pixel and a cavity pixel after
    # assembly, explicitly teaching transparency instead of marker recoloring.
    possible_lengths = list(range(max(2, left - q + 1), min(5, width - q - 1) + 1))
    length = int(rng.choice(possible_lengths))
    wrong_length = int(rng.choice([x for x in possible_lengths if x != length]))
    # A congruent cavity at an incompatible port defeats inventory-only
    # registration; an equal-area non-tileable cavity defeats guide-only choice.
    bay_holes = [hole, decoy, hole]
    bar_lengths = [length, length, wrong_length]
    for _ in range(100):
        if len(bay_holes) == count:
            break
        extra = _hole_from_heights(rng.integers(1, limit + 2, size=interior), left)
        extra_length = length if rng.random() < .65 else wrong_length
        if extra_length == length and _tile(extra, pieces)[0]:
            continue
        bay_holes.append(extra)
        bar_lengths.append(extra_length)
    if len(bay_holes) != count:
        return None
    order = list(map(int, rng.permutation(count)))
    tops = list(map(int, rng.integers(0, 4, size=count)))
    remaining_space = 30 - count * width
    gaps = []
    for _ in range(count - 1):
        gap = int(rng.integers(min(2, remaining_space) + 1))
        gaps.append(gap)
        remaining_space -= gap
    origins, x = [], 0
    for i in range(count):
        origins.append(x)
        x += width + (gaps[i] if i < count - 1 else 0)
    floor = max(tops[i] + _bbox(bay_holes[order[i]])[1] + 1 for i in range(count))
    base_thickness = int(rng.integers(1, 3))
    rack = {(r, c) for r in range(floor, floor + base_thickness) for c in range(x)}
    rack_guides = set()
    target_origin = None
    for i, kind in enumerate(order):
        top, bx = tops[i], origins[i]
        blank = _shift(bay_holes[kind], top, bx)
        rack.update({(r, c) for r in range(top, floor + 1)
                     for c in range(bx, bx + width)} - blank)
        rack_guides.update((top - 1, bx + q + j) for j in range(bar_lengths[kind]))
        if kind == 0:
            target_origin = top, bx
    roof_heights = rng.integers(1, profile["roof"] + 1, size=width)
    cap = {(r, c) for c, height in enumerate(roof_heights) for r in range(-int(height), 0)}
    if profile["overlap"]:
        cap.update((r, width - 1) for r in range(int(rng.integers(1, 4))))
    cap_guides = {(0, q + j) for j in range(length)}
    _require(rack.isdisjoint(rack_guides) and cap.isdisjoint(cap_guides), "guide collision")
    top, bx = target_origin
    moved = _shift(cap, top, bx)
    walls = rack | moved
    intended_hole = _shift(hole, top, bx)
    regions = _cavities(walls)
    if len(regions) != 1 or regions[0] != intended_hole:
        return None
    filling = {(r + top, c + bx): color for (r, c), color in target_filling.items()}
    parsed = {"bg": int(palette[0]), "wall_color": int(palette[1])}
    output = _render(parsed, {"walls": walls}, filling)
    objects = [([(rack, int(palette[1])), (rack_guides, 9)]),
               ([(cap, int(palette[1])), (cap_guides, 9)])]
    objects.extend([(set(shape), color)] for color, shape in pieces)
    return objects, output


def _scatter(rng, objects, background):
    groups = []
    for layers in objects:
        cells = set().union(*(cells for cells, _ in layers))
        t, b, l, r = _bbox(cells)
        groups.append(([(sorted(_shift(cells, -t, -l)), color) for cells, color in layers],
                       sorted(_shift(cells, -t, -l)), b - t + 1, r - l + 1))
    area = sum(len(cells) for _, cells, _, _ in groups)
    max_h, max_w = max(g[2] for g in groups), max(g[3] for g in groups)
    for attempt in range(12):
        width = int(rng.integers(max(18, max_w), 31)) if attempt < 8 else 30
        low_h = max(18, max_h + 5, (2 * area) // width + 1)
        if low_h > 30:
            continue
        height = int(rng.integers(low_h, 31)) if attempt < 8 else 30
        canvas = np.full((height, width), background, dtype=np.int8)
        blocked = np.zeros((height, width), dtype=bool)
        # Rack first; random ordering among the cap and the larger payloads.
        order = [0] + sorted(range(1, len(groups)),
                             key=lambda i: (-len(groups[i][1]), float(rng.random())))
        for i in order:
            layers, cells, h, w = groups[i]
            positions = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)
                         if all(not blocked[r + dy, c + dx] for dy, dx in cells)]
            if not positions:
                break
            edge = [(r, c) for r, c in positions if r == 0 or c == 0 or r + h == height or c + w == width]
            choices = edge if edge and rng.random() < .25 else positions
            r, c = choices[int(rng.integers(len(choices)))]
            for shape, color in layers:
                for dy, dx in shape:
                    canvas[r + dy, c + dx] = color
            for dy, dx in cells:
                for ar in range(max(0, r + dy - 1), min(height, r + dy + 2)):
                    for ac in range(max(0, c + dx - 1), min(width, c + dx + 2)):
                        blocked[ar, ac] = True
        else:
            return canvas
    return None


def _make_pair(rng, profile, palette, used):
    for _ in range(180):
        design = _design(rng, profile, palette)
        if design is None:
            continue
        objects, output = design
        inp = _scatter(rng, objects, int(palette[0]))
        if inp is None:
            continue
        k = profile["rotation"]
        inp, output = np.rot90(inp, k).copy(), np.rot90(output, k).copy()
        if _key(inp) in used:
            continue
        try:
            predicted, parsed, chosen, candidates = _infer(inp)
        except (ValueError, _SearchLimit):
            continue
        if not np.array_equal(predicted, output):
            continue
        if profile["witness"] and max(chosen["stats"]["placement_counts"]) <= 1:
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("bounded procedural construction exhausted")


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(9)
    changing_palette = bool(rng.integers(2))
    mixed_axis = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    shift = str(rng.choice(["more_ports", "deeper_cavity"]))
    puzzle, used = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for index in range(number):
            testing = split == "test"
            profile = {"bays": int(rng.integers(4, 6)) if testing and shift == "more_ports" else 3,
                       "depth": 6 if testing and shift == "deeper_cavity" else 4,
                       "roof": int(rng.integers(2, 5)),
                       "overlap": bool(rng.integers(2)),
                       "witness": split == "train" and index == 1,
                       "rotation": (rotation + index % 2 + 2 * int(rng.integers(2))) % 4
                       if mixed_axis else rotation}
            if split == "train" and index == 0:
                profile["overlap"] = False
            if split == "train" and index == 1:
                profile["overlap"] = True
            pair_palette = rng.permutation(9) if changing_palette else palette
            pair = _make_pair(rng, profile, pair_palette, used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    answer, p, c, candidates = _infer(inp)
    _require(np.array_equal(answer, out), "incorrect registration, filling or crop")
    _require(len(c["regions"]) == 1, "exactly one bay must be sealed")
    rack, cap = p["rack"]["walls"], c["moved_cap"]
    _require(int(np.sum(out == p["wall_color"])) == len(rack) + len(cap) - len(rack & cap),
             "wall union was not conserved")
    _require(np.any(out == p["bg"]), "open bays must remain unfilled")
    filling = c["answers"][0]
    cap_guide = _shift(p["cap"]["bars"][0], *c["translation"])
    _require(bool(cap_guide & c["walls"]) and bool(cap_guide & c["holes"]),
             "guide transparency must expose both wall and payload pixels")
    for color in {color for color, _ in p["pieces"]}:
        expected = sum(len(s) for col, s in p["pieces"] if col == color)
        _require(int(np.sum(out == color)) == expected, "inventory was copied, discarded or recolored")
    _require(9 not in out, "temporary guides survived")
    # Strong ablations: area alone cannot choose among compatible ports; if
    # lengths are ignored, another port admits the exact same inventory.
    _require(any(not alt["answers"] and len(alt["holes"]) == len(c["holes"])
                 for alt in candidates), "missing equal-area shape distractor")
    cap_length = len(p["cap"]["bars"][0])
    other_length = [_candidate(p, bar, offset) for bar in p["rack"]["bars"]
                    if len(bar) != cap_length for offset in (0, len(bar) - cap_length)]
    other_length = [alt for alt in other_length
                    if _bbox(alt["walls"])[1] - _bbox(alt["walls"])[0] < 30 and
                    _bbox(alt["walls"])[3] - _bbox(alt["walls"])[2] < 30]
    _require(any(alt["answers"] and
                 not np.array_equal(_render(p, alt, alt["answers"][0]), out)
                 for alt in other_length), "guides do not affect the answer")
    _require(all(len(_components(shape)) == 1 for _, shape in p["pieces"]), "disconnected inventory piece")
    stock = Counter(p["pieces"])
    shapes_by_color = {}
    for col, shape in p["pieces"]:
        shapes_by_color.setdefault(col, set()).add(shape)
    return {"axis": p["axis"], "ports": len(p["rack"]["bars"]),
            "compatible_ports": len(candidates), "bar_length": len(p["cap"]["bars"][0]),
            "overlap": len(rack & cap), "pieces": len(p["pieces"]),
            "piece_shapes": tuple(shape for _, shape in p["pieces"]),
            "repeated": any(n > 1 for n in stock.values()),
            "same_color_shapes": any(len(s) > 1 for s in shapes_by_color.values()),
            "global_fit": max(c["stats"]["placement_counts"]) > 1,
            "guide_payload_cells": len(cap_guide & c["holes"]),
            "cavity_shape": _normalise(c["holes"]), "cavity_area": len(c["holes"]),
            "cavity_depth": max(q[p["axis"]] for q in c["holes"]) -
                            min(q[p["axis"]] for q in c["holes"]) + 1,
            "palette": (p["bg"], p["wall_color"], tuple(sorted(shapes_by_color))),
            "input_shape": inp.shape, "output_shape": out.shape,
            "edge": bool(np.any(inp[0] != p["bg"]) or np.any(inp[-1] != p["bg"]) or
                         np.any(inp[:, 0] != p["bg"]) or np.any(inp[:, -1] != p["bg"]))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _count("training count", len(puzzle["train"]), 2, 10)
    _count("test count", len(puzzle["test"]), 1, 4)
    _require(len(puzzle["train"]) > len(puzzle["test"]), "invalid split counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp = _grid(pair["input"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(_analyze_pair(pair))
    _require(any(i["global_fit"] for i in infos["train"]), "training must show globally constrained fitting")
    _require(any(i["overlap"] == 0 for i in infos["train"]) and
             any(i["overlap"] > 0 for i in infos["train"]), "training must teach transparent wall union")
    _require({i["axis"] for i in infos["test"]} <= {i["axis"] for i in infos["train"]},
             "test registration axis was not demonstrated")
    _require(all(i["ports"] > max(t["ports"] for t in infos["train"]) for i in infos["test"]) or
             all(i["cavity_depth"] > max(t["cavity_depth"] for t in infos["train"]) for i in infos["test"]),
             "missing observable train/test shift")


def self_test():
    """Bounded 42-seed survey plus semantic negatives and count boundaries."""
    fingerprints, inputs, outputs, cavity_shapes, piece_shapes = set(), set(), set(), set(), set()
    all_info = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(_grid(pair[k])) for s in ("train", "test")
                               for pair in puzzle[s] for k in ("input", "output")))
        local = [_analyze_pair(pair) for s in ("train", "test") for pair in puzzle[s]]
        _require(len({i["cavity_shape"] for i in local}) > 1, "no within-puzzle geometric variation")
        all_info.extend(local)
        for i in local:
            inputs.add(i["input_shape"])
            outputs.add(i["output_shape"])
            cavity_shapes.add(i["cavity_shape"])
            piece_shapes.update(i["piece_shapes"])
    _require(len(fingerprints) == 42 and len(inputs) >= 25 and len(outputs) >= 25,
             "insufficient puzzle/dimension variation")
    _require(len(cavity_shapes) >= 100 and len(piece_shapes) >= 100, "insufficient structural variation")
    _require({i["axis"] for i in all_info} == {0, 1}, "missing registration axis")
    _require({i["ports"] for i in all_info} >= {3, 4, 5}, "missing port-count variation")
    _require(len({i["bar_length"] for i in all_info}) >= 3, "guide lengths did not vary")
    _require(len({i["pieces"] for i in all_info}) >= 3 and any(i["repeated"] for i in all_info)
             and any(i["same_color_shapes"] for i in all_info), "inventory variety missing")
    _require(any(i["edge"] for i in all_info) and any(i["overlap"] >= 2 for i in all_info),
             "boundary or overlap variation missing")
    a = generate_puzzle(713, 2, 1)
    generate_puzzle(714)
    b = generate_puzzle(713, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ("input", "output")), "nondeterministic puzzle")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(100 + nt * 5 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Colored answer ambiguity versus harmless identical-item exchange.
    slots = {(0, 0), (0, 2)}
    _require(len(_tile(slots, [(1, ((0, 0),)), (2, ((0, 0),))])[0]) == 2,
             "different colored tilings were merged")
    _require(len(_tile(slots, [(1, ((0, 0),)), (1, ((0, 0),))])[0]) == 1,
             "identical-piece exchanges count as ambiguity")
    oriented_l = {(0, 0), (1, 0), (1, 1)}
    _require(not _tile(oriented_l, [(2, ((0, 0), (0, 1), (1, 1)))])[0],
             "a piece was rotated or reflected")
    pair = a["train"][1]
    _, parsed, chosen, _ = _infer(pair["input"])
    # The wall-only A ablation and the payload deletion B ablation both fail.
    wall_only = _render(parsed, chosen, {})
    _require(not np.array_equal(wall_only, pair["output"]), "payloads have no effect")
    corrupt = {"input": pair["input"].copy(), "output": pair["output"].copy()}
    payload_color = parsed["pieces"][0][0]
    q = tuple(np.argwhere(corrupt["input"] == payload_color)[0])
    corrupt["input"][q] = parsed["bg"]
    try:
        _analyze_pair(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("missing inventory accepted")
    corrupt = {"input": pair["input"], "output": pair["output"].copy()}
    q = tuple(np.argwhere(corrupt["output"] == parsed["wall_color"])[0])
    corrupt["output"][q] = parsed["bg"]
    try:
        _analyze_pair(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("fixed wall corruption accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus count and semantic checks)")
