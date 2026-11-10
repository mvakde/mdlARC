"""Finite colored strokes interrupted by first-hit, port-directed relays.

Only NumPy and the standard library are used. solve() reads one input grid.
Construction records intended strokes; solving projects original footprints
onto beam lanes, so constructor labels are checked against input-derived results.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _step(p, direction, distance=1):
    dr, dc = DIRS[direction]
    return p[0] + dr * distance, p[1] + dc * distance


def _components(cells, diagonal=False):
    remaining, result = set(cells), []
    offsets = ([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                if (r, c) != (0, 0)] if diagonal else DIRS)
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        comp, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _runs(flags):
    groups = []
    for v in map(int, np.flatnonzero(flags)):
        if groups and groups[-1][1] == v:
            groups[-1][1] += 1
        else:
            groups.append([v, v + 1])
    return groups


def _decode(inp):
    """Find the separator, recover tokens, launcher, and actual relay cells."""
    bg = int(inp[0, 0])  # Both the panel and the scene have blank corner margins.
    candidates = [c for c in range(3, inp.shape[1] - 4)
                  if int(inp[0, c]) != bg and np.all(inp[:, c] == inp[0, c])]
    if len(candidates) != 1:
        raise ValueError("expected one full-height separator")
    sep = candidates[0]
    panel, scene = inp[:, :sep], inp[:, sep + 1:]
    rr = _runs(np.any(panel != bg, axis=1))
    cc = _runs(np.any(panel != bg, axis=0))
    if not rr or not cc or len(cc) < 2:
        raise ValueError("missing program lanes")
    size = (rr[0][1] - rr[0][0], cc[0][1] - cc[0][0])
    if (not all(b - a == size[0] for a, b in rr)
            or not all(b - a == size[1] for a, b in cc)):
        raise ValueError("inconsistent glyph boxes")
    tokens, masks = [], []
    for left, right in cc:
        for top, bottom in rr:
            block = panel[top:bottom, left:right]
            occupied = block != bg
            colors = np.unique(block[occupied])
            cells = set(zip(*np.nonzero(occupied)))
            if len(colors) != 1 or len(_components(cells, True)) != 1:
                raise ValueError("glyph must have connected monochrome ink")
            if not np.all(occupied.any(axis=0)) or not np.all(occupied.any(axis=1)):
                raise ValueError("glyph must span its tile")
            tokens.append((int(occupied.sum()), int(colors[0])))
            masks.append((occupied.shape, occupied.tobytes()))
    h, w = scene.shape
    if any(int(scene[p]) != bg for p in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
        raise ValueError("scene corners must be background")
    comps = _components(set(map(tuple, np.argwhere(scene != bg))))
    boundary = [c for c in comps if any(r in (0, h - 1) or s in (0, w - 1) for r, s in c)]
    if len(boundary) != 1:
        raise ValueError("expected one boundary launcher")
    launcher = boundary[0]
    starts = [p for p in launcher if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    if len(starts) != 1:
        raise ValueError("launcher has ambiguous boundary contact")
    start = starts[0]
    direction = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    line = [_step(start, direction, k) for k in range(len(launcher))]
    if set(line) != launcher or not 3 <= len(line) <= 5:
        raise ValueError("launcher must be a straight tail and cap")
    launch_color, cap_color = int(scene[start]), int(scene[line[-1]])
    if launch_color == cap_color or any(int(scene[p]) != launch_color for p in line[:-1]):
        raise ValueError("launcher cap must contrast with its tail")
    objects = []
    for comp in comps:
        if comp == launcher:
            continue
        palette, counts = np.unique([scene[p] for p in sorted(comp)], return_counts=True)
        if len(palette) != 2 or counts[0] == counts[1]:
            raise ValueError("relay must have a majority body and minority port")
        port_color = int(palette[np.argmin(counts)])
        port = {p for p in comp if int(scene[p]) == port_color}
        body = comp - port
        r0, r1 = min(r for r, c in comp), max(r for r, c in comp)
        c0, c1 = min(c for r, c in comp), max(c for r, c in comp)
        sides = [d for d, yes in enumerate((all(r == r0 for r, c in port),
                 all(c == c1 for r, c in port), all(r == r1 for r, c in port),
                 all(c == c0 for r, c in port))) if yes]
        if len(sides) != 1 or len(_components(body)) != 1 or not 1 <= len(port) <= 3:
            raise ValueError("relay has an ambiguous or disconnected port")
        side = sides[0]
        span = sorted(c if side % 2 == 0 else r for r, c in port)
        if span != list(range(span[0], span[0] + len(span))):
            raise ValueError("port must be contiguous")
        if not all(_step(p, (side + 2) % 4) in body for p in port):
            raise ValueError("every port cell needs an inward body neighbor")
        if not (0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1):
            raise ValueError("relays must be interior")
        objects.append({"cells": comp, "port": port, "direction": side,
                        "bbox": (r0, r1, c0, c1)})
    structural = {bg, int(inp[0, sep]), cap_color}
    relay_colors = {int(scene[p]) for o in objects for p in o["cells"]}
    if len(structural) != 3 or structural & relay_colors or launch_color in structural | relay_colors:
        raise ValueError("structural colors and paint roles must be distinct")
    if any(color in structural | relay_colors for length, color in tokens):
        raise ValueError("glyph paint cannot be a structural color")
    return {"bg": bg, "scene": scene, "tokens": tokens, "masks": masks,
            "rows": len(rr), "lanes": len(cc), "glyph_size": size,
            "launcher": set(line), "tail": set(line[:-1]), "cap": line[-1],
            "launch_color": launch_color, "direction": direction, "objects": objects}


def _execute(info, mode="normal"):
    """Input-derived execution; select the nearest projected original footprint."""
    scene, bg = info["scene"], info["bg"]
    out = np.full_like(scene, bg)
    for p in info["launcher"]:
        out[p] = info["launch_color"]
    tokens = list(info["tokens"])
    if mode == "row_major":
        tokens = [tokens[c * info["rows"] + r]
                  for r in range(info["rows"]) for c in range(info["lanes"])]
    objects = [] if mode == "ignore_relays" else info["objects"]
    origins, direction, reached, records = [info["cap"]], info["direction"], [], []
    index, budget = 0, None
    while index < len(tokens):
        length, color = tokens[index]
        if mode in ("ignore_colors", "unlimited"):
            color = info["launch_color"]
        if mode in ("unlimited", "ignore_lengths"):
            length = 100
        if budget is not None:
            length, budget = budget, None
        dr, dc = DIRS[direction]
        stops = []
        for r, c in origins:
            edge = r + 1 if direction == 0 else scene.shape[1] - c if direction == 1 else scene.shape[0] - r if direction == 2 else c + 1
            hit = (edge, -1)
            for j, obj in enumerate(objects):
                for rr, cc in obj["cells"]:
                    distance = (rr - r) * dr + (cc - c) * dc
                    transverse = (rr - r) * dc - (cc - c) * dr
                    if transverse == 0 and distance > 0:
                        hit = min(hit, (distance, j))
            stops.append(hit)
        # A future partial contact matters only if this instruction reaches it.
        near = min(d for d, j in stops)
        if near <= length and len(set(stops)) != 1:
            raise ValueError("partial-width or simultaneous multi-object collision")
        hit_distance, hit_index = stops[0] if near <= length else (length + 1, -2)
        count = min(length, hit_distance - 1)
        pixels = {_step(p, direction, k) for p in origins for k in range(1, count + 1)}
        if pixels & info["launcher"]:
            raise ValueError("a stroke cannot revisit the launcher")
        for p in pixels:
            out[p] = color
        record = {"token": index, "length": length, "direction": direction,
                  "width": len(origins), "empty_steps": count, "hit": None,
                  "discarded": 0, "pixels": pixels}
        if hit_distance <= length and hit_index >= 0:
            if hit_index in reached:
                raise ValueError("reactivation or cycle")
            reached.append(hit_index)
            obj = objects[hit_index]
            if obj["direction"] == (direction + 2) % 4:
                raise ValueError("relay cannot emit directly back toward its entry")
            for p in obj["cells"]:
                out[p] = color
            record.update(hit=hit_index, discarded=length - hit_distance,
                          outgoing=obj["direction"], port_width=len(obj["port"]))
            origins, direction = sorted(obj["port"]), obj["direction"]
            records.append(record)
            if mode == "resume" and length > hit_distance:
                budget = length - hit_distance
                continue
        else:
            records.append(record)
            if hit_distance <= length:  # Boundary consumes the rest of the program.
                break
            origins = [_step(p, direction, length) for p in origins]
        index += 1
    return out, {"reached": reached, "records": records}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the scene-pane output from this input alone."""
    return _execute(_decode(_grid(input_grid)))[0]


def _vocabulary(rng):
    gh, gw = ((2, 2), (2, 3), (3, 2), (3, 3))[int(rng.integers(4))]
    counts = (2, 3, 4) if gh * gw == 4 else (3, 4, 6) if gh * gw == 6 else (4, 6, 7)
    masks = []
    for count in counts:
        options = []
        for cells in itertools.combinations(range(gh * gw), count):
            mask = np.zeros((gh, gw), dtype=bool)
            mask.flat[list(cells)] = True
            ink = set(zip(*np.nonzero(mask)))
            if (np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))
                    and len(_components(ink, True)) == 1):
                options.append(mask)
        masks.append(options[int(rng.integers(len(options)))])
    return masks


@lru_cache(maxsize=192)
def _shape_options(incoming, outgoing, iw, ow, small):
    """Immutable geometric catalog; caching never stores seeds or examples."""
    minimum_h = max(3, iw if incoming % 2 else 3, ow + 2 if outgoing % 2 else 3)
    minimum_w = max(3, iw if incoming % 2 == 0 else 3, ow + 2 if outgoing % 2 == 0 else 3)
    options = []
    for h in range(minimum_h, max(minimum_h, 4 if small else 6) + 1):
        for w in range(minimum_w, max(minimum_w, 4 if small else 6) + 1):
            full = {(r, c) for r in range(h) for c in range(w)}
            cuts = [set(), {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}]
            for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
                cut = {(r, c)}
                if h >= 5:
                    cut.add((1 if r == 0 else h - 2, c))
                cuts.append(cut)
            cuts += [{(r, c)} for r in (0, h - 1) for c in range(1, w - 1)]
            cuts += [{(r, c)} for c in (0, w - 1) for r in range(1, h - 1)]
            for cut in cuts:
                cells = frozenset(full - cut)
                if len(_components(cells)) != 1:
                    continue
                def windows(side, width, interior):
                    limit = w if side % 2 == 0 else h
                    result = []
                    for start in range(1 if interior else 0, limit - width - (1 if interior else 0) + 1):
                        port = frozenset((0 if side == 0 else h - 1, start + k) for k in range(width)) if side % 2 == 0 else frozenset((start + k, w - 1 if side == 1 else 0) for k in range(width))
                        if port <= cells and all(_step(p, (side + 2) % 4) in cells - port for p in port):
                            result.append((start, port))
                    return result
                entries = windows((incoming + 2) % 4, iw, False)
                ports = windows(outgoing, ow, True)
                for _, port in ports:
                    if len(_components(cells - port)) == 1:
                        for entry, _ in entries:
                            options.append((cells, port, h, w, entry))
    return tuple(options)


def _shape(rng, incoming, outgoing, iw, ow, small=False):
    options = _shape_options(incoming, outgoing, iw, ow, small)
    return options[int(rng.integers(len(options)))] if options else None


def _separate(cells, occupied):
    return not any((r + dr, c + dc) in occupied for r, c in cells
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _layout(rng, lengths, hit_slots, widths, h, w, forced_direction=None,
            must_discard=False):
    """Bounded backtracking in stroke order; later objects cannot alter old hits."""
    direction = int(rng.integers(4)) if forced_direction is None else forced_direction
    tail_length = int(rng.integers(2, 4))
    transverse = int(rng.integers(2, (w if direction % 2 == 0 else h) - 2))
    start = (0, transverse) if direction == 2 else (h - 1, transverse) if direction == 0 else (transverse, 0) if direction == 1 else (transverse, w - 1)
    tail = {_step(start, direction, k) for k in range(tail_length)}
    cap = _step(start, direction, tail_length)
    launcher = tail | {cap}
    nodes = [160]
    hit_slots = set(hit_slots)
    slot_width = dict(zip(sorted(hit_slots), widths))
    def inside(p):
        return 0 <= p[0] < h and 0 <= p[1] < w
    def grow(i, origins, d, objects, occupied, painted, records):
        nodes[0] -= 1
        if nodes[0] < 0:
            return None
        if i == len(lengths):
            return objects, records, painted
        length = lengths[i]
        if i not in hit_slots:
            ray = {_step(p, d, k) for p in origins for k in range(1, length + 1)}
            if not all(inside(p) for p in ray) or ray & occupied:
                return None
            return grow(i + 1, [_step(p, d, length) for p in origins], d,
                        objects, occupied, painted | ray, records + [(ray, None)])
        for _ in range(35):
            outgoing = (d + int(rng.choice([-1, 0, 1], p=[.4, .2, .4]))) % 4
            if not objects and outgoing == d:
                continue  # Every interrupted demonstration exposes a turn.
            shape = _shape(rng, d, outgoing, len(origins), slot_width[i],
                           small=len(hit_slots) >= 3)
            if shape is None:
                continue
            mask, port, oh, ow, entry = shape
            high = length - 1 if must_discard and not objects else length
            if high < 2:
                continue
            distance = int(rng.integers(2, high + 1))
            first = min(origins)
            target = _step(first, d, distance)
            if d == 0:
                r0, c0 = target[0] - oh + 1, target[1] - entry
            elif d == 2:
                r0, c0 = target[0], target[1] - entry
            elif d == 1:
                r0, c0 = target[0] - entry, target[1]
            else:
                r0, c0 = target[0] - entry, target[1] - ow + 1
            if r0 < 1 or c0 < 1 or r0 + oh >= h or c0 + ow >= w:
                continue
            cells = {(r + r0, c + c0) for r, c in mask}
            placed_port = {(r + r0, c + c0) for r, c in port}
            ray = {_step(p, d, k) for p in origins for k in range(1, distance)}
            if (ray & occupied or cells & painted or not _separate(cells, occupied)
                    or not all(_step(p, d, distance) in cells for p in origins)):
                continue
            obj = {"cells": cells, "port": placed_port, "direction": outgoing}
            answer = grow(i + 1, sorted(placed_port), outgoing, objects + [obj],
                          occupied | cells, painted | ray | cells,
                          records + [(ray, len(objects))])
            if answer is not None:
                return answer
        return None
    route = grow(0, [cap], direction, [], launcher, launcher, [])
    if route is None:
        return None
    objects, records, painted = route
    return tail, cap, direction, objects, records, painted


def _add_decoys(rng, layout, count, h, w, max_width):
    tail, cap, direction, objects, records, painted = layout
    objects = list(objects)
    occupied = tail | {cap} | set().union(*(o["cells"] for o in objects))
    for _ in range(count):
        for attempt in range(180):
            outgoing = int(rng.integers(4))
            shape = _shape(rng, outgoing, outgoing, 1, int(rng.choice([1, max_width])), small=attempt >= 60)
            if shape is None:
                continue
            mask, port, oh, ow, entry = shape
            r0, c0 = int(rng.integers(1, h - oh)), int(rng.integers(1, w - ow))
            cells = {(r + r0, c + c0) for r, c in mask}
            if cells & painted or not _separate(cells, occupied):
                continue
            objects.append({"cells": cells, "port": {(r + r0, c + c0) for r, c in port}, "direction": outgoing})
            occupied |= cells
            break
        else:
            return None
    return tail, cap, direction, objects, records, painted


def _crosses(info, facts, different_colors=False):
    records = facts["records"]
    return any(a["pixels"] & b["pixels"] and
               (not different_colors or info["tokens"][a["token"]][1] != info["tokens"][b["token"]][1])
               for i, a in enumerate(records) for b in records[i + 1:])


def _make_pair(rng, masks, max_width, kind, roles, used, prior_colors=None,
               allow_crossings=True):
    # kind 0 calibrates all lengths; kind 1 covers interruption; kind 2 extends.
    glyph_lengths = [int(m.sum()) for m in masks]
    for attempt in range(100):
        n = 4 if kind == 0 else 6 if kind == 1 else int(rng.choice([8, 10, 12]))
        lanes = 3 if n == 12 else 2
        rows = n // lanes
        gh, gw = masks[0].shape
        ygap, xgap = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        panel_w = 1 + lanes * gw + (lanes - 1) * xgap + 1
        min_h = 1 + rows * gh + (rows - 1) * ygap + 1
        w = int(rng.integers(max(13, 30 - panel_w - 4), 30 - panel_w))
        h = int(rng.integers(max(min_h, 23 if kind else 18), 31))
        program = [0, 1, 2] + [int(rng.integers(3)) for _ in range(n - 3)]
        if kind == 0:
            program[-1] = 0  # The maximum tally vocabulary fits a 30-row scene.
        rng.shuffle(program)
        lengths = [glyph_lengths[k] for k in program]
        if kind == 0:
            available = [d for d in range(4) if sum(lengths) + 4 < (w if d % 2 else h)]
            if not available:
                continue
            forced = int(rng.choice(available))
            slots, widths = [], []
        else:
            hits = int(rng.choice([2, 3])) if kind == 1 else int(rng.choice([3, 4]))
            # Hold the desired hit count within this candidate's geometric search.
            long_slots = [i for i in range(min(2, n)) if lengths[i] >= 3]
            if not long_slots:
                continue
            first = int(rng.choice(long_slots))
            rest = list(map(int, rng.choice(np.arange(first + 1, n - 1), hits - 1, replace=False)))
            slots = sorted([first] + rest)
            widths = [max_width, 1] + [int(rng.choice([1, max_width])) for _ in range(hits - 2)]
            if rng.random() < .5:
                widths[0], widths[1] = widths[1], widths[0]
            forced = None
        for _ in range(8):
            layout = _layout(rng, lengths, slots, widths, h, w, forced, must_discard=kind != 0)
            if layout is not None:
                break
        else:
            continue
        decoys = int(rng.integers(1, 4)) if kind < 2 else int(rng.integers(0, 4))
        layout = _add_decoys(rng, layout, decoys, h, w, max_width)
        if layout is None:
            continue
        tail, cap, direction, objects, records, painted = layout
        bg, sep_color, cap_color, body_color, port_color, launch_color = roles[:6]
        paints = list(roles[5:])
        colors = []
        for k in program:
            excluded = set() if prior_colors is None else prior_colors[glyph_lengths[k]]
            choices = [v for v in paints if v not in excluded and (not colors or v != colors[-1])]
            colors.append(int(rng.choice(choices)))
        if kind == 0:
            # Nonadjacent, differently sized glyphs share paint. Color alone
            # therefore cannot identify a command's length.
            a, b = next((a, b) for a in range(n) for b in range(a + 2, n)
                        if program[a] != program[b])
            neighbors = {colors[j] for p in (a, b) for j in (p - 1, p + 1)
                         if 0 <= j < n and j not in (a, b)}
            shared = int(rng.choice([v for v in paints if v not in neighbors]))
            colors[a] = colors[b] = shared
        scene = np.full((h, w), bg, dtype=np.int8)
        for obj in objects:
            for p in obj["cells"]:
                scene[p] = port_color if p in obj["port"] else body_color
        for p in tail:
            scene[p] = launch_color
        scene[cap] = cap_color
        panel = np.full((h, panel_w), bg, dtype=np.int8)
        panel[:, -1] = sep_color
        for i, (k, color) in enumerate(zip(program, colors)):
            lane, row = divmod(i, rows)
            top, left = 1 + row * (gh + ygap), 1 + lane * (gw + xgap)
            panel[top:top + gh, left:left + gw][masks[k]] = color
        inp = np.concatenate([panel, scene], axis=1)
        if _key(inp) in used:
            continue
        out = np.full_like(scene, bg)
        for p in tail | {cap}:
            out[p] = launch_color
        for color, (ray, hit) in zip(colors, records):
            for p in ray:
                out[p] = color
            if hit is not None:
                for p in objects[hit]["cells"]:
                    out[p] = color
        info = _decode(inp)
        prediction, facts = _execute(info)
        if not np.array_equal(out, prediction):
            raise AssertionError("constructed strokes disagree with input-derived execution")
        if not allow_crossings and _crosses(info, facts):
            continue
        if kind != 0:
            if not any(r["discarded"] > 0 for r in facts["records"]):
                continue
            try:
                resume, _ = _execute(info, "resume")
            except ValueError:
                resume = None
            if resume is not None and np.array_equal(resume, out):
                continue
        used.add(_key(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("bounded construction exhausted; no simpler family is substituted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    masks = _vocabulary(rng)
    max_width = int(rng.choice([2, 3]))
    base = tuple(map(int, rng.permutation(10)))
    vary_palette = bool(rng.integers(2))
    result, used = {"train": [], "test": []}, set()
    first_colors = None
    demonstrated_crossing = False
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            kind = i if split == "train" and i < 2 else int(rng.integers(2)) if split == "train" else 2
            roles = tuple(map(int, np.roll(base, i + (nt if split == "test" else 0)))) if vary_palette else base
            pair = _make_pair(rng, masks, max_width, kind, roles, used,
                              first_colors if split == "train" and i == 1 else None,
                              allow_crossings=split == "train" or demonstrated_crossing)
            result[split].append(pair)
            if split == "train":
                info = _decode(pair["input"])
                demonstrated_crossing |= _crosses(info, _execute(info)[1], different_colors=True)
            if split == "train" and i == 0:
                first_colors = {}
                for length, color in _decode(pair["input"])["tokens"]:
                    first_colors.setdefault(length, set()).add(color)
    rng.shuffle(result["train"])
    validate_generated_puzzle(result)
    return result


def _different(info, out, mode):
    try:
        altered, _ = _execute(info, mode)
        return not np.array_equal(altered, out)
    except ValueError:
        return True


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, data = set(), {"train": [], "test": []}
    for split in data:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            expected, facts = _execute(info)
            assert np.array_equal(expected, out), "incorrect output"
            assert len(facts["records"]) == len(info["tokens"]), "generated program exits early"
            assert out.shape == info["scene"].shape
            assert info["objects"]
            assert set(map(int, np.unique(out))) <= {info["bg"], info["launch_color"]} | {c for n, c in info["tokens"]}
            all_objects = set().union(*(o["cells"] for o in info["objects"]))
            for i, obj in enumerate(info["objects"]):
                assert _separate(obj["cells"], (all_objects - obj["cells"]) | info["launcher"])
                if i not in facts["reached"]:
                    assert all(out[p] == info["bg"] for p in obj["cells"]), "unreached relay survives"
            assert all(out[p] == info["launch_color"] for p in info["launcher"])
            data[split].append((info, facts, out))
    train = data["train"]
    assert any(not f["reached"] and len(i["tokens"]) == 4 for i, f, o in train)
    assert any(len(f["reached"]) >= 2 and any(r["discarded"] > 0 for r in f["records"]) for i, f, o in train)
    assert any(len(f["reached"]) < len(i["objects"]) for i, f, o in train)
    transitions = [r for i, f, o in train for r in f["records"] if r["hit"] is not None]
    assert any(r["direction"] != r["outgoing"] for r in transitions)
    demonstrated_widths = {r["port_width"] for r in transitions}
    assert 1 in demonstrated_widths and len(demonstrated_widths) >= 2
    for mode in ("ignore_relays", "unlimited", "ignore_lengths", "resume", "row_major", "ignore_colors"):
        assert any(_different(i, o, mode) for i, f, o in train), f"shortcut survives: {mode}"
    all_data = train + data["test"]
    assert len({i["glyph_size"] for i, f, o in all_data}) == 1
    vocab = {m for i, f, o in train for m in i["masks"]}
    lengths = {n for i, f, o in train for n, c in i["tokens"]}
    assert len(vocab) == len(lengths) == 3
    colors_by_length = {n: set() for n in lengths}
    lengths_by_color = {}
    for i, f, o in train:
        for n, c in i["tokens"]:
            colors_by_length[n].add(c)
            lengths_by_color.setdefault(c, set()).add(n)
    assert all(len(cs) >= 2 for cs in colors_by_length.values())
    assert any(len(ns) >= 2 for ns in lengths_by_color.values())
    calibration = next(i for i, f, o in train if not f["reached"])
    assert {n for n, c in calibration["tokens"]} == lengths
    for i, f, o in data["test"]:
        assert set(i["masks"]) <= vocab
        assert len(i["tokens"]) > max(len(j["tokens"]) for j, ff, oo in train)
        assert len(f["reached"]) >= 3
        assert {r["port_width"] for r in f["records"] if r["hit"] is not None} <= demonstrated_widths
        if _crosses(i, f):
            assert any(_crosses(j, ff, different_colors=True) for j, ff, oo in train), "test crossing lacks color-priority evidence"


def self_test():
    """Bounded 42-seed semantic, diversity, mutation, and count checks."""
    signatures, shapes, structures, sizes, sides, widths, turns = set(), set(), set(), set(), set(), set(), set()
    programs, footprints, discarded = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                out, facts = _execute(info)
                assert np.array_equal(solve(np.array(pair["input"], copy=True)), pair["output"])
                shapes.add(pair["input"].shape)
                sizes.add(info["glyph_size"])
                sides.add(info["direction"])
                programs.add(len(info["tokens"]))
                mapping = {}
                structures.add(tuple(tuple(mapping.setdefault(int(c), len(mapping)) for c in row) for row in pair["input"]))
                for obj in info["objects"]:
                    r0, r1, c0, c1 = obj["bbox"]
                    footprints.add(tuple(sorted((r - r0, c - c0) for r, c in obj["cells"])))
                for rec in facts["records"]:
                    if rec["hit"] is not None:
                        widths.add(rec["port_width"])
                        turns.add((rec["outgoing"] - rec["direction"]) % 4)
                        discarded.add(rec["discarded"])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(structures) >= 180
    assert len(sizes) == 4 and sides == set(range(4))
    assert widths == {1, 2, 3} and {0, 1, 3} <= turns
    assert {4, 6, 8, 10, 12} <= programs
    assert len(footprints) >= 40 and len(discarded) >= 4
    a, b = generate_puzzle(8723, 2, 1), generate_puzzle(8723, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(53 + nt * 5 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in a[s]] for s in a}
    info = _decode(bad["test"][0]["input"])
    bad["test"][0]["output"][info["cap"]] = info["bg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "normalized_inputs": len(structures), "glyph_sizes": len(sizes),
            "footprints": len(footprints), "port_widths": sorted(widths),
            "program_lengths": sorted(programs)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
