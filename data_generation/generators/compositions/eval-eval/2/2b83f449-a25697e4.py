"""Ink bridges: rigid fitting installs colored routes for settling beads.

Only standard-library Python and NumPy are used. Solving reads one input only.
"""
from itertools import combinations
import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 52364957, 696024822, 11912432, 125504924, 872671643,
              109538467, 958980904, 733367533, 201651393, 768946261,
              254456637, 542192302, 464048017, 534340480, 724222638,
              695641167, 741529643, 313806061, 218260864, 160117146,
              480148751, 711902861, 389613012, 28243144, 480311470,
              378137164, 660258966, 641161520, 30613571, 297028259,
              276846177, 922900890, 448705394, 107651210, 600036093,
              965377595, 207863122, 637713461, 839934638, 662148142,
              804439555)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(frozenset(found))
    return result


def _cells(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _box(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _raw(p, op):
    r, c = p
    return ((r, c), (c, -r), (-r, -c), (-c, r),
            (r, -c), (-c, -r), (-r, c), (c, r))[op]


def _transform(regions, op):
    changed = [[_raw(p, op) for p in region] for region in regions]
    r0 = min(p[0] for region in changed for p in region)
    c0 = min(p[1] for region in changed for p in region)
    return tuple(frozenset((r - r0, c - c0) for r, c in region)
                 for region in changed)


def _shape(cells):
    r0, _, c0, _ = _box(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _canonical(cells):
    return min(_shape(_transform((cells,), op)[0]) for op in range(8))


def _neighbors(p, cells):
    return [(p[0] + dr, p[1] + dc) for dr, dc in DIRS
            if (p[0] + dr, p[1] + dc) in cells]


def _paths(cells, *, monotone=True):
    """Reject branches/loops; installed routes also forbid vertical traps."""
    paths = []
    for comp in _components(cells):
        ends = sorted(p for p in comp if len(_neighbors(p, comp)) == 1)
        if len(ends) != 2 or any(len(_neighbors(p, comp)) not in (1, 2) for p in comp):
            raise ValueError("walkable components must be simple paths")
        path, prev = [ends[0]], None
        while path[-1] != ends[1]:
            choices = [p for p in _neighbors(path[-1], comp) if p != prev]
            if len(choices) != 1:
                raise ValueError("ambiguous path")
            prev = path[-1]
            path.append(choices[0])
        steps = {int(np.sign(b[0] - a[0])) for a, b in zip(path, path[1:])
                 if a[0] != b[0]}
        if monotone and len(steps) > 1:
            raise ValueError("path contains an internal gravitational trap")
        paths.append(path)
    return paths


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=True)


def _roles(inp):
    vals, counts = np.unique(inp, return_counts=True)
    if np.sum(counts == counts.max()) != 1:
        raise ValueError("background must be the unique most frequent color")
    bg = int(vals[counts.argmax()])
    square = (inp[:-1, :-1] == inp[1:, :-1]) & (inp[:-1, :-1] == inp[:-1, 1:])
    square &= inp[:-1, :-1] == inp[1:, 1:]
    walls = set(map(int, inp[:-1, :-1][square])) - {bg}
    if len(walls) != 1:
        raise ValueError("walls must be the unique foreground color with a solid 2x2")
    wall = next(iter(walls))
    comps = _components(_cells((inp != bg) & (inp != wall)))
    plain = {int(inp[next(iter(comp))]) for comp in comps
             if len({int(inp[p]) for p in comp}) == 1}
    if len(plain) != 1:
        raise ValueError("plain track components must identify one track color")
    track = next(iter(plain))
    substrate, beads, keys = set(), {}, []
    for comp in comps:
        colors = sorted({int(inp[p]) for p in comp})
        if track in colors:
            _paths(comp)
            substrate.update(comp)
            for p in comp:
                if int(inp[p]) != track:
                    beads[p] = int(inp[p])
        else:
            if len(colors) != 2:
                raise ValueError("loose bridges must have exactly two colors")
            regions = tuple(frozenset(p for p in comp if int(inp[p]) == color)
                            for color in colors)
            if any(len(_components(region)) != 1 for region in regions):
                raise ValueError("bridge color regions must be connected")
            _paths(comp, monotone=False)
            keys.append({"cells": comp, "regions": regions, "colors": tuple(colors)})
    if not keys or not beads:
        raise ValueError("at least one bridge and one bead are required")
    return bg, wall, track, substrate, beads, keys


def _targets(inp, bg, wall):
    fragments = _components(_cells(inp == wall))
    targets = []
    for i, j in combinations(range(len(fragments)), 2):
        fixed = fragments[i] | fragments[j]
        r0, r1, c0, c1 = _box(fixed)
        rect = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
        if any(int(inp[p]) != bg for p in rect - fixed):
            continue
        slot = rect - fixed
        if not slot or len(_components(slot)) != 1:
            continue
        try:
            _paths(slot)
        except ValueError:
            continue
        targets.append({"parts": frozenset((i, j)), "rect": rect, "slot": slot})
    return fragments, targets


def _settle(inp, bg, track, substrate, beads, keys, moves, *, ink=True):
    """Stable packing and last-contact ink; also returns measurable witnesses."""
    placed = {p: color for move in moves for p, color in move["paint"].items()}
    bridge_at = {p: i for i, move in enumerate(moves) for p in move["paint"]}
    if len(placed) != sum(len(move["paint"]) for move in moves):
        raise ValueError("installed bridges overlap")
    paths = _paths(substrate | set(placed))
    endpoints = {p for path in paths for p in (path[0], path[-1])}
    if not set(beads) <= endpoints:
        raise ValueError("beads must start at complete-path endpoints")
    out = inp.copy()
    for key in keys:
        for p in key["cells"]:
            out[p] = bg
    for p in beads:
        out[p] = track
    for p, color in placed.items():
        out[p] = color
    records = []
    for path in paths:
        initial = [(p, beads[p]) for p in (path[0], path[-1]) if p in beads]
        if path[0][0] == path[-1][0]:
            records.append({"path": path, "initial": initial, "final": initial,
                            "journeys": [], "level": True, "bridges": 0})
            for p, color in initial:
                out[p] = color
            continue
        ordered = path if path[0][0] > path[-1][0] else path[::-1]
        rank = {p: i for i, p in enumerate(ordered)}
        initial.sort(key=lambda item: rank[item[0]])
        final, journeys = [], []
        for index, (p, original_color) in enumerate(initial):
            target = ordered[index]
            visited = ordered[index:rank[p] + 1][::-1]
            inks = [placed[q] for q in visited if q in placed]
            bridge_inks = []
            for q in visited:
                if q not in placed:
                    continue
                gate = bridge_at[q]
                if bridge_inks and bridge_inks[-1][0] == gate:
                    bridge_inks[-1] = (gate, placed[q])
                else:
                    bridge_inks.append((gate, placed[q]))
            color = inks[-1] if ink and inks else original_color
            out[target] = color
            final.append((target, color))
            journeys.append({"start": p, "end": target, "original": original_color,
                             "inks": inks, "bridge_inks": bridge_inks, "color": color})
        bridges = sum(bool(set(path) & set(move["paint"])) for move in moves)
        records.append({"path": ordered, "initial": initial, "final": final,
                        "journeys": journeys, "level": False, "bridges": bridges})
    return out, records, placed


def _analyze(input_grid, *, swap=True, ink=True):
    inp = _grid(input_grid)
    bg, wall, track, substrate, beads, keys = _roles(inp)
    fragments, targets = _targets(inp, bg, wall)
    if len(fragments) != 2 * len(keys):
        raise ValueError("each bridge needs two wall fragments")
    options = []
    for key in keys:
        candidates, signatures = [], set()
        for target in targets:
            sr0, _, sc0, _ = _box(target["slot"])
            for op in range(8):
                regions = _transform(key["regions"], op)
                for plug_index in (0, 1):
                    pr0, _, pc0, _ = _box(regions[plug_index])
                    shifted = tuple(_shift(region, sr0 - pr0, sc0 - pc0) for region in regions)
                    if shifted[plug_index] != target["slot"]:
                        continue
                    other = shifted[1 - plug_index]
                    if other & target["rect"]:
                        continue
                    if any(not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1])
                           or int(inp[r, c]) != bg for r, c in other):
                        continue
                    colors = key["colors"][::-1] if swap and op >= 4 else key["colors"]
                    paint = {p: colors[i] for i, region in enumerate(shifted) for p in region}
                    sig = (target["parts"], tuple(sorted(paint.items())))
                    if sig not in signatures:
                        signatures.add(sig)
                        candidates.append({"parts": target["parts"], "paint": paint,
                                           "slot": target["slot"], "rect": target["rect"],
                                           "op": op, "plug_index": plug_index})
        if not candidates:
            raise ValueError("bridge has no legal fit")
        options.append(candidates)
    answers = {}

    def search(index, used_parts, occupied, moves):
        if len(answers) > 1:
            return
        if index == len(keys):
            try:
                out, records, placed = _settle(inp, bg, track, substrate, beads, keys,
                                               moves, ink=ink)
            except ValueError:
                return
            sig = out.tobytes()
            answers.setdefault(sig, (out, moves, records, placed))
            return
        for move in options[index]:
            if used_parts & move["parts"] or occupied & set(move["paint"]):
                continue
            search(index + 1, used_parts | move["parts"],
                   occupied | set(move["paint"]), moves + [move])

    search(0, set(), set(), [])
    if len(answers) != 1:
        raise ValueError("need one unique colored fit with valid track topology")
    out, moves, records, placed = next(iter(answers.values()))
    return out, {"background": bg, "wall": wall, "track": track, "substrate": substrate,
                 "beads": beads, "keys": keys, "moves": moves, "records": records,
                 "placed": placed, "fragments": fragments}


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _analyze(input_grid)[0]


def _module(rng, small=False, fallback_index=None):
    """A separator slot and its attached bent tail, with exposed track ports."""
    for _ in range(100):
        h = int(rng.integers(3, 5 if small else 7))
        w = int(rng.integers(5, 7 if small else 8))
        a, b = map(int, rng.choice(np.arange(1, w - 1), 2, replace=False))
        cut = int(rng.integers(1, h - 1))
        if fallback_index is not None:
            h, w, a, b, cut = 3 + fallback_index, 5, 1, 2, 1
        plug_path = [(r, a) for r in range(cut + 1)]
        direction = 1 if b > a else -1
        plug_path += [(cut, c) for c in range(a + direction, b + direction, direction)]
        plug_path += [(r, b) for r in range(cut + 1, h)]
        if fallback_index is None and h >= 5 and cut <= h - 4 and rng.random() < .4:
            cut2 = int(rng.integers(cut + 2, h - 1))
            c2 = int(rng.choice([c for c in range(1, w - 1) if c != b]))
            plug_path = [p for p in plug_path if p[0] <= cut2]
            direction2 = 1 if c2 > b else -1
            plug_path += [(cut2, c) for c in range(b + direction2, c2 + direction2, direction2)]
            plug_path += [(r, c2) for r in range(cut2 + 1, h)]
            b = c2
        plug = set(plug_path)
        rect = {(r, c) for r in range(h) for c in range(w)}
        walls = rect - plug
        if len(_components(walls)) != 2:
            continue
        if not any({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= walls
                   for r, c in walls):
            continue
        down = int(rng.integers(1, 4)) if not small else 1
        side = int(rng.choice((-1, 1)))
        across = int(rng.integers(1, 5 if not small else 3))
        tail_path = [(r, b) for r in range(h, h + down)]
        tail_path += [(h + down - 1, b + side * k) for k in range(1, across + 1)]
        last_down = int(rng.integers(1, 4)) if not small else 1
        if fallback_index is not None:
            # Rebuild the tail below after selecting a compact safe geometry.
            down, side, across, last_down = (2, 1, 1, 3) if fallback_index == 0 else (1, 1, 1, 1)
            tail_path = [(r, b) for r in range(h, h + down)]
            tail_path.append((h + down - 1, b + 1))
        tail_path += [(r, b + side * across) for r in range(h + down, h + down + last_down)]
        tail = set(tail_path)
        entry = (-1, a)
        exit_cell = (tail_path[-1][0] + 1, tail_path[-1][1])
        raw_path = [entry] + plug_path + tail_path + [exit_cell]
        op = int(rng.integers(8))
        if fallback_index is not None:
            op = 1
        changed = [_raw(p, op) for p in raw_path]
        steps = {int(np.sign(v[0] - u[0])) for u, v in zip(changed, changed[1:])
                 if u[0] != v[0]}
        if len(steps) != 1:
            continue
        regions = _transform((walls, plug, tail, {entry}, {exit_cell}), op)
        wall_cells, plug, tail, en, ex = regions
        en, ex = next(iter(en)), next(iter(ex))
        # Use graph ordering after normalization rather than raw path offsets.
        try:
            full_path = _paths(plug | tail | {en, ex})[0]
        except ValueError:
            continue
        if full_path[0][0] > full_path[-1][0]:
            full_path.reverse()
        en, ex = full_path[0], full_path[-1]
        all_cells = wall_cells | plug | tail | {en, ex}
        _, bottom, _, right = _box(all_cells)
        extensions = {(r, en[1]) for r in range(0, en[0])}
        extensions |= {(r, ex[1]) for r in range(ex[0] + 1, bottom + 1)}
        if extensions & (wall_cells | plug | tail):
            continue
        ports = extensions | {en, ex}
        try:
            _paths(plug | tail | ports)
        except ValueError:
            continue
        return {"walls": wall_cells, "plug": plug, "tail": tail, "ports": ports,
                "entry": (0, en[1]), "exit": (bottom, ex[1]),
                "height": bottom + 1, "width": right + 1}
    raise ValueError("could not construct a bridge module")


def _moat(cells):
    return set(cells) | {(r + dr, c + dc) for r, c in cells for dr, dc in DIRS}


def _draw_pair(rng, count, chain_sizes, palette, witness=False, level=False, fallback=False):
    bg, wall, track, ink_a, ink_b = palette
    modules = [_module(rng, small=count == 3, fallback_index=i if fallback else None)
               for i in range(count)]
    if len({_canonical(m["plug"]) for m in modules}) != count:
        raise ValueError("sample distinguishable slots")
    chain_modules, at = [], 0
    for n in chain_sizes:
        chain_modules.append(modules[at:at + n])
        at += n
    if sum(chain_sizes) != count:
        raise ValueError("bad chain sizes")
    heights = [sum(m["height"] for m in ms) + 3 * (len(ms) - 1) + 6
               for ms in chain_modules]
    if max(heights) > 30:
        raise ValueError("chain too tall")
    height = int(rng.integers(max(heights), 31))
    width = int(rng.integers(24 if len(chain_sizes) == 2 else 19, 31))
    if fallback:
        height = width = 30
    inp = np.full((height, width), bg, dtype=np.int8)
    substrate, walls, future, sources, moves, initial_beads = set(), set(), set(), [], [], {}
    i = 0
    for chain_index, ms in enumerate(chain_modules):
        band_width = width // len(chain_sizes)
        band_left = chain_index * band_width
        max_width = max(m["width"] for m in ms)
        if max_width + 2 > band_width:
            raise ValueError("chain too wide")
        y = int(rng.integers(3, height - heights[chain_index] + 4))
        entries, exits = [], []
        for m in ms:
            x = int(rng.integers(band_left + 1, band_left + band_width - m["width"]))
            local = {k: _shift(m[k], y, x) for k in ("walls", "plug", "tail", "ports")}
            en = m["entry"][0] + y, m["entry"][1] + x
            ex = m["exit"][0] + y, m["exit"][1] + x
            entries.append(en)
            exits.append(ex)
            walls.update(local["walls"])
            substrate.update(local["ports"])
            future.update(local["plug"] | local["tail"])
            # Witness chains expose both parities and different outgoing inks.
            op = i % 2 * 4 + int(rng.integers(4)) if witness else int(rng.integers(8))
            desired_tail = ink_a if i % 2 == 0 else ink_b
            desired_plug = ink_b if desired_tail == ink_a else ink_a
            colors = (desired_plug, desired_tail)
            inverse = (0, 3, 2, 1, 4, 5, 6, 7)[op]
            key = _transform((local["plug"], local["tail"]), inverse)
            incoming_colors = colors[::-1] if op >= 4 else colors
            sources.append((key, incoming_colors))
            moves.append({"paint": {p: colors[j] for j, region in enumerate((local["plug"], local["tail"]))
                                     for p in region}, "op": op})
            i += 1
            y += m["height"] + 3
        first, last = entries[0], exits[-1]
        substrate.update((r, first[1]) for r in range(first[0] - 3, first[0]))
        substrate.update((r, last[1]) for r in range(last[0] + 1, last[0] + 4))
        for a, b in zip(exits, entries[1:]):
            middle = a[0] + 2
            substrate.update((r, a[1]) for r in range(a[0] + 1, middle + 1))
            substrate.update((middle, c) for c in range(min(a[1], b[1]), max(a[1], b[1]) + 1))
            substrate.update((r, b[1]) for r in range(middle + 1, b[0]))
        high, low = (first[0] - 3, first[1]), (last[0] + 3, last[1])
        occupancy = 2 if witness else 1 if level else int(rng.choice((0, 1, 1, 1, 2, 2, 3)))
        if occupancy in (1, 2):
            initial_beads[high] = int(rng.choice((ink_a, ink_b)))
        if occupancy in (2, 3):
            initial_beads[low] = int(rng.choice((ink_a, ink_b)))
    if walls & (substrate | future) or substrate & future:
        raise ValueError("construction overlaps")
    _paths(substrate | future)
    # Plain and sometimes occupied level tracks also visibly identify track.
    occupied = walls | substrate | future
    for decorated in ([False, True] if level else [False]):
        found = False
        for _ in range(100):
            length = int(rng.integers(3, 8))
            r = int(rng.integers(height))
            c = int(rng.integers(width - length + 1))
            line = {(r, x) for x in range(c, c + length)}
            if not _moat(line) & occupied:
                substrate.update(line)
                occupied.update(line)
                if decorated:
                    initial_beads[r, c if rng.random() < .5 else c + length - 1] = ink_a
                found = True
                break
        if not found:
            raise ValueError("level track does not fit")
    for p in walls:
        inp[p] = wall
    for p in substrate:
        inp[p] = track
    for p, color in initial_beads.items():
        inp[p] = color
    keys = []
    reserved = _moat(occupied)
    for regions, colors in sorted(sources, key=lambda item: -sum(map(len, item[0]))):
        all_cells = regions[0] | regions[1]
        _, r1, _, c1 = _box(all_cells)
        options = [(r, c) for r in range(height - r1) for c in range(width - c1)]
        rng.shuffle(options)
        for r, c in options:
            shifted = tuple(_shift(region, r, c) for region in regions)
            cells = shifted[0] | shifted[1]
            if cells & reserved:
                continue
            for j, region in enumerate(shifted):
                for p in region:
                    inp[p] = colors[j]
            keys.append({"cells": cells})
            reserved.update(_moat(cells))
            break
        else:
            raise ValueError("loose bridges do not fit")
    out, _, _ = _settle(inp, bg, track, substrate, initial_beads, keys, moves)
    if rng.random() < .5:
        inp, out = np.fliplr(inp).copy(), np.fliplr(out).copy()
    if rng.random() < .8:
        used = _cells((inp != bg) | (out != bg))
        r0, r1, c0, c1 = _box(used)
        margin = rng.integers(0, 3, size=4)
        r0, r1 = max(0, r0 - int(margin[0])), min(height - 1, r1 + int(margin[1]))
        c0, c1 = max(0, c0 - int(margin[2])), min(width - 1, c1 + int(margin[3]))
        inp, out = inp[r0:r1 + 1, c0:c1 + 1].copy(), out[r0:r1 + 1, c0:c1 + 1].copy()
    return {"input": inp, "output": out}


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 to 10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be 1..4 and smaller than num_train")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    vary_palette = rng.random() < .5
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n in (("train", num_train), ("test", num_test)):
        for index in range(n):
            witness = split == "train" and index == 0
            level = split == "train" and index == 1
            count = 2 if witness else 1 if level else 3 if split == "test" else int(rng.integers(1, 3))
            chains = [count] if witness or count == 1 or rng.random() < .7 else [count - 1, 1]
            pair_palette = tuple(map(int, rng.choice(10, 5, replace=False))) if vary_palette else palette
            for attempt in range(250):
                try:
                    pair = _draw_pair(rng, count, chains, pair_palette, witness, level,
                                      fallback=attempt >= 200)
                    predicted, info = _analyze(pair["input"])
                    if not np.array_equal(predicted, pair["output"]):
                        continue
                    if witness and not _witness(info):
                        continue
                    if split == "test" and not _query_motion(info):
                        continue
                    sig = (pair["input"].shape, pair["input"].tobytes())
                    if sig in seen:
                        continue
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded bridge construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _witness(info):
    return ({int(m["op"] >= 4) for m in info["moves"]} == {0, 1}
            and {-1, 1} <= {int(np.sign(2 * len(m["slot"]) - len(m["paint"])))
                            for m in info["moves"]}
            and any(r["bridges"] >= 2 and len(r["initial"]) == 2
                    and any(len(j["bridge_inks"]) >= 2
                            and j["bridge_inks"][0][1] != j["bridge_inks"][-1][1]
                            and j["original"] != j["color"]
                            and r["initial"][0][1] != j["color"]
                            for j in r["journeys"]) for r in info["records"]))


def _query_motion(info):
    return any(r["bridges"] >= 2
               and any(len(j["bridge_inks"]) >= 2 and j["start"] != j["end"]
                       and j["original"] != j["color"] for j in r["journeys"])
               for r in info["records"])


def _ablation_checks(inp, out, info):
    """Counter-rules must disagree for the designated teaching witness."""
    assembled = inp.copy()
    for key in info["keys"]:
        for p in key["cells"]:
            assembled[p] = info["background"]
    for p, color in info["placed"].items():
        assembled[p] = color
    assert not np.array_equal(assembled, out), "omitting A's transport must change the answer"
    no_fit, fragments, _ = _settle(inp, info["background"], info["track"], info["substrate"],
                                 info["beads"], info["keys"], [])
    real_positions = {p for r in info["records"] for p, c in r["final"]}
    fragment_positions = {p for r in fragments for p, c in r["final"]}
    assert real_positions != fragment_positions, "B's fitting must change bead grouping/destination"
    assert not np.array_equal(no_fit, out)
    no_ink = _settle(inp, info["background"], info["track"], info["substrate"],
                     info["beads"], info["keys"], info["moves"], ink=False)[0]
    assert not np.array_equal(no_ink, out), "the connecting ink rule must affect beads"
    try:
        no_exchange = _analyze(inp, swap=False)[0]
    except ValueError:
        # A counter-rule that loses uniqueness is also incompatible.
        return
    assert not np.array_equal(no_exchange, out), "B's parity must affect installed colors"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, witnesses, levels, singles, first_contact = set(), False, False, False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            sig = inp.shape, inp.tobytes()
            assert sig not in seen
            seen.add(sig)
            expected, info = _analyze(inp)
            assert np.array_equal(expected, out)
            assert not np.array_equal(inp, out)
            assert len(info["fragments"]) == 2 * len(info["moves"])
            assert all(int(out[p]) == info["wall"] for fragment in info["fragments"] for p in fragment)
            assert all(int(out[p]) == info["background"] for key in info["keys"] for p in key["cells"])
            assert sum(len(r["initial"]) for r in info["records"]) == len(info["beads"])
            assert sum(len(r["final"]) for r in info["records"]) == len(info["beads"])
            for r in info["records"]:
                for p, color in r["final"]:
                    # Generated beads stop on substrate, so occupancy is visible.
                    assert p not in info["placed"] and int(out[p]) == color
                if len(r["initial"]) == 2 and not r["level"]:
                    assert r["final"][0][0] == r["path"][0]
                    assert r["final"][1][0] == r["path"][1]
                    assert r["final"][0][1] == r["initial"][0][1]
            if split == "train":
                witnesses |= _witness(info)
                if _witness(info):
                    _ablation_checks(inp, out, info)
                levels |= any(r["level"] and r["initial"] for r in info["records"])
                singles |= any(len(r["initial"]) == 1
                               and any(j["start"] != j["end"] for j in r["journeys"])
                               for r in info["records"])
                first_contact |= any(j["inks"] and j["inks"][0] != j["inks"][-1]
                                     for r in info["records"] for j in r["journeys"])
            else:
                assert _query_motion(info), "each query must exercise multi-bridge transport and recoloring"
    assert witnesses and levels, "training needs parity, multi-bridge packing, last-ink and level witnesses"
    # A single moving bead is added by the second demonstration's sampling.
    assert singles and first_contact, "training needs single transport and a first-contact counterexample"


def self_test():
    dimensions, slots, tails, operations, counts, relations, roles = set(), set(), set(), set(), set(), set(), set()
    signatures, structures = set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((p["input"].shape, p["input"].tobytes())
                             for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                predicted, info = _analyze(inp.copy())
                assert np.array_equal(predicted, out)
                dimensions.add(inp.shape)
                roles.add((info["background"], info["wall"], info["track"]))
                counts.add(len(info["moves"]))
                mapping = {}
                normalized = tuple(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add((inp.shape, normalized))
                for key, move in zip(info["keys"], info["moves"]):
                    pi = move["plug_index"]
                    slots.add(_canonical(key["regions"][pi]))
                    tails.add(_canonical(key["regions"][1 - pi]))
                    relations.add(int(np.sign(len(key["regions"][pi]) - len(key["regions"][1 - pi]))))
                    operations.add(move["op"])
    assert len(signatures) == 42 and len(structures) >= 150
    assert len(dimensions) >= 40 and len(slots) >= 20 and len(tails) >= 5
    assert len(roles) >= 35 and counts == {1, 2, 3}
    assert operations == set(range(8)) and relations == {-1, 0, 1}
    a, b = generate_puzzle(314159, 6, 3), generate_puzzle(314159, 6, 3)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicitly exercise the compact fallback, including both query layouts.
    rng = np.random.default_rng(77)
    for count, chains, witness, level in ((1, [1], False, True), (2, [2], True, False),
                                          (3, [3], False, False), (3, [2, 1], False, False)):
        for attempt in range(40):
            try:
                pair = _draw_pair(rng, count, chains, (0, 8, 5, 1, 2), witness, level, fallback=True)
                solved, info = _analyze(pair["input"])
                assert np.array_equal(solved, pair["output"])
                if witness and not _witness(info):
                    continue
                break
            except ValueError:
                continue
        else:
            raise AssertionError("compact fallback failed")
    # A local new palette must change ink but preserve geometry and bead destinations.
    pair = next(p for p in a["train"] if _witness(_analyze(p["input"])[1]))
    old_out, old_info = _analyze(pair["input"])
    local = pair["input"].copy()
    key = old_info["keys"][0]
    fresh = sorted(set(range(10)) - set(map(int, np.unique(local))))[:2]
    color_map = dict(zip(key["colors"], fresh))
    for p in key["cells"]:
        local[p] = color_map[int(local[p])]
    local_out, local_info = _analyze(local)
    assert set(local_info["placed"]) == set(old_info["placed"])
    assert {p for r in local_info["records"] for p, c in r["final"]} == {
        p for r in old_info["records"] for p, c in r["final"]}
    for p, color in old_info["moves"][0]["paint"].items():
        assert local_info["placed"][p] == color_map[color]
    assert not np.array_equal(local_out, old_out)
    # Label corruption and malformed branching inputs must fail validation.
    damaged = generate_puzzle(0)
    damaged["test"][0]["output"] = damaged["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    for invalid in ({(1, 1), (0, 1), (2, 1), (1, 0)},
                    {(0, 0), (0, 1), (1, 0), (1, 1)},
                    {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)}):
        try:
            _paths(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("branch, cycle or gravitational trap accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
