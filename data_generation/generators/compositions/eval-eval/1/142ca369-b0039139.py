"""Billiard stencil printer: a component-counted, color-changing stamp walk.

No files, network, parent imports, latent solving parameters, or saved traces.
The self test streams 42 puzzles and retains only bounded scalar/set summaries.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "illegal color")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, group = [first], [first]
        while todo:
            r, c = todo.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
                    group.append(p)
        result.append(group)
    return result


def _crop(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty stencil")
    low, high = points.min(axis=0), points.max(axis=0) + 1
    return mask[low[0]:high[0], low[1]:high[1]].copy()


def _room_slice(d, position):
    r, c = position
    p = d["pitch"]
    return np.s_[(r+1)*p+1:(r+2)*p, c*p+1:(c+1)*p]


def _decode_top(value):
    """All roles and the chamber metric are recovered from this input alone."""
    a = _grid(value)
    wall = int(a[0, 0])
    _require(all(np.all(x == wall) for x in (a[0], a[-1], a[:, 0], a[:, -1])), "outer frame")
    rows = np.flatnonzero(np.all(a == wall, axis=1))
    _require(len(rows) >= 4, "rack plus at least two board rows required")
    pitch = int(rows[1])
    _require(pitch >= 4 and np.array_equal(rows, np.arange(0, a.shape[0], pitch)), "horizontal lattice")
    side = pitch - 1
    _require(side in (3, 4, 5), "supported chamber sizes are 3, 4, 5")
    cols = np.flatnonzero(np.all(a[pitch:] == wall, axis=0))
    _require(np.array_equal(cols, np.arange(0, a.shape[1], pitch)) and len(cols) >= 4,
             "vertical lattice")
    _require(cols[-1] == a.shape[1]-1 and rows[-1] == a.shape[0]-1, "complete lattice")
    _require(np.all(a[:pitch+1, pitch] == wall), "cartridge divider")
    nr, nc = len(rows)-2, len(cols)-1
    interiors = np.concatenate([a[(r+1)*pitch+1:(r+2)*pitch, c*pitch+1:(c+1)*pitch].ravel()
                                for r in range(nr) for c in range(nc)])
    colors, counts = np.unique(interiors, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(bg != wall and counts.max() > len(interiors)/2, "majority board background")
    d = {"input": a, "wall": wall, "bg": bg, "pitch": pitch, "side": side,
         "nr": nr, "nc": nc}
    master, counter = a[1:pitch, 1:pitch], a[1:pitch, pitch+1:-1]
    for panel in (master, counter):
        _require(bg in panel and len(np.unique(panel)) == 2 and wall not in panel, "binary cartridge")
    tile = _crop(master != bg)
    objects = _components(counter != bg)
    _require(len(objects) == len(_components(counter != bg, True)), "counter connectivity ambiguity")
    _require(len(_components(tile)) == 1, "connected stencil required")
    d.update(tile=tile, quota=len(objects), counter=counter != bg,
             master=master != bg, counter_sizes=tuple(len(x) for x in objects))
    hardware, sources = {}, []
    for r in range(nr):
        for c in range(nc):
            room = a[_room_slice(d, (r, c))]
            points = np.argwhere(room != bg)
            if not len(points):
                continue
            shades = set(map(int, room[room != bg]))
            _require(len(shades) == 1 and wall not in shades, "monochromatic hardware")
            shade = next(iter(shades))
            low, high = points.min(axis=0), points.max(axis=0)
            if len(points) == 3 and tuple(high-low) == (1, 1):
                cells = set(map(tuple, points))
                missing = next((x, y) for x in range(int(low[0]), int(high[0])+1)
                               for y in range(int(low[1]), int(high[1])+1) if (x, y) not in cells)
                elbow = tuple(map(int, low + high - missing))
                direction = elbow[0]-missing[0], elbow[1]-missing[1]
                sources.append(((r, c), direction, shade))
                hardware[r, c] = ("source", shade)
            else:
                _require(len(points) >= 3 and len(points) % 2 == 1, "odd mirror bar")
                if low[0] == high[0]:
                    axis = "H"
                elif low[1] == high[1]:
                    axis = "V"
                else:
                    raise AssertionError("unknown board object")
                _require(max(high-low)+1 == len(points), "solid mirror")
                _require(tuple((low+high)//2) == ((side-1)//2,)*2, "centered mirror")
                hardware[r, c] = (axis, shade)
    _require(len(sources) == 1, "exactly one directional head")
    d.update(hardware=hardware, source=sources[0])
    return d


def _decode(value):
    a = _grid(value)
    candidates = []
    for transposed in (False, True):
        try:
            d = _decode_top(a.T if transposed else a)
        except AssertionError:
            continue
        d["transposed"] = transposed
        candidates.append(d)
    _require(len(candidates) == 1, "no unique top/left cartridge parse")
    return candidates[0]


def _walk(d):
    """Chamber-center ray tracing; stamping consumes fuel, reflection does not."""
    position, direction, shade = d["source"]
    visited = {position}
    prints, turns = [], []
    while len(prints) < d["quota"]:
        position = position[0]+direction[0], position[1]+direction[1]
        _require(0 <= position[0] < d["nr"] and 0 <= position[1] < d["nc"], "fuel outlasts board")
        _require(position not in visited, "revisited chamber")
        visited.add(position)
        obj = d["hardware"].get(position)
        if obj is None:
            prints.append((position, shade))
        else:
            axis, new_shade = obj
            _require(axis in ("H", "V"), "head collision")
            turns.append((position, axis, shade, new_shade, len(prints)))
            shade = new_shade
            direction = (-direction[0], direction[1]) if axis == "H" else (direction[0], -direction[1])
    next_position = position[0]+direction[0], position[1]+direction[1]
    return prints, turns, visited, next_position


def _stamp(tile, side, shade, bg):
    result = np.full((side, side), bg, dtype=np.int8)
    h, w = tile.shape
    r, c = (side-h)//2, (side-w)//2
    result[r:r+h, c:c+w][tile] = shade
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    prints, _, _, _ = _walk(d)
    output = d["input"].copy()
    for position, shade in prints:
        output[_room_slice(d, position)] = _stamp(d["tile"], d["side"], shade, d["bg"])
    return output.T.copy() if d["transposed"] else output


def _grow(h, w, area, rng):
    mask = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        mask[r, c] = True
        for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= x < h and 0 <= y < w and not mask[x, y]:
                frontier.add((x, y))
    return mask


def _make_counter(h, w, n, rng, area=None):
    """Unequal connected objects in boxes separated even under 8-connectivity."""
    _require(w >= 2*n-1, "counter width")
    widths = np.ones(n, dtype=int)
    slack = w-(2*n-1)
    for _ in range(int(rng.integers(slack+1))):
        candidates = np.flatnonzero(widths < 4)
        if len(candidates):
            widths[int(rng.choice(candidates))] += 1
    free = w-(int(widths.sum())+n-1)
    gaps = np.zeros(n+1, dtype=int)
    for _ in range(free):
        gaps[int(rng.integers(n+1))] += 1
    sizes = np.ones(n, dtype=int)
    if area is not None:
        for _ in range(area-n):
            choices = np.flatnonzero(sizes < h*widths)
            sizes[int(rng.choice(choices))] += 1
    mask = np.zeros((h, w), dtype=bool)
    cursor = int(gaps[0])
    for i, width in enumerate(widths):
        width = int(width)
        amount = int(sizes[i]) if area is not None else int(rng.integers(1, min(12, h*width)+1))
        obj = _grow(h, width, amount, rng)
        if area is None and h >= 3 and width >= 3 and rng.random() < .15:
            obj[:] = True
            obj[1:-1, 1:-1] = False
        mask[:, cursor:cursor+width] = obj
        cursor += width+1+int(gaps[i+1])
    return mask


def _make_master(side, rng, teaching, alternate=False):
    if teaching:
        # Asymmetric in the room; outputs establish orientation preservation.
        tile = np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=bool) if alternate else np.array(
            [[1, 1, 1], [1, 0, 0]], dtype=bool)
        tile = np.rot90(tile, int(rng.integers(4)))
        if rng.integers(2):
            tile = tile[:, ::-1]
    else:
        h, w = int(rng.integers(1, side+1)), int(rng.integers(1, side+1))
        mode = int(rng.integers(4))
        if mode == 0:
            tile = np.ones((h, w), dtype=bool)
            if h >= 3 and w >= 3:
                tile[1:-1, 1:-1] = False
        elif mode == 1:
            tile = np.ones((h, w), dtype=bool)
        else:
            tile = _crop(_grow(h, w, int(rng.integers(1, h*w+1)), rng))
    master = np.zeros((side, side), dtype=bool)
    h, w = tile.shape
    if h == w == side and tile.all():
        tile[0, 0] = False  # keep the cartridge visibly binary
    r, c = int(rng.integers(side-h+1)), int(rng.integers(side-w+1))
    if teaching:
        r, c = side-h, side-w
    master[r:r+h, c:c+w] = tile
    return master


def _plan(nr, nc, quota, mirrors, required_axis, rng, teaching):
    """Bounded DFS over a small diagonal graph, not a parent generator."""
    starts = [(r, c, dr, dc) for r in range(nr) for c in range(nc) for dr, dc in DIRS]
    rng.shuffle(starts)
    for sr, sc, dr, dc in starts[:64]:
        nodes = [0]

        def visit(r, c, dr, dc, printed, turned, seen, path, axes):
            nodes[0] += 1
            if nodes[0] > 3000:
                return None
            if printed == quota:
                ahead = r+dr, c+dc
                if (turned == mirrors and (required_axis is None or required_axis in axes)
                        and 0 <= ahead[0] < nr and 0 <= ahead[1] < nc and ahead not in seen):
                    return path
                return None
            r, c = r+dr, c+dc
            if not (0 <= r < nr and 0 <= c < nc) or (r, c) in seen:
                return None
            actions = ["P", "H", "V"]
            rng.shuffle(actions)
            for action in actions:
                if action != "P" and (turned == mirrors or (teaching and printed == 0)):
                    continue
                if action == "P" and printed+1 == quota and turned < mirrors:
                    continue
                nd = (dr, dc) if action == "P" else ((-dr, dc) if action == "H" else (dr, -dc))
                result = visit(r, c, *nd, printed+(action == "P"), turned+(action != "P"),
                               seen | {(r, c)}, path+[((r, c), action)], axes | {action})
                if result is not None:
                    return result
            return None

        path = visit(sr, sc, dr, dc, 0, 0, {(sr, sc)}, [], set())
        if path is not None:
            return (sr, sc), (dr, dc), path
    return None


def _paint_bar(room, axis, shade, rng):
    center = (room.shape[0]-1)//2
    length = int(rng.choice([3, 5])) if room.shape[0] == 5 else 3
    low, high = center-length//2, center+length//2+1
    if axis == "H":
        room[center, low:high] = shade
    else:
        room[low:high, center] = shade


def _make_pair(rng, quota, required_axis=None, teaching=False):
    for attempt in range(24):
        side = 3 if teaching or quota >= 5 or attempt >= 12 else int(rng.choice([3, 4, 5]))
        if side == 3:
            nr = int(rng.integers(5, 7)) if quota >= 5 else int(rng.integers(3, 7))
            nc = int(rng.integers(5, 8)) if quota >= 5 else int(rng.integers(4, 8))
        elif side == 4:
            nr, nc = int(rng.integers(3, 5)), int(rng.integers(4, 6))
        else:
            nr, nc = 3, 4
        if attempt >= 12:
            nr, nc = 6, 7
        capacity = (nr*nc+1)//2
        maximum = min(4, capacity-quota-2)
        if maximum < (1 if teaching else 0):
            continue
        mirrors = 1 if teaching else int(rng.integers(2 if quota >= 7 else 0, maximum+1))
        if attempt >= 12 and not teaching:
            mirrors = 4 if quota == 8 else (2 if quota >= 5 else 1)
        plan = _plan(nr, nc, quota, mirrors, required_axis, rng, teaching)
        if plan is None:
            continue
        source, direction, path = plan
        colors = list(map(int, rng.permutation(10)))
        bg, wall, stencil_ink, counter_ink, initial = colors[:5]
        ink_pool = colors[2:]
        pitch = side+1
        a = np.full(((nr+1)*pitch+1, nc*pitch+1), wall, dtype=np.int8)
        a[1:pitch, 1:pitch] = np.where(_make_master(side, rng, teaching, quota == 3), stencil_ink, bg)
        counter = _make_counter(side, a.shape[1]-pitch-2, quota, rng, 6 if teaching else None)
        a[1:pitch, pitch+1:-1] = np.where(counter, counter_ink, bg)
        d = {"pitch": pitch}
        for r in range(nr):
            for c in range(nc):
                a[_room_slice(d, (r, c))] = bg
        room = a[_room_slice(d, source)]
        dr, dc = direction
        top, left = int(rng.integers(side-1)), int(rng.integers(side-1))
        elbow = top+(dr > 0), left+(dc > 0)
        missing = elbow[0]-dr, elbow[1]-dc
        for cell in (elbow, (elbow[0], missing[1]), (missing[0], elbow[1])):
            room[cell] = initial
        active = initial
        for position, action in path:
            if action != "P":
                if teaching:
                    shade = int(rng.choice([x for x in ink_pool if x != active]))
                else:
                    shade = active if rng.random() < .30 else int(rng.choice(ink_pool))
                _paint_bar(a[_room_slice(d, position)], action, shade, rng)
                active = shade
        # Off-route mirrors are real hardware, without a promise that all fire.
        used = {source} | {p for p, _ in path}
        final = path[-1][0]
        outgoing = direction
        for _, action in path:
            if action == "H":
                outgoing = -outgoing[0], outgoing[1]
            elif action == "V":
                outgoing = outgoing[0], -outgoing[1]
        next_position = final[0]+outgoing[0], final[1]+outgoing[1]
        available = [(r, c) for r in range(nr) for c in range(nc)
                     if (r, c) not in used and (r, c) != next_position]
        rng.shuffle(available)
        decoys = min(len(available), int(rng.integers(1 if teaching else 0, 4)))
        for position in available[:decoys]:
            _paint_bar(a[_room_slice(d, position)], str(rng.choice(["H", "V"])), int(rng.choice(ink_pool)), rng)
        # Construct the label directly from the planned walk, then verify input.
        tile = _crop(a[1:pitch, 1:pitch] != bg)
        output, active = a.copy(), initial
        for position, action in path:
            if action == "P":
                output[_room_slice(d, position)] = _stamp(tile, side, active, bg)
            else:
                active = int(a[_room_slice(d, position)][(side-1)//2, (side-1)//2])
        _require(np.array_equal(output, solve(a)), "builder/solver disagreement")
        return {"input": a, "output": output}
    raise RuntimeError("bounded route construction exhausted")


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    left_rack = bool(rng.integers(2))
    axes = ["H", "V"]
    rng.shuffle(axes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = split == "train" and index < 2
            quota = index+2 if teaching else int(rng.integers(1, 5) if split == "train" else rng.integers(5, 9))
            for _ in range(16):
                pair = _make_pair(rng, quota, axes[index] if teaching else None, teaching)
                different_direction = (not teaching or index == 0 or
                                       _decode(pair["input"])["source"][1] !=
                                       _decode(puzzle["train"][0]["input"])["source"][1])
                if _key(pair["input"]) not in seen and different_direction:
                    break
            else:
                raise RuntimeError("unique input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    if left_rack:
        puzzle = {s: [{k: v.T.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            _require(inp.shape == out.shape, "preserve canvas")
            d = _decode(inp)
            if d["transposed"]:
                inp, out = inp.T, out.T
            prints, turns, visited, ahead = _walk(d)
            _require(len(prints) == d["quota"] == len(set(p for p, _ in prints)), "print quota")
            _require(0 <= ahead[0] < d["nr"] and 0 <= ahead[1] < d["nc"]
                     and ahead not in visited and ahead not in d["hardware"], "visible fuel horizon")
            # Check unchanged hardware, rack and walls, then each stencil pixel.
            expected_change = np.zeros(inp.shape, dtype=bool)
            for position, shade in prints:
                region = _room_slice(d, position)
                actual = out[region]
                h, w = d["tile"].shape
                top, left = (d["side"]-h)//2, (d["side"]-w)//2
                for r in range(d["side"]):
                    for c in range(d["side"]):
                        occupied = top <= r < top+h and left <= c < left+w and d["tile"][r-top, c-left]
                        _require(actual[r, c] == (shade if occupied else d["bg"]), "stencil/color/centering error")
                expected_change[region] = actual != d["bg"]
            _require(np.array_equal(inp != out, expected_change), "changed nonprint cells")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "invented color")
            if split == "train":
                train.append((d, prints, turns))
    _require({t[1] for _, _, turns in train for t in turns} == {"H", "V"}, "teach both reflection axes")
    _require(len({d["source"][1] for d, _, _ in train}) >= 2, "teach multiple launch directions")
    _require(any(any((p[0]-d["source"][0][0])*d["source"][1][0] !=
                         (p[1]-d["source"][0][1])*d["source"][1][1]
                         for p, _ in prints) for d, prints, _ in train), "reflection ablation witness")
    _require({2, 3} <= {d["quota"] for d, _, _ in train}, "teach different quotas")
    _require(any(d["counter"].sum() == e["counter"].sum() and d["quota"] != e["quota"]
                 for d, _, _ in train for e, _, _ in train), "area/count counterexample")
    _require(any(t[2] != t[3] and 0 < t[4] < d["quota"] for d, _, turns in train for t in turns),
             "teach color change before and after a mirror")
    _require(any(not d["tile"].all() for d, _, _ in train), "binary stencil witness")
    _require(any(not np.array_equal(d["master"], _stamp(d["tile"], d["side"], 1, 0).astype(bool))
                 or d["master"].shape != d["tile"].shape for d, _, _ in train), "crop/registration witness")
    _require(sum(any(p not in {t[0] for t in turns} and kind[0] != "source"
                     for p, kind in d["hardware"].items()) for d, _, turns in train) >= 2,
             "unused mirror witness")


def self_test():
    """Bounded, streaming checks over 42 seeds; no execution traces are saved."""
    digests, structures, shapes, rooms, directions, quotas, tile_shapes = set(), set(), set(), set(), set(), set(), set()
    rack_orientations = set()
    axes, palette_bgs, decoys, unchanged, large_objects = set(), set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                d = _decode(a)
                prints, turns, _, _ = _walk(d)
                digest.update(bytes(a.shape)+a.tobytes()+pair["output"].tobytes())
                _require(np.array_equal(solve(a.copy()), pair["output"]), "solve mismatch")
                shapes.add(a.shape)
                rooms.add((d["nr"], d["nc"], d["side"]))
                rack_orientations.add(d["transposed"])
                directions.add(d["source"][1])
                quotas.add(d["quota"])
                tile_shapes.add(d["tile"].shape)
                palette_bgs.add(d["bg"])
                axes.update(t[1] for t in turns)
                decoys += len(d["hardware"])-1-len(turns)
                unchanged += sum(t[2] == t[3] for t in turns)
                large_objects += max(d["counter_sizes"]) > 3
                mapping = {}
                normalized = bytes(mapping.setdefault(int(x), len(mapping)) for x in a.flat)
                structures.add(hashlib.sha256(bytes(a.shape)+normalized).digest())
        digests.add(digest.digest())
        if seed in (0, 17, 41):
            other = generate_puzzle(seed)
            _require(all(np.array_equal(p[k], q[k]) for s in puzzle
                         for p, q in zip(puzzle[s], other[s]) for k in p), "nondeterminism")
    _require(len(digests) == 42 and len(structures) >= 150, "structural diversity")
    _require(len(shapes) >= 15 and len(rooms) >= 14 and len(tile_shapes) >= 8, "geometry diversity")
    _require(rack_orientations == {False, True}, "top and left racks")
    _require(directions == set(DIRS) and axes == {"H", "V"}, "direction coverage")
    _require(quotas == set(range(1, 9)) and palette_bgs == set(range(10)), "quota/palette coverage")
    _require(decoys > 50 and unchanged > 5 and large_objects > 10, "combined feature coverage")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(2026, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("invalid counts accepted")
    puzzle = generate_puzzle(0)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
