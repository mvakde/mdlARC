"""Hidden ribbon connectors: interval geometry assembles a diagonal jigsaw.

Only NumPy and the standard library are used. No files are read or written by
generation, solving, validation, or self_test. Coordinate definitions and the
full intended family are in task_spec.md.
"""

from collections import defaultdict
from itertools import combinations
import hashlib

import numpy as np


HOST_SEEDS = (0,340813260,755720588,105220653,609951666,309996636,
              519212813,952410979,453190640,685214287,422825361,567964000,
              946595249,910852038,877689085,542081799,646330734,660397402,
              206992294,385637192,844734545,306975937,217014720,333600278,
              418843811,330346079,676416318,428027649,420067265,9911598,
              280732135,135991209,197713947,146212060,571490318,615826897,
              416566172,274211271,63483245,415802462,160994769,99456569)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _components(mask):
    """Eight-connected components; sets are consumed in sorted start order."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    answer = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in remaining:
                        remaining.remove(p)
                        part.add(p)
                        stack.append(p)
        answer.append(part)
    return answer


def _tile_coordinates(w, length, slope=1):
    side = w + length + 1
    r, c = np.indices((side, side))
    if slope == 1:
        q, t = c - r + w, r + c - w
    else:
        q, t = c + r - length, r - c + length
    inside = (q >= 0) & (q <= 2*w) & (t >= 0) & (t <= 2*length)
    border = inside & ((q == 0) | (q == 2*w) | (t == 0) | (t == 2*length))
    return q, t, inside, border


def _port(strokes, level, raw=False):
    active = sorted({q for q, lo, hi, color in strokes if lo <= level <= hi})
    if len(active) < 2:
        return ()
    if raw:
        return tuple(active)
    # The filled ribbon includes internal original strokes. Its occupied
    # cross-section is the complete interval, not the unfilled contact set.
    return tuple(range(active[0], active[-1] + 1, 2))


def _orders(ports):
    """Stop after two solutions: ambiguous chains are outside the family."""
    starts = [i for i, (a, b) in enumerate(ports) if not a and b]
    ends = [i for i, (a, b) in enumerate(ports) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in ports):
        return []
    results = []

    def visit(path, left):
        if len(results) == 2:
            return
        if not left:
            if path[-1] == ends[0]:
                results.append(tuple(path))
            return
        outgoing = ports[path[-1]][1]
        if outgoing:
            for i in left:
                if ports[i][0] == outgoing:
                    visit(path + [i], tuple(j for j in left if j != i))

    visit([starts[0]], tuple(i for i in range(len(ports)) if i != starts[0]))
    return results


def _filled_tile(tile, strokes, bg, frame, w, length, slope, alternative=""):
    """Solver's pairwise overlap construction, using parsed visible intervals."""
    q, t, inside, _ = _tile_coordinates(w, length, slope)
    pairs = list(combinations(range(len(strokes)), 2))
    if alternative == "consecutive":
        order = sorted(range(len(strokes)), key=lambda i: strokes[i][0])
        pairs = list(zip(order, order[1:]))
    fill = np.zeros(tile.shape, dtype=bool)
    for i, j in pairs:
        a, b = strokes[i], strokes[j]
        if alternative == "same_color" and a[3] != b[3]:
            continue
        low, high = max(a[1], b[1]), min(a[2], b[2])
        mask = (inside & (t >= low) & (t <= high)
                & (q > min(a[0], b[0])) & (q < max(a[0], b[0])))
        if alternative == "pixel_centers":
            mask &= ((t-a[0]) % 2 == 0) & ((t-b[0]) % 2 == 0)
        fill |= mask
    output = tile.copy()
    output[fill & (tile == bg)] = frame
    return output


def _decode(value):
    """Recognize frame, field, strokes, direction, and chain from input alone."""
    grid = _grid(value)
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if counts.max() <= grid.size/2:
        raise ValueError("the exterior field must be a strict majority")
    candidates = []
    for frame in (int(x) for x in values if x != bg):
        parts = _components(grid == frame)
        if not 2 <= len(parts) <= 5:
            continue
        for slope in (1, -1):
            tiles, geometries, covered, valid = [], [], np.zeros(grid.shape, bool), True
            for part in parts:
                points = np.array(sorted(part), dtype=int)
                r0, c0 = points.min(axis=0)
                r1, c1 = points.max(axis=0)
                if r1-r0 != c1-c0:
                    valid = False
                    break
                qq = points[:, 1] - slope*points[:, 0]
                tt = points[:, 0] + slope*points[:, 1]
                dq, dt = int(np.ptp(qq)), int(np.ptp(tt))
                if dq % 2 or dt % 2 or min(dq, dt) < 8:
                    valid = False
                    break
                w, length = dq//2, dt//2
                side = w + length + 1
                if side != r1-r0+1:
                    valid = False
                    break
                tile = grid[r0:r1+1, c0:c1+1].copy()
                q, t, inside, border = _tile_coordinates(w, length, slope)
                local_part = {(r-int(r0), c-int(c0)) for r, c in part}
                if local_part != set(map(tuple, np.argwhere(border).tolist())):
                    valid = False
                    break
                if np.any(tile[inside & ~border] == frame):
                    valid = False
                    break
                section = covered[r0:r1+1, c0:c1+1]
                if np.any(section & inside):
                    valid = False
                    break
                section |= inside
                occupied = inside & ~border & (tile != bg)
                lines = defaultdict(list)
                for r, c in np.argwhere(occupied):
                    lines[int(q[r, c])].append((int(t[r, c]), int(tile[r, c])))
                strokes = []
                for pos, line in sorted(lines.items()):
                    line.sort()
                    # One uninterrupted, monochromatic run per transverse line.
                    levels = [x[0] for x in line]
                    if (pos % 2 or not 2 <= pos <= 2*w-2 or len(levels) < 2
                            or any(b-a != 2 for a, b in zip(levels, levels[1:]))
                            or levels[0] < 2 or levels[-1] > 2*length-2
                            or levels[0] % 2 or len({x[1] for x in line}) != 1):
                        valid = False
                        break
                    strokes.append((pos, levels[0], levels[-1], line[0][1]))
                if not valid or len(strokes) < 2:
                    valid = False
                    break
                tiles.append(dict(tile=tile, strokes=strokes, top=int(r0), left=int(c0)))
                geometries.append((w, length))
            if not valid or len(set(geometries)) != 1:
                continue
            if np.any((grid != bg) & ~covered):
                continue
            w, length = geometries[0]
            ports = [(_port(x["strokes"], 2), _port(x["strokes"], 2*length-2)) for x in tiles]
            orders = _orders(ports)
            if len(orders) != 1 or len(tiles)*(length+1)+w > 30:
                continue
            candidates.append(dict(bg=bg, frame=frame, slope=slope, w=w,
                                   length=length, tiles=tiles, ports=ports,
                                   order=orders[0], grid=grid))
    if len(candidates) != 1:
        raise ValueError("expected one unique framed-ribbon chain")
    return candidates[0]


def _assemble(tiles, order, bg, w, length, slope):
    n = len(tiles)
    side = n*(length+1)+w
    if side > 30:
        raise ValueError("assembled chain exceeds ARC bounds")
    output = np.full((side, side), bg, dtype=np.int8)
    _, _, inside, _ = _tile_coordinates(w, length, slope)
    span = w + length + 1
    for rank, i in enumerate(order):
        r = rank*(length+1)
        c = (rank if slope == 1 else n-1-rank)*(length+1)
        view = output[r:r+span, c:c+span]
        # Bounding squares overlap; only the lozenge itself is copied.
        view[inside] = tiles[i][inside]
    return output


def _prediction(info, alternative="", reading_order=False):
    tiles = [_filled_tile(x["tile"], x["strokes"], info["bg"], info["frame"],
                          info["w"], info["length"], info["slope"], alternative)
             for x in info["tiles"]]
    order = tuple(range(len(tiles))) if reading_order else info["order"]
    return _assemble(tiles, order, info["bg"], info["w"], info["length"], info["slope"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, interval, connector, and translation from the input."""
    return _prediction(_decode(input_grid))


def _sample_strokes(incoming, outgoing, w, length, colors, rng, special=""):
    first, last = 2, 2*length-2
    positions = list(range(2, 2*w, 2))
    front, rear = set(), set()
    for signature, target in ((incoming, front), (outgoing, rear)):
        if signature:
            a, b = signature[0], signature[-1]
            target.update((a, b))
            target.update(p for p in positions if a < p < b and rng.random() < .35)
        elif rng.random() < .45:
            target.add(int(rng.choice(positions)))
    if special == "middle":
        front, rear = {incoming[0], incoming[-1]}, {outgoing[0], outgoing[-1]}
    elif special == "start":
        front = set()
        rear = set(outgoing)  # Different visible contacts, identical filled span.
    elif special == "end":
        front = set(incoming)
        rear = set()
    strokes = []
    for pos in positions:
        if pos in front and pos in rear:
            lo, hi = first, last
        elif pos in front:
            lo, hi = first, int(rng.choice(range(4, last, 2)))
        elif pos in rear:
            lo, hi = int(rng.choice(range(4, last, 2))), last
        elif length >= 5 and (rng.random() < .32 or
                              (special == "middle" and incoming[0] < pos < incoming[-1])):
            lo = int(rng.choice(range(4, last-2, 2)))
            hi = int(rng.choice(range(lo+2, last, 2)))
        else:
            continue
        color = colors[len(strokes) % len(colors)]
        strokes.append((pos, lo, hi, color))
    return strokes


def _render_constructed(strokes, w, length, bg, frame):
    """Ground truth sweeps metadata intervals; solve instead parses and unions."""
    q, t, inside, border = _tile_coordinates(w, length)
    tile = np.full(q.shape, bg, dtype=np.int8)
    tile[border] = frame
    for pos, low, high, color in strokes:
        tile[(q == pos) & (t >= low) & (t <= high)] = color
    result = tile.copy()
    for level in range(2, 2*length-1):
        active = [pos for pos, low, high, color in strokes if low <= level <= high]
        if len(active) >= 2:
            mask = inside & (t == level) & (q > min(active)) & (q < max(active)) & (tile == bg)
            result[mask] = frame
    return tile, result


def _source_placements(n, w, length, rng):
    """Rotated lattice strips/arrays and loose physical arrays; no separators."""
    span = w + length + 1
    choices = []
    # Longitudinal and transverse strips, as well as little diagonal lattices.
    cells_options = [[(i, 0) for i in range(n)], [(0, i) for i in range(n)]]
    if n in (3, 4):
        square = [(0, 0), (0, 1), (1, 0), (1, 1)]
        if n == 3:
            square.pop(int(rng.integers(4)))
        cells_options.append(square)
    for cells in cells_options:
        coords = [(a*(length+2)-b*(w+2), a*(length+2)+b*(w+2)) for a, b in cells]
        rmin, cmin = np.min(coords, axis=0)
        coords = [(r-int(rmin), c-int(cmin)) for r, c in coords]
        h = max(r for r, c in coords)+span
        width = max(c for r, c in coords)+span
        if max(h, width) <= 30:
            choices.append((coords, h, width))
    # These arrays have independent row/column spacing and optional vacant slot.
    for nr, nc in ((1, n), (n, 1), (2, 2)):
        if nr*nc < n or (nr*nc > n and n != 3):
            continue
        max_r = min(3, (30-nr*span)//max(1, nr-1))
        max_c = min(3, (30-nc*span)//max(1, nc-1))
        if min(max_r, max_c) < 1:
            continue
        gap_r, gap_c = int(rng.integers(1, max_r+1)), int(rng.integers(1, max_c+1))
        h, width = nr*span+(nr-1)*gap_r, nc*span+(nc-1)*gap_c
        if max(h, width) > 30:
            continue
        slots = [(r*(span+gap_r), c*(span+gap_c)) for r in range(nr) for c in range(nc)]
        if len(slots) > n:
            slots.pop(int(rng.integers(len(slots))))
        choices.append((slots, h, width))
    if not choices:
        raise ValueError("no source packing fits")
    coords, h, width = choices[int(rng.integers(len(choices)))]
    extra_h, extra_w = int(rng.integers(0, min(4, 30-h)+1)), int(rng.integers(0, min(4, 30-width)+1))
    dr, dc = int(rng.integers(extra_h+1)), int(rng.integers(extra_w+1))
    return [(r+dr, c+dc) for r, c in coords], (h+extra_h, width+extra_w)


def _make_pair(rng, kind, previous_palette=None):
    for attempt in range(128):
        if kind == "interval_witness":
            n = 3
        elif kind in ("matching_witness", "combined_test"):
            n = 4
        elif kind == "ordinary_train":
            n = int(rng.choice((2, 3)))
        else:
            n = int(rng.choice((2, 3, 4), p=(.22, .25, .53)))
        configs = [(w, length) for w in range(4, 10) for length in range(4, 9)
                   if w+length+1 <= 14 and n*(length+1)+w <= 30
                   and (kind != "interval_witness" or (length >= 5 and w >= 5))
                   and (kind != "matching_witness" or w >= 6)
                   and (kind != "combined_test" or w >= 5)]
        w, length = configs[int(rng.integers(len(configs)))]
        colors = list(map(int, rng.permutation(10)))
        bg, frame = colors[0], colors[1]
        if previous_palette is not None and (bg == previous_palette[0] or frame == previous_palette[1]):
            continue
        pool = colors[2:]
        marker_count = 2 if kind == "interval_witness" else (1 if kind == "matching_witness" else int(rng.integers(1, 4)))
        pool = pool[:marker_count]
        port_pool = [tuple(range(a, b+1, 2)) for a in range(2, 2*w, 2)
                     for b in range(a+4, 2*w, 2)]
        if kind == "interval_witness":
            wide = [p for p in port_pool if len(p) >= 4]
            signature = wide[int(rng.integers(len(wide)))]
            links = [signature, signature]
        elif kind == "matching_witness":
            # Three equally wide, differently offset keys: ignoring offsets
            # makes the two intermediate pieces interchangeable.
            widths = sorted({len(p) for p in port_pool if sum(len(q) == len(p) for q in port_pool) >= 3})
            width = int(rng.choice(widths))
            same_width = [p for p in port_pool if len(p) == width]
            ids = rng.choice(len(same_width), 3, replace=False)
            links = [same_width[int(i)] for i in ids]
        elif kind == "combined_test":
            ids = rng.choice(len(port_pool), 2, replace=False)
            p, q = [port_pool[int(i)] for i in ids]
            links = ((p,p,q), (p,q,q), (p,q,p))[int(rng.integers(3))]
            links = list(links)
        else:
            links = [port_pool[int(rng.integers(len(port_pool)))] for _ in range(n-1)]
        if len(_orders(list(zip([()] + links, links + [()])))) != 1:
            continue
        original, filled = [], []
        for i in range(n):
            incoming, outgoing = (() if i == 0 else links[i-1]), (() if i == n-1 else links[i])
            special = ("start" if i == 0 else "end" if i == n-1 else "middle") if kind == "interval_witness" else ""
            strokes = _sample_strokes(incoming, outgoing, w, length, pool, rng, special)
            a, b = _render_constructed(strokes, w, length, bg, frame)
            original.append(a)
            filled.append(b)
        coords, shape = _source_placements(n, w, length, rng)
        permutation = list(map(int, rng.permutation(n)))
        source = np.full(shape, bg, dtype=np.int8)
        _, _, inside, _ = _tile_coordinates(w, length)
        span = w+length+1
        for (r, c), i in zip(coords, permutation):
            view = source[r:r+span, c:c+span]
            view[inside] = original[i][inside]
        output = _assemble(filled, tuple(range(n)), bg, w, length, 1)
        if rng.random() < .5:
            source, output = np.fliplr(source).copy(), np.fliplr(output).copy()
        pair = dict(input=source, output=output)
        try:
            info = _decode(source)
        except ValueError:
            continue
        if not np.array_equal(_prediction(info), output):
            continue
        if kind in ("interval_witness", "matching_witness") and np.array_equal(_prediction(info, reading_order=True), output):
            continue
        if kind == "interval_witness":
            raw = [(_port(x["strokes"], 2, True), _port(x["strokes"], 2*length-2, True)) for x in info["tiles"]]
            if _orders(raw):
                continue
            if any(np.array_equal(_prediction(info, alternative), output)
                   for alternative in ("consecutive", "pixel_centers", "same_color")):
                continue
        return pair, (bg, frame)
    raise RuntimeError("bounded ribbon construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    kinds = ["interval_witness", "matching_witness"] + ["ordinary_train"]*(num_train-2)
    rng.shuffle(kinds)
    puzzle, seen, previous = dict(train=[], test=[]), set(), None
    tests = ["combined_test"] + ["ordinary"]*(num_test-1)
    rng.shuffle(tests)
    for split, local_kinds in (("train", kinds), ("test", tests)):
        for kind in local_kinds:
            for attempt in range(32):
                pair, palette = _make_pair(rng, kind, previous)
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    previous = palette
                    break
            else:
                raise RuntimeError("could not obtain distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _decode(pair["input"])
    output = np.asarray(pair["output"])
    raw = [(_port(x["strokes"], 2, True), _port(x["strokes"], 2*info["length"]-2, True))
           for x in info["tiles"]]
    widths = [((len(a),) if a else (), (len(b),) if b else ()) for a, b in info["ports"]]
    return dict(n=len(info["tiles"]), w=info["w"], length=info["length"],
                shape=tuple(info["grid"].shape), slope=info["slope"], bg=info["bg"],
                frame=info["frame"], joins=tuple(info["ports"][i][1] for i in info["order"][:-1]),
                stroke_counts=tuple(len(x["strokes"]) for x in info["tiles"]),
                marker_colors=tuple(sorted({s[3] for x in info["tiles"] for s in x["strokes"]})),
                raw_fails=not bool(_orders(raw)),
                width_ambiguous=len(_orders(widths)) == 2,
                reorder=not np.array_equal(_prediction(info, reading_order=True), output),
                skip=not np.array_equal(_prediction(info, "consecutive"), output),
                subpixel=not np.array_equal(_prediction(info, "pixel_centers"), output),
                color=not np.array_equal(_prediction(info, "same_color"), output),
                fill_cells=sum(np.count_nonzero(x["tile"] != _filled_tile(x["tile"], x["strokes"], info["bg"],
                               info["frame"], info["w"], info["length"], info["slope"])) for x in info["tiles"]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), dict(train=[], test=[])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            info = _decode(a)
            assert np.array_equal(_prediction(info), b), "incorrect ribbon assembly or fill"
            # Each original stroke, including its literal color, is copied.
            for rank, i in enumerate(info["order"]):
                tile = info["tiles"][i]["tile"]
                r = rank*(info["length"]+1)
                c = (rank if info["slope"] == 1 else len(info["tiles"])-1-rank)*(info["length"]+1)
                q, t, inside, border = _tile_coordinates(info["w"], info["length"], info["slope"])
                view = b[r:r+tile.shape[0], c:c+tile.shape[1]]
                original = inside & (tile != info["bg"])
                assert np.array_equal(view[original], tile[original]), "original ink lost"
                changed = inside & (view != tile)
                assert np.all(tile[changed] == info["bg"]) and np.all(view[changed] == info["frame"])
                # Port silhouette is the filled interval, not marker count.
                filled = _filled_tile(tile, info["tiles"][i]["strokes"], info["bg"], info["frame"],
                                      info["w"], info["length"], info["slope"])
                for level, signature in zip((2, 2*info["length"]-2), info["ports"][i]):
                    if signature:
                        assert all(filled[(t == level) & (q == p)][0] != info["bg"] for p in signature)
            features[split].append(_features(pair))
    train = features["train"]
    for field in ("raw_fails", "reorder", "skip", "subpixel", "color", "width_ambiguous"):
        assert any(x[field] for x in train), "missing teaching witness: " + field
    assert all(x["fill_cells"] > 0 for values in features.values() for x in values)
    assert any(len(set(x["joins"])) >= 2 for x in train), "show offset-sensitive matching"
    assert any(len(x["marker_colors"]) == 1 for x in train), "show geometry without color labels"
    combined = lambda x: x["n"] == 4 and len(set(x["joins"])) < len(x["joins"])
    assert not any(combined(x) for x in train) and any(combined(x) for x in features["test"]), "missing combination shift"


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds streamed in memory; retain only bounded digests and feature sets."""
    fingerprints, shapes, geometries, palettes, directions, counts = set(), set(), set(), set(), set(), set()
    total, raw, skipped, reordered, repeated, layouts = 0, 0, 0, 0, 0, set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_digest(puzzle))
        assert _digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism"
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(np.asarray(pair["input"]).copy()), pair["output"])
                x = _features(pair)
                info = _decode(pair["input"])
                shapes.add(x["shape"])
                geometries.add((x["w"], x["length"]))
                palettes.add((x["bg"], x["frame"], x["marker_colors"]))
                directions.add(x["slope"])
                counts.add(x["n"])
                # Normalize away all colors to demonstrate actual geometric diversity.
                normalized = np.zeros(info["grid"].shape, np.int8)
                normalized[info["grid"] == info["frame"]] = 1
                normalized[(info["grid"] != info["bg"]) & (info["grid"] != info["frame"])] = 2
                layouts.add(hashlib.sha256(normalized.tobytes()+bytes(normalized.shape)).hexdigest())
                total += 1
                raw += x["raw_fails"]
                skipped += x["skip"]
                reordered += x["reorder"]
                repeated += len(set(x["joins"])) < len(x["joins"])
        # Solver does not mutate the caller's input.
        a = puzzle["test"][0]["input"].copy()
        saved = a.copy()
        solve(a)
        assert np.array_equal(a, saved)
    assert len(fingerprints) == 42 and len(layouts) >= 190
    assert len(shapes) >= 30 and len(geometries) >= 15 and len(palettes) >= 100
    assert directions == {-1, 1} and counts == {2, 3, 4}
    assert raw >= 42 and skipped >= 42 and reordered >= 84 and repeated >= 42
    # Every legal explicit count combination, including the two-demo minimum.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1701, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Analytical continuous-interval fixture: two length-three boundaries
    # at q=2,8 and t=2..6 bracket twelve cells, six at odd t.
    strokes = [(2, 2, 6, 3), (8, 2, 6, 7)]
    tile, result = _render_constructed(strokes, 5, 4, 0, 9)
    q, t, _, _ = _tile_coordinates(5, 4)
    expected_mask = ((t >= 2) & (t <= 6) & (q > 2) & (q < 8) & (tile == 0))
    assert np.count_nonzero(expected_mask) == 12
    assert np.array_equal(result != tile, expected_mask)
    assert np.array_equal(_filled_tile(tile, strokes, 0, 9, 5, 4, 1), result)
    assert not np.array_equal(_filled_tile(tile, strokes, 0, 9, 5, 4, 1, "pixel_centers"), result)
    p, qsig = (2,4,6), (4,6,8)
    assert len(_orders([((),p),(p,p),(p,qsig),(qsig,())])) == 1
    assert len(_orders([((),p),(p,p),(p,p),(p,())])) == 2
    # An incorrect label fails even when all grid dimensions remain legal.
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"][0,0] = (int(p["test"][0]["output"][0,0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, pairs=total, distinct_puzzles=len(fingerprints),
                normalized_inputs=len(layouts), input_shapes=len(shapes),
                diamond_geometries=len(geometries), palettes=len(palettes),
                raw_contact_failures=raw, inactive_interior_witnesses=skipped,
                reordered_pairs=reordered, repeated_join_pairs=repeated)
