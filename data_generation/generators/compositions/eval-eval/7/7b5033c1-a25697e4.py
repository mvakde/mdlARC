"""Dock and reweave a two-color ribbon. No parent code or runtime data needed."""
from __future__ import annotations

from itertools import combinations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be 0..9")
    return a


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _normalize(points):
    r0, _, c0, _ = _bbox(points)
    return [(r - r0, c - c0) for r, c in points]


def _transform(point, op):
    """Four rotations, followed by four orientation-reversing isometries."""
    r, c = point
    return ((r, c), (c, -r), (-r, -c), (-c, r),
            (r, -c), (-c, -r), (-r, c), (c, r))[op]


def _signature(cells):
    return tuple(sorted(_normalize(list(cells))))


def _orbit(cells):
    return {_signature([_transform(p, op) for p in cells]) for op in range(8)}


def _canonical(cells):
    return min(_orbit(cells))


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _trace(cells, start=None):
    """Trace an induced orthogonal chain, including every individual cell."""
    cells = set(cells)
    _require(len(cells) >= 2, "a path needs at least two cells")
    neighbors = {p: sorted((p[0] + dr, p[1] + dc) for dr, dc in DIR4
                           if (p[0] + dr, p[1] + dc) in cells) for p in cells}
    endpoints = sorted(p for p in cells if len(neighbors[p]) == 1)
    _require(len(endpoints) == 2 and all(len(n) in (1, 2) for n in neighbors.values()),
             "path must have two endpoints and no branches or side contacts")
    if start is None:
        start = endpoints[0]
    _require(start in endpoints, "traversal must start at an endpoint")
    ordered, visited, previous, current = [], set(), None, start
    while current is not None:
        _require(current not in visited, "path contains a cycle")
        ordered.append(current)
        visited.add(current)
        choices = [q for q in neighbors[current] if q != previous]
        previous, current = current, choices[0] if choices else None
    _require(visited == cells, "path is disconnected")
    return ordered


def _head_fits(source_head, slot_order):
    """The head's starting endpoint must map to the channel's starting endpoint."""
    fits = {}
    k = len(source_head)
    for destination in (slot_order, slot_order[::-1]):
        prefix = destination[:k]
        for op in range(8):
            transformed = [_transform(p, op) for p in source_head]
            dr = prefix[0][0] - transformed[0][0]
            dc = prefix[0][1] - transformed[0][1]
            placed = [(r + dr, c + dc) for r, c in transformed]
            if placed == prefix:
                # Equivalent descriptions are harmless only if direction and
                # reflection parity agree. Different parities swap the colors.
                fits[(tuple(destination), op >= 4)] = op
    return [(list(direction), bool(parity), op)
            for (direction, parity), op in fits.items()]


def _analyse(input_grid):
    inp = _grid(input_grid)
    hist = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(hist.argmax())
    _require(int(hist[bg]) > inp.size // 2, "background must occupy a strict majority")
    pieces = _components(tuple(map(int, row)) for row in np.argwhere(inp != bg))
    mono, ribbons = [], []
    for piece in pieces:
        colors = {int(inp[p]) for p in piece}
        if len(colors) == 1:
            mono.append((piece, next(iter(colors))))
        elif len(colors) == 2:
            ribbons.append((piece, tuple(sorted(colors))))
        else:
            raise ValueError("each object must be a fragment or a two-color ribbon")
    _require(len(mono) == 2 and len(ribbons) == 1, "need two target fragments and one ribbon")
    _require(mono[0][1] == mono[1][1], "target fragments must have the same color")
    fixed = mono[0][0] | mono[1][0]
    target_color = mono[0][1]
    source_cells, colors = ribbons[0]
    _require(target_color not in colors, "target color must differ from ribbon colors")
    source = _trace(source_cells)
    _require(source[0][0] != source[-1][0], "source endpoints cannot tie in row")
    if source[0][0] > source[-1][0]:
        source.reverse()
    sequence = [int(inp[p]) for p in source]
    k = next((i for i, color in enumerate(sequence) if color != sequence[0]), len(sequence))
    _require(4 <= k < len(source), "initial monochrome head must have at least four cells")
    r0, r1, c0, c1 = _bbox(fixed)
    rectangle = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    _require({p for p in rectangle if inp[p] != bg} == fixed,
             "target rectangle must contain only fixed fragments and background")
    slot = rectangle - fixed
    _require(len(slot) == len(source), "ribbon and channel lengths must agree")
    slot_order = _trace(slot)
    fits = _head_fits(source[:k], slot_order)
    _require(len(fits) == 1, "head must uniquely determine channel direction and reflection parity")
    destination, reflected, op = fits[0]
    return {"input": inp, "background": bg, "fixed": fixed, "target_color": target_color,
            "rectangle": rectangle, "slot": slot, "source": source, "colors": colors,
            "sequence": sequence, "head_length": k, "destination": destination,
            "reflected": reflected, "operation": op}


def _paint(info, sequence):
    out = np.full(info["input"].shape, info["background"], dtype=np.int8)
    for p in info["fixed"]:
        out[p] = info["target_color"]
    for p, color in zip(info["destination"], sequence):
        out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fit the rigid head, then transfer the flexible ribbon in traversal order."""
    info = _analyse(input_grid)
    sequence = info["sequence"]
    if info["reflected"]:
        a, b = info["colors"]
        sequence = [b if color == a else a for color in sequence]
    return _paint(info, sequence)


def _possible_heads(path, head_band):
    low, high = head_band
    return [k for k in range(low, min(high, len(path) - 6) + 1)
            if len(_orbit(path[:k])) == 8
            and _canonical(path[:k]) != _canonical(path[-k:])]


def _channel(rng, length_band, head_band, wandering):
    """Random separators, with induced random walks as well as stepped cuts."""
    for attempt in range(220):
        height = int(rng.integers(5, 12))
        width = int(rng.integers(7, 15))
        row = int(rng.integers(1, height - 1))
        path = [(row, 0)]
        if wandering and attempt < 140:
            path.append((row, 1))
            occupied = set(path)
            previous = (0, 1)
            while len(path) < length_band[1]:
                candidates, weights = [], []
                for dr, dc in DIR4:
                    q = (path[-1][0] + dr, path[-1][1] + dc)
                    if q in occupied or not (1 <= q[0] < height - 1 and 1 <= q[1] < width):
                        continue
                    if q[1] == width - 1 and len(path) + 1 < length_band[0]:
                        continue
                    if sum((q[0] + a, q[1] + b) in occupied for a, b in DIR4) != 1:
                        continue
                    candidates.append((q, (dr, dc)))
                    weights.append((2.0 if (dr, dc) == previous else 1.0) * (1.4 if dc == 1 else 1.0))
                if not candidates:
                    break
                probs = np.asarray(weights) / sum(weights)
                q, previous = candidates[int(rng.choice(len(candidates), p=probs))]
                path.append(q)
                occupied.add(q)
                if q[1] == width - 1:
                    break
            if path[-1][1] != width - 1:
                continue
        else:
            # Vertical corridors are separated by at least one untouched
            # column, preventing unintended four-neighbor contacts.
            columns = list(range(2, width - 2, 2))
            rng.shuffle(columns)
            columns = set(columns[:int(rng.integers(2, len(columns) + 1))])
            for c in range(1, width):
                path.append((row, c))
                if c in columns:
                    new_row = int(rng.integers(1, height - 1))
                    step = 1 if new_row > row else -1
                    for r in range(row + step, new_row + step, step):
                        path.append((r, c))
                    row = new_row
        if not length_band[0] <= len(path) <= length_band[1]:
            continue
        target = {(r, c) for r in range(height) for c in range(width)} - set(path)
        if len(_components(target)) != 2:
            continue
        if rng.random() < 0.5:
            path.reverse()
        choices = _possible_heads(path, head_band)
        if not choices:
            continue
        return target, path, int(rng.choice(choices))
    raise ValueError("bounded channel search exhausted")


def _scan_orders(path):
    """Eight raster scans, none of which consult path adjacency."""
    return [sorted(path, key=lambda p, axis=axis, sa=sa, sb=sb:
                   (sa * p[axis], sb * p[1 - axis]))
            for axis in (0, 1) for sa in (-1, 1) for sb in (-1, 1)]


def _source_path(rng, destination, k, above_mode):
    length = len(destination)
    options = []
    for op in range(8):
        head = [_transform(p, op) for p in destination[:k]]
        above = min(r for r, c in head) < head[0][0]
        if above_mode == "forbid" and above:
            continue
        if above_mode == "require" and not above:
            continue
        options.append(op)
    rng.shuffle(options)
    for op in options:
        head = _normalize([_transform(p, op) for p in destination[:k]])
        for attempt in range(65):
            path, occupied = list(head), set(head)
            previous = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
            persistence = float(rng.uniform(0.8, 4.0))
            for i in range(k, length):
                candidates, weights = [], []
                r0, r1, c0, c1 = _bbox(occupied)
                for dr, dc in DIR4:
                    q = (path[-1][0] + dr, path[-1][1] + dc)
                    if q in occupied or (above_mode == "forbid" and q[0] < path[0][0]):
                        continue
                    if sum((q[0] + a, q[1] + b) in occupied for a, b in DIR4) != 1:
                        continue
                    if max(r1, q[0]) - min(r0, q[0]) >= 18 or max(c1, q[1]) - min(c0, q[1]) >= 14:
                        continue
                    candidates.append((q, (dr, dc)))
                    weights.append((persistence if (dr, dc) == previous else 1.0) * (1.4 if dr == 1 else 1.0))
                if not candidates:
                    break
                probs = np.asarray(weights) / sum(weights)
                q, previous = candidates[int(rng.choice(len(candidates), p=probs))]
                path.append(q)
                occupied.add(q)
            if len(path) != length or path[-1][0] <= path[0][0]:
                continue
            if above_mode == "require" and min(r for r, c in path) >= path[0][0]:
                continue
            # A true flexible transfer: the complete source cannot rigidly fit.
            if _canonical(path) == _canonical(destination):
                continue
            if any(order == path for order in _scan_orders(path)):
                continue
            return _normalize(path), op
    raise ValueError("bounded source search exhausted")


def _sequence(rng, path, k, colors, run_band):
    a, b = colors
    for attempt in range(60):
        run_count = int(rng.integers(run_band[0], min(run_band[1], len(path) - k + 1) + 1))
        sizes = np.ones(run_count - 1, dtype=int)
        spare = len(path) - k - len(sizes)
        if spare:
            alpha = float(rng.choice((0.2, 0.7, 2.0)))
            sizes += rng.multinomial(spare, rng.dirichlet(np.full(len(sizes), alpha)))
        sequence = [a] * k + [b if i % 2 == 0 else a
                              for i, size in enumerate(sizes) for _ in range(int(size))]
        color_at = dict(zip(path, sequence))
        if sequence == sequence[::-1]:
            continue
        if any([color_at[p] for p in order] == sequence for order in _scan_orders(path)):
            continue
        return sequence
    raise ValueError("could not distinguish path order from raster scans")


def _pack(rng, dimensions, area):
    """Place source and destination independently; no fixed pairing layout."""
    for attempt in range(18):
        mh = max(h for h, w in dimensions)
        mw = max(w for h, w in dimensions)
        height = 30 if attempt >= 12 else int(rng.integers(mh, 31))
        width = 30 if attempt >= 12 else int(rng.integers(mw, 31))
        if height * width <= 2 * area:
            continue
        first = int(rng.integers(2))
        h, w = dimensions[first]
        top, left = int(rng.integers(height - h + 1)), int(rng.integers(width - w + 1))
        if rng.random() < 0.4:
            if rng.random() < 0.5:
                top = int(rng.choice((0, height - h)))
            else:
                left = int(rng.choice((0, width - w)))
        other = 1 - first
        oh, ow = dimensions[other]
        options = [(r, c) for r in range(height - oh + 1) for c in range(width - ow + 1)
                   if r + oh < top or top + h < r or c + ow < left or left + w < c]
        if not options:
            continue
        if rng.random() < 0.4:
            edge = [p for p in options if p[0] in (0, height - oh) or p[1] in (0, width - ow)]
            if edge:
                options = edge
        origins = [None, None]
        origins[first] = (top, left)
        origins[other] = options[int(rng.integers(len(options)))]
        return (height, width), origins
    # Guaranteed side-by-side fit: widths <=14, maximum height <=18,
    # and target area <=169. The 30x30 canvas has a strict bg majority.
    return (30, 30), [(0, 0), (0, dimensions[0][1] + 1)]


def _extend(path, segments):
    result = list(path)
    for (dr, dc), count in segments:
        for _ in range(count):
            r, c = result[-1]
            result.append((r + dr, c + dc))
    return result


def _fallback_geometry(length_band, head_band, above_mode):
    """Explicit valid shapes for every evidence regime; used only after retries."""
    long = length_band[0] >= 28
    segments = [((0, 1), 2), ((1, 0), 3), ((0, 1), 2), ((-1, 0), 4),
                ((0, 1), 2), ((1, 0), 7 if long else 5)]
    if long:
        segments += [((0, 1), 2), ((-1, 0), 6), ((0, 1), 4)]
    else:
        segments += [((0, 1), 4)]
    channel = _extend([(5, 0)], segments)
    height, width = (13, 13) if long else (11, 11)
    target = {(r, c) for r in range(height) for c in range(width)} - set(channel)
    k = 9 if head_band[0] >= 9 else 4
    source_op = 3 if above_mode == "require" else 0
    head = [_transform(p, source_op) for p in channel[:k]]
    if k == 9:
        tail = ([((-1, 0), 2), ((0, 1), 3), ((1, 0), 7), ((0, -1), 2)]
                if above_mode == "require" else
                [((0, 1), 3), ((1, 0), 6), ((0, -1), 3), ((1, 0), 2)])
    elif long:
        tail = [((0, 1), 5), ((1, 0), 7), ((0, -1), 4), ((1, 0), 5), ((0, 1), 8)]
    else:
        tail = [((0, 1), 3), ((1, 0), 6), ((0, -1), 2), ((1, 0), 4), ((0, 1), 4)]
    source = _normalize(_extend(head, tail))
    assert len(source) == len(channel) and length_band[0] <= len(source) <= length_band[1]
    assert head_band[0] <= k <= head_band[1]
    assert _trace(set(source), source[0]) == source
    assert len(_components(target)) == 2 and _canonical(source) != _canonical(channel)
    return target, channel, k, source, source_op


def _fallback_sequence(path, k, colors, run_band):
    """Finite enumeration of run boundaries, independent of random luck."""
    a, b = colors
    count = run_band[0] - 1  # Number of tail runs, all of positive length.
    tail_length = len(path) - k
    for boundaries in combinations(range(1, tail_length), count - 1):
        cuts = (0,) + boundaries + (tail_length,)
        sizes = [y - x for x, y in zip(cuts, cuts[1:])]
        sequence = [a] * k + [b if i % 2 == 0 else a
                              for i, n in enumerate(sizes) for _ in range(n)]
        color_at = dict(zip(path, sequence))
        if sequence != sequence[::-1] and all([color_at[p] for p in order] != sequence
                                              for order in _scan_orders(path)):
            return sequence
    raise AssertionError("fallback has no distinguishing color partition")


def _make_pair(rng, palette, length_band, head_band, run_band, parity,
               above_mode, lower_destination=False, force_fallback=False):
    for attempt in (90,) if force_fallback else range(91):
        try:
            if attempt == 90:
                target, channel, k, source, source_op = _fallback_geometry(length_band, head_band, above_mode)
                sequence = _fallback_sequence(source, k, palette[2:], run_band)
            else:
                target, channel, k = _channel(rng, length_band, head_band, bool(rng.random() < 0.5))
                if above_mode == "require" and len(channel) - k < 8:
                    continue
                source, source_op = _source_path(rng, channel, k, above_mode)
                sequence = _sequence(rng, source, k, palette[2:], run_band)
            dest_parity = parity ^ int(source_op >= 4)
            options = list(range(4 * dest_parity, 4 * dest_parity + 4))
            if lower_destination:
                options = [op for op in options
                           if _transform(channel[0], op)[0] > _transform(channel[-1], op)[0]]
            dest_op = int(rng.choice(options))
            transformed_target = [_transform(p, dest_op) for p in sorted(target)]
            transformed_channel = [_transform(p, dest_op) for p in channel]
            r0, r1, c0, c1 = _bbox(transformed_target)
            fixed = [(r - r0, c - c0) for r, c in transformed_target]
            destination = [(r - r0, c - c0) for r, c in transformed_channel]
            sr0, sr1, sc0, sc1 = _bbox(source)
            dimensions = [(r1 - r0 + 1, c1 - c0 + 1), (sr1 - sr0 + 1, sc1 - sc0 + 1)]
            shape, origins = _pack(rng, dimensions, len(target) + len(channel))
            tr, tc = origins[0]
            sr, sc = origins[1]
            inp = np.full(shape, palette[0], dtype=np.int8)
            out = inp.copy()
            for r, c in fixed:
                inp[r + tr, c + tc] = out[r + tr, c + tc] = palette[1]
            for (r, c), color in zip(source, sequence):
                inp[r + sr, c + sc] = color
            a, b = palette[2:]
            final_sequence = [b if color == a else a for color in sequence] if parity else sequence
            for (r, c), color in zip(destination, final_sequence):
                out[r + tr, c + tc] = color
            # Trim blank margins sometimes, only if background stays majority.
            if rng.random() < 0.7:
                used = np.argwhere((inp != palette[0]) | (out != palette[0]))
                br0, br1, bc0, bc1 = _bbox(list(map(tuple, used)))
                br0 = max(0, br0 - int(rng.integers(4)))
                bc0 = max(0, bc0 - int(rng.integers(4)))
                br1 = min(shape[0] - 1, br1 + int(rng.integers(4)))
                bc1 = min(shape[1] - 1, bc1 + int(rng.integers(4)))
                if (br1 - br0 + 1) * (bc1 - bc0 + 1) > 2 * (len(target) + len(channel)):
                    inp = inp[br0:br1 + 1, bc0:bc1 + 1].copy()
                    out = out[br0:br1 + 1, bc0:bc1 + 1].copy()
            info = _analyse(inp)
            if info["reflected"] != bool(parity) or not np.array_equal(solve(inp), out):
                continue
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded ribbon construction exhausted")


def _array_key(grid):
    a = np.ascontiguousarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _distinct_pair(pair, seen):
    if _array_key(pair["input"]) not in seen:
        return pair
    # Repack the same valid construction and enumerate canvases. At worst the
    # footprints are 18x14 and 14x14: at least 13*2=26 choices exceed the
    # maximum 13 already-used inputs. This does not alter the transformation.
    info = _analyse(pair["input"])
    tr0, tr1, tc0, tc1 = _bbox(info["fixed"])
    sr0, sr1, sc0, sc1 = _bbox(info["source"])
    th, tw = tr1 - tr0 + 1, tc1 - tc0 + 1
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    for height in range(max(th, sh), 31):
        for width in range(tw + sw + 1, 31):
            if height * width <= 2 * len(info["rectangle"]):
                continue
            inp = np.full((height, width), info["background"], dtype=np.int8)
            for r, c in info["fixed"]:
                inp[r - tr0, c - tc0] = info["target_color"]
            for (r, c), color in zip(info["source"], info["sequence"]):
                inp[r - sr0, c - sc0 + tw + 1] = color
            if _array_key(inp) not in seen:
                out = np.full_like(inp, info["background"])
                for r, c in info["rectangle"]:
                    out[r - tr0, c - tc0] = pair["output"][r, c]
                return {"input": inp, "output": out}
    raise AssertionError("distinct-canvas enumeration exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("counts must satisfy 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(("length", "head", "recurrence")))
    stable_palette = bool(rng.random() < 0.35)
    palette = [int(c) for c in rng.permutation(10)[:4]]
    allow_above = bool(rng.random() < 0.35)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            test = split == "test"
            length_band = (28, 42) if test and mode == "length" else (14, 26) if mode == "length" else (20, 36)
            head_band = (9, 12) if test and mode == "head" else (4, 7) if mode == "head" else (4, 12)
            run_band = (8, 14) if test and mode == "recurrence" else (4, 6) if mode == "recurrence" else (4, 11)
            parity = index if not test and index < 2 else 1 if test and index == 0 else int(rng.integers(2))
            above_mode = "require" if allow_above and not test and index == 1 else "allow" if allow_above else "forbid"
            local_palette = palette if stable_palette else [int(c) for c in rng.permutation(10)[:4]]
            pair = _make_pair(rng, local_palette, length_band, head_band, run_band, parity,
                              above_mode, lower_destination=not test and index < 2)
            pair = _distinct_pair(pair, seen)
            seen.add(_array_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    seq, source, destination = info["sequence"], info["source"], info["destination"]
    cuts = [0] + [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]] + [len(seq)]
    lengths = tuple(b - a for a, b in zip(cuts, cuts[1:]))
    def turns(path):
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        return sum(a != b for a, b in zip(steps, steps[1:]))
    return {"length": len(source), "head": info["head_length"], "runs": len(lengths),
            "run_lengths": lengths, "source_turns": turns(source), "destination_turns": turns(destination),
            "source_shape": _canonical(source), "destination_shape": _canonical(destination),
            "head_shape": _canonical(source[:info["head_length"]]),
            "above": min(r for r, c in source) < source[0][0],
            "lower_destination": destination[0][0] > destination[-1][0],
            "operation": info["operation"], "reflected": info["reflected"],
            "shape": info["input"].shape, "background": info["background"]}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            info = _analyse(inp)
            assert inp.shape == out.shape and np.array_equal(solve(inp), out)
            assert _array_key(inp) not in seen
            seen.add(_array_key(inp))
            infos[split].append(info)
            # Blank channel has exactly two fixed connected fragments, and
            # reweaving neither overwrites them nor leaves any channel cell empty.
            assert len(_components(info["fixed"])) == 2
            assert all(out[p] == inp[p] for p in info["fixed"])
            assert all(out[p] == info["background"] for p in info["source"])
            assert set(map(tuple, np.argwhere(out != info["background"]))) == info["rectangle"]
            assert np.count_nonzero(inp != info["background"]) == np.count_nonzero(out != info["background"])
            assert _canonical(info["source"]) != _canonical(info["slot"]), "tail must need reweaving"
            seq, source = info["sequence"], info["source"]
            color_at = dict(zip(source, seq))
            assert seq != seq[::-1], "demonstrate traversal direction"
            assert all([color_at[p] for p in order] != seq for order in _scan_orders(source)), "raster shortcut"
            # Removing A: global color counts do not recover the sequence.
            grouped = [seq[0]] * seq.count(seq[0]) + [seq[info["head_length"]]] * seq.count(seq[info["head_length"]])
            assert grouped != seq
    train = infos["train"]
    assert {i["reflected"] for i in train} == {False, True}, "show both parity rules"
    assert any(i["destination"][0][0] > i["destination"][-1][0] for i in train), "head chooses the lower end too"
    # Removing B's parity contribution changes actual labeled answers.
    assert any(i["reflected"] and not np.array_equal(_paint(i, i["sequence"]), solve(i["input"])) for i in train)
    if any(_features(i)["above"] for i in infos["test"]):
        assert any(_features(i)["above"] for i in train), "teach upper endpoint versus topmost interior"
    trf, tef = [[_features(i) for i in infos[s]] for s in ("train", "test")]
    # At least one intentional visible extrapolation, independent of a latent mode name.
    assert (min(f["length"] for f in tef) > max(f["length"] for f in trf)
            or min(f["head"] for f in tef) > max(f["head"] for f in trf)
            or min(f["runs"] for f in tef) > max(f["runs"] for f in trf))


def self_test():
    """Bounded 42-seed invariants, variation measurements, and negative checks."""
    shapes, sources, destinations, heads, operations, backgrounds = (set() for _ in range(6))
    lengths, head_lengths, run_counts, source_turns, destination_turns, puzzles = (set() for _ in range(6))
    above = lower = singleton = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(tuple(_array_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyse(pair["input"])
                f = _features(info)
                shapes.add(f["shape"])
                sources.add(f["source_shape"])
                destinations.add(f["destination_shape"])
                heads.add(f["head_shape"])
                operations.add(f["operation"])
                backgrounds.add(f["background"])
                lengths.add(f["length"])
                head_lengths.add(f["head"])
                run_counts.add(f["runs"])
                source_turns.add(f["source_turns"])
                destination_turns.add(f["destination_turns"])
                above += f["above"]
                lower += f["lower_destination"]
                singleton += 1 in f["run_lengths"]
                # Reading back along the channel recovers precisely the source
                # sequence, with only the required reflection-local exchange.
                readback = [int(pair["output"][p]) for p in info["destination"]]
                if info["reflected"]:
                    a, b = info["colors"]
                    readback = [b if c == a else a for c in readback]
                assert readback == info["sequence"]
    assert len(puzzles) == 42 and len(shapes) >= 50
    assert len(sources) >= 150 and len(destinations) >= 100 and len(heads) >= 25
    assert operations == set(range(8)) and backgrounds == set(range(10))
    assert len(lengths) >= 20 and len(head_lengths) >= 8 and len(run_counts) >= 10
    assert len(source_turns) >= 12 and len(destination_turns) >= 8
    assert above >= 8 and lower >= 80 and singleton >= 100
    p = generate_puzzle(73, 5, 2)
    q = generate_puzzle(73, 5, 2)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    assert _array_key(p["test"][0]["input"]) != _array_key(generate_puzzle(74, 5, 2)["test"][0]["input"])
    # Reordering demonstrations and changing array representation carry no rule information.
    lists = {s: [{k: v.tolist() for k, v in x.items()} for x in reversed(p[s])] for s in p}
    validate_generated_puzzle(lists)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            t = generate_puzzle(317 + nt * 7 + ne, nt, ne)
            assert len(t["train"]) == nt and len(t["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                   (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    info = _analyse(damaged["test"][0]["input"])
    point = info["destination"][-1]
    damaged["test"][0]["output"][point] = info["background"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing channel cell accepted")
    # Branches, disconnected chains, and cycles are invalid.
    for cells in ([(0, 1), (1, 0), (1, 1), (1, 2)],
                  [(0, 0), (1, 0), (4, 4), (5, 4)],
                  [(0, 0), (0, 1), (1, 0), (1, 1)]):
        try:
            _trace(cells)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed path accepted")
    # Straight channels are legal, but a straight head cannot decide parity.
    # These fixtures isolate an ambiguous head and the forbidden source row tie.
    ambiguous = np.zeros((13, 19), dtype=np.int8)
    ambiguous[2, 1:9] = ambiguous[4, 1:9] = 7
    ribbon = [(7, 11), (7, 12), (7, 13), (7, 14),
              (8, 14), (9, 14), (10, 14), (10, 15)]
    for point, color in zip(ribbon, [2] * 4 + [3] * 2 + [2] * 2):
        ambiguous[point] = color
    tied = ambiguous.copy()
    for point in ribbon:
        tied[point] = 0
    tied[9, 10:14] = 2
    tied[9, 14:18] = 3
    for grid, expected in ((ambiguous, "uniquely determine"), (tied, "cannot tie")):
        try:
            solve(grid)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError("ambiguous or tied source accepted")
    # Hand-constructed positives outside the sampler's length bands. The
    # first has tied channel endpoint rows; the second has a symmetric head
    # whose ordered endpoint correspondence still fixes reflection parity.
    fixtures = [
        (5, [(1, 0), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3),
             (3, 4), (3, 5), (2, 5), (1, 5), (1, 6), (1, 7)],
         [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3),
          (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 6)]),
        (6, [(1, 0), (1, 1), (1, 2), (2, 2), (3, 2), (4, 2),
             (4, 3), (4, 4), (3, 4), (2, 4), (2, 5), (2, 6), (2, 7)],
         [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2),
          (4, 2), (4, 3), (4, 4), (5, 4), (6, 4), (6, 5), (6, 6)])]
    for height, channel, ribbon in fixtures:
        seq = [2] * 5 + [8, 8, 2] + [8] * (len(ribbon) - 9) + [2]
        for mirror in (False, True):
            dest = [(r, 7 - c if mirror else c) for r, c in channel]
            fixed = {(r, c) for r in range(height) for c in range(8)} - set(dest)
            inp = np.zeros((20, 23), dtype=np.int8)
            out = inp.copy()
            for p0 in fixed:
                inp[p0] = out[p0] = 6
            for (r, c), color in zip(ribbon, seq):
                inp[r + 8, c + 12] = color
            for p0, color in zip(dest, seq):
                out[p0] = 10 - color if mirror else color
            original = inp.copy()
            assert np.array_equal(solve(inp), out) and np.array_equal(inp, original)
            assert _analyse(inp)["reflected"] == mirror
    # Exercise every constructive fallback combination directly, rather than
    # hoping a random seed happens to exhaust the primary sampler.
    rng = np.random.default_rng(771)
    regimes = [((14, 26), (4, 12), (4, 11)), ((28, 42), (4, 12), (4, 11)),
               ((20, 36), (4, 7), (4, 11)), ((20, 36), (9, 12), (4, 11)),
               ((20, 36), (4, 12), (4, 6)), ((20, 36), (4, 12), (8, 14))]
    for lb, hb, rb in regimes:
        for above_mode in ("require", "allow", "forbid"):
            for parity in (0, 1):
                pair = _make_pair(rng, [0, 6, 2, 8], lb, hb, rb, parity, above_mode,
                                  lower_destination=True, force_fallback=True)
                f = _features(_analyse(pair["input"]))
                assert lb[0] <= f["length"] <= lb[1] and hb[0] <= f["head"] <= hb[1]
                assert rb[0] <= f["runs"] <= rb[1] and f["reflected"] == bool(parity)
                assert f["lower_destination"] and (above_mode != "require" or f["above"])
    original = p["train"][0]
    seen = {_array_key(original["input"])}
    for _ in range(13):
        distinct = _distinct_pair(original, seen)
        assert _array_key(distinct["input"]) not in seen
        assert np.array_equal(solve(distinct["input"]), distinct["output"])
        seen.add(_array_key(distinct["input"]))
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "source_shapes_up_to_D4": len(sources), "channel_shapes_up_to_D4": len(destinations),
            "head_shapes_up_to_D4": len(heads), "lengths": sorted(lengths),
            "head_lengths": sorted(head_lengths), "run_counts": sorted(run_counts),
            "operations": sorted(operations), "above_endpoint_examples": above,
            "lower_end_docks": lower, "singleton_run_examples": singleton,
            "positive_extension_fixtures": 4, "constructive_fallback_cases": 36}
