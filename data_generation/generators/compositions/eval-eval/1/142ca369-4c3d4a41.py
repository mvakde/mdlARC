"""A comb clips ordered pins; L roof ports route the clipped cells as finite tapes.

All meanings are visible in the input. No parent generator, stored pair, seed,
or construction parameter is consulted by solve(). Standard library + NumPy.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _need(condition, message="invalid loom"):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid size")
    _need(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _need(np.all((a >= 0) & (a <= 9)), "grid colors")
    return a.astype(int, copy=True)


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (np.fliplr(b) if code >= 4 else b).copy()


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _port_cells(f, i):
    t, c = f["top"], f["lanes"][i]
    dc, _ = f["ports"][i]
    return ((t - 2, c), (t - 1, c), (t - 2, c - dc))


def _mirror_cells(m):
    r, c, axis, length = m
    rad = length // 2
    return tuple((r + (j if axis == 1 else 0),
                  c + (j if axis == 0 else 0)) for j in range(-rad, rad + 1))


def _comb_cells(f, inserted=False):
    b = f["base"]
    columns = f["lanes"] if inserted else f["teeth"]
    cells = [(b, c) for c in range(columns[0], columns[-1] + 1)]
    cells += [(r, c) for c, h in zip(columns, f["heights"])
              for r in range(b - h, b)]
    return cells


def _put(a, cells, color, bg):
    for r, c in cells:
        _need(0 <= r < a.shape[0] and 0 <= c < a.shape[1], "object outside canvas")
        _need(a[r, c] == bg, "overlapping input objects")
        a[r, c] = color


def _draw_input(f):
    a = np.full(f["shape"], f["bg"], dtype=int)
    t, b, bottom = f["top"], f["base"], f["bottom"]
    l, r, s, bg = f["left"], f["right"], f["structure"], f["bg"]
    a[t, l:r + 1] = a[bottom, l:r + 1] = s
    a[t:bottom + 1, l] = a[t:bottom + 1, r] = s
    a[b - 1:b + 1, l] = bg
    _put(a, _comb_cells(f), s, bg)
    for c, seq in zip(f["lanes"], f["stacks"]):
        for j, v in enumerate(seq, t + 1):
            _put(a, [(j, c)], v, bg)
    for i, (_, color) in enumerate(f["ports"]):
        _put(a, _port_cells(f, i), color, bg)
    for m in f["mirrors"]:
        _put(a, _mirror_cells(m), s, bg)
    return a


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], []
        while todo:
            r, c = todo.pop()
            component.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(component)
    return result


def _wall_candidates(a, bg):
    runs = {}
    for row, values in enumerate(a):
        ends = [0, *(np.flatnonzero(values[1:] != values[:-1]) + 1).tolist(), len(values)]
        for l, stop in zip(ends, ends[1:]):
            s = int(values[l])
            if s != bg and stop - l >= 7:
                runs.setdefault((s, l, stop - 1), []).append(row)
    return [(s, l, r, t, bottom) for (s, l, r), rows in runs.items()
            for t in rows for bottom in rows if bottom > t]


def _read_canonical(a, bg, wall):
    s, l, r, t, bottom = wall
    _need(t >= 3 and np.all(a[t:bottom + 1, r] == s))
    side = a[t + 1:bottom, l]
    _need(np.all((side == s) | (side == bg)))
    opening = np.flatnonzero(side == bg) + t + 1
    _need(len(opening) == 2 and opening[1] == opening[0] + 1)
    b = int(opening[1])
    d = b - t
    _need(d >= 3 and bottom - b >= 2)
    exterior = np.flatnonzero(a[b, :l] != bg)
    _need(len(exterior) >= 3)
    kl, kr = int(exterior[0]), int(exterior[-1])
    _need(kr < l - 1 and len(exterior) == kr - kl + 1)
    _need(np.all(a[b, kl:kr + 1] == s))
    teeth = tuple(int(c) for c in np.flatnonzero(a[b - 1, kl:kr + 1] == s) + kl)
    _need(2 <= len(teeth) <= 5 and teeth[0] == kl and teeth[-1] == kr)
    steps = np.diff(teeth)
    _need(steps[0] >= 2 and np.all(steps == steps[0]))
    heights = []
    for c in teeth:
        row = b - 1
        while row >= 0 and a[row, c] == s:
            row -= 1
        heights.append(b - row - 1)
    _need(all(1 <= h <= d - 2 for h in heights))
    lanes = tuple(int(c) for c in np.flatnonzero(a[t + 1, l + 1:r] != bg) + l + 1)
    _need(len(lanes) == len(teeth) and lanes[0] >= l + 2 and lanes[-1] <= r - 2)
    _need(len(set(c - k for c, k in zip(lanes, teeth))) == 1)
    stacks = []
    for c, h in zip(lanes, heights):
        segment = a[t + 1:b, c]
        indices = np.flatnonzero(segment != bg)
        _need(len(indices) and np.array_equal(indices, np.arange(len(indices))))
        seq = tuple(int(v) for v in segment[:len(indices)])
        prefix = next((j for j, v in enumerate(seq) if v != s), len(seq))
        _need(prefix < len(seq) and s not in seq[prefix:])
        _need(d - h - 1 <= len(seq) <= d - 1)
        stacks.append(seq)
    ports, port_cells = [], set()
    for c in lanes:
        color = int(a[t - 2, c])
        _need(color not in (bg, s) and a[t - 1, c] == color)
        signs = [dc for dc in (-1, 1) if 0 <= c - dc < a.shape[1]
                 and a[t - 2, c - dc] == color]
        _need(len(signs) == 1, "port must have one missing corner")
        dc = signs[0]
        cells = {(t - 2, c), (t - 1, c), (t - 2, c - dc)}
        _need(not cells & port_cells)
        port_cells.update(cells)
        ports.append((dc, color))
    upper = {tuple(int(x) for x in p) for p in np.argwhere(a[:t] != bg)} - port_cells
    _need(all(a[p] == s for p in upper), "unknown exterior objects")
    mirrors = []
    for component in _components(upper):
        rows, cols = zip(*component)
        length = len(component)
        _need(length % 2 == 1 and (len(set(rows)) == 1 or len(set(cols)) == 1))
        _need(max(max(rows) - min(rows), max(cols) - min(cols)) + 1 == length)
        _need(max(rows) <= t - 2)
        axis = 0 if len(set(rows)) == 1 else 1
        mirrors.append(((min(rows) + max(rows)) // 2,
                        (min(cols) + max(cols)) // 2, axis, length))
    f = dict(shape=a.shape, bg=bg, structure=s, top=t, base=b, bottom=bottom,
             left=l, right=r, teeth=teeth, lanes=lanes, heights=tuple(heights),
             stacks=tuple(stacks), ports=tuple(ports), mirrors=tuple(sorted(mirrors)))
    _need(np.array_equal(a, _draw_input(f)), "unexplained input pixels")
    return f


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _need(counts[bg] > a.size / 2, "background must be a unique majority")
    readings = []
    for code in range(8):
        canonical = _unorient(a, code)
        for wall in _wall_candidates(canonical, bg):
            try:
                f = _read_canonical(canonical, bg, wall)
            except ValueError:
                continue
            f["orientation"] = code
            readings.append(f)
    _need(len(readings) == 1, "scene must have one geometric interpretation")
    return readings[0]


def _contact(m, position, direction):
    r, c, axis, length = m
    row, col = position
    dr, dc = direction
    if (length == 1 or axis == 0) and col == c and r - row == dr:
        return 0
    if (length == 1 or axis == 1) and row == r and c - col == dc:
        return 1
    return None


def _trace(f, lengths, a=None):
    """Trace finite routes, including contact cells in the packet length."""
    a = _draw_input(f) if a is None else a
    paths, events, used_cells = [], [], set()
    for i, length in enumerate(lengths):
        position = (f["top"] - 2, f["lanes"][i])
        direction = (-1, f["ports"][i][0])
        path, contacts, states = [], [], set()
        for step in range(1, int(length) + 1):
            position = (position[0] + direction[0], position[1] + direction[1])
            row, col = position
            _need(0 <= row < f["top"] and 0 <= col < a.shape[1], "packet must fit above roof")
            _need(a[position] == f["bg"], "tape struck an object")
            state = position, direction
            _need(state not in states, "route loop")
            states.add(state)
            _need(position not in used_cells, "tapes must not overlap")
            used_cells.add(position)
            hits = [(j, axis) for j, m in enumerate(f["mirrors"])
                    if (axis := _contact(m, position, direction)) is not None]
            _need(len(hits) <= 1, "competing mirror contacts")
            if hits:
                j, axis = hits[0]
                direction = (-direction[0], direction[1]) if axis == 0 else (direction[0], -direction[1])
                contacts.append((j, axis, step, position, direction))
            path.append(position)
        paths.append(tuple(path))
        events.append(tuple(contacts))
    return tuple(paths), tuple(events)


def _shifts(f):
    return tuple(len(seq) - (f["base"] - f["top"] - h - 1)
                 for seq, h in zip(f["stacks"], f["heights"]))


def _settled(f, a):
    result = a.copy()
    for p in _comb_cells(f):
        result[p] = f["bg"]
    for c in f["lanes"]:
        result[f["top"] + 1:f["base"], c] = f["bg"]
    for p in _comb_cells(f, inserted=True):
        _need(result[p] == f["bg"], "key collides with unrelated object")
        result[p] = f["structure"]
    for c, seq, shift in zip(f["lanes"], f["stacks"], _shifts(f)):
        survivor = seq[shift:]
        _need(len(survivor) >= 1)
        result[f["top"] + 1:f["top"] + 1 + len(survivor), c] = survivor
    return result


def _answer(f, a=None):
    a = _draw_input(f) if a is None else a
    lengths = _shifts(f)
    paths, contacts = _trace(f, lengths, a)
    result = _settled(f, a)
    for seq, shift, path in zip(f["stacks"], lengths, paths):
        for p, v in zip(path, seq[:shift]):
            result[p] = v
    return result, paths, contacts


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    f = _parse(a)
    canonical = _unorient(a, f["orientation"])
    result, _, _ = _answer(f, canonical)
    return _orient(result, f["orientation"])


def _try_mirror(f, capacities, rng, lane, wanted_axis=None):
    """Append a reachable mirror and validate every longest planned packet."""
    a = _draw_input(f)
    paths, events = _trace(f, capacities, a)
    direction = (-1, f["ports"][lane][0])
    directions = []
    events_at = {e[2]: e[4] for e in events[lane]}
    for step in range(1, capacities[lane] + 1):
        directions.append(direction)
        if step in events_at:
            direction = events_at[step]
    last = events[lane][-1][2] if events[lane] else 0
    options = [(step, axis, length) for step in range(max(2, last + 1), capacities[lane])
               for axis in ([wanted_axis] if wanted_axis is not None else [0, 1])
               for length in (1, 3, 5)]
    rng.shuffle(options)
    for step, axis, length in options:
        point, direction = paths[lane][step - 1], directions[step - 1]
        mid = (point[0] + (direction[0] if axis == 0 else 0),
               point[1] + (direction[1] if axis == 1 else 0), axis, length)
        cells = set(_mirror_cells(mid))
        if not all(0 <= r <= f["top"] - 2 and 0 <= c < a.shape[1]
                   and a[r, c] == f["bg"] for r, c in cells):
            continue
        # Distinct structural bars remain distinct components.
        if any(0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]
               and (rr, cc) not in cells and a[rr, cc] == f["structure"]
               for r, c in cells for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))):
            continue
        trial = dict(f, mirrors=f["mirrors"] + (mid,))
        try:
            _, after = _trace(trial, capacities)
        except ValueError:
            continue
        before_signature = [(f["mirrors"][e[0]][:2], e[1], e[2]) for e in events[lane]]
        after_signature = [(trial["mirrors"][e[0]][:2], e[1], e[2]) for e in after[lane]]
        preserved = True
        for i in range(len(capacities)):
            if i == lane:
                continue
            old = [(f["mirrors"][e[0]][:2], e[1], e[2]) for e in events[i]]
            new = [(trial["mirrors"][e[0]][:2], e[1], e[2]) for e in after[i]]
            if new[:len(old)] != old:
                preserved = False
                break
        if preserved and after_signature == before_signature + [(mid[:2], axis, step)]:
            return trial
    return None


def _sequence(rng, length, structure, available, packet, teaching):
    prefix = int(rng.integers(0, min(3, length - 1) + 1))
    if teaching:
        prefix = int(rng.integers(1, min(2, packet - 2) + 1))
    palette = rng.choice(available, int(rng.integers(1, min(4, len(available)) + 1)), replace=False)
    if teaching and len(palette) == 1:
        palette = rng.choice(available, 2, replace=False)
    payload = [int(v) for v in rng.choice(palette, length - prefix)]
    if teaching:
        payload[0], payload[1] = int(palette[0]), int(palette[1])
    seq = (structure,) * prefix + tuple(payload)
    if teaching and seq[:packet] == seq[:packet][::-1]:
        seq = seq[:packet - 1] + (int(palette[1]),) + seq[packet:]
    return seq


def _hardware(rng, n, packet, colors, teaching=False):
    """Independent apparatus construction; finite route geometry is shared."""
    bg, structure = colors
    available = [v for v in range(10) if v not in colors]
    for _ in range(160):
        spacing = int(rng.choice([2, 3, 4] if n == 2 else [2, 3] if n == 3 else [2]))
        kw = spacing * (n - 1) + 1
        kl, gap, il, ir = (int(rng.integers(0, 4)), int(rng.integers(1, 6)),
                            int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        left, first = kl + kw + gap, kl + kw + gap + il
        right = first + kw - 1 + ir
        if right >= 30 or first < packet + 1:
            continue
        width = right + 1 + int(rng.integers(0, min(3, 29 - right) + 1))
        depth = int(rng.integers(packet + 5, min(14, packet + 7) + 1))
        roof_max = min(packet + 6, 27 - depth)
        if roof_max < packet + 3:
            continue
        top = int(rng.integers(packet + 3, roof_max + 1))
        base = top + depth
        bottom = base + int(rng.integers(2, min(3, 29 - base) + 1))
        height = bottom + 1 + int(rng.integers(0, min(2, 29 - bottom) + 1))
        q = rng.integers(1, min(3, depth - packet - 2) + 1, n).tolist()
        if len(set(q)) == 1:
            q[1] = 1 if q[0] != 1 else 2
        lanes = tuple(first + spacing * i for i in range(n))
        port_colors = []
        for i in range(n):
            choices = [v for v in available if not (spacing == 2 and i and v == port_colors[-1])]
            port_colors.append(int(rng.choice(choices)))
        # Leftward routes have generous space above the external key. Try
        # mixed signs too, accepting only complete nonintersecting packets.
        signs = [-1] * n
        if rng.random() < .35:
            signs = [int(rng.choice([-1, 1])) for _ in range(n)]
        seqs = tuple(_sequence(rng, depth - x - 1, structure, available,
                               packet, teaching) for x in q)
        f = dict(shape=(height, width), bg=bg, structure=structure, top=top,
                 base=base, bottom=bottom, left=left, right=right,
                 teeth=tuple(kl + spacing * i for i in range(n)), lanes=lanes,
                 heights=tuple(q), stacks=seqs,
                 ports=tuple(zip(signs, port_colors)), mirrors=())
        capacities = (packet,) * n
        try:
            a = _draw_input(f)
            _trace(f, capacities, a)
            _read_canonical(a, bg, (structure, left, right, top, bottom))
            _need(np.count_nonzero(a == bg) > a.size / 2)
        except ValueError:
            continue
        if teaching:
            first_mirror = _try_mirror(f, capacities, rng, 0, 0)
            if first_mirror is None:
                continue
            second_mirror = _try_mirror(first_mirror, capacities, rng, n - 1, 1)
            if second_mirror is None:
                continue
            f = second_mirror
        else:
            for _ in range(int(rng.integers(0, 4))):
                trial = _try_mirror(f, capacities, rng, int(rng.integers(n)))
                if trial is not None:
                    f = trial
        for _ in range(int(rng.integers(0, 3)) if teaching else 0):
            trial = _try_mirror(f, capacities, rng, int(rng.integers(n)))
            if trial is not None:
                f = trial
        a = _draw_input(f)
        if np.count_nonzero(a == bg) <= a.size / 2:
            continue
        # Parsing isn't weakened to accommodate any construction artifact.
        try:
            _read_canonical(a, bg, (structure, left, right, top, bottom))
        except ValueError:
            continue
        return f, capacities
    # Clear, deterministic geometry also provides a bounded fallback for a
    # rare unlucky search. Two point mirrors show the two normal reversals.
    spacing, kl, gap, inset = 2, 1, 3, 3
    kw = spacing * (n - 1) + 1
    first = max(kl + kw + gap + inset, packet + 1)
    left = kl + kw + gap
    right = first + kw - 1 + 2
    top, depth = packet + 3, packet + 5
    q = tuple(1 + i % 2 for i in range(n))
    lanes = tuple(first + spacing * i for i in range(n))
    f = dict(shape=(top + depth + 3, right + 1), bg=bg, structure=structure,
             top=top, base=top + depth, bottom=top + depth + 2,
             left=left, right=right, teeth=tuple(kl + spacing * i for i in range(n)),
             lanes=lanes, heights=q,
             stacks=tuple(_sequence(rng, depth - x - 1, structure, available, packet, True) for x in q),
             ports=tuple((-1, available[i % len(available)]) for i in range(n)), mirrors=())
    if teaching:
        r0 = top - 2
        f["mirrors"] = ((r0 - packet, first - packet + 1, 0, 1),
                        (r0 - 2, lanes[-1] - 3, 0, 1))
    _trace(f, (packet,) * n)
    _need(all(d <= 30 for d in f["shape"]), "fallback bounds")
    return f, (packet,) * n


def _pair(f, code):
    a = _draw_input(f)
    result, _, _ = _answer(f, a)
    return {"input": _orient(a, code), "output": _orient(result, code)}


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _unique_fallback(f, code, seen):
    """64 distinct last-payload assignments exceed the 14-pair upper bound."""
    colors = [v for v in range(10) if v not in (f["bg"], f["structure"])]
    for a in colors:
        for b in colors:
            stacks = list(f["stacks"])
            for i, color in enumerate((a, b)):
                stacks[i] = stacks[i][:-1] + (color,)
            pair = _pair(dict(f, stacks=tuple(stacks)), code)
            if _key(pair["input"]) not in seen:
                return pair
    raise AssertionError("puzzle exceeds legal pair bound")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["lanes", "length", "flat"]))
    fixed_colors = tuple(int(v) for v in rng.choice(10, 2, replace=False))
    changing_colors = bool(rng.integers(2))
    orientation = int(rng.integers(8))
    mixed_orientations = bool(rng.integers(2))
    demo_n = int(rng.integers(2, 4))
    demo_k = int(rng.integers(4, 6 if regime == "length" else 7))
    hardware, caps = _hardware(rng, demo_n, demo_k, fixed_colors, teaching=True)
    q = hardware["heights"]
    result = {"train": [], "test": []}
    for swap in (0, 1):
        heights = tuple(x + (caps[i] if (i + swap) % 2 else 0) for i, x in enumerate(q))
        result["train"].append(_pair(dict(hardware, heights=heights), orientation))
    seen = {_key(p["input"]) for p in result["train"]}
    for split, count in (("train", ntrain), ("test", ntest)):
        while len(result[split]) < count:
            for attempt in range(80):
                colors = tuple(int(v) for v in rng.choice(10, 2, replace=False)) if changing_colors else fixed_colors
                if regime == "lanes":
                    n = int(rng.integers(2, 4) if split == "train" else rng.integers(4, 6))
                else:
                    n = int(rng.integers(2, 6))
                packet = int(rng.integers(4, 6 if regime == "length" and split == "train" else 7))
                if regime == "length" and split == "test":
                    packet = int(rng.integers(6, 9))
                f, caps = _hardware(rng, n, packet, colors)
                q = f["heights"]
                if regime == "flat" and split == "test":
                    height = int(rng.integers(max(q) + 1, min(x + k for x, k in zip(q, caps)) + 1))
                    heights = (height,) * n
                else:
                    shifts = rng.integers(0, packet + 1, n).tolist()
                    shifts[int(rng.integers(n))] = packet
                    heights = tuple(x + shift for x, shift in zip(q, shifts))
                    if regime == "flat" and split == "train" and len(set(heights)) == 1:
                        continue
                f["heights"] = heights
                code = int(rng.integers(8)) if mixed_orientations else orientation
                pair = _pair(f, code)
                if _key(pair["input"]) in seen:
                    continue
                seen.add(_key(pair["input"]))
                result[split].append(pair)
                break
            else:
                pair = _unique_fallback(f, code, seen)
                seen.add(_key(pair["input"]))
                result[split].append(pair)
    rng.shuffle(result["train"])
    validate_generated_puzzle(result)
    return result


def _features(a):
    f = _parse(a)
    canonical = _unorient(a, f["orientation"])
    lengths = _shifts(f)
    _, paths, events = _answer(f, canonical)
    changed_route = False
    for i, path in enumerate(paths):
        dc = f["ports"][i][0]
        straight = tuple((f["top"] - 2 - j, f["lanes"][i] + j * dc)
                         for j in range(1, len(path) + 1))
        changed_route |= set(path) != set(straight)
    prefixes = [seq[:shift] for seq, shift in zip(f["stacks"], lengths)]
    return dict(f=f, lengths=lengths, paths=paths, events=events,
                axes={e[1] for ev in events for e in ev}, changed_route=changed_route,
                ordered=any(len(set(p)) >= 2 and p != p[::-1] for p in prefixes),
                multicolor=any(len({v for v in p if v != f["structure"]}) >= 2 for p in prefixes),
                structural_loss=any(f["structure"] in p for p in prefixes))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _features(a)
            f = info["f"]
            canonical, canonical_out = _unorient(a, f["orientation"]), _unorient(out, f["orientation"])
            expected, paths, _ = _answer(f, canonical)
            assert np.array_equal(expected, canonical_out), "wrong settled hardware or overflow tape"
            assert np.array_equal(np.bincount(a.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10)), "colored material not conserved"
            assert not np.array_equal(a, out)
            assert all(canonical_out[p] == f["structure"] for p in _comb_cells(f, True))
            assert all(canonical_out[p] == f["bg"] for p in _comb_cells(f))
            for i, (c, seq, shift, path) in enumerate(zip(f["lanes"], f["stacks"], info["lengths"], paths)):
                assert len(path) == shift and shift >= 0
                assert tuple(canonical_out[p] for p in path) == seq[:shift]
                survivor = tuple(int(v) for v in canonical_out[f["top"] + 1:f["base"] - f["heights"][i], c])
                assert survivor == seq[shift:]
            for i in range(len(f["ports"])):
                assert all(canonical_out[p] == canonical[p] for p in _port_cells(f, i))
            assert all(canonical_out[p] == canonical[p] for m in f["mirrors"] for p in _mirror_cells(m))
            if split == "train":
                training.append(info)
    visible_axes = {e[1] for x in training for ev, length in zip(x["events"], x["lengths"])
                    for e in ev if e[2] < length}
    assert visible_axes == {0, 1}, "both faces need a visible post-contact segment"
    assert any(x["changed_route"] for x in training), "A-omitting straight export must differ"
    assert any(x["ordered"] and x["multicolor"] for x in training), "ordered payload evidence"
    assert any(x["structural_loss"] for x in training), "structural prefix must be routed"
    assert any(0 in x["lengths"] and max(x["lengths"]) > 0 for x in training), "stationary/moving witness"
    assert any(sum(x["lengths"]) != sum(x["f"]["heights"]) for x in training), "pin length must affect clipping"
    comparisons = {}
    for x in training:
        f = x["f"]
        apparatus = tuple((k, f[k]) for k in sorted(f) if k != "heights")
        comparisons.setdefault(apparatus, set()).add(f["heights"])
    assert any(len(profiles) >= 2 for profiles in comparisons.values()), "missing shared-hardware comparison"


def _puzzle_signature(puzzle):
    compact = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    return hashlib.sha256(json.dumps(compact, separators=(",", ":")).encode()).hexdigest()


def _structural_signature(a):
    roles, encoded = {}, []
    for v in a.flat:
        encoded.append(roles.setdefault(int(v), len(roles)))
    return a.shape, bytes(encoded)


def _regressions():
    # Literal geometry with a three-cell horizontal mirror and a point mirror.
    # Roof=9, base=18; lengths 7/6, tooth heights 5/5 => shifts 4/3.
    f = dict(shape=(21, 24), bg=0, structure=5, top=9, base=18, bottom=20,
             left=9, right=23, teeth=(1, 5), lanes=(12, 16), heights=(5, 5),
             stacks=((5, 1, 2, 3, 4, 4, 4), (5, 6, 7, 8, 8, 8)),
             ports=((-1, 9), (-1, 1)), mirrors=((3, 9, 0, 3), (5, 13, 0, 1)))
    a = _draw_input(f)
    # Fully specify the expected answer without _answer/_settled/_trace.
    expected = a.copy()
    expected[13:18, 1] = expected[13:18, 5] = 0
    expected[18, 1:6] = 0
    expected[10:18, 12] = expected[10:18, 16] = 0
    expected[18, 12:17] = 5
    expected[13:18, 12] = expected[13:18, 16] = 5
    expected[10:13, 12] = (4, 4, 4)
    expected[10:13, 16] = (8, 8, 8)
    for p, color in (((6, 11), 5), ((5, 10), 1), ((4, 9), 2), ((5, 8), 3),
                     ((6, 15), 5), ((5, 14), 6), ((4, 15), 7)):
        expected[p] = color
    assert np.array_equal(solve(a), expected)
    for code in range(8):
        mapping = np.array([(i * 3 + code) % 10 for i in range(10)])
        assert np.array_equal(solve(_orient(mapping[a], code)), _orient(mapping[expected], code))
    # Same pins with exact contact: neither port writes any tape.
    still = dict(f, heights=(1, 2))
    still_a = _draw_input(still)
    assert _shifts(still) == (0, 0)
    assert np.array_equal(solve(still_a)[:9], still_a[:9])
    # Exercise finite fallbacks, including fresh colors at full pair capacity.
    fallback = dict(f, mirrors=())
    seen = set()
    for _ in range(14):
        fresh = _unique_fallback(fallback, 0, seen)
        seen.add(_key(fresh["input"]))
        assert np.array_equal(solve(fresh["input"]), fresh["output"])
    assert len(seen) == 14
    # Input-only solver rejects a gapped pin and a broken source port.
    for row, col in ((11, 12), (7, 13)):
        bad = a.copy()
        bad[row, col] = 0
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed apparatus accepted")


def self_test():
    _regressions()
    signatures, structures, shapes, lanes, spacings, depths = set(), set(), set(), set(), set(), set()
    backgrounds, structural_colors, orientations, packet_lengths, bar_lengths, route_turns = set(), set(), set(), set(), set(), set()
    pairs = stationary = moving = flat = multicolor = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        for split in puzzle:
            for pair in puzzle[split]:
                a = np.asarray(pair["input"])
                info = _features(a)
                f = info["f"]
                assert np.array_equal(solve(a.copy()), pair["output"])
                structures.add(_structural_signature(a))
                shapes.add(a.shape)
                lanes.add(len(f["lanes"]))
                spacings.add(f["lanes"][1] - f["lanes"][0])
                depths.add(f["base"] - f["top"])
                backgrounds.add(f["bg"])
                structural_colors.add(f["structure"])
                orientations.add(f["orientation"])
                packet_lengths.update(info["lengths"])
                bar_lengths.update(m[3] for m in f["mirrors"])
                route_turns.update(len(e) for e in info["events"])
                pairs += 1
                stationary += info["lengths"].count(0)
                moving += sum(s > 0 for s in info["lengths"])
                flat += len(set(f["heights"])) == 1
                multicolor += info["multicolor"]
    assert len(signatures) == 42 and len(structures) >= 120 and len(shapes) >= 40
    assert lanes == {2, 3, 4, 5} and spacings >= {2, 3}
    assert len(depths) >= 5 and orientations == set(range(8))
    assert backgrounds == set(range(10)) and structural_colors == set(range(10))
    assert packet_lengths >= set(range(9)) and bar_lengths >= {1, 3, 5}
    assert route_turns >= {0, 1, 2} and stationary >= 80 and moving >= 200
    assert flat >= 8 and multicolor >= 80
    assert _puzzle_signature(generate_puzzle(0)) == _puzzle_signature(generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(0)
    wrong = {s: [dict(p) for p in sample[s]] for s in sample}
    wrong["test"][0]["output"] = np.asarray(wrong["test"][0]["output"]).copy()
    wrong["test"][0]["output"][0, 0] = (int(wrong["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(wrong)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                structures=len(structures), shapes=len(shapes), lanes=sorted(lanes),
                spacings=sorted(spacings), depths=sorted(depths),
                packet_lengths=sorted(packet_lengths), mirror_lengths=sorted(bar_lengths),
                turns=sorted(route_turns), stationary_lanes=stationary,
                moving_lanes=moving, flat_examples=flat, multicolor_examples=multicolor)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
