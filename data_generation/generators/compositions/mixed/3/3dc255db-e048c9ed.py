"""Hollow frames export the units digit of their payload adjacency square.

Only NumPy and the standard library are used. Generation paints labels directly;
the solver separately reads monochromatic components and their cross-sections.
No parent module, saved example, seed, or generation record is used by solve().
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


_DIRECTIONS = ("U", "R", "D", "L")
_STEP = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}
_LENGTHS = (1, 2, 3, 4, 5, 6)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _components(cells, diagonal=True):
    remaining = set(cells)
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _canonical(p, direction):
    r, c = p
    return {"U": (r, c), "R": (-c, r), "D": (-r, -c), "L": (c, -r)}[direction]


def _world(p, direction):
    y, x = p
    return {"U": (y, x), "R": (x, -y), "D": (-y, -x), "L": (-x, y)}[direction]


def _axis(cells):
    """A singleton is a valid bar with zero internal adjacencies."""
    if not cells:
        return None
    rows, cols = zip(*cells)
    if len(set(rows)) == 1 and max(cols) - min(cols) + 1 == len(cells):
        return "H"
    if len(set(cols)) == 1 and max(rows) - min(rows) + 1 == len(cells):
        return "V"
    return None


def _adjacencies(cells):
    return sum((r + 1, c) in cells for r, c in cells) + sum(
        (r, c + 1) in cells for r, c in cells)


def _forward(shape, tip, direction, limit=None):
    # In particular, zero MUST return no cells, even with ample clearance.
    if limit == 0:
        return ()
    h, w = shape
    dr, dc = _STEP[direction]
    r, c = tip
    cells = []
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
        if limit is not None and len(cells) == limit:
            break
    return tuple(cells)


def _envelope(frame, direction):
    """Read an open, monotone two-arm envelope without a stored stencil.

    Before the hollow opens there can be a stem and solid cap. Afterwards
    every cross-section has two contiguous bands. Both outer edges move
    monotonically away from the tip. Diagonal arm connections are allowed.
    """
    local = {_canonical(p, direction) for p in frame}
    ys = [y for y, _ in local]
    first, last = min(ys), max(ys)
    if last - first < 2:
        return None
    levels = [sorted(x for y, x in local if y == level)
              for level in range(first, last + 1)]
    if len(levels[0]) != 1 or any(not row for row in levels):
        return None
    tip_x = levels[0][0]
    left = right = tip_x
    opened = False
    cavity = set()
    for y, row in zip(range(first + 1, last + 1), levels[1:]):
        runs = []
        for x in row:
            if not runs or x != runs[-1][-1] + 1:
                runs.append([x])
            else:
                runs[-1].append(x)
        if row[0] > left or row[-1] < right:
            return None
        left, right = row[0], row[-1]
        if len(runs) == 1 and not opened:
            continue
        if len(runs) != 2:
            return None
        opened = True
        cavity.update(_world((y, x), direction)
                      for x in range(runs[0][-1] + 1, runs[1][0]))
    if not opened or not left < tip_x < right:
        return None
    return _world((first, tip_x), direction), cavity


def _read(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    k = int(np.argmax(counts))
    if int(counts[k]) * 2 <= a.size:
        raise ValueError("background must occupy strictly more than half the input")
    bg = int(values[k])
    objects = []
    for color in (int(v) for v in values if v != bg):
        for frame in _components(map(tuple, np.argwhere(a == color))):
            if len(frame) < 5:
                continue
            possibilities = []
            for direction in _DIRECTIONS:
                geometry = _envelope(frame, direction)
                if geometry is None:
                    continue
                tip, cavity = geometry
                palette = {int(a[p]) for p in cavity} - {bg}
                if len(palette) != 1 or color in palette:
                    continue
                payload_color = next(iter(palette))
                payload = {p for p in cavity if int(a[p]) == payload_color}
                n = len(payload)
                if n not in _LENGTHS or _axis(payload) is None or len(cavity) <= n:
                    continue
                adjacency = _adjacencies(payload)
                if adjacency != n - 1:
                    continue
                rs, cs = zip(*frame)
                box = {(r, c) for r in range(min(rs), max(rs) + 1)
                       for c in range(min(cs), max(cs) + 1)}
                if any(a[p] != bg for p in box - frame - cavity):
                    continue
                digit = adjacency ** 2 % 10
                ray = _forward(a.shape, tip, direction, digit)
                if any(a[p] != bg for p in ray):
                    continue
                possibilities.append(dict(frame=frame, cavity=cavity, payload=payload,
                    frame_color=color, payload_color=payload_color, n=n,
                    adjacency=adjacency, digit=digit, direction=direction, tip=tip,
                    box=box, ray=ray, available=len(_forward(a.shape, tip, direction))))
            if len(possibilities) > 1:
                raise ValueError("ambiguous frame direction")
            objects.extend(possibilities)
    if not objects:
        raise ValueError("no valid hollow frame with a payload bar")
    objects.sort(key=lambda o: o["tip"])
    boxes, rays = set(), set()
    for o in objects:
        if boxes & o["box"] or rays & set(o["ray"]):
            raise ValueError("objects or emitted rays overlap")
        boxes.update(o["box"])
        rays.update(o["ray"])
    if boxes & rays:
        raise ValueError("a ray enters a frame region")
    return bg, objects


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every edit from this input alone; literal colors carry no code."""
    a = _grid(input_grid)
    bg, objects = _read(a)
    result = a.copy()
    for o in objects:
        for p in o["payload"]:
            result[p] = bg
        for p in o["ray"]:
            result[p] = o["payload_color"]
    return result


def _profile(rng, anchor=False, n=1):
    # Independent arm walks give thin/thick, symmetric/asymmetric caps.
    if anchor:
        left, right = [(3, 4), (4, 3), (4, 4)][int(rng.integers(3))]
        tail = int(rng.integers(7, 9))
    else:
        left, right = (int(v) for v in rng.integers(1, 5, size=2))
        if rng.random() < .3:
            right = left
        # Leave a visibly hollow cell even in a one-cell-wide cavity.
        tail = int(rng.integers(max(2, n + 1), max(2, n + 1) + 3))
    stem = int(rng.integers(0, 3 if not anchor else 2))
    walks = [(0, 0)] * (stem + 1)
    l = r = 0
    symmetric = left == right and rng.random() < .5
    while l < left or r < right:
        dl = l < left and (symmetric or r == right or rng.random() < .8)
        dr = r < right and (symmetric or l == left or rng.random() < .8)
        if not dl and not dr:
            dl = l < left
            dr = not dl
        l, r = l + int(dl), r + int(dr)
        walks.append((-l, r))
    slope_end = len(walks)
    walks += [(-left, right)] * tail
    thick = rng.random() < .4
    frame, cavity = set(), set()
    for y, (lo, hi) in enumerate(walks):
        t = 2 if thick and y < slope_end else 1
        li, ri = lo + t - 1, hi - t + 1
        if ri - li < 2:
            frame.update((y, x + left) for x in range(lo, hi + 1))
        else:
            frame.update((y, x + left) for x in range(lo, li + 1))
            frame.update((y, x + left) for x in range(ri, hi + 1))
            cavity.update((y, x + left) for x in range(li + 1, ri))
    return dict(frame=frame, cavity=cavity, tip=(0, left))


def _spec(rng, n, direction, colors, profile=None, mode="free", axis=None):
    profile = profile if profile is not None else _profile(rng, n=n)
    cavity = profile["cavity"]
    choices = {"V": [], "H": []}
    for r, c in sorted(cavity):
        for name, (dr, dc) in (("V", (1, 0)), ("H", (0, 1))):
            bar = {(r + j * dr, c + j * dc) for j in range(n)}
            if bar <= cavity:
                choices[name].append(bar)
    if axis is None or not choices[axis]:
        axis = str(rng.choice([k for k, v in choices.items() if v]))
    payload = choices[axis][int(rng.integers(len(choices[axis])))]
    all_cells = profile["frame"] | cavity
    transformed = {p: _world(p, direction) for p in all_cells}
    r0 = min(r for r, _ in transformed.values())
    c0 = min(c for _, c in transformed.values())

    def transform(p):
        r, c = transformed[p]
        return r - r0, c - c0

    frame = {transform(p) for p in profile["frame"]}
    digit = (n - 1) ** 2 % 10
    if mode == "clip" and digit <= 1:
        mode = "edge"
    return dict(frame=frame, payload={transform(p) for p in payload},
                tip=transform(profile["tip"]), direction=direction,
                frame_color=int(colors[0]), payload_color=int(colors[1]), n=n,
                digit=digit, mode=mode, height=max(r for r, _ in frame) + 1,
                width=max(c for _, c in frame) + 1)


def _positions(spec, h, w, rng):
    choices = []
    for r in range(h - spec["height"] + 1):
        for c in range(w - spec["width"] + 1):
            tr, tc = r + spec["tip"][0], c + spec["tip"][1]
            room = {"U": tr, "R": w - 1 - tc, "D": h - 1 - tr, "L": tc}[spec["direction"]]
            d, mode = spec["digit"], spec["mode"]
            if ((mode == "free" and room < max(2, d + 1))
                    or (mode == "exact" and room != d)
                    or (mode == "clip" and not 0 < room < d)
                    or (mode == "edge" and room != 0)):
                continue
            choices.append((r, c))
    return [choices[int(i)] for i in rng.permutation(len(choices))]


def _pack(rng, h, w, specs, required, stripes=False):
    """Bounded placement; optional extras may be omitted when they do not fit."""
    boxes, guards, rays = set(), set(), set()
    placed = []
    stripe_origins = {}
    if stripes:
        direction = specs[0]["direction"]
        if any(s["direction"] != direction for s in specs[:required]):
            return None
        vertical = direction in ("U", "D")
        sizes = [s["width"] if vertical else s["height"] for s in specs[:required]]
        capacity = w if vertical else h
        spare = capacity - sum(sizes) - required + 1
        if spare < 0:
            return None
        gaps = rng.multinomial(spare, np.ones(required + 1) / (required + 1))
        offset = int(gaps[0])
        for j, index in enumerate(rng.permutation(required)):
            index = int(index)
            stripe_origins[index] = offset
            offset += sizes[index] + 1 + int(gaps[j + 1])
    for index, s in enumerate(specs):
        accepted = None
        for r, c in _positions(s, h, w, rng):
            if index in stripe_origins:
                cross = c if s["direction"] in ("U", "D") else r
                if cross != stripe_origins[index]:
                    continue
            box = {(r + y, c + x) for y in range(s["height"]) for x in range(s["width"])}
            if box & (guards | rays):
                continue
            tip = (r + s["tip"][0], c + s["tip"][1])
            guard_ray = set(_forward((h, w), tip, s["direction"], s["digit"] + 1))
            if guard_ray & (boxes | rays):
                continue
            accepted = (s, r, c, tip)
            boxes.update(box)
            guards.update((y + dy, x + dx) for y, x in box
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1))
            rays.update(guard_ray)
            break
        if accepted is None:
            if index < required:
                return None
        else:
            placed.append(accepted)
    return placed


def _scene(rng, specs, required, background, mark=False):
    min_h = min_w = 7
    for s in specs[:required]:
        extra = max(2, s["digit"] + 1) if s["mode"] == "free" else s["digit"] if s["mode"] == "exact" else 0
        vertical = s["direction"] in ("U", "D")
        min_h = max(min_h, s["height"] + (extra if vertical else 0))
        min_w = max(min_w, s["width"] + (0 if vertical else extra))
    if len({s["direction"] for s in specs[:required]}) == 1:
        vertical = specs[0]["direction"] in ("U", "D")
        cross = sum(s["width"] if vertical else s["height"] for s in specs[:required]) + required - 1
        if vertical:
            min_w = max(min_w, cross)
        else:
            min_h = max(min_h, cross)
    _require(min_h <= 30 and min_w <= 30, "mandatory geometry too large")
    h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    placed = _pack(rng, h, w, specs, required)
    if placed is None:
        placed = _pack(rng, h, w, specs, required, stripes=True)
    if placed is None:
        h = w = 30
        placed = _pack(rng, h, w, specs, required, stripes=True)
    _require(placed is not None, "mandatory examples could not be packed")
    inp = np.full((h, w), background, dtype=np.int8)
    out = inp.copy()
    protected = set()
    for s, r, c, tip in placed:
        for y, x in s["frame"]:
            inp[r + y, c + x] = out[r + y, c + x] = s["frame_color"]
        for y, x in s["payload"]:
            inp[r + y, c + x] = s["payload_color"]
        for p in _forward(inp.shape, tip, s["direction"], s["digit"]):
            out[p] = s["payload_color"]
        protected.update((r + y, c + x) for y in range(-2, s["height"] + 2)
                         for x in range(-2, s["width"] + 2))
        protected.update(_forward(inp.shape, tip, s["direction"]))
    # Exterior bars can share a payload color, but are never exported.
    mark_count = 1 if mark else int(rng.choice([0, 0, 1, 2]))
    for _ in range(mark_count):
        options = []
        length = int(rng.integers(2, 4))
        for r in range(h):
            for c in range(w):
                for dr, dc in ((0, 1), (1, 0)):
                    bar = {(r + j * dr, c + j * dc) for j in range(length)}
                    if (all(0 <= y < h and 0 <= x < w for y, x in bar)
                            and not bar & protected):
                        options.append(bar)
        if not options:
            if mark:
                # A singleton still provides unambiguous preservation evidence.
                options = [{(r, c)} for r in range(h) for c in range(w)
                           if (r, c) not in protected]
            if not options:
                break
        bar = options[int(rng.integers(len(options)))]
        color = placed[0][0]["payload_color"] if mark else int(rng.choice(
            [v for v in range(10) if v != background]))
        for p in bar:
            inp[p] = out[p] = color
        protected.update((r + dy, c + dx) for r, c in bar
                         for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    # Compare two implementations: direct painting versus input reconstruction.
    bg, observed = _read(inp)
    signature = lambda o: (o["tip"], o["direction"], o["n"], o["frame_color"], o["payload_color"])
    expected = {(tip, s["direction"], s["n"], s["frame_color"], s["payload_color"])
                for s, _, _, tip in placed}
    _require(bg == background and {signature(o) for o in observed} == expected,
             "construction and input reader disagree")
    _require(np.array_equal(solve(inp), out), "direct label and solver disagree")
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _profile_key(o):
    y0, x0 = _canonical(o["tip"], o["direction"])
    return tuple(sorted((y - y0, x - x0) for y, x in (
        _canonical(p, o["direction"]) for p in o["frame"])))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return a reproducible puzzle; counts are explicit integers, never latent."""
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    directions = [str(d) for d in rng.permutation(_DIRECTIONS)]
    # Teach two perpendicular directions, then hold out the opposite signs.
    first = directions[0]
    perpendicular = [d for d in _DIRECTIONS if _STEP[d][0] != 0] if first in ("R", "L") else ["R", "L"]
    second = str(rng.choice(perpendicular))
    train_dirs = [first, second]
    test_dirs = [d for d in _DIRECTIONS if d not in train_dirs]
    colors = tuple(int(v) for v in rng.choice(10, 2, replace=False))
    bg_options = [v for v in range(10) if v not in colors]
    bg0, bg1 = (int(v) for v in rng.choice(bg_options, 2, replace=False))
    anchor = _profile(rng, anchor=True)
    plans = [[3, 4], [1, 5]] + [[] for _ in range(nt - 2)]
    for j, n in enumerate((2, 6)):
        plans[2 + j if nt >= 4 else 2 if nt == 3 else j].append(n)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            entries = plans[index] if split == "train" else []
            bg = (bg0 if index == 0 else bg1) if entries else int(rng.integers(10))
            direction = train_dirs[index % 2] if split == "train" else str(rng.choice(test_dirs))
            specs = [_spec(rng, n, direction, colors, profile=anchor,
                           axis="V" if n == 3 else "H" if n == 4 else None)
                     for n in entries]
            required = len(specs)
            if not required:
                n = int(rng.choice((3, 4, 5, 6))) if split == "test" else int(rng.choice(_LENGTHS))
                local_colors = tuple(int(v) for v in rng.choice([c for c in range(10) if c != bg], 2, replace=False))
                specs.append(_spec(rng, n, direction, local_colors))
                required = 1
            target = max(required, int(rng.choice([1, 1, 2, 2, 3, 4])))
            color_mode = str(rng.choice(["local", "frame", "payload", "swap"]))
            for j in range(len(specs), target):
                palette = [c for c in range(10) if c != bg]
                f, p = specs[0]["frame_color"], specs[0]["payload_color"]
                local_colors = (tuple(int(v) for v in rng.choice(palette, 2, replace=False))
                    if color_mode == "local" else (f, int(rng.choice([v for v in palette if v != f])))
                    if color_mode == "frame" else (int(rng.choice([v for v in palette if v != p])), p)
                    if color_mode == "payload" else (p, f) if j % 2 else (f, p))
                pool = train_dirs if split == "train" else _DIRECTIONS
                mode = str(rng.choice(["free", "free", "exact", "clip", "edge"]))
                specs.append(_spec(rng, int(rng.choice(_LENGTHS)), str(rng.choice(pool)), local_colors, mode=mode))
            for _ in range(8):
                pair = _scene(rng, specs, required, bg, mark=split == "train" and index == 0)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded attempts did not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == b.shape, "output shape changed")
    bg, objects = _read(a)
    _require(1 <= len(objects) <= 4, "object count outside generated family")
    expected = a.copy()
    owned, changed = set(), set()
    for o in objects:
        _require(_adjacencies(o["payload"]) == o["n"] - 1, "payload is not an adjacency bar")
        _require(o["digit"] == (o["n"] - 1) ** 2 % 10, "wrong arithmetic")
        for p in o["payload"]:
            expected[p] = bg
        for p in o["ray"]:
            expected[p] = o["payload_color"]
        owned.update(o["frame"] | o["payload"])
        changed.update(o["payload"] | set(o["ray"]))
        _require(all(b[p] == o["frame_color"] for p in o["frame"]), "frame changed")
        if o["available"] > o["digit"]:
            after = _forward(a.shape, o["tip"], o["direction"], o["digit"] + 1)[-1]
            _require(b[after] == bg, "complete ray has no blank stopping witness")
    _require(np.array_equal(b, expected), "wrong edits or exterior preservation")
    _require(np.count_nonzero(a != b) == len(changed), "edits overlap or are invisible")
    exterior = set(map(tuple, np.argwhere(a != bg))) - owned
    marks = _components(exterior)
    _require(all(_axis(m) is not None and len(m) <= 3 for m in marks), "invalid exterior marks")
    return bg, objects, marks


def validate_generated_puzzle(puzzle):
    """Check returned cells, arithmetic, membership, evidence and the held-out turn."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal counts")
    seen, taught, matched, train_dirs = set(), set(), {}, set()
    perpendicular_witness = exterior_witness = False
    test_records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            bg, objects, marks = _inspect_pair(pair)
            a = np.asarray(pair["input"])
            k = _key(a)
            _require(k not in seen, "duplicate input")
            seen.add(k)
            if split == "test":
                test_records.append(objects)
                continue
            exterior_witness |= bool(marks)
            for o in objects:
                train_dirs.add(o["direction"])
                full = o["available"] > o["digit"]
                if full:
                    taught.add(o["n"])
                    mk = (_profile_key(o), o["direction"], o["frame_color"], o["payload_color"])
                    matched.setdefault(mk, set()).add(o["n"])
                aligned_axis = "V" if o["direction"] in ("U", "D") else "H"
                perpendicular_witness |= o["digit"] > 0 and _axis(o["payload"]) != aligned_axis
    _require(taught == set(_LENGTHS), "training must fully demonstrate all six counts")
    _require(any({3, 4} <= counts for counts in matched.values()), "no controlled 3-versus-4 comparison")
    _require(perpendicular_witness and exterior_witness, "frame direction/membership is not demonstrated")
    _require(len(train_dirs) == 2 and any(d in ("U", "D") for d in train_dirs)
             and any(d in ("R", "L") for d in train_dirs), "training must teach perpendicular tips")
    for objects in test_records:
        _require(any(o["direction"] not in train_dirs for o in objects), "missing held-out direction shift")
        _require(any(o["digit"] > 0 and o["available"] > o["digit"] for o in objects), "test hides all arithmetic stopping")
        _require(all(o["n"] in taught for o in objects), "untaught test arithmetic case")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds and all 30 legal count requests; never save bulk grids."""
    fingerprints, shapes, profiles, backgrounds = set(), set(), set(), set()
    orientations, lengths, object_counts, ray_modes = set(), set(), set(), Counter()
    pairs = shared = mixed = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        if seed in (0, 17, 41):
            _require(_fingerprint(generate_puzzle(seed)) == _fingerprint(puzzle), "nondeterministic seed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                bg, objects, _ = _inspect_pair(pair)
                pairs += 1
                shapes.add(a.shape)
                backgrounds.add(bg)
                object_counts.add(len(objects))
                mixed += len({o["direction"] for o in objects}) > 1
                shared += len({o["frame_color"] for o in objects}) < len(objects) or len({o["payload_color"] for o in objects}) < len(objects)
                for o in objects:
                    profiles.add(_profile_key(o))
                    orientations.add(o["direction"])
                    lengths.add(o["n"])
                    mode = "arithmetic_zero" if o["digit"] == 0 else "edge_zero" if o["available"] == 0 else "clipped" if o["available"] < o["digit"] else "exact" if o["available"] == o["digit"] else "free"
                    ray_modes[mode] += 1
                original = a.copy()
                _require(np.array_equal(solve(a), b) and np.array_equal(a, original), "solver mismatch or mutation")
                # Palette permutation must change colors, never the arithmetic length.
                if seed < 3:
                    palette = np.array([(v + 3) % 10 for v in range(10)], dtype=np.int8)
                    for turn in range(4):
                        _require(np.array_equal(solve(np.rot90(palette[a], turn)),
                                                np.rot90(palette[b], turn)), "rotation/recolor equivariance")
    _require(len(fingerprints) == 42 and len(shapes) >= 40 and len(profiles) >= 40, "insufficient structural variation")
    _require(backgrounds == set(range(10)) and orientations == set(_DIRECTIONS), "missing colors or directions")
    _require(lengths == set(_LENGTHS) and object_counts >= {1, 2, 3}, "missing count variation")
    _require(set(ray_modes) == {"arithmetic_zero", "edge_zero", "clipped", "exact", "free"}, "missing boundary/zero cases")
    _require(mixed > 0 and shared > 0, "missing compatible structural combinations")
    explicit = 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(10000 + 10 * nt + ns, nt, ns)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns), "counts ignored")
            explicit += 1
    for nt, ns in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "frame_profiles": len(profiles),
            "backgrounds": sorted(backgrounds), "directions": sorted(orientations),
            "payload_counts": sorted(lengths), "object_counts": sorted(object_counts),
            "ray_modes": dict(sorted(ray_modes.items())), "mixed_direction_pairs": mixed,
            "shared_role_color_pairs": shared, "legal_count_combinations": explicit}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
