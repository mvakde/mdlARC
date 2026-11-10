"""Seal and dye: endpoint growth makes the regions addressed by an edge key.

Only Python's standard library and NumPy are used. No files are read or written.
All interpretation uses the supplied input; generation metadata is never needed.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
SIDES = ("top", "left", "bottom", "right")
HOST_SEEDS = (0, 912486822, 172958989, 819981250, 511345116, 228567898,
              906633920, 324647737, 253874605, 585317423, 445053177,
              802121289, 257848569, 797240817, 383191601, 141880643,
              877806122, 362506332, 21353833, 1908652, 193269704,
              845017804, 852211890, 550794790, 933210171, 204254692,
              191286734, 254891977, 874507909, 936428555, 725308039,
              882307829, 928510, 327531297, 459678288, 157360067,
              961330467, 767181089, 498145194, 583223297, 955637018,
              972757760)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, dtype=bool)
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += DIRECTIONS
    result = []
    for r, c in np.argwhere(mask):
        p = (int(r), int(c))
        if seen[p]:
            continue
        seen[p] = True
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if _inside(q, mask.shape) and mask[q] and not seen[q]:
                    seen[q] = True
                    stack.append(q)
        result.append(cells)
    return result


def _regions(grid, bg):
    result = []
    h, w = grid.shape
    for cells in _components(grid == bg):
        edge = any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
        boundary = set()
        for r, c in cells:
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if _inside(p, grid.shape) and grid[p] != bg:
                    boundary.add(p)
        result.append((cells, edge, {int(grid[p]) for p in boundary}))
    return result


def _diamond(center, radius):
    r, c = center
    return [(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius]


def _cap(center, radius, direction):
    r, c = center
    dr, dc = direction
    return [(r + dr * (k + 1), c + dc * (radius - k + 1)) for k in range(radius + 1)]


def _read_key(grid, bg):
    candidates = []
    h, w = grid.shape
    for cells in _components(grid != bg, diagonal=True):
        rr, cc = zip(*cells)
        a, b, c, d = min(rr), max(rr), min(cc), max(cc)
        sh, sw = b - a + 1, d - c + 1
        if len(cells) != sh * sw:
            continue
        lanes = []
        # Length >=3 makes the strip direction unambiguous even at a corner.
        if sh == 2 and 3 <= sw <= 7:
            if a == 0:
                lanes.append(("top", grid[a, c:d + 1], grid[b, c:d + 1]))
            if b == h - 1:
                lanes.append(("bottom", grid[b, c:d + 1], grid[a, c:d + 1]))
        if sw == 2 and 3 <= sh <= 7:
            if c == 0:
                lanes.append(("left", grid[a:b + 1, c], grid[a:b + 1, d]))
            if d == w - 1:
                lanes.append(("right", grid[a:b + 1, d], grid[a:b + 1, c]))
        for side, outer, inner in lanes:
            sources, fills = tuple(map(int, outer)), tuple(map(int, inner))
            if len(set(sources)) == len(sources):
                candidates.append(dict(side=side, mapping=dict(zip(sources, fills)),
                                       cells=set(cells), sources=sources, fills=fills))
    _require(len(candidates) == 1, "expected one isolated edge key of length 3..7")
    return candidates[0]


def _parse(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > grid.size // 2, "background must occupy a strict majority")
    key = _read_key(grid, bg)
    h, w = grid.shape
    emitters = []
    # Recover complete, hollow Manhattan diamonds; cap colors are local roles.
    for radius in (2, 3, 4):
        if min(h, w) < 2 * radius + 1:
            continue
        shape = (h - 2 * radius, w - 2 * radius)
        offsets = _diamond((0, 0), radius)
        dr, dc = offsets[0]
        colors = grid[radius + dr:radius + dr + shape[0],
                      radius + dc:radius + dc + shape[1]]
        mask = colors != bg
        for dr, dc in offsets[1:]:
            mask &= grid[radius + dr:radius + dr + shape[0],
                         radius + dc:radius + dc + shape[1]] == colors
        for rr, cc in np.argwhere(mask):
            center = (int(rr) + radius, int(cc) + radius)
            r, c = center
            interior = [(r + x, c + y) for x in range(-radius + 1, radius)
                        for y in range(-radius + 1, radius) if abs(x) + abs(y) < radius]
            if not all(grid[p] == bg for p in interior):
                continue
            structural = int(colors[rr, cc])
            caps = {}
            for direction in DIRECTIONS:
                cells = _cap(center, radius, direction)
                if all(_inside(p, grid.shape) for p in cells):
                    values = {int(grid[p]) for p in cells}
                    if len(values) == 1 and next(iter(values)) not in (bg, structural):
                        caps[direction] = next(iter(values))
            if caps:
                emitters.append(dict(center=center, radius=radius, structural=structural, caps=caps))
    _require(1 <= len(emitters) <= 3, "expected 1..3 capped diamond emitters")
    outlines = set()
    for e in emitters:
        cells = set(_diamond(e["center"], e["radius"]))
        _require(not outlines & cells and not key["cells"] & cells, "overlapping emitters")
        outlines.update(cells)
    return dict(bg=bg, key=key, emitters=emitters)


def _complete(grid, analysis):
    """All rays inspect ORIGINAL ink, so there is no processing-order effect."""
    claims, ports = {}, []
    for e in analysis["emitters"]:
        for direction, color in e["caps"].items():
            cap = _cap(e["center"], e["radius"], direction)
            paths, stops = [], []
            for endpoint in (cap[0], cap[-1]):
                dr, dc = direction
                p = (endpoint[0] + dr, endpoint[1] + dc)
                path = []
                while _inside(p, grid.shape) and grid[p] == analysis["bg"]:
                    _require(p not in claims, "ray continuations must be disjoint")
                    claims[p] = color
                    path.append(p)
                    p = (p[0] + dr, p[1] + dc)
                stops.append(p if _inside(p, grid.shape) else None)
                paths.append(path)
            ports.append(dict(direction=direction, color=color, cap=cap, paths=paths, stops=stops,
                              center=e["center"], radius=e["radius"]))
    completed = grid.copy()
    for p, color in claims.items():
        completed[p] = color
    return completed, claims, ports


def _interpret(grid):
    analysis = _parse(grid)
    bg, mapping = analysis["bg"], analysis["key"]["mapping"]
    original_regions = _regions(grid, bg)
    exterior = np.zeros(grid.shape, dtype=bool)
    old_mapped = []
    for cells, edge, colors in original_regions:
        if edge:
            for p in cells:
                exterior[p] = True
        elif len(colors) == 1 and next(iter(colors)) in mapping:
            old_mapped.append(cells)
    completed, claims, ports = _complete(grid, analysis)
    output = completed.copy()
    new_regions, filled, unlisted = [], [], []
    for cells, edge, colors in _regions(completed, bg):
        if edge or not exterior[cells[0]]:
            continue
        _require(all(exterior[p] for p in cells), "inconsistent original region")
        new_regions.append((cells, colors))
        if len(colors) == 1:
            color = next(iter(colors))
            if color in mapping:
                for p in cells:
                    output[p] = mapping[color]
                filled.append((cells, color))
            else:
                unlisted.append((cells, color))
    analysis.update(completed=completed, claims=claims, ports=ports, exterior=exterior,
                    original_regions=original_regions, old_mapped=old_mapped,
                    new_regions=new_regions, filled=filled, unlisted=unlisted)
    return output, analysis


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete endpoint rails, then dye only newly sealed monochromatic holes."""
    return _interpret(_grid(input_grid))[0]


def _rotate(direction, turns):
    dr, dc = direction
    for _ in range(turns % 4):
        dr, dc = -dc, dr
    return dr, dc


def _wall(center, radius, direction, profile):
    """A monotone diagonal/stepped catcher, joining the two endpoint rails."""
    dr, dc = direction
    anchors = [(r + dr * t, c + dc * t)
               for (r, c), t in zip(_cap(center, radius, direction), profile)]
    wall = [anchors[0]]
    for a, b in zip(anchors, anchors[1:]):
        distance = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        for k in range(1, distance + 1):
            wall.append((a[0] + (b[0] - a[0]) * k // distance,
                         a[1] + (b[1] - a[1]) * k // distance))
    return wall, anchors


def _module(rng, radius, directions, roles, structural, max_length, simple=False):
    ink = {p: structural for p in _diamond((0, 0), radius)}
    expected = set(ink)
    specs = []
    for j, (direction, role) in enumerate(zip(directions, roles)):
        color = 5 if role == "unlisted" else 2 if j == 0 else int(rng.choice([2, 6, 7]))
        cap = _cap((0, 0), radius, direction)
        for p in cap:
            _require(p not in ink, "module cap overlap")
            ink[p] = color
        length = 3 if simple else int(rng.integers(2, max_length + 1))
        profile = [length] * (radius + 1)
        if not simple and rng.random() < .7:
            for k in range(1, radius + 1):
                profile[k] = max(2, min(max_length, profile[k - 1] + int(rng.integers(-1, 2))))
        # The first teaching channel always exposes a nonflat far boundary.
        if role == "chain" and not simple:
            profile = [max(2, min(max_length - 1, length))] * (radius + 1)
            profile[radius // 2] += 1
        wall, anchors = _wall((0, 0), radius, direction, profile)
        if role == "open":
            # Remove a whole short cross-section, while keeping both rail stops.
            gap = anchors[radius // 2]
            wall = [p for p in wall if max(abs(p[0] - gap[0]), abs(p[1] - gap[1])) > 0]
        for p in wall:
            _require(p not in ink, "module wall overlap")
            ink[p] = color
        expected.update(wall)
        for endpoint, distance in zip((cap[0], cap[-1]), (profile[0], profile[-1])):
            for t in range(1, distance):
                expected.add((endpoint[0] + direction[0] * t, endpoint[1] + direction[1] * t))
        expected.update(cap)
        specs.append((direction, color, role))
    rr, cc = zip(*expected)
    offset = (-min(rr), -min(cc))
    normalized = {(r + offset[0], c + offset[1]): color for (r, c), color in ink.items()}
    return dict(ink=normalized, shape=(max(rr) - min(rr) + 1, max(cc) - min(cc) + 1),
                center=offset, radius=radius, structural=structural, specs=specs)


def _mapping(rng, modules, chain=False, fill_only=False):
    active = {color for m in modules for _, color, role in m["specs"] if role != "unlisted"}
    forced = active | {m["structural"] for m in modules}
    if chain:
        forced.add(3)
    choices = [c for c in range(1, 10) if c != 5 and c not in forced]
    rng.shuffle(choices)
    length = int(rng.integers(max(3, len(forced)), 8))
    sources = sorted(forced) + choices[:length - len(forced)]
    mapping = {c: int(rng.choice([d for d in range(1, 10) if c not in active or d != c])) for c in sources}
    if chain:
        mapping[2], mapping[3] = 3, 4
    if fill_only:
        mapping[modules[0]["structural"]] = 5
    order = list(map(int, rng.permutation(sources)))
    return order, [mapping[c] for c in order]


def _pack(rng, modules, sources, fills, fallback=False):
    maxh = max(m["shape"][0] for m in modules)
    maxw = max(m["shape"][1] for m in modules)
    ink_count = sum(len(m["ink"]) for m in modules) + 2 * len(sources)
    for attempt in range(18):
        if fallback or attempt >= 10:
            h = w = 30
        else:
            h = int(rng.integers(max(10, maxh), 31))
            w = int(rng.integers(max(10, maxw, len(sources)), 31))
            if rng.random() < .15:
                h = w = max(h, w)
        if h * w <= 2 * ink_count:
            continue
        start = int(rng.integers(w - len(sources) + 1))
        grid = np.zeros((h, w), dtype=np.int8)
        grid[0, start:start + len(sources)] = sources
        grid[1, start:start + len(sources)] = fills
        blocked = np.zeros_like(grid, dtype=bool)
        blocked[:3, max(0, start - 1):min(w, start + len(sources) + 1)] = True
        placements = [None] * len(modules)
        for i in sorted(range(len(modules)), key=lambda k: -np.prod(modules[k]["shape"])):
            mh, mw = modules[i]["shape"]
            positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)
                         if not np.any(blocked[r:r + mh, c:c + mw])]
            if not positions:
                break
            r, c = positions[int(rng.integers(len(positions)))]
            placements[i] = (r, c)
            blocked[max(0, r - 1):min(h, r + mh + 1), max(0, c - 1):min(w, c + mw + 1)] = True
        else:
            expected = []
            for m, (r, c) in zip(modules, placements):
                for (dr, dc), color in m["ink"].items():
                    grid[r + dr, c + dc] = color
                expected.append(((r + m["center"][0], c + m["center"][1]), m["radius"],
                                 {d: color for d, color, _ in m["specs"]}))
            return grid, expected
    return None


def _make_pair(rng, max_radius, turns, lesson=None, curriculum="branch", is_test=False):
    for attempt in range(18):
        simple = attempt >= 12
        if lesson is not None:
            directions, roles = lesson
            n = 1
        elif curriculum == "multi":
            n = int(rng.integers(2, 4)) if is_test else 1
        else:
            n = 1 if curriculum == "branch" and is_test else int(rng.choice([1, 1, 2, 3]))
        modules = []
        for i in range(n):
            if lesson is not None:
                dirs, kinds = list(directions), list(roles)
            elif n == 1:
                if curriculum == "branch":
                    count = int(rng.integers(3, 5)) if is_test else int(rng.integers(1, 3))
                else:
                    count = int(rng.integers(1, 5))
                dirs = [DIRECTIONS[j] for j in rng.choice(4, count, replace=False)]
                kinds = ["closed"] + [str(rng.choice(["closed", "open", "unlisted"])) for _ in dirs[1:]]
            else:
                dirs = [DIRECTIONS[int(rng.integers(4))]]
                kinds = ["closed" if i == 0 else str(rng.choice(["closed", "open", "unlisted"]))]
            radius = max_radius if lesson is not None else 2 if n > 1 or simple else int(rng.integers(2, max_radius + 1))
            structural = 1 if i == 0 else int(rng.choice([1, 8, 9]))
            length = 4 if n > 1 else 6 if radius <= 3 else 5
            modules.append(_module(rng, radius, dirs, kinds, structural, length, simple))
        sources, fills = _mapping(rng, modules, chain=lesson is not None and "chain" in roles,
                                  fill_only=lesson is not None and "unlisted" in roles)
        packed = _pack(rng, modules, sources, fills, fallback=simple)
        if packed is None:
            continue
        grid, expected = packed
        try:
            output, analysis = _interpret(grid)
            recovered = [(e["center"], e["radius"], e["caps"]) for e in analysis["emitters"]]
            if sorted(recovered, key=lambda x: x[0]) != sorted(expected, key=lambda x: x[0]):
                continue
            f = _features(grid, output, analysis)
            if not f["filled"] or not f["old_mapped"]:
                continue
            if lesson is not None:
                if "chain" in roles and (not f["chain"] or not f["open"]):
                    continue
                if "unlisted" in roles and not f["unlisted"]:
                    continue
            palette = rng.permutation(10).astype(np.int8)
            return {"input": palette[np.rot90(grid, turns)].copy(),
                    "output": palette[np.rot90(output, turns)].copy()}
        except ValueError:
            continue
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    _require(num_train > num_test, "num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    maximum = int(rng.choice([2, 3, 4], p=[.4, .35, .25]))
    curriculum = str(rng.choice(["branch", "multi", "key"]))
    rotations = list(map(int, rng.permutation(4)))
    world_directions = [DIRECTIONS[j] for j in rng.permutation(4)]
    lessons = [([_rotate(d, -rotations[0]) for d in world_directions[:2]], ["chain", "open"]),
               ([_rotate(d, -rotations[1]) for d in world_directions[2:]], ["closed", "unlisted"])]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            turns = rotations[i] if split == "train" and i < 2 else int(rng.integers(4))
            if curriculum == "key" and split == "train" and i >= 2:
                turns = rotations[int(rng.integers(2))]
            if split == "test" and (curriculum == "key" or rng.random() < .65):
                turns = rotations[2 + i % 2]
            lesson = lessons[i] if split == "train" and i < 2 else None
            for _ in range(12):
                pair = _make_pair(rng, maximum, turns, lesson, curriculum, split == "test")
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-input construction exhausted")
    # Shuffle teaching positions so example index does not encode a rule.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, out, a=None):
    if a is None:
        _, a = _interpret(inp)
    mapping, bg = a["key"]["mapping"], a["bg"]
    open_ports = 0
    for p in a["ports"]:
        if p["color"] not in mapping:
            continue
        dr, dc = p["direction"]
        probes = [(r + dr, c + dc) for r, c in p["cap"][1:-1]]
        open_ports += any(_inside(q, inp.shape) and inp[q] == bg and out[q] == bg
                          and a["completed"][q] == bg for q in probes)
    return dict(side=a["key"]["side"], key_length=len(mapping), background=bg,
                emitters=len(a["emitters"]), radii=tuple(e["radius"] for e in a["emitters"]),
                max_caps=max(len(e["caps"]) for e in a["emitters"]),
                ports=len(a["ports"]), directions={p["direction"] for p in a["ports"]
                                                  if min(map(len, p["paths"])) >= 1},
                lengths=tuple(len(path) for p in a["ports"] for path in p["paths"]),
                filled=len(a["filled"]), unlisted=len(a["unlisted"]), open=open_ports,
                old_mapped=len(a["old_mapped"]),
                chain=any(mapping[c] in mapping and mapping[mapping[c]] != mapping[c] for _, c in a["filled"]),
                irregular=sum(len(cells) != (max(p[0] for p in cells) - min(p[0] for p in cells) + 1) *
                              (max(p[1] for p in cells) - min(p[1] for p in cells) + 1) for cells, _ in a["filled"]),
                unequal=sum(len(p["paths"][0]) != len(p["paths"][1]) for p in a["ports"]),
                changed=int(np.count_nonzero(inp != out)))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _count(a, 2, 10, "train length")
    _count(b, 1, 4, "test length")
    _require(a > b, "train length must exceed test length")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output dimensions changed")
            key = _fingerprint(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            predicted, analysis = _interpret(inp)
            _require(np.array_equal(predicted, out), "incorrect output")
            bg = analysis["bg"]
            _require(np.array_equal(inp[inp != bg], out[inp != bg]), "original ink changed")
            _require(all(analysis["exterior"][p] for p in analysis["claims"]),
                     "a ray crossed into an original enclosure")
            # Explicit checks on the two coupled mechanisms and temporal gate.
            for cells, edge, _ in analysis["original_regions"]:
                if not edge:
                    _require(all(out[p] == bg for p in cells), "a pre-existing hole changed")
            for cells, color in analysis["unlisted"]:
                _require(all(out[p] == bg for p in cells), "an unlisted new hole was dyed")
            for p, color in analysis["claims"].items():
                _require(out[p] == color, "endpoint rail was mapped or overwritten")
            f = _features(inp, out, analysis)
            _require(f["filled"] > 0 and f["changed"] > 0, "each scene needs a dyed new enclosure")
            _require(f["old_mapped"] > 0, "missing temporal preservation contrast")
            _require(not np.array_equal(out, analysis["completed"]), "B contribution is vacuous")
            features[split].append(f)
    train = features["train"]
    _require(set.union(*(f["directions"] for f in train)) == set(DIRECTIONS), "missing endpoint direction lesson")
    for name in ("chain", "open", "unlisted", "irregular"):
        _require(any(f[name] for f in train), "missing training distinction: " + name)
    _require(len({f["side"] for f in train}) >= 2, "need two directions for the key convention")
    _require(max(r for f in train for r in f["radii"]) >= max(r for f in features["test"] for r in f["radii"]),
             "test radius exceeds demonstrated scale")
    for f in features["test"]:
        _require(f["emitters"] > max(t["emitters"] for t in train)
                 or f["max_caps"] > max(t["max_caps"] for t in train)
                 or f["side"] not in {t["side"] for t in train},
                 "missing structural train/test shift")


def _fixture(open_wall=False, unlisted=False, escape=False):
    """Small hand-checkable scene. Its expected labels are not made by solve."""
    inp = np.zeros((14, 16), dtype=np.int8)
    inp[:2, 11:14] = [[1, 6 if unlisted else 2, 3], [5, 7 if unlisted else 3, 4]]
    for p in _diamond((3, 3), 2):
        inp[p] = 1
    for p in ((4, 6), (5, 5), (6, 4)):
        inp[p] = 2
    if not escape:
        for p in ((7, 9), (8, 8), (9, 7)):
            if not open_wall or p != (8, 8):
                inp[p] = 2
    expected = inp.copy()
    if escape:
        rays = [(4 + t, 6 + t) for t in range(1, 10)] + [(6 + t, 4 + t) for t in range(1, 8)]
    else:
        rays = [(5, 7), (6, 8), (7, 5), (8, 6)]
    for p in rays:
        expected[p] = 2
    if not open_wall and not unlisted and not escape:
        for p in ((5, 6), (6, 5), (6, 6), (6, 7), (7, 6), (7, 7), (7, 8), (8, 7)):
            expected[p] = 3
    return inp, expected


def self_test():
    """Bounded streaming suite: 42 seeds, all count choices, explicit contrasts.

    No sampling payloads or execution traces are saved. Only aggregate features
    and hashes are retained in memory; the largest live puzzle has 14 pairs.
    """
    for args in ({}, {"open_wall": True}, {"unlisted": True}, {"escape": True}):
        inp, expected = _fixture(**args)
        for turns in range(4):
            for shift in (0, 7):
                x, y = (np.rot90(g, turns) + shift for g in (inp, expected))
                x, y = x % 10, y % 10
                untouched = x.copy()
                assert np.array_equal(solve(x), y)
                assert np.array_equal(x, untouched)
    # Previously completed walls do not trigger any dye, even when mapped.
    inp, expected = _fixture()
    sealed = inp.copy()
    for p in ((5, 7), (6, 8), (7, 5), (8, 6)):
        sealed[p] = 2
    assert np.array_equal(solve(sealed), sealed)
    # A complete NW cap touching both edges contributes no new rail cells.
    # Its middle must not grow into (0,0); the SE channel is still informative.
    inp, expected = _fixture()
    for p in ((2, 0), (1, 1), (0, 2)):
        inp[p] = expected[p] = 6
    assert np.array_equal(solve(inp), expected) and solve(inp)[0, 0] == 0
    # A differently colored stopping wall has mixed ownership and receives no dye.
    inp, expected = _fixture(open_wall=True)
    for p in ((7, 9), (8, 8), (9, 7)):
        inp[p] = expected[p] = 6
    assert np.array_equal(solve(inp), expected)

    hashes, shapes, geometries = set(), set(), set()
    coverage = {"radii": set(), "emitters": set(), "ports": set(), "sides": set(),
                "backgrounds": set(), "key_lengths": set()}
    unequal = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                _, a = _interpret(inp)
                f = _features(inp, out, a)
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                shapes.add(inp.shape)
                geometries.add((inp.shape, tuple((e["center"], e["radius"], tuple(e["caps"])) for e in a["emitters"])))
                coverage["radii"].update(f["radii"])
                coverage["emitters"].add(f["emitters"])
                coverage["ports"].add(f["ports"])
                coverage["sides"].add(f["side"])
                coverage["backgrounds"].add(f["background"])
                coverage["key_lengths"].add(f["key_length"])
                unequal += f["unequal"]
                assert np.array_equal(solve(inp), out)
        hashes.add(digest.digest())
    assert len(hashes) == 42 and len(shapes) >= 40 and len(geometries) >= 150
    assert coverage["radii"] == {2, 3, 4}
    assert coverage["emitters"] == {1, 2, 3}
    assert {1, 2, 3, 4} <= coverage["ports"]
    assert coverage["sides"] == set(SIDES)
    assert coverage["backgrounds"] == set(range(10))
    assert coverage["key_lengths"] == set(range(3, 8)) and unequal >= 10
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(12345 + nt * 5 + ne, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(_fingerprint(x[k]) == _fingerprint(y[k]) for s in a
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    pair = broken["test"][0]
    p = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][p] = pair["input"][p]
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged label accepted")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "shapes": len(shapes),
            "geometry_signatures": len(geometries), "unequal_endpoint_lengths": unequal,
            "coverage": {k: sorted(v) for k, v in coverage.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
