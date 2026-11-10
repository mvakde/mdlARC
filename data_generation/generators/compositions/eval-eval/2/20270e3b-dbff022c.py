"""Cooperative stained glass: registration, ownership, and mapped consensus.

Only standard-library modules and NumPy are used. No files are read or written
by the public API, and no parent implementation is imported.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad grid colors")
    return a.astype(np.int8, copy=False)


def _count(value, low, high, name):
    _require(not isinstance(value, (bool, np.bool_)) and
             isinstance(value, (int, np.integer)), name + " must be an integer")
    _require(low <= int(value) <= high, name + " out of range")
    return int(value)


def _components(mask, diagonal=False):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    steps = _EIGHT if diagonal else _FOUR
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _signature(grid):
    a = _grid(grid)
    return a.shape, a.tobytes()


def _read_key(grid, cells):
    t, b, l, r = _bounds(cells)
    h, w = b - t + 1, r - l + 1
    _require(len(cells) == h * w, "key is not dense")
    _require(len({int(grid[q]) for q in cells}) >= 2, "key must be multicolor")
    choices = []
    if h == 2 and w >= 3:
        if t == 0:
            choices.append(("top", grid[t, l:r+1], grid[b, l:r+1]))
        if b == grid.shape[0] - 1:
            choices.append(("bottom", grid[b, l:r+1], grid[t, l:r+1]))
    if w == 2 and h >= 3:
        if l == 0:
            choices.append(("left", grid[t:b+1, l], grid[t:b+1, r]))
        if r == grid.shape[1] - 1:
            choices.append(("right", grid[t:b+1, r], grid[t:b+1, l]))
    _require(len(choices) == 1, "key must have one boundary-to-inward direction")
    side, sources, fills = choices[0]
    sources, fills = tuple(map(int, sources)), tuple(map(int, fills))
    _require(len(set(sources)) == len(sources), "duplicate key sources")
    _require(all(c not in (0, 9) for c in sources + fills), "reserved key color")
    return dict(zip(sources, fills)), side


def _parse(input_grid):
    grid = _grid(input_grid)
    components = _components(grid != 0, True)
    _require(len(components) == 3, "need one key and two connected fragments")
    key_parts = [p for p in components if not any(grid[q] == 9 for q in p)]
    _require(len(key_parts) == 1, "need one isolated key")
    mapping, side = _read_key(grid, key_parts[0])
    parts = [p for p in components if p is not key_parts[0]]
    bars = []
    for part in parts:
        bar = {q for q in part if grid[q] == 9}
        _require(len(bar) >= 2, "guides need at least two cells")
        t, b, l, r = _bounds(bar)
        _require((t == b and len(bar) == r-l+1) or
                 (l == r and len(bar) == b-t+1), "guide is not a straight bar")
        bars.append(bar)
    _require(len(bars[0]) == len(bars[1]), "incongruent guides")
    axis = 0 if len({r for r, c in bars[0]}) == 1 else 1
    records = []
    staged = {}
    for part, bar in zip(parts, bars):
        converted = {(p[axis], p[1-axis]) for p in bar}
        _require(len({r for r, c in converted}) == 1, "nonparallel guides")
        br, _, bc, _ = _bounds(converted)
        walls = {(q[axis]-br, q[1-axis]-bc): int(grid[q]) for q in part if grid[q] != 9}
        _require(bool(walls), "empty wall fragment")
        negative = all(r < 0 for r, c in walls)
        positive = all(r > 0 for r, c in walls)
        _require(negative != positive, "walls must lie strictly on one guide side")
        row = -1 if negative else 1
        _require(any((row, c) in walls for c in range(len(bar))), "unbacked guide")
        records.append((negative, walls, part))
    _require(records[0][0] != records[1][0], "backing sides must oppose")
    records.sort(key=lambda x: not x[0])
    for owner, (_, walls, part) in enumerate(records):
        for q in part:
            if grid[q] != 9:
                staged[q] = (int(grid[q]), owner)
    return {"mapping": mapping, "key_side": side, "axis": axis,
            "length": len(bars[0]), "records": records, "staged": staged}


def _aligned(parsed, seam=-1):
    fields = {}
    for owner, (_, walls, _) in enumerate(parsed["records"]):
        for (r, c), color in walls.items():
            q = (r + (seam if owner else 0), c)
            _require(q not in fields, "conflicting assembled walls")
            fields[q] = (color, owner)
    return fields


def _canvas(fields):
    t, b, l, r = _bounds(fields)
    # Keep a real exterior frame; crop boundaries never supply walls.
    ink = np.zeros((b-t+3, r-l+3), dtype=np.int8)
    owners = np.full(ink.shape, -1, dtype=np.int8)
    for (y, x), (color, owner) in fields.items():
        ink[y-t+1, x-l+1] = color
        owners[y-t+1, x-l+1] = owner
    return ink, owners, (t, l)


def _render(fields, mapping, *, axis=0, cooperative=True, policy="consensus",
            diagonal_background=False):
    ink, owners, origin = _canvas(fields)
    out, regions = ink.copy(), []
    h, w = ink.shape
    for part in _components(ink == 0, diagonal_background):
        if any(r in (0, h-1) or c in (0, w-1) for r, c in part):
            continue
        boundary = {(r+dr, c+dc) for r, c in part for dr, dc in _FOUR
                    if ink[r+dr, c+dc] != 0}
        colors = {int(ink[q]) for q in boundary}
        own = {int(owners[q]) for q in boundary}
        mapped = colors <= mapping.keys()
        destinations = {mapping[c] for c in colors if c in mapping}
        eligible = (own == {0, 1} or not cooperative) and mapped and len(destinations) == 1
        fill = next(iter(destinations)) if eligible else None
        if policy == "none":
            fill = None
        elif policy == "first" and own == {0, 1}:
            # Deliberately wrong: ignore the veto from conflicting/unlisted walls.
            fill = next((mapping[c] for c in sorted(colors) if c in mapping), None)
        elif policy == "recursive" and fill is not None:
            visited = set()
            while fill in mapping and fill not in visited:
                visited.add(fill)
                fill = mapping[fill]
        if fill is not None:
            for q in part:
                out[q] = fill
        regions.append({"cells": part, "boundary": boundary, "colors": colors,
                        "owners": own, "mapped": mapped, "destinations": destinations,
                        "eligible": eligible, "fill": fill})
    result = out[1:-1, 1:-1].copy()
    if axis:
        result = result.T.copy()
    return result, regions


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete result from input pixels alone; no cached state."""
    parsed = _parse(input_grid)
    return _render(_aligned(parsed), parsed["mapping"], axis=parsed["axis"])[0]


def _open_witness(fields, mapping):
    """A sealable exterior one-cell doorway between compatible fragment arcs."""
    ink, owners, (t, l) = _canvas(fields)
    exterior = set()
    for part in _components(ink == 0):
        if (0, 0) in part:
            exterior = part
            break
    base, _ = _render(fields, mapping)
    base_paint = np.count_nonzero(base) - len(fields)
    for r, c in sorted(exterior):
        if r == 0 or c == 0 or r == ink.shape[0]-1 or c == ink.shape[1]-1:
            continue
        neighbors = [(r+dr, c+dc) for dr, dc in _EIGHT if ink[r+dr, c+dc] != 0]
        colors = {int(ink[q]) for q in neighbors}
        if {int(owners[q]) for q in neighbors} != {0, 1} or not colors <= mapping.keys():
            continue
        if len({mapping[c] for c in colors}) != 1:
            continue
        q = neighbors[0]
        sealed = dict(fields)
        sealed[(r+t-1, c+l-1)] = (int(ink[q]), int(owners[q]))
        candidate, _ = _render(sealed, mapping)
        if np.count_nonzero(candidate) - len(sealed) > base_paint:
            return True
    return False


def _analyze(pair, details=False):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    parsed = _parse(inp)
    fields, mapping = _aligned(parsed), parsed["mapping"]
    expected, regions = _render(fields, mapping, axis=parsed["axis"])
    _require(np.array_equal(out, expected), "wrong assembly, consensus, fill, or crop")
    _require(not np.any(out == 9), "guide leaked into output")
    ordinary, owners, _ = _canvas(fields)
    canonical_out = out.T if parsed["axis"] else out
    ordinary = ordinary[1:-1, 1:-1]
    _require(out.size <= 900 and all(n <= 30 for n in out.shape), "output too large")
    _require(np.array_equal(canonical_out[ordinary != 0], ordinary[ordinary != 0]), "wall loss")
    _require(int(np.count_nonzero(ordinary)) == len(fields), "wall collision")
    filled = [p for p in regions if p["eligible"]]
    shared = [p for p in regions if p["owners"] == {0, 1}]
    old = [p for p in regions if len(p["owners"]) == 1 and p["mapped"] and
           len(p["destinations"]) == 1 and len(p["colors"]) > 1]
    _require(any(len(p["colors"]) > 1 for p in filled), "missing mixed-source cooperative fill")
    _require(any(not p["eligible"] for p in shared), "missing shared veto contrast")
    _require(bool(old), "missing old mapped-equivalent multicolor window")
    info = {"input_shape": inp.shape, "output_shape": out.shape,
            "axis": parsed["axis"], "guide_length": parsed["length"],
            "key_side": parsed["key_side"], "key_length": len(mapping),
            "walls": len(fields), "regions": len(regions),
            "filled_regions": len(filled), "painted_cells": sum(len(p["cells"]) for p in filled),
            "mapped_veto": any(p["mapped"] and len(p["destinations"]) > 1 for p in shared),
            "unmapped_veto": any(not p["mapped"] for p in shared),
            "old_multicolor": bool(old),
            "irregular_fill": any(len(p["cells"]) !=
                (_bounds(p["cells"])[1]-_bounds(p["cells"])[0]+1) *
                (_bounds(p["cells"])[3]-_bounds(p["cells"])[2]+1) for p in filled)}
    if details:
        eight = _render(fields, mapping, axis=parsed["axis"], diagonal_background=True)[0]
        info["four_vs_eight"] = not np.array_equal(eight, out)
        info["open_bay"] = _open_witness(fields, mapping)
    return info


def _loop(rng, kind, small=False):
    if kind == "diamond":
        radius = 2 if small else int(rng.integers(2, 4))
        shell = {(r, c) for r in range(-radius, radius+1) for c in range(2*radius+1)
                 if abs(r) + abs(c-radius) == radius}
        cavity = {(r, c) for r in range(-radius+1, radius) for c in range(1, 2*radius)
                  if abs(r) + abs(c-radius) < radius}
    else:
        up, down = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        width = int(rng.integers(1 if small else 2, 4 if small else 5))
        cavity = set()
        for r in range(-up, down+1):
            lo, hi = 1, width
            if r not in (0, 1) and width >= 3:
                lo += int(rng.integers(2))
                hi -= int(rng.integers(2))
            cavity.update((r, c) for c in range(lo, hi+1))
        steps = _EIGHT if rng.random() < .4 else _FOUR
        shell = {(r+dr, c+dc) for r, c in cavity for dr, dc in steps} - cavity
    # Seam rows span the whole width, fixing the registration tangent origin.
    return shell, cavity


def _geometry(rng, n, length, palette, lesson, open_bay):
    a, b, c, d, e = palette[:5]
    extra = int(rng.integers(3))
    sources = list(palette[:4]) + list(palette[5:5+extra])
    mapping = {a: d, b: d, c: e, d: e}
    for source in sources[4:]:
        mapping[source] = int(rng.choice(palette))
    fields, room_data, x = {}, [], 0
    for index in range(n):
        kind = "diamond" if index == 0 and lesson == 1 else str(rng.choice(["profile", "profile", "diamond"]))
        shell, cavity = _loop(rng, kind, small=n == 4)
        top, bottom, left, right = _bounds(shell)
        # _loop has left zero at the seam; translation never changes its shape.
        _require(left == 0, "unexpected loop origin")
        upper = a if rng.random() < .5 else b
        lower = b if upper == a else a
        if index == 1:
            lower = e if lesson == 1 else c
        elif index >= 2 and not (open_bay and index == n-1):
            lower = int(rng.choice([a, b, c, e]))
        shifted = {(r, q+x) for r, q in shell}
        hole = {(r, q+x) for r, q in cavity}
        if open_bay and index == n-1:
            # Open the outermost wall directly into unbounded exterior space.
            door = max(q for r, q in shifted if r == 0)
            shifted.remove((0, door))
        for r, q in shifted:
            fields[(r, q)] = (upper if r <= 0 else lower, 0 if r <= 0 else 1)
        room_data.append((shifted, hole, upper, lower))
        x += right + 1 + int(rng.integers(1, 3))
    width = max(q for r, q in fields) + 1
    _require(width <= 26, "network too wide")
    top = min(r for r, q in fields) - 2
    bottom = max(r for r, q in fields) + 2
    for q in range(width):
        fields[(top, q)] = (a if rng.random() < .5 else b, 0)
        fields[(bottom, q)] = (a if rng.random() < .5 else b, 1)
    for shell, hole, upper, lower in room_data:
        first, last, _, _ = _bounds(shell)
        candidates = sorted(q for r, q in shell if r == first)
        stem = int(rng.choice(candidates))
        for r in range(top, first+1):
            fields[(r, stem)] = (upper, 0)
        candidates = sorted(q for r, q in shell if r == last)
        stem = int(rng.choice(candidates))
        for r in range(last, bottom+1):
            fields[(r, stem)] = (lower, 1)
    # A pre-existing multicolor window carried entirely by just one fragment.
    owner = int(rng.integers(2))
    if rng.random() < .5:
        radius = int(rng.integers(1, 3))
        w = 2*radius+1
        left = int(rng.integers(width-w+1))
        center = top-radius if owner == 0 else bottom+radius
        old_shell = {(r, q+left) for r in range(center-radius, center+radius+1)
                     for q in range(w) if abs(r-center)+abs(q-radius) == radius}
        old_hole = {(r, q+left) for r in range(center-radius+1, center+radius)
                    for q in range(1, w-1) if abs(r-center)+abs(q-radius) < radius}
    else:
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        left = int(rng.integers(width-w+1))
        first = top-h+1 if owner == 0 else bottom
        old_shell = {(r+first, q+left) for r in range(h) for q in range(w)
                     if r in (0, h-1) or q in (0, w-1)}
        old_hole = {(r+first, q+left) for r in range(1, h-1) for q in range(1, w-1)}
    for i, q in enumerate(sorted(old_shell)):
        fields[q] = (a if i % 2 == 0 else b, owner)
    # Make both equivalent colors visible on the actual old cavity boundary.
    old_boundary = sorted({(r+dr, q+dc) for r, q in old_hole for dr, dc in _FOUR} & old_shell)
    for i, q in enumerate(old_boundary):
        fields[q] = (a if i % 2 == 0 else b, owner)
    expected, _ = _render(fields, mapping)
    t, _, l, _ = _bounds(fields)
    def at(q):
        return int(expected[q[0]-t, q[1]-l])
    _require(all(at(q) == d for q in room_data[0][1]), "construction lost shared paint witness")
    _require(all(at(q) == 0 for q in room_data[1][1]), "construction lost veto witness")
    _require(all(at(q) == 0 for q in old_hole), "construction filled an old window")
    if open_bay:
        _require(all(at(q) == 0 for q in room_data[-1][1]), "open bay is not exterior")
    groups = []
    for own in (0, 1):
        group = {q: color for q, (color, index) in fields.items() if index == own}
        for col in range(length):
            group[(1 if own == 0 else 0, col)] = 9
        gt, gb, gl, gr = _bounds(group)
        groups.append(({(r-gt, q-gl): color for (r, q), color in group.items()},
                       gb-gt+1, gr-gl+1))
    return groups, mapping, expected


def _stage(rng, groups, mapping):
    side = str(rng.choice(["top", "bottom", "left", "right"]))
    limh = 27 if side in ("top", "bottom") else 30
    limw = 27 if side in ("left", "right") else 30
    order = list(map(int, rng.permutation(2)))
    first, second = [groups[i] for i in order]
    _, ah, aw = first
    _, bh, bw = second
    gap, jitter = int(rng.integers(1, 4)), int(rng.integers(-2, 3))
    options = [(ah+gap, jitter, "vertical"), (jitter, aw+gap, "horizontal"),
               (ah+gap, aw+gap, "diagonal")]
    legal = []
    for dy, dx, mode in options:
        t, b = min(0, dy), max(ah, dy+bh)
        l, r = min(0, dx), max(aw, dx+bw)
        if b-t <= limh and r-l <= limw:
            legal.append((dy, dx, mode, t, b, l, r))
    if not legal:
        # Maximal-width networks still fit vertically without tangent jitter.
        dy, dx = ah+1, 0
        if dy+bh <= limh and max(aw, bw) <= limw:
            legal.append((dy, dx, "vertical", 0, dy+bh, 0, max(aw, bw)))
    _require(bool(legal), "no bounded fragment staging")
    dy, dx, mode, t, b, l, r = legal[int(rng.integers(len(legal)))]
    height, width = b-t, r-l
    pads = []
    for spare in (limh-height, limw-width):
        p = int(rng.integers(min(2, spare)+1))
        q = int(rng.integers(min(2, spare-p)+1))
        pads.extend((p, q))
    pt, pb, pl, pr = pads
    h = height+pt+pb+(3 if side in ("top", "bottom") else 0)
    w = width+pl+pr+(3 if side in ("left", "right") else 0)
    grid = np.zeros((h, w), dtype=np.int8)
    base_r = pt-t+(3 if side == "top" else 0)
    base_c = pl-l+(3 if side == "left" else 0)
    for (cells, _, _), (dr, dc) in zip((first, second), ((0, 0), (dy, dx))):
        for (r0, c0), color in cells.items():
            grid[r0+base_r+dr, c0+base_c+dc] = color
    entries = list(mapping.items())
    entries = [entries[int(i)] for i in rng.permutation(len(entries))]
    axis = w if side in ("top", "bottom") else h
    start = int(rng.integers(axis-len(entries)+1))
    for i, (source, fill) in enumerate(entries):
        p = start+i
        if side == "top":
            grid[0, p], grid[1, p] = source, fill
        elif side == "bottom":
            grid[-1, p], grid[-2, p] = source, fill
        elif side == "left":
            grid[p, 0], grid[p, 1] = source, fill
        else:
            grid[p, -1], grid[p, -2] = source, fill
    return grid


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and
             isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    interpolation = bool(rng.integers(2))
    base_turn = int(rng.integers(4))
    stable_palette = bool(rng.integers(2))
    common_palette = tuple(map(int, rng.permutation(np.arange(1, 9))))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else -1
            if split == "train":
                n = 3 if lesson == 0 else 2 if lesson == 1 else int(rng.integers(2, 5 if interpolation else 4))
                length = (2 if index % 2 == 0 else 4) if interpolation else int(rng.integers(2, 5))
                turn = (base_turn+index % 2+2*int(rng.integers(2))) % 4
            else:
                n = int(rng.integers(2, 5)) if interpolation else 4
                length = 3 if interpolation else int(rng.integers(2, 5))
                turn = int(rng.integers(4))
            open_bay = lesson == 0 or (n >= 3 and rng.random() < .4)
            for attempt in range(8):
                palette = common_palette if stable_palette else tuple(map(int, rng.permutation(np.arange(1, 9))))
                try:
                    groups, mapping, output = _geometry(rng, n, length, palette, lesson, open_bay)
                    grid = _stage(rng, groups, mapping)
                    pair = {"input": np.rot90(grid, turn).copy(),
                            "output": np.rot90(output, turn).copy()}
                    _analyze(pair)
                    sig = _signature(pair["input"])
                    if sig in seen:
                        continue
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
                except ValueError:
                    if attempt == 7:
                        raise
            else:
                raise ValueError("bounded unique construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _count(a, 2, 10, "num_train")
    _count(b, 1, 4, "num_test")
    _require(a > b, "invalid count relationship")
    seen, lessons = set(), []
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            sig = _signature(pair["input"])
            _require(sig not in seen, "duplicate puzzle input")
            seen.add(sig)
            info = _analyze(pair, details=split == "train" and i < 2)
            if split == "train":
                lessons.append(info)
    for feature in ("mapped_veto", "unmapped_veto", "old_multicolor", "four_vs_eight", "open_bay"):
        _require(any(p.get(feature, False) for p in lessons), "missing training lesson: " + feature)
    _require({p["axis"] for p in lessons} == {0, 1}, "training must show both guide axes")
    return True


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = _grid(pair[name])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed streaming survey, with genuine semantic contrasts."""
    signatures, shapes, output_shapes, structures = set(), set(), set(), set()
    sides, lengths, key_lengths, axes, paint_counts, region_counts = [set() for _ in range(6)]
    witnesses = set()
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                inp, out = pair["input"], pair["output"]
                _require(np.array_equal(solve(inp.copy()), out), "solve mismatch")
                info = _analyze(pair)
                shapes.add(inp.shape)
                output_shapes.add(out.shape)
                sides.add(info["key_side"])
                lengths.add(info["guide_length"])
                key_lengths.add(info["key_length"])
                axes.add(info["axis"])
                paint_counts.add(info["painted_cells"])
                region_counts.add(info["regions"])
                roles = {0: 0, 9: 1}
                normalized = []
                for color in inp.flat:
                    c = int(color)
                    if c not in roles:
                        roles[c] = len(roles)
                    normalized.append(roles[c])
                structures.add((inp.shape, bytes(normalized)))
                if split != "train":
                    continue
                p = _parse(inp)
                fields, mapping = _aligned(p), p["mapping"]
                alternatives = {
                    "without_lookup_consensus": _render(fields, mapping, axis=p["axis"], policy="first")[0],
                    "without_cooperation": _render(fields, mapping, axis=p["axis"], cooperative=False)[0],
                    "without_filling": _render(fields, mapping, axis=p["axis"], policy="none")[0],
                    "recursive_lookup": _render(fields, mapping, axis=p["axis"], policy="recursive")[0],
                    "eight_neighbor_background": _render(fields, mapping, axis=p["axis"], diagonal_background=True)[0],
                    "without_registration": _render(p["staged"], mapping, policy="none")[0],
                    "coincident_guides": _render(_aligned(p, 0), mapping, axis=p["axis"])[0],
                    "opposite_seam": _render(_aligned(p, 1), mapping, axis=p["axis"])[0],
                    "reverse_key": _render(fields, {v: k for k, v in mapping.items()}, axis=p["axis"])[0]}
                painted_guides = dict(fields)
                for row in (0, -1):
                    for col in range(p["length"]):
                        painted_guides.setdefault((row, col), (9, 0))
                alternatives["opaque_guides"] = _render(painted_guides, mapping, axis=p["axis"])[0]
                witnesses.update(name for name, wrong in alternatives.items() if not np.array_equal(wrong, out))
        if seed in (0, 17, 41):
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic seed")
    _require(len(signatures) == 42, "seed diversity failure")
    _require(len(structures) >= 190 and len(shapes) >= 40 and len(output_shapes) >= 30, "structural variation too small")
    _require(sides == {"top", "bottom", "left", "right"} and axes == {0, 1}, "orientation coverage")
    _require(lengths == {2, 3, 4} and key_lengths == {4, 5, 6}, "length coverage")
    # Two to four chambers, intervening regions and an old window yield 4..8
    # cavities before an optional exterior opening removes one of them.
    _require(len(paint_counts) >= 20 and len(region_counts) >= 5, "topology variation too small")
    _require(len(witnesses) == 10, "semantic alternative not distinguished")
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(100+train*4+test, train, test)
            _require(len(p["train"]) == train and len(p["test"]) == test, "explicit count failure")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(0, 2, 1)
    broken["test"][0]["output"] = broken["test"][0]["output"].copy()
    arr = broken["test"][0]["output"]
    q = tuple(np.argwhere(arr != 0)[0])
    arr[q] = 0
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("wall deletion accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "color_normalized_structures": len(structures), "key_sides": sorted(sides),
            "guide_lengths": sorted(lengths), "key_lengths": sorted(key_lengths),
            "painted_cell_counts": len(paint_counts), "cavity_counts": sorted(region_counts),
            "rejected_alternatives": sorted(witnesses)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
