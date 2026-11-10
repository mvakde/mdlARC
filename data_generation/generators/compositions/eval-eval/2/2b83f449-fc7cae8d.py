"""Calibrated terminal compaction: a visible frame controls a rewired network.

Only NumPy and the standard library are used. No parent code or files are loaded.
The solver deliberately leaves the extracted plate in its original orientation.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMAL = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
NEIGHBORS = ((-1, 0), (0, 1), (1, 0), (0, -1))
# These matrices act on (row, column) vectors; translations do not affect rank.
MATRICES = (
    ((1, 0), (0, 1)), ((0, -1), (1, 0)),
    ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
    ((1, 0), (0, -1)), ((-1, 0), (0, 1)),
    ((0, 1), (1, 0)), ((0, -1), (-1, 0)),
)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _adjacent(index):
    return SIDES[index], SIDES[(index + 1) % 4]


def _edge(a, side):
    return {"T": a[0, :], "R": a[:, -1], "B": a[-1, :], "L": a[:, 0]}[side]


def _matrix(inner, outer):
    """Unique signed coordinate frame from the two identity matches."""
    m = np.zeros((2, 2), dtype=int)
    for side, color in inner.items():
        target = next(s for s, c in outer.items() if c == color)
        m += np.outer(NORMAL[target], NORMAL[side])
    assert tuple(map(tuple, m.tolist())) in MATRICES
    return m


def _arrangement(outer_index, inner_index, reverse, markers):
    inner = dict(zip(_adjacent(inner_index), markers))
    order = markers[::-1] if reverse else markers
    outer = dict(zip(_adjacent(outer_index), order))
    return inner, outer, _matrix(inner, outer)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, component = [start], []
        while todo:
            p = todo.pop()
            component.append(p)
            for dr, dc in NEIGHBORS:
                q = (p[0] + dr, p[1] + dc)
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        components.append(component)
    return components


def _ordered_paths(mask):
    paths = []
    for component in _components(mask):
        cells = set(component)
        adjacency = {p: [(p[0] + dr, p[1] + dc) for dr, dc in NEIGHBORS
                         if (p[0] + dr, p[1] + dc) in cells] for p in cells}
        assert all(len(ns) in (1, 2) for ns in adjacency.values()), "branch or isolated cell"
        ends = sorted(p for p, ns in adjacency.items() if len(ns) == 1)
        assert len(ends) == 2, "closed circuit"
        path, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            nxt = [p for p in adjacency[current] if p != previous]
            assert len(nxt) == 1
            previous, current = current, nxt[0]
            path.append(current)
        assert len(path) == len(cells)
        paths.append(path)
    return paths


def _rewire(core, bg, track, bead, divider):
    """Parse horizontal lanes and apply simultaneous agreeing 3x3 writes."""
    h, w = core.shape
    assert h >= 3 and h % 2 == 1 and w >= 7
    assert np.all(np.isin(core[::2], (bg, track, bead)))
    assert np.all(np.isin(core[1::2], (bg, divider)))
    assert np.all(np.isin(core[::2, [0, w - 1]], (track, bead)))
    writes, centers, separator_counts, overlaps = {}, [], [], 0
    for r in range(1, h, 2):
        xs = np.flatnonzero(core[r] == divider).tolist()
        starts = [c for c in xs if c - 1 not in xs]
        separator_counts.append(len(starts))
        for left in starts:
            c = left + 1
            assert 2 <= c <= w - 3
            assert all(x in xs for x in (c - 1, c, c + 1)) and c + 2 not in xs
            assert np.all(np.isin(core[[r - 1, r + 1], c - 1:c + 2], (track, bead)))
            assert core[r - 1, c] == core[r + 1, c] == track
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    value = divider if cc == c else track
                    if (rr, cc) in writes:
                        assert writes[rr, cc] == value, "conflicting divider neighborhoods"
                        overlaps += 1
                    writes[rr, cc] = value
            centers.append((r, c))
    assert centers
    breaks = []
    for r in range(0, h, 2):
        xs = np.flatnonzero(core[r] == bg).tolist()
        assert all(2 <= c <= w - 3 for c in xs)
        assert all(b - a >= 3 for a, b in zip(xs, xs[1:]))
        breaks.extend((r, c) for c in xs)
    structure = core.copy()
    structure[structure == bead] = track
    for p, value in writes.items():
        structure[p] = value
    paths = _ordered_paths(structure == track)
    endpoints = {p for path in paths for p in (path[0], path[-1])}
    beads = set(map(tuple, np.argwhere(core == bead).tolist()))
    assert beads <= endpoints, "beads must start at terminals"
    for r, c in endpoints:
        assert r % 2 == 0
        assert c in (0, w - 1) or core[r, c - 1] == bg or core[r, c + 1] == bg
    return structure, paths, {"centers": centers, "breaks": breaks,
                              "separator_counts": separator_counts, "overlaps": overlaps}


def _decode_plate(plate, bg, missing_bead=None):
    colors = set(map(int, np.unique(plate)))
    if missing_bead is not None:
        colors.add(missing_bead)
    assert len(colors) == 4 and bg in colors
    loc = np.argwhere(plate != bg)
    r0, c0 = map(int, loc.min(axis=0))
    r1, c1 = map(int, loc.max(axis=0) + 1)
    crop = plate[r0:r1, c0:c1]
    matches = []
    for transpose in (False, True):
        core = crop.T if transpose else crop
        if core.shape[0] < 3 or core.shape[0] % 2 != 1 or core.shape[1] < 7:
            continue
        ds = set(map(int, np.unique(core[1::2]))) - {bg}
        if len(ds) != 1:
            continue
        divider = next(iter(ds))
        remaining = colors - {bg, divider}
        for track in sorted(remaining):
            bead = next(iter(remaining - {track}))
            try:
                structural, paths, metrics = _rewire(core, bg, track, bead, divider)
            except AssertionError:
                continue
            result = plate.copy()
            result[r0:r1, c0:c1] = structural.T if transpose else structural
            def to_plate(p):
                return (r0 + p[1], c0 + p[0]) if transpose else (r0 + p[0], c0 + p[1])
            converted = [[to_plate(p) for p in path] for path in paths]
            matches.append(dict(structural=result, paths=converted, background=bg,
                                track=track, bead=bead, divider=divider, axis=int(transpose),
                                lanes=(core.shape[0] + 1) // 2, width=core.shape[1],
                                padding=(r0, c0, plate.shape[0] - r1, plate.shape[1] - c1),
                                **metrics))
    assert len(matches) == 1, "ambiguous network roles or lane axis"
    return matches[0]


def _decode_input(input_grid):
    a = _grid(input_grid)
    assert min(a.shape) >= 8
    bg = int(a[1, 1])
    # A clear inner border distinguishes the local rulers from canvas controls.
    assert all(np.all(line == bg) for line in (a[1, 1:-1], a[-2, 1:-1],
                                               a[1:-1, 1], a[1:-1, -2]))
    outer = {}
    for side in SIDES:
        values = _edge(a, side)[1:-1]
        if len(np.unique(values)) == 1 and int(values[0]) != bg:
            outer[side] = int(values[0])
        else:
            assert np.all(values == bg)
    assert len(outer) == 2 and len(set(outer.values())) == 2
    assert any(set(outer) == set(_adjacent(i)) for i in range(4))
    bars = []
    for color in sorted(outer.values()):
        loc = np.argwhere(a[2:-2, 2:-2] == color) + 2
        assert len(loc) >= 3
        rows, cols = np.unique(loc[:, 0]), np.unique(loc[:, 1])
        if len(rows) == 1:
            assert len(loc) == cols[-1] - cols[0] + 1
            bars.append(("H", color, int(rows[0]), int(cols[0]), int(cols[-1])))
        else:
            assert len(cols) == 1 and len(loc) == rows[-1] - rows[0] + 1
            bars.append(("V", color, int(cols[0]), int(rows[0]), int(rows[-1])))
    horizontal = [b for b in bars if b[0] == "H"]
    vertical = [b for b in bars if b[0] == "V"]
    assert len(horizontal) == len(vertical) == 1
    _, hc, hr, hs, he = horizontal[0]
    _, vc, vcol, vs, ve = vertical[0]
    h, w = ve - vs + 1, he - hs + 1
    assert hr in (vs - 1, ve + 1) and vcol in (hs - 1, he + 1)
    r, c = vs, hs
    inner = {"T" if hr == vs - 1 else "B": hc, "L" if vcol == hs - 1 else "R": vc}
    assert r >= 2 and c >= 2 and r + h <= a.shape[0] - 2 and c + w <= a.shape[1] - 2
    plate = a[r:r + h, c:c + w].copy()
    assert not set(map(int, np.unique(plate))) & set(outer.values())
    # Reconstruct the entire permitted canvas. This rejects noise, stray guide
    # cells, alternate crops, and extra payloads without frequency assumptions.
    reconstructed = _canvas(plate, bg, inner, outer, a.shape, (r, c))
    assert np.array_equal(a, reconstructed), "invalid frame, rulers, moat or outside content"
    info = _decode_plate(plate, bg)
    info.update(plate=plate, inner=inner, outer=outer, matrix=_matrix(inner, outer),
                bounds=(r, c, h, w))
    return info


def _preferred(path, matrix):
    first = tuple(int(x) for x in matrix @ np.array(path[0]))
    last = tuple(int(x) for x in matrix @ np.array(path[-1]))
    return path if first > last else path[::-1]


def _targets(paths, beads, matrix, retain_ties=False):
    targets, single, collision, tie, tie_moved = set(), 0, 0, 0, 0
    for path in paths:
        here = beads & set(path)
        assert here <= {path[0], path[-1]}
        count = len(here)
        primary_equal = int((matrix @ (np.array(path[0]) - path[-1]))[0]) == 0
        if retain_ties and primary_equal:
            packed = here
        else:
            packed = set(_preferred(path, matrix)[:count])
        targets.update(packed)
        single += count == 1 and here != packed
        collision += count == 2 and len(path) > 2
        tie += bool(count and primary_equal)
        tie_moved += bool(count and primary_equal and here != packed)
    assert len(targets) == len(beads)
    return targets, dict(single=single, collision=collision, tie=tie, tie_moved=tie_moved)


def _answer(info):
    beads = set(map(tuple, np.argwhere(info["plate"] == info["bead"]).tolist()))
    targets, details = _targets(info["paths"], beads, info["matrix"])
    out = info["structural"].copy()
    for p in targets:
        out[p] = info["bead"]
    return out, beads, targets, details


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode every convention from this input, then compact in its local pose."""
    return _answer(_decode_input(input_grid))[0]


def _canvas(plate, bg, inner, outer, shape, origin):
    a = np.full(shape, bg, dtype=np.int8)
    # Fixed draw order makes shared control corners unambiguous but irrelevant.
    for side in SIDES:
        if side in outer:
            _edge(a, side)[:] = outer[side]
    r, c = origin
    h, w = plate.shape
    a[r:r + h, c:c + w] = plate
    for side, color in inner.items():
        if side == "T":
            a[r - 1, c:c + w] = color
        elif side == "B":
            a[r + h, c:c + w] = color
        elif side == "L":
            a[r:r + h, c - 1] = color
        else:
            a[r:r + h, c + w] = color
    return a


def _frame(rng, plate, palette, arrangement):
    bg, _, _, _, m0, m1 = palette
    inner, outer, _ = _arrangement(*arrangement, (m0, m1))
    h, w = plate.shape
    H = int(rng.integers(h + 5, min(30, h + 9) + 1))
    W = int(rng.integers(w + 5, min(30, w + 9) + 1))
    r = int(rng.integers(2 + ("T" in inner), H - h - ("B" in inner) - 1))
    c = int(rng.integers(2 + ("L" in inner), W - w - ("R" in inner) - 1))
    return _canvas(plate, bg, inner, outer, (H, W), (r, c))


def _base(lanes, width, centers, breaks, palette):
    bg, track, _, divider = palette[:4]
    a = np.full((2 * lanes - 1, width), bg, dtype=np.int8)
    a[::2] = track
    for r, c in centers:
        a[r, c - 1:c + 2] = divider
    for p in breaks:
        a[p] = bg
    return a


def _break_pool(lanes, width, centers, breaks):
    return [(r, c) for r in range(0, 2 * lanes - 1, 2) for c in range(2, width - 2)
            if all(abs(c - cc) >= 2 for rr, cc in centers if abs(rr - r) == 1)
            and all(abs(c - cc) >= 3 for rr, cc in breaks if rr == r)]


def _sample_plate(rng, palette, size, teaching=False):
    bg, track, bead, divider = palette[:4]
    if size == "small":
        lanes, width = int(rng.integers(3, 6)), int(rng.integers(7, 17))
    elif size == "large":
        lanes, width = int(rng.integers(6, 11)), int(rng.integers(14, 24))
    else:
        lanes, width = int(rng.integers(3, 11)), int(rng.integers(7, 25))
    for attempt in range(36):
        centers, previous = [], []
        for row in range(1, 2 * lanes - 1, 2):
            chosen = []
            wanted = int(rng.integers(min(4, (width - 1) // 4) + 1))
            for raw in rng.permutation(np.arange(2, width - 2)):
                c = int(raw)
                if len(chosen) == wanted:
                    break
                if all(abs(c - p) >= 4 for p in chosen) and all(abs(c - p) >= 2 for p in previous):
                    chosen.append(c)
            centers.extend((row, c) for c in sorted(chosen))
            previous = chosen
        if not centers:
            centers = [(_pick(rng, list(range(1, 2 * lanes - 1, 2))), int(rng.integers(2, width - 2)))]
        if attempt == 35:
            # A size-preserving constructive fallback: one randomly located
            # switch and otherwise intact lanes, never an unbounded retry.
            centers = [(_pick(rng, list(range(1, 2 * lanes - 1, 2))), int(rng.integers(2, width - 2)))]
        breaks = []
        raw = _base(lanes, width, centers, breaks, palette)
        structure = raw.copy()
        for r, c in centers:
            structure[r - 1:r + 2, c - 1:c + 2] = track
            structure[r - 1:r + 2, c] = divider
        # Open actual cycles, wherever they occur, with visible lane breaks.
        valid = True
        for component in _components(structure == track):
            cells = set(component)
            degrees = [sum((r + dr, c + dc) in cells for dr, dc in NEIGHBORS) for r, c in cells]
            if any(d not in (1, 2) for d in degrees):
                valid = False
                break
            if all(d == 2 for d in degrees):
                candidates = [p for p in _break_pool(lanes, width, centers, breaks) if p in cells]
                if not candidates:
                    valid = False
                    break
                breaks.append(_pick(rng, candidates))
        if not valid:
            continue
        for _ in range(int(rng.integers(5)) if attempt != 35 else 0):
            pool = _break_pool(lanes, width, centers, breaks)
            if pool:
                breaks.append(_pick(rng, pool))
        raw = _base(lanes, width, centers, breaks, palette)
        try:
            _, paths, _ = _rewire(raw, bg, track, bead, divider)
        except AssertionError:
            continue
        if teaching and not (any(p[0][0] == p[-1][0] for p in paths)
                             and any(p[0][1] == p[-1][1] for p in paths)):
            continue
        if rng.random() < .5:
            raw = np.fliplr(raw)
        raw = np.rot90(raw, int(rng.integers(4))).copy()
        # Guide lengths, rather than ink bounds, retain these blank strips.
        padding = [(0, 0), (0, 0)]
        if teaching or rng.random() < .65:
            axis = int(rng.integers(2))
            total = int(rng.integers(1, min(2, 25 - raw.shape[axis]) + 1))
            before = int(rng.integers(total + 1))
            padding[axis] = (before, total - before)
        raw = np.pad(raw, padding, constant_values=bg)
        assert max(raw.shape) <= 25
        return raw
    raise RuntimeError("bounded network construction exhausted")


def _populate(rng, plate, palette, matrix, alternate=None):
    info = _decode_plate(plate, palette[0], missing_bead=palette[2])
    paths = info["paths"]
    beads = set()
    probability = float(rng.uniform(.12, .95))
    for path in paths:
        beads.update(p for p in (path[0], path[-1]) if rng.random() < probability)
    if alternate is not None:
        divergent = [p for p in paths if len(p) >= 4 and _preferred(p, matrix)[0] != _preferred(p, alternate)[0]]
        assert divergent
        single = _pick(rng, divergent)
        beads.difference_update((single[0], single[-1]))
        beads.add(_preferred(single, matrix)[-1])
        doubles = [p for p in paths if p is not single and len(p) > 2]
        collision = _pick(rng, doubles)
        beads.update((collision[0], collision[-1]))
        ties = [(p, m) for p in paths if p is not single and p is not collision
                for m in (matrix, alternate)
                if int((m @ (np.array(p[0]) - p[-1]))[0]) == 0]
        if ties:
            path, m = _pick(rng, ties)
            beads.difference_update((path[0], path[-1]))
            beads.add(_preferred(path, m)[-1])
        # A strict priority witness disagrees with the sum of virtual row and
        # column. Make it active without disturbing single/collision witnesses.
        priority = _priority_paths(paths, (matrix, alternate))
        assert priority
        path = _pick(rng, priority)
        if not (beads & {path[0], path[-1]}):
            beads.add(path[int(rng.integers(2)) * (len(path) - 1)])
    if not beads:
        path = _pick(rng, paths)
        beads.add(path[0])
    out = plate.copy()
    for p in beads:
        out[p] = palette[2]
    return out


def _priority_paths(paths, matrices):
    result = []
    for path in paths:
        for matrix in matrices:
            delta = matrix @ (np.array(path[0]) - path[-1])
            if delta[0] * delta[1] < 0 and abs(delta[1]) > abs(delta[0]):
                result.append(path)
                break
    return result


def _measure(inp):
    info = _decode_input(inp)
    answer, beads, targets, details = _answer(info)
    original_paths = _ordered_paths(np.isin(info["plate"], (info["track"], info["bead"])))
    original_targets, _ = _targets(original_paths, beads, info["matrix"])
    fixed_targets, _ = _targets(info["paths"], beads, np.eye(2, dtype=int))
    details.update(topology_sensitive=targets != original_targets,
                   frame_sensitive=targets != fixed_targets,
                   priority_witness=any(beads & {p[0], p[-1]}
                                        for p in _priority_paths(info["paths"], (info["matrix"],))),
                   bead_count=len(beads), path_count=len(info["paths"]),
                   empty_paths=sum(not (beads & set(p)) for p in info["paths"]))
    return info, answer, details


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)[:6]))
    stable = bool(rng.random() < .25)
    size_regime = int(rng.integers(3))
    train_size = ("all", "small", "large")[size_regime]
    test_size = ("all", "large", "small")[size_regime]
    all_specs = [(o, i, rev) for o in range(4) for i in range(4) for rev in (False, True)]
    # Opposite-sign target normals make a correspondence reversal observable
    # on pure-axis endpoint pairs. All corner combinations remain legal later.
    teaching = [s for s in all_specs if s[0] in (0, 2)
                and tuple(map(tuple, _arrangement(*s, palette[4:])[2].tolist())) in (MATRICES[1], MATRICES[3])]
    first = _pick(rng, teaching)
    second = (first[0], first[1], not first[2])
    held = _pick(rng, [(o, i) for o in range(4) for i in range(4) if (o, i) != first[:2]])
    m0 = _arrangement(*first, palette[4:])[2]
    m1 = _arrangement(*second, palette[4:])[2]
    for attempt in range(30):
        if attempt < 29:
            plate = _sample_plate(rng, palette, train_size, teaching=True)
        else:
            # Guaranteed teaching topology: two staggered switches provide a
            # diagonal-ended path, while intact lanes and side returns provide
            # both kinds of primary tie. Its pose is visibly calibrated.
            n, width = (6, 14) if train_size == "large" else (4, 11)
            base = _base(n, width, [(1, 3), (3, 5)], [], palette)
            candidates = [np.rot90(a, k).copy() for a in (base, np.fliplr(base)) for k in range(4)]
            plate = next(a for a in candidates if _priority_paths(
                _decode_plate(a, palette[0], palette[2])["paths"], (m0, m1)))
            plate = np.pad(plate, ((1, 0), (0, 0)), constant_values=palette[0])
        bare_info = _decode_plate(plate, palette[0], palette[2])
        if not _priority_paths(bare_info["paths"], (m0, m1)):
            continue
        plate = _populate(rng, plate, palette, m0, m1)
        inp0 = _frame(rng, plate, palette, first)
        # Exact controlled contrast: only the two outer marker identities swap.
        d0 = _decode_input(inp0)
        outer1 = {s: palette[5] if color == palette[4] else palette[4] for s, color in d0["outer"].items()}
        inp1 = _canvas(plate, palette[0], d0["inner"], outer1, inp0.shape, d0["bounds"][:2])
        records = [_measure(inp0), _measure(inp1)]
        features = ("single", "collision", "tie_moved", "priority_witness", "topology_sensitive", "frame_sensitive")
        if all(any(d[2][f] for d in records) for f in features) and not np.array_equal(records[0][1], records[1][1]):
            break
    else:
        raise RuntimeError("bounded teaching construction exhausted")
    puzzle = {"train": [{"input": inp0, "output": records[0][1]},
                         {"input": inp1, "output": records[1][1]}], "test": []}
    seen = {_key(inp0), _key(inp1)}
    for split, count in (("train", nt - 2), ("test", ne)):
        options = [s for s in all_specs if (s[:2] == held) == (split == "test")]
        for _ in range(count):
            for attempt in range(24):
                pal = palette if stable else tuple(map(int, rng.permutation(10)[:6]))
                spec = _pick(rng, options)
                m = _arrangement(*spec, pal[4:])[2]
                plate = _sample_plate(rng, pal, train_size if split == "train" else test_size)
                plate = _populate(rng, plate, pal, m)
                inp = _frame(rng, plate, pal, spec)
                if _key(inp) not in seen:
                    seen.add(_key(inp))
                    puzzle[split].append({"input": inp, "output": solve(inp)})
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    # Teaching witnesses need not occupy privileged positions in a puzzle.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info, expected, details = _measure(inp)
            assert out.shape == info["plate"].shape, "output pose and guide extent are retained"
            assert np.array_equal(out, expected), "incorrect compaction or rewiring"
            assert np.count_nonzero(out == info["bead"]) == details["bead_count"]
            # Per-component conservation, contiguous terminal runs, and exactly
            # preserved nonbead structure are stronger than a grid shape check.
            for path in info["paths"]:
                n_in = sum(info["plate"][p] == info["bead"] for p in path)
                occupied = {p for p in path if out[p] == info["bead"]}
                assert occupied == set(_preferred(path, info["matrix"])[:n_in])
            clean = out.copy()
            clean[clean == info["bead"]] = info["track"]
            assert np.array_equal(clean, info["structural"])
            records[split].append((info, details))
    train = records["train"]
    for feature in ("single", "collision", "tie_moved", "priority_witness", "topology_sensitive", "frame_sensitive"):
        assert any(d[feature] for _, d in train), f"missing training witness: {feature}"
    assert any(any(info["padding"]) for info, _ in train), "guide-defined blank extent not demonstrated"
    contrasts = [(a, b) for i, (a, _) in enumerate(train) for b, _ in train[i + 1:]
                 if np.array_equal(a["plate"], b["plate"]) and a["inner"] == b["inner"]
                 and set(a["outer"]) == set(b["outer"]) and a["outer"] != b["outer"]
                 and not np.array_equal(_answer(a)[0], _answer(b)[0])]
    assert contrasts, "missing exact identity-correspondence contrast"
    train_arrangements = {(tuple(sorted(a["outer"])), tuple(sorted(a["inner"]))) for a, _ in train}
    assert all((tuple(sorted(a["outer"])), tuple(sorted(a["inner"]))) not in train_arrangements
               for a, _ in records["test"]), "test arrangement should be held out"


def self_test():
    """42 streamed seeds, meaningful invariants, deterministic and mutation checks.

    No grids, traces or logs are saved. One fixture, the current sample, and
    at most one temporary check copy are retained.
    Returned measurements are descriptive, not independent semantic approval.
    """
    # A small hand-worked regression makes the new tie convention and the
    # unchanged output pose explicit, independently of sampled labels.
    plate = np.full((5, 9), 0, dtype=np.int8)
    plate[::2] = 8
    plate[1, 3:6] = 7
    for p in ((2, 0), (0, 8), (2, 8), (4, 0)):
        plate[p] = 3
    inner = {"T": 1, "L": 2}
    for outer, rows in (
        ({"T": 2, "R": 1}, ("388878833", "000878000", "888878888", "000000000", "888888883")),
        ({"T": 1, "R": 2}, ("888878888", "000878000", "388878833", "000000000", "388888888")),
    ):
        inp = _canvas(plate, 0, inner, outer, (10, 14), (3, 3))
        expected = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
        assert np.array_equal(solve(inp), expected)
    shapes, core_shapes, layouts, palettes, matrices, axes = set(), set(), set(), set(), set(), set()
    lanes, dividers, breaks, bead_counts, padding = set(), set(), set(), set(), set()
    features = Counter()
    digests = set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed == 0:
            first = puzzle
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info, expected, details = _measure(pair["input"])
                assert np.array_equal(expected, pair["output"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                digest.update(pair["input"].tobytes())
                shapes.add(pair["input"].shape)
                core_shapes.add(info["plate"].shape)
                roles = (info["background"], info["track"], info["bead"], info["divider"])
                palettes.add(roles)
                normalized = np.zeros(info["plate"].shape, dtype=np.int8)
                for j, color in enumerate(roles):
                    normalized[info["plate"] == color] = j
                layouts.add(_key(normalized))
                matrices.add(tuple(map(tuple, info["matrix"].tolist())))
                axes.add(info["axis"])
                lanes.add(info["lanes"])
                dividers.add(len(info["centers"]))
                breaks.add(len(info["breaks"]))
                bead_counts.add(details["bead_count"])
                padding.add(info["padding"])
                features["overlapping_pairs"] += info["overlaps"] > 0
                features["busy_pairs"] += max(info["separator_counts"]) >= 2
                features["blank_separator_pairs"] += 0 in info["separator_counts"]
                for key in ("single", "collision", "tie_moved", "priority_witness", "topology_sensitive", "frame_sensitive", "empty_paths"):
                    features[key] += bool(details[key])
        digests.add(digest.digest())
        if seed < 3:
            repeat = generate_puzzle(seed)
            assert all(_key(p[k]) == _key(q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], repeat[s]) for k in ("input", "output"))
            del repeat
    assert len(digests) == 42
    assert matrices == set(MATRICES) and axes == {0, 1}
    assert len(shapes) >= 50 and len(core_shapes) >= 40 and len(layouts) >= 120
    assert len(palettes) >= 50 and len(lanes) >= 7
    assert len(dividers) >= 10 and len(breaks) >= 5 and len(bead_counts) >= 12
    assert len(padding) >= 10
    assert features["overlapping_pairs"] and features["busy_pairs"] and features["blank_separator_pairs"]
    assert all(features[k] >= 42 for k in ("single", "collision", "tie_moved", "priority_witness", "topology_sensitive", "frame_sensitive"))
    # Every explicit legal count is checked by the host; exercise boundaries
    # here too, without retaining 32 reconstructible puzzles at once.
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(12345, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Rejection checks use one copy per mutation, bounded by one puzzle.
    for kind in ("cell", "extent", "guide"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        pair = bad["test"][0]
        if kind == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif kind == "extent":
            pair["output"] = pair["output"][1:, :].copy()
        else:
            info = _decode_input(pair["input"])
            r, c, h, w = info["bounds"]
            side = next(iter(info["inner"]))
            p = {"T": (r - 1, c), "B": (r + h, c), "L": (r, c - 1), "R": (r, c + w)}[side]
            pair["input"][p] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"mutation accepted: {kind}")
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), plate_shapes=len(core_shapes),
                normalized_layouts=len(layouts), palettes=len(palettes), frames=len(matrices),
                lane_counts=sorted(lanes), divider_counts=sorted(dividers),
                break_counts=sorted(breaks), bead_counts=sorted(bead_counts), features=dict(features))
