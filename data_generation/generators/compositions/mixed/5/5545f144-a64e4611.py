"""Exhaust a tip-led journey, then mark every empty clipped 3x3 center.

Only standard-library code and NumPy are used. No files are read or written.
"""
from __future__ import annotations

import hashlib
import numpy as np

HIGHLIGHT = 3
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=False)


def _rot(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _arrow(span, extension, tail, closed, solid):
    """A geometric grammar, not a mask indexed by area; canonical facing right."""
    cells = {(0, -i) for i in range(extension)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((r, 1 - extension - depth) for r in rows)
    for k in range(1, tail + 1):
        cells.update((r, 1 - extension - span - k) for r in (-span, span))
    if closed:
        cells.update((r, 1 - extension - span - tail) for r in range(-span, span + 1))
    return tuple(sorted(cells))


def _footprint(mask, tip, direction):
    return {(tip[0] + r, tip[1] + c) for r, c in
            (_rot(p, direction - 1) for p in mask)}


def _fits(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _direction(p, q):
    dr, dc = q[0] - p[0], q[1] - p[1]
    _require((dr == 0) != (dc == 0), "route leg is not orthogonal")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _groups(cells):
    remaining, result = set(cells), []
    while remaining:
        group = {remaining.pop()}
        todo = list(group)
        while todo:
            r, c = todo.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        group.add(q)
                        todo.append(q)
        result.append(group)
    return result


def _identify(cells):
    """Find a unique leading cell and symmetry axis from the actual bitmap."""
    matches = []
    for d, (dr, dc) in enumerate(DIRS):
        projection = max(r * dr + c * dc for r, c in cells)
        front = [p for p in cells if p[0] * dr + p[1] * dc == projection]
        if len(front) != 1:
            continue
        tip = front[0]
        offsets = {_rot((r - tip[0], c - tip[1]), 1 - d) for r, c in cells}
        if max(abs(r) for r, c in offsets) and {(-r, c) for r, c in offsets} == offsets:
            matches.append((tuple(sorted(offsets)), tip, d))
    _require(len(matches) == 1, "mover must have one geometrically identifiable tip")
    return matches[0]


def _split(inp):
    colors = set(map(int, np.unique(inp)))
    _require(HIGHLIGHT not in colors, "highlight is reserved for output")
    _require(len(colors) in (2, 3), "expected background, ink, and optional divider")
    if len(colors) == 2:
        boards, axis, divider = [inp], "single", None
    else:
        candidates = []
        for color in sorted(colors):
            for dim in (0, 1):
                lines = np.flatnonzero(np.all(inp == color, axis=1 - dim)).tolist()
                if not 1 <= len(lines) <= 3:
                    continue
                ends = [-1] + lines + [inp.shape[dim]]
                sizes = [b - a - 1 for a, b in zip(ends, ends[1:])]
                if min(sizes) <= 0 or len(set(sizes)) != 1:
                    continue
                parts = [inp[a + 1:b, :] if dim == 0 else inp[:, a + 1:b]
                         for a, b in zip(ends, ends[1:])]
                if all(color not in b for b in parts):
                    candidates.append((parts, "vertical" if dim == 0 else "horizontal", color))
        _require(len(candidates) == 1, "divider segmentation must be unique")
        boards, axis, divider = candidates[0]
    values, counts = np.unique(boards[0], return_counts=True)
    _require(len(values) == 2 and counts.max() > boards[0].size // 2, "background must be a strict majority")
    bg = int(values[np.argmax(counts)])
    ink = int(values[np.argmin(counts)])
    _require(all(set(map(int, np.unique(b))) == {bg, ink} for b in boards), "inconsistent panel palette")
    return boards, bg, ink, axis, divider


def _route(tip, waypoints):
    remaining, points, directions = set(waypoints), [tip], []
    while remaining:
        rays = {}
        for q in remaining:
            if q[0] == tip[0] or q[1] == tip[1]:
                rays.setdefault(_direction(tip, q), []).append(q)
        _require(len(rays) == 1, "each route position needs exactly one occupied ray")
        d, candidates = next(iter(rays.items()))
        tip = min(candidates, key=lambda q: abs(q[0] - tip[0]) + abs(q[1] - tip[1]))
        points.append(tip)
        directions.append(d)
        remaining.remove(tip)
    return points, directions


def _board(shape, bg, ink, mask, tip, facing, waypoints=()):
    cells = _footprint(mask, tip, facing)
    _require(_fits(cells, shape), "mover leaves the board")
    a = np.full(shape, bg, dtype=np.int8)
    for r, c in cells | set(waypoints):
        a[r, c] = ink
    return a


def _empty_centers(foreground):
    """Outside the board contributes no foreground: this is clipping."""
    h, w = foreground.shape
    padded = np.pad(foreground, 1, constant_values=False)
    near = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + h, dc:dc + w]
    return ~near


def _highlight(terminal, bg):
    out = terminal.copy()
    out[_empty_centers(terminal != bg)] = HIGHLIGHT
    return out


def _analyze(input_grid):
    inp = _grid(input_grid)
    boards, bg, ink, axis, divider = _split(inp)
    groups = _groups(zip(*np.where(boards[0] == ink)))
    movers = [g for g in groups if len(g) > 1]
    _require(len(movers) == 1, "first panel must separate the mover from singleton waypoints")
    mask, tip, initial = _identify(movers[0])
    waypoints = {next(iter(g)) for g in groups if len(g) == 1}
    _require(len(waypoints) >= 2, "need at least two waypoints")
    points, ds = _route(tip, waypoints)
    _require(ds[0] == initial, "initial travel must follow the tip")
    indices, next_panel = [], 0
    for i, p in enumerate(points):
        facing = initial if i == 0 else ds[i - 1]
        cells = _footprint(mask, p, facing)
        _require(_fits(cells, boards[0].shape), "intermediate mover leaves board")
        _require(all(max(abs(q[0] - r), abs(q[1] - c)) > 1
                     for q in points[i + 1:] for r, c in cells), "future waypoint touches mover")
        state = _board(boards[0].shape, bg, ink, mask, p, facing, points[i + 1:])
        if i < len(points) - 1 and next_panel < len(boards) and np.array_equal(state, boards[next_panel]):
            indices.append(i)
            next_panel += 1
    _require(len(indices) == len(boards) and indices[0] == 0, "panels are not advancing snapshots of one journey")
    terminal = _board(boards[0].shape, bg, ink, mask, points[-1], ds[-1])
    return dict(boards=boards, bg=bg, ink=ink, axis=axis, divider=divider,
                mask=mask, points=points, directions=ds, snapshots=indices, terminal=terminal)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only the supplied input; there is no generation state or palette key."""
    a = _analyze(input_grid)
    return _highlight(a["terminal"], a["bg"])


def _options(points, footprints, mask, shape):
    options = []
    r, c = points[-1]
    forbidden = np.zeros(shape, dtype=bool)
    for fp in footprints:
        for y, x in fp:
            forbidden[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
    for d, (dr, dc) in enumerate(DIRS):
        first = _footprint(mask, points[0], d) if not footprints else None
        if first is not None and not _fits(first, shape):
            continue
        blocked = forbidden
        if first is not None:
            blocked = forbidden.copy()
            for y, x in first:
                blocked[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
        offsets = _footprint(mask, (0, 0), d)
        lowr, highr = -min(y for y, x in offsets), shape[0] - 1 - max(y for y, x in offsets)
        lowc, highc = -min(x for y, x in offsets), shape[1] - 1 - max(x for y, x in offsets)
        for distance in range(2, max(shape)):
            q = (r + dr * distance, c + dc * distance)
            if not (lowr <= q[0] <= highr and lowc <= q[1] <= highc):
                continue
            if blocked[q]:
                continue
            legal = True
            for p, nxt in zip(points[:-1], points[1:]):
                if p[0] == q[0] or p[1] == q[1]:
                    # A later point may only extend an existing occupied ray,
                    # beyond its already scheduled nearest point.
                    cross = (q[0] - p[0]) * (nxt[1] - p[1]) - (q[1] - p[1]) * (nxt[0] - p[0])
                    dot = (q[0] - nxt[0]) * (nxt[0] - p[0]) + (q[1] - nxt[1]) * (nxt[1] - p[1])
                    if cross != 0 or dot <= 0:
                        legal = False
                        break
            if legal:
                options.append((q, d, first))
    return options


def _grow(rng, shape, mask, count, need_turn):
    for _ in range(18):
        points = [(int(rng.integers(shape[0])), int(rng.integers(shape[1])))]
        footprints, ds = [], []
        for _ in range(count):
            options = _options(points, footprints, mask, shape)
            if not options:
                break
            # Sample directions first so long rays do not dominate short ones.
            choices = sorted({option[1] for option in options})
            d = int(rng.choice(choices))
            choices = [option for option in options if option[1] == d]
            q, d, first = choices[int(rng.integers(len(choices)))]
            if first is not None:
                footprints.append(first)
            footprints.append(_footprint(mask, q, d))
            points.append(q)
            ds.append(d)
        if len(ds) == count and (not need_turn or ds[0] != ds[-1]):
            return points
    return None


def _combine(boards, axis, divider):
    if axis == "single":
        return boards[0].copy()
    dim = 0 if axis == "vertical" else 1
    shape = (1, boards[0].shape[1]) if dim == 0 else (boards[0].shape[0], 1)
    bar = np.full(shape, divider, dtype=np.int8)
    pieces = []
    for b in boards:
        if pieces:
            pieces.append(bar)
        pieces.append(b)
    return np.concatenate(pieces, axis=dim)


def _fallback(mask, count, axis):
    """Finite geometric construction, used only when random growth is cramped."""
    depth = -min(c for r, c in mask)
    span = max(r for r, c in mask)
    if axis == "single":
        shape = (30, 30)
        points = [(depth, depth)]
        for i in range(count):
            r, c = points[-1]
            points.append((r, c + 3) if i % 2 == 0 else (r + 3, c))
    else:
        shape = (30, 14)
        points = [(depth, span)]
        points += [(depth + 3 * i, span) for i in range(1, count)]
        points.append((points[-1][0], max(depth + 2, span + 3)))
        if axis == "vertical":
            shape = shape[::-1]
            points = [(c, r) for r, c in points]
    return shape, points


def _make_pair(rng, mask, count, axis, panels, palette, test=False, need_turn=False):
    points = None
    for _ in range(3):
        if axis == "single":
            low = 25 if test else 16
            shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        else:
            narrow = (30 - panels + 1) // panels
            shape = (int(rng.integers(21, 31)), int(rng.integers(max(8, narrow - 2), narrow + 1)))
            if axis == "vertical":
                shape = shape[::-1]
        points = _grow(rng, shape, mask, count, need_turn)
        if points is not None:
            break
    if points is None:
        shape, points = _fallback(mask, count, axis)
        # The fallback narrow dimension supports two panels.
        panels = min(panels, 2)
    ds = [_direction(p, q) for p, q in zip(points, points[1:])]
    # Translate the complete execution, sometimes right onto a board edge.
    all_cells = set(points)
    for i, p in enumerate(points):
        all_cells |= _footprint(mask, p, ds[max(0, i - 1)])
    minr, maxr = min(r for r, c in all_cells), max(r for r, c in all_cells)
    minc, maxc = min(c for r, c in all_cells), max(c for r, c in all_cells)
    dr = int(rng.integers(-minr, shape[0] - maxr))
    dc = int(rng.integers(-minc, shape[1] - maxc))
    points = [(r + dr, c + dc) for r, c in points]
    bg, ink, divider = palette
    indices = [0] + sorted(map(int, rng.choice(np.arange(1, count), panels - 1, replace=False)))
    boards = [_board(shape, bg, ink, mask, points[i], ds[max(0, i - 1)], points[i + 1:]) for i in indices]
    terminal = _board(shape, bg, ink, mask, points[-1], ds[-1])
    return {"input": _combine(boards, axis, divider), "output": _highlight(terminal, bg)}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _count(seed, 0, 2**128 - 1, "seed")
    ntrain = _count(num_train, 2, 10, "num_train")
    ntest = _count(num_test, 1, 4, "num_test")
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    primary = str(rng.choice(["horizontal", "vertical"]))
    puzzle, seen, first_mask = {"train": [], "test": []}, set(), None
    for split, number in (("train", ntrain), ("test", ntest)):
        for index in range(number):
            test = split == "test"
            if not test and index == 0:
                # A tiny enclosed eligible component explicitly rejects B v2's
                # minimum-component-size rule and a bounding-box exclusion.
                mask = _arrow(2, int(rng.integers(1, 3)), int(rng.integers(3, 6)), True, False)
                first_mask = mask
                count, axis, panels = int(rng.integers(4, 6)), primary, 2
            else:
                pool = tuple(dict.fromkeys(
                    _arrow(span, extension, tail, closed, solid)
                    for span in range(1, 5 if test else 4)
                    for tail in range(6 if test else 5)
                    for extension in (1, 2) for closed in (False, True)
                    for solid in (False, True)))
                pool = tuple(mask for mask in pool if mask != first_mask)
                mask = pool[int(rng.integers(len(pool)))]
                count = int(rng.integers(6, 10) if test else rng.integers(2, 4) if index == 1 else rng.integers(2, 6))
                axis = "single" if test or index == 1 else str(rng.choice(["single", primary]))
                panels = 1 if axis == "single" else 2
                depth = -min(c for r, c in mask)
                if axis != "single" and count >= 3 and depth <= 3 and rng.random() < .45:
                    panels = 3
            for _ in range(10):
                palette = tuple(map(int, rng.choice([c for c in range(10) if c != HIGHLIGHT], 3, replace=False)))
                pair = _make_pair(rng, mask, count, axis, panels, palette, test,
                                  need_turn=(not test and index == 0))
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _cavity_centers(a):
    foreground = a["terminal"] != a["bg"]
    h, w = foreground.shape
    reachable = set()
    todo = [(r, c) for r in range(h) for c in range(w)
            if (r in (0, h - 1) or c in (0, w - 1)) and not foreground[r, c]]
    reachable.update(todo)
    while todo:
        r, c = todo.pop()
        for dr, dc in DIRS:
            q = r + dr, c + dc
            if 0 <= q[0] < h and 0 <= q[1] < w and not foreground[q] and q not in reachable:
                reachable.add(q)
                todo.append(q)
    return sum((int(r), int(c)) not in reachable
               for r, c in zip(*np.where(_empty_centers(foreground))))


def validate_generated_puzzle(puzzle) -> None:
    """Check motion, snapshot suffixes, local semantics, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            terminal, bg, ink = a["terminal"], a["bg"], a["ink"]
            assert out.shape == terminal.shape
            assert np.array_equal(out, _highlight(terminal, bg))
            assert np.sum(out == ink) == len(a["mask"]), "waypoints must all be consumed"
            assert np.array_equal(out == ink, terminal == ink), "preserve exact terminal mask"
            assert np.any(out == bg) and np.sum(out == HIGHLIGHT) > 9
            assert not np.array_equal(out, _highlight(a["boards"][0], bg)), "motion must affect the highlight"
            # Direct clipped-window checks provide local witnesses independently
            # of the vectorized neighborhood implementation (not another solver).
            for r in range(out.shape[0]):
                for c in range(out.shape[1]):
                    window = terminal[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                    clear = bool(np.all(window == bg))
                    assert (out[r, c] == HIGHLIGHT) == clear
                    if terminal[r, c] != bg:
                        assert out[r, c] == terminal[r, c]
            # At least one foreground-free border neighborhood is highlighted.
            assert any(np.any(v == HIGHLIGHT) for v in (out[0], out[-1], out[:, 0], out[:, -1]))
            # Diagonal foreground matters: Manhattan/cross neighborhoods fail.
            fg = terminal != bg
            cross_near = fg.copy()
            cross_near[1:] |= fg[:-1]
            cross_near[:-1] |= fg[1:]
            cross_near[:, 1:] |= fg[:, :-1]
            cross_near[:, :-1] |= fg[:, 1:]
            assert np.any(~cross_near & (out == bg))
            info[split].append(a)
    train = info["train"]
    assert len({a["mask"] for a in train}) >= 2
    assert len({len(a["directions"]) for a in train}) >= 2
    assert any(a["directions"][0] != a["directions"][-1] for a in train)
    assert len(train[0]["boards"]) >= 2 and len(train[1]["boards"]) == 1
    assert 1 <= _cavity_centers(train[0]) <= 4, "teach tiny hollow interiors"
    assert all(len(a["directions"]) > max(len(b["directions"]) for b in train) for a in info["test"])
    assert all(len(a["boards"]) == 1 for a in info["test"])


def self_test() -> None:
    """One puzzle at a time, 42 seeds, no saved grids or execution traces."""
    signatures, shapes, masks, routes, palettes = set(), set(), set(), set(), set()
    firsts, lasts, snapshots, cavity_sizes = set(), set(), set(), set()
    same_facing = mixed_turns = straight = edge = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                digest.update(repr(pair["input"].shape).encode())
                digest.update(pair["input"].tobytes())
                shapes.add(a["terminal"].shape)
                masks.add(a["mask"])
                palettes.add((a["bg"], a["ink"], a["divider"]))
                ds, pts = a["directions"], a["points"]
                firsts.add(ds[0])
                lasts.add(ds[-1])
                snapshots.add(len(a["boards"]))
                cavity_sizes.add(_cavity_centers(a))
                # Translation-, rotation-, and reflection-normalized route.
                variants = []
                for sign in (-1, 1):
                    for turns in range(4):
                        variants.append(tuple(_rot((r - pts[0][0], sign * (c - pts[0][1])), turns) for r, c in pts))
                routes.add(min(variants))
                turns = {(b - a) % 4 for a, b in zip(ds, ds[1:]) if a != b}
                same_facing += ds[0] == ds[-1]
                mixed_turns += 1 in turns and 3 in turns
                straight += any(a == b for a, b in zip(ds, ds[1:]))
                cells = _footprint(a["mask"], pts[-1], ds[-1])
                h, w = a["terminal"].shape
                edge += any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 30
    assert len(masks) >= 35 and len(routes) >= 150 and len(palettes) >= 100
    assert firsts == lasts == set(range(4)) and {1, 2} <= snapshots
    assert {0, 1, 2, 3} <= cavity_sizes
    assert same_facing and mixed_turns and straight and edge
    # Every legal explicit count combination, plus determinism at the maximum.
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(100 + ntrain * 5 + ntest, ntrain, ntest)
            assert (len(p["train"]), len(p["test"])) == (ntrain, ntest)
    p, q = generate_puzzle(987, 10, 4), generate_puzzle(987, 10, 4)
    assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Negative examples: no-motion label, damaged label, and repeated snapshot.
    p = generate_puzzle(42, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("damaged label accepted")
    a = _analyze(p["train"][0]["input"])
    repeated = _combine([a["boards"][0]] * 2, a["axis"], a["divider"])
    try:
        solve(repeated)
    except ValueError:
        pass
    else:
        raise AssertionError("nonadvancing snapshots accepted")
