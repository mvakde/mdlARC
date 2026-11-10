"""A tip-led, distance-clocked printing ruler. Standard library + NumPy only."""
from __future__ import annotations

import hashlib
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ABLATIONS = ("terminal_only", "no_remote", "no_odd_clear", "leg_reset",
             "fixed_heading", "width_one", "late_flashes", "current_objects")
TEACHING = ("odd_object_erased", "ordinary_object_overwritten", "remote_object_highlighted",
            "outside_object_preserved", "outside_background_preserved")


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid size"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "ARC colors required"
    return a.astype(np.int8, copy=False)


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert (dr == 0) != (dc == 0), "distinct orthogonal references required"
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _components(cells):
    unseen, result = set(cells), []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack, part = [root], {root}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in unseen:
                        unseen.remove(q)
                        part.add(q)
                        stack.append(q)
        result.append(part)
    return result


def _cells(mask, tip, facing):
    return {(tip[0] + dr, tip[1] + dc)
            for dr, dc in (_rotate(p, facing - 1) for p in mask)}


def _order(tip, waypoints):
    remaining, points = set(waypoints), [tip]
    while remaining:
        rays = {}
        for q in remaining:
            if q[0] == tip[0] or q[1] == tip[1]:
                rays.setdefault(_direction(tip, q), []).append(q)
        assert len(rays) == 1, "exactly one occupied ray is required"
        candidates = next(iter(rays.values()))
        tip = min(candidates, key=lambda q: _distance(points[-1], q))
        points.append(tip)
        remaining.remove(tip)
    return points


def _events(points):
    """A bend is printed once, facing along its incoming leg; zero is at the nose."""
    facing = _direction(points[0], points[1])
    yield 0, 0, points[0][0], points[0][1], facing
    clock = 0
    for a, b in zip(points, points[1:]):
        facing = _direction(a, b)
        dr, dc = DIRS[facing]
        for local in range(1, _distance(a, b) + 1):
            clock += 1
            yield clock, local, a[0] + local * dr, a[1] + local * dc, facing


def _bar(r, c, facing, span):
    dr, dc = DIRS[(facing + 1) % 4]
    return [(r + j * dr, c + j * dc) for j in range(-span, span + 1)]


def _decode(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    assert 4 <= len(colors) <= 6, "background, body, nose, and 1-3 object colors"
    bg = int(colors[np.argmax(counts)])
    assert counts.max() > g.size / 2, "background must be a strict majority"
    noses = [int(c) for c, n in zip(colors, counts) if n == 1 and c != bg]
    assert len(noses) == 1, "nose is the unique color appearing once"
    flash = noses[0]
    tip = tuple(map(int, np.argwhere(g == flash)[0]))
    adjacent = {int(g[r, c]) for r in range(max(0, tip[0] - 1), min(g.shape[0], tip[0] + 2))
                for c in range(max(0, tip[1] - 1), min(g.shape[1], tip[1] + 2))}
    adjacent -= {bg, flash}
    assert len(adjacent) == 1, "nose must touch only its body color"
    ink = next(iter(adjacent))
    parts = _components(map(tuple, np.argwhere((g == ink) | (g == flash)).tolist()))
    movers = [p for p in parts if len(p) > 1]
    assert len(movers) == 1 and tip in movers[0], "one separated arrow required"
    mover = movers[0]
    matches = []
    for facing in range(4):
        mask = {_rotate((r - tip[0], c - tip[1]), 1 - facing) for r, c in mover}
        if (all(c <= 0 for r, c in mask)
                and {p for p in mask if p[1] == 0} == {(0, 0)}
                and {(-r, c) for r, c in mask} == mask
                and max(abs(r) for r, c in mask) >= 1):
            matches.append((tuple(sorted(mask)), facing))
    assert len(matches) == 1, "unique arrow axis required"
    mask, initial = matches[0]
    span = max(abs(r) for r, c in mask)
    waypoints = {next(iter(p)) for p in parts if len(p) == 1}
    assert len(waypoints) >= 1, "at least one waypoint required"
    points = _order(tip, waypoints)
    assert _direction(points[0], points[1]) == initial, "first leg must follow the nose"
    for _, _, r, c, facing in _events(points):
        # No clipping: both the moving arrow and its ruler must fit at every step.
        footprint = _cells(mask, (r, c), facing) | set(_bar(r, c, facing, span))
        assert all(0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
                   for rr, cc in footprint), "sweep leaves board"
    # Also fit the outgoing rotation at each bend, before the next unit step.
    for a, b in zip(points, points[1:]):
        assert all(0 <= r < g.shape[0] and 0 <= c < g.shape[1]
                   for r, c in _cells(mask, a, _direction(a, b))), "turn leaves board"
    fg = ~np.isin(g, (bg, ink, flash))
    return dict(bg=bg, ink=ink, flash=flash, tip=tip, initial=initial,
                mask=mask, span=span, points=points, fg=fg)


def _paint(g, a, variant=None):
    """Chronological printing; variants are only counterexample checks."""
    out = g.copy()
    out[(g == a["ink"]) | (g == a["flash"])] = a["bg"]
    pending = np.zeros(g.shape, dtype=bool)
    for clock, local, r, c, facing in _events(a["points"]):
        if variant == "terminal_only":
            break
        phase = (local if variant == "leg_reset" else clock) % 6
        if variant == "fixed_heading":
            facing = a["initial"]
        span = 0 if variant == "width_one" else a["span"]
        color = a["bg"] if phase % 2 else a["flash"] if phase == 4 else a["ink"]
        if not (variant == "no_odd_clear" and phase % 2):
            for rr, cc in _bar(r, c, facing, span):
                if 0 <= rr < len(g) and 0 <= cc < g.shape[1]:
                    out[rr, cc] = color
        if phase == 4 and variant != "no_remote":
            eligible = a["fg"] & (out == g) if variant == "current_objects" else a["fg"]
            if facing % 2 == 0:
                if variant == "late_flashes":
                    pending[r, eligible[r]] = True
                else:
                    out[r, eligible[r]] = a["flash"]
            else:
                if variant == "late_flashes":
                    pending[eligible[:, c], c] = True
                else:
                    out[eligible[:, c], c] = a["flash"]
    out[pending] = a["flash"]
    final_facing = _direction(a["points"][-2], a["points"][-1])
    if variant == "fixed_heading":
        final_facing = a["initial"]
    for r, c in _cells(a["mask"], a["points"][-1], final_facing):
        if 0 <= r < len(g) and 0 <= c < g.shape[1]:
            out[r, c] = a["ink"]
    out[a["points"][-1]] = a["flash"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    return _paint(g, _decode(g))


def _pixel_prediction(g, a):
    """Check the painter by comparing last bar-write and flash times per pixel."""
    last_bar = np.full(g.shape, -1, dtype=int)
    bar_value = np.full(g.shape, a["bg"], dtype=np.int8)
    row_flash = np.full(g.shape[0], -1, dtype=int)
    col_flash = np.full(g.shape[1], -1, dtype=int)
    for clock, _, r, c, facing in _events(a["points"]):
        color = a["bg"] if clock % 2 else a["flash"] if clock % 6 == 4 else a["ink"]
        for rr, cc in _bar(r, c, facing, a["span"]):
            last_bar[rr, cc] = clock
            bar_value[rr, cc] = color
        if clock % 6 == 4:
            (row_flash if facing % 2 == 0 else col_flash)[r if facing % 2 == 0 else c] = clock
    out = np.where((g == a["ink"]) | (g == a["flash"]), a["bg"], g).astype(np.int8)
    out = np.where(last_bar >= 0, bar_value, out).astype(np.int8)
    flash_time = np.maximum(row_flash[:, None], col_flash[None, :])
    out[a["fg"] & (flash_time >= 0) & (flash_time >= last_bar)] = a["flash"]
    final = _direction(a["points"][-2], a["points"][-1])
    for r, c in _cells(a["mask"], a["points"][-1], final):
        out[r, c] = a["ink"]
    out[a["points"][-1]] = a["flash"]
    return out


def _arrow(span, extension, hollow, tail):
    depth = extension + span - 1 + tail
    cells = {(0, -j) for j in range(depth + 1)}
    for j in range(1, span + 1):
        rows = (-j, j) if hollow else range(-j, j + 1)
        cells.update((r, -(extension - 1 + j)) for r in rows)
    return tuple(sorted(cells))


def _valid_extension(points, q, initial_cells):
    if any(max(abs(q[0] - r), abs(q[1] - c)) <= 1
           for r, c in initial_cells | set(points[1:])):
        return False
    for a, b in zip(points, points[1:]):
        if q[0] == a[0] or q[1] == a[1]:
            if q == a or _direction(a, q) != _direction(a, b) or _distance(a, q) <= _distance(a, b):
                return False
    return True


def _route(rng, shape, mask, count, long_first=False, fallback=False):
    radius = max(max(abs(r), abs(c)) for r, c in mask)
    lo = radius
    hi = (shape[0] - 1 - radius, shape[1] - 1 - radius)
    if fallback:
        # Parameterized geometry, used only after bounded random proposals.
        if long_first:
            points = [(lo, lo), (lo, lo + 12), (lo + 5, lo + 12), (lo + 5, lo + 7)]
        else:
            points = [(lo, lo)]
            for i in range(count):
                r, c = points[-1]
                points.append((r + (3 if i % 2 else 0), c + (0 if i % 2 else 3)))
        return points
    for _ in range(24):
        initial = int(rng.integers(4))
        dr, dc = DIRS[initial]
        axis = 0 if dr else 1
        max_step = hi[axis] - lo
        minimum = 12 if long_first else 3
        if max_step < minimum:
            continue
        length = int(rng.integers(minimum, max_step + 1))
        start = [int(rng.integers(lo, x + 1)) for x in hi]
        positive = dr + dc > 0
        start[axis] = int(rng.integers(lo, hi[axis] - length + 1)) if positive else int(rng.integers(lo + length, hi[axis] + 1))
        points = [tuple(start), (start[0] + dr * length, start[1] + dc * length)]
        initial_cells = _cells(mask, points[0], initial)
        if not _valid_extension(points[:1], points[1], initial_cells):
            continue
        while len(points) <= count:
            options = []
            for facing, (dr, dc) in enumerate(DIRS):
                if facing == (_direction(points[-2], points[-1]) + 2) % 4:
                    continue
                for length in range(3, max(shape)):
                    q = (points[-1][0] + dr * length, points[-1][1] + dc * length)
                    if not (lo <= q[0] <= hi[0] and lo <= q[1] <= hi[1]):
                        continue
                    if _valid_extension(points, q, initial_cells):
                        options.append(q)
            if not options:
                break
            points.append(options[int(rng.integers(len(options)))])
        if len(points) == count + 1:
            if not long_first or len({_direction(a, b) for a, b in zip(points, points[1:])}) > 1:
                return points
    return None


def _objects(rng, allowed, count, dense=False):
    """Natural growth, rectangles, elbows and open/closed outlines; no phase stamp."""
    h, w = allowed.shape
    labels = np.zeros((h, w), dtype=np.int8)
    target = int(h * w * rng.uniform(.14 if dense else .06, .28 if dense else .22))
    for _ in range(int(rng.integers(3, 10))):
        color = int(rng.integers(1, count + 1))
        budget = int(rng.integers(5, max(6, target // 2 + 1)))
        kind = int(rng.integers(4))
        cells = set()
        if kind == 0:
            r, c = int(rng.integers(h)), int(rng.integers(w))
            cells.add((r, c))
            frontier = [(r, c)]
            for _ in range(budget * 4):
                if len(cells) >= budget:
                    break
                r, c = frontier[int(rng.integers(len(frontier)))]
                dr, dc = DIRS[int(rng.integers(4))]
                q = (r + dr, c + dc)
                if 0 <= q[0] < h and 0 <= q[1] < w and q not in cells:
                    cells.add(q)
                    frontier.append(q)
        else:
            rh = int(rng.integers(2, min(h, max(4, int(np.sqrt(budget)) * 2)) + 1))
            cw = int(rng.integers(2, min(w, max(4, budget // rh + 2)) + 1))
            r0, c0 = int(rng.integers(h - rh + 1)), int(rng.integers(w - cw + 1))
            for r in range(rh):
                for c in range(cw):
                    if kind == 1 or (kind == 2 and (r == rh - 1 or c == 0)) or (kind == 3 and (r in (0, rh - 1) or c in (0, cw - 1))):
                        cells.add((r0 + r, c0 + c))
        for r, c in cells:
            if allowed[r, c]:
                labels[r, c] = color
        if np.count_nonzero(labels) >= target:
            break
    # Every object role has at least two cells; only the nose is a singleton color.
    for color in range(1, count + 1):
        if np.count_nonzero(labels == color) < 2:
            labels[labels == color] = 0
    return labels


def _evidence(g, out, a):
    swept = np.zeros(g.shape, dtype=bool)
    flashed = np.zeros(g.shape, dtype=bool)
    for clock, _, r, c, facing in _events(a["points"]):
        for rr, cc in _bar(r, c, facing, a["span"]):
            swept[rr, cc] = True
        if clock % 6 == 4:
            if facing % 2 == 0:
                flashed[r, :] = True
            else:
                flashed[:, c] = True
    terminal = np.zeros(g.shape, dtype=bool)
    final = _direction(a["points"][-2], a["points"][-1])
    for r, c in _cells(a["mask"], a["points"][-1], final):
        terminal[r, c] = True
    return dict(odd_object_erased=int(np.count_nonzero(a["fg"] & (out == a["bg"]))),
                ordinary_object_overwritten=int(np.count_nonzero(a["fg"] & (out == a["ink"]) & ~terminal)),
                remote_object_highlighted=int(np.count_nonzero(a["fg"] & (out == a["flash"]) & ~swept & ~terminal)),
                outside_object_preserved=int(np.count_nonzero(a["fg"] & (out == g) & ~swept & ~terminal)),
                outside_background_preserved=int(np.count_nonzero((g == a["bg"]) & (out == a["bg"]) & flashed & ~swept & ~terminal)))


def _make_pair(rng, palette, span, count, teach=False, fallback=False):
    mask = _arrow(span, int(rng.integers(1, 3)), bool(rng.integers(2)), int(rng.integers(3)))
    radius = max(max(abs(r), abs(c)) for r, c in mask)
    minimum = max(20, 2 * radius + 14) if teach else 2 * radius + (16 if count >= 5 else 9)
    shape = (int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31)))
    points = _route(rng, shape, mask, count, long_first=teach, fallback=fallback)
    if points is None:
        return None
    initial = _direction(points[0], points[1])
    bg, ink, flash = palette[:3]
    g = np.full(shape, bg, dtype=np.int8)
    mover = _cells(mask, points[0], initial)
    allowed = np.ones(shape, dtype=bool)
    for r, c in mover | set(points[1:]):
        allowed[r, c] = False
    r, c = points[0]
    allowed[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = False
    labels = _objects(rng, allowed, int(rng.integers(1, 4)), dense=teach or fallback)
    if not labels.any():
        return None
    for label in range(1, 4):
        g[labels == label] = palette[2 + label]
    for r, c in mover | set(points[1:]):
        g[r, c] = ink
    g[points[0]] = flash
    try:
        info = _decode(g)
    except AssertionError:
        return None
    output = _paint(g, info)
    if teach and not all(not np.array_equal(output, _paint(g, info, v)) for v in ABLATIONS):
        return None
    if teach and not all(_evidence(g, output, info)[k] > 0 for k in TEACHING):
        return None
    # Random D4 transforms also apply to the constructive fallback.
    turns = int(rng.integers(4))
    if rng.random() < .5:
        g, output = g[:, ::-1], output[:, ::-1]
    g, output = np.rot90(g, turns).copy(), np.rot90(output, turns).copy()
    return {"input": g, "output": output}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    shared = bool(rng.random() < .2)
    common_palette = rng.permutation(10).tolist()
    first_span = int(rng.integers(1, 3))
    puzzle, seen = {"train": [], "test": []}, set()
    first_shape = None
    for split, total in (("train", nt), ("test", ne)):
        for index in range(total):
            count = (3 if index == 0 else 2 if index == 1 else int(rng.integers(2, 5))) if split == "train" else int(rng.integers(5, 8))
            span = first_span if split == "train" and index == 0 else 3 - first_span if split == "train" and index == 1 else int(rng.integers(1, 3))
            palette = common_palette if shared else rng.permutation(10).tolist()
            for attempt in range(96):
                pair = _make_pair(rng, palette, span, count, teach=split == "train" and index == 0, fallback=attempt >= 80)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                if split == "train" and index == 1 and pair["input"].shape == first_shape:
                    continue
                break
            else:
                raise RuntimeError("bounded scanner construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                first_shape = pair["input"].shape
    return puzzle


def _measure(pair):
    g, out = _grid(pair["input"]), _grid(pair["output"])
    a = _decode(g)
    ds = [_direction(x, y) for x, y in zip(a["points"], a["points"][1:])]
    lengths = [_distance(x, y) for x, y in zip(a["points"], a["points"][1:])]
    bends = [(y - x) % 4 for x, y in zip(ds, ds[1:]) if x != y]
    ablations = {v: int(np.count_nonzero(out != _paint(g, a, v))) for v in ABLATIONS}
    variants = []
    for reflect in (1, -1):
        for turns in range(4):
            variants.append(tuple(_rotate((r - a["tip"][0], reflect * (c - a["tip"][1])), turns) for r, c in a["points"]))
    return dict(shape=g.shape, mask=a["mask"], span=a["span"], waypoints=len(ds),
                length=sum(lengths), distances=tuple(lengths), turns=len(bends),
                mixed_turns=1 in bends and 3 in bends, straight_steps=sum(x == y for x, y in zip(ds, ds[1:])),
                initial=ds[0], final=ds[-1], palette=(a["bg"], a["ink"], a["flash"]),
                object_colors=tuple(int(c) for c in np.unique(g[a["fg"]])),
                object_cells=int(a["fg"].sum()), route=min(variants), ablations=ablations,
                evidence=_evidence(g, out, a))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape and not np.array_equal(g, out)
            assert _key(g) not in seen, "duplicate input"
            seen.add(_key(g))
            a = _decode(g)
            assert np.array_equal(out, _pixel_prediction(g, a)), "wrong chronological pixel writes"
            final_facing = _direction(a["points"][-2], a["points"][-1])
            terminal = _cells(a["mask"], a["points"][-1], final_facing)
            assert out[a["points"][-1]] == a["flash"]
            assert all(out[r, c] == a["ink"] for r, c in terminal - {a["points"][-1]}), "rigid terminal mask"
            assert set(np.unique(out)) <= set(np.unique(g)), "no hidden output color"
            records[split].append(_measure(pair))
    train = records["train"]
    assert len({m["mask"] for m in train}) >= 2, "demonstrate actual shape preservation"
    assert len({m["span"] for m in train}) >= 2, "demonstrate ruler width from wings"
    assert len({m["shape"] for m in train}) >= 2
    assert len({m["waypoints"] for m in train}) >= 2
    assert any(m["distances"][0] >= 12 and m["turns"] for m in train), "show two clock cycles and a bend"
    for variant in ABLATIONS:
        assert any(m["ablations"][variant] > 0 for m in train), f"shortcut not distinguished: {variant}"
    for witness in TEACHING:
        assert any(m["evidence"][witness] > 0 for m in train), f"missing teaching witness: {witness}"
    assert min(m["waypoints"] for m in records["test"]) > max(m["waypoints"] for m in train), "longer held-out journeys"


def self_test():
    """Stream 42 seeds, retaining only small sets of signatures and feature values."""
    signatures, shapes, routes, masks = set(), set(), {"train": set(), "test": set()}, set()
    headings = {s: [set(), set()] for s in ("train", "test")}
    waypoint_counts = {s: set() for s in headings}
    widths, palettes, lengths, turns, mixed, straight = set(), [set(), set(), set()], set(), set(), 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        first = p["train"][0]
        info = _decode(first["input"])
        first_leg = dict(info, points=info["points"][:2])
        assert not np.array_equal(first["output"], _paint(first["input"], first_leg)), "later route must matter"
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                for grid in pair.values():
                    digest.update(bytes(grid.shape))
                    digest.update(grid.tobytes())
                m = _measure(pair)
                shapes.add(m["shape"])
                routes[split].add(m["route"])
                masks.add(m["mask"])
                headings[split][0].add(m["initial"])
                headings[split][1].add(m["final"])
                waypoint_counts[split].add(m["waypoints"])
                widths.add(m["span"])
                lengths.add(m["length"])
                turns.add(m["turns"])
                mixed += m["mixed_turns"]
                straight += m["straight_steps"] > 0
                for i, c in enumerate(m["palette"]):
                    palettes[i].add(c)
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 40
    assert all(len(routes[s]) >= (35 if s == "test" else 80) for s in routes)
    assert len(masks) >= 10 and widths == {1, 2}
    assert all(all(d == set(range(4)) for d in headings[s]) for s in headings)
    assert waypoint_counts == {"train": {2, 3, 4}, "test": {5, 6, 7}}
    assert all(c == set(range(10)) for c in palettes)
    assert len(lengths) >= 30 and len(turns) >= 4 and mixed >= 5 and straight >= 5
    # A new width/mask, boundary flashes, one-waypoint journeys, and redundant
    # collinear instructions exercise the rule beyond the sampled templates.
    custom = np.zeros((30, 30), dtype=np.int8)
    custom_mask = _arrow(3, 2, True, 1)
    for r, c in _cells(custom_mask, (15, 5), 1) | {(15, 20), (22, 20)}:
        custom[r, c] = 1
    custom[15, 5] = 2
    for point in ((16, 0), (17, 0), (22, 29), (23, 29)):
        custom[point] = 3
    result = solve(custom)
    assert (result[16, 0], result[17, 0], result[22, 29], result[23, 29]) == (2, 3, 2, 3)
    subdivided = custom.copy()
    subdivided[15, 11] = 1
    assert np.array_equal(solve(subdivided), result), "waypoint count must not control phase"
    one_waypoint = custom.copy()
    one_waypoint[22, 20] = 0
    assert solve(one_waypoint)[15, 20] == 2
    for k in range(4):
        assert np.array_equal(solve(np.rot90(custom, k)), np.rot90(result, k)), "rotation covariance"
    assert np.array_equal(solve(custom[:, ::-1]), result[:, ::-1]), "reflection covariance"
    a, b = generate_puzzle(2718, 2, 1), generate_puzzle(2718, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + 10 * nt + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    try:
        _order((4, 4), {(4, 8), (8, 4)})
    except AssertionError:
        pass
    else:
        raise AssertionError("junction accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "routes": {s: len(v) for s, v in routes.items()}, "masks": len(masks),
            "mixed_turn_pairs": mixed, "straight_continuation_pairs": straight}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
