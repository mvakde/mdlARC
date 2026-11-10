"""Turning keys: checkpoint traversal with orientation-dependent fork capture.

Only NumPy and the Python standard library are used. No files are read by the
generator or solver. self_test holds its 42-seed sweep in memory.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _rot(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _mask(span, shaft, solid, tail):
    """Right-facing, symmetric arrow; its tip is (0,0)."""
    cells = {(0, -j) for j in range(shaft)}
    for j in range(1, span + 1):
        cross = range(-j, j + 1) if solid else (-j, j)
        cells.update((r, 1 - shaft - j) for r in cross)
    for j in range(tail):
        cells.update((r, -shaft - span - j) for r in (-span, span))
    return frozenset(cells)


def _foot(mask, tip, facing):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rot(p, facing - 1) for p in mask)}


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert (dr == 0) != (dc == 0), "distinct orthogonal checkpoints required"
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _components(cells, diagonal=False):
    remaining, result = set(cells), []
    neighbors = ([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if dr or dc] if diagonal else DIRS)
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _identify_arrow(cells):
    possibilities = []
    for tip in sorted(cells):
        for facing in range(4):
            local = {_rot((r - tip[0], c - tip[1]), 1 - facing) for r, c in cells}
            if (max(c for r, c in local) == 0
                    and {p for p in local if p[1] == 0} == {(0, 0)}
                    and max(abs(r) for r, c in local) > 0
                    and {(-r, c) for r, c in local} == local):
                possibilities.append((frozenset(local), tip, facing))
    assert len(possibilities) == 1, "arrow must have one symmetric axis and projecting tip"
    return possibilities[0]


def _parse_fork(cells):
    """Infer the base and three positive rectangular spokes, in any direction."""
    possibilities = []
    for inward in range(4):
        out = DIRS[(inward + 2) % 4]
        perp = (out[1], -out[0])
        kbase = min(r * out[0] + c * out[1] for r, c in cells)
        qbase = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * out[0] + c * out[1] - kbase,
                  r * perp[0] + c * perp[1] - qbase) for r, c in cells}
        width = max(q for k, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        heights = [sum((k, q) in local for k in range(1, 31)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, length in enumerate(heights) for k in range(1, length + 1)}
        if local != expected:
            continue
        runs, q = [], 0
        while q < width:
            if heights[q] == 0:
                q += 1
                continue
            start, length = q, heights[q]
            while q < width and heights[q] == length:
                q += 1
            runs.append((start, q, length))
        if (len(runs) != 3 or runs[0][1] >= runs[1][0] or runs[1][1] >= runs[2][0]
                or runs[1][0] + runs[1][1] != width):
            continue
        possibilities.append(dict(cells=set(cells), inward=inward, out=out, perp=perp,
                                  kbase=kbase, qbase=qbase, width=width,
                                  runs=tuple(runs), span=runs[1][1] - runs[1][0],
                                  stop=kbase + runs[1][2] + 1))
    assert len(possibilities) == 1, "receiver needs exactly three positive spokes and a unique base"
    return possibilities[0]


def _decode_route(tip, waypoints):
    points, directions, remaining = [tip], [], set(waypoints)
    while remaining:
        rays = {}
        for q in remaining:
            if q[0] == tip[0] or q[1] == tip[1]:
                d = _direction(tip, q)
                rays.setdefault(d, []).append(q)
        assert len(rays) == 1, "the remaining waypoints must occupy exactly one ray"
        d, candidates = next(iter(rays.items()))
        q = min(candidates, key=lambda x: abs(x[0] - tip[0]) + abs(x[1] - tip[1]))
        directions.append(d)
        points.append(q)
        remaining.remove(q)
        tip = q
    return points, directions


def _read_input(grid):
    grid = np.asarray(grid)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert counts.max() > grid.size / 2 and bg != 0 and 0 not in values
    forks, arrow_parts = [], []
    for color in values:
        if color == bg:
            continue
        cells = set(map(tuple, np.argwhere(grid == color)))
        # Forks are recognized first. Some symmetric long-center forks also
        # have a projecting extremity; that does not make them moving arrows.
        for component in _components(cells):
            try:
                fork = _parse_fork(component)
            except AssertionError:
                continue
            fork["color"] = int(color)
            forks.append(fork)
            cells -= component
        if cells:
            arrow_parts.append((int(color), cells))
    assert len(arrow_parts) == 1 and 2 <= len(forks) <= 4
    ink, cells = arrow_parts[0]
    components = _components(cells, diagonal=True)
    movers = [c for c in components if len(c) > 1]
    assert len(movers) == 1
    mask, tip, facing = _identify_arrow(movers[0])
    waypoints = {next(iter(c)) for c in components if len(c) == 1}
    assert len(waypoints) >= 2
    points, ds = _decode_route(tip, waypoints)
    assert ds[0] == facing
    assert all(f["color"] != ink for f in forks)
    structural = set().union(*(f["cells"] for f in forks))
    return dict(bg=bg, ink=ink, forks=forks, mask=mask, points=points,
                directions=ds, facing=facing, structural=structural)


def _sweep_between(mask, a, b, facing):
    d = _direction(a, b)
    dr, dc = DIRS[d]
    length = abs(a[0] - b[0]) + abs(a[1] - b[1])
    return set().union(*(_foot(mask, (a[0] + i * dr, a[1] + i * dc), facing)
                         for i in range(length + 1)))


def _dock_candidate(fork, cells, structural, shape, *, ignore_span=False):
    """Rigid insertion with the arrival orientation preserved, even sideways."""
    out, perp = fork["out"], fork["perp"]
    qs = {r * perp[0] + c * perp[1] for r, c in cells}
    lo, hi = min(qs), max(qs)
    a, b, _ = fork["runs"][1]
    flo, fhi = fork["qbase"] + a, fork["qbase"] + b - 1
    if ignore_span:
        if lo + hi != flo + fhi:
            return None
    elif qs != set(range(flo, fhi + 1)):
        return None
    lead = min(r * out[0] + c * out[1] for r, c in cells)
    distance = lead - fork["stop"]
    if distance <= 0:
        return None
    swept = set()
    for i in range(distance + 1):
        moved = _shift(cells, (-i * out[0], -i * out[1]))
        if not _inside(moved, shape) or moved & structural:
            return None
        swept |= moved
    final = _shift(cells, (-distance * out[0], -distance * out[1]))
    # Matching a full projection guarantees a leading cell touches the stop.
    return dict(final=final, swept=swept, distance=distance)


def _execute(grid, analysis, *, ignore_span=False, ignore_capture=False):
    mask, points, ds = analysis["mask"], analysis["points"], analysis["directions"]
    swept = set()
    for i, tip in enumerate(points):
        facing = analysis["facing"] if i == 0 else ds[i - 1]
        cells = _foot(mask, tip, facing)
        assert _inside(cells, grid.shape) and not cells & analysis["structural"]
        swept |= cells
        matches = []
        if not ignore_capture:
            for fi, fork in enumerate(analysis["forks"]):
                candidate = _dock_candidate(fork, cells, analysis["structural"], grid.shape,
                                            ignore_span=ignore_span)
                if candidate is not None:
                    matches.append((fi, candidate))
        if matches:
            # Valid inputs never need nearest-fork, color, or directional ties.
            assert len(matches) == 1, "two receivers fit at the same checkpoint"
            fi, candidate = matches[0]
            swept |= candidate["swept"]
            final = candidate["final"]
            break
        if i + 1 == len(points):
            final, fi = cells, None
            break
        # Rotation is a discrete quarter-turn at the checkpoint. Include both
        # endpoint masks, not imaginary 45-degree rasterized orientations.
        leg = _sweep_between(mask, tip, points[i + 1], ds[i])
        assert _inside(leg, grid.shape) and not leg & analysis["structural"]
        swept |= leg
    output = np.full(grid.shape, analysis["bg"], dtype=np.int8)
    _paint(output, swept - final, 0)
    for fork in analysis["forks"]:
        _paint(output, fork["cells"], fork["color"])
    _paint(output, final, analysis["ink"])
    return output, dict(checkpoint=i, fork=fi, final=final, swept=swept, facing=facing)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, route, capture, and output cell from this input only."""
    grid = np.asarray(input_grid)
    return _execute(grid, _read_input(grid))[0]


def _grow_route(rng, mask, count, need_turn):
    """Sample independent leg lengths and turn choices; no route template bank."""
    shape = (30, 30)
    for _ in range(36):
        points = [(int(rng.integers(6, 24)), int(rng.integers(6, 24)))]
        ds, footprints = [], []
        while len(ds) < count:
            options = []
            for d, (dr, dc) in enumerate(DIRS):
                if ds and d == (ds[-1] + 2) % 4:
                    continue
                for distance in range(3, 16):
                    q = points[-1][0] + distance * dr, points[-1][1] + distance * dc
                    body = _foot(mask, q, d)
                    initial = _foot(mask, points[0], d) if not ds else footprints[0]
                    if not _inside(body | initial, shape):
                        continue
                    if not (3 <= q[0] <= 26 and 3 <= q[1] <= 26):
                        continue
                    if any(q in _halo(f) for f in (footprints or [initial])):
                        continue
                    # A new point must preserve the unique occupied ray at all
                    # earlier checkpoints. Collinear continuation is allowed.
                    valid = True
                    for p, nxt in zip(points[:-1], points[1:]):
                        if p[0] == q[0] or p[1] == q[1]:
                            if (_direction(p, q) != _direction(p, nxt)
                                    or abs(q[0] - p[0]) + abs(q[1] - p[1])
                                    <= abs(nxt[0] - p[0]) + abs(nxt[1] - p[1])):
                                valid = False
                                break
                    if valid:
                        options.append((d, q, body, initial))
            if not options:
                break
            d, q, body, initial = options[int(rng.integers(len(options)))]
            if not ds:
                footprints.append(initial)
            ds.append(d)
            points.append(q)
            footprints.append(body)
        if len(ds) == count and (not need_turn or ds[0] != ds[1]):
            return points, ds
    return None


def _prefix_sweep(mask, points, ds, capture):
    cells = _foot(mask, points[0], ds[0])
    for i in range(capture):
        cells |= _sweep_between(mask, points[i], points[i + 1], ds[i])
    return cells


def _place_fork(rng, cells, directions, wrong, forbidden, structural, shape, length_mode=None):
    """A receiver is constructed relative to the actual oriented key shadow."""
    for _ in range(65):
        inward = int(rng.choice(directions))
        out = DIRS[(inward + 2) % 4]
        perp = out[1], -out[0]
        qs = [r * perp[0] + c * perp[1] for r, c in cells]
        span = max(qs) - min(qs) + 1
        fspan = (span - 2 if span >= 3 else span + 2) if wrong else span
        guard = 1 if rng.random() < 0.85 else 2
        clearance = int(rng.integers(1, 3))
        width = fspan + 2 * (guard + clearance)
        q0 = guard + clearance
        qbase = min(qs) + (span - fspan) // 2 - q0
        if length_mode == "short_center":
            lengths = (int(rng.integers(3, 6)), 1, int(rng.integers(3, 6)))
        elif length_mode == "long_center":
            lengths = (int(rng.integers(1, 4)), int(rng.integers(4, 6)), int(rng.integers(1, 4)))
        else:
            lengths = tuple(map(int, rng.integers(1, 6, size=3)))
        distance = int(rng.integers(2, 9))
        kbase = min(r * out[0] + c * out[1] for r, c in cells) - distance - lengths[1] - 1
        local = {(0, q) for q in range(width)}
        for qa, qb, length in ((0, guard, lengths[0]), (q0, q0 + fspan, lengths[1]),
                               (width - guard, width, lengths[2])):
            local |= {(k, q) for k in range(1, length + 1) for q in range(qa, qb)}
        body = {((kbase + k) * out[0] + (qbase + q) * perp[0],
                 (kbase + k) * out[1] + (qbase + q) * perp[1]) for k, q in local}
        if not _inside(body, shape) or body & forbidden or _halo(body) & structural:
            continue
        fork = _parse_fork(body)
        candidate = _dock_candidate(fork, cells, structural | body, shape, ignore_span=wrong)
        if candidate is not None:
            return fork, candidate
    return None


def _make_pair(rng, index, test):
    # The first two demonstrations contrast sideways and nose-first docking,
    # different visible masks, and one versus two waypoint moves before capture.
    if not test and index == 0:
        mask = _mask(1, 1, bool(rng.integers(2)), 0)  # length 2, breadth 3
    elif not test and index == 1:
        mask = _mask(1, 2, bool(rng.integers(2)), 1)  # length 4, breadth 3
    else:
        mask = _mask(int(rng.integers(1, 3)), int(rng.integers(1, 3)),
                     bool(rng.integers(2)), int(rng.integers(2)))
    capture = int(rng.integers(3, 5)) if test else (1 if index % 2 == 0 else 2)
    count = capture + int(rng.integers(2, 4))
    for _ in range(180):
        grown = _grow_route(rng, mask, count, need_turn=capture >= 2)
        if grown is None:
            continue
        points, ds = grown
        prefix = _prefix_sweep(mask, points, ds, capture)
        visible = _foot(mask, points[0], ds[0]) | set(points[1:])
        if prefix & set(points[capture + 1:]):
            continue
        # All initial singletons must remain visibly separate, including those
        # intentionally left unvisited after capture.
        if any(q in _halo(_foot(mask, points[0], ds[0])) for q in points[1:]):
            continue
        arrival = _foot(mask, points[capture], ds[capture - 1])
        mode = "side" if capture == 1 else "head" if not test and index == 1 else str(rng.choice(["side", "head"]))
        dirs = ([(ds[capture - 1] + 1) % 4, (ds[capture - 1] - 1) % 4]
                if mode == "side" else [ds[capture - 1]])
        length_mode = ("short_center" if index == 0 else "long_center" if index == 1 else None) if not test else None
        placed = _place_fork(rng, arrival, dirs, False, prefix | visible, set(), (30, 30), length_mode)
        if placed is None:
            continue
        catcher, docking = placed
        # Docking must not sweep through any *other* waypoint; route markers are
        # never used as a collision exception in the generated examples.
        if (set(points) - {points[capture]}) & docking["swept"]:
            continue
        structures = set(catcher["cells"])
        reserved = prefix | docking["swept"] | visible
        # A centered but wrong-size receiver is a real counterexample to the
        # center-alignment-only shortcut, at the initial checkpoint.
        wrong = _place_fork(rng, _foot(mask, points[0], ds[0]), list(range(4)), True,
                            reserved, structures, (30, 30))
        if wrong is None:
            continue
        decoy, decoy_docking = wrong
        structures |= decoy["cells"]
        forks = [catcher, decoy]
        # Tests add a correctly fitting receiver at a later waypoint. The first
        # matching checkpoint wins; following all waypoints would choose later.
        if test or index == 1 or (index >= 2 and rng.random() < 0.55):
            late_index = len(points) - 1
            late = _place_fork(rng, _foot(mask, points[late_index], ds[-1]), list(range(4)),
                               False, reserved | visible, structures, (30, 30))
            if late is None:
                continue
            forks.append(late[0])
            structures |= late[0]["cells"]
        palette = list(map(int, rng.choice(np.arange(1, 10), 2 + len(forks), replace=False)))
        bg, ink = palette[:2]
        grid = np.full((30, 30), bg, dtype=np.int8)
        for fork, color in zip(forks, palette[2:]):
            fork["color"] = color
            _paint(grid, fork["cells"], color)
        _paint(grid, visible, ink)
        try:
            a = _read_input(grid)
            output, result = _execute(grid, a)
            if result["checkpoint"] != capture or a["forks"][result["fork"]]["cells"] != catcher["cells"]:
                continue
            if result["final"] != docking["final"]:
                continue
            no_size, bad = _execute(grid, a, ignore_span=True)
            if bad["checkpoint"] != 0 or np.array_equal(no_size, output):
                continue
        except AssertionError:
            continue
        # Independently paint the constructed execution, before its metadata is
        # discarded. This checks the generic input parser against construction.
        intended = np.full(grid.shape, bg, dtype=np.int8)
        expected_swept = prefix | docking["swept"]
        _paint(intended, expected_swept - docking["final"], 0)
        for fork in forks:
            _paint(intended, fork["cells"], fork["color"])
        _paint(intended, docking["final"], ink)
        assert np.array_equal(output, intended)
        # Crop only during input generation, using the whole full-route geometry
        # and all output footprints; keep independent random background margins.
        all_cells = visible | structures | expected_swept | decoy_docking["swept"]
        for i, p in enumerate(points):
            all_cells |= _foot(mask, p, ds[max(0, i - 1)])
        if len(forks) == 3:
            all_cells |= late[1]["swept"]
        r0, r1 = min(r for r, c in all_cells), max(r for r, c in all_cells)
        c0, c1 = min(c for r, c in all_cells), max(c for r, c in all_cells)
        r0 = max(0, r0 - int(rng.integers(4)))
        c0 = max(0, c0 - int(rng.integers(4)))
        r1 = min(29, r1 + int(rng.integers(4)))
        c1 = min(29, c1 + int(rng.integers(4)))
        inp, out = grid[r0:r1 + 1, c0:c1 + 1].copy(), output[r0:r1 + 1, c0:c1 + 1].copy()
        if np.count_nonzero(inp == bg) <= inp.size / 2:
            continue
        assert np.array_equal(solve(inp), out)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded placement budget exhausted")


def _valid_count(value, low, high):
    return (isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            and low <= int(value) <= high)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _valid_count(num_train, 2, 10) or not _valid_count(num_test, 1, 4) or num_train <= num_test:
        raise ValueError("counts must be integers: 2<=train<=10, 1<=test<=4, train>test")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(int(count)):
            for _ in range(12):
                pair = _make_pair(rng, i, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct-input budget exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    grid, output = np.asarray(pair["input"]), np.asarray(pair["output"])
    a = _read_input(grid)
    expected, result = _execute(grid, a)
    assert np.array_equal(output, expected), "incorrect trajectory, final mask, or receiver preservation"
    assert output.shape == grid.shape and not np.array_equal(grid, output)
    assert result["fork"] is not None and 1 <= result["checkpoint"] < len(a["points"]) - 1
    assert np.count_nonzero(output == a["ink"]) == len(a["mask"]), "the rigid arrow must survive exactly"
    assert set(map(int, np.unique(output))) - set(map(int, np.unique(grid))) == {0}
    for fork in a["forks"]:
        assert min(run[2] for run in fork["runs"]) >= 1
        assert all(int(output[p]) == fork["color"] for p in fork["cells"])
    final = result["final"]
    assert all(int(output[p]) == 0 for p in result["swept"] - final)
    # Each checkpoint's hypothetical mask remains on the board, even after the
    # early stop. The unexecuted suffix is route evidence, not more motion.
    for i, p in enumerate(a["points"]):
        body = _foot(a["mask"], p, a["directions"][max(0, i - 1)])
        assert _inside(body, grid.shape)
        assert not (set(a["points"][i + 1:]) & _halo(body))
    wrong_output, wrong = _execute(grid, a, ignore_span=True)
    assert wrong["checkpoint"] == 0 and not np.array_equal(wrong_output, output)
    # Omit A's traversal: only the initial checkpoint can be considered. There
    # is no exact receiver there, whereas the real process eventually captures.
    initial = _foot(a["mask"], a["points"][0], a["facing"])
    assert all(_dock_candidate(f, initial, a["structural"], grid.shape) is None for f in a["forks"])
    assert final != initial
    # Omitting B entirely would consume all waypoints and place an arrow at the
    # final singleton; this necessarily differs from the early captured mask.
    route_final = _foot(a["mask"], a["points"][-1], a["directions"][-1])
    assert route_final != final
    fork = a["forks"][result["fork"]]
    return a, result, (fork["inward"] - result["facing"]) % 4


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert _valid_count(len(puzzle["train"]), 2, 10) and _valid_count(len(puzzle["test"]), 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for value in pair.values():
                g = np.asarray(value)
                assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
                assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(_pair_info(pair))
    train, test = infos["train"], infos["test"]
    assert train[0][2] in (1, 3) and train[1][2] == 0
    assert train[0][0]["mask"] != train[1][0]["mask"]
    assert train[0][1]["checkpoint"] == 1 and train[1][1]["checkpoint"] == 2
    assert train[1][0]["directions"][0] != train[1][0]["directions"][1]
    first_fork = train[0][0]["forks"][train[0][1]["fork"]]
    second_fork = train[1][0]["forks"][train[1][1]["fork"]]
    assert first_fork["runs"][1][2] < min(first_fork["runs"][i][2] for i in (0, 2))
    assert second_fork["runs"][1][2] > max(second_fork["runs"][i][2] for i in (0, 2))
    assert all(r[1]["checkpoint"] > max(t[1]["checkpoint"] for t in train) for r in test)
    assert all(len(r[0]["forks"]) == 3 for r in test)
    for pair, (a, result, mode) in [(puzzle["train"][1], train[1])] + list(zip(puzzle["test"], test)):
        late = _foot(a["mask"], a["points"][-1], a["directions"][-1])
        assert any(_dock_candidate(f, late, a["structural"], np.asarray(pair["input"]).shape)
                   is not None for fi, f in enumerate(a["forks"]) if fi != result["fork"])


def self_test() -> dict:
    """42 seeds, no saved grids or execution traces; bounded negative checks."""
    signatures, shapes, masks, routes, geometries, modes, colors = set(), set(), set(), set(), set(), set(), set()
    initial_ds, final_ds, captures, spans = set(), set(), set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, result, mode = _pair_info(pair)
                pairs += 1
                shapes.add(np.asarray(pair["input"]).shape)
                masks.add(tuple(sorted(a["mask"])))
                routes.add(tuple((b[0] - a["points"][0][0], b[1] - a["points"][0][1]) for b in a["points"]))
                modes.add(mode)
                colors.add((a["bg"], a["ink"]))
                initial_ds.add(a["facing"])
                final_ds.add(result["facing"])
                captures.add(result["checkpoint"])
                for fork in a["forks"]:
                    geometries.add((fork["width"], fork["runs"]))
                    spans.add(fork["span"])
                digest.update(bytes(pair["input"].shape))
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 15 and len(masks) >= 8 and len(routes) >= 150
    assert len(geometries) >= 100 and {0, 1, 3} <= modes
    assert initial_ds == final_ds == set(range(4)) and captures == {1, 2, 3, 4}
    assert len(spans) >= 5 and len(colors) >= 35
    p, q = generate_puzzle(91, 2, 1), generate_puzzle(91, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    try:
        _decode_route((3, 3), {(3, 6), (6, 3)})
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous waypoint junction accepted")
    malformed = {(0, c) for c in range(7)} | {(r, c) for r in (1, 2) for c in (0, 6)}
    try:
        _parse_fork(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError("zero center spoke accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures), shapes=len(shapes),
                masks=len(masks), relative_routes=len(routes), fork_geometries=len(geometries),
                docking_modes=sorted(modes), capture_checkpoints=sorted(captures),
                spans=sorted(spans), palettes=len(colors))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
