"""Routed corner restoration. Only NumPy and the Python standard library.

The decoder reads geometry and observations from one input. Construction uses
its own path-to-stencil plan to paint labels; it never calls solve for a label.
No examples, seeds, palettes, or templates are retained by the solver.
"""

from itertools import combinations
import hashlib
import numpy as np


_D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
_DOMAIN = tuple((u, v) for u in range(-1, 2) for v in range(-1, 2))
_SEEDS = (0, 149026050, 495904818, 828201827, 644301787, 234260412,
          48800595, 269069749, 519940353, 428472645, 602585745, 114581829,
          918104848, 127840488, 45585936, 840985705, 339221609, 247675794,
          118408400, 939181557, 298800813, 498683675, 80332220, 541950264,
          608728743, 12550906, 166484864, 205724982, 782669020, 394934101,
          563151583, 375277155, 173152842, 215032245, 118291050, 545562929,
          38287846, 104907287, 496138755, 647682455, 331697078, 65257706)


def _add(p, d, k=1):
    return p[0] + k * d[0], p[1] + k * d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        found, todo = {start}, [start]
        while todo:
            p = todo.pop()
            for d in _D4:
                q = _add(p, d)
                if q in unseen:
                    unseen.remove(q)
                    found.add(q)
                    todo.append(q)
        result.append(found)
    return result


def _paths(mask):
    """Every wire pixel must belong to one simple path, never a loop/branch."""
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    for component in _components(cells):
        neighbors = {p: sorted(_add(p, d) for d in _D4 if _add(p, d) in component)
                     for p in component}
        assert all(len(q) in (1, 2) for q in neighbors.values()), "branch or isolated wire"
        ends = sorted(p for p in component if len(neighbors[p]) == 1)
        assert len(ends) == 2, "closed wire loop"
        path, previous = [ends[0]], None
        while path[-1] != ends[1]:
            p = path[-1]
            nxt = [q for q in neighbors[p] if q != previous]
            assert len(nxt) == 1
            previous = p
            path.append(nxt[0])
        assert len(path) == len(component)
        result.append(path)
    return sorted(result, key=lambda path: (path[0], path[-1]))


def _connected8(pattern):
    if not pattern:
        return False
    seen, todo = {min(pattern)}, [min(pattern)]
    while todo:
        u, v = todo.pop()
        for du, dv in _DOMAIN:
            q = u + du, v + dv
            if q in pattern and q not in seen:
                seen.add(q)
                todo.append(q)
    return len(seen) == len(pattern)


def _transpose(pattern):
    return frozenset((v, u) for u, v in pattern)


def _positions(port, pattern):
    r, c = port["corner"]
    dy, dx = port["orientation"]
    return {(r + dy * u, c + dx * v) for u, v in pattern}


def _outline(port):
    r, c = port["corner"]
    dy, dx = port["orientation"]
    return {(r + dy * k, c) for k in range(4)} | {(r, c + dx * k) for k in range(4)}


def _turn(inp, centers, axis, track, gate):
    # axis is the direction of the INPUT bars, normal the OUTPUT bars.
    normal = (axis[1], axis[0])
    writes = {}
    for center in centers:
        for u in (-1, 0, 1):
            for v in (-1, 0, 1):
                p = _add(_add(center, normal, u), axis, v)
                assert _inside(p, inp.shape)
                value = gate if v == 0 else track
                assert p not in writes or writes[p] == value, "conflicting local rewiring"
                writes[p] = value
    out = inp.copy()
    for p, value in writes.items():
        out[p] = value
    return out


def _read_candidate(inp, background, gate):
    bars = _components(tuple(map(int, p)) for p in np.argwhere(inp == gate))
    assert bars
    axes, centers, track_colors = set(), [], set()
    for bar in bars:
        assert len(bar) == 3
        rows, cols = {p[0] for p in bar}, {p[1] for p in bar}
        assert len(rows) == 1 or len(cols) == 1
        axis = (0, 1) if len(rows) == 1 else (1, 0)
        center = sorted(bar)[1]
        assert bar == {_add(center, axis, k) for k in (-1, 0, 1)}
        normal = axis[1], axis[0]
        for sign in (-1, 1):
            points = [_add(_add(center, normal, sign), axis, k) for k in (-1, 0, 1)]
            assert all(_inside(p, inp.shape) for p in points)
            values = {int(inp[p]) for p in points}
            assert len(values) == 1
            track_colors.update(values)
        axes.add(axis)
        centers.append(center)
    assert len(axes) == len(track_colors) == 1
    axis, track = next(iter(axes)), next(iter(track_colors))
    assert len({background, gate, track}) == 3
    structural = _turn(inp, centers, axis, track, gate)
    paths = _paths(structural == track)
    assert paths
    ports, frames, frame_cells = {}, set(), set()
    for path in paths:
        for endpoint, neighbor in ((path[0], path[1]), (path[-1], path[-2])):
            outward = endpoint[0] - neighbor[0], endpoint[1] - neighbor[1]
            corner = _add(endpoint, outward, 2)
            line = [_add(corner, outward, k) for k in (2, 3)]
            assert all(_inside(p, inp.shape) for p in line)
            frame = int(inp[line[0]])
            assert frame not in (background, gate, track) and all(inp[p] == frame for p in line)
            possibilities = []
            for side in (-1, 1):
                perp = side * outward[1], -side * outward[0]
                arm = [_add(corner, perp, k) for k in (2, 3)]
                if all(_inside(p, inp.shape) and inp[p] == frame for p in arm):
                    dy, dx = outward[0] + perp[0], outward[1] + perp[1]
                    port = {"corner": corner, "orientation": (dy, dx), "endpoint": endpoint}
                    if all(_inside(p, inp.shape) for p in _outline(port)):
                        possibilities.append(port)
            assert len(possibilities) == 1, "missing/ambiguous corner arms"
            port = possibilities[0]
            assert corner not in {p["corner"] for p in ports.values()}
            outline = _outline(port)
            assert not outline & frame_cells, "overlapping brackets"
            frame_cells.update(outline)
            frames.add(frame)
            ports[endpoint] = port
    assert len(frames) == 1
    frame = frames.pop()
    roles = {background, gate, track, frame}
    accent_cells = {tuple(map(int, p)) for p in np.argwhere(~np.isin(inp, list(roles)))}
    assigned, corner_boxes = set(), set()
    for port in ports.values():
        box = _positions(port, _DOMAIN)
        assert all(_inside(p, inp.shape) for p in box)
        assert not box & corner_boxes, "overlapping ornament neighborhoods"
        assert not box & set(map(tuple, np.argwhere(inp == track)))
        corner_boxes.update(box)
        observed = box & accent_cells
        colors = {int(inp[p]) for p in observed}
        assert len(colors) <= 1, "a corner must have a single local accent color"
        r, c = port["corner"]
        dy, dx = port["orientation"]
        port["clue"] = frozenset(((y - r) * dy, (x - c) * dx) for y, x in observed)
        port["color"] = next(iter(colors)) if colors else None
        assigned.update(observed)
    assert assigned == accent_cells, "accent not associated with a corner"
    assert all(tuple(p) in frame_cells for p in np.argwhere(inp == frame))
    assert all(inp[p] == frame or p in accent_cells for p in frame_cells)
    answer = structural.copy()
    stencils, missing, reflection_witness = [], 0, False
    gate_depths = []
    for path in paths:
        ends = [ports[path[0]], ports[path[-1]]]
        clues = [p["clue"] for p in ends if p["clue"]]
        stencil = max(clues, key=len) if clues else frozenset()
        if stencil:
            assert 3 <= len(stencil) <= 7 and _connected8(stencil)
            assert all(clue <= stencil for clue in clues), "incompatible routed copies"
        for port in ends:
            port["stencil"] = stencil
            if port["clue"]:
                missing += len(stencil - port["clue"])
                for p in _positions(port, stencil):
                    assert inp[p] in (background, frame, port["color"])
                    answer[p] = port["color"]
        if len(clues) == 2 and len(clues[0]) != len(clues[1]) and stencil != _transpose(stencil):
            a, b = ends[0]["orientation"], ends[1]["orientation"]
            reflection_witness |= (a[0] == b[0]) != (a[1] == b[1])
        stencils.append(stencil)
        cells = set(path)
        gate_depths.append(sum(any(_add(center, axis, k) in cells for k in (-1, 1))
                               for center in centers))
    # Check the substantive routing witness on ORNAMENTS, not merely on bars.
    original = _paths(inp == track)
    original_partner = {}
    for path in original:
        assert path[0] in ports and path[-1] in ports
        original_partner[path[0]] = ports[path[-1]]
        original_partner[path[-1]] = ports[path[0]]
    routing_witness = any(port["clue"] and port["clue"] < port["stencil"]
                          and original_partner[end]["clue"] != port["stencil"]
                          for end, port in ports.items())
    return answer, {"background": background, "track": track, "frame": frame,
                    "gate": gate, "axis": axis, "centers": centers,
                    "ports": ports, "paths": paths, "stencils": stencils,
                    "structural": structural, "missing": missing,
                    "reflection_witness": reflection_witness,
                    "routing_witness": routing_witness, "gate_depths": gate_depths}


def _analyze(input_grid):
    inp = np.asarray(input_grid)
    assert inp.ndim == 2 and all(1 <= n <= 30 for n in inp.shape)
    assert np.issubdtype(inp.dtype, np.integer) and np.all((inp >= 0) & (inp <= 9))
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    assert background == 0 and np.count_nonzero(counts == counts[background]) == 1
    candidates = []
    for gate in map(int, np.unique(inp)):
        if gate == background:
            continue
        try:
            candidates.append(_read_candidate(inp, background, gate))
        except AssertionError:
            continue
    assert len(candidates) == 1, "input does not have a unique routed-corner interpretation"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every role, path, corner, stencil and local color from this input."""
    return _analyze(input_grid)[0]


def _sample_pattern(rng, required, asymmetric=False, incomparable=None):
    for _ in range(100):
        size = int(rng.integers(3, 8))
        pattern = {required}
        while len(pattern) < size:
            frontier = sorted(p for p in _DOMAIN if p not in pattern and any(
                max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in pattern))
            pattern.add(frontier[int(rng.integers(len(frontier)))])
        pattern = frozenset(pattern)
        if asymmetric and pattern == _transpose(pattern):
            continue
        if incomparable is not None and (pattern <= incomparable or incomparable <= pattern):
            continue
        return pattern
    # Finite exhaustive rescue over the same domain, not a template dictionary.
    for n in range(3, 8):
        for values in combinations(_DOMAIN, n):
            pattern = frozenset(values)
            if required not in pattern or not _connected8(pattern):
                continue
            if asymmetric and pattern == _transpose(pattern):
                continue
            if incomparable is None or not (pattern <= incomparable or incomparable <= pattern):
                return pattern
    raise AssertionError("no compatible stencil")


def _build_wires(shape, rows, left, right, gates, track, gate):
    base = np.zeros(shape, dtype=np.int8)
    for y, a, b in zip(rows, left, right):
        base[y, a:b + 1] = track
    # Clear main rails before adding approaches, so construction is simultaneous.
    for separator, c in gates:
        for y in (rows[separator], rows[separator + 1]):
            base[y, c - 1:c + 2] = 0
    centers = []
    for separator, c in gates:
        a, b = rows[separator], rows[separator + 1]
        m = (a + b) // 2
        base[a:m, c - 2] = track
        base[a:m, c + 2] = track
        base[m + 1:b + 1, c - 2] = track
        base[m + 1:b + 1, c + 2] = track
        base[m - 1, c - 1:c + 2] = track
        base[m + 1, c - 1:c + 2] = track
        base[m, c - 1:c + 2] = gate
        centers.append((m, c))
    return base, centers


def _geometry(rng, lanes, gate_count, palette, fan=False, independent=True):
    track, gate, frame = palette[:3]
    for _ in range(100):
        width = int(rng.integers(29 if fan else (27 if gate_count > 1 else 23), 31))
        rows = [int(rng.integers(3, 5))]
        for _ in range(lanes - 1):
            rows.append(rows[-1] + (7 if lanes == 4 else int(rng.integers(7, 10))))
        height = rows[-1] + int(rng.integers(4, 6))
        if height > 30:
            continue
        lc = list(map(int, rng.integers(3, 5, size=lanes)))
        rc = [width - int(rng.integers(4, 6)) for _ in rows]
        left, right = [c + 2 for c in lc], [c - 2 for c in rc]
        lo, hi = max(left) + 3, min(right) - 3
        cols = list(range(lo, hi + 1))
        separated = [(a, b) for a in cols for b in cols if abs(a - b) >= 5]
        if gate_count == 1:
            gates = [(int(rng.integers(lanes - 1)), int(rng.choice(cols)))]
        elif fan:
            triples = [(a, b, c) for a in cols for b in cols for c in cols if a + 5 <= b and b + 5 <= c]
            if not triples:
                continue
            a, b, c = triples[int(rng.integers(len(triples)))]
            busy = int(rng.integers(2))
            gates = [(busy, a), (busy, c), (1 - busy, b)]
        elif independent and gate_count == 2 and lanes == 4 and rng.random() < .35:
            gates = [(0, int(rng.choice(cols))), (2, int(rng.choice(cols)))]
        else:
            if not separated:
                continue
            a, b = separated[int(rng.integers(len(separated)))]
            start = int(rng.integers(lanes - gate_count))
            gates = [(start, a), (start + 1, b)]
            if gate_count == 3:
                choices = [c for c in cols if abs(c - b) >= 5]
                gates.append((start + 2, int(rng.choice(choices))))
        base, centers = _build_wires((height, width), rows, left, right, gates, track, gate)
        port_plan, outlines = {}, set()
        valid = True
        for i, y in enumerate(rows):
            for c, dx, end in ((lc[i], -1, left[i]), (rc[i], 1, right[i])):
                port = {"corner": (y, c), "orientation": (int(rng.choice((-1, 1))), dx),
                        "endpoint": (y, end)}
                outline = _outline(port)
                if outline & outlines or any(not _inside(p, base.shape) or base[p] != 0 for p in outline):
                    valid = False
                    break
                outlines.update(outline)
                port_plan[y, end] = port
            if not valid:
                break
        if not valid:
            continue
        for p in outlines:
            base[p] = frame
        try:
            paths = _paths(_turn(base, centers, (0, 1), track, gate) == track)
            assert len(paths) == lanes
            assert {p for path in paths for p in (path[0], path[-1])} == set(port_plan)
            # Also check the original rails remain simple and port-terminated.
            assert len(_paths(base == track)) == lanes
        except AssertionError:
            continue
        return base, centers, port_plan, paths
    raise AssertionError("bounded geometry construction exhausted")


def _make_pair(rng, lanes, gate_count, palette, witness, offset, fan=False):
    for _ in range(60):
        base, centers, ports, paths = _geometry(rng, lanes, gate_count, palette, fan,
                                               independent=not offset)
        gate_cells = {p for y, x in centers for p in ((y, x - 1), (y, x + 1))}
        active = [i for i, path in enumerate(paths) if set(path) & gate_cells]
        rng.shuffle(active)
        if offset:
            # Teach actual restoration through TWO gates, not just the presence
            # of two switches somewhere in the demonstration.
            long_route = max(active, key=lambda i: len(set(paths[i]) & gate_cells))
            assert len(set(paths[long_route]) & gate_cells) == 2
            active.remove(long_route)
            active.insert(0, long_route)
        if witness:
            informative = [i for i in active if (
                (ports[paths[i][0]]["orientation"][0] == ports[paths[i][-1]]["orientation"][0])
                != (ports[paths[i][0]]["orientation"][1] == ports[paths[i][-1]]["orientation"][1]))]
            if not informative:
                continue
            active.remove(informative[0])
            active.insert(0, informative[0])
        assert len(active) >= 2
        common = (0, 0) if witness else _DOMAIN[int(rng.integers(9))]
        if offset:
            common = _DOMAIN[int(rng.choice([i for i, p in enumerate(_DOMAIN) if p != (0, 0)]))]
        first = _sample_pattern(rng, common, asymmetric=witness)
        second = _sample_pattern(rng, common, incomparable=first)
        forced = {active[0]: first, active[1]: second}
        extra = [i for i in range(len(paths)) if i not in forced]
        if offset:
            assert len(extra) == 2
        inp = base.copy()
        # Labels are painted from construction plans, not the decoder's answer.
        out = _turn(base, centers, (0, 1), palette[0], palette[1])
        local_colors = list(palette[3:])
        for i, path in enumerate(paths):
            ends = [ports[path[0]], ports[path[-1]]]
            if int(rng.integers(2)):
                ends.reverse()
            if i in forced:
                stencil, mode = forced[i], "partial"
            else:
                stencil = _sample_pattern(rng, _DOMAIN[int(rng.integers(9))])
                if offset:
                    mode = "partial" if i == extra[0] else "single"
                else:
                    mode = str(rng.choice(("partial", "full", "single", "empty"), p=(.55, .15, .18, .12)))
            for j, port in enumerate(ends):
                if mode == "empty" or (mode == "single" and j == 1):
                    continue
                if i in forced:
                    color = local_colors[0]
                elif offset:
                    color = local_colors[1 + j] if mode == "partial" else local_colors[1]
                else:
                    color = int(rng.choice(local_colors))
                clue = stencil
                if j == 1 and mode == "partial":
                    if i in forced:
                        clue = frozenset({common})
                    else:
                        choices = sorted(stencil)
                        k = int(rng.integers(1, len(choices)))
                        clue = frozenset(choices[int(a)] for a in rng.permutation(len(choices))[:k])
                for p in _positions(port, stencil):
                    assert out[p] in (0, palette[2])
                    out[p] = color
                for p in _positions(port, clue):
                    assert inp[p] in (0, palette[2])
                    inp[p] = color
        rotation = int(rng.integers(4))
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        prediction, info = _analyze(inp)
        assert np.array_equal(prediction, out), "construction and input decoder disagree"
        if not info["routing_witness"] or (witness and not info["reflection_witness"]):
            continue
        if gate_count == 3 and max(info["gate_depths"]) < 3:
            continue
        return {"input": inp, "output": out}
    raise AssertionError("bounded informative-pair construction exhausted")


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable = bool(rng.integers(2))
    fixed = tuple(map(int, rng.choice(np.arange(1, 10), 6, replace=False)))
    witness_index, offset_index = map(int, rng.permutation(nt)[:2])
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            for _ in range(20):
                palette = fixed if stable else tuple(map(int, rng.choice(np.arange(1, 10), 6, replace=False)))
                if split == "test":
                    fan = bool(rng.integers(2))
                    lanes, gates = (3 if fan else 4), 3
                else:
                    fan = False
                    gates = 2 if i == offset_index else (1 if i == witness_index else int(rng.integers(1, 3)))
                    lanes = (4 if i == offset_index else
                             int(rng.integers(2, 4)) if i == witness_index else
                             int(rng.integers(gates + 1, 5)))
                pair = _make_pair(rng, lanes, gates, palette,
                                  split == "train" and i == witness_index,
                                  split == "train" and i == offset_index, fan)
                key = pair["input"].shape, pair["input"].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, reflection = set(), False
    single_mark = local_color = offset_clue = chained = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and np.any(inp != out)
            key = inp.shape, inp.astype(np.int8).tobytes()
            assert key not in seen
            seen.add(key)
            predicted, info = _analyze(inp)
            assert np.array_equal(predicted, out), "wrong output"
            assert info["routing_witness"] and info["missing"] >= 4
            nonempty = [s for s in info["stencils"] if s]
            assert any(not (a <= b or b <= a) for a in nonempty for b in nonempty)
            accent = ~np.isin(inp, [info[k] for k in ("background", "track", "frame", "gate")])
            assert np.array_equal(inp[accent], out[accent]), "an existing clue was erased"
            assert np.count_nonzero(out == info["gate"]) == np.count_nonzero(inp == info["gate"])
            assert np.any(out != info["structural"]), "omitting stencil completion had no effect"
            assert set(map(int, np.unique(inp))) == set(map(int, np.unique(out)))
            if split == "train":
                assert 1 <= len(info["centers"]) <= 2
                reflection |= info["reflection_witness"]
                for index, path in enumerate(info["paths"]):
                    ends = [info["ports"][p] for p in (path[0], path[-1])]
                    chained |= (info["gate_depths"][index] == 2 and all(p["clue"] for p in ends)
                                and ends[0]["clue"] != ends[1]["clue"])
                    single_mark |= sum(bool(p["clue"]) for p in ends) == 1
                    local_color |= (all(p["clue"] for p in ends) and ends[0]["color"] != ends[1]["color"]
                                    and ends[0]["clue"] != ends[1]["clue"])
                offset_clue |= any(p["clue"] and p["clue"] < p["stencil"] and (0, 0) not in p["clue"]
                                   for p in info["ports"].values())
            else:
                assert len(info["centers"]) == max(info["gate_depths"]) == 3
    assert reflection, "training must distinguish reflections from quarter turns"
    assert single_mark and local_color and offset_clue and chained, "missing educational training witness"


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                grid = np.asarray(pair[key], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _countermodel_predictions(inp, info):
    """Concrete omissions, keeping correct divider pixels in every candidate.

    These are diagnostic alternatives, never used by solve or construction.
    The original-wire alternative leaves an incompatible pair unresolved.
    """
    original_wires = info["structural"].copy()
    for path in _paths(inp == info["track"]):
        ports = [info["ports"][p] for p in (path[0], path[-1])]
        clues = [p["clue"] for p in ports if p["clue"]]
        pattern = max(clues, key=len) if clues else frozenset()
        if not all(clue <= pattern for clue in clues):
            continue
        for port in ports:
            if port["clue"]:
                for p in _positions(port, pattern):
                    original_wires[p] = port["color"]
    rotation = info["structural"].copy()
    for path in info["paths"]:
        marked = [info["ports"][p] for p in (path[0], path[-1]) if info["ports"][p]["clue"]]
        if not marked:
            continue
        source = max(marked, key=lambda port: len(port["clue"]))
        sy, sx = source["orientation"]
        for port in marked:
            dy, dx = port["orientation"]
            pattern = source["clue"] if sy * sx == dy * dx else _transpose(source["clue"])
            for p in _positions(port, pattern):
                rotation[p] = port["color"]
    return original_wires, info["structural"], rotation


def self_test():
    """Stream 42 seeds; retain only bounded aggregate sets, never all puzzles."""
    signatures, shapes, structures, patterns, palettes, axes = set(), set(), set(), set(), set(), set()
    lanes_seen, mark_counts, clue_sizes, gate_counts = set(), set(), set(), set()
    absent_corner = asymmetric = repeated = unchanged_port = fan = 0
    pairs = 0
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        rotation_refuted = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                before = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, before), "solve mutated its input"
                _, info = _analyze(inp)
                original_wires, no_completion, rotation = _countermodel_predictions(inp, info)
                assert not np.array_equal(original_wires, pair["output"]), "original wiring shortcut survived"
                assert not np.array_equal(no_completion, pair["output"]), "no-completion shortcut survived"
                if split == "train":
                    rotation_refuted |= not np.array_equal(rotation, pair["output"])
                pairs += 1
                shapes.add(inp.shape)
                lanes_seen.add(len(info["paths"]))
                axes.add(info["axis"])
                gate_counts.add(len(info["centers"]))
                palettes.add(tuple(info[k] for k in ("track", "gate", "frame")))
                normalized = np.zeros(inp.shape, dtype=np.int8)
                for value, role in enumerate(("track", "gate", "frame"), 1):
                    normalized[inp == info[role]] = value
                normalized[~np.isin(inp, [0, info["track"], info["gate"], info["frame"]])] = 4
                structures.add(hashlib.sha256(bytes(inp.shape) + normalized.tobytes()).hexdigest())
                fan += int(len({c[0] for c in info["centers"]}) < len(info["centers"]) if info["axis"] == (0, 1)
                           else len({c[1] for c in info["centers"]}) < len(info["centers"]))
                colors = [p["color"] for p in info["ports"].values() if p["clue"]]
                repeated += int(len(colors) > len(set(colors)))
                for path in info["paths"]:
                    mark_counts.add(sum(bool(info["ports"][p]["clue"]) for p in (path[0], path[-1])))
                for port in info["ports"].values():
                    clue_sizes.add(len(port["clue"]))
                    if port["stencil"]:
                        patterns.add(tuple(sorted(port["stencil"])))
                        absent_corner += int((0, 0) not in port["stencil"])
                        asymmetric += int(port["stencil"] != _transpose(port["stencil"]))
                    unchanged_port += int(not port["clue"])
        assert rotation_refuted, "rotation convention was not refuted in training"
    assert len(signatures) == 42 and pairs == 210
    assert len(shapes) >= 45 and len(structures) >= 180 and len(patterns) >= 75
    assert len(palettes) >= 30 and axes == {(0, 1), (1, 0)}
    assert lanes_seen == {2, 3, 4} and gate_counts == {1, 2, 3}
    assert mark_counts == {0, 1, 2} and {0, 1, 2, 3, 4, 5, 6, 7} <= clue_sizes
    assert absent_corner and asymmetric and repeated == pairs and unchanged_port and fan
    assert _signature(generate_puzzle(12345, 6, 3)) == _signature(generate_puzzle(12345, 6, 3))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(7000 + 10 * nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    corrupt = generate_puzzle(0)
    old = int(corrupt["test"][0]["output"][0, 0])
    corrupt["test"][0]["output"][0, 0] = (old + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": pairs, "shapes": len(shapes),
            "normalized_structures": len(structures), "stencils": len(patterns),
            "role_palettes": len(palettes), "lane_counts": sorted(lanes_seen),
            "gate_counts": sorted(gate_counts), "marked_ports_per_path": sorted(mark_counts),
            "clue_sizes": sorted(clue_sizes), "fan_scenes": fan,
            "independent_semantic_check": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
