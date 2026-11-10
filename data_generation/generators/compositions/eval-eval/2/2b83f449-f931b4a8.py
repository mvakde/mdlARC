"""Rewired ribbon loom: dual-scale color tapes transported in path coordinates.

Only numpy and the standard library are used. No files are read or written by
generation, solving, validation, or self_test. See task_spec.md for the rule.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))
REQUIRED = frozenset((
    "old_partner", "tile_carrier", "stretch_stencil", "fuse_first",
    "opaque_zero", "ignore_stencil", "far_phase", "reverse_tapes",
    "global_phase", "manhattan", "center_resize", "gravity", "tie",
    "visible_carrier", "visible_overwrite", "uneven", "bent",
))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a


def _paths(mask):
    """Return ordered, disjoint simple paths; reject branches and cycles."""
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    adj = {p: [q for dr, dc in DIRS
               if (q := (p[0] + dr, p[1] + dc)) in cells] for p in cells}
    if any(len(q) not in (1, 2) for q in adj.values()):
        raise ValueError("isolated or branching track")
    seen, paths = set(), []
    for start in sorted(p for p in cells if len(adj[p]) == 1):
        if start in seen:
            continue
        path, previous, current = [], None, start
        while current is not None:
            if current in seen:
                raise ValueError("repeated path cell")
            seen.add(current)
            path.append(current)
            choices = [q for q in adj[current] if q != previous]
            previous, current = current, (choices[0] if choices else None)
        paths.append(path)
    if seen != cells:
        raise ValueError("closed track cycle")
    return paths


def _rewire(core, track, gate):
    h, w = core.shape
    if h < 3 or h % 2 != 1 or w < 7 or not np.all(core[::2] == track):
        raise ValueError("expected unbroken parallel lanes")
    if not np.all(np.isin(core[1::2], (0, gate))):
        raise ValueError("unexpected separator ink")
    writes, centers, counts, overlaps = {}, [], [], 0
    for r in range(1, h, 2):
        xs = set(map(int, np.flatnonzero(core[r] == gate)))
        starts = sorted(c for c in xs if c - 1 not in xs)
        counts.append(len(starts))
        for c in starts:
            if c < 1 or c + 3 >= w or not {c, c + 1, c + 2} <= xs or c + 3 in xs:
                raise ValueError("gate must be an interior three-cell bar")
            center = c + 1
            centers.append((r, center))
            for rr in range(r - 1, r + 2):
                for cc in range(center - 1, center + 2):
                    value = gate if cc == center else track
                    if (rr, cc) in writes:
                        if writes[rr, cc] != value:
                            raise ValueError("conflicting gate rotations")
                        overlaps += 1
                    writes[rr, cc] = value
    if not centers:
        raise ValueError("at least one gate is required")
    structural = core.copy()
    for p, value in writes.items():
        structural[p] = value
    paths = _paths(structural == track)
    endpoints = {p for path in paths for p in (path[0], path[-1])}
    expected = {(r, c) for r in range(0, h, 2) for c in (0, w - 1)}
    if endpoints != expected:
        raise ValueError("every endpoint must have exactly one tape")
    return structural, paths, centers, counts, overlaps


def _world(info, p):
    r, c = p[0] + info["r0"], p[1] + info["n"] + 1
    return (r, c) if info["axis"] == 0 else (c, r)


def _orient(info, path, high=False, right_tie=False):
    def priority(p):
        r, c = _world(info, p)
        return (-r if high else r, c if right_tie else -c)
    return path if priority(path[0]) > priority(path[-1]) else path[::-1]


def _tape(info, endpoint):
    r = info["r0"] + endpoint[0]
    n = info["n"]
    if endpoint[1] == 0:
        return info["view"][r, :n][::-1]
    return info["view"][r, -n:]


def _set_tape(info, endpoint, tape):
    _tape(info, endpoint)[:] = tape


def _analyze(input_grid):
    a = _grid(input_grid)
    candidates = []
    for axis in (0, 1):
        g = a if axis == 0 else a.T
        locations = np.argwhere(g == 9)
        if len(locations) < 4:
            continue
        columns = sorted(set(map(int, locations[:, 1])))
        if len(columns) != 2:
            continue
        left, right = columns
        n = left
        if not (2 <= n <= 5 and g.shape[1] - right - 1 == n):
            continue
        r0, r1 = int(locations[:, 0].min()), int(locations[:, 0].max()) + 1
        h = r1 - r0
        if h < 3 or h % 2 != 1:
            continue
        expected = {(r, c) for r in range(r0, r1, 2) for c in columns}
        if set(map(tuple, locations)) != expected:
            continue
        if np.any(g[:r0]) or np.any(g[r1:]) or r0 + g.shape[0] - r1 > 3:
            continue
        if np.any(g[r0 + 1:r1:2, :left + 1]) or np.any(g[r0 + 1:r1:2, right:]):
            continue
        core = g[r0:r1, left + 1:right]
        if core.shape[1] < 7:
            continue
        tracks = np.unique(core[::2])
        gates = set(map(int, np.unique(core[1::2]))) - {0}
        if len(tracks) != 1 or len(gates) != 1:
            continue
        track, gate = int(tracks[0]), next(iter(gates))
        if track in (0, 9) or gate in (0, 9, track):
            continue
        try:
            structural, paths, centers, counts, overlaps = _rewire(core, track, gate)
        except ValueError:
            continue
        tapes = np.hstack((g[r0:r1:2, :n], g[r0:r1:2, -n:]))
        if np.any(np.isin(tapes, (track, gate, 9))):
            continue
        info = dict(array=a, view=g, axis=axis, n=n, r0=r0, r1=r1,
                    core=core, structural=structural, paths=paths,
                    centers=centers, counts=counts, overlaps=overlaps,
                    track=track, gate=gate)
        if any(len(p) < n or np.any(_tape(info, _orient(info, p)[0]) == 0) for p in paths):
            continue
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous ribbon interpretation")
    return candidates[0]


def _paint(info, mode="correct"):
    """Input-reading implementation, including explicit counterfactual rules."""
    g = info["view"].copy()
    n, w = info["n"], info["core"].shape[1]
    region = g[info["r0"]:info["r1"], n + 1:-n - 1]
    region[:] = info["structural"]
    offset = 0
    for raw in info["paths"]:
        path = _orient(info, raw, high=mode == "high", right_tie=mode == "right_tie")
        carrier, stencil = _tape(info, path[0]), _tape(info, path[-1])
        if mode == "old_partner":
            stencil = _tape(info, (path[0][0], w - 1 - path[0][1]))
        if mode == "reverse_tapes":
            carrier, stencil = carrier[::-1], stencil[::-1]
        length = len(path)
        for i, p in enumerate(path):
            ci, si = i * n // length, i % n
            if mode == "tile_carrier":
                ci = i % n
            elif mode == "stretch_stencil":
                si = ci
            elif mode == "far_phase":
                si = (length - 1 - i) % n
            elif mode == "global_phase":
                si = (i + offset) % n
            elif mode == "manhattan":
                d = abs(p[0] - path[0][0]) + abs(p[1] - path[0][1])
                ci = min(n - 1, d * n // length)
            elif mode == "center_resize":
                ci = min(n - 1, (2 * i + 1) * n // (2 * length))
            base, ink = int(carrier[ci]), int(stencil[si])
            if mode == "fuse_first":
                ink = int(stencil[ci])
            if mode == "ignore_stencil":
                ink = 0
            region[p] = ink if ink != 0 or mode == "opaque_zero" else base
        offset += length
    return g if info["axis"] == 0 else g.T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the grid: fixed down/left priority, visible roles and tapes."""
    return _paint(_analyze(input_grid)).astype(np.int8)


def _evidence(info, output):
    result = set()
    for mode in REQUIRED - {"gravity", "tie", "visible_carrier", "visible_overwrite", "uneven", "bent"}:
        if not np.array_equal(_paint(info, mode), output):
            result.add(mode)
    if not np.array_equal(_paint(info, "high"), output):
        result.add("gravity")
    if not np.array_equal(_paint(info, "right_tie"), output):
        result.add("tie")
    n = info["n"]
    for raw in info["paths"]:
        path = _orient(info, raw)
        c, s = _tape(info, path[0]), _tape(info, path[-1])
        for i in range(len(path)):
            ink, base = int(s[i % n]), int(c[i * n // len(path)])
            if ink == 0:
                result.add("visible_carrier")
            elif ink != base:
                result.add("visible_overwrite")
        if len(path) % n and len(np.unique(c)) > 1:
            result.add("uneven")
        if len(path) > abs(path[0][0] - path[-1][0]) + abs(path[0][1] - path[-1][1]) + 1:
            result.add("bent")
    return result


def _base(lanes, width, centers, track, gate):
    a = np.zeros((2 * lanes - 1, width), dtype=np.int8)
    a[::2] = track
    for r, c in centers:
        a[r, c - 1:c + 2] = gate
    return a


def _layout(rng, lanes, width, track, gate, reserve=False):
    # At most 48 candidate topologies, then a guaranteed open one-gate layout.
    for _ in range(48):
        rows, centers = [], []
        blank = int(rng.choice((0, lanes - 2))) if reserve and lanes >= 3 else -1
        for j in range(lanes - 1):
            previous = rows[-1] if rows else []
            wanted = 0 if j == blank else int(rng.integers(min(4, (width - 1) // 4) + 1))
            row = []
            for c in rng.permutation(np.arange(2, width - 2)):
                c = int(c)
                if len(row) >= wanted:
                    break
                if all(abs(c - x) >= 2 for x in previous) and all(abs(c - x) >= 4 for x in row):
                    row.append(c)
            rows.append(row)
            centers.extend((2 * j + 1, c) for c in sorted(row))
        if not centers:
            continue
        base = _base(lanes, width, centers, track, gate)
        try:
            _rewire(base, track, gate)
            return base
        except ValueError:
            pass
    j = 0 if not reserve or lanes == 2 else (1 if blank == 0 else 0)
    return _base(lanes, width, [(2 * j + 1, width // 2)], track, gate)


def _carrier(rng, n, palette):
    kind = int(rng.integers(4))
    k = int(rng.integers(1, min(n, len(palette)) + 1))
    inks = rng.choice(palette, k, replace=False)
    if kind == 0:
        tape = np.full(n, inks[0], dtype=np.int8)
    elif kind == 1:
        tape = inks[(np.arange(n) + int(rng.integers(k))) % k]
    elif kind == 2:
        cuts = sorted(rng.choice(np.arange(1, n), k - 1, replace=False))
        tape = inks[np.searchsorted(cuts, np.arange(n), side="right")]
    else:
        tape = rng.choice(inks, n)
    return np.asarray(tape, dtype=np.int8)


def _stencil(rng, n, palette):
    tape = _carrier(rng, n, palette)
    mode = int(rng.integers(6))
    if mode == 0:
        tape[:] = 0
    elif mode == 1:
        pass  # Opaque tapes are legal; geometry still determines their role.
    elif mode == 2:
        tape[(np.arange(n) + int(rng.integers(2))) % 2 == 0] = 0
    else:
        keep = int(rng.integers(1, n))
        tape[rng.choice(n, n - keep, replace=False)] = 0
    return tape


def _sample_pair(rng, shift, split, axis):
    if shift == 0:
        n = int(rng.integers(2, 6))
        lanes = int(rng.integers(3, 6) if split == "train" else rng.integers(6, 15))
    else:
        long_tapes = (split == "test") == (shift == 1)
        n = int(rng.choice((4, 5) if long_tapes else (2, 3)))
        lanes = int(rng.integers(3, 9) if split == "train" else rng.integers(2, 15))
    cap = 30 - 2 * n - 2
    width = int(rng.integers(7, min(cap, 15) + 1) if split == "train" else rng.integers(7, cap + 1))
    track, gate = map(int, rng.choice(np.arange(1, 9), 2, replace=False))
    colors = [c for c in range(1, 9) if c not in (track, gate)]
    palette = rng.choice(colors, int(rng.integers(2, 7)), replace=False)
    base = _layout(rng, lanes, width, track, gate, reserve=rng.random() < .65)
    if rng.random() < .5:
        base = np.fliplr(base)
    before = int(rng.integers(min(3, 30 - base.shape[0]) + 1))
    after = int(rng.integers(min(3 - before, 30 - base.shape[0] - before) + 1))
    a = np.zeros((base.shape[0] + before + after, width + 2 * n + 2), dtype=np.int8)
    a[before:before + base.shape[0], n + 1:-n - 1] = base
    for r in range(before, before + base.shape[0], 2):
        a[r, n] = a[r, -n - 1] = 9
        a[r, :n] = a[r, -n:] = palette[0]
    if axis:
        a = a.T.copy()
    info = _analyze(a)
    for path in info["paths"]:
        path = _orient(info, path)
        _set_tape(info, path[0], _carrier(rng, n, palette))
        _set_tape(info, path[-1], _stencil(rng, n, palette))
    return {"input": a, "output": _paint(info).astype(np.int8)}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalized(grid):
    a = np.asarray(grid)
    roles = {}
    return a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _train_diverse(pairs):
    infos = [_analyze(p["input"]) for p in pairs]
    return (len({i["n"] for i in infos}) >= 2
            and len({_normalized(i["core"]) for i in infos}) >= 2)


def _teaching_fallback(rng, shift, axis):
    """Two constructive witnesses, tested for all three shifts and both axes.

    Each has two returning U paths and a straight path. Distinct carrier
    symbols and a stencil with only its first symbol opaque expose both
    scales, while the two core widths prevent identical phase coincidences.
    """
    ns = ((2, 5), (2, 3), (4, 5))[shift]
    colors = list(map(int, rng.permutation(np.arange(1, 9))))
    track, gate, payload = colors[0], colors[1], colors[2:]
    pairs = []
    for j, n in enumerate(ns):
        width = 11 + 2 * j
        a = np.zeros((5, width + 2 * n + 2), dtype=np.int8)
        a[:, n + 1:-n - 1] = _base(3, width, [(1, 3 + j)], track, gate)
        a[::2, n] = a[::2, -n - 1] = 9
        a[::2, :n] = a[::2, -n:] = payload[0]
        if axis:
            a = a.T.copy()
        info = _analyze(a)
        for raw in info["paths"]:
            path = _orient(info, raw)
            stencil = np.zeros(n, dtype=np.int8)
            stencil[0] = payload[-1]
            _set_tape(info, path[0], np.array(payload[:n], dtype=np.int8))
            _set_tape(info, path[-1], stencil)
        pairs.append({"input": a, "output": _paint(info).astype(np.int8)})
    assert REQUIRED <= set().union(*(_evidence(_analyze(p["input"]), p["output"]) for p in pairs))
    assert _train_diverse(pairs)
    return pairs


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    shift, axis = int(rng.integers(3)), int(rng.integers(2))
    candidates, covers, seen = [], [], set()
    train = None
    # Bounded selection uses visible teaching evidence, never query outputs.
    for _ in range(128):
        pair = _sample_pair(rng, shift, "train", axis)
        signature = _key(pair["input"])
        if signature in seen:
            continue
        info = _analyze(pair["input"])
        cover = _evidence(info, pair["output"])
        for j in rng.permutation(len(candidates)):
            j = int(j)
            if REQUIRED <= cover | covers[j] and _train_diverse([candidates[j], pair]):
                train = [candidates[j], pair]
                break
        if train is not None:
            break
        candidates.append(pair)
        covers.append(cover)
        seen.add(signature)
    if train is None:
        train = _teaching_fallback(rng, shift, axis)
    puzzle = {"train": train, "test": []}
    seen = {_key(p["input"]) for p in train}
    for split, count in (("train", nt), ("test", nq)):
        while len(puzzle[split]) < count:
            for _ in range(32):
                pair = _sample_pair(rng, shift, split, axis)
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness selection failed")
            seen.add(signature)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, coverage = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            expected = _paint(info)
            assert np.array_equal(expected, b), "incorrect ribbon painting"
            # Separate local and path-level invariants, reconstructed from grids.
            canonical = b if info["axis"] == 0 else b.T
            n = info["n"]
            region = canonical[info["r0"]:info["r1"], n + 1:-n - 1]
            structural = info["structural"]
            mask = structural == info["track"]
            assert np.all(region[~mask] == structural[~mask])
            assert np.all(~np.isin(region[mask], (0, 9, info["track"], info["gate"])))
            untouched = np.ones(a.shape, dtype=bool)
            view = untouched if info["axis"] == 0 else untouched.T
            view[info["r0"]:info["r1"], n + 1:-n - 1] = False
            assert np.array_equal(a[untouched], b[untouched]), "changed tapes, markers, or margins"
            for raw in info["paths"]:
                path = _orient(info, raw)
                carrier, stencil = _tape(info, path[0]), _tape(info, path[-1])
                # Interval expansion + native repeats is a second array-level
                # check of the two scale conventions, not an independent solver.
                base = np.repeat(carrier, np.diff((np.arange(n + 1) * len(path) + n - 1) // n))
                tiled = np.resize(stencil, len(path))
                colors = np.where(tiled != 0, tiled, base)
                assert len(base) == len(path)
                assert np.array_equal(np.array([region[p] for p in path]), colors)
            if split == "train":
                coverage |= _evidence(info, b)
    assert REQUIRED <= coverage, "missing training witnesses: " + str(sorted(REQUIRED - coverage))
    assert _train_diverse(puzzle["train"]), "training must vary tape length and topology"


def family_features(puzzle):
    """Bounded descriptive records; reconstructed from actual inputs."""
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze(pair["input"])
            ordered = [_orient(info, p) for p in info["paths"]]
            carriers = [_tape(info, p[0]) for p in ordered]
            stencils = [_tape(info, p[-1]) for p in ordered]
            records.append(dict(split=split, shape=list(info["array"].shape), n=info["n"],
                lanes=(info["core"].shape[0] + 1) // 2, width=info["core"].shape[1],
                axis=info["axis"], gates=len(info["centers"]),
                max_row_gates=max(info["counts"]), overlaps=info["overlaps"],
                path_lengths=[len(p) for p in ordered],
                empty_stencils=sum(not np.any(s) for s in stencils),
                opaque_stencils=sum(np.all(s != 0) for s in stencils),
                partial_periods=sum(len(p) % info["n"] != 0 for p in ordered),
                carrier_inks=sorted({len(np.unique(c)) for c in carriers}),
                stencil_inks=sorted({len(np.unique(s[s != 0])) for s in stencils}),
                roles=[info["track"], info["gate"]],
                topology=hashlib.sha256(repr(_normalized(info["core"])).encode()).hexdigest()))
    return records


def self_test():
    # A fully specified small example checks tape order, endpoint priority,
    # exact path lengths, uneven replication, transparency, and gate rotation.
    a = np.zeros((5, 15), dtype=np.int8)
    a[:, 4:11] = _base(3, 7, [(1, 3)], 8, 5)
    a[::2, 3] = a[::2, 11] = 9
    a[0, :3], a[0, 12:] = [0, 7, 0], [0, 0, 6]
    a[2, :3], a[2, 12:] = [3, 2, 1], [4, 2, 3]
    a[4, :3], a[4, 12:] = [1, 2, 4], [0, 6, 0]
    original = a.copy()
    predicted = solve(a)
    # Both U paths have seven cells. Low-left tape reads 1,2,3.
    info = _analyze(a)
    left_path = next(_orient(info, p) for p in info["paths"] if (2, 0) in p)
    reg = predicted[:, 4:11]
    assert len(left_path) == 7
    assert [int(reg[p]) for p in left_path] == [1, 7, 1, 2, 7, 3, 3]
    assert np.array_equal(predicted[:3, 7], [5, 5, 5])
    assert np.array_equal(a, original), "solver modified its input"
    # Recoloring all nonstructural color identities commutes with solving;
    # zero transparency and the color-9 bracket stay fixed.
    permutation = np.array([0, 6, 3, 8, 2, 7, 1, 4, 5, 9], dtype=np.int8)
    assert np.array_equal(solve(permutation[a]), permutation[predicted])
    for shift in range(3):
        for axis in (0, 1):
            fallback = _teaching_fallback(np.random.default_rng(300 + shift + 3 * axis), shift, axis)
            cover = set().union(*(_evidence(_analyze(p["input"]), p["output"]) for p in fallback))
            assert REQUIRED <= cover and _train_diverse(fallback)
    # Invalid topology and input conventions must fail before painting.
    for centers in ([(1, 3), (3, 3)], [(1, 2), (1, 6)]):
        try:
            _rewire(_base(3 if len({r for r, _ in centers}) == 2 else 2,
                          11, centers, 8, 5), 8, 5)
        except ValueError:
            pass
        else:
            raise AssertionError("bad topology accepted")
    signatures, topologies, shapes, ns, axes, roles = set(), set(), set(), set(), set(), [set(), set()]
    row_counts, gate_counts, lengths, carrier_inks, stencil_inks = set(), set(), set(), set(), set()
    empty = opaque = overlap = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for record in family_features(puzzle):
            topologies.add(record["topology"])
            shapes.add(tuple(record["shape"]))
            ns.add(record["n"])
            axes.add(record["axis"])
            for j in range(2):
                roles[j].add(record["roles"][j])
            row_counts.add(record["max_row_gates"])
            gate_counts.add(record["gates"])
            lengths.update(record["path_lengths"])
            carrier_inks.update(record["carrier_inks"])
            stencil_inks.update(record["stencil_inks"])
            empty += record["empty_stencils"]
            opaque += record["opaque_stencils"]
            overlap += record["overlaps"] > 0
        # Every query is solved again after copying only its input.
        for pair in puzzle["test"]:
            assert np.array_equal(solve(np.array(pair["input"].tolist())), pair["output"])
    assert len(signatures) == 42 and len(topologies) >= 80 and len(shapes) >= 50
    assert ns == {2, 3, 4, 5} and axes == {0, 1}
    assert all(r == set(range(1, 9)) for r in roles)
    assert {1, 2, 3} <= row_counts and len(gate_counts) >= 10 and len(lengths) >= 30
    assert {1, 2, 3} <= carrier_inks and {0, 1, 2, 3} <= stencil_inks
    assert empty > 0 and opaque > 0 and overlap > 0
    first = generate_puzzle(123456, 2, 1)
    second = generate_puzzle(123456, 2, 1)
    assert all(_key(p[k]) == _key(q[k]) for s in first for p, q in zip(first[s], second[s]) for k in p)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.2), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
