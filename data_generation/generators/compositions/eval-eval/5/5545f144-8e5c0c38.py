"""Mirror courier: collect locally reflected cargo and deliver its body-frame union.

Only NumPy and the standard library are used. Generation and checks keep bounded
in-memory records; no files, execution traces, logs, or worker processes are made.
"""
from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _rot(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _place(cells, tip, direction):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rot(p, direction - 1) for p in cells)}


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if (dr == 0) == (dc == 0):
        raise ValueError("route references must be distinct and orthogonal")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _components(cells):
    todo, result = set(cells), []
    while todo:
        start = min(todo)
        todo.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in todo:
                        todo.remove(q)
                        found.add(q)
                        stack.append(q)
        result.append(found)
    return result


def _mover_geometry(cells):
    """Infer a projecting tip and bilateral body, without a motif lookup."""
    matches = []
    for tip in sorted(cells):
        for d in range(4):
            body = {_rot((r - tip[0], c - tip[1]), 1 - d) for r, c in cells}
            if (max(c for r, c in body) == 0
                    and {p for p in body if p[1] == 0} == {(0, 0)}
                    and any(r != 0 for r, c in body)
                    and {(-r, c) for r, c in body} == body):
                matches.append((tuple(sorted(body)), tip, d))
    if len(matches) != 1:
        raise ValueError("courier must have exactly one projecting symmetric tip")
    return matches[0]


def _route(origin, waypoints):
    remaining, ports, directions = set(waypoints), [], []
    position = origin
    while remaining:
        rays = {}
        for q in sorted(remaining):
            if q[0] == position[0] or q[1] == position[1]:
                rays.setdefault(_direction(position, q), []).append(q)
        if len(rays) != 1:
            raise ValueError("each step needs exactly one occupied orthogonal ray")
        d, choices = next(iter(rays.items()))
        q = min(choices, key=lambda p: _cheb(p, position))
        ports.append(q)
        directions.append(d)
        remaining.remove(q)
        position = q
    return ports, directions


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("a grid must be a 1..30 integer matrix of colors 0..9")
    return g


def _analyze(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    if len(colors) != 3 or counts.max() <= g.size // 2:
        raise ValueError("need a majority background, courier ink, and cargo ink")
    bg = int(colors[int(counts.argmax())])
    foreground = sorted(set(map(int, colors)) - {bg})
    candidates = []
    for ink in foreground:
        cells = set(map(tuple, np.argwhere(g == ink).tolist()))
        pieces = _components(cells)
        bodies = [c for c in pieces if len(c) > 1]
        if len(bodies) != 1 or len(bodies[0]) < 3:
            continue
        try:
            mask, origin, initial = _mover_geometry(bodies[0])
            points = {next(iter(c)) for c in pieces if len(c) == 1}
            if len(points) < 2:
                continue
            ports, ds = _route(origin, points)
            if ds[0] != initial:
                continue
            if any(not _inside(_place(mask, p, d), g.shape) for p, d in zip(ports, ds)):
                continue
        except ValueError:
            continue
        candidates.append((ink, mask, origin, initial, ports, ds))
    if len(candidates) != 1:
        raise ValueError("courier and ports must be geometrically identifiable")
    ink, mask, origin, initial, ports, ds = candidates[0]
    cargo = next(c for c in foreground if c != ink)
    clouds = [set() for _ in ports]
    for q in map(tuple, np.argwhere(g == cargo).tolist()):
        distances = [_cheb(q, p) for p in ports]
        nearest = min(distances)
        if distances.count(nearest) != 1:
            raise ValueError("cargo may not be tied between ports")
        clouds[distances.index(nearest)].add(q)
    if not all(clouds):
        raise ValueError("every port must have a visible cargo cloud")
    return dict(shape=g.shape, bg=bg, ink=ink, cargo=cargo, mask=mask,
                origin=origin, initial=initial, ports=ports, directions=ds, clouds=clouds)


def _local_cloud(a, index, direction=None):
    p = a["ports"][index]
    d = a["directions"][index] if direction is None else direction
    return {_rot((r - p[0], c - p[1]), 1 - d) for r, c in a["clouds"][index]}


def _supported(local):
    return {p for p in local if (-p[0], p[1]) in local}


def _delivery(a, variant="rule"):
    """Counterfactuals are for validator evidence, never for choosing the rule."""
    carried, raw = set(), set()
    last = len(a["ports"]) - 1
    for i, (p, d) in enumerate(zip(a["ports"], a["directions"])):
        frame = (a["initial"] if variant == "fixed_initial" else
                 a["directions"][-1] if variant == "fixed_terminal" else d)
        local = _local_cloud(a, i, frame)
        raw |= local
        selected = local if variant == "omit_B" else _supported(local)
        if variant != "last_only" or i == last:
            carried |= selected
        if variant == "rule":
            if set(a["mask"]) & carried:
                raise ValueError("cargo delivery may not overwrite the courier body")
            if not _inside(_place(carried, p, d), a["shape"]):
                raise ValueError("carried cargo must fit at every arrival")
    if variant == "filter_after_merge":
        carried = _supported(raw)
    tip, d = a["ports"][-1], a["directions"][-1]
    cargo_cells = _place(carried, tip, d)
    body = _place(a["mask"], tip, d)
    if not _inside(body | cargo_cells, a["shape"]):
        return None  # A counterfactual can fail to fit; no clipping is performed.
    out = np.full(a["shape"], a["bg"], dtype=np.int8)
    for q in cargo_cells:
        out[q] = a["cargo"]
    for q in body:
        out[q] = a["ink"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, journey, reflection lines, cargo, and final pose from input."""
    return _delivery(_analyze(input_grid))


def _arrow(span, extension, filled, tails):
    body = {(0, -i) for i in range(extension)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if filled else (-depth, depth)
        body.update((r, 1 - extension - depth) for r in rows)
    for tail in range(1, tails + 1):
        body.update((r, 1 - extension - span - tail) for r in (-span, span))
    return tuple(sorted(body))


_MASKS = tuple(dict.fromkeys(_arrow(s, e, f, t)
                           for s in (1, 2) for e in (1, 2)
                           for f in (False, True) for t in (0, 1)))


def _valid_extension(points, footprints, mask, shape, q, d):
    if any(_cheb(q, p) < 6 for p in points):
        return False
    if not (2 <= q[0] < shape[0] - 2 and 2 <= q[1] < shape[1] - 2):
        return False
    body = _place(mask, q, d)
    if not _inside(body, shape):
        return False
    past = footprints if footprints else [_place(mask, points[0], d)]
    if not all(_inside(b, shape) for b in past):
        return False
    if any(_cheb(q, p) <= 2 for b in past for p in b):
        return False
    for p, successor in zip(points[:-1], points[1:]):
        if p[0] == q[0] or p[1] == q[1]:
            if _direction(p, q) != _direction(p, successor) or _cheb(p, q) <= _cheb(p, successor):
                return False
    return True


def _make_route(rng, mask, count, shape):
    # At most 56 short walks; one sequential worker, no recursive search tree.
    for _ in range(56):
        points = [(int(rng.integers(2, shape[0] - 2)), int(rng.integers(2, shape[1] - 2)))]
        footprints, ds = [], []
        for step in range(count):
            by_direction = {}
            for d, (dr, dc) in enumerate(_DIRS):
                choices = []
                for distance in range(6, max(shape)):
                    q = points[-1][0] + dr * distance, points[-1][1] + dc * distance
                    if _valid_extension(points, footprints, mask, shape, q, d):
                        choices.append(q)
                if choices:
                    by_direction[d] = choices
            if not by_direction:
                break
            dirs = sorted(by_direction)
            # Straight runs remain possible; no fixed turn handedness or lattice.
            d = int(rng.choice(dirs))
            q = by_direction[d][int(rng.integers(len(by_direction[d])))]
            if not footprints:
                footprints.append(_place(mask, points[0], d))
            points.append(q)
            ds.append(d)
            footprints.append(_place(mask, q, d))
        if len(ds) == count and len(set(ds)) > 1:
            return shape, points, ds
    # Guaranteed bounded fallback, with quarter-turns and variable padding in
    # the normal sampler. Only used if the random continuous-coordinate walks fail.
    shape = (30, 30)
    points = [(4, 4), (4, 12), (12, 12), (12, 20), (20, 20), (20, 27)][:count + 1]
    turns = int(rng.integers(4))
    for _ in range(turns):
        points = [(c, 29 - r) for r, c in points]
    ds = [_direction(p, q) for p, q in zip(points, points[1:])]
    return shape, points, ds


def _sample_clouds(rng, mask, count):
    # Mirrored pairs in a compact body frame; the input-only solver has no
    # radius constant and can also handle larger legal clouds.
    allowed = {(r, c) for r in range(-2, 3) for c in range(-2, 3)} - set(mask)
    pairs = sorted((r, c) for r, c in allowed if r > 0 and (-r, c) in allowed)
    ghost_index, exclusive_index = map(int, rng.choice(len(pairs), 2, replace=False))
    ghost, exclusive = pairs[ghost_index], pairs[exclusive_index]
    clouds, cores = [], []
    for i in range(count):
        pool = [p for p in pairs if p != ghost and (i == 0 or p != exclusive)]
        k = int(rng.integers(2, min(6, len(pool)) + 1))
        picked = {pool[int(j)] for j in rng.choice(len(pool), k, replace=False)}
        if i == 0:
            picked.add(exclusive)
        core = {q for r, c in picked for q in ((r, c), (-r, c))}
        core |= {p for p in sorted(allowed) if p[0] == 0 and rng.random() < .35}
        # Each port has some multi-cell material so the lone courier is a
        # visible geometric role, even when other supported cargo is isolated.
        if not any(len(c) > 1 for c in _components(core)):
            return None
        noise = set()
        if i < 2:
            noise.add(ghost if i == 0 else (-ghost[0], ghost[1]))
        vacancies = [p for p in pairs if p not in picked and p != ghost]
        amount = 0 if rng.random() < .25 else int(rng.integers(1, min(3, len(vacancies)) + 1)) if vacancies else 0
        if amount:
            for j in rng.choice(len(vacancies), amount, replace=False):
                r, c = vacancies[int(j)]
                noise.add((r if rng.random() < .5 else -r, c))
        clouds.append(core | noise)
        cores.append(core)
    return clouds, cores


def _make_pair(rng, mask, count, palette):
    shape = (int(rng.integers(18, 31)), int(rng.integers(18, 31)))
    shape, points, ds = _make_route(rng, mask, count, shape)
    for _ in range(96):
        sampled = _sample_clouds(rng, mask, count)
        if sampled is None:
            continue
        clouds, cores = sampled
        bg, ink, cargo = palette
        inp = np.full(shape, bg, dtype=np.int8)
        for p in _place(mask, points[0], ds[0]) | set(points[1:]):
            inp[p] = ink
        for tip, d, cloud in zip(points[1:], ds, clouds):
            for p in _place(cloud, tip, d):
                if inp[p] != bg:
                    raise AssertionError("constructor overwrote a marker or courier")
                inp[p] = cargo
        # Construction uses sampled mirror pairs and rigid transforms directly;
        # it does not call solve to manufacture the label.
        collected = set().union(*cores)
        out = np.full(shape, bg, dtype=np.int8)
        for p in _place(collected, points[-1], ds[-1]):
            out[p] = cargo
        for p in _place(mask, points[-1], ds[-1]):
            out[p] = ink
        a = _analyze(inp)
        if any(np.array_equal(out, _delivery(a, variant))
               for variant in ("fixed_initial", "fixed_terminal")):
            continue
        assert np.array_equal(out, _delivery(a))
        return {"input": inp, "output": out}
    return None


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _fingerprint(grid):
    g = np.asarray(grid, dtype=np.uint8)
    return hashlib.sha256(bytes(g.shape) + g.tobytes()).digest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    first_mask, previous_bg = None, None
    for split, size in (("train", num_train), ("test", num_test)):
        for i in range(size):
            count = (2 + i if i < 2 else int(rng.integers(2, 4))) if split == "train" else int(rng.integers(4, 6))
            for _ in range(32):
                masks = [m for m in _MASKS if not (split == "train" and i == 1 and m == first_mask)]
                mask = masks[int(rng.integers(len(masks)))]
                bg = int(rng.choice([c for c in range(10) if c != previous_bg]))
                rest = rng.choice([c for c in range(10) if c != bg], 2, replace=False)
                palette = bg, int(rest[0]), int(rest[1])
                pair = _make_pair(rng, mask, count, palette)
                if pair is not None and _fingerprint(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded courier construction exhausted")
            if split == "train" and i == 0:
                first_mask = mask
            previous_bg = bg
            seen.add(_fingerprint(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(pair):
    a = _analyze(pair["input"])
    locals_ = [_local_cloud(a, i) for i in range(len(a["ports"]))]
    good = [_supported(c) for c in locals_]
    points = [a["origin"]] + a["ports"]
    ds = a["directions"]
    changes = tuple((b - x) % 4 for x, b in zip(ds, ds[1:]) if x != b)
    return dict(shape=a["shape"], ports=len(ds), mask=a["mask"], initial=a["initial"],
                final=ds[-1], palette=(a["bg"], a["ink"], a["cargo"]),
                route=tuple((r - points[0][0], c - points[0][1]) for r, c in points),
                distances=tuple(_cheb(x, y) for x, y in zip(points, points[1:])),
                turns=changes, mixed_turns=1 in changes and 3 in changes,
                straight=any(x == y for x, y in zip(ds, ds[1:])),
                noise=sum(len(c - k) for c, k in zip(locals_, good)),
                collected=len(set().union(*good)),
                cargo_shapes=tuple(tuple(sorted(k)) for k in good),
                axis_cells=sum(sum(r == 0 for r, c in k) for k in good),
                supported_islands=sum(sum(len(p) == 1 for p in _components(k)) for k in good))


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
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            a = _analyze(inp)
            assert np.array_equal(_delivery(a), out), "incorrect transported cargo or terminal courier"
            assert set(map(tuple, np.argwhere(out == a["ink"]).tolist())) == _place(a["mask"], a["ports"][-1], a["directions"][-1])
            assert set(map(int, np.unique(out))) == {a["bg"], a["ink"], a["cargo"]}
            assert not np.array_equal(inp, out)
            assert len(set(a["directions"])) >= 2, "demonstrate rotation of the local mirror"
            points = [a["origin"]] + a["ports"]
            assert all(_cheb(x, y) >= 6 for i, x in enumerate(points) for y in points[i + 1:])
            for i, p in enumerate(a["ports"]):
                local = _local_cloud(a, i)
                assert local and max(max(abs(r), abs(c)) for r, c in local) <= 2
                assert not (local & set(a["mask"]))
                assert len(_supported(local)) >= 4, "each port needs reflected evidence"
                assert any(len(c) > 1 for c in _components(local))
            for variant in ("omit_B", "filter_after_merge", "last_only", "fixed_initial", "fixed_terminal"):
                assert not np.array_equal(out, _delivery(a, variant)), f"uninformative {variant} ablation"
            records[split].append(_measure(pair))
    train, test = records["train"], records["test"]
    assert {m["ports"] for m in train} == {2, 3}
    assert all(m["ports"] in (4, 5) for m in test)
    assert len({m["mask"] for m in train}) >= 2
    assert len({m["palette"][0] for m in train}) >= 2
    return None


def self_test():
    """42 seeds, bounded signatures/metrics, counterfactual and mutation checks."""
    records, hashes = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                digest.update(_fingerprint(pair["input"]))
                records.append(_measure(pair))
        hashes.add(digest.digest())
        if seed < 2:
            again = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for split in p
                       for x, y in zip(p[split], again[split]) for k in x)
    assert len(hashes) == 42
    assert {m["ports"] for m in records} == {2, 3, 4, 5}
    assert {m["initial"] for m in records} == {0, 1, 2, 3}
    assert {m["final"] for m in records} == {0, 1, 2, 3}
    assert len({m["shape"] for m in records}) >= 45
    assert len({m["mask"] for m in records}) >= 12
    assert len({m["route"] for m in records}) >= 170
    assert len({c for m in records for c in m["cargo_shapes"]}) >= 150
    assert {m["palette"][i] for m in records for i in range(3)} == set(range(10))
    assert all({m["palette"][i] for m in records} == set(range(10)) for i in range(3))
    assert any(m["mixed_turns"] for m in records) and any(m["straight"] for m in records)
    assert any(m["initial"] == m["final"] for m in records)
    assert any(m["axis_cells"] > 0 for m in records)
    assert any(m["supported_islands"] > 0 for m in records)
    assert any(len(set(m["distances"])) > 1 for m in records)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Palette equivariance and four geometric rotations of a solved input.
    p = generate_puzzle(73, 2, 1)
    pair = p["test"][0]
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], turns)), np.rot90(pair["output"], turns))
    permutation = np.random.default_rng(902).permutation(10)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # A legal input beyond the sampler: a wider arrow and radius-three cargo.
    # This checks that solve does not quietly depend on _MASKS or radius two.
    wide = _arrow(3, 2, False, 1)
    assert wide not in _MASKS
    origin, ports, ds = (6, 6), [(6, 17), (20, 17)], [1, 2]
    core_one = {(-3, 1), (3, 1), (-2, 2), (2, 2), (0, 3)}
    core_two = {(-1, 2), (1, 2), (-3, -1), (3, -1)}
    probe = np.zeros((30, 30), dtype=np.int8)
    for q in _place(wide, origin, 1) | set(ports):
        probe[q] = 2
    for tip, d, cloud in zip(ports, ds, (core_one | {(1, 3)}, core_two | {(-1, 3)})):
        for q in _place(cloud, tip, d):
            assert probe[q] == 0
            probe[q] = 8
    expected = np.zeros_like(probe)
    for q in _place(core_one | core_two, ports[-1], ds[-1]):
        expected[q] = 8
    for q in _place(wide, ports[-1], ds[-1]):
        expected[q] = 2
    assert np.array_equal(solve(probe), expected)
    tied = probe.copy()
    tied[13, 16] = 8  # Same square distance from the two ports.
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("tied port assignment accepted")
    # Mutation: losing even one delivered reflected cell must be rejected.
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    a = _analyze(corrupted["test"][0]["input"])
    out = corrupted["test"][0]["output"]
    q = tuple(np.argwhere(out == a["cargo"])[0])
    out[q] = a["bg"]
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt delivery accepted")
    try:
        _route((3, 3), {(3, 9), (9, 3)})
    except ValueError:
        pass
    else:
        raise AssertionError("route priority invented at junction")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all counts, counterfactuals, rotations, and mutations")
