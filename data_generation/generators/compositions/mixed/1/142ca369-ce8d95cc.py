"""Trace the visible optics, then contract equal consecutive full vectors.

The sampler engineers equal colored vectors with interlaced ray pairs.  The
solver knows nothing about that engineering: it parses and traces the input.
No files, parent modules, caches, or generation metadata are used by solve.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] * 2 <= a.size:
        raise ValueError("background must occupy a strict input majority")
    remaining = set(map(tuple, np.argwhere(a != bg)))
    sources, mirrors = [], []
    while remaining:
        start = min(remaining)
        color = int(a[start])
        remaining.remove(start)
        todo, cells = [start], set()
        while todo:
            r, c = todo.pop()
            cells.add((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and a[p] == color:
                    remaining.remove(p)
                    todo.append(p)
        rs, cs = zip(*cells)
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if len(cells) == 3 and r1-r0 == c1-c0 == 1:
            missing = next(iter({(r, c) for r in (r0, r1)
                                for c in (c0, c1)} - cells))
            elbow = (r0+r1-missing[0], c0+c1-missing[1])
            direction = (elbow[0]-missing[0], elbow[1]-missing[1])
            sources.append({"cells": cells, "color": color,
                            "elbow": elbow, "direction": direction})
        elif (len(cells) % 2 == 1 and (r0 == r1 or c0 == c1)
              and len(cells) == max(r1-r0, c1-c0)+1):
            axis = None if len(cells) == 1 else (0 if r0 == r1 else 1)
            mirrors.append({"cells": cells, "color": color, "axis": axis,
                            "midpoint": ((r0+r1)//2, (c0+c1)//2)})
        else:
            raise ValueError("ambiguous source or reflector component")
    if not sources:
        raise ValueError("no L source")
    return bg, sources, mirrors


def _trace(a, *, ignore_mirrors=False):
    """Cellwise A optics; reject conflicts instead of assigning a priority."""
    bg, sources, mirrors = _parse(a)
    out = a.copy()
    painted, routes, used = {}, [], set()
    h, w = a.shape
    for source in sources:
        dr, dc = source["direction"]
        r, c = source["elbow"]
        color = source["color"]
        visited, contacts, path = set(), [], []
        while True:
            r, c = r+dr, c+dc
            if not (0 <= r < h and 0 <= c < w):
                break
            state = r, c, dr, dc
            if state in visited:
                raise ValueError("ray loop")
            visited.add(state)
            if a[r, c] != bg:
                raise ValueError("ray hit an object instead of a midpoint face")
            hits = []
            if not ignore_mirrors:
                for j, mirror in enumerate(mirrors):
                    mr, mc = mirror["midpoint"]
                    for axis in (0, 1):
                        if mirror["axis"] not in (None, axis):
                            continue
                        if ((axis == 0 and c == mc and mr-r == dr)
                                or (axis == 1 and r == mr and mc-c == dc)):
                            hits.append((j, axis))
            if len(hits) > 1:
                raise ValueError("ambiguous simultaneous midpoint contacts")
            if hits:
                j, axis = hits[0]
                before = color
                color = mirrors[j]["color"]
                used.add(j)
                contacts.append((j, axis, (r, c), before, color))
                if axis == 0:
                    dr = -dr
                else:
                    dc = -dc
            if (r, c) in painted and painted[r, c] != color:
                raise ValueError("conflicting ray colors; no priority is defined")
            painted[r, c] = color
            out[r, c] = color
            path.append(((r, c), color))
        routes.append({"path": path, "contacts": contacts})
    if not ignore_mirrors and used != set(range(len(mirrors))):
        raise ValueError("unused reflector")
    return out, {"background": bg, "sources": sources,
                 "mirrors": mirrors, "routes": routes}


def _run_indices(a, axis):
    vectors = a if axis == 0 else a.T
    keep = np.ones(len(vectors), dtype=bool)
    keep[1:] = np.any(vectors[1:] != vectors[:-1], axis=1)
    return np.flatnonzero(keep)


def _compress(a):
    # Both equivalence relations are evaluated on the completed full grid.
    return a[np.ix_(_run_indices(a, 0), _run_indices(a, 1))].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    completed, _ = _trace(_grid(input_grid))
    return _compress(completed)


def _l_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r-dr, c), (r, c-dc)}


def _bar(midpoint, axis, length):
    r, c = midpoint
    return {(r+(d if axis == 1 else 0), c+(d if axis == 0 else 0))
            for d in range(-(length//2), length//2+1)}


def _arm(rng, shape, focus, side, terminal, background, mirrored,
         required_axis=None, changed=None, restrict=False):
    """Build a source backwards from its required outgoing diagonal.

    Only the builder uses the invisible target; solve rediscovers every ray.
    All objects and pre-reflection flights stay above the duplicate waist.
    """
    h, w = shape
    R, C = focus
    target = (R, C+side)
    outgoing = (1, 1 if side == 0 else -1)

    def allowed(cells):
        return all(0 <= r <= R-3 and 0 <= c < w
                   and (not restrict or (c < C if side == 0 else c > C+1))
                   for r, c in cells)

    if not mirrored:
        options = []
        for d in range(3, 29):
            elbow = (target[0]-d, target[1]-d*outgoing[1])
            cells = _l_cells(elbow, outgoing)
            if allowed(cells):
                options.append([(cells, terminal)])
        return None if not options else options[int(rng.integers(len(options)))]

    # Geometry choices are counted before sampling; there is no unbounded search.
    options = []
    axes = (0, 1) if required_axis is None else (required_axis,)
    for d in range(3, 26):
        contact = (target[0]-d, target[1]-d*outgoing[1])
        for axis in axes:
            incoming = list(outgoing)
            incoming[axis] *= -1
            midpoint = list(contact)
            midpoint[axis] += incoming[axis]
            for length in (1, 3, 5, 7):
                cells = _bar(tuple(midpoint), axis, length)
                if not allowed(cells) or not allowed({contact}):
                    continue
                for flight in range(2, 18):
                    elbow = (contact[0]-flight*incoming[0],
                             contact[1]-flight*incoming[1])
                    source = _l_cells(elbow, incoming)
                    if allowed(source) and source.isdisjoint(cells):
                        options.append((cells, source))
    if not options:
        return None
    mirror, source = options[int(rng.integers(len(options)))]
    if changed is None:
        changed = bool(rng.integers(2))
    source_color = (int(rng.choice([c for c in range(10)
                                  if c not in (terminal, background)]))
                    if changed else terminal)
    return [(mirror, terminal), (source, source_color)]


def _nonbackground_run(a, bg, axis):
    vectors = a if axis == 0 else a.T
    return any(np.array_equal(vectors[i-1], vectors[i])
               and np.any(vectors[i] != bg) for i in range(1, len(vectors)))


def _separated_repeat(a, bg, axis):
    vectors = a if axis == 0 else a.T
    seen = {}
    for i, row in enumerate(vectors):
        key = row.tobytes()
        if np.any(row != bg) and key in seen and i-seen[key] > 1:
            if any(not np.array_equal(vectors[j], row)
                   for j in range(seen[key]+1, i)):
                return True
        seen[key] = i
    return False


def _make_pair(rng, bundles, mirror_count, swap, *, axis=None, changed=None):
    for attempt in range(256):
        # Variable half-integer foci share a row, with even offsets. Opposite
        # colored diagonals therefore cross between cells, never ambiguously.
        R = int(rng.integers(7 if mirror_count else 4, 17))
        steps = [int(rng.choice((4, 6))) for _ in range(bundles-1)]
        span = sum(steps)
        min_width = max(12, span+14)
        if min_width > 30:
            continue
        w = int(rng.integers(min_width, 31))
        h = int(rng.integers(R+4, 31))
        left = int(rng.integers(5, w-span-6))
        centers = [left]
        for step in steps:
            centers.append(centers[-1]+step)
        background = int(rng.integers(10))
        palette = [c for c in range(10) if c != background]
        terminal = rng.choice(palette, size=bundles, replace=False)
        # Some multibundle scenes reuse a terminal color. Equality, not color
        # identity or source order, still determines contraction.
        if bundles > 1 and rng.random() < .25:
            terminal[-1] = terminal[0]
        selected = set(int(x) for x in rng.choice(2*bundles, size=mirror_count,
                                                replace=False))
        a = np.full((h, w), background, dtype=np.int8)
        good = True
        for j, C in enumerate(centers):
            for side in (0, 1):
                serial = 2*j+side
                objects = _arm(rng, a.shape, (R, C), side, int(terminal[j]),
                               background, serial in selected,
                               axis if serial in selected else None, changed,
                               restrict=bundles == 1)
                if objects is None:
                    good = False
                    break
                for cells, color in objects:
                    if any(a[p] != background for p in cells):
                        good = False
                        break
                    for p in cells:
                        a[p] = color
                if not good:
                    break
            if not good:
                break
        if not good:
            continue
        try:
            full, info = _trace(a)
        except ValueError:
            continue
        if (len(info["sources"]) != 2*bundles
                or len(info["mirrors"]) != mirror_count
                or sum(len(route["contacts"]) for route in info["routes"]) != mirror_count
                or any(len(route["contacts"]) > 1 for route in info["routes"])
                or not np.array_equal(full[R], full[R+1])
                or not np.array_equal(full[R-1], full[R+2])
                or not np.any(full[R] != background)):
            continue
        if bundles == 1 and not _nonbackground_run(full, background, 1):
            continue
        if mirror_count and axis is not None and not any(contact[1] == axis
                for route in info["routes"] for contact in route["contacts"]):
            continue
        if mirror_count and changed is not None and not any((contact[3] != contact[4]) == changed
                for route in info["routes"] for contact in route["contacts"]):
            continue
        if swap:
            a = a.T.copy()
        if rng.integers(2):
            a = a[::-1].copy()
        if rng.integers(2):
            a = a[:, ::-1].copy()
        return {"input": a, "output": solve(a)}
    # A deterministic optical construction is the bounded fallback. It keeps
    # the requested count, reflection axis, color-change evidence, and waist.
    return _fallback(rng, bundles, mirror_count, swap, axis, changed)


def _fallback(rng, bundles, mirror_count, swap, axis, changed):
    """Bounded geometric enumeration, preserving every requested mechanism."""
    from itertools import combinations

    R, w, h = 15, 30, 24
    centers = [8+4*j for j in range(bundles)]
    palette = [int(c) for c in rng.permutation(10)]
    bg = palette[0]
    change = bool(rng.integers(2)) if changed is None else changed
    # A source's extra two L cells must miss all the other beams. Varying these
    # three distances finds that clearance without changing any optical rule.
    for direct in (4, 2, 6, 3, 5):
        for distance in (5, 4, 6, 3, 7):
            for flight in (2, 3, 4):
                for selected_tuple in combinations(range(2*bundles), mirror_count):
                    selected = set(selected_tuple)
                    a = np.full((h, w), bg, dtype=np.int8)
                    good, serial_changed = True, 0
                    for j, C in enumerate(centers):
                        for side in (0, 1):
                            dc, color = (1 if side == 0 else -1), palette[1+j]
                            tc = C+side
                            if 2*j+side in selected:
                                point = (R-distance, tc-distance*dc)
                                mid = (point[0], point[1]-dc)
                                elbow = (point[0]-flight, point[1]+flight*dc)
                                direction = (1, -dc)
                                shade = palette[5+serial_changed] if change else color
                                serial_changed += 1
                                objects = [({mid}, color), (_l_cells(elbow, direction), shade)]
                            else:
                                elbow = (R-direct, tc-direct*dc)
                                objects = [(_l_cells(elbow, (1, dc)), color)]
                            for cells, shade in objects:
                                if any(not (0 <= r < h and 0 <= c < w)
                                       or a[r, c] != bg for r, c in cells):
                                    good = False
                                    break
                                for cell in cells:
                                    a[cell] = shade
                            if not good:
                                break
                        if not good:
                            break
                    if not good:
                        continue
                    try:
                        full, info = _trace(a)
                    except ValueError:
                        continue
                    if (len(info["sources"]) != 2*bundles
                            or len(info["mirrors"]) != mirror_count
                            or any(len(route["contacts"]) > 1 for route in info["routes"])
                            or not _nonbackground_run(full, bg, 0)
                            or not _separated_repeat(full, bg, 0)
                            or (bundles == 1 and not _nonbackground_run(full, bg, 1))):
                        continue
                    if axis == 0:
                        a = a.T.copy()
                    if swap:
                        a = a.T.copy()
                    if rng.integers(2):
                        a = a[::-1].copy()
                    if rng.integers(2):
                        a = a[:, ::-1].copy()
                    return {"input": a, "output": solve(a)}
    raise RuntimeError("bounded construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    swap = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    # Even the minimum two demonstrations show both reflection axes, changed
    # and unchanged colors, both contractions, and separated repeated vectors.
    specs = [("train", 1, 1, swap, 1, True),
             ("train", 2, 1, swap, 0, False)]
    for _ in range(ntrain-2):
        k = int(rng.integers(1, 3))
        specs.append(("train", k, int(rng.integers(0, min(k+1, 3))),
                      bool(rng.integers(2)), None, None))
    for i in range(ntest):
        k = 3+(i % 2)
        specs.append(("test", k, int(rng.integers(0, 3)),
                      bool(rng.integers(2)), None, None))
    for split, k, m, transpose, axis, changed in specs:
        for _ in range(8):
            pair = _make_pair(rng, k, m, transpose, axis=axis, changed=changed)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            # There are at most nine training inputs of this source count.
            # Ten palette rotations have distinct majority backgrounds, so one
            # must be new even if repeated random construction was unlucky.
            for offset in range(1, 11):
                candidate = {role: ((array.astype(int)+offset) % 10).astype(np.int8)
                             for role, array in pair.items()}
                key = _key(candidate["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(candidate)
                    break
            else:
                raise RuntimeError("uniqueness invariant violated")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    full, info = _trace(a)
    bg = info["background"]
    ri, ci = _run_indices(full, 0), _run_indices(full, 1)
    assert np.array_equal(out, full[np.ix_(ri, ci)])
    # Re-expand the output using both original axis partitions; exact full
    # reconstruction checks retained crossings, objects, and every tail cell.
    rr = np.diff(np.r_[ri, full.shape[0]])
    cr = np.diff(np.r_[ci, full.shape[1]])
    assert np.array_equal(np.repeat(np.repeat(out, rr, axis=0), cr, axis=1), full)
    assert np.array_equal(_compress(out), out)
    assert np.array_equal(full[a != bg], a[a != bg])
    assert set(np.unique(out)) <= set(np.unique(a))
    assert np.any(full != a), "no actual ray drawing"
    assert out.shape != a.shape, "B compression is a no-op"
    assert not np.array_equal(_compress(a), out), "A is unnecessary"
    active = {_axis for _axis in (0, 1) if _nonbackground_run(full, bg, _axis)}
    separated = {_axis for _axis in (0, 1) if _separated_repeat(full, bg, _axis)}
    assert active and separated, "colored waist or separated recurrence missing"
    assert any(_separated_repeat(out, bg, ax) for ax in (0, 1))
    contacts = [c for route in info["routes"] for c in route["contacts"]]
    return {"shape": a.shape, "output_shape": out.shape, "background": bg,
            "sources": len(info["sources"]), "mirrors": len(info["mirrors"]),
            "active": active, "separated": separated,
            "axes": {c[1] for c in contacts},
            "changed": any(c[3] != c[4] for c in contacts),
            "unchanged": any(c[3] == c[4] for c in contacts),
            "lengths": {len(m["cells"]) for m in info["mirrors"]},
            "directions": {s["direction"] for s in info["sources"]},
            "run_lengths": set(int(n) for n in np.r_[rr, cr])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            features[split].append(_analyze_pair(pair))
    training = features["train"]
    assert set().union(*(f["axes"] for f in training)) == {0, 1}
    assert set().union(*(f["active"] for f in training)) == {0, 1}
    assert any(f["changed"] for f in training)
    assert any(f["unchanged"] for f in training)
    assert {f["sources"] for f in training} >= {2, 4}
    assert all(f["sources"] > max(t["sources"] for t in training)
               for f in features["test"]), "missing count generalization"


def _regressions():
    a = np.zeros((10, 10), dtype=np.int8)
    a[0, 1], a[1, 0], a[1, 1] = 1, 1, 1
    a[0, 8], a[1, 8], a[1, 9] = 1, 1, 1
    full, _ = _trace(a)
    expected = np.array([[int(c) for c in row] for row in (
        "0100000010", "1100000011", "0010000100", "0001001000",
        "0000110000", "0000110000", "0001001000", "0010000100",
        "0100000010", "1000000001")], dtype=np.int8)
    assert np.array_equal(full, expected)
    assert np.array_equal(solve(a), expected[np.ix_([0, 1, 2, 3, 4, 6, 7, 8, 9],
                                                    [0, 1, 2, 3, 4, 6, 7, 8, 9])])
    # Adjacent versus separated equality, including two runs of the same value.
    probe = np.array([[1, 1, 2, 2, 1, 1], [1, 1, 2, 2, 1, 1],
                      [3, 3, 4, 4, 3, 3], [1, 1, 2, 2, 1, 1]], dtype=np.int8)
    assert np.array_equal(_compress(probe), [[1, 2, 1], [3, 4, 3], [1, 2, 1]])
    # A singleton at (4,6) is reached from its left at (4,5).
    b = np.zeros((9, 10), dtype=np.int8)
    for p in _l_cells((1, 2), (1, 1)):
        b[p] = 2
    b[4, 6] = 7
    filled, info = _trace(b)
    assert filled[3, 4] == 2 and filled[4, 5] == 7
    assert all(filled[p] == 7 for p in ((5, 4), (6, 3), (7, 2), (8, 1)))
    assert info["routes"][0]["contacts"][0][1] == 1
    # Changing a ray's direction is visible even after contraction.
    assert not np.array_equal(_compress(b), _compress(filled))
    # Every rectangular symmetry commutes with this geometric rule.
    for array in (a, b, probe):
        if array is probe:
            continue
        for rotation in range(4):
            for flip in (False, True):
                transform = lambda x: np.fliplr(np.rot90(x, rotation)).copy() if flip else np.rot90(x, rotation).copy()
                assert np.array_equal(solve(transform(array)), transform(solve(array)))


def self_test():
    """42 streaming seeds; bounded memory, no written samples or logs."""
    _regressions()
    for bundles in range(1, 5):
        for mirrors in range(min(2, bundles)+1):
            for axis in (0, 1):
                p = _fallback(np.random.default_rng(100+bundles), bundles,
                              mirrors, bool(axis), axis, bool(axis))
                f = _analyze_pair(p)
                assert f["sources"] == 2*bundles and f["mirrors"] == mirrors
                if mirrors:
                    assert f["axes"] == {0} and f["changed"] == bool(axis)
    shapes, structures, backgrounds, sources, mirrors = set(), set(), set(), set(), set()
    lengths, directions, run_lengths, signatures = set(), set(), set(), set()
    counts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _analyze_pair(pair)
                a = pair["input"]
                shapes.add(f["shape"])
                backgrounds.add(f["background"])
                sources.add(f["sources"])
                mirrors.add(f["mirrors"])
                lengths.update(f["lengths"])
                directions.update(f["directions"])
                run_lengths.update(f["run_lengths"])
                roles, normalized = {}, []
                for color in a.flat:
                    normalized.append(roles.setdefault(int(color), len(roles)))
                structures.add((a.shape, bytes(normalized)))
                digest.update(bytes(a.shape)+a.tobytes()+pair["output"].tobytes())
                counts["pairs"] += 1
                counts["both_active"] += f["active"] == {0, 1}
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 50 and len(structures) >= 150
    assert backgrounds == set(range(10)) and sources >= {2, 4, 6}
    assert mirrors >= {0, 1, 2} and lengths >= {1, 3, 5, 7}
    assert directions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert max(run_lengths) >= 5 and counts["both_active"] >= 42
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            p = generate_puzzle(1000+ntrain*10+ntest, ntrain, ntest)
            assert len(p["train"]) == ntrain and len(p["test"]) == ntest
    for ntrain, ntest in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2),
                         (3, 4), (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for split in ("train", "test")
               for a, b in zip(p[split], q[split]) for k in ("input", "output"))
    # Negative validation probe: a damaged retained crossing must be rejected.
    q["test"][0]["output"][0, 0] = (int(q["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
