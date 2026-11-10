"""A hub's whole motif is transported by distance along a repaired, switched tape.

Fixed geometric roles: 0 background, 5 hub/scaffold, 8 repair, 9 switch.
The solver reads only its input. Construction plans are used for sampling and
ground-truth rendering, never retained or supplied to solve.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


BG, GUIDE, REPAIR, SWITCH = 0, 5, 8, 9
PAYLOAD = (1, 2, 3, 4, 6, 7)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 84086875, 993740410, 801661046, 549720117, 429071328,
              998928600, 473016565, 333756735, 19321270, 541522638,
              489344368, 957595846, 589253821, 838117849, 203357698,
              291629541, 233699, 893842182, 430409689, 162138968,
              960803831, 951752245, 963714316, 457997345, 910470129,
              672315232, 326748692, 401953645, 19738111, 464408253,
              670402334, 72344738, 996392751, 849138947, 442009608,
              775076359, 186536657, 541628011, 127008132, 432698787,
              668250168)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + i*d[0], a[1] + i*d[1]) for i in range(n + 1)]


def _polyline(vertices):
    result = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _ray(p, d, shape):
    result = []
    while _inside(p := _add(p, d), shape):
        result.append(p)
    return result


def _components(cells):
    remaining, groups = set(cells), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        group, stack = {first}, [first]
        while stack:
            for q in _neighbors(stack.pop(), remaining):
                remaining.remove(q)
                group.add(q)
                stack.append(q)
        groups.append(group)
    return groups


def _halo(cells):
    return {(r+dr, c+dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _mask(h, w, kind):
    result = np.ones((h, w), dtype=bool)
    if kind == "clipped":
        result[0, 0] = result[0, -1] = result[-1, 0] = result[-1, -1] = False
    elif kind == "cross":
        result[:] = False
        result[h//2, :] = True
        result[:, w//2] = True
    return result


def _hub(t, l, h, w, kind):
    return {(t+int(r), l+int(c)) for r, c in zip(*np.nonzero(_mask(h, w, kind)))}


def _hub_kind(cells):
    rs, cs = zip(*cells)
    t, l = min(rs), min(cs)
    h, w = max(rs)-t+1, max(cs)-l+1
    if h not in (3, 5) or w not in (3, 5):
        return None
    for kind in ("rectangle", "clipped", "cross"):
        if cells == _hub(t, l, h, w, kind):
            return (t, l, h, w), kind
    return None


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "bad dimensions"
    assert np.issubdtype(g.dtype, np.integer), "noninteger cells"
    assert np.all((0 <= g) & (g <= 9)), "bad color"
    return g.astype(np.int8, copy=True)


def _read_input(value):
    """Recover hubs, complete seed words, damaged routes and switches from pixels."""
    g = _grid(value)
    hub_records = []
    for component in _components(map(tuple, np.argwhere(g == GUIDE))):
        match = _hub_kind(component)
        if match:
            box, kind = match
            hub_records.append({"hub": component, "box": box, "kind": kind})
    assert hub_records, "no recognizable hub"
    hubs = set().union(*(r["hub"] for r in hub_records))
    occupied = set(map(tuple, np.argwhere((g != BG) & (g != SWITCH)))) - hubs
    # The sole allowed unmarked damage is an isolated missing straight cell.
    # Infer all bridges simultaneously, using surviving route cells only.
    holes = set()
    for p in map(tuple, np.argwhere(g == BG)):
        axes = []
        for d in ((1, 0), (0, 1)):
            if _add(p, d) in occupied and _add(p, (-d[0], -d[1])) in occupied:
                axes.append(d)
        assert len(axes) <= 1, "ambiguous hole"
        if axes:
            holes.add(p)
    cells = occupied | holes
    assert cells
    roots = []
    for rec in hub_records:
        t, l, h, w = rec["box"]
        faces = ((t, l+w//2), (t+h//2, l+w-1),
                 (t+h-1, l+w//2), (t+h//2, l))
        choices = []
        for face, d in zip(faces, DIRS):
            q = _add(face, d)
            if _inside(q, g.shape) and int(g[q]) in PAYLOAD:
                word, seed_cells = [], []
                while _inside(q, g.shape) and int(g[q]) in PAYLOAD:
                    word.append(int(g[q]))
                    seed_cells.append(q)
                    q = _add(q, d)
                if word and len(word) <= 5 and q in cells and int(g[q]) == GUIDE:
                    choices.append((word, seed_cells, d))
        assert len(choices) == 1, "hub needs one centered, intact, delimited word"
        word, seed_cells, d = choices[0]
        roots.append({**rec, "word": tuple(word), "seed": seed_cells, "root_direction": d})
    used = set()
    for rec in roots:
        start = rec["seed"][0]
        assert len(_neighbors(start, cells)) == 1, "source is not a route endpoint"
        walk, previous, current = [], None, start
        for _ in range(g.size):
            assert current not in walk, "route loop"
            walk.append(current)
            following = [q for q in _neighbors(current, cells) if q != previous]
            if not following:
                break
            assert len(following) == 1, "branching or touching routes"
            previous, current = current, following[0]
        else:
            raise AssertionError("route too long")
        assert len(walk) >= len(rec["word"]) + 3
        assert walk[:len(rec["seed"])] == rec["seed"]
        assert _boundary(walk[-1], g.shape), "old tape must end on the boundary"
        assert all(not _boundary(p, g.shape) for p in walk[:-1]), "route touches edge before its endpoint"
        assert not used.intersection(walk), "two hubs share a route"
        used.update(walk)
        rec["walk"], rec["index"] = walk, {p: i for i, p in enumerate(walk)}
        assert all(int(g[p]) in (GUIDE, REPAIR, BG) for p in walk[len(rec["word"]):])
        rec["switch"] = None
    assert used == cells, "unattached material"
    for q in map(tuple, np.argwhere(g == SWITCH)):
        choices = []
        for rec in roots:
            walk, indexes = rec["walk"], rec["index"]
            for d in DIRS:
                pivot = (q[0]-d[0], q[1]-d[1])
                k = indexes.get(pivot, -1)
                if not (len(rec["word"]) + 1 <= k < len(walk) - 1):
                    continue
                if walk[k+1] != (pivot[0]-d[0], pivot[1]-d[1]):
                    continue
                incoming = (pivot[0]-walk[k-1][0], pivot[1]-walk[k-1][1])
                if incoming[0]*d[0] + incoming[1]*d[1] != 0:
                    continue
                choices.append((rec, {"pivot": pivot, "index": k, "direction": d,
                                      "marker": q, "ray": _ray(pivot, d, g.shape)}))
        assert len(choices) == 1, "switch has no unique forward corner"
        rec, edit = choices[0]
        assert rec["switch"] is None, "at most one switch per route"
        rec["switch"] = edit
    final_cells = set(hubs)
    for rec in roots:
        edit = rec["switch"]
        kept = rec["walk"] if edit is None else rec["walk"][:edit["index"]+1]
        new = [] if edit is None else edit["ray"]
        assert not set(kept).intersection(new), "ray hits its retained prefix"
        assert not final_cells.intersection(kept), "retained tapes overlap"
        final_cells.update(kept)
        assert not final_cells.intersection(new), "replacement ray hits retained material"
        final_cells.update(new)
    # Check all retained material, including sources visited after a ray.
    kept_all = set().union(*(set(r["walk"] if r["switch"] is None else
                                r["walk"][:r["switch"]["index"]+1]) for r in roots))
    for rec in roots:
        if rec["switch"]:
            assert not set(rec["switch"]["ray"]) & (hubs | kept_all)
    return g, roots, holes


def _paint_plan(shape, records, *, ignore_motif=False, ignore_switch=False):
    """Simultaneous rendering. Also supplies two explicit concept ablations."""
    result = np.full(shape, BG, dtype=np.int8)
    occupied = set()
    for rec in records:
        for p in rec["hub"]:
            assert p not in occupied
            occupied.add(p)
            result[p] = GUIDE
    for rec in records:
        word, walk = rec["word"], rec["walk"]
        edit = None if ignore_switch else rec["switch"]
        kept = walk if edit is None else walk[:edit["index"]+1]
        new = [] if edit is None else edit["ray"]
        # The new route's index follows the last retained index without restart.
        for i, p in enumerate(kept + new):
            assert p not in occupied, "final routes cross"
            occupied.add(p)
            result[p] = word[0] if ignore_motif else word[i % len(word)]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve with input pixels alone; do not mutate the supplied grid."""
    g, records, _ = _read_input(input_grid)
    return _paint_plan(g.shape, records)


def _turn_indexes(walk):
    return [i for i in range(1, len(walk)-1)
            if (walk[i][0]-walk[i-1][0], walk[i][1]-walk[i-1][1]) !=
               (walk[i+1][0]-walk[i][0], walk[i+1][1]-walk[i][1])]


def _valid_walk(walk):
    cells = set(walk)
    return (len(cells) == len(walk) and len(walk) >= 2 and
            all(len(_neighbors(p, cells)) == (1 if i in (0, len(walk)-1) else 2)
                for i, p in enumerate(walk)))


def _word(rng, length, style="random"):
    palette = list(map(int, rng.choice(PAYLOAD, int(rng.integers(2, 7)), replace=False)))
    if style == "aba":
        return (palette[0], palette[1], palette[0])
    if style == "distinct":
        return tuple(map(int, rng.choice(PAYLOAD, length, replace=False)))
    word = list(map(int, rng.choice(palette, length)))
    # Requested lengths are actual primitive periods, so the longer-word
    # curriculum cannot collapse to an already familiar constant/AB cycle.
    if any(length % p == 0 and all(word[i] == word[i % p] for i in range(length))
           for p in range(1, length)):
        word[-1] = next(c for c in palette if c != word[-1])
    return tuple(word)


def _select_switch(rec, shape, rng, allowed, require=False):
    walk = rec["walk"]
    candidates = []
    for k in _turn_indexes(walk):
        if k < len(rec["word"]) + 1:
            continue
        p, nxt = walk[k:k+2]
        d = (p[0]-nxt[0], p[1]-nxt[1])
        q = _add(p, d)
        if d not in allowed or not _inside(q, shape) or q in walk or q in _halo(rec["hub"]):
            continue
        ray = _ray(p, d, shape)
        if not set(ray) & (set(walk[:k+1]) | rec["hub"]):
            candidates.append({"pivot": p, "index": k, "direction": d, "marker": q, "ray": ray})
    if not candidates:
        return False if require else True
    rec["switch"] = candidates[int(rng.integers(len(candidates)))]
    return True


def _random_record(rng, shape, period, bends, occupied, switched, allowed):
    """Independent rectilinear segments, with sampled hubs throughout the canvas."""
    for _ in range(85):
        h, w = int(rng.choice((3, 5))), int(rng.choice((3, 5)))
        t, l = int(rng.integers(shape[0]-h+1)), int(rng.integers(shape[1]-w+1))
        kind = str(rng.choice(("rectangle", "clipped", "cross")))
        hub = _hub(t, l, h, w, kind)
        if _halo(hub) & occupied:
            continue
        di = int(rng.integers(4))
        d = DIRS[di]
        face = ((t, l+w//2), (t+h//2, l+w-1), (t+h-1, l+w//2), (t+h//2, l))[di]
        stem_len = period + int(rng.integers(2, 6))
        stem = _ray(face, d, shape)[:stem_len]
        if len(stem) != stem_len or _boundary(stem[-1], shape):
            continue
        if _halo(set(stem)) & occupied:
            continue
        walk, previous = stem, d
        for _ in range(bends):
            options = [(-previous[1], previous[0]), (previous[1], -previous[0])]
            rng.shuffle(options)
            proposals = []
            for nd in options:
                for length in rng.permutation(np.arange(3, 11)):
                    q = (walk[-1][0]+int(length)*nd[0], walk[-1][1]+int(length)*nd[1])
                    if not _inside(q, shape) or _boundary(q, shape):
                        continue
                    added = _line(walk[-1], q)[1:]
                    candidate = walk + added
                    if (_valid_walk(candidate) and not set(added) & _halo(hub) and
                            not _halo(set(added)) & occupied):
                        proposals.append((candidate, nd))
                        break
            if not proposals:
                break
            walk, previous = proposals[int(rng.integers(len(proposals)))]
        # A last straight segment reaches any compatible boundary.
        endings = [previous, (-previous[1], previous[0]), (previous[1], -previous[0])]
        rng.shuffle(endings)
        for nd in endings:
            tail = _ray(walk[-1], nd, shape)
            candidate = walk + tail
            if (tail and _valid_walk(candidate) and not set(tail) & _halo(hub) and
                    not _halo(set(tail)) & occupied):
                rec = {"hub": hub, "box": (t, l, h, w), "kind": kind,
                       "word": _word(rng, period), "walk": candidate, "switch": None}
                if switched and not _select_switch(rec, shape, rng, allowed, require=True):
                    continue
                return rec
    return None


def _transform(pair, rotation, reflect=False):
    return {k: np.ascontiguousarray(np.rot90(np.fliplr(v) if reflect else v, rotation))
            for k, v in pair.items()}


def _damage_and_render(rng, shape, records, *, marked_pivot=False, bend_marker=False,
                       deleted_marker=False, force_gap=False):
    inp = np.full(shape, BG, dtype=np.int8)
    occupied = set()
    for rec in records:
        for p in rec["hub"]:
            if p in occupied:
                return None
            occupied.add(p)
            inp[p] = GUIDE
        for i, p in enumerate(rec["walk"]):
            if p in occupied:
                return None
            occupied.add(p)
            inp[p] = rec["word"][i] if i < len(rec["word"]) else GUIDE
    for rec in records:
        if rec["switch"]:
            q = rec["switch"]["marker"]
            if inp[q] != BG:
                return None
            inp[q] = SWITCH
    try:
        out = _paint_plan(shape, records)
        _, recovered, holes = _read_input(inp)
        assert not holes
        assert {tuple(r["walk"]) for r in recovered} == {tuple(r["walk"]) for r in records}
        assert np.array_equal(solve(inp), out)
    except AssertionError:
        return None
    required = set()
    for rec in records:
        p = len(rec["word"])
        turns = [i for i in _turn_indexes(rec["walk"]) if i > p]
        if bend_marker and turns:
            required.add(rec["walk"][turns[0]])
        edit = rec["switch"]
        if edit and marked_pivot:
            required.add(edit["pivot"])
        if edit and deleted_marker:
            old = rec["walk"][edit["index"]+1:]
            required.add(old[int(rng.integers(len(old)))])
        eligible = rec["walk"][p+1:]
        if eligible:
            n = min(len(eligible), int(rng.integers(1, 5)))
            required.update(eligible[int(i)] for i in rng.choice(len(eligible), n, replace=False))
    for p in required:
        inp[p] = REPAIR
    # Sparse omissions are restricted to straight guide cells with intact
    # immediate neighbors. The pixel parser must recover the exact original walk.
    gap_candidates = []
    for rec in records:
        walk, period = rec["walk"], len(rec["word"])
        turns = set(_turn_indexes(walk))
        for i in range(period+2, len(walk)-1):
            p = walk[i]
            if i in turns or any(inp[q] != GUIDE for q in walk[i-1:i+2]):
                continue
            if any(_inside(q := _add(p, d), shape) and inp[q] == SWITCH for d in DIRS):
                continue
            gap_candidates.append(p)
    rng.shuffle(gap_candidates)
    desired = int(rng.integers(1 if force_gap else 0, 4))
    gaps = 0
    for p in gap_candidates:
        if gaps >= desired:
            break
        inp[p] = BG
        try:
            _, recovered, _ = _read_input(inp)
            assert {tuple(r["walk"]) for r in recovered} == {tuple(r["walk"]) for r in records}
            assert np.array_equal(solve(inp), out)
        except AssertionError:
            inp[p] = GUIDE
        else:
            gaps += 1
    if force_gap and gaps == 0:
        return None
    try:
        assert np.array_equal(solve(inp), out)
    except AssertionError:
        return None
    return {"input": inp, "output": out}


def _winding_witness(rng, period=3, direction=None):
    """One curriculum witness: a new ray cuts through its obsolete winding suffix."""
    for _ in range(40):
        shape = (int(rng.integers(21, 31)), int(rng.integers(22, 31)))
        h, w = shape
        hh, hw = int(rng.choice((3, 5))), int(rng.choice((3, 5)))
        rh = h - hh//2 - int(rng.integers(2, 4))
        left = int(rng.integers(0, 3))
        first_c = left + hw
        cp = int(rng.integers(max(first_c+period+2, 12), w-6))
        rp = int(rng.integers(7, rh-3))
        cl = int(rng.integers(max(first_c+1, 5), cp-3))
        ru = int(rng.integers(1, rp-3))
        cr = int(rng.integers(cp+3, w-1))
        kind = str(rng.choice(("rectangle", "clipped", "cross")))
        hub = _hub(rh-hh//2, left, hh, hw, kind)
        walk = _polyline([(rh, first_c), (rh, cp), (rp, cp), (rp, cl),
                          (ru, cl), (ru, cr), (h-1, cr)])
        word = _word(rng, period, "distinct")
        k = walk.index((rp, cp))
        # Teach that the switch does not reset the clock, and that a shortened
        # suffix has a different phase where it revisits an old suffix cell.
        crossing = (rp, cr)
        new_i = k + cr-cp
        if (k+1) % period == 0 or word[new_i % period] == word[walk.index(crossing) % period]:
            continue
        edit = {"pivot": (rp, cp), "index": k, "direction": (0, 1),
                "marker": (rp, cp+1), "ray": _ray((rp, cp), (0, 1), shape)}
        rec = {"hub": hub, "box": (rh-hh//2, left, hh, hw), "kind": kind,
               "word": word, "walk": walk, "switch": edit}
        pair = _damage_and_render(rng, shape, [rec], marked_pivot=True,
                                  deleted_marker=True, force_gap=True)
        if pair:
            # np.rot90 rotates a rightward ray to up, left, down, right.
            rotation = int(rng.integers(4)) if direction is None else ((0, 1), (-1, 0), (0, -1), (1, 0)).index(direction)
            return _transform(pair, rotation)
    # A guaranteed geometry with nonzero phase offset for periods 3..5.
    shape = (28, 30)
    cp, rp, cl, ru, cr, rh = 17, 9, 10, 3, 24, 24
    for rp in range(8, 13):
        walk = _polyline([(rh, 5), (rh, cp), (rp, cp), (rp, cl),
                          (ru, cl), (ru, cr), (27, cr)])
        k = walk.index((rp, cp))
        if (k+1) % period and (walk.index((rp, cr)) - (k+cr-cp)) % period:
            break
    rec = {"hub": _hub(23, 2, 3, 3, "clipped"), "box": (23, 2, 3, 3),
           "kind": "clipped", "word": _word(rng, period, "distinct"),
           "walk": walk, "switch": {"pivot": (rp, cp), "index": k,
           "direction": (0, 1), "marker": (rp, cp+1), "ray": _ray((rp, cp), (0, 1), shape)}}
    pair = _damage_and_render(rng, shape, [rec], marked_pivot=True, deleted_marker=True, force_gap=True)
    assert pair is not None
    rotation = int(rng.integers(4)) if direction is None else ((0, 1), (-1, 0), (0, -1), (1, 0)).index(direction)
    return _transform(pair, rotation)


def _fallback(rng, count, periods, switches, allowed, anchor=False):
    """Finite escape layout; independent opposite-facing stems, no final crossings."""
    shape = (30, 30)
    configurations = [((6, 0, 3, 3), [(7, 3), (7, 11), (29, 11)]),
                      ((22, 27, 3, 3), [(23, 26), (23, 18), (29, 18)]),
                      ((0, 23, 3, 3), [(3, 24), (12, 24), (12, 29)])]
    if anchor:
        assert count == 1 and switches == 0
        configurations = [((3, 1, 3, 3), [(4, 4), (4, 16), (16, 16), (16, 8), (29, 8)])]
    recs = []
    for i in range(count):
        box, vertices = configurations[i]
        kind = str(rng.choice(("rectangle", "clipped", "cross")))
        walk = _polyline(vertices)
        rec = {"hub": _hub(*box, kind), "box": box, "kind": kind,
               "walk": walk, "word": _word(rng, periods[i], "aba" if anchor else "random"),
               "switch": None}
        if i < switches:
            # The two switched paths point up before orientation transformation.
            assert i < 2
            p = vertices[1]
            rec["switch"] = {"pivot": p, "index": walk.index(p), "direction": (-1, 0),
                             "marker": _add(p, (-1, 0)), "ray": _ray(p, (-1, 0), shape)}
        recs.append(rec)
    pair = _damage_and_render(rng, shape, recs, marked_pivot=bool(switches),
                              bend_marker=True, force_gap=anchor)
    assert pair is not None, "finite fallback failed"
    if switches:
        wanted = allowed[int(rng.integers(len(allowed)))]
        rotation = ((-1, 0), (0, -1), (1, 0), (0, 1)).index(wanted)
    else:
        rotation = int(rng.integers(4))
    return _transform(pair, rotation)


def _make_pair(rng, count, periods, switches, allowed=DIRS, anchor=False):
    for _ in range(24):
        low = 13 if count == 1 else 24
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        recs, occupied = [], set()
        for i in range(count):
            bends = int(rng.integers(1 if anchor else 0, 5 if count == 1 else 4))
            rec = _random_record(rng, shape, periods[i], bends, occupied, i < switches, allowed)
            if rec is None:
                break
            if anchor:
                rec["word"] = _word(rng, 3, "aba")
            marker = set() if rec["switch"] is None else {rec["switch"]["marker"]}
            occupied.update(rec["hub"] | set(rec["walk"]) | marker)
            recs.append(rec)
        if len(recs) != count:
            continue
        pair = _damage_and_render(rng, shape, recs, marked_pivot=rng.random() < .65,
                                  bend_marker=anchor, deleted_marker=bool(switches), force_gap=anchor)
        if pair is None:
            continue
        if anchor:
            f = _features(pair)
            if not f["bend_phase"] or not f["manhattan_phase"]:
                continue
        return pair
    return _fallback(rng, count, periods, switches, allowed, anchor)


def _count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return int(value)


def _key(grid):
    g = np.ascontiguousarray(grid, dtype=np.int8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))  # source count, multi-switch, orientation, word length
    held = DIRS[int(rng.integers(4))]
    allowed = tuple(d for d in DIRS if regime != 2 or d != held)
    train = [_make_pair(rng, 1, [3], 0, anchor=True),
             _winding_witness(rng, direction=allowed[int(rng.integers(len(allowed)))])]
    for _ in range(a-2):
        count = 1 if regime == 0 else int(rng.choice((1, 1, 2, 2, 3)))
        switches = int(rng.integers(0, min(count, 1 if regime == 1 else 2)+1))
        periods = list(map(int, rng.integers(1, 4 if regime == 3 else 6, count)))
        train.append(_make_pair(rng, count, periods, switches, allowed))
    test = []
    for _ in range(b):
        count = int(rng.choice((2, 2, 3))) if regime in (0, 1, 2) else int(rng.integers(1, 4))
        switches = 2 if regime == 1 else int(rng.integers(1, min(count, 2)+1))
        periods = list(map(int, rng.integers(4 if regime == 3 else 2, 6, count)))
        test.append(_make_pair(rng, count, periods, switches, (held,) if regime == 2 else DIRS))
    # Finite duplicate repair. Even a one-color motif has six recolorings;
    # two independent guide/repair toggles raise that to 24 inputs, exceeding
    # the maximum 14 exclusions in one puzzle. Geometry and answers are fixed.
    seen = set()
    for pair in train + test:
        if _key(pair["input"]) in seen:
            _, roots, _ = _read_input(pair["input"])
            sites = [p for r in roots for p in r["walk"][len(r["word"])+1:]
                     if pair["input"][p] in (GUIDE, REPAIR)][:2]
            assert len(sites) == 2
            found = False
            for bits in itertools.product((GUIDE, REPAIR), repeat=2):
                base = pair["input"].copy()
                for point, value in zip(sites, bits):
                    base[point] = value
                for values in itertools.permutations(PAYLOAD):
                    mapping = np.arange(10, dtype=np.int8)
                    mapping[list(PAYLOAD)] = values
                    candidate = {"input": mapping[base], "output": mapping[pair["output"]]}
                    if _key(candidate["input"]) not in seen:
                        pair.update(candidate)
                        found = True
                        break
                if found:
                    break
            else:
                raise AssertionError("finite distinct-input space exhausted")
        seen.add(_key(pair["input"]))
    rng.shuffle(train)
    result = {"train": train, "test": test}
    validate_generated_puzzle(result)
    return result


def _features(pair):
    inp, roots, holes = _read_input(pair["input"])
    out = _grid(pair["output"])
    switched = [r for r in roots if r["switch"] is not None]
    return {"shape": tuple(inp.shape), "sources": len(roots), "switches": len(switched),
            "periods": tuple(len(r["word"]) for r in roots),
            "words": tuple(r["word"] for r in roots),
            "hub_kinds": tuple(r["kind"] for r in roots),
            "hub_sizes": tuple(r["box"][2:] for r in roots),
            "turns": sum(len(_turn_indexes(r["walk"])) for r in roots),
            "directions": tuple(r["switch"]["direction"] for r in switched),
            "holes": len(holes), "markers": int(np.count_nonzero(inp == REPAIR)),
            "marked_pivots": sum(inp[r["switch"]["pivot"]] == REPAIR for r in switched),
            "old_bends": tuple(len(_turn_indexes(r["walk"][r["switch"]["index"]:])) for r in switched),
            "ray_lengths": tuple(len(r["switch"]["ray"]) for r in switched),
            "ray_old_hits": sum(len(set(r["switch"]["ray"]) &
                                    set(r["walk"][r["switch"]["index"]+1:])) for r in switched),
            "deleted_markers": sum(sum(inp[p] == REPAIR for p in r["walk"][r["switch"]["index"]+1:])
                                   for r in switched),
            "whole_word": any(len(r["word"]) == 3 and r["word"][0] == r["word"][2] != r["word"][1]
                              for r in roots),
            "bend_phase": any(r["switch"] is None and
                              any(i % len(r["word"]) != 0 for i in _turn_indexes(r["walk"]))
                              and len(set(r["word"])) > 1 for r in roots),
            "manhattan_phase": any(r["switch"] is None and
                                   any(r["word"][i % len(r["word"])] !=
                                       r["word"][(abs(p[0]-r["walk"][0][0]) +
                                                   abs(p[1]-r["walk"][0][1])) % len(r["word"])]
                                       for i, p in enumerate(r["walk"])) for r in roots),
            "no_reset": any(any(out[p] != r["word"][j % len(r["word"])]
                                for j, p in enumerate(r["switch"]["ray"])) for r in switched),
            "a_changes_answer": not np.array_equal(out, _paint_plan(inp.shape, roots, ignore_motif=True)),
            "b_changes_answer": not np.array_equal(out, _paint_plan(inp.shape, roots, ignore_switch=True))}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            _, roots, holes = _read_input(inp)
            expected = _paint_plan(inp.shape, roots)
            assert np.array_equal(out, expected), "wrong phase, route edit, repair, or preserved cell"
            assert not np.any(np.isin(out, (REPAIR, SWITCH)))
            hubs = set().union(*(r["hub"] for r in roots))
            assert set(map(tuple, np.argwhere(out == GUIDE))) == hubs, "only hubs remain gray"
            for r in roots:
                assert all(out[p] == inp[p] for p in r["walk"][:len(r["word"])]), "seed changed"
                for p in r["hub"]:
                    assert inp[p] == out[p] == GUIDE
            assert all(out[p] in PAYLOAD or out[p] == BG for p in holes)
            # Rendering in reversed source order must produce the same answer.
            assert np.array_equal(out, _paint_plan(inp.shape, list(reversed(roots))))
            if split == "train":
                training.append(_features(pair))
            else:
                testing.append(_features(pair))
    assert any(f["whole_word"] and f["holes"] and f["bend_phase"] and f["manhattan_phase"]
               for f in training), "teach whole-word repair and graph-distance phase"
    assert any(f["marked_pivots"] and f["deleted_markers"] and f["ray_old_hits"] and f["no_reset"]
               for f in training), "teach switched phase transport and bent deletion"
    assert any(f["a_changes_answer"] for f in training), "A must affect a demonstration"
    assert any(f["b_changes_answer"] for f in training), "B must affect a demonstration"
    train_dirs = {d for f in training for d in f["directions"]}
    test_dirs = {d for f in testing for d in f["directions"]}
    shifts = (max(f["sources"] for f in testing) > max(f["sources"] for f in training),
              max(f["switches"] for f in testing) > max(f["switches"] for f in training),
              bool(test_dirs - train_dirs),
              max(n for f in testing for n in f["periods"]) >
              max(n for f in training for n in f["periods"]))
    assert any(shifts), "missing structural train/test shift"


def _puzzle_key(p):
    return tuple(tuple((_key(q["input"]), _key(q["output"])) for q in p[s]) for s in ("train", "test"))


def self_test() -> dict:
    """Bounded 42-seed semantic checks and pixel-derived diversity measurements."""
    puzzles, features = [], []
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed)), "nondeterminism"
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                before = pair["input"].copy()
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert np.array_equal(before, pair["input"]), "solver mutated input"
                features.append(_features(pair))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert {f["sources"] for f in features} == {1, 2, 3}
    assert {f["switches"] for f in features} >= {0, 1, 2}
    assert {n for f in features for n in f["periods"]} == {1, 2, 3, 4, 5}
    assert {d for f in features for d in f["directions"]} == set(DIRS)
    assert {k for f in features for k in f["hub_kinds"]} == {"rectangle", "clipped", "cross"}
    assert len({f["shape"] for f in features}) >= 30
    assert len({f["turns"] for f in features}) >= 5
    assert len({n for f in features for n in f["ray_lengths"]}) >= 10
    assert any(f["holes"] >= 2 for f in features)
    assert any(f["ray_old_hits"] for f in features)
    assert any(f["old_bends"] and max(f["old_bends"]) >= 3 for f in features)
    assert any(f["sources"] == 2 and f["switches"] == 2 for f in features)
    # Validate every legal count combination. Reject nonintegers and explicit None.
    for a in range(2, 11):
        for b in range(1, min(a-1, 4)+1):
            p = generate_puzzle(12345, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Rotation/reflection invariance is a semantic check of source-relative phase.
    p = puzzles[0]
    for k in range(4):
        transformed = {s: [_transform(q, k, True) for q in p[s]] for s in p}
        validate_generated_puzzle(transformed)
    # Reject wrong answers at preserved cells, reconstructed gaps, pivots, deleted
    # suffix cells, and new rays. Full-array comparisons cover all other sites.
    sites = set()
    q = p["train"][next(i for i, x in enumerate(p["train"]) if _features(x)["ray_old_hits"])]
    _, roots, holes = _read_input(q["input"])
    sites.update(list(sorted(holes))[:1])
    sites.add(min(roots[0]["hub"]))
    e = roots[0]["switch"]
    sites.update((e["pivot"], e["ray"][0], roots[0]["walk"][-1]))
    for point in sites:
        mutated = {s: [dict(x) for x in p[s]] for s in p}
        index = next(i for i, x in enumerate(p["train"]) if x is q)
        mutated["train"][index]["output"] = q["output"].copy()
        mutated["train"][index]["output"][point] = (int(q["output"][point])+1) % 10
        try:
            validate_generated_puzzle(mutated)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt output accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": len(features),
            "shapes": len({f["shape"] for f in features}),
            "source_counts": sorted({f["sources"] for f in features}),
            "switch_counts": sorted({f["switches"] for f in features}),
            "periods": sorted({n for f in features for n in f["periods"]}),
            "turn_counts": sorted({f["turns"] for f in features}),
            "hub_kinds": sorted({k for f in features for k in f["hub_kinds"]}),
            "ray_lengths": sorted({n for f in features for n in f["ray_lengths"]}),
            "semantic_scope": "author-owned solver and invariants; no independent or blind solver"}


if __name__ == "__main__":
    report = self_test()
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report, separators=(",", ":")))
