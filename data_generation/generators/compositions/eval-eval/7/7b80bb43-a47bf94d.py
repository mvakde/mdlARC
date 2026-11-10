"""Compatibility-gated cable repair. Standard library + NumPy only.

Inputs encode all conventions: terminal stencils identify the two port roles,
diagonal deviations propose repairs, and endpoint agreement authorizes them.
No parent module, seed, stored example, or generation metadata is used by solve.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert ((a >= 0) & (a <= 9)).all()
    return a.astype(np.int8, copy=True)


def _at(a, p):
    return 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1] and bool(a[p])


def _strong(mask):
    result = np.zeros_like(mask)
    for m, out in ((mask, result), (mask.T, result.T)):
        triple = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for k in range(3):
            out[:, k:k + triple.shape[1]] |= triple
    return result


def _proposals(mask):
    """Visible three-cell gaps, two-cell spurs, and straight support."""
    strong = _strong(mask)
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        brackets = m[:, :-4] & m[:, 4:]
        brackets &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(brackets):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            if not (_at(mask, (r - dr, c - dc)) or _at(mask, (r + 5*dr, c + 5*dc))):
                continue
            matches = []
            for end in (0, 4):
                ar, ac = r + end*dr, c + end*dc
                toward = 1 if end == 0 else -1
                for side in (-1, 1):
                    sr, sc = toward*dr + side*dc, toward*dc + side*dr
                    spur = tuple((ar + k*sr, ac + k*sc) for k in (1, 2))
                    if (all(_at(mask, p) and not strong[p] for p in spur)
                            and not _at(mask, (ar + 3*sr, ac + 3*sc))):
                        matches.append(spur)
            if matches:
                assert len(matches) == 1, "multiple deviations at one gap"
                result.append(dict(gap=tuple((r + k*dr, c + k*dc) for k in (1, 2, 3)),
                                   spur=matches[0], axis="h" if dc else "v"))
    return result


def _hypothesis(mask):
    """Temporarily reconnect proposals; erase weak diagonal-only clutter."""
    proposals = _proposals(mask)
    filled = mask.copy()
    anchors = np.zeros_like(mask)
    for item in proposals:
        for p in item["gap"]:
            filled[p] = True
        a, b = item["gap"][0], item["gap"][-1]
        dr, dc = (0, 1) if item["axis"] == "h" else (1, 0)
        anchors[a[0] - dr, a[1] - dc] = True
        anchors[b[0] + dr, b[1] + dc] = True
    weak = mask & ~_strong(filled) & ~anchors
    erased = np.zeros_like(mask)
    for dc in (-1, 1):
        a = weak[:-1, :-1] if dc == 1 else weak[:-1, 1:]
        b = weak[1:, 1:] if dc == 1 else weak[1:, :-1]
        linked = a & b
        if dc == 1:
            erased[:-1, :-1] |= linked
            erased[1:, 1:] |= linked
        else:
            erased[:-1, 1:] |= linked
            erased[1:, :-1] |= linked
    return filled & ~erased, proposals, erased


def _motif(pattern, bg, label):
    return np.where(pattern, label, bg).astype(np.int8)


def _block(block, bg, source):
    if np.all(block == bg):
        return "blank", None
    values = set(map(int, np.unique(block))) - {bg}
    assert len(values) == 1
    label = next(iter(values))
    if source and np.all(block == label):
        return "solid", label
    pattern = DIAMOND if source else CHECKER
    assert np.array_equal(block, _motif(pattern, bg, label))
    return "diamond" if source else "checker", label


def _trace(mask, crosses, sources, receivers):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    ends = set(sources + receivers)
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in DIRS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    for p, ns in neighbors.items():
        assert len(ns) == (1 if p in ends else 4 if p in crosses else 2)
    seen, routes, permutation = set(), [], []
    receiver_index = {p: j for j, p in enumerate(receivers)}
    for start in sources:
        previous, p, path, states = None, start, [], set()
        while True:
            assert (previous, p) not in states, "cycle"
            states.add((previous, p))
            path.append(p)
            seen.add(p)
            if p in receiver_index:
                permutation.append(receiver_index[p])
                break
            assert p == start or p not in sources, "returned to source bank"
            ns = neighbors[p]
            if previous is None:
                q = ns[0]
            elif p in crosses:
                q = (2*p[0] - previous[0], 2*p[1] - previous[1])
                assert q in ns
            else:
                q = next(q for q in ns if q != previous)
            previous, p = p, q
        routes.append(path)
    assert sorted(permutation) == list(range(len(sources)))
    assert seen == cells, "unattached fragments in hypothesized wiring"
    return routes, permutation


def _view(a, turns, flip):
    b = np.rot90(a, turns)
    return b[:, ::-1] if flip else b


def _unview(a, turns, flip):
    if flip:
        a = a[:, ::-1]
    return np.rot90(a, -turns).copy()


def _parse(input_grid):
    original = _grid(input_grid)
    bg = int(original[0, 0])
    assert all(int(original[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    candidates = []
    for turns in range(4):
        for flip in (False, True):
            g = _view(original, turns, flip)
            for wire in set(map(int, np.unique(g))) - {bg}:
                positions = np.argwhere(g == wire)
                s, e = int(positions[:, 1].min()), int(positions[:, 0].max())
                if s < 3 or e + 4 > g.shape[0]:
                    continue
                rs = list(map(int, np.flatnonzero(g[:, s] == wire)))
                cs = list(map(int, np.flatnonzero(g[e] == wire)))
                if not 3 <= len(rs) == len(cs) <= 5:
                    continue
                if (min(rs) < 1 or max(rs) >= e - 3 or min(cs) <= s + 1
                        or max(cs) + 1 >= g.shape[1]
                        or min(np.diff(rs)) < 3 or min(np.diff(cs)) < 4):
                    continue
                try:
                    footprint = np.zeros_like(g, dtype=bool)
                    banks = [[], []]
                    for r in rs:
                        footprint[r-1:r+2, s-3:s] = True
                        banks[0].append(_block(g[r-1:r+2, s-3:s], bg, True))
                    for c in cs:
                        footprint[e+1:e+4, c-1:c+2] = True
                        banks[1].append(_block(g[e+1:e+4, c-1:c+2], bg, False))
                    assert any(k == "diamond" for k, _ in banks[0])
                    assert any(k == "checker" for k, _ in banks[1])
                    field = g.copy()
                    field[footprint] = bg
                    others = set(map(int, np.unique(field))) - {bg, wire}
                    assert len(others) <= 1
                    cross = next(iter(others)) if others else None
                    labels = {v for bank in banks for _, v in bank if v is not None}
                    assert labels.isdisjoint({bg, wire, cross})
                    assert np.all(field[:, :s] == bg) and np.all(field[e+1:] == bg)
                    mask, proposals, erased = _hypothesis(field != bg)
                    crosses = set(map(tuple, np.argwhere(field == cross).tolist())) if cross is not None else set()
                    routes, perm = _trace(mask, crosses, [(r, s) for r in rs], [(e, c) for c in cs])
                    active, colors = [], []
                    for i, j in enumerate(perm):
                        values = {banks[0][i][1], banks[1][j][1]} - {None}
                        assert values
                        active.append(len(values) == 1)
                        colors.append(next(iter(values)) if len(values) == 1 else None)
                    owners = {p: i for i, path in enumerate(routes) for p in path if p not in crosses}
                    for item in proposals:
                        indices = {owners[p] for p in item["gap"]}
                        assert len(indices) == 1
                        item["route"] = indices.pop()
                    candidates.append(dict(grid=g, bg=bg, wire=wire, cross=cross, field=field,
                                           footprint=footprint, rs=rs, cs=cs, s=s, e=e, banks=banks,
                                           routes=routes, permutation=perm, active=active, colors=colors,
                                           proposals=proposals, erased=erased, turns=turns, flip=flip,
                                           crossings=len(crosses)))
                except (AssertionError, KeyError, StopIteration):
                    continue
    assert len(candidates) == 1, f"ambiguous or invalid input: {len(candidates)} interpretations"
    return candidates[0]


def _answer(info):
    out = info["grid"].copy()
    bg, wire = info["bg"], info["wire"]
    out[info["erased"]] = bg
    for item in info["proposals"]:
        if info["active"][item["route"]]:
            for p in item["gap"]:
                out[p] = wire
    for i, j in enumerate(info["permutation"]):
        a, b = info["banks"][0][i][1], info["banks"][1][j][1]
        if info["active"][i]:
            a = b = info["colors"][i]
        if a is not None:
            r, s = info["rs"][i], info["s"]
            out[r-1:r+2, s-3:s] = _motif(DIAMOND, bg, a)
        if b is not None:
            e, c = info["e"], info["cs"][j]
            out[e+1:e+4, c-1:c+2] = _motif(CHECKER, bg, b)
    return _unview(out, info["turns"], info["flip"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from input cells alone."""
    return _answer(_parse(input_grid))


def _geometry(rng, n, serial):
    s = int(rng.integers(4, 7))
    top = int(rng.integers(3, 6))
    rs = [top]
    for k in range(1, n):
        hi = min(5, 20 - rs[-1] - 3*(n-1-k))
        rs.append(rs[-1] + int(rng.integers(3, hi+1)))
    cs = [int(rng.integers(s+3, min(s+5, 26-4*(n-1))+1))]
    for k in range(1, n):
        hi = min(6, 26 - cs[-1] - 4*(n-1-k))
        cs.append(cs[-1] + int(rng.integers(4, hi+1)))
    e = rs[-1] + int(rng.integers(6, min(11, 26-rs[-1])+1))
    h = e + 4 + int(rng.integers(min(2, 30-e-4)+1))
    w = cs[-1] + 3 + int(rng.integers(min(3, 30-cs[-1]-3)+1))
    perm = list(map(int, rng.permutation(n)))
    if serial:
        # The upper cable's long lateral run has space for two distinct faults.
        old = perm.index(n-1)
        perm[0], perm[old] = perm[old], perm[0]
    paths = []
    for r, j in zip(rs, perm):
        c = cs[j]
        path = [(r, x) for x in range(s, c+1)] + [(y, c) for y in range(r+1, e+1)]
        paths.append(path)
    paths = _wander(rng, paths, (h, w), s, e, serial)
    visits = Counter(p for path in paths for p in path)
    clean = np.zeros((h, w), dtype=bool)
    crossing = np.zeros_like(clean)
    for p, count in visits.items():
        clean[p] = True
        crossing[p] = count == 2
        assert count <= 2
    return clean, crossing, paths, perm, rs, cs, s, e


def _wander(rng, paths, shape, s, e, serial):
    """Add bounded rectangular excursions, preserving individually traced wires.

    A proposal changes actual path geometry, not the endpoint permutation. It is
    retained only when the visible graph traces precisely those constructed paths.
    """
    sources, receivers = [p[0] for p in paths], [p[-1] for p in paths]
    for i in map(int, rng.permutation(len(paths))):
        if (serial and i == 0) or rng.random() < .35:
            continue
        wanted = int(rng.integers(1, 3))
        for _ in range(18):
            if not wanted:
                break
            path = paths[i]
            start = int(rng.integers(max(1, len(path)-5)))
            length = int(rng.integers(4, 9))
            if start + length >= len(path):
                continue
            a, b = path[start], path[start+length]
            dr, dc = path[start+1][0]-a[0], path[start+1][1]-a[1]
            if path[start:start+length+1] != [(a[0]+k*dr, a[1]+k*dc) for k in range(length+1)]:
                continue
            side, offset = int(rng.choice([-1, 1])), int(rng.integers(2, 4))
            pr, pc = side*dc, side*dr
            detour = [a]
            detour += [(a[0]+k*pr, a[1]+k*pc) for k in range(1, offset+1)]
            detour += [(a[0]+offset*pr+k*dr, a[1]+offset*pc+k*dc) for k in range(1, length+1)]
            detour += [(b[0]+k*pr, b[1]+k*pc) for k in range(offset-1, -1, -1)]
            if not all(0 <= r < e and s < c < shape[1]-1 for r, c in detour[1:-1]):
                continue
            candidate = path[:start] + detour + path[start+length+1:]
            if len(set(candidate)) != len(candidate):
                continue
            trial_paths = paths.copy()
            trial_paths[i] = candidate
            visits = Counter(p for pth in trial_paths for p in pth)
            if max(visits.values()) > 2:
                continue
            mask = np.zeros(shape, dtype=bool)
            for p in visits:
                mask[p] = True
            try:
                traced, _ = _trace(mask, {p for p, count in visits.items() if count == 2}, sources, receivers)
                assert traced == trial_paths
                assert _strong(mask)[mask].all()
            except (AssertionError, StopIteration):
                continue
            paths = trial_paths
            wanted -= 1
    return paths


def _damage_options(clean, crossing, paths, s, e):
    result = [[] for _ in paths]
    for i, path in enumerate(paths):
        cells = set(path)
        for r, c in path:
            for dr, dc in ((0, 1), (1, 0)):
                run = [(r+k*dr, c+k*dc) for k in range(5)]
                if not all(p in cells and not crossing[p] for p in run):
                    continue
                if (r-dr, c-dc) not in cells and (r+5*dr, c+5*dc) not in cells:
                    continue
                gap = tuple(run[1:4])
                minus = clean.copy()
                for p in gap:
                    minus[p] = False
                for end in (0, 4):
                    ar, ac = run[end]
                    toward = 1 if end == 0 else -1
                    for side in (-1, 1):
                        sr, sc = toward*dr + side*dc, toward*dc + side*dr
                        spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                        if all(0 <= y < e-1 and s < x < clean.shape[1] and not clean[y, x]
                               and not any(_at(minus, (y+dy, x+dx)) for dy, dx in DIRS)
                               for y, x in spur):
                            result[i].append(dict(gap=gap, spur=spur, route=i, axis="h" if dc else "v"))
    return result


def _faults(rng, clean, options, mandatory, serial):
    inp, chosen = clean.copy(), []
    demands = ([0, 0] if serial else []) + [i for i in mandatory if not serial or i != 0]
    demands += [i for i in range(len(options)) if rng.random() < .55 and i not in demands]
    for i in demands:
        found = False
        for index in rng.permutation(len(options[i])):
            item = options[i][int(index)]
            if serial and i == 0 and len([x for x in chosen if x["route"] == 0]) < 2 and item["axis"] != "h":
                continue
            if any(not inp[p] for p in item["gap"]) or any(inp[p] for p in item["spur"]):
                continue
            trial = inp.copy()
            for p in item["gap"]:
                trial[p] = False
            for p in item["spur"]:
                trial[p] = True
            try:
                restored, proposed, _ = _hypothesis(trial)
            except AssertionError:
                continue
            wanted = {x["gap"] for x in chosen + [item]}
            if np.array_equal(restored, clean) and {x["gap"] for x in proposed} == wanted:
                inp = trial
                chosen.append(item)
                found = True
                break
        if not found and (i in mandatory or (serial and i == 0)):
            raise AssertionError("no room for required fault")
    assert not serial or sum(x["route"] == 0 for x in chosen) >= 2
    return inp, chosen


def _clutter(rng, mask, clean, s, e, wanted):
    for _ in range(wanted):
        for _ in range(25):
            r, c = int(rng.integers(1, e-2)), int(rng.integers(s+2, mask.shape[1]-1))
            dc = int(rng.choice([-1, 1]))
            cells = [(r+k, c+k*dc) for k in range(int(rng.integers(2, 4)))]
            if not all(0 <= y < e-1 and s < x < mask.shape[1]
                       and not mask[y, x] and not clean[y, x]
                       and not any(_at(mask, (y+dy, x+dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
                       for y, x in cells):
                continue
            trial = mask.copy()
            for p in cells:
                trial[p] = True
            try:
                restored, _, _ = _hypothesis(trial)
            except AssertionError:
                continue
            if np.array_equal(restored, clean):
                mask = trial
                break
    return mask


def _make_pair(rng, params, split, index):
    is_test = split == "test"
    n = 5 if is_test and params["shift"] == "count" else int(rng.integers(3, 5))
    serial = is_test and params["shift"] == "serial"
    for _ in range(64):
        try:
            clean, crossing, paths, perm, rs, cs, s, e = _geometry(rng, n, serial)
            order = list(map(int, rng.permutation(n)))
            forward, reverse, rejected = order[:3]
            active = [True] * n
            active[rejected] = False
            for i in order[3:]:
                active[i] = bool(rng.integers(2))
            options = _damage_options(clean, crossing, paths, s, e)
            im, chosen = _faults(rng, clean, options, [forward, reverse, rejected], serial)
            im = _clutter(rng, im, clean, s, e, int(rng.integers(3)))
            break
        except AssertionError:
            continue
    else:
        raise RuntimeError("bounded geometry construction exhausted")
    palette = list(map(int, rng.permutation(10))) if params["vary"] else params["palette"]
    bg, wire, cross = palette[:3]
    label_pool = palette[3:3+int(rng.integers(2, 8))]
    labels = list(map(int, rng.choice(label_pool, n)))
    labels[reverse] = next(c for c in label_pool if c != labels[forward])
    inp = np.where(im, wire, bg).astype(np.int8)
    out = np.where(clean, wire, bg).astype(np.int8)
    inp[crossing] = out[crossing] = cross
    for item in chosen:
        if not active[item["route"]]:
            for p in item["gap"]:
                out[p] = bg
    for i, j in enumerate(perm):
        label = labels[i]
        if not active[i]:
            other = int(rng.choice([c for c in label_pool if c != label]))
            mode = "diamond" if index % 2 == 0 else "solid"
            a, b = label, other
        elif i == forward:
            mode, a, b = ("solid", label, None) if index % 2 == 0 else ("diamond", label, label)
        elif i == reverse:
            mode, a, b = ("blank", None, label) if index % 2 == 0 else ("solid", label, None)
        else:
            mode, a, b = [("solid", label, None), ("blank", None, label),
                          ("diamond", label, label)][int(rng.integers(3))]
        r, c = rs[i], cs[j]
        inp[r-1:r+2, s-3:s] = bg if a is None else a if mode == "solid" else _motif(DIAMOND, bg, a)
        inp[e+1:e+4, c-1:c+2] = bg if b is None else _motif(CHECKER, bg, b)
        if active[i]:
            a = b = label
        out[r-1:r+2, s-3:s] = _motif(DIAMOND, bg, a)
        out[e+1:e+4, c-1:c+2] = _motif(CHECKER, bg, b)
    turns = params["rotation"] if params["shift"] == "orientation" else int(rng.integers(4))
    if is_test and params["shift"] == "orientation":
        turns = (turns + 1) % 4
    flip = bool(rng.integers(2))
    return {"input": _view(inp, turns, flip).copy(), "output": _view(out, turns, flip).copy()}


def _count(value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"count must be an integer in {lo}..{hi}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    params = dict(palette=list(map(int, rng.permutation(10))), vary=bool(rng.integers(2)),
                  shift=str(rng.choice(["serial", "count", "orientation"])), rotation=int(rng.integers(4)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", n), ("test", t)):
        for index in range(count):
            for _ in range(32):
                pair = _make_pair(rng, params, split, index)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    return puzzle


def _inspect(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _parse(inp)
    assert np.array_equal(_answer(info), out), "full input-derived answer disagrees"
    canonical_out = _view(out, info["turns"], info["flip"])
    accepted = rejected = 0
    for item in info["proposals"]:
        yes = info["active"][item["route"]]
        accepted += yes
        rejected += not yes
        assert all(canonical_out[p] == (info["wire"] if yes else info["bg"]) for p in item["gap"])
        assert all(canonical_out[p] == info["bg"] for p in item["spur"])
    assert accepted and rejected, "both gate outcomes must be visible"
    assert np.all(canonical_out[info["erased"]] == info["bg"])
    changed = canonical_out != info["grid"]
    allowed = info["erased"] | info["footprint"]
    for item in info["proposals"]:
        for p in item["gap"]:
            allowed[p] = True
    assert not np.any(changed & ~allowed), "unrelated drawing changed"
    # A damaged compatible cable with a blank partner makes both parents affect
    # the very same route, rather than separate objects with unrelated answers.
    assert any(info["active"][x["route"]]
               and (info["banks"][0][x["route"]][0] == "blank"
                    or info["banks"][1][info["permutation"][x["route"]]][0] == "blank")
               for x in info["proposals"])
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, t = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, evidence = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            info = _inspect(pair)
            if split == "train":
                for i, j in enumerate(info["permutation"]):
                    a, b = info["banks"][0][i], info["banks"][1][j]
                    evidence.add(a[0])
                    if info["active"][i]:
                        evidence.add("reverse" if a[0] == "blank" else "forward" if b[0] == "blank" else "both")
                    else:
                        evidence.add("conflict")
    assert {"solid", "diamond", "reverse", "forward", "both", "conflict"} <= evidence


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for k in ("input", "output"):
                a = np.asarray(pair[k], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, streaming 42-seed audit, plus equivariance and bad-label tests."""
    shapes, palettes, permutations, structures, hashes = set(), set(), set(), set(), set()
    stats = Counter()
    routes, faults, axes, cross_counts, bends = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect(pair)
                stats["pairs"] += 1
                stats["test_pairs"] += split == "test"
                shapes.add(pair["input"].shape)
                palettes.add((info["bg"], info["wire"], info["cross"]))
                permutations.add(tuple(info["permutation"]))
                routes.add(len(info["routes"]))
                counts = Counter(x["route"] for x in info["proposals"])
                faults.update(counts.values())
                axes.update(x["axis"] for x in info["proposals"])
                cross_counts.add(info["crossings"])
                for path in info["routes"]:
                    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
                    bends.add(sum(a != b for a, b in zip(directions, directions[1:])))
                role_map = {}
                normalized = bytes(role_map.setdefault(int(c), len(role_map)) for c in pair["input"].flat)
                structures.add((pair["input"].shape, hashlib.sha256(normalized).hexdigest()))
                for x in info["proposals"]:
                    stats["accepted_gaps" if info["active"][x["route"]] else "rejected_gaps"] += 1
                stats["nonidentity_pairings"] += info["permutation"] != list(range(len(info["routes"])))
                stats["repeated_source_labels"] += len({v for _, v in info["banks"][0] if v is not None}) < sum(v is not None for _, v in info["banks"][0])
    assert len(hashes) == 42 and routes == {3, 4, 5}
    assert len(shapes) >= 35 and len(palettes) >= 25 and len(permutations) >= 20
    assert {1, 2} <= faults and axes == {"h", "v"}
    assert 0 in cross_counts and max(cross_counts) >= 6
    assert 1 in bends and max(bends) >= 5
    assert stats["nonidentity_pairings"] >= 100 and stats["repeated_source_labels"] > 0
    assert len(structures) >= 180
    p = generate_puzzle(0)
    assert _digest(p) == _digest(generate_puzzle(0))
    pair = p["train"][0]
    for turns in range(4):
        for flip in (False, True):
            assert np.array_equal(solve(_view(pair["input"], turns, flip)), _view(pair["output"], turns, flip))
    palette = np.array([9, 7, 4, 0, 1, 6, 8, 2, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
    for n, t in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(901+n, n, t))
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    info = _parse(pair["input"])
    # Two explicit parent-omission counterfactuals: leave a genuine gap open,
    # or fill a conflicting route's locally identical proposed repair.
    for active in (True, False):
        item = next(x for x in info["proposals"] if info["active"][x["route"]] == active)
        mutated = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
        bad = _view(mutated["train"][0]["output"], info["turns"], info["flip"])
        for q in item["gap"]:
            bad[q] = info["bg"] if active else info["wire"]
        try:
            validate_generated_puzzle(mutated)
        except AssertionError:
            stats["rejected_mutations"] += 1
        else:
            raise AssertionError("omitted parent concept accepted")
    stats.update(distinct_puzzles=len(hashes), distinct_shapes=len(shapes), structural_inputs=len(structures),
                 role_palettes=len(palettes), permutations=len(permutations), max_crossings=max(cross_counts),
                 max_bends=max(bends))
    return dict(stats)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
