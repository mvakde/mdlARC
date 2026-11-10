"""Repair/reroute a thin network, then take the parity-selected top corner.

Only NumPy and the standard library are used. No parent module or artifact is
loaded at runtime. The fixed annotation roles make solve genuinely input-only.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


BACKGROUND, REPAIR, DIRECTION = 0, 2, 4
PAYLOAD = (1, 3, 5, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 283697064, 940987291, 461610936, 825506079, 697763983,
              262948621, 496833201, 923961079, 743805320, 71431805,
              452841471, 502638596, 34012779, 399485690, 450088940,
              913417035, 156295604, 717638776, 182985254, 22893876,
              512025782, 130875110, 224478358, 956391112, 461496908,
              487991661, 736055797, 208781521, 601880567, 995062622,
              519369644, 397618298, 537143187, 72092162, 332287355,
              627764122, 34648368, 46876184, 510150563, 111082742, 287722913)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, n):
    return 0 <= p[0] < n and 0 <= p[1] < n


def _boundary(p, n):
    return p[0] in (0, n - 1) or p[1] in (0, n - 1)


def _neighbors(p, cells):
    return [_add(p, d) for d in DIRS if _add(p, d) in cells]


def _line(a, b):
    if a[0] != b[0] and a[1] != b[1]:
        raise ValueError("non-orthogonal segment")
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [(a[0] + i * dr, a[1] + i * dc)
            for i in range(max(abs(b[0] - a[0]), abs(b[1] - a[1])) + 1)]


def _polyline(vertices):
    walk = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        walk.extend(_line(a, b)[1:])
    if len(set(walk)) != len(walk):
        raise ValueError("self-intersecting route")
    cells = set(walk)
    if any(len(_neighbors(p, cells)) > 2 for p in cells):
        raise ValueError("route touches itself")
    return walk


def _ray(p, d, n):
    cells = set()
    q = _add(p, d)
    while _inside(q, n):
        cells.add(q)
        q = _add(q, d)
    return cells


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _reconstruct(inp):
    """Marked waypoints, then bracketed gaps of one or two cells.

    A crossing belongs geometrically to both paths. Its visible input owner is
    preserved; generation never damages that owner or hides an ambiguous end.
    """
    work = inp.copy()
    pending = {tuple(map(int, p)) for p in np.argwhere(inp == REPAIR)}
    for _ in range(len(pending) + 1):
        progress = False
        for p in sorted(pending):
            candidates = {int(work[q]) for d in DIRS
                          if _inside(q := _add(p, d), len(inp)) and int(work[q]) in PAYLOAD}
            if len(candidates) != 1:
                candidates = set()
                for d in ((1, 0), (0, 1)):
                    a, b = _add(p, d), _add(p, (-d[0], -d[1]))
                    if (_inside(a, len(inp)) and _inside(b, len(inp))
                            and work[a] == work[b] and int(work[a]) in PAYLOAD):
                        candidates.add(int(work[a]))
            if len(candidates) == 1:
                work[p] = next(iter(candidates))
                pending.remove(p)
                progress = True
        if not progress:
            break
    if pending:
        raise ValueError("repair marker lacks a unique color witness")
    colors = sorted(set(map(int, np.unique(work))) & set(PAYLOAD))
    layers = {c: set(map(tuple, np.argwhere(work == c).tolist())) for c in colors}
    if not layers:
        raise ValueError("no trace")
    for color, cells in layers.items():
        original = set(cells)
        for p in sorted(original):
            for d in ((1, 0), (0, 1)):
                perp = (-d[1], d[0])

                def turns(x):
                    return (_add(x, perp) in original
                            or _add(x, (-perp[0], -perp[1])) in original)

                for distance in (2, 3):
                    q = p[0] + distance * d[0], p[1] + distance * d[1]
                    if q not in original or (turns(p) and turns(q)):
                        continue
                    between = _line(p, q)[1:-1]
                    if not any(inp[x] == DIRECTION for x in between):
                        cells.update(between)
    return layers


def _find_edits(inp, layers):
    edits = []
    n = len(inp)
    for marker in map(tuple, np.argwhere(inp == DIRECTION).tolist()):
        candidates = []
        for d in DIRS:
            pivot = marker[0] - d[0], marker[1] - d[1]
            start = pivot[0] - d[0], pivot[1] - d[1]
            for color, cells in layers.items():
                if pivot not in cells or start not in cells or marker in cells:
                    continue
                perp = (-d[1], d[0])
                if not any(_add(pivot, t) in cells for t in (perp, (-perp[0], -perp[1]))):
                    continue
                previous, current, old = pivot, start, []
                while current not in old and len(old) < inp.size:
                    old.append(current)
                    following = [q for q in _neighbors(current, cells) if q != previous]
                    if not following:
                        if _boundary(current, n):
                            candidates.append({"color": color, "pivot": pivot, "direction": d,
                                               "old": set(old), "walk": (pivot, *old),
                                               "new": _ray(pivot, d, n)})
                        break
                    if len(following) != 1:
                        break
                    previous, current = current, following[0]
        if len(candidates) != 1:
            raise ValueError("direction marker lacks one unique terminal branch")
        edits.append(candidates[0])
    for i, a in enumerate(edits):
        for b in edits[i + 1:]:
            if a["color"] == b["color"]:
                if (a["old"] & b["old"] or a["pivot"] in b["old"]
                        or b["pivot"] in a["old"]):
                    raise ValueError("conflicting terminal edits")
            elif a["new"] & b["new"]:
                raise ValueError("different new rays may not overlap")
    return edits


def _render(inp, layers, edits=(), *, delete=True, draw=True):
    kept = {c: set(s) for c, s in layers.items()}
    if delete:
        for e in edits:
            kept[e["color"]].difference_update(e["old"])
    out = np.zeros_like(inp)
    owners = {}
    for c, cells in kept.items():
        for p in cells:
            owners.setdefault(p, []).append(c)
    for p, cs in owners.items():
        if len(cs) == 1:
            out[p] = cs[0]
        elif int(inp[p]) in cs:
            out[p] = inp[p]
        else:
            raise ValueError("crossing ownership is not visible")
    if draw:
        for e in edits:
            for p in e["new"]:
                out[p] = e["color"]
    return out


def _analyze(value):
    inp = _grid(value)
    if inp.shape[0] != inp.shape[1] or len(inp) < 2:
        raise ValueError("the canvas must be square and at least 2x2")
    layers = _reconstruct(inp)
    edits = _find_edits(inp, layers)
    repaired = _render(inp, layers)
    final = _render(inp, layers, edits)
    return inp, layers, edits, repaired, final


def _crop(grid, parity=None):
    odd = len(grid) % 2 if parity is None else parity
    return grid[:2, :2].copy() if odd else grid[:2, -2:].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All reconstruction and simultaneous routing precedes parity extraction."""
    return _crop(_analyze(input_grid)[4])


def _make_edit(color, old_vertices, retained_vertices, direction, n):
    old = _polyline(old_vertices)
    retained = _polyline(retained_vertices)
    pivot = old[0]
    if retained[0] != pivot or set(old[1:]) & set(retained):
        raise ValueError("old branch must meet retained path only at pivot")
    cells = set(old) | set(retained)
    if any(len(_neighbors(p, cells)) > 2 for p in cells if p != pivot):
        raise ValueError("ambiguous old branch")
    marker = _add(pivot, direction)
    if marker in cells:
        raise ValueError("occupied directional marker")
    edit = {"color": color, "pivot": pivot, "direction": direction,
            "old": set(old[1:]), "walk": tuple(old), "new": _ray(pivot, direction, n)}
    return cells, edit


def _corner(rng, n, kind, bends, route_color, stable_color, right):
    """Sample corner-reaching paths; no fixed whole-puzzle layout is copied."""
    m = (n - 3) // 2
    t = int(rng.integers(2))
    s = 1 - t
    if kind == "horizontal":
        row = int(rng.integers(2))
        col = int(rng.integers(4, m - 1))
        pivot = (row, col)
        old = [pivot, (row, m), (n - 1, m)]
        retained = [pivot, (int(rng.integers(4, n - 2)), col)]
        d = (0, -1)
        end = int(rng.integers(3, n - 3))
        stable = [(0, s), (end, s)]
        if rng.random() < .5:
            stable.append((end, int(rng.integers(2, col - 1))))
        repair = (1 - row, s)
    elif kind == "grow":
        top_limit = n - 10 if bends >= 4 else n - 6
        row = int(rng.integers(4, max(5, top_limit + 1)))
        pivot = (row, t)
        if bends >= 4 and m >= t + 6 and row + 6 < n - 1:
            old = [pivot, (row + 3, t), (row + 3, m),
                   (row + 6, m), (row + 6, t + 3), (n - 1, t + 3)]
        elif bends >= 2:
            old = [pivot, (row + 3, t), (row + 3, m), (n - 1, m)]
        else:
            old = [pivot, (n - 1, t)]
        retained = [pivot, (row, int(rng.integers(3, m + 1)))]
        d = (-1, 0)
        end = int(rng.integers(2, row - 1))
        stable = [(0, s), (end, s)]
        if rng.random() < .5:
            stable.append((end, int(rng.integers(3, m + 1))))
        repair = (1, s)
    else:  # Erase an old route whose boundary end lies in the candidate crop.
        row = int(rng.integers(9, n - 3))
        if bends >= 4 and m >= t + 6:
            pivot = (row, m)
            old = [pivot, (row - 3, m), (row - 3, t + 3),
                   (3, t + 3), (3, t), (0, t)]
            retained = [pivot, (row, m - 3)]
        elif bends >= 2:
            col = int(rng.integers(3, m - 1))
            pivot = (row, col)
            bend_row = int(rng.integers(3, row - 2))
            old = [pivot, (bend_row, col), (bend_row, t), (0, t)]
            retained = [pivot, (row, m)]
        else:
            pivot = (row, t)
            old = [pivot, (0, t)]
            retained = [pivot, (row, int(rng.integers(3, m + 1)))]
        d = (1, 0)
        end = int(rng.integers(2, row - 2))
        stable = [(0, s), (end, s)]
        if rng.random() < .4:
            stable.append((end, int(rng.integers(3, m + 1))))
        repair = (1, s)
    route, edit = _make_edit(route_color, old, retained, d, n)
    stable = set(_polyline(stable))
    layers = {stable_color: stable, route_color: route}
    if right:
        flip = lambda p: (p[0], n - 1 - p[1])
        layers = {c: {flip(p) for p in cells} for c, cells in layers.items()}
        edit = dict(edit, pivot=flip(edit["pivot"]),
                    direction=(d[0], -d[1]), old={flip(p) for p in edit["old"]},
                    walk=tuple(flip(p) for p in edit["walk"]),
                    new={flip(p) for p in edit["new"]})
        repair = flip(repair)
    return layers, edit, repair


def _extra(rng, n, color, redirect):
    col = int(rng.integers(3, n - 3))
    row = n - 2 if redirect and rng.random() < .25 else int(rng.integers(5, n - 3))
    candidates = [c for c in range(2, n - 2) if abs(c - col) >= 3]
    col2 = int(rng.choice(candidates))
    if redirect:
        pivot = (row, col)
        cells, edit = _make_edit(color, [pivot, (0, col)],
                                 [pivot, (row, col2)], (1, 0), n)
        return cells, edit
    vertices = [(int(rng.integers(2, row)), col), (row, col), (row, col2)]
    if rng.random() < .6:
        vertices.append((int(rng.integers(2, row - 1)), col2))
    return set(_polyline(vertices)), None


def _edit_key(edits):
    return {(e["color"], e["pivot"], e["direction"], frozenset(e["old"])) for e in edits}


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _construct(rng, n, selected_kind, bends, extras, used):
    for _attempt in range(160):
        try:
            colors = list(map(int, rng.permutation(PAYLOAD)))
            selected_right = n % 2 == 0
            kinds = {selected_right: selected_kind,
                     not selected_right: "erase" if selected_kind != "erase" else
                     str(rng.choice(["grow", "horizontal"]))}
            layers, edits, forced = {}, [], []
            for index, right in enumerate((False, True)):
                ls, edit, repair = _corner(rng, n, kinds[right], bends,
                                          colors[2 * index], colors[2 * index + 1], right)
                layers.update(ls)
                edits.append(edit)
                forced.append(repair)
            for index in range(extras):
                c = colors[4 + index]
                cells, edit = _extra(rng, n, c, bool(rng.random() < .55))
                layers[c] = cells
                if edit is not None:
                    edits.append(edit)
            complete = np.zeros((n, n), dtype=np.int8)
            multiplicity = np.zeros_like(complete)
            for c, cells in layers.items():
                for p in cells:
                    complete[p] = c
                    multiplicity[p] += 1
            if multiplicity.max() > 2:
                continue
            # Truth uses constructed paths and edits, never solve's inferred ones.
            final = _render(complete, layers, edits)
            inp = complete.copy()
            markers = [_add(e["pivot"], e["direction"]) for e in edits]
            if len(set(markers)) != len(markers) or any(inp[p] != 0 for p in markers):
                continue
            for p in markers:
                inp[p] = DIRECTION
            if any(multiplicity[p] != 1 for p in forced):
                continue
            for p in forced:
                inp[p] = REPAIR

            def recover_ok():
                recovered = _reconstruct(inp)
                return (recovered == layers
                        and _edit_key(_find_edits(inp, recovered)) == _edit_key(edits))

            if not recover_ok():
                continue
            sites = [p for cells in layers.values() for p in cells
                     if multiplicity[p] == 1 and inp[p] not in (REPAIR, DIRECTION)]
            sites.sort()
            rng.shuffle(sites)
            # At most 12 damage proposals: bounded time, no saved intermediates.
            marker_budget = int(rng.integers(1, 6))
            gap_budget = int(rng.integers(0, 5))
            for p in sites[:12]:
                choices = ([REPAIR] if marker_budget else []) + ([0] if gap_budget else [])
                if not choices:
                    break
                replacement = int(rng.choice(choices))
                previous = int(inp[p])
                inp[p] = replacement
                try:
                    acceptable = recover_ok()
                except ValueError:
                    acceptable = False
                if not acceptable:
                    inp[p] = previous
                elif replacement == REPAIR:
                    marker_budget -= 1
                else:
                    gap_budget -= 1
            analysis = _analyze(inp)
            if not np.array_equal(analysis[4], final):
                continue
            out = _crop(final)
            # Both corners carry repair work; the selected one must also carry
            # routing work, so neither inherited operation is answer-neutral.
            if (np.array_equal(out, _crop(inp))
                    or np.array_equal(out, _crop(analysis[3]))
                    or np.array_equal(out, _crop(final, 1 - n % 2))):
                continue
            key = _signature(inp)
            if key in used:
                continue
            used.add(key)
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded construction exhausted for side %d" % n)


def _integer(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
            or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    first_parity = int(rng.integers(2))
    used, train, test = set(), [], []
    sizes = {0: [], 1: []}
    for i in range(num_train):
        parity = (first_parity + i) % 2
        options = [n for n in range(16, 25) if n % 2 == parity and n not in sizes[parity]]
        if not options:
            options = [n for n in range(16, 25) if n % 2 == parity]
        n = int(rng.choice(options))
        sizes[parity].append(n)
        kind = ("horizontal", "erase", "erase", "grow")[i % 4]
        bends = (0, 2, 0, 2)[i % 4]
        train.append(_construct(rng, n, kind, bends, int(rng.integers(0, 3)), used))
    for i in range(num_test):
        parity = (int(rng.integers(2)) + i) % 2
        n = int(rng.choice([v for v in range(20, 31) if v % 2 == parity]))
        # The first query extends the demonstrated 0/1/2-turn rule to four
        # turns. Both addition and deletion queries occur, so mode is no cue.
        kind = str(rng.choice(["grow", "erase"] if i == 0 else
                              ["grow", "erase", "horizontal"]))
        test.append(_construct(rng, n, kind, 4 if i == 0 else int(rng.choice([2, 4])),
                               int(rng.integers(1, 4)), used))
    rng.shuffle(train)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _bends(walk):
    ds = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _features(pair):
    inp, layers, edits, repaired, final = _analyze(pair["input"])
    selected = {(r, c) for r in (0, 1)
                for c in ((0, 1) if len(inp) % 2 else (len(inp) - 2, len(inp) - 1))}
    features = {"side": len(inp), "parity": len(inp) % 2, "payloads": len(layers),
            "redirects": len(edits), "repairs": int(np.sum(inp == REPAIR)),
            "gaps": int(np.sum((inp == 0) & (repaired != 0))),
            "directions": [e["direction"] for e in edits],
            "bends": [_bends(e["walk"]) for e in edits],
            "ray_lengths": [len(e["new"]) for e in edits],
            "marked_pivots": sum(inp[e["pivot"]] == REPAIR for e in edits),
            "deleted_markers": sum(sum(inp[p] == REPAIR for p in e["old"]) for e in edits),
            "crossings": sum(sum(p in s for s in layers.values()) > 1
                             for p in set().union(*layers.values())),
            "new_crossings": sum(any(p in s for c, s in layers.items() if c != e["color"])
                                 for e in edits for p in e["new"]),
            "crop_added": sum(repaired[p] != final[p] and final[p] != 0 for p in selected),
            "crop_deleted": sum(repaired[p] != 0 and final[p] == 0 for p in selected)}
    return {k: int(v) if isinstance(v, np.integer) else v for k, v in features.items()}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "counts")
    seen, parities, train_features = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _, layers, edits, repaired, final = _analyze(inp)
            _require(out.shape == (2, 2), "2x2 output")
            _require(np.array_equal(out, _crop(final)), "wrong reconstructed parity crop")
            _require(not np.any(np.isin(final, (REPAIR, DIRECTION))), "annotation survived")
            _require(np.array_equal(final, _render(inp, layers, list(reversed(edits)))),
                     "edit order matters")
            _require(_signature(inp) not in seen, "duplicate input")
            seen.add(_signature(inp))
            _require(not np.array_equal(out, _crop(inp)), "raw crop shortcut succeeds")
            _require(not np.array_equal(out, _crop(repaired)), "routing is answer-neutral")
            _require(not np.array_equal(out, _crop(final, 1 - len(inp) % 2)),
                     "parity is answer-neutral")
            # Repair contributes visible literal annotation replacement in the
            # selected block, independent of the separate routing modification.
            _require(np.any((_crop(inp) == REPAIR) & np.isin(out, PAYLOAD)),
                     "selected corner has no surviving repair witness")
            raw = {c: set(map(tuple, np.argwhere(inp == c).tolist())) for c in layers}
            _require(not np.array_equal(out, _crop(_render(inp, raw, edits))),
                     "repair is answer-neutral even with perfect routing supplied")
            if split == "train":
                parities.add(len(inp) % 2)
                train_features.append(_features(pair))
    _require(parities == {0, 1}, "both parity branches must be demonstrated")
    _require(any(f["crop_added"] for f in train_features), "no ray-addition demonstration")
    _require(any(f["crop_deleted"] for f in train_features), "no branch-deletion demonstration")
    _require(any(max(f["bends"]) >= 2 and f["crop_deleted"] for f in train_features),
             "no answer-relevant bent deletion demonstration")
    if n >= 4:
        _require(all(len({f["side"] for f in train_features if f["parity"] == p}) >= 2
                     for p in (0, 1)), "two sizes of each parity required")


def _regression_tests():
    # Hand-constructed full-canvas expectations exercise information that the
    # tiny crop cannot reveal. They do not use any sampled seed or parent pair.
    a = np.zeros((9, 9), dtype=np.int8)
    a[:, 3] = 3
    a[2, 1:6] = 6
    a[:3, 5] = 6
    a[2:6, 1] = 6
    a[5, 1:5] = 6
    a[6, 1] = DIRECTION
    a[5, 1] = a[2, 5] = a[0, 3] = REPAIR
    a[7, 3] = 0
    expected = np.zeros_like(a)
    expected[:, 3] = 3
    expected[5, 1:5] = 6
    expected[6:, 1] = 6
    _require(np.array_equal(_analyze(a)[4], expected),
             "marked pivot/bend/endpoint, gap, and crossing-restoration regression")

    b = np.zeros((9, 9), dtype=np.int8)
    b[:8, 5] = 3
    b[7, 2:6] = 3
    b[4:8, 7] = 6
    b[4, 8] = 6
    b[4, 6] = b[8, 5] = DIRECTION
    expected = np.zeros_like(b)
    expected[7, 2:6] = 3
    expected[8, 5] = 3
    expected[4:8, 7] = 6
    expected[4, :7] = 6
    _, ls, edits, _, final = _analyze(b)
    _require(np.array_equal(final, expected), "new ray crossing another removed branch")
    _require(np.array_equal(_render(b, ls, list(reversed(edits))), expected),
             "simultaneous edit regression")


def self_test():
    """42 supplied seeds, streaming measurements, no files or retained puzzles."""
    _regression_tests()
    signatures, structures, sizes, palettes, directions = set(), set(), set(), set(), set()
    payload_counts, redirects, bends, rays = set(), set(), set(), set()
    totals = {k: 0 for k in ("gaps", "marked_pivots", "deleted_markers", "crossings", "new_crossings")}
    test_parities = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                digest.update(bytes([len(inp)]))
                digest.update(inp.tobytes())
                f = _features(pair)
                sizes.add(f["side"])
                palettes.add(tuple(np.unique(inp)))
                payload_counts.add(f["payloads"])
                redirects.add(f["redirects"])
                bends.update(f["bends"])
                rays.update(f["ray_lengths"])
                directions.update(f["directions"])
                for k in totals:
                    totals[k] += f[k]
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add(hashlib.sha256(bytes([len(inp)]) + normalized).hexdigest())
                _require(np.array_equal(solve(inp.tolist()), pair["output"]), "list solve")
                if split == "test":
                    test_parities.add(f["parity"])
        signatures.add(digest.digest())
        # A corrupt label must fail even if it changes a previously blank cell.
        bad = {s: [dict(p) for p in puzzle[s]] for s in puzzle}
        bad["test"][0]["output"] = puzzle["test"][0]["output"].copy()
        bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupt label accepted")
    _require(len(signatures) == 42 and len(structures) >= 190, "structural diversity")
    _require(len(sizes) >= 12 and len(palettes) >= 20, "scale/palette diversity")
    _require(test_parities == {0, 1} and directions == set(DIRS), "parity/direction coverage")
    _require(len(payload_counts) >= 3 and len(redirects) >= 3, "path/count diversity")
    _require({0, 1, 2} <= bends and max(bends) >= 4, "bent-route diversity")
    _require(min(rays) == 1 and max(rays) >= 15, "boundary-ray diversity")
    _require(all(v > 0 for v in totals.values()), "repair/crossing variation missing")
    for a, b in ((2, 1), (10, 4), (7, 3)):
        p, q = generate_puzzle(1234 + a, a, b), generate_puzzle(1234 + a, a, b)
        _require(all(_signature(x[r]) == _signature(y[r])
                     for s in p for x, y in zip(p[s], q[s]) for r in ("input", "output")),
                 "determinism")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_structures": len(structures),
            "sides": sorted(sizes), "payload_counts": sorted(payload_counts),
            "redirect_counts": sorted(redirects), "old_bends": sorted(bends),
            "ray_lengths": [min(rays), max(rays)], "totals": totals}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
