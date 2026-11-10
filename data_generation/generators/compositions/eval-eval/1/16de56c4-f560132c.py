"""Rhythm-locked mosaic: jointly register fragments by shape and cadence.

No parent modules, files, random state, or examples are consulted by solve.
Construction paints a known assembly; solving enumerates visible registrations.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape), "invalid grid shape")
    _require(np.issubdtype(array.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= array) & (array <= 9))), "invalid color")
    return array.astype(np.int8, copy=True)


def _key(array):
    return array.shape, array.tobytes()


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, found = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _N4:
                q = r + dr, c + dc
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
                    found.append(q)
        components.append(np.asarray(found, dtype=int))
    return components


def _crop_piece(grid, coords):
    low, high = coords.min(axis=0), coords.max(axis=0) + 1
    piece = np.full(tuple(high - low), -1, dtype=np.int8)
    local = coords - low
    piece[local[:, 0], local[:, 1]] = grid[coords[:, 0], coords[:, 1]]
    return piece


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "blank border required")
    components = _components(grid != bg)
    _require(len(components) == 4, "four disconnected fragments required")
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    _require(len(colors) == 7, "carrier, two inks, four key hues required")
    carrier = int(colors[np.argmax(counts)])
    singletons = colors[counts == 1]
    _require(len(singletons) == 4, "four singleton key hues required")
    points = np.argwhere(np.isin(grid, singletons))
    r, c = points.min(axis=0)
    _require(set(map(tuple, points)) == {(r, c), (r, c+1), (r+1, c), (r+1, c+1)}, "key must be a two by two block")
    key = grid[r:r+2, c:c+2].copy().reshape(-1)
    inks = [int(color) for color in colors if color != carrier and color not in singletons]
    _require(len(inks) == 2, "two marker inks required")
    inks.sort(key=lambda color: np.count_nonzero(grid == color), reverse=True)
    ref, terminal = inks
    _require(np.count_nonzero(grid == ref) > np.count_nonzero(grid == terminal) >= 2, "unreadable ink roles")
    anchor = [i for i, comp in enumerate(components) if any(np.array_equal(p, (r, c)) for p in comp)]
    _require(len(anchor) == 1, "missing anchor")
    order = anchor + [i for i in range(4) if i != anchor[0]]
    pieces = [_crop_piece(grid, components[i]) for i in order]
    for piece in pieces:
        _require(np.count_nonzero(piece == carrier) > np.count_nonzero((piece >= 0) & (piece != carrier)), "each fragment needs a majority carrier")
    area = sum(int(np.count_nonzero(p >= 0)) for p in pieces)
    side = math.isqrt(area)
    _require(side * side == area and 1 <= side <= 30, "fragment area must be square")
    return {"pieces": pieces, "side": side, "key": key, "bg": bg, "carrier": carrier, "ref": ref, "terminal": terminal}


def _placements(pieces, side):
    """All corner-constrained exact covers, retaining decorated rotations.

    A corner fixes the bounding-box translation. At most 6*4**3=384
    complete assignments exist; no unbounded placement search is needed.
    """
    corners = ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))
    table = {}
    for i, piece in enumerate(pieces):
        rotations = (0,) if i == 0 else range(4)
        variants, seen = [], set()
        for turns in rotations:
            rotated = np.rot90(piece, turns).copy()
            h, w = rotated.shape
            signature = _key(rotated)
            if h > side or w > side or signature in seen:
                continue
            seen.add(signature)
            rr, cc = np.nonzero(rotated >= 0)
            variants.append((turns, h, w, rr, cc, rotated[rr, cc]))
        for corner in ((0,) if i == 0 else (1, 2, 3)):
            choices = []
            for turns, h, w, local_rr, local_cc, values in variants:
                r0 = 0 if corner < 2 else side-h
                c0 = 0 if corner % 2 == 0 else side-w
                rr, cc = local_rr+r0, local_cc+c0
                mask = sum(1 << (int(r)*side + int(c)) for r, c in zip(rr, cc))
                hits = [bool(mask & (1 << (r*side+c))) for r, c in corners]
                if sum(hits) != 1 or not hits[corner]:
                    continue
                choices.append((mask, rr, cc, values, i, turns))
            table[i, corner] = choices
    full = (1 << (side*side)) - 1
    for anchor in table.get((0, 0), []):
        def visit(corner, remaining, occupied, chosen):
            if corner == 4:
                if occupied == full:
                    yield chosen
                return
            for i in remaining:
                for candidate in table[i, corner]:
                    if not occupied & candidate[0]:
                        yield from visit(corner+1, tuple(j for j in remaining if j != i), occupied | candidate[0], chosen + [candidate])
        yield from visit(1, (1, 2, 3), anchor[0], [anchor])


def _signals(assembled, owner, ref, terminal, axis):
    rows = assembled if axis == "H" else assembled.T
    regions = owner if axis == "H" else owner.T
    side = len(rows)
    signals = []
    for line, values in enumerate(rows):
        refs = np.flatnonzero(values == ref).tolist()
        ends = np.flatnonzero(values == terminal).tolist()
        if not refs and not ends:
            continue
        if len(refs) != 2 or len(ends) > 1:
            return None
        a, b = refs
        period = b-a
        phase = list(range(a % period, side, period))
        if len(phase) < 3:
            return None
        t = ends[0] if ends else None
        if t is not None:
            if a < t < b or (t-a) % period:
                return None
            if regions[line, t] == regions[line, a] == regions[line, b]:
                return None
        elif len({int(regions[line, x]) for x in phase}) < 2:
            return None
        signals.append((line, a, b, t, period))
    if len(signals) < 3 or sum(s[3] is not None for s in signals) < 2 or all(s[3] is not None for s in signals):
        return None
    return signals


def _paint(side, owner, signals, axis, carrier, ref, key, mode="normal"):
    out = np.full((side, side), carrier, dtype=np.int8)
    view = out if axis == "H" else out.T
    regions = owner if axis == "H" else owner.T
    for line, a, b, t, period in signals:
        if mode == "solid":
            period = 1
        if t is None:
            positions, color = range(a % period, side, period), ref
        else:
            low, high = min(a, t), max(b, t)
            if mode == "unbounded":
                low, high = a % period, side-1
            positions = range(low, high+1, period)
            corner = int(regions[line, t])
            if mode == "source_key":
                corner = int(regions[line, a])
            color = int(key[corner])
        view[line, list(positions)] = color
    return out


def _registered(analysis, limit=2):
    pieces, side = analysis["pieces"], analysis["side"]
    solutions, signatures = [], set()
    geometry_count = 0
    for assignment in _placements(pieces, side):
        geometry_count += 1
        assembled = np.full((side, side), -1, dtype=np.int8)
        owner = assembled.copy()
        for corner, candidate in enumerate(assignment):
            _, rr, cc, values, _, _ = candidate
            assembled[rr, cc], owner[rr, cc] = values, corner
        for axis in ("H", "V"):
            signals = _signals(assembled, owner, analysis["ref"], analysis["terminal"], axis)
            if signals is None:
                continue
            signature = (owner.tobytes(), assembled.tobytes(), axis)
            if signature in signatures:
                continue
            signatures.add(signature)
            output = _paint(side, owner, signals, axis, analysis["carrier"], analysis["ref"], analysis["key"])
            solutions.append({"owner": owner, "assembled": assembled, "axis": axis, "signals": signals, "output": output,
                              "rotations": tuple(c[5] for c in assignment), "assignment": tuple(c[4] for c in assignment)})
            if len(solutions) >= limit:
                return solutions, geometry_count
    return solutions, geometry_count


def _analyze(input_grid):
    analysis = _parse(input_grid)
    solutions, geometry_count = _registered(analysis)
    _require(len(solutions) == 1, "shape and rhythm must select exactly one registration")
    analysis.update(solutions[0])
    analysis["geometry_count"] = geometry_count
    return analysis


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve only from visible components, key, ink counts, and registration."""
    return _analyze(input_grid)["output"].copy()


def _partition(rng, side, kind):
    split = int(rng.integers(3, side-2))
    upper, lower = int(rng.integers(3, side-1)), int(rng.integers(2, side-1))
    owner = np.empty((side, side), dtype=np.int8)
    for r in range(side):
        if kind and r not in (0, split):
            if r < split:
                upper = int(np.clip(upper + rng.integers(-1, 2), 2, side-2))
            else:
                lower = int(np.clip(lower + rng.integers(-1, 2), 2, side-2))
        cut = upper if r < split else lower
        left = 0 if r < split else 2
        owner[r, :cut], owner[r, cut:] = left, left+1
    if kind == 2:
        # Local seam transfers produce tabs/arms without disconnected pieces.
        for _ in range(side * 3):
            r, c = int(rng.integers(1, side-1)), int(rng.integers(1, side-1))
            old = int(owner[r, c])
            dr, dc = _N4[int(rng.integers(4))]
            new = int(owner[r+dr, c+dc])
            if new == old:
                continue
            owner[r, c] = new
            if len(_components(owner == old)) != 1 or len(_components(owner == new)) != 1:
                owner[r, c] = old
    # Rotate cut geometry independently of the eventual rhythm axis.
    owner = np.rot90(owner, int(rng.integers(4))).copy()
    corner_values = [int(owner[r, c]) for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    relabel = np.asarray([corner_values.index(i) for i in range(4)], dtype=np.int8)
    return relabel[owner]


def _key_locations(owner):
    return [(r, c) for r in range(len(owner)-1) for c in range(len(owner)-1) if np.all(owner[r:r+2, c:c+2] == 0)]


def _line_options(regions, forbidden, line, bounded, solid, dotted, split, target, cap):
    """One immutable geometry/role candidate list, in the original order."""
    side = len(regions)
    region, blocked = regions[line], forbidden[line]
    options = []
    for d in range(1, min(cap, (side-1)//2)+1):
        if (solid and d != 1) or (dotted and d < 2):
            continue
        for a in range(side-d):
            b = a+d
            if blocked[a] or blocked[b] or (split and region[a] == region[b]):
                continue
            phase = range(a % d, side, d)
            if len(phase) < 3:
                continue
            if not bounded:
                if dotted and not (a-d >= 0 and b+d < side):
                    continue
                if len({region[x] for x in phase}) >= 2:
                    options.append((line, a, b, None, d))
                continue
            for t in phase:
                if a <= t <= b or blocked[t] or region[t] == region[a] == region[b]:
                    continue
                if target is not None and region[t] != target:
                    continue
                if solid and (region[t] == region[a] or min(a, t) == 0 or max(b, t) == side-1 or min(abs(t-a), abs(t-b)) < 2*d):
                    continue
                options.append((line, a, b, t, d))
    return options


def _mark(rng, owner, key_location, axis, count, split_witness, training, target_corners=None, candidate_cache=None):
    side = len(owner)
    # This cache lives only for one owner/key/axis attempt. It contains no
    # sampled examples and is never consulted by the input-only solver.
    if candidate_cache is None:
        candidate_cache = {}
    if "context" not in candidate_cache:
        forbidden = np.zeros_like(owner, dtype=bool)
        kr, kc = key_location
        forbidden[kr:kr+2, kc:kc+2] = True
        regions = owner if axis == "H" else owner.T
        forbidden = forbidden if axis == "H" else forbidden.T
        candidate_cache["context"] = regions.tolist(), forbidden.tolist()
    regions, forbidden = candidate_cache["context"]
    rows = rng.permutation(side).tolist()
    flags = [True, True, False] + [bool(rng.integers(2)) for _ in range(count-3)]
    records = []
    for i, bounded in enumerate(flags):
        options = []
        role = (bounded, i == 0, i == 2, split_witness and i == 3,
                target_corners[i] if target_corners is not None and i < 2 else None,
                4 if training else 5)
        for line in rows:
            cache_key = (line, role)
            if cache_key not in candidate_cache:
                candidate_cache[cache_key] = _line_options(regions, forbidden, line, *role)
            options.extend(candidate_cache[cache_key])
        if not options:
            return None
        record = options[int(rng.integers(len(options)))]
        records.append(record)
        rows.remove(record[0])
    return records


def _scatter(rng, pieces, background):
    # Shelf groups are shuffled independently of output corners; both axes,
    # linear rows, 3+1 layouts and unequal margins are available when they fit.
    order = rng.permutation(4).tolist()
    layouts = []
    for transpose in (False, True):
        dims = [p.shape[::-1] if transpose else p.shape for p in pieces]
        for grouping in ((4,), (3, 1), (1, 3), (2, 2), (2, 1, 1), (1, 1, 2), (1, 1, 1, 1)):
            gap = int(rng.integers(1, 4))
            groups, start = [], 0
            for size in grouping:
                groups.append(order[start:start+size])
                start += size
            widths = [sum(dims[i][1] for i in group)+gap*(len(group)-1) for group in groups]
            heights = [max(dims[i][0] for i in group) for group in groups]
            h, w = sum(heights)+gap*(len(groups)-1), max(widths)
            if h > 28 or w > 28:
                continue
            offsets, row = [None]*4, 0
            for group, gh, gw in zip(groups, heights, widths):
                col = int(rng.integers(w-gw+1))
                for i in group:
                    rr = row+int(rng.integers(gh-dims[i][0]+1))
                    offsets[i] = (col, rr) if transpose else (rr, col)
                    col += dims[i][1]+gap
                row += gh+gap
            layouts.append((offsets, (w, h) if transpose else (h, w)))
    if not layouts:
        return None
    offsets, extent = layouts[int(rng.integers(len(layouts)))]
    margins = [(int(rng.integers(1, min(3, 30-n))), 1) for n in extent]
    shape = tuple(extent[j]+sum(margins[j]) for j in (0, 1))
    grid = np.full(shape, background, dtype=np.int8)
    for p, (r, c) in zip(pieces, offsets):
        r, c = r+margins[0][0], c+margins[1][0]
        rr, cc = np.nonzero(p >= 0)
        grid[rr+r, cc+c] = p[rr, cc]
    return grid


def _make_pair(rng, side, kind, axis, count, rotations, split_witness, training, target_corners=None, registration_witness=False, avoid_key=None):
    for _ in range(80):
        owner = _partition(rng, side, kind)
        locations = _key_locations(owner)
        if not locations:
            continue
        key_location = locations[int(rng.integers(len(locations)))]
        # Eight disjoint roles, permuted afresh per example, including black.
        palette = rng.choice(10, 8, replace=False).astype(np.int8)
        bg, carrier, ref, terminal = map(int, palette[:4])
        key = palette[4:].copy()
        if avoid_key is not None and np.array_equal(key, avoid_key):
            key = np.roll(key, 1)
        candidate_cache = {}
        for _ in range(30):
            signals = _mark(rng, owner, key_location, axis, count, split_witness, training, target_corners, candidate_cache)
            if signals is None:
                continue
            assembled = np.full((side, side), carrier, dtype=np.int8)
            r, c = key_location
            assembled[r:r+2, c:c+2] = key.reshape(2, 2)
            view = assembled if axis == "H" else assembled.T
            for line, a, b, t, _ in signals:
                view[line, [a, b]] = ref
                if t is not None:
                    view[line, t] = terminal
            pieces = [_crop_piece(assembled, np.argwhere(owner == corner)) for corner in range(4)]
            if any(np.count_nonzero(p == carrier) <= np.count_nonzero((p >= 0) & (p != carrier)) for p in pieces):
                continue
            analysis = {"pieces": pieces, "side": side, "key": key, "bg": bg, "carrier": carrier, "ref": ref, "terminal": terminal}
            registered, geometry_count = _registered(analysis)
            if len(registered) != 1 or not np.array_equal(registered[0]["owner"], owner) or registered[0]["axis"] != axis:
                continue
            if registration_witness and geometry_count <= 1:
                continue
            expected = _paint(side, owner, signals, axis, carrier, ref, key)
            oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
            grid = _scatter(rng, oriented, bg)
            if grid is not None:
                return {"input": grid, "output": expected}
    raise RuntimeError("bounded rhythm-locked construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not _integer(num_test) or not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("counts must be integers: 2..10 train, 1..4 test, train > test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    fixed_side = int(rng.integers(9, 14))
    first_axis = "H" if rng.integers(2) == 0 else "V"
    demonstrated_corners = rng.permutation(4).tolist()
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for i in range(number):
            training = split == "train"
            if regime == 0:
                side = int(rng.integers(9, 12) if training else rng.integers(12, 14))
            elif regime == 1:
                side = int(rng.integers(11, 14) if training else rng.integers(9, 12))
            elif regime == 2:
                side = fixed_side
            else:
                side = int(rng.integers(9, 14))
            axis = first_axis if i == 0 and training else ("V" if first_axis == "H" else "H") if i == 1 and training else ("H" if rng.integers(2) == 0 else "V")
            count = (3 if i == 0 else 4) if training and i < 2 else int(rng.integers(3 if training else 4, 6 if training else 7))
            kind = 0 if training and i == 0 else int(rng.integers(1, 3)) if training and i == 1 else int(rng.integers(3))
            rotations = [0, 1, 2, 3] if training and i == 0 else [0]+rng.integers(0, 4, 3).tolist()
            for _ in range(10):
                targets = demonstrated_corners[2*i:2*i+2] if training and i < 2 else None
                avoid_key = _parse(puzzle["train"][0]["input"])["key"] if training and i == 1 else None
                pair = _make_pair(rng, side, kind, axis, count, rotations, training and i == 1, training, targets,
                                  registration_witness=training and i == 0, avoid_key=avoid_key)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(analysis):
    a = analysis
    side, axis, signals = a["side"], a["axis"], a["signals"]
    owner = a["owner"] if axis == "H" else a["owner"].T
    bounded = [s for s in signals if s[3] is not None]
    return {"side": side, "axis": axis, "lines": len(signals), "bounded": len(bounded),
            "periods": tuple(s[4] for s in signals), "geometry_count": a["geometry_count"],
            "split_references": sum(owner[line, x] != owner[line, y] for line, x, y, _, _ in signals),
            "stop": any(any(x < min(p, t) or x > max(q, t) for x in range(p % d, side, d)) for _, p, q, t, d in bounded),
            "both_sides": any(t is None and p-d >= 0 and q+d < side for _, p, q, t, d in signals),
            "terminal_directions": tuple(sorted({-1 if t < p else 1 for _, p, _, t, _ in bounded})),
            "terminal_corners": tuple(sorted({int(owner[line, t]) for line, _, _, t, _ in bounded})),
            "areas": tuple(int(np.count_nonzero(a["owner"] == i)) for i in range(4)),
            "irregular": sum(np.count_nonzero(p >= 0) != p.size for p in a["pieces"]),
            "rotations": a["rotations"], "bg": a["bg"], "carrier": a["carrier"]}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen, train_features, train_keys = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            a = _analyze(grid)
            _require(np.array_equal(a["output"], output), "wrong registered completion")
            _require(np.any(output == a["carrier"]) and np.any(output == a["ref"]), "boundary and blank witnesses required")
            f = _features(a)
            _require(f["stop"] and f["both_sides"], "missing stopping/extrapolation evidence")
            # Counterfactuals check the inherited mechanisms on actual pixels.
            for mode in ("solid", "unbounded", "source_key"):
                wrong = _paint(a["side"], a["owner"], a["signals"], a["axis"], a["carrier"], a["ref"], a["key"], mode)
                _require(not np.array_equal(wrong, output), "A-omission shortcut survived")
            if split == "train":
                train_features.append(f)
                train_keys.add(tuple(map(int, a["key"])))
    _require({f["axis"] for f in train_features} == {"H", "V"}, "training must teach both axes")
    _require(any(f["geometry_count"] > 1 for f in train_features), "training must expose phase-based registration")
    _require(any(f["split_references"] for f in train_features), "training must expose cross-seam reference pairs")
    _require(any(1 in f["periods"] for f in train_features) and any(max(f["periods"]) > 1 for f in train_features), "spacing witnesses missing")
    _require(len(train_keys) > 1, "key must vary in training")
    _require(set().union(*(set(f["terminal_corners"]) for f in train_features)) == {0, 1, 2, 3}, "all four endpoint addresses need training support")


def self_test():
    """42 streaming seeds, boundary counts, metamorphic and negative checks."""
    shapes, cuts, axes, periods, counts, backgrounds, endpoints, rotations = (set() for _ in range(8))
    signatures = set()
    multi_geometry = split_refs = irregular = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                signature.update(pair["input"].tobytes())
                a = _analyze(pair["input"])
                f = _features(a)
                shapes.add(pair["input"].shape)
                cuts.add(a["owner"].tobytes())
                axes.add(f["axis"])
                periods.update(f["periods"])
                counts.add(f["lines"])
                backgrounds.add(f["bg"])
                endpoints.update(f["terminal_corners"])
                rotations.update(f["rotations"])
                multi_geometry += f["geometry_count"] > 1
                split_refs += f["split_references"] > 0
                irregular += f["irregular"] > 0
        signatures.add(signature.hexdigest())
    _require(len(signatures) == 42 and len(shapes) >= 40 and len(cuts) >= 100, "structural diversity collapsed")
    _require(axes == {"H", "V"} and {1, 2, 3, 4, 5} <= periods and {3, 4, 5, 6} <= counts, "rhythm diversity missing")
    _require(len(backgrounds) == 10 and endpoints == {0, 1, 2, 3} and rotations == {0, 1, 2, 3}, "roles or rotations missing")
    _require(multi_geometry >= 42 and split_refs >= 42 and irregular >= 80, "family interactions missing")
    # The host exhaustively exercises all 30 legal counts before self_test.
    # Retain boundary regressions here without duplicating that entire sweep.
    for tr, te in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000+tr*10+te, tr, te)
        _require(len(p["train"]) == tr and len(p["test"]) == te, "explicit counts ignored")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(1, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    q = generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in ("input", "output")), "nondeterministic seed")
    pair = p["train"][0]
    permutation = np.array([7, 0, 9, 2, 8, 1, 6, 4, 3, 5], dtype=np.int8)
    _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]), "palette covariance failed")
    # Changing a visibly addressed legend entry must change its completed line.
    a = _analyze(pair["input"])
    grid = pair["input"].copy()
    corner = _features(a)["terminal_corners"][0]
    old = int(a["key"][corner])
    spare = next(c for c in range(10) if c not in grid)
    grid[grid == old] = spare
    _require(not np.array_equal(solve(grid), pair["output"]), "B key has no causal effect")
    damaged = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
