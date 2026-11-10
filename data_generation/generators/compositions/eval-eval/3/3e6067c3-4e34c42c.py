"""Itinerary-controlled translational stitching of two-station segments.

Only standard Python and NumPy are used. The solver reads pixels exclusively;
construction coordinates never enter decoding or validation.
"""
from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer 1..30 grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(a, bg):
    remaining = set(zip(*np.nonzero(a != bg)))
    groups = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, group = [first], []
        while stack:
            r, c = stack.pop()
            group.append((int(r), int(c)))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _square_points(top, s):
    return {(top[0]+r, top[1]+c) for r in range(s) for c in range(s)}


def _station(block, bg, carrier):
    s = len(block)
    label = int(block[s//2, s//2])
    if label in (bg, carrier):
        raise ValueError("station label uses a structural color")
    corners = [(0, 0), (0, s-1), (s-1, 0), (s-1, s-1)]
    if any(int(block[q]) in (bg, carrier) for q in corners):
        raise ValueError("invalid fingerprint corner")
    if all(int(block[q]) == label for q in corners):
        raise ValueError("a fingerprint needs an accent distinct from its label")
    widths = []
    for p in range(1, s-1):
        if (s-p) % 2:
            continue
        d = (s-p)//2
        expected = np.full((s, s), carrier, dtype=np.int8)
        expected[d:d+p, d:d+p] = label
        for q in corners:
            expected[q] = block[q]
        if np.array_equal(block, expected):
            widths.append(p)
    if len(widths) != 1:
        raise ValueError("unclear centered label patch")
    return tuple(int(v) for v in block.flat), label, widths[0]


def _read(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])):
        raise ValueError("background must surround the input")
    tokens, parts = [], []
    common = None
    bottoms = []
    for group in _components(a, bg):
        if len(group) == 1:
            tokens.append(group[0])
            continue
        rs, cs = zip(*group)
        r0, c0 = min(rs), min(cs)
        h, w = max(rs)-r0+1, max(cs)-c0+1
        s = min(h, w)
        if not 3 <= s <= 5 or max(h, w) <= 2*s:
            raise ValueError("component is not a two-station segment")
        horizontal = h == s
        endpoints = [(0, 0), (0, w-s) if horizontal else (h-s, 0)]
        local = {(r-r0, c-c0): int(a[r, c]) for r, c in group}
        block0 = a[r0:r0+s, c0:c0+s]
        carrier = int(block0[0, 1])
        if carrier == bg:
            raise ValueError("station carrier cannot be background")
        ends = []
        for r, c in endpoints:
            block = a[r0+r:r0+r+s, c0+c:c0+c+s]
            glyph, label, p = _station(block, bg, carrier)
            ends.append((r, c, glyph, label))
            signature = (s, p, carrier)
            if common is not None and common != signature:
                raise ValueError("segment geometry/roles must be common within a pair")
            common = signature
        p = common[1]
        d = (s-p)//2
        band = ([(r, c) for r in range(d, d+p) for c in range(s, w-s)]
                if horizontal else
                [(r, c) for r in range(s, h-s) for c in range(d, d+p)])
        expected_points = _square_points(endpoints[0], s) | _square_points(endpoints[1], s) | set(band)
        if set(local) != expected_points or any(local[q] != carrier for q in band):
            raise ValueError("extra material or an incorrect gap band")
        parts.append(dict(cells=local, ends=ends, band=band, s=s, p=p))
        bottoms.append(max(rs))
    if not 2 <= len(parts) <= 8 or len(tokens) != len(parts)+1:
        raise ValueError("need E segments and E+1 key tokens")
    tokens.sort(key=lambda q: q[1])
    if (len({q[0] for q in tokens}) != 1
            or any(b[1]-a0[1] != 2 for a0, b in zip(tokens, tokens[1:]))
            or tokens[0][0] < max(bottoms)+2):
        raise ValueError("key must be detached below the segments")
    key = [int(a[q]) for q in tokens]
    if any(v in (bg, common[2]) for v in key):
        raise ValueError("invalid key color")
    # Exact duplicate segments would make piece assignments interchangeable.
    signatures = [tuple(sorted(part["cells"].items())) for part in parts]
    if len(set(signatures)) != len(signatures):
        raise ValueError("duplicate interchangeable segments")
    return dict(grid=a, bg=bg, carrier=common[2], s=common[0], p=common[1],
                parts=parts, key=key, tokens=tokens)


def _render(cells, bg):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    h = max(r for r, c in cells)-r0+1
    w = max(c for r, c in cells)-c0+1
    a = np.full((h, w), bg, dtype=np.int8)
    for (r, c), color in cells.items():
        a[r-r0, c-c0] = color
    return a


def _search(info, *, centers_only=False, use_key=True, limit=2):
    """Search assignments, not a compactness or maximal-overlap heuristic.

    centers_only is a diagnostic ablation: it retains the previous station at
    an unequal-pattern join. It is never used to choose the actual answer.
    use_key=False is another diagnostic: preserve exact station joins and
    all-piece consumption, but permit any sequence of center labels.
    """
    parts, key, s = info["parts"], info["key"], info["s"]
    found = []

    def visit(used, current, previous_glyph, union, paint, route, stations):
        t = len(route)
        if len(found) >= limit:
            return
        if t == len(parts):
            result = dict(union)
            for points, color in paint:
                for q in points:
                    result[q] = color
            found.append(dict(output=_render(result, info["bg"]), union=union,
                              paint=paint, route=route, stations=stations))
            return
        for i, part in enumerate(parts):
            if i in used:
                continue
            for orientation in (0, 1):
                start, end = part["ends"][orientation], part["ends"][1-orientation]
                if use_key and (start[3], end[3]) != (key[t], key[t+1]):
                    continue
                if t and not centers_only and start[2] != previous_glyph:
                    continue
                dr, dc = current[0]-start[0], current[1]-start[1]
                moved = {(r+dr, c+dc): v for (r, c), v in part["cells"].items()}
                shared = _square_points(current, s) if t else set()
                if set(moved) & set(union) != shared:
                    continue
                if not centers_only and any(moved[q] != union[q] for q in shared):
                    continue
                combined = dict(union)
                combined.update({q: v for q, v in moved.items() if q not in shared})
                ys, xs = zip(*combined)
                if max(ys)-min(ys) >= 30 or max(xs)-min(xs) >= 30:
                    continue
                target = (end[0]+dr, end[1]+dc)
                points = [(r+dr, c+dc) for r, c in part["band"]]
                visit(used | {i}, target, end[2], combined,
                      paint+[(points, start[3])], route+[(i, orientation)],
                      (stations if t else [(current, start[2], start[3])])
                      + [(target, end[2], end[3])])

    visit(set(), (0, 0), None, {}, [], [], [])
    return found


def _itinerary_changes_assembly(info):
    """Detect a different exact-fingerprint assembly without the color key.

    Compare the neutral union, so a reversal that changes only source colors
    cannot satisfy this check. For four-piece demonstrations the 100-result
    bound exceeds the complete set of trails in the sampled fingerprint graph.
    """
    neutral = _render(info["union"], info["bg"])
    return any(not np.array_equal(_render(v["union"], info["bg"]), neutral)
               for v in _search(info, use_key=False, limit=100))


def _decode(input_grid):
    info = _read(input_grid)
    solutions = _search(info)
    if len(solutions) != 1:
        raise ValueError("itinerary must have one complete noncolliding reconstruction")
    info.update(solutions[0])
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the input pixels."""
    return _decode(input_grid)["output"].copy()


def _glyph(rng, s, p, label, carrier, palette):
    a = np.full((s, s), carrier, dtype=np.int8)
    d = (s-p)//2
    a[d:d+p, d:d+p] = label
    corners = [(0, 0), (0, s-1), (s-1, 0), (s-1, s-1)]
    values = [int(v) for v in rng.choice(palette, 4)]
    if all(v == label for v in values):
        values[int(rng.integers(4))] = int(rng.choice([v for v in palette if v != label]))
    for q, v in zip(corners, values):
        a[q] = v
    return a


def _segment(a, b, step, carrier, p):
    """Create one local segment; its first station begins at (0,0)."""
    s = len(a)
    dy, dx = step
    cells = {}
    for block, (y, x) in ((a, (0, 0)), (b, step)):
        for r in range(s):
            for c in range(s):
                cells[y+r, x+c] = int(block[r, c])
    d = (s-p)//2
    if dy == 0:
        band = [(r, c) for r in range(d, d+p)
                for c in range(min(0, dx)+s, max(0, dx))]
    else:
        band = [(r, c) for r in range(min(0, dy)+s, max(0, dy))
                for c in range(d, d+p)]
    cells.update({q: carrier for q in band})
    return cells, band


def _normalize(cells):
    y = min(r for r, c in cells)
    x = min(c for r, c in cells)
    return {(r-y, c-x): v for (r, c), v in cells.items()}


def _geometry(rng, count, s, p):
    # Walk construction uses plain occupied cells, independent of fingerprint
    # decoding. Nonadjacent visits cannot overlap or cross earlier material.
    block = np.ones((s, s), dtype=np.int8)
    union = {(r, c): 1 for r in range(s) for c in range(s)}
    positions, steps = [(0, 0)], []
    for _ in range(count):
        candidates = [(dy*(s+gap), dx*(s+gap))
                      for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0))
                      for gap in range(1, 6)]
        accepted = False
        for k in rng.permutation(len(candidates)):
            step = candidates[int(k)]
            cells, _ = _segment(block, block, step, 1, p)
            y, x = positions[-1]
            moved = {(r+y, c+x): v for (r, c), v in cells.items()}
            if set(moved) & set(union) != _square_points((y, x), s):
                continue
            new = dict(union)
            new.update(moved)
            ys, xs = zip(*new)
            if max(ys)-min(ys) >= 28 or max(xs)-min(xs) >= 28:
                continue
            union = new
            positions.append((y+step[0], x+step[1]))
            steps.append(step)
            accepted = True
            break
        if not accepted:
            return None
    if len({dy == 0 for dy, dx in steps}) < 2:
        return None
    return positions, steps


def _pack(parts, key, bg, rng):
    """Packing has no relationship to solved order; no fragment is rotated."""
    parts = [_normalize(cells) for cells in parts]
    shapes = [(max(r for r, c in cells)+1, max(c for r, c in cells)+1) for cells in parts]
    minh = max(h for h, w in shapes)+4
    minw = max(max(w for h, w in shapes)+2, 2*len(key)+1)
    if minh > 30 or minw > 30:
        return None
    canvases = [(int(rng.integers(minh, 31)), int(rng.integers(minw, 31))) for _ in range(8)]
    canvases += [(30, 30)]*8
    for h, w in canvases:
        boxes, placements = [], {}
        order = sorted(range(len(parts)), key=lambda i: -(shapes[i][0]*shapes[i][1]+float(rng.random())*20))
        for i in order:
            ph, pw = shapes[i]
            feasible = []
            for r in range(1, h-ph-2):
                for c in range(1, w-pw):
                    if all(r+ph < y or y+hh < r or c+pw < x or x+ww < c
                           for y, x, hh, ww in boxes):
                        feasible.append((r, c))
            if not feasible:
                break
            r, c = feasible[int(rng.integers(len(feasible)))]
            boxes.append((r, c, ph, pw))
            placements[i] = (r, c)
        if len(placements) != len(parts):
            continue
        bottom = max(r+ph for r, c, ph, pw in boxes)
        ky = int(rng.integers(bottom+1, h-1))
        kx = int(rng.integers(1, w-(2*len(key)-1)))
        a = np.full((h, w), bg, dtype=np.int8)
        for i, cells in enumerate(parts):
            r0, c0 = placements[i]
            for (r, c), v in cells.items():
                a[r+r0, c+c0] = v
        a[ky, kx:kx+2*len(key):2] = key
        return a
    return None


def _make_pair(rng, roles, count, s, p, require_ablation):
    bg, carrier = roles
    available = [v for v in range(10) if v not in roles]
    for _ in range(160):
        palette = [int(v) for v in rng.choice(available, int(rng.integers(3, 8)), replace=False)]
        # Sample occurrence structure independently of the geometric walk.
        # Repeated visits may be internal or terminal, adjacent-cycle lengths
        # may differ, and long itineraries may repeat several identities.
        ntypes = int(rng.integers(max(4, count-2), count+1))
        occurrences = list(range(ntypes))
        occurrences += [int(rng.integers(ntypes)) for _ in range(count+1-ntypes)]
        types = [int(v) for v in rng.permutation(occurrences)]
        if any(a == b for a, b in zip(types, types[1:])):
            continue
        nlabels = int(rng.integers(3, min(ntypes-1, len(palette))+1))
        label_values = palette[:nlabels] + [int(rng.choice(palette[:nlabels])) for _ in range(ntypes-nlabels)]
        labels = dict(enumerate(int(v) for v in rng.permutation(label_values)))
        glyphs = {}
        for t in sorted(set(types)):
            for _ in range(40):
                block = _glyph(rng, s, p, labels[t], carrier, palette)
                if not any(np.array_equal(block, old) for old in glyphs.values()):
                    glyphs[t] = block
                    break
            else:
                break
        if len(glyphs) != len(set(types)):
            continue
        geometry = _geometry(rng, count, s, p)
        if geometry is None:
            continue
        positions, steps = geometry
        parts, intended, paint = [], {}, []
        for i, step in enumerate(steps):
            cells, band = _segment(glyphs[types[i]], glyphs[types[i+1]], step, carrier, p)
            parts.append(_normalize(cells))
            y, x = positions[i]
            intended.update({(r+y, c+x): v for (r, c), v in cells.items()})
            paint.append(([(r+y, c+x) for r, c in band], labels[types[i]]))
        for band, color in paint:
            for q in band:
                intended[q] = color
        expected = _render(intended, bg)
        key = [labels[t] for t in types]
        inp = _pack(parts, key, bg, rng)
        if inp is None:
            continue
        try:
            info = _decode(inp)
        except ValueError:
            continue
        if not np.array_equal(info["output"], expected):
            continue
        if require_ablation:
            alternatives = _search(info, centers_only=True, limit=40)
            if not any(not np.array_equal(v["output"], expected) for v in alternatives):
                continue
            if not _itinerary_changes_assembly(info):
                continue
        return {"input": inp, "output": expected}
    raise RuntimeError("bounded construction could not find a unique stitchable itinerary")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    fixed_roles = tuple(int(v) for v in rng.choice(10, 2, replace=False))
    stable = bool(rng.integers(2))
    widths = [int(v) for v in rng.permutation([1, 3])]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n in (("train", nt), ("test", nq)):
        for i in range(n):
            for _ in range(12):
                roles = fixed_roles if stable else tuple(int(v) for v in rng.choice(10, 2, replace=False))
                if split == "train":
                    count = 4 if i < 2 else int(rng.integers(4, 6))
                    s = 5 if i < 2 else int(rng.integers(3, 6))
                    p = widths[i] if i < 2 else int(rng.choice([v for v in range(1, s-1) if (s-v) % 2 == 0]))
                else:
                    count = int(rng.integers(6, 8))
                    s = int(rng.choice([3, 3, 4, 5]))
                    p = int(rng.choice([v for v in range(1, s-1) if (s-v) % 2 == 0]))
                pair = _make_pair(rng, roles, count, s, p, split == "train" and i == 0)
                signature = _grid_key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce distinct input grids")
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalized_structure(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(v), len(roles)) for v in a.flat)


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _decode(inp)
    assert np.array_equal(info["output"], out), "incorrect station union/source bands/crop"
    assert not np.array_equal(inp, out)
    assert len(info["route"]) == len(info["parts"])
    assert len({i for i, o in info["route"]}) == len(info["parts"])
    # Every visit is a distinct geometric station; only adjacent copies merge.
    assert len({q for q, glyph, label in info["stations"]}) == len(info["key"])
    glyphs = [glyph for q, glyph, label in info["stations"]]
    assert len(set(glyphs)) < len(glyphs), "nonadjacent repeat missing"
    center = (info["s"]//2)*info["s"] + info["s"]//2
    assert any(a != b and a[center] == b[center]
               for a in glyphs for b in glyphs), "same-label unequal fingerprints missing"
    assert any(a != b for a, b in zip(info["key"], info["key"][1:]))
    for points, color in info["paint"]:
        assert len(points) >= info["p"] and all(info["union"][q] == info["carrier"] for q in points)
    # Reversing the itinerary changes source coloring on these same pieces.
    # A separate first-demonstration check below establishes an assembly change.
    changed = inp.copy()
    for q, color in zip(info["tokens"], info["key"][::-1]):
        changed[q] = color
    reversed_output = solve(changed)
    assert not np.array_equal(reversed_output, out), "key direction has no effect"
    station_cells = set().union(*(_square_points(q, info["s"]) for q, glyph, label in info["stations"]))
    band_cells = set(q for points, color in info["paint"] for q in points)
    assert not station_cells & band_cells
    assert len(info["union"]) == len(info["key"])*info["s"]**2 + len(band_cells)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            signature = _grid_key(pair["input"])
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            infos[split].append(info)
    train, test = infos["train"], infos["test"]
    assert train[0]["s"] == train[1]["s"] == 5
    assert {train[0]["p"], train[1]["p"]} == {1, 3}
    assert max(len(a["parts"]) for a in train) < min(len(a["parts"]) for a in test)
    for a in train[:2]:
        assert len({q[0] for q, g, l in a["stations"]}) > 1
        assert len({q[1] for q, g, l in a["stations"]}) > 1
    alternatives = _search(train[0], centers_only=True, limit=40)
    assert any(not np.array_equal(v["output"], train[0]["output"]) for v in alternatives), "B omission was not distinguished"
    assert _itinerary_changes_assembly(train[0]), "A omission did not change the assembly"


def self_test():
    """Bounded semantic, metamorphic, and structural sweep across 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos, shapes, structures, roles, gaps, fingerprints, layouts = [], set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _decode(pair["input"])
                infos.append(a)
                shapes.add((pair["input"].shape, pair["output"].shape))
                structures.add(_normalized_structure(pair["output"]))
                roles.add((a["bg"], a["carrier"]))
                gaps.update(len(points)//a["p"] for points, color in a["paint"])
                fingerprints.update(glyph for q, glyph, label in a["stations"])
                first = a["stations"][0][0]
                layouts.add(tuple((q[0]-first[0], q[1]-first[1]) for q, g, l in a["stations"]))
    assert len(shapes) >= 60
    assert len(structures) >= 150
    assert len(roles) >= 30
    assert len(fingerprints) >= 200
    assert len(layouts) >= 100
    assert gaps == {1, 2, 3, 4, 5}
    assert {a["s"] for a in infos} == {3, 4, 5}
    assert {a["p"] for a in infos} == {1, 2, 3}
    assert {len(a["parts"]) for a in infos} == {4, 5, 6, 7}
    assert any(max(Counter(g for q, g, l in a["stations"]).values()) >= 3 for a in infos)
    assert all(len({_normalized_structure(p["output"]) for p in task["train"]}) >= 2 for task in puzzles)
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            task = generate_puzzle(1000+10*nt+nq, nt, nq)
            assert len(task["train"]) == nt and len(task["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    pair = puzzles[0]["train"][0]
    info = _decode(pair["input"])
    mapping = np.array([7, 2, 9, 5, 0, 4, 8, 1, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    rng = np.random.default_rng(98765)
    repacked = _pack([p["cells"] for p in info["parts"]][::-1], info["key"], info["bg"], rng)
    assert repacked is not None and np.array_equal(solve(repacked), pair["output"])
    # An unambiguous two-piece family member verifies input-only support beyond
    # the normal recurring-cycle construction (pair validation is narrower).
    glyphs = [_glyph(rng, 3, 1, label, 1, [2, 3, 4]) for label in (2, 3, 4)]
    small_parts = [_segment(glyphs[0], glyphs[1], (0, 4), 1, 1)[0],
                   _segment(glyphs[1], glyphs[2], (5, 0), 1, 1)[0]]
    small = _pack(small_parts, [2, 3, 4], 0, rng)
    small_info = _decode(small)
    assert len(small_info["stations"]) == 3
    assert small_info["output"].shape == (8, 7)
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("incorrect output accepted")


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
