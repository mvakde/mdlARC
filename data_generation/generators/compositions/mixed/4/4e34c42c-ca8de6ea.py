"""Register transparent point fragments, then interleave their two rings.

No parent generator, file, seed, or example is consulted by solve().
The thin frames delimit fragments; only their interior colored points register.
"""
from collections import Counter
import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 918397025, 855520616, 923159644, 423010328, 223416943,
              790448087, 895133926, 75412416, 359635037, 347836813, 882543611,
              288389554, 594057267, 215947692, 547205058, 496650138, 815379905,
              611538782, 796095792, 34153649, 585571440, 470111429, 813193166,
              480279536, 618821537, 327228904, 167173246, 601818978, 920467543,
              773794770, 472560580, 522258581, 182673612, 780057461, 720294452,
              109877477, 465926939, 510274169, 125113972, 782552057, 676064811)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid cells must be integers in 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return {(r-r0, c-c0): v for (r, c), v in cells.items()}


def _components(points):
    remaining = set(points)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _decode(input_grid):
    """Recover sheets from their outlines, not from generation coordinates."""
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    _require(counts.max()*2 > grid.size, "background must occupy over half the input")
    background = int(colors[counts.argmax()])
    candidates = []
    for color in colors:
        color = int(color)
        if color == background:
            continue
        groups = _components(zip(*np.nonzero(grid == color)))
        if not 2 <= len(groups) <= 4:
            continue
        boxes = []
        for group in groups:
            r0, r1 = min(r for r, c in group), max(r for r, c in group)
            c0, c1 = min(c for r, c in group), max(c for r, c in group)
            if r1-r0 < 2 or c1-c0 < 2:
                break
            border = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                      if r in (r0, r1) or c in (c0, c1)}
            if group != border:
                break
            boxes.append((r0, c0, r1, c1))
        else:
            parts, owned = [], set()
            for r0, c0, r1, c1 in boxes:
                interior = {(r, c) for r in range(r0+1, r1) for c in range(c0+1, c1)}
                if owned & interior:
                    break
                owned.update(interior)
                part = {(r-r0-1, c-c0-1): int(grid[r, c]) for r, c in interior
                        if grid[r, c] != background}
                if not 3 <= len(part) <= 7 or color in part.values():
                    break
                parts.append(_normalize(part))
            else:
                data = set(zip(*np.nonzero((grid != background) & (grid != color))))
                if data <= owned:
                    candidates.append((background, color, parts))
    _require(len(candidates) == 1, "need one unambiguous outline color and 2..4 sheets")
    return candidates[0]


def _matches(a, b):
    """All exact >=3-point, >=2-color translations; no greedy overlap score."""
    votes = Counter((r-u, c-v) for (r, c), x in a.items()
                    for (u, v), y in b.items() if x == y)
    result = []
    for (dr, dc), count in sorted(votes.items()):
        if count < 3:
            continue
        overlap = [((r, c), (r-dr, c-dc)) for r, c in a if (r-dr, c-dc) in b]
        if (len(overlap) >= 3 and len({a[p] for p, q in overlap}) >= 2
                and all(a[p] == b[q] for p, q in overlap)):
            result.append((dr, dc))
    return result


def _register(parts):
    """Unique connected translation solution, before imposing ring geometry."""
    n = len(parts)
    adjacency = [[] for _ in parts]
    for i in range(n):
        for j in range(i+1, n):
            deltas = _matches(parts[i], parts[j])
            if deltas:
                adjacency[i].append((j, deltas))
                adjacency[j].append((i, [(-r, -c) for r, c in deltas]))
    offsets, solutions = {0: (0, 0)}, []
    visits = 0

    def search():
        nonlocal visits
        visits += 1
        _require(visits <= 2000, "registration search exceeded family bound")
        if len(solutions) > 1:
            return
        if len(offsets) == n:
            union, placed = {}, []
            for i, part in enumerate(parts):
                dr, dc = offsets[i]
                layer = {(r+dr, c+dc): v for (r, c), v in part.items()}
                for p, v in layer.items():
                    if p in union and union[p] != v:
                        return
                    union[p] = v
                placed.append(layer)
            solutions.append((_normalize(union), tuple(offsets[i] for i in range(n)), placed))
            return
        options = []
        for j in range(n):
            if j in offsets:
                continue
            domain = None
            for i, deltas in adjacency[j]:
                if i in offsets:
                    possible = {(offsets[i][0]-dr, offsets[i][1]-dc) for dr, dc in deltas}
                    domain = possible if domain is None else domain & possible
            if domain is not None:
                if not domain:
                    return
                options.append((len(domain), j, domain))
        if not options:
            return
        _, j, domain = min(options, key=lambda x: (x[0], x[1]))
        for position in sorted(domain):
            offsets[j] = position
            search()
            del offsets[j]

    search()
    _require(len(solutions) == 1, "disconnected, conflicting, or ambiguous registration")
    union, offsets, placed = solutions[0]
    exclusive = tuple(len(set(p)-set().union(*(set(q) for j, q in enumerate(placed) if j != i)))
                      for i, p in enumerate(placed))
    return union, offsets, exclusive


def _rings(cells):
    _require(len(cells) == 9, "assembled union must contain nine points")
    ys, xs = sorted({r for r, c in cells}), sorted({c for r, c in cells})
    _require(len(ys) == len(xs) == 5, "need five row and column levels")
    cy, cx = ys[2], xs[2]
    oy, ox, iy, ix = cy-ys[0], cx-xs[0], cy-ys[1], cx-xs[1]
    _require(0 < iy < oy and 0 < ix < ox and
             ys == [cy-oy, cy-iy, cy, cy+iy, cy+oy] and
             xs == [cx-ox, cx-ix, cx, cx+ix, cx+ox], "rings must be concentric")
    positions = ((cy-oy, cx-ox), (cy-oy, cx+ox), (cy+oy, cx+ox), (cy+oy, cx-ox),
                 (cy-iy, cx-ix), (cy-iy, cx+ix), (cy+iy, cx+ix), (cy+iy, cx-ix), (cy, cx))
    _require(set(positions) == set(cells), "foreground must be the two rings plus center")
    return tuple(cells[p] for p in positions), (oy, ox, iy, ix)


def _interleave(points):
    # points: clockwise outer UL,UR,LR,LL; clockwise inner UL,UR,LR,LL; center.
    a, b, c, d, e, f, g, h, z = points
    return np.array([[a, e, b], [h, z, f], [d, g, c]], dtype=np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read outlines, register repeated patterns, union, then cyclically merge."""
    _, _, parts = _decode(input_grid)
    union, _, _ = _register(parts)
    points, _ = _rings(union)
    return _interleave(points)


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and
                 isinstance(value, (int, np.integer)) and low <= int(value) <= high,
                 "illegal example count")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _coordinates(geometry):
    oy, ox, iy, ix = geometry
    return ((0, 0), (0, 2*ox), (2*oy, 2*ox), (2*oy, 0),
            (oy-iy, ox-ix), (oy-iy, ox+ix), (oy+iy, ox+ix), (oy+iy, ox-ix), (oy, ox))


def _sheet(part, background, frame, padding=(0, 0, 0, 0)):
    part = _normalize(part)
    top, bottom, left, right = padding
    h = max(r for r, c in part)+3+top+bottom
    w = max(c for r, c in part)+3+left+right
    grid = np.full((h, w), background, dtype=np.int8)
    grid[[0, -1], :] = frame
    grid[:, [0, -1]] = frame
    for (r, c), v in part.items():
        grid[r+1+top, c+1+left] = v
    return grid


def _pack(sheets, background, rng):
    """Variable shelf packing, permuted and optionally transposed as a layout.

    Transposing the PACKING does not transpose any sheet's contents.
    At most four 14x14 sheets always fit in a 29x29 two-by-two layout.
    """
    order = [sheets[int(i)] for i in rng.permutation(len(sheets))]
    n = len(order)
    layouts = []
    for columns in range(1, n+1):
        rows, positions, top = [], [], 0
        for start in range(0, n, columns):
            row = order[start:start+columns]
            height, left = max(a.shape[0] for a in row), 0
            for a in row:
                positions.append((top, left))
                left += a.shape[1]+1
            rows.append(left-1)
            top += height+1
        h, w = top-1, max(rows)
        if h <= 30 and w <= 30:
            layouts.append((h, w, positions))
    _require(bool(layouts), "sheets cannot fit")
    h, w, positions = layouts[int(rng.integers(len(layouts)))]
    total_ink = sum(int(np.count_nonzero(a != background)) for a in order)
    # Add margins until the background is a strict majority.
    hh, ww = h, w
    while hh*ww <= 2*total_ink:
        if hh < 30 and (hh <= ww or ww == 30):
            hh += 1
        else:
            _require(ww < 30, "cannot make background dominant")
            ww += 1
    hh += int(rng.integers(min(3, 30-hh)+1))
    ww += int(rng.integers(min(3, 30-ww)+1))
    top = int(rng.integers(hh-h+1))
    left = int(rng.integers(ww-w+1))
    canvas = np.full((hh, ww), background, dtype=np.int8)
    for a, (r, c) in zip(order, positions):
        canvas[top+r:top+r+a.shape[0], left+c:left+c+a.shape[1]] = a
    return canvas


def _make_pair(rng, geometry, n, distinct, separate_center=False):
    background, frame, *palette = [int(v) for v in rng.permutation(10)]
    if distinct:
        points = tuple(palette) + (palette[int(rng.integers(8))],)
    elif separate_center:
        # The first demonstration labels all eight perimeter transfers. This
        # next one gives the center its own color, so copying it separately is
        # observable despite the eight-color data budget imposed by outlines.
        size = int(rng.integers(2, 8))
        points = tuple(int(v) for v in rng.choice(palette[:size], 8)) + (palette[-1],)
    else:
        size = int(rng.integers(3, 9))
        points = tuple(int(v) for v in rng.choice(palette[:size], 9))
    cells = dict(zip(_coordinates(geometry), points))
    for _ in range(160):
        sizes = [6, 6] if n == 2 else [int(rng.integers(4, 8)) for _ in range(n)]
        subsets = [set(int(x) for x in rng.choice(9, size, replace=False)) for size in sizes]
        if len(set.union(*subsets)) != 9 or len({tuple(sorted(s)) for s in subsets}) != n:
            continue
        # No sheet supplies an intact ring. Both streams really need merging.
        if any(set(range(4)) <= s or set(range(4, 8)) <= s for s in subsets):
            continue
        raw = [{p: cells[p] for i, p in enumerate(cells) if i in subset} for subset in subsets]
        parts = [_normalize(p) for p in raw]
        try:
            union, offsets, exclusive = _register(parts)
        except ValueError:
            continue
        if union != cells or not any(exclusive):
            continue
        if distinct and len(set(offsets)) == 1:
            continue
        sheets = []
        for p in raw:
            normalized = _normalize(p)
            h = max(r for r, c in normalized)+3
            w = max(c for r, c in normalized)+3
            pads = [int(v) for v in rng.integers(0, 2, size=4)]
            # Individual padding is independent of registration coordinates.
            if h+sum(pads[:2]) > 14:
                pads[int(rng.integers(2))] = 0
            if w+sum(pads[2:]) > 14:
                pads[2+int(rng.integers(2))] = 0
            sheets.append(_sheet(p, background, frame, pads))
        inp = _pack(sheets, background, rng)
        return {"input": inp, "output": _interleave(points)}
    raise ValueError("could not sample uniquely registering fragment subsets")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    result, seen, train_geometry = {"train": [], "test": []}, set(), set()
    for index in range(nt+nq):
        training = index < nt
        for _ in range(80):
            if index == 0:
                outer = int(rng.integers(2, 4))
                geometry = (outer, outer, 1, 1)
                n = 2
            else:
                oy, ox = [int(v) for v in rng.integers(2, 5 if training else 6, size=2)]
                iy, ix = int(rng.integers(1, oy)), int(rng.integers(1, ox))
                geometry = (oy, ox, iy, ix)
                if index == 1 and oy == ox and iy == ix:
                    continue
                if not training and geometry in train_geometry:
                    continue
                n = 3 if index == 1 else (int(rng.integers(2, 5)) if training else 4)
            try:
                pair = _make_pair(rng, geometry, n, index == 0, index == 1)
            except ValueError:
                continue
            key = _key(pair["input"])
            if key in seen:
                continue
            seen.add(key)
            result["train" if training else "test"].append(pair)
            if training:
                train_geometry.add(geometry)
            break
        else:
            raise RuntimeError("bounded distinct-pair generation exhausted")
    return result


def _inspect(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    bg, frame, parts = _decode(inp)
    union, offsets, exclusive = _register(parts)
    points, geometry = _rings(union)
    _require(np.array_equal(out, _interleave(points)), "incorrect clockwise interleaving")
    _require(all(3 <= len(p) <= 7 for p in parts) and any(exclusive),
             "a fragment must supply information unavailable in the others")
    _require(set(out.flat) == set(union.values()) and frame not in out,
             "data palette must be preserved and outlines discarded")
    positions = _coordinates(geometry)
    # The placed layers use part-0 coordinates; normalize all using the union.
    placed = [{(r+offsets[i][0], c+offsets[i][1]) for r, c in part}
              for i, part in enumerate(parts)]
    r0 = min(r for layer in placed for r, c in layer)
    c0 = min(c for layer in placed for r, c in layer)
    layers = [{(r-r0, c-c0) for r, c in layer} for layer in placed]
    _require(all(not set(positions[:4]) <= layer and not set(positions[4:8]) <= layer
                 for layer in layers), "no individual sheet may supply a complete ring")
    # Explicit omission-B counterfactual: concatenate both clockwise streams.
    concat = np.full((3, 3), bg, dtype=np.int8)
    perimeter = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
    for p, v in zip(perimeter, points[:8]):
        concat[p] = v
    concat[1, 1] = points[8]
    unregistered, collision = {}, False
    for part in parts:
        for p, v in part.items():
            if p in unregistered and unregistered[p] != v:
                collision = True
            unregistered[p] = v
    try:
        naive = _interleave(_rings(unregistered)[0])
        a_necessary = collision or not np.array_equal(out, naive)
    except ValueError:
        a_necessary = True
    return {"bg": bg, "frame": frame, "n": len(parts), "geometry": geometry,
            "sizes": tuple(len(p) for p in parts), "colors": len(set(points)),
            "center_unique": points[8] not in points[:8],
            "exclusive": exclusive, "offsets": offsets,
            "a_necessary": a_necessary, "b_necessary": not np.array_equal(out, concat)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    # Distribution and demonstration requirements, separate from solve's rule.
    first, second = infos["train"][:2]
    _require(first["n"] == 2 and first["colors"] == 8 and first["b_necessary"],
             "first demonstration must disambiguate the eight perimeter transfers")
    _require(len(set(first["offsets"])) > 1 and first["a_necessary"],
             "first demonstration must require a nonzero registration offset")
    _require(first["geometry"][0] == first["geometry"][1] and
             first["geometry"][2] == first["geometry"][3], "first demonstration must have square rings")
    _require(second["n"] == 3 and (second["geometry"][0] != second["geometry"][1] or
             second["geometry"][2] != second["geometry"][3]), "second demonstration must extend geometry")
    _require(second["center_unique"], "second demonstration must identify the center separately")
    train_geometry = {x["geometry"] for x in infos["train"]}
    _require(all(x["n"] == 4 and x["geometry"] not in train_geometry for x in infos["test"]),
             "tests must combine four fragments with unseen geometry")


def self_test():
    """42 seeds, streamed statistics, metamorphic and negative checks; no files."""
    digests, shapes, geometries, sizes, backgrounds, frames, palettes, counts = (set() for _ in range(8))
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                info = _inspect(pair)
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                digest.update(repr(_key(pair["input"])).encode())
                shapes.add(pair["input"].shape)
                geometries.add(info["geometry"])
                sizes.update(info["sizes"])
                backgrounds.add(info["bg"])
                frames.add(info["frame"])
                palettes.add(info["colors"])
                counts.add(info["n"])
        digests.add(digest.digest())
        first = p["train"][0]
        # Relocating every sheet and globally renaming colors must not change
        # the correspondence. The complete intermediate union has nine cells.
        bg, frame, parts = _decode(first["input"])
        repacked = _pack([_sheet(x, bg, frame) for x in parts], bg, np.random.default_rng(seed+71))
        assert np.array_equal(solve(repacked), first["output"])
        lookup = np.array([3, 9, 1, 7, 0, 5, 8, 4, 6, 2], dtype=np.int8)
        assert np.array_equal(solve(lookup[repacked]), lookup[first["output"]])
        assert all(len(part) < 9 for part in parts)  # No individual sheet supplies the answer.
    assert len(digests) == 42 and len(shapes) >= 30 and len(geometries) >= 50
    assert counts == {2, 3, 4} and len(sizes) >= 4 and len(palettes) >= 4
    assert len(backgrounds) == len(frames) == 10
    # A hand-specified semantic fixture checks the directed inner transfers
    # and separate center, without constructing the answer via _interleave.
    positions = _coordinates((3, 4, 1, 2))
    values = (1, 2, 3, 4, 5, 6, 1, 2, 7)
    cells = dict(zip(positions, values))
    masks = ((0, 1, 4, 5, 7, 8), (2, 3, 4, 5, 6, 8))
    parts = [{positions[i]: cells[positions[i]] for i in mask} for mask in masks]
    fixture = _pack([_sheet(part, 0, 9) for part in parts], 0, np.random.default_rng(51))
    expected = np.array([[1, 5, 2], [2, 7, 6], [4, 1, 3]], dtype=np.int8)
    assert np.array_equal(solve(fixture), expected)
    changed = fixture.copy()
    changed[changed == 7] = 8
    expected[1, 1] = 8
    assert np.array_equal(solve(changed), expected)
    assert all(_key(a[k]) == _key(b[k]) for split in ("train", "test")
               for a, b in zip(generate_puzzle(17)[split], generate_puzzle(17)[split])
               for k in ("input", "output"))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(103, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    try:
        _register([{(0, 0): 1, (0, 1): 2, (0, 2): 1},
                   {(0, 0): 1, (0, 1): 2, (0, 2): 1, (0, 3): 2, (0, 4): 1}])
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous periodic signatures accepted")
    bad = generate_puzzle(0)
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"valid": True, "self_test_seeds": 42}, separators=(",", ":")))
