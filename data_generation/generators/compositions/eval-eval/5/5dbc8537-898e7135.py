"""Restore a miniature anchored mosaic using enlarged patterned rectangles.

No input conventions depend on a seed, split, generation state or examples.
All diagnostics are in memory. No parent generator is imported or executed.
"""

import hashlib
import math
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NODE_LIMIT = 20000


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return tuple(a.shape), a.astype(np.uint8).tobytes()


def _variants(a, reflections=True):
    found = {}
    for base in (a, np.fliplr(a)) if reflections else (a,):
        for q in range(4):
            b = np.rot90(base, q).copy()
            found.setdefault(_key(b), b)
    return list(found.values())


def _canonical(a):
    return min(_key(b) for b in _variants(a))


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        todo, component = [start], []
        while todo:
            r, c = todo.pop()
            component.append((r, c))
            for dr, dc in _N4:
                p = r + dr, c + dc
                if p in cells:
                    cells.remove(p)
                    todo.append(p)
        result.append(component)
    return result


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _covers(mask, clues, background, tiles, node_limit=_NODE_LIMIT):
    """Enumerate up to two distinct colored answers, with finite D4 inventory.

    A budget exhaustion is an unresolved search, never a uniqueness proof.
    Dead-state memoization stores only states with no completion.
    """
    coords = np.argwhere(mask)
    n = len(coords)
    indices = np.full(mask.shape, -1, dtype=int)
    indices[mask] = np.arange(n)
    groups, lookup = [], {}
    for tile in tiles:
        key = _canonical(tile)
        if key not in lookup:
            lookup[key] = len(groups)
            groups.append({"base": tile, "count": 0})
        groups[lookup[key]]["count"] += 1
    placements, by_cell = [], [[] for _ in range(n)]
    for gi, group in enumerate(groups):
        for tile in _variants(group["base"]):
            h, w = tile.shape
            for r in range(mask.shape[0] - h + 1):
                for c in range(mask.shape[1] - w + 1):
                    if not np.all(mask[r:r+h, c:c+w]):
                        continue
                    samples = clues[r:r+h, c:c+w]
                    if np.any((samples != background) & (samples != tile)):
                        continue
                    ii = tuple(int(i) for i in indices[r:r+h, c:c+w].flat)
                    bits = sum(1 << i for i in ii)
                    pi = len(placements)
                    placements.append((gi, bits, ii, tile, r, c))
                    for i in ii:
                        by_cell[i].append(pi)
    remaining = [g["count"] for g in groups]
    rendered = np.zeros(n, dtype=np.uint8)
    full, nodes, done = (1 << n) - 1, 0, True
    outcomes, dead, path = {}, set(), []

    def visit(covered):
        nonlocal nodes, done
        if not done or len(outcomes) >= 2:
            return False
        nodes += 1
        if nodes > node_limit:
            done = False
            return False
        if covered == full:
            if any(remaining):
                return False
            outcomes.setdefault(rendered.tobytes(), tuple(path))
            return True
        state = covered, tuple(remaining)
        if state in dead:
            return False
        best = None
        for i in range(n):
            if covered & (1 << i):
                continue
            options = [pi for pi in by_cell[i]
                       if remaining[placements[pi][0]] and not covered & placements[pi][1]]
            if not options:
                dead.add(state)
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        successful = False
        for pi in best:
            gi, bits, ii, tile, r, c = placements[pi]
            remaining[gi] -= 1
            rendered[list(ii)] = tile.ravel()
            path.append(pi)
            successful |= visit(covered | bits)
            path.pop()
            remaining[gi] += 1
            if not done or len(outcomes) >= 2:
                return successful
        if not successful:
            dead.add(state)
        return successful

    visit(0)
    return outcomes, placements, groups, done, nodes


def _recover(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= inp.size // 2:
        raise ValueError("Canvas background must be a strict majority")
    answers = []
    for wall in range(10):
        if wall == bg:
            continue
        for component in _components(inp == wall):
            r0, r1, c0, c1 = _bounds(component)
            board = inp[r0:r1, c0:c1]
            if min(board.shape) < 4:
                continue
            if not all(np.all(edge == wall) for edge in
                       (board[0], board[-1], board[:, 0], board[:, -1])):
                continue
            if not np.any(board == bg):
                continue
            outside = np.ones(inp.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            if np.any((inp == wall) & outside):
                continue
            moat = np.zeros(inp.shape, dtype=bool)
            moat[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
            if np.any((inp != bg) & outside & moat):
                continue
            fragments = _components((inp != bg) & outside)
            if not 2 <= len(fragments) <= 6:
                continue
            mask = board != wall
            area = int(mask.sum())
            external_area = sum(map(len, fragments))
            if not area or external_area % area:
                continue
            ratio = external_area // area
            k = math.isqrt(ratio)
            if k*k != ratio or not 2 <= k <= 4 or max(board.shape)*k > 30:
                continue
            tiles, locations = [], []
            for cells in fragments:
                a, b, c, d = _bounds(cells)
                tile = inp[a:b, c:d]
                h, w = tile.shape
                if h*w != len(cells) or h % k or w % k:
                    break
                small = tile[::k, ::k]
                if not np.array_equal(tile, np.repeat(np.repeat(small, k, 0), k, 1)):
                    break
                if np.any((small == bg) | (small == wall)):
                    break
                tiles.append(small.copy())
                locations.append((a, b, c, d))
            else:
                outcomes, placements, groups, done, nodes = _covers(mask, board, bg, tiles)
                if not done:
                    raise ValueError("Exact-cover budget exhausted")
                if len(outcomes) > 1:
                    raise ValueError("Ambiguous colored mosaic")
                if not outcomes:
                    continue
                data, path = next(iter(outcomes.items()))
                logical = board.copy()
                logical[mask] = np.frombuffer(data, dtype=np.uint8)
                output = np.repeat(np.repeat(logical, k, 0), k, 1)
                labels = np.full(mask.shape, -1, dtype=int)
                cavities = _components(mask)
                for j, cells in enumerate(cavities):
                    for p in cells:
                        labels[p] = j
                cavity_counts = [0] * len(cavities)
                reflection = rotation = 0
                hidden_per_tile = []
                for pi in path:
                    gi, bits, ii, tile, r, c = placements[pi]
                    cavity_counts[int(labels[r, c])] += 1
                    h, w = tile.shape
                    hidden_per_tile.append(int(np.sum(board[r:r+h, c:c+w] == bg)))
                    if groups[gi]["count"] == 1:
                        base = groups[gi]["base"]
                        if not np.array_equal(tile, base):
                            if _key(tile) in {_key(a) for a in _variants(base, False)}:
                                rotation += 1
                            else:
                                reflection += 1
                anchors = int(np.sum(mask & (board != bg)))
                info = {"background": bg, "wall": wall, "board": board,
                        "mask": mask, "logical": logical, "tiles": tiles,
                        "scale": k, "board_bounds": (r0, r1, c0, c1),
                        "board_shape": tuple(board.shape), "tile_count": len(tiles),
                        "areas": tuple(sorted(t.size for t in tiles)),
                        "dimensions": tuple(sorted(t.shape for t in tiles)),
                        "anchors": anchors, "area": area,
                        "hidden_per_tile": tuple(hidden_per_tile),
                        "cavities": len(cavities), "max_cavity_tiles": max(cavity_counts),
                        "reflection": reflection, "rotation": rotation, "nodes": nodes,
                        "repeated": len(groups) < len(tiles),
                        "above": any(a < r0 for a, b, c, d in locations),
                        "below": any(b > r1 for a, b, c, d in locations),
                        "left": any(c < c0 for a, b, c, d in locations),
                        "right": any(d > c1 for a, b, c, d in locations)}
                answers.append((output, info))
    if len(answers) != 1:
        raise ValueError("Expected one framed board and colored completion")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the answer entirely from visible input cells."""
    return _recover(input_grid)[0]


def _grow_rectangles(rng, bh, bw, count, witness):
    occupied = np.zeros((bh, bw), dtype=bool)
    rects = []
    for i in range(count):
        if witness and i < 2:
            h, w = 2, 3
        elif i and rng.random() < .24:
            h, w = rects[int(rng.integers(len(rects)))][2:]
        else:
            h = int(rng.integers(1, min(4, bh-2) + 1))
            w = int(rng.integers(1, min(4, bw-2) + 1))
        options = []
        for r in range(1, bh-h):
            for c in range(1, bw-w):
                if np.any(occupied[r:r+h, c:c+w]):
                    continue
                touches = (np.any(occupied[r-1, c:c+w]) or np.any(occupied[r+h, c:c+w])
                           or np.any(occupied[r:r+h, c-1]) or np.any(occupied[r:r+h, c+w]))
                # The first contact is compulsory; later islands are permitted.
                if i == 1 and not touches:
                    continue
                if i > 1 and not touches and rng.random() < .85:
                    continue
                options.append((r, c))
        if not options:
            return None
        r, c = options[int(rng.integers(len(options)))]
        occupied[r:r+h, c:c+w] = True
        rects.append((r, c, h, w))
    return occupied, rects


def _pattern(rng, shape, palette, chiral=False):
    for _ in range(30):
        mode = int(rng.integers(5))
        if mode == 0 and not chiral:
            tile = np.full(shape, rng.choice(palette), dtype=np.int8)
        elif mode == 1 and not chiral:
            yy, xx = np.indices(shape)
            tile = np.array(palette, dtype=np.int8)[(yy+xx) % len(palette)]
        else:
            tile = rng.choice(palette, size=shape).astype(np.int8)
        if not chiral or len(_variants(tile)) == 8:
            return tile
    # Shape is 2x3 for this branch; this matrix has eight distinct D4 images.
    a, b, c = palette[:3]
    return np.array([[a, b, a], [c, a, b]], dtype=np.int8)


def _add_anchors(rng, mask, logical, bg, tiles, rects):
    clues = logical.copy()
    clues[mask] = bg
    owner = np.full(mask.shape, -1, dtype=int)
    for i, (r, c, h, w) in enumerate(rects):
        owner[r:r+h, c:c+w] = i
    counts = [h*w for r, c, h, w in rects]
    initial = [(int(r), int(c)) for r, c in np.argwhere(mask)
               if counts[int(owner[r, c])] > 1]
    r, c = initial[int(rng.integers(len(initial)))]
    clues[r, c] = logical[r, c]
    counts[int(owner[r, c])] -= 1
    expected = logical[mask].astype(np.uint8).tobytes()
    for _ in range(int(mask.sum()) // 2 + 1):
        outcomes, _, _, done, _ = _covers(mask, clues, bg, tiles)
        if done and set(outcomes) == {expected}:
            return clues
        if int(np.sum(mask & (clues != bg))) >= int(mask.sum()) // 2:
            return None
        wrong = next((a for a in outcomes if a != expected), None)
        difference = mask.copy()
        if wrong is not None:
            difference[mask] = np.frombuffer(wrong, dtype=np.uint8) != logical[mask]
        candidates = [(int(r), int(c)) for r, c in np.argwhere(difference & (clues == bg))
                      if counts[int(owner[r, c])] > 1]
        if not candidates:
            return None
        r, c = candidates[int(rng.integers(len(candidates)))]
        clues[r, c] = logical[r, c]
        counts[int(owner[r, c])] -= 1
    return None


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return []
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _scatter(rng, board, tiles, k, bg):
    objects = [board] + [np.repeat(np.repeat(t, k, 0), k, 1) for t in tiles]
    for attempt in range(24):
        height, width = ((int(rng.integers(18, 31)), int(rng.integers(18, 31)))
                         if attempt < 12 else (30, 30))
        canvas = np.full((height, width), bg, dtype=np.int8)
        blocked = np.zeros(canvas.shape, dtype=bool)
        # Random ties and arbitrary origins; no relationship to destination order.
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(rng.random())))
        for i in order:
            obj = objects[i]
            h, w = obj.shape
            origins = _free_origins(blocked, h, w)
            if len(origins) == 0:
                break
            r, c = map(int, origins[int(rng.integers(len(origins)))])
            canvas[r:r+h, c:c+w] = obj
            blocked[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True
        else:
            if np.count_nonzero(canvas == bg) > canvas.size // 2:
                return canvas
    return None


def _make_pair(rng, k, roles, witness=False):
    bg, wall = roles
    for _ in range(240):
        limit = min(12, 30 // k)
        bh, bw = int(rng.integers(6, limit+1)), int(rng.integers(6, limit+1))
        count = int(rng.integers(2, 5 if witness else 7))
        grown = _grow_rectangles(rng, bh, bw, count, witness)
        if grown is None:
            continue
        mask, rects = grown
        if int(mask.sum()) < 7 or int(mask.sum())*k*k + bh*bw > 370:
            continue
        palette = [c for c in range(10) if c not in roles]
        rng.shuffle(palette)
        palette = palette[:int(rng.integers(3, 7))]
        pats = []
        for i, (r, c, h, w) in enumerate(rects):
            repeats = [t for t in pats if t.shape == (h, w)]
            if not witness and repeats and rng.random() < .30:
                tile = repeats[int(rng.integers(len(repeats)))].copy()
            else:
                tile = _pattern(rng, (h, w), palette, witness and i < 2)
            pats.append(tile)
        if witness and _canonical(pats[0]) == _canonical(pats[1]):
            continue
        logical = np.full((bh, bw), wall, dtype=np.int8)
        tiles = []
        for i, ((r, c, h, w), tile) in enumerate(zip(rects, pats)):
            logical[r:r+h, c:c+w] = tile
            if witness and i == 0:
                loose = np.fliplr(tile).copy()
            elif witness and i == 1:
                loose = np.rot90(tile, 1).copy()
            else:
                variants = _variants(tile)
                loose = variants[int(rng.integers(len(variants)))].copy()
            tiles.append(loose)
        board = _add_anchors(rng, mask, logical, bg, tiles, rects)
        if board is None:
            continue
        canvas = _scatter(rng, board, tiles, k, bg)
        if canvas is None:
            continue
        output = np.repeat(np.repeat(logical, k, 0), k, 1)
        try:
            prediction, info = _recover(canvas)
        except ValueError:
            continue
        if (not np.array_equal(prediction, output) or info["max_cavity_tiles"] < 2
                or min(info["hidden_per_tile"]) < 1):
            continue
        if witness and (not info["reflection"] or not info["rotation"]):
            continue
        return {"input": canvas, "output": output}
    raise RuntimeError("Bounded mosaic construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))
    train_scales = ([2]*nt if mode == 0 else [3]*nt if mode == 1
                    else [2, 3] + [int(rng.choice([2, 3])) for _ in range(nt-2)])
    test_scales = [3 if mode == 0 else 2 if mode == 1 else 4]*ne
    puzzle, seen = {"train": [], "test": []}, set()
    previous_roles, first_geometry = None, None
    for i, k in enumerate(train_scales + test_scales):
        palette = list(map(int, rng.permutation(10)))
        roles = tuple(palette[:2])
        while previous_roles is not None and any(a == b for a, b in zip(roles, previous_roles)):
            palette = palette[1:] + palette[:1]
            roles = tuple(palette[:2])
        previous_roles = roles
        for _ in range(12):
            pair = _make_pair(rng, k, roles, witness=i == 0)
            info = _recover(pair["input"])[1]
            geometry = (info["mask"].shape, info["mask"].tobytes())
            if _key(pair["input"]) not in seen and (i != 1 or geometry != first_geometry):
                break
        else:
            raise RuntimeError("Distinct mosaic construction exhausted")
        if i == 0:
            first_geometry = geometry
        seen.add(_key(pair["input"]))
        puzzle["train" if i < nt else "test"].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, infos = set(), []
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = _grid(pair["input"]), _grid(pair["output"])
        assert _key(inp) not in seen
        seen.add(_key(inp))
        expected, f = _recover(inp)
        assert np.array_equal(expected, out)
        assert f["max_cavity_tiles"] >= 2
        assert 1 <= f["anchors"] <= f["area"] // 2
        assert min(f["hidden_per_tile"]) >= 1
        assert sum(t.size for t in f["tiles"]) == f["area"]
        assert np.all(f["logical"][f["board"] != f["background"]]
                      == f["board"][f["board"] != f["background"]])
        infos.append(f)
    first = infos[0]
    assert first["reflection"] and first["rotation"]
    erased = first["board"].copy()
    erased[first["mask"]] = first["background"]
    outcomes, _, _, done, _ = _covers(first["mask"], erased, first["background"], first["tiles"])
    assert len(outcomes) >= 2, "First demonstration must defeat geometry-only packing"
    assert len({(f["background"], f["wall"]) for f in infos[:nt]}) >= 2
    assert (infos[0]["mask"].shape, infos[0]["mask"].tobytes()) != (
        infos[1]["mask"].shape, infos[1]["mask"].tobytes())
    train_scales = {f["scale"] for f in infos[:nt]}
    assert all(f["scale"] not in train_scales for f in infos[nt:])


def _signature(p):
    return tuple((_key(q["input"]), _key(q["output"]))
                 for s in ("train", "test") for q in p[s])


def self_test():
    """Bounded seed suite; no artifacts, logs, retained puzzles or hidden state."""
    signatures, shapes, silhouettes, patterns, schedules = set(), set(), set(), set(), set()
    features = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in p:
            for pair in p[split]:
                for grid in pair.values():
                    digest.update(bytes(grid.shape))
                    digest.update(grid.astype(np.uint8).tobytes())
        signatures.add(digest.hexdigest())
        ff = [_recover(q["input"])[1] for s in p for q in p[s]]
        schedules.add((tuple(sorted({f["scale"] for f in ff[:4]})), ff[4]["scale"]))
        for f in ff:
            shapes.add(f["board_shape"])
            silhouettes.add((f["mask"].shape, f["mask"].tobytes()))
            patterns.update(_canonical(t) for t in f["tiles"])
            features.append({name: f[name] for name in
                             ("scale", "dimensions", "tile_count", "repeated", "cavities",
                              "above", "below", "left", "right")})
    assert len(signatures) == 42 and len(schedules) == 3
    assert len(shapes) >= 25 and len(silhouettes) >= 150 and len(patterns) >= 200
    assert {f["scale"] for f in features} == {2, 3, 4}
    assert len({f["dimensions"] for f in features}) >= 60
    assert {f["tile_count"] for f in features} >= {2, 3, 4, 5}
    assert any(f["repeated"] for f in features)
    assert any(f["cavities"] > 1 for f in features)
    assert all(any(f[d] for f in features) for d in ("above", "below", "left", "right"))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1) + 1):
            p = generate_puzzle(10000 + 10*nt + ne, nt, ne)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    a = generate_puzzle(8271, 2, 1)
    assert _signature(a) == _signature(generate_puzzle(8271, 2, 1))
    assert _signature(a) != _signature(generate_puzzle(8272, 2, 1))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Exact-cover ambiguity and finite inventory, independent of generation.
    mask = np.ones((1, 4), dtype=bool)
    clues = np.zeros((1, 4), dtype=np.int8)
    red = np.array([[2, 2]], dtype=np.int8)
    blue = np.array([[3, 3]], dtype=np.int8)
    assert not _covers(mask, clues, 0, [red])[0]
    assert len(_covers(mask, clues, 0, [red, red])[0]) == 1
    assert len(_covers(mask, clues, 0, [red, blue])[0]) == 2
    clues[0, 0] = 2
    assert len(_covers(mask, clues, 0, [red, blue])[0]) == 1
    # A wrong output and a broken macrocell must be rejected.
    wrong = {s: [{k: v.copy() for k, v in q.items()} for q in a[s]] for s in a}
    wrong["test"][0]["output"][0, 0] = (int(wrong["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(wrong)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Incorrect output accepted")
    inp = a["test"][0]["input"].copy()
    f = _recover(inp)[1]
    r0, r1, c0, c1 = f["board_bounds"]
    outside = np.ones(inp.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    r, c = np.argwhere((inp != f["background"]) & outside)[0]
    inp[r, c] = next(x for x in range(10) if x not in
                     (int(inp[r, c]), f["background"], f["wall"]))
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("Broken macrocell accepted")
