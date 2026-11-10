"""Thread batches address molds by cavity count and dye their ordered cavities.

Only standard-library modules and NumPy are used.  Construction supplies labels
from its objects; solve and validation reconstruct everything from input pixels.
"""
from itertools import permutations
import numpy as np


STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else STEPS
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in steps:
                q = (p[0] + dr, p[1] + dc)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(sorted(cells))
    return sorted(result, key=lambda cells: cells[0])


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _trace(cells):
    cells = set(cells)
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in STEPS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    ends = sorted(p for p in cells if len(neighbors[p]) == 1)
    if len(ends) != 2 or any(len(v) not in (1, 2) for v in neighbors.values()):
        raise ValueError("thread must be an induced simple chain")
    if ends[0][0] == ends[1][0]:
        raise ValueError("thread endpoints must have different rows")
    ordered, previous, current = [], None, ends[0]
    while current is not None:
        if current in ordered:
            raise ValueError("cyclic thread")
        ordered.append(current)
        choices = [q for q in neighbors[current] if q != previous]
        previous, current = current, choices[0] if choices else None
    if len(ordered) != len(cells):
        raise ValueError("disconnected thread")
    return ordered


def _records(grid, path, delimiter):
    if int(grid[path[-1]]) != delimiter:
        raise ValueError("both endpoints must carry the delimiter")
    records, current = [], []
    for p in path[1:]:
        if int(grid[p]) == delimiter:
            records.append(current)
            current = []
        else:
            current.append(p)
    if current or len(records) not in (2, 3, 4):
        raise ValueError("expected two to four closed batches")
    if any(len(r) > 5 for r in records) or len({len(r) for r in records}) != len(records):
        raise ValueError("batch lengths must be distinct and in 0..5")
    if not any(len(r) >= 2 for r in records):
        raise ValueError("at least one ordered multi-cell batch is required")
    return records


def _decode(value):
    grid = _grid(value)
    hist = np.bincount(grid.ravel(), minlength=10)
    bg = int(hist.argmax())
    if int(hist[bg]) <= grid.size // 2:
        raise ValueError("background must have a strict majority")
    objects = _components(grid != bg)
    threads, molds = [], {}
    boxes = []
    for cells in objects:
        box = _box(cells)
        boxes.append(box)
        try:
            path = _trace(cells)
        except ValueError:
            path = None
        if path is not None:
            threads.append(path)
            continue
        r, s, c, d = box
        mold = grid[r:s, c:d]
        if not all(4 <= n <= 8 for n in mold.shape):
            raise ValueError("mold dimensions must be 4..8")
        color = int(mold[0, 0])
        if color == bg or not np.all(_perimeter(mold) == color):
            raise ValueError("mold needs a complete monochrome perimeter")
        if not set(map(int, np.unique(mold))) <= {bg, color}:
            raise ValueError("mold must be binary")
        if len(_components(mold != bg)) != 1:
            raise ValueError("mold foreground must be connected")
        holes = _components(mold == bg)
        k = len(holes)
        if k > 5 or k in molds:
            raise ValueError("mold cavity counts must be distinct and in 0..5")
        molds[k] = {"box": box, "color": color, "mask": mold == bg,
                    "holes": [[(r + rr, c + cc) for rr, cc in hole] for hole in holes]}
    if len(threads) != 1 or len(molds) not in (2, 3, 4):
        raise ValueError("expected one thread and two to four molds")
    for i, (a, b, c, d) in enumerate(boxes):
        for e, f, g, h in boxes[i + 1:]:
            if not (b < e or f < a or d < g or h < c):
                raise ValueError("object bounding boxes need a blank-cell gap")
    path = threads[0]
    delimiter = int(grid[path[0]])
    records = _records(grid, path, delimiter)
    if set(map(len, records)) != set(molds):
        raise ValueError("batch lengths and cavity counts must match bijectively")
    return {"grid": grid, "background": bg, "delimiter": delimiter,
            "path": path, "records": records, "molds": molds}


def _render(info, orders=None):
    grid, bg = info["grid"], info["background"]
    out = grid.copy()
    for p in info["path"]:
        out[p] = bg
    for record in info["records"]:
        tokens = record if orders is None else orders(record)
        for token, hole in zip(tokens, info["molds"][len(record)]["holes"]):
            for p in hole:
                out[p] = int(grid[token])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from the input; never consult generation."""
    return _render(_decode(input_grid))


def _spatial_failures(info):
    """Eight global row/column sweeps, applied within correctly identified batches."""
    expected = _render(info)
    result = []
    for primary in (0, 1):
        for first in (-1, 1):
            for second in (-1, 1):
                def order(record, p=primary, f=first, s=second):
                    return sorted(record, key=lambda q: (f * q[p], s * q[1 - p]))
                result.append(not np.array_equal(_render(info, order), expected))
    return result


def _evidence(info):
    masks = [m["mask"] for m in info["molds"].values()]
    areas = {k: int(m["mask"].sum()) for k, m in info["molds"].items()}
    payloads = [[int(info["grid"][p]) for p in r] for r in info["records"]]
    return {
        "sweeps": _spatial_failures(info),
        "area_inversion": any(a < b and areas[a] > areas[b] for a in areas for b in areas),
        "diagonal": any(len(_components(m, True)) < len(_components(m)) for m in masks),
        "repetition": any(any(a == b for a, b in zip(v, v[1:])) and len(set(v)) > 1
                          for v in payloads),
        "direction": any(v != v[::-1] for v in payloads),
        "above": min(r for r, _ in info["path"]) < info["path"][0][0],
        "nonpositional": [len(r) for r in info["records"]] !=
                         sorted(info["molds"], key=lambda k:
                                (info["molds"][k]["box"][0], info["molds"][k]["box"][2])),
    }


def _make_holes(rng, count, large=False, witness=False):
    low = 5 if count >= 3 or witness else 4
    if large:
        h, w = 8, int(rng.integers(max(6, low), 9))
    else:
        h, w = int(rng.integers(low, 8)), int(rng.integers(low, 8))
    mask = np.zeros((h, w), dtype=bool)
    if count == 0:
        pass
    elif witness and count == 1 or count == 1 and rng.random() < 0.30:
        mask[1:-1, 1:-1] = True
    elif witness and count == 2:
        mask[1, 1] = mask[2, 2] = mask[2, 3] = True
    else:
        # Seed independent holes while preserving connected foreground.
        success = False
        for _ in range(24):
            mask[:] = False
            for index in rng.permutation((h - 2) * (w - 2)):
                r, c = divmod(int(index), w - 2)
                r, c = r + 1, c + 1
                if any(mask[r + dr, c + dc] for dr, dc in STEPS):
                    continue
                mask[r, c] = True
                if len(_components(~mask)) != 1:
                    mask[r, c] = False
                if int(mask.sum()) == count:
                    success = True
                    break
            if success:
                break
        if not success:
            mask[:] = False
            # Every >=3x3 interior admits these five isolated holes with a
            # connected rim. Smaller count-1/2 interiors use opposite corners.
            cells = ([(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)] if count >= 3
                     else [(1, 1), (h - 2, w - 2)])
            for p in cells[:count]:
                mask[p] = True
        target = int(rng.integers(count, (h - 2) * (w - 2) + 1))
        for _ in range((h - 2) * (w - 2)):
            if int(mask.sum()) >= target:
                break
            options = []
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if mask[r, c] or not any(mask[r + dr, c + dc] for dr, dc in STEPS):
                        continue
                    trial = mask.copy()
                    trial[r, c] = True
                    if len(_components(trial)) == count and len(_components(~trial)) == 1:
                        options.append((r, c))
            if not options:
                break
            mask[options[int(rng.integers(len(options)))]] = True
    if rng.random() < 0.5:
        mask = mask.T
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    assert len(_components(mask)) == count and len(_components(~mask)) == 1
    return mask.copy()


def _walk(rng, length, allow_above):
    for _ in range(96):
        path = [(0, 0)]
        used = set(path)
        persistence = float(rng.choice((0.8, 2.0, 8.0)))
        previous = None
        for _ in range(length - 1):
            options, weights = [], []
            for step in STEPS:
                q = (path[-1][0] + step[0], path[-1][1] + step[1])
                if q in used or sum((q[0] + dr, q[1] + dc) in used for dr, dc in STEPS) != 1:
                    continue
                options.append((q, step))
                weights.append(persistence if step == previous else 1.0)
            if not options:
                break
            weights = np.asarray(weights)
            q, previous = options[int(rng.choice(len(options), p=weights / weights.sum()))]
            path.append(q)
            used.add(q)
        if len(path) != length or path[0][0] == path[-1][0]:
            continue
        if path[0][0] > path[-1][0]:
            path.reverse()
        r0, r1, c0, c1 = _box(path)
        if r1 - r0 > 12 or c1 - c0 > 12:
            continue
        if not allow_above and r0 < path[0][0]:
            continue
        return [(r - r0, c - c0) for r, c in path]
    # Compact induced serpentine: constructive fallback fits every length 6..19.
    path = [(0, 0)]
    for dr, dc in ((1, 0), (0, 1), (1, 0), (1, 0), (0, -1), (0, -1),
                   (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0),
                   (0, -1), (0, -1), (1, 0), (1, 0), (0, 1), (0, 1)):
        path.append((path[-1][0] + dr, path[-1][1] + dc))
    path = path[:length]
    r0, _, c0, _ = _box(path)
    return [(r - r0, c - c0) for r, c in path]


def _thread_image(path, sequence, bg):
    r0, r1, c0, c1 = _box(path)
    a = np.full((r1 - r0, c1 - c0), bg, dtype=np.int8)
    for p, color in zip(path, sequence):
        a[p[0] - r0, p[1] - c0] = color
    return a


def _free_origins(boxes, h, w):
    rr, cc = np.indices((31 - h, 31 - w))
    free = np.ones(rr.shape, dtype=bool)
    for r, c, bh, bw in boxes:
        free &= (rr + h < r) | (rr > r + bh) | (cc + w < c) | (cc > c + bw)
    return np.argwhere(free)


def _pack(rng, objects, bg, shelf_order=None):
    boxes = None
    for _ in range(24 if shelf_order is None else 0):
        order = list(map(int, rng.permutation(len(objects))))
        trial, placed = [], {}
        for index in order:
            h, w = objects[index].shape
            free = _free_origins(trial, h, w)
            if len(free) == 0:
                break
            r, c = map(int, free[int(rng.integers(len(free)))])
            placed[index] = (r, c)
            trial.append((r, c, h, w))
        if len(placed) == len(objects):
            boxes = placed
            break
    if boxes is None:
        # With one <=12x12 thread and <=4 molds of side <=8, a shelf exists.
        orders = permutations(range(len(objects))) if shelf_order is None else (shelf_order,)
        for order in orders:
            r, c, shelf, placed = 0, 0, 0, {}
            for index in order:
                h, w = objects[index].shape
                if c + w > 30:
                    r, c, shelf = r + shelf + 1, 0, 0
                if r + h > 30:
                    break
                placed[index] = (r, c)
                c += w + 1
                shelf = max(shelf, h)
            if len(placed) == len(objects):
                boxes = placed
                break
    assert boxes is not None
    grid = np.full((30, 30), bg, dtype=np.int8)
    for i, a in enumerate(objects):
        r, c = boxes[i]
        grid[r:r + a.shape[0], c:c + a.shape[1]] = a
    r0 = min(r for r, c in boxes.values())
    c0 = min(c for r, c in boxes.values())
    r1 = max(boxes[i][0] + a.shape[0] for i, a in enumerate(objects))
    c1 = max(boxes[i][1] + a.shape[1] for i, a in enumerate(objects))
    # Random margins include edge contact; expand if needed for background identity.
    r0 = max(0, r0 - int(rng.integers(4)))
    c0 = max(0, c0 - int(rng.integers(4)))
    r1 = min(30, r1 + int(rng.integers(4)))
    c1 = min(30, c1 + int(rng.integers(4)))
    foreground = int(np.count_nonzero(grid != bg))
    while (r1 - r0) * (c1 - c0) <= 2 * foreground:
        edges = [i for i, ok in enumerate((r0 > 0, c0 > 0, r1 < 30, c1 < 30)) if ok]
        edge = edges[int(rng.integers(len(edges)))]
        if edge == 0:
            r0 -= 1
        elif edge == 1:
            c0 -= 1
        elif edge == 2:
            r1 += 1
        else:
            c1 += 1
    return grid[r0:r1, c0:c1].copy()


def _make_pair(rng, counts, bg, large=False, witness=False, repeated=False,
               allow_above=False):
    colors = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    delimiter, palette = colors[0], colors[1:]
    pool = palette[:int(rng.integers(3, 9))]
    masks = {k: _make_holes(rng, k, large, witness and k in (1, 2)) for k in counts}
    frames = {k: delimiter if rng.random() < 0.20 else int(rng.choice(pool)) for k in counts}
    dyes = {k: list(map(int, rng.choice(pool, k))) for k in counts}
    if witness:
        dyes[1] = [int(rng.choice([c for c in pool if c != frames[1]]))]
        dyes[2] = list(map(int, rng.choice(pool, 2, replace=False)))
        dyes[3] = list(map(int, rng.choice(pool, 3, replace=False)))
    if repeated:
        choices = list(map(int, rng.choice(pool, 2, replace=False)))
        dyes[3] = [choices[0], choices[0], choices[1]]
    order = list(map(int, rng.permutation(counts)))
    sequence = [delimiter]
    for k in order:
        sequence.extend(dyes[k])
        sequence.append(delimiter)
    # Witness is about *answers*, not hashes or turn counts. Try diverse walks,
    # then use a small proven distinguishing path if random growth misses it.
    path = None
    for _ in range(64 if witness else 1):
        candidate = _walk(rng, len(sequence), allow_above)
        if witness:
            cursor, records = 1, []
            for k in order:
                records.append(candidate[cursor:cursor + k])
                cursor += k + 1
            thread = _thread_image(candidate, sequence, bg)
            # Compare token sequences directly: holes in this local diagnostic
            # are not yet placed, so _render is deliberately not used.
            failed = []
            for primary in (0, 1):
                for f in (-1, 1):
                    for s in (-1, 1):
                        failed.append(any([int(thread[p]) for p in sorted(rec, key=lambda p:
                            (f * p[primary], s * p[1 - primary]))] != dyes[k]
                            for k, rec in zip(order, records)))
            above = min(r for r, c in candidate) < candidate[0][0]
            if not all(failed) or (allow_above and not above):
                continue
        path = candidate
        break
    if path is None:
        order = [3, 1, 2]
        if allow_above:
            path = [(0, 0), (-1, 0), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
                    (0, 2), (1, 2), (2, 2), (3, 2)]
        else:
            path = [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (4, 1),
                    (4, 0), (4, -1), (4, -2), (5, -2)]
        r0, _, c0, _ = _box(path)
        path = [(r - r0, c - c0) for r, c in path]
        sequence = [delimiter]
        for k in order:
            sequence.extend(dyes[k])
            sequence.append(delimiter)
    objects = [_thread_image(path, sequence, bg)]
    mold_order = list(map(int, rng.permutation(counts)))
    for k in mold_order:
        a = np.full(masks[k].shape, frames[k], dtype=np.int8)
        a[masks[k]] = bg
        objects.append(a)
    inp = _pack(rng, objects, bg)
    # Ground truth is constructed from the supplied dyes/masks; use decoding
    # only to recover placements after packing, not to infer its payload colors.
    info = _decode(inp)
    if witness and not _evidence(info)["nonpositional"]:
        # Keep the proven thread and payloads unchanged. A path-first shelf
        # with molds in reverse address order is feasible under the witness
        # size bounds and guarantees a nonpositional routing demonstration.
        shelf_order = [0] + [1 + mold_order.index(k) for k in order[::-1]]
        inp = _pack(rng, objects, bg, shelf_order=shelf_order)
        info = _decode(inp)
    out = inp.copy()
    for p in info["path"]:
        out[p] = bg
    for k, m in info["molds"].items():
        for dye, hole in zip(dyes[k], m["holes"]):
            for p in hole:
                out[p] = dye
    return {"input": inp, "output": out}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, v, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("count", "scale")))
    allow_above = bool(rng.integers(2))
    shared_bg = bool(rng.integers(2))
    base_bg = int(rng.integers(10))
    puzzle, used = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            witness = split == "train" and i == 0
            repeated = split == "train" and i == 1
            if witness:
                counts = [1, 2, 3]
            elif repeated:
                other = int(rng.choice((0, 1, 2)))
                counts = [other, 3]
            else:
                cap = 5 if split == "test" and shift == "count" else 3
                n = int(rng.integers(2, min(4, cap + 1) + 1))
                required = [int(rng.choice((4, 5)))] if cap == 5 else [int(rng.choice((2, 3)))]
                available = [k for k in range(cap + 1) if k not in required]
                counts = sorted(required + list(map(int, rng.choice(available, n - 1, replace=False))))
            bg = base_bg if shared_bg else int(rng.integers(10))
            pair = _make_pair(rng, counts, bg, large=split == "test" and shift == "scale",
                              witness=witness, repeated=repeated, allow_above=allow_above)
            if _key(pair["input"]) in used:
                # Finite, rule-preserving color-permutation fallback. There are
                # 90 choices for two present colors and at most 13 prior inputs.
                palette = list(map(int, np.unique(pair["input"])))
                source = palette[:2] + [c for c in range(10) if c not in palette[:2]]
                found = False
                for a in range(10):
                    for b in range(10):
                        if a == b:
                            continue
                        target = [a, b] + [c for c in range(10) if c not in (a, b)]
                        lookup = np.empty(10, dtype=np.int8)
                        lookup[source] = target
                        fresh = {k: lookup[v] for k, v in pair.items()}
                        if _key(fresh["input"]) not in used:
                            pair, found = fresh, True
                            break
                    if found:
                        break
                assert found
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check topology, addressed dyeing, preserved pixels and training evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(out, _render(info)), "wrong dyeing or thread erasure"
            for record in info["records"]:
                for token, hole in zip(record, info["molds"][len(record)]["holes"]):
                    assert all(int(out[p]) == int(inp[token]) for p in hole)
            assert all(int(out[p]) == info["background"] for p in info["path"])
            protected = np.ones(inp.shape, dtype=bool)
            for p in info["path"]:
                protected[p] = False
            for m in info["molds"].values():
                for hole in m["holes"]:
                    for p in hole:
                        protected[p] = False
            assert np.array_equal(inp[protected], out[protected])
            infos[split].append(info)
    evidence = [_evidence(i) for i in infos["train"]]
    assert all(any(e["sweeps"][j] for e in evidence) for j in range(8)), "spatial sweep shortcut survives"
    for name in ("area_inversion", "diagonal", "repetition", "direction", "nonpositional"):
        assert any(e[name] for e in evidence), f"missing {name} teaching witness"
    if any(_evidence(i)["above"] for i in infos["test"]):
        assert any(e["above"] and e["direction"] for e in evidence)
    # A real visible test shift, as opposed to a latent mode flag.
    train_counts = set().union(*(set(i["molds"]) for i in infos["train"]))
    train_size = max(max(m["mask"].shape) for i in infos["train"] for m in i["molds"].values())
    assert any(set(i["molds"]) - train_counts or any(max(m["mask"].shape) > train_size
               for m in i["molds"].values()) for i in infos["test"]), "missing train/test shift"


def self_test():
    """Bounded 42-seed variation and mutation checks; no files or network."""
    shapes, walks, masks, count_sets, backgrounds, palettes, fingerprints = [set() for _ in range(7)]
    zeros, rectangles, upward, same_dye, shifts = 0, 0, 0, 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        train_max_count, test_max_count, train_max_size, test_max_size = 0, 0, 0, 0
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                shapes.add(pair["input"].shape)
                origin = info["path"][0]
                walks.add(tuple((r - origin[0], c - origin[1]) for r, c in info["path"]))
                backgrounds.add(info["background"])
                palettes.add(len(np.unique(pair["input"])))
                count_sets.add(tuple(sorted(info["molds"])))
                zeros += 0 in info["molds"]
                upward += any(b[0] < a[0] for a, b in zip(info["path"], info["path"][1:]))
                flat = [int(info["grid"][p]) for r in info["records"] for p in r]
                same_dye += len(set(flat)) < len(flat)
                for k, m in info["molds"].items():
                    masks.add((m["mask"].shape, m["mask"].tobytes()))
                    rectangles += m["mask"].shape[0] != m["mask"].shape[1]
                    if split == "train":
                        train_max_count = max(train_max_count, k)
                        train_max_size = max(train_max_size, max(m["mask"].shape))
                    else:
                        test_max_count = max(test_max_count, k)
                        test_max_size = max(test_max_size, max(m["mask"].shape))
        shifts.add("count" if test_max_count > train_max_count else "scale")
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                       for k in ("input", "output"))
    assert len(fingerprints) == 42
    assert len(shapes) >= 20 and len(walks) >= 90 and len(masks) >= 150
    assert len(count_sets) >= 15 and len(backgrounds) == 10 and len(palettes) >= 4
    assert zeros >= 10 and rectangles >= 80 and upward >= 30 and same_dye >= 100
    assert shifts == {"count", "scale"}
    # A semantic family member outside the sampler's thread-box bound. This
    # straight 19-cell thread also reuses its delimiter as a mold foreground.
    fixture = np.zeros((30, 25), dtype=np.int8)
    fixture_counts = [5, 2, 4, 3]
    stream = [9]
    fixture_dyes = {}
    for k in fixture_counts:
        fixture_dyes[k] = [1 + j % 4 for j in range(k)]
        stream += fixture_dyes[k] + [9]
    fixture[1:20, 1] = stream
    expected = fixture.copy()
    expected[1:20, 1] = 0
    for k, (r, c) in zip((2, 3, 4, 5), ((1, 5), (1, 15), (13, 5), (13, 15))):
        holes = ([(1, 1), (2, 2)] if k == 2 else
                 [(1, 1), (1, 3), (3, 1)] if k == 3 else
                 [(1, 1), (1, 3), (3, 1), (3, 3)] if k == 4 else
                 [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)])
        fixture[r:r + 5, c:c + 5] = k + 4
        expected[r:r + 5, c:c + 5] = k + 4
        for color, (dr, dc) in zip(fixture_dyes[k], holes):
            fixture[r + dr, c + dc] = 0
            expected[r + dr, c + dc] = color
    original = fixture.copy()
    assert np.array_equal(solve(fixture), expected)
    assert np.array_equal(fixture, original), "solve mutated the input"
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(100 + nt * 5 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(0, 2, 1)
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    p = bad["test"][0]
    point = _decode(p["input"])["path"][0]
    p["output"][point] = (int(p["output"][point]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted erasure accepted")
    # Turning an internal separator into a dye preserves the visible chain,
    # but breaks its unique cavity-address correspondence and must be rejected.
    malformed = sample["train"][0]["input"].copy()
    parsed = _decode(malformed)
    internal = [p for p in parsed["path"][1:-1] if int(malformed[p]) == parsed["delimiter"]]
    replacement = next(c for c in range(10) if c not in (parsed["background"], parsed["delimiter"]))
    malformed[internal[0]] = replacement
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("broken batch addressing accepted")
    # Endpoint tie, branch, cycle and malformed palette are rejected input-only.
    for cells in ([(0, 0), (0, 1), (0, 2)], [(0, 1), (1, 0), (1, 1), (1, 2)],
                  [(0, 0), (0, 1), (1, 0), (1, 1)]):
        try:
            _trace(cells)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed chain accepted")
    return {"seeds": 42, "puzzles": len(fingerprints), "input_shapes": len(shapes),
            "thread_geometries": len(walks), "cavity_masks": len(masks),
            "count_sets": len(count_sets), "background_colors": len(backgrounds)}
