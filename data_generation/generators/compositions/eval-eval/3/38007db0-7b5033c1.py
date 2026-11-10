"""Unspool the altered thread in a repeating fabric.

Only NumPy and the standard library are used. Generation never writes files.
The solver infers the template and the entire route from the supplied pixels.
"""

from collections import Counter, defaultdict
import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 323559327, 2819329, 551576795, 551166064, 666284063,
              342787805, 56631824, 29431867, 334321771, 128018570, 84283167,
              250544646, 157761392, 376460141, 386395696, 552354444, 585505601,
              503820464, 519740003, 176084582, 436580272, 187162560, 416332331,
              887981241, 296927364, 507266974, 445372477, 290249804, 638181239,
              756223774, 955901256, 778770265, 140339647, 652606464, 515653229,
              201339277, 663458507, 354335329, 807524852, 934705046, 792020930)
_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_TILE_SHAPES = tuple((h, w) for h in range(2, 6) for w in range(2, 6))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be two dimensional with axes in 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be in 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        _require(isinstance(value, (int, np.integer)) and
                 not isinstance(value, (bool, np.bool_)), "counts must be integers")
        _require(low <= value <= high, "counts outside supported bounds")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _trace(cells):
    """Order an induced orthogonal chain from its higher endpoint."""
    cells = set(cells)
    _require(2 <= len(cells) <= 30, "thread length must be in 2..30")
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    _require(all(len(v) in (1, 2) for v in neighbors.values()),
             "thread cannot branch or contain isolated cells")
    ends = sorted(p for p, v in neighbors.items() if len(v) == 1)
    _require(len(ends) == 2 and ends[0][0] != ends[1][0],
             "thread needs two endpoints in different rows")
    path, seen, previous, current = [], set(), None, ends[0]
    while current is not None:
        _require(current not in seen, "thread contains a cycle")
        path.append(current)
        seen.add(current)
        onward = [p for p in neighbors[current] if p != previous]
        previous, current = current, onward[0] if onward else None
    _require(len(path) == len(cells) and path[-1] == ends[1],
             "all changed cells must belong to one chain")
    return path


def _unpack(input_grid):
    g = _grid(input_grid)
    lattice = int(g[0, 0])
    rows = np.flatnonzero(np.all(g == lattice, axis=1))
    cols = np.flatnonzero(np.all(g == lattice, axis=0))
    _require(len(rows) >= 4 and len(cols) >= 4, "need at least 3 by 3 tiles")
    _require(rows[0] == cols[0] == 0 and rows[-1] == g.shape[0] - 1
             and cols[-1] == g.shape[1] - 1, "lattice must frame the grid")
    _require(bool(np.all(np.diff(rows) == rows[1])) and
             bool(np.all(np.diff(cols) == cols[1])), "lattice spacing must be regular")
    ph, pw = int(rows[1] - 1), int(cols[1] - 1)
    _require(2 <= ph <= 5 and 2 <= pw <= 5, "tile axes must be in 2..5")
    nr, nc = len(rows) - 1, len(cols) - 1
    tiles = [g[r+1:r+1+ph, c+1:c+1+pw].copy() for r in rows[:-1] for c in cols[:-1]]
    _require(all(np.all(t != lattice) for t in tiles), "lattice color is reserved")
    freq = Counter(t.tobytes() for t in tiles)
    template_bytes, count = freq.most_common(1)[0]
    _require(count * 2 > len(tiles), "one exact tile must have a strict majority")
    template = np.frombuffer(template_bytes, dtype=np.int8).reshape(ph, pw).copy()
    palette = set(map(int, np.unique(template)))
    _require(2 <= len(palette) <= 4, "template needs 2..4 interior colors")
    raw = np.concatenate([np.concatenate(tiles[r*nc:(r+1)*nc], axis=1)
                          for r in range(nr)], axis=0)
    _require(set(map(int, np.unique(raw))) <= palette,
             "changes must use existing template colors")
    baseline = np.tile(template, (nr, nc))
    path = _trace(tuple(map(int, p)) for p in np.argwhere(raw != baseline))
    groups = defaultdict(list)
    for p in path:
        groups[(p[0] // ph, p[1] // pw)].append(p)
    _require(len(groups) >= 2, "thread must cross a tile seam")
    return dict(grid=g, lattice=lattice, ph=ph, pw=pw, nr=nr, nc=nc,
                template=template, raw=raw, baseline=baseline, path=path,
                groups=dict(groups), palette=palette, master_count=count)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Copy the actual altered-cell colors in seam-free graph order."""
    info = _unpack(input_grid)
    return np.asarray([info["raw"][p] for p in info["path"]], dtype=np.int8).reshape(-1, 1)


def _assemble(raw, ph, pw, lattice):
    nr, nc = raw.shape[0] // ph, raw.shape[1] // pw
    grid = np.full((nr * (ph + 1) + 1, nc * (pw + 1) + 1), lattice, dtype=np.int8)
    for r in range(nr):
        for c in range(nc):
            grid[1+r*(ph+1):(r+1)*(ph+1), 1+c*(pw+1):(c+1)*(pw+1)] = \
                raw[r*ph:(r+1)*ph, c*pw:(c+1)*pw]
    return grid


def _walk(rng, length):
    path, occupied, previous = [(0, 0)], {(0, 0)}, None
    persistence = float(rng.choice((0.4, 1.0, 3.0, 9.0)))
    for _ in range(length - 1):
        candidates, weights = [], []
        for dr, dc in _DIRS:
            p = (path[-1][0] + dr, path[-1][1] + dc)
            if p in occupied:
                continue
            if sum((p[0] + a, p[1] + b) in occupied for a, b in _DIRS) != 1:
                continue
            candidates.append((p, (dr, dc)))
            weights.append(persistence if (dr, dc) == previous else 1.0)
        if not candidates:
            return None
        weights = np.asarray(weights, dtype=float)
        p, previous = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
        path.append(p)
        occupied.add(p)
    if path[0][0] == path[-1][0]:
        return None
    return path if path[0][0] < path[-1][0] else path[::-1]


def _fallback(length, above):
    if above:
        # A high interior corner, distinct endpoint rows, and a returning foot.
        segments = (((-1, 0), 1), ((0, 1), 4), ((1, 0), length-8), ((0, -1), 2))
        path = [(0, 0)]
        for (dr, dc), count in segments:
            for _ in range(count):
                path.append((path[-1][0] + dr, path[-1][1] + dc))
        return path
    # A variable-length serpentine chain with one empty row between rails.
    width = min(5, max(2, length // 3))
    path, direction = [(0, 0)], 1
    while len(path) < length:
        for _ in range(width - 1):
            if len(path) == length:
                return path
            path.append((path[-1][0], path[-1][1] + direction))
        for _ in range(2):
            if len(path) == length:
                return path
            path.append((path[-1][0] + 1, path[-1][1]))
        direction *= -1
    return path


def _orders(path):
    """Raster alternatives, including the column-sweep shortcut in B's review."""
    result = [sorted(path), sorted(path, reverse=True)]
    columns = sorted({c for _, c in path})
    for cs in (columns, columns[::-1]):
        for last_up in (False, True):
            result.append([p for c in cs for p in sorted(
                (p for p in path if p[1] == c), reverse=last_up and c == cs[-1])])
    return result


def _layout(rng, path, ph, pw, balanced, exhaustive=False):
    origin_r, origin_c = min(r for r, _ in path), min(c for _, c in path)
    path = [(r-origin_r, c-origin_c) for r, c in path]
    bh, bw = max(r for r, _ in path) + 1, max(c for _, c in path) + 1
    boards = [(nr, nc) for nr in range(3, 29 // (ph+1) + 1)
              for nc in range(3, 29 // (pw+1) + 1) if nr*ph >= bh and nc*pw >= bw]
    for i in rng.permutation(len(boards)):
        nr, nc = boards[int(i)]
        dh, dw = nr*ph-bh+1, nc*pw-bw+1
        total = dh * dw
        # Exhaustive fallback has at most 729 placements per board; ordinary
        # sampling tries eight, independent of how much blank fabric remains.
        choices = rng.permutation(total) if exhaustive else rng.choice(total, min(total, 8), replace=False)
        for index in choices:
            top, left = divmod(int(index), dw)
            placed = [(r+top, c+left) for r, c in path]
            groups = defaultdict(list)
            for r, c in placed:
                groups[(r//ph, c//pw)].append((r, c))
            if not 2 <= len(groups) < nr*nc/2:
                continue
            anchors = [v for v in groups.values() if len(v) >= 2 and len(v) % 2 == 0]
            if balanced and not anchors:
                continue
            anchor = anchors[int(rng.integers(len(anchors)))] if anchors else []
            return nr, nc, placed, anchor
    return None


def _geometry(rng, length, ph, pw, witness):
    for _ in range(192):
        path = _walk(rng, length)
        if path is None:
            continue
        if witness and min(r for r, _ in path) >= path[0][0]:
            continue
        if witness and any(path == order for order in _orders(path)):
            continue
        layout = _layout(rng, path, ph, pw, witness)
        if layout is not None:
            return layout
    path = _fallback(length, witness)
    # Mirror the constructive safety net without changing endpoint direction.
    if rng.random() < 0.5:
        path = [(r, -c) for r, c in path]
    layout = _layout(rng, path, ph, pw, witness, exhaustive=True)
    _require(layout is not None, "constructive thread did not fit a legal board")
    return layout


def _template(rng, ph, pw, colors):
    k = len(colors)
    rr, cc = np.indices((ph, pw))
    style = int(rng.integers(5))
    if style == 0:
        roles = rng.integers(k, size=(ph, pw))
    elif style == 1:
        roles = (rr + cc + int(rng.integers(k))) % k
    elif style == 2:
        roles = (rr + 2*cc + int(rng.integers(k))) % k
    elif style == 3:
        roles = np.minimum(np.minimum(rr, ph-1-rr), np.minimum(cc, pw-1-cc)) % k
    else:
        roles = (rr if rng.random() < 0.5 else cc) % k
    roles = roles.astype(int)
    # Every chosen interior role actually occurs, even on a small symmetric tile.
    for i, index in enumerate(rng.choice(ph*pw, k, replace=False)):
        roles.flat[int(index)] = i
    return np.asarray(colors, dtype=np.int8)[roles]


def _color_fabric(rng, nr, nc, path, anchor, ph, pw, colors, witness):
    # Reserve the chosen tile's whole changed set for a count-preserving
    # exchange. This makes an actual exceptional bitmap invisible to counting.
    balanced = bool(anchor) and (witness or rng.random() < 0.65)
    anchor_local = [(r % ph, c % pw) for r, c in anchor] if balanced else []
    available = [(r, c) for r in range(ph) for c in range(pw) if (r, c) not in anchor_local]
    k = min(len(colors), 2 + len(available)) if balanced else len(colors)
    colors = colors[:k]
    for _ in range(96):
        template = _template(rng, ph, pw, colors)
        if balanced:
            shuffled = list(anchor_local)
            rng.shuffle(shuffled)
            for i, p in enumerate(shuffled):
                template[p] = colors[i % 2]
            # Keep any extra palette roles outside the balanced exchange.
            for color, i in zip(colors[2:], rng.permutation(len(available))):
                template[available[int(i)]] = color
        baseline = np.tile(template, (nr, nc))
        raw = baseline.copy()
        for p in path:
            before = int(baseline[p])
            raw[p] = int(rng.choice([c for c in colors if c != before]))
        if balanced:
            for p in anchor:
                raw[p] = colors[1] if baseline[p] == colors[0] else colors[0]
        sequence = [int(raw[p]) for p in path]
        if witness:
            if sequence == sequence[::-1]:
                continue
            if any(sequence == [int(raw[p]) for p in order] for order in _orders(path)):
                continue
            runs = [c for i, c in enumerate(sequence) if i == 0 or c != sequence[i-1]]
            if len(runs) == len(set(runs)):
                continue
        return raw, np.asarray(sequence, dtype=np.int8).reshape(-1, 1)
    raise ValueError("could not color a distinguishing witness in the bounded budget")


def _constructive_witness(length, ph, pw, colors):
    # A finite safety net, checked for all 16 tile shapes and five witness
    # lengths in self_test. Color equality, not literal numeric order, controls
    # acceptance, so the same proof applies under every palette reassignment.
    fallback_rng = np.random.default_rng(0)
    layout = _layout(fallback_rng, _fallback(length, True), ph, pw, True, exhaustive=True)
    _require(layout is not None, "constructive witness placement")
    nr, nc, path, anchor = layout
    raw, output = _color_fabric(fallback_rng, nr, nc, path, anchor, ph, pw, colors, True)
    return nr, nc, raw, output


def _make_unique(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # At most thirteen prior inputs. With at least three distinct input colors,
    # these twenty cyclic/reflected color maps produce distinct candidates.
    for sign in (1, -1):
        for shift in range(10):
            lut = np.asarray([(sign*c + shift) % 10 for c in range(10)], dtype=np.int8)
            candidate = {role: lut[a] for role, a in pair.items()}
            if _key(candidate["input"]) not in seen:
                return candidate
    raise ValueError("finite uniqueness fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)),
             "seed must be an integer")
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(("length", "retile")))
    shapes = [tuple(map(int, x)) for x in rng.choice(_TILE_SHAPES, 3, replace=False)]
    witness_index = int(rng.integers(nt))
    companion = (witness_index + 1) % nt
    shared = bool(rng.random() < 0.25)
    puzzle_palette = list(map(int, rng.permutation(10)[:5]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            witness = split == "train" and index == witness_index
            if split == "train":
                shape_index = 0 if witness else 1 if index == companion else int(rng.integers(2))
                length = int(rng.integers(14, 19)) if witness else int(rng.integers(6, 19))
            else:
                shape_index = int(rng.integers(2)) if mode == "length" else 2
                length = int(rng.integers(20, 31)) if mode == "length" else int(rng.integers(10, 25))
            ph, pw = shapes[shape_index]
            nr, nc, path, anchor = _geometry(rng, length, ph, pw, witness)
            palette = puzzle_palette if shared else list(map(int, rng.permutation(10)[:5]))
            lattice = palette[0]
            k = 3 if witness else int(rng.integers(2, min(4, ph*pw) + 1))
            try:
                raw, output = _color_fabric(rng, nr, nc, path, anchor, ph, pw, palette[1:1+k], witness)
            except ValueError:
                if not witness:
                    raise
                nr, nc, raw, output = _constructive_witness(length, ph, pw, palette[1:4])
            pair = {"input": _assemble(raw, ph, pw, lattice), "output": output}
            pair = _make_unique(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    path, raw, baseline = info["path"], info["raw"], info["baseline"]
    sequence = [int(raw[p]) for p in path]
    runs = [c for i, c in enumerate(sequence) if i == 0 or c != sequence[i-1]]
    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    balanced = 0
    for group in info["groups"].values():
        balanced += Counter(int(raw[p]) for p in group) == Counter(int(baseline[p]) for p in group)
    return dict(length=len(path), turns=sum(a != b for a, b in zip(steps, steps[1:])),
                above=min(r for r, _ in path) < path[0][0], balanced=balanced,
                nonpalindrome=sequence != sequence[::-1],
                recurrence=len(runs) > len(set(runs)),
                defeats_raster=all(sequence != [int(raw[p]) for p in order] for order in _orders(path)),
                defeats_color_sort=all(sequence != sorted(sequence, reverse=x) for x in (False, True)),
                deletion=any(raw[p] == info["template"][0, 0] for p in path),
                uniform_background_fails=all(set(map(tuple, np.argwhere(raw != c))) != set(path)
                                             for c in info["palette"]),
                seams=sum(a[0]//info["ph"] != b[0]//info["ph"] or
                          a[1]//info["pw"] != b[1]//info["pw"] for a, b in zip(path, path[1:])),
                reentry=any(sum((r//info["ph"], c//info["pw"]) == tile and
                               (i == 0 or (path[i-1][0]//info["ph"], path[i-1][1]//info["pw"]) != tile)
                               for i, (r, c) in enumerate(path)) > 1 for tile in info["groups"]),
                signature=tuple((r-path[0][0], c-path[0][1]) for r, c in path))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            info = _unpack(pair["input"])
            output = _grid(pair["output"])
            expected = np.asarray([info["raw"][p] for p in info["path"]], dtype=np.int8).reshape(-1, 1)
            _require(np.array_equal(output, expected), "output must copy every altered color in path order")
            key = _key(info["grid"])
            _require(key not in seen, "inputs must be distinct")
            seen.add(key)
            if split == "train":
                train_features.append(_features(info))
    _require(any(f["above"] and f["balanced"] and f["nonpalindrome"] and f["recurrence"] and
                 f["uniform_background_fails"] and
                 f["defeats_raster"] and f["defeats_color_sort"] for f in train_features),
             "training must teach a balanced, non-raster, higher-endpoint thread")


def _digest(puzzle):
    serial = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Bounded, in-memory checks over the 42 specified host seeds."""
    shapes, tile_shapes, geometries, motifs, palettes, lengths, turns = (set() for _ in range(7))
    metrics = Counter()
    puzzle_hashes = set()
    # Solver-domain boundaries that the ordinary sampler does not need to use.
    for template, path, current_color in (
            (np.array([[1, 2], [2, 1]], dtype=np.int8), [(1, 1), (2, 1)], None),
            (np.array([[1, 2], [3, 1]], dtype=np.int8), [(0, 0), (1, 0), (2, 0)], 2)):
        raw = np.tile(template, (3, 3))
        for p in path:
            raw[p] = current_color if current_color is not None else 3-int(raw[p])
        inp = _assemble(raw, 2, 2, 0)
        expected = np.asarray([raw[p] for p in path], dtype=np.int8).reshape(-1, 1)
        _require(np.array_equal(solve(inp), expected), "short or monochrome boundary thread")
    for ph, pw in _TILE_SHAPES:
        for length in range(14, 19):
            nr, nc, raw, output = _constructive_witness(length, ph, pw, [2, 7, 0])
            inp = _assemble(raw, ph, pw, 9)
            info = _unpack(inp)
            f = _features(info)
            _require(f["above"] and f["balanced"] and f["defeats_raster"] and f["nonpalindrome"],
                     "constructive witness regression")
            _require(np.array_equal(solve(inp), output), "constructive readout regression")
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _digest(puzzle)
        _require(digest == _digest(generate_puzzle(seed)), "seed reproducibility")
        puzzle_hashes.add(digest)
        tr, te = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _unpack(pair["input"])
                f = _features(info)
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "input-only solve mismatch")
                shapes.add(info["grid"].shape)
                tile_shapes.add((info["ph"], info["pw"]))
                geometries.add((info["ph"], info["pw"], info["nr"], info["nc"]))
                motifs.add(_key(info["template"]))
                palettes.add((info["lattice"], tuple(sorted(info["palette"]))))
                lengths.add(f["length"])
                turns.add(f["turns"])
                metrics.update({name: int(f[name]) for name in ("above", "balanced", "reentry", "defeats_raster")})
                metrics["pairs"] += 1
                metrics[f"interior_colors_{len(info['palette'])}"] += 1
                metrics["max_seams"] = max(metrics["max_seams"], f["seams"])
                (tr if split == "train" else te).append(info)
        _require(len({(i["ph"], i["pw"]) for i in tr}) >= 2, "two training tessellations")
        _require(all(len(i["path"]) > max(len(t["path"]) for t in tr) for i in te) or
                 all((i["ph"], i["pw"]) not in {(t["ph"], t["pw"]) for t in tr} for i in te),
                 "test must extend length or use an unseen tile shape")
    _require(len(puzzle_hashes) == 42, "42 distinct puzzles")
    _require(len(shapes) >= 45 and len(geometries) >= 70, "board geometry diversity")
    _require(len(tile_shapes) == 16 and len(lengths) >= 20 and len(turns) >= 9,
             "tile, thread-length and folding diversity")
    _require(len(motifs) >= 130 and len(palettes) >= 90, "template and role diversity")
    _require(metrics["above"] >= 42 and metrics["balanced"] >= 42 and metrics["reentry"] >= 4,
             "meaningful difference and topology variation")
    _require(all(metrics[f"interior_colors_{k}"] > 0 for k in (2, 3, 4)), "palette-size variation")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1) + 1):
            p = generate_puzzle(73, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Malformed chains: tied endpoints, branch, cycle, disconnected extra chain.
    for cells in (((0, 0), (0, 1), (0, 2)),
                  ((0, 1), (1, 0), (1, 1), (1, 2)),
                  ((0, 0), (0, 1), (1, 0), (1, 1)),
                  ((0, 0), (1, 0), (4, 4), (5, 4))):
        try:
            _trace(cells)
        except ValueError:
            continue
        raise AssertionError("malformed thread accepted")
    base = generate_puzzle(0, 2, 1)
    for defect in ("output", "lattice", "hole", "duplicate"):
        bad = {s: [{k: a.copy() for k, a in p.items()} for p in base[s]] for s in base}
        pair = bad["test"][0]
        info = _unpack(pair["input"])
        if defect == "output":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif defect == "lattice":
            pair["input"][0, 0] = (info["lattice"] + 1) % 10
        elif defect == "hole":
            p = info["path"][len(info["path"]) // 2]
            raw = info["raw"].copy()
            raw[p] = info["baseline"][p]
            pair["input"] = _assemble(raw, info["ph"], info["pw"], info["lattice"])
        else:
            bad["test"][0] = {k: a.copy() for k, a in bad["train"][0].items()}
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError("corruption accepted: " + defect)
    return dict(seeds=42, distinct_puzzles=len(puzzle_hashes), input_shapes=len(shapes),
                tile_shapes=len(tile_shapes), board_geometries=len(geometries),
                templates=len(motifs), palettes=len(palettes), lengths=sorted(lengths),
                turn_counts=sorted(turns), **dict(metrics))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
