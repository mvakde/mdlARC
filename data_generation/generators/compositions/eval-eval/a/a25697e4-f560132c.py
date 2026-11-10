"""Mirror-stitched channels: joint packing, material exchange, corner flooding.

Only NumPy and the standard library are used. No files, parent code, seeds, or
generation metadata are consulted by solve(). Internal material labels 0/1 are
arbitrary and interchangeable; they are not literal ARC colors.
"""
from collections import Counter
import hashlib
import json
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10
            and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 / 1..4, with train > test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.add(p)
        result.append(cells)
    return result


def _crop(a, mask):
    cells = np.argwhere(mask)
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    return a[lo[0]:hi[0], lo[1]:hi[1]].copy(), tuple(map(int, lo))


def _transform(a, op):
    """0..3: rotations; 4..7: left/right reflection followed by rotation."""
    return np.rot90(np.fliplr(a) if op >= 4 else a, op % 4).copy()


def _border(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _parse(inp):
    grid = _grid(inp)
    border = _border(grid)
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(np.count_nonzero(border == bg) > border.size / 2, "unreadable background")
    comps = _components(grid != bg)
    _require(len(comps) == 4, "need four separated tiles")
    for i, a in enumerate(comps):
        halo = {(r + dr, c + dc) for r, c in a
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        _require(all(not (halo & b) for b in comps[i + 1:]), "tiles need a blank-cell moat")
    # The four singleton legend colors form a 2x2 patch surrounded by a
    # one-cell, monochrome 4x4 ring. The ring explicitly restores its material.
    legends = []
    for index, comp in enumerate(comps):
        count = Counter(int(grid[p]) for p in comp)
        for r, c in sorted(p for p in comp if count[int(grid[p])] == 1):
            patch = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
            if not patch <= comp or len({int(grid[p]) for p in patch}) != 4:
                continue
            if not all(count[int(grid[p])] == 1 for p in patch):
                continue
            ring = {(y, x) for y in range(r - 1, r + 3) for x in range(c - 1, c + 3)} - patch
            if not ring <= comp or len({int(grid[p]) for p in ring}) != 1:
                continue
            legends.append((index, r, c, int(grid[min(ring)])))
    _require(len(legends) == 1, "need one readable ringed legend")
    anchor, r, c, under = legends[0]
    key = grid[r:r + 2, c:c + 2].copy().reshape(-1)
    key_colors = set(map(int, key))
    pigments = sorted(set(map(int, np.unique(grid))) - key_colors - {bg})
    _require(len(key_colors) == 4 and len(pigments) == 2 and under in pigments,
             "legend, materials, and background must be disjoint")
    _require(all(sum(int(grid[p]) in key_colors for p in comp) == (4 if i == anchor else 0)
                 for i, comp in enumerate(comps)), "legend colors must be local singleton entries")
    material = np.full(grid.shape, -1, dtype=np.int8)
    for j, color in enumerate(pigments):
        material[grid == color] = j
    material[r:r + 2, c:c + 2] = pigments.index(under)
    order = [anchor] + [i for i in range(4) if i != anchor]
    tiles = []
    for i in order:
        mask = np.zeros(grid.shape, dtype=bool)
        for p in comps[i]:
            mask[p] = True
        tile, _ = _crop(np.where(mask, material, -1), mask)
        _require(np.all(tile[tile >= 0] <= 1), "unrecognized material")
        tiles.append(tile)
    total = sum(int(np.count_nonzero(t >= 0)) for t in tiles)
    side = int(total ** 0.5)
    _require(side * side == total, "total area must be square")
    return {"tiles": tiles, "side": side, "key": key, "background": bg,
            "pigments": pigments, "legend_location": (r, c),
            "legend_local": (r - min(p[0] for p in comps[anchor]),
                             c - min(p[1] for p in comps[anchor])), "input_shape": grid.shape}


def _channels(material):
    """Return corner ownership, or None if a source is missing or conflicts."""
    n = len(material)
    corners = ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1))
    groups = _components(material == 0) + _components(material == 1)
    if len(groups) != 4:
        return None
    owner = np.full(material.shape, -1, dtype=np.int8)
    for group in groups:
        hits = [i for i, p in enumerate(corners) if p in group]
        if len(hits) != 1:
            return None
        for p in group:
            owner[p] = hits[0]
    return owner if np.all(owner >= 0) else None


def _mask(a, side, row=0, col=0):
    value = 0
    for r, c in np.argwhere(a):
        value |= 1 << ((int(r) + row) * side + int(c) + col)
    return value


def _options(tile, side, corner, swap_policy=(0, 1)):
    corners = (0, side - 1, side * (side - 1), side * side - 1)
    result, seen = [], set()
    for op in range(8):
        a = _transform(tile, op)
        if swap_policy[int(op >= 4)]:
            a[a >= 0] = 1 - a[a >= 0]
        h, w = a.shape
        if h > side or w > side:
            continue
        row, col = (0 if corner < 2 else side - h), (0 if corner % 2 == 0 else side - w)
        mask = _mask(a >= 0, side, row, col)
        if [i for i, p in enumerate(corners) if mask >> p & 1] != [corner]:
            continue
        zero = _mask(a == 0, side, row, col)
        signature = (mask, zero)
        if signature not in seen:
            seen.add(signature)
            result.append({"mask": mask, "zero": zero, "array": a, "row": row,
                           "col": col, "op": op})
    return result


def _paint_material(side, moves, exchange=True):
    material = np.full((side, side), -1, dtype=np.int8)
    tile_owner = np.full((side, side), -1, dtype=np.int8)
    for corner, move in enumerate(moves):
        a = move["array"].copy()
        if not exchange and move["op"] >= 4:
            a[a >= 0] = 1 - a[a >= 0]
        r, c = move["row"], move["col"]
        mask = a >= 0
        material[r:r + len(a), c:c + a.shape[1]][mask] = a[mask]
        tile_owner[r:r + len(a), c:c + a.shape[1]][mask] = corner
    return material, tile_owner


def _reconstruct(analysis, swap_policy=(0, 1), required_output=None):
    tiles, n = analysis["tiles"], analysis["side"]
    anchor = tiles[0]
    _require(all(d <= n for d in anchor.shape) and anchor[0, 0] >= 0, "anchor misses top-left")
    amask = _mask(anchor >= 0, n)
    corners = (0, n - 1, n * (n - 1), n * n - 1)
    _require([i for i, p in enumerate(corners) if amask >> p & 1] == [0], "anchor corner ambiguity")
    first = {"mask": amask, "array": anchor, "row": 0, "col": 0, "op": 0, "piece": 0}
    options = {(p, q): _options(tiles[p], n, q, swap_policy) for p in range(1, 4) for q in range(1, 4)}
    full = (1 << (n * n)) - 1
    answers = {}
    packing_count = 0

    def visit(q, remaining, occupied, moves):
        nonlocal packing_count
        if len(answers) > 1 or (required_output is not None and answers):
            return
        if q == 4:
            if occupied != full:
                return
            packing_count += 1
            material, tile_owner = _paint_material(n, moves)
            channels = _channels(material)
            if channels is None:
                return
            out = analysis["key"][channels].astype(np.int8)
            if required_output is not None and not np.array_equal(out, required_output):
                return
            answers.setdefault(out.tobytes(), (out, moves, material, tile_owner, channels))
            return
        for p in remaining:
            for option in options[p, q]:
                if option["mask"] & occupied:
                    continue
                move = dict(option, piece=p)
                visit(q + 1, tuple(x for x in remaining if x != p), occupied | move["mask"], moves + [move])

    visit(1, (1, 2, 3), amask, [first])
    _require(bool(answers), "no channel-valid packing")
    _require(len(answers) == 1, "multiple distinct colored assemblies")
    out, moves, material, tile_owner, channels = next(iter(answers.values()))
    return dict(analysis, output=out, moves=moves, material=material,
                tile_owner=tile_owner, channels=channels, geometric_packings=packing_count)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible tiles, their material patterns, and their legend only."""
    return _reconstruct(_parse(input_grid))["output"].copy()


def _partition(rng, n, y, x, protected, safe_material):
    owner = np.empty((n, n), dtype=np.int8)
    owner[:y, :x], owner[:y, x:], owner[y:, :x], owner[y:, x:] = 0, 1, 2, 3
    corners = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}
    areas = [int(np.count_nonzero(owner == j)) for j in range(4)]
    target = int(rng.integers(n, 4 * n + 1))
    changes = 0
    for _ in range(5 * n * n):
        r, c = int(rng.integers(n)), int(rng.integers(n))
        if (r, c) in protected or (r, c) in corners:
            continue
        old = int(owner[r, c])
        neighbors = [(r + dr, c + dc) for dr, dc in DIR4
                     if 0 <= r + dr < n and 0 <= c + dc < n]
        choices = sorted({int(owner[p]) for p in neighbors} - {old})
        if not choices or areas[old] <= 4:
            continue
        new = choices[int(rng.integers(len(choices)))]
        if safe_material and any(owner[p] == (new ^ 3) for p in neighbors):
            continue
        owner[r, c] = new
        if len(_components(owner == old)) != 1:
            owner[r, c] = old
            continue
        areas[old] -= 1
        areas[new] += 1
        changes += 1
        if changes >= target:
            break
    return owner


def _geometry(rng, n):
    cuts = [int(rng.integers(4, n - 1)) for _ in range(4)]
    pr = int(rng.integers(min(cuts[0], cuts[2]) - 3))
    pc = int(rng.integers(min(cuts[1], cuts[3]) - 3))
    protected = {(r, c) for r in range(pr, pr + 4) for c in range(pc, pc + 4)}
    pieces = _partition(rng, n, cuts[0], cuts[1], protected, False)
    channels = _partition(rng, n, cuts[2], cuts[3], protected, True)
    material = ((channels // 2) ^ (channels % 2)).astype(np.int8)
    _require(_channels(material) is not None, "construction lost four corner channels")
    # Independently chosen cuts must produce channels spanning tile seams.
    mixed = sum(len(set(material[pieces == j].tolist())) == 2 for j in range(4))
    _require(mixed >= 2, "need tiles carrying more than one output region")
    return pieces, channels, material, (pr + 1, pc + 1)


def _scatter(rng, tiles):
    # Random shelf partitions include rows, columns, 3+1, and uneven 2+2.
    # A 2x2 fallback fits: 2*13 + one gap + two blank borders = 29.
    for attempt in range(70):
        transpose = bool(rng.integers(2))
        dims = [tuple(reversed(t.shape)) if transpose else t.shape for t in tiles]
        order = list(map(int, rng.permutation(4)))
        if attempt == 69:
            groups, gap = [order[:2], order[2:]], 1
        else:
            groups, current = [], []
            for i, p in enumerate(order):
                current.append(p)
                if i == 3 or rng.random() < 0.45:
                    groups.append(current)
                    current = []
            gap = int(rng.integers(1, 4))
        heights = [max(dims[p][0] for p in g) for g in groups]
        widths = [sum(dims[p][1] for p in g) + gap * (len(g) - 1) for g in groups]
        h, w = sum(heights) + gap * (len(groups) - 1), max(widths)
        if h > 28 or w > 28:
            continue
        offsets, r = [None] * 4, 0
        for group, gh, gw in zip(groups, heights, widths):
            c = int(rng.integers(w - gw + 1))
            for p in group:
                rr = r + int(rng.integers(gh - dims[p][0] + 1))
                offsets[p] = (c, rr) if transpose else (rr, c)
                c += dims[p][1] + gap
            r += gh + gap
        if transpose:
            h, w = w, h
        pads = []
        for length in (h, w):
            before = int(rng.integers(1, min(4, 30 - length)))
            after = int(rng.integers(1, min(4, 31 - length - before)))
            pads.append((before, after))
        offsets = [(r + pads[0][0], c + pads[1][0]) for r, c in offsets]
        return offsets, (h + sum(pads[0]), w + sum(pads[1]))
    raise AssertionError("bounded shelf placement exhausted")


def _witnesses(a):
    """Pixel-derived ablations and measures, independent of sampled schedules."""
    material, tile_owner, channels = a["material"], a["tile_owner"], a["channels"]
    without_exchange, _ = _paint_material(a["side"], a["moves"], exchange=False)
    wrong_channels = _channels(without_exchange)
    parity_matters = wrong_channels is None or not np.array_equal(wrong_channels, channels)
    whole_piece_wrong = not np.array_equal(tile_owner, channels)
    crossing = sum(len(set(tile_owner[channels == q].tolist())) > 1 for q in range(4))
    mixed = sum(len(set(channels[tile_owner == q].tolist())) > 1 for q in range(4))
    seam_edges = sum(tile_owner[r, c] != tile_owner[r + dr, c + dc]
                     and material[r, c] == material[r + dr, c + dc]
                     for r in range(a["side"]) for c in range(a["side"])
                     for dr, dc in ((1, 0), (0, 1))
                     if r + dr < a["side"] and c + dc < a["side"])
    return {"parity_matters": parity_matters, "whole_piece_wrong": whole_piece_wrong,
            "crossing_channels": crossing, "mixed_tiles": mixed, "seam_edges": int(seam_edges),
            "reflections": sum(m["op"] >= 4 for m in a["moves"])}


def _make_pair(rng, side, reflected_count, rotations, forced_reflected=None):
    for _ in range(140):
        try:
            pieces, channel_owner, material, legend = _geometry(rng, side)
            palette = list(map(int, rng.permutation(10)))
            bg, pigments, key = palette[0], palette[1:3], np.array(palette[3:7], dtype=np.int8)
            ops = list(rotations)
            reflected = (list(forced_reflected) if forced_reflected is not None else
                         list(map(int, rng.choice([1, 2, 3], reflected_count, replace=False))))
            for p in reflected:
                ops[p] += 4
            tiles, anchor_origin = [], None
            for q in range(4):
                tile, origin = _crop(np.where(pieces == q, material, -1), pieces == q)
                if q == 0:
                    anchor_origin = origin
                tile = _transform(tile, ops[q])
                if ops[q] >= 4:
                    tile[tile >= 0] = 1 - tile[tile >= 0]
                tiles.append(tile)
            offsets, shape = _scatter(rng, tiles)
            inp = np.full(shape, bg, dtype=np.int8)
            for tile, (r, c) in zip(tiles, offsets):
                dest = inp[r:r + len(tile), c:c + tile.shape[1]]
                for k in (0, 1):
                    dest[tile == k] = pigments[k]
            kr = offsets[0][0] + legend[0] - anchor_origin[0]
            kc = offsets[0][1] + legend[1] - anchor_origin[1]
            inp[kr:kr + 2, kc:kc + 2] = key.reshape(2, 2)
            out = key[channel_owner].astype(np.int8)
            a = _reconstruct(_parse(inp))
            w = _witnesses(a)
            if (np.array_equal(a["output"], out) and w["parity_matters"]
                    and w["whole_piece_wrong"] and w["crossing_channels"] >= 2
                    and w["mixed_tiles"] >= 2 and w["reflections"] == reflected_count):
                return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise AssertionError("bounded geometry/uniqueness construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # One reflection per demonstration; tests combine two or three. Scale can
    # grow or shrink, so more simultaneous exchanges are the invariant shift.
    grow = bool(rng.integers(2))
    puzzle, seen, first_key = {"train": [], "test": []}, set(), None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            lo, hi = ((8, 11) if (split == "train") == grow else (11, 14))
            side = int(rng.integers(lo, hi))
            reflections = 1 if split == "train" else int(rng.integers(2, 4))
            # Across the first two demonstrations, all four rotation amounts
            # are seen before composition with reflection is needed at test.
            rotations = ([0, 0, 1, 2] if i == 0 else [0, 1, 2, 3]) if split == "train" else [0] + list(map(int, rng.integers(0, 4, 3)))
            for _ in range(8):
                forced = ([1] if i == 0 else [2]) if split == "train" and i < 2 else None
                pair = _make_pair(rng, side, reflections, rotations, forced)
                signature = (pair["input"].shape, pair["input"].tobytes())
                key = tuple(map(int, _parse(pair["input"])["key"]))
                if split == "train" and i == 1 and key == first_key:
                    continue
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    if split == "train" and i == 0:
                        first_key = key
                    break
            else:
                raise AssertionError("bounded duplicate rejection exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_keys, sides = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = (inp.shape, inp.tobytes())
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            a = _reconstruct(_parse(inp))
            _require(np.array_equal(out, a["output"]), "incorrect output")
            _require(set(map(int, np.unique(out))) == set(map(int, a["key"])), "four source colors required")
            w = _witnesses(a)
            _require(w["parity_matters"], "exchange ablation does not change channels")
            _require(w["whole_piece_wrong"] and w["mixed_tiles"] >= 2
                     and w["crossing_channels"] >= 2 and w["seam_edges"] > 0,
                     "must route color across tile boundaries")
            _require(w["reflections"] == 1 if split == "train" else w["reflections"] in (2, 3),
                     "missing demonstrated-to-combined reflection shift")
            sides[split].append(a["side"])
            if split == "train":
                train_keys.add(tuple(map(int, a["key"])))
    _require(len(train_keys) >= 2, "legend must change in training")
    _require(max(sides["train"]) < min(sides["test"])
             or max(sides["test"]) < min(sides["train"]), "missing scale transfer")
    # Label compatibility, not failure of an arbitrarily selected alternative
    # placement: search for ANY assembly explaining all demonstrations under
    # each competing parity-to-exchange table.
    compatible = []
    for policy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for pair in puzzle["train"]:
            try:
                _reconstruct(_parse(pair["input"]), policy, np.asarray(pair["output"]))
            except AssertionError:
                break
        else:
            compatible.append(policy)
    _require(compatible == [(0, 1)], "training does not identify reflection-only exchange")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, in-memory 42-seed semantic/variation checks; no saved grids."""
    signatures, shapes, sides, backgrounds, legends = set(), set(), set(), set(), set()
    piece_shapes, channel_shapes, operations, regimes, crossings = set(), set(), set(), set(), set()
    local_legends, mixed_counts, area_ratios, fills = set(), set(), [], []
    multisolutions = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_digest(p))
        tr = [pair["output"].shape[0] for pair in p["train"]]
        te = [pair["output"].shape[0] for pair in p["test"]]
        regimes.add("grow" if min(te) > max(tr) else "shrink")
        for split in ("train", "test"):
            for pair in p[split]:
                a = _reconstruct(_parse(pair["input"]))
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "input-only solve failed")
                shapes.add(pair["input"].shape)
                sides.add(a["side"])
                backgrounds.add(a["background"])
                legends.add(a["legend_location"])
                local_legends.add(a["legend_local"])
                operations.update(m["op"] for m in a["moves"])
                crossings.add(_witnesses(a)["crossing_channels"])
                mixed_counts.add(_witnesses(a)["mixed_tiles"])
                areas = [int(np.count_nonzero(t >= 0)) for t in a["tiles"]]
                area_ratios.append(max(areas) / min(areas))
                fills.extend(float(np.count_nonzero(t >= 0) / t.size) for t in a["tiles"])
                multisolutions += a["geometric_packings"] > 1
                piece_shapes.add(a["tile_owner"].tobytes())
                channel_shapes.add(a["channels"].tobytes())
    _require(len(signatures) == 42, "seeds must vary the whole puzzle")
    _require(len(shapes) >= 25 and sides == set(range(8, 14)), "size diversity")
    _require(backgrounds == set(range(10)) and len(legends) >= 25, "palette/legend diversity")
    _require(len(piece_shapes) >= 190 and len(channel_shapes) >= 190, "structural diversity")
    _require(operations == set(range(8)) and regimes == {"grow", "shrink"}, "operation/regime diversity")
    _require(max(crossings) == 4, "channels must sometimes traverse all tile neighborhoods")
    p = generate_puzzle(91, 2, 1)
    _require(_digest(p) == _digest(generate_puzzle(91, 2, 1)), "nondeterministic")
    fixture = p["test"][0]
    a = _parse(fixture["input"])
    r, c = a["legend_location"]
    changed = fixture["input"].copy()
    new_key = np.roll(a["key"], 1)
    changed[r:r + 2, c:c + 2] = new_key.reshape(2, 2)
    expected = fixture["output"].copy()
    for old, new in zip(a["key"], new_key):
        expected[fixture["output"] == old] = new
    _require(not np.array_equal(expected, fixture["output"])
             and np.array_equal(solve(changed), expected), "legend counterfactual failed")
    changed = fixture["input"].copy()
    u, v = a["pigments"]
    changed[fixture["input"] == u], changed[fixture["input"] == v] = v, u
    _require(np.array_equal(solve(changed), fixture["output"]), "material names must be interchangeable")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "square_sides": sorted(sides), "piece_partitions": len(piece_shapes),
            "channel_partitions": len(channel_shapes), "operations": sorted(operations),
            "backgrounds": sorted(backgrounds), "legend_locations": len(legends),
            "local_legend_locations": len(local_legends), "mixed_tile_counts": sorted(mixed_counts),
            "crossing_channel_counts": sorted(crossings), "max_piece_area_ratio": max(area_ratios),
            "min_piece_bbox_fill": min(fills), "channel_disambiguated_pairs": multisolutions}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
