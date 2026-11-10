"""Count-expanded stencils compete for shaped molds.

Only NumPy and the standard library are used.  No parent module, example,
seed, or generation metadata is consulted by solve().
"""
from __future__ import annotations

import numpy as np

_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= x <= 30 for x in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in (_D8 if diagonal else _D4):
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _bbox(cells):
    a = np.array(sorted(cells), dtype=int)
    r0, c0 = a.min(axis=0)
    r1, c1 = a.max(axis=0) + 1
    return int(r0), int(r1), int(c0), int(c1)


def _crop(mask):
    r0, r1, c0, c1 = _bbox(set(map(tuple, np.argwhere(mask))))
    return mask[r0:r1, c0:c1].copy(), (r0, r1, c0, c1)


def _transform(a, op):
    # First reflect left/right for operations 4..7; then rotate counterclockwise.
    return np.rot90(a[:, ::-1] if op >= 4 else a, op % 4).copy()


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _orbit(mask):
    return {_key(_transform(mask, op)) for op in range(8)}


def _no_holes(mask):
    return len(_components(~np.pad(mask, 1, constant_values=False))) == 1


def _expand(tile, count, axis, primary=None, secondary=None, local_swap=True):
    geometry_only = primary is None and secondary is None
    shape = list(tile.shape)
    shape[axis] = 1
    gap = np.full(shape, False if geometry_only else secondary,
                  dtype=bool if geometry_only else np.int8)
    pieces = []
    for i in range(count):
        if i:
            pieces.append(gap)
        # Copies alternately face opposite sides of the ribbon.  A local
        # reflection exchanges the colors by the same parity rule as a global fit.
        mask = np.flip(tile, axis=1 - axis) if i % 2 else tile
        if geometry_only:
            unit = mask
        else:
            p, q = (secondary, primary) if i % 2 and local_swap else (primary, secondary)
            unit = np.where(mask, p, q).astype(np.int8)
        pieces.append(unit)
    return np.concatenate(pieces, axis=axis)


def _parse(g):
    """Recover the sole two-color source, framed molds, and marker-9 tokens."""
    sources, molds, tokens = [], [], []
    for cells in _components(g != 0):
        colors = tuple(sorted({int(g[p]) for p in cells}))
        r0, r1, c0, c1 = box = _bbox(cells)
        block = g[r0:r1, c0:c1]
        if colors == (9,):
            tokens.append(cells)
        elif len(colors) == 2 and 9 not in colors and np.all(block != 0):
            if any(len(_components(block == c)) != 1 for c in colors):
                raise ValueError("each source color must form one connected region")
            sources.append({"cells": cells, "box": box, "block": block, "colors": colors})
        elif len(colors) == 1 and colors[0] != 9:
            color = colors[0]
            if (min(block.shape) < 4 or not np.all(block[0] == color)
                    or not np.all(block[-1] == color) or not np.all(block[:, 0] == color)
                    or not np.all(block[:, -1] == color)
                    or not np.any(block[1:-1, 1:-1] == 0)):
                raise ValueError("a mold needs a complete one-cell outer frame and vacancies")
            molds.append({"box": box, "color": color, "slot": block[1:-1, 1:-1] == 0})
        else:
            raise ValueError("unrecognized component")
    if len(sources) != 1 or not 2 <= len(molds) <= 3 or not 1 <= len(tokens) <= 6:
        raise ValueError("expected one source, two or three molds, and one to six tokens")
    source = sources[0]
    if (len({m["color"] for m in molds}) != 1
            or molds[0]["color"] in source["colors"]):
        raise ValueError("mold ink must be a separate, shared color")
    # All objects have a moat; diagonal contact must not redefine the token count.
    if len(_components(g == 9, True)) != len(tokens):
        raise ValueError("counter objects touch diagonally")
    all_boxes = [source["box"]] + [m["box"] for m in molds] + [_bbox(t) for t in tokens]
    for i, a in enumerate(all_boxes):
        for b in all_boxes[i + 1:]:
            if not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]):
                raise ValueError("object bounding boxes need at least one blank row or column")
    tiles = []
    for color in source["colors"]:
        tile, crop_box = _crop(source["block"] == color)
        if min(tile.shape) < 2 or tile.all() or not _no_holes(tile):
            raise ValueError("source crops must have two-dimensional, non-solid, hole-free shapes")
        tiles.append({"tile": tile, "color": color, "crop_box": crop_box})
    if any(len(_orbit(t["tile"])) != 8 for t in tiles) or _orbit(tiles[0]["tile"]) & _orbit(tiles[1]["tile"]):
        raise ValueError("source color shapes must be asymmetric and noncongruent")
    return {"source": source, "tiles": tiles, "molds": molds, "tokens": tokens,
            "count": len(tokens), "area": sum(map(len, tokens))}


def _matches(d, count=None):
    """Exact whole-ribbon fits; neither output labels nor seed are available."""
    n = d["count"] if count is None else count
    found = {}
    for ti, t in enumerate(d["tiles"]):
        for axis in (0, 1):
            axial = n * (t["tile"].shape[axis] + 1) - 1
            if axial > 28:
                continue
            mask = _expand(t["tile"], n, axis)
            for op in range(8):
                fitted = _transform(mask, op)
                for mi, mold in enumerate(d["molds"]):
                    if np.array_equal(fitted, mold["slot"]):
                        # count=1 has two equivalent axis descriptions.
                        signature = (mi, ti, op, _key(fitted))
                        found.setdefault(signature, {"mold": mi, "tile": ti, "axis": axis,
                                                      "op": op, "count": n})
    return list(found.values())


def _paint(g, d, move, swap_policy=(False, True), local_swap=True):
    out = g.copy()
    r0, r1, c0, c1 = d["source"]["box"]
    out[r0:r1, c0:c1] = 0
    for token in d["tokens"]:
        for p in token:
            out[p] = 0
    t = d["tiles"][move["tile"]]
    primary = t["color"]
    secondary = next(c for c in d["source"]["colors"] if c != primary)
    if swap_policy[int(move["op"] >= 4)]:
        primary, secondary = secondary, primary
    ribbon = _transform(_expand(t["tile"], move["count"], move["axis"], primary, secondary,
                                local_swap=local_swap), move["op"])
    r0, r1, c0, c1 = d["molds"][move["mold"]]["box"]
    out[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = ribbon
    return out


def _decode(g):
    d = _parse(g)
    matches = _matches(d)
    outputs = {}
    for move in matches:
        out = _paint(g, d, move)
        outputs.setdefault(_key(out), (out, move))
    if len(outputs) != 1:
        raise ValueError("the counted ribbon must select one unique colored mold filling")
    out, move = next(iter(outputs.values()))
    return out, d, move


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the uniquely matched print using information in the input alone."""
    return _decode(_grid(input_grid))[0]


def _grow(h, w, area, rng):
    m = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        m[r, c] = True
        frontier.update((r + dr, c + dc) for dr, dc in _D4
                        if 0 <= r + dr < h and 0 <= c + dc < w and not m[r + dr, c + dc])
    return m


def _source_mask(rng, small=False):
    for _ in range(400):
        h = int(rng.integers(3, 5 if small else 7))
        w = int(rng.integers(3, 6 if small else 8))
        mask = _grow(h, w, int(rng.integers(4, h * w - 3)), rng)
        if len(_components(~mask)) != 1:
            continue
        a, _ = _crop(mask)
        b, _ = _crop(~mask)
        if (min(a.shape) >= 2 and min(b.shape) >= 2 and len(_orbit(a)) == len(_orbit(b)) == 8
                and not (_orbit(a) & _orbit(b)) and _no_holes(a) and _no_holes(b)
                and (a.shape != mask.shape or b.shape != mask.shape)):
            return mask
    # Finite, valid asymmetric fallback; both regions have different exact shapes.
    return np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)


def _token_masks(rng, n, total=None):
    if total is None:
        sizes = [int(rng.integers(1, 10)) for _ in range(n)]
    else:
        sizes = [1] * n
        for _ in range(total - n):
            possible = [i for i, v in enumerate(sizes) if v < 9]
            sizes[possible[int(rng.integers(len(possible)))]] += 1
    masks = []
    for area in sizes:
        style = int(rng.integers(4))
        if style == 0 and area <= 7:
            mask = np.ones((1, area), dtype=bool)
        elif style == 1 and area == 8:
            mask = np.ones((3, 3), dtype=bool)
            mask[1, 1] = False
        else:
            mask = _grow(3, 3, area, rng)
        mask, _ = _crop(mask)
        masks.append(_transform(mask, int(rng.integers(8))))
    return masks


def _pack(pieces, rng):
    """Reserve complete bounding rectangles plus a moat, with no pairing bands."""
    mh, mw = max(p.shape[0] for p in pieces), max(p.shape[1] for p in pieces)
    total_area = sum(p.size for p in pieces)
    for attempt in range(32):
        if attempt < 8:
            h, w = int(rng.integers(max(12, mh), 31)), int(rng.integers(max(12, mw), 31))
            if h * w < total_area * 1.35:
                continue
        else:
            h = w = 30
        occupied = np.zeros((h, w), dtype=bool)
        origins = [None] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: -pieces[i].size + rng.random())
        for i in order:
            ph, pw = pieces[i].shape
            options = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            rng.shuffle(options)
            if rng.random() < 0.4:
                options.sort(key=lambda p: not (p[0] in (0, h - ph) or p[1] in (0, w - pw)))
            for r, c in options:
                if occupied[r:r + ph, c:c + pw].any():
                    continue
                origins[i] = r, c
                occupied[max(0, r - 1):min(h, r + ph + 1), max(0, c - 1):min(w, c + pw + 1)] = True
                break
            else:
                break
        else:
            g = np.zeros((h, w), dtype=np.int8)
            for piece, (r, c) in zip(pieces, origins):
                g[r:r + piece.shape[0], c:c + piece.shape[1]] = piece
            # Independently crop empty margins, retaining some zero padding.
            r0, r1, c0, c1 = _bbox(set(map(tuple, np.argwhere(g != 0))))
            margins = rng.integers(0, 4, size=4)
            r0, r1 = max(0, r0 - int(margins[0])), min(h, r1 + int(margins[1]))
            c0, c1 = max(0, c0 - int(margins[2])), min(w, c1 + int(margins[3]))
            return g[r0:r1, c0:c1].copy()
    return None


def _mold(mask, ink):
    # Connected secondary space and the frame make one monochrome mold object.
    return np.pad(np.where(mask, 0, ink).astype(np.int8), 1, constant_values=ink)


def _shape_decoy(unit, forbidden, rng):
    """Keep area and bounding dimensions while changing exact vacancy geometry."""
    ones, zeros = np.argwhere(unit), np.argwhere(~unit)
    for _ in range(160):
        candidate = unit.copy()
        a = tuple(ones[int(rng.integers(len(ones)))])
        b = tuple(zeros[int(rng.integers(len(zeros)))])
        candidate[a], candidate[b] = False, True
        if (len(_components(candidate)) == 1 and _no_holes(candidate)
                and _crop(candidate)[0].shape == unit.shape
                and not any(_orbit(candidate) & orbit for orbit in forbidden)):
            return candidate
    return None


def _make_pair(rng, n, palette, witness=None, parity=None, source=None, region=None):
    for attempt in range(100):
        raw = source if source is not None else _source_mask(rng, small=n >= 4 or attempt >= 70)
        p, q, ink = palette
        ri = int(rng.integers(2)) if region is None else region
        unit, _ = _crop(raw if ri == 0 else ~raw)
        axes = [axis for axis in (0, 1) if n * (unit.shape[axis] + 1) - 1 <= 26]
        if not axes:
            continue
        axis = axes[int(rng.integers(len(axes)))]
        op = int(rng.integers(4)) + 4 * (int(rng.integers(2)) if parity is None else parity)
        main = _mold(_transform(_expand(unit, n, axis), op), ink)
        wrong_counts = [k for k in range(1, 7) if k != n and k * (unit.shape[axis] + 1) - 1 <= 26]
        # First witnesses explicitly exchange counts 2 and 3 at equal ink area.
        wrong_n = 5 - n if witness is not None else wrong_counts[int(rng.integers(len(wrong_counts)))]
        decoy_op = int(rng.integers(8))
        decoy = _mold(_transform(_expand(unit, wrong_n, axis), decoy_op), ink)
        pieces = [np.where(raw, p, q).astype(np.int8), main, decoy]
        if witness is not None or (attempt < 70 and rng.random() < 0.4):
            other, _ = _crop(~raw if ri == 0 else raw)
            changed = _shape_decoy(unit, (_orbit(unit), _orbit(other)), rng)
            if changed is None:
                continue
            # Identical count, vacancy area, interior dimensions and macro orientation.
            # Only the small shape differs, so counting holes or measuring size fails.
            pieces.append(_mold(_transform(_expand(changed, n, axis), op), ink))
        pieces += [np.where(t, 9, 0).astype(np.int8) for t in _token_masks(rng, n, 12 if witness is not None else None)]
        g = _pack(pieces, rng)
        if g is None:
            continue
        try:
            predicted, d, move = _decode(g)
        except ValueError:
            continue
        if witness is not None and move["mold"] != witness:
            continue
        if d["tiles"][move["tile"]]["color"] != (p if ri == 0 else q) or move["op"] != op:
            continue
        # Construct the label directly from the chosen geometry; solve above is a check.
        out = g.copy()
        r0, r1, c0, c1 = d["source"]["box"]
        out[r0:r1, c0:c1] = 0
        out[g == 9] = 0
        primary, secondary = (p, q) if ri == 0 else (q, p)
        if op >= 4:
            primary, secondary = secondary, primary
        colored = _transform(_expand(unit, n, axis, primary, secondary), op)
        r0, r1, c0, c1 = d["molds"][move["mold"]]["box"]
        out[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = colored
        _require(np.array_equal(predicted, out), "constructed print disagrees with input reconstruction")
        return {"input": g, "output": out}
    raise RuntimeError("bounded distinct-mold construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("counts require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    stable = rng.random() < 0.3
    palette = tuple(map(int, rng.choice(np.arange(1, 9), 3, replace=False)))
    shared_source = _source_mask(rng, small=True)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", num_train), ("test", num_test)):
        for index in range(amount):
            witness = split == "train" and index < 2
            n = index + 2 if witness else int(rng.integers(1, 4) if split == "train" else rng.integers(4, 7))
            pair_palette = palette if stable or witness else tuple(map(int, rng.choice(np.arange(1, 9), 3, replace=False)))
            for _ in range(12):
                pair = _make_pair(rng, n, pair_palette, index if witness else None,
                                  index if witness else None, shared_source if witness else None,
                                  index if witness else None)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, training = set(), []
    policies = {(p, local) for p in ((False, False), (False, True), (True, False), (True, True))
                for local in (False, True)}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(g.shape == out.shape and _key(g) not in seen, "shape or duplicate input")
            seen.add(_key(g))
            predicted, d, move = _decode(g)
            _require(np.array_equal(predicted, out), "wrong output")
            _require(not np.array_equal(g, out), "identity")
            _require(not np.any(out == 9), "consumed counter tokens remain")
            # Verify every printed cell through inverse transformation and modular indexing,
            # without using concatenation or generation metadata.
            r0, r1, c0, c1 = d["molds"][move["mold"]]["box"]
            printed = out[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
            inv = (0, 3, 2, 1, 4, 5, 6, 7)[move["op"]]
            unit_frame = _transform(printed, inv)
            tile = d["tiles"][move["tile"]]
            a = tile["tile"].shape[move["axis"]]
            primary = tile["color"]
            secondary = next(c for c in d["source"]["colors"] if c != primary)
            if move["op"] >= 4:
                primary, secondary = secondary, primary
            for r in range(unit_frame.shape[0]):
                for c in range(unit_frame.shape[1]):
                    axial_coord = (r, c)[move["axis"]]
                    coord = axial_coord % (a + 1)
                    copy_index = axial_coord // (a + 1)
                    at = (coord, c) if move["axis"] == 0 else (r, coord)
                    local = tile["tile"]
                    p, q = primary, secondary
                    if copy_index % 2:
                        local = np.flip(local, axis=1 - move["axis"])
                        p, q = q, p
                    wanted = secondary if coord == a else p if local[at] else q
                    _require(unit_frame[r, c] == wanted, "tile phase, separator, or recoloring")
            # A wrong-count near match must exist in a different destination.
            alternatives = [m for k in range(1, 7) if k != d["count"] for m in _matches(d, k)]
            _require(any(m["mold"] != move["mold"] for m in alternatives), "counter has no selection effect")
            if split == "train":
                training.append((g, out, d, move))
                policies = {(p, local) for p, local in policies
                            if np.array_equal(_paint(g, d, move, p, local), out)}
    _require(policies == {((False, True), True)},
             "training must demonstrate local and global reflection swapping")
    _require(len({d["count"] for _, _, d, _ in training}) >= 2, "training count evidence")
    _require(any(d["area"] > d["count"] for _, _, d, _ in training), "pixel-count shortcut")
    _require(any(a[2]["area"] == b[2]["area"] and a[2]["count"] != b[2]["count"]
                 for a in training for b in training), "equal-area/different-count witness")
    _require(any(d["tiles"][m["tile"]]["tile"].shape != d["source"]["block"].shape
                 for _, _, d, m in training), "tight-cropping witness")
    _require(any(any(i != m["mold"] and len(_components(mold["slot"])) == d["count"]
                     and mold["slot"].shape == d["molds"][m["mold"]]["slot"].shape
                     and int(mold["slot"].sum()) == int(d["molds"][m["mold"]]["slot"].sum())
                     for i, mold in enumerate(d["molds"]))
                 for _, _, d, m in training), "same-count/area/dimensions shape decoy")
    _require({m["mold"] for _, _, _, m in training} >= {0, 1}, "fixed scan-order shortcut")
    # The same physical two-color source appears with each color chosen as ink.
    _require(any(_key(a[2]["source"]["block"]) == _key(b[2]["source"]["block"])
                 and a[2]["tiles"][a[3]["tile"]]["color"] != b[2]["tiles"][b[3]["tile"]]["color"]
                 for a in training for b in training), "shape-selected ink role witness")


def self_test():
    """Bounded 42-seed semantics/diversity checks plus small negative fixtures."""
    puzzles, structures, source_shapes, tile_shapes, canvases, colors = set(), set(), set(), set(), set(), set()
    counts, axes, operations, mold_counts, token_shapes, region_areas = set(), set(), set(), set(), set(), set()
    crop_witnesses = edges = unequal_tokens = 0
    first = None
    for seed in range(42):
        p = generate_puzzle(seed)
        if first is None:
            first = p
        puzzles.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
        for s in ("train", "test"):
            for pair in p[s]:
                out, d, m = _decode(pair["input"])
                _require(np.array_equal(out, pair["output"]), "self-test solve")
                counts.add(d["count"])
                axes.add(m["axis"])
                operations.add(m["op"])
                mold_counts.add(len(d["molds"]))
                canvases.add(pair["input"].shape)
                colors.add(d["source"]["colors"] + (d["molds"][0]["color"],))
                source_shapes.add(d["source"]["block"].shape)
                tile = d["tiles"][m["tile"]]["tile"]
                tile_shapes.add(tile.shape)
                region_areas.add(int(tile.sum()))
                structures.add(min(_orbit(tile)))
                crop_witnesses += tile.shape != d["source"]["block"].shape
                unequal_tokens += len({len(t) for t in d["tokens"]}) > 1
                for t in d["tokens"]:
                    r0, r1, c0, c1 = _bbox(t)
                    token_shapes.add(_key(pair["input"][r0:r1, c0:c1] == 9))
                for mold in d["molds"]:
                    r0, r1, c0, c1 = mold["box"]
                    edges += r0 == 0 or c0 == 0 or r1 == out.shape[0] or c1 == out.shape[1]
        if seed in (0, 17, 41):
            q = generate_puzzle(seed)
            _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s])
                         for k in ("input", "output")), "determinism")
    _require(len(puzzles) == 42 and len(structures) >= 50, "structural variation")
    _require(len(canvases) >= 25 and len(source_shapes) >= 8 and len(tile_shapes) >= 8, "geometry variation")
    _require(counts == set(range(1, 7)) and axes == {0, 1} and operations == set(range(8)), "count/transform variety")
    _require(mold_counts == {2, 3} and len(token_shapes) >= 20 and len(colors) >= 25, "object and palette variation")
    _require(crop_witnesses > 50 and edges > 0 and unequal_tokens > 50 and len(region_areas) >= 10, "family coverage")
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(12345, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    corrupt["test"][0]["output"].flat[0] = (int(corrupt["test"][0]["output"].flat[0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Two identical count-compatible molds must not be resolved by arbitrary order.
    g = np.zeros((20, 20), dtype=np.int8)
    raw = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
    tile, _ = _crop(raw)
    mold = _mold(_expand(tile, 2, 1), 4)
    g[0:3, 0:4] = np.where(raw, 2, 3)
    g[0, 8:10] = 9
    g[2, 8:11] = 9
    g[5:5 + mold.shape[0], :mold.shape[1]] = mold
    g[12:12 + mold.shape[0], :mold.shape[1]] = mold
    try:
        solve(g)
    except ValueError as exc:
        _require("unique colored" in str(exc), "ambiguous fixture rejected for wrong reason")
    else:
        raise AssertionError("ambiguous molds accepted")
    return {"seeds": 42, "pairs": 210, "source_structures_up_to_symmetry": len(structures),
            "canvas_shapes": len(canvases), "source_rectangle_shapes": len(source_shapes),
            "tile_shapes": len(tile_shapes), "counter_shapes": len(token_shapes),
            "palettes": len(colors), "counts": sorted(counts), "operations": sorted(operations),
            "mold_counts": sorted(mold_counts), "crop_witnesses": crop_witnesses,
            "edge_touching_molds": edges, "unequal_counter_sizes": unequal_tokens}
