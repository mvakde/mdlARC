"""Corner stamp interference: reflected, self-similar stamps with XOR ink.

Only NumPy and the standard library are used. All inference is input-only.
"""
from collections import Counter

import numpy as np


_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_TILE_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _normalized(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _orient(a, q):
    return a[::(-1 if q & 2 else 1), ::(-1 if q & 1 else 1)]


def _corners(rect):
    t, l, b, r = rect
    return ((t, l), (t, r), (b, l), (b, r))


def _outline(rect):
    t, l, b, r = rect
    return ({(y, x) for y in (t, b) for x in range(l, r + 1)}
            | {(y, x) for x in (l, r) for y in range(t, b + 1)})


def _target(rect, q, u, v):
    y, x = _corners(rect)[q]
    dy, dx = _DIRECTIONS[q]
    return y + dy * (u + 1), x + dx * (v + 1)


def _runs(indices):
    result = []
    for raw in indices:
        i = int(raw)
        if result and i == result[-1][1] + 1:
            result[-1] = (result[-1][0], i)
        else:
            result.append((i, i))
    return result


def _lattice(a):
    divider = int(a[0, 0])
    rr = _runs(np.flatnonzero(np.all(a == divider, axis=1)))
    cc = _runs(np.flatnonzero(np.all(a == divider, axis=0)))
    assert len(rr) >= 3 and len(cc) >= 3, "Missing divider lattice"
    assert rr[0][0] == cc[0][0] == 0
    assert rr[-1][1] == a.shape[0] - 1 and cc[-1][1] == a.shape[1] - 1
    thickness = {j - i + 1 for i, j in rr + cc}
    assert len(thickness) == 1
    d = thickness.pop()
    heights = {rr[i + 1][0] - rr[i][1] - 1 for i in range(len(rr) - 1)}
    widths = {cc[i + 1][0] - cc[i][1] - 1 for i in range(len(cc) - 1)}
    assert len(heights) == len(widths) == 1
    h, w = heights.pop(), widths.pop()
    assert (h, w) in _TILE_SHAPES and d in (1, 2)
    tiles = np.empty((len(rr) - 1, len(cc) - 1, h, w), dtype=np.int8)
    for y in range(tiles.shape[0]):
        for x in range(tiles.shape[1]):
            tiles[y, x] = a[rr[y][1]+1:rr[y+1][0], cc[x][1]+1:cc[x+1][0]]
    assert not np.any(tiles == divider), "Divider color occurs inside a tile"
    return (h, w, d, tiles.shape[0], tiles.shape[1]), divider, tiles


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            y, x = todo.pop()
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    todo.append(p)
        result.append(component)
    return result


def _write_tile(a, geometry, position, tile):
    h, w, d, _, _ = geometry
    r, c = position
    y, x = d + r * (h + d), d + c * (w + d)
    a[y:y+h, x:x+w] = tile


def _prediction(a, geometry, bg, ink, mask, marks, *, fragments=False,
                operation="xor", reflect_tiles=True):
    """Reconstruct corner sources and collect interior demands separately."""
    out = a.copy()
    demands = {}
    for rect, q, clue in marks:
        local = clue if fragments else mask
        glyph_mask = _orient(local, q) if reflect_tiles else local
        glyph = np.where(glyph_mask, ink, bg).astype(np.int8)
        _write_tile(out, geometry, _corners(rect)[q], glyph)
        for u, v in np.argwhere(local):
            p = _target(rect, q, int(u), int(v))
            demands.setdefault(p, []).append(glyph_mask)
    parity = {}
    for p, layers in demands.items():
        if operation == "xor":
            m = np.logical_xor.reduce(np.stack(layers), axis=0)
        elif operation == "union":
            m = np.logical_or.reduce(np.stack(layers), axis=0)
        elif operation == "last":
            m = layers[-1]
        else:
            raise ValueError("Unknown operation")
        parity[p] = m
        _write_tile(out, geometry, p, np.where(m, ink, bg).astype(np.int8))
    return out, demands, parity


def _infer(value):
    a = _grid(value)
    geometry, divider, tiles = _lattice(a)
    h, w, _, rows, cols = geometry
    records = {}
    for y in range(rows):
        for x in range(cols):
            tile = tiles[y, x]
            records.setdefault(_key(tile), [tile.copy(), []])[1].append((y, x))
    uniform = [record for record in records.values() if len(np.unique(record[0])) == 1]
    nonuniform = [record for record in records.values() if len(np.unique(record[0])) > 1]
    assert len(uniform) == 1 and len(nonuniform) >= 2
    frame = int(uniform[0][0][0, 0])
    ordinary, ordinary_positions = max(nonuniform, key=lambda item: len(item[1]))
    assert len(ordinary_positions) > sum(len(ps) for tile, ps in nonuniform if _key(tile) != _key(ordinary))
    ordinary_colors = set(map(int, np.unique(ordinary)))
    assert len(ordinary_colors) in (2, 3) and frame not in ordinary_colors
    border = {(y, x) for y in range(rows) for x in range(cols)
              if not np.array_equal(tiles[y, x], ordinary)}
    rects = []
    for component in _components(border):
        ys, xs = zip(*component)
        rect = min(ys), min(xs), max(ys), max(xs)
        t, l, b, r = rect
        assert b-t-1 >= h and r-l-1 >= w, "Frame cannot contain the stencil window"
        assert component == _outline(rect), "Nonordinary tiles do not form a closed frame"
        corners = set(_corners(rect))
        assert all(np.all(tiles[p] == frame) for p in component - corners)
        rects.append(rect)
    rects.sort()
    assert 1 <= len(rects) <= 2
    extra_colors = set().union(*(set(map(int, np.unique(t))) for t, _ in nonuniform)) - ordinary_colors
    assert len(extra_colors) == 1
    ink = extra_colors.pop()
    assert ink not in (frame, divider)
    marks, backgrounds = [], set()
    for rect in rects:
        for q, p in enumerate(_corners(rect)):
            tile = tiles[p]
            if np.all(tile == frame):
                continue
            mask = tile == ink
            assert 0 < mask.sum() < h*w
            assert len(np.unique(tile[~mask])) == 1
            bg = int(tile[~mask][0])
            assert bg in ordinary_colors
            backgrounds.add(bg)
            marks.append((rect, q, _orient(mask, q).copy()))
    assert len(backgrounds) == 1 and len(marks) >= 2
    bg = backgrounds.pop()
    candidates = {_key(m): m for _, _, m in marks
                  if all(np.all(~clue | m) for _, _, clue in marks)}
    assert len(candidates) == 1, "No intact compatible corner stencil"
    mask = next(iter(candidates.values())).copy()
    assert 3 <= mask.sum() < h*w
    assert any(not np.array_equal(clue, mask) for _, _, clue in marks), "No damaged corner"
    predicted, demands, parity = _prediction(a, geometry, bg, ink, mask, marks)
    interior = {(y, x) for t, l, b, r in rects for y in range(t+1, b) for x in range(l+1, r)}
    assert set(demands) <= interior
    assert np.any(predicted != a)
    singles = [p for p, layers in demands.items() if len(layers) == 1]
    collisions = [p for p, layers in demands.items() if len(layers) > 1]
    collision_witness = any(
        np.any(np.logical_or.reduce(np.stack(demands[p]), axis=0) & ~parity[p])
        and np.any(parity[p])
        and all(not np.array_equal(parity[p], layer) for layer in demands[p])
        for p in collisions)
    background_witness = any(np.any((ordinary != bg) & ~parity[p]) for p in singles)
    info = dict(geometry=geometry, divider=divider, frame=frame, ordinary=ordinary,
                bg=bg, ink=ink, mask=mask, marks=marks, frames=rects,
                demands=demands, parity=parity, singles=singles, collisions=collisions,
                collision_witness=collision_witness, background_witness=background_witness,
                holes=interior-set(demands), tile_array=tiles)
    return predicted, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer lattice, frames, intact stencil, requests and ink parity from input."""
    return _infer(input_grid)[0]


def _assemble(geometry, palette, ordinary, mask, plans, rng, sparse=False):
    h, w, d, rows, cols = geometry
    divider, frame, bg, ink = palette[:4]
    a = np.full((rows*(h+d)+d, cols*(w+d)+d), divider, dtype=np.int8)
    for y in range(rows):
        for x in range(cols):
            _write_tile(a, geometry, (y, x), ordinary)
    marked = [(rect, q) for rect, qs in plans for q in qs]
    assert len(marked) >= 2
    reference = int(rng.integers(len(marked)))
    full = {reference}
    for i in range(len(marked)):
        if i != reference and rng.random() < .22:
            full.add(i)
    if len(full) == len(marked):
        full.remove(next(i for i in range(len(marked)) if i != reference))
    for rect, _ in plans:
        for p in _outline(rect):
            _write_tile(a, geometry, p, np.full((h, w), frame, dtype=np.int8))
    active = np.argwhere(mask)
    marks = []
    for i, (rect, q) in enumerate(marked):
        clue = mask.copy()
        if i not in full:
            n = 1 if sparse else int(rng.integers(1, len(active)))
            clue[:] = False
            for u, v in active[rng.choice(len(active), n, replace=False)]:
                clue[u, v] = True
        _write_tile(a, geometry, _corners(rect)[q], np.where(_orient(clue, q), ink, bg).astype(np.int8))
        marks.append((rect, q, clue))
    # Construction paints each complete oriented tile into an XOR accumulator.
    # The inference routine must reconstruct all of this information from a.
    out = a.copy()
    accumulated, touched = {}, set()
    for rect, q, _ in marks:
        tile_mask = _orient(mask, q)
        _write_tile(out, geometry, _corners(rect)[q], np.where(tile_mask, ink, bg).astype(np.int8))
        for u in range(h):
            for v in range(w):
                if mask[u, v]:
                    p = _target(rect, q, u, v)
                    if p not in accumulated:
                        accumulated[p] = np.zeros((h, w), dtype=bool)
                    accumulated[p] ^= tile_mask
                    touched.add(p)
    for p in touched:
        _write_tile(out, geometry, p, np.where(accumulated[p], ink, bg).astype(np.int8))
    return {"input": a, "output": out}


def _sample_mask(rng, shape):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    mask.flat[rng.choice(h*w, int(rng.integers(3, h*w)), replace=False)] = True
    return mask


def _ordinary(rng, shape, palette):
    colors = (palette[2], *palette[4:])
    labels = np.concatenate((np.arange(len(colors)), rng.integers(len(colors), size=np.prod(shape)-len(colors))))
    rng.shuffle(labels)
    return np.asarray(colors, dtype=np.int8)[labels.reshape(shape)]


def _random_plan(rng, shape, thickness, test, witness, witness_axis=1):
    h, w = shape
    max_rows, max_cols = (30-thickness)//(h+thickness), (30-thickness)//(w+thickness)
    # Two separate frames are possible only on an axis with two small windows.
    multi_axes = [i for i, (side, maximum) in enumerate(((h, max_rows), (w, max_cols)))
                  if 2*(side+2)+1 <= maximum]
    if multi_axes and not witness and rng.random() < .24:
        axis = int(rng.choice(multi_axes))
        rows = max_rows if axis == 0 else int(rng.integers(h+2, max_rows+1))
        cols = max_cols if axis == 1 else int(rng.integers(w+2, max_cols+1))
        if axis == 0:
            rects = []
            for top in (0, rows-h-2):
                extent = int(rng.integers(w+2, cols+1))
                left = int(rng.integers(cols-extent+1))
                rects.append((top, left, top+h+1, left+extent-1))
        else:
            rects = []
            for left in (0, cols-w-2):
                extent = int(rng.integers(h+2, rows+1))
                top = int(rng.integers(rows-extent+1))
                rects.append((top, left, top+extent-1, left+w+1))
    else:
        rh = int(rng.integers(h+2, max_rows+1))
        cw = int(rng.integers(w+2, max_cols+1))
        rows = int(rng.integers(rh, max_rows+1))
        cols = int(rng.integers(cw, max_cols+1))
        top = int(rng.integers(rows-rh+1))
        left = int(rng.integers(cols-cw+1))
        rects = [(top, left, top+rh-1, left+cw-1)]
    plans = []
    for i, rect in enumerate(rects):
        n = int(rng.choice((3, 4))) if test and i == 0 else (2 if i == 0 else int(rng.integers(5)))
        if not test and i > 0:
            n = min(n, 2)
        if n == 2 and witness:
            q = int(rng.integers(4))
            qs = [q, q ^ witness_axis]
        else:
            qs = list(map(int, rng.choice(4, n, replace=False)))
        plans.append((rect, qs))
    return (h, w, thickness, rows, cols), plans


def _acceptable(pair, seen, first, index, test):
    if _key(pair["input"]) in seen:
        return False
    _, f = _infer(pair["input"])
    if not np.array_equal(solve(pair["input"]), pair["output"]):
        return False
    if not test and index == 0:
        if not f["singles"] or not f["holes"] or not f["background_witness"]:
            return False
        if f["mask"].shape[0] == f["mask"].shape[1] and np.array_equal(f["mask"], f["mask"].T):
            return False
        if np.array_equal(f["mask"], _orient(f["mask"], 1)):
            return False
        no_reflection = _prediction(pair["input"], f["geometry"], f["bg"], f["ink"],
                                    f["mask"], f["marks"], reflect_tiles=False)[0]
        if np.array_equal(no_reflection, pair["output"]):
            return False
    if not test and index == 1:
        if not f["collision_witness"] or not f["singles"] or not f["holes"] or not f["background_witness"]:
            return False
        if _key(f["mask"]) == _key(first["mask"]) or _normalized(f["ordinary"]) == _normalized(first["ordinary"]):
            return False
    return True


def _fallback(rng, palette, seen, first, index, test):
    # Finite, spacious construction. The missing corner can be any of four
    # positions; each has an adjacent stamp collision with partial cancellation.
    for absent in rng.permutation(4):
        mask = np.ones((2, 2), dtype=bool)
        mask.flat[int(absent)] = False
        for code in rng.permutation(np.arange(1, 15)):
            ordinary = np.array([palette[4] if int(code) & (1 << i) else palette[2]
                                 for i in range(4)], dtype=np.int8).reshape(2, 2)
            for top in range(3):
                for left in range(3):
                    rect = top, left, top+4, left+4
                    qs = [0, 1, 2] if test else ([0, 2] if index == 1 else [0, 1])
                    plans = [(rect, qs)]
                    pair = _assemble((2, 2, 1, 7, 7), palette[:5], ordinary, mask, plans, rng, sparse=test)
                    if _acceptable(pair, seen, first, index, test):
                        return pair
    raise RuntimeError("Constructive fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.random() < .25)
    stable_size = bool(rng.random() < .25)
    base_palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    base_shape = _TILE_SHAPES[int(rng.integers(4))]
    # Tests activate more corners; half also use a familiar stencil with new
    # damage, while the others introduce an input-visible new stencil.
    familiar_test = bool(rng.integers(2))
    puzzle, seen, train_info = {"train": [], "test": []}, set(), []
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            test = split == "test"
            pair = None
            for attempt in range(240):
                palette = base_palette if stable_palette else tuple(map(int, rng.choice(10, 6, replace=False)))
                # Two or three ordinary colors, independently of stencil density.
                palette = palette[:5 + int(rng.random() < .25)]
                shape = base_shape if stable_size else _TILE_SHAPES[int(rng.integers(4))]
                if test and familiar_test:
                    mask = train_info[index % len(train_info)]["mask"].copy()
                    shape = mask.shape
                else:
                    mask = _sample_mask(rng, shape)
                    if test and any(_key(mask) == _key(f["mask"]) for f in train_info):
                        continue
                thickness = int(rng.choice((1, 2), p=(.8, .2)))
                geometry, plans = _random_plan(rng, shape, thickness, test, not test and index < 2,
                                               witness_axis=2 if index == 1 else 1)
                marked_count = sum(len(qs) for _, qs in plans)
                ordinary_count = geometry[3]*geometry[4] - sum(len(_outline(rect)) for rect, _ in plans)
                if ordinary_count <= marked_count:
                    continue
                ordinary = _ordinary(rng, shape, palette)
                pair = _assemble(geometry, palette, ordinary, mask, plans, rng, sparse=test and rng.random() < .5)
                if _acceptable(pair, seen, train_info[0] if train_info else None, index, test):
                    break
                pair = None
            if pair is None:
                pair = _fallback(rng, base_palette, seen, train_info[0] if train_info else None, index, test)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if not test:
                train_info.append(_infer(pair["input"])[1])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            expected, f = _infer(a)
            assert np.array_equal(expected, b), "Output differs from input-only parity inference"
            geometry, _, output_tiles = _lattice(b)
            assert geometry == f["geometry"]
            corners = {_corners(rect)[q] for rect, q, _ in f["marks"]}
            allowed = corners | set(f["demands"])
            for y in range(output_tiles.shape[0]):
                for x in range(output_tiles.shape[1]):
                    p = y, x
                    if p not in allowed:
                        assert np.array_equal(output_tiles[p], f["tile_array"][p])
                    elif p in f["demands"]:
                        assert np.array_equal(output_tiles[p], np.where(f["parity"][p], f["ink"], f["bg"]))
            fragment_answer = _prediction(a, geometry, f["bg"], f["ink"], f["mask"], f["marks"], fragments=True)[0]
            assert not np.array_equal(fragment_answer, b), "Ignoring shared-template recovery must change the answer"
            restored_only = a.copy()
            for rect, q, _ in f["marks"]:
                _write_tile(restored_only, geometry, _corners(rect)[q], np.where(_orient(f["mask"], q), f["ink"], f["bg"]))
            assert not np.array_equal(restored_only, b), "Omitting scale transfer must change the answer"
            per_frame = Counter(rect for rect, _, _ in f["marks"])
            if split == "train":
                assert max(per_frame.values()) <= 2
            else:
                assert max(per_frame.values()) >= 3
            features[split].append(f)
    first, second = features["train"][:2]
    assert _key(first["mask"]) != _key(second["mask"])
    assert _normalized(first["ordinary"]) != _normalized(second["ordinary"])
    assert first["background_witness"] and first["singles"] and first["holes"]
    if first["mask"].shape[0] == first["mask"].shape[1]:
        assert not np.array_equal(first["mask"], first["mask"].T)
    first_pair = puzzle["train"][0]
    unreflected = _prediction(_grid(first_pair["input"]), first["geometry"], first["bg"], first["ink"],
                             first["mask"], first["marks"], reflect_tiles=False)[0]
    assert not np.array_equal(unreflected, first_pair["output"])
    for f, axis in ((first, 1), (second, 2)):
        qs = [q for rect, q, _ in f["marks"] if rect == f["frames"][0]]
        assert len(qs) == 2 and qs[0] ^ qs[1] == axis
        assert not np.array_equal(f["mask"], _orient(f["mask"], axis))
    assert second["collision_witness"] and second["singles"] and second["holes"] and second["background_witness"]
    pair = puzzle["train"][1]
    for operation in ("union", "last"):
        alternative = _prediction(_grid(pair["input"]), second["geometry"], second["bg"], second["ink"],
                                  second["mask"], second["marks"], operation=operation)[0]
        assert not np.array_equal(alternative, pair["output"])


def _signature(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded semantic, adversarial, count, determinism and diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    fs = [_infer(pair["input"])[1] for p in puzzles for s in ("train", "test") for pair in p[s]]
    assert {_key(f["mask"])[0] for f in fs} == set(_TILE_SHAPES)
    assert {f["geometry"][2] for f in fs} == {1, 2}
    assert len({_key(f["mask"]) for f in fs}) >= 35
    assert len({_normalized(f["ordinary"]) for f in fs}) >= 40
    assert len({f["geometry"] for f in fs}) >= 40
    assert {len(f["frames"]) for f in fs} == {1, 2}
    assert {q for f in fs for _, q, _ in f["marks"]} == {0, 1, 2, 3}
    assert max(len(layers) for f in fs for layers in f["demands"].values()) >= 3
    assert any(not np.any(m) for f in fs for m in f["parity"].values())
    assert any(np.array_equal(f["mask"], f["mask"].T) for f in fs if f["mask"].shape[0] == f["mask"].shape[1])
    assert any(0 not in np.unique(pair["input"]) for p in puzzles for s in p for pair in p[s])
    assert any(30 in pair["input"].shape for p in puzzles for s in p for pair in p[s])
    probe = puzzles[0]["test"][0]["input"].copy()
    saved = probe.copy()
    assert np.array_equal(solve(probe), puzzles[0]["test"][0]["output"])
    assert np.array_equal(probe, saved), "solve mutated its input"
    # Hold stencil, geometry, roles and requested corners fixed; only source
    # damage varies. This checks structural diversity conditional on a design.
    fixed_mask = np.array([[0, 1], [1, 1]], dtype=bool)
    fixed_ordinary = np.array([[5, 3], [3, 5]], dtype=np.int8)
    fixed = [_assemble((2, 2, 1, 7, 7), (1, 2, 3, 4, 5), fixed_ordinary,
                       fixed_mask, [((1, 1, 5, 5), [0, 1])], np.random.default_rng(s))
             for s in range(42)]
    assert len({_key(p["input"]) for p in fixed}) >= 8
    assert len({_key(p["output"]) for p in fixed}) == 1
    assert all(np.array_equal(solve(p["input"]), p["output"]) for p in fixed)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(9000+10*nt+nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Force the finite construction, rather than trusting that it is reachable.
    rng = np.random.default_rng(123)
    f0 = _fallback(rng, (1, 2, 3, 4, 5, 6), set(), None, 0, False)
    f1 = _fallback(rng, (1, 2, 3, 4, 5, 6), {_key(f0["input"])}, _infer(f0["input"])[1], 1, False)
    ft = _fallback(rng, (1, 2, 3, 4, 5, 6), {_key(f0["input"]), _key(f1["input"])}, _infer(f0["input"])[1], 0, True)
    validate_generated_puzzle({"train": [f0, f1], "test": [ft]})
    # Reject a wrong overlap rule, an untouched-region edit, a damaged divider,
    # and loss of the intact source (two incompatible maximal clues).
    import copy
    for defect in ("union", "outside", "divider", "reference"):
        bad = copy.deepcopy(puzzles[0])
        pair = bad["train"][1]
        a, f = _infer(pair["input"])
        if defect == "union":
            pair["output"] = _prediction(pair["input"], f["geometry"], f["bg"], f["ink"], f["mask"], f["marks"], operation="union")[0]
        elif defect in ("outside", "divider"):
            target = pair["output"] if defect == "outside" else pair["input"]
            target[0, 0] = (int(target[0, 0])+1) % 10
        else:
            rect, q, _ = next(mark for mark in f["marks"] if np.array_equal(mark[2], f["mask"]))
            partial = next(clue for _, _, clue in f["marks"] if not np.array_equal(clue, f["mask"]))
            altered = f["mask"].copy()
            removed = tuple(np.argwhere(partial)[0])
            altered[removed] = False
            # Select a formerly inactive pixel, not the pixel just removed.
            added = int(np.flatnonzero(~f["mask"])[0])
            altered.flat[added] = True
            _write_tile(pair["input"], f["geometry"], _corners(rect)[q], np.where(_orient(altered, q), f["ink"], f["bg"]))
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"Accepted defect: {defect}")
