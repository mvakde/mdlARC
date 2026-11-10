"""Mirror molds with handed pigments. Standard library and NumPy only.

The solver knows no seed or generator state. Labels are constructed directly
from a sampled substrate and master placements, then checked from input pixels.
"""
from __future__ import annotations

import copy
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "non-integer grid"
    assert np.all((0 <= a) & (a <= 9)), "invalid color"
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(frozenset(cells))
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _orient(a, op):
    """0..3 rotations, 4..7 a left/right reflection followed by rotation."""
    return np.rot90(np.fliplr(a) if op >= 4 else a, op % 4).copy()


def _canonical(a):
    return min(_signature(_orient(a != 0, op)) for op in range(8))


def _swap(a, colors):
    out = a.copy()
    x, y = colors
    out[a == x], out[a == y] = y, x
    return out


def _frame(a):
    candidates = []
    for color in np.unique(a):
        if color == 0:
            continue
        rr, cc = np.nonzero(a == color)
        r, s, c, d = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        if s - r < 8 or d - c < 8:
            continue
        plate = a[r:s, c:d]
        if (np.all(plate[0] == color) and np.all(plate[-1] == color)
                and np.all(plate[:, 0] == color) and np.all(plate[:, -1] == color)
                and len(np.unique(plate[1:-1, 1:-1])) >= 3):
            candidates.append((int(color), (r, s, c, d)))
    assert len(candidates) == 1, "the mold must have one identifiable complete frame"
    return candidates[0]


def _occlusion(plate, frame_color):
    candidates = []
    for color in np.unique(plate):
        if color in (0, frame_color):
            continue
        points = np.argwhere(plate == color)
        r, s, c, d = _bbox(map(tuple, points))
        if len(points) >= 2 and (s-r)*(d-c) == len(points):
            if 0 < r < s < plate.shape[0] and 0 < c < d < plate.shape[1]:
                candidates.append((int(color), (r, s, c, d)))
    assert len(candidates) == 1, "occlusion must be the unique interior rectangular color"
    return candidates[0]


def _restore(plate, mask_color):
    """Use all supported centered axial reflections; never infer from labels."""
    axes = []
    for axis in (0, 1):
        mate = np.flip(plate, axis)
        visible = (plate != mask_color) & (mate != mask_color)
        interior = visible[1:-1, 1:-1]
        if (interior.sum() >= (plate.shape[0]-2)*(plate.shape[1]-2)*.25
                and np.array_equal(plate[visible], mate[visible])):
            axes.append(axis)
    assert axes, "no supported centered reflection"
    full = plate.copy()
    h, w = plate.shape
    for r, c in np.argwhere(plate == mask_color):
        rows = (r, h-1-r) if 0 in axes else (r,)
        cols = (c, w-1-c) if 1 in axes else (c,)
        values = {int(plate[y, x]) for y in rows for x in cols if plate[y, x] != mask_color}
        assert len(values) == 1, "erased orbit or conflicting witnesses"
        full[r, c] = values.pop()
    for axis in axes:
        assert np.array_equal(full, np.flip(full, axis)), "reconstruction violates symmetry"
    return full, tuple(axes)


def _analyse(input_grid):
    a = _grid(input_grid)
    frame_color, frame_box = _frame(a)
    r, s, c, d = frame_box
    plate = a[r:s, c:d].copy()
    mask_color, mask_box = _occlusion(plate, frame_color)
    ground, axes = _restore(plate, mask_color)
    outside = a != 0
    outside[r:s, c:d] = False
    keys = []
    for cells in _components(outside):
        colors = tuple(sorted({int(a[p]) for p in cells}))
        assert len(colors) == 2, "each loose master must have two colors"
        assert not (set(colors) & set(map(int, np.unique(plate)))), "master colors and mold colors overlap"
        for color in colors:
            local = {p for p in cells if a[p] == color}
            rr, ss, cc, dd = _bbox(local)
            region_mask = np.zeros((ss-rr, dd-cc), dtype=bool)
            for y, x in local:
                region_mask[y-rr, x-cc] = True
            assert len(_components(region_mask)) == 1, "disconnected pigment region"
        kr, ks, kc, kd = _bbox(cells)
        tile = np.zeros((ks-kr, kd-kc), dtype=np.int8)
        for y, x in cells:
            tile[y-kr, x-kc] = a[y, x]
        keys.append({"colors": colors, "cells": cells, "tile": tile, "box": (kr, ks, kc, kd)})
    assert 1 <= len(keys) <= 3, "need one to three loose masters"
    assert len({_canonical(k["tile"]) for k in keys}) == len(keys), "congruent distinct masters"
    cavities = _components(ground == 0)
    assert cavities, "no cavities"
    out, plain = ground.copy(), ground.copy()
    moves, use_counts = [], [0] * len(keys)
    for cells in cavities:
        cr, cs, cc, cd = _bbox(cells)
        assert cr > 0 and cc > 0 and cs < ground.shape[0] and cd < ground.shape[1], "cavity touches frame"
        silhouette = np.zeros((cs-cr, cd-cc), dtype=bool)
        for y, x in cells:
            silhouette[y-cr, x-cc] = True
        choices = {}
        for i, key in enumerate(keys):
            for op in range(8):
                tile = _orient(key["tile"], op)
                if tile.shape != silhouette.shape or not np.array_equal(tile != 0, silhouette):
                    continue
                colored = _swap(tile, key["colors"]) if op >= 4 else tile
                choices.setdefault((i, _signature(colored)), []).append((op, colored, tile))
        assert len(choices) == 1, "cavity needs one unique final colored casting"
        (key_index, _), descriptions = next(iter(choices.items()))
        op, colored, unswapped = descriptions[0]
        out[cr:cs, cc:cd][silhouette] = colored[silhouette]
        plain[cr:cs, cc:cd][silhouette] = unswapped[silhouette]
        moves.append({"key": key_index, "ops": tuple(v[0] for v in descriptions),
                      "box": (cr, cs, cc, cd), "cells": cells})
        use_counts[key_index] += 1
    assert all(n >= 2 for n in use_counts), "each master must have repeated castings"
    assert not np.any(out == 0) and mask_color not in out
    return out, {"frame": frame_box, "frame_color": frame_color, "mask": mask_box,
                 "mask_color": mask_color, "ground": ground, "plate": plate,
                 "axes": axes, "keys": keys, "moves": moves, "uses": tuple(use_counts),
                 "plain": plain}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore and cast the framed mold, using only the queried input."""
    return _analyse(input_grid)[0]


def _master(rng, area=None):
    """Two connected pigments grown together; chiral silhouettes make parity clear."""
    for _ in range(250):
        total = int(rng.integers(7, 14)) if area is None else area
        first = int(rng.integers(2, total-1))
        regions = [{(1, 1)}, {(1, 2)}]
        wanted = (first, total-first)
        for _ in range(total-2):
            available = [i for i in (0, 1) if len(regions[i]) < wanted[i]]
            i = int(rng.choice(available))
            used = regions[0] | regions[1]
            frontier = sorted({(r+dr, c+dc) for r, c in regions[i] for dr, dc in _D4
                               if 0 <= r+dr < 4 and 0 <= c+dc < 5} - used)
            if not frontier:
                break
            weights = np.array([1 + sum((r+dr, c+dc) in regions[i] for dr, dc in _D4)
                                for r, c in frontier], dtype=float)
            regions[i].add(frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))])
        if tuple(map(len, regions)) != wanted:
            continue
        r, s, c, d = _bbox(regions[0] | regions[1])
        tile = np.zeros((s-r, d-c), dtype=np.int8)
        for i, region in enumerate(regions, 1):
            for y, x in region:
                tile[y-r, x-c] = i
        if len({_signature(_orient(tile != 0, op)) for op in range(8)}) == 8:
            return tile
    # Guaranteed chiral fallback, with adjustable connected pigment boundary.
    tile = np.array([[1, 1, 0, 0], [0, 1, 1, 2], [0, 0, 2, 2]], dtype=np.int8)
    if area is not None and area != 7:
        # Equal-area requests use ten cells; this fixed geometry has orbit eight.
        tile = np.array([[1, 1, 1, 0], [0, 1, 1, 2], [2, 2, 2, 2]], dtype=np.int8)
    return tile


def _place_base(rng, tiles, shape):
    occupied = np.zeros(shape, dtype=bool)
    origins = [None] * len(tiles)
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    for i in order:
        h, w = tiles[i].shape
        options = [(r, c) for r in range(1, shape[0]-h) for c in range(1, shape[1]-w)]
        rng.shuffle(options)
        for r, c in options:
            if not occupied[r:r+h, c:c+w].any():
                origins[i] = (r, c)
                occupied[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True
                break
        else:
            return None
    return origins


def _texture(rng, shape, palette, style):
    colors = np.asarray(palette, dtype=np.int8)
    h, w = shape
    if style == 0:
        base = rng.choice(colors, size=shape)
    elif style == 1:
        motif = rng.choice(colors, size=(2, 3))
        base = np.tile(motif, ((h+1)//2, (w+2)//3))[:h, :w].copy()
    elif style == 2:
        small = rng.choice(colors, size=((h+1)//2, (w+1)//2))
        base = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)[:h, :w].copy()
    else:
        weights = np.arange(1, len(colors)+1, dtype=float) ** 2
        base = rng.choice(colors, size=shape, p=weights/weights.sum())
    for i, color in enumerate(colors):
        base.flat[i] = color
    return base


def _make_mold(rng, count, layout, equal_area=False):
    roles = list(map(int, rng.permutation(np.arange(1, 10))))
    frame_color, mask_color = roles[:2]
    texture_palette = [frame_color] + roles[2:2+int(rng.integers(1, 3))]
    available = [v for v in roles if v not in texture_palette and v != mask_color]
    shared = equal_area or rng.random() < .55
    shared_colors = tuple(map(int, rng.choice(available, 2, replace=False)))
    pairs = [shared_colors if shared else tuple(map(int, rng.choice(available, 2, replace=False)))
             for _ in range(count)]
    templates, source_tiles, source_parities, signatures = [], [], [], set()
    for i in range(count):
        for _ in range(100):
            tile = _master(rng, 10 if equal_area else None)
            if _canonical(tile) not in signatures:
                signatures.add(_canonical(tile))
                break
        else:
            raise AssertionError("could not make distinct master silhouettes")
        literal = np.zeros(tile.shape, dtype=np.int8)
        for role, color in enumerate(pairs[i], 1):
            literal[tile == role] = color
        hole_op, source_op = int(rng.integers(8)), int(rng.integers(8))
        templates.append(_orient(literal, hole_op))
        source_tiles.append(_orient(literal, source_op))
        source_parities.append(int(hole_op >= 4) ^ int(source_op >= 4))
    if layout == "HV":
        minimum = 7 if count == 1 else 10
        shape = (int(rng.integers(minimum, 12)), int(rng.integers(minimum, 12)))
    elif layout == "H":
        shape = (int(rng.integers(7, 11)), int(rng.integers(12, 21)))
    else:
        shape = (int(rng.integers(12, 21)), int(rng.integers(7, 11)))
    origins = _place_base(rng, templates, shape)
    if origins is None:
        raise AssertionError("base packing failed")
    base = _texture(rng, shape, texture_palette, int(rng.integers(4)))
    cast = base.copy()
    owners = np.full(shape, -1, dtype=np.int8)
    for i, (tile, (r, c)) in enumerate(zip(templates, origins)):
        h, w = tile.shape
        mask = tile != 0
        base[r:r+h, c:c+w][mask] = 0
        pigment = _swap(tile, pairs[i]) if source_parities[i] else tile
        cast[r:r+h, c:c+w][mask] = pigment[mask]
        owners[r:r+h, c:c+w][mask] = i
    lengths = [2*n-int(rng.integers(2)) if axis in layout else n
               for n, axis in zip(shape, "HV")]
    coordinates = [np.minimum(np.arange(n), n-1-np.arange(n)) if axis in layout else np.arange(n)
                   for n, axis in zip(lengths, "HV")]
    rr, cc = coordinates[0][:, None], coordinates[1][None, :]
    field, expected, own = base[rr, cc], cast[rr, cc], owners[rr, cc]
    reflected = np.zeros(tuple(lengths), dtype=bool)
    if "H" in layout:
        reflected ^= np.arange(lengths[0])[:, None] >= shape[0]
    if "V" in layout:
        reflected ^= np.arange(lengths[1])[None, :] >= shape[1]
    for i, colors in enumerate(pairs):
        m = (own == i) & reflected
        expected[m] = _swap(expected, colors)[m]
    ground = np.full((lengths[0]+2, lengths[1]+2), frame_color, dtype=np.int8)
    output = ground.copy()
    ground[1:-1, 1:-1], output[1:-1, 1:-1] = field, expected
    return ground, output, source_tiles, mask_color


def _mask_mold(rng, ground, mask_color, mode):
    cavities = _components(ground == 0)
    for _ in range(120):
        transpose = bool(rng.integers(2))
        work = ground.T if transpose else ground
        h, w = work.shape
        if mode == "simple":
            cavity = cavities[int(rng.integers(len(cavities)))]
            y, x = sorted(cavity)[int(rng.integers(len(cavity)))]
            y, x = (x, y) if transpose else (y, x)
            mh, mw = int(rng.integers(2, 7)), int(rng.integers(2, 7))
            r = int(np.clip(y-rng.integers(mh), 1, h-mh-1))
            c = int(np.clip(x-rng.integers(mw), 1, w-mw-1))
            box = r, r+mh, c, c+mw
        else:
            eligible = [cells for cells in _components(work == 0)
                        if max(y for y, _ in cells) < h//2 and max(x for _, x in cells) < w//2]
            if not eligible:
                continue
            cavity = eligible[int(rng.integers(len(eligible)))]
            r0, r1, c0, c1 = _bbox(cavity)
            if mode == "partial_seam":
                r = int(rng.integers(r0, r1))
                c = int(rng.integers(c0, c1))
                d = min(c+int(rng.integers(1, 3)), w//2)
            else:
                r, c, d = max(1, r0-1), max(1, c0-1), min(w//2, c1+1)
            box = r, h-r, c, d
        r, s, c, d = box
        selected = work[r:s, c:d]
        if not np.any(selected == 0) or not np.any(selected != 0):
            continue
        masked = work.copy()
        masked[r:s, c:d] = mask_color
        masked = masked.T.copy() if transpose else masked
        try:
            found, _ = _occlusion(masked, int(ground[0, 0]))
            rebuilt, axes = _restore(masked, found)
        except AssertionError:
            continue
        if not np.array_equal(rebuilt, ground):
            continue
        if mode != "simple" and len(axes) != 2:
            continue
        # All simple examples expose a partially concealed, incomplete cavity.
        covered = masked == mask_color
        if mode == "simple" and not any(0 < sum(covered[p] for p in cells) < len(cells) for cells in cavities):
            continue
        if mode == "full_seam" and sum(all(covered[p] for p in cells) for cells in cavities) < 2:
            continue
        return masked
    raise AssertionError("mask construction failed")


def _pack_scene(rng, plate, keys):
    ph, pw = plate.shape
    for attempt in range(35):
        if attempt >= 25:
            h = w = 30
            pr = pc = 0
        else:
            h = int(rng.integers(max(ph, 19), 31))
            w = int(rng.integers(max(pw, 19), 31))
            pr, pc = int(rng.integers(h-ph+1)), int(rng.integers(w-pw+1))
        scene = np.zeros((h, w), dtype=np.int8)
        scene[pr:pr+ph, pc:pc+pw] = plate
        occupied = np.zeros((h, w), dtype=bool)
        occupied[max(0, pr-1):min(h, pr+ph+1), max(0, pc-1):min(w, pc+pw+1)] = True
        order = list(range(len(keys)))
        rng.shuffle(order)
        for i in order:
            tile = keys[i]
            kh, kw = tile.shape
            options = [(r, c) for r in range(h-kh+1) for c in range(w-kw+1)]
            rng.shuffle(options)
            for r, c in options:
                mask = tile != 0
                if occupied[r:r+kh, c:c+kw][mask].any():
                    continue
                scene[r:r+kh, c:c+kw][mask] = tile[mask]
                for y, x in np.argwhere(mask):
                    for dy, dx in ((0, 0),) + _D4:
                        yy, xx = r+y+dy, c+x+dx
                        if 0 <= yy < h and 0 <= xx < w:
                            occupied[yy, xx] = True
                break
            else:
                break
        else:
            # Remove unused outer margins, preserving the complete mold and keys.
            points = np.argwhere(scene != 0)
            r, s, c, d = _bbox(map(tuple, points))
            padding = rng.integers(0, 3, size=4)
            r, s = max(0, r-int(padding[0])), min(h, s+int(padding[1]))
            c, d = max(0, c-int(padding[2])), min(w, d+int(padding[3]))
            return scene[r:s, c:d].copy()
    raise AssertionError("scene packing failed")


def _pair(rng, count, layout, mode, seen, equal_area=False):
    for _ in range(180):
        try:
            ground, expected, keys, mask_color = _make_mold(rng, count, layout, equal_area)
            plate = _mask_mold(rng, ground, mask_color, mode)
            inp = _pack_scene(rng, plate, keys)
            if _signature(inp) in seen:
                continue
            predicted, record = _analyse(inp)
            if not np.array_equal(predicted, expected):
                continue
            # Require a visible color-exchange witness in every scene.
            if np.array_equal(record["plain"], expected):
                continue
            return {"input": inp, "output": expected.copy()}
        except AssertionError:
            continue
    raise AssertionError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if not _integer(value) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    count_shift = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(int(num_train)):
        count = i+1 if i < 2 else int(rng.integers(1, 3 if count_shift else 4))
        mode = "partial_seam" if i == 1 and count_shift else "simple"
        layout = "HV" if i < 2 or mode != "simple" else str(rng.choice(["H", "V", "HV"]))
        pair = _pair(rng, count, layout, mode, seen, equal_area=i == 1)
        puzzle["train"].append(pair)
        seen.add(_signature(pair["input"]))
    for _ in range(int(num_test)):
        count = 3 if count_shift else int(rng.integers(1, 4))
        pair = _pair(rng, count, "HV", "full_seam", seen)
        puzzle["test"].append(pair)
        seen.add(_signature(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Schema, reconstruction, fitting, replication and demonstration checks."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train_records = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _signature(a) not in seen, "duplicate input"
            seen.add(_signature(a))
            predicted, record = _analyse(a)
            assert np.array_equal(predicted, b), "incorrect casting or crop"
            assert b.shape == record["plate"].shape
            assert np.array_equal(b[0], record["plate"][0]) and np.array_equal(b[-1], record["plate"][-1])
            assert np.array_equal(b[:, 0], record["plate"][:, 0]) and np.array_equal(b[:, -1], record["plate"][:, -1])
            assert np.array_equal(b[record["ground"] != 0], record["ground"][record["ground"] != 0])
            r, s, c, d = record["mask"]
            hidden = np.zeros(b.shape, dtype=bool)
            hidden[r:s, c:d] = True
            assert np.any(hidden & (record["ground"] == 0)), "occlusion hides no cavity"
            assert np.any(hidden & (record["ground"] != 0)), "occlusion hides no substrate"
            assert not np.array_equal(b, record["ground"]), "omitting master fitting changes nothing"
            assert not np.array_equal(b, record["plain"]), "omitting color exchange changes nothing"
            for axis in record["axes"]:
                for move in record["moves"]:
                    mirrored = frozenset((b.shape[0]-1-y, x) if axis == 0 else (y, b.shape[1]-1-x)
                                         for y, x in move["cells"])
                    mate = [m for m in record["moves"] if m["cells"] == mirrored]
                    assert len(mate) == 1 and mate[0]["key"] == move["key"], "broken cavity orbit"
                    colors = record["keys"][move["key"]]["colors"]
                    x, y = colors
                    for yy, xx in move["cells"]:
                        my, mx = (b.shape[0]-1-yy, xx) if axis == 0 else (yy, b.shape[1]-1-xx)
                        assert int(b[my, mx]) == (y if int(b[yy, xx]) == x else x), "wrong reflected pigment"
            if split == "train":
                train_records.append(record)
    assert any(len(r["axes"]) == 2 for r in train_records), "both axial mechanisms need demonstration"
    assert any(len(r["keys"]) == 1 and
               any(0 < sum(r["plate"][p] == r["mask_color"] for p in m["cells"]) < len(m["cells"])
                   for m in r["moves"]) for r in train_records), "need a one-master partial-cavity witness"
    # Equal-size, noncongruent masters establish exact shape rather than area/color matching.
    assert any(any(np.count_nonzero(a["tile"]) == np.count_nonzero(b["tile"])
                   and _canonical(a["tile"]) != _canonical(b["tile"])
                   and a["colors"] == b["colors"]
                   for i, a in enumerate(r["keys"]) for b in r["keys"][i+1:]) for r in train_records)


def self_test():
    """Bounded 42-seed structural diversity, counterfactuals and negative fixtures."""
    puzzles, dimensions, mold_shapes, shapes, partitions, axes, master_counts = set(), set(), set(), set(), set(), set(), set()
    masks, palettes, operations, styles = set(), set(), set(), set()
    covered_cavities, full_hidden, off_axis = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_signature(p["input"]) for split in ("train", "test") for p in puzzle[split])
        puzzles.add(signature)
        if seed in (0, 21, 41):
            again = generate_puzzle(seed)
            assert signature == tuple(_signature(p["input"]) for split in ("train", "test") for p in again[split])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                out, r = _analyse(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                dimensions.add(pair["input"].shape)
                mold_shapes.add(out.shape)
                axes.add(r["axes"])
                master_counts.add(len(r["keys"]))
                masks.add((r["mask"][1]-r["mask"][0], r["mask"][3]-r["mask"][2]))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                styles.add(round(float(np.mean(r["ground"][1:] == r["ground"][:-1])), 1))
                for key in r["keys"]:
                    shapes.add(_canonical(key["tile"]))
                    normalized = np.zeros(key["tile"].shape, dtype=np.int8)
                    for i, color in enumerate(key["colors"], 1):
                        normalized[key["tile"] == color] = i
                    partitions.add(min(_signature(_orient(normalized, op)) for op in range(8)))
                mr, ms, mc, md = r["mask"]
                for move in r["moves"]:
                    operations.update(move["ops"])
                    n = sum(mr <= y < ms and mc <= x < md for y, x in move["cells"])
                    covered_cavities += n > 0
                    full_hidden += n == len(move["cells"])
                if split == "test":
                    # At least one direct axis cannot recover every mask pixel; the other can.
                    hidden = r["plate"] == r["mask_color"]
                    off_axis += any(np.any(hidden & np.flip(hidden, axis)) for axis in r["axes"])
    assert len(puzzles) == 42
    assert len(dimensions) >= 45 and len(mold_shapes) >= 20
    assert len(shapes) >= 70 and len(partitions) >= 100
    assert axes == {(0,), (1,), (0, 1)} and master_counts == {1, 2, 3}
    assert operations == set(range(8)) and len(masks) >= 35 and len(palettes) >= 30
    assert len(styles) >= 4 and covered_cavities >= 100 and full_hidden >= 84 and off_axis == 42
    # Validate every legal count request without expensive repeated diversity checks.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(90210, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(81723, 2, 1)
    bad = copy.deepcopy(p)
    bad["test"][0]["output"][1, 1] = (int(bad["test"][0]["output"][1, 1])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
    # A common rigid transform of the entire scene preserves relative parity.
    # Transforming a master alone by reflection changes its cast pigments.
    for op in (1, 4, 7):
        transformed = {sp: [{k: _orient(v, op) for k, v in pair.items()} for pair in p[sp]] for sp in p}
        validate_generated_puzzle(transformed)
    # Color renaming fixes zero, the sole fixed literal role.
    perm = np.array([0] + list(range(9, 0, -1)), dtype=np.int8)
    renamed = {sp: [{k: perm[v] for k, v in pair.items()} for pair in p[sp]] for sp in p}
    validate_generated_puzzle(renamed)
    # Fully erased symmetry orbits cannot be guessed from texture statistics.
    field = np.array([[1, 2, 2, 1], [3, 4, 4, 3], [3, 4, 4, 3], [1, 2, 2, 1]], dtype=np.int8)
    field[1:3, 1:3] = 9
    try:
        _restore(field, 9)
    except AssertionError:
        pass
    else:
        raise AssertionError("fully erased orbit accepted")
    # A bar's outline cannot determine its unequal internal pigment partition.
    # Different rotations give different colored results: no arbitrary tie-break.
    ambiguous = np.zeros((14, 14), dtype=np.int8)
    ground = np.full((10, 12), 2, dtype=np.int8)
    ground[0] = ground[-1] = 1
    ground[:, 0] = ground[:, -1] = 1
    ground[2, 2:5] = ground[7, 2:5] = 0
    ground[1:3, 1:3] = 9
    ambiguous[:10, :12] = ground
    ambiguous[12, 3:6] = [3, 3, 4]
    try:
        solve(ambiguous)
    except AssertionError as error:
        assert "unique final colored casting" in str(error)
    else:
        raise AssertionError("competing colored castings accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "input_shapes": len(dimensions),
            "mold_shapes": len(mold_shapes), "master_silhouettes": len(shapes),
            "pigment_partitions": len(partitions), "mask_shapes": len(masks),
            "covered_cavities": int(covered_cavities), "fully_hidden_cavities": int(full_hidden),
            "seam_shift_tests": off_axis}
