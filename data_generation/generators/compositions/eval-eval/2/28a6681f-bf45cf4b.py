"""Communicating pouring tiles: a charged blueprint expands into a fluid device.

Only NumPy and the standard library are used. No evidence files are read. Sampling
works in memory, retains no traces, and caps every construction retry. The parser
and solver use the input's visible color overlap, never a seed or training label.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or a.dtype.kind not in "iu" or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    """Four-neighbor components, returned as Boolean masks."""
    h, w = mask.shape
    remaining = set(map(tuple, np.argwhere(mask)))
    answer = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        part = np.zeros(mask.shape, dtype=bool)
        while stack:
            r, c = stack.pop()
            part[r, c] = True
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        answer.append(part)
    return answer


def _crop_foreground(a, bg):
    r, c = np.nonzero(a != bg)
    if not len(r):
        return None
    return a[r.min():r.max() + 1, c.min():c.max() + 1].copy()


def _read_input(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(counts.max()) * 2 <= a.size:
        raise ValueError("input background must be a strict majority")
    candidates = {}
    # A full blank separator groups even disconnected blueprint cells. Try all
    # blank cuts and deduplicate identical tight crops; placement is irrelevant.
    for axis in (0, 1):
        for cut in range(a.shape[axis]):
            line = a[cut, :] if axis == 0 else a[:, cut]
            if np.any(line != bg):
                continue
            first = a[:cut, :] if axis == 0 else a[:, :cut]
            second = a[cut + 1:, :] if axis == 0 else a[:, cut + 1:]
            x, y = _crop_foreground(first, bg), _crop_foreground(second, bg)
            if x is None or y is None:
                continue
            for blueprint, tile in ((x, y), (y, x)):
                bc = set(map(int, np.unique(blueprint))) - {bg}
                tc = set(map(int, np.unique(tile))) - {bg}
                common = bc & tc
                if len(bc) != 2 or len(tc) < 3 or len(common) != 1:
                    continue
                water = next(iter(common))
                signal = next(iter(bc - common))
                if signal in tc:
                    continue
                mh, mw = blueprint.shape
                sh, sw = tile.shape
                if mh * sh > 30 or mw * sw > 30:
                    continue
                key = (_key(blueprint), _key(tile), bg, water, signal)
                candidates[key] = {"blueprint": blueprint, "tile": tile,
                                   "background": bg, "water": water, "signal": signal}
    if len(candidates) != 1:
        raise ValueError("expected one separated blueprint/tile interpretation")
    return next(iter(candidates.values()))


def _assemble(info):
    bp, tile = info["blueprint"], info["tile"]
    bg, water = info["background"], info["water"]
    sh, sw = tile.shape
    dry = tile.copy()
    dry[dry == water] = bg
    arena = np.full((bp.shape[0] * sh, bp.shape[1] * sw), bg, dtype=np.int8)
    for r, c in np.argwhere(bp != bg):
        arena[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = (
            tile if bp[r, c] == water else dry)
    return arena


def _settle(arena, bg, water, reverse=False, top_first=False):
    """Fall, then seek a supported horizontal outlet; never cross a solid.

    Valid family members have complete levels and route-safe supplies. Reverse
    tie/pour orders are checked during construction and validation, so these
    implementation orders do not encode a hidden rule for partial water levels.
    """
    out = arena.copy()
    sources = sorted(map(tuple, np.argwhere(arena == water)),
                     key=lambda p: (p[0] if top_first else -p[0],
                                    -p[1] if reverse else p[1]))
    out[out == water] = bg
    h, w = out.shape
    for row, col in sources:
        row, col = int(row), int(col)
        while row > 0 and out[row, col] == water:
            row -= 1
        if out[row, col] != bg:
            raise ValueError("source trapped under an already full ceiling")
        for _ in range(h + 1):
            if row + 1 < h and out[row + 1, col] == bg:
                row += 1
                continue
            left = right = col
            while left > 0 and out[row, left - 1] == bg:
                left -= 1
            while right + 1 < w and out[row, right + 1] == bg:
                right += 1
            exits = [] if row + 1 == h else [c for c in range(left, right + 1)
                                           if out[row + 1, c] == bg]
            if not exits:
                out[row, col] = water
                break
            col = min(exits, key=lambda c: (abs(c - col), -c if reverse else c))
            row += 1
        else:
            raise ValueError("downward fall did not terminate")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, expand the charged blueprint, and settle its liquid."""
    info = _read_input(input_grid)
    return _settle(_assemble(info), info["background"], info["water"])


def _safe_routes(solid, pool):
    """Cells whose every dry downhill/lateral outlet leads into this pool.

    Pool cells are absorbing. At a supported cell every outlet along its free
    horizontal interval must be safe, rather than preferring a left/right exit.
    """
    h, w = solid.shape
    safe = np.zeros(solid.shape, dtype=bool)
    for r in range(h - 1, -1, -1):
        c = 0
        while c < w:
            if solid[r, c]:
                c += 1
                continue
            end = c
            while end + 1 < w and not solid[r, end + 1]:
                end += 1
            outlets = []
            for j in range(c, end + 1):
                if pool[r, j]:
                    outlets.append(True)
                elif r + 1 < h and not solid[r + 1, j]:
                    outlets.append(bool(safe[r + 1, j]))
            all_safe = bool(outlets) and all(outlets)
            for j in range(c, end + 1):
                if pool[r, j]:
                    safe[r, j] = True
                elif r + 1 < h and not solid[r + 1, j]:
                    safe[r, j] = safe[r + 1, j]
                else:
                    safe[r, j] = all_safe
            c = end + 1
    return safe


def _branches(pool):
    # Add rows from the bottom. The live union count is exactly the number of
    # pockets in the pool restricted to this and lower rows. This avoids
    # repeatedly flood-filling all lower cuts for every candidate basin.
    h, w = pool.shape
    parent = {}
    count = maximum = 0

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in range(h - 1, -1, -1):
        for c in np.flatnonzero(pool[r]):
            x = r * w + int(c)
            parent[x] = x
            count += 1
            for y in (x - 1 if c else -1, x + w):
                if y in parent:
                    a, b = root(x), root(y)
                    if a != b:
                        parent[a] = b
                        count -= 1
        maximum = max(maximum, count)
    return maximum


def _transforms(a):
    return [f(np.rot90(a, k)) for k in range(4) for f in (lambda x: x, np.fliplr)]


def _asymmetric(a):
    return all(not np.array_equal(a, b) for b in _transforms(a)[1:])


def _tile_terrain(rng, shape, raised):
    sh, sw = shape
    tile = np.zeros(shape, dtype=np.int8)
    tile[-1, :] = 2
    tile[:, 0] = tile[:, -1] = 2
    # Matched side ports are genuine holes, including at seams between copies.
    bottom_port = int(rng.integers(1, sh - 2)) if raised else sh - 2
    top_port = max(1, bottom_port - int(rng.integers(1, 3)) + 1)
    tile[top_port:bottom_port + 1, 0] = 0
    tile[top_port:bottom_port + 1, -1] = 0
    # Grounded ridges vary lower pocket capacity without blocking the port row.
    for c in range(1, sw - 1):
        extra = int(rng.integers(0, sh - 1 - bottom_port)) if raised else 0
        if extra:
            tile[sh - 1 - extra:, c] = 2
    if rng.random() < .5:
        tile = tile[:, ::-1].copy()
    return tile


def _macro_mask(rng, shape, upper_shelf=False):
    h, w = shape
    rr, cc = np.indices(shape)
    style = int(rng.integers(5))
    if style == 0:
        mask = rng.random(shape) < float(rng.uniform(.25, .8))
    elif style == 1:
        mask = np.zeros(shape, dtype=bool)
        for r in range(h):
            start = int(rng.integers(w))
            length = int(rng.integers(1, w + 1))
            mask[r, start:min(w, start + length)] = True
    elif style == 2:
        mask = ((rr + cc) % int(rng.integers(2, 4)) == 0)
    elif style == 3:
        mask = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
        mask &= rng.random(shape) < .85
    else:
        mask = rng.random(shape) < .85
    # One full shelf supplies a communicating basin; its height is variable.
    anchor = 0 if upper_shelf else int(rng.integers(1, h))
    mask[anchor, :] = True
    mask[0, int(rng.integers(w))] = True
    mask[-1, int(rng.integers(w))] = True
    if np.all(mask):
        row = h - 1 if upper_shelf else 0
        mask[row, int(rng.integers(w))] = False
        if not mask[row].any():
            mask[row, 0] = True
    if rng.random() < .5:
        mask = mask[:, ::-1].copy()
    return mask


def _expand_dry(mask, tile):
    h, w = mask.shape
    sh, sw = tile.shape
    arena = np.zeros((h * sh, w * sw), dtype=np.int8)
    for r, c in np.argwhere(mask):
        arena[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = tile
    return arena


def _pool_candidates(solid, limit, raised):
    rows = np.arange(solid.shape[0])[:, None]
    candidates = []
    for top in range(1, solid.shape[0]):
        for pool in _components(~solid & (rows >= top)):
            if not pool[top].any() or not 2 <= int(pool.sum()) <= limit:
                continue
            if raised and _branches(pool) < 2:
                continue
            candidates.append(pool)
    return candidates


def _charge_construction(rng, mask, dry_tile, raised, asymmetric, excluded_volume=None):
    """Choose a complete receiving basin, then factor its volume into charges.

    A charge pattern is shared by all wet tiles. Every selected offset in every
    charged copy must safely drain to the same visible, complete receiving basin.
    This is a sampling certificate only; it is not passed to solve().
    """
    sh, sw = dry_tile.shape
    base = _expand_dry(mask, dry_tile)
    solid = base != 0
    active = list(map(tuple, np.argwhere(mask)))
    candidates = _pool_candidates(solid, int(len(active) * (dry_tile == 0).sum()), raised)
    rng.shuffle(candidates)
    for pool in candidates[:80]:
        if asymmetric and _branches(pool) != 1:
            continue
        if asymmetric:
            top = int(np.nonzero(pool)[0].min())
            if not np.any(~solid & ~pool & (np.arange(pool.shape[0])[:, None] > top)):
                continue
        volume = int(pool.sum())
        if volume == excluded_volume:
            continue
        safe = _safe_routes(solid, pool)
        offsets = [safe[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw]
                   & (dry_tile == 0) for r, c in active]
        divisors = [k for k in range(1, min(len(active) - 1, 12) + 1) if volume % k == 0]
        for k in rng.permutation(divisors):
            k = int(k)
            charge = volume // k
            if charge < 2:
                continue
            good = [i for i, eligible in enumerate(offsets) if int(eligible.sum()) >= charge]
            if len(good) < k:
                continue
            for _ in range(8):
                chosen = list(map(int, rng.choice(good, k, replace=False)))
                common = np.logical_and.reduce([offsets[i] for i in chosen])
                coords = np.argwhere(common)
                if len(coords) < charge:
                    continue
                picked = coords[rng.choice(len(coords), charge, replace=False)]
                tile = dry_tile.copy()
                tile[picked[:, 0], picked[:, 1]] = 1
                bp = np.where(mask, 9, 0).astype(np.int8)
                for i in chosen:
                    bp[active[i]] = 1
                if asymmetric and not _asymmetric(bp):
                    continue
                dry_hit = any(np.any(pool[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw])
                              for i, (r, c) in enumerate(active) if i not in chosen)
                if not dry_hit:
                    continue
                initial = base.copy()
                for i in chosen:
                    r, c = active[i]
                    initial[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = tile
                if np.array_equal(initial == 1, pool):
                    continue
                source_rows = np.argwhere(initial == 1)[:, 0].sum()
                if np.argwhere(pool)[:, 0].sum() <= source_rows:
                    continue
                expected = base.copy()
                expected[pool] = 1
                try:
                    if not all(np.array_equal(_settle(initial, 0, 1, rev, top), expected)
                               for rev in (False, True) for top in (False, True)):
                        continue
                except ValueError:
                    continue
                return bp, tile, expected
    return None


def _paint_tile(rng, tile, bg, water, payload):
    result = np.full(tile.shape, bg, dtype=np.int8)
    result[tile == 1] = water
    cells = np.argwhere(tile == 2)
    # Spatial color patches do not affect impermeability. Each payload color
    # has an anchor, so there are visibly at least two stationary colors.
    anchors = cells[rng.choice(len(cells), len(payload), replace=False)]
    distances = ((cells[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    result[cells[:, 0], cells[:, 1]] = np.asarray(payload, dtype=np.int8)[labels]
    return result


def _layout(rng, bp, tile, bg):
    # Always use a single blank separator and unambiguous tight encodings.
    # Extra padding enforces a majority background, independent of literal color.
    horizontal = bool(rng.integers(2))
    reverse = bool(rng.integers(2))
    gap = int(rng.integers(1, 5))
    first, second = (tile, bp) if reverse else (bp, tile)
    if horizontal:
        core_h = max(first.shape[0], second.shape[0])
        core_w = first.shape[1] + gap + second.shape[1]
    else:
        core_h = first.shape[0] + gap + second.shape[0]
        core_w = max(first.shape[1], second.shape[1])
    h = core_h + int(rng.integers(2, min(9, 30 - core_h) + 1))
    w = core_w + int(rng.integers(2, min(9, 30 - core_w) + 1))
    ink = int((bp != bg).sum() + (tile != bg).sum())
    while h * w <= 2 * ink:
        if h <= w and h < 30 or w == 30:
            h += 1
        else:
            w += 1
    canvas = np.full((h, w), bg, dtype=np.int8)
    if horizontal:
        c0 = int(rng.integers(w - core_w + 1))
        positions = [(int(rng.integers(h - first.shape[0] + 1)), c0),
                     (int(rng.integers(h - second.shape[0] + 1)), c0 + first.shape[1] + gap)]
    else:
        r0 = int(rng.integers(h - core_h + 1))
        positions = [(r0, int(rng.integers(w - first.shape[1] + 1))),
                     (r0 + first.shape[0] + gap, int(rng.integers(w - second.shape[1] + 1)))]
    for encoding, (r, c) in zip((first, second), positions):
        canvas[r:r + encoding.shape[0], c:c + encoding.shape[1]] = encoding
    return canvas


def _make_pair(rng, bg, water, split, index, previous_tile, excluded_volume=None):
    for _ in range(96):
        mh = (2 if index == 0 else 3 if index == 1 else int(rng.integers(2, 4))) if split == "train" else int(rng.integers(4, 7))
        mw = int(rng.integers(3 if split == "train" and index == 0 else 2,
                              6 if split == "train" else 8))
        sh = int(rng.integers(4, min(7, 30 // mh) + 1))
        sw = int(rng.integers(3, min(7, 30 // mw) + 1))
        if previous_tile is not None and (sh, sw) == previous_tile:
            continue
        raised = (index == 1 if index < 2 and split == "train" else bool(rng.integers(2)))
        mask = _macro_mask(rng, (mh, mw), upper_shelf=split == "train" and index == 0)
        dry = _tile_terrain(rng, (sh, sw), raised)
        constructed = _charge_construction(rng, mask, dry, raised,
                                           asymmetric=split == "train" and index == 0,
                                           excluded_volume=excluded_volume)
        if constructed is None:
            continue
        normalized_bp, normalized_tile, expected = constructed
        palette = [int(x) for x in rng.permutation([c for c in range(10) if c not in (bg, water)])]
        signal, payload = palette[0], palette[1:1 + int(rng.integers(2, 5))]
        tile = _paint_tile(rng, normalized_tile, bg, water, payload)
        if split == "train" and index == 0 and not _asymmetric(tile):
            continue
        bp = np.full(mask.shape, bg, dtype=np.int8)
        bp[normalized_bp == 9] = signal
        bp[normalized_bp == 1] = water
        inp = _layout(rng, bp, tile, bg)
        try:
            info = _read_input(inp)
        except ValueError:
            continue
        if not (np.array_equal(info["blueprint"], bp)
                and np.array_equal(info["tile"], tile)):
            continue
        arena = _assemble(info)
        out = arena.copy()
        out[out == water] = bg
        out[expected == 1] = water
        return {"input": inp, "output": out}, (sh, sw)
    raise RuntimeError("bounded construction exhausted without a certified pair")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not 2 <= num_train <= 10
            or not _integer(num_test) or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    bg, water = map(int, rng.choice(10, 2, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_tile = None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            for _ in range(8):
                excluded = (int((puzzle["train"][0]["output"] == water).sum())
                            if split == "train" and index == 1 else None)
                pair, tile_shape = _make_pair(rng, bg, water, split, index,
                                             previous_tile, excluded)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            previous_tile = tile_shape
    return puzzle


def _inspect(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _read_input(inp)
    bg, water = info["background"], info["water"]
    bp, tile = info["blueprint"], info["tile"]
    sh, sw = tile.shape
    initial = _assemble(info)
    assert out.shape == initial.shape
    terrain = (initial != bg) & (initial != water)
    assert np.array_equal(out[terrain], initial[terrain]), "terrain changed"
    assert np.array_equal((out != bg) & (out != water), terrain), "terrain support changed"
    charged = int((bp == water).sum())
    unit = int((tile == water).sum())
    volume = int((out == water).sum())
    assert unit >= 2 and volume == charged * unit > 0, "charge volume not conserved"
    assert not np.array_equal(initial, out), "gravity has no effect"
    assert np.argwhere(out == water)[:, 0].sum() > np.argwhere(initial == water)[:, 0].sum()
    pools = _components(out == water)
    assigned = np.zeros(out.shape, dtype=bool)
    for pool in pools:
        top = int(np.nonzero(pool)[0].min())
        free = ~terrain & (np.arange(out.shape[0])[:, None] >= top)
        closure = next(p for p in _components(free) if np.any(p & pool))
        assert np.array_equal(closure, pool), "partial meniscus or leaking basin"
        sources = (initial == water) & _safe_routes(terrain, pool)
        assert int(sources.sum()) == int(pool.sum()), "ambiguous source allocation"
        assert not np.any(assigned & sources)
        assigned |= sources
    assert np.array_equal(assigned, initial == water), "source outside safe catchments"
    assert all(np.array_equal(_settle(initial, bg, water, rev, top), out)
               for rev in (False, True) for top in (False, True)), "pour order matters"
    assert np.array_equal(solve(inp), out)
    dry_wet = sum(int(np.count_nonzero(out[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] == water))
                  for r, c in np.argwhere(bp == info["signal"]))
    assert dry_wet > 0, "no hydraulic interaction with dry copies"
    # The independent-copy shortcut leaves every dry tile water-free. Its
    # counterfactual is computed explicitly as well as rejected by dry_wet.
    local = initial.copy()
    for r, c in np.argwhere(bp != bg):
        sl = np.s_[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw]
        local[sl] = _settle(initial[sl], bg, water)
    assert not np.array_equal(local, out), "independent tile shortcut survives"
    return {"roles": (bg, water), "mask_shape": bp.shape, "tile_shape": tile.shape,
            "input_shape": inp.shape, "output_shape": out.shape,
            "active": int((bp != bg).sum()), "charged": charged,
            "unit_volume": unit, "volume": volume, "dry_wet_cells": dry_wet,
            "branches": max(_branches(p) for p in pools), "pools": len(pools),
            "lower_dry_space": bool(np.any(~terrain & (out != water) &
                                           (np.arange(out.shape[0])[:, None] >
                                            int(np.nonzero(out == water)[0].min())))),
            "empty_blocks": int((bp == bg).sum()),
            "in_gap_water": bool(any(np.any(out[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] == water)
                                     for r, c in np.argwhere(bp == bg)))}


def _validate(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, roles = set(), set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            stats = _inspect(pair)
            records[split].append(stats)
            roles.add(stats["roles"])
    assert len(roles) == 1, "background/liquid convention changed within puzzle"
    train, test = records["train"], records["test"]
    assert all(r["empty_blocks"] > 0 for r in train + test)
    assert len({r["mask_shape"] for r in train}) >= 2
    assert len({r["tile_shape"] for r in train}) >= 2
    assert len({r["volume"] for r in train}) >= 2, "show that volume is example-specific"
    assert train[0]["branches"] == 1 and train[1]["branches"] >= 2
    assert train[0]["lower_dry_space"], "demonstrate reachable rather than global bottom filling"
    first = _read_input(puzzle["train"][0]["input"])
    assert _asymmetric(first["blueprint"]) and _asymmetric(first["tile"])
    assert min(r["mask_shape"][0] for r in test) > max(r["mask_shape"][0] for r in train)
    return records


def validate_generated_puzzle(puzzle):
    _validate(puzzle)


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds and boundary counts; retain only bounded summaries."""
    signatures, masks, tiles, inputs, outputs, backgrounds, waters = (set() for _ in range(7))
    units, volumes, charges, records = set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        features = _validate(puzzle)
        if seed in (0, 41):
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair, stats in zip(puzzle[split], features[split]):
                records.append(stats)
                info = _read_input(pair["input"])
                bp, tile = info["blueprint"], info["tile"]
                bg, water = stats["roles"]
                masks.add(_key(np.where(bp == bg, 0, np.where(bp == water, 1, 2)).astype(np.int8)))
                tiles.add(_key(np.where(tile == bg, 0, np.where(tile == water, 1, 2)).astype(np.int8)))
                inputs.add(stats["input_shape"])
                outputs.add(stats["output_shape"])
                backgrounds.add(bg)
                waters.add(water)
                units.add(stats["unit_volume"])
                volumes.add(stats["volume"])
                charges.add(stats["charged"])
    assert len(signatures) == 42
    assert len(masks) >= 100 and len(tiles) >= 100
    assert len(inputs) >= 30 and len(outputs) >= 30
    assert len(backgrounds) >= 8 and len(waters) >= 8
    assert len(units) >= 10 and len(volumes) >= 20 and len(charges) >= 4
    assert sum(r["branches"] >= 2 for r in records) >= 42
    # The supplied verifier exhausts all 30 legal combinations. Keep this
    # standalone self-test bounded by checking the three extreme settings.
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
        validate_generated_puzzle(puzzle)
        assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.0, 1), (True, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    valid = generate_puzzle(0, 2, 1)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in valid[s]] for s in valid}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("incorrect output was accepted")
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(signatures),
            "mask_patterns": len(masks), "tile_patterns": len(tiles),
            "input_shapes": len(inputs), "output_shapes": len(outputs),
            "background_colors": len(backgrounds), "water_colors": len(waters),
            "charge_sizes": len(units), "volumes": len(volumes),
            "charged_counts": sorted(charges),
            "joined_cases": sum(r["branches"] >= 2 for r in records),
            "gap_water_cases": sum(r["in_gap_water"] for r in records),
            "boundary_count_settings": 3}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
