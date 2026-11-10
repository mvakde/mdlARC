"""Shattered mirror panel: recover engraving, print it in corner-indexed inks.

Only Python's standard library and NumPy are used.  The solver enumerates
visible packings and mirror constraints; it has no access to sampled plans.
"""
from __future__ import annotations

import itertools
from math import isqrt
import numpy as np


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int16, copy=True)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while cells:
        root = min(cells)
        cells.remove(root)
        stack, found = [root], [root]
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBORS:
                q = r + dr, c + dc
                if q in cells:
                    cells.remove(q)
                    found.append(q)
                    stack.append(q)
        result.append(np.asarray(found, dtype=np.int16))
    return result


def _border(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _parse(a):
    border = _border(a)
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(np.count_nonzero(border == bg) > len(border) / 2, "background must dominate border")
    comps = _components(a != bg)
    _require(len(comps) == 4, "four connected shards required")
    for x, y in itertools.combinations(comps, 2):
        gap = np.max(np.abs(x[:, None] - y[None, :]), axis=2).min()
        _require(gap >= 2, "shards need a background-cell gap")
    colors, counts = np.unique(a[a != bg], return_counts=True)
    singles = colors[counts == 1]
    _require(len(colors) == 7 and len(singles) == 4, "three panel colors and four singleton key inks required")
    points = np.argwhere(np.isin(a, singles))
    r, c = points.min(axis=0)
    _require(set(map(tuple, points.tolist())) == {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}, "ordered 2x2 key required")
    key = a[r:r + 2, c:c + 2].copy()
    anchor = [i for i, comp in enumerate(comps) if (comp == (r, c)).all(axis=1).any()]
    _require(len(anchor) == 1, "key must belong to a shard")
    comps = [comps[anchor[0]]] + [p for i, p in enumerate(comps) if i != anchor[0]]
    pieces = []
    for comp in comps:
        lo, hi = comp.min(axis=0), comp.max(axis=0) + 1
        tile = np.full(tuple(hi - lo), -1, dtype=np.int16)
        q = comp - lo
        tile[q[:, 0], q[:, 1]] = a[comp[:, 0], comp[:, 1]]
        pieces.append(tile)
    area = sum(len(comp) for comp in comps)
    n = isqrt(area)
    _require(n * n == area and 4 <= n <= 30, "shard area must be square")
    return {"background": bg, "pieces": pieces, "key": key, "side": n,
            "panel_colors": tuple(int(v) for v in colors if v not in singles),
            "components": comps}


def _placement(tile, n, corner, rotation):
    work = np.rot90(tile, rotation)
    h, w = work.shape
    if h > n or w > n:
        return None
    row = 0 if corner < 2 else n - h
    col = 0 if corner % 2 == 0 else n - w
    coords = np.argwhere(work >= 0)
    rr, cc = coords[:, 0] + row, coords[:, 1] + col
    hits = {(int(r), int(c)) for r, c in zip(rr, cc)} & {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}
    target = ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1))[corner]
    if hits != {target}:
        return None
    bits = sum(1 << (int(r) * n + int(c)) for r, c in zip(rr, cc))
    return {"bits": bits, "rows": rr, "cols": cc,
            "values": work[coords[:, 0], coords[:, 1]], "rotation": rotation}


def _packings(parsed):
    n, pieces = parsed["side"], parsed["pieces"]
    first = _placement(pieces[0], n, 0, 0)
    if first is None:
        return
    table = {}
    for i in range(1, 4):
        for corner in range(1, 4):
            table[i, corner] = []
            seen = set()
            for k in range(4):
                p = _placement(pieces[i], n, corner, k)
                if p is not None:
                    sig = (p["bits"], p["values"].tobytes())
                    if sig not in seen:
                        table[i, corner].append(p)
                        seen.add(sig)
    board = np.full((n, n), -1, dtype=np.int16)
    owner = board.copy()
    board[first["rows"], first["cols"]] = first["values"]
    owner[first["rows"], first["cols"]] = 0
    full = (1 << (n * n)) - 1

    def visit(corner, remaining, occupied, turns, identities):
        if corner == 4:
            if occupied == full:
                yield board.copy(), owner.copy(), tuple(turns), tuple(identities)
            return
        for i in remaining:
            for p in table[i, corner]:
                if p["bits"] & occupied:
                    continue
                board[p["rows"], p["cols"]] = p["values"]
                owner[p["rows"], p["cols"]] = corner
                yield from visit(corner + 1, tuple(j for j in remaining if j != i), occupied | p["bits"], turns + [p["rotation"]], identities + [i])
                board[p["rows"], p["cols"]] = -1
                owner[p["rows"], p["cols"]] = -1

    yield from visit(1, (1, 2, 3), first["bits"], [0], [0])


def _orbit(r, c, n, axes):
    rows = {r, n - 1 - r} if "H" in axes else {r}
    cols = {c, n - 1 - c} if "V" in axes else {c}
    return {(rr, cc) for rr in rows for cc in cols}


def _recover(board, owner, key, panel_colors, mask_color):
    mask = board == mask_color
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return None
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    r0, c0 = map(int, lo)
    r1, c1 = map(int, hi)
    if (r1 - r0) * (c1 - c0) != len(coords) or len(coords) > board.size * .45:
        return None
    other = [c for c in panel_colors if c != mask_color]
    counts = [int(np.count_nonzero(board == c)) for c in other]
    if min(counts) < 2 or counts[0] == counts[1]:
        return None
    substrate, glyph = (other[0], other[1]) if counts[0] > counts[1] else (other[1], other[0])
    visible = (board == substrate) | (board == glyph)
    axes = []
    for axis, mate_board, mate_visible in (("H", board[::-1], visible[::-1]), ("V", board[:, ::-1], visible[:, ::-1])):
        witnesses = visible & mate_visible
        # Distinct coordinates, enough visible evidence of both engraving states.
        rr, cc = np.indices(board.shape)
        witnesses &= (rr != len(board) - 1 - rr) if axis == "H" else (cc != len(board) - 1 - cc)
        if witnesses.sum() >= max(8, board.size // 5) and np.count_nonzero(board[witnesses] == glyph) >= 2 and np.count_nonzero(board[witnesses] == substrate) >= 2 and np.array_equal(board[witnesses], mate_board[witnesses]):
            axes.append(axis)
    if not axes:
        return None
    # Validate the whole field, including constraints joined through a missing
    # legend cell. Pairwise visible comparisons alone can miss a diagonal
    # conflict in a four-cell H/V orbit.
    rr, cc = np.indices(board.shape)
    folded_r = np.minimum(rr, len(board) - 1 - rr) if "H" in axes else rr
    folded_c = np.minimum(cc, len(board) - 1 - cc) if "V" in axes else cc
    labels = (folded_r * len(board) + folded_c)[visible]
    low = np.full(board.size, 10, dtype=np.int16)
    high = np.full(board.size, -1, dtype=np.int16)
    np.minimum.at(low, labels, board[visible])
    np.maximum.at(high, labels, board[visible])
    if np.any((high >= 0) & (low != high)):
        return None
    recovered = np.empty((r1 - r0, c1 - c0), dtype=np.int16)
    composed = 0
    cross_shard = 0
    half_turn_disagreements = 0
    for r, c in coords:
        orbit = _orbit(int(r), int(c), len(board), axes)
        values = {int(board[q]) for q in orbit if visible[q]}
        if len(values) != 1:
            return None
        cross_shard += int(not any(visible[q] and owner[q] == owner[r, c] for q in orbit))
        recovered[r - r0, c - c0] = values.pop()
        half_turn = len(board) - 1 - int(r), len(board) - 1 - int(c)
        half_turn_disagreements += int(visible[half_turn] and board[half_turn] != recovered[r - r0, c - c0])
        direct = []
        if "H" in axes:
            direct.append((len(board) - 1 - int(r), int(c)))
        if "V" in axes:
            direct.append((int(r), len(board) - 1 - int(c)))
        composed += int(not any(visible[q] for q in direct))
    local_owner = owner[r0:r1, c0:c1]
    answer = np.where(recovered == glyph, key.reshape(-1)[local_owner], substrate).astype(np.int8)
    return {"answer": answer, "raw_crop": recovered, "owner_crop": local_owner.copy(),
            "bbox": (r0, r1, c0, c1), "mask_color": mask_color,
            "substrate": substrate, "glyph": glyph, "axes": tuple(axes),
            "composed_cells": composed, "board": board.copy(), "owner": owner.copy(),
            "cross_shard_cells": cross_shard,
            "half_turn_disagreements": half_turn_disagreements,
            "active_roles": set(int(x) for x in local_owner[recovered == glyph]),
            "blank_cells": int(np.count_nonzero(recovered == substrate))}


def _analyze(input_grid):
    parsed = _parse(_grid(input_grid))
    answers, interpretations, geometric = {}, [], 0
    seen_boards = set()
    for board, owner, turns, identities in _packings(parsed):
        geometric += 1
        sig = (board.tobytes(), owner.tobytes())
        if sig in seen_boards:
            continue
        seen_boards.add(sig)
        for marker in parsed["panel_colors"]:
            record = _recover(board, owner, parsed["key"], parsed["panel_colors"], marker)
            if record is None:
                continue
            record.update({"turns": turns, "identities": identities})
            answers[_signature(record["answer"])] = record["answer"]
            interpretations.append(record)
    _require(len(answers) == 1, "input must determine one answer from supported packings and mirrors")
    # Several indistinguishable interpretations are legal only if their answers agree.
    record = interpretations[0]
    record.update(parsed)
    record["geometric_packings"] = geometric
    record["interpretations"] = len(interpretations)
    record["answer"] = next(iter(answers.values())).copy()
    return record


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer exclusively from this grid's geometry and pixels."""
    return _analyze(input_grid)["answer"].copy()


def _key_sites(owner):
    return [(r, c) for r in range(len(owner) - 1) for c in range(len(owner) - 1)
            if np.all(owner[r:r + 2, c:c + 2] == 0)]


def _connected(owner):
    return all(len(_components(owner == i)) == 1 for i in range(4))


def _partition(rng, n, style):
    if style == 0:
        cut = int(rng.integers(2, n - 1))
        top, bottom = int(rng.integers(2, n - 1)), int(rng.integers(1, n))
        owner = np.empty((n, n), dtype=np.int16)
        for r in range(n):
            if r >= 2 and rng.random() < .7:
                if r < cut:
                    top = int(np.clip(top + rng.choice([-1, 1]), 1, n - 1))
                else:
                    bottom = int(np.clip(bottom + rng.choice([-1, 1]), 1, n - 1))
            x = top if r < cut else bottom
            owner[r, :x], owner[r, x:] = (0, 1) if r < cut else (2, 3)
    else:
        owner = np.full((n, n), -1, dtype=np.int16)
        owner[:2, :2], owner[:2, -2:], owner[-2:, :2], owner[-2:, -2:] = 0, 1, 2, 3
        frontier = [set() for _ in range(4)]
        for i in range(4):
            for r, c in np.argwhere(owner == i):
                for dr, dc in NEIGHBORS:
                    if 0 <= r + dr < n and 0 <= c + dc < n and owner[r + dr, c + dc] == -1:
                        frontier[i].add((int(r + dr), int(c + dc)))
        weights = rng.dirichlet(np.full(4, 2.0)) + .04
        for _ in range(n * n - 16):
            live = [i for i in range(4) if frontier[i]]
            p = weights[live] / weights[live].sum()
            i = int(rng.choice(live, p=p))
            pool = sorted(frontier[i])
            r, c = pool[int(rng.integers(len(pool)))]
            owner[r, c] = i
            for f in frontier:
                f.discard((r, c))
            for dr, dc in NEIGHBORS:
                q = r + dr, c + dc
                if 0 <= q[0] < n and 0 <= q[1] < n and owner[q] == -1:
                    frontier[i].add(q)
    # Boundary exchanges produce tabs, returning arms, and nonstraight seams.
    corners = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}
    for _ in range(n * 2):
        r, c = int(rng.integers(n)), int(rng.integers(n))
        if (r, c) in corners:
            continue
        old = int(owner[r, c])
        choices = [int(owner[r + dr, c + dc]) for dr, dc in NEIGHBORS if 0 <= r + dr < n and 0 <= c + dc < n]
        owner[r, c] = int(rng.choice(choices))
        if not _connected(owner) or not _key_sites(owner) or min(np.count_nonzero(owner == i) for i in range(4)) < 4:
            owner[r, c] = old
    return owner


def _shadow_window(rng, n, key_site, orientation):
    kr, kc = key_site
    # Query the opposite side of the legend; span the other reflection seam.
    if orientation == "wide":
        r0, r1 = n - kr - 2, n - kr
        c0, c1 = min(kc, n - kc - 2), max(kc + 2, n - kc)
        r0 = max(n // 2, r0 - int(rng.integers(2)))
        r1 = min(n, r1 + int(rng.integers(2)))
    else:
        c0, c1 = n - kc - 2, n - kc
        r0, r1 = min(kr, n - kr - 2), max(kr + 2, n - kr)
        c0 = max(n // 2, c0 - int(rng.integers(2)))
        c1 = min(n, c1 + int(rng.integers(2)))
    if r0 >= r1 or c0 >= c1:
        return None
    if (r1 - r0) * (c1 - c0) > n * n * .45:
        return None
    if not (c1 - c0 > r1 - r0 if orientation == "wide" else r1 - r0 > c1 - c0):
        return None
    return r0, r1, c0, c1


def _direct_window(rng, n, orientation):
    if orientation == "wide":
        h = int(rng.integers(1, max(2, n // 2)))
        w = int(rng.integers(max(h + 1, n // 2), n + 1))
        r0, c0 = int(rng.integers(0, n // 2 - h + 1)), int(rng.integers(0, n - w + 1))
    else:
        w = int(rng.integers(1, max(2, n // 2)))
        h = int(rng.integers(max(w + 1, n // 2), n + 1))
        r0, c0 = int(rng.integers(0, n - h + 1)), int(rng.integers(0, n // 2 - w + 1))
    return r0, r0 + h, c0, c0 + w


def _pattern(rng, n, axes, style):
    rr, cc = np.indices((n, n))
    r = np.minimum(rr, n - 1 - rr) if "H" in axes else rr
    c = np.minimum(cc, n - 1 - cc) if "V" in axes else cc
    source = rng.random((n, n)) < float(rng.uniform(.20, .32))
    if style == 1:
        source = np.repeat(np.repeat(rng.random(((n + 1) // 2, (n + 1) // 2)) < .25, 2, axis=0), 2, axis=1)[:n, :n]
    elif style == 2:
        source = ((rr + 2 * cc + int(rng.integers(5))) % int(rng.integers(3, 6)) == 0)
        source ^= rng.random((n, n)) < .08
    return source[r, c].astype(np.int16)


def _scatter(rng, tiles, bg, mode):
    dims = [a.shape for a in tiles]
    offsets = None
    if mode == 0:
        h, w = int(rng.integers(20, 31)), int(rng.integers(20, 31))
        forbidden = np.zeros((h, w), dtype=bool)
        placed = [None] * 4
        for i in rng.permutation(4):
            i = int(i)
            q = np.argwhere(tiles[i] >= 0)
            for _ in range(90):
                r, c = int(rng.integers(h - dims[i][0] + 1)), int(rng.integers(w - dims[i][1] + 1))
                x = q + (r, c)
                if forbidden[x[:, 0], x[:, 1]].any():
                    continue
                placed[i] = (r, c)
                for dr, dc in itertools.product((-1, 0, 1), repeat=2):
                    forbidden[np.clip(x[:, 0] + dr, 0, h - 1), np.clip(x[:, 1] + dc, 0, w - 1)] = True
                break
            else:
                break
        if all(x is not None for x in placed):
            offsets = placed
    if offsets is None:
        choices = []
        for transpose in (False, True):
            ds = [d[::-1] if transpose else d for d in dims]
            for order in itertools.permutations(range(4)):
                for bits in range(8):
                    groups, group = [], []
                    for j, i in enumerate(order):
                        group.append(i)
                        if j == 3 or bits & (1 << j):
                            groups.append(group)
                            group = []
                    gap = int(rng.integers(1, 4))
                    hs = [max(ds[i][0] for i in g) for g in groups]
                    ws = [sum(ds[i][1] for i in g) + gap * (len(g) - 1) for g in groups]
                    h, w = sum(hs) + gap * (len(groups) - 1), max(ws)
                    if max(h, w) > 28:
                        continue
                    pos, row = [None] * 4, 0
                    for g, hh, ww in zip(groups, hs, ws):
                        col = int(rng.integers(w - ww + 1))
                        for i in g:
                            rr = row + int(rng.integers(hh - ds[i][0] + 1))
                            pos[i] = (col, rr) if transpose else (rr, col)
                            col += ds[i][1] + gap
                        row += hh + gap
                    linear = len(groups) == 1
                    choices.append((pos, linear))
        _require(bool(choices), "bounded shelf layout must exist")
        pool = [p for p in choices if p[1]] if mode == 1 else choices
        if not pool:
            pool = choices
        offsets = pool[int(rng.integers(len(pool)))][0]
    minimum = np.min(offsets, axis=0)
    offsets = [tuple(np.asarray(x) - minimum) for x in offsets]
    extent = [max(offsets[i][a] + dims[i][a] for i in range(4)) for a in (0, 1)]
    pads = []
    for size in extent:
        room = 30 - size
        before = int(rng.integers(0, min(3, room) + 1))
        after = int(rng.integers(0, min(3, room - before) + 1))
        pads.append((before, after))
    shape = tuple(extent[a] + sum(pads[a]) for a in (0, 1))
    grid = np.full(shape, bg, dtype=np.int16)
    for i, tile in enumerate(tiles):
        q = np.argwhere(tile >= 0)
        x = q + (offsets[i][0] + pads[0][0], offsets[i][1] + pads[1][0])
        grid[x[:, 0], x[:, 1]] = tile[q[:, 0], q[:, 1]]
    if np.count_nonzero(_border(grid) == bg) <= len(_border(grid)) / 2:
        # A guaranteed blank frame fits: shelf layouts reserve two cells,
        # and a failed free placement is retried as a shelf placement.
        if mode == 0:
            return _scatter(rng, tiles, bg, 2)
        coords = np.argwhere(grid != bg)
        lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
        trimmed = grid[lo[0]:hi[0], lo[1]:hi[1]]
        return np.pad(trimmed, 1, constant_values=bg).astype(np.int16)
    return grid


def _make_pair(rng, n, orientation, direct, required_roles, palette, first=False):
    bg, substrate, glyph, marker = palette[:4]
    key = np.asarray(palette[4:], dtype=np.int16).reshape(2, 2)
    axes = ("H",) if direct and orientation == "wide" else (("V",) if direct else ("H", "V"))
    for attempt in range(240):
        owner = _partition(rng, n, int(rng.integers(2)))
        sites = _key_sites(owner)
        if not sites:
            continue
        kr, kc = sites[int(rng.integers(len(sites)))]
        bbox = _direct_window(rng, n, orientation) if direct else _shadow_window(rng, n, (kr, kc), orientation)
        if bbox is None:
            continue
        r0, r1, c0, c1 = bbox
        erased = np.zeros((n, n), dtype=bool)
        erased[r0:r1, c0:c1] = True
        legend = np.zeros_like(erased)
        legend[kr:kr + 2, kc:kc + 2] = True
        if np.any(erased & legend):
            continue
        visible = ~(erased | legend)
        if any(not any(visible[q] for q in _orbit(int(r), int(c), n, axes)) for r, c in np.argwhere(erased)):
            continue
        if not required_roles <= set(int(x) for x in owner[erased]):
            continue
        for _ in range(16):
            bits = _pattern(rng, n, axes, int(rng.integers(3)))
            # Require real signals from every needed output-corner role.
            active = set(int(x) for x in owner[erased & (bits == 1)])
            if not required_roles <= active or len(active) < 2 or not np.any(erased & (bits == 0)):
                continue
            count0, count1 = np.count_nonzero(visible & (bits == 0)), np.count_nonzero(visible & (bits == 1))
            if count1 < 2 or count0 <= count1:
                continue
            full = np.where(bits, glyph, substrate).astype(np.int16)
            board = full.copy()
            board[erased] = marker
            board[kr:kr + 2, kc:kc + 2] = key
            recovered = _recover(board, owner, key, (substrate, glyph, marker), marker)
            if recovered is None or recovered["cross_shard_cells"] < 1 or (not direct and recovered["composed_cells"] < 1):
                continue
            if direct and len(recovered["axes"]) != 1:
                continue
            if direct and recovered["half_turn_disagreements"] < 1:
                continue
            tiles = []
            for i in range(4):
                coords = np.argwhere(owner == i)
                lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
                tile = np.full(tuple(hi - lo), -1, dtype=np.int16)
                q = coords - lo
                tile[q[:, 0], q[:, 1]] = board[coords[:, 0], coords[:, 1]]
                rotation = 0 if i == 0 else (i if first else int(rng.integers(4)))
                tiles.append(np.rot90(tile, rotation).copy())
            inp = _scatter(rng, tiles, bg, int(rng.integers(3))).astype(np.int8)
            expected = np.where(bits[r0:r1, c0:c1], key.reshape(-1)[owner[r0:r1, c0:c1]], substrate).astype(np.int8)
            try:
                analyzed = _analyze(inp)
            except AssertionError:
                continue
            if not np.array_equal(analyzed["answer"], expected):
                continue
            if not required_roles <= analyzed["active_roles"] or analyzed["cross_shard_cells"] < 1 or (not direct and analyzed["composed_cells"] < 1):
                continue
            return {"input": inp, "output": expected}
    raise AssertionError("bounded construction exhausted; no ambiguous fallback is emitted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not _integer(num_test) or not 2 <= int(num_train) <= 10 or not 1 <= int(num_test) <= 4 or num_train <= num_test:
        raise ValueError("require integer counts 2..10 / 1..4 with num_train > num_test")
    rng = np.random.default_rng(seed)
    orientation = "wide" if rng.integers(2) else "tall"
    opposite = "tall" if orientation == "wide" else "wide"
    shift = int(rng.integers(3))  # scale increase, equal range, or scale decrease
    if shift == 0:
        train_range, test_range = (6, 10), (10, 13)
    elif shift == 1:
        train_range, test_range = (7, 12), (7, 12)
    else:
        train_range, test_range = (9, 13), (6, 10)
    required = ({0, 1}, {2, 3}) if orientation == "wide" else ({0, 2}, {1, 3})
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(int(num_train) + int(num_test)):
        train = i < num_train
        orient = orientation if train else opposite
        span = train_range if train else test_range
        need = required[i] if i < 2 else set()
        for _ in range(20):
            n = int(rng.integers(*span))
            palette = [int(x) for x in rng.choice(10, 8, replace=False)]
            if i == 1 and tuple(palette[4:]) == tuple(_parse(puzzle["train"][0]["input"])["key"].flat):
                palette[4], palette[5] = palette[5], palette[4]
            pair = _make_pair(rng, n, orient, i == 0, need, palette, first=i == 0)
            sig = _signature(pair["input"])
            if sig not in seen:
                seen.add(sig)
                puzzle["train" if train else "test"].append(pair)
                break
        else:
            raise AssertionError("unable to construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "counts")
    seen, training_roles, training_axes = set(), set(), set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            a = _analyze(inp)
            _require(np.array_equal(a["answer"], out), "incorrect reconstructed ink window")
            _require(a["mask_color"] not in out and a["glyph"] not in out and a["background"] not in out, "discarded roles in output")
            _require(a["blank_cells"] >= 1 and len(a["active_roles"]) >= 2, "both parents must affect the answer")
            _require(a["cross_shard_cells"] >= 1, "a query cell must need another assembled shard")
            # A-only raw pigment restoration and B-only solid key painting both fail.
            _require(not np.array_equal(a["raw_crop"], out), "corner inks unused")
            _require(not np.array_equal(a["key"].reshape(-1)[a["owner_crop"]], out), "engraving unused")
            records[split].append(a)
            if split == "train":
                training_roles.update(a["active_roles"])
                training_axes.update(a["axes"])
    _require(training_roles == {0, 1, 2, 3}, "demonstrate every key position")
    _require(training_axes == {"H", "V"}, "demonstrate both mirror directions")
    _require(any(len(a["axes"]) == 1 for a in records["train"]), "distinguish half-turn-only recovery")
    _require(any(len(a["axes"]) == 1 and a["half_turn_disagreements"] > 0 for a in records["train"]), "half-turn copying must give a wrong query cell")
    _require(any(a["composed_cells"] > 0 for a in records["train"]), "demonstrate composed recovery behind the legend")
    _require(len({tuple(a["key"].flat) for a in records["train"]}) >= 2, "show variable visible key inks")


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded checks on 42 seeds, geometry, ablations, and damaging mutations."""
    signatures, shapes, areas, cuts, keys, backgrounds, sides, boxes, turns = (set() for _ in range(9))
    composed = 0
    ambiguous_geometry = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        if seed in (0, 21, 41):
            _require(_puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed)), "determinism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list input solve")
                shapes.add(pair["input"].shape)
                areas.add(tuple(sorted(np.count_nonzero(a["owner"] == i) for i in range(4))))
                cuts.add(a["owner"].astype(np.int8).tobytes())
                keys.add(tuple(a["key"].flat))
                backgrounds.add(a["background"])
                sides.add(a["side"])
                boxes.add(pair["output"].shape)
                turns.update(a["turns"])
                composed += a["composed_cells"] > 0
                ambiguous_geometry += a["geometric_packings"] > 1
    _require(len(signatures) == 42, "42 distinct puzzles")
    _require(len(shapes) >= 25 and len(areas) >= 50 and len(cuts) >= 100, "structural diversity")
    _require(len(keys) >= 150 and backgrounds == set(range(10)), "palette roles")
    _require(len(sides) >= 6 and len(boxes) >= 20 and turns == {0, 1, 2, 3}, "geometry and motion diversity")
    _require(composed >= 120, "composition mechanism absent")
    _require(ambiguous_geometry > 0, "joint packing constraints unexercised")
    # The host verifier exercises every legal count combination; repeat only
    # boundaries here to keep this 42-seed self-test bounded.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(991 + nt * 10 + nq, nt, nq)
        _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A color bijection preserves every semantic role, including nonblack backgrounds.
    p = generate_puzzle(37, 2, 1)
    perm = np.roll(np.arange(10), 3)
    recolored = {s: [{k: perm[v] for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(recolored)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    # A center rectangle erases a complete H/V orbit. A second fixture makes
    # conflicting diagonal witnesses invisible to pairwise axis comparisons.
    n = 8
    rr, cc = np.indices((n, n))
    r, c = np.minimum(rr, n - 1 - rr), np.minimum(cc, n - 1 - cc)
    bits = (r * 2 + c) % 4 == 0
    full = np.where(bits, 2, 1).astype(np.int16)
    owner = 2 * (rr >= n // 2) + (cc >= n // 2)
    key = np.asarray([[4, 5], [6, 7]], dtype=np.int16)
    erased = full.copy()
    erased[:2, :2] = key
    erased[3:5, 3:5] = 3
    _require(_recover(erased, owner, key, (1, 2, 3), 3) is None, "fully erased orbit accepted")
    conflicting = full.copy()
    conflicting[1:3, 1:3] = key
    conflicting[5:7, 5:7] = 3
    conflicting[2, 5] = 3 - int(conflicting[2, 5])
    _require(_recover(conflicting, owner, key, (1, 2, 3), 3) is None, "conflicting composed witnesses accepted")


if __name__ == "__main__":
    self_test()
