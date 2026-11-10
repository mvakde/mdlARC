"""Taut stencil chains: unary slack, motif-aware clearance, transparent copies.

The solver reads only the supplied grid. Sampling choices change presentation
and curriculum, never the transformation. See task_spec.md for the exact rule.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "invalid grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((g >= 0) & (g <= 9))), "invalid grid colors")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _bbox(cells):
    return (min(p[0] for p in cells), max(p[0] for p in cells),
            min(p[1] for p in cells), max(p[1] for p in cells))


def _components(cells, directions=DIR8):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        part, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    part.add(p)
                    todo.append(p)
        result.append(part)
    return result


def _local(p, endpoint, forward, lateral):
    r, c = p[0] - endpoint[0], p[1] - endpoint[1]
    return r * forward[0] + c * forward[1], r * lateral[0] + c * lateral[1]


def _global(p, endpoint, forward, lateral):
    u, v = p
    return (endpoint[0] + u * forward[0] + v * lateral[0],
            endpoint[1] + u * forward[1] + v * lateral[1])


def _local_stencil(stamp, forward, lateral):
    """Change coordinates only; copies retain the stamp's SCREEN orientation."""
    points = [_local((r, c), (0, 0), forward, lateral)
              for r in range(stamp.shape[0]) for c in range(stamp.shape[1])]
    u0, u1, v0, v1 = _bbox(points)
    tile = np.empty((u1 - u0 + 1, v1 - v0 + 1), dtype=np.int8)
    for (r, c), (u, v) in zip(np.ndindex(stamp.shape), points):
        tile[u - u0, v - v0] = stamp[r, c]
    return tile


def _clearance(tile, background, obstacle, rectangle=False):
    """Forbidden lateral translations, accounting for each motif row's notch."""
    h, w = tile.shape
    _require(all(np.any(row != background) for row in tile), "empty stencil axial row")
    left = [0 if rectangle else int(np.flatnonzero(row != background)[0]) for row in tile]
    clearance = {}
    for u, v in obstacle:
        _require(u >= h, "obstacle must be beyond the first stencil band")
        band, row = divmod(u, h)
        clearance[band] = max(clearance.get(band, -1000), v + 1 - left[row])
    return clearance


def _route(clearance, length):
    """Least unit-slope envelope with a supported parallel or diagonal tangent."""
    peak = max(clearance.values())
    tangent = max(v - k for k, v in clearance.items())
    _require(peak == 1 or (peak > 1 and tangent == 1), "unsupported incoming tangent")
    first_peak = min(k for k, v in clearance.items() if v == peak)
    n = max(length, max(clearance) + 4)
    path = [1 if peak == 1 and k <= first_peak else
            max(v - abs(k - t) for t, v in clearance.items()) for k in range(n)]
    slopes = [b - a for a, b in zip(path, path[1:])]
    _require(path[0] == 1, "attachment must be fixed")
    _require(all(s in (-1, 0, 1) for s in slopes) and
             slopes == sorted(slopes, reverse=True), "unsupported nonconvex route")
    return path[:length]


def _sweep_route(clearance, length):
    """Equivalent constructive sweeps, used when creating labels."""
    n = max(length, max(clearance) + 4)
    heights = [-1000] * n
    for k, v in clearance.items():
        heights[k] = v
    for k in range(1, n):
        heights[k] = max(heights[k], heights[k - 1] - 1)
    for k in range(n - 2, -1, -1):
        heights[k] = max(heights[k], heights[k + 1] - 1)
    if max(clearance.values()) == 1:
        first = min(k for k, v in clearance.items() if v == 1)
        heights[:first + 1] = [1] * (first + 1)
    return heights[:length]


def _read_input(g):
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(max(counts)) > g.size / 2, "background must be a strict majority")
    parts = _components(map(tuple, np.argwhere(g != bg)))
    _require(len(parts) == 3, "expected an attached assembly, an obstacle, and a stencil")
    palettes = [set(int(g[p]) for p in part) for part in parts]
    candidates = []
    for ai, assembly in enumerate(parts):
        if len(palettes[ai]) != 2:
            continue
        for oi, obstacle in enumerate(parts):
            if oi == ai or len(palettes[oi]) != 1:
                continue
            si = 3 - ai - oi
            if len(palettes[si]) < 2 or any(palettes[x] & palettes[y]
                                           for x, y in ((ai, oi), (ai, si), (oi, si))):
                continue
            sb = _bbox(parts[si])
            stamp = g[sb[0]:sb[1] + 1, sb[2]:sb[3] + 1].copy()
            if set(map(int, np.unique(stamp))) - {bg} != palettes[si]:
                continue
            if min(stamp.shape) < 2 or any(stamp[p] == bg for p in
                                         ((0, 0), (0, -1), (-1, 0), (-1, -1))):
                continue
            for support_color in sorted(palettes[ai]):
                support = {p for p in assembly if int(g[p]) == support_color}
                slack = assembly - support
                if not 3 <= len(support) <= 8 or not 2 <= len(slack) <= 9:
                    continue
                if len(_components(slack)) != 1:
                    continue
                if len({p[1] for p in support}) == 1:
                    positive = (1, 0)
                elif len({p[0] for p in support}) == 1:
                    positive = (0, 1)
                else:
                    continue
                if len(_components(support, DIR4)) != 1:
                    continue
                for endpoint, f in ((max(support), positive),
                                    (min(support), (-positive[0], -positive[1]))):
                    for lateral in ((f[1], -f[0]), (-f[1], f[0])):
                        local_slack = {_local(p, endpoint, f, lateral) for p in slack}
                        if ((0, 1) not in local_slack or
                                any(u > 0 or v < 1 for u, v in local_slack) or
                                {p for p in local_slack if p[0] == 0} != {(0, 1)}):
                            continue
                        tile = _local_stencil(stamp, f, lateral)
                        obs = {_local(p, endpoint, f, lateral) for p in obstacle}
                        try:
                            clearance = _clearance(tile, bg, obs)
                            route = _route(clearance, len(slack))
                        except ValueError:
                            continue
                        payload = {}
                        blocks = set()
                        for k, y in enumerate(route):
                            for a, b in np.ndindex(tile.shape):
                                p = _global((k * tile.shape[0] + a, y + b), endpoint, f, lateral)
                                blocks.add(p)
                                if tile[a, b] != bg:
                                    payload[p] = int(tile[a, b])
                        # Source stencil is an instruction, never part of a copy.
                        source_box = {(r, c) for r in range(sb[0], sb[1] + 1)
                                      for c in range(sb[2], sb[3] + 1)}
                        if blocks & source_box or set(payload) & (support | obstacle):
                            continue
                        candidates.append(dict(background=bg, support=support,
                                               support_color=support_color, slack=slack,
                                               signal=int(g[next(iter(slack))]), obstacle=obstacle,
                                               obstacle_color=int(g[next(iter(obstacle))]),
                                               source=parts[si], stamp=stamp, tile=tile,
                                               source_bbox=sb, endpoint=endpoint, forward=f,
                                               lateral=lateral, local_obstacle=obs,
                                               local_slack=local_slack, clearance=clearance,
                                               route=route, payload=payload, blocks=blocks))
    _require(len(candidates) == 1, "input must have one geometric reading")
    return candidates[0]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    info = _read_input(g)
    out = np.full(g.shape, info["background"], dtype=np.int8)
    for p in info["support"] | info["obstacle"]:
        out[p] = g[p]
    for p, color in info["payload"].items():
        if _inside(p, out.shape):
            out[p] = color
    return out


def _stencil(rng, shape, bg, payload, notch_rows=(), solid=False):
    h, w = shape
    r, c = np.indices(shape)
    style = int(rng.integers(5))
    if style == 0:
        labels = r + c
    elif style == 1:
        labels = r if rng.random() < .5 else c
    elif style == 2:
        labels = np.minimum.reduce((r, c, h - 1 - r, w - 1 - c))
    elif style == 3:
        labels = (2 * r + c) % len(payload)
    else:
        labels = rng.integers(len(payload), size=shape)
    tile = np.asarray(payload, dtype=np.int8)[labels % len(payload)].copy()
    # Corner identities make unchanged orientation visible, without a palette row.
    tile[0, 0], tile[-1, -1], tile[0, -1] = payload[0], payload[1], payload[-1]
    if not solid:
        for a in range(1, h - 1):
            if rng.random() < .35:
                width = int(rng.integers(1, min(3, w)))
                tile[a, :width] = bg
        for a in notch_rows:
            tile[a, 0] = bg
            tile[a, 1:] = np.asarray(payload)[(a + np.arange(w - 1)) % len(payload)]
        for a in range(1, h - 1):
            for b in range(1, w - 1):
                if rng.random() < .13 and (a not in notch_rows or b > 1):
                    tile[a, b] = bg
    return tile


def _slack(rng, length, support_length, witness=False):
    cells = {(0, 1)}
    depth = max(2, support_length - 1)
    width = int(rng.integers(3, 7))
    usable_width = width - 1 if witness else width
    depth = max(depth, (length - 2) // usable_width + 1)
    first = (-1, 2) if witness or rng.random() < .5 else (-1, 1)
    cells.add(first)
    style = int(rng.integers(3))
    while len(cells) < length:
        candidates = sorted({(u + du, v + dv) for u, v in cells for du, dv in DIR8
                             if -depth <= u + du <= -1 and 1 <= v + dv <= width} - cells)
        if witness:
            candidates = [p for p in candidates if p[1] >= 2]
        scores = []
        for p in candidates:
            neighbors = sum((p[0] + du, p[1] + dv) in cells for du, dv in DIR8)
            scores.append((neighbors + 1) ** 2 if style == 0 else
                          1 / (neighbors + 1) if style == 1 else 1)
        weights = np.asarray(scores, dtype=float)
        cells.add(candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))])
    return cells


def _raw_obstacle(rng, h):
    near = int(rng.integers(h, 2 * h + 1))
    height = int(rng.integers(1, min(8, 2 * h + 1)))
    width = int(rng.integers(1, 5))
    kind = int(rng.integers(6))
    if kind == 0:
        raw = {(0, 0)}
    elif kind == 1:
        raw = {(a, 0) for a in range(height)}
    elif kind == 2:
        raw = {(0, b) for b in range(width)}
    elif kind == 3:
        raw = {(a, b) for a in range(height) for b in range(width)}
    elif kind == 4:
        raw = {(a, b) for a in range(height)
               for b in range(max(1, width - abs(a - height // 2)))}
    else:
        raw = {(a, min(width - 1, a // 2)) for a in range(height)}
    return {(a + near, b) for a, b in raw}


def _transform(g, turns, mirror):
    x = np.rot90(g, turns)
    return (np.fliplr(x) if mirror else x).copy()


def _asymmetric(tile):
    return all(not np.array_equal(tile, _transform(tile, k, mirror))
               for k in range(4) for mirror in (False, True) if k or mirror)


def _place(rng, tile, bg, role_colors, support, slack, obstacle, path,
           turns, mirror, clip=False):
    h, w = tile.shape
    copied = {(k * h + a, y + b): int(tile[a, b])
              for k, y in enumerate(path) for a, b in np.ndindex(tile.shape)
              if tile[a, b] != bg}
    blocks = {(k * h + a, y + b) for k, y in enumerate(path)
              for a, b in np.ndindex(tile.shape)}
    assembly = support | slack | obstacle
    bounds = _bbox(assembly | blocks)
    layout = int(rng.integers(4))
    gap = int(rng.integers(2, 5))
    if layout < 2:
        su = int(rng.integers(bounds[0], bounds[1] - h + 2))
        sv = bounds[3] + gap + 1 if layout == 0 else bounds[2] - gap - w
    else:
        su = bounds[1] + gap + 1 if layout == 2 else bounds[0] - gap - h
        sv = int(rng.integers(bounds[2], max(bounds[2] + 1, bounds[3] - w + 2)))
    source_box = {(su + a, sv + b) for a, b in np.ndindex(tile.shape)}
    all_input = assembly | source_box
    full_bounds = list(_bbox(all_input | blocks))
    input_bounds = _bbox(all_input)
    clipped_edge = None
    if clip:
        options = [edge for edge in range(4) if
                   any((p[edge // 2] < input_bounds[edge] if edge % 2 == 0 else
                        p[edge // 2] > input_bounds[edge]) for p in copied)]
        if not options:
            raise ValueError("no visible clipping edge")
        clipped_edge = int(rng.choice(options))
        full_bounds[clipped_edge] = input_bounds[clipped_edge]
    for axis in range(2):
        size = full_bounds[axis * 2 + 1] - full_bounds[axis * 2] + 1
        if size > 30:
            raise ValueError("scene does not fit")
        budget = int(rng.integers(0, min(30 - size, 5) + 1))
        for _ in range(budget):
            edges = [2 * axis + e for e in (0, 1) if 2 * axis + e != clipped_edge]
            edge = int(rng.choice(edges))
            full_bounds[edge] += -1 if edge % 2 == 0 else 1
    u0, u1, v0, v1 = full_bounds
    inp = np.full((u1 - u0 + 1, v1 - v0 + 1), bg, dtype=np.int8)
    for cells, color in zip((support, slack, obstacle), role_colors):
        for u, v in cells:
            inp[u - u0, v - v0] = color
    inp[su - u0:su - u0 + h, sv - v0:sv - v0 + w] = tile
    out = np.full_like(inp, bg)
    for cells, color in ((support, role_colors[0]), (obstacle, role_colors[2])):
        for u, v in cells:
            out[u - u0, v - v0] = color
    for (u, v), color in copied.items():
        p = u - u0, v - v0
        if _inside(p, out.shape):
            _require(out[p] == bg, "payload collides with preserved material")
            out[p] = color
    _require(np.count_nonzero(inp == bg) > inp.size / 2, "insufficient background")
    return {"input": _transform(inp, turns, mirror), "output": _transform(out, turns, mirror)}


def _make_pair(rng, parameters, split, index, palette):
    witness = split == "train" and index < 2
    regime = parameters["regime"]
    clip = ((split == "train" and index == 2) or
            (split == "test" and parameters["clipping"] and rng.random() < .6))
    notch_test = split == "test" and regime != 3 and rng.random() < .45
    for attempt in range(180):
        if witness:
            h, w = parameters["witness_shape"]
            length = parameters["witness_lengths"][index]
        else:
            if regime == 1 and split == "test":
                h, w, length = int(rng.integers(2, 4)), int(rng.integers(2, 6)), int(rng.integers(6, 9))
            elif regime == 3 and split == "test":
                h, w, length = 2, int(rng.integers(4, 6)), int(rng.integers(3, 8))
            else:
                h, w = map(int, rng.integers(2, 6, size=2))
                if regime == 3 and split == "train":
                    h = int(rng.integers(3, 6))
                upper = 5 if regime == 1 and split == "train" else min(9, 24 // h)
                lower = 3 if regime == 2 and split == "train" else 2
                length = 2 if regime == 2 and split == "test" else int(rng.integers(lower, upper + 1))
            if notch_test:
                h = 3
        support_length = int(rng.integers(3, 7))
        support = {(-a, 0) for a in range(support_length)}
        slack = _slack(rng, length, support_length, witness)
        bg, sc, tc, oc = map(int, palette[:4])
        payload = list(map(int, palette[4:4 + int(rng.integers(2, 6))]))
        if witness:
            rows = parameters["witness_rows"]
            tile = _stencil(rng, (h, w), bg, payload, rows if index == 0 else (), solid=index == 1)
            if index == 0 and not _asymmetric(tile):
                continue
            obstacle = {(h + a, 1) for a in rows}
        elif notch_test:
            tile = _stencil(rng, (h, w), bg, payload, (1,))
            band = int(rng.integers(1, min(4, length)))
            obstacle = {(band * h + 1, 1)}
        else:
            tile = _stencil(rng, (h, w), bg, payload)
            obstacle = _raw_obstacle(rng, h)
            raw = _clearance(tile, bg, obstacle)
            parallel = rng.random() < .4
            offset = 1 - (max(raw.values()) if parallel else max(v - k for k, v in raw.items()))
            obstacle = {(u, v + offset) for u, v in obstacle}
        try:
            clearance = _clearance(tile, bg, obstacle)
            route = _route(clearance, length)
            _require(route == _sweep_route(clearance, length), "clearance implementations disagree")
            if regime == 0:
                turns = int(rng.choice((0, 2) if split == "train" else (1, 3)))
            else:
                turns = int(rng.integers(4))
            pair = _place(rng, tile, bg, (sc, tc, oc), support, slack, obstacle,
                          _sweep_route(clearance, length), turns, bool(rng.integers(2)), clip)
            info = _read_input(pair["input"])
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "construction mismatch")
            if witness:
                _require(all(_inside(p, pair["input"].shape) for p in info["blocks"]), "witness clipped")
            if clip:
                _require(any(not _inside(p, pair["input"].shape) for p in info["payload"]), "clipping lost")
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and low <= value <= high, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    h, w = int(rng.integers(3, 5)), int(rng.integers(3, 5))
    rows = list(range(1, h - 1))
    if h == 4 and rng.random() < .5:
        rows = [int(rng.integers(1, 3))]
    first_length = int(rng.integers(3, 5))
    parameters = dict(regime=int(rng.integers(4)), witness_shape=(h, w),
                      witness_rows=rows, witness_lengths=[first_length, first_length + 1],
                      clipping=num_train >= 3)
    palette = rng.permutation(10)
    stable = bool(rng.random() < .3)
    seen = set()
    puzzle = {"train": [], "test": []}
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            colors = palette if stable else rng.permutation(10)
            for _ in range(24):
                pair = _make_pair(rng, parameters, split, index, colors)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _metrics(pair):
    g = np.asarray(pair["input"])
    i = _read_input(g)
    obs = i["local_obstacle"]
    ob = _bbox(obs)
    rectangular = len(obs) == (ob[1] - ob[0] + 1) * (ob[3] - ob[2] + 1)
    hidden = [p for p in i["payload"] if not _inside(p, g.shape)]
    try:
        box_route = _route(_clearance(i["tile"], i["background"], obs, rectangle=True), len(i["slack"]))
        notch_changes_route = box_route != i["route"]
    except ValueError:
        notch_changes_route = True
    under_hole = bool((i["blocks"] - set(i["payload"])) & i["obstacle"])
    return dict(input_shape=g.shape, tile_shape=i["tile"].shape, background=i["background"],
                axis="vertical" if i["forward"][0] else "horizontal", length=len(i["slack"]),
                support_length=len(i["support"]), route=tuple(i["route"]),
                obstacle="point" if len(obs) == 1 else "rectangle" if rectangular else "stepped",
                silhouette=tuple(sorted((u - ob[0], v - ob[2]) for u, v in obs)),
                bands=len(i["clearance"]), parallel=max(i["clearance"].values()) == 1,
                holes=int(np.count_nonzero(i["tile"] == i["background"])),
                clipped=bool(hidden), needs_diagonal=len(_components(i["slack"], DIR4)) > 1,
                notch_changes_route=notch_changes_route, under_hole=under_hole,
                full=all(_inside(p, g.shape) for p in i["blocks"]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, metrics = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions must be preserved")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            i = _read_input(inp)
            bg, tile, length = i["background"], i["tile"], len(i["slack"])
            _require(np.array_equal(out, solve(inp)), "incorrect reconstructed output")
            _require(i["route"] == _sweep_route(i["clearance"], length), "route mismatch")
            _require(all(out[p] == inp[p] for p in i["support"] | i["obstacle"]), "fixed material changed")
            _require(all(out[p] == bg for p in i["source"]), "source stencil not erased")
            _require(i["signal"] not in out, "slack instruction color leaked into output")
            attachment = _global((0, 1), i["endpoint"], i["forward"], i["lateral"])
            _require(out[attachment] == tile[0, 0], "first copy lost its anchor")
            visible = {p: c for p, c in i["payload"].items() if _inside(p, inp.shape)}
            _require(len(i["payload"]) == length * int(np.count_nonzero(tile != bg)), "copies overlap")
            _require(sum(int(out[p] != bg) for p in np.ndindex(out.shape)) ==
                     len(i["support"]) + len(i["obstacle"]) + len(visible), "extra or missing payload")
            for color in set(map(int, tile.ravel())) - {bg}:
                _require(np.count_nonzero(out == color) == sum(c == color for c in visible.values()),
                         "motif color count not preserved after clipping")
            metrics[split].append(_metrics(pair))
    train = metrics["train"]
    _require(any(m["full"] and m["under_hole"] and m["notch_changes_route"] for m in train),
             "missing stencil-aware transparent-clearance lesson")
    _require(any(m["full"] and not m["parallel"] and not m["holes"] for m in train),
             "missing solid diagonal-clearance lesson")
    _require(any(m["full"] and m["needs_diagonal"] for m in train), "missing full unary-count lesson")
    _require(any(_asymmetric(_read_input(_grid(p["input"]))["tile"]) for p in puzzle["train"]),
             "missing unchanged-orientation lesson")
    _require(len({m["length"] for m in train}) >= 2, "count variation not demonstrated")
    if any(m["clipped"] for m in metrics["test"]):
        _require(any(m["clipped"] for m in train), "test clipping needs a demonstration")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded structural and semantic checks over 42 seeds and all count pairs."""
    # Exact normalized witnesses. The notch changes the trajectory before copying.
    assert _route({1: 1}, 5) == [1, 1, 0, -1, -2]
    assert _route({1: 2}, 5) == [1, 2, 1, 0, -1]
    assert _route({2: 3, 3: 3}, 6) == [1, 2, 3, 3, 2, 1]
    tile = np.array([[2, 3, 2], [3, 2, 3], [2, 2, 3]], dtype=np.int8)
    stepped = {(3, 1), (4, 1), (5, 0), (6, 0), (7, -1), (8, -1), (9, -2)}
    rectangle = {(u, v) for u in range(3, 10) for v in range(-2, 2)}
    assert _route(_clearance(tile, 0, stepped), 6) == [1, 2, 1, 0, -1, -2]
    assert _route(_clearance(tile, 0, rectangle), 6) == [1, 2, 2, 2, 1, 0]
    records, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        assert _equal(p, generate_puzzle(seed))
        signatures.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                m = _metrics(pair)
                records.append((split, m))
                # Equivariance exercises the input-only coordinate conversion.
                if seed < 4:
                    for turns in range(4):
                        for mirror in (False, True):
                            assert np.array_equal(solve(_transform(pair["input"], turns, mirror)),
                                                  _transform(pair["output"], turns, mirror))
    assert len(signatures) == 42
    rows = [m for _, m in records]
    assert {m["background"] for m in rows} == set(range(10))
    assert len({m["input_shape"] for m in rows}) >= 45
    assert len({m["tile_shape"] for m in rows}) >= 12
    assert len({(m["tile_shape"], m["route"], m["silhouette"]) for m in rows}) >= 90
    assert {m["obstacle"] for m in rows} == {"point", "rectangle", "stepped"}
    assert max(m["bands"] for m in rows) >= 3
    assert min(m["length"] for m in rows) == 2 and max(m["length"] for m in rows) >= 8
    for split in ("train", "test"):
        ms = [m for s, m in records if s == split]
        assert {m["axis"] for m in ms} == {"vertical", "horizontal"}
        assert {m["parallel"] for m in ms} == {False, True}
        assert any(m["clipped"] for m in ms) and any(m["full"] for m in ms)
        assert any(m["under_hole"] for m in ms)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(912, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 2.3):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    p = generate_puzzle(7, 2, 1)
    permutation = np.random.default_rng(84).permutation(10)
    for pair in p["train"] + p["test"]:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    damaged = generate_puzzle(8, 2, 1)
    i = _read_input(damaged["test"][0]["input"])
    damaged["test"][0]["output"][min(i["obstacle"])] = i["background"]
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted erased obstacle")
    return {"seeds": 42, "pairs": len(records),
            "input_shapes": len({m["input_shape"] for m in rows}),
            "tile_shapes": len({m["tile_shape"] for m in rows}),
            "structural_signatures": len({(m["tile_shape"], m["route"], m["silhouette"]) for m in rows}),
            "length_range": [min(m["length"] for m in rows), max(m["length"] for m in rows)],
            "clipped_by_split": dict(Counter(s for s, m in records if m["clipped"])),
            "transparent_overlap_by_split": dict(Counter(s for s, m in records if m["under_hole"]))}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
