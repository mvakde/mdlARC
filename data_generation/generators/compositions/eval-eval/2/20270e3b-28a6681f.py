"""Registered sluices: assemble a chamber network and release its liquid.

Only NumPy and the standard library are used. Solver parsing, registration and
pouring use input pixels exclusively. Generation first selects complete stable
basins, then constructs safe supplies; it does not store a seed-specific answer.
"""
from collections import Counter
import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(not 1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(cells, diagonal=False):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            directions = ((_DIRECTIONS + ((-1, -1), (-1, 1), (1, -1), (1, 1)))
                          if diagonal else _DIRECTIONS)
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _cells(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _shift(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _box(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _enclosed(wall):
    t, b, l, r = _box(wall)
    free = {(y, x) for y in range(t - 1, b + 2)
            for x in range(l - 1, r + 2)} - wall
    return [p for p in _components(free)
            if all(t <= y <= b and l <= x <= r for y, x in p)]


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if (not np.all(g[0] == bg) or not np.all(g[-1] == bg)
            or not np.all(g[:, 0] == bg) or not np.all(g[:, -1] == bg)):
        raise ValueError("the display must have a uniform empty border")
    colors = list(map(int, np.unique(g)))
    if len(colors) != 4:
        raise ValueError("a scene has four distinct semantic colors")
    by_color = {v: _cells(g == v) for v in colors}
    candidates = []
    for marker in colors:
        if marker == bg:
            continue
        bars = _components(by_color[marker])
        lengths = Counter(map(len, bars))
        if (not 1 <= len(lengths) <= 3 or any(n != 2 for n in lengths.values())
                or any(len({c for r, c in bar}) != 1
                       or _box(bar)[1] - _box(bar)[0] + 1 != len(bar)
                       for bar in bars)):
            continue
        for solid in colors:
            if solid in (bg, marker):
                continue
            water = next(c for c in colors if c not in (bg, marker, solid))
            pieces = _components(by_color[solid] | by_color[marker], diagonal=True)
            if len(pieces) != len(lengths) + 1:
                continue
            records, matched = [], {}
            good = True
            for index, piece in enumerate(pieces):
                cavities = _enclosed(piece)
                own = [bar for bar in bars if bar <= piece]
                if len(cavities) != 1 or not 1 <= len(own) <= 2:
                    good = False
                    break
                cavity = cavities[0]
                t, b, l, r = _box(piece)
                sides = set()
                ports = []
                for bar in own:
                    y, _, x, _ = _box(bar)
                    side = -1 if x == l else 1 if x == r else 0
                    if (not side or side in sides
                            or not all((rr, cc - side) in cavity for rr, cc in bar)
                            or any((rr, cc + side) in piece for rr, cc in bar)
                            or (y - 1, x) not in piece
                            or (y + len(bar), x) not in piece):
                        good = False
                        break
                    sides.add(side)
                    port = {"length": len(bar), "row": y, "col": x,
                            "side": side, "cells": bar, "owner": index}
                    ports.append(port)
                    matched.setdefault(len(bar), []).append(port)
                records.append({"wall": piece & by_color[solid],
                                "cavity": cavity, "ports": ports,
                                "source": cavity & by_color[water], "box": (t, b, l, r)})
                if not good:
                    break
            if not good or set.union(*(p["cavity"] for p in records)) & by_color[marker]:
                continue
            if set.union(*(p["source"] for p in records)) != by_color[water]:
                continue
            outgoing, incoming = {}, {}
            for length in sorted(matched):
                ports = matched[length]
                if len(ports) != 2 or {p["side"] for p in ports} != {-1, 1}:
                    good = False
                    break
                left = next(p for p in ports if p["side"] == 1)
                right = next(p for p in ports if p["side"] == -1)
                a, b = left["owner"], right["owner"]
                if a == b or a in outgoing or b in incoming:
                    good = False
                    break
                outgoing[a] = (b, left, right)
                incoming[b] = a
            roots = [i for i in range(len(records)) if i not in incoming]
            if not good or len(roots) != 1:
                continue
            order, i = [], roots[0]
            while i not in order:
                order.append(i)
                if i not in outgoing:
                    break
                i = outgoing[i][0]
            if len(order) != len(records) or order[-1] in outgoing:
                continue
            candidates.append({"colors": (bg, solid, water, marker),
                               "pieces": records, "order": order, "links": outgoing})
    if len(candidates) != 1:
        raise ValueError("input must uniquely identify chamber, liquid and guide roles")
    return candidates[0]


def _assemble(parsed, ignore_rows=False, keep_plugs=False):
    offsets = {parsed["order"][0]: (0, 0)}
    for i in parsed["order"][:-1]:
        j, a, b = parsed["links"][i]
        dr, dc = offsets[i]
        row = (parsed["pieces"][i]["box"][0] + dr
               - parsed["pieces"][j]["box"][0]) if ignore_rows else a["row"] + dr - b["row"]
        offsets[j] = row, a["col"] + dc + 1 - b["col"]
    groups = []
    for i in parsed["order"]:
        p, (dr, dc) = parsed["pieces"][i], offsets[i]
        groups.append({"wall": _shift(p["wall"], dr, dc),
                       "cavity": _shift(p["cavity"], dr, dc),
                       "source": _shift(p["source"], dr, dc),
                       "guides": _shift(set.union(*(x["cells"] for x in p["ports"])), dr, dc)})
    wall = set.union(*(p["wall"] for p in groups))
    guides = set.union(*(p["guides"] for p in groups))
    source = set.union(*(p["source"] for p in groups))
    t, b, l, r = _box(wall)
    if max(b - t + 1, r - l + 1) > 30:
        raise ValueError("registered vessel exceeds ARC dimensions")
    for p in groups:
        for role in p:
            p[role] = _shift(p[role], -t, -l)
    bg, solid, water, _ = parsed["colors"]
    scene = np.full((b - t + 1, r - l + 1), bg, dtype=np.int8)
    for y, x in _shift(wall | (guides if keep_plugs else set()), -t, -l):
        scene[y, x] = solid
    for y, x in _shift(source, -t, -l):
        if scene[y, x] != bg:
            raise ValueError("source collides with registered walls")
        scene[y, x] = water
    return scene, groups


def _pour(scene, bg, water, reverse=False, top_first=False):
    """Discrete downward flow; no transport across a rim before it is reached."""
    result = scene.copy()
    points = sorted(_cells(scene == water),
                    key=lambda p: (p[0] if top_first else -p[0],
                                   -p[1] if reverse else p[1]))
    result[result == water] = bg
    h, w = result.shape
    for row, col in points:
        while row > 0 and result[row, col] == water:
            row -= 1
        if result[row, col] != bg:
            raise ValueError("supply exceeds a sealed local capacity")
        for _ in range(h + 1):
            if row + 1 < h and result[row + 1, col] == bg:
                row += 1
                continue
            left = right = col
            while left > 0 and result[row, left - 1] == bg:
                left -= 1
            while right + 1 < w and result[row, right + 1] == bg:
                right += 1
            exits = [] if row + 1 == h else [c for c in range(left, right + 1)
                                                  if result[row + 1, c] == bg]
            if not exits:
                result[row, col] = water
                break
            col = min(exits, key=lambda c: (abs(c - col), -c if reverse else c))
            row += 1
        else:
            raise AssertionError("fall did not terminate")
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All roles, ownership, matches and water volume are recovered from input."""
    parsed = _parse(input_grid)
    scene, _ = _assemble(parsed)
    bg, _, water, _ = parsed["colors"]
    return _pour(scene, bg, water)


def _capture(solid, pools):
    """Bitset of all dry downward destinations, not a hidden chosen direction."""
    h, w = solid.shape
    unknown = 1 << len(pools)
    fixed = np.zeros((h, w), dtype=np.int64)
    for i, pool in enumerate(pools):
        for p in pool:
            fixed[p] = 1 << i
    labels = np.zeros_like(fixed)
    for row in range(h - 1, -1, -1):
        c = 0
        while c < w:
            if solid[row, c]:
                c += 1
                continue
            end = c
            while end + 1 < w and not solid[row, end + 1]:
                end += 1
            outlets = 0
            for x in range(c, end + 1):
                outlets |= int(fixed[row, x])
                if row + 1 < h and not solid[row + 1, x]:
                    outlets |= int(labels[row + 1, x])
            for x in range(c, end + 1):
                if fixed[row, x]:
                    labels[row, x] = fixed[row, x]
                elif row + 1 < h and not solid[row + 1, x]:
                    labels[row, x] = labels[row + 1, x]
                else:
                    labels[row, x] = outlets or unknown
            c = end + 1
    return labels


def _pool_options(interior, groups):
    options = []
    for level in range(_box(interior)[0], _box(interior)[1] + 1):
        for pool in _components({p for p in interior if p[0] >= level}):
            if len(pool) >= 2 and min(r for r, c in pool) == level:
                covers = frozenset(i for i, g in enumerate(groups) if pool & g["cavity"])
                options.append((pool, covers))
    return options


def _choose_targets(rng, groups, mode):
    interior = set.union(*(p["cavity"] | p["guides"] for p in groups))
    options = _pool_options(interior, groups)
    n = len(groups)
    pools, covered = [], set()
    if mode in ("joined", "mixed"):
        big = [(p, ids) for p, ids in options
               if (len(ids) == n if mode == "joined" else 2 <= len(ids) < n)]
        if not big:
            raise ValueError("no requested communicating basin")
        # Prefer the lowest complete connected level, where actual spill matters.
        highest = max(min(r for r, c in p) for p, ids in big)
        big = [(p, ids) for p, ids in big if min(r for r, c in p) >= highest - 1]
        pool, ids = big[int(rng.integers(len(big)))]
        pools.append(pool)
        covered.update(ids)
    for i, group in enumerate(groups):
        if i in covered:
            continue
        bottom_gate = max(r for r, c in group["guides"])
        small = [p for p, ids in options if ids == {i}
                 and min(r for r, c in p) >= bottom_gate + 2]
        if not small:
            raise ValueError("no low basin below the local gates")
        pools.append(small[int(rng.integers(len(small)))])
    for i, a in enumerate(pools):
        touch = a | {(r + dr, c + dc) for r, c in a for dr, dc in _DIRECTIONS}
        if any(touch & b for b in pools[i + 1:]):
            raise ValueError("selected bodies touch")
    return pools


def _chamber(rng, h, w, left_length, right_length):
    ceiling = np.ones(w - 2, dtype=int)
    floor = np.full(w - 2, h - 2, dtype=int)
    style = int(rng.integers(4))
    if style == 1:
        floor[1:-1] -= np.arange(1, w - 3) % 3
    elif style == 2:
        floor[1:-1] -= rng.integers(0, 4, max(0, w - 4))
    elif style == 3:
        floor[1:-1] -= rng.integers(1, max(2, h // 2), max(0, w - 4))
    if rng.random() < .65:
        ceiling[1:-1] += rng.integers(0, 3, max(0, w - 4))
    cavity = {(r, c + 1) for c in range(w - 2)
              for r in range(int(ceiling[c]), int(floor[c]) + 1)}
    wall = {(r + dr, c + dc) for r, c in cavity for dr, dc in _DIRECTIONS} - cavity
    ports = []
    for side, length in ((-1, left_length), (1, right_length)):
        if not length:
            continue
        first = max(2, min(h - length - 2, h // 2 - length // 2 + int(rng.integers(-1, 2))))
        col = 0 if side == -1 else w - 1
        bar = {(r, col) for r in range(first, first + length)}
        assert bar <= wall
        wall -= bar
        ports.append({"length": length, "row": first, "col": col,
                      "side": side, "cells": bar})
    return {"wall": wall, "cavity": cavity, "source": set(), "ports": ports,
            "box": (0, h - 1, 0, w - 1)}


def _layout(rng, pieces, colors, staging):
    """Stage intact fragments with one or more blank cells between their boxes."""
    n = len(pieces)
    permutation = list(map(int, rng.permutation(n)))
    boxes = [p["box"] for p in pieces]
    sizes = [(b - t + 1, r - l + 1) for t, b, l, r in boxes]
    positions = {}
    gap = int(rng.integers(1, 4))
    if staging == "row":
        x = 0
        for i in permutation:
            positions[i] = int(rng.integers(0, 4)), x
            x += sizes[i][1] + gap
    elif staging == "column":
        y = 0
        for i in permutation:
            positions[i] = y, int(rng.integers(0, 4))
            y += sizes[i][0] + gap
    else:
        # Two rows; row/column assignment is unrelated to assembled order.
        upper = permutation[:(n + 1) // 2]
        lower = permutation[(n + 1) // 2:]
        y = 0
        for row in (upper, lower):
            x = int(rng.integers(0, 4))
            for i in row:
                positions[i] = y, x
                x += sizes[i][1] + gap
            if row:
                y += max(sizes[i][0] for i in row) + gap
    h = max(positions[i][0] + sizes[i][0] for i in range(n))
    w = max(positions[i][1] + sizes[i][1] for i in range(n))
    if h + 2 > 30 or w + 2 > 30:
        raise ValueError("staging exceeds display bounds")
    pads = [int(rng.integers(1, min(4, 30 - h))),
            int(rng.integers(1, min(4, 30 - w)))]
    extra_h = int(rng.integers(1, min(4, 31 - h - pads[0])))
    extra_w = int(rng.integers(1, min(4, 31 - w - pads[1])))
    bg, solid, water, marker = colors
    g = np.full((h + pads[0] + extra_h, w + pads[1] + extra_w), bg, dtype=np.int8)
    for i, piece in enumerate(pieces):
        y, x = positions[i]
        t, b, l, r = piece["box"]
        for cells, color in ((piece["wall"], solid), (piece["source"], water),
                             (set.union(*(p["cells"] for p in piece["ports"])), marker)):
            for rr, cc in cells:
                g[rr - t + y + pads[0], cc - l + x + pads[1]] = color
    return g


def _supply(rng, eligible, target, style):
    points = sorted(eligible)
    outside = sorted(eligible - target)
    if len(points) < len(target) or not outside:
        raise ValueError("not enough safe supply space")
    anchor = outside[int(rng.integers(len(outside)))]
    noise = rng.random(len(points))
    def score(i):
        r, c = points[i]
        dr, dc = abs(r - anchor[0]), abs(c - anchor[1])
        return (dr + dc, 3 * dr + .1 * dc, .1 * dr + 3 * dc,
                0, 4 * (points[i] not in target))[style] + noise[i]
    chosen = {points[i] for i in sorted(range(len(points)), key=score)[:len(target)]}
    if anchor not in chosen:
        chosen.remove(max(chosen))
        chosen.add(anchor)
    return chosen


def _lowest_fill(interior, volume):
    # Explicit shortcut used only in evidence checks, never by solve.
    return set(sorted(interior, key=lambda p: (-p[0], p[1]))[:volume])


def _make_pair(rng, colors, n, mode, size, staging, used, avoid_volume=None):
    for attempt in range(180):
        lengths = list(map(int, rng.choice(np.arange(1, 5), n - 1, replace=False)))
        pieces = []
        for i in range(n):
            h = int(rng.integers(*size[0]))
            w = int(rng.integers(*size[1]))
            pieces.append(_chamber(rng, h, w, lengths[i - 1] if i else 0,
                                   lengths[i] if i + 1 < n else 0))
        # Use the same pixel registration invariant on a conceptual uncolored
        # layout. This record is discarded before returning the puzzle.
        links = {}
        for i in range(n - 1):
            a = next(p for p in pieces[i]["ports"] if p["side"] == 1)
            b = next(p for p in pieces[i + 1]["ports"] if p["side"] == -1)
            links[i] = (i + 1, a, b)
        parsed = {"colors": colors, "pieces": pieces, "order": list(range(n)), "links": links}
        try:
            scene, groups = _assemble(parsed)
            actual_mode = "joined" if mode == "mixed" and (n == 2 or attempt >= 100) else mode
            pools = _choose_targets(rng, groups, actual_mode)
            bg, solid, water, marker = colors
            labels = _capture(scene == solid, pools)
            union_cavities = set.union(*(p["cavity"] for p in groups))
            source = set()
            for index, pool in enumerate(pools):
                eligible = {p for p in union_cavities if labels[p] == 1 << index}
                owners = [i for i, g in enumerate(groups) if pool & g["cavity"]]
                if len(owners) > 1:
                    omissions = [i for i in owners
                                 if len(eligible - groups[i]["cavity"]) >= len(pool)
                                 and (eligible - groups[i]["cavity"]) - pool]
                    if not omissions:
                        raise ValueError("cannot demonstrate actual cross-gate transfer")
                    omit = omissions[int(rng.integers(len(omissions)))]
                    eligible -= groups[omit]["cavity"]
                source |= _supply(rng, eligible, pool, int(rng.integers(5)))
            target = set.union(*pools)
            if len(target) == avoid_volume:
                continue
            interior = set.union(*(p["cavity"] | p["guides"] for p in groups))
            if actual_mode == "low":
                if len({min(r for r, c in p) for p in pools}) < 2:
                    continue
                if target == _lowest_fill(interior, len(source)):
                    continue
            if source == target:
                continue
            scene_source = scene.copy()
            out = scene.copy()
            for p in source:
                scene_source[p] = water
            for p in target:
                out[p] = water
            if not all(np.array_equal(_pour(scene_source, bg, water, rev, top), out)
                       for rev in (False, True) for top in (False, True)):
                continue
            # Pull the safe supplies back into each original fragment.
            for piece, group in zip(pieces, groups):
                a = min(piece["cavity"])
                b = min(group["cavity"])
                piece["source"] = _shift(source & group["cavity"], a[0] - b[0], a[1] - b[1])
            try:
                inp = _layout(rng, pieces, colors, staging)
            except ValueError:
                # A long stack may not fit the display; changing staging does
                # not change the registered vessel or the requested size shift.
                inp = _layout(rng, pieces, colors, "tiles" if n > 2 else "row")
            if _key(inp) in used:
                continue
            p = _parse(inp)
            reconstructed, _ = _assemble(p)
            if not np.array_equal(reconstructed, scene_source):
                raise AssertionError("staging changed registration or source accounting")
            # Neither a top-edge join nor rigid copying may explain a pair.
            wrong, _ = _assemble(p, ignore_rows=True)
            if np.array_equal(wrong, scene_source):
                continue
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded chamber construction exhausted")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.choice(10, 4, replace=False)))
    changing_solid = bool(rng.integers(2))
    regime = str(rng.choice(["network", "placement", "scale"]))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            n = 2 if index == 0 and split == "train" else int(rng.integers(2, 4))
            if regime == "network":
                n = (2 if index == 0 else 3) if split == "train" else 4
            if split == "train":
                mode = "low" if index == 0 else "joined" if index == 1 else str(rng.choice(["low", "joined", "mixed"]))
            else:
                mode = str(rng.choice(["low", "joined", "mixed"]))
            size = ((9, 13), (5, 8))
            if regime == "scale":
                size = ((8, 11), (5, 7)) if split == "train" else ((11, 15), (6, 9))
            stage = str(rng.choice(["row", "column", "tiles"]))
            if regime == "placement":
                # Three fragments fit in a row at these widths. Tests instead
                # use a two-row arrangement, so bounds repairs cannot erase the
                # intended placement shift.
                stage = "row" if split == "train" else "tiles"
                if split == "test":
                    n = 3
            colors = palette.copy()
            if changing_solid:
                colors[1] = int(rng.choice([c for c in range(10) if c not in (colors[0], colors[2], colors[3])]))
            avoid_volume = (int((puzzle["train"][0]["input"] == colors[2]).sum())
                            if split == "train" and index == 1 else None)
            pair = _make_pair(rng, tuple(colors), n, mode, size, stage, used, avoid_volume)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    return tuple(grid.shape), grid.tobytes()


def _inspect(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    parsed = _parse(inp)
    scene, groups = _assemble(parsed)
    bg, solid, water, marker = parsed["colors"]
    assert out.shape == scene.shape
    assert np.array_equal(out == solid, scene == solid), "assembled terrain changed"
    assert marker not in np.unique(out), "temporary guides survived"
    assert int((inp == water).sum()) == int((out == water).sum()) > 0
    assert np.array_equal(_pour(scene, bg, water), out), "wrong flow or registration"
    assert not np.array_equal(scene, out), "rigid water-copy shortcut survives"
    walls, wet = _cells(scene == solid), _cells(out == water)
    guides = set.union(*(p["guides"] for p in groups))
    interior = set.union(*(p["cavity"] | p["guides"] for p in groups))
    assert wet <= interior, "liquid escaped the vessel"
    pools = _components(wet)
    for pool in pools:
        top = min(r for r, c in pool)
        closure = next(p for p in _components({q for q in interior if q[0] >= top}) if p & pool)
        assert closure == pool, "partial level, leak or unsupported liquid"
    labels = _capture(scene == solid, pools)
    source = _cells(scene == water)
    for i, pool in enumerate(pools):
        assert sum(labels[p] == 1 << i for p in source) == len(pool), "unsafe catchment or wrong allocation"
    assert all(labels[p] > 0 and int(labels[p]).bit_count() == 1
               and labels[p] < 1 << len(pools) for p in source)
    # Since every chamber has an intact roof, bottom and outer wall, crop is
    # uniquely the wall bounding rectangle, regardless of settled pool levels.
    assert _box(walls) == (0, out.shape[0] - 1, 0, out.shape[1] - 1)
    wrong, _ = _assemble(parsed, ignore_rows=True)
    assert not np.array_equal(wrong, scene), "guide heights do not matter"
    transfer = any(not (source & g["cavity"]) and bool(wet & g["cavity"]) for g in groups)
    levels = {min(r for r, c in p) for p in pools}
    return {"colors": parsed["colors"], "chambers": len(groups),
            "lengths": tuple(sorted({p["length"] for g in parsed["pieces"] for p in g["ports"]})),
            "pools": len(pools), "levels": tuple(sorted(levels)),
            "volume": len(wet), "transfer": transfer,
            "dry_guides": len(guides - wet), "wet_guides": len(guides & wet),
            "global_shortcut_differs": wet != _lowest_fill(interior, len(wet)),
            "source_components": len(_components(source)),
            "lateral": not np.array_equal((scene == water).sum(axis=0), (out == water).sum(axis=0)),
            "normalized": _key(np.where(inp == bg, 0, np.where(inp == solid, 1,
                                         np.where(inp == water, 2, 3))).astype(np.int8))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, records = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _inspect(pair)
            roles.add((info["colors"][0], info["colors"][2], info["colors"][3]))
            records[split].append(info)
    assert len(roles) == 1, "background, liquid and guide roles vary within puzzle"
    low, joined = records["train"][:2]
    assert low["pools"] >= 2 and len(low["levels"]) >= 2
    assert low["global_shortcut_differs"] and low["wet_guides"] == 0
    assert joined["pools"] == 1 and joined["transfer"] and joined["wet_guides"] > 0
    assert len({p["volume"] for p in records["train"]}) >= 2


def self_test():
    """Bounded 42-seed diversity, invariant, mutation and interface checks."""
    signatures, structures, shapes, volumes, colors = set(), set(), set(), set(), set()
    chamber_counts, lengths, records = set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                          for split in ("train", "test") for p in puzzle[split])
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect(pair)
                records.append(info)
                structures.add(info["normalized"])
                shapes.add(tuple(np.asarray(pair["input"]).shape))
                volumes.add(info["volume"])
                colors.add(info["colors"])
                chamber_counts.add(info["chambers"])
                lengths.update(info["lengths"])
                p = _parse(pair["input"])
                scene, _ = _assemble(p)
                bg, solid, water, marker = p["colors"]
                for reverse in (False, True):
                    for top in (False, True):
                        assert np.array_equal(_pour(scene, bg, water, reverse, top), pair["output"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
        # Mutation tests exercise conservation and topology, not just equality
        # with the solver used to generate the outputs.
        altered = {s: [{k: _grid(v) for k, v in pair.items()} for pair in puzzle[s]] for s in puzzle}
        out = altered["train"][0]["output"]
        water = _parse(altered["train"][0]["input"])["colors"][2]
        out[tuple(np.argwhere(out == water)[0])] = _parse(altered["train"][0]["input"])["colors"][0]
        try:
            validate_generated_puzzle(altered)
        except AssertionError:
            pass
        else:
            raise AssertionError("water deletion was accepted")
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 25 and len(volumes) >= 25 and len(colors) >= 30
    assert chamber_counts == {2, 3, 4} and lengths == {1, 2, 3, 4}
    assert sum(r["transfer"] for r in records) >= 42
    assert sum(len(r["levels"]) > 1 for r in records) >= 42
    assert sum(r["lateral"] for r in records) >= 100
    assert sum(r["source_components"] > 1 for r in records) >= 100
    a, b = generate_puzzle(13), generate_puzzle(13)
    assert all(np.array_equal(pa[k], pb[k]) for s in a for pa, pb in zip(a[s], b[s]) for k in pa)
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(100 + train * 10 + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
