"""Liquid mirrors: conserved fluid creates the midpoint reflectors.

Only the standard library and NumPy are used. No parent code or saved examples
are loaded. All semantics, including the material swatch, are input-visible.
"""
from __future__ import annotations

import json
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        part, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    part.add(p)
                    todo.append(p)
        result.append(part)
    return result


def _square(cells):
    if len(cells) != 4:
        return False
    rows, cols = zip(*cells)
    return max(rows)-min(rows) == max(cols)-min(cols) == 1


def _parse(inp):
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if counts.max() * 2 <= inp.size:
        raise ValueError("background must be a strict majority")
    objects = [(int(color), part) for color in values if color != bg
               for part in _components(inp == color)]
    swatches = [(color, part) for color, part in objects if _square(part)]
    if len(swatches) != 1:
        raise ValueError("need one isolated 2x2 material swatch")
    water, gauge = swatches[0]
    fluid = (inp == water).copy()
    for p in gauge:
        fluid[p] = False
    if not fluid.any():
        raise ValueError("need liquid outside the swatch")
    terrain = np.zeros(inp.shape, dtype=bool)
    sources = []
    for color, cells in objects:
        if color == water:
            continue
        rows, cols = zip(*cells)
        if len(cells) == 3 and max(rows)-min(rows) == max(cols)-min(cols) == 1:
            box = {(r, c) for r in (min(rows), max(rows))
                   for c in (min(cols), max(cols))}
            missing = next(iter(box-cells))
            elbow = (min(rows)+max(rows)-missing[0], min(cols)+max(cols)-missing[1])
            sources.append({"cells": cells, "elbow": elbow, "color": color,
                            "direction": (elbow[0]-missing[0], elbow[1]-missing[1])})
        elif len(cells) >= 5:
            for p in cells:
                terrain[p] = True
        else:
            raise ValueError("non-liquid objects must be Ls or terrain of at least five cells")
    if not sources or not terrain.any():
        raise ValueError("need emitters and receiving terrain")
    opaque = terrain.copy()
    for obj in sources:
        for p in obj["cells"]:
            opaque[p] = True
    for p in gauge:
        opaque[p] = True
    return bg, water, gauge, fluid, terrain, sources, opaque


def _settle(fluid, opaque, reverse=False, top_first=False):
    """Fall, seek a lateral downward outlet when supported, and accumulate.

    The accepted domain has complete horizontal levels and unique allocations.
    The two ordering switches check that no numerical tie-break is semantic.
    """
    h, w = fluid.shape
    wet = np.zeros(fluid.shape, dtype=bool)
    drops = sorted(map(tuple, np.argwhere(fluid)),
                   key=lambda p: (p[0] if top_first else -p[0], -p[1] if reverse else p[1]))
    for r, c in drops:
        r, c = int(r), int(c)
        while r >= 0 and wet[r, c]:
            r -= 1
        if r < 0 or opaque[r, c]:
            raise ValueError("liquid trapped under a ceiling or exceeding capacity")
        for _ in range(h+1):
            if r+1 < h and not (opaque[r+1, c] or wet[r+1, c]):
                r += 1
                continue
            left = right = c
            while left > 0 and not (opaque[r, left-1] or wet[r, left-1]):
                left -= 1
            while right+1 < w and not (opaque[r, right+1] or wet[r, right+1]):
                right += 1
            outlets = ([] if r+1 == h else
                       [x for x in range(left, right+1) if not (opaque[r+1, x] or wet[r+1, x])])
            if not outlets:
                wet[r, c] = True
                break
            c = min(outlets, key=lambda x: (abs(x-c), -x if reverse else x))
            r += 1
        else:
            raise ValueError("fall failed to terminate")
    return wet


def _surfaces(wet, strict=True):
    surfaces = []
    for pool in _components(wet):
        top = min(r for r, _ in pool)
        cols = sorted(c for r, c in pool if r == top)
        full = cols == list(range(cols[0], cols[-1]+1))
        valid = full and len(cols) >= 3 and len(cols) % 2 == 1
        if not valid:
            if strict:
                raise ValueError("each settled pool needs one contiguous odd-width surface of at least three cells")
            continue
        surfaces.append({"row": top, "left": cols[0], "right": cols[-1],
                         "center": cols[len(cols)//2], "cells": pool})
    return surfaces


def _trace(shape, wet, opaque, sources, water, *, strict=True, reflect=True):
    h, w = shape
    contacts = {(s["row"]-1, s["center"]): i
                for i, s in enumerate(_surfaces(wet, strict))}
    painted, routes = {}, []
    for source in sources:
        r, c = source["elbow"]
        dr, dc = source["direction"]
        color = source["color"]
        path, hits = [], []
        r, c = r+dr, c+dc
        while 0 <= r < h and 0 <= c < w:
            if opaque[r, c] or wet[r, c]:
                reason = "water" if wet[r, c] else "solid"
                break
            if reflect and dr == 1 and (r, c) in contacts:
                color = water
                dr = -1
                hits.append((contacts[r, c], (r, c)))
            p = (r, c)
            if p in painted and painted[p] != color:
                raise ValueError("incompatible light colors at a shared cell")
            painted[p] = color
            path.append((p, color))
            r, c = r+dr, c+dc
        else:
            reason = "boundary"
        routes.append({"path": path, "hits": hits, "end": reason})
    return painted, routes


def _details(value):
    inp = _grid(value)
    bg, water, gauge, fluid, terrain, sources, opaque = _parse(inp)
    wet = _settle(fluid, opaque)
    surfaces = _surfaces(wet)
    painted, routes = _trace(inp.shape, wet, opaque, sources, water)
    out = inp.copy()
    out[fluid] = bg
    out[wet] = water
    for p, color in painted.items():
        out[p] = color
    return out, {"bg": bg, "water": water, "gauge": gauge, "fluid": fluid,
                 "terrain": terrain, "opaque": opaque, "sources": sources,
                 "wet": wet, "surfaces": surfaces, "painted": painted, "routes": routes}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive material roles, settle the water, and trace all light from input."""
    return _details(input_grid)[0]


def _l_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r-dr, c), (r, c-dc)}


def _place_l(dry, sources, elbow, direction, color, bg):
    cells = _l_cells(elbow, direction)
    h, w = dry.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or dry[r, c] != bg:
            return None
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if p not in cells and 0 <= p[0] < h and 0 <= p[1] < w and dry[p] == color:
                return None
    a = dry.copy()
    for p in cells:
        a[p] = color
    obj = {"cells": cells, "elbow": elbow, "direction": direction, "color": color}
    return a, sources+[obj]


def _add_ray(rng, dry, sources, wet, bg, water, colors, kind, pool=None, dc=None, color=None):
    surfaces = _surfaces(wet)
    existing_path = set(_trace(dry.shape, wet, dry != bg, sources, water)[0])
    candidates = []
    if kind in ("hit", "miss"):
        s = surfaces[pool]
        for sign in ([dc] if dc is not None else [-1, 1]):
            # Offsets toward the incoming side leave the next diagonal cell
            # in the pool. Wider offsets also avoid an existing centered L's
            # one-cell arms, allowing absorbed misses in multi-ray scenes.
            offsets = [0] if kind == "hit" else range(1, (s["right"]-s["left"])//2+1)
            for offset in offsets:
                target = (s["row"]-1, s["center"]-sign*offset)
                for distance in range(2, 11):
                    candidates.append(((target[0]-distance, target[1]-sign*distance), (1, sign)))
    else:
        for r in range(2, min(9, dry.shape[0]-1)):
            for c in range(1, dry.shape[1]-1):
                for sign in (-1, 1):
                    candidates.append(((r, c), (-1, sign)))
    rng.shuffle(candidates)
    for elbow, direction in candidates:
        if _l_cells(elbow, direction) & existing_path:
            continue
        shade = int(rng.choice(colors)) if color is None else color
        trial = _place_l(dry, sources, elbow, direction, shade, bg)
        if trial is None:
            continue
        a, launches = trial
        try:
            _, routes = _trace(a.shape, wet, a != bg, launches, water)
        except ValueError:
            continue
        route = routes[-1]
        good = (len(route["hits"]) == 1 and route["hits"][0][0] == pool) if kind == "hit" else (
                    not route["hits"] and route["end"] == ("water" if kind == "miss" else "boundary"))
        if good and len(route["path"]) >= 3:
            return a, launches
    return None


def _cup(dry, left, width, surface, floors, color, bg):
    """Irregular cup with a contiguous solid floor and high outer rims."""
    right = left+width+1
    bottom = max(floors)
    cells = set()
    for c, floor in enumerate(floors, left+1):
        cells.update((r, c) for r in range(floor, bottom+1))
    for c in (left, right):
        cells.update((r, c) for r in range(surface-1, bottom+1))
    if any(dry[p] != bg for p in cells):
        raise ValueError("overlapping cups")
    for p in cells:
        dry[p] = color
    return {"left": left+1, "right": right-1, "surface": surface, "floors": list(floors)}


def _target(shape, cups, lower=0):
    wet = np.zeros(shape, dtype=bool)
    for cup in cups:
        for c, floor in enumerate(cup["floors"], cup["left"]):
            wet[cup["surface"]+lower:floor, c] = True
    return wet


def _geometry(rng, bg, terrain_colors, count, teaching=False):
    h = int(rng.integers(18, 31)) if count > 1 else int(rng.integers(15, 30))
    w = int(rng.integers(max(18, count*6+1), 31))
    dry = np.full((h, w), bg, dtype=np.int8)
    widths = [int(rng.choice([5, 7, 9, 11] if teaching else [3, 5, 7, 9])) for _ in range(count)]
    while sum(x+2 for x in widths)+count-1 > w-2:
        i = int(np.argmax(widths))
        widths[i] -= 2
    slack = w-sum(x+2 for x in widths)-(count-1)
    gaps = np.bincount(rng.integers(0, count+1, size=slack), minlength=count+1)
    left = int(gaps[0])
    cups = []
    for i, width in enumerate(widths):
        s = int(rng.integers(7, h-6))
        floors = list(map(int, s+rng.integers(3, 7, size=width)))
        # A raised internal ridge makes the lowest layers split into pockets.
        if teaching or rng.random() < .65:
            floors[width//2] = s+3
            floors[0] = floors[-1] = s+6
        color = int(rng.choice(terrain_colors))
        cups.append(_cup(dry, left, width, s, floors, color, bg))
        left += width+2+1+int(gaps[i+1])
    return dry, cups


def _gauge(rng, dry, sources, wets, bg, water, extra_paths=()):
    h, w = dry.shape
    options = [(r, c) for r in range(0, min(5, h-1)) for c in range(w-1)]
    rng.shuffle(options)
    paths = set(extra_paths)
    for wet in wets:
        paths.update(_trace(dry.shape, wet, dry != bg, sources, water)[0])
    for r, c in options:
        cells = {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}
        if cells & paths or any(dry[p] != bg or any(wet[p] for wet in wets) for p in cells):
            continue
        a = dry.copy()
        for p in cells:
            a[p] = water
        return a, cells
    return None


def _supply(rng, dry, cups, intended, bg, water, gauge, style):
    """Sample source shapes in each cup's vertical catchment, not target masks."""
    for _ in range(35):
        fluid = np.zeros(dry.shape, dtype=bool)
        for cup in cups:
            part = np.zeros(dry.shape, dtype=bool)
            eligible = np.zeros(dry.shape, dtype=bool)
            for c, floor in enumerate(cup["floors"], cup["left"]):
                part[:, c] = intended[:, c]
                eligible[1:floor, c] = True
            eligible &= dry == bg
            coords = np.argwhere(eligible)
            volume = int(part.sum())
            if len(coords) < volume:
                break
            jitter = rng.random(len(coords))
            if style == 0:  # left reservoir: transfer across the raised ridge
                score = 3*(coords[:, 1]-cup["left"])+.1*coords[:, 0]+jitter
            elif style == 1:  # dispersed drops
                score = jitter
            elif style == 2:  # low and partly settled
                score = -coords[:, 0]+jitter
            elif style == 3:  # a compact band high in the catchment
                score = coords[:, 0]+.15*abs(coords[:, 1]-(cup["left"]+cup["right"])/2)+jitter
            else:  # a vertical filament/cluster
                col = int(rng.integers(cup["left"], cup["right"]+1))
                score = 3*abs(coords[:, 1]-col)+jitter
            selected = coords[np.argsort(score)[:volume]]
            fluid[selected[:, 0], selected[:, 1]] = True
            if not np.any(fluid & eligible & ~intended):
                break
        else:
            # The swatch must remain the only isolated square, in every color.
            if any(_square(cells) for cells in _components(fluid)):
                continue
            try:
                if not all(np.array_equal(_settle(fluid, dry != bg, rev, order), intended)
                           for rev in (False, True) for order in (False, True)):
                    continue
                inp = dry.copy()
                inp[fluid] = water
                if int((inp == bg).sum())*2 <= inp.size:
                    continue
                out, info = _details(inp)
                if not any(r["hits"] for r in info["routes"]):
                    continue
                frozen = _trace(inp.shape, fluid, info["opaque"], info["sources"], water, strict=False)[0]
                if frozen == info["painted"]:
                    continue
                return {"input": inp, "output": out}
            except ValueError:
                continue
    return None


def _demonstrations(rng, bg, water, terrain_colors, light_colors):
    primary = int(rng.choice(light_colors))
    direction = int(rng.choice([-1, 1]))
    for _ in range(80):
        base, cups = _geometry(rng, bg, terrain_colors, 1, True)
        high = _target(base.shape, cups)
        low = _target(base.shape, cups, 1)
        a = _add_ray(rng, base, [], high, bg, water, light_colors, "hit", 0, direction, primary)
        if a is None:
            continue
        first, sources = a
        # The identical primary emitter now passes one column away from center.
        try:
            old_route = _trace(base.shape, low, first != bg, sources, water)[1][0]
        except ValueError:
            continue
        if old_route["hits"] or old_route["end"] != "water":
            continue
        b = _add_ray(rng, first, sources, low, bg, water, light_colors, "hit", 0, -direction)
        if b is None:
            continue
        second, launches = b
        # An upward launch establishes rotation decoding and boundary escape.
        b = _add_ray(rng, second, launches, low, bg, water, light_colors, "escape")
        if b is None:
            continue
        second, launches = b
        primary_path = _trace(first.shape, high, first != bg, sources, water)[0]
        key = _gauge(rng, second, launches, [low], bg, water, primary_path)
        if key is None:
            continue
        keyed, gauge = key
        first = first.copy()
        for p in gauge:
            first[p] = water
        p = _supply(rng, first, cups, high, bg, water, gauge, 0)
        q = _supply(rng, keyed, cups, low, bg, water, gauge, 0)
        if p is not None and q is not None:
            return [p, q]
    raise RuntimeError("failed to construct the controlled demonstrations")


def _ordinary(rng, bg, water, terrain_colors, light_colors, test):
    for _ in range(100):
        count = int(rng.integers(2, 4) if test else rng.integers(1, 4))
        dry, cups = _geometry(rng, bg, terrain_colors, count)
        intended = _target(dry.shape, cups)
        sources = []
        pool_order = list(range(count))
        rng.shuffle(pool_order)
        okay = True
        for pool in pool_order:
            trial = _add_ray(rng, dry, sources, intended, bg, water, light_colors, "hit", pool)
            if trial is None:
                okay = False
                break
            dry, sources = trial
        if not okay:
            continue
        wanted = int(rng.integers(3, 6) if test else rng.integers(1, 6))
        for _ in range(max(0, wanted-len(sources))):
            kind = str(rng.choice(["hit", "miss", "escape"]))
            pool = int(rng.integers(count)) if kind != "escape" else None
            trial = _add_ray(rng, dry, sources, intended, bg, water, light_colors, kind, pool)
            if trial is not None:
                dry, sources = trial
        if test and len(sources) < 3:
            continue
        key = _gauge(rng, dry, sources, [intended], bg, water)
        if key is None:
            continue
        dry, gauge = key
        pair = _supply(rng, dry, cups, intended, bg, water, gauge, int(rng.integers(5)))
        if pair is not None:
            return pair
    raise RuntimeError("failed to construct a diverse scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= int(num_train) <= 10 or not 1 <= int(num_test) <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, with train > test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    bg, water = palette[:2]
    terrain_colors, light_colors = palette[2:5], palette[5:]
    puzzle = {"train": _demonstrations(rng, bg, water, terrain_colors, light_colors), "test": []}
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        while len(puzzle[split]) < count:
            pair = _ordinary(rng, bg, water, terrain_colors, light_colors, split == "test")
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _closure(free, start):
    h, w = free.shape
    seen, todo = {start}, [start]
    while todo:
        r, c = todo.pop()
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= p[0] < h and 0 <= p[1] < w and free[p] and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen


def validate_generated_puzzle(puzzle):
    """Reconstruct material and optical invariants from returned arrays."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, infos = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _details(inp)
            assert np.array_equal(expected, out), "output disagrees with input-derived rule"
            assert np.any(inp != out)
            roles.add((info["bg"], info["water"]))
            infos[split].append(info)
            assert int(info["fluid"].sum()) == int(info["wet"].sum())
            assert all(np.array_equal(_settle(info["fluid"], info["opaque"], rev, order), info["wet"])
                       for rev in (False, True) for order in (False, True)), "ambiguous liquid allocation"
            assert np.argwhere(info["wet"])[:, 0].sum() > np.argwhere(info["fluid"])[:, 0].sum()
            assert np.array_equal(inp[info["opaque"]], out[info["opaque"]])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            for surface in info["surfaces"]:
                # Full flood closure below each meniscus rules out unsupported
                # cells, partial levels, open rims, and inaccessible low filling.
                free = ~info["opaque"] & (np.arange(inp.shape[0])[:, None] >= surface["row"])
                assert _closure(free, (surface["row"], surface["center"])) == surface["cells"]
            assert any(route["hits"] for route in info["routes"])
            assert {index for route in info["routes"] for index, _ in route["hits"]} == set(range(len(info["surfaces"])))
            assert all(len(route["hits"]) <= 1 for route in info["routes"])
            for route, source in zip(info["routes"], info["sources"]):
                previous = source["elbow"]
                dr, dc = source["direction"]
                color = source["color"]
                hit_cells = {point for _, point in route["hits"]}
                for point, shade in route["path"]:
                    assert point == (previous[0]+dr, previous[1]+dc)
                    assert not info["opaque"][point] and not info["wet"][point]
                    if point in hit_cells:
                        assert dr == 1
                        dr, color = -1, info["water"]
                    assert shade == color and out[point] == shade
                    previous = point
            # Removing A leaves the material settled but no optical marks.
            material_only = inp.copy()
            material_only[info["fluid"]] = info["bg"]
            material_only[info["wet"]] = info["water"]
            assert not np.array_equal(material_only, out)
            # Removing B must change the OPTICAL answer, beyond moving fluid.
            frozen = _trace(inp.shape, info["fluid"], info["opaque"], info["sources"],
                            info["water"], strict=False)[0]
            assert frozen != info["painted"]
            if split == "test":
                assert len(info["surfaces"]) >= 2 and len(info["sources"]) >= 3
    assert len(roles) == 1
    first, second = infos["train"][:2]
    assert len(first["surfaces"]) == len(second["surfaces"]) == 1
    assert second["surfaces"][0]["row"] == first["surfaces"][0]["row"]+1
    assert np.array_equal(first["terrain"], second["terrain"])
    primary = first["sources"][0]
    index = next(i for i, source in enumerate(second["sources"]) if source == primary)
    assert first["routes"][0]["hits"] and not second["routes"][index]["hits"]
    assert second["routes"][index]["end"] == "water"
    assert any(s["direction"][0] == -1 for s in second["sources"])
    assert any(r["hits"] for r in second["routes"])


def _regressions():
    # Explicit expected grids for a center hit and a one-row-lower miss.
    for depth in (2, 1):
        inp = np.zeros((13, 15), dtype=np.int8)
        inp[0:2, 0:2] = 1
        inp[6:11, 4] = inp[6:11, 10] = 5
        inp[10, 4:11] = 5
        inp[1, 2] = inp[2, 1] = inp[2, 2] = 2
        inp[3:3+depth, 5:10] = 1
        expected = inp.copy()
        expected[3:3+depth, 5:10] = 0
        expected[10-depth:10, 5:10] = 1
        if depth == 2:
            for p in ((3, 3), (4, 4), (5, 5), (6, 6)):
                expected[p] = 2
            for p in ((7, 7), (6, 8), (5, 9), (4, 10), (3, 11), (2, 12), (1, 13), (0, 14)):
                expected[p] = 1
        else:
            for n in range(3, 9):
                expected[n, n] = 2
        assert np.array_equal(solve(inp), expected)
        # Mirror horizontally: direction decoding, miss, and reflection rotate.
        assert np.array_equal(solve(inp[:, ::-1]), expected[:, ::-1])
        # Color permutation must commute with solving; zero can be material.
        permutation = np.array([7, 0, 9, 2, 8, 3, 4, 1, 5, 6])
        assert np.array_equal(solve(permutation[inp]), permutation[expected])


def self_test():
    """Bounded 42-seed variation, physics, counterfactual, and API checks."""
    _regressions()
    shapes, structures, volumes, widths, floors, colors, counts = (set() for _ in range(7))
    metrics = {"pairs": 0, "multi_pool": 0, "miss": 0, "lateral": 0,
               "branched": 0, "different_levels": 0, "retained": 0}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                _, info = _details(inp)
                assert np.array_equal(solve(inp.copy()), pair["output"])
                shapes.add(inp.shape)
                roles = {info["bg"]: 0, info["water"]: 1}
                normalized = np.array([roles.setdefault(int(x), len(roles)) for x in inp.flat], dtype=np.uint8)
                structures.add((inp.shape, normalized.tobytes()))
                volumes.add(int(info["wet"].sum()))
                widths.update(s["right"]-s["left"]+1 for s in info["surfaces"])
                colors.add((info["bg"], info["water"]))
                counts.add(len(info["sources"]))
                metrics["pairs"] += 1
                metrics["multi_pool"] += len(info["surfaces"]) > 1
                metrics["miss"] += any(not r["hits"] and r["end"] == "water" for r in info["routes"])
                metrics["lateral"] += not np.array_equal(info["fluid"].sum(axis=0), info["wet"].sum(axis=0))
                metrics["different_levels"] += len({s["row"] for s in info["surfaces"]}) > 1
                metrics["retained"] += bool(np.any(info["fluid"] & info["wet"]))
                branched = False
                for surface in info["surfaces"]:
                    profile = tuple(sum((r, c) in surface["cells"] for r in range(inp.shape[0]))
                                    for c in range(surface["left"], surface["right"]+1))
                    floors.add(profile)
                    below = np.zeros(inp.shape, dtype=bool)
                    for r, c in surface["cells"]:
                        if r > surface["row"]+2:
                            below[r, c] = True
                    branched |= len(_components(below)) > 1
                metrics["branched"] += branched
    assert len(signatures) == 42 and len(structures) >= 190
    assert len(shapes) >= 45 and len(volumes) >= 30 and len(floors) >= 80
    assert widths >= {3, 5, 7, 9, 11} and counts >= {1, 2, 3, 4, 5}
    assert len(colors) >= 30
    assert metrics["multi_pool"] >= 60 and metrics["miss"] >= 42
    assert metrics["lateral"] >= 100 and metrics["different_levels"] >= 40
    assert metrics["branched"] >= 30 and metrics["retained"] >= 60
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    for counts_pair in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(2026, *counts_pair)
        assert (len(p["train"]), len(p["test"])) == counts_pair
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    report = {"seeds": 42, "shapes": len(shapes), "structures": len(structures),
              "volumes": len(volumes), "floor_profiles": len(floors),
              "surface_widths": list(map(int, sorted(widths))), "source_counts": sorted(counts), **metrics}
    return report


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
