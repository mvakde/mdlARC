"""Elected blueprints: shape-conditioned voting chooses a multiscale assembly.

Only standard-library modules and NumPy are used. No files or random state are
consulted by solve. Generation constructs its labels from explicit assembly
plans; validation recovers the plans from the returned input grids.
"""
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from itertools import combinations

import numpy as np


_TILE_SHAPES = ((2, 3), (3, 2), (2, 4), (4, 2),
                (3, 3), (3, 4), (4, 3), (4, 4))
_BALLOTS = ((2, 1), (2, 1), (3, 1), (3, 2), (2, 1, 1),
            (3, 2, 1), (3, 2, 2), (4, 3), (4, 2, 1),
            (2, 1, 1, 1), (4, 3, 2))


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _norm(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-y, c-x) for r, c in cells))


@lru_cache(maxsize=4096)
def _rotations(cells):
    result, current = set(), _norm(cells)
    for _ in range(4):
        result.add(current)
        current = _norm((c, -r) for r, c in current)
    return tuple(sorted(result))


@lru_cache(maxsize=4096)
def _variants(cells):
    return tuple(sorted(set(_rotations(cells)) |
                        set(_rotations(_norm((r, -c) for r, c in cells)))))


def _signature(mask):
    return min(_variants(_norm(np.argwhere(mask))))


def _extent(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _points(mask):
    return tuple((int(r), int(c)) for r, c in np.argwhere(mask))


def _connected(mask):
    points = set(_points(mask))
    if not points:
        return False
    todo, reached = [min(points)], set()
    while todo:
        r, c = todo.pop()
        if (r, c) in reached:
            continue
        reached.add((r, c))
        todo.extend(p for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                    if p in points and p not in reached)
    return reached == points


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _check_counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or
                not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def _origins(mask, rows, cols, positions):
    """Every supplied position must be active; the full padded window fits."""
    h, w = mask.shape
    return [(y, x) for y in range(rows-h+1) for x in range(cols-w+1)
            if all(0 <= r-y < h and 0 <= c-x < w and mask[r-y, c-x]
                   for r, c in positions)]


def _absolute(mask, origin):
    return {(origin[0]+r, origin[1]+c) for r, c in _points(mask)}


def _fragment(mask, rows, cols, origin, size, rng, also=()):
    shared = mask.copy()
    for other in also:
        shared &= other
    points = _points(shared)
    if len(points) < size:
        return None
    # At most C(11,5)=462 combinations in generic generation. No subset cache
    # or reconstructible trace payload is retained on disk.
    subsets = list(combinations(points, size))
    for index in rng.permutation(len(subsets)):
        local = subsets[int(index)]
        supplied = tuple((origin[0]+r, origin[1]+c) for r, c in local)
        if all(_origins(m, rows, cols, supplied) == [origin]
               for m in (mask, *also)):
            return supplied
    return None


def _variant_mask(mask, rng):
    h, w = mask.shape
    options = [v for v in _variants(_norm(_points(mask)))
               if _extent(v)[0] <= h and _extent(v)[1] <= w]
    cells = _choice(rng, options)
    mh, mw = _extent(cells)
    y, x = int(rng.integers(h-mh+1)), int(rng.integers(w-mw+1))
    result = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        result[y+r, x+c] = True
    return result


def _grow_mask(rng, shape, count):
    h, w = shape
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(count-1):
        frontier = sorted({(y, x) for r, c in cells
                           for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                           if 0 <= y < h and 0 <= x < w} - cells)
        cells.add(_choice(rng, frontier))
    mask = np.zeros(shape, dtype=bool)
    for p in cells:
        mask[p] = True
    return mask


def _texture(rng, shape, background, texture, masks):
    for _ in range(12):
        labels = rng.random(shape) < float(rng.uniform(.3, .8))
        if (labels.any() and not labels.all() and
                all(np.any(labels & ~mask) for mask in masks)):
            return np.where(labels, texture, background).astype(np.int8)
    # One background pixel can be chosen so no singleton inactive region is
    # lost. There are at most three prohibited pixels, and >=6 tile pixels.
    prohibited = {tuple(np.argwhere(~m)[0]) for m in masks if int((~m).sum()) == 1}
    allowed = [(r, c) for r in range(shape[0]) for c in range(shape[1])
               if (r, c) not in prohibited]
    tile = np.full(shape, texture, dtype=np.int8)
    tile[_choice(rng, allowed)] = background
    return tile


def _blank(geometry, divider, ordinary):
    h, w, t, rows, cols = geometry
    a = np.full((rows*(h+t)+t, cols*(w+t)+t), divider, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            y, x = t+r*(h+t), t+c*(w+t)
            a[y:y+h, x:x+w] = ordinary
    return a


def _put(grid, geometry, position, tile):
    h, w, t, _, _ = geometry
    y, x = t+position[0]*(h+t), t+position[1]*(w+t)
    grid[y:y+h, x:x+w] = tile


def _assemble(geometry, divider, background, ordinary, plans):
    inp = _blank(geometry, divider, ordinary)
    for plan in plans:
        for position, color, mask in plan["voters"]:
            _put(inp, geometry, position, np.where(mask, color, background))
    out = inp.copy()
    occupied = set()
    for plan in plans:
        active = _absolute(plan["mask"], plan["origin"])
        assert not occupied & active
        occupied |= active
        tile = np.where(plan["mask"], plan["color"], background)
        for position in active:
            _put(out, geometry, position, tile)
    return {"input": inp, "output": out}


def _dense_mask(rng, shape, count):
    mask = np.zeros(shape, dtype=bool)
    mask.flat[rng.choice(mask.size, count, replace=False)] = True
    return mask


def _teaching_layout(rng, a, b, companion, geometry):
    h, w, t, rows, cols = geometry
    origins = [(r, c) for r in range(rows-h+1) for c in range(cols-w+1)]
    layouts = []
    for oa in origins:
        union = _absolute(a | b, oa)
        for oc in origins:
            if not union & _absolute(companion, oc):
                layouts.append((oa, oc))
    if not layouts:
        return None
    for index in rng.permutation(len(layouts)):
        oa, oc = layouts[int(index)]
        focus = _fragment(a, rows, cols, oa, 7, rng, also=(b,))
        if focus is None:
            continue
        for size in (3, 4):
            other = _fragment(companion, rows, cols, oc, size, rng)
            if other is not None:
                return oa, oc, focus, other
    return None


def _teaching_pairs(rng, stats):
    for attempt in range(49):
        h, w = _choice(rng, ((3, 4), (4, 3)))
        rows = int(rng.integers(6, 8)) if h == 3 else int(rng.integers(4, 6))
        cols = int(rng.integers(6, 8)) if w == 3 else int(rng.integers(4, 6))
        geometry = h, w, 1, rows, cols
        count = int(rng.integers(8, 11))
        if attempt == 48:
            # Finite constructive fallback with a chiral ten-cell design.
            a = np.array([[0, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=bool)
            companion = np.array([[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=bool)
            h, w, geometry = 3, 4, (3, 4, 1, 7, 5)
            if stats is not None:
                stats["teaching_fallbacks"] += 1
        else:
            a = _dense_mask(rng, (h, w), count)
            if (not _connected(a) or _extent(_norm(_points(a))) != (h, w)):
                continue
            companion = None
            for _ in range(24):
                trial = _dense_mask(rng, (h, w), count)
                if (_connected(trial) and _extent(_norm(_points(trial))) == (h, w)
                        and _signature(trial) != _signature(a)):
                    companion = trial
                    break
            if companion is None:
                continue
        b = np.fliplr(a).copy()
        if (_norm(_points(b)) in _rotations(_norm(_points(a))) or
                int((a & b).sum()) < 7 or np.array_equal(a, b)):
            continue
        layout = _teaching_layout(rng, a, b, companion, geometry)
        if layout is not None:
            break
    else:
        raise AssertionError("constructive teaching layout failed")
    oa, oc, focus, other = layout
    palette = list(map(int, rng.permutation(10)))
    divider, background, texture, ca, cb, cc, cd = palette[:7]
    ordinary = _texture(rng, (h, w), background, texture, (a, b, companion))
    focus = [focus[int(i)] for i in rng.permutation(7)]
    # The first spatial voter stays unchanged across the reballot, excluding
    # a 'copy the first candidate' explanation of both demonstrations.
    if focus[2] == min(focus):
        focus[2], focus[-1] = focus[-1], focus[2]
    third_pose = _variant_mask(a, rng)
    other = [other[int(i)] for i in rng.permutation(len(other))]
    other_votes = [(p, cd, companion) for p in other[:-1]]
    other_votes.append((other[-1], ca, _variant_mask(companion, rng)))
    poses = {ca: a, cb: b, cc: third_pose}
    pairs = []
    for elected, colors in ((ca, (ca, ca, ca, cb, cb, cc, cc)),
                             (cb, (ca, ca, cb, cb, cb, cc, cc))):
        plans = [{"origin": oa, "mask": poses[elected], "color": elected,
                  "voters": [(p, c, poses[c]) for p, c in zip(focus, colors)]},
                 {"origin": oc, "mask": companion, "color": cd, "voters": other_votes}]
        pairs.append(_assemble(geometry, divider, background, ordinary, plans))
    return pairs


def _random_geometry(rng):
    h, w = _choice(rng, _TILE_SHAPES)
    t = 2 if rng.random() < .2 else 1
    def axis(side):
        maximum = (30-t)//(side+t)
        return maximum if rng.random() < .5 else int(rng.integers(side, maximum+1))
    return h, w, t, axis(h), axis(w)


def _random_pair(rng, known, is_test, stats):
    for _ in range(24):
        geometry = _random_geometry(rng)
        h, w, t, rows, cols = geometry
        cap = min(3, (rows*cols-1)//6)
        if cap < 1:
            continue
        count = int(rng.integers(1, cap+1))
        if t == 2 and h == w == 4:
            count = 1
        palette = list(map(int, rng.permutation(10)))
        divider, background, texture = palette[:3]
        inks = palette[3:]
        occupied, shapes, plans, supplied_count = set(), set(), [], 0
        for class_index in range(count):
            found = False
            for _ in range(40):
                maximum = min(11, h*w-1, rows*cols//count)
                if maximum < 4:
                    break
                area = int(rng.integers(4, maximum+1))
                mask = _grow_mask(rng, (h, w), area)
                signature = _signature(mask)
                if signature in shapes or (is_test and class_index == 0 and signature in known):
                    continue
                origins = [(r, c) for r in range(rows-h+1) for c in range(cols-w+1)
                           if not _absolute(mask, (r, c)) & occupied]
                if not origins:
                    continue
                budget = (rows*cols-1)//2 - supplied_count - 3*(count-class_index-1)
                ballots = [b for b in _BALLOTS if sum(b) < area and sum(b) <= budget]
                if not ballots:
                    continue
                for bi in rng.permutation(len(ballots)):
                    ballot = ballots[int(bi)]
                    for oi in rng.permutation(len(origins))[:8]:
                        origin = origins[int(oi)]
                        positions = _fragment(mask, rows, cols, origin, sum(ballot), rng)
                        if positions is not None:
                            found = True
                            break
                    if found:
                        break
                if found:
                    colors = list(map(int, rng.choice(inks, len(ballot), replace=False)))
                    poses = [mask] + [_variant_mask(mask, rng) for _ in colors[1:]]
                    ballots_flat = [(c, pose) for c, pose, n in zip(colors, poses, ballot)
                                    for _ in range(n)]
                    shuffled = rng.permutation(len(positions))
                    voters = [(positions[int(pi)], c, pose)
                              for pi, (c, pose) in zip(shuffled, ballots_flat)]
                    plans.append({"origin": origin, "mask": mask, "color": colors[0],
                                  "voters": voters})
                    occupied |= _absolute(mask, origin)
                    shapes.add(signature)
                    supplied_count += len(positions)
                    break
            if not found:
                break
        if len(plans) == count:
            ordinary = _texture(rng, (h, w), background, texture,
                                [p["mask"] for p in plans])
            return _assemble(geometry, divider, background, ordinary, plans)
    if stats is not None:
        stats["random_fallbacks"] += 1
    return _fallback_pair(rng, known if is_test else set())


def _fallback_pair(rng, forbidden):
    """Single connected novel stamp, with no crowding, under a finite search."""
    geometry = (3, 4, 1, 7, 5)
    h, w, t, rows, cols = geometry
    for code in rng.permutation(np.arange(1, 4095)):
        n = int(code).bit_count()
        if not 4 <= n <= 11:
            continue
        mask = np.array([(int(code) >> i) & 1 for i in range(12)], dtype=bool).reshape(h, w)
        if not _connected(mask) or _signature(mask) in forbidden:
            continue
        origin = int(rng.integers(rows-h+1)), int(rng.integers(cols-w+1))
        positions = _fragment(mask, rows, cols, origin, n-1, rng)
        if positions is None:
            continue
        palette = list(map(int, rng.permutation(10)))
        divider, background, texture, winner, loser = palette[:5]
        pose = _variant_mask(mask, rng)
        voters = [(p, winner, mask) for p in positions[:-1]] + [(positions[-1], loser, pose)]
        ordinary = _texture(rng, (h, w), background, texture, (mask,))
        return _assemble(geometry, divider, background, ordinary,
                         [{"origin": origin, "mask": mask, "color": winner, "voters": voters}])
    raise AssertionError("no novel conservative blueprint available")


def _generate(seed, num_train, num_test, stats=None):
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle = {"train": _teaching_pairs(rng, stats), "test": []}
    seen = {_key(p["input"]) for p in puzzle["train"]}
    known = set()
    for p in puzzle["train"]:
        known.update(_infer(p["input"])[1]["classes"])
    for split, n in (("train", int(num_train)-2), ("test", int(num_test))):
        for _ in range(n):
            for _ in range(12):
                pair = _random_pair(rng, known, split == "test", stats)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise AssertionError("bounded unique-input construction exhausted")
            puzzle[split].append(pair)
            seen.add(key)
            if split == "train":
                known.update(_infer(pair["input"])[1]["classes"])
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    return _generate(seed, num_train, num_test)


def _runs(indices):
    groups = []
    for index in map(int, indices):
        if groups and index == groups[-1][1]+1:
            groups[-1] = groups[-1][0], index
        else:
            groups.append((index, index))
    return groups


def _lattice(grid):
    divider = int(grid[0, 0])
    ys = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    xs = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    assert len(ys) >= 3 and len(xs) >= 3, "incomplete divider lattice"
    assert ys[0][0] == xs[0][0] == 0
    assert ys[-1][1] == grid.shape[0]-1 and xs[-1][1] == grid.shape[1]-1
    thicknesses = {b-a+1 for a, b in ys+xs}
    assert len(thicknesses) == 1
    t = thicknesses.pop()
    heights = {ys[i+1][0]-ys[i][1]-1 for i in range(len(ys)-1)}
    widths = {xs[i+1][0]-xs[i][1]-1 for i in range(len(xs)-1)}
    assert len(heights) == len(widths) == 1
    h, w = heights.pop(), widths.pop()
    assert (h, w) in _TILE_SHAPES and t in (1, 2)
    return (h, w, t, len(ys)-1, len(xs)-1), divider


def _infer(value):
    grid = _grid(value)
    geometry, divider = _lattice(grid)
    h, w, t, rows, cols = geometry
    records = {}
    for r in range(rows):
        for c in range(cols):
            y, x = t+r*(h+t), t+c*(w+t)
            tile = grid[y:y+h, x:x+w]
            assert not np.any(tile == divider), "divider in a tile"
            records.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
    ordinary, ordinary_positions = max(records.values(), key=lambda item: len(item[1]))
    assert len(ordinary_positions) * 2 > rows*cols, "ordinary tile needs a strict majority"
    ordinary_colors = set(map(int, np.unique(ordinary)))
    assert len(ordinary_colors) == 2
    classes = {}
    backgrounds = set()
    for tile, positions in records.values():
        if np.array_equal(tile, ordinary):
            continue
        inks = set(map(int, np.unique(tile))) - ordinary_colors
        assert len(inks) == 1, "one novel ink role per glyph"
        ink = inks.pop()
        mask = tile == ink
        assert 4 <= int(mask.sum()) <= 11 and not mask.all()
        assert _connected(mask), "glyph ink must be one connected shape"
        rest = set(map(int, np.unique(tile[~mask])))
        assert len(rest) == 1 and rest <= ordinary_colors
        backgrounds |= rest
        signature = _signature(mask)
        data = classes.setdefault(signature, {"votes": Counter(), "candidates": {}, "positions": set()})
        assert ink not in data["candidates"], "one consistent pose per shape/color"
        data["votes"][ink] += len(positions)
        data["candidates"][ink] = {"tile": tile, "mask": mask, "positions": tuple(positions)}
        data["positions"].update(positions)
    assert 1 <= len(classes) <= 3 and len(backgrounds) == 1
    background = backgrounds.pop()
    output, occupied = grid.copy(), set()
    for data in classes.values():
        votes = data["votes"]
        assert len(votes) >= 2 and sum(votes.values()) >= 3
        high = max(votes.values())
        winners = [color for color, n in votes.items() if n == high]
        assert len(winners) == 1, "tied shape-conditioned vote"
        winner = winners[0]
        tile = data["candidates"][winner]["tile"]
        mask = data["candidates"][winner]["mask"]
        assert len(data["positions"]) < int(mask.sum()), "a strict fragment is required"
        origins = _origins(mask, rows, cols, data["positions"])
        assert len(origins) == 1, "blueprint translation must be unique"
        origin = origins[0]
        active = _absolute(mask, origin)
        assert not occupied & active, "completed active footprints overlap"
        occupied |= active
        assert np.any(ordinary[~mask] != background), "whole-tile background copying must matter"
        for position in active:
            _put(output, geometry, position, tile)
        data.update(winner=winner, tile=tile, mask=mask, origin=origin, active=active,
                    missing=active-data["positions"])
    return output, {"geometry": geometry, "divider": divider, "ordinary": ordinary,
                    "background": background, "classes": classes, "active": occupied}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all elections and completions from the input alone."""
    return _infer(input_grid)[0]


def _teaching_witness(first, second):
    assert first["geometry"] == second["geometry"]
    assert np.array_equal(first["ordinary"], second["ordinary"])
    assert set(first["classes"]) == set(second["classes"])
    h, w, t, rows, cols = first["geometry"]
    witnesses = []
    for signature, a in first["classes"].items():
        b = second["classes"][signature]
        assert a["positions"] == b["positions"]
        if a["winner"] == b["winner"] or a["active"] == b["active"]:
            continue
        assert sorted(a["votes"].values()) == sorted(b["votes"].values()) == [2, 2, 3]
        assert a["origin"] == b["origin"]
        assert _origins(b["mask"], rows, cols, a["positions"]) == [b["origin"]]
        assert _norm(_points(b["mask"])) not in _rotations(_norm(_points(a["mask"])))
        others = set().union(*(d["active"] for s, d in first["classes"].items() if s != signature))
        assert not others & (a["active"] | b["active"])
        # A different exact shape of equal area AND equal normalized box must
        # win a different color in the same demonstration.
        assert any(s != signature and len(s) == len(signature) and
                   sorted(_extent(s)) == sorted(_extent(signature)) and
                   d["winner"] != a["winner"]
                   for s, d in first["classes"].items())
        first_position = min(a["positions"])
        original_tile = next(c["tile"] for c in a["candidates"].values()
                             if first_position in c["positions"])
        reballot_tile = next(c["tile"] for c in b["candidates"].values()
                             if first_position in c["positions"])
        assert np.array_equal(original_tile, reballot_tile)
        witnesses.append(signature)
    assert witnesses, "training must expose election-dependent geometry"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            expected, data = _infer(inp)
            assert np.array_equal(out, expected), "wrong election, footprint, or exact tile copy"
            assert not np.array_equal(inp, out)
            # Explicitly check preservation outside assemblies at coarse scale.
            h, w, t, rows, cols = data["geometry"]
            for r in range(rows):
                for c in range(cols):
                    if (r, c) not in data["active"]:
                        y, x = t+r*(h+t), t+c*(w+t)
                        assert np.array_equal(inp[y:y+h, x:x+w], out[y:y+h, x:x+w])
            for d in data["classes"].values():
                assert d["missing"]
                assert any(color != d["winner"] for color in d["votes"])
            analyses[split].append(data)
    _teaching_witness(*analyses["train"][:2])
    known = set().union(*(set(a["classes"]) for a in analyses["train"]))
    assert all(set(a["classes"])-known for a in analyses["test"]), "test must include a novel class"


def _puzzle_hash(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        digest.update(bytes((len(puzzle[split]),)))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _must_reject(function, *args):
    try:
        function(*args)
    except (AssertionError, ValueError):
        return
    raise AssertionError("damaged family member was accepted")


def self_test():
    """Stream 42 seeds; retain only small sets of signatures and measurements."""
    stats = {"teaching_fallbacks": 0, "random_fallbacks": 0}
    fingerprints, dimensions, geometries, masks, ballots, fragments, poses = (set() for _ in range(7))
    counts, thicknesses, palettes, holes = set(), set(), set(), 0
    tile_shapes, ink_areas = set(), set()
    pairs = 0
    windows_overlap = 0
    for seed in range(42):
        puzzle = _generate(seed, 4, 1, stats)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_hash(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                _, data = _infer(pair["input"])
                dimensions.add(pair["input"].shape)
                geometries.add(data["geometry"])
                thicknesses.add(data["geometry"][2])
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                counts.add(len(data["classes"]))
                h, w = data["geometry"][:2]
                tile_shapes.add((h, w))
                windows = []
                for signature, d in data["classes"].items():
                    masks.add(signature)
                    ink_areas.add(len(signature))
                    poses.add(_key(d["mask"]))
                    ballots.add(tuple(sorted(d["votes"].values(), reverse=True)))
                    fragments.add((signature, tuple(sorted((r-d["origin"][0], c-d["origin"][1])
                                                          for r, c in d["positions"]))))
                    y, x = d["origin"]
                    windows.append({(r, c) for r in range(y, y+h) for c in range(x, x+w)})
                    # Background holes are allowed, and are measured without
                    # constructing or retaining full execution traces.
                    background_points = set(_points(~d["mask"]))
                    while background_points:
                        todo = [background_points.pop()]
                        component = set(todo)
                        while todo:
                            r, c = todo.pop()
                            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                                if p in background_points:
                                    background_points.remove(p)
                                    component.add(p)
                                    todo.append(p)
                        holes += not any(r in (0, h-1) or c in (0, w-1) for r, c in component)
                windows_overlap += any(a & b for a, b in combinations(windows, 2))
                pairs += 1
    assert len(fingerprints) == 42
    assert len(dimensions) >= 20 and len(geometries) >= 25
    assert len(masks) >= 45 and len(poses) >= 80 and len(fragments) >= 100
    assert {1, 2, 3} <= counts and thicknesses == {1, 2}
    assert len(ballots) >= 7 and len(palettes) >= 20
    assert windows_overlap > 0 and holes > 0
    assert tile_shapes == set(_TILE_SHAPES)
    assert ink_areas == set(range(4, 12))
    p = generate_puzzle(0)
    assert _puzzle_hash(p) == _puzzle_hash(generate_puzzle(0))
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(100+ntrain, ntrain, ntest))
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (None, 1), (4, None), (4, False)):
        _must_reject(generate_puzzle, 0, *args)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    _must_reject(validate_generated_puzzle, bad)
    inp = p["train"][0]["input"]
    _, data = _infer(inp)
    focus = next(d for d in data["classes"].values()
                 if sorted(d["votes"].values()) == [2, 2, 3])
    tied = inp.copy()
    _put(tied, data["geometry"], focus["candidates"][focus["winner"]]["positions"][0], data["ordinary"])
    _must_reject(solve, tied)
    mixed = inp.copy()
    winner_positions = focus["candidates"][focus["winner"]]["positions"]
    _put(mixed, data["geometry"], winner_positions[0], np.fliplr(focus["tile"]))
    _must_reject(solve, mixed)
    # An ink-only paste incorrectly retains texture outside the winning mask.
    wrong = p["train"][0]["output"].copy()
    position = min(focus["missing"])
    h, w, t, _, _ = data["geometry"]
    dr, dc = np.argwhere((data["ordinary"] != data["background"]) & ~focus["mask"])[0]
    y, x = t+position[0]*(h+t)+int(dr), t+position[1]*(w+t)+int(dc)
    wrong[y, x] = data["ordinary"][dr, dc]
    assert not np.array_equal(solve(inp), wrong)
    bad_copy = {s: list(p[s]) for s in p}
    bad_copy["train"][0] = {"input": inp, "output": wrong}
    _must_reject(validate_generated_puzzle, bad_copy)
    # Exercise conservative constructions directly even when no sampled retry
    # budget was exhausted. These do not alter the sampled fallback counts.
    a = np.array([[0, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=bool)
    c = np.array([[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=bool)
    assert _norm(_points(np.fliplr(a))) not in _rotations(_norm(_points(a)))
    assert _teaching_layout(np.random.default_rng(200), a, np.fliplr(a), c,
                            (3, 4, 1, 7, 5)) is not None
    fallback = _fallback_pair(np.random.default_rng(200), masks)
    inferred, fallback_data = _infer(fallback["input"])
    assert np.array_equal(inferred, fallback["output"])
    assert set(fallback_data["classes"])-masks
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "canvas_shapes": len(dimensions), "geometries": len(geometries),
            "shape_classes": len(masks), "elected_masks": len(poses),
            "fragments": len(fragments), "ballots": sorted(ballots),
            "class_counts": sorted(counts), "thicknesses": sorted(thicknesses),
            "tile_shapes": sorted(tile_shapes), "ink_areas": sorted(ink_areas),
            "palettes": len(palettes), "mask_holes": holes,
            "overlapping_windows": windows_overlap, "fallback_probes_passed": True, **stats}
