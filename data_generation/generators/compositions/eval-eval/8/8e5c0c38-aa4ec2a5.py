"""Mirror-voted reservoirs: topology scopes same-color reflection votes.

Only NumPy and the standard library are used. Importing, generation, validation,
and self_test do not write files, cache examples, or use global random state.
"""
from collections import Counter
import hashlib

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _components(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _closed_regions(wall, neighbors=N4):
    h, w = wall.shape
    return [cells for cells in _components(~wall, neighbors)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)]


def _mask(shape, cells):
    out = np.zeros(shape, bool)
    for p in cells:
        out[p] = True
    return out


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _roles(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] * 2 <= grid.size:
        raise ValueError("background must occupy more than half the input")
    counts[bg] = 0
    maximum = counts.max()
    walls = np.flatnonzero(counts == maximum)
    if maximum == 0 or len(walls) != 1:
        raise ValueError("wall must be the unique most frequent non-background color")
    return bg, int(walls[0])


def _room_votes(grid, cells, bg, wall_color):
    r0, r1, c0, c1 = _bounds(cells)
    axis2 = c0 + c1
    if any((r, axis2 - c) not in cells for r, c in cells):
        raise ValueError("each sealed chamber must have a vertical mirror axis")
    totals, support = Counter(), Counter()
    for r, c in cells:
        color = int(grid[r, c])
        if color not in (bg, wall_color):
            totals[color] += 1
            if grid[r, axis2 - c] == color:
                support[color] += 1
    maximum = max(support.values(), default=0)
    winners = [color for color in support if support[color] == maximum]
    if maximum and len(winners) != 1:
        raise ValueError("positive reflection-support ties are outside the family")
    return {"cells": cells, "bounds": (r0, r1, c0, c1), "axis2": axis2,
            "totals": dict(totals), "support": dict(support),
            "winner": winners[0] if maximum else bg, "maximum": maximum}


def _analyze(input_grid):
    grid = _grid(input_grid)
    bg, wc = _roles(grid)
    wall = grid == wc
    rooms = [_room_votes(grid, cells, bg, wc) for cells in _closed_regions(wall)]
    inside = _mask(grid.shape, set().union(*(r["cells"] for r in rooms)))
    return {"grid": grid, "background": bg, "wall_color": wc, "wall": wall,
            "inside": inside, "outside": ~wall & ~inside, "rooms": rooms}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, chambers, axes and winning pigments from this input only."""
    analysis = _analyze(input_grid)
    out = analysis["grid"].copy()
    for room in analysis["rooms"]:
        for p in room["cells"]:
            out[p] = room["winner"]
    return out


def _dilate(mask, amount=1):
    out = mask.copy()
    for _ in range(amount):
        padded = np.pad(out, 1)
        out = np.logical_or.reduce([padded[1 + r:1 + r + out.shape[0],
                                            1 + c:1 + c + out.shape[1]]
                                    for r, c in ((0, 0),) + N8])
    return out


def _shape(rng, height, width, rectangular=False):
    """Connected row profiles; their empty spaces have no diagonal-only necks."""
    cuts = np.zeros(height, int)
    maximum_cut = max(0, (width - 3) // 2)
    style = 0 if rectangular else int(rng.integers(5))
    if style == 1:
        center = float(rng.uniform(0, height - 1))
        cuts = np.minimum(maximum_cut, (np.abs(np.arange(height) - center)).astype(int))
    elif style == 2:
        cuts = rng.integers(0, maximum_cut + 1, height)
    elif style == 3:
        cuts = np.minimum(maximum_cut, np.minimum(np.arange(height), np.arange(height)[::-1]))
    elif style == 4:
        cut = int(rng.integers(maximum_cut + 1))
        for r in range(height):
            cut = int(np.clip(cut + rng.integers(-1, 2), 0, maximum_cut))
            cuts[r] = cut
    cuts[int(rng.integers(height))] = 0
    mask = np.zeros((height, width), bool)
    for r, cut in enumerate(cuts):
        mask[r, cut:width - cut] = True
    return mask


def _orbits(mask):
    h, w = mask.shape
    pairs = [((r, c), (r, w - 1 - c)) for r, c in np.argwhere(mask).tolist()
             if 2 * c < w - 1]
    axes = [(r, w // 2) for r in range(h) if w % 2 and mask[r, w // 2]]
    return pairs, axes


def _speckles(rng, hole, kind, pigments, preferred=None):
    """Choose explicit reflected orbits, then add single-sided outliers."""
    pairs, axis = _orbits(hole)
    pairs = [pairs[int(i)] for i in rng.permutation(len(pairs))]
    colors = (list(map(int, rng.choice(pigments, 3, replace=False))) if preferred is None
              else [preferred, *map(int, rng.choice([c for c in pigments if c != preferred], 2, replace=False))])
    winner, rival, third = colors
    paint = {}
    cursor = 0

    def paired(color, count):
        nonlocal cursor
        for a, b in pairs[cursor:cursor + count]:
            paint[a] = paint[b] = color
        cursor += count

    def single(color, count):
        nonlocal cursor
        for orbit in pairs[cursor:cursor + count]:
            paint[orbit[int(rng.integers(2))]] = color
        cursor += count

    if kind == "generic":
        choices = ["minority", "contest", "empty", "paired"] + (["axis", "weighted"] if axis else [])
        kind = str(rng.choice(choices))
    if kind == "minority":
        paired(winner, 2)
        single(rival, 5)
    elif kind == "contest":
        paired(winner, 2)
        paired(rival, 1)
        single(rival, 3)
        if len(pairs) > cursor and rng.random() < .6:
            single(third, 1)
    elif kind == "axis":
        if not axis:
            raise ValueError("axis demonstration needs an odd chamber width")
        paint[axis[int(rng.integers(len(axis)))]] = winner
        single(rival, 4)
    elif kind == "weighted":
        # Pixel support, not number of orbits: two pairs give four votes;
        # three distinct self-reflecting axis cells give only three.
        paired(winner, 2)
        for i in rng.choice(len(axis), 3, replace=False):
            paint[axis[int(i)]] = rival
        single(rival, 2)
    elif kind == "empty":
        single(rival, int(rng.integers(2, 6)))
        if len(pairs) > cursor:
            single(third, 1)
    elif kind == "paired":
        paired(winner, int(rng.integers(1, min(3, len(pairs) - 2) + 1)))
        single(rival, int(rng.integers(1, 3)))
    else:
        raise ValueError("unknown pigment construction")
    return paint


def _tile(rng, kind, pigments, *, opened=False, small=False, preferred=None):
    if small:
        height, width, thickness = 4, (6 if kind == "minority" else 5), 1
        hole = _shape(rng, height, width, rectangular=True)
    else:
        height = int(rng.integers(4, 8))
        widths = ((5, 7, 9) if kind in ("axis", "weighted")
                  else (6, 8) if kind == "minority" else (5, 6, 7, 8, 9))
        width = int(rng.choice(widths))
        thickness = int(rng.choice([1, 1, 1, 2]))
        hole = _shape(rng, height, width)
        if len(_orbits(hole)[0]) < 8:
            hole[:] = True
    paint = _speckles(rng, hole, kind, pigments, preferred)
    hole = np.pad(hole, thickness)
    wall = _dilate(hole, thickness) & ~hole
    paint = {(r + thickness, c + thickness): color for (r, c), color in paint.items()}
    if opened:
        # One orthogonal tunnel connects the bay to the exterior. Paint is
        # transparent for topology, so none of its votes can seal this door.
        col = thickness + width // 2
        wall[:thickness, col] = False
    return {"wall": wall, "hole": np.zeros_like(hole) if opened else hole, "paint": paint}


def _pack(rng, tiles, shape, margin, gap, attempts=5):
    h, w = shape
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i]["wall"].size)
    for _ in range(attempts):
        occupied = np.zeros(shape, bool)
        offsets = [None] * len(tiles)
        for i in order:
            th, tw = tiles[i]["wall"].shape
            if th + 2 * margin > h or tw + 2 * margin > w:
                break
            prefix = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            sums = prefix[th:, tw:] - prefix[:-th, tw:] - prefix[th:, :-tw] + prefix[:-th, :-tw]
            valid = sums == 0
            if margin:
                valid[:margin] = valid[-margin:] = False
                valid[:, :margin] = valid[:, -margin:] = False
            choices = np.argwhere(valid)
            if not len(choices):
                break
            r, c = map(int, choices[int(rng.integers(len(choices)))])
            offsets[i] = (r, c)
            occupied[max(0, r - gap):min(h, r + th + gap),
                     max(0, c - gap):min(w, c + tw + gap)] = True
        else:
            return offsets
    return None


def _join_walls(rng, wall, expected_holes, protected):
    """Optional corridors join housings without creating or merging chambers."""
    for _ in range(int(rng.integers(0, 4))):
        parts = _components(wall)
        if len(parts) < 2:
            break
        ia, ib = map(int, rng.choice(len(parts), 2, replace=False))
        a, b = sorted(parts[ia]), sorted(parts[ib])
        distance, p, q = min((abs(r - rr) + abs(c - cc), (r, c), (rr, cc))
                             for r, c in a for rr, cc in b)
        if distance > 12:
            continue
        candidate = wall.copy()
        (r, c), (rr, cc) = p, q
        if rng.random() < .5:
            candidate[min(r, rr):max(r, rr) + 1, c] = True
            candidate[rr, min(c, cc):max(c, cc) + 1] = True
        else:
            candidate[r, min(c, cc):max(c, cc) + 1] = True
            candidate[min(r, rr):max(r, rr) + 1, cc] = True
        if np.any(candidate & expected_holes) or any(candidate[p] for p in protected):
            continue
        closed = _closed_regions(candidate)
        actual = _mask(wall.shape, set().union(*closed))
        diagonal = _mask(wall.shape, set().union(*_closed_regions(candidate, N8)))
        if np.array_equal(actual, expected_holes) and np.array_equal(actual, diagonal):
            wall = candidate
    return wall


def _teaching_link(wall, expected_holes, protected):
    """Find one topology-preserving link between two closed-room housings."""
    parts = _components(wall)
    owners = Counter()
    for room in _closed_regions(wall):
        boundary = {(r + dr, c + dc) for r, c in room for dr, dc in N4} - room
        for i, part in enumerate(parts):
            if part & boundary:
                owners[i] += 1
    if max(owners.values(), default=0) >= 2:
        return wall
    routes = []
    for i in owners:
        for j in owners:
            if i < j:
                routes.append(min((abs(r-rr)+abs(c-cc),(r,c),(rr,cc))
                                  for r,c in sorted(parts[i]) for rr,cc in sorted(parts[j])))
    for _, (r,c), (rr,cc) in sorted(routes):
        for direction in (0, 1):
            candidate = wall.copy()
            if direction:
                candidate[min(r,rr):max(r,rr)+1,c] = True
                candidate[rr,min(c,cc):max(c,cc)+1] = True
            else:
                candidate[r,min(c,cc):max(c,cc)+1] = True
                candidate[min(r,rr):max(r,rr)+1,cc] = True
            if np.any(candidate & expected_holes) or any(candidate[p] for p in protected):
                continue
            if all(np.array_equal(_mask(wall.shape, set().union(*_closed_regions(candidate, neighbors))),
                                  expected_holes) for neighbors in (N4, N8)):
                return candidate
    return None


def _make_pair(rng, kinds, opened, palette, compact=False):
    bg, wc = map(int, palette[:2])
    pigments = list(map(int, palette[2:]))
    construction = [(kind, False) for kind in kinds] + ([("minority", True)] if opened else [])
    teaching = kinds == ["axis", "contest", "weighted"]
    preferred = [pigments[i] if teaching and i < 3 else None for i in range(len(construction))]
    tiles = [_tile(rng, kind, pigments, opened=op, small=compact, preferred=preferred[i])
             for i, (kind, op) in enumerate(construction)]
    offsets = None
    shape = (30, 30)
    for _ in range(0 if compact else 3):
        shape = tuple(map(int, rng.integers(18, 31, 2)))
        offsets = _pack(rng, tiles, shape, int(rng.integers(0, 3)), int(rng.integers(1, 3)))
        if offsets is not None:
            break
    if offsets is None and not compact:
        shape = (30, 30)
        offsets = _pack(rng, tiles, shape, 0, 1, attempts=8)
    if offsets is None:
        # Six <=6x8 tiles fit in two shelves, preserving all teaching cases.
        # This fallback reduces size, never drops a required room or marker.
        tiles = [_tile(rng, kind, pigments, opened=op, small=True, preferred=preferred[i])
                 for i, (kind, op) in enumerate(construction)]
        shape = (int(rng.integers(17, 25)), int(rng.integers(25, 31)))
        offsets = [(1 + 8 * (i // 3), 1 + 8 * (i % 3)) for i in range(len(tiles))]
    wall = np.zeros(shape, bool)
    holes = np.zeros(shape, bool)
    paint = {}
    for tile, (r, c) in zip(tiles, offsets):
        th, tw = tile["wall"].shape
        wall[r:r + th, c:c + tw] |= tile["wall"]
        holes[r:r + th, c:c + tw] |= tile["hole"]
        paint.update({(r + rr, c + cc): value for (rr, cc), value in tile["paint"].items()})
    # Keep the first teaching bay readable: corridors are allowed only if
    # their topology is unchanged, and never pass through a colored speckle.
    wall = _join_walls(rng, wall, holes, paint)
    if teaching:
        linked = _teaching_link(wall, holes, paint)
        if linked is None:
            _require(not compact, "compact teaching link must exist")
            return _make_pair(rng, kinds, opened, palette, compact=True)
        wall = linked
    inp = np.full(shape, bg, np.int8)
    inp[wall] = wc
    for p, value in paint.items():
        inp[p] = value
    exterior = ~(wall | holes)
    candidates = np.argwhere(exterior & (inp == bg))
    if len(candidates):
        for i in rng.choice(len(candidates), min(int(rng.integers(0, 5)), len(candidates)), replace=False):
            inp[tuple(candidates[int(i)])] = int(rng.choice(pigments))
    # Dense thick-wall packings can fit geometrically while obscuring the
    # majority-background cue. Add visible exterior space, then (only if a
    # full 30x30 canvas is still too dense) use the bounded thin-wall fallback.
    if 2 * int(np.count_nonzero(inp != bg)) >= inp.size:
        padding = ((0, 30 - shape[0]), (0, 30 - shape[1]))
        inp = np.pad(inp, padding, constant_values=bg)
        wall = np.pad(wall, padding)
        holes = np.pad(holes, padding)
        shape = inp.shape
        if 2 * int(np.count_nonzero(inp != bg)) >= inp.size:
            _require(not compact, "compact density bound violated")
            return _make_pair(rng, kinds, opened, palette, compact=True)
    _require(_roles(inp) == (bg, wc), "construction lost its visible role convention")
    _require(np.array_equal(_mask(shape, set().union(*_closed_regions(wall))), holes),
             "placement altered chamber topology")
    return {"input": inp, "output": solve(inp)}


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _signature(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if split == "train" and index == 0:
                kinds, opened = ["minority", "empty"], True
            elif split == "train" and index == 1:
                kinds, opened = ["axis", "contest", "weighted"], bool(rng.integers(2))
            else:
                n = int(rng.integers(4, 6) if split == "test" else rng.integers(1, 4))
                kinds, opened = ["generic"] * n, bool(rng.integers(2))
            palette = rng.permutation(10)
            pair = _make_pair(rng, kinds, opened, palette)
            key = _signature(pair["input"])
            if key in seen:
                # More available exterior cells than the maximum 13 previous
                # pairs guarantee a distinct transparent, preserved marker.
                a = _analyze(pair["input"])
                for p in map(tuple, np.argwhere(a["outside"] & (pair["input"] == a["background"])).tolist()):
                    candidate = pair["input"].copy()
                    candidate[p] = int(palette[2])
                    if _signature(candidate) not in seen:
                        pair = {"input": candidate, "output": solve(candidate)}
                        key = _signature(candidate)
                        break
                else:
                    raise AssertionError("no distinct exterior-marker variant")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _features(analysis):
    bg, wc, grid = analysis["background"], analysis["wall_color"], analysis["grid"]
    rooms = analysis["rooms"]
    minority = [r for r in rooms if r["maximum"] and
                max(r["totals"].values()) > r["totals"][r["winner"]]]
    axial = [r for r in rooms if r["maximum"] == 1 and r["axis2"] % 2 == 0]
    orbit_disagreement = 0
    for room in rooms:
        orbit_support = Counter()
        for r, c in room["cells"]:
            color = int(grid[r, c])
            if (color not in (bg, wc) and 2 * c <= room["axis2"]
                    and grid[r, room["axis2"] - c] == color):
                orbit_support[color] += 1
        if room["maximum"] and any(n > orbit_support[room["winner"]]
                                   for color, n in orbit_support.items() if color != room["winner"]):
            orbit_disagreement += 1
    bay = 0
    # A pigment in exterior space bracketed by the same housing's walls is
    # direct open-bay evidence. This does not classify holes by row gaps.
    exterior_paint = analysis["outside"] & (grid != bg)
    for component in _components(analysis["wall"]):
        r0, r1, c0, c1 = _bounds(component)
        for r, c in np.argwhere(exterior_paint).tolist():
            if (r0 < r < r1 and c0 < c < c1
                    and any((r, cc) in component for cc in range(c0, c))
                    and any((r, cc) in component for cc in range(c + 1, c1 + 1))):
                bay += 1
    housing_counts, distinct_housing = [], False
    for component in _components(analysis["wall"]):
        owned = [room for room in rooms if any((r + dr, c + dc) in component
                                               for r, c in room["cells"] for dr, dc in N4)]
        housing_counts.append(len(owned))
        distinct_housing |= len({room["winner"] for room in owned if room["maximum"]}) >= 2
    return {"rooms": len(rooms), "minority": len(minority), "axial": len(axial),
            "empty": sum(not r["maximum"] for r in rooms), "bay_markers": bay,
            "housings": len(housing_counts), "shared_housing": max(housing_counts, default=0),
            "distinct_housing": distinct_housing,
            "parities": {r["axis2"] % 2 for r in rooms},
            "orbit_disagreement": orbit_disagreement,
            "contests": sum(sum(s > 0 for s in r["support"].values()) >= 2 for r in rooms)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, features = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyze(inp)
            _require(bool(a["rooms"]), "no sealed reservoir")
            # Background connectivity is cross-checked with eight neighbors;
            # this rejects apparent rooms that leak only through a corner.
            eight = _closed_regions(a["wall"], N8)
            _require({frozenset(c) for c in eight} == {frozenset(r["cells"]) for r in a["rooms"]},
                     "diagonally ambiguous topology")
            _require(np.array_equal(out[~a["inside"]], inp[~a["inside"]]),
                     "walls, open bays or exterior changed")
            for room in a["rooms"]:
                values = {int(out[p]) for p in room["cells"]}
                _require(values == {room["winner"]}, "reservoir not uniformly filled with its supported winner")
                if room["maximum"]:
                    _require(room["winner"] not in (a["background"], a["wall_color"]), "invalid pigment")
                    _require(room["support"][room["winner"]] == room["maximum"], "unsupported winner")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "new literal color")
            features[split].append(_features(a))
    training = features["train"]
    for name in ("minority", "axial", "empty", "bay_markers", "contests", "orbit_disagreement"):
        _require(any(f[name] for f in training), "missing training contrast: " + name)
    _require(set().union(*(f["parities"] for f in training)) == {0, 1},
             "both integral and half-integral axes need training evidence")
    _require(any(f["distinct_housing"] for f in training),
             "training must distinguish chamber voting from whole-housing voting")
    _require(max(f["rooms"] for f in features["test"]) > max(f["rooms"] for f in training),
             "missing count recombination shift")


def self_test():
    """42 seeds, all legal count combinations, mutation and hand-made regressions.

    Bounded memory: retain scalar counts and hashes, never the full seed batch.
    No files, process workers, retry logs, journals, or execution traces.
    """
    shapes, silhouettes, fingerprints, backgrounds, walls, parities = set(), set(), set(), set(), set(), set()
    counts, totals = set(), Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in p:
            for pair in p[split]:
                inp = pair["input"]
                a = _analyze(inp)
                f = _features(a)
                shapes.add(inp.shape)
                backgrounds.add(a["background"])
                walls.add(a["wall_color"])
                counts.add(f["rooms"])
                parities.update(f["parities"])
                for name in ("minority", "axial", "empty", "bay_markers", "contests", "orbit_disagreement"):
                    totals[name] += f[name]
                totals["shared_housing_pairs"] += f["shared_housing"] > 1
                totals["pairs"] += 1
                digest.update(bytes(inp.shape) + inp.tobytes())
                _require(np.array_equal(solve(inp.copy()), pair["output"]), "solve disagreement")
                for room in a["rooms"]:
                    r0, r1, c0, c1 = room["bounds"]
                    local = _mask((r1 - r0 + 1, c1 - c0 + 1),
                                  {(r - r0, c - c0) for r, c in room["cells"]})
                    silhouettes.add((local.shape, local.tobytes()))
        fingerprints.add(digest.digest())
        if seed < 2:
            q = generate_puzzle(seed)
            _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                         for k in ("input", "output")), "nondeterministic")
    _require(len(fingerprints) == 42 and len(shapes) >= 30 and len(silhouettes) >= 60,
             "insufficient measured structural variation")
    _require(backgrounds == walls == set(range(10)), "palette roles do not vary")
    _require(parities == {0, 1} and counts == {1, 2, 3, 4, 5}, "missing parity/count variation")
    _require(totals["shared_housing_pairs"] >= 10, "multi-room housings not exercised")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "count mismatch")
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None), (4, 1.5)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A rectangular hand fixture gives four votes to color 2, five unmatched
    # cells to color 3, and a supported exterior pair to color 4.
    fixture = np.zeros((12, 15), int)
    fixture[2:9, 3:12] = 5
    fixture[3:8, 4:11] = 0
    for p in ((3, 4), (3, 10), (4, 5), (4, 9)):
        fixture[p] = 2
    for p in ((5, 4), (5, 5), (5, 6), (6, 4), (6, 5)):
        fixture[p] = 3
    fixture[10, 2] = fixture[10, 12] = 4
    expected = fixture.copy()
    expected[3:8, 4:11] = 2
    _require(np.array_equal(solve(fixture), expected), "minority/exterior regression")
    opened = fixture.copy()
    opened[2, 7] = 0
    _require(np.array_equal(solve(opened), opened), "open-bay enclosure regression")
    axial = fixture.copy()
    axial[3:8, 4:11] = 0
    axial[5, 7] = 8
    axial[4, 4] = axial[6, 4] = 9
    expected = axial.copy()
    expected[3:8, 4:11] = 8
    _require(np.array_equal(solve(axial), expected), "self-support regression")
    empty = axial.copy()
    empty[5, 7] = 0
    expected = empty.copy()
    expected[3:8, 4:11] = 0
    _require(np.array_equal(solve(empty), expected), "no-support clearing regression")
    p = generate_puzzle(812)
    for zone in ("inside", "outside", "wall"):
        q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        a = _analyze(q["train"][0]["input"])
        point = tuple(np.argwhere(a[zone])[0])
        out = q["train"][0]["output"]
        out[point] = (int(out[point]) + 1) % 10
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted a mutated " + zone)
    return {"seeds": 42, "pairs": totals["pairs"], "grid_shapes": len(shapes),
            "cavity_shapes": len(silhouettes), "room_counts": sorted(counts),
            "axis_parities": sorted(parities), "features": dict(totals)}
