"""Ray-linked instruction chambers: original-input topology controls execution.

Only standard-library modules and NumPy are used. The solver has no knowledge
of the seed, generation layouts, or any other example in the puzzle.
"""

from functools import lru_cache
from itertools import combinations
import numpy as np


# A small geometric vocabulary, fixed throughout the family. These arrowheads
# are rotated together with the scene; their occurrence colors carry stroke ink.
_EAST = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]], dtype=bool)
_GLYPHS = {
    tuple(np.rot90(_EAST, k).ravel()): direction
    for k, direction in enumerate(((0, 1), (-1, 0), (0, -1), (1, 0)))
}
_STOP = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
_GLYPHS[tuple(_STOP.ravel())] = None
_MASKS = {direction: np.array(bits, dtype=bool).reshape(3, 3)
          for bits, direction in _GLYPHS.items()}
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = _STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _STEPS
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(cells))
    return result


def _holes(wall):
    """Four-connected holes of one wall component, ignoring internal glyph ink."""
    passable = np.pad(~wall, 1, constant_values=True)
    exterior = np.zeros(passable.shape, dtype=bool)
    stack = [(0, 0)]
    exterior[0, 0] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in _STEPS:
            rr, cc = r + dr, c + dc
            if (0 <= rr < passable.shape[0] and 0 <= cc < passable.shape[1]
                    and passable[rr, cc] and not exterior[rr, cc]):
                exterior[rr, cc] = True
                stack.append((rr, cc))
    return _components(~exterior[1:-1, 1:-1] & ~wall)


def _glyph(cells):
    rr, cc = zip(*cells)
    r0, c0 = min(rr), min(cc)
    if max(rr) - r0 != 2 or max(cc) - c0 != 2:
        return None
    bits = np.zeros((3, 3), dtype=bool)
    for r, c in cells:
        bits[r-r0, c-c0] = True
    key = tuple(bits.ravel())
    if key not in _GLYPHS:
        return None
    return (r0+1, c0+1), _GLYPHS[key]


def _key(inp, bg):
    """Locate the bounded edge strip; other objects may use the same edge."""
    h, w = inp.shape
    candidates = []
    for cells in _components(inp != bg, diagonal=True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1-r0+1, c1-c0+1
        if len(cells) != sh*sw or len({int(inp[q]) for q in cells}) < 2:
            continue
        lanes = []
        if sh == 2 and sw >= 2:
            if r0 == 0:
                lanes.append(("top", inp[r0, c0:c1+1], inp[r1, c0:c1+1]))
            if r1 == h-1:
                lanes.append(("bottom", inp[r1, c0:c1+1], inp[r0, c0:c1+1]))
        if sw == 2 and sh >= 2:
            if c0 == 0:
                lanes.append(("left", inp[r0:r1+1, c0], inp[r0:r1+1, c1]))
            if c1 == w-1:
                lanes.append(("right", inp[r0:r1+1, c1], inp[r0:r1+1, c0]))
        for side, outer, inner in lanes:
            sources, fills = list(map(int, outer)), list(map(int, inner))
            if len(set(sources)) == len(sources):
                candidates.append({"side": side, "cells": set(cells),
                                   "mapping": dict(zip(sources, fills))})
    if len(candidates) != 1:
        raise ValueError("edge key must have one unambiguous inward direction")
    return candidates[0]


def _read(inp):
    colors, counts = np.unique(inp, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be the uniquely most frequent color")
    bg = int(colors[np.argmax(counts)])
    key = _key(inp, bg)
    ink = inp != bg
    for q in key["cells"]:
        ink[q] = False
    rooms, wall_cards = [], []
    occupied = np.zeros(inp.shape, dtype=bool)
    starts = []
    for color in colors:
        if color == bg:
            continue
        for cells in _components((inp == color) & ink, diagonal=True):
            rr, cc = zip(*cells)
            r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
            if r1-r0 < 4 or c1-c0 < 4:
                continue  # Instruction glyphs and the singleton seed are smaller.
            wall = np.zeros((r1-r0+1, c1-c0+1), dtype=bool)
            for r, c in cells:
                wall[r-r0, c-c0] = True
            hole_groups = _holes(wall)
            # The card list is for validation/measurements only, not execution.
            card_glyphs = []
            for other in colors:
                if other in (bg, color):
                    continue
                patch = (inp[r0:r1+1, c0:c1+1] == other)
                for group in _components(patch, diagonal=True):
                    decoded = _glyph(group)
                    if decoded is not None:
                        anchor, direction = decoded
                        card_glyphs.append(((anchor[0]+r0, anchor[1]+c0), direction))
            if len(card_glyphs) == 1:
                wall_cards.append({"anchor": card_glyphs[0][0], "wall": int(color),
                                   "closed": bool(hole_groups)})
            for group in hole_groups:
                interior = np.zeros(inp.shape, dtype=bool)
                for r, c in group:
                    interior[r+r0, c+c0] = True
                if np.any(occupied & interior):
                    raise ValueError("nested or overlapping chambers are unsupported")
                contents = []
                for other in colors:
                    if other in (bg, color):
                        continue
                    for part in _components(interior & (inp == other), diagonal=True):
                        contents.append((int(other), part, _glyph(part)))
                glyphs = [item for item in contents if item[2] is not None]
                extras = [item for item in contents if item[2] is None]
                if len(glyphs) != 1 or len(extras) > 1 or any(len(x[1]) != 1 for x in extras):
                    raise ValueError("each closed chamber needs one glyph and at most one seed")
                glyph_color, glyph_cells, (anchor, direction) = glyphs[0]
                room = {"anchor": anchor, "wall": int(color), "glyph_color": glyph_color,
                        "direction": direction, "interior": interior,
                        "wall_cells": cells, "glyph_cells": glyph_cells,
                        "area": int(interior.sum())}
                if extras:
                    starts.append(len(rooms))
                    room["seed"] = extras[0][1][0]
                    if extras[0][0] in key["mapping"]:
                        raise ValueError("seed color must be separate from source roles")
                occupied |= interior
                rooms.append(room)
    if len(starts) != 1 or rooms[starts[0]]["wall"] not in key["mapping"]:
        raise ValueError("one eligible closed chamber must contain the seed")
    if len({r["anchor"] for r in rooms}) != len(rooms):
        raise ValueError("chamber anchors must be distinct")
    return {"bg": bg, "key": key, "rooms": rooms, "start": starts[0],
            "protected": occupied, "cards": wall_cards}


def _distance(a, b, direction):
    dr, dc = direction
    rr, cc = b[0]-a[0], b[1]-a[1]
    t = rr*dr + cc*dc
    return t if t > 0 and (rr, cc) == (t*dr, t*dc) else None


def _trace(info):
    rooms, mapping = info["rooms"], info["key"]["mapping"]
    eligible = [i for i, room in enumerate(rooms) if room["wall"] in mapping]
    route, strokes = [], []
    current = info["start"]
    while True:
        if current in route:
            raise ValueError("execution must terminate without revisiting a chamber")
        route.append(current)
        room = rooms[current]
        if room["direction"] is None:
            break
        targets = []
        for i in eligible:
            distance = _distance(room["anchor"], rooms[i]["anchor"], room["direction"])
            if distance is not None:
                targets.append((distance, i))
        if not targets:
            raise ValueError("a nonterminal command has no eligible ray target")
        distance, target = min(targets)
        strokes.append((current, target, distance))
        current = target
    return route, strokes


def _execute(inp, info):
    out = inp.copy()
    route, strokes = _trace(info)
    for i in route:
        room = info["rooms"][i]
        out[room["interior"] & (inp == info["bg"])] = info["key"]["mapping"][room["wall"]]
    for source, target, distance in strokes:
        room = info["rooms"][source]
        r, c = room["anchor"]
        dr, dc = room["direction"]
        for t in range(1, distance+1):
            q = (r+t*dr, c+t*dc)
            if inp[q] == info["bg"] and not info["protected"][q]:
                out[q] = room["glyph_color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Execute the visible, enclosure-scoped program from its singleton seed."""
    inp = _grid(input_grid)
    return _execute(inp, _read(inp))


def _neighbors(active):
    result = {}
    for a in active:
        result[a] = []
        for direction in _STEPS:
            targets = [(_distance(a, b, direction), b) for b in active]
            targets = [(t, b) for t, b in targets if t is not None]
            if targets:
                t, b = min(targets)
                result[a].append((b, direction))
    return result


def _skipped(path, sites):
    skipped = set()
    for a, b in zip(path, path[1:]):
        direction = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
        distance = _distance(a, b, direction)
        skipped.update(q for q in sites if (t := _distance(a, q, direction)) is not None
                       and t < distance)
    return skipped


def _crossed_sites(path, sites):
    counts = {}
    for a, b in zip(path, path[1:]):
        for q in _skipped((a, b), sites):
            counts[q] = counts.get(q, 0)+1
    return {q for q, count in counts.items() if count > 1}


@lru_cache(maxsize=1)
def _lesson_catalog():
    """Finite geometrical search, used only for sampling the first demonstration."""
    sites = tuple((r, c) for r in range(3) for c in range(3))
    options = []
    for active_tuple in combinations(sites, 6):
        active = set(active_tuple)
        neighbors = _neighbors(active)

        def visit(path, directions):
            if len(path) == 5:
                skipped = _skipped(path, sites)
                if len(set(directions)) == 4 and len(skipped) >= 2:
                    options.append((active_tuple, tuple(path), tuple(sorted(skipped))))
                return
            for b, direction in neighbors[path[-1]]:
                if b not in path:
                    visit(path+[b], directions+[direction])

        for a in active_tuple:
            visit([a], [])
    if not options:
        raise RuntimeError("no teaching layout")
    return tuple(options)


def _sample_route(rng, nr, nc, length):
    sites = tuple((r, c) for r in range(nr) for c in range(nc))
    for _ in range(48):
        count = min(len(sites)-2, length+int(rng.integers(1, 3)))
        active = {sites[int(i)] for i in rng.choice(len(sites), count, replace=False)}
        neighbors = _neighbors(active)
        budget = [12000]

        def visit(path):
            budget[0] -= 1
            if len(path) == length:
                return tuple(path)
            if budget[0] <= 0:
                return None
            choices = neighbors[path[-1]]
            for index in rng.permutation(len(choices)):
                b, direction = choices[int(index)]
                if b not in path:
                    answer = visit(path+[b])
                    if answer is not None:
                        return answer
            return None

        for index in rng.permutation(len(active)):
            a = sorted(active)[int(index)]
            path = visit([a])
            if path is not None:
                return sites, active, path
    # Guaranteed connected serpentine path; two trailing sites are inactive,
    # and at least one active site is not visited. This branch remains bounded.
    snake = [(r, c) for r in range(nr)
             for c in (range(nc) if r % 2 == 0 else range(nc-1, -1, -1))]
    active = set(snake[:length+1])
    return sites, active, tuple(snake[:length])


def _geometry(rng, nr, nc, lesson):
    for _ in range(32):
        ygaps = list(map(int, rng.integers(8 if lesson else 6, 10 if lesson else (9 if nr == 4 else 11), nr-1)))
        xgaps = list(map(int, rng.integers(8 if lesson else 6, 10 if lesson else (9 if nc == 4 else 11), nc-1)))
        ry = min(4, (min(ygaps)-2)//2, (26-sum(ygaps))//2)
        rx = min(4, (min(xgaps)-2)//2, (29-sum(xgaps))//2)
        if min(ry, rx) >= (3 if lesson else 2):
            break
    else:
        ygaps, xgaps = [6]*(nr-1), [6]*(nc-1)
        ry = rx = 2
    min_h, min_w = 4+2*ry+sum(ygaps), 1+2*rx+sum(xgaps)
    extra_h = int(rng.integers(0, min(3, 30-min_h)+1))
    extra_w = int(rng.integers(0, min(4, 30-min_w)+1))
    top = int(rng.integers(extra_h+1))
    left = int(rng.integers(extra_w+1))
    rows = [3+ry+top]
    cols = [rx+left]
    for gap in ygaps:
        rows.append(rows[-1]+gap)
    for gap in xgaps:
        cols.append(cols[-1]+gap)
    return (min_h+extra_h, min_w+extra_w), rows, cols, (ry, rx)


def _wall(radii, kind):
    ry, rx = radii
    h, w = 2*ry+1, 2*rx+1
    rr, cc = np.indices((h, w))
    if kind == "diamond":
        assert ry == rx
        return abs(rr-ry)+abs(cc-rx) == ry
    wall = np.zeros((h, w), dtype=bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    if kind == "clipped":
        wall[0, 0] = wall[0, -1] = wall[-1, 0] = wall[-1, -1] = False
    elif kind == "profile":
        # A concave wall makes a notch in one corner of the cavity, outside
        # the central 3x3 instruction footprint.
        wall[1, -2] = True
        wall[0, -1] = False
    elif kind == "thick":
        wall[1, 1:-1] = True
        wall[1:-1, 1] = True
    elif kind == "open":
        wall[0, rx] = False
    return wall


def _make_pair(rng, split, index, turns):
    lesson = split == "train" and index == 0
    if lesson:
        catalog = _lesson_catalog()
        active_tuple, path, skipped = catalog[int(rng.integers(len(catalog)))]
        active = set(active_tuple)
        sites = tuple((r, c) for r in range(3) for c in range(3))
        nr = nc = 3
        inactive = sorted(set(sites)-active)
        open_site, inert_site = map(tuple, rng.permutation(np.array(skipped))[:2])
    else:
        length = (int(rng.integers(4, 7)) if index == 1 else int(rng.integers(3, 7))) if split == "train" else int(rng.integers(7, 11))
        layouts = [(r, c) for r in range(2, 5) for c in range(2, 5)
                   if r*c >= length+3]
        nr, nc = layouts[int(rng.integers(len(layouts)))]
        sites, active, path = _sample_route(rng, nr, nc, length)
        inactive = sorted(set(sites)-active)
        chosen = rng.choice(len(inactive), 2, replace=False)
        open_site, inert_site = (inactive[int(i)] for i in chosen)
    shape, row_levels, col_levels, radius_max = _geometry(rng, nr, nc, lesson)
    inp = np.zeros(shape, dtype=np.int8)
    k = 3 if lesson else int(rng.integers(2, 5))
    sources = list(range(2, k+2))
    inert_fill, inert_absent = k+2, k+3
    glyph_colors = [k+4, k+5]
    mapping = {sources[0]: sources[1], sources[1]: inert_fill}
    for source in sources[2:]:
        mapping[source] = int(rng.choice(sources+[inert_fill]+glyph_colors))
    if k >= 3:
        mapping[sources[-1]] = sources[-1]
    key_order = list(map(int, rng.permutation(sources)))
    start = int(rng.integers(1 if k == 2 else 0, shape[1]-k+(0 if k == 2 else 1)))
    inp[0, start:start+k] = key_order
    inp[1, start:start+k] = [mapping[c] for c in key_order]
    card_sites = set(active) | {open_site, inert_site}
    # Every potential crossing has a card at its anchor. The original center
    # glyph pixel then protects the crossing, avoiding an untaught paint-order
    # convention. This affects placement only, never the execution rule.
    card_sites |= _crossed_sites(path, sites)
    for site in inactive:
        if site not in card_sites and rng.random() < .55:
            card_sites.add(site)
    path_index = {site: i for i, site in enumerate(path)}
    for site in sorted(card_sites):
        r, c = row_levels[site[0]], col_levels[site[1]]
        ry, rx = (radius_max if lesson and site in path[:3] else
                  tuple(int(rng.integers(2, cap+1)) for cap in radius_max))
        if site == open_site:
            kind, color = "open", int(rng.choice(sources))
        elif site == inert_site:
            kind, color = str(rng.choice(["rectangle", "clipped"])), inert_fill
        elif site not in active:
            kind, color = str(rng.choice(["rectangle", "clipped"])), inert_absent
        else:
            kinds = ["rectangle", "clipped"] + (["profile", "thick"] if min(ry, rx) >= 3 else [])
            if ry == rx and ry >= 3:
                kinds.append("diamond")
            kind = str(rng.choice(kinds))
            color = int(rng.choice(sources))
            if site in path_index:
                j = path_index[site]
                if j < len(sources):
                    color = sources[j]
            if lesson and site == path[0]:
                kind = "clipped"
            if lesson and site == path[1]:
                kind = "profile"
            if lesson and site == path[2]:
                kind = "diamond"
        wall = _wall((ry, rx), kind)
        # Reflections preserve arbitrary rectangular dimensions. Square walls
        # additionally admit quarter-turns independently of the glyph direction.
        wall = np.rot90(wall, int(rng.integers(4)) if ry == rx else 2*int(rng.integers(2)))
        if rng.random() < .5:
            wall = np.fliplr(wall)
        patch = inp[r-ry:r+ry+1, c-rx:c+rx+1]
        patch[wall] = color
        if site in path_index:
            j = path_index[site]
            if j == len(path)-1:
                direction = None
            else:
                nxt = path[j+1]
                direction = (int(np.sign(nxt[0]-site[0])), int(np.sign(nxt[1]-site[1])))
        else:
            direction = (_STEPS+(None,))[int(rng.integers(5))]
        glyph_color = glyph_colors[path_index[site] % 2] if site in path_index else int(rng.choice(glyph_colors))
        inp[r-1:r+2, c-1:c+2][_MASKS[direction]] = glyph_color
        if site == path[0]:
            # Pick a visibly separate-color singleton from a blank interior cell.
            hole = _holes(wall)[0]
            choices = [(r-ry+rr, c-rx+cc) for rr, cc in hole
                       if inp[r-ry+rr, c-rx+cc] == 0]
            q = choices[int(rng.integers(len(choices)))]
            inp[q] = 1
    # Whole-scene rotations expose the boundary-to-inward direction on every side.
    inp = np.rot90(inp, turns).copy()
    palette = rng.permutation(10).astype(np.int8)
    inp = palette[inp]
    return {"input": inp, "output": solve(inp)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base_turn = int(rng.integers(4))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            # At least two inward directions are taught; tests may use any side.
            turns = (base_turn+index) % 4 if split == "train" and index < 2 else int(rng.integers(4))
            for _ in range(24):
                pair = _make_pair(rng, split, index, turns)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _features(inp, info=None):
    info = _read(inp) if info is None else info
    route, strokes = _trace(info)
    mapping = info["key"]["mapping"]
    visited = [info["rooms"][i] for i in route]
    skipped_open = skipped_inert = False
    for source, target, distance in strokes:
        room = info["rooms"][source]
        for card in info["cards"]:
            t = _distance(room["anchor"], card["anchor"], room["direction"])
            if t is not None and t < distance:
                skipped_open |= not card["closed"] and card["wall"] in mapping
                skipped_inert |= card["closed"] and card["wall"] not in mapping
    def irregular(room):
        rr, cc = np.nonzero(room["interior"])
        return room["area"] != (rr.max()-rr.min()+1)*(cc.max()-cc.min()+1)
    return {"shape": inp.shape, "side": info["key"]["side"], "bg": info["bg"],
            "key_length": len(mapping), "rooms": len(info["rooms"]),
            "cards": len(info["cards"]), "visited": len(route),
            "directions": set(r["direction"] for r in visited if r["direction"] is not None),
            "open_skip": skipped_open, "inert_skip": skipped_inert,
            "unvisited": sum(i not in route and r["wall"] in mapping
                             for i, r in enumerate(info["rooms"])),
            "chain": any(mapping[r["wall"]] in mapping
                         and mapping[mapping[r["wall"]]] != mapping[r["wall"]]
                         for r in visited),
            "self_map": any(mapping[r["wall"]] == r["wall"] for r in visited),
            "irregular": sum(irregular(r) for r in visited),
            "diagonal_wall": any(len(_components(_wall_mask(inp.shape, r["wall_cells"]))) > 1
                                 for r in visited),
            "areas": tuple(sorted(r["area"] for r in info["rooms"])),
            "border_wall": any(q[0] in (0, inp.shape[0]-1) or q[1] in (0, inp.shape[1]-1)
                               for r in info["rooms"] for q in r["wall_cells"])}


def _wall_mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    for q in cells:
        result[q] = True
    return result


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, measured = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _read(inp)
            route, strokes = _trace(info)
            assert np.array_equal(out, _execute(inp, info)), "incorrect execution"
            assert np.array_equal(inp[inp != info["bg"]], out[inp != info["bg"]]), "input ink changed"
            assert np.any(inp != out), "example has no transformation"
            painted = set()
            for source, target, distance in strokes:
                room = info["rooms"][source]
                r, c = room["anchor"]
                dr, dc = room["direction"]
                for t in range(1, distance+1):
                    q = (r+t*dr, c+t*dc)
                    if inp[q] == info["bg"] and not info["protected"][q]:
                        assert q not in painted, "exterior strokes overlap"
                        painted.add(q)
            for i, room in enumerate(info["rooms"]):
                blank = room["interior"] & (inp == info["bg"])
                assert np.any(blank), "chamber must expose some fill cells"
                expected = info["key"]["mapping"][room["wall"]] if i in route else info["bg"]
                assert np.all(out[blank] == expected), "wrong chamber eligibility or visitation"
            # Explicit ablations: omitting A fills every mapped hole, including
            # an unvisited one; omitting B's lookup fills a visited hole with its wall.
            all_fill = inp.copy()
            for room in info["rooms"]:
                if room["wall"] in info["key"]["mapping"]:
                    all_fill[room["interior"] & (inp == info["bg"])] = info["key"]["mapping"][room["wall"]]
            assert not np.array_equal(all_fill, out), "path contribution is dispensable"
            wall_fill = out.copy()
            for i in route:
                room = info["rooms"][i]
                wall_fill[room["interior"] & (inp == info["bg"])] = room["wall"]
            assert not np.array_equal(wall_fill, out), "directed lookup is dispensable"
            measured[split].append(_features(inp, info))
    train, test = measured["train"], measured["test"]
    assert min(f["visited"] for f in test) > max(f["visited"] for f in train)
    assert set.union(*(f["directions"] for f in train)) == set(_STEPS)
    assert all(f["unvisited"] >= 1 and f["chain"] for f in train+test)
    assert train[0]["open_skip"] and train[0]["inert_skip"]
    assert train[0]["diagonal_wall"] and train[0]["irregular"]
    assert train[0]["self_map"]
    assert len({f["side"] for f in train}) >= 2


def _puzzle_fingerprint(puzzle):
    return tuple((s, _fingerprint(p["input"]), _fingerprint(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def _regression_scene():
    """Hand-planned scene with unequal spacings and explicitly specified edits."""
    inp = np.zeros((27, 29), dtype=np.int8)
    inp[:2, 10:12] = [[2, 3], [3, 4]]
    anchors = ((6, 4), (6, 24), (19, 24), (19, 12), (12, 12))
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0), None)
    cards = [(q, 2 if j % 2 == 0 else 3, directions[j], 6+j % 2,
              "diamond" if j == 1 else "rectangle") for j, q in enumerate(anchors)]
    cards += [((6, 14), 3, (0, -1), 6, "open"),
              ((12, 24), 4, None, 7, "rectangle"),
              ((24, 4), 2, None, 7, "rectangle")]
    interiors = []
    for (r, c), color, direction, glyph_color, kind in cards:
        radius = 3 if kind == "diamond" else 2
        mask = _wall((radius, radius), kind)
        inp[r-radius:r+radius+1, c-radius:c+radius+1][mask] = color
        inp[r-1:r+2, c-1:c+2][_MASKS[direction]] = glyph_color
        interior = np.zeros(inp.shape, dtype=bool)
        if kind == "diamond":
            for rr in range(r-2, r+3):
                for cc in range(c-2, c+3):
                    interior[rr, cc] = abs(rr-r)+abs(cc-c) < 3
        elif kind != "open":
            interior[r-1:r+2, c-1:c+2] = True
        interiors.append(interior)
    inp[5, 4] = 1
    expected = inp.copy()
    for j in range(5):
        expected[interiors[j] & (inp == 0)] = 3 if j % 2 == 0 else 4
    protected = np.logical_or.reduce(interiors)
    for j, (a, b) in enumerate(zip(anchors, anchors[1:])):
        dr, dc = directions[j]
        for t in range(1, abs(b[0]-a[0])+abs(b[1]-a[1])+1):
            q = (a[0]+dr*t, a[1]+dc*t)
            if inp[q] == 0 and not protected[q]:
                expected[q] = 6+j % 2
    assert expected[6, 23] == 4  # Destination is 4 after exactly one lookup.
    assert expected[24, 4] == 7  # The unvisited terminal glyph stays intact.
    assert expected[23, 4] == 0  # Its cavity remains blank.
    assert expected[11, 24] == 0  # Fill-only inert chamber is skipped.
    assert expected[6, 10] == 6 and expected[15, 24] == 7
    assert np.array_equal(solve(inp), expected)


def self_test():
    """Bounded semantic regressions, mutation rejection, and 42-seed variation."""
    _regression_scene()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_fingerprint(p) for p in puzzles}) == 42
    assert _puzzle_fingerprint(puzzles[0]) == _puzzle_fingerprint(generate_puzzle(0))
    records = [_features(pair["input"]) for p in puzzles
               for split in ("train", "test") for pair in p[split]]
    assert len({f["shape"] for f in records}) >= 30
    assert {f["side"] for f in records} == {"top", "left", "bottom", "right"}
    assert {f["bg"] for f in records} == set(range(10))
    assert {f["key_length"] for f in records} == {2, 3, 4}
    assert len({f["cards"] for f in records}) >= 6
    assert {f["visited"] for f in records} >= set(range(3, 11))
    assert len({f["areas"] for f in records}) >= 30
    assert sum(f["border_wall"] for f in records) >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(100+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Rotation and a color permutation must commute with the transformation.
    pair = puzzles[0]["train"][0]
    palette = np.array([7, 1, 9, 3, 0, 8, 4, 2, 6, 5], dtype=np.int8)
    for turns in range(4):
        a = palette[np.rot90(pair["input"], turns)]
        b = palette[np.rot90(pair["output"], turns)]
        assert np.array_equal(solve(a), b)
    for damage in ("fill", "unvisited", "stroke"):
        broken = {s: [{k: v.copy() for k, v in x.items()} for x in puzzles[0][s]]
                  for s in ("train", "test")}
        pair = broken["train"][0]
        inp, out = pair["input"], pair["output"]
        info = _read(inp)
        route, strokes = _trace(info)
        if damage == "stroke":
            candidates = np.argwhere((inp == info["bg"]) & ~info["protected"] & (out != inp))
            q = tuple(candidates[0])
        else:
            index = route[0] if damage == "fill" else next(i for i, r in enumerate(info["rooms"])
                        if i not in route and r["wall"] in info["key"]["mapping"])
            q = tuple(np.argwhere(info["rooms"][index]["interior"] & (inp == info["bg"]))[0])
        out[q] = (int(out[q])+1) % 10
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("semantic mutation accepted")
    return {"seeds": 42, "pairs": len(records),
            "shapes": len({f["shape"] for f in records}),
            "card_counts": sorted({f["cards"] for f in records}),
            "visited_counts": sorted({f["visited"] for f in records}),
            "cavity_area_profiles": len({f["areas"] for f in records})}
