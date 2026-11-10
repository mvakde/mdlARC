"""Addressed texture wells: radial plate identities, a directed key, and masks.

The solver reads only pixels. Generation uses normalized construction colors and
then permutes all ten colors; none of those construction colors is a convention.
No samples, traces, caches, or other files are written by this module.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
SIDES = ("top", "left", "bottom", "right")
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _components(mask, diagonal=False):
    """Deterministic connected components; diagonal grouping is for ink only."""
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    steps = STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else STEPS
    result = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack, cells = [(int(r), int(c))], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _regions(a, bg):
    h, w = a.shape
    regions = []
    for cells in _components(a == bg):
        edge, boundary = False, set()
        for r, c in cells:
            edge |= r in (0, h - 1) or c in (0, w - 1)
            for dr, dc in STEPS:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w and a[y, x] != bg:
                    boundary.add((y, x))
        regions.append(dict(cells=cells, edge=edge, boundary=boundary))
    return regions


def _analyze(input_grid):
    """Recover hub, radial addresses, local edge key, and cavity ownership."""
    a = _grid(input_grid)
    h, w = a.shape
    values, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    bg = int(values[np.argmax(counts)])
    components = _components(a != bg, diagonal=True)
    hubs = [cells[0] for cells in components if len(cells) == 1]
    _require(len(hubs) == 1, "expected one isolated one-cell hub")
    hr, hc = hubs[0]
    carrier = int(a[hr, hc])
    _require(5 <= hr < h - 5 and 5 <= hc < w - 5, "hub footprint outside canvas")
    plate_cells, plates = set(), {}
    for dr, dc in DIRECTIONS:
        r, c = hr + 4 * dr - 1, hc + 4 * dc - 1
        tile = a[r:r + 3, c:c + 3]
        if np.all(tile == bg):
            continue
        _require(np.all(tile != bg), "incomplete printing plate")
        _require(np.count_nonzero(tile == carrier) >= 5, "plate carrier is unclear")
        address = int(tile[1 - dr, 1 - dc])
        _require(address not in (bg, carrier) and address not in plates, "ambiguous inward address")
        cells = {(r + y, c + x) for y in range(3) for x in range(3)}
        _require(any(set(comp) == cells for comp in components), "plate touches another object")
        plate_cells.update(cells)
        plates[address] = dict(tile=tile.copy(), direction=(dr, dc), cells=cells)
    _require(3 <= len(plates) <= 8, "expected three to eight plates")
    source_cells = plate_cells | {(hr, hc)}
    expected = np.zeros((11, 11), dtype=bool)
    for r, c in source_cells:
        expected[r - hr + 5, c - hc + 5] = True
    _require(np.array_equal(a[hr - 5:hr + 6, hc - 5:hc + 6] != bg, expected),
             "unexpected ink in the hub's footprint")

    candidates = []
    for cells in components:
        if set(cells) & source_cells:
            continue
        r0, r1, c0, c1 = _bounds(cells)
        sh, sw = r1 - r0, c1 - c0
        if len(cells) != sh * sw or len({int(a[q]) for q in cells}) < 2:
            continue
        lanes = []
        if sh == 2 and 3 <= sw <= 6:
            if r0 == 0:
                lanes.append(("top", a[r0, c0:c1], a[r1 - 1, c0:c1]))
            if r1 == h:
                lanes.append(("bottom", a[r1 - 1, c0:c1], a[r0, c0:c1]))
        if sw == 2 and 3 <= sh <= 6:
            if c0 == 0:
                lanes.append(("left", a[r0:r1, c0], a[r0:r1, c1 - 1]))
            if c1 == w:
                lanes.append(("right", a[r0:r1, c1 - 1], a[r0:r1, c0]))
        for side, outer, inner in lanes:
            sources, addresses = tuple(map(int, outer)), tuple(map(int, inner))
            if len(set(sources)) == len(sources) and set(addresses) <= set(plates):
                _require(carrier not in sources, "carrier cannot name a contour")
                candidates.append(dict(side=side, mapping=dict(zip(sources, addresses)),
                                       cells=set(cells), sources=sources, addresses=addresses))
    _require(len(candidates) == 1, "expected one unambiguous local edge key")
    key = candidates[0]
    contours, owner_at = [], {}
    for cells in components:
        if set(cells) & (source_cells | key["cells"]):
            continue
        colors = {int(a[q]) for q in cells}
        _require(len(colors) == 1 and carrier not in colors, "contours must be monochromatic")
        color = next(iter(colors))
        index = len(contours)
        for q in cells:
            owner_at[q] = index
        contours.append(dict(color=color, cells=cells, holes=[]))
    _require(bool(contours), "no contours")
    cavities = []
    for region in _regions(a, bg):
        if region["edge"]:
            continue
        owners = {owner_at.get(q, -1) for q in region["boundary"]}
        _require(len(owners) == 1 and -1 not in owners, "mixed or noncontour cavity boundary")
        owner = next(iter(owners))
        contours[owner]["holes"].append(region["cells"])
        cavities.append(dict(cells=region["cells"], color=contours[owner]["color"]))
    return dict(background=bg, carrier=carrier, hub=(hr, hc), plates=plates,
                source_cells=source_cells, key=key, contours=contours, cavities=cavities)


def _render(a, info, *, halfturn=True, uniform=False):
    out = a.copy()
    for cavity in info["cavities"]:
        address = info["key"]["mapping"].get(cavity["color"])
        if address is None:
            continue
        tile = info["plates"][address]["tile"]
        if halfturn:
            tile = tile[::-1, ::-1]
        r0, _, c0, _ = _bounds(cavity["cells"])
        rr, cc = np.array(cavity["cells"], dtype=int).T
        out[rr, cc] = address if uniform else tile[(rr - r0) % 3, (cc - c0) % 3]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Print the selected half-turned texture into each eligible closed cavity."""
    a = _grid(input_grid)
    return _render(a, _analyze(a))


def _dihedral(tile):
    return [np.rot90(tile, k) for k in range(4)] + [np.rot90(tile.T, k) for k in range(4)]


def _plate(rng, direction, address, decoy, symmetric=False, minimal=False):
    dr, dc = direction
    inward = (1 - dr, 1 - dc)
    if symmetric:
        tile = np.full((3, 3), 9, dtype=np.int8)
        tile[inward] = tile[2 - inward[0], 2 - inward[1]] = address
        options = [p for p in ((0, 0), (0, 1), (0, 2), (1, 0))
                   if p not in (inward, (2 - inward[0], 2 - inward[1]))]
        p = options[int(rng.integers(len(options)))]
        tile[p] = tile[2 - p[0], 2 - p[1]] = decoy
        return tile
    for _ in range(24):
        tile = np.full((3, 3), 9, dtype=np.int8)
        tile[inward] = address
        positions = [p for p in range(9) if p != 3 * inward[0] + inward[1]]
        rng.shuffle(positions)
        count = 1 if minimal else int(rng.integers(1, 4))
        tile.flat[positions[0]] = decoy
        for p in positions[1:count]:
            tile.flat[p] = address if rng.random() < .5 else int(rng.integers(1, 9))
        if len({view.tobytes() for view in _dihedral(tile)}) == 8:
            return tile
    # A unique inward mark and a differently colored off-axis mark always suffice.
    for p in range(9):
        tile = np.full((3, 3), 9, dtype=np.int8)
        tile[inward] = address
        if p == 3 * inward[0] + inward[1]:
            continue
        tile.flat[p] = decoy
        if len({view.tobytes() for view in _dihedral(tile)}) == 8:
            return tile
    raise AssertionError("no asymmetric plate")


def _dilate(mask, diagonal=False):
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    if diagonal:
        out[1:, 1:] |= mask[:-1, :-1]
        out[1:, :-1] |= mask[:-1, 1:]
        out[:-1, 1:] |= mask[1:, :-1]
        out[:-1, :-1] |= mask[1:, 1:]
    return out


def _shape(rng, kind, small=False):
    if kind == "diamond":
        n = 2 if small else int(rng.integers(1, 4))
        yy, xx = np.indices((2 * n + 1, 2 * n + 1))
        ink = abs(yy - n) + abs(xx - n) == n
    elif kind == "multi":
        h, w = (5, 5) if small else (int(rng.integers(5, 9)), int(rng.integers(5, 10)))
        ink = np.ones((h, w), dtype=bool)
        ink[1:-1, 1:-1] = False
        ink[:, w // 2] = True
        if not small and h >= 7 and rng.random() < .5:
            ink[h // 2, :w // 2 + 1] = True
    else:
        if kind == "wide":
            h, w = (4, 4) if small else ((4, 4), (4, 5), (5, 4))[int(rng.integers(3))]
        elif kind == "profile":
            h, w = (3, 3) if small else tuple(map(int, rng.integers(3, 6, size=2)))
        else:
            h, w = (3, 3) if small else (int(rng.integers(1, 6)), int(rng.integers(1, 7)))
        cavity = np.ones((h, w), dtype=bool)
        if kind == "profile":
            # Row-convex, connected, nonrectangular; the wall can meet diagonally.
            for y in range(h):
                if rng.random() < .6:
                    cavity[y, 0] = False
                if rng.random() < .6:
                    cavity[y, -1] = False
            cavity[0, 0] = False
            cavity[-1, -1] = True
        cavity = np.pad(cavity, 1)
        ink = _dilate(cavity, diagonal=kind != "profile") & ~cavity
        if kind == "open":
            # One missing wall cell gives a four-neighbor escape route.
            ink[0, ink.shape[1] // 2] = False
    ink = np.rot90(ink, int(rng.integers(4)))
    if rng.random() < .5:
        ink = ink[:, ::-1]
    r0, r1, c0, c1 = _bounds(list(zip(*np.nonzero(ink))))
    return ink[r0:r1, c0:c1].copy()


def _put_key(a, side, sources, addresses, start):
    for j, (source, address) in enumerate(zip(sources, addresses)):
        p = start + j
        if side == "top":
            a[0, p], a[1, p] = source, address
        elif side == "bottom":
            a[-1, p], a[-2, p] = source, address
        elif side == "left":
            a[p, 0], a[p, 1] = source, address
        else:
            a[p, -1], a[p, -2] = source, address


def _positions(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _block(blocked, r, c, h, w):
    blocked[max(0, r - 1):min(blocked.shape[0], r + h + 1),
            max(0, c - 1):min(blocked.shape[1], c + w + 1)] = True


def _pick_position(rng, positions, h, w, sh, sw):
    border = positions[(positions[:, 0] == 0) | (positions[:, 1] == 0) |
                       (positions[:, 0] + sh == h) | (positions[:, 1] + sw == w)]
    choices = border if len(border) and rng.random() < .45 else positions
    return tuple(map(int, choices[int(rng.integers(len(choices)))]))


def _pack(rng, source, masks, colors, sources, addresses, side):
    """At most twelve layouts, then one small guaranteed packing."""
    for attempt in range(12):
        low = 18 if attempt < 5 else 26
        h, w = map(int, rng.integers(low, 31, size=2))
        if rng.random() < .16:
            h = w = max(h, w)
        a = np.zeros((h, w), dtype=np.int8)
        axis = w if side in ("top", "bottom") else h
        start = int(rng.integers(axis - len(sources) + 1))
        _put_key(a, side, sources, addresses, start)
        blocked = _dilate(a != 0, diagonal=True)
        places = _positions(blocked, 11, 11)
        if not len(places):
            continue
        r, c = _pick_position(rng, places, h, w, 11, 11)
        a[r:r + 11, c:c + 11] = source
        _block(blocked, r, c, 11, 11)
        for j in sorted(range(len(masks)), key=lambda i: -masks[i].size):
            mask = masks[j]
            sh, sw = mask.shape
            places = _positions(blocked, sh, sw)
            if not len(places):
                break
            r, c = _pick_position(rng, places, h, w, sh, sw)
            a[r:r + sh, c:c + sw][mask] = colors[j]
            _block(blocked, r, c, sh, sw)
        else:
            return a
    return None


def _make_pair(rng, side, lesson, plate_range, target_range):
    primary_direction = DIRECTIONS[int(rng.integers(8))]
    if lesson is not None:
        choices = [d for d in DIRECTIONS if bool(d[0] * d[1]) == (lesson == 1)]
        primary_direction = choices[int(rng.integers(len(choices)))]
    count = int(rng.integers(plate_range[0], plate_range[1] + 1))
    others = [d for d in DIRECTIONS if d != primary_direction]
    rng.shuffle(others)
    directions = [primary_direction] + others[:count - 1]
    ids = list(map(int, rng.choice(np.arange(1, 9), size=count, replace=False)))
    primary_id = ids[0]
    source = np.zeros((11, 11), dtype=np.int8)
    source[5, 5] = 9
    tiles_by_id = {}
    for j, (direction, address) in enumerate(zip(directions, ids)):
        tile = _plate(rng, direction, address, ids[(j + 1) % count],
                      symmetric=j != 0 and rng.random() < .22,
                      minimal=j == 0 and lesson == 0)
        tiles_by_id[address] = tile
        dr, dc = direction
        r, c = 4 + 4 * dr, 4 + 4 * dc
        source[r:r + 3, c:c + 3] = tile
    nkey = int(rng.integers(3, 7))
    next_id = None
    if lesson == 0:
        # A visible chain teaches object-reference lookup exactly once. The
        # primary plate has only two detail colors, so a third address exists
        # whose own plate necessarily has a different texture.
        next_id = int(rng.choice([c for c in ids if c not in tiles_by_id[primary_id]]))
    primary_wall = int(rng.choice([c for c in range(1, 9) if c not in (primary_id, next_id)]))
    prefix = [primary_wall, primary_id] if lesson == 0 else [primary_wall]
    other_sources = [c for c in range(1, 9) if c not in prefix]
    rng.shuffle(other_sources)
    sources = prefix + other_sources[:nkey - len(prefix)]
    # In lessons, the primary wall is absent from the inward lane: reversing the
    # key cannot explain its printed cavity. Other mappings can overlap freely.
    fill_choices = [c for c in ids if lesson is None or c != primary_wall]
    address_prefix = [primary_id, next_id] if lesson == 0 else [primary_id]
    addresses = address_prefix + list(map(int, rng.choice(fill_choices, nkey - len(address_prefix))))
    mapping = dict(zip(sources, addresses))
    inert = [c for c in range(1, 9) if c not in mapping]
    closed_kinds = ("rect", "rect", "profile", "diamond", "multi")
    kind = "wide" if lesson == 0 else "profile" if lesson == 1 else str(rng.choice(closed_kinds))
    specs = [(primary_wall, kind)]
    if lesson == 0:
        specs += [(int(rng.choice(inert)), str(rng.choice(closed_kinds))), (primary_wall, "open")]
    elif lesson == 1:
        specs += [(int(rng.choice(sources)), "diamond"), (int(rng.choice(inert)), "rect")]
    target_count = max(len(specs), int(rng.integers(target_range[0], target_range[1] + 1)))
    while len(specs) < target_count:
        role = int(rng.integers(5))
        color = int(rng.choice(inert if role == 0 else sources))
        specs.append((color, "open" if role == 1 else str(rng.choice(closed_kinds))))
    rng.shuffle(specs)
    order = rng.permutation(nkey)
    sources, addresses = [sources[j] for j in order], [addresses[j] for j in order]
    masks = [_shape(rng, kind) for color, kind in specs]
    colors = [color for color, kind in specs]
    a = _pack(rng, source, masks, colors, sources, addresses, side)
    if a is None:
        # No rejection loop: all required topology lessons fit these six slots.
        a = np.zeros((30, 30), dtype=np.int8)
        _put_key(a, "top", sources, addresses, int(rng.integers(7)))
        a[3:14, 1:12] = source
        slots = ((4, 15), (4, 23), (12, 15), (12, 23), (21, 2), (21, 11))
        for (color, kind), (r, c) in zip(specs, slots):
            mask = _shape(rng, kind, small=True)
            a[r:r + mask.shape[0], c:c + mask.shape[1]][mask] = color
        a = np.rot90(a, SIDES.index(side)).copy()
    palette = rng.permutation(10).astype(np.int8)
    a = palette[a]
    return dict(input=a, output=solve(a))


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"illegal {name}")
    return int(value)


def _signature(a):
    return a.shape, a.astype(np.uint8, copy=False).tobytes()


def _make_distinct(pair, seen):
    if _signature(pair["input"]) not in seen:
        return pair
    info = _analyze(pair["input"])
    source_order = [info["background"], info["carrier"]] + [c for c in range(10)
                    if c not in (info["background"], info["carrier"])]
    # Ninety different visible background/carrier assignments exceed 13 prior
    # inputs even at the largest legal request. No further scenes are retained.
    for bg in range(10):
        for carrier in range(10):
            if bg == carrier:
                continue
            target_order = [bg, carrier] + [c for c in range(10) if c not in (bg, carrier)]
            lut = np.empty(10, dtype=np.int8)
            lut[source_order] = target_order
            a = lut[pair["input"]]
            if _signature(a) not in seen:
                return dict(input=a, output=lut[pair["output"]])
    raise AssertionError("distinct palette assignments exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    sides = [SIDES[j] for j in rng.permutation(4)]
    shift = str(rng.choice(("key_side", "plate_count", "contour_count")))
    train_pool = sides[:2] if shift == "key_side" else sides
    train_sides = [sides[i % 2] if i < 2 else str(rng.choice(train_pool)) for i in range(n)]
    test_sides = [str(rng.choice(sides[2:] if shift == "key_side" else sides)) for _ in range(m)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count, side_list in (("train", n, train_sides), ("test", m, test_sides)):
        for i in range(count):
            lesson = i if split == "train" and i < 2 else None
            plate_range = ((3, 5) if split == "train" else (6, 8)) if shift == "plate_count" else (3, 8)
            target_range = ((3, 4) if split == "train" else (5, 6)) if shift == "contour_count" else (2, 6)
            pair = _make_distinct(_make_pair(rng, side_list[i], lesson, plate_range, target_range), seen)
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a, b, info):
    mapping, plates = info["key"]["mapping"], info["plates"]
    f = dict(rotation=not np.array_equal(b, _render(a, info, halfturn=False)),
             uniform=not np.array_equal(b, _render(a, info, uniform=True)),
             unlisted=False, open=False, irregular=False, diagonal=False,
             repeat=False, direction=False, decoy=False, cardinal=False, corner=False,
             transform=False, chain=False, repeat_rows=False, repeat_columns=False)
    exterior8 = set()
    for region in _components(a == info["background"], True):
        if any(y in (0, a.shape[0] - 1) or x in (0, a.shape[1] - 1) for y, x in region):
            exterior8.update(region)
    for contour in info["contours"]:
        color = contour["color"]
        if color in mapping and not contour["holes"]:
            f["open"] = True
        if color not in mapping and contour["holes"]:
            f["unlisted"] = True
    for cavity in info["cavities"]:
        color = cavity["color"]
        if color not in mapping:
            continue
        address = mapping[color]
        plate = plates[address]
        dr, dc = plate["direction"]
        f["corner" if dr * dc else "cardinal"] = True
        f["direction"] |= color not in info["key"]["addresses"]
        f["decoy"] |= any(np.any(p["tile"] == address) for k, p in plates.items() if k != address)
        r0, r1, c0, c1 = _bounds(cavity["cells"])
        if address in mapping:
            next_tile = plates[mapping[address]]["tile"]
            f["chain"] |= any(b[r, c] != next_tile[2 - (r - r0) % 3, 2 - (c - c0) % 3]
                              for r, c in cavity["cells"])
        cells = set(cavity["cells"])
        f["transform"] |= (all((r0 + y, c0 + x) in cells for y in range(3) for x in range(3))
                            and len({t.tobytes() for t in _dihedral(plate["tile"])}) == 8)
        f["irregular"] |= len(cavity["cells"]) < (r1 - r0) * (c1 - c0)
        f["repeat"] |= r1 - r0 > 3 or c1 - c0 > 3
        f["repeat_rows"] |= r1 - r0 > 3
        f["repeat_columns"] |= c1 - c0 > 3
        # A cavity that reaches exterior under eight-neighbor flood is evidence
        # specifically for four-neighbor enclosure (e.g. a diagonal diamond).
        f["diagonal"] |= bool(exterior8.intersection(cavity["cells"]))
    return f


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split is not a list")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid pair counts")
    seen, lessons = set(), []
    profiles = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "output shape changed")
            sig = _signature(a)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            info = _analyze(a)
            profiles[split].append((info["key"]["side"], len(info["plates"]), len(info["contours"])))
            assigned = np.zeros(a.shape, dtype=bool)
            active = 0
            # Scalar coordinates cross-check the vectorized modulo renderer.
            for cavity in info["cavities"]:
                address = info["key"]["mapping"].get(cavity["color"])
                if address is None:
                    continue
                active += 1
                tile = info["plates"][address]["tile"]
                r0, _, c0, _ = _bounds(cavity["cells"])
                for r, c in cavity["cells"]:
                    _require(b[r, c] == tile[2 - (r - r0) % 3, 2 - (c - c0) % 3],
                             "wrong plate, phase, or half-turn")
                    assigned[r, c] = True
            _require(active > 0, "scene has no active closed well")
            _require(np.array_equal(a[~assigned], b[~assigned]), "unassigned pixels changed")
            _require(np.all(a[assigned] == info["background"]) and np.all(b[assigned] != info["background"]),
                     "printing must replace background only")
            _require(np.array_equal(np.unique(a), np.unique(b)), "palette changed")
            if split == "train":
                lessons.append(_features(a, b, info))
    for feature in ("rotation", "uniform", "unlisted", "open", "irregular", "diagonal",
                    "repeat", "direction", "decoy", "cardinal", "corner", "transform", "chain",
                    "repeat_rows", "repeat_columns"):
        _require(any(f[feature] for f in lessons), "missing training witness: " + feature)
    train, test = profiles["train"], profiles["test"]
    held_out_side = (len({f[0] for f in train}) >= 2 and
                     not ({f[0] for f in train} & {f[0] for f in test}))
    count_shift = any(min(f[j] for f in test) > max(f[j] for f in train) for j in (1, 2))
    _require(held_out_side or count_shift, "missing observable train/test shift")


def self_test():
    """Bounded 42-seed sweep, semantic mutations, symmetries, and API checks."""
    shapes, layouts, structures, plate_counts, target_counts = set(), set(), set(), set(), set()
    backgrounds, key_sides, cavity_shapes, signatures = set(), set(), set(), set()
    repeated_ids = symmetric_plates = multi = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                digest.update(a.tobytes())
                digest.update(b.tobytes())
                _require(np.array_equal(solve(a.copy()), b), "solver mismatch")
                info = _analyze(a)
                shapes.add(a.shape)
                layouts.add((a.shape, info["hub"], info["key"]["side"]))
                plate_counts.add(len(info["plates"]))
                target_counts.add(len(info["contours"]))
                backgrounds.add(info["background"])
                key_sides.add(info["key"]["side"])
                roles = {info["background"]: 0, info["carrier"]: 1}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, hashlib.sha256(normalized).digest()))
                for address, plate in info["plates"].items():
                    tile = plate["tile"]
                    repeated_ids += np.count_nonzero(tile == address) > 1
                    symmetric_plates += np.array_equal(tile, tile[::-1, ::-1])
                for contour in info["contours"]:
                    multi += len(contour["holes"]) > 1
                    for cells in contour["holes"]:
                        r0, _, c0, _ = _bounds(cells)
                        cavity_shapes.add(tuple(sorted((r - r0, c - c0) for r, c in cells)))
        signatures.add(digest.digest())
    _require(len(signatures) == 42, "seed diversity collapsed")
    _require(len(shapes) >= 35 and len(layouts) >= 100 and len(structures) >= 180, "geometry diversity collapsed")
    _require(plate_counts == set(range(3, 9)), "missing plate counts")
    _require(min(target_counts) == 2 and max(target_counts) == 6, "missing target counts")
    _require(backgrounds == set(range(10)) and key_sides == set(SIDES), "visible role diversity missing")
    _require(len(cavity_shapes) >= 35 and repeated_ids > 30 and symmetric_plates > 30 and multi > 15,
             "motif or topology diversity missing")
    sample = generate_puzzle(314159, 2, 1)
    again = generate_puzzle(314159, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in sample for x, y in zip(sample[s], again[s])
                 for k in ("input", "output")), "nondeterministic generation")
    # Rotating a scene transports its radial addresses and its edge key together.
    # Texture phase remains the cavity's own upper-left bounding-box coordinate,
    # so compute the expected phase explicitly rather than rotating its label.
    a = sample["train"][0]["input"]
    for turn in range(4):
        rotated = np.rot90(a, turn)
        info = _analyze(rotated)
        out = solve(rotated)
        for cavity in info["cavities"]:
            address = info["key"]["mapping"].get(cavity["color"])
            if address is not None:
                tile = np.rot90(info["plates"][address]["tile"], 2)
                r0, _, c0, _ = _bounds(cavity["cells"])
                for r, c in cavity["cells"]:
                    _require(out[r, c] == tile[(r - r0) % 3, (c - c0) % 3], "orientation failure")
        perm = np.array([7, 0, 4, 2, 9, 1, 8, 6, 3, 5])
        _require(np.array_equal(solve(perm[rotated]), perm[out]), "color equivariance failed")
    for n, m in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(111 + n + m, n, m)
        _require(len(p["train"]) == n and len(p["test"]) == m, "count ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for mutation in ("ink", "cavity", "exterior"):
        broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in sample[s]] for s in sample}
        pair = broken["test"][0]
        info = _analyze(pair["input"])
        if mutation == "ink":
            q = info["hub"]
        elif mutation == "cavity":
            q = next(c["cells"][0] for c in info["cavities"] if c["color"] in info["key"]["mapping"])
        else:
            q = next(region["cells"][0] for region in _regions(pair["input"], info["background"]) if region["edge"])
        pair["output"][q] = (int(pair["output"][q]) + 1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("output mutation accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), canvas_shapes=len(shapes),
                layouts=len(layouts), normalized_structures=len(structures),
                plate_counts=sorted(plate_counts), contour_counts=sorted(target_counts),
                cavity_shapes=len(cavity_shapes), repeated_address_tiles=int(repeated_ids),
                halfturn_symmetric_tiles=int(symmetric_plates), multi_cavity_contours=int(multi))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
