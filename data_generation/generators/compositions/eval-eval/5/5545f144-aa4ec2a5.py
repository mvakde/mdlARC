"""Rotating repair stamps: movement changes the topology of broken chambers.

Only NumPy and the Python standard library are required. No parent module,
stored puzzle, generation metadata, or palette learned from examples is used
by solve(). See task_spec.md for the family rather than one seed's layout.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _regions(cells, neighbors=DIRS):
    unseen, result = set(cells), []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        component, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _place(offsets, tip, direction):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rotate(p, direction - 1) for p in offsets)}


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "route references must be distinct and aligned")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _pointed_mask(cells, tip):
    """The differently colored tip identifies the anchor; geometry gives facing."""
    matches = []
    for direction in range(4):
        offsets = {_rotate((r - tip[0], c - tip[1]), 1 - direction) for r, c in cells}
        if any(c > 0 for r, c in offsets):
            continue
        if {p for p in offsets if p[1] == 0} != {(0, 0)}:
            continue
        if max(abs(r) for r, c in offsets) == 0:
            continue
        if {(-r, c) for r, c in offsets} != offsets:
            continue
        matches.append((tuple(sorted(offsets)), direction))
    _require(len(matches) == 1, "stamp needs a unique pointing axis")
    return matches[0]


def _read_input(inp):
    """Recover all four color roles from visible connectivity and geometry."""
    inp = _grid(inp)
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 4 and max(counts) > inp.size // 2, "four roles and majority background required")
    bg = int(colors[np.argmax(counts)])
    colors = [int(c) for c in colors if c != bg]
    by_color = {c: set(map(tuple, np.argwhere(inp == c).tolist())) for c in colors}
    components = {c: _regions(by_color[c]) for c in colors}
    candidates = []
    for ink in colors:
        if len(components[ink]) != 1 or not 3 <= len(by_color[ink]) <= 20:
            continue
        for beacon in colors:
            if beacon == ink or len(by_color[beacon]) < 3:
                continue
            if any(len(c) != 1 for c in _regions(by_color[beacon], N8)):
                continue
            attached = [p for p in sorted(by_color[beacon])
                        if any((p[0] + dr, p[1] + dc) in by_color[ink] for dr, dc in DIRS)]
            if len(attached) != 1:
                continue
            tip = attached[0]
            try:
                mask, facing = _pointed_mask(by_color[ink] | {tip}, tip)
            except AssertionError:
                continue
            wall = next(c for c in colors if c not in (ink, beacon))
            if len(components[wall]) < 2:
                continue
            candidates.append(dict(bg=bg, ink=ink, beacon=beacon, wall=wall,
                                   mask=mask, initial=tip, facing=facing,
                                   waypoints=by_color[beacon] - {tip},
                                   walls=by_color[wall], original_objects=components[wall]))
    _require(len(candidates) == 1, "color roles must be uniquely visible")
    a = candidates[0]
    remaining, points, directions = set(a["waypoints"]), [a["initial"]], []
    while remaining:
        p = points[-1]
        rays = {}
        for q in sorted(remaining):
            if q[0] == p[0] or q[1] == p[1]:
                rays.setdefault(_direction(p, q), []).append(q)
        _require(len(rays) == 1, "route has a junction or disconnected suffix")
        d, options = next(iter(rays.items()))
        q = min(options, key=lambda q: _distance(p, q))
        remaining.remove(q)
        points.append(q)
        directions.append(d)
    _require(directions[0] == a["facing"], "first leg must follow the tip")
    a.update(points=points, directions=directions, shape=inp.shape)
    return a


def _topology(cells, shape, neighbors=DIRS):
    """Exterior includes a virtual border: the canvas edge is not a wall."""
    h, w = shape
    occupied = np.zeros((h + 2, w + 2), bool)
    for r, c in cells:
        _require(0 <= r < h and 0 <= c < w, "material outside canvas")
        occupied[r + 1, c + 1] = True
    exterior = np.zeros_like(occupied)
    exterior[0, 0] = True
    queue = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h + 2 and 0 <= cc < w + 2 and not occupied[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                queue.append((rr, cc))
    holes = set(map(tuple, np.argwhere((~occupied & ~exterior)[1:-1, 1:-1]).tolist()))
    bodies = _regions(cells)
    owner = {p: i for i, body in enumerate(bodies) for p in body}
    cavities, hollow = [], set()
    for region in _regions(holes):
        adjacent = {owner[q] for r, c in region for dr, dc in DIRS
                    if (q := (r + dr, c + dc)) in owner}
        _require(len(adjacent) == 1, "cavity must belong to one connected body")
        i = next(iter(adjacent))
        cavities.append((i, region))
        hollow.add(i)
    return dict(bodies=bodies, holes=holes, cavities=cavities, hollow=hollow)


def _deposits(a):
    result = []
    for q, d in zip(a["points"][1:], a["directions"]):
        cells = _place(a["mask"], q, d)
        _require(all(0 <= r < a["shape"][0] and 0 <= c < a["shape"][1] for r, c in cells),
                 "every complete stamp must fit")
        result.append(cells)
    return result


def _finish(cells, shape, bg, wall, ink, beacon):
    t = _topology(cells, shape)
    out = np.full(shape, bg, np.int8)
    for i, body in enumerate(t["bodies"]):
        color = ink if i in t["hollow"] else wall
        for p in body:
            out[p] = color
    for p in t["holes"]:
        out[p] = beacon
    return out, t


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Move the visible template, then classify the material it has repaired."""
    a = _read_input(input_grid)
    material = set(a["walls"])
    for cells in _deposits(a):
        material.update(cells)
    return _finish(material, a["shape"], a["bg"], a["wall"], a["ink"], a["beacon"])[0]


def _stamp(rng, span):
    extension = int(rng.integers(1, 3))
    tail = int(rng.integers(0, 3))
    cells = {(0, -c) for c in range(extension + tail + 1)}
    cells.update((r, -extension) for r in range(-span, span + 1))
    # Extra shoulders give equal-area templates with different geometry.
    if span == 2 and extension == 2 and rng.random() < .5:
        cells.update({(-1, -1), (1, -1)})
    if tail and rng.random() < .45:
        cells.update((r, -extension - tail) for r in (-span, span))
        for c in range(extension + 1, extension + tail + 1):
            cells.update({(-span, -c), (span, -c)})
    _require(len(cells) <= 21, "stamp exceeds extraction range")
    return tuple(sorted(cells)), extension


def _station(rng, span, extension, kind, minimal=False, tail=0):
    """A broken connected chamber with a gate at the stamp's rear column.

    Every cavity is open in the input. Sealing most of a gate may leave a
    one-cell leak; a two-room chamber may be repaired only partly.
    """
    up = 1 if kind == "open" else 0 if minimal else int(rng.integers(0, 2))
    down = 1 if kind in ("leaky", "partial") else 0 if minimal else int(rng.integers(0, 2))
    r0, r1 = -span - 1 - up, span + 1 + down
    lt = 1 if minimal else int(rng.integers(1, 3))
    # Leave an interior corridor beyond the longest tail: open bays do not
    # accidentally split into extra sealed pockets when shoulders are printed.
    width = 5 if minimal else int(rng.integers(max(5, tail + lt + 3), 9))
    left, right = -extension - width + 1, -extension
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(left, right + 1)}
    top = -span - (1 if kind == "open" else 0 if minimal else int(rng.integers(up + 1)))
    bottom = span + (1 if kind in ("leaky", "partial") else 0 if minimal else int(rng.integers(down + 1)))
    cavity = {(r, c) for r in range(top, bottom + 1) for c in range(left + lt, right)}
    # Off-center stepped pockets, while keeping the entire gate exposed.
    if not minimal and width >= 6 and rng.random() < .55:
        cut = int(rng.integers(1, max(2, width - lt - 2)))
        side = top if rng.random() < .5 else bottom
        cavity -= {(side, c) for c in range(left + lt, left + lt + cut)}
    multiple = kind in ("multi", "partial")
    if multiple:
        cavity = {p for p in cavity if p[0] != 0}
    cells -= cavity
    cells -= {(r, right) for r in range(-span, span + 1)}
    if kind in ("leaky", "partial"):
        cells.discard((span + 1, right))
    if kind == "open":
        cells.discard((-span - 1, right))
    if not minimal:
        # A rectangular lobe attaches away from the gate.
        if rng.random() < .75:
            length, thick = int(rng.integers(1, 4)), int(rng.integers(1, 3))
            if rng.random() < .5:
                cells.update((r0 - k, c) for k in range(1, length + 1)
                             for c in range(left, left + thick))
            else:
                cells.update((r, left - k) for k in range(1, length + 1)
                             for r in range(r1 - thick + 1, r1 + 1))
    _require(len(_regions(cells)) == 1, "station disconnected")
    return cells


def _bbox(cells):
    return min(r for r, c in cells), max(r for r, c in cells), min(c for r, c in cells), max(c for r, c in cells)


def _reserve(occupied, cells):
    r0, r1, c0, c1 = _bbox(cells)
    result = occupied.copy()
    result[max(0, r0 - 1):r1 + 2, max(0, c0 - 1):c1 + 2] = True
    return result


def _compatible_future(points, q):
    if q in points:
        return False
    for p, nxt in zip(points[:-1], points[1:]):
        if q[0] == p[0] or q[1] == p[1]:
            if _direction(p, q) != _direction(p, nxt) or _distance(p, q) <= _distance(p, nxt):
                return False
    return True


def _layout(rng, shape, mask, templates, teaching):
    """Bounded geometric search, not a stored path bank or a spiral lattice."""
    oriented = [[(_place(template, (0, 0), d) | _place(mask, (0, 0), d))
                 for d in range(4)] for template in templates]
    boxes = [[_bbox(cells) for cells in row] for row in oriented]
    budget = [1000]

    def grow(points, ds, occupied):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        i = len(ds)
        if i == len(templates):
            return points, ds
        if not ds:
            allowed = [initial]
        elif teaching == 1 and i == 1:
            allowed = [ds[0]]  # nearest of two collinear stations, then a turn
        else:
            allowed = list(map(int, rng.permutation(4)))
        if (teaching == 0 and i == 1) or (teaching == 1 and i == 2):
            allowed = [d for d in allowed if d != ds[0]]
        for d in allowed:
            dr, dc = DIRS[d]
            for distance in rng.permutation(np.arange(6, max(shape))):
                q = (points[-1][0] + dr * int(distance), points[-1][1] + dc * int(distance))
                if not _compatible_future(points, q):
                    continue
                a, b, c, e = boxes[i][d]
                r0, r1, c0, c1 = q[0] + a, q[0] + b, q[1] + c, q[1] + e
                if r0 < 0 or c0 < 0 or r1 >= shape[0] or c1 >= shape[1]:
                    continue
                if occupied[r0:r1 + 1, c0:c1 + 1].any():
                    continue
                cells = {(q[0] + r, q[1] + c) for r, c in oriented[i][d]}
                result = grow(points + [q], ds + [d], _reserve(occupied, cells))
                if result is not None:
                    return result
        return None

    for _ in range(14):
        initial = int(rng.integers(4))
        tip = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
        cells = _place(mask, tip, initial)
        if any(r < 0 or c < 0 or r >= shape[0] or c >= shape[1] for r, c in cells):
            continue
        budget[0] = 300
        result = grow([tip], [], _reserve(np.zeros(shape, bool), cells))
        if result is not None:
            return result
    return None


def _render_scene(shape, palette, mask, templates, points, ds):
    bg, wall, ink, beacon = palette
    inp = np.full(shape, bg, np.int8)
    material = set()
    for template, tip, d in zip(templates, points[1:], ds):
        cells = _place(template, tip, d)
        material.update(cells)
        for p in cells:
            inp[p] = wall
    for p in _place(mask, points[0], ds[0]):
        inp[p] = ink
    inp[points[0]] = beacon
    for p in points[1:]:
        _require(inp[p] == bg, "beacon overlaps a wall")
        inp[p] = beacon
    for q, d in zip(points[1:], ds):
        material.update(_place(mask, q, d))
    out, _ = _finish(material, shape, bg, wall, ink, beacon)
    return dict(input=inp, output=out)


def _make_pair(rng, palette, span, kinds, teaching=None):
    mask, extension = _stamp(rng, span)
    tail = -min(c for r, c in mask) - extension
    n = len(kinds)
    low = {2: 17, 3: 22, 4: 26, 5: 29}[n]
    for _ in range(10):
        templates = [_station(rng, span, extension, kind, tail=tail) for kind in kinds]
        compatible = True
        for template, kind in zip(templates, kinds):
            filled = _place(template | set(mask), (15, 15), 1)
            t = _topology(filled, (30, 30))
            nh = len(t["cavities"])
            if not (nh >= 2 if kind == "multi" else nh >= 1 if kind in ("closed", "partial") else nh == 0):
                compatible = False
                break
        if not compatible:
            continue
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        result = _layout(rng, shape, mask, templates, teaching)
        if result is None:
            continue
        try:
            pair = _render_scene(shape, palette, mask, templates, *result)
            _inspect_pair(pair)
        except AssertionError:
            continue
        return pair
    # Constructive fallback preserves the template and all teaching classes.
    # Random rotation/reflection/margins still apply outside this function.
    templates = [_station(rng, span, extension, kind, minimal=True, tail=tail) for kind in kinds]
    route = [(4, 4), (4, 13), (4, 25), (16, 25), (16, 10), (27, 10)]
    points = [route[0], route[2], route[3]] if teaching == 0 else route[:n + 1]
    ds = [_direction(p, q) for p, q in zip(points, points[1:])]
    pair = _render_scene((30, 30), palette, mask, templates, points, ds)
    _inspect_pair(pair)
    return pair


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _fingerprint(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    variable_palette = bool(rng.random() < .65)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            if split == "train" and i == 0:
                span, kinds, teaching = 1, ["closed", "open"], 0
            elif split == "train" and i == 1:
                span, kinds, teaching = 2, ["multi", "leaky", "partial"], 1
            else:
                span, teaching = int(rng.integers(1, 3)), None
                n = int(rng.choice([4, 5], p=[.8, .2])) if split == "test" else int(rng.integers(2, 4))
                kinds = [str(rng.choice(["closed", "multi", "partial", "open", "leaky"])) for _ in range(n)]
                if split == "test":
                    kinds[0], kinds[1] = "closed", "open"
                rng.shuffle(kinds)
            if variable_palette:
                palette = tuple(map(int, rng.choice(10, 4, replace=False)))
            for attempt in range(16):
                pair = _make_pair(rng, palette, span, kinds, teaching)
                turns, flip = int(rng.integers(4)), bool(rng.integers(2))
                pair = {k: np.rot90(v, turns).copy() for k, v in pair.items()}
                if flip:
                    pair = {k: np.fliplr(v).copy() for k, v in pair.items()}
                key = _fingerprint(pair["input"])
                if key not in seen:
                    break
            else:
                # At most 13 prior inputs, versus 10*9*8*7 visible palettes.
                # No change of geometric rule is needed for uniqueness.
                original, found = pair, False
                from itertools import permutations
                for new_palette in permutations(range(10), 4):
                    color_map = np.arange(10, dtype=np.int8)
                    for old, new in zip(palette, new_palette):
                        color_map[old] = new
                    pair = {k: color_map[v] for k, v in original.items()}
                    key = _fingerprint(pair["input"])
                    if key not in seen:
                        found = True
                        break
                _require(found, "uniqueness fallback exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "canvas dimensions changed")
    a = _read_input(inp)
    initial_cells = _place(a["mask"], a["initial"], a["facing"])
    objects = a["original_objects"]
    _require(len(objects) == len(a["waypoints"]), "one broken chamber per waypoint required")
    _require(len(_regions(a["walls"], N8)) == len(objects), "diagonal wall contact")
    original_topology = _topology(a["walls"], inp.shape)
    _require(not original_topology["holes"], "all input chambers must initially be open")
    _require(all(max(abs(r - rr), abs(c - cc)) > 1 for r, c in initial_cells for rr, cc in a["walls"]),
             "initial stamp touches a chamber")
    deposits = _deposits(a)
    contacted = []
    for stamp in deposits:
        contact = stamp | {(r + dr, c + dc) for r, c in stamp for dr, dc in DIRS}
        ids = [i for i, body in enumerate(objects) if contact & body]
        _require(len(ids) == 1, "each stamp must attach to exactly its own chamber wall")
        contacted.append(ids[0])
    _require(len(set(contacted)) == len(objects), "chambers must be visited once each")
    material = a["walls"] | set().union(*deposits)
    expected, t = _finish(material, inp.shape, a["bg"], a["wall"], a["ink"], a["beacon"])
    _require(len(t["bodies"]) == len(objects), "deposits must not merge distinct chambers")
    _require(len(_regions(material, N8)) == len(objects), "diagonal repaired-body contact")
    _require(t["holes"] == _topology(material, inp.shape, N8)["holes"], "diagonal-only cavity leak")
    _require(len(_regions(t["holes"])) == len(_regions(t["holes"], N8)), "diagonal cavity contact")
    _require(np.array_equal(out, expected), "incorrect deposition, whole-body recoloring, or cavity filling")
    _require(np.array_equal(solve(inp.copy()), out), "input-only solve disagrees")
    _require(not np.array_equal(inp, out), "identity example")
    rotation_changes_topology = False
    for tip, direction, stamp, body_id in zip(a["points"][1:], a["directions"], deposits, contacted):
        if direction == a["facing"]:
            continue
        # Translate this chamber into a padded local scene so this comparison
        # tests topology rather than an accidental wrong-orientation edge clip.
        body = {(15 + r - tip[0], 15 + c - tip[1]) for r, c in objects[body_id]}
        correct = body | _place(a["mask"], (15, 15), direction)
        wrong = body | _place(a["mask"], (15, 15), a["facing"])
        rotation_changes_topology |= _topology(correct, (30, 30))["holes"] != _topology(wrong, (30, 30))["holes"]
    a.update(deposits=deposits, material=material, topology=t,
             rotation_changes_topology=rotation_changes_topology,
             cavity_counts=[sum(j == i for j, region in t["cavities"]) for i in range(len(objects))])
    return a


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a = _inspect_pair(pair)
            key = _fingerprint(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(a)
    train, test = analyses["train"], analyses["test"]
    _require(len({a["mask"] for a in train}) >= 2, "training must show different visible templates")
    _require(any(a["directions"][0] != a["directions"][-1] for a in train), "training must show rotation")
    _require(any(any(x == y for x, y in zip(a["directions"], a["directions"][1:])) for a in train),
             "training must show nearest collinear waypoint first")
    _require(any(max(a["cavity_counts"]) >= 2 for a in train), "multiple cavities must be demonstrated")
    _require(any(0 in a["cavity_counts"] and max(a["cavity_counts"]) > 0 for a in train),
             "training must contrast sealed and still-open repairs")
    _require(any(a["rotation_changes_topology"] for a in train), "rotation must change a training enclosure")
    _require(all(len(a["waypoints"]) > max(len(t["waypoints"]) for t in train) for a in test),
             "test must compose a longer route than training")


def _normalized_mask(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _features(pair):
    a = _inspect_pair(pair)
    points, ds, t = a["points"], a["directions"], a["topology"]
    turns = {(y - x) % 4 for x, y in zip(ds, ds[1:]) if x != y}
    cavities = [region for i, region in t["cavities"]]
    irregular = any(len(region) != (b - r + 1) * (d - c + 1)
                    for region in cavities for r, b, c, d in [_bbox(region)])
    # A component can have a sealed room and an exterior bay at the same time.
    partial = False
    for i in t["hollow"]:
        body = t["bodies"][i]
        r0, r1, c0, c1 = _bbox(body)
        filled = body | set().union(*(region for j, region in t["cavities"] if j == i))
        for rows in (False, True):
            for line in range(c0 if rows else r0, (c1 if rows else r1) + 1):
                positions = sorted(r if rows else c for r, c in filled if (c if rows else r) == line)
                if any(y - x > 1 for x, y in zip(positions, positions[1:])):
                    partial = True
    return dict(shape=a["shape"], mask=a["mask"], count=len(ds), first=ds[0], last=ds[-1],
                distances=tuple(_distance(p, q) for p, q in zip(points, points[1:])),
                mixed_turns=1 in turns and 3 in turns,
                route=tuple((r - points[0][0], c - points[0][1]) for r, c in points),
                cavities=tuple(sorted(a["cavity_counts"])), irregular=irregular, partial=partial,
                station_masks=tuple(_normalized_mask(body) for body in a["original_objects"]),
                palette=(a["bg"], a["wall"], a["ink"], a["beacon"]))


def _encode(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in pair.items()} for pair in puzzle[s]]
                       for s in ("train", "test")}, separators=(",", ":"))


def self_test() -> dict:
    """Bounded 42-seed sweep plus mechanism ablations and corruption checks."""
    records, puzzles = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(_encode(p))
        records.extend(_features(pair) for split in ("train", "test") for pair in p[split])
    _require(len(puzzles) == 42, "seed diversity")
    _require(len({f["shape"] for f in records}) >= 25, "canvas diversity")
    _require(len({f["mask"] for f in records}) >= 8, "template geometry diversity")
    _require(len({f["route"] for f in records}) >= 100, "route diversity")
    _require({f["count"] for f in records} >= {2, 3, 4, 5}, "route count diversity")
    _require({f["first"] for f in records} == set(range(4)), "initial direction diversity")
    _require({f["last"] for f in records} == set(range(4)), "terminal direction diversity")
    _require(any(f["mixed_turns"] for f in records), "mixed turn senses missing")
    _require(sum(f["partial"] for f in records) >= 42, "partial repairs missing")
    _require(sum(f["irregular"] for f in records) >= 20, "cavity geometry too narrow")
    silhouettes = {m for f in records for m in f["station_masks"]}
    _require(len(silhouettes) >= 100, "station silhouette diversity")
    _require(_encode(generate_puzzle(317, 7, 3)) == _encode(generate_puzzle(317, 7, 3)), "nondeterminism")
    # The host covers every legal count combination; retain bounded boundary
    # checks here, alongside the 42 complete default puzzles above.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(700 + nt * 10 + nq, nt, nq)
        _require(len(p["train"]) == nt and len(p["test"]) == nq, "count not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    a = _inspect_pair(p["train"][0])
    pair = p["train"][0]
    # B ablation: deposit correctly, but never detect or fill cavities.
    no_topology = np.full(a["shape"], a["bg"], np.int8)
    for q in a["material"]:
        no_topology[q] = a["wall"]
    _require(not np.array_equal(no_topology, pair["output"]), "B is redundant")
    # A ablation: move the tip but keep the template's initial orientation.
    material = set(a["walls"])
    for tip in a["points"][1:]:
        material.update(_place(a["mask"], tip, a["facing"]))
    if all(0 <= r < a["shape"][0] and 0 <= c < a["shape"][1] for r, c in material):
        wrong = _finish(material, a["shape"], a["bg"], a["wall"], a["ink"], a["beacon"])[0]
        _require(not np.array_equal(wrong, pair["output"]), "rotation is redundant")
    # A tip-only patch cannot seal the back-column gate.
    material = a["walls"] | set(a["waypoints"])
    wrong = _finish(material, a["shape"], a["bg"], a["wall"], a["ink"], a["beacon"])[0]
    _require(not np.array_equal(wrong, pair["output"]), "exact stamp shape is redundant")
    # Remove one cavity-fill pixel; the validator must reject it.
    q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    cell = min(a["topology"]["holes"])
    q["train"][0]["output"][cell] = a["bg"]
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted cavity accepted")
    return dict(seeds=42, pairs=len(records), puzzles=len(puzzles),
                shapes=len({f["shape"] for f in records}),
                stamp_masks=len({f["mask"] for f in records}),
                routes=len({f["route"] for f in records}), station_masks=len(silhouettes),
                partial_repairs=sum(f["partial"] for f in records),
                irregular_cavities=sum(f["irregular"] for f in records))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
