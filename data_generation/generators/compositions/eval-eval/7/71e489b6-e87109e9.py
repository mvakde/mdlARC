"""Defect relays: recover damaged plates, then trace keyed light through holes.

Only standard-library modules and NumPy are used. No parent code or data is
imported. The constructor's geometric renderer and the input solver use separate
tracing implementations. Experiments and reports are bounded and opt-in.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 566091756, 540163009, 285160596, 730297941, 885783119,
              372599533, 528399606, 769582014, 846181147, 120829962,
              996174380, 978165696, 679993691, 118825715, 736148112,
              454075801, 134119305, 325492955, 107752481, 304619809,
              251579009, 714493670, 432409140, 578279066, 225571844,
              613444122, 913708039, 443426009, 546816911, 61944912,
              826652796, 310024594, 302896328, 877209101, 562532899,
              611915222, 656007227, 357429370, 844437548, 96322655,
              213922750)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_))
                 and isinstance(value, (int, np.integer))
                 and low <= value <= high, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer)
             and np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _dilate(a):
    h, w = a.shape
    p = np.pad(a, 1)
    return np.logical_or.reduce([p[y:y+h, x:x+w]
                                 for y in range(3) for x in range(3)])


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in DIRECTIONS:
                q = y+dy, x+dx
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _header(rng, width, turns, frame, background, height=3):
    """Separate framed panels; bar attachment encodes relative handedness."""
    colors = list(map(int, rng.permutation(sorted(turns))))
    widths = np.full(len(colors), 5, dtype=int)
    _require(width >= int(widths.sum()), "header does not fit")
    widths += rng.multinomial(width-int(widths.sum()),
                              np.full(len(colors), 1/len(colors)))
    band = np.full((height, width), background, dtype=np.int8)
    band[[0, -1], :] = frame
    x = 0
    for color, extent in zip(colors, widths):
        band[:, x] = band[:, x+extent-1] = frame
        column = x+1 if turns[color] == -1 else x+extent-2
        band[1:-1, column] = color
        x += int(extent)
    return band


def _parse(input_grid):
    """Recover all geometry and roles using only the visible input."""
    g = _grid(input_grid)
    frame = int(g[0, 0])
    _require(np.all(g[0] == frame), "missing top frame")
    bottoms = np.flatnonzero(np.all(g[1:] == frame, axis=1))+1
    _require(len(bottoms) > 0, "missing bottom frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < g.shape[0]-1, "invalid header height")
    inside = g[1:bottom]
    _require(np.all(inside == inside[0]), "inconsistent header rows")
    row = inside[0]
    _require(row[0] == row[-1] == frame, "missing panel side walls")
    intervals, x = [], 0
    while x < len(row):
        if row[x] == frame:
            x += 1
            continue
        start = x
        while x < len(row) and row[x] != frame:
            x += 1
        intervals.append((start, x))
    _require(2 <= len(intervals) <= 4
             and all(b-a >= 3 for a, b in intervals), "invalid panels")
    background = int(row[intervals[0][0]+1])
    turns = {}
    for a, b in intervals:
        _require(np.all(row[a+1:b-1] == background), "invalid panel interior")
        left, right = int(row[a]), int(row[b-1])
        _require((left == background) != (right == background), "ambiguous bar")
        color, turn = (right, 1) if left == background else (left, -1)
        _require(color not in (background, frame) and color not in turns,
                 "duplicate or invalid material key")
        turns[color] = turn
    field = g[bottom+1:].copy()
    present = set(map(int, np.unique(field)))
    source_colors = present-{background}-set(turns)
    _require(frame not in present and len(source_colors) == 1,
             "source color must be unique and unkeyed")
    source_color = next(iter(source_colors))
    source_cells = np.argwhere(field == source_color)
    _require(len(source_cells) == 1, "source must be a single pixel")
    source = tuple(map(int, source_cells[0]))
    owner = np.full(field.shape, -1, dtype=np.int8)
    extras = np.zeros(field.shape, dtype=bool)
    rectangles = []
    for color in sorted(turns):
        for cells in _components(field == color):
            if len(cells) <= 2:
                yy, xx = zip(*cells)
                extras[yy, xx] = True
                continue
            points = np.asarray(cells)
            y, x = map(int, points.min(axis=0))
            v, u = map(int, points.max(axis=0)+1)
            _require(v-y >= 3 and u-x >= 3, "plate must span at least 3x3")
            block = field[y:v, x:u]
            _require(np.all((block == color) | (block == background)),
                     "plate bounding boxes overlap other objects")
            _require(all(field[a, b] == color
                         for a, b in ((y, x), (y, u-1), (v-1, x), (v-1, u-1))),
                     "intact plate corners are required")
            _require(len(cells)*4 >= 3*(v-y)*(u-x), "plate has excessive damage")
            _require(np.all(owner[y:v, x:u] == -1), "overlapping plates")
            owner[y:v, x:u] = color
            rectangles.append((y, v, x, u, color))
    _require(len(rectangles) >= 2, "at least two plates required")
    # A one-cell field gap separates every genuine plate. Tiny components do
    # not have enough support to extend a plate or become a new plate.
    plate_union = owner >= 0
    for y, v, x, u, _ in rectangles:
        area = plate_union[max(0, y-1):v+1, max(0, x-1):u+1]
        _require(int(area.sum()) == (v-y)*(u-x), "plates need a field gap")
    _require(not np.any(extras & _dilate(plate_union)), "speck attached to plate")
    holes = plate_union & (field == background)
    _require(holes.any(), "missing relays")
    return dict(field=field, background=background, frame=frame,
                source_color=source_color, source=source, turns=turns,
                owner=owner, holes=holes, extras=extras,
                rectangles=rectangles, crop=bottom+1)


def _walk(info, *, use_turns=True, raw_specks=False, mapping=None):
    """Cell-step input tracer. Paint is never consulted as routing geometry."""
    field, holes = info["field"], info["holes"]
    h, w = field.shape
    paths = np.zeros((h, w), dtype=bool)
    active = np.zeros((h, w), dtype=bool)
    chains, headings = [], []
    for initial in range(4):
        y, x = info["source"]
        direction, seen, chain = initial, set(), []
        while 0 <= y < h and 0 <= x < w:
            state = y, x, direction
            _require(state not in seen, "cyclic relay route")
            seen.add(state)
            paths[y, x] = True
            relay = bool(holes[y, x]) or (raw_specks and info["extras"][y, x])
            if relay:
                color = int(info["owner"][y, x] if holes[y, x] else field[y, x])
                turn = info["turns"][color]
                if mapping is not None:
                    turn = mapping[turn]
                if holes[y, x]:
                    active[y, x] = True
                chain.append((y, x))
                headings.append((turn, direction))
                if use_turns:
                    direction = (direction+turn) % 4
            dy, dx = DIRECTIONS[direction]
            y, x = y+dy, x+dx
        chains.append(tuple(chain))
    return paths, active, tuple(chains), tuple(headings)


def _paint_input(info, paths, active, *, all_holes=False, halo=True,
                 erase=True, orthogonal=False):
    result = info["field"].copy()
    if erase:
        result[info["extras"]] = info["background"]
    illumination = paths.copy()
    if halo:
        centers = info["holes"] if all_holes else active
        if orthogonal:
            p = np.pad(centers, 1)
            h, w = centers.shape
            illumination |= (p[:-2, 1:w+1] | p[2:, 1:w+1]
                             | p[1:h+1, :-2] | p[1:h+1, 2:])
        else:
            illumination |= _dilate(centers)
    result[illumination] = info["source_color"]
    # All missing cells, including unreachable neighbors, retain hollow centers.
    result[info["holes"]] = info["background"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from a single input: no examples, generation state, or seed."""
    info = _parse(input_grid)
    paths, active, _, _ = _walk(info)
    return _paint_input(info, paths, active)


def _geometric_trace(shape, source, holes, owner, turns):
    """Construction tracer: jump to the nearest hole on a geometric ray."""
    h, w = shape
    paths = np.zeros(shape, dtype=bool)
    active = set()
    chains = []
    for initial in range(4):
        p, direction, seen, chain = source, initial, set(), []
        while True:
            state = p, direction
            _require(state not in seen, "cyclic construction route")
            seen.add(state)
            y, x = p
            dy, dx = DIRECTIONS[direction]
            candidates = []
            for q in holes:
                a, b = q
                distance = (a-y)*dy+(b-x)*dx
                if distance > 0 and ((dy and b == x) or (dx and a == y)):
                    candidates.append((distance, q))
            if candidates:
                _, q = min(candidates)
            else:
                q = ((0 if dy < 0 else h-1), x) if dy else (y, (0 if dx < 0 else w-1))
            a, b = q
            paths[min(y, a):max(y, a)+1, min(x, b):max(x, b)+1] = True
            if not candidates:
                break
            active.add(q)
            chain.append(q)
            direction = (direction+turns[int(owner[q])]) % 4
            p = q
        chains.append(tuple(chain))
    return paths, active, tuple(chains)


def _construct_output(owner, holes, extras, source, turns, colors):
    """Render known construction masks, independently of the input parser."""
    background, source_color = colors
    result = np.where(owner >= 0, owner, background).astype(np.int8)
    result[source] = source_color
    paths, active, chains = _geometric_trace(owner.shape, source, holes, owner, turns)
    result[paths] = source_color
    for y, x in active:
        result[max(0, y-1):y+2, max(0, x-1):x+2] = source_color
    for p in holes:
        result[p] = background
    return result, paths, active, chains


def _pack_plates(rng, h, w, materials, count):
    owner = np.full((h, w), -1, dtype=np.int8)
    rectangles = []
    for _ in range(130):
        if len(rectangles) >= count:
            break
        rh, rw = int(rng.integers(4, min(10, h)+1)), int(rng.integers(4, min(11, w)+1))
        y, x = int(rng.integers(h-rh+1)), int(rng.integers(w-rw+1))
        if np.any(owner[max(0, y-1):y+rh+1, max(0, x-1):x+rw+1] >= 0):
            continue
        color = materials[len(rectangles) % len(materials)]
        owner[y:y+rh, x:x+rw] = color
        rectangles.append((y, y+rh, x, x+rw, color))
    allowed = owner >= 0
    for y, v, x, u, _ in rectangles:
        allowed[y, x] = allowed[y, u-1] = allowed[v-1, x] = allowed[v-1, u-1] = False
    return owner, rectangles, allowed


def _ahead(p, q, direction):
    dy, dx = DIRECTIONS[direction]
    distance = (q[0]-p[0])*dy+(q[1]-p[1])*dx
    return distance if distance > 0 and ((dy and q[1] == p[1]) or (dx and q[0] == p[0])) else 0


def _candidate(rng, test, palette, depth, clustered, anchor_side=None):
    background, source_color, frame = palette[:3]
    n = int(rng.integers(2, 5))
    materials = list(map(int, palette[3:3+n]))
    turns = {color: int(rng.choice((-1, 1))) for color in materials}
    first_side = int(rng.choice((-1, 1))) if anchor_side is None else anchor_side
    turns[materials[0]], turns[materials[1]] = first_side, -first_side
    rng.shuffle(materials)
    band_height = int(rng.integers(3, 6))
    h = int(rng.integers(16, 31-band_height))
    w = int(rng.integers(max(16, 5*n), 31))
    owner, rectangles, allowed = _pack_plates(rng, h, w, materials, int(rng.integers(3, 8)))
    _require(len(rectangles) >= 3, "not enough plate space")
    points = list(map(tuple, np.argwhere(allowed).tolist()))
    q = points[int(rng.integers(len(points)))]
    # Place the source on a real unobstructed field site aligned with a hole.
    possible = []
    forbidden_source = _dilate(owner >= 0)
    for direction, (dy, dx) in enumerate(DIRECTIONS):
        for distance in range(3, max(h, w)):
            p = q[0]-dy*distance, q[1]-dx*distance
            if 0 <= p[0] < h and 0 <= p[1] < w and not forbidden_source[p]:
                possible.append((p, direction))
    _require(bool(possible), "no source alignment")
    source, incoming = possible[int(rng.integers(len(possible)))]
    holes = {q}
    # Grow one recursive route. It can revisit a plate or travel to another
    # material; the field's intact parts are deliberately transparent.
    for step in range(depth-1):
        direction = (incoming+turns[int(owner[q])]) % 4
        candidates = [p for p in points if p not in holes and _ahead(q, p, direction) >= 2]
        _require(bool(candidates), "route growth reached edge")
        if step == 0:
            opposite = [p for p in candidates if turns[int(owner[p])] != turns[int(owner[q])]]
            if opposite:
                candidates = opposite
        p = candidates[int(rng.integers(len(candidates)))]
        holes.add(p)
        q, incoming = p, direction
    _, _, _, chains = _construct_output(owner, holes, set(), source, turns,
                                        (background, source_color))
    maximum = max(map(len, chains))
    _require((maximum >= 4 if test else 2 <= maximum <= 3), "route depth mismatch")
    output, paths, active, chains = _construct_output(owner, holes, set(), source, turns,
                                                      (background, source_color))
    active_colors = {int(owner[p]) for p in active}
    _require(len({turns[color] for color in active_colors}) == 2,
             "both turn meanings must be exercised")
    if anchor_side is not None:
        _require(set(palette[3:5]) <= active_colors, "anchor keys must be reached")
    # Off-route defects test reachability. Adding one cannot change an existing
    # path because its center is outside the entire old path union.
    inactive = [p for p in points if not paths[p] and p not in holes]
    _require(bool(inactive), "no inactive defect site")
    adjacent = [p for p in inactive if any(max(abs(p[0]-a), abs(p[1]-b)) == 1 for a, b in active)]
    if clustered:
        _require(bool(adjacent), "no nearby inactive defect")
    first = (adjacent if clustered or (adjacent and rng.random() < .65) else inactive)
    chosen = first[int(rng.integers(len(first)))]
    holes.add(chosen)
    for _ in range(int(rng.integers(0, 4))):
        options = [p for p in inactive if p not in holes]
        if options:
            holes.add(options[int(rng.integers(len(options)))])
    # The same intact bounding boxes must be recoverable after adding defects.
    for y, v, x, u, _ in rectangles:
        count = sum(y <= a < v and x <= b < u for a, b in holes)
        _require(4*count <= (v-y)*(u-x), "too many defects in one plate")
    occupied = owner >= 0
    occupied[source] = True
    extra_sites = ~_dilate(occupied)
    on = list(map(tuple, np.argwhere(extra_sites & paths).tolist()))
    off = list(map(tuple, np.argwhere(extra_sites & ~paths).tolist()))
    _require(bool(on) and bool(off), "need on-route and off-route specks")
    extras = {}
    for choices in (on, off):
        choices = [p for p in choices if all(max(abs(p[0]-a), abs(p[1]-b)) > 1 for a, b in extras)]
        _require(bool(choices), "specks lack separation")
        p = choices[int(rng.integers(len(choices)))]
        extras[p] = int(rng.choice(materials))
    for _ in range(int(rng.integers(0, 4))):
        choices = [p for p in on+off if all(max(abs(p[0]-a), abs(p[1]-b)) > 1 for a, b in extras)]
        if choices:
            p = choices[int(rng.integers(len(choices)))]
            extras[p] = int(rng.choice(materials))
    field = np.where(owner >= 0, owner, background).astype(np.int8)
    for p in holes:
        field[p] = background
    for p, color in extras.items():
        field[p] = color
    field[source] = source_color
    output, _, _, _ = _construct_output(owner, holes, extras, source, turns,
                                       (background, source_color))
    band = _header(rng, w, turns, frame, background, band_height)
    pair = {"input": np.vstack((band, field)), "output": output}
    info = _parse(pair["input"])
    _require(np.array_equal(info["owner"], owner), "damaged plate not recoverable")
    _require(np.array_equal(solve(pair["input"]), output), "construction/solve mismatch")
    return pair


def _make_pair(rng, test, palette, depth, clustered, anchor_side=None):
    for _ in range(240):
        try:
            return _candidate(rng, test, palette, depth, clustered, anchor_side)
        except ValueError:
            continue
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return one deterministic puzzle with a longer-route test shift."""
    nt, nq = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_))
             and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    # Stable roles or locally encoded changing palettes are both coherent.
    vary_palette = bool(rng.integers(2))
    first_side = int(rng.choice((-1, 1)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        test = split == "test"
        for i in range(count):
            for _ in range(12):
                anchor = not test and i < 2
                local_palette = (list(map(int, rng.permutation(10)))
                                 if vary_palette and not anchor else palette)
                depth = int(rng.integers(4, 7)) if test else 2+i % 2
                side = first_side * (1 if i == 0 else -1) if anchor else None
                pair = _make_pair(rng, test, local_palette, depth,
                                  clustered=(i == 1 or test), anchor_side=side)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, paths, active, chains):
    holes = info["holes"]
    active_colors = {int(info["owner"][tuple(p)]) for p in np.argwhere(active)}
    edge = np.zeros_like(holes)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    adjacent_inactive = holes & ~active & _dilate(active)
    return dict(shape=info["field"].shape, plates=len(info["rectangles"]),
                keys=len(info["turns"]), holes=int(holes.sum()),
                active=int(active.sum()), inactive=int((holes & ~active).sum()),
                extras=int(info["extras"].sum()),
                max_turns=max(map(len, chains)),
                active_colors=len(active_colors),
                adjacent_inactive=bool(adjacent_inactive.any()),
                boundary_hole=any(holes[y:v, x:u][[0, -1]].any()
                                  or holes[y:v, x:u][:, [0, -1]].any()
                                  for y, v, x, u, _ in info["rectangles"]),
                edge_halo=bool((active & edge).any()),
                on_route_extra=bool((info["extras"] & paths).any()),
                off_route_extra=bool((info["extras"] & ~paths).any()),
                palette=(info["background"], info["source_color"], info["frame"]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training_features, observed_turns = set(), [], {}
    alternatives = Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _parse(inp)
            paths, active, chains, headings = _walk(info)
            expected = _paint_input(info, paths, active)
            _require(np.array_equal(out, expected), "incorrect solved output")
            _require(out.shape == info["field"].shape, "wrong crop")
            _require(np.all(out[info["holes"]] == info["background"]), "filled hollow relay")
            f = _features(info, paths, active, chains)
            _require(f["inactive"] and f["on_route_extra"] and f["off_route_extra"], "missing contrastive evidence")
            _require({turn for turn, _ in headings} == {-1, 1}, "unused turn meaning")
            if split == "train":
                training_features.append(f)
                for y, x in np.argwhere(active):
                    color = int(info["owner"][y, x])
                    observed_turns.setdefault(color, set()).add(info["turns"][color])
                _require(2 <= f["max_turns"] <= 3, "training route depth")
                straight_paths, straight_active, _, _ = _walk(info, use_turns=False)
                wrong = {
                    "ignore_routing": _paint_input(info, paths, active, all_holes=True),
                    "ignore_halos": _paint_input(info, paths, active, halo=False),
                    "ignore_cleanup": _paint_input(info, paths, active, erase=False),
                    "ignore_turns": _paint_input(info, straight_paths, straight_active),
                    "four_neighbors": _paint_input(info, paths, active, orthogonal=True),
                }
                empty = dict(info, holes=np.zeros_like(info["holes"]))
                no_relays, _, _, _ = _walk(empty)
                wrong["ignore_scaffold"] = _paint_input(info, no_relays, np.zeros_like(active))
                for name, prediction in wrong.items():
                    alternatives[name] += not np.array_equal(prediction, out)
            else:
                _require(f["max_turns"] >= 4 and f["adjacent_inactive"], "test shift/evidence absent")
    _require(any(f["adjacent_inactive"] for f in training_features), "teach protected inactive neighbors")
    _require(any(sides == {-1, 1} for sides in observed_turns.values()),
             "teach that material color alone does not fix its turn")
    _require(all(alternatives[name] > 0 for name in
                 ("ignore_routing", "ignore_halos", "ignore_cleanup", "ignore_turns", "four_neighbors", "ignore_scaffold")),
             "demonstrations admit a tested shortcut")


def _fixture():
    """Small hand-designed regression with a manually specified answer."""
    bg, light, frame, a, b = 0, 8, 5, 2, 3
    field = np.zeros((13, 18), dtype=np.int8)
    field[4:9, 4:9] = a
    field[0:4, 4:11] = b
    # Leave a gap between plates in this fixture.
    field[3, 4:11] = bg
    holes = ((6, 6), (1, 6), (7, 7), (2, 9))
    for p in holes:
        field[p] = bg
    field[6, 0] = light
    field[6, 2], field[10, 13] = a, b
    turns = {a: -1, b: 1}
    band = _header(np.random.default_rng(101), 18, turns, frame, bg)
    expected = field.copy()
    expected[10, 13] = bg
    expected[:, 0] = light
    expected[6, :7] = light
    expected[1:7, 6] = light
    expected[1, 6:] = light
    expected[5:8, 5:8] = light
    expected[0:3, 5:8] = light
    for p in holes:
        expected[p] = bg
    return {"input": np.vstack((band, field)), "output": expected}


def self_test():
    """Bounded 42-seed semantic and variation checks; retain no pair corpus."""
    fixture = _fixture()
    assert np.array_equal(solve(fixture["input"]), fixture["output"])
    q = _parse(fixture["input"])
    # Rotation preserves relative handedness; reflection swaps it. Rebuild only
    # the unchanged key syntax, never rotate the instruction language itself.
    for k in range(4):
        for reflect in (False, True):
            field = np.rot90(q["field"], k)
            answer = np.rot90(fixture["output"], k)
            turns = q["turns"]
            if reflect:
                field, answer = np.fliplr(field), np.fliplr(answer)
                turns = {c: -t for c, t in turns.items()}
            band = _header(np.random.default_rng(102), field.shape[1], turns,
                           q["frame"], q["background"])
            assert np.array_equal(solve(np.vstack((band, field))), answer)
    measurements, layouts, shapes, palettes = [], set(), set(), set()
    fingerprints = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                paths, active, chains, _ = _walk(info)
                points = set(map(tuple, np.argwhere(info["holes"]).tolist()))
                geometric, activated, geometry_chains = _geometric_trace(
                    info["field"].shape, info["source"], points,
                    info["owner"], info["turns"])
                assert np.array_equal(paths, geometric)
                assert activated == set(map(tuple, np.argwhere(active).tolist()))
                assert chains == geometry_chains
                f = _features(info, paths, active, chains)
                measurements.append(f)
                shapes.add(f["shape"])
                palettes.add(f["palette"])
                # Normalize color identity while retaining plate geometry.
                layout = np.where(info["owner"] >= 0, 1, 0).astype(np.int8)
                layouts.add(_signature(layout))
                digest.update(pair["input"].tobytes())
        fingerprints.add(digest.hexdigest())
        if seed == 0:
            replay = generate_puzzle(seed)
            assert all(np.array_equal(a[key], b[key]) for s in puzzle
                       for a, b in zip(puzzle[s], replay[s]) for key in a)
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(layouts) >= 180 and len(palettes) >= 30
    assert len({f["plates"] for f in measurements}) >= 3
    assert len({f["holes"] for f in measurements}) >= 6
    assert len({f["active"] for f in measurements}) >= 3
    assert len({f["extras"] for f in measurements}) >= 3
    assert {2, 3, 4} <= {f["keys"] for f in measurements}
    assert any(f["boundary_hole"] for f in measurements)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42+nt*10+nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(measurements),
            "field_shapes": len(shapes), "plate_layouts": len(layouts),
            "role_triples": len(palettes),
            "plate_counts": sorted({f["plates"] for f in measurements}),
            "hole_counts": sorted({f["holes"] for f in measurements}),
            "active_counts": sorted({f["active"] for f in measurements}),
            "max_route_turns": max(f["max_turns"] for f in measurements),
            "boundary_hole_pairs": sum(f["boundary_hole"] for f in measurements),
            "clipped_halo_pairs": sum(f["edge_halo"] for f in measurements)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
