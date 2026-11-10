"""Stencil couriers: spatially addressed, locally reflected corner repairs.

Only the visible input is used by solve. No parent code or saved examples are
imported. Generation labels use the sampled geometry; validation reparses grids.
"""
from __future__ import annotations

from collections import deque
import numpy as np


DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
PORTS = ((-1, -1), (-1, 0), (-1, 1), (-1, 2),
         (0, -1), (0, 2), (1, -1), (1, 2),
         (2, -1), (2, 0), (2, 1), (2, 2))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _direction(offset):
    return tuple(-1 if x == -1 else 1 if x == 2 else 0 for x in offset)


def _footprint(core, shape):
    r, c = core
    return {(y, x) for y in range(r - 1, r + 3) for x in range(c - 1, c + 3)
            if _inside((y, x), shape)}


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r + 1)} | {(b, x) for x in range(l, r + 1)} |
            {(y, l) for y in range(t, b + 1)} | {(y, r) for y in range(t, b + 1)})


def _corners(rect, kind, index):
    t, l, b, r = rect
    return [(y, x, q, kind, index)
            for q, (y, x) in enumerate(((t, l), (t, r), (b, l), (b, r)))]


def _source_corners(core, index):
    r, c = core
    return _corners((r - 1, c - 1, r + 2, c + 2), "source", index)


def _positions(corner, pattern):
    r, c, q = corner[:3]
    dr, dc = DIRS[q]
    return {(r + dr * u, c + dc * v) for u, v in pattern}


def _connected(points, diagonal=False):
    if not points:
        return False
    seen = {min(points)}
    todo = list(seen)
    steps = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)
             if (y or x) and (diagonal or abs(y) + abs(x) == 1)]
    while todo:
        p = todo.pop()
        for step in steps:
            q = _add(p, step)
            if q in points and q not in seen:
                seen.add(q)
                todo.append(q)
    return len(seen) == len(points)


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    queue.append(q)
        result.append(component)
    return result


def _parse_with_structure(a, bg, structure, candidates):
    shape = a.shape
    sources, occupied, cores, ring_structure = [], set(), set(), set()
    for core, color in sorted(candidates):
        foot = _footprint(core, shape)
        _require(not foot & occupied, "overlapping source footprints")
        occupied |= foot
        r, c = core
        cores |= {(r + y, c + x) for y in (0, 1) for x in (0, 1)}
        ports = []
        for offset in PORTS:
            p = _add(core, offset)
            if not _inside(p, shape):
                continue
            if a[p] == bg:
                ports.append((p, _direction(offset), offset))
            else:
                _require(a[p] == structure, "mixed ring")
                ring_structure.add(p)
        sources.append({"core": core, "color": color, "ports": ports})
    remaining = {tuple(map(int, p)) for p in np.argwhere(a == structure)} - ring_structure
    rectangles = []
    for component in _components(remaining):
        t = min(y for y, _ in component)
        l = min(x for _, x in component)
        b = max(y for y, _ in component)
        r = max(x for _, x in component)
        rect = (t, l, b, r)
        _require(b - t >= 6 and r - l >= 6 and component == _outline(rect),
                 "receiving frames must be closed rectangles of at least 7x7")
        rectangles.append(rect)
    _require(rectangles and 1 <= len(sources) <= 4, "missing sources or receivers")
    corners = [p for i, rect in enumerate(rectangles) for p in _corners(rect, "receiver", i)]
    corners += [p for i, s in enumerate(sources) for p in _source_corners(s["core"], i)
                if _inside(p[:2], shape)]
    marks = {}
    for raw in np.argwhere((a != bg) & (a != structure)):
        p = tuple(map(int, raw))
        if p in cores:
            continue
        distances = [(p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in corners]
        _require(distances.count(min(distances)) == 1, "ambiguous ornament association")
        corner = corners[distances.index(min(distances))]
        dr, dc = DIRS[corner[2]]
        local = ((p[0] - corner[0]) * dr, (p[1] - corner[1]) * dc)
        _require(all(-3 <= x <= -1 for x in local), "ornaments must be in the outer corner quadrant")
        marks.setdefault(corner, []).append((local, int(a[p])))
    marks = {q: (items[0][1], frozenset(p for p, _ in items)) for q, items in marks.items()
             if len({color for _, color in items}) == 1}
    # Detect mixed-color ornaments rather than silently dropping their cells.
    covered = cores | ring_structure | set().union(*(_outline(r) for r in rectangles))
    covered |= set().union(*(_positions(q, p) for q, (_, p) in marks.items())) if marks else set()
    _require(covered == {tuple(map(int, p)) for p in np.argwhere(a != bg)}, "unaccounted foreground")
    for index, source in enumerate(sources):
        refs = [(q, color, p) for q, (color, p) in marks.items()
                if q[3:] == ("source", index)]
        _require(len(refs) == 1, "each source needs exactly one intact reference")
        q, color, pattern = refs[0]
        _require(color == source["color"] and 3 <= len(pattern) <= 8 and
                 _connected(pattern, True), "invalid source stencil")
        source.update(reference=q, pattern=pattern)
    payload_colors = {s["color"] for s in sources}
    for q, (color, pattern) in marks.items():
        if q[3] == "receiver":
            _require(len(pattern) <= 8 and color in payload_colors, "invalid receiving clue")
    return {"background": bg, "structure": structure, "sources": sources,
            "rectangles": rectangles, "corners": corners, "marks": marks}


def _parse(a):
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[bg]) == 1, "background must be uniquely most frequent")
    groups = {}
    h, w = a.shape
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(a[r, c])
            if color == bg or not np.all(a[r:r + 2, c:c + 2] == color):
                continue
            ring = [int(a[_add((r, c), o)]) for o in PORTS if _inside(_add((r, c), o), a.shape)]
            nonbg = [x for x in ring if x != bg]
            if len(nonbg) >= 4 and len(nonbg) < len(ring) and len(set(nonbg)) == 1 and nonbg[0] != color:
                groups.setdefault(nonbg[0], []).append(((r, c), color))
    interpretations = []
    for structure, candidates in groups.items():
        try:
            interpretations.append(_parse_with_structure(a, bg, structure, candidates))
        except ValueError:
            continue
    _require(len(interpretations) == 1, "missing or ambiguous input interpretation")
    return interpretations[0]


def _beam_paths(a, info):
    """Ports emit spatial rays; all launcher footprints and frame outlines stop them."""
    feet = [_footprint(s["core"], a.shape) for s in info["sources"]]
    frame_cells = set().union(*(_outline(r) for r in info["rectangles"]))
    by_position = {q[:2]: q for q in info["corners"] if q[3] == "receiver"}
    beams = []
    for index, source in enumerate(info["sources"]):
        other_sources = set().union(*(foot for i, foot in enumerate(feet) if i != index))
        for port, direction, _ in source["ports"]:
            p, path, hit = port, [], None
            while _inside(p, a.shape):
                if p in other_sources:
                    break
                if p in frame_cells:
                    hit = by_position.get(p)
                    break
                _require(int(a[p]) in (info["background"], source["color"]),
                         "beam encounters incompatible foreground")
                path.append(p)
                p = _add(p, direction)
            beams.append({"source": index, "direction": direction, "path": path, "hit": hit})
    return beams


def _check_beams(beams):
    occupied, diagonals = set(), set()
    for beam in beams:
        path = beam["path"]
        _require(path and not occupied & set(path), "rays may not overlap or meet")
        occupied |= set(path)
        for p, q in zip(path, path[1:]):
            if p[0] != q[0] and p[1] != q[1]:
                midpoint = _add(p, q)
                slope = 1 if (p[0] - q[0]) * (p[1] - q[1]) > 0 else -1
                _require((midpoint, -slope) not in diagonals, "rays may not cross between cells")
                diagonals.add((midpoint, slope))
    addressed = [b["hit"] for b in beams if b["hit"] is not None]
    _require(len(addressed) == len(set(addressed)), "multiple beams address one corner")


def _render(a, info, beams, repair=True, rotation=False, color_only=False, unconditional=False):
    out = a.copy()
    for beam in beams:
        color = info["sources"][beam["source"]]["color"]
        for p in beam["path"]:
            out[p] = color
    if not repair:
        return out
    deliveries = [(b["hit"], b["source"]) for b in beams
                  if b["hit"] in info["marks"] and b["hit"][3] == "receiver"]
    if unconditional:
        deliveries = [(q, next(i for i, s in enumerate(info["sources"]) if s["color"] == color))
                      for q, (color, _) in info["marks"].items() if q[3] == "receiver"]
    source_cells = set().union(*(_footprint(s["core"], a.shape) |
                                 _positions(s["reference"], s["pattern"])
                                for s in info["sources"]))
    stamp_cells = set()
    for corner, index in deliveries:
        source = info["sources"][index]
        color, clue = info["marks"][corner]
        _require(source["color"] == color, "delivery ink must match receiving clue")
        if color_only:
            source = next(s for s in info["sources"] if s["color"] == color)
        pattern = source["pattern"]
        if rotation and ((source["reference"][2] in (1, 2)) != (corner[2] in (1, 2))):
            pattern = frozenset((v, u) for u, v in pattern)
        if not (rotation or color_only or unconditional):
            _require(clue <= pattern, "receiving clue is incompatible with routed stencil")
        positions = _positions(corner, pattern)
        _require(all(_inside(p, a.shape) for p in positions), "delivered stencil must fit")
        _require(not positions & source_cells, "delivered stencil overlaps a source")
        other_clues = set().union(*(_positions(q, p) for q, (_, p) in info["marks"].items()
                                   if q[3] == "receiver" and q != corner))
        _require(not positions & other_clues, "delivered stencil overlaps another receiving clue")
        _require(not positions & stamp_cells, "deliveries overlap")
        stamp_cells |= positions
        for p in positions:
            _require(int(a[p]) in (info["background"], color), "delivery overwrites unrelated foreground")
            _require(int(out[p]) in (info["background"], color), "delivery conflicts with a beam")
            out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer geometry, references, and addresses from this input alone."""
    a = _grid(input_grid)
    info = _parse(a)
    beams = _beam_paths(a, info)
    _check_beams(beams)
    return _render(a, info, beams)


def _sample_pattern(rng, asymmetric=False):
    domain = {(u, v) for u in (-3, -2, -1) for v in (-3, -2, -1)}
    for _ in range(100):
        size = int(rng.integers(3, 9))
        pattern = {(-1, -1)}
        while len(pattern) < size:
            options = [p for p in sorted(domain - pattern)
                       if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in pattern)]
            pattern.add(options[int(rng.integers(len(options)))])
        if not asymmetric or pattern != {(v, u) for u, v in pattern}:
            return frozenset(pattern)
    raise RuntimeError("stencil sampling exhausted")


def _base_grid(shape, rectangles, sources):
    a = np.zeros(shape, dtype=np.int8)
    for rect in rectangles:
        for p in _outline(rect):
            a[p] = 1
    for index, source in enumerate(sources):
        r, c = source["core"]
        a[r:r + 2, c:c + 2] = source["color"]
        for o in PORTS:
            p = _add(source["core"], o)
            if _inside(p, shape) and o not in source["offsets"]:
                a[p] = 1
        for p in _positions(source["reference"], source["pattern"]):
            _require(_inside(p, shape), "reference stencil extends outside canvas")
            a[p] = source["color"]
    return a


def _known_info(a, rectangles, sources, marks=None):
    src = []
    for s in sources:
        item = dict(s)
        item["ports"] = [(_add(s["core"], o), _direction(o), o) for o in s["offsets"]
                         if _inside(_add(s["core"], o), a.shape)]
        src.append(item)
    return {"background": 0, "structure": 1, "sources": src, "rectangles": rectangles,
            "corners": [q for i, r in enumerate(rectangles) for q in _corners(r, "receiver", i)] +
                       [q for i, s in enumerate(src) for q in _source_corners(s["core"], i)
                        if _inside(q[:2], a.shape)], "marks": marks or {}}


def _rectangle_scene(rng, count, shape):
    rectangles = []
    h, w = shape
    for _ in range(count):
        for attempt in range(100):
            sh, sw = map(int, rng.integers(7, 13 if count == 1 else 10, 2))
            t = int(rng.integers(3, h - sh - 2))
            l = int(rng.integers(3, w - sw - 2))
            rect = (t, l, t + sh - 1, l + sw - 1)
            if all(rect[2] + 5 < r[0] or r[2] + 5 < rect[0] or
                   rect[3] + 5 < r[1] or r[3] + 5 < rect[1] for r in rectangles):
                rectangles.append(rect)
                break
        else:
            raise ValueError("rectangle packing exhausted")
    return rectangles


def _place_source(rng, shape, rectangles, sources, target, pattern, color, witness=False):
    index = len(sources)
    options = [(o, distance) for o in PORTS for distance in range(4, 13)]
    rng.shuffle(options)
    blocked = {(y, x) for t, l, b, r in rectangles
               for y in range(t - 1, b + 2) for x in range(l - 1, r + 2)}
    blocked |= set().union(*(_footprint(s["core"], shape) for s in sources)) if sources else set()
    for offset, distance in options:
        dr, dc = _direction(offset)
        port = (target[0] - distance * dr, target[1] - distance * dc)
        core = (port[0] - offset[0], port[1] - offset[1])
        if not (0 <= core[0] < shape[0] - 1 and 0 <= core[1] < shape[1] - 1):
            continue
        if _footprint(core, shape) & blocked:
            continue
        refs = _source_corners(core, index)
        rng.shuffle(refs)
        for ref in refs:
            if witness and ((ref[2] in (1, 2)) == (target[2] in (1, 2))):
                continue
            positions = _positions(ref, pattern)
            if not all(_inside(p, shape) for p in positions) or positions & blocked:
                continue
            source = {"core": core, "color": color, "offsets": (offset,),
                      "reference": ref, "pattern": pattern, "target": target}
            candidate = sources + [source]
            a = _base_grid(shape, rectangles, candidate)
            try:
                info = _known_info(a, rectangles, candidate)
                beams = _beam_paths(a, info)
                _check_beams(beams)
                _require(all(b["hit"] == candidate[b["source"]]["target"] for b in beams), "blocked address")
                # Every reference must be segmented correctly from the actual input.
                _parse(a)
                return source
            except ValueError:
                continue
    raise ValueError("launcher placement exhausted")


def _finish_scene(rng, shape, rectangles, sources, active_count, witness):
    a = _base_grid(shape, rectangles, sources)
    info = _known_info(a, rectangles, sources)
    marks = {}
    active = {s["target"] for s in sources[:active_count]}
    unmarked_hits = {s["target"] for s in sources[active_count:]}
    for source in sources[:active_count]:
        pattern = sorted(source["pattern"])
        if witness:
            clue = frozenset({(-1, -1)})
        else:
            rng.shuffle(pattern)
            clue = frozenset(pattern[:int(rng.integers(1, len(pattern)))])
        marks[source["target"]] = (source["color"], clue)
    inactive_options = [q for q in info["corners"] if q[3] == "receiver" and
                        q not in active and q not in unmarked_hits]
    _require(inactive_options, "no inactive receiving corner")
    rng.shuffle(inactive_options)
    for q in inactive_options[:int(rng.integers(1, min(3, len(inactive_options)) + 1))]:
        s = sources[int(rng.integers(len(sources)))]
        pattern = sorted(s["pattern"])
        rng.shuffle(pattern)
        clue = frozenset(pattern[:int(rng.integers(1, min(3, len(pattern))))])
        marks[q] = (s["color"], clue)
    for q, (color, clue) in marks.items():
        for p in _positions(q, clue):
            _require(_inside(p, shape) and a[p] == 0, "receiving clue conflicts")
            a[p] = color
    # Additional ports broadcast the same shape, or simply emit to a boundary/side.
    for _ in range(int(rng.integers(1, 6))):
        index = int(rng.integers(len(sources)))
        s = sources[index]
        choices = [o for o in PORTS if o not in s["offsets"] and _inside(_add(s["core"], o), shape)]
        if not choices or len(s["offsets"]) >= 3:
            continue
        o = choices[int(rng.integers(len(choices)))]
        p = _add(s["core"], o)
        if a[p] != 1:
            continue
        a[p] = 0
        previous = s["offsets"]
        s["offsets"] += (o,)
        try:
            trial = _known_info(a, rectangles, sources, marks)
            beams = _beam_paths(a, trial)
            _check_beams(beams)
            _render(a, trial, beams)
            # Keep partial, unaddressed copies visibly unchanged by incidental ink.
            reached = {b["hit"] for b in beams}
            ink = set().union(*(set(b["path"]) for b in beams))
            _require(all(not _positions(q, s0["pattern"]) & ink
                         for q in marks if q not in reached for s0 in sources), "beam masks inactive clue")
        except ValueError:
            s["offsets"] = previous
            a[p] = 1
    info = _known_info(a, rectangles, sources, marks)
    beams = _beam_paths(a, info)
    _check_beams(beams)
    out = _render(a, info, beams)
    _require(np.array_equal(solve(a), out), "construction/parser disagreement")
    metrics = _features(a, out)
    _require(metrics["repairs"] >= 2 and metrics["inactive"] >= 1, "insufficient addressing evidence")
    _require(metrics["missing"] >= 2, "repairs hidden by rays")
    parsed = _parse(a)
    _require(_inactive_untouched(a, out, parsed, _beam_paths(a, parsed)),
             "incidental ink at an inactive corner")
    if witness:
        _require(metrics["color_shortcut_diff"], "color-only lookup accidentally works")
        _require(metrics["rotation_diff"], "reflection convention is not exposed")
    return {"input": a, "output": out}


def _make_pair(rng, frame_count, source_count, unmarked_hit=False, witness=False):
    for attempt in range(90):
        shape = tuple(map(int, rng.integers(23 if frame_count == 1 else 27, 31, 2)))
        try:
            rectangles = _rectangle_scene(rng, frame_count, shape)
            all_corners = [q for i, rect in enumerate(rectangles) for q in _corners(rect, "receiver", i)]
            if frame_count == 1:
                q = int(rng.integers(4))
                targets = [all_corners[q], all_corners[q ^ int(rng.choice((1, 2)))]]
                targets += [x for x in all_corners if x not in targets][:source_count - 2]
            else:
                targets = [all_corners[int(rng.integers(4))], all_corners[4 + int(rng.integers(4))]]
                rest = [q for q in all_corners if q not in targets]
                rng.shuffle(rest)
                targets += rest[:source_count - 2]
            patterns = [_sample_pattern(rng, witness and i < 2) for i in range(source_count)]
            _require(patterns[0] != patterns[1], "identical courier shapes")
            colors = [2, 2] + [int(rng.choice((2, 3, 4))) for _ in range(source_count - 2)]
            sources = []
            for i, (target, pattern, color) in enumerate(zip(targets, patterns, colors)):
                sources.append(_place_source(rng, shape, rectangles, sources, target, pattern, color,
                                             witness and i == 0))
            active_count = source_count - 1 if unmarked_hit else source_count
            return _finish_scene(rng, shape, rectangles, sources, active_count, witness)
        except ValueError:
            continue
    # Spacious, explicit fallback; positions still vary by a grid symmetry and palette.
    # It preserves the requested frame count, inactive clue, and teaching mechanisms.
    for _ in range(100):
        shape = (30, 30)
        rectangles = [(6, 12, 13, 19)] if frame_count == 1 else [(6, 12, 13, 19), (19, 12, 26, 19)]
        cs = [_corners(rect, "receiver", i) for i, rect in enumerate(rectangles)]
        targets = [cs[0][0], cs[0][2] if frame_count == 1 else cs[1][0], cs[0][1]]
        cores = [(6, 2), (13 if frame_count == 1 else 19, 2), (6, 27)]
        sources = []
        for i in range(source_count):
            refq = 1 if i < 2 else 0
            pattern = _sample_pattern(rng, witness and i < 2)
            sources.append({"core": cores[i], "color": 2 if i < 2 else 3,
                            "offsets": ((0, 2),) if i < 2 else ((0, -1),),
                            "reference": _source_corners(cores[i], i)[refq],
                            "pattern": pattern, "target": targets[i]})
        if sources[0]["pattern"] == sources[1]["pattern"]:
            continue
        try:
            return _finish_scene(rng, shape, rectangles, sources,
                                 source_count - 1 if unmarked_hit else source_count, witness)
        except ValueError:
            continue
    raise RuntimeError("bounded scene construction exhausted")


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    stable_palette = rng.random() < .3
    palette = rng.permutation(10).astype(np.int8)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            witness = split == "train" and index == 0
            unmarked = split == "train" and index == 1
            # The held-out relation is routing across separate receiving frames.
            # All demonstrations use one frame; every test addresses two frames.
            frame_count = 2 if split == "test" else 1
            source_count = 3 if unmarked else int(rng.choice((2, 3)))
            if witness:
                source_count = 2
            for _ in range(12):
                pair = _make_pair(rng, frame_count, source_count, unmarked, witness)
                rotation = int(rng.integers(4))
                reflection = bool(rng.integers(2))
                roles = palette if stable_palette else rng.permutation(10).astype(np.int8)
                for name in ("input", "output"):
                    a = np.rot90(pair[name], rotation)
                    if reflection:
                        a = np.fliplr(a)
                    pair[name] = np.ascontiguousarray(roles[a])
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    order = rng.permutation(nt)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_diff(a, out, info, beams, **kwargs):
    try:
        return not np.array_equal(_render(a, info, beams, **kwargs), out)
    except ValueError:
        return True


def _inactive_untouched(a, out, info, beams):
    reached = {b["hit"] for b in beams}
    window = {(u, v) for u in (-3, -2, -1) for v in (-3, -2, -1)}
    return all(all(a[p] == out[p] for p in _positions(q, window) if _inside(p, a.shape))
               for q in info["marks"] if q[3] == "receiver" and q not in reached)


def _features(a, out):
    info = _parse(a)
    beams = _beam_paths(a, info)
    reached = {b["hit"] for b in beams}
    active = [q for q in info["marks"] if q[3] == "receiver" and q in reached]
    inactive = [q for q in info["marks"] if q[3] == "receiver" and q not in reached]
    ink = set().union(*(set(b["path"]) for b in beams))
    missing = set()
    for beam in beams:
        if beam["hit"] in active:
            s = info["sources"][beam["source"]]
            missing |= {p for p in _positions(beam["hit"], s["pattern"])
                        if a[p] == info["background"] and p not in ink}
    return {"shape": a.shape, "frames": len(info["rectangles"]), "sources": len(info["sources"]),
            "repairs": len(active), "inactive": len(inactive), "missing": len(missing),
            "unmarked_hits": sum(b["hit"] is not None and b["hit"] not in info["marks"] for b in beams),
            "ports": tuple(len(s["ports"]) for s in info["sources"]),
            "directions": tuple(sorted({b["direction"] for b in beams})),
            "patterns": tuple(tuple(sorted(s["pattern"])) for s in info["sources"]),
            "sizes": tuple(len(s["pattern"]) for s in info["sources"]),
            "frame_sizes": tuple((r[2] - r[0] + 1, r[3] - r[1] + 1) for r in info["rectangles"]),
            "reference_orientations": tuple(s["reference"][2] for s in info["sources"]),
            "target_orientations": tuple(q[2] for q in active),
            "clue_sizes": tuple(len(info["marks"][q][1]) for q in active + inactive),
            "clipped_sources": sum(len(_footprint(s["core"], a.shape)) < 16 for s in info["sources"]),
            "broadcast": any(sum(b["source"] == i and b["hit"] in active for b in beams) > 1
                             for i in range(len(info["sources"]))),
            "roles": (info["background"], info["structure"]),
            "color_shortcut_diff": _alternative_diff(a, out, info, beams, color_only=True),
            "unconditional_diff": _alternative_diff(a, out, info, beams, unconditional=True),
            "rotation_diff": _alternative_diff(a, out, info, beams, rotation=True),
            "beams_only_diff": _alternative_diff(a, out, info, beams, repair=False)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape and _key(a) not in seen, "duplicate input or changed dimensions")
            seen.add(_key(a))
            info = _parse(a)
            beams = _beam_paths(a, info)
            _check_beams(beams)
            _require(np.array_equal(_render(a, info, beams), out), "incorrect output")
            _require(np.any(a != out), "identity pair")
            _require(np.array_equal(a[a != info["background"]], out[a != info["background"]]),
                     "original foreground changed")
            _require(set(np.unique(a)) == set(np.unique(out)), "palette changed")
            _require(_inactive_untouched(a, out, info, beams), "inactive ornament neighborhood changed")
            f = _features(a, out)
            _require(f["repairs"] >= 2 and f["inactive"] >= 1 and f["missing"] >= 2,
                     "missing family evidence")
            _require(f["beams_only_diff"] and f["unconditional_diff"], "parent ablation ineffective")
            if split == "train":
                train_features.append(f)
                _require(f["frames"] == 1, "training frame-count regime changed")
            else:
                _require(f["frames"] == 2, "test must route across two receiving frames")
                touched_frames = {b["hit"][4] for b in beams if b["hit"] in info["marks"]}
                _require(touched_frames == {0, 1}, "both test receivers must participate")
    _require(any(f["color_shortcut_diff"] and f["rotation_diff"] for f in train_features),
             "training must expose shape routing and reflection")
    _require(any(f["unmarked_hits"] for f in train_features), "training must expose unmarked stopping")


def _regressions():
    # One courier broadcasts to two differently oriented corners. Its stencil
    # has no (-1,-1) pixel, unlike the usual sampled growth anchor. One closed
    # receiving frame touches the canvas edge. This exercises solver scope
    # beyond generation quotas without relying on any parent fixture.
    pattern = frozenset({(-1, -2), (-2, -1), (-2, -3), (-3, -3)})
    rectangles = [(12, 22, 18, 28), (0, 12, 6, 18)]
    source = {"core": (12, 12), "color": 2, "offsets": ((-1, 0), (0, 2)),
              "reference": _source_corners((12, 12), 0)[3], "pattern": pattern}
    a = _base_grid((30, 30), rectangles, [source])
    targets = [_corners(rectangles[0], "receiver", 0)[0],
               _corners(rectangles[1], "receiver", 1)[2]]
    inactive = _corners(rectangles[0], "receiver", 0)[2]
    for q in targets + [inactive]:
        for p in _positions(q, {(-1, -2)}):
            a[p] = 2
    expected = a.copy()
    expected[7:12, 12] = 2
    expected[12, 14:22] = 2
    for q in targets:
        for p in _positions(q, pattern):
            expected[p] = 2
    for rotation in range(4):
        for reflection in (False, True):
            inp, out = np.rot90(a, rotation), np.rot90(expected, rotation)
            if reflection:
                inp, out = np.fliplr(inp), np.fliplr(out)
            palette = np.array([7, 0, 5, 1, 2, 3, 4, 6, 8, 9], dtype=np.int8)
            inp, out = palette[inp], palette[out]
            _require(np.array_equal(solve(inp), out), "broadcast/no-anchor/edge regression")
    # Association ties and unrelated foreground are rejected, not resolved by
    # numeric color priority. A remote pixel cannot become a hidden instruction.
    noisy = a.copy()
    noisy[29, 0] = 3
    try:
        solve(noisy)
    except ValueError:
        pass
    else:
        raise AssertionError("unrelated foreground accepted")


def self_test():
    """Bounded 42-seed diversity and corruption audit; returns compact measurements."""
    _regressions()
    regression = generate_puzzle(815736105)
    validate_generated_puzzle(regression)
    features, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(pair["input"]) for split in puzzle.values() for pair in split)
        signatures.add(signature)
        for split in puzzle.values():
            for pair in split:
                a, out = pair["input"], pair["output"]
                before = a.copy()
                _require(np.array_equal(solve(a), out) and np.array_equal(a, before), "solver mutates input")
                features.append(_features(a, out))
    repeated = generate_puzzle(0)
    _require(tuple(_key(p["input"]) for s in repeated.values() for p in s) in signatures, "nondeterminism")
    _require(len(signatures) == 42, "cross-seed collapse")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(50618768 + 17 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in repeated[s]] for s in repeated}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    patterns = {p for f in features for p in f["patterns"]}
    shapes = {f["shape"] for f in features}
    directions = {d for f in features for d in f["directions"]}
    _require(len(patterns) >= 40 and len(shapes) >= 25, "structural variation collapsed")
    _require({1, 2} <= {f["frames"] for f in features} and {2, 3} <= {f["sources"] for f in features},
             "object counts collapsed")
    _require(len(directions) == 8, "direction variation collapsed")
    _require({3, 4, 5, 6, 7, 8} <= {n for f in features for n in f["sizes"]}, "stencil sizes collapsed")
    _require({0, 1, 2, 3} <= {q for f in features for q in f["target_orientations"]}, "corner coverage collapsed")
    _require(any(max(f["ports"]) > 1 for f in features), "no multiport sources")
    return {"seeds": 42, "pairs": len(features), "distinct_puzzles": len(signatures),
            "distinct_stencils": len(patterns), "canvas_shapes": len(shapes),
            "directions": len(directions), "frame_sizes": len({s for f in features for s in f["frame_sizes"]}),
            "broadcast_pairs": sum(f["broadcast"] for f in features),
            "clipped_sources": sum(f["clipped_sources"] for f in features),
            "verification": "generator-owned checks; no independent semantic oracle"}
