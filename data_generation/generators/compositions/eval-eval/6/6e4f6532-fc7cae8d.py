"""Reassemble ruler-bearing fragments by registration; extract their picture.

Only standard library and NumPy are used. The decoder searches input-derived
placements and certifies a unique final picture, not a unique symmetry ID.
"""
from collections import Counter, deque
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMAL = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
STEPS = tuple(NORMAL.values())
NEAR = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _point(p, t):
    r, c = p
    if t >= 4:
        c = -c
    for _ in range(t % 4):
        r, c = -c, r
    return r, c


def _transform(cells, t):
    raw = {_point(p, t): v for p, v in cells.items()}
    r0 = min(r for r, _ in raw)
    c0 = min(c for _, c in raw)
    return {(r - r0, c - c0): v for (r, c), v in raw.items()}


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]): v for (r, c), v in cells.items()}


def _normal_points(points):
    r0 = min(r for r, _ in points)
    c0 = min(c for _, c in points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _canonical(points):
    return min(_normal_points({_point(p, t) for p in points}) for t in range(8))


def _components(points, steps=NEAR):
    pending = set(points)
    result = []
    while pending:
        first = min(pending)
        pending.remove(first)
        component, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in pending:
                    pending.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _edge(a, side):
    return {"T": a[0, :], "R": a[:, -1],
            "B": a[-1, :], "L": a[:, 0]}[side]


def _controls(a):
    edge_colors = {}
    for s in SIDES:
        values = np.unique(_edge(a, s)[1:-1])
        if len(values) != 1:
            raise ValueError("edge cores must be uniform")
        edge_colors[s] = int(values[0])
    counts = Counter(edge_colors.values())
    backgrounds = [c for c, n in counts.items() if n == 2]
    if len(backgrounds) != 1 or sorted(counts.values()) != [1, 1, 2]:
        raise ValueError("need two adjacent distinct controls and two blank sides")
    bg = backgrounds[0]
    controls = {s: c for s, c in edge_colors.items() if c != bg}
    ids = [SIDES.index(s) for s in controls]
    if (ids[0] - ids[1]) % 4 not in (1, 3):
        raise ValueError("controls must be adjacent")
    if any(int(a[p]) != bg for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))):
        raise ValueError("corners must be background")
    return bg, controls


def _segment(cells, color):
    """Return axis, line coordinate, interval, and its outward normal."""
    marks = {p for p, v in cells.items() if v == color}
    other = [p for p, v in cells.items() if v != color]
    if len(marks) < 2 or not other:
        return None
    rs = {r for r, _ in marks}
    cs = {c for _, c in marks}
    if len(rs) == 1:
        line = next(iter(rs))
        lo, hi = min(cs), max(cs)
        if hi - lo + 1 != len(marks):
            return None
        if all(r > line for r, _ in other):
            side = "T"
        elif all(r < line for r, _ in other):
            side = "B"
        else:
            return None
        return "H", line, lo, hi, side
    if len(cs) == 1:
        line = next(iter(cs))
        lo, hi = min(rs), max(rs)
        if hi - lo + 1 != len(marks):
            return None
        if all(c > line for _, c in other):
            side = "L"
        elif all(c < line for _, c in other):
            side = "R"
        else:
            return None
        return "V", line, lo, hi, side
    return None


def _parse(a):
    bg, controls = _controls(a)
    H, W = a.shape
    interior = {(int(r), int(c)) for r, c in np.argwhere(a != bg)
                if 0 < r < H - 1 and 0 < c < W - 1}
    sources, targets = [], []
    for comp in _components(interior):
        cells = {p: int(a[p]) for p in comp}
        if len(set(cells.values())) == 1:
            targets.append(cells)
        else:
            sources.append(cells)
    if not 2 <= len(sources) <= 4 or len(sources) != len(targets):
        raise ValueError("need 2..4 fragments and equally many detached stencils")
    reg_colors = {v for target in targets for v in target.values()}
    if len(reg_colors) != 1:
        raise ValueError("detached stencils must share one registration color")
    reg = next(iter(reg_colors))
    if reg == bg or reg in controls.values():
        raise ValueError("registration must differ from background and guides")
    markers = set(controls.values())
    for cells in sources:
        values = set(cells.values())
        guides = values & markers
        if (reg not in values or len(guides) != 1
                or not values - markers - {reg, bg}
                or len(_components(cells, STEPS)) != 1):
            raise ValueError("fragment roles or connectivity are invalid")
        if _segment(cells, next(iter(guides))) is None:
            raise ValueError("each fragment needs one exposed straight ruler segment")
    return dict(background=bg, controls=controls, registration=reg,
                sources=sources, targets=[set(t) for t in targets], shape=a.shape)


def _candidates(source, info):
    reg = info["registration"]
    color = next(iter(set(source.values()) & set(info["controls"].values())))
    seen, choices = set(), []
    H, W = info["shape"]
    for t in range(8):
        cells = _transform(source, t)
        marks = {p for p, v in cells.items() if v == reg}
        for index, target in enumerate(info["targets"]):
            if _normal_points(marks) != _normal_points(target):
                continue
            delta = tuple(b - x for b, x in zip(min(target), min(marks)))
            placed = _shift(cells, delta)
            if any(not (0 < r < H - 1 and 0 < c < W - 1) for r, c in placed):
                continue
            # A piece may not swallow another detached registration stencil.
            foreign = set().union(*(p for j, p in enumerate(info["targets"]) if j != index))
            if placed.keys() & foreign:
                continue
            signature = index, tuple(sorted(placed.items()))
            if signature in seen:
                continue
            seen.add(signature)
            choices.append((index, placed, color, _segment(placed, color), t))
    return choices


def _joined_guides(cells, markers):
    segments = {c: _segment(cells, c) for c in markers}
    # _segment needs all other pixels inward; the other guide includes the
    # diagonal corner endpoint, which is inward of each guide's supporting line.
    if any(s is None for s in segments.values()):
        return None
    hs = [(c, s) for c, s in segments.items() if s[0] == "H"]
    vs = [(c, s) for c, s in segments.items() if s[0] == "V"]
    if len(hs) != 1 or len(vs) != 1:
        return None
    hc, (_, hr, c0, c1, hside) = hs[0]
    vc, (_, vcol, r0, r1, vside) = vs[0]
    if (vcol != (c0 - 1 if vside == "L" else c1 + 1)
            or hr != (r0 - 1 if hside == "T" else r1 + 1)):
        return None
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if min(h, w) < 4:
        return None
    for (r, c), value in cells.items():
        if value not in markers and not (r0 <= r <= r1 and c0 <= c <= c1):
            return None
    return (r0, c0, h, w), {hside: hc, vside: vc}


def _orientation(guides, controls):
    matrix = np.zeros((2, 2), dtype=int)
    for side, color in guides.items():
        target = next(s for s, v in controls.items() if v == color)
        matrix += np.outer(NORMAL[target], NORMAL[side])
    if not np.array_equal(matrix @ matrix.T, np.eye(2, dtype=int)):
        raise ValueError("guide/control correspondence must define a rigid symmetry")
    return matrix


def _orient(array, matrix):
    positions = np.indices(array.shape).reshape(2, -1)
    moved = matrix @ positions
    moved -= moved.min(axis=1, keepdims=True)
    out = np.empty(tuple(moved.max(axis=1) + 1), dtype=np.int8)
    out[moved[0], moved[1]] = array.ravel()
    return out


def _asymmetric(array):
    signatures = set()
    for t in range(8):
        matrix = np.column_stack((_point((1, 0), t), _point((0, 1), t)))
        signatures.add(_key(_orient(array, matrix)))
    return len(signatures) == 8


def _finish(cells, info):
    markers = set(info["controls"].values())
    joined = _joined_guides(cells, markers)
    if joined is None:
        return None
    (r0, c0, h, w), guides = joined
    payload = np.full((h, w), info["background"], dtype=np.int8)
    for (r, c), v in cells.items():
        if v not in markers:
            payload[r - r0, c - c0] = v
    matrix = _orientation(guides, info["controls"])
    return _orient(payload, matrix), dict(cells=cells, bounds=(r0, c0, h, w),
                                        guides=guides, payload=payload, matrix=matrix)


def _reconstruct(info, limit=2):
    choices = [_candidates(source, info) for source in info["sources"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    answers = {}

    def search(depth, used, cells, lines, selected):
        if len(answers) >= limit:
            return
        if depth == len(order):
            finished = _finish(cells, info)
            if finished is not None:
                out, detail = finished
                detail["selected"] = list(selected)
                answers.setdefault(_key(out), (out, detail))
            return
        for target, placed, color, segment, t in choices[order[depth]]:
            if target in used or cells.keys() & placed.keys():
                continue
            axis, line, lo, hi, side = segment
            previous = lines.get(color)
            if previous is not None and previous != (axis, line, side):
                continue
            if any(other != color and (axis == desc[0] or side == desc[2])
                   for other, desc in lines.items()):
                continue
            search(depth + 1, used | {target}, {**cells, **placed},
                   {**lines, color: (axis, line, side)},
                   selected + [(order[depth], target, t)])

    search(0, set(), {}, {}, [])
    return list(answers.values())


def _decode(a):
    info = _parse(a)
    answers = _reconstruct(info)
    if len(answers) != 1:
        raise ValueError("registration and ruler continuity must determine one output")
    out, assembly = answers[0]
    return out, {**info, **assembly}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict only from visible input: no generation state or training lookup."""
    return _decode(_grid(input_grid))[0].copy()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _splits(rng, length, number):
    sizes = [2] * number
    for _ in range(length - sum(sizes)):
        sizes[int(rng.integers(number))] += 1
    offset, result = 0, []
    for size in sizes:
        result.append((offset, offset + size - 1))
        offset += size
    return result


def _path(rng, allowed, start, goals):
    queue = deque([start])
    previous = {start: None}
    end = None
    directions = [STEPS[int(i)] for i in rng.permutation(4)]
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = previous[end]
    return result


def _marks(rng, allowed, blocked, mode=None):
    if mode is None and rng.random() < .6:
        # Registration shapes are not a fixed vocabulary. Grow a bounded
        # connected or diagonally connected cluster directly in the private
        # picture region; its position and pose are independent of body growth.
        eligible = allowed - blocked
        origins = sorted(eligible)
        rng.shuffle(origins)
        wanted = int(rng.integers(1, 6))
        steps = NEAR if rng.random() < .45 else STEPS
        for start in origins[:24]:
            points = {start}
            for _ in range(wanted - 1):
                frontier = sorted({(r + dr, c + dc) for r, c in points
                                   for dr, dc in steps} & eligible - points)
                if not frontier:
                    break
                points.add(_pick(rng, frontier))
            if len(points) == wanted:
                return points
    templates = ({(0, 0)}, {(0, 0), (0, 1)}, {(0, 0), (1, 1)},
                 {(0, 0), (0, 1), (1, 0)},
                 {(0, 0), (0, 1), (0, 2)},
                 {(0, 0), (0, 1), (1, 0), (1, 1)})
    indices = [mode] if mode is not None else list(rng.permutation(len(templates)))
    for index in indices:
        poses = [_normal_points({_point(p, t) for p in templates[index]}) for t in range(8)]
        rng.shuffle(poses)
        origins = sorted(allowed)
        rng.shuffle(origins)
        for pose in poses:
            for r0, c0 in origins:
                points = {(r0 + r, c0 + c) for r, c in pose}
                if points <= allowed and not points & blocked:
                    return points
    return None


def _make_fragments(rng, h, w, nh, nv, palette, witness, blank):
    bg, reg, horizontal, vertical = palette[:4]
    motifs = palette[4:4 + int(rng.integers(1, 4))]
    bars = [(horizontal, {(-1, c) for c in range(lo, hi + 1)})
            for lo, hi in _splits(rng, w, nh)]
    bars += [(vertical, {(r, -1) for r in range(lo, hi + 1)})
             for lo, hi in _splits(rng, h, nv)]
    # Voronoi ownership is only a construction tool, not a solving convention.
    # It leaves connected room for each ruler fragment's independently grown body.
    priority = [int(v) for v in rng.permutation(len(bars))]
    rank = {v: i for i, v in enumerate(priority)}
    allowed = [set() for _ in bars]
    for r in range(h):
        for c in range(w):
            if (blank == "B" and r == h - 1) or (blank == "R" and c == w - 1):
                continue
            distances = [min(abs(r - rr) + abs(c - cc) for rr, cc in bar)
                         for _, bar in bars]
            owner = min(range(len(bars)), key=lambda i: (distances[i], rank[i]))
            allowed[owner].add((r, c))
    fragments, blocked = [], set()
    for index, (color, bar) in enumerate(bars):
        region = allowed[index]
        mode = (1 if index == 0 else 2 if index == 1 else 0) if witness else None
        marks = _marks(rng, region, blocked, mode)
        if marks is None:
            return None
        blocked.update(_halo(marks))
        near_bar = {p for p in region if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in bar)}
        if not near_bar:
            return None
        core = set()
        for p in sorted(marks):
            path = _path(rng, region, p, core | near_bar)
            if path is None:
                return None
            core.update(path)
        budget = max(len(marks) + 3, int(len(region) * float(rng.uniform(.25, .85))))
        for _ in range(len(region)):
            if len(core) >= budget:
                break
            frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in STEPS}
                              & region - core)
            if not frontier:
                break
            core.add(_pick(rng, frontier))
        if len(core - marks) < 2:
            return None
        cells = {p: motifs[0] for p in core}
        for p in sorted(core - marks):
            if rng.random() < .25:
                cells[p] = int(_pick(rng, motifs))
        cells.update({p: reg for p in marks})
        cells.update({p: color for p in bar})
        if len(_components(cells, STEPS)) != 1:
            return None
        fragments.append(cells)
    return fragments


def _write_controls(a, bg, controls):
    a[0, :] = bg
    a[-1, :] = bg
    a[:, 0] = bg
    a[:, -1] = bg
    for side, color in controls.items():
        _edge(a, side)[1:-1] = color


def _construct(rng, count, palette, pose, outer, witness=False, split_mode=None):
    if split_mode is None:
        nh = 1 if count == 2 else int(rng.integers(1, count))
    else:
        nh = split_mode
    nv = count - nh
    # Large full bars coexist with short split bars; aspect and count are not tied.
    h = int(rng.integers(max(6, 3 * nv), 13))
    w = int(rng.integers(max(6, 3 * nh), 13))
    blank = str(_pick(rng, ("B", "R"))) if witness or rng.random() < .7 else None
    fragments = _make_fragments(rng, h, w, nh, nv, palette, witness, blank)
    if fragments is None:
        return None
    all_cells = {p: v for fragment in fragments for p, v in fragment.items()}
    raw = {_point(p, pose): v for p, v in all_cells.items()}
    low_r, low_c = min(r for r, _ in raw), min(c for _, c in raw)
    extent = (max(r for r, _ in raw) - low_r + 1, max(c for _, c in raw) - low_c + 1)
    minimum = 20 if count == 2 else 25
    H, W = int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31))
    origin = (int(rng.integers(2, H - extent[0] - 1)),
              int(rng.integers(2, W - extent[1] - 1)))
    finals = []
    for fragment in fragments:
        finals.append({(r - low_r + origin[0], c - low_c + origin[1]): v
                       for p, v in fragment.items() for r, c in [_point(p, pose)]})
    bg, reg = palette[:2]
    controls = dict(zip(outer, palette[2:4]))
    inp = np.full((H, W), bg, dtype=np.int8)
    _write_controls(inp, bg, controls)
    target_points = {p for f in finals for p, v in f.items() if v == reg}
    for p in target_points:
        inp[p] = reg
    blocked = _halo(target_points)
    sources = []
    for index in [int(v) for v in rng.permutation(count)]:
        found = False
        for t in [int(v) for v in rng.permutation(8)]:
            piece = _transform(finals[index], t)
            ph = max(r for r, _ in piece) + 1
            pw = max(c for _, c in piece) + 1
            origins = [(r, c) for r in range(2, H - ph - 1)
                       for c in range(2, W - pw - 1)]
            rng.shuffle(origins)
            for start in origins:
                source = _shift(piece, start)
                if not source.keys() & blocked:
                    sources.append(source)
                    blocked.update(_halo(source))
                    found = True
                    break
            if found:
                break
        if not found:
            return None
    for source in sources:
        for p, v in source.items():
            inp[p] = v
    try:
        info = _parse(inp)
        expected = _finish({p: v for f in finals for p, v in f.items()}, info)[0]
        result, detail = _decode(inp)
    except (ValueError, TypeError):
        return None
    if not np.array_equal(expected, result):
        return None
    if witness:
        patterns = [_canonical(t) for t in info["targets"]]
        if not any(len(a) == len(b) and a != b for i, a in enumerate(patterns) for b in patterns[i + 1:]):
            return None
        # Every possible correspondence reflection must change the picture.
        if not _asymmetric(result):
            return None
    return {"input": inp, "output": expected}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    used = set()
    stable = bool(rng.random() < .3)
    base_palette = [int(c) for c in rng.permutation(10)]
    count_shift = bool(rng.integers(2))
    first_pose = int(rng.integers(8))
    # First output is a quarter-turn of the reconstructed sheet; swapping only
    # the two outer identities then supplies an exact reflection contrast.
    initial_guides = {_point(NORMAL["T"], first_pose): base_palette[2],
                      _point(NORMAL["L"], first_pose): base_palette[3]}
    rotation = int(_pick(rng, (1, 3)))
    outer = tuple(next(s for s in SIDES if NORMAL[s] == _point(n, rotation))
                  for n in initial_guides)
    first_outer_set = set(outer)
    corners = [tuple((SIDES[i], SIDES[(i + 1) % 4])) for i in range(4)]
    held = _pick(rng, [c for c in corners if set(c) != first_outer_set])
    for index in range(nt + ne):
        split = "train" if index < nt else "test"
        if index == 1:
            pair = {k: v.copy() for k, v in puzzle["train"][0].items()}
            bg, controls = _controls(pair["input"])
            values = list(controls.values())
            _write_controls(pair["input"], bg, dict(zip(controls, values[::-1])))
            pair["output"] = solve(pair["input"])
        else:
            palette = base_palette if stable or index == 0 else [int(c) for c in rng.permutation(10)]
            if index == 0:
                count, pose, outer_pair = 3, first_pose, outer
            else:
                count = 4 if split == "test" and count_shift else int(rng.integers(2, 4 if count_shift else 5))
                pose = int(rng.integers(8))
                pool = [c for c in corners if count_shift or set(c) != set(held)]
                outer_pair = tuple(held) if split == "test" and not count_shift else tuple(_pick(rng, pool))
                if rng.random() < .5:
                    outer_pair = outer_pair[::-1]
            for _ in range(160):
                pair = _construct(rng, count, palette, pose, outer_pair, witness=index == 0)
                if pair is not None and _key(pair["input"]) not in used:
                    break
            else:
                raise RuntimeError("bounded fragment construction exhausted")
        if _key(pair["input"]) in used:
            raise RuntimeError("duplicate input")
        used.add(_key(pair["input"]))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    predicted, info = _decode(inp)
    if not np.array_equal(predicted, out):
        raise ValueError("output violates registration/ruler reconstruction")
    payload, bg = info["payload"], info["background"]
    patterns = [_canonical(t) for t in info["targets"]]
    occupied = np.argwhere(payload != bg)
    low, high = occupied.min(axis=0), occupied.max(axis=0) + 1
    tight = payload[low[0]:high[0], low[1]:high[1]]
    old = set().union(*(set(s) for s in info["sources"]))
    body = {p for p, v in info["cells"].items() if v not in info["controls"].values()}
    r, c, h, w = info["bounds"]
    stationary = _orient(inp[r:r + h, c:c + w], info["matrix"])
    return dict(info=info, shape=inp.shape, output_shape=out.shape,
                count=len(info["sources"]), patterns=patterns,
                same_count_distinct=any(len(a) == len(b) and a != b for i, a in enumerate(patterns) for b in patterns[i + 1:]),
                blank_edge=tight.shape != payload.shape,
                docking_needed=not np.array_equal(stationary, out),
                orientation_needed=not np.array_equal(payload, out),
                overlap=len(old & body),
                guide_fragment_counts=tuple(sorted(Counter(
                    next(iter(set(s.values()) & set(info["controls"].values())))
                    for s in info["sources"]).values())),
                registration_pose_choices=tuple(len(_candidates(s, info)) for s in info["sources"]),
                density=len(body) / payload.size,
                matrix=tuple(int(v) for v in info["matrix"].ravel()),
                background=bg, palette=tuple(int(v) for v in np.unique(inp)),
                piece_shapes=tuple((max(r for r, _ in s) - min(r for r, _ in s) + 1,
                                    max(c for _, c in s) - min(c for _, c in s) + 1) for s in info["sources"]))


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    _count(len(puzzle["train"]), "training count", 2, 10)
    _count(len(puzzle["test"]), "test count", 1, 4)
    if len(puzzle["train"]) <= len(puzzle["test"]):
        raise ValueError("need more training than test examples")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("pair must have exactly input and output")
            record = _measure(pair)
            key = _key(_grid(pair["input"]))
            if key in seen:
                raise ValueError("duplicate input")
            seen.add(key)
            # Lossless payload: guides are absent; registration/body cells remain.
            info = record["info"]
            if not np.array_equal(np.sort(info["payload"].ravel()), np.sort(_grid(pair["output"]).ravel())):
                raise ValueError("payload histogram changed")
            if set(np.unique(pair["output"])) & set(info["controls"].values()):
                raise ValueError("ruler cells must be removed")
            carried = Counter(v for source in info["sources"] for v in source.values()
                              if v not in info["controls"].values())
            returned = Counter(int(v) for v in _grid(pair["output"]).ravel()
                               if int(v) != info["background"])
            if carried != returned:
                raise ValueError("every registration and picture cell must survive exactly once")
            records[split].append(record)
    train = records["train"]
    if not any(r["same_count_distinct"] and r["docking_needed"] for r in train):
        raise ValueError("training must expose registration geometry and docking")
    if not any(r["blank_edge"] for r in train):
        raise ValueError("training must expose the exact measured blank margin")
    if not any(r["orientation_needed"] for r in train):
        raise ValueError("training must expose orientation")
    a, b = train[:2]
    ia, ib = a["info"], b["info"]
    if (set(ia["controls"]) != set(ib["controls"])
            or ia["controls"] == ib["controls"]
            or a["matrix"] == b["matrix"]
            or np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"])):
        raise ValueError("first demonstrations must contrast marker identity correspondence")
    ai, bi = _grid(puzzle["train"][0]["input"]), _grid(puzzle["train"][1]["input"])
    if not np.array_equal(ai[1:-1, 1:-1], bi[1:-1, 1:-1]):
        raise ValueError("correspondence contrast must keep all interior evidence fixed")
    if not _asymmetric(_grid(puzzle["train"][0]["output"])):
        raise ValueError("the correspondence contrast must use an asymmetric picture")
    if max(a["guide_fragment_counts"]) < 2 or not any(n > 1 for n in a["registration_pose_choices"]):
        raise ValueError("training must expose fragment joining and unresolved registration poses")
    max_train_count = max(r["count"] for r in train)
    taught_corners = {frozenset(r["info"]["controls"]) for r in train}
    count_shift = all(r["count"] > max_train_count for r in records["test"])
    corner_shift = all(frozenset(r["info"]["controls"]) not in taught_corners for r in records["test"])
    if not (count_shift or corner_shift):
        raise ValueError("tests must extend fragment count or control corner")
    return None


def self_test():
    records, puzzle_keys = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_keys.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for p in puzzle[split]:
                r = _measure(p)
                r["split"] = split
                records.append(r)
                original = p["input"].copy()
                assert np.array_equal(solve(original), p["output"])
                assert np.array_equal(original, p["input"])
        if seed < 2:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
                       for a, b in zip(puzzle[s], repeat[s]) for k in ("input", "output"))
    assert len(puzzle_keys) == 42
    assert {r["count"] for r in records} == {2, 3, 4}
    assert len({r["shape"] for r in records}) >= 20
    assert len({r["output_shape"] for r in records}) >= 20
    assert len({r["matrix"] for r in records}) == 8
    assert len({p for r in records for p in r["patterns"]}) >= 12
    assert len({s for r in records for s in r["piece_shapes"]}) >= 20
    assert {r["background"] for r in records} == set(range(10))
    assert any(r["overlap"] for r in records)
    assert any(not r["blank_edge"] for r in records)
    assert all(r["docking_needed"] for r in records if r["split"] == "train")
    for nt, ne in ((2, 1), (10, 4), (10, 1), (5, 4)):
        validate_generated_puzzle(generate_puzzle(817 + nt, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    good = generate_puzzle(9)
    # Rotation/reflection of the whole scene rotates/reflects the returned
    # picture, and an arbitrary palette permutation just permutes its cells.
    for t in range(8):
        matrix = np.column_stack((_point((1, 0), t), _point((0, 1), t)))
        pair = good["test"][0]
        assert np.array_equal(solve(_orient(pair["input"], matrix)),
                              _orient(pair["output"], matrix))
    color_map = np.array([8, 4, 9, 0, 7, 2, 6, 1, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(color_map[good["test"][0]["input"]]),
                          color_map[good["test"][0]["output"]])
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in good[s]] for s in good}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    # The exact crop cannot silently become a tight foreground crop.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in good[s]] for s in good}
    info = _decode(bad["train"][0]["input"])[1]
    p = bad["train"][0]["output"]
    points = np.argwhere(p != info["background"])
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    bad["train"][0]["output"] = p[lo[0]:hi[0], lo[1]:hi[1]].copy()
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("tight crop accepted")
    # A hole in a long ruler destroys the promised straight uninterrupted
    # segment and/or the unique guide reconstruction; it is not ignored noise.
    bad_input = good["train"][0]["input"].copy()
    source = max(info["sources"], key=lambda s: max(
        sum(v == color for v in s.values()) for color in info["controls"].values()))
    color = next(iter(set(source.values()) & set(info["controls"].values())))
    ruler = sorted(p for p, v in source.items() if v == color)
    bad_input[ruler[len(ruler) // 2]] = info["background"]
    try:
        solve(bad_input)
    except ValueError:
        pass
    else:
        raise AssertionError("broken ruler accepted")


if __name__ == "__main__":
    self_test()
    print("42-seed family self-test passed")
