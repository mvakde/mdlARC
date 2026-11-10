"""Contour paint gauges: sparse topology controls an embedded layer code.

Only NumPy and the standard library are used. Generation paints from its proposed
polygon; solve reconstructs everything from the returned input. No file I/O,
parent imports, retained puzzle state, or execution trace storage occurs here.
"""

from collections import deque
import hashlib
import numpy as np


_DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_FOUR = ((-1, 0), (0, -1), (0, 1), (1, 0))
_FAMILIES = ("rectangle", "square", "diamond", "chamfer", "elbow", "notch", "neck")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = max(abs(dr), abs(dc))
    if not n or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise ValueError("unaligned contour vertices")
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return [(a[0] + k * sr, a[1] + k * sc) for k in range(n + 1)]


def _edges(points):
    occupied, edges = set(points), []
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            try:
                line = _segment(a, points[j])
            except ValueError:
                continue
            if not occupied.intersection(line[1:-1]):
                edges.append((i, j))
    return edges


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0])
            or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1])
            or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _cycle(points):
    """Unique noncrossing full cycle; no latent order or shortest-tour heuristic."""
    if not 4 <= len(points) <= 12 or len(set(points)) != len(points):
        raise ValueError("contour must have 4..12 distinct landmarks")
    edges = _edges(points)
    adjacency = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    if any(len(a) < 2 for a in adjacency):
        raise ValueError("no closed contour")
    conflicts = [set() for _ in edges]
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges[:i]):
            if not {a, b}.intersection((c, d)) and _intersect(points[a], points[b], points[c], points[d]):
                conflicts[i].add(j)
                conflicts[j].add(i)
    solutions, visits = {}, [0]

    def visit(order, used, selected):
        visits[0] += 1
        if visits[0] > 30000:
            raise ValueError("contour search budget exceeded")
        if len(solutions) > 1:
            return
        if len(order) == len(points):
            closing = [k for v, k in adjacency[order[-1]] if v == 0]
            if closing and not conflicts[closing[0]].intersection(selected):
                key = tuple(sorted(selected + [closing[0]]))
                solutions[key] = order
            return
        for vertex, edge in adjacency[order[-1]]:
            if vertex not in used and not conflicts[edge].intersection(selected):
                visit(order + [vertex], used | {vertex}, selected + [edge])

    visit([0], {0}, [])
    if len(solutions) != 1:
        raise ValueError("no unique simple closed contour")
    signature, order = next(iter(solutions.items()))
    boundary = set()
    for i in signature:
        a, b = edges[i]
        boundary.update(_segment(points[a], points[b]))
    return [points[i] for i in order], boundary, edges


def _geometry(polygon, shape):
    """Integer cell centers inside/on a continuous polygon, including all strokes."""
    boundary = set()
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        boundary.update(_segment(a, b))
    mask = np.zeros(shape, dtype=bool)
    for r in range(min(p[0] for p in polygon), max(p[0] for p in polygon) + 1):
        for c in range(min(p[1] for p in polygon), max(p[1] for p in polygon) + 1):
            inside = False
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                if (a[0] > r) != (b[0] > r):
                    crossing = a[1] + (r - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
                    if c < crossing:
                        inside = not inside
            mask[r, c] = inside or (r, c) in boundary
    return mask, boundary


def _distances(mask, boundary, neighbors=_FOUR):
    depth = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque(sorted(boundary))
    for p in boundary:
        depth[p] = 0
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and depth[rr, cc] == -1:
                depth[rr, cc] = depth[r, c] + 1
                queue.append((rr, cc))
    if np.any(mask & (depth < 0)):
        raise ValueError("disconnected silhouette")
    return depth


def _decode(input_grid):
    """Find the single cap, directed gauge and residual same-color contour."""
    inp = _grid(input_grid)
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if int(max(counts)) * 2 <= inp.size:
        raise ValueError("background must occupy more than half the input")
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(inp == c)] for c in values if c != bg}
    solutions, cache = {}, {}
    h, w = inp.shape
    for color, source in groups.items():
        if len(source) < 4:
            continue
        for cap_color, caps in groups.items():
            if cap_color == color or len(caps) != 1:
                continue
            cap = caps[0]
            exceptions = [p for c, ps in groups.items() if c not in (color, cap_color) for p in ps]
            if not exceptions:
                continue
            for dr, dc in _DIRECTIONS:
                anchor = (cap[0] + dr, cap[1] + dc)
                if not (0 <= anchor[0] < h and 0 <= anchor[1] < w) or inp[anchor] != color:
                    continue
                steps = []
                for r, c in exceptions:
                    k = (r - anchor[0]) * dr if dr else (c - anchor[1]) * dc
                    if k < 1 or k > 14 or (anchor[0] + k * dr, anchor[1] + k * dc) != (r, c):
                        break
                    steps.append(k)
                else:
                    ray = [anchor]
                    for k in range(1, 15):
                        p = (anchor[0] + k * dr, anchor[1] + k * dc)
                        if not (0 <= p[0] < h and 0 <= p[1] < w) or inp[p] in (bg, cap_color):
                            break
                        ray.append(p)
                        if k < max(steps) or k < 2:
                            continue
                        points = tuple(sorted(set(source).difference(ray[1:])))
                        if not 4 <= len(points) <= 12:
                            continue
                        if points not in cache:
                            try:
                                order, boundary, edges = _cycle(list(points))
                                mask, _ = _geometry(order, inp.shape)
                                depth = _distances(mask, boundary)
                                cache[points] = (order, boundary, edges, mask, depth)
                            except ValueError:
                                cache[points] = None
                        entry = cache[points]
                        if entry is None:
                            continue
                        order, boundary, edges, mask, depth = entry
                        if mask[cap] or int(depth.max()) != k or any(depth[p] != i for i, p in enumerate(ray)):
                            continue
                        code = tuple(int(inp[p]) for p in ray)
                        key = (color, cap, (dr, dc), points, code)
                        solutions[key] = {"background": bg, "color": color, "cap_color": cap_color,
                                          "cap": cap, "direction": (dr, dc), "anchor": anchor,
                                          "points": points, "polygon": order, "boundary": boundary,
                                          "edges": edges, "mask": mask, "depth": depth, "code": code,
                                          "ray": tuple(ray)}
    if len(solutions) != 1:
        raise ValueError(f"expected one recoverable paint gauge, found {len(solutions)}")
    return next(iter(solutions.values()))


def _paint(inp, info, code=None, depth=None, mask=None):
    code = np.asarray(info["code"] if code is None else code, dtype=np.int8)
    depth = info["depth"] if depth is None else depth
    mask = info["mask"] if mask is None else mask
    out = np.asarray(inp).copy()
    out[mask] = code[np.minimum(depth[mask], len(code) - 1)]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    return _paint(inp, _decode(inp))


def _transform(points, rotation, reflected):
    result = []
    for r, c in points:
        if reflected:
            c = -c
        for _ in range(rotation):
            r, c = c, -r
        result.append((r, c))
    lo_r, lo_c = min(r for r, c in result), min(c for r, c in result)
    return [(r - lo_r, c - lo_c) for r, c in result]


def _proposal(rng, family, d):
    def integer(low, high):
        return int(rng.integers(low, high + 1))

    if family in ("rectangle", "square", "chamfer"):
        h = integer(2 * d, min(27, 2 * d + 4))
        w = h if family == "square" else integer(2 * d, min(27, 2 * d + 7))
        if family != "chamfer":
            points = [(0, 0), (0, w), (h, w), (h, 0)]
        else:
            a, b, c, e = [integer(0, max(1, min(h, w) // 3)) for _ in range(4)]
            points = [(0, a), (0, w - b), (b, w), (h - c, w),
                      (h, w - c), (h, e), (h - e, 0), (a, 0)]
            points = [p for i, p in enumerate(points) if p != points[i - 1]]
    elif family == "diamond":
        points = [(0, d), (d, 2 * d), (2 * d, d), (d, 0)]
    elif family in ("elbow", "notch"):
        h, w = [integer(min(27, 3 * d), min(27, 3 * d + 4)) for _ in range(2)]
        if family == "elbow":
            a, b = integer(d, min(w - 2, 2 * d)), integer(d, min(h - 2, 2 * d))
            points = [(0, 0), (0, a), (b, a), (b, w), (h, w), (h, 0)]
        else:
            a = integer(max(2, w // 4), max(2, w // 2 - 1))
            b = integer(a + 1, w - 2)
            cut = integer(d, min(h - 2, 2 * d))
            points = [(0, 0), (0, a), (cut, a), (cut, b), (0, b), (0, w), (h, w), (h, 0)]
    else:
        # Offset the lobes vertically. Symmetric equal-height lobes create
        # competing aligned full cycles and are intentionally not accepted.
        h = integer(2 * d, min(25, 2 * d + 3))
        a = min(2 * d + integer(0, 1), 11)
        gap = integer(2, 4)
        right = min(2 * d + integer(0, 1), 27 - a - gap)
        start = integer(1, min(3, h - 3))
        end = min(27, start + h + integer(-1, 1))
        top = integer(start + 1, max(start + 1, h // 2))
        bottom = integer(top + 1, min(h - 1, top + 3))
        w = a + gap + right
        points = [(0, 0), (0, a), (top, a), (top, a + gap), (start, a + gap), (start, w),
                  (end, w), (end, a + gap), (bottom, a + gap), (bottom, a), (h, a), (h, 0)]
    return _transform(points, integer(0, 3), bool(integer(0, 1)))


def _gauge_options(points, mask, depth, directions):
    d, options = int(depth.max()), []
    for anchor in points:
        for dr, dc in directions:
            cap = (anchor[0] - dr, anchor[1] - dc)
            if 0 <= cap[0] < mask.shape[0] and 0 <= cap[1] < mask.shape[1] and mask[cap]:
                continue
            ray = [(anchor[0] + k * dr, anchor[1] + k * dc) for k in range(d + 1)]
            if all(0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and depth[r, c] == k
                   for k, (r, c) in enumerate(ray)):
                # A gauge may not consume another landmark.
                if not set(points).intersection(ray[1:]):
                    options.append((cap, (dr, dc), ray))
    return options


def _sample_code(rng, d, source, available, mode):
    length = d + 1
    if mode == "repeat":
        a, b = map(int, rng.choice(available, 2, replace=False))
        code = [source, a, a] + [int(rng.choice([source, a])) for _ in range(length - 4)] + [b]
    elif mode == "return":
        a = int(rng.choice(available))
        others = [int(c) for c in available if c != a]
        code = [source, a] + [int(rng.choice([source, a] + others[:int(rng.integers(0, 3))]))
                               for _ in range(length - 3)] + [source]
    else:
        k = int(rng.integers(2, min(8, length) + 1))
        palette = [source] + [int(c) for c in rng.choice(available, k - 1, replace=False)]
        style = int(rng.integers(4))
        if style == 0:
            # Ordered broad bands, including a possibly thick outer band.
            thickness = np.ones(k, dtype=int)
            thickness += rng.multinomial(length - k, np.full(k, 1 / k))
            code = np.repeat(palette, thickness).astype(int).tolist()
        elif style == 1:
            code = [palette[i % k] for i in range(length)]
        else:
            code = palette + [int(rng.choice(palette)) for _ in range(length - k)]
            tail = code[1:]
            rng.shuffle(tail)
            code = [source] + tail
    return tuple(code)


def _concave(points):
    signs = {int(np.sign(_cross(points[i - 1], p, points[(i + 1) % len(points)])))
             for i, p in enumerate(points)}
    return -1 in signs and 1 in signs


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _make_pair(rng, families, depth_range, directions, mode=None, background=None):
    """Bounded geometry proposals; accepted only when visible reconstruction agrees."""
    for attempt in range(240):
        family = str(rng.choice(families))
        target = int(rng.integers(depth_range[0], depth_range[1] + 1))
        points = _proposal(rng, family, target)
        shape = (max(p[0] for p in points) + 1, max(p[1] for p in points) + 1)
        try:
            order, boundary, _ = _cycle(points)
            intended = set().union(*(set(_segment(a, b)) for a, b in zip(points, points[1:] + points[:1])))
            if boundary != intended:
                continue
            mask, _ = _geometry(points, shape)
            depth = _distances(mask, boundary)
        except ValueError:
            continue
        d = int(depth.max())
        if not depth_range[0] <= d <= depth_range[1]:
            continue
        if mode == "return" and not _concave(order):
            continue
        options = _gauge_options(points, mask, depth, directions)
        if not options:
            continue
        cap, direction, ray = options[int(rng.integers(len(options)))]
        lo_r, lo_c = min(0, cap[0]), min(0, cap[1])
        full_h = max(shape[0] - 1, cap[0]) - lo_r + 1
        full_w = max(shape[1] - 1, cap[1]) - lo_c + 1
        if max(full_h, full_w) > 30:
            continue
        extra_r = int(rng.integers(0, min(6, 30 - full_h) + 1))
        extra_c = int(rng.integers(0, min(6, 30 - full_w) + 1))
        shift_r = int(rng.integers(extra_r + 1)) - lo_r
        shift_c = int(rng.integers(extra_c + 1)) - lo_c
        shift = lambda p: (p[0] + shift_r, p[1] + shift_c)
        points, ray, cap = [shift(p) for p in points], [shift(p) for p in ray], shift(cap)
        bg = int(rng.integers(10)) if background is None else background
        palette = [int(c) for c in rng.permutation(10) if c != bg]
        source, cap_color = palette[:2]
        code = _sample_code(rng, d, source, palette[2:], mode)
        inp = np.full((full_h + extra_r, full_w + extra_c), bg, dtype=np.int8)
        for p in points:
            inp[p] = source
        for p, c in zip(ray, code):
            inp[p] = c
        inp[cap] = cap_color
        # Labels use the sampled ordered polygon, not the input-only decoder.
        mask, boundary = _geometry(points, inp.shape)
        depth = _distances(mask, boundary)
        out = inp.copy()
        out[mask] = np.asarray(code, dtype=np.int8)[depth[mask]]
        try:
            info = _decode(inp)
        except ValueError:
            continue
        if set(info["points"]) != set(points) or info["code"] != code:
            continue
        if mode == "return" and np.array_equal(_alternative(inp, info, "eight"), out):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError(f"bounded geometry construction exhausted: {families}/{depth_range}/{directions}")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # Presentation regimes never alter semantics or hide a solving convention.
    regime = int(rng.integers(2))
    held = _DIRECTIONS[int(rng.integers(8))]
    train_directions = tuple(v for v in _DIRECTIONS if regime == 1 or v != held)
    train_depths = (3, 5) if regime == 1 else (3, 7)
    test_depths = (6, 9) if regime == 1 else (2, 8)
    test_directions = _DIRECTIONS if regime == 1 else (held,)
    puzzle, seen = {"train": [], "test": []}, set()
    first_direction, first_background = None, None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            mode = ("repeat" if index == 0 else "return" if index == 1 else None) if split == "train" else None
            families = ("square",) if mode == "repeat" else ("elbow", "notch", "neck") if mode == "return" else _FAMILIES
            directions = train_directions if split == "train" else test_directions
            if mode == "return":
                directions = tuple(v for v in directions if v != first_direction)
            depths = train_depths if split == "train" else test_depths
            depths = (max(4, depths[0]), depths[1]) if mode == "repeat" else depths
            bg = ((first_background + int(rng.integers(1, 10))) % 10) if mode == "return" else None
            for retry in range(12):
                pair = _make_pair(rng, families, depths, directions, mode, bg)
                fingerprint = _key(pair["input"])
                if fingerprint not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(fingerprint)
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                info = _decode(pair["input"])
                first_direction, first_background = info["direction"], info["background"]
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(inp, info, name):
    code, mask, boundary = info["code"], info["mask"], info["boundary"]
    if name == "uniform":
        return _paint(inp, info, [code[0]])
    if name == "reverse":
        return _paint(inp, info, code[::-1])
    if name == "collapse":
        compact = [c for i, c in enumerate(code) if i == 0 or c != code[i - 1]]
        return _paint(inp, info, compact)
    if name == "eight":
        return _paint(inp, info, depth=_distances(mask, boundary, _DIRECTIONS))
    if name == "chords":
        all_lines = set(boundary)
        points = info["points"]
        for a, b in info["edges"]:
            all_lines.update(_segment(points[a], points[b]))
        # Indiscriminate alignments become false depth-zero contour strokes.
        all_lines = {p for p in all_lines if mask[p]}
        return _paint(inp, info, depth=_distances(mask, all_lines))
    if name == "box":
        rr, cc = np.nonzero(mask)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        box = np.zeros(mask.shape, dtype=bool)
        box[r0:r1 + 1, c0:c1 + 1] = True
        border = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if r in (r0, r1) or c in (c0, c1)}
        return _paint(inp, info, depth=_distances(box, border), mask=box)
    raise ValueError(name)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "train/test keys")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "pair lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    alternatives = {name: False for name in ("uniform", "reverse", "collapse", "eight", "chords", "box")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "input/output keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _check(inp.shape == out.shape, "canvas preservation")
            fingerprint = _key(inp)
            _check(fingerprint not in seen, "duplicate input")
            seen.add(fingerprint)
            info = _decode(inp)
            infos[split].append(info)
            # Reconstruct rings by synchronous frontier growth rather than indexing
            # the decoder's distance array. This also checks every occupied depth.
            expected, assigned = inp.copy(), set(info["boundary"])
            front = set(assigned)
            for index, color in enumerate(info["code"]):
                _check(bool(front), "empty gauge depth")
                for p in front:
                    expected[p] = color
                nxt = set()
                for r, c in front:
                    for dr, dc in _FOUR:
                        p = (r + dr, c + dc)
                        if 0 <= p[0] < inp.shape[0] and 0 <= p[1] < inp.shape[1] and info["mask"][p] and p not in assigned:
                            nxt.add(p)
                assigned.update(nxt)
                front = nxt
            _check(not front and len(assigned) == int(info["mask"].sum()), "code must cover the entire silhouette exactly")
            _check(np.array_equal(out, expected), "incorrect contour-layer painting")
            _check(np.array_equal(inp[inp != info["background"]], out[inp != info["background"]]), "existing pixels changed")
            _check(np.any(inp != out), "identity example")
            _check(2 <= len(info["code"]) - 1 <= 14, "depth range")
            _check(len(set(info["code"])) >= 2, "uninformative gauge")
            if split == "train":
                for name in alternatives:
                    alternatives[name] |= not np.array_equal(out, _alternative(inp, info, name))
    _check(all(alternatives.values()), f"training leaves shortcut unrefuted: {alternatives}")
    train, test = infos["train"], infos["test"]
    _check(any(_concave(i["polygon"]) for i in train), "no demonstrated concavity")
    _check(len({i["direction"] for i in train}) >= 2, "need two gauge directions")
    _check(len({i["background"] for i in train}) >= 2, "need varied backgrounds")
    novel_direction = all(i["direction"] not in {t["direction"] for t in train} for i in test)
    deeper = min(len(i["code"]) for i in test) > max(len(i["code"]) for i in train)
    _check(novel_direction or deeper, "missing observable train/test shift")


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, bounded summaries, semantic fixtures and corruption checks."""
    signatures, shapes, geometry, colors, backgrounds, directions, depths = [set() for _ in range(7)]
    concave = split_cores = unit_edges = diagonal = 0
    shifts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        train_info = [_decode(p["input"]) for p in puzzle["train"]]
        test_info = [_decode(p["input"]) for p in puzzle["test"]]
        shifts.add("depth" if min(len(i["code"]) for i in test_info) > max(len(i["code"]) for i in train_info) else "direction")
        for pair in puzzle["train"] + puzzle["test"]:
            inp = pair["input"]
            info = _decode(inp)
            _check(np.array_equal(solve(inp), pair["output"]), "solve mismatch")
            shapes.add(inp.shape)
            points = info["points"]
            lo_r, lo_c = min(r for r, c in points), min(c for r, c in points)
            geometry.add(tuple((r - lo_r, c - lo_c) for r, c in points))
            colors.update(int(c) for c in np.unique(inp))
            backgrounds.add(info["background"])
            directions.add(info["direction"])
            depths.add(len(info["code"]) - 1)
            concave += _concave(info["polygon"])
            poly = info["polygon"]
            unit_edges += sum(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(poly, poly[1:] + poly[:1]))
            diagonal += any(a[0] != b[0] and a[1] != b[1] for a, b in zip(poly, poly[1:] + poly[:1]))
            core = {tuple(map(int, p)) for p in np.argwhere(info["depth"] >= max(1, int(info["depth"].max()) - 1))}
            if core:
                visited, todo = {next(iter(core))}, [next(iter(core))]
                while todo:
                    r, c = todo.pop()
                    for dr, dc in _FOUR:
                        p = (r + dr, c + dc)
                        if p in core and p not in visited:
                            visited.add(p)
                            todo.append(p)
                split_cores += len(visited) < len(core)
    _check(len(signatures) == 42 and len(shapes) >= 50 and len(geometry) >= 100, "insufficient structural variation")
    _check(colors == backgrounds == set(range(10)), "palette/background coverage")
    _check(directions == set(_DIRECTIONS) and len(depths) >= 6, "gauge variation")
    _check(concave >= 42 and diagonal > 0 and split_cores > 0 and unit_edges > 0, "missing shape features")
    _check(shifts == {"depth", "direction"}, "shift regimes")
    # Exact semantic and size-boundary fixtures, including depth 14 beyond the
    # ordinary sampler. Their answers are explicit contour-distance formulas.
    for d in (2, 3, 14):
        code = np.array([1] + [2 + (i % 7) for i in range(d)], dtype=np.int8)
        square = np.zeros((2 * d + 2, 2 * d + 2), dtype=np.int8)
        square[0, 0] = 9
        for p in ((1, 1), (1, 2 * d + 1), (2 * d + 1, 2 * d + 1), (2 * d + 1, 1)):
            square[p] = 1
        for k, color in enumerate(code):
            square[1 + k, 1 + k] = color
        expected = square.copy()
        for r in range(1, 2 * d + 2):
            for c in range(1, 2 * d + 2):
                expected[r, c] = code[min(r - 1, c - 1, 2 * d + 1 - r, 2 * d + 1 - c)]
        diamond = np.zeros((2 * d + 2, 2 * d + 1), dtype=np.int8)
        diamond[0, d] = 9
        for p in ((1, d), (d + 1, 2 * d), (2 * d + 1, d), (d + 1, 0)):
            diamond[p] = 1
        for k, color in enumerate(code):
            diamond[1 + k, d] = color
        diamond_expected = diamond.copy()
        for r in range(1, 2 * d + 2):
            for c in range(2 * d + 1):
                distance = abs(r - d - 1) + abs(c - d)
                if distance <= d:
                    diamond_expected[r, c] = code[d - distance]
        for inp, out in ((square, expected), (diamond, diamond_expected)):
            for turns in range(4):
                for reflect in (False, True):
                    a, b = np.rot90(inp, turns), np.rot90(out, turns)
                    if reflect:
                        a, b = np.fliplr(a), np.fliplr(b)
                    _check(np.array_equal(solve(a.tolist()), b), "explicit square/diamond fixture")
    _check(_signature(generate_puzzle(12345)) == _signature(generate_puzzle(12345)), "nondeterminism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 10 * nt + nq, nt, nq)
            _check((len(p["train"]), len(p["test"])) == (nt, nq), "counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    original = generate_puzzle(17)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    p = bad["test"][0]
    changed = tuple(np.argwhere(p["input"] != p["output"])[0])
    p["output"][changed] = p["input"][changed]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted layer accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal count combinations)")
