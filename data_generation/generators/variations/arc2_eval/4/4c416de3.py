"""Complete locally colored frame-corner stencils, with observable references.

The source stencils are transpose-symmetric. Extended asymmetric stencils use
axis reflections, and an adjacent-orientation training witness teaches this.
Geometry, colors, stencil and damage vary per input; no stencil dictionary is
shared with the solver. The input parser below never reads a generated output.
"""

import hashlib
import numpy as np


_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_SOURCE_HASHES = {
    "0c7e2774da1c1a728283161da678acedadaf76f456224b1fb8c98a1d4a6c2140",
    "85e541fd2bc623c971976422a86683a7e87b3ac70a3148a4d8a7552db3b2ceac",
    "2db069dcf51832333752c397c455d56784ca2cd89ae6a92d0fb24d16b78ac8d0",
    "9b0c7748d344df88482c5b8cbd62c24c087080ad2a9904d653a6848e20b9409a",
}


def _validate_count(name, value, lo, hi):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")


def _resolve_counts(seed, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        nt = 2 + seed % 9
        nq = 1 + (seed + seed // 9) % min(4, nt - 1)
    elif num_train is None:
        nq = int(num_test)
        nt = nq + 1 + seed % (10 - nq)
    elif num_test is None:
        nt = int(num_train)
        nq = 1 + seed % min(4, nt - 1)
    else:
        nt, nq = int(num_train), int(num_test)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    return int(nt), int(nq)


def _point_connected(points):
    if not points:
        return False
    seen = {min(points)}
    todo = list(seen)
    while todo:
        u, v = todo.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                q = (u + du, v + dv)
                if q in points and q not in seen:
                    seen.add(q)
                    todo.append(q)
    return len(seen) == len(points)


def _transpose(pattern):
    return frozenset((v, u) for u, v in pattern)


def _sample_stencil(rng, radius, symmetric):
    # Keep two recognizable frame-arm pixels beyond the ornament. Interior
    # and exterior cells can still reach radius three; the corner is optional.
    domain = {(u, v) for u in range(-radius, radius + 1)
              for v in range(-radius, radius + 1)
              if not ((u == 0 and abs(v) >= 2) or (v == 0 and abs(u) >= 2))}
    for _ in range(80):
        size = int(rng.integers(3, min(15, len(domain) + 1)))
        start = (int(rng.integers(-1, 2)), int(rng.integers(-1, 2)))
        pattern = {start}
        if symmetric:
            pattern.add(start[::-1])
        while len(pattern) < size:
            choices = []
            for q in sorted(domain - pattern):
                extension = {q, q[::-1]} if symmetric else {q}
                new = pattern | extension
                if len(new) <= size and _point_connected(new):
                    choices.append(new)
            if not choices:
                break
            pattern = choices[int(rng.integers(len(choices)))]
        if (len(pattern) >= 3 and _point_connected(pattern)
                and (pattern == _transpose(pattern)) == symmetric):
            return frozenset(pattern)
    return (frozenset({(-1, -1), (1, 1), (-1, 0), (0, -1), (0, 0)})
            if symmetric else frozenset({(-1, 0), (0, 0), (1, 1)}))


def _corners(rect, shape):
    t, l, b, r = rect
    h, w = shape
    return [(y, x, q) for q, (y, x) in enumerate(((t, l), (t, r), (b, l), (b, r)))
            if 0 <= y < h and 0 <= x < w]


def _outline(rect, shape):
    t, l, b, r = rect
    h, w = shape
    return {(y, x) for y in range(max(0, t), min(h, b + 1))
            for x in range(max(0, l), min(w, r + 1))
            if y in (t, b) or x in (l, r)}


def _positions(corner, pattern, shape=None):
    y, x, q = corner
    dy, dx = _DIRS[q]
    points = {(y + dy * u, x + dx * v) for u, v in pattern}
    if shape is not None:
        h, w = shape
        points = {(r, c) for r, c in points if 0 <= r < h and 0 <= c < w}
    return points


def _detect_geometry(inp, background, frame):
    """Find corners from visible arms, then join arms into closed or cut frames.

    Accent pixels may cover an outline, including its corner. No colored
    anchor is required. Invisible frame extents are represented only by the
    relevant canvas edge, and are never inferred or used for completion.
    """
    h, w = inp.shape
    candidates = []
    for y, x in np.argwhere(inp != background):
        y, x = int(y), int(x)
        for q, (dy, dx) in enumerate(_DIRS):
            if not (0 <= y + 3 * dy < h and 0 <= x + 3 * dx < w):
                continue
            if inp[y + 2 * dy, x] != frame or inp[y, x + 2 * dx] != frame:
                continue
            if all(inp[y + k * dy, x] != background and
                   inp[y, x + k * dx] != background for k in (1, 3)):
                candidates.append((y, x, q))
    assert candidates, "no recognizable frame corners"
    rectangles = set()
    for y, x, q in candidates:
        dy, dx = _DIRS[q]
        vertical = [a for a, b, o in candidates
                    if b == x and o == (q ^ 2) and (a - y) * dy >= 4]
        horizontal = [b for a, b, o in candidates
                      if a == y and o == (q ^ 1) and (b - x) * dx >= 4]
        other_y = min(vertical, key=lambda a: abs(a - y)) if vertical else (-1 if dy < 0 else h)
        other_x = min(horizontal, key=lambda b: abs(b - x)) if horizontal else (-1 if dx < 0 else w)
        rect = (min(y, other_y), min(x, other_x), max(y, other_y), max(x, other_x))
        rectangles.add(rect)
    rectangles = sorted(rectangles)
    expected_corners = [c for rect in rectangles for c in _corners(rect, inp.shape)]
    assert len(expected_corners) == len(set(expected_corners))
    assert set(expected_corners) == set(candidates), "ambiguous frame intersections"
    structure = set()
    for rect in rectangles:
        outline = _outline(rect, inp.shape)
        assert not structure & outline, "frames overlap"
        assert all(inp[p] != background for p in outline), "broken frame arms"
        structure |= outline
    assert all(tuple(p) in structure for p in np.argwhere(inp == frame))
    return rectangles, sorted(candidates), structure


def _read_input(inp, rotation=False):
    """Recover scene and largest compatible visible stencil from input alone."""
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    # A structural color must explain *all* long frame arms. Enumerating roles
    # also handles heavily reused accents which outnumber the frame pixels.
    solutions = []
    for frame in range(10):
        if frame == background or counts[frame] < 6:
            continue
        try:
            rects, corners, structure = _detect_geometry(inp, background, frame)
            assigned = {corner: [] for corner in corners}
            for y, x in np.argwhere((inp != background) & (inp != frame)):
                distances = [(int(y) - a) ** 2 + (int(x) - b) ** 2 for a, b, _ in corners]
                distance = min(distances)
                assert distances.count(distance) == 1, "clue has two equally near corners"
                corner = corners[distances.index(distance)]
                assert max(abs(int(y) - corner[0]), abs(int(x) - corner[1])) <= 3
                assigned[corner].append((int(y), int(x)))
            occurrences = []
            for corner, coords in assigned.items():
                if not coords:
                    continue
                colors = {int(inp[p]) for p in coords}
                assert len(colors) == 1, "two accent colors assigned to one corner"
                y, x, q = corner
                dy, dx = _DIRS[q]
                pattern = frozenset(((a - y) * dy, (b - x) * dx) for a, b in coords)
                if rotation and q in (1, 2):
                    pattern = _transpose(pattern)
                occurrences.append((corner, colors.pop(), pattern))
            assert len(occurrences) >= 2
            maximal = {p for _, _, p in occurrences
                       if all(other <= p for _, _, other in occurrences)}
            assert len(maximal) == 1, "no intact compatible stencil"
            pattern = maximal.pop()
            assert 3 <= len(pattern) <= 14 and _point_connected(pattern)
            pred = inp.copy()
            occupied = set()
            full = 0
            clipped = 0
            for corner, color, clue in occurrences:
                local = _transpose(pattern) if rotation and corner[2] in (1, 2) else pattern
                entire = _positions(corner, local)
                visible = _positions(corner, local, inp.shape)
                assert visible and not visible & occupied, "completed ornaments collide"
                # Geometry, rather than color-component membership, identifies
                # each copy. Even completed pixels must have a unique corner.
                for a, b in visible:
                    distances = [(a - y) ** 2 + (b - x) ** 2 for y, x, _ in corners]
                    idx = corners.index(corner)
                    assert distances.count(min(distances)) == 1 and distances[idx] == min(distances)
                    assert int(inp[a, b]) in (background, frame, color)
                own = next(rect for rect in rects if corner in _corners(rect, inp.shape))
                assert not visible & (structure - _outline(own, inp.shape))
                occupied |= visible
                for point in visible:
                    pred[point] = color
                full += int(clue == pattern and visible == entire)
                clipped += int(visible != entire)
            assert full >= 1, "no entirely visible reference"
            solutions.append((pred, {"background": background, "frame": frame,
                "rectangles": rects, "corners": corners, "occurrences": occurrences,
                "stencil": pattern, "intact": full, "clipped_ornaments": clipped}))
        except AssertionError:
            continue
    assert len(solutions) == 1, "input has missing or ambiguous geometric interpretation"
    return solutions[0]


def solve(train_pairs, test_input):
    """Prediction uses only training demonstrations and the query input."""
    conventions = []
    for rotation in (False, True):
        try:
            if all(np.array_equal(_read_input(p["input"], rotation)[0], p["output"])
                   for p in train_pairs):
                conventions.append(rotation)
        except AssertionError:
            pass
    assert conventions, "inconsistent training demonstrations"
    answers = []
    for rotation in conventions:
        try:
            answers.append(_read_input(test_input, rotation)[0])
        except AssertionError:
            pass
    assert answers and all(np.array_equal(a, answers[0]) for a in answers)
    return answers[0]


def _sample_rect(rng, shape, radius, small, clipping, first):
    h, w = shape
    minimum = 2 * radius + 3
    max_h = min(h - 2 * radius, minimum + (3 if small else 12))
    max_w = min(w - 2 * radius, minimum + (3 if small else 12))
    sh = int(rng.integers(minimum, max_h + 1))
    sw = int(rng.integers(minimum, max_w + 1))
    t = int(rng.integers(radius, h - radius - sh + 1))
    l = int(rng.integers(radius, w - radius - sw + 1))
    b, r = t + sh - 1, l + sw - 1
    if clipping and rng.random() < .64:
        edge = int(rng.choice(clipping))
        # A one-edge cut leaves two genuine visible corners. The first frame
        # always has at least two corners so even a one-frame scene is useful.
        if edge == 0:
            t, b = -1, int(rng.integers(minimum - 1, h - radius))
        elif edge == 1:
            l, r = -1, int(rng.integers(minimum - 1, w - radius))
        elif edge == 2:
            b, t = h, int(rng.integers(radius, h - minimum + 1))
        else:
            r, l = w, int(rng.integers(radius, w - minimum + 1))
        perpendicular = [e for e in clipping if e % 2 != edge % 2]
        if not first and perpendicular and rng.random() < .45:
            second = int(rng.choice(perpendicular))
            if second == 0:
                t = -1
            elif second == 1:
                l = -1
            elif second == 2:
                b = h
            else:
                r = w
    # Sometimes a visible corner is near the canvas edge: its ornament can be
    # cut, but all real frame corners still have long inward arms.
    if rng.random() < .18:
        if 0 <= t < h and t == radius:
            t = int(rng.integers(radius + 1))
        if 0 <= l < w and l == radius:
            l = int(rng.integers(radius + 1))
    return t, l, b, r


def _separated_rect(a, b, radius, shape):
    h, w = shape
    ta, la, ba, ra = max(0, a[0]), max(0, a[1]), min(h - 1, a[2]), min(w - 1, a[3])
    tb, lb, bb, rb = max(0, b[0]), max(0, b[1]), min(h - 1, b[2]), min(w - 1, b[3])
    gap = 2 * radius + 2
    return ba + gap <= tb or bb + gap <= ta or ra + gap <= lb or rb + gap <= la


def _draw_scene(rng, symmetric, clipping, roles, crowded, witness=False):
    radius = 1 if crowded else int(rng.choice((1, 1, 1, 2, 2, 3)))
    minimum = 4 * radius + 3
    if crowded:
        shape = (int(rng.integers(26, 31)), int(rng.integers(26, 31)))
    else:
        h = int(rng.integers(max(10, minimum), 31))
        w = h if rng.random() < .35 else int(rng.integers(max(10, minimum), 31))
        shape = (h, w)
    pattern = _sample_stencil(rng, radius, symmetric)
    target = int(rng.integers(4, 7)) if crowded else int(rng.integers(1, 5))
    frames = []
    for index in range(target):
        for attempt in range(100):
            rect = _sample_rect(rng, shape, radius, crowded or attempt > 40, clipping, index == 0)
            if witness and index == 0 and len(_corners(rect, shape)) < 4:
                continue
            if all(_separated_rect(rect, other, radius, shape) for other in frames):
                frames.append(rect)
                break
        else:
            break
    assert frames
    all_corners = [(fid, corner) for fid, rect in enumerate(frames) for corner in _corners(rect, shape)]
    marked = []
    for fid, rect in enumerate(frames):
        candidates = _corners(rect, shape)
        lo = 2 if fid == 0 else 0
        count = int(rng.integers(lo, len(candidates) + 1))
        selected = [candidates[int(i)] for i in rng.permutation(len(candidates))[:count]]
        if witness and fid == 0:
            q = int(rng.integers(4))
            selected = list(dict.fromkeys([candidates[q], candidates[q ^ 1]] + selected))
        marked.extend((fid, c) for c in selected)
    positions = [_positions(c, pattern, shape) for _, c in marked]
    full_options = [i for i, (_, c) in enumerate(marked)
                    if len(positions[i]) == len(pattern)]
    assert full_options and len(marked) >= 2 and all(positions)
    reference = int(rng.choice(full_options))
    full_indices = {reference}
    for i in full_options:
        if rng.random() < .3:
            full_indices.add(i)
    if len(full_indices) == len(marked):
        full_indices.remove(next(i for i in sorted(full_indices) if i != reference))
    if roles is None:
        background, frame = map(int, rng.choice(10, 2, replace=False))
    else:
        background, frame = roles
    palette = [c for c in range(10) if c not in (background, frame)]
    rng.shuffle(palette)
    palette_size = int(rng.integers(1, min(8, len(marked)) + 1))
    colors = [int(rng.choice(palette[:palette_size])) for _ in marked]
    base = np.full(shape, background, dtype=np.int8)
    for rect in frames:
        for point in _outline(rect, shape):
            base[point] = frame
    inp, out = base.copy(), base.copy()
    occupied = set()
    loss = float(rng.uniform(.12, .9))
    for i, ((fid, corner), points, color) in enumerate(zip(marked, positions, colors)):
        assert not occupied & points
        occupied |= points
        for j, rect in enumerate(frames):
            if j != fid:
                assert not points & _outline(rect, shape)
        for point in points:
            out[point] = color
        retained = sorted(points)
        rng.shuffle(retained)
        if i not in full_indices:
            # No distinguished anchor cell or mandatory singleton/almost-full
            # pair. Independent deletion masks can be disconnected and offset.
            n = int(rng.binomial(len(points), 1 - loss))
            retained = retained[:max(1, min(len(points) - 1, n))]
        for point in retained:
            inp[point] = color
    assert np.any(inp != out)
    assert int(np.argmax(np.bincount(inp.ravel(), minlength=10))) == background
    predicted, info = _read_input(inp)
    assert np.array_equal(predicted, out)
    assert set(info["corners"]) == {c for _, c in all_corners}
    assert len(info["rectangles"]) == len(frames)
    if witness:
        try:
            alternative = _read_input(inp, True)[0]
        except AssertionError:
            alternative = None
        assert alternative is None or not np.array_equal(alternative, out)
    return {"input": inp, "output": out}


def _pair_hash(pair):
    inp, out = pair["input"], pair["output"]
    return hashlib.sha256(bytes(inp.shape) + inp.astype(np.int8).tobytes()
                          + out.astype(np.int8).tobytes()).hexdigest()


def _input_key(inp):
    return inp.shape, inp.astype(np.int8).tobytes()


def _rescue_pair(rng, symmetric, roles, used_shapes, used_stencils):
    """Guaranteed spacious construction if free placement exhausts its budget."""
    patterns = ({(-1, 0), (0, -1), (0, 0), (1, 1)},
                {(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)}) if symmetric else (
                    {(-1, 0), (0, 0), (1, 1)},
                    {(-1, -1), (-1, 0), (0, 0), (1, 1)})
    pattern = next((frozenset(p) for p in patterns if frozenset(p) not in used_stencils),
                   frozenset(patterns[0]))
    shape = next((n, n) for n in range(16, 31) if (n, n) not in used_shapes)
    background, frame = roles if roles is not None else map(int, rng.choice(10, 2, replace=False))
    colors = [c for c in range(10) if c not in (background, frame)]
    rng.shuffle(colors)
    rect = (4, 4, shape[0] - 5, shape[1] - 5)
    inp = np.full(shape, background, dtype=np.int8)
    for p in _outline(rect, shape):
        inp[p] = frame
    out = inp.copy()
    for i, corner in enumerate(_corners(rect, shape)[:3]):
        for point in _positions(corner, pattern):
            out[point] = colors[i]
        for point in _positions(corner, pattern if i == 0 else {(1, 1)}):
            inp[point] = colors[i]
    assert np.array_equal(_read_input(inp)[0], out)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    nt, nq = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Reflection is the invariant extension. Symmetric-only puzzles need no
    # distinction from rotation; mixed puzzles contain an informative witness.
    mixed = bool(rng.random() < .58)
    regime = int(rng.choice((0, 0, 0, 1, 2)))  # same, boundary shift, count shift
    held_edge = int(rng.integers(4))
    stable_roles = tuple(map(int, rng.choice(10, 2, replace=False))) if rng.random() < .25 else None
    puzzle = {"train": [], "test": []}
    seen = set()
    shapes, stencils = set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            witness = mixed and split == "train" and index == 0
            symmetric = not mixed or (not witness and rng.random() < .60)
            clipping = tuple(e for e in range(4) if e != held_edge) if regime == 1 and split == "train" else tuple(range(4))
            # Fully visible scenes are common in either split. No mandatory
            # clipping or shape shift is imposed on a puzzle.
            if rng.random() < .4:
                clipping = ()
            crowded = bool((regime == 2 and split == "test") or rng.random() < .16)
            for attempt in range(80):
                try:
                    pair = _draw_scene(rng, symmetric, clipping, stable_roles, crowded, witness)
                    key = _input_key(pair["input"])
                    assert key not in seen and _pair_hash(pair) not in _SOURCE_HASHES
                    info = _read_input(pair["input"])[1]
                    if split == "train" and index == 1:
                        assert pair["input"].shape not in shapes
                        assert info["stencil"] not in stencils
                    break
                except AssertionError:
                    continue
            else:
                pair = _rescue_pair(rng, symmetric, stable_roles, shapes, stencils)
                key = _input_key(pair["input"])
                assert key not in seen and _pair_hash(pair) not in _SOURCE_HASHES
                info = _read_input(pair["input"])[1]
            seen.add(key)
            shapes.add(pair["input"].shape)
            stencils.add(info["stencil"])
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen = set()
    rotation_fits = True
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and np.any(inp != out)
            key = _input_key(inp)
            assert key not in seen and _pair_hash(pair) not in _SOURCE_HASHES
            seen.add(key)
            prediction, info = _read_input(inp)
            assert np.array_equal(prediction, out), "input-only prediction differs"
            changed = inp != out
            assert np.all(np.isin(inp[changed], [info["background"], info["frame"]]))
            assert set(np.unique(inp)) == set(np.unique(out))
            infos[split].append(info)
            if split == "train" and rotation_fits:
                try:
                    rotation_fits = np.array_equal(_read_input(inp, True)[0], out)
                except AssertionError:
                    rotation_fits = False
    assert len({p["input"].shape for p in puzzle["train"]}) >= 2
    assert len({x["stencil"] for x in infos["train"]}) >= 2
    # All still-plausible conventions must agree, or be ruled out by the
    # test input itself. This explicitly rejects the review's countermodel.
    if rotation_fits:
        for pair in puzzle["test"]:
            try:
                alternative = _read_input(pair["input"], True)[0]
            except AssertionError:
                continue
            assert np.array_equal(alternative, pair["output"]), "untaught reflection convention"


def _features(pair):
    """Measurements reconstructed from grids, never sampled plan metadata."""
    inp = pair["input"]
    _, info = _read_input(inp)
    pattern = info["stencil"]
    rects, occurrences = info["rectangles"], info["occurrences"]
    h, w = inp.shape
    frame_marks = [sum(c in _corners(rect, inp.shape) for c, _, _ in occurrences) for rect in rects]
    clips = [sum((t < 0, l < 0, b >= h, r >= w)) for t, l, b, r in rects]
    clue_counts = [len(p) for _, _, p in occurrences]
    partial = [p for _, _, p in occurrences if p != pattern]
    categories = set()
    for p in partial:
        for u, v in p:
            categories.add("inside" if u > 0 and v > 0 else
                           "row_arm" if u == 0 and v >= 0 else
                           "col_arm" if v == 0 and u >= 0 else "outside")
    colors = [c for _, c, _ in occurrences]
    frame_corners = [_corners(rect, inp.shape) for rect in rects]
    footprint = (max(u for u, _ in pattern) - min(u for u, _ in pattern) + 1,
                 max(v for _, v in pattern) - min(v for _, v in pattern) + 1)
    return {"shape": inp.shape, "frame_count": len(rects), "frame_marks": frame_marks,
            "frame_shapes": [(min(h - 1, b) - max(0, t) + 1,
                              min(w - 1, r) - max(0, l) + 1) for t, l, b, r in rects],
            "clipping": clips, "stencil_size": len(pattern),
            "radius": max(max(abs(u), abs(v)) for u, v in pattern),
            "footprint": footprint, "density": len(pattern) / (footprint[0] * footprint[1]),
            "symmetric": pattern == _transpose(pattern), "corner_in_stencil": (0, 0) in pattern,
            "intact": info["intact"], "clue_counts": clue_counts,
            "partial_counts": [len(p) for p in partial],
            "offset_clues": sum((0, 0) not in p for p in partial),
            "clue_categories": sorted(categories), "reused_colors": len(set(colors)) < len(colors),
            "accent_count": len(set(colors)), "background": info["background"], "frame": info["frame"],
            "clipped_ornaments": info["clipped_ornaments"],
            "clipped_reference": any(p == pattern and clips[i] > 0
                                     for c, _, p in occurrences for i, rect in enumerate(rects)
                                     if c in _corners(rect, inp.shape)),
            "same_frame_reuse": any(sum(color == k and c in cs for c, color, _ in occurrences) > 1
                                    for cs in frame_corners for k in set(colors)),
            "cross_frame_reuse": any(k == l and not any(c in cs and d in cs for cs in frame_corners)
                                     for c, k, _ in occurrences for d, l, _ in occurrences if c != d),
            "intact_partial_reuse": any(k == l and p == pattern and q != pattern
                                        for _, k, p in occurrences for _, l, q in occurrences),
            "adjacent_marks": any(c[2] ^ d[2] in (1, 2)
                                  for cs in frame_corners for c, _, _ in occurrences
                                  for d, _, _ in occurrences if c in cs and d in cs),
            "boundary_edges": sorted({e for t, l, b, r in rects for e, cut in
                                      enumerate((t < 0, l < 0, b >= h, r >= w)) if cut}),
            "orientations": sorted({c[2] for c, _, _ in occurrences})}


def _signature(puzzle):
    return tuple(_pair_hash(pair) for split in ("train", "test") for pair in puzzle[split])


def _regression_test():
    # Minimal version of the seed-11/14/23/27/35 countermodel: diagonal-only
    # asymmetric demonstrations admit rotation, but an adjacent query differs.
    # Build observations explicitly; the prediction routine sees no metadata.
    def example(n, pattern, orientations):
        inp = np.full((n, n), 7, dtype=np.int8)
        rect = (3, 3, n - 4, n - 4)
        for p in _outline(rect, inp.shape):
            inp[p] = 0
        out = inp.copy()
        corners = _corners(rect, inp.shape)
        for i, q in enumerate(orientations):
            for p in _positions(corners[q], pattern):
                out[p] = 2 + i
            for p in _positions(corners[q], pattern if i == 0 else {(0, 0)}):
                inp[p] = 2 + i
        return {"input": inp, "output": out}

    a = frozenset({(-1, 0), (0, 0), (1, 1)})
    b = frozenset({(-1, -1), (-1, 0), (0, 0), (1, 1)})
    train = [example(15, a, (0, 3)), example(17, b, (1, 2))]
    query = example(16, a, (2, 0))
    assert all(np.array_equal(_read_input(p["input"], True)[0], p["output"]) for p in train)
    assert not np.array_equal(_read_input(query["input"], True)[0], query["output"])
    try:
        solve(train, query["input"].copy())
    except AssertionError:
        pass
    else:
        raise AssertionError("diagonal-only convention ambiguity was missed")
    try:
        validate_generated_puzzle({"train": train, "test": [query]})
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted the review countermodel")
    witness = example(18, b, (0, 1, 3))
    assert np.array_equal(solve(train + [witness], query["input"]), query["output"])
    # Symmetry makes the two conventions identical, as in every source pair.
    symmetric = frozenset({(-1, 0), (0, -1), (0, 0), (1, 1)})
    symmetric_query = example(16, symmetric, (2, 0))
    assert np.array_equal(solve([example(15, symmetric, (0, 3))], symmetric_query["input"]),
                          symmetric_query["output"])
    for symmetric in (False, True):
        rescue = _rescue_pair(np.random.default_rng(91), symmetric, (6, 0), set(), set())
        assert np.array_equal(_read_input(rescue["input"])[0], rescue["output"])
    damaged = {"train": train + [witness], "test": [query]}
    damaged["test"][0] = {k: v.copy() for k, v in query.items()}
    damaged["test"][0]["output"][0, 0] = 4
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("unrelated output change was accepted")


def self_test() -> None:
    _regression_test()
    assert _signature(generate_puzzle(777, 6, 3)) == _signature(generate_puzzle(777, 6, 3))
    assert _signature(generate_puzzle(777, 6, 3)) != _signature(generate_puzzle(778, 6, 3))
    for i, (nt, nq) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                 (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + i, nt, nq)
        assert nt is None or len(puzzle["train"]) == nt
        assert nq is None or len(puzzle["test"]) == nq
        validate_generated_puzzle(puzzle)
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(99, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    features, counts, signatures, stencil_shapes = [], set(), set(), set()
    split_features = {"train": [], "test": []}
    puzzle_features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        signatures.add(_signature(puzzle))
        current = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                feature = _features(pair)
                features.append(feature)
                split_features[split].append(feature)
                current[split].append(feature)
                stencil_shapes.add(tuple(sorted(_read_input(pair["input"])[1]["stencil"])))
        puzzle_features.append(current)
        for pair in puzzle["test"]:
            # Infer before comparing: solve receives no test labels or seed.
            prediction = solve([{k: v.copy() for k, v in p.items()} for p in puzzle["train"]],
                               pair["input"].copy())
            assert np.array_equal(prediction, pair["output"])
    assert len({a for a, _ in counts}) >= 4 and len({b for _, b in counts}) >= 3
    assert len(signatures) == 42 and len(stencil_shapes) >= 60
    assert len({x["shape"] for x in features}) >= 60
    assert {x["frame_count"] for x in features} >= {1, 2, 3, 4}
    assert max(x["frame_count"] for x in features) >= 5
    assert {x["radius"] for x in features} >= {1, 2, 3}
    assert len({x["stencil_size"] for x in features}) >= 8
    assert {v for x in features for v in x["frame_marks"]} >= {0, 1, 2, 3, 4}
    assert {v for x in features for v in x["clipping"]} >= {0, 1, 2}
    assert {v for x in features for v in x["clue_categories"]} == {"inside", "outside", "row_arm", "col_arm"}
    assert any(x["frame"] == 0 for x in features)
    assert any(x["intact"] > 1 for x in features)
    assert any(x["clipped_reference"] for x in features)
    assert any(x["clipped_ornaments"] for x in features)
    assert sum(x["symmetric"] for x in features) > len(features) / 3
    assert any(not x["symmetric"] for x in features)
    assert any(not x["corner_in_stencil"] for x in features)
    assert any(x["offset_clues"] and x["reused_colors"] and max(x["clipping"]) for x in features)
    assert any(x["accent_count"] == 1 and x["frame_count"] == 1 for x in features)
    assert any(30 in x["shape"] for x in features)
    assert any(min(x["shape"]) <= 12 for x in features)
    for split, rows in split_features.items():
        assert {x["shape"][0] == x["shape"][1] for x in rows} == {False, True}
        assert any(30 in x["shape"] for x in rows)
        assert any(max(x["clipping"]) == 0 for x in rows)
        assert any(max(x["clipping"]) > 0 for x in rows)
        assert len({x["frame_count"] for x in rows}) >= 4
        assert len({x["stencil_size"] for x in rows}) >= 8
    for key in ("same_frame_reuse", "cross_frame_reuse", "intact_partial_reuse", "adjacent_marks"):
        assert any(x[key] for x in features), key
    assert any(1 not in x["partial_counts"] for x in features)
    assert any(x["stencil_size"] - 1 not in x["partial_counts"] for x in features)
    assert len({x["footprint"] for x in features}) >= 12
    assert len({x["density"] for x in features}) >= 12
    assert any(x["footprint"][0] != x["footprint"][1] for x in features)
    assert any(a == b == 5 for x in features for a, b in x["frame_shapes"])
    assert max(max(a, b) for x in features for a, b in x["frame_shapes"]) >= 16
    assert len({(x["shape"], x["frame_count"], x["stencil_size"]) for x in features}) >= 200
    assert any(max(x["frame_count"] for x in p["test"]) > max(x["frame_count"] for x in p["train"])
               for p in puzzle_features)
    assert any(set(e for x in p["test"] for e in x["boundary_edges"]) -
               set(e for x in p["train"] for e in x["boundary_edges"]) for p in puzzle_features)
    assert any(all(x["symmetric"] for split in p.values() for x in split) for p in puzzle_features)
    assert any(len({(x["background"], x["frame"]) for split in p.values() for x in split}) == 1
               for p in puzzle_features)
