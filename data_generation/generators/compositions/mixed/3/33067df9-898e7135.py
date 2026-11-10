"""Run-coded rectangle inventory -> enlarged, exactly filled blueprint.

Only NumPy and the standard library are required. No parent generator, file,
seed, training example, or generation metadata is consulted by solve().
"""
from collections import Counter, defaultdict
import hashlib
import math

import numpy as np


HOST_SEEDS = (0, 302227336, 753947178, 823799564, 141415716, 859489935,
              319684658, 233196019, 800950017, 387652436, 240960289,
              338506294, 475936224, 934513510, 828329465, 453504601,
              263787805, 378960343, 861205815, 237335671, 520898807,
              681552225, 515268915, 967666515, 779478813, 217866812,
              269015968, 929692979, 667662606, 717579126, 839973683,
              744324325, 365223268, 304396475, 365719669, 128845622,
              28615231, 658035050, 674066075, 817328430, 643041170,
              474910205)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _runs(row, bg):
    result, c = [], 0
    while c < len(row):
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        end = c + 1
        while end < len(row) and int(row[end]) == color:
            end += 1
        result.append((color, c, end))
        c = end
    return result


def _rectangles(coarse, bg, merge=True):
    """Return (color, top, left, height, width); endpoints are half-open."""
    rows = [_runs(row, bg) for row in coarse]
    result = []
    for r, runs in enumerate(rows):
        before = set(rows[r - 1]) if r else set()
        for color, left, end in runs:
            if merge and (color, left, end) in before:
                continue
            depth = 1
            while merge and r + depth < len(rows) and (color, left, end) in rows[r + depth]:
                depth += 1
            result.append((color, r, left, depth, end - left))
    return result


def _features(coarse, bg):
    rows = [_runs(row, bg) for row in coarse]
    unequal = any(a == b and (l, e) != (m, f) and max(l, m) < min(e, f)
                  for upper, lower in zip(rows, rows[1:])
                  for a, l, e in upper for b, m, f in lower)
    exact = any(set(a) & set(b) for a, b in zip(rows, rows[1:]))
    return exact, unequal


def _group_pieces(rects, scale):
    groups = defaultdict(list)
    for color, r, c, h, w in rects:
        if h % scale or w % scale:
            raise ValueError("a merged rectangle does not share the common scale")
        groups[tuple(sorted((h // scale, w // scale)))].append(color)
    return groups


def _recover(input_grid):
    """Read one self-contained input and reject missing or ambiguous covers."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("the outer canvas margin must be background")
    candidates = []
    # Source marks are isolated by the lattice. Only a solid blueprint border
    # can give a component with both a tall and a wide complete border.
    for structural in range(10):
        if structural == bg:
            continue
        for component in _components(a == structural):
            r0, r1, c0, c1 = _bounds(component)
            board = a[r0:r1, c0:c1]
            if min(board.shape) < 4:
                continue
            if not (np.all(board[0] == structural) and np.all(board[-1] == structural)
                    and np.all(board[:, 0] == structural) and np.all(board[:, -1] == structural)
                    and np.all((board == structural) | (board == bg))):
                continue
            holes = _components(board == bg)
            if len(holes) < 2:
                continue
            hole_rects = []
            for cells in holes:
                y0, y1, x0, x1 = _bounds(cells)
                if len(cells) != (y1-y0) * (x1-x0):
                    break
                hole_rects.append((y0, y1, x0, x1))
            else:
                candidates.append((structural, (r0, r1, c0, c1), board.copy(), hole_rects))
    if len(candidates) != 1:
        raise ValueError("expected one rectangular blueprint with rectangular cavities")
    structural, board_bounds, board, holes = candidates[0]
    r0, r1, c0, c1 = board_bounds
    outside = a.copy()
    outside[r0:r1, c0:c1] = bg
    cells = list(zip(*np.nonzero(outside != bg)))
    if not cells or np.any(outside == structural):
        raise ValueError("the structural color is reserved for the blueprint")
    y0, y1, x0, x1 = _bounds(cells)
    # All payload lies on one pitch-two lattice, independently translated.
    if any((r-y0) % 2 or (c-x0) % 2 for r, c in cells):
        raise ValueError("source payload is not on one sparse lattice")
    # Require one separating background row or column between bounding boxes.
    if not (y1 < r0 or r1 < y0 or x1 < c0 or c1 < x0):
        raise ValueError("source and blueprint bounding boxes must be separated")
    coarse = outside[y0:y1:2, x0:x1:2].copy()
    rects = _rectangles(coarse, bg)
    hole_area = sum((b-a_) * (d-c) for a_, b, c, d in holes)
    source_area = int(np.count_nonzero(coarse != bg))
    if source_area % hole_area:
        raise ValueError("inventory and cavity areas do not imply an integer scale")
    ratio = source_area // hole_area
    scale = math.isqrt(ratio)
    if scale < 2 or scale * scale != ratio or max(board.shape) * scale > 30:
        raise ValueError("invalid common enlargement factor")
    groups = _group_pieces(rects, scale)
    hole_groups = defaultdict(list)
    for bounds in holes:
        a_, b, c, d = bounds
        hole_groups[tuple(sorted((b-a_, d-c)))].append(bounds)
    if set(groups) != set(hole_groups) or any(len(groups[k]) != len(hole_groups[k]) for k in groups):
        raise ValueError("piece inventory does not exactly cover the cavities")
    if any(len(set(colors)) != 1 for colors in groups.values()):
        raise ValueError("congruent differently colored pieces give ambiguous answers")
    logical = board.copy()
    for dims, bounds_list in hole_groups.items():
        color = groups[dims][0]
        for a_, b, c, d in bounds_list:
            logical[a_:b, c:d] = color
    result = np.repeat(np.repeat(logical, scale, 0), scale, 1)
    rotation = False
    for dims in groups:
        members = [p for p in rects if tuple(sorted((p[3] // scale, p[4] // scale))) == dims]
        if len(members) == 1:
            _, _, _, h, w = members[0]
            a_, b, c, d = hole_groups[dims][0]
            rotation |= (h // scale, w // scale) != (b-a_, d-c)
    # Is area alone sufficient to determine colors? An equal-area pair of
    # noncongruent rectangles with different colors is a counterexample.
    equal_area = any(h*w == j*k and (h, w) != (j, k) and groups[(h, w)][0] != groups[(j, k)][0]
                     for h, w in groups for j, k in groups)
    positional = board.copy()
    for p, (a_, b, c, d) in zip(rects, holes):
        positional[a_:b, c:d] = p[0]
    info = {"background": bg, "structural": structural, "scale": scale,
            "board": board, "board_bounds": board_bounds, "holes": holes,
            "coarse": coarse, "source_bounds": (y0, y1, x0, x1),
            "rectangles": rects, "classes": tuple(sorted(groups)),
            "piece_count": len(rects), "rotation": rotation,
            "equal_area_distinct_colors": equal_area,
            "positional_wrong": not np.array_equal(positional, logical),
            "exact": _features(coarse, bg)[0], "unequal": _features(coarse, bg)[1],
            "logical_source_shape": (coarse.shape[0] // scale, coarse.shape[1] // scale)}
    return result, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return only the enlarged filled board, using the input alone."""
    return _recover(input_grid)[0]


def _trim(a, bg):
    cells = list(zip(*np.nonzero(a != bg)))
    y0, y1, x0, x1 = _bounds(cells)
    return a[y0:y1, x0:x1].copy()


def _source(rng, bg, colors, role, scale, large):
    """Sample at logical source scale; amplify before exposing run evidence."""
    if role == "partial":
        h = int(rng.integers(3, 5))
        w = int(rng.integers(3, 5))
        base = np.full((h, w), bg, dtype=np.int8)
        # Two adjoining same-color rectangles with unequal, overlapping spans.
        split = int(rng.integers(1, h))
        spans = [(l, e) for l in range(w) for e in range(l+1, w+1)]
        pairs = [(a, b) for a in spans for b in spans
                 if a != b and max(a[0], b[0]) < min(a[1], b[1])]
        first, second = pairs[int(rng.integers(len(pairs)))]
        base[:split, first[0]:first[1]] = colors[0]
        base[split:, second[0]:second[1]] = colors[0]
        # Sometimes add a third differently colored rectangle in free space.
        if rng.random() < .5:
            free = np.argwhere(base == bg)
            if len(free):
                r, c = map(int, free[int(rng.integers(len(free)))])
                base[r, c] = colors[1]
    elif role == "area":
        # Same area, different side lengths: fitting cannot use area alone.
        base = np.full((4, 4), bg, dtype=np.int8)
        if rng.random() < .5:
            base[0, :] = colors[0]
            left = int(rng.integers(3))
            base[1:3, left:left+2] = colors[1]
        else:
            base[:, 0] = colors[0]
            top = int(rng.integers(3))
            base[top:top+2, 1:3] = colors[1]
        if rng.random() < .4:
            free = np.argwhere(base == bg)
            r, c = map(int, free[int(rng.integers(len(free)))])
            base[r, c] = colors[2]
    else:
        cap = min(4, 14 // scale)
        if large:
            h, w = int(rng.integers(5, 8)), int(rng.integers(2, 5))
            if rng.random() < .5:
                h, w = w, h
        else:
            h, w = int(rng.integers(2, cap+1)), int(rng.integers(2, cap+1))
        base = np.full((h, w), bg, dtype=np.int8)
        palette = colors[:int(rng.integers(1, 5))]
        # Randomly placed rectangles can touch or overwrite: the final maximal
        # runs, rather than the construction's paint operations, define pieces.
        for _ in range(int(rng.integers(2, 9))):
            y0 = int(rng.integers(h))
            x0 = int(rng.integers(w))
            y1 = int(rng.integers(y0+1, h+1))
            x1 = int(rng.integers(x0+1, w+1))
            base[y0:y1, x0:x1] = int(rng.choice(palette))
        if large:
            base[0, 0] = colors[0]
            base[-1, -1] = colors[0]
    if rng.random() < .5:
        base = base[::-1]
    if rng.random() < .5:
        base = base[:, ::-1]
    base = _trim(base, bg)
    rects = _rectangles(base, bg)
    if not 2 <= len(rects) <= 8:
        return None
    groups = _group_pieces(rects, 1)
    if any(len(set(v)) > 1 for v in groups.values()):
        return None
    raw = np.repeat(np.repeat(base, scale, 0), scale, 1)
    if max(raw.shape) > 14:
        return None
    return raw


def _blueprint(rng, rects, scale, bg, structural, force_rotation):
    limit = min(12, 30 // scale)
    sizes = [(p[3] // scale, p[4] // scale) for p in rects]
    for _ in range(70):
        bh, bw = int(rng.integers(4, limit+1)), int(rng.integers(4, limit+1))
        occupied = np.zeros((bh, bw), dtype=bool)
        placed = [None] * len(rects)
        order = sorted(range(len(rects)), key=lambda i: (-sizes[i][0]*sizes[i][1], float(rng.random())))
        for i in order:
            h, w = sizes[i]
            if (force_rotation and i == 0) or rng.random() < .5:
                h, w = w, h
            positions = [(r, c) for r in range(1, bh-h) for c in range(1, bw-w)]
            for j in rng.permutation(len(positions)):
                r, c = positions[int(j)]
                if not np.any(occupied[r-1:r+h+1, c-1:c+w+1]):
                    placed[i] = (r, r+h, c, c+w)
                    occupied[r:r+h, c:c+w] = True
                    break
            if placed[i] is None:
                break
        else:
            board = np.full((bh, bw), structural, dtype=np.int8)
            logical = board.copy()
            for p, (a, b, c, d) in zip(rects, placed):
                board[a:b, c:d] = bg
                logical[a:b, c:d] = p[0]
            return board, logical
    return None


def _pack(rng, raw, board, bg):
    source = np.full((2*raw.shape[0]-1, 2*raw.shape[1]-1), bg, dtype=np.int8)
    source[::2, ::2] = raw
    sh, sw = source.shape
    bh, bw = board.shape
    options = []
    if sh + bh + 1 <= 28 and max(sw, bw) <= 28:
        options.append("vertical")
    if sw + bw + 1 <= 28 and max(sh, bh) <= 28:
        options.append("horizontal")
    if not options:
        return None
    direction = str(rng.choice(options))
    if direction == "vertical":
        height = int(rng.integers(sh+bh+3, 31))
        width = int(rng.integers(max(sw, bw)+2, 31))
        gap = int(rng.integers(1, height-sh-bh-1))
        top = int(rng.integers(1, height-sh-bh-gap))
        if rng.random() < .5:
            sy, by = top, top+sh+gap
        else:
            by, sy = top, top+bh+gap
        sx = int(rng.integers(1, width-sw))
        bx = int(rng.integers(1, width-bw))
    else:
        height = int(rng.integers(max(sh, bh)+2, 31))
        width = int(rng.integers(sw+bw+3, 31))
        gap = int(rng.integers(1, width-sw-bw-1))
        left = int(rng.integers(1, width-sw-bw-gap))
        if rng.random() < .5:
            sx, bx = left, left+sw+gap
        else:
            bx, sx = left, left+bw+gap
        sy = int(rng.integers(1, height-sh))
        by = int(rng.integers(1, height-bh))
    canvas = np.full((height, width), bg, dtype=np.int8)
    canvas[sy:sy+sh, sx:sx+sw] = source
    canvas[by:by+bh, bx:bx+bw] = board
    return canvas


def _make_pair(rng, scale, role="random", large=False):
    for _ in range(450):
        palette = list(map(int, rng.permutation(10)))
        bg, structural, colors = palette[0], palette[1], palette[2:]
        raw = _source(rng, bg, colors, role, scale, large)
        if raw is None:
            continue
        rects = _rectangles(raw, bg)
        blueprint = _blueprint(rng, rects, scale, bg, structural, role == "area")
        if blueprint is None:
            continue
        board, logical = blueprint
        canvas = _pack(rng, raw, board, bg)
        if canvas is None:
            continue
        expected = np.repeat(np.repeat(logical, scale, 0), scale, 1)
        result, info = _recover(canvas)
        if not np.array_equal(result, expected):
            raise AssertionError("construction disagrees with input-only solve")
        if role == "partial" and not info["unequal"]:
            continue
        if role == "area" and not (info["equal_area_distinct_colors"] and info["rotation"] and info["positional_wrong"]):
            continue
        return {"input": canvas, "output": expected}
    raise RuntimeError("bounded pair construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))
    if mode == 0:
        train_scales = [2] * nt
        test_scales = [int(rng.choice((3, 4))) for _ in range(ns)]
    else:
        train_scales = [2, 3] + [int(rng.choice((2, 3))) for _ in range(nt-2)]
        test_scales = [4 if mode == 1 else 2] * ns
    roles = ["partial", "area"] + ["random"] * (nt-2)
    train_specs = list(zip(train_scales, roles))
    rng.shuffle(train_specs)
    seen, puzzle = set(), {"train": [], "test": []}
    specs = [("train", k, role, False) for k, role in train_specs]
    specs += [("test", k, "random", mode == 2) for k in test_scales]
    for split, scale, role, large in specs:
        for _ in range(20):
            pair = _make_pair(rng, scale, role, large)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("could not make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, input-only reconstruction, exact cover and teaching cases."""
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train/test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    nt = _count(len(puzzle["train"]), "num_train", 2, 10)
    ns = _count(len(puzzle["test"]), "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("pair must have exactly input/output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            if _key(inp) in seen:
                raise ValueError("duplicate inputs")
            seen.add(_key(inp))
            expected, info = _recover(inp)
            if not np.array_equal(out, expected):
                raise ValueError("output disagrees with input-only rule")
            if not 2 <= info["piece_count"] <= 8:
                raise ValueError("generated inventory must have 2..8 pieces")
            if not 2 <= info["scale"] <= 4:
                raise ValueError("generated scale must be 2..4")
            # Cell conservation plus dimensions is stronger than comparing solve
            # twice: every sparse payload cell belongs to exactly one rectangle.
            raw, bg, k = info["coarse"], info["background"], info["scale"]
            occupancy = np.zeros(raw.shape, dtype=int)
            for color, r, c, h, w in info["rectangles"]:
                assert np.all(raw[r:r+h, c:c+w] == color)
                occupancy[r:r+h, c:c+w] += 1
            assert np.array_equal(occupancy, (raw != bg).astype(int))
            assert np.count_nonzero(raw != bg) == sum((b-a)*(d-c) for a, b, c, d in info["holes"]) * k*k
            # Omitting A's span merge must fail the one-piece-per-cavity cover.
            assert len(_rectangles(raw, bg, False)) > len(info["holes"])
            structural_mask = np.repeat(np.repeat(info["board"] == info["structural"], k, 0), k, 1)
            assert np.all(out[structural_mask] == info["structural"])
            assert not np.any(out == bg)
            infos[split].append(info)
    train = infos["train"]
    if not any(i["unequal"] and i["exact"] for i in train):
        raise ValueError("training must distinguish exact spans from partial overlaps")
    if not any(i["equal_area_distinct_colors"] and i["rotation"] and i["positional_wrong"] for i in train):
        raise ValueError("training must distinguish shape fit from area/order shortcuts")
    train_scales = {i["scale"] for i in train}
    train_shapes = {i["logical_source_shape"] for i in train}
    for i in infos["test"]:
        if i["scale"] in train_scales and i["logical_source_shape"] in train_shapes:
            raise ValueError("test must hold out scale or source geometry")
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, meaningful counterfactuals, no files or unbounded retention."""
    digests, input_shapes, board_shapes, source_shapes, inventories = set(), set(), set(), set(), set()
    scales, backgrounds, counts, areas = set(), set(), set(), set()
    directions, palette_zero = set(), False
    pair_count = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                out, info = _recover(pair["input"].copy())
                assert np.array_equal(out, pair["output"])
                raw, bg, k = info["coarse"], info["background"], info["scale"]
                input_shapes.add(pair["input"].shape)
                board_shapes.add(info["board"].shape)
                source_shapes.add(info["logical_source_shape"])
                inventories.add(tuple(sorted((p[3]//k, p[4]//k) for p in info["rectangles"])))
                scales.add(k)
                backgrounds.add(bg)
                counts.add(info["piece_count"])
                areas.update(p[3]*p[4]//(k*k) for p in info["rectangles"])
                palette_zero |= bg != 0 and np.any(raw == 0)
                br0, br1, bc0, bc1 = info["board_bounds"]
                for r, c in zip(*np.nonzero(pair["input"] != bg)):
                    if br0 <= r < br1 and bc0 <= c < bc1:
                        continue
                    if r < br0:
                        directions.add("above")
                    if r >= br1:
                        directions.add("below")
                    if c < bc0:
                        directions.add("left")
                    if c >= bc1:
                        directions.add("right")
                if info["unequal"]:
                    # A component interpretation must fuse at least two pieces.
                    component_count = sum(len(_components(raw == color))
                                          for color in np.unique(raw) if color != bg)
                    assert component_count < len(info["rectangles"])
                if info["equal_area_distinct_colors"] and info["rotation"]:
                    source_dims = Counter((p[3]//k, p[4]//k) for p in info["rectangles"])
                    hole_dims = Counter((b-a, d-c) for a, b, c, d in info["holes"])
                    assert source_dims != hole_dims, "quarter-turn rotation must matter"
    assert len(digests) == 42 and pair_count == 210
    assert scales == {2, 3, 4} and len(backgrounds) >= 8 and palette_zero
    assert len(input_shapes) >= 30 and len(board_shapes) >= 25
    assert len(source_shapes) >= 15 and len(inventories) >= 50
    assert len(counts) >= 5 and max(areas) >= 8
    assert directions == {"above", "below", "left", "right"}
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    # Every legal explicit count combination, with one live puzzle at a time.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(701, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Explicit ablations: exact-span decoding and quarter-turn dimension fit.
    example = np.array([[4, 4, 0], [4, 4, 0], [0, 4, 4]], dtype=int)
    assert [(p[3], p[4]) for p in _rectangles(example, 0)] == [(2, 2), (1, 2)]
    assert len(_components(example == 4)) == 1
    assert len(_rectangles(example, 0, False)) == 3
    # Congruent rectangles of different colors are ambiguous. The same geometry
    # becomes unambiguous with one color; corrupting one site then breaks scale.
    base = np.array([[3, 3, 0, 4, 4], [3, 3, 0, 4, 4]], dtype=np.int8)
    raw = np.repeat(np.repeat(base, 2, 0), 2, 1)
    board = np.full((5, 9), 2, dtype=np.int8)
    board[1:3, 1:3] = board[1:3, 5:7] = 0
    diagnostic = _pack(np.random.default_rng(17), raw, board, 0)
    try:
        solve(diagnostic)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous colored cover accepted")
    diagnostic[diagnostic == 4] = 3
    expected, info = _recover(diagnostic)
    assert expected.shape == (10, 18)
    sy, _, sx, _ = info["source_bounds"]
    diagnostic[sy, sx] = 0
    try:
        solve(diagnostic)
    except ValueError:
        pass
    else:
        raise AssertionError("broken inventory scale accepted")
    # Wrong labels must fail even when bounds and colors are valid.
    p = generate_puzzle(11, 2, 1)
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(digests),
            "input_shapes": len(input_shapes), "blueprint_shapes": len(board_shapes),
            "source_shapes": len(source_shapes), "inventories": len(inventories),
            "scales": sorted(scales), "piece_counts": sorted(counts),
            "max_piece_area": max(areas), "independent_solver": False}
