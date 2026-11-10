"""Registered shadow stencils: block-scale matching, compass docking, rays.

No generation state is used by solve. Input recovery searches the common
scale and all eight rigid poses, and deduplicates complete rendered answers.
"""
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N E S W


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grid must be a 1..30 integer rectangle with colors 0..9")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    unseen = set(zip(*np.nonzero(mask)))
    result = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        todo, cells = [root], {root}
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    todo.append(p)
        result.append(cells)
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _normal(cells):
    r0, _, c0, _ = _bounds(cells)
    return frozenset((int(r-r0), int(c-c0)) for r, c in cells)


def _transform(tile, t):
    return np.rot90(np.fliplr(tile) if t >= 4 else tile, t % 4).copy()


def _shape_class(cells):
    r0, r1, c0, c1 = _bounds(cells)
    mask = np.zeros((r1-r0, c1-c0), dtype=np.uint8)
    for r, c in cells:
        mask[r-r0, c-c0] = 1
    return min(tuple(sorted(zip(*np.nonzero(_transform(mask, t))))) for t in range(8))


def _tabs(tile, bg, body, reg, sides):
    core = set(zip(*np.nonzero((tile == body) | (tile == reg))))
    if not core or len(_components((tile == body) | (tile == reg))) != 1:
        raise ValueError("stencil core must be connected")
    r0, r1, c0, c1 = _bounds(core)
    tabs = []
    for r, c in zip(*np.nonzero((tile != bg) & (tile != body) & (tile != reg))):
        color = int(tile[r, c])
        if color not in sides:
            raise ValueError("unknown stencil color")
        ds = [d for d, ok in enumerate((r == r0-1 and c0 <= c < c1,
                                        c == c1 and r0 <= r < r1,
                                        r == r1 and c0 <= c < c1,
                                        c == c0-1 and r0 <= r < r1)) if ok]
        if len(ds) != 1:
            raise ValueError("tab must project one cell beyond the core envelope")
        d = ds[0]
        dr, dc = DIRS[d]
        if (r-dr, c-dc) not in core:
            raise ValueError("tab must touch its core")
        tabs.append(((int(r), int(c)), d, color))
    if not tabs or not np.any(tile == reg) or not np.any(tile == body):
        raise ValueError("stencil needs body, registration and light tabs")
    # A color names one compass direction, even when its tab is a patch.
    if any(len({d for _, d, v in tabs if v == color}) > 1 for color in sides):
        raise ValueError("one light identity cannot occupy two sides")
    return core, tabs


def _shadow(tile, bg, body, reg, sides, transparent=False, through_gaps=False):
    """Return core colors after inward rays; labels are omitted."""
    core, tabs = _tabs(tile, bg, body, reg, sides)
    r0, r1, c0, c1 = _bounds(core)
    painted = {p: int(tile[p]) for p in core}
    lit, blocked_beyond = set(), 0
    for (r, c), d, color in tabs:
        dr, dc = DIRS[d]
        p = r-dr, c-dc
        while r0 <= p[0] < r1 and c0 <= p[1] < c1:
            if p not in core:
                if not through_gaps:
                    break
                p = p[0]-dr, p[1]-dc
                continue
            if tile[p] == reg:
                q = p[0]-dr, p[1]-dc
                if q in core and tile[q] == body:
                    blocked_beyond += 1
                if not transparent:
                    break
            else:
                lit.add(p)
            p = p[0]-dr, p[1]-dc
    for p in lit:
        painted[p] = bg
    return painted, lit, blocked_beyond


def _decode_piece(a, cells, k, bg):
    r0, r1, c0, c1 = _bounds(cells)
    h, w = r1-r0, c1-c0
    if h % k or w % k:
        return None
    crop = a[r0:r1, c0:c1]
    tile = crop[::k, ::k].copy()
    if not np.array_equal(crop, np.repeat(np.repeat(tile, k, 0), k, 1)):
        return None
    if np.count_nonzero(tile != bg) * k*k != len(cells):
        return None
    return tile


def _boards(a, bg):
    """The corner color is structural; four open side runs form a compass."""
    result = []
    for component in _components(a != bg):
        r0, r1, c0, c1 = _bounds(component)
        b = a[r0:r1, c0:c1]
        if min(b.shape) < 5:
            continue
        structure = int(b[0, 0])
        if structure == bg or not (b[0, -1] == b[-1, 0] == b[-1, -1] == structure):
            continue
        edges = (b[0, 1:-1], b[1:-1, -1], b[-1, 1:-1], b[1:-1, 0])
        sides = tuple(int(edge[0]) for edge in edges)
        if len(set(sides) | {bg, structure}) != 6:
            continue
        if any(not np.all(edge == color) for edge, color in zip(edges, sides)):
            continue
        interior_colors = set(map(int, np.unique(b[1:-1, 1:-1]))) - {bg, structure}
        if len(interior_colors) != 1:
            continue
        reg = next(iter(interior_colors))
        if reg in sides:
            continue
        holemask = np.zeros(b.shape, dtype=bool)
        holemask[1:-1, 1:-1] = b[1:-1, 1:-1] != structure
        holes = _components(holemask)
        if not 1 <= len(holes) <= 4:
            continue
        if any(not any(b[p] == reg for p in hole) or not any(b[p] == bg for p in hole)
               for hole in holes):
            continue
        result.append((b.copy(), (r0, r1, c0, c1), structure, reg, sides, holes))
    return result


def _placements(tile, board, holes, bg, body, reg, sides, compass=True,
                transparent=False, rotations_only=False):
    result, seen = [], set()
    for t in range(4 if rotations_only else 8):
        pose = _transform(tile, t)
        core, tabs = _tabs(pose, bg, body, reg, sides)
        if compass and any(sides[d] != color for _, d, color in tabs):
            continue
        core_pattern = _normal(core)
        pr0, _, pc0, _ = _bounds(core)
        painted, lit, occluded = _shadow(pose, bg, body, reg, sides, transparent)
        for j, hole in enumerate(holes):
            if core_pattern != _normal(hole):
                continue
            hr0, _, hc0, _ = _bounds(hole)
            delta = hr0-pr0, hc0-pc0
            shifted_marks = {(r+delta[0], c+delta[1]) for r, c in core if pose[r, c] == reg}
            target_marks = {p for p in hole if board[p] == reg}
            if shifted_marks != target_marks:
                continue
            cells = {(r+delta[0], c+delta[1]): v for (r, c), v in painted.items()}
            signature = j, tuple(sorted(cells.items()))
            if signature in seen:
                continue
            seen.add(signature)
            result.append({"hole": j, "cells": cells, "t": t,
                           "lit": len(lit), "occluded": occluded,
                           "dark": sum(v == body for v in painted.values()),
                           "tab_sides": len({d for _, d, _ in tabs}),
                           "tab_count": len(tabs)})
    return result


def _answers(info, compass=True, transparent=False, rotations_only=False, limit=2):
    choices = [_placements(tile, info["board"], info["holes"], info["bg"], info["body"],
                           info["reg"], info["sides"], compass, transparent, rotations_only)
               for tile in info["pieces"]]
    order = sorted(range(len(choices)), key=lambda j: len(choices[j]))
    solutions = {}

    def visit(depth, used, cells, selected):
        if len(solutions) >= limit:
            return
        if depth == len(order):
            logical = info["board"].copy()
            for p, v in cells.items():
                logical[p] = v
            signature = _key(logical)
            solutions.setdefault(signature, (logical, selected))
            return
        for candidate in choices[order[depth]]:
            if candidate["hole"] not in used:
                visit(depth+1, used | {candidate["hole"]},
                      {**cells, **candidate["cells"]}, selected + [candidate])
    visit(0, set(), {}, [])
    return list(solutions.values())


def _recover(a):
    bg = int(np.argmax(np.bincount(a.ravel(), minlength=10)))
    if np.count_nonzero(a == bg) <= a.size//2:
        raise ValueError("canvas background must be a strict majority")
    accepted = []
    for board, bounds, structure, reg, sides, holes in _boards(a, bg):
        r0, r1, c0, c1 = bounds
        outside = np.ones(a.shape, dtype=bool)
        outside[r0:r1, c0:c1] = False
        moat = np.zeros(a.shape, dtype=bool)
        moat[max(0, r0-1):min(a.shape[0], r1+1), max(0, c0-1):min(a.shape[1], c1+1)] = True
        if np.any((a != bg) & outside & moat):
            continue
        objects = _components((a != bg) & outside)
        large = [cells for cells in objects if len(cells) >= 4]
        noise = [cells for cells in objects if len(cells) < 4]
        if len(large) != len(holes):
            continue
        extra = set(int(a[p]) for cells in large for p in cells) - {bg, reg, structure, *sides}
        if len(extra) != 1:
            continue
        body = next(iter(extra))
        for k in range(2, min(4, 30//max(board.shape))+1):
            tiles = [_decode_piece(a, cells, k, bg) for cells in large]
            if any(tile is None for tile in tiles):
                continue
            try:
                for tile in tiles:
                    if structure in tile:
                        raise ValueError("structure is not a stencil color")
                    _tabs(tile, bg, body, reg, sides)
            except ValueError:
                continue
            info = {"board": board, "bounds": bounds, "structure": structure,
                    "bg": bg, "body": body, "reg": reg, "sides": sides,
                    "holes": holes, "pieces": tiles, "scale": k,
                    "noise": sum(map(len, noise)), "noise_sizes": tuple(map(len, noise))}
            answers = _answers(info)
            if len(answers) > 1:
                raise ValueError("ambiguous shadow assignment")
            if answers:
                logical, selected = answers[0]
                info["selected"] = selected
                accepted.append((np.repeat(np.repeat(logical, k, 0), k, 1), info))
    if len(accepted) != 1:
        raise ValueError("input needs one blueprint, block scale and shadow answer")
    return accepted[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover scale, board, stencil assignment and light rays from input only."""
    return _recover(_grid(input_grid))[0]


def _grow(rng, area, span=3):
    cells = {(int(rng.integers(span)), int(rng.integers(span)))}
    while len(cells) < area:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in DIRS
                           if 0 <= r+dr < span and 0 <= c+dc < span} - cells)
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return set(_normal(cells))


def _stencil(rng, shape, palette, witness=False, force_two=False, single_mark=False):
    bg, structure, body, reg, *sides = palette[:8]
    h = max(r for r, c in shape)+1
    w = max(c for r, c in shape)+1
    tile = np.full((h+2, w+2), bg, dtype=int)
    for r, c in shape:
        tile[r+1, c+1] = body
    if witness:
        marks = {(h//2, w//2)}
    else:
        points = sorted(shape)
        count = 1 if single_mark else int(rng.integers(1, min(3, len(shape)-2)+1))
        marks = {points[int(j)] for j in rng.choice(len(points), count, replace=False)}
    for r, c in marks:
        tile[r+1, c+1] = reg
    if witness:
        labels = {0: [w//2], 3: [0]}
    else:
        # Two adjacent named sides always certify handedness. Additional
        # random tabs add beams without introducing a color mixing rule.
        first = int(rng.integers(4))
        selected = {first, (first+1) % 4}
        if not force_two:
            for d in range(4):
                if rng.random() < .32:
                    selected.add(d)
        labels = {}
        for d in sorted(selected):
            positions = sorted({c if d in (0, 2) else r for r, c in shape
                                if (r == 0 if d == 0 else c == w-1 if d == 1
                                    else r == h-1 if d == 2 else c == 0)})
            size = int(rng.integers(1, min(2, len(positions))+1))
            labels[d] = [positions[int(j)] for j in rng.choice(len(positions), size, replace=False)]
    for d, positions in labels.items():
        for x in positions:
            p = ((0, x+1), (x+1, w+1), (h+1, x+1), (x+1, 0))[d]
            tile[p] = sides[d]
    colored = zip(*np.nonzero(tile != bg))
    r0, r1, c0, c1 = _bounds(set(colored))
    return tile[r0:r1, c0:c1].copy()


def _place_holes(rng, shapes, h, w):
    occupied, placements = set(), []
    # Independent shuffled positions; exact silhouettes, not rectangular slots.
    for shape in shapes:
        sh = max(r for r, c in shape)+1
        sw = max(c for r, c in shape)+1
        origins = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)]
        rng.shuffle(origins)
        for r0, c0 in origins:
            cells = {(r+r0, c+c0) for r, c in shape}
            halo = {(r+dr, c+dc) for r, c in cells for dr, dc in ((0, 0), *DIRS)}
            if not halo & occupied:
                placements.append((r0, c0))
                occupied.update(cells)
                break
        else:
            return None
    return placements


def _gap_stencil(rng, palette):
    """A six-cell bent fork explicitly shows light stopping at an interior gap."""
    bg, _, body, reg, *sides = palette[:8]
    tile = np.full((5, 5), bg, dtype=int)
    for r, c in ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)):
        tile[r+1, c+1] = body
    tile[2, 2] = reg
    tile[1, 0], tile[4, 2] = sides[3], sides[2]
    tile = _transform(tile, int(rng.integers(8)))
    _, tabs = _tabs(tile, bg, body, reg, sides)
    # The fixture's physical pose varies; relabel its lamps to the locally
    # visible compass before it becomes a solved destination stencil.
    for p, d, _ in tabs:
        tile[p] = sides[d]
    r0, r1, c0, c1 = _bounds(set(zip(*np.nonzero(tile != bg))))
    return tile[r0:r1, c0:c1].copy()


def _free(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return []
    s = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]
    return np.argwhere(counts == 0)


def _pack(rng, board, pieces, k, palette):
    bg = palette[0]
    rasters = [board] + [np.repeat(np.repeat(p, k, 0), k, 1) for p in pieces]
    for attempt in range(35):
        h, w = (int(rng.integers(20, 31)), int(rng.integers(20, 31))) if attempt < 15 else (30, 30)
        canvas = np.full((h, w), bg, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        # Board can be anywhere. Largest-first ordering aids packing without
        # imposing an object rank or shelf in the input.
        order = sorted(range(len(rasters)), key=lambda j: (-rasters[j].size, float(rng.random())))
        for j in order:
            raster = rasters[j]
            sh, sw = raster.shape
            origins = _free(blocked, sh, sw)
            if not len(origins):
                break
            r0, c0 = map(int, origins[int(rng.integers(len(origins)))])
            canvas[r0:r0+sh, c0:c0+sw] = raster
            blocked[max(0, r0-1):min(h, r0+sh+1), max(0, c0-1):min(w, c0+sw+1)] = True
        else:
            # Optional small clusters cannot be k-by-k stencil macrocells.
            for _ in range(int(rng.integers(0, 5))):
                n = int(rng.integers(1, 4))
                shape = [(0, 0)] if n == 1 else [(0, 0), (0, 1)] if n == 2 else [(0, 0), (0, 1), (1, 0)]
                mask = np.zeros((max(r for r, c in shape)+1, max(c for r, c in shape)+1), dtype=bool)
                for p in shape:
                    mask[p] = True
                mask = np.rot90(mask, int(rng.integers(4)))
                nh, nw = mask.shape
                free = _free(blocked, nh, nw)
                if not len(free):
                    continue
                r, c = map(int, free[int(rng.integers(len(free)))])
                color = int(rng.choice([palette[2], palette[8], palette[9]]))
                canvas[r:r+nh, c:c+nw][mask] = color
                blocked[max(0, r-1):min(h, r+nh+1), max(0, c-1):min(w, c+nw+1)] = True
            if np.count_nonzero(canvas == bg) > canvas.size//2:
                return canvas
    return None


def _make_pair(rng, palette, k, count, span, demonstration=None):
    bg, structure, body, reg, *sides = palette[:8]
    for attempt in range(150):
        if demonstration == "compass":
            shapes = [{(r, c) for r in range(3) for c in range(3)},
                      {(r, c) for r in range(2) for c in range(3)}]
        elif demonstration == "reflection":
            # Equal areas and one registration cell each defeat area/count matching.
            shapes = [{(r, c) for r in range(2) for c in range(3)},
                      {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)}]
        else:
            shapes = []
            for j in range(count):
                if shapes and rng.random() < .2:
                    shapes.append(set(shapes[int(rng.integers(len(shapes)))]))
                else:
                    area = int(rng.integers(4, min(11, span*span)+1))
                    shapes.append(_grow(rng, area, span))
            if span == 4 and not any(max(max(r for r, c in s), max(c for r, c in s)) == 3 for s in shapes):
                continue
        count = len(shapes)
        # Random destination silhouettes need no permanent motif dictionary.
        poses = []
        for shape in shapes:
            sh = max(r for r, c in shape)+1
            sw = max(c for r, c in shape)+1
            mask = np.zeros((sh, sw), dtype=int)
            for p in shape:
                mask[p] = 1
            pose = _transform(mask, int(rng.integers(8)))
            poses.append(set(zip(*np.nonzero(pose))))
        gap_tile = _gap_stencil(rng, palette) if demonstration == "reflection" else None
        if gap_tile is not None:
            core, _ = _tabs(gap_tile, bg, body, reg, sides)
            poses[1] = set(_normal(core))
        limit = 30//k
        lower = 6 if count == 1 else 7 if count == 2 else 9
        if lower > limit:
            continue
        h = int(rng.integers(lower, limit+1))
        w = int(rng.integers(lower, limit+1))
        origins = _place_holes(rng, poses, h, w)
        if origins is None:
            continue
        board = np.full((h, w), structure, dtype=int)
        board[0, 1:-1], board[1:-1, -1] = sides[0], sides[1]
        board[-1, 1:-1], board[1:-1, 0] = sides[2], sides[3]
        final, pieces, all_lit, all_dark, blockers = board.copy(), [], 0, 0, 0
        for j, (shape, origin) in enumerate(zip(poses, origins)):
            witness = demonstration == "compass" and j == 0
            tile = gap_tile if gap_tile is not None and j == 1 else _stencil(
                rng, shape, palette, witness,
                force_two=(k >= 3 or demonstration is not None),
                single_mark=(demonstration == "reflection"))
            core, tabs = _tabs(tile, bg, body, reg, sides)
            cr0, _, cc0, _ = _bounds(core)
            shadow, lit, occluded = _shadow(tile, bg, body, reg, sides)
            all_lit += len(lit)
            all_dark += sum(v == body for v in shadow.values())
            blockers += occluded
            for (r, c), value in shadow.items():
                p = r-cr0+origin[0], c-cc0+origin[1]
                board[p] = reg if tile[r, c] == reg else bg
                final[p] = value
            t = int(rng.integers(8))
            if j == 0 and demonstration == "compass":
                t = int(rng.integers(1, 4))
            elif j == 0 and demonstration == "reflection":
                t = int(rng.integers(4, 8))
            pieces.append(_transform(tile, t))
        if not all_lit or not all_dark or (demonstration == "compass" and not blockers):
            continue
        inp = _pack(rng, board, pieces, k, palette)
        if inp is None:
            continue
        expected = np.repeat(np.repeat(final, k, 0), k, 1)
        try:
            predicted, info = _recover(inp)
            if not np.array_equal(predicted, expected):
                continue
            if demonstration == "compass":
                if len(_answers(info, compass=False)) < 2:
                    continue
                transparent = _answers(info, transparent=True)
                if not transparent or np.array_equal(transparent[0][0], final):
                    continue
            if demonstration == "reflection" and _answers(info, rotations_only=True):
                continue
        except ValueError:
            continue
        return {"input": inp, "output": expected}
    raise RuntimeError("bounded shadow-stencil construction exhausted")


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    curriculum = int(rng.integers(3))  # novel scale, higher count, larger shapes
    puzzle, used = {"train": [], "test": []}, set()
    previous_bg = None
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            palette = list(map(int, rng.permutation(10)))
            if palette[0] == previous_bg:
                palette[0], palette[8] = palette[8], palette[0]
            previous_bg = palette[0]
            demonstration = ("compass" if index == 0 else "reflection" if index == 1 else None) if split == "train" else None
            if split == "train":
                k = 2 if index == 0 else 3 if index == 1 else int(rng.choice([2, 3]))
                count, span = (2, 3) if demonstration else (int(rng.integers(1, 3)), 3)
            elif curriculum == 0:
                k, count, span = 4, int(rng.integers(1, 3)), 3
                if count == 2:
                    span = 2
            elif curriculum == 1:
                k, count, span = 2, int(rng.integers(3, 5)), 3
            else:
                k, count, span = 2, int(rng.integers(1, 4)), 4
            for _ in range(20):
                pair = _make_pair(rng, palette, k, count, span, demonstration)
                if _key(pair["input"]) not in used:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(a):
    expected, info = _recover(a)
    selected = info["selected"]
    logical = expected[::info["scale"], ::info["scale"]]
    info["compass_needed"] = len(_answers(info, compass=False)) > 1
    transparent = _answers(info, transparent=True)
    info["occlusion_needed"] = bool(transparent and not np.array_equal(transparent[0][0], logical))
    info["reflection_needed"] = not bool(_answers(info, rotations_only=True))
    info["rotation"] = any(0 < p["t"] < 4 for p in selected)
    info["gap_needed"] = False
    for tile in info["pieces"]:
        for t in range(8):
            pose = _transform(tile, t)
            _, tabs = _tabs(pose, info["bg"], info["body"], info["reg"], info["sides"])
            if any(info["sides"][d] != color for _, d, color in tabs):
                continue
            shadow = _shadow(pose, info["bg"], info["body"], info["reg"], info["sides"])[0]
            alternative = _shadow(pose, info["bg"], info["body"], info["reg"], info["sides"], through_gaps=True)[0]
            info["gap_needed"] |= shadow != alternative
    info["lit"] = sum(p["lit"] for p in selected)
    info["dark"] = sum(p["dark"] for p in selected)
    return expected, info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            expected, info = _measure(a)
            assert np.array_equal(expected, out), "incorrect shadow output"
            k, board = info["scale"], info["board"]
            assert out.shape == (board.shape[0]*k, board.shape[1]*k)
            logical = out[::k, ::k]
            structure = np.ones(board.shape, dtype=bool)
            for hole in info["holes"]:
                for p in hole:
                    structure[p] = False
            assert np.array_equal(logical[structure], board[structure]), "structure changed"
            assert np.array_equal(logical == info["reg"], board == info["reg"]), "registration changed"
            assert info["lit"] > 0 and info["dark"] > 0, "light and shadow both required"
            assert all(size <= 3 for size in info["noise_sizes"])
            assert 1 <= len(info["holes"]) <= 4 and 2 <= k <= 4
            if split == "train":
                train.append(info)
    assert any(i["compass_needed"] for i in train), "missing compass witness"
    assert any(i["occlusion_needed"] for i in train), "missing blocker witness"
    assert any(i["gap_needed"] for i in train), "missing ray-gap witness"
    assert any(i["reflection_needed"] for i in train), "missing reflection witness"
    assert any(i["rotation"] for i in train), "missing rotation witness"
    assert len({i["scale"] for i in train}) >= 2
    assert len({i["bg"] for i in train}) >= 2
    assert any(len({len(h) for h in i["holes"]}) < len(i["holes"])
               and len({_shape_class(h) for h in i["holes"]}) > 1 for i in train), "missing equal-area shape witness"


def self_test():
    """Bounded 42-seed suite, family ablations and malformed-input checks."""
    records, keys = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])
        assert signature not in keys
        keys.add(signature)
        for split in ("train", "test"):
            for p in puzzle[split]:
                expected, info = _measure(p["input"])
                assert np.array_equal(solve(p["input"].tolist()), p["output"])
                records.append((split, p["input"].shape, info))
    assert len({shape for _, shape, _ in records}) >= 30
    assert {i["scale"] for _, _, i in records} == {2, 3, 4}
    assert {len(i["holes"]) for _, _, i in records} == {1, 2, 3, 4}
    assert len({_shape_class(h) for _, _, i in records for h in i["holes"]}) >= 25
    assert len({i["bg"] for _, _, i in records}) == 10
    assert {p["tab_sides"] for _, _, i in records for p in i["selected"]} >= {2, 3, 4}
    assert any(len(set(i["noise_sizes"])) > 1 for _, _, i in records)
    assert any(i["noise"] == 0 for _, _, i in records)
    assert any(len({_shape_class(h) for h in i["holes"]}) < len(i["holes"]) for _, _, i in records)
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            p = generate_puzzle(901+ntrain*10+ntest, ntrain, ntest)
            assert len(p["train"]) == ntrain and len(p["test"]) == ntest
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    a, b = generate_puzzle(1234), generate_puzzle(1234)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # Palette equivariance and global spatial transforms must not change the rule.
    p = a["train"][0]
    permutation = np.array([7, 4, 8, 1, 9, 0, 6, 3, 5, 2])
    assert np.array_equal(solve(permutation[p["input"]]), permutation[p["output"]])
    for t in (1, 4, 7):
        assert np.array_equal(solve(_transform(p["input"], t)), _transform(p["output"], t))
    # Changing a tab's identity while retaining its position corrupts the compass.
    inp = p["input"].copy()
    _, info = _recover(inp)
    r0, r1, c0, c1 = info["bounds"]
    outside = np.ones(inp.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    candidates = np.argwhere(outside & (inp == info["sides"][0]))
    if len(candidates):
        inp[tuple(candidates[0])] = info["sides"][1]
        try:
            solve(inp)
        except ValueError:
            pass
        else:
            raise AssertionError("broken macrocell accepted")
    wrong = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    wrong["test"][0]["output"][0, 0] = (int(wrong["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return {"seeds": 42, "pairs": len(records), "shape_classes": len({_shape_class(h) for _, _, i in records for h in i["holes"]}),
            "input_shapes": len({shape for _, shape, _ in records})}
