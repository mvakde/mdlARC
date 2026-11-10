"""Route-constrained macrocell jigsaw. Standard library and NumPy only.

No file IO, source fixtures, persistent caches or generation state in solve.
"""
from __future__ import annotations

import hashlib
import math
from itertools import combinations

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _N4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _normalize(cells):
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _orientations(a, reflections=True):
    seen, result = set(), []
    for flip in range(2 if reflections else 1):
        for q in range(4):
            b = np.rot90(a[:, ::-1] if flip else a, q).copy()
            key = b.shape, b.tobytes()
            if key not in seen:
                seen.add(key)
                result.append(b)
    return result


def _mask(cells):
    cells = _normalize(cells)
    a = np.zeros((max(r for r, c in cells) + 1,
                  max(c for r, c in cells) + 1), dtype=np.int8)
    for r, c in cells:
        a[r, c] = 1
    return a


def _class(cells):
    return min((b.shape, b.tobytes()) for b in _orientations(_mask(cells)))


def _spine(cells):
    """The assembled comb has a unique long horizontal row and short teeth."""
    rows = {}
    for r, c in cells:
        rows.setdefault(r, []).append(c)
    length = max(map(len, rows.values()))
    candidates = [r for r, cs in rows.items() if len(cs) == length]
    if length < 4 or len(candidates) != 1:
        raise ValueError("not a horizontal comb")
    f = candidates[0]
    lo, hi = min(rows[f]), max(rows[f])
    teeth = [(r, c) for r, c in cells if r != f]
    if (hi - lo + 1 != length or not 1 <= len(teeth) <= 3
            or any(abs(r - f) != 1 or not lo <= c <= hi for r, c in teeth)):
        raise ValueError("invalid comb teeth")
    return f, lo, hi


def _command(tile, bg, bar, marker):
    cells = tuple(map(tuple, np.argwhere(tile != bg)))
    f, lo, hi = _spine(cells)
    if any(int(tile[r, c]) != bar for r, c in cells if r != f):
        raise ValueError("teeth must have the permanent color")
    marks = tuple(map(int, np.flatnonzero(tile[f] == marker)))
    n = len(marks)
    if not 1 <= n or hi - lo + 1 < 2 * n:
        raise ValueError("invalid unary count")
    low = marks == tuple(lo + 2 * j for j in range(n))
    high = marks == tuple(hi - 2 * j for j in reversed(range(n)))
    if low == high:
        raise ValueError("markers must start at exactly one endpoint")
    return n, 1 if low else -1


def _parse(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= a.size // 2:
        raise ValueError("canvas background must occupy a strict majority")
    candidates = []
    for floor in np.flatnonzero(counts):
        floor = int(floor)
        if floor == bg:
            continue
        points = np.argwhere(a == floor)
        r0, r1, c0, c1 = _bounds(points)
        b = a[r0:r1, c0:c1]
        if min(b.shape) < 6:
            continue
        border = np.zeros(b.shape, dtype=bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        exceptions = np.argwhere(border & (b != floor))
        if len(exceptions) != 1:
            continue
        sr, sc = map(int, exceptions[0])
        route = int(b[sr, sc])
        if (route in (bg, floor) or counts[route] != 1
                or sum((sr == 0, sc == 0, sr == b.shape[0] - 1,
                        sc == b.shape[1] - 1)) != 1
                or not np.all((b == floor) | (b == bg) | (b == route))):
            continue
        candidates.append((floor, route, (r0, r1, c0, c1), b.copy()))
    if len(candidates) != 1:
        raise ValueError("expected one miniature board")
    floor, route, bounds, physical_board = candidates[0]
    r0, r1, c0, c1 = bounds
    sr, sc = map(int, np.argwhere(physical_board == route)[0])
    # np.rot90(canonical_down, q) gives down, right, up, left respectively.
    q = 0 if sr == 0 else 1 if sc == 0 else 2 if sr == physical_board.shape[0] - 1 else 3
    board = np.rot90(physical_board, -q).copy()
    initial = int(np.argwhere(board == route)[0, 1])
    holes = _components(board == bg)
    if not 1 <= len(holes) <= 3:
        raise ValueError("expected one to three sockets")
    slots = []
    for cells in holes:
        if any(r in (0, board.shape[0] - 1) or c in (0, board.shape[1] - 1)
               for r, c in cells):
            raise ValueError("sockets must be enclosed")
        f, lo, hi = _spine(cells)
        slots.append({"cells": cells, "row": f, "lo": lo, "hi": hi,
                      "mask": _mask(cells), "class": _class(cells)})
    slots.sort(key=lambda s: s["row"])
    if len({s["row"] for s in slots}) != len(slots):
        raise ValueError("sockets must be sequential")
    outside = np.ones(a.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    moat = np.zeros(a.shape, dtype=bool)
    moat[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
    if np.any(outside & moat & (a != bg)):
        raise ValueError("board needs a blank moat")
    pieces = _components(outside & (a != bg))
    if len(pieces) != len(slots):
        raise ValueError("one loose tile per socket is required")
    unit_area = sum(len(s["cells"]) for s in slots)
    total_area = sum(map(len, pieces))
    ratio, remainder = divmod(total_area, unit_area)
    k = math.isqrt(ratio)
    if remainder or k < 2 or k * k != ratio or max(board.shape) * k > 30:
        raise ValueError("no integral common enlargement scale")
    tiles = []
    for cells in pieces:
        pr0, pr1, pc0, pc1 = _bounds(cells)
        ph, pw = pr1-pr0, pc1-pc0
        if ph % k or pw % k:
            raise ValueError("tile bounds do not fit the common scale")
        crop = a[pr0:pr1, pc0:pc1]
        blocks = crop.reshape(ph//k, k, pw//k, k).transpose(0, 2, 1, 3)
        tile = blocks[:, :, 0, 0].copy()
        if not np.all(blocks == tile[:, :, None, None]):
            raise ValueError("broken macrocell")
        if np.count_nonzero(tile != bg) * k * k != len(cells):
            raise ValueError("loose objects have overlapping bounding boxes")
        tiles.append(tile)
    colors = set(map(int, a[outside & (a != bg)]))
    if len(colors) != 2 or colors & {bg, floor, route}:
        raise ValueError("loose tiles need two distinct local colors")
    roles = []
    for bar in sorted(colors):
        marker = next(c for c in colors if c != bar)
        options, rotation_options, ns, rotations, reflections = [], [], [], 0, 0
        for slot in slots:
            local, local_rotations = set(), set()
            for i, tile in enumerate(tiles):
                rot_fits, d4_fits = False, False
                for transformed in _orientations(tile):
                    if (transformed.shape != slot["mask"].shape
                            or not np.array_equal(transformed != bg, slot["mask"] != 0)):
                        continue
                    try:
                        n, side = _command(transformed, bg, bar, marker)
                    except ValueError:
                        continue
                    local.add((i, n, side))
                    d4_fits = True
                for transformed in _orientations(tile, False):
                    if (transformed.shape == slot["mask"].shape
                            and np.array_equal(transformed != bg, slot["mask"] != 0)):
                        rot_fits = True
                        try:
                            n, side = _command(transformed, bg, bar, marker)
                        except ValueError:
                            continue
                        local_rotations.add((i, n, side))
                if d4_fits and not rot_fits:
                    reflections += 1
                if d4_fits and (tile.shape != slot["mask"].shape
                               or not np.array_equal(tile != bg, slot["mask"] != 0)):
                    rotations += 1
            options.append(tuple(sorted(local)))
            rotation_options.append(tuple(sorted(local_rotations)))
        if all(options):
            for tile in tiles:
                ns.append(int(np.count_nonzero(tile == marker)))
            roles.append((bar, marker, options, rotation_options, tuple(ns), rotations, reflections))
    if len(roles) != 1:
        raise ValueError("permanent and marker roles are not uniquely readable")
    bar, marker, options, rotation_options, ns, rotations, reflections = roles[0]
    obstacle = board == bg
    return {"board": board, "slots": slots, "options": options,
            "rotation_options": rotation_options, "tile_counts": ns,
            "obstacle": obstacle, "initial": initial, "q": q, "scale": k,
            "palette": (bg, floor, route, bar, marker), "bounds": bounds,
            "input_shape": a.shape, "rotation_fits": rotations,
            "reflection_fits": reflections}


def _geometric_count(info, cap=2):
    """Count distinct command strings admitted by jigsaw matching alone."""
    sequences = set()
    def visit(index, used, prefix):
        if len(sequences) >= cap:
            return
        if index == len(info["slots"]):
            sequences.add(prefix)
            return
        for i, n, side in info["options"][index]:
            if i not in used:
                visit(index + 1, used | {i}, prefix + ((n, side),))
    visit(0, set(), ())
    return len(sequences)


def _routes(info, ignore_contours=False, constant_distance=None, wrong_endpoint=False,
            rotations_only=False):
    """Bounded assignment search, retaining at most two different route masks."""
    h, w = info["board"].shape
    slots, obstacles = info["slots"], info["obstacle"]
    options = info["rotation_options"] if rotations_only else info["options"]
    if ignore_contours:
        broad = tuple((i, n, side) for i, n in enumerate(info["tile_counts"])
                      for side in (-1, 1))
        options = [broad] * len(slots)
    solutions, seen = [], set()
    def visit(index, used, current, start, previous, path, commands):
        if len(solutions) >= 2:
            return
        if index == len(slots):
            final = path.copy()
            final[start:, current] = True
            if np.any(final & obstacles):
                return
            key = final.tobytes()
            if key not in seen:
                seen.add(key)
                solutions.append((final, commands))
            return
        slot = slots[index]
        f, lo, hi = slot["row"], slot["lo"], slot["hi"]
        for i, n, side in options[index]:
            if i in used:
                continue
            d = n + 1 if constant_distance is None else constant_distance
            sign = -side if wrong_endpoint else side
            turn = f - d
            target = hi + d if sign == 1 else lo - d
            if not (turn >= 1 and turn > previous and turn >= start
                    and lo <= current <= hi and 0 <= target < w):
                continue
            following = path.copy()
            following[start:turn + 1, current] = True
            following[turn, min(current, target):max(current, target) + 1] = True
            if np.any(following & obstacles):
                continue
            visit(index + 1, used | {i}, target, turn, f, following,
                  commands + ((i, n, side),))
    visit(0, set(), info["initial"], 0, -1, np.zeros((h, w), dtype=bool), ())
    return solutions


def _recover(input_grid):
    info = _parse(input_grid)
    solutions = _routes(info)
    if len(solutions) != 1:
        raise ValueError("assembly must have exactly one feasible route")
    path, commands = solutions[0]
    bg, floor, route, bar, marker = info["palette"]
    logical = info["board"].copy()
    logical[info["obstacle"]] = bar
    logical[path] = route
    physical = np.rot90(logical, info["q"])
    result = np.repeat(np.repeat(physical, info["scale"], 0), info["scale"], 1)
    info["path"], info["commands"] = path, commands
    info["reflection_needed"] = not bool(_routes(info, rotations_only=True))
    return result, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the queried input only; all conventions are visible geometric roles."""
    return _recover(input_grid)[0]


def _comb_vocabulary(length, tooth_count):
    # Procedural vocabulary, not a list of worked-example templates.
    groups = {}
    sites = [(r, c) for r in (-1, 1) for c in range(length)]
    for teeth in combinations(sites, tooth_count):
        cells = _normalize(tuple((0, c) for c in range(length)) + teeth)
        groups.setdefault(_class(cells), cells)
    return list(groups.values())


def _horizontal_variants(shape):
    result = []
    for mask in _orientations(_mask(shape)):
        cells = tuple(map(tuple, np.argwhere(mask)))
        try:
            _spine(cells)
        except ValueError:
            continue
        result.append(cells)
    return result


def _random_plan(rng, scale, kind, maximum):
    limit = 30 // scale
    count = 2 if kind in ("coupled", "shape") else int(rng.choice((1, 2, 3), p=(.18, .57, .25)))
    if count == 3 and limit < 12:
        count = 2
    ns = ([1, 2] if kind == "coupled" else
          ([1, 1] if scale == 3 else [1, maximum]) if kind == "shape" else
          list(map(int, rng.integers(1, maximum + 1, size=count))))
    rng.shuffle(ns)
    # Three is feasible only on the larger logical boards.
    length_min = max(4, 2 * max(ns))
    length_max = min(7, limit - max(ns) - 2)
    if length_min > length_max:
        return None
    length = int(rng.integers(length_min, length_max + 1))
    tooth_count = int(rng.choice((1, 2, 3), p=(.55, .3, .15)))
    # Restrict vocabulary construction to the chosen, small geometric slice.
    vocabulary = _comb_vocabulary(length, tooth_count)
    if kind == "shape":
        if len(vocabulary) < 2:
            return None
        selected = [vocabulary[int(i)] for i in rng.choice(len(vocabulary), 2, replace=False)]
    elif kind == "coupled" or rng.random() < .55:
        selected = [vocabulary[int(rng.integers(len(vocabulary)))]] * count
    else:
        selected = [vocabulary[int(rng.integers(len(vocabulary)))]]
        for _ in range(count - 1):
            l = int(rng.integers(length_min, length_max + 1))
            pool = _comb_vocabulary(l, int(rng.integers(1, 3)))
            selected.append(pool[int(rng.integers(len(pool)))])
    min_h = sum(n + 2 for n in ns) + 2
    if min_h > limit:
        return None
    h = int(rng.integers(max(8, min_h), limit + 1))
    w = int(rng.integers(max(length + max(ns) + 2, 8), limit + 1))
    spare = h - min_h
    allocation = rng.multinomial(spare, np.full(count + 1, 1 / (count + 1)))
    current = int(rng.integers(1, w - 1))
    initial, previous = current, -1
    cells_list, commands, turns = [], [], []
    for index, (shape, n) in enumerate(zip(selected, ns)):
        variants = _horizontal_variants(shape)
        cells = variants[int(rng.integers(len(variants)))]
        sf, slo, shi = _spine(cells)
        span = shi - slo + 1
        turn = (1 if index == 0 else previous + 1) + int(allocation[index])
        f = turn + n + 1
        placements = []
        for lo in range(1, w - span):
            hi = lo + span - 1
            if not lo <= current <= hi:
                continue
            for side in (-1, 1):
                target = hi + n + 1 if side == 1 else lo - n - 1
                if 0 <= target < w:
                    placed = tuple((r + f - sf, c + lo - slo) for r, c in cells)
                    if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in placed):
                        placements.append((placed, side, target))
        if not placements:
            return None
        placed, side, target = placements[int(rng.integers(len(placements)))]
        cells_list.append(placed)
        commands.append((n, side))
        turns.append(turn)
        previous, current = f, target
    footprint = np.zeros((h, w), dtype=bool)
    for cells in cells_list:
        for r, c in cells:
            footprint[r, c] = True
    if len(_components(footprint)) != count:
        return None
    path = np.zeros((h, w), dtype=bool)
    current, start = initial, 0
    for cells, (n, side), turn in zip(cells_list, commands, turns):
        _, lo, hi = _spine(cells)
        target = hi + n + 1 if side == 1 else lo - n - 1
        path[start:turn + 1, current] = True
        path[turn, min(current, target):max(current, target) + 1] = True
        current, start = target, turn
    path[start:, current] = True
    if np.any(path & footprint):
        return None
    return (h, w), initial, cells_list, commands, path


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    prefix = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = prefix[h:, w:] - prefix[:-h, w:] - prefix[h:, :-w] + prefix[:-h, :-w]
    return np.argwhere(sums == 0)


def _pack(rng, board, tiles, bg):
    objects = [board] + tiles
    # Whole bounding boxes and a one-cell moat are reserved; no interleaving.
    for attempt in range(20):
        h, w = (30, 30) if attempt >= 10 else (int(rng.integers(22, 31)), int(rng.integers(22, 31)))
        canvas = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        order = [0] + sorted(range(1, len(objects)), key=lambda i: -objects[i].size)
        for i in order:
            obj = objects[i]
            oh, ow = obj.shape
            origins = _free_origins(blocked, oh, ow)
            if not len(origins):
                break
            r, c = map(int, origins[int(rng.integers(len(origins)))])
            canvas[r:r+oh, c:c+ow] = obj
            blocked[max(0, r-1):r+oh+1, max(0, c-1):c+ow+1] = True
        else:
            if np.count_nonzero(canvas == bg) > canvas.size // 2:
                return canvas
    return None


def _make_pair(rng, scale, q, kind, maximum):
    for attempt in range(250):
        plan = _random_plan(rng, scale, kind, maximum)
        if plan is None:
            continue
        shape, initial, cells_list, commands, path = plan
        if kind == "coupled" and {side for n, side in commands} != {-1, 1}:
            continue
        bg, floor, route, bar, marker = map(int, rng.choice(10, 5, replace=False))
        board = np.full(shape, floor, dtype=np.int8)
        board[0, initial] = route
        tiles = []
        for index, (cells, (n, side)) in enumerate(zip(cells_list, commands)):
            tile = np.full(_mask(cells).shape, bg, dtype=np.int8)
            for r, c in _normalize(cells):
                tile[r, c] = bar
            f, lo, hi = _spine(_normalize(cells))
            for j in range(n):
                tile[f, lo + 2 * j if side == 1 else hi - 2 * j] = marker
            choices = _orientations(tile)
            if kind == "shape" and index == 0:
                rotations = {(x.shape, (x != bg).tobytes()) for x in _orientations(tile, False)}
                reflected = [x for x in choices if (x.shape, (x != bg).tobytes()) not in rotations]
                choices = reflected or choices
            tile = choices[int(rng.integers(len(choices)))]
            tiles.append(np.repeat(np.repeat(tile, scale, 0), scale, 1))
            for r, c in cells:
                board[r, c] = bg
        logical = board.copy()
        logical[logical == bg] = bar
        logical[path] = route
        output = np.repeat(np.repeat(np.rot90(logical, q), scale, 0), scale, 1)
        # Packing preserves contour/macrocell information and adds no commands.
        canvas = _pack(rng, np.rot90(board, q).copy(), tiles, bg)
        if canvas is None:
            continue
        try:
            recovered, info = _recover(canvas)
        except ValueError:
            continue
        if not np.array_equal(recovered, output):
            raise AssertionError("construction disagrees with input reconstruction")
        if kind == "coupled" and _geometric_count(info) < 2:
            continue
        if kind == "shape" and (len(_routes(info, ignore_contours=True)) < 2
                                or not info["reflection_needed"]):
            continue
        return {"input": canvas, "output": output}
    raise RuntimeError(f"bounded construction exhausted for {scale}/{kind}/{maximum}")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    directions = list(map(int, rng.permutation(4)))
    scale_mode = int(rng.integers(4))
    maximum = 3 if scale_mode != 3 and rng.random() < .4 else 2
    train_scales = ([2] * num_train if scale_mode in (0, 2) else
                    [3] * num_train if scale_mode == 3 else
                    [int(rng.choice((2, 3))) for _ in range(num_train)])
    if maximum == 3:
        train_scales[1] = 2
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            q = directions[index % 2] if split == "train" else directions[2 + index % 2]
            scale = train_scales[index] if split == "train" else (
                3 if scale_mode in (2, 3) else 2 if scale_mode == 0 else int(rng.choice((2, 3))))
            kind = "coupled" if split == "train" and index == 0 else "shape" if split == "train" and index == 1 else "varied"
            pair_max = min(maximum, 2) if scale == 3 else maximum
            for _ in range(8):
                pair = _make_pair(rng, scale, q, kind, pair_max)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(nq, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("pair must have exactly input and output")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            if _key(a) in seen:
                raise ValueError("duplicate input")
            seen.add(_key(a))
            expected, info = _recover(a)
            if not np.array_equal(expected, b):
                raise ValueError("output is not the unique input-derived solution")
            k = info["scale"]
            blocks = b.reshape(b.shape[0]//k, k, b.shape[1]//k, k).transpose(0, 2, 1, 3)
            if not np.all(blocks == blocks[:, :, :1, :1]):
                raise ValueError("output must preserve the common block scale")
            bg, floor, route, bar, marker = info["palette"]
            if marker in b or bg in b:
                raise ValueError("temporary markers and canvas must be absent")
            if np.count_nonzero(b == bar) != k*k*np.count_nonzero(info["obstacle"]):
                raise ValueError("assembled tile material was lost")
            infos[split].append(info)
    training = infos["train"]
    if not any(_geometric_count(x) >= 2 for x in training):
        raise ValueError("training needs a genuinely route-resolved assembly")
    if not any(len({s["class"] for s in x["slots"]}) >= 2
               and len(_routes(x, ignore_contours=True)) == 2 for x in training):
        raise ValueError("training needs a contour-dependent routing witness")
    if not any(x["reflection_needed"] for x in training):
        raise ValueError("training needs visible reflection evidence")
    demonstrated = {n for x in training for _, n, side in x["commands"]}
    if not {1, 2} <= demonstrated:
        raise ValueError("two distinct clearances must be demonstrated")
    if {side for x in training for _, n, side in x["commands"]} != {-1, 1}:
        raise ValueError("both endpoint meanings must be demonstrated")
    if not {n for x in infos["test"] for _, n, side in x["commands"]} <= demonstrated:
        raise ValueError("test counts need training evidence")
    if {x["q"] for x in infos["test"]} & {x["q"] for x in training}:
        raise ValueError("generated tests reserve unseen forward orientations")


def self_test():
    """Stream 42 seeds; retain hashes and small feature sets, never grid corpora."""
    hashes, inputs, outputs, classes, scales, counts, dirs = set(), set(), set(), set(), set(), set(), set()
    features = {"coupled": 0, "shape": 0, "repeated_sides": 0, "reflection": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256(repr(_puzzle_key(puzzle)).encode()).hexdigest()
        hashes.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _, info = _recover(pair["input"])
                inputs.add(tuple(pair["input"].shape))
                outputs.add(tuple(pair["output"].shape))
                scales.add(info["scale"])
                counts.add(len(info["slots"]))
                dirs.add((split, info["q"]))
                classes.update(s["class"] for s in info["slots"])
                features["coupled"] += _geometric_count(info) >= 2
                features["shape"] += len(_routes(info, ignore_contours=True)) == 2
                features["reflection"] += bool(info["reflection_needed"])
                signs = [side for _, n, side in info["commands"]]
                features["repeated_sides"] += any(a == b for a, b in zip(signs, signs[1:]))
        if seed == 0:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
            bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzle[s]] for s in puzzle}
            bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
            try:
                validate_generated_puzzle(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("corrupt output accepted")
    assert len(hashes) == 42 and len(inputs) >= 12 and len(outputs) >= 12
    assert len(classes) >= 15 and scales == {2, 3} and counts == {1, 2, 3}
    assert dirs == {(s, q) for s in ("train", "test") for q in range(4)}
    assert features["coupled"] >= 42 and features["shape"] >= 42
    assert features["reflection"] >= 42 and features["repeated_sides"] > 0
    # Literal one-socket regression: independent of random plans and their labels.
    # The reflected loose L comb fits only after a reflection. Its marks travel
    # with it; the route turns at row 3 and runs down column 7.
    board = np.full((9, 10), 2, dtype=np.int8)
    board[0, 3] = 3
    board[5, 2:6] = 0
    board[4, 2] = 0
    tile = np.array([[4, 0, 0, 0], [1, 4, 4, 4]], dtype=np.int8)
    expected = board.copy()
    expected[expected == 0] = 4
    expected[:4, 3] = expected[3, 3:8] = expected[3:, 7] = 3
    for q in range(4):
        for k in (2, 3):
            fixture = np.zeros((30, 30), dtype=np.int8)
            oriented_board = np.rot90(board, q)
            bh, bw = oriented_board.shape
            fixture[1:1+bh, 1:1+bw] = oriented_board
            loose = np.rot90(tile[:, ::-1], q)
            loose = np.repeat(np.repeat(loose, k, 0), k, 1)
            lh, lw = loose.shape
            fixture[15:15+lh, 15:15+lw] = loose
            answer = np.repeat(np.repeat(np.rot90(expected, q), k, 0), k, 1)
            assert np.array_equal(solve(fixture), answer)
            damaged = fixture.copy()
            rr, cc = np.argwhere((damaged != 0) & (np.indices(damaged.shape)[0] >= 15))[0]
            damaged[rr, cc] = 0
            try:
                solve(damaged)
            except ValueError:
                pass
            else:
                raise AssertionError("broken macrocell accepted")
    # A centered tooth permits both marker endpoints, with both exits feasible.
    # Reject rather than choose an arbitrary reflection or spatial ordering.
    ambiguous = np.zeros((30, 30), dtype=np.int8)
    b = np.full((9, 10), 2, dtype=np.int8)
    b[0, 3] = 3
    b[5, 2:7] = b[4, 4] = 0
    ambiguous[1:10, 1:11] = b
    t = np.array([[0, 0, 4, 0, 0], [1, 4, 4, 4, 4]], dtype=np.int8)
    ambiguous[15:19, 15:25] = np.repeat(np.repeat(t, 2, 0), 2, 1)
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("two feasible routes accepted")
    for a, b in ((2, 1), (10, 4)):
        p = generate_puzzle(2100 + a, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "input_shapes": len(inputs),
            "output_shapes": len(outputs), "comb_classes": len(classes),
            "scales": sorted(scales), "slot_counts": sorted(counts), "features": features}
