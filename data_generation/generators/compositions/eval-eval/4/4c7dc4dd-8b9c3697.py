"""Ranked docking keys: a local rank analogy selects geometrically fitting cargo.

Only Python's standard library and NumPy are used. No parent modules are loaded.
Construction paints logical footprints; solve() reparses the input pixels.
"""
from itertools import permutations
import json

import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_PERMUTATIONS = tuple(permutations(range(3)))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid must be 2D with dimensions 1..30")
    _require(np.issubdtype(grid.dtype, np.integer) and
             np.all((0 <= grid) & (grid <= 9)), "grid must contain integer colors 0..9")
    return grid.astype(np.int8, copy=True)


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _components(mask):
    unseen = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRECTIONS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def _rectangle(r0, c0, r1, c1):
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _outline(r0, c0, r1, c1):
    return (_rectangle(r0, c0, r0 + 1, c1) |
            _rectangle(r1 - 1, c0, r1, c1) |
            _rectangle(r0, c0, r1, c0 + 1) |
            _rectangle(r0, c1 - 1, r1, c1))


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _fork(cells, color):
    """Recover a comb in its own rotating coordinate system, without templates.

    k increases outward from the base; q increases to the right when open-up.
    This fixes stripe chirality under rotation, rather than reflection.
    """
    possibilities = []
    for out in _DIRECTIONS:
        perp = out[1], -out[0]
        kmin = min(r * out[0] + c * out[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * out[0] + c * out[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum(k > 0 and qq == q for k, qq in local) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, length in enumerate(lengths)
                     for k in range(1, length + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            a, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((a, q, length))
        if (len(runs) != 3 or len({x[2] for x in runs}) != 3 or
                any(runs[i][1] >= runs[i + 1][0] for i in (0, 1))):
            continue
        a, b, center = runs[1]
        if a + b != width or (b - a) % 3:
            continue
        prongs = tuple(x[2] for x in runs)
        ranks = tuple(sorted(prongs).index(x) for x in prongs)
        possibilities.append(dict(cells=cells, color=color, out=out, perp=perp,
                                  kmin=kmin, qmin=qmin, width=width, runs=tuple(runs),
                                  lengths=prongs, ranks=ranks, q0=a, span=b - a,
                                  stop=center + 1))
    _require(len(possibilities) == 1, "fork needs one base and three distinct positive prongs")
    return possibilities[0]


def _point(fork, k, q):
    k, q = k + fork["kmin"], q + fork["qmin"]
    return (k * fork["out"][0] + q * fork["perp"][0],
            k * fork["out"][1] + q * fork["perp"][1])


def _coordinates(fork, cells):
    return {(r * fork["out"][0] + c * fork["out"][1] - fork["kmin"],
             r * fork["perp"][0] + c * fork["perp"][1] - fork["qmin"])
            for r, c in cells}


def _stripe_code(grid, fork, cells):
    """Read a rectangle relative to a fork, allowing any position or stripe width."""
    local = _coordinates(fork, cells)
    ks, qs = sorted({k for k, _ in local}), sorted({q for _, q in local})
    if (not qs or len(qs) % 3 or local != {(k, q) for k in ks for q in qs} or
            ks != list(range(ks[0], ks[-1] + 1)) or
            qs != list(range(qs[0], qs[-1] + 1))):
        return None
    thickness = len(qs) // 3
    code = []
    for band in range(3):
        colors = {int(grid[_point(fork, k, q)]) for k in ks
                  for q in qs[band * thickness:(band + 1) * thickness]}
        if len(colors) != 1:
            return None
        code.append(colors.pop())
    return tuple(code) if len(set(code)) == 3 else None


def _valid_key(grid, cells):
    r0, c0, r1, c1 = _box(cells)
    if cells != _rectangle(r0, c0, r1, c1):
        return False
    patch = grid[r0:r1, c0:c1]
    for array in (patch, patch.T):
        if array.shape[1] % 3:
            continue
        bands = np.split(array, 3, axis=1)
        if all(np.all(band == band[0, 0]) for band in bands) and len({int(band[0, 0]) for band in bands}) == 3:
            return True
    return False


def _calibration(grid, bg):
    frames = []
    for color in np.unique(grid):
        if int(color) == bg:
            continue
        for cells in _components(grid == color):
            r0, c0, r1, c1 = _box(cells)
            if min(r1 - r0, c1 - c0) >= 7 and cells == _outline(r0, c0, r1, c1):
                frames.append((r0, c0, r1, c1, int(color)))
    _require(len(frames) == 1, "expected one isolated reference frame")
    r0, c0, r1, c1, frame_color = frame = frames[0]
    interior = _rectangle(r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    forks = []
    for color in np.unique(grid[r0 + 1:r1 - 1, c0 + 1:c1 - 1]):
        if int(color) in (bg, frame_color):
            continue
        mask = np.zeros(grid.shape, bool)
        for p in interior:
            mask[p] = int(grid[p]) == int(color)
        for cells in _components(mask):
            if cells != _rectangle(*_box(cells)):
                forks.append(_fork(cells, int(color)))
    _require(len(forks) == 1, "reference must contain one fork")
    fork = forks[0]
    q0, span, stop = fork["q0"], fork["span"], fork["stop"]
    first = {_point(fork, stop, q) for q in range(q0, q0 + span)}
    _require(first <= interior, "reference key outside frame")
    code = _stripe_code(grid, fork, first)
    _require(code is not None and not (set(code) & {bg, frame_color, fork["color"]}),
             "reference key must have three distinct role colors")
    cargo, k = set(), stop
    while k <= 30:
        row = {_point(fork, k, q) for q in range(q0, q0 + span)}
        if not row <= interior or _stripe_code(grid, fork, row) != code:
            break
        cargo |= row
        k += 1
    row = {_point(fork, k, q) for q in range(q0, q0 + span)}
    _require(row <= interior, "reference needs a visible trail")
    colors = {int(grid[p]) for p in row}
    _require(len(colors) == 1, "reference trail must be uniform")
    trail = colors.pop()
    _require(trail not in {bg, frame_color, fork["color"], *code}, "trail role must be distinct")
    trace = set()
    while k <= 30:
        row = {_point(fork, k, q) for q in range(q0, q0 + span)}
        if not row <= interior or any(int(grid[p]) != trail for p in row):
            break
        trace |= row
        k += 1
    _require(bool(trace), "reference trail may not be empty")
    occupied = {p for p in interior if int(grid[p]) != bg}
    _require(occupied == fork["cells"] | cargo | trace, "extraneous reference content")
    rank_colors = [None] * 3
    for rank, color in zip(fork["ranks"], code):
        rank_colors[rank] = color
    return dict(frame=frame, area=_rectangle(r0, c0, r1, c1), fork=fork,
                cargo=cargo, trace=trace, trail=trail, rank_colors=tuple(rank_colors))


def _geometry(fork, cells, static, shape):
    local = _coordinates(fork, cells)
    qs = {q for _, q in local}
    if qs != set(range(fork["q0"], fork["q0"] + fork["span"])):
        return None
    k0, k1 = min(k for k, _ in local), max(k for k, _ in local)
    if k0 <= max(fork["lengths"]) or k0 <= fork["stop"]:
        return None
    route = {_point(fork, k, q) for k in range(fork["stop"], k1 + 1) for q in qs}
    if (not route.isdisjoint(static) or
            any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in route)):
        return None
    final = {_point(fork, k, q) for k in range(fork["stop"], fork["stop"] + k1 - k0 + 1)
             for q in qs}
    return dict(route=route, final=final, distance=k0 - fork["stop"],
                length=k1 - k0 + 1, k0=k0, k1=k1)


def _analyze(grid):
    bg = int(grid[0, 0])
    _require(np.all(grid[0] == bg) and np.all(grid[-1] == bg) and
             np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg),
             "outer edge must be background")
    reference = _calibration(grid, bg)
    outside = grid != bg
    for p in reference["area"]:
        outside[p] = False
    forks, keys = [], []
    palette = set(reference["rank_colors"])
    for cells in _components(outside):
        colors = {int(grid[p]) for p in cells}
        if len(colors) == 1:
            color = colors.pop()
            _require(color not in palette | {reference["trail"], reference["frame"][4]},
                     "structural and coding roles must be distinct")
            forks.append(_fork(cells, color))
        else:
            _require(colors == palette and _valid_key(grid, cells),
                     "loose keys must be solid three-color rectangles")
            keys.append(dict(cells=cells))
    _require(bool(forks) and bool(keys), "need query forks and loose keys")
    _require(reference["fork"]["color"] not in palette, "structural role collision")
    static = set(reference["area"])
    for fork in forks:
        static |= fork["cells"]
        fork["code"] = tuple(reference["rank_colors"][rank] for rank in fork["ranks"])
    matches, geometric, used, swept = [], [], set(), set()
    for fi, fork in enumerate(forks):
        candidates = []
        geo = []
        for ti, token in enumerate(keys):
            candidate = _geometry(fork, token["cells"], static, grid.shape)
            if candidate is not None:
                candidate = dict(candidate, fi=fi, ti=ti)
                geo.append(candidate)
                if _stripe_code(grid, fork, token["cells"]) == fork["code"]:
                    candidates.append(candidate)
        _require(len(candidates) <= 1, "multiple keys satisfy code and geometry")
        geometric.append(geo)
        if candidates:
            match = candidates[0]
            _require(match["ti"] not in used and swept.isdisjoint(match["route"]),
                     "key assignments or swept regions conflict")
            used.add(match["ti"])
            swept |= match["route"]
            matches.append(match)
    return dict(background=bg, reference=reference, forks=forks, keys=keys,
                static=static, matches=matches, geometric=geometric, used=used)


def _render(grid, analysis, matches=None):
    output = grid.copy()
    for key in analysis["keys"]:
        _paint(output, key["cells"], analysis["background"])
    for match in analysis["matches"] if matches is None else matches:
        _paint(output, match["route"] - match["final"], analysis["reference"]["trail"])
        fork = analysis["forks"][match["fi"]]
        for r, c in analysis["keys"][match["ti"]]["cells"]:
            destination = (r - match["distance"] * fork["out"][0],
                           c - match["distance"] * fork["out"][1])
            output[destination] = grid[r, c]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every convention, selection, and output cell from this input only."""
    grid = _grid(input_grid)
    return _render(grid, _analyze(grid))


def _rank_code(lengths, mapping):
    return tuple(mapping[sorted(lengths).index(length)] for length in lengths)


def _local_structure(lengths, span, guard, gap, tail, color):
    width = span + 2 * (guard + gap + tail)
    q0 = tail + guard + gap
    cells = {(0, q): color for q in range(width)}
    for a, b, length in ((tail, tail + guard, lengths[0]),
                          (q0, q0 + span, lengths[1]),
                          (width - tail - guard, width - tail, lengths[2])):
        cells.update({(k, q): color for k in range(1, length + 1) for q in range(a, b)})
    return cells, width, q0


def _local_key(cells, k0, length, q0, span, code):
    cells.update({(k, q): code[(q - q0) // (span // 3)]
                  for k in range(k0, k0 + length) for q in range(q0, q0 + span)})


def _local_array(cells, depth, width, bg):
    result = np.full((depth, width), bg, np.int8)
    for (k, q), color in cells.items():
        result[depth - 1 - k, q] = color
    return result


def _reference_module(rng, maps, palette, order):
    bg, frame, trail, structural = palette[:4]
    lengths = tuple((1, 2, 3)[rank] for rank in order)
    structure, width, q0 = _local_structure(lengths, 3, 1, 1, 0, structural)
    key_length, trail_length = map(int, rng.integers(1, 3, size=2))
    depth = max(max(lengths), lengths[1] + key_length + trail_length) + 1
    result = []
    for mapping in maps:
        cells = dict(structure)
        code = _rank_code(lengths, mapping)
        _local_key(cells, lengths[1] + 1, key_length, q0, 3, code)
        cells.update({(k, q): trail for k in range(lengths[1] + key_length + 1, depth)
                      for q in range(q0, q0 + 3)})
        interior = _local_array(cells, depth, width, bg)
        array = np.full((depth + 4, width + 4), bg, np.int8)
        array[2:-2, 2:-2] = interior
        _paint(array, _outline(0, 0, *array.shape), frame)
        result.append(array)
    return result


def _query_module(rng, available, maps, palette, order, rotation, mode, hard, thick=False):
    depth_limit, cross_limit = available if rotation % 2 == 0 else available[::-1]
    span = 6 if thick else 3
    structural = int(rng.choice(palette[3:5]))
    choices = [(g, gap, tail) for g in (1, 2) for gap in (1, 2) for tail in (0, 1)
               if span + 2 * (g + gap + tail) <= cross_limit]
    guard, gap, tail = choices[int(rng.integers(len(choices)))]
    if mode in ("switch", "lose"):
        sorted_lengths = (1, 3, 4)
    else:
        sorted_lengths = tuple(sorted(map(int, rng.choice(np.arange(1, 7 if hard else 5),
                                                          3, replace=False))))
        if hard and thick:
            sorted_lengths = (1, int(rng.integers(2, 5)), int(rng.integers(5, 7)))
    lengths = tuple(sorted_lengths[rank] for rank in order)
    structure, width, q0 = _local_structure(lengths, span, guard, gap, tail, structural)
    codes = [_rank_code(lengths, mapping) for mapping in maps]
    other_codes = [tuple(maps[0][i] for i in p) for p in _PERMUTATIONS]
    alternatives = [code for code in other_codes if code not in codes]
    if mode == "lose":
        cargo_codes = [codes[0], alternatives[int(rng.integers(len(alternatives)))]]
    elif mode == "inactive":
        cargo_codes = [alternatives[int(i)] for i in rng.choice(len(alternatives), 2, replace=False)]
    else:
        cargo_codes = codes.copy()
    if rng.integers(2):
        cargo_codes.reverse()
    start, separation = max(lengths) + 2, int(rng.integers(1, 3))
    lens = list(map(int, rng.integers(1, 3, size=2)))
    if mode == "switch" and lens[0] == lens[1]:
        lens[1] = 3 - lens[0]
    while start + sum(lens) + separation > depth_limit:
        if max(lens) > 1:
            lens[lens.index(max(lens))] -= 1
        else:
            separation -= 1
    _require(separation >= 1, "packing too shallow")
    starts = (start, start + lens[0] + separation)
    depth = starts[1] + lens[1]
    original = dict(structure)
    for k0, length, code in zip(starts, lens, cargo_codes):
        _local_key(original, k0, length, q0, span, code)
    outputs = []
    for code in codes:
        cells = dict(structure)
        if code in cargo_codes:
            index = cargo_codes.index(code)
            stop, length = lengths[1] + 1, lens[index]
            end = starts[index] + length
            cells.update({(k, q): palette[2] for k in range(stop + length, end)
                          for q in range(q0, q0 + span)})
            _local_key(cells, stop, length, q0, span, code)
        outputs.append(np.rot90(_local_array(cells, depth, width, palette[0]), rotation).copy())
    input_array = np.rot90(_local_array(original, depth, width, palette[0]), rotation).copy()
    return input_array, outputs


def _extra_key(rng, grids, analyses, code, thickness):
    """Place a code-correct key that fails physical matching for every fork."""
    grid = grids[0]
    fork = analyses[0]["forks"][0]
    raw = {(k * fork["out"][0] + q * fork["perp"][0],
            k * fork["out"][1] + q * fork["perp"][1]): code[q // thickness]
           for k in range(1) for q in range(3 * thickness)}
    r0, c0, r1, c1 = _box(raw)
    normalized = {(r - r0, c - c0): color for (r, c), color in raw.items()}
    dh, dw = r1 - r0, c1 - c0
    protected = set(analyses[0]["static"])
    for analysis in analyses:
        for match in analysis["matches"]:
            protected |= match["route"]
    protected = _halo(protected | set(map(tuple, np.argwhere(grid != analyses[0]["background"]))))
    choices = []
    for r in range(1, grid.shape[0] - dh):
        for c in range(1, grid.shape[1] - dw):
            cells = {(r + y, c + x) for y, x in normalized}
            if not cells.isdisjoint(protected):
                continue
            if any(_geometry(f, cells, analyses[0]["static"], grid.shape) is not None
                   for f in analyses[0]["forks"]):
                continue
            choices.append((r, c))
    _require(bool(choices), "no spare space for physical distractor")
    r, c = choices[int(rng.integers(len(choices)))]
    for array in grids:
        for (y, x), color in normalized.items():
            array[r + y, c + x] = color


def _scene(rng, hard=False, twins=False, index=0):
    """Bounded rejection only excludes collisions or accidental foreign matches."""
    for _ in range(80):
        palette = tuple(map(int, rng.permutation(10)[:8]))
        bg = palette[0]
        mapping = palette[5:8]
        alternative = tuple(mapping[i] for i in _PERMUTATIONS[int(rng.integers(1, 6))])
        maps = (mapping, alternative)
        h, w = map(int, rng.integers(27, 31, size=2))
        rh = int(rng.integers(12, h - 14 + 1))
        cw = int(rng.integers(12, w - 14 + 1))
        slots = [(1, 1, rh, cw), (1, cw + 2, rh, w - cw - 3),
                 (rh + 2, 1, h - rh - 3, cw),
                 (rh + 2, cw + 2, h - rh - 3, w - cw - 3)]
        slots = [slots[int(i)] for i in rng.permutation(4)]
        grids = [np.full((h, w), bg, np.int8) for _ in range(2)]
        outputs = [g.copy() for g in grids]
        ref_order = _PERMUTATIONS[int(rng.integers(6))]
        reference = _reference_module(rng, maps, palette, ref_order)
        rotation = int(rng.integers(4))
        reference = [np.rot90(array, rotation).copy() for array in reference]
        count = 3 if hard else 2
        query_orders = [p for p in _PERMUTATIONS if p != ref_order]
        rng.shuffle(query_orders)
        first_rotation = int(rng.integers(4))
        try:
            for i in range(count + 1):
                r, c, sh, sw = slots[i]
                if i == 0:
                    arrays = reference
                    result_arrays = reference
                else:
                    if twins:
                        mode = "switch" if i == 1 else "lose"
                    else:
                        inactive = bool(index % 2 or rng.integers(2))
                        mode = "inactive" if i == count and inactive else "active"
                    turn = (first_rotation + (i - 1) * int(rng.choice([1, 3]))) % 4
                    array, result_arrays = _query_module(
                        rng, (sh, sw), maps, palette, query_orders[i - 1], turn,
                        mode, hard, thick=hard and (i == 1 or rng.random() < .3))
                    arrays = [array, array]
                dh, dw = arrays[0].shape
                _require(dh <= sh and dw <= sw, "module does not fit")
                y, x = r + int(rng.integers(sh - dh + 1)), c + int(rng.integers(sw - dw + 1))
                for j in range(2):
                    grids[j][y:y + dh, x:x + dw] = arrays[j]
                    outputs[j][y:y + dh, x:x + dw] = result_arrays[j]
            analyses = [_analyze(g) for g in grids]
            _require(all(len(a["forks"]) == count for a in analyses), "touching modules")
            _require(all(np.array_equal(_render(g, a), o)
                         for g, a, o in zip(grids, analyses, outputs)), "foreign key assignment")
            # Distractors use the parsed first fork's code and orientation.
            # Both twin calibrations receive a code-correct, geometrically invalid key.
            for j in range(2):
                code = analyses[j]["forks"][0]["code"]
                _extra_key(rng, grids, analyses, code, 1)
            if not hard:
                _extra_key(rng, grids, analyses, analyses[0]["forks"][0]["code"], 2)
            final = [_analyze(g) for g in grids]
            _require(all(np.array_equal(_render(g, a), o)
                         for g, a, o in zip(grids, final, outputs)), "extra key changes assignment")
            return [{"input": g, "output": o} for g, o in zip(grids, outputs)]
        except ValueError:
            continue
    raise RuntimeError("could not pack a noninteracting ranked-key scene in 80 attempts")


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    first, second = _scene(rng, twins=True)
    puzzle = {"train": [first, second], "test": []}
    seen = {_signature(first["input"]), _signature(second["input"])}
    for split, count in (("train", a - 2), ("test", b)):
        for index in range(count):
            for _ in range(20):
                pair = _scene(rng, hard=split == "test", index=index)[0]
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == output.shape, "output must preserve canvas dimensions")
            signature = _signature(grid)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            analysis = _analyze(grid)
            reference, forks, keys = analysis["reference"], analysis["forks"], analysis["keys"]
            _require(np.array_equal(output, _render(grid, analysis)), "incorrect ranked docking output")
            _require(np.array_equal(solve(grid), output), "input-only solver disagreement")
            _require(not np.array_equal(grid, output), "identity scene")
            _require(all(int(output[p]) == int(grid[p]) for p in analysis["static"]), "fixed object changed")
            objects = [f["cells"] for f in forks] + [k["cells"] for k in keys] + [reference["area"]]
            occupied = set()
            for cells in objects:
                _require(occupied.isdisjoint(_halo(cells)), "input objects touch")
                occupied |= cells
            _require(all(_stripe_code(grid, f, key["cells"]) is not None
                         for key in keys for f in forks
                         if _geometry(f, key["cells"], analysis["static"], grid.shape) is not None),
                     "aligned key has malformed stripes")
            for match in analysis["matches"]:
                fork, key = forks[match["fi"]], keys[match["ti"]]
                _require(match["distance"] > 0 and match["route"].isdisjoint(fork["cells"]), "bad route")
                _require(_stripe_code(output, fork, match["final"]) == fork["code"], "cargo code changed")
                _require(all(int(output[p]) == reference["trail"]
                             for p in match["route"] - match["final"]), "trail footprint")
                for r, c in key["cells"]:
                    dest = r - match["distance"] * fork["out"][0], c - match["distance"] * fork["out"][1]
                    _require(int(output[dest]) == int(grid[r, c]), "cargo is not rigidly translated")
            _require(len(keys) > len(analysis["matches"]) > 0, "need accepted and rejected keys")
            # At least one color-correct distractor fails geometry, for every scene.
            distractors = [(fi, ti) for fi, f in enumerate(forks) for ti, key in enumerate(keys)
                           if _stripe_code(grid, f, key["cells"]) == f["code"] and
                           _geometry(f, key["cells"], analysis["static"], grid.shape) is None]
            _require(bool(distractors), "missing geometry ablation witness")
            _require(any(all(int(output[p]) == analysis["background"] for p in keys[ti]["cells"])
                         for _, ti in distractors),
                     "geometry-free retention must give different output cells")
            if split == "train":
                _require(len(forks) == 2 and all(f["span"] == 3 for f in forks), "training geometry")
                _require(max(max(f["lengths"]) for f in forks) <= 4, "training length range")
                training.append((grid, output, analysis))
            else:
                _require(len(forks) == 3 and len(analysis["matches"]) >= 2, "test count shift")
                _require(any(f["span"] == 6 for f in forks), "test thickness shift")
                _require(max(max(f["lengths"]) for f in forks) >= 5, "test length shift")
    g0, o0, a0 = training[0]
    g1, o1, a1 = training[1]
    _require(g0.shape == g1.shape and a0["reference"]["frame"] == a1["reference"]["frame"], "twins frame")
    changed = set(map(tuple, np.argwhere(g0 != g1)))
    _require(bool(changed) and changed <= a0["reference"]["cargo"], "twins must change only calibration cargo")
    outside = np.ones(g0.shape, bool)
    for p in a0["reference"]["area"]:
        outside[p] = False
    _require(np.any(o0[outside] != o1[outside]), "rank-reference ablation failed")
    _require((len(a0["matches"]), len(a1["matches"])) == (2, 1), "twins activation evidence")
    _require({m["length"] for a in (a0, a1) for m in a["matches"]} == {1, 2},
             "twins must demonstrate both one-cell and two-cell cargo lengths")
    _require(len({f["out"] for f in a0["forks"]}) == 2 and
             a0["forks"][0]["out"][0] != a0["forks"][1]["out"][0], "orientation evidence")
    orders = [f["ranks"] for f in a0["forks"]]
    _require(len(set(orders)) == 2 and a0["reference"]["fork"]["ranks"] not in orders, "rank transfer evidence")
    _require(any(len(_coordinates(f, key["cells"])) >= 6 and
                 _stripe_code(g0, f, key["cells"]) == f["code"] and
                 len({q for _, q in _coordinates(f, key["cells"])}) == 6
                 for f in a0["forks"] for key in a0["keys"]), "width-six training distractor")
    # A nearest-only geometry rule ignores the reference and must fail a twin.
    nearest = [min(options, key=lambda m: m["k0"]) for options in a0["geometric"] if options]
    blind0, blind1 = _render(g0, a0, nearest), _render(g1, a1, nearest)
    _require(not np.array_equal(blind0[outside], o0[outside]) or
             not np.array_equal(blind1[outside], o1[outside]), "rank ablation unexpectedly succeeds")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and
               all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    shapes, geometries, layouts, mappings, codes, thicknesses = (set() for _ in range(6))
    distances, lengths, orders, active, structures, palettes = (set() for _ in range(6))
    signatures = set()
    first_puzzle = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_equal(puzzle, generate_puzzle(seed)), "nondeterministic seed")
        if seed == 0:
            first_puzzle = puzzle
        signatures.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                analysis = _analyze(grid)
                shapes.add(grid.shape)
                palettes.add(tuple(map(int, np.unique(grid))))
                mappings.add(analysis["reference"]["rank_colors"])
                active.add((len(analysis["forks"]), len(analysis["matches"])))
                layouts.add(tuple((f["out"], _box(f["cells"])) for f in analysis["forks"]))
                normalized = {analysis["background"]: 0}
                structures.add((grid.shape, tuple(normalized.setdefault(int(x), len(normalized)) for x in grid.flat)))
                for fork in analysis["forks"]:
                    geometries.add((fork["width"], fork["runs"]))
                    orders.add(fork["ranks"])
                    codes.add(fork["code"])
                    lengths.update(fork["lengths"])
                    thicknesses.add(fork["span"] // 3)
                distances.update(m["distance"] for m in analysis["matches"])
    _require(len(signatures) == 42 and len(structures) >= 150 and len(layouts) >= 150, "low scene variation")
    _require(len(geometries) >= 60 and len(mappings) >= 30 and len(codes) >= 30, "low semantic variation")
    _require(len(shapes) >= 12 and len(palettes) >= 20 and len(distances) >= 5, "low shape/palette/motion variation")
    _require(orders == set(_PERMUTATIONS) and thicknesses == {1, 2} and lengths == set(range(1, 7)),
             "missing rank/thickness/length variation")
    _require({(2, 1), (2, 2), (3, 2), (3, 3)} <= active, "missing activation cases")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(2000 + nt * 10 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Equivariance uses the same data-only solve path on unfamiliar layouts/palettes.
    rng = np.random.default_rng(742)
    color_map = rng.permutation(10)
    for pair in first_puzzle["train"] + first_puzzle["test"]:
        for turn in range(4):
            grid = color_map[np.rot90(pair["input"], turn)]
            expected = color_map[np.rot90(pair["output"], turn)]
            _require(np.array_equal(solve(grid), expected), "rotation/recoloring equivariance")
            _require(np.array_equal(solve(grid.tolist()), expected), "nested-list input")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first_puzzle[s]] for s in first_puzzle}
    pair = damaged["test"][0]
    analysis = _analyze(pair["input"])
    cell = min(analysis["matches"][0]["route"] - analysis["matches"][0]["final"])
    pair["output"][cell] = analysis["background"]
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted trail accepted")
    malformed = first_puzzle["train"][0]["input"].copy()
    reference = _analyze(malformed)["reference"]
    # Destroy the unique three-color reference code without touching its frame.
    bad_color, replacement = reference["rank_colors"][:2]
    for p in reference["cargo"]:
        if int(malformed[p]) == bad_color:
            malformed[p] = replacement
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed reference accepted")
    # A second aligned key with the correct code is ambiguous, not a nearest-key tie.
    ambiguous = first_puzzle["train"][0]["input"].copy()
    analysis = _analyze(ambiguous)
    fork = analysis["forks"][0]
    options = analysis["geometric"][0]
    _require(len(options) >= 2, "missing aligned ambiguity diagnostic")
    for option in options:
        for r, c in analysis["keys"][option["ti"]]["cells"]:
            q = r * fork["perp"][0] + c * fork["perp"][1] - fork["qmin"]
            ambiguous[r, c] = fork["code"][(q - fork["q0"]) // (fork["span"] // 3)]
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous matching keys accepted")
    for prongs in ((1, 1, 3), (1, 0, 3)):
        local, width, _ = _local_structure(prongs, 3, 1, 1, 0, 2)
        try:
            _fork(set(local), 2)
        except ValueError:
            pass
        else:
            raise AssertionError("tied or absent middle prong accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                color_normalized_scenes=len(structures), layouts=len(layouts),
                fork_geometries=len(geometries), rank_color_maps=len(mappings),
                required_codes=len(codes), travel_distances=sorted(distances),
                prong_lengths=sorted(lengths), stripe_thicknesses=sorted(thicknesses),
                rank_orders=len(orders), activation_cases=sorted(active),
                all_legal_counts=True, reference_and_geometry_ablations=True,
                independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
