"""Line-qualified ballots: exact addresses, geometric quorums, shape elections.

The scene contains two carrier rectangles: an ink-filled token lattice and a
two-color cavity stencil. Exterior connected glyphs are both addresses and
ballots. Only addresses lying on complete queried board lines may vote.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import itertools

import numpy as np


def _norm(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - y, c - x) for r, c in cells))


def _rot(shape):
    return _norm((c, -r) for r, c in shape)


def _flip(shape):
    return _norm((r, -c) for r, c in shape)


@lru_cache(maxsize=1024)
def _rotations(shape):
    result = set()
    for _ in range(4):
        result.add(shape)
        shape = _rot(shape)
    return tuple(sorted(result))


@lru_cache(maxsize=1024)
def _variants(shape):
    return tuple(sorted(set(_rotations(shape)) | set(_rotations(_flip(shape)))))


def _canon(shape):
    return _variants(_norm(shape))[0]


def _size(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _components(mask):
    """Four-neighbor components, independent of color."""
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        todo, cells = [(y, x)], []
        seen[y, x] = True
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        result.append(tuple(cells))
    return result


def _hole_free(shape):
    h, w = _size(shape)
    mask = np.ones((h + 2, w + 2), bool)
    for r, c in shape:
        mask[r + 1, c + 1] = False
    return len(_components(mask)) == 1


@lru_cache(maxsize=1)
def _shape_bank():
    """All connected, hole-free masks on a 3x3 canvas; no shape dictionary."""
    bank = set()
    for code in range(1, 512):
        mask = np.array([(code >> i) & 1 for i in range(9)], bool).reshape(3, 3)
        if len(_components(mask)) == 1:
            shape = _canon(tuple(map(tuple, np.argwhere(mask))))
            if _hole_free(shape):
                bank.add(shape)
    return tuple(sorted(bank))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _anchor_pair(rng, preferred=None):
    bank = _shape_bank()
    choices = [s for s in bank if 4 <= len(s) <= 6 and
               _flip(s) not in _rotations(s) and any(
                   t != s and len(t) == len(s) and
                   sorted(_size(t)) == sorted(_size(s)) and
                   len(_rotations(t)) > 1 for t in bank)]
    if preferred:
        narrowed = [s for s in choices if s in preferred]
        if narrowed:
            choices = narrowed
    anchor = _choose(rng, choices)
    companion = _choose(rng, [s for s in bank if s != anchor and len(s) == len(anchor)
                             and sorted(_size(s)) == sorted(_size(anchor))
                             and len(_rotations(s)) > 1])
    return anchor, companion


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _check_grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger cells"
    assert np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a.astype(np.int8, copy=False)


def _complete_lines(selected, rows, cols):
    selected = set(selected)
    lines = []
    for r in range(rows):
        line = {(r, c) for c in range(cols)}
        if line <= selected:
            lines.append(line)
    for c in range(cols):
        line = {(r, c) for r in range(rows)}
        if line <= selected:
            lines.append(line)
    return lines, set().union(*lines) if lines else set()


def _plurality(counts):
    if not counts:
        return None
    high = max(counts.values())
    winners = [c for c, n in counts.items() if n == high]
    assert len(winners) == 1, "supported cavity has a tied election"
    return winners[0]


def _read(input_grid):
    """Recover every address, line, ballot and cavity using input pixels only."""
    a = _check_grid(input_grid)
    bg = int(a[0, 0])
    assert np.all(a[[0, -1]] == bg) and np.all(a[:, [0, -1]] == bg), "outer margin"
    rectangles = []
    for color in np.unique(a):
        if color == bg:
            continue
        for cells in _components(a == color):
            ys, xs = zip(*cells)
            y, x, b, e = min(ys), min(xs), max(ys) + 1, max(xs) + 1
            box = a[y:b, x:e]
            if min(box.shape) < 5:
                continue
            if not (np.all(box[[0, -1]] == color) and np.all(box[:, [0, -1]] == color)):
                continue
            if not np.any(box == bg):
                continue
            rectangles.append((int(color), (y, x, b, e), box))
    assert len(rectangles) == 2 and rectangles[0][0] == rectangles[1][0], "two carrier boxes"
    carrier = rectangles[0][0]
    stencils = [v for v in rectangles if set(map(int, np.unique(v[2]))) == {bg, carrier}]
    assert len(stencils) == 1, "one blank cavity stencil"
    stencil = stencils[0]
    board = next(v for v in rectangles if v is not stencil)
    box = board[2]
    row_walls = np.flatnonzero(np.all(box == carrier, axis=1)).tolist()
    col_walls = np.flatnonzero(np.all(box == carrier, axis=0)).tolist()
    assert row_walls[0] == col_walls[0] == 0
    assert row_walls[-1] == box.shape[0] - 1 and col_walls[-1] == box.shape[1] - 1
    dy, dx = np.diff(row_walls), np.diff(col_walls)
    assert len(set(dy.tolist())) == len(set(dx.tolist())) == 1
    assert 4 <= dy[0] <= 5 and 4 <= dx[0] <= 5
    nr, nc = len(row_walls) - 1, len(col_walls) - 1
    assert 2 <= nr <= 4 and 2 <= nc <= 4
    tokens, lookup = {}, {}
    for r, c in itertools.product(range(nr), range(nc)):
        slot = box[row_walls[r] + 1:row_walls[r + 1],
                   col_walls[c] + 1:col_walls[c + 1]]
        ink = set(map(int, np.unique(slot))) - {bg}
        assert len(ink) == 1 and carrier not in ink
        shape = _norm(tuple(map(tuple, np.argwhere(slot != bg))))
        assert len(_components(slot != bg)) == 1 and _hole_free(shape)
        assert max(_size(shape)) <= 3
        ys, xs = np.where(slot != bg)
        assert abs(int(ys.min()) - (slot.shape[0] - 1 - int(ys.max()))) <= 1
        assert abs(int(xs.min()) - (slot.shape[1] - 1 - int(xs.max()))) <= 1
        token = shape, ink.pop()
        assert token not in lookup, "exact colored tokens must be unique"
        tokens[r, c] = token
        lookup[token] = (r, c)
    outside = a.copy()
    for _, (y, x, b, e), _ in rectangles:
        outside[y:b, x:e] = bg
    for _, (y, x, b, e), _ in rectangles:
        assert np.all(outside[y - 1:b + 1, x - 1:e + 1] == bg), "rectangle moat"
    objects = []
    selected = set()
    for cells in _components(outside != bg):
        colors = {int(outside[r, c]) for r, c in cells}
        assert len(colors) == 1, "touching or multicolored exterior glyphs"
        token = _norm(cells), colors.pop()
        assert token in lookup, "every ballot must have an exact board address"
        pos = lookup[token]
        selected.add(pos)
        objects.append((pos, token, cells))
    assert objects, "nonempty ballot set"
    # A one-cell moat makes segmentation visually unambiguous, including diagonals.
    occupied = {p: i for i, (_, _, cells) in enumerate(objects) for p in cells}
    for p, i in occupied.items():
        for dr, dc in itertools.product((-1, 0, 1), repeat=2):
            assert occupied.get((p[0] + dr, p[1] + dc), i) == i, "ballot moat"
    lines, eligible = _complete_lines(selected, nr, nc)
    votes = defaultdict(Counter)
    for pos, (shape, color), cells in objects:
        if pos in eligible:
            votes[_canon(shape)][color] += 1
    cavities = _components(stencil[2] == bg)
    assert cavities and all(_hole_free(_norm(c)) for c in cavities)
    assert all(max(_size(_norm(c))) <= 3 for c in cavities)
    cavity_ids = {p: j for j, cells in enumerate(cavities) for p in cells}
    for (r, c), j in cavity_ids.items():
        for dr, dc in itertools.product((-1, 0, 1), repeat=2):
            assert cavity_ids.get((r + dr, c + dc), j) == j, "cavity separation"
    cavity_shapes = {_canon(c) for c in cavities}
    assert cavity_shapes <= {_canon(s) for s, color in tokens.values()}, "unknown cavity class"
    winners = {s: _plurality(votes[s]) for s in cavity_shapes}
    return dict(grid=a, bg=bg, carrier=carrier, board_bbox=board[1],
                stencil_bbox=stencil[1], stencil=stencil[2], cavities=cavities,
                tokens=tokens, objects=objects, selected=selected, lines=lines,
                eligible=eligible, votes=votes, winners=winners, rows=nr, cols=nc)


def _paint(info, counts):
    out = info["stencil"].copy()
    for cells in info["cavities"]:
        color = _plurality(counts.get(_canon(cells), {}))
        if color is not None:
            for r, c in cells:
                out[r, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    return _paint(info, info["votes"])


def _counterfactual(info, mode):
    """Small omission checks, not an independent semantic oracle."""
    eligible = info["eligible"]
    if mode in ("orientation_blind_lookup", "shape_only_lookup", "ink_only_lookup"):
        def lookup_key(shape, color):
            if mode == "shape_only_lookup":
                return shape
            if mode == "ink_only_lookup":
                return color
            return _canon(shape), color
        keys = {lookup_key(shape, color) for _, (shape, color), _ in info["objects"]}
        selection = {p for p, (s, c) in info["tokens"].items() if lookup_key(s, c) in keys}
        _, eligible = _complete_lines(selection, info["rows"], info["cols"])
    counts = defaultdict(Counter)
    seen = set()
    for pos, (shape, color), cells in info["objects"]:
        if mode != "ignore_lines" and pos not in eligible:
            continue
        if mode == "one_vote_per_address" and pos in seen:
            continue
        seen.add(pos)
        key = len(shape) if mode == "area" else (
            min(_rotations(shape)) if mode == "rotation_only" else _canon(shape))
        weight = sum(pos in line for line in info["lines"]) if mode == "per_line_counting" else 1
        counts[key][color] += weight
    out = info["stencil"].copy()
    try:
        for cells in info["cavities"]:
            key = len(cells) if mode == "area" else (
                min(_rotations(_norm(cells))) if mode == "rotation_only" else _canon(cells))
            color = _plurality(counts.get(key, {}))
            if color is not None:
                for r, c in cells:
                    out[r, c] = color
    except AssertionError:
        return None
    return out


def _differs(info, out, mode):
    predicted = _counterfactual(info, mode)
    return predicted is None or not np.array_equal(predicted, out)


def _fresh_token(rng, bank, inks, used):
    choices = [(v, c) for s in bank for v in _variants(s) for c in inks if (v, c) not in used]
    token = _choose(rng, choices)
    used.add(token)
    return token


def _plan(rng, kind, preferred=None):
    """Construct semantic witnesses before any rendering or parsing."""
    anchor, companion = _anchor_pair(rng, preferred)
    extra = _choose(rng, [s for s in _shape_bank() if s not in (anchor, companion)])
    colors = list(map(int, rng.permutation(10)))
    bg, carrier, inks = colors[0], colors[1], colors[2:]
    a, b, c, d, e, f, g = inks[:7]
    nr = 4 if kind == "double" else (3 if kind in ("lesson", "cross", "near") else int(rng.integers(3, 5)))
    nc = 3 if kind in ("cross", "double") else (4 if kind in ("lesson", "near") else int(rng.integers(3, 5)))
    tokens, used = {}, set()
    shapes = [anchor, companion, extra]

    def put(pos, shape, ink):
        token = shape, ink
        assert token not in used
        used.add(token)
        tokens[pos] = token

    specs = []
    if kind == "cross":
        put((0, 0), anchor, a)
        put((0, 1), anchor, b)
        put((0, 2), companion, d)
        put((1, 0), companion, e)
        put((2, 0), extra, f)
        put((1, 1), anchor, c)
        specs = [((0, 0), 2), ((0, 1), 3), ((0, 2), 2),
                 ((1, 0), 1), ((2, 0), 1), ((1, 1), 4)]
        if rng.random() < .5:
            fourth = _choose(rng, [s for s in _shape_bank() if s not in shapes])
            used.remove(tokens[1, 0])
            put((1, 0), fourth, e)
            shapes.append(fourth)
    elif kind == "double":
        put((0, 0), anchor, a)
        put((1, 0), anchor, b)
        put((0, 1), companion, d)
        put((1, 1), companion, e)
        put((0, 2), extra, f)
        put((1, 2), extra, g)
        put((2, 0), anchor, c)
        specs = [((0, 0), 2), ((1, 0), 1), ((0, 1), 2), ((1, 1), 1),
                 ((0, 2), 2), ((1, 2), 1), ((2, 0), 3)]
    else:
        put((0, 0), anchor, a)
        put((0, 1), anchor, b)
        if nc == 4:
            put((0, 2), anchor, c)
        put((0, nc - 1), companion, d)
        alias = _choose(rng, [s for s in _variants(anchor) if s != anchor])
        put((1, 0), alias, b)
        for col in range(1, nc - 1):
            token = _fresh_token(rng, (extra,), inks[4:], used)
            tokens[1, col] = token
        companion_alias = _choose(rng, [s for s in _rotations(companion) if s != companion])
        put((1, nc - 1), companion_alias, d)
        if kind in ("near", "scatter", "partial"):
            # n-1 queried row cells plus one elsewhere: same distinct count as a line.
            # Omitting one of the same-mask colored competitors also exposes
            # why occupancy shape alone is insufficient for exact lookup.
            omitted = int(rng.integers(nc - 1))
            specs = [((0, col), int(rng.integers(1, 4))) for col in range(nc) if col != omitted]
            specs.append(((1, int(rng.integers(nc))), int(rng.integers(1, 4))))
        else:
            win = 3 if kind == "lesson" else int(rng.integers(2, 4))
            loss = 2 if kind == "lesson" else int(rng.integers(1, win))
            specs = [((0, 0), win), ((0, 1), loss)]
            if nc == 4:
                specs.append(((0, 2), 2 if kind == "lesson" else int(rng.integers(1, win))))
            specs.append(((0, nc - 1), 1))
            # These near-line votes overturn the raw plurality, but cannot vote.
            specs.append(((1, 0), win + 1))
            specs.extend(((1, col), 1) for col in range(1, nc - 1))
    for pos in itertools.product(range(nr), range(nc)):
        if pos not in tokens:
            tokens[pos] = _fresh_token(rng, tuple(shapes), inks, used)
    if kind in ("scatter", "partial"):
        occupancy = nc if kind == "scatter" else int(rng.integers(1, min(nr, nc)))
        all_positions = list(tokens)
        for _ in range(64):
            selected = {all_positions[int(j)] for j in rng.choice(len(all_positions), occupancy, replace=False)}
            lines, _ = _complete_lines(selected, nr, nc)
            diagonal = nr == nc and selected in ({(j, j) for j in range(nr)},
                                                  {(j, nc - 1 - j) for j in range(nr)})
            if not lines and not diagonal:
                specs = [(p, int(rng.integers(1, 4))) for p in sorted(selected)]
                break
        else:
            raise AssertionError("nonline sampling exhausted")
    # Moving whole rows/columns preserves the quorum rule while changing every
    # edge/interior address, the missing position, and the crossing location.
    row_map, col_map = rng.permutation(nr), rng.permutation(nc)
    remap = lambda p: (int(row_map[p[0]]), int(col_map[p[1]]))
    tokens = {remap(p): token for p, token in tokens.items()}
    specs = [(remap(p), n) for p, n in specs]
    selected = {p for p, n in specs}
    lines, eligible = _complete_lines(selected, nr, nc)
    counts = defaultdict(Counter)
    for pos, n in specs:
        if pos in eligible:
            shape, ink = tokens[pos]
            counts[_canon(shape)][ink] += n
    # The third class is deliberately silent in a single-line lesson, but is
    # supported at the crossing. Optional repeats are independent of ballots.
    if kind not in ("lesson", "cross", "double") and rng.random() < .35:
        shapes = shapes[:2]
    cavity_specs = [_flip(anchor)] + [_choose(rng, _variants(s)) for s in shapes[1:]]
    repeat = int(rng.integers(0, 7 - len(cavity_specs)))
    cavity_specs.extend(_choose(rng, _variants(_choose(rng, shapes))) for _ in range(repeat))
    winners = {s: _plurality(counts[s]) for s in shapes}
    return dict(bg=bg, carrier=carrier, tokens=tokens, specs=specs,
                cavity_specs=cavity_specs, winners=winners, rows=nr, cols=nc)


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = False


def _positions(available, shape):
    h, w = _size(shape)
    nr, nc = available.shape[0] - h + 1, available.shape[1] - w + 1
    if nr <= 0 or nc <= 0:
        return []
    valid = np.ones((nr, nc), bool)
    for r, c in shape:
        valid &= available[r:r + nr, c:c + nc]
    return np.argwhere(valid)


def _stencil(rng, plan, max_width):
    shapes = plan["cavity_specs"]
    columns = _choose(rng, [k for k in (1, 2, 3) if 4 * k + 1 <= max_width
                            and 4 * ((len(shapes) + k - 1) // k) + 1 <= 17])
    base_h, base_w = 4 * ((len(shapes) + columns - 1) // columns) + 1, 4 * columns + 1
    h = base_h + int(rng.integers(0, min(2, 17 - base_h) + 1))
    w = base_w + int(rng.integers(0, min(2, max_width - base_w) + 1))
    placements = None
    for _ in range(8):
        available = np.ones((h, w), bool)
        available[[0, -1]] = False
        available[:, [0, -1]] = False
        candidate = []
        for shape in sorted(shapes, key=lambda s: (len(s), _size(s)), reverse=True):
            positions = _positions(available, shape)
            if not len(positions):
                break
            y, x = map(int, _choose(rng, positions))
            cells = tuple((y + r, x + c) for r, c in shape)
            candidate.append((shape, cells))
            _block(available, cells)
        if len(candidate) == len(shapes):
            placements = candidate
            break
    if placements is None:
        placements = []
        for i, shape in enumerate(shapes):
            r, c = divmod(i, columns)
            sh, sw = _size(shape)
            y, x = 1 + 4 * r + int(rng.integers(4 - sh)), 1 + 4 * c + int(rng.integers(4 - sw))
            placements.append((shape, tuple((y + yy, x + xx) for yy, xx in shape)))
    box = np.full((h, w), plan["carrier"], np.int8)
    out = box.copy()
    for shape, cells in placements:
        winner = plan["winners"][_canon(shape)]
        for r, c in cells:
            box[r, c] = plan["bg"]
            out[r, c] = plan["bg"] if winner is None else winner
    assert len(_components(box == plan["carrier"])) == 1
    return box, out


def _render(rng, plan, orientation):
    nr, nc = plan["rows"], plan["cols"]
    # Rectangular slot padding varies where the scene has room.
    ph = 4 if nr == 3 and rng.random() < .35 else 3
    pw = 4 if nc == 3 and rng.random() < .35 else 3
    board_h, board_w = nr * (ph + 1) + 1, nc * (pw + 1) + 1
    stem, out = _stencil(rng, plan, 30 - board_w - 3)
    width_min = board_w + stem.shape[1] + 3
    width = int(rng.integers(width_min, 31))
    board_y = 1
    stem_y = 1 + int(rng.integers(0, min(2, max(0, board_h - stem.shape[0])) + 1))
    board_x = 1
    stem_x = width - stem.shape[1] - 1
    top_end = max(board_y + board_h, stem_y + stem.shape[0])
    ballots = [plan["tokens"][p] for p, count in plan["specs"] for _ in range(count)]
    ballots = [ballots[int(i)] for i in rng.permutation(len(ballots))]
    columns = (width - 1) // 4
    tile_rows = (len(ballots) + columns - 1) // columns
    minimum_height = top_end + 1 + 4 * tile_rows
    # More horizontal room preserves every ballot in narrow/tall scenes.
    if minimum_height > 30:
        width = 30
        stem_x = width - stem.shape[1] - 1
        columns = (width - 1) // 4
        tile_rows = (len(ballots) + columns - 1) // columns
        minimum_height = top_end + 1 + 4 * tile_rows
    assert minimum_height <= 30, "semantic content must fit"
    height = int(rng.integers(minimum_height, 31))
    inp = np.full((height, width), plan["bg"], np.int8)
    board = np.full((board_h, board_w), plan["carrier"], np.int8)
    for (r, c), (shape, ink) in plan["tokens"].items():
        y, x = 1 + r * (ph + 1), 1 + c * (pw + 1)
        board[y:y + ph, x:x + pw] = plan["bg"]
        sh, sw = _size(shape)
        yy, xx = y + (ph - sh) // 2, x + (pw - sw) // 2
        for dr, dc in shape:
            board[yy + dr, xx + dc] = ink
    inp[board_y:board_y + board_h, board_x:board_x + board_w] = board
    inp[stem_y:stem_y + stem.shape[0], stem_x:stem_x + stem.shape[1]] = stem
    placements = None
    for _ in range(6):
        available = np.ones(inp.shape, bool)
        available[:top_end + 1] = False
        available[-1] = False
        available[:, [0, -1]] = False
        candidate = []
        # Large glyphs first improves free placement without imposing a lattice.
        for shape, ink in sorted(ballots, key=lambda v: len(v[0]), reverse=True):
            positions = _positions(available, shape)
            if not len(positions):
                break
            y, x = map(int, _choose(rng, positions))
            cells = tuple((y + r, x + c) for r, c in shape)
            candidate.append((ink, cells))
            _block(available, cells)
        if len(candidate) == len(ballots):
            placements = candidate
            break
    if placements is None:
        placements = []
        slots = list(map(int, rng.permutation(tile_rows * columns)))[:len(ballots)]
        for (shape, ink), index in zip(ballots, slots):
            r, c = divmod(index, columns)
            sh, sw = _size(shape)
            y = top_end + 1 + r * 4 + int(rng.integers(4 - sh))
            x = 1 + c * 4 + int(rng.integers(4 - sw))
            placements.append((ink, tuple((y + yy, x + xx) for yy, xx in shape)))
    for ink, cells in placements:
        for r, c in cells:
            inp[r, c] = ink
    return {"input": _orient(inp, orientation), "output": _orient(out, orientation)}


def _orient(grid, orientation):
    return np.rot90(np.fliplr(grid) if orientation >= 4 else grid, orientation % 4).copy()


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
        if not low <= int(value) <= high:
            raise ValueError("counts outside permitted ranges")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    orientation0 = int(rng.integers(8))
    puzzle, seen, taught_shapes = {"train": [], "test": []}, set(), set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            if split == "train":
                schedule = ("lesson", "near", "lesson", "cross", "partial", "scatter", "double")
                kind = schedule[i] if i < len(schedule) else _choose(rng, ("row", "near", "cross", "double", "scatter", "partial"))
                orientation = orientation0 if i < 2 else (orientation0 + i - 1) % 4 + 4 * (orientation0 // 4)
            else:
                test_kinds = ["row", "near", "cross", "double", "scatter"]
                if nt >= 5:
                    test_kinds.append("partial")
                kind = _choose(rng, test_kinds) if nt >= 4 else _choose(rng, ("lesson", "near"))
                # With two demonstrations, successful tests retain the witnessed axis.
                orientation = orientation0 if nt == 2 else int(rng.integers(8))
            preferred = set(_shape_bank()) - taught_shapes if split == "test" else None
            for _ in range(12):
                plan = _plan(rng, kind, preferred)
                pair = _render(rng, plan, orientation)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("distinct-input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                taught_shapes.update(_canon(s) for s in plan["cavity_specs"])
    # The returned position of an example does not encode its outcome.
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _read(inp)
            assert np.array_equal(_paint(info, info["votes"]), out), "incorrect crop or election"
            assert np.all(out[info["stencil"] == info["carrier"]] == info["carrier"])
            # The two carrier components must account for all carrier pixels.
            area = 0
            for bbox in (info["board_bbox"], info["stencil_bbox"]):
                y, x, b, e = bbox
                area += int(np.count_nonzero(inp[y:b, x:e] == info["carrier"]))
            assert area == int(np.count_nonzero(inp == info["carrier"]))
            analyses[split].append((info, out))
    train = analyses["train"]
    assert any(i["lines"] for i, out in train), "complete-line lesson"
    assert any(not i["lines"] and len(i["selected"]) >= min(i["rows"], i["cols"])
               for i, out in train), "full-count nonline lesson"
    for mode in ("ignore_lines", "one_vote_per_address", "area", "rotation_only", "orientation_blind_lookup",
                 "shape_only_lookup", "ink_only_lookup"):
        assert any(_differs(i, out, mode) for i, out in train), "missing witness: " + mode
    assert any(any(len(v) >= 3 and max(v.values()) * 2 <= sum(v.values())
                   and sorted(v.values(), reverse=True)[1] >= 2 for v in i["votes"].values())
               for i, out in train), "non-majority plurality lesson"
    if any(len(i["lines"]) > 1 for i, out in analyses["test"]):
        assert any(len(i["lines"]) > 1 for i, out in train), "multi-line evidence"
    if len(train) >= 4:
        assert any(_differs(i, out, "per_line_counting") for i, out in train), "intersection deduplication lesson"
    if any(i["rows"] == i["cols"] == 3 for i, out in analyses["test"]):
        assert any(i["rows"] == i["cols"] == 3 for i, out in train), "shorter-line evidence"
    def axes(i):
        return {"row" if len({p[0] for p in line}) == 1 else "column" for line in i["lines"]}
    taught_axes = set().union(*(axes(i) for i, out in train))
    assert all(axes(i) <= taught_axes for i, out in analyses["test"]), "untaught winning axis"
    return None


def self_test():
    """Bounded 42-seed coverage, exact replay, omission and metamorphic checks."""
    signatures, structures, shapes, palettes, dimensions, elections = set(), set(), set(), set(), set(), set()
    boards, cavity_counts, ballot_counts, outcomes = set(), set(), set(), Counter()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        repeated = generate_puzzle(seed)
        signature = tuple((_key(p["input"]), _key(p["output"])) for s in puzzle for p in puzzle[s])
        assert signature == tuple((_key(p["input"]), _key(p["output"])) for s in repeated for p in repeated[s])
        signatures.add(signature)
        for split in puzzle:
            for pair in puzzle[split]:
                i = _read(pair["input"])
                dimensions.add((pair["input"].shape, pair["output"].shape))
                boards.add((i["rows"], i["cols"]))
                palettes.add((i["bg"], i["carrier"]))
                shapes.update(i["winners"])
                cavity_counts.add(len(i["cavities"]))
                ballot_counts.add(len(i["objects"]))
                elections.update(tuple(sorted(v.values(), reverse=True)) for v in i["votes"].values())
                roles = {}
                structures.add(tuple(tuple(roles.setdefault(int(c), len(roles)) for c in row)
                                     for row in pair["input"]))
                outcomes[(split, len(i["lines"]))] += 1
                assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(dimensions) >= 40 and len(palettes) >= 30 and len(shapes) >= 20
    assert {(3, 3), (3, 4), (4, 3)} <= boards
    assert len(cavity_counts) >= 4 and len(ballot_counts) >= 6 and len(elections) >= 6
    assert all(outcomes[split, n] > 0 for split in ("train", "test") for n in (0, 1, 2))
    for orientation in range(8):
        p = puzzles[0]
        transformed = {s: [{k: _orient(v, orientation) for k, v in pair.items()} for pair in p[s]] for s in p}
        validate_generated_puzzle(transformed)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9017 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A supported cavity corruption and a carrier corruption must both fail.
    for carrier_cell in (False, True):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        for pair in damaged["train"]:
            i = _read(pair["input"])
            cells = np.argwhere(pair["output"] == i["carrier"] if carrier_cell else
                                ((pair["output"] != i["bg"]) & (pair["output"] != i["carrier"])))
            if len(cells):
                r, c = cells[0]
                pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
                break
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt output accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), normalized_inputs=len(structures),
                dimension_pairs=len(dimensions), shape_classes=len(shapes),
                role_palettes=len(palettes), board_sizes=sorted(boards),
                cavity_counts=sorted(cavity_counts), ballot_counts=sorted(ballot_counts),
                ballot_patterns=sorted(elections), outcomes={str(k): v for k, v in outcomes.items()})


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
