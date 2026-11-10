"""Windows that agree: topology selects a shell; exact lookup qualifies it.

Only standard library and NumPy. No filesystem reads, parent imports, saved
examples, seed-dependent solving, or persistent diagnostic payloads.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _EIGHT if diagonal else _FOUR
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        pending, cells = [first], []
        while pending:
            r, c = pending.pop()
            cells.append((r, c))
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    pending.append(p)
        result.append(cells)
    return result


def _box(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _trim(array, occupied):
    a, b, c, d = _box(list(map(tuple, np.argwhere(occupied).tolist())))
    return array[a:b, c:d].copy()


def _perimeter(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _orient(grid, index):
    return np.rot90(np.fliplr(grid) if index >= 4 else grid, index % 4).copy()


def _glyph_key(glyph, background):
    occupied = glyph != background
    assert np.any(occupied), "empty glyph"
    crop = _trim(glyph, occupied)
    inks = np.unique(crop[crop != background])
    assert len(inks) == 1 and max(crop.shape) <= 2
    return crop.shape, int(inks[0]), (crop != background).tobytes()


def _glyph_bank():
    bank = {}
    for bits in range(1, 16):
        mask = np.array([(bits >> j) & 1 for j in range(4)], dtype=bool).reshape(2, 2)
        crop = _trim(mask, mask)
        bank[(crop.shape, crop.tobytes())] = crop
    return tuple(bank.values())


def _paint_glyph(grid, r, c, mask, ink, footprint=2):
    h, w = mask.shape
    block = grid[r + (footprint-h)//2:r + (footprint-h)//2 + h,
                 c + (footprint-w)//2:c + (footprint-w)//2 + w]
    block[mask] = ink


def _line_kind(positions, rows, cols):
    positions = set(positions)
    for r in range(rows):
        if positions == {r*cols+c for c in range(cols)}:
            return "row"
    for c in range(cols):
        if positions == {r*cols+c for r in range(rows)}:
            return "column"
    return None


def _board_tokens(rng, rows, cols, inks, selected):
    """A selected key has both color-only and shape-only unselected aliases."""
    bank = _glyph_bank()
    shape_index = int(rng.integers(len(bank)))
    other_index = _pick(rng, [i for i in range(len(bank)) if i != shape_index])
    anchor = int(_pick(rng, sorted(selected)))
    outside = list(map(int, rng.choice([i for i in range(rows*cols) if i not in selected], 2, replace=False)))
    assignments = {anchor: (shape_index, 0), outside[0]: (shape_index, 1),
                   outside[1]: (other_index, 0)}
    pool = [(s, i) for s in range(len(bank)) for i in range(len(inks))
            if (s, i) not in assignments.values()]
    rng.shuffle(pool)
    result = []
    for i in range(rows*cols):
        shape, color = assignments[i] if i in assignments else pool.pop()
        result.append((bank[shape].copy(), inks[color]))
    return result


@lru_cache(maxsize=16)
def _room_topologies(k, q):
    """Finite constructive room/corridor possibilities, not a motif bank."""
    edges = ((0, 1), (2, 3), (0, 2), (1, 3))
    options = []
    for bits in range(1, 16):
        rooms = tuple(i for i in range(4) if bits & (1 << i))
        if len(rooms) < max(k, q):
            continue
        available = [e for e in edges if all(i in rooms for i in e)]
        for e_bits in range(1 << len(available)):
            joins = tuple(e for j, e in enumerate(available) if e_bits & (1 << j))
            groups = [{i} for i in rooms]
            for a, b in joins:
                ga = next(g for g in groups if a in g)
                gb = next(g for g in groups if b in g)
                if ga is not gb:
                    groups.remove(ga)
                    groups.remove(gb)
                    groups.append(ga | gb)
            if len(groups) == k:
                options.append((rooms, joins))
    assert options
    return tuple(options)


def _transform_windows(mask, anchors, orientation):
    markers = np.zeros(mask.shape, dtype=np.int8)
    for i, (r, c) in enumerate(anchors, 1):
        markers[r:r+2, c:c+2] = i
    markers = _orient(markers, orientation)
    transformed = []
    for i in range(1, len(anchors)+1):
        a, _, c, _ = _box(list(map(tuple, np.argwhere(markers == i).tolist())))
        transformed.append((a, c))
    return _orient(mask, orientation), transformed


def _windows(rng, side, count, query_count, special=False, bar=False):
    if special or bar:
        # A two-key window, optionally with a diagonally touching EMPTY singleton.
        assert side == 8 and query_count == 2 and count == (2 if special else 1)
        mask = np.zeros((side, side), dtype=bool)
        mask[1:3, 1:6] = True
        if special:
            mask[3, 6] = True
        anchors = [(1, 1), (1, 4)]
    else:
        rooms, joins = _pick(rng, _room_topologies(count, query_count))
        offset_r = int(rng.integers(side-6)) + 1
        offset_c = int(rng.integers(side-6)) + 1
        locations = [(offset_r+3*(i//2), offset_c+3*(i%2)) for i in range(4)]
        mask = np.zeros((side, side), dtype=bool)
        for i in rooms:
            r, c = locations[i]
            mask[r:r+2, c:c+2] = True
        for a, b in joins:
            r, c = locations[a]
            if b-a == 1:
                mask[r:r+2, c+2] = True
            else:
                mask[r+2, c:c+2] = True
        chosen = list(map(int, rng.choice(rooms, query_count, replace=False)))
        anchors = [locations[i] for i in chosen]
        # Empty rooms can be tiny; occupied 2x2 footprints remain intact.
        for i in rooms:
            if i not in chosen and rng.random() < .65:
                r, c = locations[i]
                for dr, dc in ((0, 1), (1, 0), (1, 1)):
                    trial = mask.copy()
                    trial[r+dr, c+dc] = False
                    if len(_components(trial)) == count:
                        mask = trial
        if count == 1 and rng.random() < .30:
            mask[1:-1, 1:-1] = True
        else:
            budget = int(rng.integers((side-2)**2-int(mask.sum())+1))
            for cell in rng.permutation((side-2)**2):
                if budget <= 0:
                    break
                r, c = divmod(int(cell), side-2)
                r, c = r+1, c+1
                if mask[r, c] or not any(mask[r+dr, c+dc] for dr, dc in _FOUR):
                    continue
                trial = mask.copy()
                trial[r, c] = True
                if len(_components(trial)) == count:
                    mask = trial
                    budget -= 1
    assert len(_components(mask)) == count
    return _transform_windows(mask, anchors, int(rng.integers(8)))


def _stamp(rng, side, count, positions, tokens, bg, shell, used, special=False, ring=False, bar=False):
    for attempt in range(48):
        mask, anchors = _windows(rng, side, count, len(positions), special=special, bar=bar)
        if ring:
            mask[1:-1, 1:-1] = True
        if mask.tobytes() not in used:
            break
        assert not special and not ring and not bar, "a fixed teaching mask unexpectedly repeats"
    else:
        # Finite local fallback. No glyph footprint can be removed.
        protected = np.zeros_like(mask)
        for r, c in anchors:
            protected[r:r+2, c:c+2] = True
        found = False
        for r, c in itertools.product(range(1, side-1), repeat=2):
            if protected[r, c]:
                continue
            trial = mask.copy()
            trial[r, c] = not trial[r, c]
            if len(_components(trial)) == count and trial.tobytes() not in used:
                mask, found = trial, True
                break
        assert found, "distinct shell space exhausted"
    used.add(mask.tobytes())
    grid = np.full((side, side), shell, dtype=np.int8)
    grid[mask] = bg
    order = list(map(int, rng.permutation(positions)))
    for (r, c), p in zip(anchors, order):
        glyph, ink = tokens[p]
        _paint_glyph(grid, r, c, glyph, ink)
    return {"grid": grid, "mask": mask, "count": count}


def _pack(rng, shapes):
    """Bounded random placement; all supported configurations have a shelf fallback."""
    for _ in range(40):
        boxes = [None]*len(shapes)
        order = sorted(range(len(shapes)), key=lambda i: -shapes[i][0]*shapes[i][1])
        for i in order:
            h, w = shapes[i]
            origins = []
            for r in range(31-h):
                for c in range(31-w):
                    if all(b is None or r+h < b[0] or b[0]+b[2] < r
                           or c+w < b[1] or b[1]+b[3] < c for b in boxes):
                        origins.append((r, c))
            if not origins:
                break
            r, c = _pick(rng, origins)
            boxes[i] = (r, c, h, w)
        if all(b is not None for b in boxes):
            return boxes
    # A horizontal 9x17 board, 10x20 receiver mosaic and five 8x8 stamps
    # fit in 29x29: one donor beside the mosaic, one beside the board,
    # and three along the bottom. Transpose the placement for vertical boards.
    bh, bw = shapes[0]
    transposed = bh > bw
    h, w = min(bh, bw), max(bh, bw)
    t = min(shapes[1])
    s = shapes[2][0]
    boxes = [(t+1, 0, h, w), (0, 0, t, 2*t), (0, 2*t+1, s, s),
             (t+1, w+1, s, s)]
    boxes += [(t+h+2, i*(s+1), s, s) for i in range(len(shapes)-4)]
    if transposed:
        boxes = [(c, r, w, h) for r, c, h, w in boxes]
    assert all(h == shapes[i][0] and w == shapes[i][1] and r+h <= 30 and c+w <= 30
               for i, (r, c, h, w) in enumerate(boxes))
    return boxes


def _world(rng, rows, cols, side, counts=None, candidates=4, teaching=False, reversal=False):
    colors = list(map(int, rng.permutation(10)))
    bg, divider, shell, carrier = colors[:4]
    inks = colors[4:4+int(rng.integers(2, 7))]
    row = int(rng.integers(rows))
    col = int(rng.integers(cols))
    row_positions = [row*cols+c for c in range(cols)]
    col_positions = [r*cols+col for r in range(rows)]
    short, long = sorted((row_positions, col_positions), key=len)
    tokens = _board_tokens(rng, rows, cols, inks, set(short))
    counts = ([2, 1] if teaching else list(map(int, rng.choice(range(1, 5), 2, replace=False)))) if counts is None else list(counts)
    assert len(counts) == len(set(counts)) == 2
    board = np.full((4*rows+1, 4*cols+1), bg, dtype=np.int8)
    board[::4, :] = divider
    board[:, ::4] = divider
    for p, (glyph, ink) in enumerate(tokens):
        r, c = divmod(p, cols)
        _paint_glyph(board, 4*r+1, 4*c+1, glyph, ink, footprint=3)
    stamps, used = [], set()
    for i, selection in enumerate((short, long)):
        stamps.append(_stamp(rng, side, counts[i], selection, tokens, bg, shell, used,
                             special=teaching and i == 0, ring=teaching and i == 1,
                             bar=reversal and i == 0))
    valid = {stamp["count"]: stamp["mask"].copy() for stamp in stamps}
    decoy_indices = [0, 1] + ([int(rng.integers(2))] if candidates == 5 else [])
    for i in decoy_indices:
        selection = (short, long)[i]
        q = len(selection) if teaching or rng.random() < .7 else int(rng.integers(1, max(rows, cols)+1))
        options = [p for p in itertools.combinations(range(rows*cols), q)
                   if _line_kind(p, rows, cols) is None]
        failure = _pick(rng, options)
        stamps.append(_stamp(rng, side, counts[i], failure, tokens, bg, shell, used))
    rng.shuffle(stamps)
    receiver_shape = (side+2, 2*(side+2)) if rows < cols else (2*(side+2), side+2)
    receiver = np.full(receiver_shape, carrier, dtype=np.int8)
    objects = [board, receiver]+[s["grid"] for s in stamps]
    boxes = _pack(rng, [g.shape for g in objects])
    low_r, low_c = min(b[0] for b in boxes), min(b[1] for b in boxes)
    high_r = max(b[0]+b[2] for b in boxes)
    high_c = max(b[1]+b[3] for b in boxes)
    # Crop with independently varied retained margins, including zero margins.
    low_r = int(rng.integers(low_r+1))
    low_c = int(rng.integers(low_c+1))
    high_r = int(rng.integers(high_r, 31))
    high_c = int(rng.integers(high_c, 31))
    canvas = np.full((high_r-low_r, high_c-low_c), bg, dtype=np.int8)
    for (r, c, h, w), obj in zip(boxes, objects):
        canvas[r-low_r:r-low_r+h, c-low_c:c-low_c+w] = obj
    r, c, _, _ = boxes[1]
    return {"canvas": canvas, "receiver": (r-low_r, c-low_c), "side": side,
            "bg": bg, "carrier": carrier, "shell": shell, "counts": counts,
            "orientation": int(rng.integers(8)), "stamps": stamps,
            "valid": valid, "rows": rows, "cols": cols, "receiver_shape": receiver_shape}


def _render(world, rng):
    # Construction uses its own valid mask map, without invoking the solver.
    grid = world["canvas"].copy()
    r, c = world["receiver"]
    s = world["side"]
    carrier, bg = world["carrier"], world["bg"]
    output = np.full(world["receiver_shape"], carrier, dtype=np.int8)
    for i, count in enumerate(map(int, rng.permutation(world["counts"]))):
        tr, tc = (0, i*(s+2)) if world["rows"] < world["cols"] else (i*(s+2), 0)
        dots = np.zeros((s, s), dtype=bool)
        for p in rng.permutation(s*s):
            a, b = divmod(int(p), s)
            if not any(0 <= a+dr < s and 0 <= b+dc < s and dots[a+dr, b+dc] for dr, dc in _FOUR):
                dots[a, b] = True
                if int(dots.sum()) == count:
                    break
        grid[r+tr+1:r+tr+s+1, c+tc+1:c+tc+s+1][dots] = bg
        mask = world["valid"][count]
        output[tr+1:tr+s+1, tc+1:tc+s+1][~mask] = world["shell"]
    return {"input": _orient(grid, world["orientation"]),
            "output": _orient(output, world["orientation"])}


def _grid(value):
    array = np.asarray(value)
    assert array.ndim == 2 and all(1 <= n <= 30 for n in array.shape)
    assert np.issubdtype(array.dtype, np.integer) and np.all((0 <= array) & (array <= 9))
    return array.astype(np.int8, copy=False)


def _outer_boxes(grid, bg):
    boxes = set(_box(c) for c in _components(grid != bg))
    return sorted(b for b in boxes if not any(a != b and a[0] <= b[0] and a[1] >= b[1]
                  and a[2] <= b[2] and a[3] >= b[3] for a in boxes))


def _parse_background(grid, bg):
    boxes = _outer_boxes(grid, bg)
    assert len(boxes) in (6, 7)
    for i, (a, b, c, d) in enumerate(boxes):
        for e, f, g, h in boxes[i+1:]:
            assert b < e or f < a or d < g or h < c
    boards = [b for b in boxes if (b[1]-b[0]-1) % 4 == 0 and (b[3]-b[2]-1) % 4 == 0
              and sorted(((b[1]-b[0]-1)//4, (b[3]-b[2]-1)//4)) in ([2, 3], [2, 4])]
    assert len(boards) == 1
    a, b, c, d = boards[0]
    board = grid[a:b, c:d]
    assert (board.shape[0]-1) % 4 == (board.shape[1]-1) % 4 == 0
    rows, cols = (board.shape[0]-1)//4, (board.shape[1]-1)//4
    assert sorted((rows, cols)) in ([2, 3], [2, 4])
    divider = int(board[0, 0])
    assert divider != bg and np.all(board[::4] == divider) and np.all(board[:, ::4] == divider)
    lookup, shapes, inks = {}, [], []
    for r, c in np.ndindex(rows, cols):
        slot = board[4*r+1:4*r+4, 4*c+1:4*c+4]
        key = _glyph_key(slot, bg)
        assert key not in lookup
        assert key[1] != divider
        # Translation-normalized and centered in the displayed board slot.
        rr, cc = np.where(slot != bg)
        # Whole-scene symmetries can move the extra padding to the other side.
        assert abs(int(rr.min())-(2-int(rr.max()))) <= 1
        assert abs(int(cc.min())-(2-int(cc.max()))) <= 1
        lookup[key] = r*cols+c
        shapes.append((key[0], key[2]))
        inks.append(key[1])
    squares = [b for b in boxes if b[1]-b[0] == b[3]-b[2]]
    sizes = {b[1]-b[0] for b in squares}
    assert len(sizes) == 1 and len(squares) in (4, 5)
    s = next(iter(sizes))
    assert s in (7, 8)
    receivers = [b for b in boxes if b not in boards and b not in squares]
    assert len(receivers) == 1
    a, b, c, d = receivers[0]
    receiver = grid[a:b, c:d]
    assert sorted(receiver.shape) == [s+2, 2*(s+2)]
    carrier = int(receiver[0, 0])
    assert np.all(_perimeter(receiver) == carrier)
    assert set(map(int, np.unique(receiver))) == {bg, carrier}
    tiles = []
    for i in range(2):
        r, c = (0, i*(s+2)) if receiver.shape[0] < receiver.shape[1] else (i*(s+2), 0)
        tile = receiver[r:r+s+2, c:c+s+2]
        assert np.all(_perimeter(tile) == carrier)
        dots = tile[1:-1, 1:-1] == bg
        requested = int(dots.sum())
        assert 1 <= requested <= 4
        assert not np.any(dots[:-1] & dots[1:]) and not np.any(dots[:, :-1] & dots[:, 1:])
        tiles.append({"r": r, "c": c, "requested": requested})
    assert len({tile["requested"] for tile in tiles}) == 2
    stamps, shell_colors = [], set()
    for a, b, c, d in squares:
        stamp = grid[a:b, c:d]
        shell = int(stamp[0, 0])
        shell_colors.add(shell)
        assert np.all(_perimeter(stamp) == shell)
        windows = stamp != shell
        components = _components(windows)
        assert 1 <= len(components) <= 4
        labels = np.full((s, s), -1, dtype=np.int8)
        for i, cells in enumerate(components):
            for r, c in cells:
                labels[r, c] = i
        query, occupancies = [], [0]*len(components)
        token_boxes = []
        for ink in map(int, np.unique(stamp)):
            if ink in (bg, shell):
                continue
            assert ink in inks
            for cells in _components(stamp == ink, diagonal=True):
                r0, r1, c0, c1 = _box(cells)
                token = np.full((r1-r0, c1-c0), bg, dtype=np.int8)
                for r, c in cells:
                    token[r-r0, c-c0] = ink
                key = _glyph_key(token, bg)
                assert key in lookup
                query.append(lookup[key])
                chamber = int(labels[cells[0]])
                assert chamber >= 0 and all(int(labels[p]) == chamber for p in cells)
                occupancies[chamber] += 1
                token_boxes.append((r0, r1, c0, c1))
        for i, (a, b, c, d) in enumerate(token_boxes):
            for e, f, g, h in token_boxes[i+1:]:
                assert b < e or f < a or d < g or h < c
        assert 1 <= len(query) <= max(rows, cols) and len(query) == len(set(query))
        selected = set(query)
        relevant_shape = any(shapes[p] == shapes[q] for p in selected for q in range(rows*cols) if q not in selected)
        relevant_ink = any(inks[p] == inks[q] and shapes[p] != shapes[q]
                           for p in selected for q in range(rows*cols) if q not in selected)
        stamps.append({"mask": windows, "count": len(components), "query": tuple(sorted(query)),
                       "line": _line_kind(query, rows, cols), "occupancies": occupancies,
                       "eight_count": len(_components(windows, diagonal=True)),
                       "shape_relevant": relevant_shape, "ink_relevant": relevant_ink,
                       "area": int(windows.sum())})
    assert len(shell_colors) == 1
    shell = next(iter(shell_colors))
    assert len({bg, divider, shell, carrier}) == 4
    assert not set(inks) & {bg, divider, shell, carrier}
    assert len({stamp["mask"].tobytes() for stamp in stamps}) == len(stamps)
    line_stamps = [stamp for stamp in stamps if stamp["line"]]
    assert len(line_stamps) == 2 and {stamp["line"] for stamp in line_stamps} == {"row", "column"}
    assert len({stamp["count"] for stamp in line_stamps}) == 2
    assert {tile["requested"] for tile in tiles} == {stamp["count"] for stamp in line_stamps}
    for tile in tiles:
        chosen = [stamp for stamp in line_stamps if stamp["count"] == tile["requested"]]
        assert len(chosen) == 1
        tile["chosen"] = chosen[0]
        same_count = [stamp for stamp in stamps if stamp["count"] == tile["requested"]]
        assert len(same_count) >= 2 and any(not stamp["line"] for stamp in same_count)
    return {"bg": bg, "shell": shell, "carrier": carrier,
            "requested": tuple(tile["requested"] for tile in tiles),
            "side": s, "rows": rows, "cols": cols, "stamps": stamps, "tiles": tiles,
            "receiver_shape": receiver.shape, "receiver_box": receivers[0], "boxes": boxes}


def _analyze(input_grid):
    grid = _grid(input_grid)
    interpretations = []
    for bg in map(int, np.unique(grid)):
        try:
            interpretations.append(_parse_background(grid, bg))
        except (AssertionError, ValueError, IndexError, StopIteration):
            continue
    assert len(interpretations) == 1, "input needs a unique framed-object interpretation"
    return interpretations[0]


def _candidate_output(info, stamp):
    side = info["side"]
    result = np.full((side+2, side+2), info["carrier"], dtype=np.int8)
    result[1:-1, 1:-1][~stamp["mask"]] = info["shell"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer background, topology, lookup, eligibility and answer from pixels."""
    info = _analyze(input_grid)
    return _answer(info)


def _answer(info):
    result = np.full(info["receiver_shape"], info["carrier"], dtype=np.int8)
    size = info["side"]+2
    for tile in info["tiles"]:
        r, c = tile["r"], tile["c"]
        result[r:r+size, c:c+size] = _candidate_output(info, tile["chosen"])
    return result


def _count(value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError("illegal example count")
    return int(value)


def _unique_pair(pair, seen):
    key = pair["input"].shape, pair["input"].tobytes()
    if key not in seen:
        return pair
    # Two present colors have 90 ordered distinct images. There are at most
    # thirteen earlier inputs, so this finite fallback must find a new input.
    first = list(map(int, np.unique(pair["input"])))[:2]
    source_order = first+[i for i in range(10) if i not in first]
    for a, b in itertools.permutations(range(10), 2):
        target_order = [a, b]+[i for i in range(10) if i not in (a, b)]
        table = np.empty(10, dtype=np.int8)
        table[source_order] = target_order
        result = {k: table[v] for k, v in pair.items()}
        if (result["input"].shape, result["input"].tobytes()) not in seen:
            return result
    raise AssertionError("finite uniqueness bound violated")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    configurations, taught_counts, seen = [], set(), set()
    first_config = first_orientation = None
    for index in range(nt):
        if index == 1:
            rows, cols, side = first_config
        else:
            long = int(rng.integers(3, 5))
            rows, cols = (2, long) if rng.random() < .5 else (long, 2)
            side = 8 if index == 0 or long == 4 else int(rng.integers(7, 9))
        counts = [1, 2] if index == 1 else None
        candidates = 4 if index < 2 else int(rng.integers(4, 6))
        world = _world(rng, rows, cols, side, counts=counts, candidates=candidates,
                       teaching=index == 0, reversal=index == 1)
        if index == 0:
            first_config = rows, cols, side
            first_orientation = world["orientation"]
        elif index == 1:
            world["orientation"] = first_orientation
        pair = _unique_pair(_render(world, rng), seen)
        puzzle["train"].append(pair)
        taught_counts.update(world["counts"])
        seen.add((pair["input"].shape, pair["input"].tobytes()))
        configurations.append((rows, cols, side))
    for index in range(ne):
        rows, cols, side = _pick(rng, configurations)
        counts = list(map(int, rng.choice(sorted(taught_counts), 2, replace=False)))
        candidates = 5 if index == 0 else int(rng.integers(4, 6))
        world = _world(rng, rows, cols, side, counts=counts, candidates=candidates)
        pair = _unique_pair(_render(world, rng), seen)
        seen.add((pair["input"].shape, pair["input"].tobytes()))
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = inp.shape, inp.tobytes()
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze(inp)
            assert np.array_equal(_answer(info), out), "wrong shell transplant"
            # Omitting either predicate leaves different possible tile crops.
            for tile in info["tiles"]:
                for candidates in ([s for s in info["stamps"] if s["line"]],
                                   [s for s in info["stamps"] if s["count"] == tile["requested"]]):
                    assert len({_candidate_output(info, s).tobytes() for s in candidates}) >= 2
            analyses[split].append(info)
    train = analyses["train"]
    chosen = [tile["chosen"] for a in train for tile in a["tiles"]]
    assert {s["line"] for s in chosen} == {"row", "column"}, "both axes must be taught"
    assert any(s["count"] != len(s["query"]) for s in chosen), "key-count shortcut unrefuted"
    assert any(0 in s["occupancies"] and max(s["occupancies"]) > 1 for s in chosen)
    assert any(s["eight_count"] < s["count"] for s in chosen), "connectivity witness missing"
    assert any(s["shape_relevant"] and s["ink_relevant"] for s in chosen)
    assert any(x["count"] < y["count"] and x["area"] > y["area"]
               for a in train for x in a["stamps"] if x["line"]
               for y in a["stamps"] if y["line"]), "area-order inversion missing"
    assert any(x["count"] < y["count"] and x["area"] < y["area"]
               for a in train for x in a["stamps"] if x["line"]
               for y in a["stamps"] if y["line"]), "reverse-area shortcut unrefuted"
    assert any(all(not np.array_equal(s["mask"], _orient(s["mask"], j)) for j in range(1, 8))
               for s in chosen), "asymmetric copy witness missing"
    count_lengths = {count: {len(s["query"]) for s in chosen if s["count"] == count}
                     for count in (1, 2)}
    assert all(len(lengths) >= 2 for lengths in count_lengths.values()), "count-to-line-length shortcut unrefuted"
    count_axes = {count: {s["line"] for s in chosen if s["count"] == count} for count in (1, 2)}
    assert all(axes == {"row", "column"} for axes in count_axes.values())
    requested = {count for a in train for count in a["requested"]}
    configs = {(a["side"], tuple(sorted((a["rows"], a["cols"])))) for a in train}
    for a in analyses["test"]:
        assert set(a["requested"]) <= requested
        assert (a["side"], tuple(sorted((a["rows"], a["cols"])))) in configs
    return None


def self_test():
    """Bounded 42-seed coverage, contracts, metamorphisms and corruption checks."""
    metrics = {k: set() for k in ("shape", "board", "side", "count", "candidates", "mask", "selection", "palette")}
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                fingerprint.append(pair["input"].tobytes())
                metrics["shape"].add(pair["input"].shape)
                metrics["board"].add((info["rows"], info["cols"]))
                metrics["side"].add(info["side"])
                metrics["count"].update(info["requested"])
                metrics["candidates"].add(len(info["stamps"]))
                metrics["palette"].add(len(np.unique(pair["input"])))
                for stamp in info["stamps"]:
                    metrics["mask"].add((info["side"], stamp["mask"].tobytes()))
                    metrics["selection"].add((info["rows"], info["cols"], stamp["query"]))
        fingerprints.add(tuple(fingerprint))
    assert len(fingerprints) == 42
    # Six/seven framed regions constrain canvas sizes near ARC's upper bound.
    # Topological and relational diversity are checked separately below.
    assert len(metrics["shape"]) >= 8 and len(metrics["mask"]) >= 100
    assert len(metrics["selection"]) >= 40
    assert len(metrics["board"]) == 4 and metrics["side"] == {7, 8}
    assert metrics["count"] == {1, 2, 3, 4} and metrics["candidates"] == {4, 5}
    assert len(metrics["palette"]) >= 3
    sample = generate_puzzle(0, 2, 1)
    repeated = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for split in sample
               for p, q in zip(sample[split], repeated[split]) for k in p)
    pair = sample["test"][0]
    recolored = _unique_pair(pair, {(pair["input"].shape, pair["input"].tobytes())})
    assert not np.array_equal(recolored["input"], pair["input"])
    assert np.array_equal(solve(recolored["input"]), recolored["output"])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    for orientation in range(8):
        transformed = {s: [{k: _orient(v, orientation) for k, v in p.items()}
                           for p in sample[s]] for s in sample}
        validate_generated_puzzle(transformed)
    permutation = np.array([9, 4, 7, 0, 3, 1, 8, 2, 5, 6], dtype=np.int8)
    validate_generated_puzzle({s: [{k: permutation[v] for k, v in p.items()}
                                  for p in sample[s]] for s in sample})
    for defect in ("output", "duplicate", "dots"):
        damaged = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
        if defect == "output":
            damaged["test"][0]["output"][1, 1] = (int(damaged["test"][0]["output"][1, 1])+1) % 10
        elif defect == "duplicate":
            damaged["train"][1] = {k: v.copy() for k, v in damaged["train"][0].items()}
        else:
            pair = damaged["test"][0]
            a = _analyze(pair["input"])
            r0, r1, c0, c1 = a["receiver_box"]
            block = pair["input"][r0:r1, c0:c1]
            block[block == a["bg"]] = a["carrier"]
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError("accepted damaged " + defect)
    return {"seeds": 42, "distinct_puzzles": len(fingerprints),
            "variation": {k: len(v) for k, v in metrics.items()},
            "independent_solver": False}
