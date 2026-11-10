"""Mirror-cancel loom: exact lookup, a completed-line hinge, XOR tile exchange.

Only standard library and NumPy are used. solve reads its input alone.
Construction sends punches forward; solve looks up reflected source coordinates.
"""
from functools import lru_cache
import itertools

import numpy as np


HOST_SEEDS = (0, 529635760, 875136783, 414283905, 641997319, 372949995,
              204591668, 589770151, 926094860, 282347044, 856703187,
              830717010, 662058873, 540657286, 45610593, 44872173,
              115857524, 692464886, 230436637, 714929386, 621753039,
              243227607, 518426597, 686460328, 763641084, 469506320,
              429135953, 690876169, 637862634, 745698439, 105574433,
              154041930, 921600785, 280064598, 117536455, 774389369,
              329150754, 967495750, 193254946, 51375506, 732150405, 861092641)


def _key(array):
    a = np.asarray(array, dtype=np.int64)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer rectangles of size 1..30, colors 0..9")
    return a.astype(np.int64, copy=False)


def _trim(a, mask):
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("empty token")
    lo, hi = points.min(0), points.max(0) + 1
    return a[lo[0]:hi[0], lo[1]:hi[1]]


def _lines(selected):
    lines = [("row", int(r)) for r in np.flatnonzero(selected.all(axis=1))]
    lines += [("col", int(c)) for c in np.flatnonzero(selected.all(axis=0))]
    return lines


def _mask(shape, positions):
    result = np.zeros(shape, dtype=bool)
    for r, c in positions:
        result[r, c] = True
    return result


def _reflected(point, line):
    r, c = point
    axis, index = line
    return (2 * index - r, c) if axis == "row" else (r, 2 * index - c)


def _forward_output(tile, selected):
    """Construction: each source sends one punch, plus one reflected punch."""
    rows, cols = selected.shape
    lines = _lines(selected)
    if len(lines) > 1:
        raise ValueError("more than one complete line")
    hits = np.zeros(selected.shape, dtype=int)
    for r, c in np.argwhere(selected):
        hits[r, c] += 1
        if lines:
            rr, cc = _reflected((int(r), int(c)), lines[0])
            if 0 <= rr < rows and 0 <= cc < cols:
                hits[rr, cc] += 1
    colors = np.unique(tile)
    swapped = colors.sum() - tile
    d = len(tile)
    out = np.empty((rows * d, cols * d), dtype=int)
    for r, c in np.ndindex(selected.shape):
        out[r*d:(r+1)*d, c*d:(c+1)*d] = swapped if hits[r, c] % 2 else tile
    return out


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    boxes = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        points, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    points.append(p)
                    stack.append(p)
        a = np.array(points)
        lo, hi = a.min(0), a.max(0) + 1
        boxes.append(tuple(map(int, (lo[0], lo[1], hi[0], hi[1]))))
    return boxes


def _lattice(box, frame, background):
    row_edges = np.flatnonzero(np.all(box == frame, axis=1))
    col_edges = np.flatnonzero(np.all(box == frame, axis=0))
    if len(row_edges) < 2 or len(col_edges) < 2:
        raise ValueError("missing lattice edges")
    step = int(row_edges[1] - row_edges[0])
    if (step not in (3, 4) or row_edges[0] != 0 or col_edges[0] != 0
            or row_edges[-1] != len(box)-1 or col_edges[-1] != box.shape[1]-1
            or not np.all(np.diff(row_edges) == step)
            or not np.all(np.diff(col_edges) == step)):
        raise ValueError("irregular glyph lattice")
    rows, cols, g = len(row_edges)-1, len(col_edges)-1, step-1
    tokens = []
    expected = np.full(box.shape, background, dtype=int)
    expected[row_edges, :] = frame
    expected[:, col_edges] = frame
    for r, c in np.ndindex((rows, cols)):
        token = box[r*step+1:(r+1)*step, c*step+1:(c+1)*step].copy()
        inks = np.unique(token[token != background])
        if len(inks) > 1 or frame in inks:
            raise ValueError("glyphs must have one nonstructural ink")
        if len(inks):
            cropped = _trim(token, token != background)
            centered = np.full((g, g), background, dtype=int)
            y, x = (g-cropped.shape[0])//2, (g-cropped.shape[1])//2
            centered[y:y+cropped.shape[0], x:x+cropped.shape[1]] = cropped
            if not np.array_equal(token, centered):
                raise ValueError("glyph placement must be centered")
        tokens.append(token)
        expected[r*step+1:(r+1)*step, c*step+1:(c+1)*step] = token
    if not np.array_equal(box, expected):
        raise ValueError("extra ink on lattice")
    return rows, cols, g, tokens


def _parse_canonical(grid):
    background = int(grid[0, 0])
    dividers = [r for r in range(len(grid))
                if grid[r, 0] != background and np.all(grid[r] == grid[r, 0])]
    if len(dividers) != 1:
        raise ValueError("expected one full-width divider")
    sep = dividers[0]
    frame = int(grid[sep, 0])
    header_boxes = sorted(_components(grid[:sep] == frame), key=lambda b: b[1])
    board_boxes = _components(grid[sep+1:] == frame)
    if len(header_boxes) != 2 or len(board_boxes) != 1:
        raise ValueError("expected tile, query lattice and board")
    tile_box, query_box = header_boxes
    r0, c0, r1, c1 = tile_box
    framed = grid[r0:r1, c0:c1]
    d = r1-r0-2
    if (not 2 <= d <= 4 or framed.shape != (d+2, d+2)
            or not np.all(framed[[0, -1]] == frame)
            or not np.all(framed[:, [0, -1]] == frame)):
        raise ValueError("tile needs explicit square frame")
    tile = framed[1:-1, 1:-1].copy()
    if len(np.unique(tile)) != 2 or frame in tile:
        raise ValueError("tile must have two nonframe colors")
    if c1 >= query_box[1]:
        raise ValueError("header objects must be separated")
    br0, bc0, br1, bc1 = board_boxes[0]
    br0, br1 = br0+sep+1, br1+sep+1
    board_box = (br0, bc0, br1, bc1)
    rows, cols, g, tokens = _lattice(grid[br0:br1, bc0:bc1], frame, background)
    if rows not in (3, 4, 5) or cols not in (3, 4, 5):
        raise ValueError("board dimensions must be 3..5")
    lookup, trimmed = {}, set()
    for p, token in enumerate(tokens):
        if np.all(token == background):
            raise ValueError("board cells may not be empty")
        identity = _key(token)
        trimmed_identity = _key(_trim(token, token != background))
        if identity in lookup or trimmed_identity in trimmed:
            raise ValueError("ambiguous board token")
        lookup[identity] = divmod(p, cols)
        trimmed.add(trimmed_identity)
    qr0, qc0, qr1, qc1 = query_box
    qr, qc, qg, queries = _lattice(grid[qr0:qr1, qc0:qc1], frame, background)
    if qr not in (1, 2, 3) or qg != g:
        raise ValueError("query and board must use the same glyph canvas")
    positions, occupied_slots = [], []
    for slot, token in enumerate(queries):
        if np.all(token == background):
            continue
        identity = _key(token)
        if identity not in lookup:
            raise ValueError("query has no exact board match")
        positions.append(lookup[identity])
        occupied_slots.append(slot)
    if not 1 <= len(positions) <= 9 or len(set(positions)) != len(positions):
        raise ValueError("query must contain 1..9 distinct tokens")
    selected = _mask((rows, cols), positions)
    lines = _lines(selected)
    if len(lines) > 1:
        raise ValueError("multiple completed lines are outside the family")
    expected = np.full(grid.shape, background, dtype=int)
    expected[sep] = frame
    for y0, x0, y1, x1 in (tile_box, query_box, board_box):
        expected[y0:y1, x0:x1] = grid[y0:y1, x0:x1]
    if not np.array_equal(expected, grid):
        raise ValueError("unexpected pixels outside framed objects")
    return dict(tile=tile, selected=selected, line=lines[0] if lines else None,
                rows=rows, cols=cols, g=g, background=background, frame=frame,
                tokens=tokens, queries=queries, positions=positions,
                occupied_slots=occupied_slots, query_box=query_box,
                tray_shape=(qr, qc), canonical=grid)


def _parse(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for turn in range(4):
        try:
            info = _parse_canonical(np.rot90(grid, turn))
        except (ValueError, IndexError):
            continue
        info["turn"] = turn
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("scene orientation must be unambiguous")
    return candidates[0]


def _inverse_parity(info):
    selected = info["selected"]
    result = selected.copy()
    if info["line"]:
        axis, h = info["line"]
        for r, c in np.ndindex(selected.shape):
            sr, sc = (2*h-r, c) if axis == "row" else (r, 2*h-c)
            if 0 <= sr < info["rows"] and 0 <= sc < info["cols"]:
                result[r, c] ^= selected[sr, sc]
    return result


def _woven(info):
    """Dense fabric in the visibly recovered canonical coordinate frame."""
    tile, d = info["tile"], len(info["tile"])
    parity = _inverse_parity(info)
    rr, cc = np.indices((info["rows"]*d, info["cols"]*d))
    original = tile[rr % d, cc % d]
    colors = np.unique(tile)
    output = np.where(parity[rr//d, cc//d], int(colors.sum())-original, original)
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all output-affecting values from the displayed input only."""
    info = _parse(input_grid)
    return np.rot90(_woven(info), -info["turn"]).copy()


@lru_cache(maxsize=2)
def _shape_bank(g):
    bank = {}
    for code in range(1, 1 << (g*g)):
        mask = np.array([(code >> i) & 1 for i in range(g*g)], dtype=bool).reshape(g, g)
        crop = _trim(mask, mask)
        centered = np.zeros((g, g), dtype=bool)
        y, x = (g-crop.shape[0])//2, (g-crop.shape[1])//2
        centered[y:y+crop.shape[0], x:x+crop.shape[1]] = crop
        bank[_key(centered)] = centered
    return tuple(bank.values())


def _colors(rng):
    bg, frame = map(int, rng.choice(10, 2, replace=False))
    inks = list(map(int, rng.choice([c for c in range(10) if c not in (bg, frame)],
                                  int(rng.integers(3, 7)), replace=False)))
    return bg, frame, inks


def _tile(rng, d, background, frame):
    colors = rng.choice([c for c in range(10) if c != frame], 2, replace=False)
    for _ in range(64):
        bits = rng.random((d, d)) < rng.uniform(.2, .8)
        if (bits.any() and not bits.all() and not np.array_equal(bits, np.flipud(bits))
                and not np.array_equal(bits, np.fliplr(bits))):
            return np.where(bits, colors[0], colors[1]).astype(int)
    bits = np.zeros((d, d), dtype=bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(int)


def _board(rng, shape, g, background, inks, selected):
    bank = _shape_bank(g)
    first, second = map(int, rng.choice(len(bank), 2, replace=False))
    positions = list(map(tuple, np.argwhere(selected)))
    outside = list(map(tuple, np.argwhere(~selected)))
    if len(positions) < 2 or not outside:
        special = {}
    else:
        special = {positions[0]: (first, inks[0]), positions[1]: (second, inks[0]),
                   outside[int(rng.integers(len(outside)))]: (first, inks[1])}
    unused = [(s, ink) for s in range(len(bank)) for ink in inks
              if (s, ink) not in special.values()]
    rng.shuffle(unused)
    tokens = []
    for r, c in np.ndindex(shape):
        s, ink = special[(r, c)] if (r, c) in special else unused.pop()
        tokens.append(np.where(bank[s], ink, background).astype(int))
    return tokens


def _witness_selection(rng, n, axis):
    h = int(rng.integers(1, n-1))
    a, b = map(int, rng.choice(n, 2, replace=False))
    positions = [(h, c) for c in range(n)] + [(h-1, a), (h+1, a), (h-1, b)]
    if axis == "col":
        positions = [(c, r) for r, c in positions]
    return _mask((n, n), positions)


def _scatter(rng, rows, cols, count, diagonal=False):
    max_count = rows*cols-max(rows, cols)
    count = min(count, max_count)
    mandatory = [(i, i) for i in range(rows)] if diagonal and rows == cols else []
    count = max(count, len(mandatory))
    pool = [(r, c) for r, c in np.ndindex((rows, cols)) if (r, c) not in mandatory]
    for _ in range(96):
        extra = rng.choice(len(pool), count-len(mandatory), replace=False)
        selected = _mask((rows, cols), mandatory+[pool[int(i)] for i in extra])
        if not _lines(selected):
            return selected
    # Exclude a set covering every row and column, then choose a subset.
    excluded = ({(c % rows, c) for c in range(cols)} if rows <= cols
                else {(r, r % cols) for r in range(rows)})
    if mandatory:
        # The cyclic complement covers all axes without excluding the diagonal.
        excluded = {(r, (r+1) % cols) for r in range(rows)}
    available = [p for p in pool if p not in excluded]
    return _mask((rows, cols), mandatory+available[:count-len(mandatory)])


def _full_selection(rng, rows, cols, axis, edge=False, extra_count=None):
    extent = rows if axis == "row" else cols
    h = int(rng.choice([0, extent-1])) if edge else int(rng.integers(1, extent-1))
    line = ([(h, c) for c in range(cols)] if axis == "row"
            else [(r, h) for r in range(rows)])
    extras = [(r, c) for r, c in np.ndindex((rows, cols)) if (r, c) not in line]
    if not edge:
        # The initial lessons/test regime teaches interior reflection without
        # clipping. Off-center interior axes can otherwise clip distant cells.
        extras = [p for p in extras if all(0 <= x < bound for x, bound in
                  zip(_reflected(p, (axis, h)), (rows, cols)))]
    count = int(rng.integers(1, 5)) if extra_count is None else extra_count
    max_extra = (min((rows-2)*cols, (rows-1)*(cols-1)) if axis == "row" else
                 min((cols-2)*rows, (cols-1)*(rows-1)))
    count = min(count, max_extra, len(extras))
    def permitted(selected):
        return (_lines(selected) == [(axis, h)] and
                _inverse_parity(dict(selected=selected, line=(axis, h), rows=rows, cols=cols)).any())
    for _ in range(96):
        chosen = [extras[int(i)] for i in rng.choice(len(extras), count, replace=False)]
        selected = _mask((rows, cols), line+chosen)
        if permitted(selected):
            return selected
    for chosen in itertools.combinations(extras, count):
        selected = _mask((rows, cols), line+list(chosen))
        if permitted(selected):
            return selected
    raise ValueError("no compatible single-line stencil")


def _layout_options(shape, g, d, count):
    rows, cols = shape
    options = []
    for qr in (1, 2, 3):
        for spare in (0, 1, 2):
            qc = (count+spare+qr-1)//qr
            if qr*qc > count+2:
                continue
            hh = max(d+2, qr*(g+1)+1)
            for margin, gap, hg, bgap in itertools.product((1, 2), (1, 2), (0, 1), (0, 1)):
                width = max(d+2+gap+qc*(g+1)+1, cols*(g+1)+1)+2*margin
                height = margin+hh+hg+1+bgap+rows*(g+1)+1+margin
                if width <= 30 and height <= 30:
                    options.append((qr, qc, margin, gap, hg, bgap, hh, height, width))
    return options


def _render(rng, shape, g, tile, tokens, query_indices, background, frame, orientation):
    rows, cols = shape
    choices = _layout_options(shape, g, len(tile), len(query_indices))
    if not choices:
        raise ValueError("layout does not fit")
    qr, qc, margin, gap, hg, bgap, hh, height, width = choices[int(rng.integers(len(choices)))]
    inp = np.full((height, width), background, dtype=int)
    d, step = len(tile), g+1
    ty = margin+int(rng.integers(hh-(d+2)+1))
    tx = margin
    inp[ty:ty+d+2, tx:tx+d+2] = frame
    inp[ty+1:ty+d+1, tx+1:tx+d+1] = tile
    qx = tx+d+2+gap
    qy = margin+int(rng.integers(hh-(qr*step+1)+1))
    query_order = list(query_indices)
    rng.shuffle(query_order)
    occupied = sorted(map(int, rng.choice(qr*qc, len(query_order), replace=False)))
    query_map = dict(zip(occupied, query_order))
    def draw_lattice(y, x, nr, nc, cell_tokens):
        view = inp[y:y+nr*step+1, x:x+nc*step+1]
        view[::step, :] = frame
        view[:, ::step] = frame
        for p, token in cell_tokens.items():
            r, c = divmod(p, nc)
            view[r*step+1:(r+1)*step, c*step+1:(c+1)*step] = token
    draw_lattice(qy, qx, qr, qc, {p: tokens[j] for p, j in query_map.items()})
    sep = margin+hh+hg
    inp[sep] = frame
    by = sep+1+bgap
    bw = cols*step+1
    bx = int(rng.integers(margin, width-bw-margin+1))
    draw_lattice(by, bx, rows, cols, dict(enumerate(tokens)))
    positions = [divmod(p, cols) for p in query_indices]
    out = _forward_output(tile, _mask(shape, positions))
    return {"input": np.rot90(inp, orientation).copy(),
            "output": np.rot90(out, orientation).copy()}


def _new_pair(rng, shape, axis, edge, count, g, d, orientation):
    selected = (_scatter(rng, *shape, count) if axis is None else
                _full_selection(rng, *shape, axis, edge=edge, extra_count=count))
    bg, frame, inks = _colors(rng)
    tile = _tile(rng, d, bg, frame)
    tokens = _board(rng, shape, g, bg, inks, selected)
    indices = [int(r)*shape[1]+int(c) for r, c in np.argwhere(selected)]
    return _render(rng, shape, g, tile, tokens, indices, bg, frame, orientation)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    n, d = int(rng.integers(4, 6)), int(rng.integers(2, 5))
    g = 2 if n == 5 else int(rng.integers(2, 4))
    axis = str(rng.choice(["row", "col"]))
    other = "col" if axis == "row" else "row"
    orientation = int(rng.integers(4))
    bg, frame, inks = _colors(rng)
    tile = _tile(rng, d, bg, frame)
    selected = _witness_selection(rng, n, axis)
    tokens = _board(rng, (n, n), g, bg, inks, selected)
    indices = [int(r)*n+int(c) for r, c in np.argwhere(selected)]
    first = _render(rng, (n, n), g, tile, tokens, indices, bg, frame, orientation)
    scatter = _scatter(rng, n, n, len(indices), diagonal=True)
    target = [int(r)*n+int(c) for r, c in np.argwhere(scatter)]
    rng.shuffle(target)
    rearranged = [None]*(n*n)
    for old, new in zip(indices, target):
        rearranged[new] = tokens[old].copy()
    rest = [tokens[p].copy() for p in range(n*n) if p not in indices]
    rng.shuffle(rest)
    for p in range(n*n):
        if rearranged[p] is None:
            rearranged[p] = rest.pop()
    # Reuse the entire visible header (including bag order and empty cells),
    # while retaining the second board. Independent layouts would confound the
    # intentionally controlled rearrangement lesson.
    f = _parse(first["input"])
    canon_first = f["canonical"]
    # Render second again using first's scaffold and replace only board glyphs.
    canon_second = canon_first.copy()
    frame_rows = [r for r in range(len(canon_first)) if np.all(canon_first[r] == frame)]
    sep = frame_rows[0]
    b = _components(canon_first[sep+1:] == frame)[0]
    by, bx = b[0]+sep+1, b[1]
    for p, token in enumerate(rearranged):
        r, c = divmod(p, n)
        canon_second[by+r*(g+1)+1:by+(r+1)*(g+1),
                     bx+c*(g+1)+1:bx+(c+1)*(g+1)] = token
    second = {"input": np.rot90(canon_second, orientation).copy(),
              "output": np.rot90(_forward_output(tile, scatter), orientation).copy()}
    train = [first, second]
    tile_alt = int(rng.choice([v for v in (2, 3, 4) if v != d]))
    if nt >= 3:
        train.append(_new_pair(rng, (n, n), other, False, 1, g, tile_alt, orientation))
    alt_shape = (n, n)
    if nt >= 4:
        alt_shape = (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
        if alt_shape == (n, n):
            alt_shape = (3, n)
        train.append(_new_pair(rng, alt_shape, str(rng.choice([axis, other])), True,
                               1, 2, tile_alt, (orientation+1) % 4))
    for _ in range(4, nt):
        shape = (n, n) if rng.random() < .5 else alt_shape
        local_axis = str(rng.choice([axis, other])) if rng.random() < .55 else None
        count = int(rng.integers(1, 5)) if local_axis is None else int(rng.integers(1, 4))
        train.append(_new_pair(rng, shape, local_axis, bool(rng.integers(2)), count,
                               2, int(rng.integers(2, 5)), int(rng.integers(4))))
    infos = [_parse(p["input"]) for p in train]
    row_choices = sorted({i["rows"] for i in infos})
    col_choices = sorted({i["cols"] for i in infos})
    axes = sorted({i["line"][0] for i in infos if i["line"]})
    negative_counts = sorted({len(i["positions"]) for i in infos if not i["line"]})
    test = []
    seen = {_key(p["input"]) for p in train}
    for _ in range(nq):
        for attempt in range(32):
            shape = (int(rng.choice(row_choices)), int(rng.choice(col_choices)))
            local_axis = str(rng.choice(axes)) if rng.random() < .6 else None
            count = int(rng.integers(1, 5)) if local_axis else int(rng.choice(negative_counts))
            if local_axis and nt == 2:
                count = 3
            if local_axis is None:
                compatible = [(r, c) for r in row_choices for c in col_choices
                              if r*c-max(r, c) >= count]
                shape = compatible[int(rng.integers(len(compatible)))]
            local_g = g if nt < 4 else 2
            local_d = d if nt < 3 else int(rng.integers(2, 5))
            turn = orientation if nt < 4 else int(rng.integers(4))
            pair = _new_pair(rng, shape, local_axis, nt >= 4 and bool(rng.integers(2)),
                             count, local_g, local_d, turn)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("could not construct distinct test input")
        seen.add(_key(pair["input"]))
        test.append(pair)
    puzzle = {"train": [train[int(i)] for i in rng.permutation(nt)],
              "test": [test[int(i)] for i in rng.permutation(nq)]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _shape_identity(token, bg):
    return _key(_trim(token != bg, token != bg))


def _color_witness(info):
    shapes = [_shape_identity(t, info["background"]) for t in info["tokens"]]
    chosen = {r*info["cols"]+c for r, c in info["positions"]}
    return any(shapes[p] == shapes[q] for p in chosen
               for q in range(len(shapes)) if q not in chosen)


def _clipped(info):
    return bool(info["line"] and any(
        not (0 <= p[0] < info["rows"] and 0 <= p[1] < info["cols"])
        for p in (_reflected(point, info["line"]) for point in info["positions"])))


def _bag(info):
    return tuple(sorted(_key(q) for q in info["queries"] if np.any(q != info["background"])))


def validate_generated_puzzle(puzzle):
    """Validate labels and teaching evidence reconstructed from returned pixels."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse(inp)
            infos[split].append(info)
            canonical_out = np.rot90(out, info["turn"])
            assert np.array_equal(canonical_out, _woven(info)), "incorrect woven output"
            tile, d = info["tile"], len(info["tile"])
            assert canonical_out.shape == (info["rows"]*d, info["cols"]*d)
            swapped = int(np.unique(tile).sum())-tile
            for r, c in np.ndindex(info["selected"].shape):
                block = canonical_out[r*d:(r+1)*d, c*d:(c+1)*d]
                assert np.array_equal(block, tile) or np.array_equal(block, swapped)
            assert _inverse_parity(info).any(), "uninformative constant fabric"
    training = infos["train"]
    assert any(i["line"] for i in training) and any(not i["line"] for i in training)
    assert any(_color_witness(i) for i in training), "color-sensitive lookup not taught"
    assert any(i["line"] and np.any(_inverse_parity(i) & ~i["selected"])
               and np.any(i["selected"] & ~_inverse_parity(i)) for i in training), "missing fold witness"
    assert any(i["line"] and any(i["selected"][p] and not _inverse_parity(i)[p]
               for p in i["positions"] if (p[0] if i["line"][0] == "row" else p[1]) != i["line"][1])
               for i in training), "satellite cancellation not taught"
    assert any(a["line"] and not b["line"] and _bag(a) == _bag(b)
               and _key(a["tile"]) == _key(b["tile"])
               and a["selected"].shape == b["selected"].shape
               and {_key(t) for t in a["tokens"]} == {_key(t) for t in b["tokens"]}
               for a in training for b in training), "missing controlled board rearrangement"
    axes = {i["line"][0] for i in training if i["line"]}
    negative_counts = {len(i["positions"]) for i in training if not i["line"]}
    for info in infos["test"]:
        assert not info["line"] or info["line"][0] in axes, "unwitnessed fold axis"
        assert not _clipped(info) or any(_clipped(i) for i in training), "clipping not taught"
        assert info["line"] or len(info["positions"]) in negative_counts, "new negative query count"


def _features(pair):
    i = _parse(pair["input"])
    parity = _inverse_parity(i)
    return dict(board_shape=i["selected"].shape, tile_size=len(i["tile"]), glyph_size=i["g"],
                tray=i["tray_shape"], rotation=i["turn"], count=len(i["positions"]),
                axis=i["line"][0] if i["line"] else "none", clipped=_clipped(i),
                added=int(np.sum(parity & ~i["selected"])),
                cancelled=int(np.sum(i["selected"] & ~parity)), odd=int(parity.sum()),
                inks=len({int(t[t != i["background"]][0]) for t in i["tokens"]}),
                palette=tuple(map(int, np.unique(pair["input"]))),
                stencil=_key(i["selected"]), tile=_key(i["tile"]))


def self_test():
    """Bounded structural, semantic and metamorphic checks over 42 host seeds."""
    records, signatures = [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = tuple((_key(p["input"]), _key(p["output"]))
                          for s in ("train", "test") for p in puzzle[s])
        if seed in (HOST_SEEDS[0], HOST_SEEDS[1], HOST_SEEDS[-1]):
            repeated = generate_puzzle(seed)
            assert signature == tuple((_key(p["input"]), _key(p["output"]))
                                      for s in ("train", "test") for p in repeated[s])
        signatures.add(signature)
        records.extend(_features(p) for s in ("train", "test") for p in puzzle[s])
    assert len(signatures) == 42
    assert {f["tile_size"] for f in records} == {2, 3, 4}
    assert {f["glyph_size"] for f in records} == {2, 3}
    assert {f["rotation"] for f in records} == {0, 1, 2, 3}
    assert {f["axis"] for f in records} == {"row", "col", "none"}
    assert len({f["board_shape"] for f in records}) >= 6
    assert len({f["tray"] for f in records}) >= 6
    assert len({f["stencil"] for f in records}) >= 100
    assert len({f["tile"] for f in records}) >= 100
    assert len({f["palette"] for f in records}) >= 15
    assert len({f["count"] for f in records}) >= 5
    assert any(f["clipped"] for f in records)
    assert any(f["added"] and f["cancelled"] for f in records)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(193254946, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    baseline = generate_puzzle(0)
    for pair in baseline["train"]+baseline["test"]:
        inp, out = pair["input"], pair["output"]
        for turn in range(4):
            assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn))
        permutation = np.random.default_rng(300).permutation(10)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        info = _parse(inp)
        shuffled = info["canonical"].copy()
        y0, x0, _, _ = info["query_box"]
        step, qc = info["g"]+1, info["tray_shape"][1]
        for p, token in enumerate(reversed(info["queries"])):
            r, c = divmod(p, qc)
            shuffled[y0+r*step+1:y0+(r+1)*step, x0+c*step+1:x0+(c+1)*step] = token
        assert np.array_equal(solve(np.rot90(shuffled, -info["turn"])), out)
    # Hand-calculated macro parity: a rail, symmetric pair, and unpaired point.
    s = _mask((4, 4), [(1, c) for c in range(4)]+[(0, 0), (2, 0), (0, 3)])
    info = dict(selected=s, rows=4, cols=4, line=("row", 1))
    assert np.array_equal(_inverse_parity(info), _mask((4, 4), [(0, 3), (2, 3)]))
    s = _mask((4, 4), [(0, c) for c in range(4)]+[(2, 1)])
    info.update(selected=s, line=("row", 0))
    assert np.array_equal(_inverse_parity(info), _mask((4, 4), [(2, 1)]))
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in baseline[s]] for s in baseline}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": len(records), "board_shapes": len({f["board_shape"] for f in records}),
            "tray_shapes": len({f["tray"] for f in records}),
            "distinct_stencils": len({f["stencil"] for f in records}),
            "tile_sizes": sorted({f["tile_size"] for f in records}),
            "query_counts": sorted({f["count"] for f in records})}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
