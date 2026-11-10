"""Certified weave: exact token lookup, complete lines, intersection packing.

Two guide-associated query banks certify different axes of the SAME board.
Their guide/control identities also determine the orientation of the packed
intersection mosaic. All interpretation is recoverable from the input.
"""

from itertools import combinations
import numpy as np


# Sides are top, right, bottom, left. Transform k is a counterclockwise
# rotation by k%4, preceded by a left/right reflection when k>=4.
_MAPS = ((0, 1, 2, 3), (3, 0, 1, 2), (2, 3, 0, 1), (1, 2, 3, 0),
         (0, 3, 2, 1), (3, 2, 1, 0), (2, 1, 0, 3), (1, 0, 3, 2))
_NORMALS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((a >= 0) & (a <= 9))), "grid cells must be integers 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _transform(a, k):
    return np.rot90(np.fliplr(a) if k >= 4 else a, k % 4).copy()


def _side(a, side):
    return (a[0], a[:, -1], a[-1], a[:, 0])[side]


def _paint_side(a, side, color):
    _side(a, side)[1:-1] = color


def _paint_guide(a, side, color, r, c, h, w):
    if side == 0:
        a[r - 1, c:c + w] = color
    elif side == 1:
        a[r:r + h, c + w] = color
    elif side == 2:
        a[r + h, c:c + w] = color
    else:
        a[r:r + h, c - 1] = color


def _bank_boxes(shape, bounds, side, sh, sw):
    """Aligned tiles, nearest the guide first; at most three layers."""
    H, W = shape
    r, c, h, w = bounds
    for depth in range(1, 4):
        if side == 0:
            starts = [(r - 1 - depth * sh, cc) for cc in range(c, c + w, sw)]
        elif side == 2:
            starts = [(r + h + 1 + (depth - 1) * sh, cc)
                      for cc in range(c, c + w, sw)]
        elif side == 3:
            starts = [(rr, c - 1 - depth * sw) for rr in range(r, r + h, sh)]
        else:
            starts = [(rr, c + w + 1 + (depth - 1) * sw)
                      for rr in range(r, r + h, sh)]
        if any(rr < 1 or cc < 1 or rr + sh > H - 1 or cc + sw > W - 1
               for rr, cc in starts):
            break
        yield depth, starts


def _parse_lattice(a, bg, outer, guides, bounds, n, sh, sw):
    r, c, h, w = bounds
    expected = np.full(a.shape, bg, dtype=np.int8)
    for side, color in outer.items():
        _paint_side(expected, side, color)
    for side, color in guides.items():
        _paint_guide(expected, side, color, *bounds)
    lookup, board = {}, []
    for i, j in np.ndindex(n, n):
        rr, cc = r + i * sh, c + j * sw
        tile = a[rr:rr + sh, cc:cc + sw].copy()
        _require(np.all(tile[-1] == bg) and np.all(tile[:, -1] == bg),
                 "tiles must retain their trailing background gutters")
        ink_positions = np.argwhere(tile != bg)
        _require(len(ink_positions) > 0 and len(np.unique(tile[tile != bg])) == 1,
                 "each board tile must contain one nonempty monochrome glyph")
        _require(tuple(ink_positions.min(axis=0)) == (0, 0),
                 "glyphs must be translation normalized to the tile origin")
        _require(not set(np.unique(tile)) & set(outer.values()),
                 "marker colors cannot be token inks")
        token_key = _key(tile)
        _require(token_key not in lookup, "board tokens must be exact-copy unique")
        lookup[token_key] = (i, j)
        board.append(tile)
        expected[rr:rr + sh, cc:cc + sw] = tile
    queries, depths = {}, {}
    for side in guides:
        order, occupied_layers = [], []
        for depth, starts in _bank_boxes(a.shape, bounds, side, sh, sw):
            for rr, cc in starts:
                tile = a[rr:rr + sh, cc:cc + sw]
                if np.all(tile == bg):
                    continue
                token_key = _key(tile)
                _require(token_key in lookup, "query chip must exactly copy a board tile")
                order.append(lookup[token_key])
                occupied_layers.append(depth)
                expected[rr:rr + sh, cc:cc + sw] = tile
        _require(order and len(set(order)) == len(order),
                 "each bank needs distinct nonempty query chips")
        queries[side] = order
        depths[side] = max(occupied_layers)
    _require(np.array_equal(a, expected), "unexpected ink outside the guide-defined lattice and banks")
    row_side = next(s for s in guides if s in (1, 3))
    col_side = next(s for s in guides if s in (0, 2))
    row_query, col_query = set(queries[row_side]), set(queries[col_side])
    rows = [i for i in range(n) if all((i, j) in row_query for j in range(n))]
    cols = [j for j in range(n) if all((i, j) in col_query for i in range(n))]
    _require(rows and cols, "each bank must certify at least one complete line")
    operations = [k for k, mapping in enumerate(_MAPS)
                  if all(outer.get(mapping[s]) == color for s, color in guides.items())]
    _require(len(operations) == 1, "the two side identities must determine one symmetry")
    return dict(grid=a, bg=bg, outer=outer, guides=guides, bounds=bounds,
                n=n, sh=sh, sw=sw, board=board, queries=queries, depths=depths,
                row_side=row_side, col_side=col_side, rows=rows, cols=cols,
                transform=operations[0])


def _decode(value):
    a = _grid(value)
    H, W = a.shape
    _require(min(H, W) >= 10, "scene is too small for the family")
    bg = int(a[0, 0])
    _require(all(int(a[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))),
             "the four outer corners must be background")
    outer = {}
    for side in range(4):
        core = _side(a, side)[1:-1]
        if len(np.unique(core)) == 1 and int(core[0]) != bg:
            outer[side] = int(core[0])
    _require(len(outer) == 2 and len(set(outer.values())) == 2 and
             (list(outer)[0] - list(outer)[1]) % 2 == 1,
             "two distinct adjacent outer controls are required")
    bars = []
    for color in outer.values():
        positions = np.argwhere(a[1:-1, 1:-1] == color) + 1
        _require(len(positions) >= 9, "missing interior guide")
        rr, cc = positions[:, 0], positions[:, 1]
        if len(set(rr)) == 1 and sorted(map(int, cc)) == list(range(int(cc.min()), int(cc.max()) + 1)):
            bars.append(("H", int(rr[0]), int(cc.min()), int(cc.max()), color))
        elif len(set(cc)) == 1 and sorted(map(int, rr)) == list(range(int(rr.min()), int(rr.max()) + 1)):
            bars.append(("V", int(cc[0]), int(rr.min()), int(rr.max()), color))
        else:
            raise ValueError("interior marker pixels must be two isolated straight bars")
    _require({b[0] for b in bars} == {"H", "V"}, "guides must be perpendicular")
    hb = next(b for b in bars if b[0] == "H")
    vb = next(b for b in bars if b[0] == "V")
    _, hr, c0, c1, hc = hb
    _, vc, r0, r1, vc_color = vb
    h, w = r1 - r0 + 1, c1 - c0 + 1
    hs = 0 if hr == r0 - 1 else 2 if hr == r1 + 1 else None
    vs = 3 if vc == c0 - 1 else 1 if vc == c1 + 1 else None
    _require(hs is not None and vs is not None, "guides must span adjacent board sides exactly")
    bounds, guides = (r0, c0, h, w), {hs: hc, vs: vc_color}
    _require(r0 >= 2 and c0 >= 2 and r1 <= H - 3 and c1 <= W - 3,
             "board and banks must be separated from the outer frame")
    candidates = []
    for n in range(3, 6):
        if h % n or w % n or h // n not in (3, 4) or w // n not in (3, 4):
            continue
        try:
            candidates.append(_parse_lattice(a, bg, outer, guides, bounds, n, h // n, w // n))
        except ValueError:
            continue
    _require(len(candidates) == 1, "the tile lattice must have one valid interpretation")
    return candidates[0]


def _packed(info, rows=None, cols=None):
    rows = info["rows"] if rows is None else rows
    cols = info["cols"] if cols is None else cols
    sh, sw = info["sh"], info["sw"]
    r, c, h, w = info["bounds"]
    board = info["grid"][r:r + h, c:c + w]
    rr = [p for i in rows for p in range(i * sh, (i + 1) * sh)]
    cc = [p for j in cols for p in range(j * sw, (j + 1) * sw)]
    return board[np.ix_(rr, cc)].copy()


def _coordinate_output(info, packed=None):
    """Move pixels using the signed axes inferred from the input controls."""
    payload = _packed(info) if packed is None else packed
    matrix = np.zeros((2, 2), dtype=int)
    for side, color in info["guides"].items():
        target = next(s for s, value in info["outer"].items() if value == color)
        matrix += np.outer(_NORMALS[target], _NORMALS[side])
    coordinates = matrix @ np.indices(payload.shape).reshape(2, -1)
    coordinates -= coordinates.min(axis=1, keepdims=True)
    result = np.empty(tuple(coordinates.max(axis=1) + 1), dtype=np.int8)
    result[coordinates[0], coordinates[1]] = payload.ravel()
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode exact query copies and controls from this input alone."""
    return _coordinate_output(_decode(input_grid))


def _mask_bank(gh, gw):
    bank = {}
    for code in range(1, 1 << (gh * gw)):
        mask = np.array([(code >> i) & 1 for i in range(gh * gw)], bool).reshape(gh, gw)
        rr, cc = np.where(mask)
        crop = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        canonical = np.zeros((gh, gw), bool)
        canonical[:crop.shape[0], :crop.shape[1]] = crop
        bank[canonical.tobytes()] = canonical
    return list(bank.values())


def _choose(rng, options):
    return options[int(rng.integers(len(options)))]


def _palette(rng):
    colors = list(map(int, rng.permutation(10)))
    return colors[0], colors[1:3], colors[3:3 + int(rng.integers(3, 6))]


def _line_set(rng, n, count, gappy=False):
    options = list(combinations(range(n), count))
    if gappy and count == 2:
        options = [x for x in options if x[1] - x[0] > 1]
    return list(_choose(rng, options))


def _make_board(rng, n, gh, gw, bg, inks, rows, cols):
    masks = _mask_bank(gh, gw)
    full_bounds = [m for m in masks if m[-1].any() and m[:, -1].any()]
    shape = _choose(rng, full_bounds)
    # The two missing decoy chips are aliases of an intersection chip in
    # occupancy only. Shape-only lookup will falsely complete both lines.
    bad_row = _choose(rng, [i for i in range(n) if i not in rows]) if len(rows) < n else None
    bad_col = _choose(rng, [j for j in range(n) if j not in cols]) if len(cols) < n else None
    anchor = (rows[0], cols[0])
    fixed = {anchor: (shape, inks[0])}
    if bad_row is not None:
        fixed[bad_row, cols[0]] = (shape, inks[1])
    if bad_col is not None:
        fixed[rows[0], bad_col] = (shape, inks[2])
    occupied = {(m.tobytes(), ink) for m, ink in fixed.values()}
    if bad_row is not None and bad_col is not None:
        # Make ink-only failure constructive, not a probabilistic consequence
        # of the random palette: both missing-chip inks also occur on different
        # queried shapes in their respective certified lines.
        row_witness = (rows[0], next(j for j in range(n) if j not in (cols[0], bad_col)))
        col_witness = (next(i for i in range(n) if i not in (rows[0], bad_row)), cols[0])
        for position, ink in ((row_witness, inks[1]), (col_witness, inks[2])):
            choices = full_bounds if position == (0, 0) else masks
            witness = _choose(rng, [m for m in choices if (m.tobytes(), ink) not in occupied])
            fixed[position] = (witness, ink)
            occupied.add((witness.tobytes(), ink))
    pool = [(m, ink) for m in masks for ink in inks if (m.tobytes(), ink) not in occupied]
    # A full-bounds token in the first slot disambiguates 3x4 versus 4x3
    # possible factorings of a twelve-cell board extent. Its shape is random.
    if (0, 0) not in fixed:
        item = _choose(rng, [(m, ink) for m, ink in pool if m[-1].any() and m[:, -1].any()])
        fixed[0, 0] = item
        pool = [(m, ink) for m, ink in pool if (m.tobytes(), ink) != (item[0].tobytes(), item[1])]
    rng.shuffle(pool)
    board = []
    for position in np.ndindex(n, n):
        mask, ink = fixed[position] if position in fixed else pool.pop()
        tile = np.full((gh + 1, gw + 1), bg, np.int8)
        tile[:gh, :gw][mask] = ink
        board.append(tile)
    return board, bad_row, bad_col


def _queries(rng, n, rows, cols, bad_row, bad_col, decoys):
    row_query = {(i, j) for i in rows for j in range(n)}
    col_query = {(i, j) for j in cols for i in range(n)}
    if decoys == "near":
        if bad_row is not None and len(rows) < 3:
            row_query.update((bad_row, j) for j in range(n) if j != cols[0])
        if bad_col is not None and len(cols) < 3:
            col_query.update((i, bad_col) for i in range(n) if i != rows[0])
    elif decoys == "scatter":
        # Add chips while preserving exactly the intended certificate sets.
        for query, axis, certified in ((row_query, 0, rows), (col_query, 1, cols)):
            options = [p for p in np.ndindex(n, n) if p not in query]
            rng.shuffle(options)
            for p in options[:int(rng.integers(0, n + 1))]:
                if len(query) >= 3 * n:
                    break
                candidate = query | {p}
                if any(all(((i, j) if axis == 0 else (j, i)) in candidate
                           for j in range(n)) for i in range(n) if i not in certified):
                    continue
                query.add(p)
    return row_query, col_query


def _render(rng, n, gh, gw, palette, inner, operation, rows, cols, decoys):
    bg, markers, inks = palette
    sh, sw = gh + 1, gw + 1
    h, w = n * sh, n * sw
    board, bad_row, bad_col = _make_board(rng, n, gh, gw, bg, inks, rows, cols)
    rq, cq = _queries(rng, n, rows, cols, bad_row, bad_col, decoys)
    dh, dv = (len(cq) + n - 1) // n, (len(rq) + n - 1) // n
    if dh < 3 and rng.random() < .3:
        dh += 1
    if dv < 3 and rng.random() < .3:
        dv += 1
    core_h, core_w = h + 1 + dh * sh, w + 1 + dv * sw
    H, W = int(rng.integers(core_h + 4, 31)), int(rng.integers(core_w + 4, 31))
    u, v = int(rng.integers(2, H - core_h - 1)), int(rng.integers(2, W - core_w - 1))
    hs, vs = inner
    r = u + dh * sh + 1 if hs == 0 else u
    c = v + dv * sw + 1 if vs == 3 else v
    bounds = r, c, h, w
    guides = {hs: markers[0], vs: markers[1]}
    outer = {_MAPS[operation][side]: color for side, color in guides.items()}
    a = np.full((H, W), bg, np.int8)
    for side, color in outer.items():
        _paint_side(a, side, color)
    for side, color in guides.items():
        _paint_guide(a, side, color, *bounds)
    for i, j in np.ndindex(n, n):
        a[r + i * sh:r + (i + 1) * sh, c + j * sw:c + (j + 1) * sw] = board[i * n + j]
    for side, query, layers in ((hs, cq, dh), (vs, rq, dv)):
        places = [p for depth, starts in _bank_boxes(a.shape, bounds, side, sh, sw)
                  if depth <= layers for p in starts]
        indices = list(map(int, rng.choice(len(places), len(query), replace=False)))
        chips = sorted(query)
        rng.shuffle(chips)
        for index, (i, j) in zip(indices, chips):
            rr, cc = places[index]
            a[rr:rr + sh, cc:cc + sw] = board[i * n + j]
    payload = np.block([[board[i * n + j] for j in cols] for i in rows])
    return {"input": a, "output": _transform(payload, operation)}


def _count(value, name, low, high):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
             and low <= int(value) <= high, f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    _require(nt > ne, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 6))
    sizes = [g for g in (2, 3) if (n + 3) * (g + 1) + 5 <= 30]
    gh, gw = _choose(rng, sizes), _choose(rng, sizes)
    palette, stable = _palette(rng), bool(rng.random() < .55)
    corners = [(hs, vs) for hs in (0, 2) for vs in (1, 3)]
    held = _choose(rng, corners)
    teaching_inner = _choose(rng, [x for x in corners if x != held])
    operation = _choose(rng, [1, 3])
    first_counts = _choose(rng, [(1, 2), (2, 1)])
    first = _render(rng, n, gh, gw, palette, teaching_inner, operation,
                    _line_set(rng, n, first_counts[0], gappy=first_counts[0] == 2),
                    _line_set(rng, n, first_counts[1], gappy=first_counts[1] == 2), "near")
    second = {k: v.copy() for k, v in first.items()}
    info = _decode(first["input"])
    controls = list(info["outer"].items())
    for (side, _), (_, replacement) in zip(controls, controls[::-1]):
        _paint_side(second["input"], side, replacement)
    flipped_operation = next(k for k, m in enumerate(_MAPS)
                             if all(_side(second["input"], m[s])[1] == color
                                    for s, color in info["guides"].items()))
    second["output"] = _transform(_packed(info), flipped_operation)
    train, test = [first, second], []
    seen = {_key(p["input"]) for p in train}
    for index in range(2, nt):
        if index == 2:
            kr, kc = first_counts[::-1]
            decoys = "near"
        elif index == 3:
            kr, kc = _choose(rng, [(1, 1), (1, 1), (3, 1), (1, 3), (2, 2)])
            decoys = "scatter"
        else:
            kr, kc = int(rng.integers(1, min(3, n) + 1)), int(rng.integers(1, min(3, n) + 1))
            decoys = _choose(rng, ["near", "scatter", "none"])
        local = palette if stable else _palette(rng)
        for _ in range(32):
            candidate = _render(rng, n, gh, gw, local, _choose(rng, [x for x in corners if x != held]),
                                int(rng.integers(8)), _line_set(rng, n, kr, gappy=kr == 2),
                                _line_set(rng, n, kc, gappy=kc == 2), decoys)
            if _key(candidate["input"]) not in seen:
                break
        else:
            raise RuntimeError("bounded distinct-input construction exhausted")
        seen.add(_key(candidate["input"]))
        train.append(candidate)
    teaching = [_decode(p["input"]) for p in train]
    counts_r = sorted({len(i["rows"]) for i in teaching})
    counts_c = sorted({len(i["cols"]) for i in teaching})
    seen_counts = {(len(i["rows"]), len(i["cols"])) for i in teaching}
    novel_counts = [(r, c) for r in counts_r for c in counts_c if (r, c) not in seen_counts]
    for index in range(ne):
        kr, kc = (_choose(rng, novel_counts) if novel_counts and index == 0 else
                  (_choose(rng, counts_r), _choose(rng, counts_c)))
        for _ in range(32):
            candidate = _render(rng, n, gh, gw, palette if stable else _palette(rng), held,
                                int(rng.integers(8)), _line_set(rng, n, kr),
                                _line_set(rng, n, kc), _choose(rng, ["near", "scatter", "none"]))
            if _key(candidate["input"]) not in seen:
                break
        else:
            raise RuntimeError("bounded distinct-input construction exhausted")
        seen.add(_key(candidate["input"]))
        test.append(candidate)
    puzzle = {"train": [train[int(i)] for i in rng.permutation(nt)],
              "test": [test[int(i)] for i in rng.permutation(ne)]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_lines(info, mode):
    n, bg = info["n"], info["bg"]
    def feature(tile):
        return (tile != bg).tobytes() if mode == "shape" else int(tile[tile != bg][0])
    matched = {}
    for side, positions in info["queries"].items():
        if mode in ("shape", "color"):
            keys = {feature(info["board"][i * n + j]) for i, j in positions}
            matched[side] = {(i, j) for i, j in np.ndindex(n, n)
                             if feature(info["board"][i * n + j]) in keys}
        else:
            matched[side] = set(positions)
    rq, cq = matched[info["row_side"]], matched[info["col_side"]]
    if mode == "merged":
        rq = cq = rq | cq
    if mode == "any":
        return sorted({i for i, _ in rq}), sorted({j for _, j in cq})
    return ([i for i in range(n) if all((i, j) in rq for j in range(n))],
            [j for j in range(n) if all((i, j) in cq for i in range(n))])


def _has_near(info, axis):
    side = info["row_side"] if axis == 0 else info["col_side"]
    query, n = set(info["queries"][side]), info["n"]
    return any(sum(((i, j) if axis == 0 else (j, i)) in query for j in range(n)) == n - 1
               for i in range(n))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(ne, "num_test", 1, 4)
    _require(nt > ne, "training must outnumber tests")
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _decode(inp)
            information[split].append(info)
            _require(np.array_equal(_coordinate_output(info), out), "wrong packed mosaic or side alignment")
            _require(np.array_equal(np.sort(_packed(info).ravel()), np.sort(out.ravel())),
                     "packed tile pixels must be preserved exactly")
            # A zero-width certificate cannot sneak into a well-shaped output.
            wanted = (len(info["rows"]) * info["sh"], len(info["cols"]) * info["sw"])
            if info["transform"] % 2:
                wanted = wanted[::-1]
            _require(out.shape == wanted, "tile gutters and line counts define the output extent")
    train, test = information["train"], information["test"]
    _require(len({(i["n"], i["sh"], i["sw"]) for i in train + test}) == 1,
             "one puzzle uses one tile lattice convention")
    _require(any(_has_near(i, 0) and _has_near(i, 1) for i in train),
             "training must demonstrate rejection of nearly complete lines on both axes")
    _require(any(any(b - a > 1 for a, b in zip(i["rows"], i["rows"][1:])) or
                 any(b - a > 1 for a, b in zip(i["cols"], i["cols"][1:])) for i in train),
             "training must show deletion of gaps between certified lines")
    contrasts = [(a, b) for x, a in enumerate(train) for b in train[x + 1:]
                 if a["grid"].shape == b["grid"].shape and
                 np.array_equal(a["grid"][1:-1, 1:-1], b["grid"][1:-1, 1:-1]) and
                 set(a["outer"]) == set(b["outer"]) and a["outer"] != b["outer"]]
    _require(contrasts and any(a["transform"] in (1, 3) or b["transform"] in (1, 3)
                               for a, b in contrasts), "training needs a quarter-turn/reflective identity reversal")
    _require(any(not np.array_equal(_coordinate_output(a), _coordinate_output(b)) for a, b in contrasts),
             "the marker reversal must change the actual answer")
    for mode in ("shape", "color", "any", "merged"):
        _require(any(_alternative_lines(i, mode) != (i["rows"], i["cols"]) for i in train),
                 "training does not distinguish the " + mode + " shortcut")
    _require(any(not np.array_equal(_packed(i), _coordinate_output(i)) for i in train),
             "omitting parent B must change a training answer")
    axes_r, axes_c = {len(i["rows"]) for i in train}, {len(i["cols"]) for i in train}
    train_corners = {tuple(sorted(i["guides"])) for i in train}
    for i in test:
        _require(len(i["rows"]) in axes_r and len(i["cols"]) in axes_c,
                 "each tested axis cardinality must be demonstrated")
        _require(tuple(sorted(i["guides"])) not in train_corners,
                 "tests must use a held-out bank/guide corner")


def self_test():
    """Bounded checks on 42 seeds, all explicit counts, and semantic mutations."""
    records, signatures, shapes, selections, palettes = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        _require(all(np.array_equal(p[k], q[k]) for s in puzzle
                     for p, q in zip(puzzle[s], again[s]) for k in p), "seed is not deterministic")
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                records.append((split, info))
                shapes.add(pair["input"].shape)
                selections.add((info["n"], tuple(info["rows"]), tuple(info["cols"])))
                palettes.add((info["bg"], tuple(sorted(info["outer"].values()))))
                # Query order is irrelevant: exchange two occupied chips within
                # a bank without changing which full lines it certifies.
                changed = pair["input"].copy()
                side = info["row_side"]
                chips = [(rr, cc) for _, starts in _bank_boxes(changed.shape, info["bounds"], side,
                                                              info["sh"], info["sw"])
                         for rr, cc in starts if np.any(changed[rr:rr + info["sh"], cc:cc + info["sw"]] != info["bg"])]
                (r1, c1), (r2, c2) = chips[:2]
                x = changed[r1:r1 + info["sh"], c1:c1 + info["sw"]].copy()
                y = changed[r2:r2 + info["sh"], c2:c2 + info["sw"]].copy()
                changed[r1:r1 + info["sh"], c1:c1 + info["sw"]] = y
                changed[r2:r2 + info["sh"], c2:c2 + info["sw"]] = x
                _require(np.array_equal(solve(changed), pair["output"]), "query permutation changed output")
    _require(len(signatures) == 42 and len(shapes) >= 20 and len(selections) >= 45 and len(palettes) >= 20,
             "insufficient measured family variation")
    _require({i["n"] for _, i in records} == {3, 4, 5}, "missing board size")
    _require({(i["sh"], i["sw"]) for _, i in records} >= {(3, 3), (3, 4), (4, 3), (4, 4)},
             "missing rectangular or larger glyph canvases")
    for split in ("train", "test"):
        _require({i["transform"] for s, i in records if s == split} == set(range(8)),
                 "missing dihedral symmetry in " + split)
        _require({tuple(sorted(i["guides"])) for s, i in records if s == split} ==
                 {(0, 1), (0, 3), (1, 2), (2, 3)}, "missing guide corner")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + nt * 10 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (3, False), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0)
    for defect in ("output_cell", "tight_crop", "guide", "query", "duplicate"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        p = bad["test"][0]
        info = _decode(p["input"])
        if defect == "output_cell":
            p["output"][0, 0] = (int(p["output"][0, 0]) + 1) % 10
        elif defect == "tight_crop":
            rr, cc = np.where(p["output"] != info["bg"])
            p["output"] = p["output"][rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()
        elif defect == "guide":
            r, c, h, w = info["bounds"]
            side = info["col_side"]
            p["input"][r - 1 if side == 0 else r + h, c + w // 2] = info["bg"]
        elif defect == "query":
            side = info["row_side"]
            target = _key(info["board"][info["rows"][0] * info["n"] + info["cols"][0]])
            for _, starts in _bank_boxes(p["input"].shape, info["bounds"], side, info["sh"], info["sw"]):
                occupied = [(r, c) for r, c in starts
                            if _key(p["input"][r:r + info["sh"], c:c + info["sw"]]) == target]
                if occupied:
                    r, c = occupied[0]
                    p["input"][r:r + info["sh"], c:c + info["sw"]] = info["bg"]
                    break
        else:
            bad["train"][1] = {k: v.copy() for k, v in bad["train"][0].items()}
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError("validator accepted " + defect)
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "certificate_selections": len(selections),
            "local_role_palettes": len(palettes), "symmetries": 8}


if __name__ == "__main__":
    print(self_test())
