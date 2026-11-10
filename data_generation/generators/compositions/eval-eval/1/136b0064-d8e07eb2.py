"""Line-bundle plotter: exact lookup turns complete board lines into strokes.

No training data or generation metadata is used by solve. The displayed board
is both an address dictionary and the source of each stroke's ordered pigments.
"""

import itertools

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is outside its permitted range")
    return value


def _counts(rng, num_train, num_test):
    # Explicit defaults are 4/1. None is additionally supported for local tools.
    nt = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    nq = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b
             and (nt is None or a == nt) and (nq is None or b == nq)]
    if not legal:
        raise ValueError("num_train must exceed num_test")
    return legal[int(rng.integers(len(legal)))] if nt is None or nq is None else (nt, nq)


def _key(array):
    array = np.asarray(array)
    return array.shape, array.astype(np.int8).tobytes()


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any(array < 0) or np.any(array > 9)):
        raise ValueError("grids must be integer rectangles of size 1..30, colors 0..9")
    return array


def _runs(present):
    result = []
    for v in np.flatnonzero(present):
        v = int(v)
        if result and result[-1][1] == v:
            result[-1][1] += 1
        else:
            result.append([v, v + 1])
    return result


def _line(positions, rows, cols):
    """An exact full row/column, without extras; query order is immaterial."""
    selected = set(positions)
    for r in range(rows):
        ordered = [r * cols + c for c in range(cols)]
        if selected == set(ordered):
            return "row", ordered
    for c in range(cols):
        ordered = [r * cols + c for r in range(rows)]
        if selected == set(ordered):
            return "col", ordered
    return None, []


def _parse(input_grid):
    """Infer the board, bundle lattice, palette, pane and seed from pixels."""
    inp = _grid(input_grid)
    bg = int(inp[0, 0])
    bars = [c for c in range(1, inp.shape[1] - 1)
            if int(inp[0, c]) != bg and np.all(inp[:, c] == inp[0, c])]
    if len(bars) != 1:
        raise ValueError("expected one full-height source/pane divider")
    divider = bars[0]
    sep = int(inp[0, divider])
    panel, pane = inp[:, :divider], inp[:, divider + 1:]
    seed_cells = np.argwhere(pane != bg)
    if seed_cells.shape != (1, 2) or int(seed_cells[0, 0]) != 0:
        raise ValueError("the pane must contain just one top-row seed")
    start = int(seed_cells[0, 1])
    marker = int(pane[0, start])
    if marker == sep:
        raise ValueError("seed and divider roles must differ")
    cuts = [r for r in range(len(panel)) if np.all(panel[r] == sep)]
    if len(cuts) != 1:
        raise ValueError("expected one board/program horizontal divider")
    cut = cuts[0]
    inventory = panel[:cut]
    rr = _runs(np.any(inventory != bg, axis=1))
    cc = _runs(np.any(inventory != bg, axis=0))
    rows, cols = len(rr), len(cc)
    if not (2 <= rows <= 4 and 2 <= cols <= 4):
        raise ValueError("board dimensions must be 2..4")
    gh, gw = rr[0][1] - rr[0][0], cc[0][1] - cc[0][0]
    capacity = max(rows, cols)
    if not (2 <= gh <= 3 and 2 <= gw <= 3):
        raise ValueError("glyph canvases must be 2..3 cells per axis")
    if (rr != [[1 + r * (gh + 1), 1 + r * (gh + 1) + gh] for r in range(rows)]
            or cc != [[1 + c * (gw + 1), 1 + c * (gw + 1) + gw] for c in range(cols)]
            or cut != rows * (gh + 1) + 1
            or divider != capacity * (gw + 1) + 1):
        raise ValueError("irregular board lattice or margins")

    expected = np.full(inp.shape, bg, dtype=np.int8)
    expected[:, divider] = sep
    expected[cut, :divider] = sep
    expected[0, divider + 1 + start] = marker
    tokens, inks, lookup = [], [], {}

    def token_at(top, left, allow_empty=False):
        token = inp[top:top + gh, left:left + gw]
        mask = token != bg
        if not np.any(mask) and allow_empty:
            return None
        ink = np.unique(token[mask])
        if (token.shape != (gh, gw) or len(ink) != 1
                or int(ink[0]) in (sep, marker)
                or not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1))):
            raise ValueError("glyphs must be monochrome and occupy every canvas row/column")
        expected[top:top + gh, left:left + gw] = token
        return token.copy(), int(ink[0])

    for r, c in np.ndindex((rows, cols)):
        token, ink = token_at(1 + r * (gh + 1), 1 + c * (gw + 1))
        key = _key(token)
        if key in lookup:
            raise ValueError("board keys must be unique in both shape and ink")
        lookup[key] = r * cols + c
        tokens.append(token)
        inks.append(ink)
    program0 = cut + 1
    bands = _runs(np.any(panel[program0:] != bg, axis=1))
    if not bands or bands != [[1 + j * (gh + 1), 1 + j * (gh + 1) + gh]
                             for j in range(len(bands))]:
        raise ValueError("irregular or empty bundle program")
    if program0 + len(bands) * (gh + 1) >= len(inp):
        raise ValueError("missing bottom margin")
    queries, slots = [], []
    for j in range(len(bands)):
        query, occupied = [], []
        for c in range(capacity):
            tok = token_at(program0 + 1 + j * (gh + 1), 1 + c * (gw + 1), True)
            if tok is not None:
                key = _key(tok[0])
                if key not in lookup:
                    raise ValueError("a query has no exact board copy")
                query.append(lookup[key])
                occupied.append(c)
        if not query or len(query) != len(set(query)):
            raise ValueError("a bundle must contain distinct, nonempty board keys")
        queries.append(query)
        slots.append(occupied)
    if not np.array_equal(inp, expected):
        raise ValueError("unexpected ink outside the board and bundle canvases")
    return {"input": inp, "pane": pane.copy(), "bg": bg, "sep": sep,
            "marker": marker, "start": start, "rows": rows, "cols": cols,
            "gh": gh, "gw": gw, "divider": divider, "cut": cut,
            "program0": program0, "tokens": tokens, "inks": inks,
            "queries": queries, "slots": slots,
            "kinds": [_line(q, rows, cols)[0] for q in queries]}


def _plot(pane, start, bg, rows, cols, inks, queries,
          chain=True, gate=True, board_order=True, blank_step=False):
    """The three switches are used only for explicit shortcut counterchecks."""
    out = pane.copy()
    endpoint_r, endpoint_c = 0, start
    for query in queries:
        kind, ordered = _line(query, rows, cols)
        if kind is None:
            if blank_step:
                endpoint_r += 1
            if gate:
                continue
            kind, ordered = "row", sorted(query)
        if not board_order:
            ordered = list(query)
        r, c = (endpoint_r, endpoint_c) if chain else (0, start)
        for offset, index in enumerate(ordered):
            y = r + 1 + (offset if kind == "col" else 0)
            x = c + (offset if kind == "row" else 0)
            if not (0 <= y < out.shape[0] and 0 <= x < out.shape[1]):
                raise ValueError("a stroke extends outside the drawing pane")
            out[y, x] = inks[index]
        endpoint_r = r + (len(ordered) if kind == "col" else 1)
        endpoint_c = c + (len(ordered) - 1 if kind == "row" else 0)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _plot(info["pane"], info["start"], info["bg"], info["rows"],
                 info["cols"], info["inks"], info["queries"])


def _shape_bank(gh, gw):
    # Connectivity is deliberately irrelevant. Solid, hollow and diagonal masks
    # are included whenever their size admits them. All have full bounding boxes.
    bank = []
    for bits in range(1, 1 << (gh * gw)):
        mask = np.array([(bits >> i) & 1 for i in range(gh * gw)], bool).reshape(gh, gw)
        if np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)):
            bank.append(mask)
    return bank


def _board_signature(board):
    return tuple((_key(mask), int(ink)) for mask, ink in board)


def _make_board(rng, rows, cols, bank, palette, primary_row, forbidden):
    """Fresh keys, with selected/unselected shape and color confounders."""
    last = None
    for _ in range(32):
        offset = int(rng.integers(len(palette)))
        colors = [palette[(r + c + offset) % len(palette)]
                  for r, c in np.ndindex((rows, cols))]
        line = [primary_row * cols + c for c in range(cols)]
        outside = [i for i in range(rows * cols) if i not in line]
        # Preserve balanced per-ink capacities, but break the cyclic spatial
        # pattern. All lines stay multicolored and the selected row retains an
        # off-row ink competitor. The original arrangement is a finite fallback.
        for _ in range(16):
            shuffled = list(colors)
            rng.shuffle(shuffled)
            multi_rows = all(len({shuffled[r * cols + c] for c in range(cols)}) > 1
                             for r in range(rows))
            multi_cols = all(len({shuffled[r * cols + c] for r in range(rows)}) > 1
                             for c in range(cols))
            if multi_rows and multi_cols and any(shuffled[i] == shuffled[j]
                                                for i in line for j in outside):
                colors = shuffled
                break
        eligible = [i for i in line if any(colors[i] == colors[j] for j in outside)]
        p = int(rng.choice(eligible))
        q = int(rng.choice([j for j in outside if colors[j] != colors[p]]))
        special = [i for i, m in enumerate(bank) if np.all(m)
                   or np.array_equal(m, np.indices(m.shape).sum(0) % 2 == 0)
                   or (m.shape == (3, 3) and int(m.sum()) == 8 and not m[1, 1])]
        shared = int(rng.choice(special if rng.random() < .5 else range(len(bank))))
        ids, used = {}, {ink: set() for ink in palette}
        ids[p] = ids[q] = shared
        used[colors[p]].add(shared)
        used[colors[q]].add(shared)
        for i in rng.permutation(rows * cols):
            i = int(i)
            if i not in ids:
                options = [k for k in range(len(bank)) if k not in used[colors[i]]]
                ids[i] = int(rng.choice(options))
                used[colors[i]].add(ids[i])
        board = [(bank[ids[i]].copy(), colors[i]) for i in range(rows * cols)]
        if _board_signature(board) not in forbidden:
            forbidden.add(_board_signature(board))
            return board
        last = (ids, colors)
    # Deterministic, finite completion: relabel only used masks injectively.
    # At least two mask IDs are used, even on 2x2 boards; the smallest bank has
    # seven entries, hence at least 42 distinct relabelings for <=14 pairs.
    ids, colors = last
    old = sorted(set(ids.values()))
    for targets in itertools.permutations(range(len(bank)), len(old)):
        remap = dict(zip(old, targets))
        board = [(bank[remap[ids[i]]].copy(), colors[i]) for i in range(rows * cols)]
        signature = _board_signature(board)
        if signature not in forbidden:
            forbidden.add(signature)
            return board
    raise RuntimeError("exhausted board vocabulary")


def _near_miss(rng, rows, cols, axis=None, inks=None, axis_index=None):
    # A complete line minus one key plus one off-line key. This has exactly the
    # same occupancy as a winning bundle, so token-count shortcuts fail.
    axes = [axis] if axis is not None else ["row", "col"]
    options, same_ink = [], []
    for kind in axes:
        for k in range(rows if kind == "row" else cols):
            if axis_index is not None and k != axis_index:
                continue
            line = ([k * cols + c for c in range(cols)] if kind == "row"
                    else [r * cols + k for r in range(rows)])
            for missing in line:
                for extra in range(rows * cols):
                    if extra not in line:
                        pos = [p for p in line if p != missing] + [extra]
                        if _line(pos, rows, cols)[0] is None:
                            options.append(pos)
                            if inks is not None and inks[missing] == inks[extra]:
                                same_ink.append(pos)
    choices = same_ink or options
    return choices[int(rng.integers(len(choices)))].copy()


def _program(rng, rows, cols, count, primary_row, row_first, inks):
    rquery = [primary_row * cols + c for c in range(cols)]
    if rows == cols == 2:
        # On the smallest board, a same-ink off-row replacement may complete a
        # column. Instead teach that the same ink multiset can mean either axis.
        choices = [c for c in range(cols)
                   if sorted(inks[p] for p in rquery)
                   == sorted(inks[r * cols + c] for r in range(rows))]
        col = int(rng.choice(choices))
    else:
        col = int(rng.integers(cols))
    cquery = [r * cols + col for r in range(rows)]
    near = _near_miss(rng, rows, cols, "row", inks, primary_row)
    queries = [rquery, near, cquery] if row_first else [cquery, near, rquery]
    while len(queries) < count:
        kind = int(rng.integers(3))
        if kind == 0:
            r = int(rng.integers(rows))
            q = [r * cols + c for c in range(cols)]
        elif kind == 1:
            c = int(rng.integers(cols))
            q = [r * cols + c for r in range(rows)]
        else:
            q = _near_miss(rng, rows, cols, inks=inks)
        queries.insert(int(rng.integers(len(queries) + 1)), q)
    # Every primary-row query is visibly shuffled in a way that changes its
    # cyclic color sequence. Other groups include both ordered and shuffled keys.
    for j, q in enumerate(queries):
        if q == rquery:
            queries[j] = q[1:] + q[:1]
        elif rng.random() < .75:
            rng.shuffle(q)
    return queries


def _render_pair(rng, rows, cols, gh, gw, board, queries, roles):
    bg, sep, marker = roles
    capacity = max(rows, cols)
    panel_w = capacity * (gw + 1) + 1
    cut = rows * (gh + 1) + 1
    program0 = cut + 1
    base_h = (rows + len(queries)) * (gh + 1) + 3
    height = base_h + int(rng.integers(min(2, 30 - base_h) + 1))
    active = [_line(q, rows, cols) for q in queries]
    displacement = sum(cols - 1 for kind, _ in active if kind == "row")
    width_limit = 30 - panel_w - 1
    width = displacement + 1 + int(rng.integers(min(5, width_limit - displacement - 1) + 1))
    start = int(rng.integers(width - displacement))
    inp = np.full((height, panel_w + 1 + width), bg, np.int8)
    inp[:, panel_w] = sep
    inp[cut, :panel_w] = sep
    inp[0, panel_w + 1 + start] = marker
    for i, (mask, ink) in enumerate(board):
        r, c = divmod(i, cols)
        block = inp[1 + r * (gh + 1):1 + r * (gh + 1) + gh,
                    1 + c * (gw + 1):1 + c * (gw + 1) + gw]
        block[mask] = ink
    for j, query in enumerate(queries):
        slots = sorted(int(v) for v in rng.choice(capacity, len(query), replace=False))
        for slot, index in zip(slots, query):
            mask, ink = board[index]
            top = program0 + 1 + j * (gh + 1)
            left = 1 + slot * (gw + 1)
            block = inp[top:top + gh, left:left + gw]
            block[mask] = ink
    # Construction uses the sampled board and references. solve independently
    # recovers those references from pixels; neither receives a latent decoder.
    out = _plot(inp[:, panel_w + 1:], start, bg, rows, cols,
                [ink for _, ink in board], queries)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    nt, nq = _counts(rng, num_train, num_test)
    layouts = [(r, c, gh, gw) for r in range(2, 5) for c in range(2, 5)
               for gh in (2, 3) for gw in (2, 3)
               if (r + 4) * (gh + 1) + 3 <= 30]
    rows, cols, gh, gw = layouts[int(rng.integers(len(layouts)))]
    pane_limit = 30 - (max(rows, cols) * (gw + 1) + 1) - 1
    # Two mandatory groups are a column and a failure. Even if every other
    # group is horizontal, its full displacement must fit beside the panel.
    max_commands = min(6, 27 // (gh + 1) - rows,
                       2 + (pane_limit - 1) // (cols - 1))
    train_cap = 3 if max_commands == 4 else int(rng.integers(3, 5))
    roles = tuple(int(v) for v in rng.choice(10, 3, replace=False))
    available = [v for v in range(10) if v not in roles]
    palette = [int(v) for v in rng.choice(available, int(rng.integers(3, 7)), replace=False)]
    bank = _shape_bank(gh, gw)
    forbidden, result = set(), {"train": [], "test": []}
    first_row_first = bool(rng.integers(2))
    for split, count in (("train", nt), ("test", nq)):
        for j in range(count):
            commands = (3 if j < 2 else int(rng.integers(3, train_cap + 1))) if split == "train" else int(rng.integers(train_cap + 1, max_commands + 1))
            primary_row = int(rng.integers(rows))
            board = _make_board(rng, rows, cols, bank, palette, primary_row, forbidden)
            row_first = (first_row_first if j == 0 else not first_row_first) if split == "train" and j < 2 else bool(rng.integers(2))
            queries = _program(rng, rows, cols, commands, primary_row, row_first,
                               [ink for _, ink in board])
            result[split].append(_render_pair(rng, rows, cols, gh, gw, board, queries, roles))
    validate_generated_puzzle(result)
    return result


def _different(info, expected, **switches):
    try:
        altered = _plot(info["pane"], info["start"], info["bg"], info["rows"],
                        info["cols"], info["inks"], info["queries"], **switches)
    except ValueError:
        return True  # The competing rule even leaves the supplied pane.
    return not np.array_equal(altered, expected)


def _confounders(info):
    masks = [_key(t != info["bg"]) for t in info["tokens"]]
    shape_needed = color_needed = False
    for query, kind in zip(info["queries"], info["kinds"]):
        if kind is None:
            continue
        outside = set(range(len(masks))) - set(query)
        color_needed |= any(masks[p] == masks[q] and info["inks"][p] != info["inks"][q]
                            for p in query for q in outside)
        shape_needed |= any(masks[p] != masks[q] and info["inks"][p] == info["inks"][q]
                            for p in query for q in outside)
    return shape_needed, color_needed


def _matching_changes(info, expected, compare_shape):
    """Replace a selected key by a confusable unselected key, then execute."""
    masks = [_key(t != info["bg"]) for t in info["tokens"]]
    for j, (query, kind) in enumerate(zip(info["queries"], info["kinds"])):
        if kind is None:
            continue
        for p in query:
            for q in sorted(set(range(len(masks))) - set(query)):
                confusable = ((masks[p] == masks[q] and info["inks"][p] != info["inks"][q])
                              if compare_shape else
                              (masks[p] != masks[q] and info["inks"][p] == info["inks"][q]))
                if confusable:
                    queries = [list(group) for group in info["queries"]]
                    queries[j] = [q if v == p else v for v in query]
                    changed = dict(info, queries=queries)
                    if _different(changed, expected):
                        return True
    return False


def _ink_bag_contrast(info):
    bags = [tuple(sorted(info["inks"][p] for p in q)) for q in info["queries"]]
    for i, a in enumerate(info["kinds"]):
        for j, b in enumerate(info["kinds"]):
            if bags[i] == bags[j] and a != b and (a is not None or b is not None):
                return True
    return False


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must contain exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _integer(nt, "num_train", 2, 10)
    _integer(nq, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    seen, boards, conventions, lengths = set(), set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("pairs must contain exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(solve(inp.copy()), out), "incorrect output"
            assert out.shape == info["pane"].shape
            assert np.array_equal(out[0], info["pane"][0]), "seed or top margin changed"
            assert {"row", "col", None} <= set(info["kinds"])
            assert any(k is None and any(t is not None for t in info["kinds"][i + 1:])
                       for i, k in enumerate(info["kinds"])), "skip behavior is not observable"
            assert all(_confounders(info)), "shape-and-color lookup is not demonstrated"
            assert _matching_changes(info, out, True), "omitting ink does not change the answer"
            assert _matching_changes(info, out, False), "omitting shape does not change the answer"
            assert _ink_bag_contrast(info), "ink multisets could classify all bundle commands"
            assert _different(info, out, chain=False), "chaining can be omitted"
            assert _different(info, out, gate=False), "completion gate can be omitted"
            assert _different(info, out, board_order=False), "query order can replace board order"
            assert _different(info, out, blank_step=True), "failed commands could advance the pen"
            # Stroke count, pigments and blank margins are substantive invariants.
            active = [_line(q, info["rows"], info["cols"]) for q in info["queries"]]
            expected_pixels = 1 + sum(len(q) for k, q in active if k)
            assert int(np.count_nonzero(out != info["bg"])) == expected_pixels
            boardkey = tuple(_key(t) for t in info["tokens"])
            assert boardkey not in boards, "generated pairs must have fresh board maps"
            boards.add(boardkey)
            conventions.add(tuple(info[k] for k in ("bg", "sep", "marker", "rows", "cols", "gh", "gw")))
            lengths[split].append(len(info["queries"]))
    assert len(conventions) == 1, "puzzle-level palette roles and lattice changed"
    assert min(lengths["test"]) > max(lengths["train"]), "missing longer-program test shift"


def puzzle_features(puzzle):
    infos = [_parse(p["input"]) for s in ("train", "test") for p in puzzle[s]]
    return {"board": (infos[0]["rows"], infos[0]["cols"]),
            "glyph": (infos[0]["gh"], infos[0]["gw"]),
            "roles": (infos[0]["bg"], infos[0]["sep"], infos[0]["marker"]),
            "shapes": tuple(sorted({_key(t != i["bg"]) for i in infos for t in i["tokens"]})),
            "programs": tuple(tuple(i["kinds"]) for i in infos),
            "lengths": tuple(len(i["queries"]) for i in infos),
            "input_shapes": tuple(p["input"].shape for s in ("train", "test") for p in puzzle[s]),
            "pane_widths": tuple(i["pane"].shape[1] for i in infos),
            "line_locations": tuple((k, tuple(q)) for i in infos for query in i["queries"]
                                    for k, q in [_line(query, i["rows"], i["cols"])] if k),
            "blank_slots": tuple(len(i["slots"][j]) < max(i["rows"], i["cols"])
                                 for i in infos for j in range(len(i["queries"])))}


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed diversity, inference, metamorphic and mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = [puzzle_features(p) for p in puzzles]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({f["board"] for f in features}) >= 8
    assert len({f["glyph"] for f in features}) == 4
    assert len({f["roles"] for f in features}) >= 30
    assert len({f["programs"] for f in features}) >= 30
    assert len({shape for f in features for shape in f["input_shapes"]}) >= 30
    assert len({w for f in features for w in f["pane_widths"]}) >= 10
    assert {n for f in features for n in f["lengths"]} >= {3, 4, 5, 6}
    assert {v for f in features for v in f["blank_slots"]} == {False, True}
    for seed, puzzle in enumerate(puzzles):
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                # A global color permutation must commute with the transform.
                perm = np.array([(c + 3) % 10 for c in range(10)], np.int8)
                assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
                info = _parse(pair["input"])
                # Reversing keys AND their empty-slot placements within every
                # bundle preserves the answer; bundle order itself does not.
                shuffled = pair["input"].copy()
                for j in range(len(info["queries"])):
                    top = info["program0"] + 1 + j * (info["gh"] + 1)
                    blocks = [shuffled[top:top + info["gh"],
                                       1 + c * (info["gw"] + 1):1 + c * (info["gw"] + 1) + info["gw"]].copy()
                              for c in range(max(info["rows"], info["cols"]))]
                    for c, block in enumerate(reversed(blocks)):
                        left = 1 + c * (info["gw"] + 1)
                        shuffled[top:top + info["gh"], left:left + info["gw"]] = block
                assert np.array_equal(solve(shuffled), pair["output"])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for nt, nq in ((None, None), (None, 4), (2, None)):
        generate_puzzle(77, nt, nq)
    # Hand-specified geometric regression, independent of random layouts.
    pane = np.zeros((12, 8), np.int8)
    pane[0, 1] = 9
    expected = pane.copy()
    expected[1, 1:4] = [1, 2, 3]
    expected[2:4, 3] = [2, 5]
    assert np.array_equal(_plot(pane, 1, 0, 2, 3, [1, 2, 3, 4, 5, 6],
                                [[2, 0, 1], [0, 1, 5], [4, 1]]), expected)
    base = puzzles[0]
    for damage in ("pigment", "seed", "duplicate"):
        changed = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        if damage == "duplicate":
            changed["train"][1] = {k: v.copy() for k, v in changed["train"][0].items()}
        else:
            pair = changed["test"][0]
            info = _parse(pair["input"])
            y, x = (0, info["start"]) if damage == "seed" else tuple(np.argwhere(pair["output"][1:] != info["bg"])[0] + [1, 0])
            pair["output"][y, x] = info["bg"]
        try:
            validate_generated_puzzle(changed)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted " + damage)


if __name__ == "__main__":
    self_test()
    print("42-seed self_test passed")
