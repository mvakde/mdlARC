"""Perforated ring code: exact query lookup closes ordered color ribbons.

Only standard library and NumPy are used. No parent module or stored example is
read. Generation labels know selected coordinates; solve discovers them again
from the input's visible lattice and exact colored glyphs.
"""

from functools import lru_cache
import math

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _arc_grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "colors must be 0..9")
    return grid.astype(np.int8, copy=False)


def _key(array):
    return array.shape, array.astype(np.int8, copy=False).tobytes()


def _trim(mask):
    rr, cc = np.where(mask)
    _require(len(rr) > 0, "empty mask")
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]


def _center(mask, gh, gw):
    crop = _trim(mask)
    result = np.zeros((gh, gw), dtype=bool)
    r, c = (gh - crop.shape[0]) // 2, (gw - crop.shape[1]) // 2
    result[r:r + crop.shape[0], c:c + crop.shape[1]] = crop
    return result


@lru_cache(maxsize=9)
def _shape_bank(gh, gw):
    """All nonempty occupancy patterns, unique after translation trimming."""
    bank = {}
    for number in range(1, 1 << (gh * gw)):
        mask = np.array([(number >> i) & 1 for i in range(gh * gw)],
                        dtype=bool).reshape(gh, gw)
        crop = _trim(mask)
        bank[_key(crop)] = _center(crop, gh, gw)
    # Explicit mass for solid, narrow, disconnected, and hollow shapes comes
    # from the general bank; generation also sometimes reserves one of these.
    return tuple(bank.values())


@lru_cache(maxsize=1)
def _layouts():
    result = []
    for gh in range(1, 4):
        for gw in range(2, 5):
            bank_size = len(_shape_bank(gh, gw))
            for columns in range(2, 7):
                if columns * (gw + 1) + 1 > 30 or bank_size < 2 * columns:
                    continue
                for rows in range(2, 13):
                    qmax = min(rows, 27 // (gh + 1) - rows)
                    if qmax >= 1 and rows * columns <= bank_size * 8:
                        result.append((rows, columns, gh, gw, qmax))
    return tuple(result)


def _parse(input_grid):
    """Recover board, tray, exact matches and closure from pixels alone."""
    grid = _arc_grid(input_grid)
    height, width = grid.shape
    _require(height >= 9 and width >= 7, "scene is too small")
    bg = int(grid[0, 0])
    _require(bool(np.all(grid[0] == bg)), "missing background header")
    divider = int(grid[1, 0])
    _require(divider != bg, "background and divider must differ")
    horizontal = [r for r in range(1, height) if np.all(grid[r] == divider)]
    _require(horizontal[0] == 1 and horizontal[-1] == height - 1,
             "missing outer lattice boundary")
    double = [i for i in range(len(horizontal) - 1)
              if horizontal[i + 1] - horizontal[i] == 1]
    _require(len(double) == 1, "there must be one doubled horizontal boundary")
    split = double[0]
    _require(split >= 2, "need at least two board ribbons")
    pitch = horizontal[1] - horizontal[0]
    gh = pitch - 1
    _require(1 <= gh <= 3, "invalid glyph height")
    _require(all(horizontal[i + 1] - horizontal[i] == pitch
                 for i in range(len(horizontal) - 1) if i != split),
             "irregular horizontal lattice")
    rows = split
    query_rows = len(horizontal) - split - 2
    _require(2 <= rows <= 12 and 1 <= query_rows <= rows,
             "invalid ribbon or tray row count")
    vertical = [c for c in range(width) if np.all(grid[1:, c] == divider)]
    _require(len(vertical) >= 3 and vertical[0] == 0 and vertical[-1] == width - 1,
             "missing vertical lattice boundary")
    wpitch = vertical[1] - vertical[0]
    gw = wpitch - 1
    columns = len(vertical) - 1
    _require(2 <= gw <= 4 and 2 <= columns <= 6,
             "invalid glyph width or ribbon length")
    _require(all(vertical[i + 1] - vertical[i] == wpitch
                 for i in range(columns)), "irregular vertical lattice")
    _require(len(_shape_bank(gh, gw)) >= 2 * columns, "insufficient shape capacity")
    _require(height == (rows + query_rows) * pitch + 3
             and width == columns * wpitch + 1, "inconsistent scene dimensions")

    # Reconstruct the entire scaffold to reject noise/marker coincidences.
    scaffold = np.full(grid.shape, divider, dtype=np.int8)
    scaffold[0] = bg
    board, lookup, codes, shapes, inks, slices = [], {}, [], [], [], []
    trimmed = set()

    def token_record(block, allow_empty):
        occupied = block != bg
        if not np.any(occupied):
            _require(allow_empty, "board token cannot be empty")
            return None
        colors = np.unique(block[occupied])
        _require(len(colors) == 1 and int(colors[0]) != divider,
                 "glyph must have one nonstructural ink")
        _require(np.array_equal(occupied, _center(occupied, gh, gw)),
                 "glyph must be translation-normalized and centered")
        return int(colors[0]), _key(_trim(occupied))

    for r in range(rows):
        row_inks = []
        for c in range(columns):
            rs = slice(2 + r * pitch, 2 + r * pitch + gh)
            cs = slice(1 + c * wpitch, 1 + c * wpitch + gw)
            block = grid[rs, cs]
            record = token_record(block, False)
            ink, shape = record
            exact = _key(block)
            normalized = (shape, ink)
            _require(exact not in lookup and normalized not in trimmed,
                     "board tokens must be unique including after trimming")
            lookup[exact] = (r, c)
            trimmed.add(normalized)
            row_inks.append(ink)
            board.append(block.copy())
            shapes.append(shape)
            inks.append(ink)
            slices.append((rs, cs))
            scaffold[rs, cs] = block
        _require(len(set(row_inks)) == 1, "ribbon inks must be uniform")
        codes.append(row_inks[0])

    tray_top = horizontal[split + 1]
    order, blanks, query_slices = [], [], []
    for r in range(query_rows):
        for c in range(columns):
            rs = slice(tray_top + 1 + r * pitch, tray_top + 1 + r * pitch + gh)
            cs = slice(1 + c * wpitch, 1 + c * wpitch + gw)
            block = grid[rs, cs]
            record = token_record(block, True)
            query_slices.append((rs, cs))
            scaffold[rs, cs] = block
            if record is None:
                blanks.append(r * columns + c)
            else:
                _require(_key(block) in lookup, "query has no exact board copy")
                order.append(lookup[_key(block)])
    _require(len(order) > 0 and len(set(order)) == len(order),
             "query keys must be nonempty and distinct")
    _require(np.array_equal(scaffold, grid), "unexpected cells outside the glyph slots")
    selected = set(order)
    closed = [all((r, c) in selected for c in range(columns)) for r in range(rows)]
    layers = np.array([codes[r] if closed[r] else bg for r in range(rows)], dtype=np.int8)
    return {"grid": grid, "bg": bg, "divider": divider, "rows": rows,
            "columns": columns, "gh": gh, "gw": gw, "query_rows": query_rows,
            "board": board, "codes": codes, "shapes": shapes, "inks": inks,
            "board_slices": slices, "query_slices": query_slices,
            "order": order, "selected": selected, "blanks": blanks,
            "closed": closed, "layers": layers}


def _rings(layers):
    side = 2 * len(layers)
    edge = np.minimum(np.arange(side), np.arange(side)[::-1])
    return np.asarray(layers, dtype=np.int8)[np.minimum(edge[:, None], edge[None, :])]


def _wrap(layers):
    """Validation/construction alternative: grow one border at a time."""
    result = np.full((2, 2), int(layers[-1]), dtype=np.int8)
    for color in layers[-2::-1]:
        result = np.pad(result, 1, constant_values=int(color))
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _rings(info["layers"])


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _requested_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is outside its permitted range")
    return int(value)


def _anchor_layout(rng, lesson, different_rows=None):
    choices = []
    for rows, columns, gh, gw, qmax in _layouts():
        if not 4 <= rows <= 7 or gh > 2 or rows == different_rows:
            continue
        # First lesson: two closed ribbons, one near-complete neighbor,
        # and one complete column. Second: three closed ribbons and a column.
        count = 3 * columns + rows - (4 if lesson == 0 else 3)
        query_rows = math.ceil(count / columns)
        if lesson == 0 and count == query_rows * columns:
            query_rows += 1  # Empty-slot semantics must be exposed.
        if query_rows <= qmax:
            choices.append((rows, columns, gh, gw, query_rows))
    _require(bool(choices), "no feasible teaching layout")
    # Sample height and ribbon count before remaining axes, so large shape
    # banks and their many possible widths do not erase short glyphs.
    gh = _choose(rng, sorted({v[2] for v in choices}))
    choices = [v for v in choices if v[2] == gh]
    rows = _choose(rng, sorted({v[0] for v in choices}))
    return _choose(rng, [v for v in choices if v[0] == rows])


def _sample_codes(rng, rows, columns, bank_size, available, lesson=None, repeat_start=None):
    capacity = bank_size // columns
    minimum = max(3 if lesson is not None else 1, math.ceil(rows / capacity))
    maximum = min(len(available), rows - (lesson is not None))
    _require(minimum <= maximum, "infeasible ribbon palette")
    richness = int(rng.integers(minimum, maximum + 1))
    palette = list(map(int, rng.choice(available, richness, replace=False)))
    counts = {color: 0 for color in palette}
    code = [None] * rows

    def put(index, color):
        code[index] = color
        counts[color] += 1

    other_row = None
    if lesson is not None:
        k = repeat_start
        put(k, palette[0])
        put(k + 1, palette[0])
        put(rows - 1, palette[2])
        other_row = _choose(rng, [r for r in range(rows - 1) if r not in (k, k + 1)])
        put(other_row, palette[1])
        empty = [r for r in range(rows) if code[r] is None]
        rng.shuffle(empty)
        for color in palette[3:]:
            put(empty.pop(), color)
    else:
        positions = list(map(int, rng.permutation(rows)))
        for index, color in zip(positions, palette):
            put(index, color)
    for index in range(rows):
        if code[index] is None:
            choices = [c for c in palette if counts[c] < capacity]
            put(index, int(_choose(rng, choices)))
    assert max(counts.values()) <= capacity
    return code, other_row


def _selection(rng, rows, columns, max_tokens, lesson=None, repeat_start=None):
    all_positions = [(r, c) for r in range(rows) for c in range(columns)]
    if lesson is not None:
        k = repeat_start
        col = int(rng.integers(columns))
        active = {k, rows - 1} if lesson == 0 else {k, k + 1, rows - 1}
        selected = {(r, c) for r in active for c in range(columns)}
        selected.update((r, col) for r in range(rows))
        if lesson == 0:
            missing = int(_choose(rng, [c for c in range(columns) if c != col]))
            selected.update((k + 1, c) for c in range(columns) if c != missing)
        assert len(selected) <= max_tokens
        return selected, col

    mode = int(rng.integers(5))
    if mode in (0, 1):
        maximum = min(rows, max_tokens // columns)
        count = int(rng.integers(1, maximum + 1))
        active = list(map(int, rng.choice(rows, count, replace=False)))
        selected = {(r, c) for r in active for c in range(columns)}
        if mode == 1 and len(selected) < max_tokens:
            remaining = [p for p in all_positions if p not in selected]
            extra = int(rng.integers(min(len(remaining), max_tokens - len(selected)) + 1))
            if extra:
                indices = rng.choice(len(remaining), extra, replace=False)
                selected.update(remaining[int(i)] for i in indices)
    elif mode == 2:
        # Near miss: one ribbon is exactly one key short; optional scatter
        # cannot add the missing key back.
        row = int(rng.integers(rows))
        missing = int(rng.integers(columns))
        selected = {(row, c) for c in range(columns) if c != missing}
        remaining = [p for p in all_positions if p[0] != row]
        extra_max = min(len(remaining), max_tokens - len(selected))
        extra = int(rng.integers(extra_max + 1))
        if extra:
            selected.update(remaining[int(i)] for i in rng.choice(len(remaining), extra, replace=False))
    elif mode == 3 and rows <= max_tokens:
        col = int(rng.integers(columns))
        selected = {(r, col) for r in range(rows)}
    else:
        count = int(rng.integers(1, min(rows * columns, max_tokens) + 1))
        selected = {all_positions[int(i)] for i in rng.choice(len(all_positions), count, replace=False)}
    return selected, None


def _make_board(rng, code, columns, gh, gw, bg, selected, lesson, repeat_start, other_row):
    bank = _shape_bank(gh, gw)
    assignments = {}
    inks = list(dict.fromkeys(code))
    # Build the repeated ink first; its selected row supplies the confusers.
    if lesson == 0:
        ink = code[repeat_start]
        inks.remove(ink)
        inks.insert(0, ink)
    for ink in inks:
        positions = [(r, c) for r in range(len(code)) for c in range(columns) if code[r] == ink]
        forced = {}
        if lesson == 0 and ink == code[other_row]:
            # Every missing key in this differently colored ribbon has the
            # shape of a queried key. Shape-only union therefore closes it.
            for c in range(columns):
                if (other_row, c) not in selected:
                    forced[(other_row, c)] = assignments[(repeat_start, c)]
        reserved = set(forced.values())
        candidates = [i for i in range(len(bank)) if i not in reserved]
        needed = len(positions) - len(forced)
        chosen = list(map(int, rng.choice(candidates, needed, replace=False)))
        # Sometimes replace one random shape with an allowed distinctive mask.
        hollow_witness = ink == inks[0] and gh >= 3 and gw >= 3
        if needed and (hollow_witness or rng.random() < .35):
            style = 2 if hollow_witness else int(rng.integers(4))
            special = np.ones((gh, gw), dtype=bool)
            if style == 1:
                special = np.indices((gh, gw)).sum(0) % 2 == 0
            elif style == 2:
                special[1:-1, 1:-1] = False
            elif style == 3:
                special[:, 1:] = False
            normalized = _key(_center(special, gh, gw))
            index = next(i for i, mask in enumerate(bank) if _key(mask) == normalized)
            if index not in reserved and index not in chosen:
                chosen[0] = index
        rng.shuffle(chosen)
        for position in positions:
            assignments[position] = forced[position] if position in forced else chosen.pop()
    board = []
    for r in range(len(code)):
        for c in range(columns):
            token = np.full((gh, gw), bg, dtype=np.int8)
            token[bank[assignments[(r, c)]]] = code[r]
            board.append(token)
    return board


def _render_pair(rng, layout, bg, divider, available, lesson=None):
    rows, columns, gh, gw, query_rows = layout
    k = int(rng.integers(rows - 2)) if lesson is not None else None
    code, other_row = _sample_codes(rng, rows, columns, len(_shape_bank(gh, gw)),
                                   available, lesson, k)
    selected, _ = _selection(rng, rows, columns, query_rows * columns, lesson, k)
    board = _make_board(rng, code, columns, gh, gw, bg, selected,
                        lesson, k, other_row)
    pitch, wpitch = gh + 1, gw + 1
    height = (rows + query_rows) * pitch + 3
    inp = np.full((height, columns * wpitch + 1), divider, dtype=np.int8)
    inp[0] = bg
    for r in range(rows):
        for c in range(columns):
            inp[2 + r * pitch:2 + r * pitch + gh,
                1 + c * wpitch:1 + c * wpitch + gw] = board[r * columns + c]
    tray_top = 2 + rows * pitch
    order = sorted(selected)
    if lesson is not None or rng.random() < .8:
        rng.shuffle(order)
        if len(order) > 1 and order == sorted(selected):
            order = order[1:] + order[:1]
    occupied = sorted(map(int, rng.choice(query_rows * columns, len(order), replace=False)))
    query = dict(zip(occupied, order))
    for r in range(query_rows):
        for c in range(columns):
            block = np.full((gh, gw), bg, dtype=np.int8)
            position = query.get(r * columns + c)
            if position is not None:
                block = board[position[0] * columns + position[1]]
            inp[tray_top + 1 + r * pitch:tray_top + 1 + r * pitch + gh,
                1 + c * wpitch:1 + c * wpitch + gw] = block
    layers = [code[r] if all((r, c) in selected for c in range(columns)) else bg
              for r in range(rows)]
    return {"input": inp, "output": _wrap(layers)}


def _sample_general_layout(rng, allowed_rows=None):
    choices = [v for v in _layouts() if allowed_rows is None or v[0] in allowed_rows]
    _require(bool(choices), "no feasible ordinary layout")
    rows = int(_choose(rng, sorted({v[0] for v in choices})))
    choices = [v for v in choices if v[0] == rows]
    gh = int(_choose(rng, sorted({v[2] for v in choices})))
    choices = [v for v in choices if v[2] == gh]
    rows, columns, gh, gw, qmax = _choose(rng, choices)
    query_rows = int(rng.integers(1, qmax + 1))
    return rows, columns, gh, gw, query_rows


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _requested_count(num_train, "num_train", 2, 10)
    nq = _requested_count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(int(seed))
    bg, divider = list(map(int, rng.choice(10, 2, replace=False)))
    available = [c for c in range(10) if c not in (bg, divider)]
    train, test, seen = [], [], set()
    first_rows = None
    for index in range(nt + nq):
        lesson = index if index < 2 else None
        if lesson is not None:
            layout = _anchor_layout(rng, lesson, first_rows if index == 1 else None)
        else:
            allowed = None
            if index >= nt:
                witnessed = {_parse(p["input"])["rows"] for p in train}
                allowed = set(range(2, 13)) - witnessed
            layout = _sample_general_layout(rng, allowed)
        for attempt in range(64):
            pair = _render_pair(rng, layout, bg, divider, available, lesson)
            fingerprint = _key(pair["input"])
            if fingerprint not in seen:
                seen.add(fingerprint)
                break
        else:
            raise RuntimeError("bounded input uniqueness attempts exhausted")
        if index == 0:
            first_rows = layout[0]
        (train if index < nt else test).append(pair)
    rng.shuffle(train)
    rng.shuffle(test)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation_layers(info, mode):
    rows, columns = info["rows"], info["columns"]
    if mode == "all":
        return np.asarray(info["codes"], dtype=np.int8)
    if mode == "color":
        queried = {info["codes"][r] for r, _ in info["selected"]}
        selected = {(r, c) for r in range(rows) for c in range(columns)
                    if info["codes"][r] in queried}
    elif mode == "shape":
        queried = {info["shapes"][r * columns + c] for r, c in info["selected"]}
        selected = {(r, c) for r in range(rows) for c in range(columns)
                    if info["shapes"][r * columns + c] in queried}
    elif mode == "any":
        return np.asarray([info["codes"][r] if any((r, c) in info["selected"]
                                                  for c in range(columns)) else info["bg"]
                           for r in range(rows)], dtype=np.int8)
    else:
        raise ValueError("unknown ablation")
    return np.asarray([info["codes"][r] if all((r, c) in selected for c in range(columns))
                       else info["bg"] for r in range(rows)], dtype=np.int8)


def _training_checks(infos, pairs):
    assert len({i["rows"] for i in infos}) >= 2, "size dependence needs two ribbon counts"
    assert any(i["blanks"] for i in infos), "blank query slots must be taught"
    assert any(any(i["closed"][r] != i["closed"][s] and i["codes"][r] == i["codes"][s]
                   for r in range(i["rows"]) for s in range(r + 1, i["rows"])) for i in infos), \
        "same-ink closed/incomplete contrast must be taught"
    assert any(any(i["closed"][r] and i["closed"][r + 1]
                       and i["codes"][r] == i["codes"][r + 1]
                   for r in range(i["rows"] - 1)) for i in infos), "repeated completed bands must be taught"
    assert any(any(not i["closed"][r]
                       and sum((r, c) in i["selected"] for c in range(i["columns"])) == i["columns"] - 1
                   for r in range(i["rows"])) for i in infos), "near-miss lesson is missing"
    assert any(any(all((r, c) in i["selected"] for r in range(i["rows"]))
                   for c in range(i["columns"])) and not all(i["closed"]) for i in infos), \
        "completed-column counterexample must be taught"
    for mode in ("all", "color", "shape", "any"):
        assert any(not np.array_equal(_wrap(_ablation_layers(i, mode)), p["output"])
                   for i, p in zip(infos, pairs)), "training permits the " + mode + " shortcut"
    assert any(not np.array_equal(_wrap(i["layers"][::-1]), p["output"])
               for i, p in zip(infos, pairs)), "reverse traversal remains possible"
    assert any(not np.array_equal(np.repeat(i["layers"], 2)[:, None].repeat(2 * i["rows"], axis=1),
                                 p["output"]) for i, p in zip(infos, pairs)), "flat stripes remain possible"
    # Run compression also changes the required dimensions.
    assert any(len(i["layers"]) != len(i["layers"][np.r_[True, i["layers"][1:] != i["layers"][:-1]]])
               for i in infos), "run compression remains possible"


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), []
    pairs = puzzle["train"] + puzzle["test"]
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = _arc_grid(pair["input"]), _arc_grid(pair["output"])
        fingerprint = _key(inp)
        assert fingerprint not in seen, "duplicate input"
        seen.add(fingerprint)
        info = _parse(inp)
        assert out.shape == (2 * info["rows"],) * 2, "wrong ring-code size"
        assert np.array_equal(out, _wrap(info["layers"])), "wrong closure, depth or frame pixels"
        assert set(map(int, np.unique(out))) <= {info["bg"]} | set(info["codes"])
        assert np.array_equal(out, out.T) and np.array_equal(out, out[::-1])
        infos.append(info)
    assert len({(i["bg"], i["divider"]) for i in infos}) == 1, "structural roles change within puzzle"
    _training_checks(infos[:nt], puzzle["train"])
    witnessed = {i["rows"] for i in infos[:nt]}
    assert all(i["rows"] not in witnessed for i in infos[nt:]), "test length must be unseen"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split])
                 for split in ("train", "test"))


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def self_test() -> dict:
    """Bounded 42-seed diversity, explicit counts, and semantic counterfactuals."""
    signatures, structures, dimensions, glyph_sizes, ribbon_counts = set(), set(), set(), set(), set()
    column_counts, statuses, palettes, colors, roles = set(), set(), set(), set(), set()
    disconnected = holes = solids = 0
    examples = 0
    for seed in range(42):
        # Exercise every test count and a range of legal training counts.
        nq = 1 + seed % 4
        nt = max(nq + 1, 2 + seed % 9)
        puzzle = generate_puzzle(seed, nt, nq)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, nt, nq)), "seed is nondeterministic"
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                info = _parse(inp)
                before = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, before), "solve mutated its input"
                dimensions.add(inp.shape)
                glyph_sizes.add((info["gh"], info["gw"]))
                ribbon_counts.add(info["rows"])
                column_counts.add(info["columns"])
                palettes.add(len(set(info["codes"])))
                colors.update(map(int, np.unique(inp)))
                roles.add((info["bg"], info["divider"]))
                normalized = {}
                code = tuple(normalized.setdefault(c, len(normalized)) for c in info["codes"])
                structures.add((info["rows"], info["columns"], code, tuple(info["closed"])))
                if split == "test":
                    statuses.add(sum(info["closed"]))
                for token in info["board"]:
                    mask = _trim(token != info["bg"])
                    disconnected += not _connected(mask)
                    solids += bool(np.all(mask))
                    holes += (mask.shape[0] >= 3 and mask.shape[1] >= 3
                              and np.all(mask[0]) and np.all(mask[-1])
                              and np.all(mask[:, 0]) and np.all(mask[:, -1])
                              and not np.all(mask[1:-1, 1:-1]))
                # Moving query keys among arbitrary slots must leave closure
                # and output unchanged, including moving the blanks.
                permuted = inp.copy()
                blocks = [inp[rs, cs].copy() for rs, cs in info["query_slices"]]
                order = np.random.default_rng(seed + examples + 123).permutation(len(blocks))
                for (rs, cs), index in zip(info["query_slices"], order):
                    permuted[rs, cs] = blocks[int(index)]
                assert np.array_equal(solve(permuted), pair["output"])
                examples += 1
    assert len(signatures) == 42
    assert len(structures) >= 120 and len(dimensions) >= 35
    assert ribbon_counts == set(range(2, 13))
    assert column_counts == set(range(2, 7))
    assert len(glyph_sizes) >= 6 and palettes >= {1, 2, 3, 4, 5, 6}
    assert colors == set(range(10)) and len(roles) >= 20
    assert {0, 1, 2} <= statuses
    assert disconnected > 50 and solids > 50 and holes > 0

    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")

    baseline = generate_puzzle(0)
    chosen = next(p for p in baseline["train"]
                  if _parse(p["input"])["blanks"] and _near_complete_repeat(_parse(p["input"])) is not None)
    info = _parse(chosen["input"])
    closed_row, near_row, missing_col = _near_complete_repeat(info)
    assert info["blanks"]
    completed = chosen["input"].copy()
    query_slice = info["query_slices"][info["blanks"][0]]
    completed[query_slice] = info["board"][near_row * info["columns"] + missing_col]
    expected_layers = info["layers"].copy()
    expected_layers[near_row] = info["codes"][near_row]
    assert np.array_equal(solve(completed), _wrap(expected_layers)), "one missing key should close exactly one annulus"

    # The unchanged query now selects a different arrangement in its own board.
    rearranged = chosen["input"].copy()
    near_slice = info["board_slices"][near_row * info["columns"] + missing_col]
    closed_slice = info["board_slices"][closed_row * info["columns"] + missing_col]
    near_token, closed_token = rearranged[near_slice].copy(), rearranged[closed_slice].copy()
    rearranged[near_slice], rearranged[closed_slice] = closed_token, near_token
    moved_layers = info["layers"].copy()
    moved_layers[closed_row] = info["bg"]
    moved_layers[near_row] = info["codes"][near_row]
    assert np.array_equal(solve(rearranged), _wrap(moved_layers))
    assert not np.array_equal(solve(rearranged), chosen["output"])

    for defect in ("frame", "duplicate", "query", "scaffold"):
        damaged = {split: [{k: np.asarray(v).copy() for k, v in p.items()} for p in baseline[split]]
                   for split in ("train", "test")}
        pair = damaged["test"][0]
        parsed = _parse(pair["input"])
        if defect == "frame":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif defect == "duplicate":
            damaged["train"][1] = {k: v.copy() for k, v in damaged["train"][0].items()}
        elif defect == "query":
            occupied = [s for s in parsed["query_slices"] if np.any(pair["input"][s] != parsed["bg"])]
            pair["input"][occupied[0]] = parsed["divider"]
        else:
            pair["input"][1, 0] = parsed["bg"]
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("validator accepted " + defect)
    return {"seeds": 42, "pairs": examples, "distinct_puzzles": len(signatures),
            "structural_signatures": len(structures), "input_shapes": len(dimensions),
            "glyph_canvases": sorted(glyph_sizes), "ribbon_counts": sorted(ribbon_counts),
            "column_counts": sorted(column_counts), "palette_sizes": sorted(palettes),
            "test_closed_ribbon_counts": sorted(statuses), "structural_palettes": len(roles),
            "disconnected_tokens": int(disconnected), "solid_tokens": int(solids),
            "hollow_rectangles": int(holes), "independent_solver": False}


def _near_complete_repeat(info):
    for near_row in range(info["rows"]):
        missing = [c for c in range(info["columns"]) if (near_row, c) not in info["selected"]]
        if len(missing) != 1:
            continue
        for closed_row in range(info["rows"]):
            if info["closed"][closed_row] and info["codes"][closed_row] == info["codes"][near_row]:
                return closed_row, near_row, missing[0]
    return None


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
