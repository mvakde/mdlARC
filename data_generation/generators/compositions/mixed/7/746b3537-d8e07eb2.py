"""Collapse repeated slot vectors, then perform exact glyph lookup and line status.

The solver reads pixels only. Generation labels are rendered from the sampled
logical board and selection; validation reconstructs those from the input.
No files, traces, caches, network calls, or non-NumPy dependencies are used.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


HIGHLIGHT = 3
FAILURE = 2


def _key(array):
    a = np.asarray(array, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _line_kind(selected, rows, cols):
    chosen = set(selected)
    for r in range(rows):
        if chosen == {(r, c) for c in range(cols)}:
            return "row"
    for c in range(cols):
        if chosen == {(r, c) for r in range(rows)}:
            return "col"
    return None


def _contains_line(selected, rows, cols):
    chosen = set(selected)
    return (any({(r, c) for c in range(cols)} <= chosen for r in range(rows))
            or any({(r, c) for r in range(rows)} <= chosen for c in range(cols)))


def _run_indices(board, axis):
    # Compare ENTIRE vectors of complete colored slots, including their glyphs.
    vectors = np.moveaxis(board, axis, 0)
    starts = [0]
    for i in range(1, len(vectors)):
        if not np.array_equal(vectors[i], vectors[i - 1]):
            starts.append(i)
    lengths = tuple(b - a for a, b in zip(starts, starts[1:] + [len(vectors)]))
    return starts, lengths


def _compress(board):
    rr, row_runs = _run_indices(board, 0)
    cc, col_runs = _run_indices(board, 1)
    return board[np.ix_(rr, cc)].copy(), row_runs, col_runs


def _assemble(board, query, background, divider, status_height):
    """Pack a slot board and its unexpanded tray into their canonical scaffold."""
    rows, cols, side, _ = board.shape
    width = max(cols, len(query)) * side
    end = side + 1 + rows * side
    result = np.full((end + 1 + status_height, width), background, np.int8)
    result[side] = divider
    result[end] = divider
    result[side + 1:end, :cols * side] = board.transpose(0, 2, 1, 3).reshape(rows * side, cols * side)
    for c, block in enumerate(query):
        result[:side, c * side:(c + 1) * side] = block
    return result


def _token_description(block, background):
    side = len(block)
    if not (np.all(block[[0, -1]] == background)
            and np.all(block[:, [0, -1]] == background)):
        raise ValueError("glyph slots require a one-cell neutral border")
    core = block[1:-1, 1:-1]
    inks = np.unique(core[core != background])
    if len(inks) != 1 or int(inks[0]) == HIGHLIGHT:
        raise ValueError("each glyph must have one non-highlight ink")
    rr, cc = np.where(core != background)
    if (int(rr.min()) != (side - 2 - int(rr.max() - rr.min() + 1)) // 2
            or int(cc.min()) != (side - 2 - int(cc.max() - cc.min() + 1)) // 2):
        raise ValueError("small glyphs must be consistently centered")
    return int(inks[0]), _key(core != background)


def _analyze(input_grid):
    a = _grid(input_grid)
    background = int(a[0, 0])
    if background in (HIGHLIGHT, FAILURE):
        raise ValueError("neutral color must differ from both status colors")
    bands = [r for r in range(len(a)) if a[r, 0] != background and np.all(a[r] == a[r, 0])]
    if len(bands) != 2:
        raise ValueError("expected exactly two complete divider rows")
    side, end = bands
    if (side not in (4, 5) or int(a[side, 0]) != int(a[end, 0])
            or int(a[side, 0]) in (HIGHLIGHT, FAILURE)
            or (end - side - 1) % side or a.shape[1] % side):
        raise ValueError("invalid tray or slot geometry")
    status_height = len(a) - end - 1
    physical_rows = (end - side - 1) // side
    width_slots = a.shape[1] // side
    if status_height not in (1, 2) or physical_rows < 2:
        raise ValueError("invalid board or status height")
    if not np.all(a[end + 1:] == background):
        raise ValueError("input status must be blank")
    all_blocks = a[side + 1:end].reshape(physical_rows, side, width_slots, side).transpose(0, 2, 1, 3)
    occupied = np.any(all_blocks != background, axis=(2, 3))
    physical_cols = int(occupied[0].sum())
    if (physical_cols < 2 or not np.all(occupied[:, :physical_cols])
            or np.any(occupied[:, physical_cols:])):
        raise ValueError("the board must be a left-aligned nonempty rectangle")
    physical = all_blocks[:, :physical_cols].copy()
    for block in physical.reshape(-1, side, side):
        _token_description(block, background)
    logical, row_runs, col_runs = _compress(physical)
    rows, cols = logical.shape[:2]
    if not (2 <= rows <= 4 and 2 <= cols <= 4):
        raise ValueError("compressed board dimensions must be 2..4")
    lookup = {}
    descriptions = {}
    for r, c in np.ndindex(rows, cols):
        key = _key(logical[r, c])
        if key in lookup:
            raise ValueError("exact glyphs must be unique AFTER vector compression")
        lookup[key] = (r, c)
        descriptions[r, c] = _token_description(logical[r, c], background)
    capacity = max(rows, cols)
    if width_slots != max(physical_cols, capacity):
        raise ValueError("unexpected canvas padding")
    query = [a[:side, c * side:(c + 1) * side].copy() for c in range(capacity)]
    if not np.all(a[:side, capacity * side:] == background):
        raise ValueError("tray padding must be empty")
    selected, order = set(), []
    for block in query:
        if np.all(block == background):
            continue
        key = _key(block)
        if key not in lookup or lookup[key] in selected:
            raise ValueError("query tokens must be distinct exact logical-board keys")
        selected.add(lookup[key])
        order.append(lookup[key])
    if not 1 <= len(selected) <= capacity:
        raise ValueError("query must contain 1..capacity distinct glyphs")
    divider = int(a[side, 0])
    if not np.array_equal(_assemble(physical, query, background, divider, status_height), a):
        raise ValueError("unexpected pixels outside the scaffold")
    return {"grid": a, "background": background, "divider": divider,
            "side": side, "status_height": status_height, "physical": physical,
            "logical": logical, "row_runs": row_runs, "col_runs": col_runs,
            "rows": rows, "cols": cols, "query": query, "selected": selected,
            "order": order, "descriptions": descriptions,
            "kind": _line_kind(selected, rows, cols)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve entirely from the displayed exact slot vectors and query glyphs."""
    info = _analyze(input_grid)
    board = info["logical"]
    result = _assemble(board, info["query"], info["background"],
                       info["divider"], info["status_height"])
    side, rows, cols = info["side"], info["rows"], info["cols"]
    end = side + 1 + rows * side
    win = info["kind"] is not None
    # Pixel reconstruction differs from generation's selected-block renderer.
    for r, c in np.ndindex(result.shape):
        if r > end:
            result[r, c] = HIGHLIGHT if win else FAILURE
        elif result[r, c] == info["background"]:
            if r < side and win:
                result[r, c] = HIGHLIGHT
            elif side < r < end and c < cols * side:
                slot = ((r - side - 1) // side, c // side)
                if slot in info["selected"]:
                    result[r, c] = HIGHLIGHT
    return result


def _shape_bank(side):
    dim = side - 2
    bank = {}
    for code in range(1, 1 << (dim * dim)):
        mask = np.array([(code >> k) & 1 for k in range(dim * dim)], bool).reshape(dim, dim)
        rr, cc = np.where(mask)
        crop = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        centered = np.zeros((dim, dim), bool)
        r, c = (dim - crop.shape[0]) // 2, (dim - crop.shape[1]) // 2
        centered[r:r + crop.shape[0], c:c + crop.shape[1]] = crop
        bank[_key(centered)] = centered
    return list(bank.values())


def _new_board(rng, rows, cols, side, background, inks, selected, bank):
    # Force two different exact-matching lessons involving a queried token:
    # same shape/different ink, and same ink/different shape, both unselected.
    cells = list(np.ndindex(rows, cols))
    target = sorted(selected)[int(rng.integers(len(selected)))]
    outside = [p for p in cells if p not in selected]
    rng.shuffle(outside)
    shape_a, shape_b = map(int, rng.choice(len(bank), 2, replace=False))
    ink_a, ink_b = map(int, rng.choice(inks, 2, replace=False))
    assignments = {target: (shape_a, ink_a), outside[0]: (shape_a, ink_b),
                   outside[1]: (shape_b, ink_a)}
    pool = [(s, int(ink)) for s in range(len(bank)) for ink in inks
            if (s, int(ink)) not in assignments.values()]
    take = rng.choice(len(pool), len(cells) - 3, replace=False)
    remainder = iter(pool[int(k)] for k in take)
    board = np.full((rows, cols, side, side), background, np.int8)
    for p in cells:
        shape, ink = assignments[p] if p in assignments else next(remainder)
        board[p][1:-1, 1:-1][bank[shape]] = ink
    return board


def _selection(rng, rows, cols, kind, axis):
    if kind in ("row", "col"):
        index = int(rng.integers(rows if kind == "row" else cols))
        return ({(index, c) for c in range(cols)} if kind == "row"
                else {(r, index) for r in range(rows)})
    cells = list(np.ndindex(rows, cols))
    line_length = cols if axis == "row" else rows
    if kind == "partial":
        count = int(rng.integers(1, min(rows, cols)))
    else:
        count = line_length
    candidates = []
    # At most C(16,4)=1820 candidates; no unbounded rejection or retries.
    for combination in itertools.combinations(cells, count):
        chosen = set(combination)
        if _contains_line(chosen, rows, cols):
            continue
        if rows == cols and rows >= 3 and (chosen == {(i, i) for i in range(rows)}
                                           or chosen == {(i, cols - 1 - i) for i in range(rows)}):
            continue
        if kind == "near":
            if axis == "row":
                overlap = max(sum(r == i for r, _ in chosen) for i in range(rows))
            else:
                overlap = max(sum(c == i for _, c in chosen) for i in range(cols))
            if overlap != line_length - 1:
                continue
        candidates.append(combination)
    if not candidates:
        raise RuntimeError("no compatible negative selection")
    return set(candidates[int(rng.integers(len(candidates)))])


def _multiplicities(rng, count, limit, active, training):
    values = np.ones(count, dtype=int)
    if active:
        budget = min(limit - count, 2) if training else limit - count
        extras = int(rng.integers(1, budget + 1))
        for _ in range(extras):
            values[int(rng.integers(count))] += 1
    return tuple(map(int, values))


def _render_label(logical, query, selected, background, divider, status_height):
    out = _assemble(logical, query, background, divider, status_height)
    side = logical.shape[2]
    for r, c in selected:
        block = out[side + 1 + r * side:side + 1 + (r + 1) * side,
                    c * side:(c + 1) * side]
        block[block == background] = HIGHLIGHT
    win = _line_kind(selected, *logical.shape[:2]) is not None
    if win:
        tray = out[:side]
        tray[tray == background] = HIGHLIGHT
    out[-status_height:] = HIGHLIGHT if win else FAILURE
    return out


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    side = int(rng.choice((4, 5)))
    status_height = int(rng.integers(1, 3))
    max_rows = (30 - side - 2 - status_height) // side
    rows = int(rng.integers(2, min(4, max_rows - 1) + 1))
    cols = int(rng.integers(2, 5))
    capacity = max(rows, cols)
    first_axis = str(rng.choice(("row", "col")))
    other_axis = "col" if first_axis == "row" else "row"
    first_mode = "cols" if first_axis == "row" else "rows"
    other_mode = "rows" if first_mode == "cols" else "cols"
    schedule = [(first_axis, first_axis, first_mode), ("near", first_axis, other_mode)]
    if num_train >= 3:
        schedule.append((other_axis, other_axis, other_mode))
    if num_train >= 4:
        schedule.append(("partial", first_axis, str(rng.choice(("rows", "cols")))))
    while len(schedule) < num_train:
        schedule.append((str(rng.choice(("row", "col", "near", "scatter", "partial"))),
                         str(rng.choice(("row", "col"))), str(rng.choice(("rows", "cols")))))
    test_kinds = [first_axis, "near", "scatter"]
    if num_train >= 3:
        test_kinds.append(other_axis)
    if num_train >= 4:
        test_kinds.append("partial")
    for _ in range(num_test):
        schedule.append((str(rng.choice(test_kinds)), first_axis, "both"))
    varying_palette = bool(rng.integers(2))
    colors = [c for c in range(10) if c not in (HIGHLIGHT, FAILURE)]
    fixed_background, fixed_divider = map(int, rng.choice(colors, 2, replace=False))
    bank = _shape_bank(side)
    seen, pairs = set(), []
    for index, (kind, axis, mode) in enumerate(schedule):
        training = index < num_train
        # Each attempt uses only a tiny in-memory pair. Bounded uniqueness retries.
        for _ in range(16):
            if varying_palette:
                background, divider = map(int, rng.choice(colors, 2, replace=False))
            else:
                background, divider = fixed_background, fixed_divider
            pool = [c for c in range(10) if c not in (background, HIGHLIGHT)]
            inks = list(map(int, rng.choice(pool, int(rng.integers(2, 7)), replace=False)))
            selected = _selection(rng, rows, cols, kind, axis)
            logical = _new_board(rng, rows, cols, side, background, inks, selected, bank)
            query = [np.full((side, side), background, np.int8) for _ in range(capacity)]
            order = sorted(selected)
            if rng.random() < .8:
                rng.shuffle(order)
            slots = sorted(map(int, rng.choice(capacity, len(order), replace=False)))
            for slot, p in zip(slots, order):
                query[slot] = logical[p].copy()
            row_runs = _multiplicities(rng, rows, max_rows, mode in ("rows", "both"), training)
            col_runs = _multiplicities(rng, cols, 30 // side, mode in ("cols", "both"), training)
            physical = np.repeat(np.repeat(logical, row_runs, axis=0), col_runs, axis=1)
            inp = _assemble(physical, query, background, divider, status_height)
            if _key(inp) not in seen:
                break
        else:
            raise RuntimeError("bounded input uniqueness attempts exhausted")
        seen.add(_key(inp))
        pairs.append({"input": inp, "output": _render_label(
            logical, query, selected, background, divider, status_height)})
    puzzle = {"train": [pairs[int(i)] for i in rng.permutation(num_train)],
              "test": [pairs[num_train + int(i)] for i in rng.permutation(num_test)]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _matching_witnesses(info):
    selected = info["selected"]
    descriptions = info["descriptions"]
    outside = [p for p in descriptions if p not in selected]
    shape_necessary = any(descriptions[p][0] == descriptions[q][0]
                          and descriptions[p][1] != descriptions[q][1]
                          for p in selected for q in outside)
    color_necessary = any(descriptions[p][1] == descriptions[q][1]
                          and descriptions[p][0] != descriptions[q][0]
                          for p in selected for q in outside)
    return shape_necessary, color_necessary


def _first_copy_line(info):
    first_positions = {}
    physical = info["physical"]
    for p in np.ndindex(*physical.shape[:2]):
        first_positions.setdefault(_key(physical[p]), p)
    selected = [first_positions[_key(info["logical"][p])] for p in info["selected"]]
    return _line_kind(selected, *physical.shape[:2])


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(solve(inp), out), "incorrect compression, marking, or status"
            assert _matching_witnesses(info) == (True, True), "exact shape AND color must matter"
            logical = info["logical"]
            reconstructed = np.repeat(np.repeat(logical, info["row_runs"], axis=0), info["col_runs"], axis=1)
            assert np.array_equal(reconstructed, info["physical"])
            assert np.array_equal(_compress(logical)[0], logical), "compression is idempotent"
            rr, _ = _run_indices(info["physical"], 0)
            cc, _ = _run_indices(info["physical"][rr], 1)
            assert np.array_equal(info["physical"][rr][:, cc], logical)
            cc, _ = _run_indices(info["physical"], 1)
            rr, _ = _run_indices(info["physical"][:, cc], 0)
            assert np.array_equal(info["physical"][:, cc][rr], logical), "axis order must commute"
            repeated_rows = max(info["row_runs"]) > 1
            repeated_cols = max(info["col_runs"]) > 1
            assert (repeated_rows != repeated_cols) if split == "train" else (repeated_rows and repeated_cols)
            assert info["kind"] or not _contains_line(info["selected"], info["rows"], info["cols"])
            assert out.shape[0] == info["side"] * (info["rows"] + 1) + 2 + info["status_height"]
            assert out.shape[1] == max(info["rows"], info["cols"]) * info["side"]
            assert info["physical"].shape[:2] != logical.shape[:2]
            assert not np.array_equal(inp, out), "compression and marking must change the answer"
            analyses[split].append(info)
    train = analyses["train"]
    all_info = train + analyses["test"]
    assert len({(i["rows"], i["cols"], i["side"], i["status_height"]) for i in all_info}) == 1
    axes = {i["kind"] for i in train if i["kind"]}
    assert axes and any(not i["kind"] for i in train)
    assert {max(i["row_runs"]) > 1 for i in train} == {False, True}
    assert {max(i["col_runs"]) > 1 for i in train} == {False, True}
    assert any(i["kind"] and _first_copy_line(i) is None for i in train), "run collapse must repair false line length"
    assert any(len(a["selected"]) == len(b["selected"]) for a in train if a["kind"] for b in train if not b["kind"])
    if len(train) >= 3:
        assert axes == {"row", "col"}
    for info in analyses["test"]:
        assert not info["kind"] or info["kind"] in axes
        if len(info["selected"]) < min(info["rows"], info["cols"]):
            assert any(len(i["selected"]) < min(i["rows"], i["cols"]) for i in train)


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed streaming suite; no retained puzzle corpus or disk IO."""
    signatures, dimensions, geometries, palettes, run_patterns = set(), set(), set(), set(), set()
    selections, test_kinds, glyphs, shapes = set(), set(), set(), set()
    baseline = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        signatures.add(_fingerprint(puzzle))
        if seed == 0:
            baseline = puzzle
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                dimensions.add((info["rows"], info["cols"], info["side"]))
                geometries.add((pair["input"].shape, pair["output"].shape))
                palettes.add((info["background"], info["divider"]))
                run_patterns.add((info["row_runs"], info["col_runs"]))
                selections.add((info["rows"], info["cols"], tuple(sorted(info["selected"]))))
                glyphs.update(_key(b) for b in info["logical"].reshape(-1, info["side"], info["side"]))
                shapes.update(d[1] for d in info["descriptions"].values())
                if split == "test":
                    test_kinds.add(info["kind"] or "failure")
                # Multiplicities carry no semantic information after compression.
                unexpanded = _assemble(info["logical"], info["query"], info["background"],
                                       info["divider"], info["status_height"])
                assert np.array_equal(solve(unexpanded), pair["output"])
                # Changing query order changes the tray, never lookup or status.
                reversed_query = list(reversed(info["query"]))
                reordered = _assemble(info["physical"], reversed_query, info["background"],
                                      info["divider"], info["status_height"])
                expected = _render_label(info["logical"], reversed_query, info["selected"],
                                         info["background"], info["divider"], info["status_height"])
                assert np.array_equal(solve(reordered), expected)
    assert len(signatures) == 42
    assert len(dimensions) >= 12 and len(geometries) >= 40
    assert len(palettes) >= 20 and len(run_patterns) >= 65
    assert len(selections) >= 55 and len(glyphs) >= 100 and len(shapes) >= 50
    assert test_kinds == {"row", "col", "failure"}
    legal_counts = 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            generated = generate_puzzle(4400 + train * 10 + test, train, test)
            assert len(generated["train"]) == train and len(generated["test"]) == test
            legal_counts += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Validator must detect errors in labels and input uniqueness.
    for defect in ("status", "ink", "divider", "duplicate"):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in baseline[s]] for s in baseline}
        pair = damaged["test"][0]
        if defect == "status":
            pair["output"][-1, 0] = (int(pair["output"][-1, 0]) + 1) % 10
        elif defect == "ink":
            info = _analyze(pair["input"])
            r, c = sorted(info["selected"])[0]
            ir, ic = np.argwhere(info["logical"][r, c] != info["background"])[0]
            pair["output"][info["side"] + 1 + r * info["side"] + ir, c * info["side"] + ic] = info["background"]
        elif defect == "divider":
            info = _analyze(pair["input"])
            pair["output"][info["side"], 0] = info["background"]
        else:
            damaged["train"][1] = {k: v.copy() for k, v in damaged["train"][0].items()}
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("validator accepted " + defect)
    return {"seeds": 42, "distinct_puzzles": len(signatures), "legal_count_combinations": legal_counts,
            "logical_geometries": len(dimensions), "input_output_geometries": len(geometries),
            "background_divider_assignments": len(palettes), "vector_run_patterns": len(run_patterns),
            "selection_sets": len(selections), "exact_glyphs": len(glyphs), "glyph_shapes": len(shapes),
            "test_outcomes": sorted(test_kinds), "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
