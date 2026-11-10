"""Exact colored-token lookup, slot marking, and whole-row/column status.

Geometry and input palettes are visible. Only the two output colors are learned
conventions. Validation reconstructs the scene from pixels, without RNG state.
"""

import hashlib
import itertools

import numpy as np


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _requested_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside its permitted range")
    return int(value)


def _choose_counts(seed, rng, num_train, num_test):
    nt = _requested_count(num_train, "num_train", 2, 10)
    nq = _requested_count(num_test, "num_test", 1, 4)
    legal = [(t, q) for q in range(1, 5) for t in range(q + 1, 11)
             if (nt is None or t == nt) and (nq is None or q == nq)]
    if not legal:
        raise ValueError("num_train must exceed num_test")
    return legal[seed % len(legal)] if nt is None and nq is None else legal[int(rng.integers(len(legal)))]


# n, glyph height/width, padding height/width, side margin, board spacer,
# status thickness. The bound applies to the complete scene, before rotation.
_LAYOUTS = tuple(
    (n, gh, gw, ph, pw, margin, gap, status)
    for n in (3, 4, 5)
    for gh, gw in itertools.product(range(2, 5), repeat=2)
    for ph, pw in itertools.product((1, 2), repeat=2)
    for margin, gap, status in itertools.product((1, 2), (1, 2), (1, 2, 3))
    if (n + 1) * (gh + 2 * ph) + 2 + gap + status <= 30
    and n * (gw + 2 * pw) + 2 * margin <= 30
)
_SOURCE_LAYOUT = (4, 3, 3, 1, 1, 1, 1, 2)


def _sample_layout(rng, inventory=None, different_n=None):
    choices = [v for v in _LAYOUTS
               if (inventory is None or v[:3] == inventory)
               and (different_n is None or v[0] != different_n)]
    # Sampling n first prevents the many small-board layouts dominating.
    n = int(rng.choice(sorted({v[0] for v in choices})))
    choices = [v for v in choices if v[0] == n]
    return choices[int(rng.integers(len(choices)))]


def _geometry(layout):
    n, gh, gw, ph, pw, margin, gap, status = layout
    sh, sw = gh + 2 * ph, gw + 2 * pw
    board0 = sh + 1 + gap
    end = board0 + n * sh
    return n, sh, sw, margin, board0, end, end + 1 + status


def _orient(grid, orientation):
    return np.rot90(np.fliplr(grid) if orientation >= 4 else grid,
                    orientation % 4).copy()


def _trim(array, occupied):
    rr, cc = np.where(occupied)
    assert len(rr)
    return array[rr.min():rr.max() + 1, cc.min():cc.max() + 1]


def _key(array):
    return array.shape, array.tobytes()


def _shape_key(token, background):
    mask = token != background
    return _key(_trim(mask, mask))


def _center_mask(mask, gh, gw):
    crop = _trim(mask, mask)
    result = np.zeros((gh, gw), dtype=bool)
    r, c = (gh - crop.shape[0]) // 2, (gw - crop.shape[1]) // 2
    result[r:r + crop.shape[0], c:c + crop.shape[1]] = crop
    return result


def _shape_bank(rng, gh, gw):
    # Exhaust small canvases; compose fresh random bit patterns for larger ones.
    area = gh * gw
    codes = range(1, 1 << area) if area <= 9 else rng.integers(1, 1 << area, size=320)
    special = [np.ones((gh, gw), bool), np.indices((gh, gw)).sum(0) % 2 == 0]
    ring = np.ones((gh, gw), bool)
    ring[1:-1, 1:-1] = False
    special.append(ring)
    narrow = np.zeros((gh, gw), bool)
    narrow[:, 0] = True
    special.append(narrow)
    bank = {}
    for mask in special:
        canonical = _center_mask(mask, gh, gw)
        bank[_shape_key(canonical, False)] = canonical
    for code in codes:
        mask = np.array([(int(code) >> i) & 1 for i in range(area)], bool).reshape(gh, gw)
        canonical = _center_mask(mask, gh, gw)
        bank[_shape_key(canonical, False)] = canonical
    return list(bank.values())


def _input_colors(rng, highlight, failure, ink_count=None):
    bg = int(rng.choice([c for c in range(10) if c not in (highlight, failure)]))
    divider = int(rng.choice([c for c in range(10) if c not in (bg, highlight, failure)]))
    available = [c for c in range(10) if c not in (bg, highlight)]
    # The two local role/ink coincidences are independently optional.
    for role in (divider, failure):
        if rng.random() < .45 and (ink_count is None or len(available) > ink_count):
            available.remove(role)
    if ink_count is None:
        ink_count = int(rng.integers(1, len(available) + 1))
    inks = [int(x) for x in rng.choice(available, ink_count, replace=False)]
    return bg, divider, inks


def _make_board(rng, layout, inks, ensure_confusable=False):
    n, gh, gw = layout[:3]
    size = n * n
    bank = _shape_bank(rng, gh, gw)
    minimum = (size + len(inks) - 1) // len(inks)
    maximum = min(size, len(bank))
    assert minimum <= maximum
    mode = int(rng.integers(3))
    if mode == 0:
        count = maximum
    elif mode == 1:
        count = max(minimum, maximum - int(rng.integers(1, 5)))
    else:
        count = int(rng.integers(minimum, max(minimum, maximum // 2) + 1))
    if ensure_confusable:
        assert len(inks) >= 2
        count = min(count, size - 1)
    indices = list(map(int, rng.choice(len(bank), count, replace=False)))
    # Give solid, disconnected, hollow, and narrow shapes real probability.
    special = int(rng.integers(4))
    if special not in indices:
        indices[0] = special
    chosen = [bank[i] for i in indices]
    pairs = [(j % count, j % len(inks)) for j in range(max(count, len(inks)))]
    remaining = [(s, c) for s in range(count) for c in range(len(inks)) if (s, c) not in pairs]
    rng.shuffle(remaining)
    pairs.extend(remaining[:size - len(pairs)])
    rng.shuffle(pairs)
    return [(chosen[s].copy(), inks[c]) for s, c in pairs]


def _confusable(board):
    keys = [_key(mask) for mask, _ in board]
    return {i for i, key in enumerate(keys) if keys.count(key) > 1}


def _teach_color(rng, board, selected):
    """Expose a selected and an unselected copy of the same occupancy mask."""
    keys = [_key(mask) for mask, _ in board]
    groups = [{j for j, other in enumerate(keys) if other == key} for key in dict.fromkeys(keys)]
    if any(group & set(selected) and group - set(selected) for group in groups):
        return board
    for group in groups:
        if len(group) > 1 and group <= set(selected):
            move = sorted(group)[-1]
            outside = [p for p in range(len(board)) if p not in selected and p not in group]
            target = int(rng.choice(outside))
            board[move], board[target] = board[target], board[move]
            return board
    assert not _confusable(board)
    return board


def _line_kind(positions, n):
    pos = set(positions)
    if len(pos) != n:
        return None
    if len({p // n for p in pos}) == 1:
        return "row"
    if len({p % n for p in pos}) == 1:
        return "col"
    return None


def _diagonal(positions, n):
    return set(positions) in ({i * n + i for i in range(n)},
                              {i * n + n - 1 - i for i in range(n)})


def _selection(rng, n, kind, used, required=(), edge=None):
    def permitted(pos):
        return ((n, tuple(sorted(pos))) not in used and not _diagonal(pos, n)
                and (not required or bool(set(pos) & set(required))))
    if kind in ("row", "col"):
        lines = ([[r * n + c for c in range(n)] for r in range(n)] if kind == "row"
                 else [[r * n + c for r in range(n)] for c in range(n)])
        rng.shuffle(lines)
        if edge is not None:
            # Prefer the requested edge/interior shift, but never lose a
            # required line demonstration to its color-confusion constraint.
            lines.sort(key=lambda line: ((line[0] // n if kind == "row" else line[0] % n)
                                        in (0, n - 1)) != edge)
        for line in lines:
            if permitted(line):
                return line
        kind = "scatter"
    if kind == "near":
        options = []
        for axis, index, missing in itertools.product(range(2), range(n), range(n)):
            line = [index * n + j if axis == 0 else j * n + index for j in range(n)]
            kept = line[:missing] + line[missing + 1:]
            options.extend(kept + [extra] for extra in range(n * n) if extra not in line)
        rng.shuffle(options)
        for pos in options:
            if permitted(pos):
                return pos
        kind = "scatter"
    count = int(rng.integers(1, n)) if kind == "partial" else n
    for _ in range(64):
        pos = list(map(int, rng.choice(n * n, count, replace=False)))
        if _line_kind(pos, n) is None and permitted(pos):
            return pos
    # A finite fallback rather than a rejection loop that can fail legal seeds.
    for k in [count] + [j for j in range(1, n + 1) if j != count]:
        for pos in itertools.combinations(range(n * n), k):
            if _line_kind(pos, n) is None and permitted(pos):
                return list(pos)
    raise AssertionError("selection space exhausted")


def _query_order(rng, positions, n):
    order = sorted(positions)
    if rng.random() < .7:
        rng.shuffle(order)
    occupied = sorted(map(int, rng.choice(n, len(order), replace=False)))
    query = [None] * n
    for slot, pos in zip(occupied, order):
        query[slot] = pos
    return query


def _render_pair(layout, board, query, bg, divider, highlight, failure, orientation):
    n, sh, sw, margin, board0, end, height = _geometry(layout)
    _, gh, gw, ph, pw, _, _, _ = layout
    inp = np.full((height, n * sw + 2 * margin), bg, np.int8)
    inp[sh, :] = divider
    inp[end, :] = divider
    for index, (mask, ink) in enumerate(board):
        r, c = divmod(index, n)
        block = inp[board0 + r * sh + ph:board0 + r * sh + ph + gh,
                    margin + c * sw + pw:margin + c * sw + pw + gw]
        block[mask] = ink
    for slot, index in enumerate(query):
        if index is not None:
            mask, ink = board[index]
            block = inp[ph:ph + gh, margin + slot * sw + pw:margin + slot * sw + pw + gw]
            block[mask] = ink
    selected = [p for p in query if p is not None]
    out = inp.copy()
    for p in selected:
        r, c = divmod(p, n)
        block = out[board0 + r * sh:board0 + (r + 1) * sh,
                    margin + c * sw:margin + (c + 1) * sw]
        block[block == bg] = highlight
    win = _line_kind(selected, n) is not None
    if win:
        tray = out[:sh]
        tray[tray == bg] = highlight
    out[end + 1:] = highlight if win else failure
    return {"input": _orient(inp, orientation), "output": _orient(out, orientation)}


def _rearrange(rng, board, old_selected=None, new_selected=None):
    if old_selected is None:
        return [board[int(i)] for i in rng.permutation(len(board))]
    result = [None] * len(board)
    for old, new in zip(old_selected, new_selected):
        result[new] = board[old]
    rest = [board[i] for i in range(len(board)) if i not in old_selected]
    rng.shuffle(rest)
    for i in range(len(result)):
        if result[i] is None:
            result[i] = rest.pop()
    return result


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    nt, nq = _choose_counts(seed, rng, num_train, num_test)
    highlight, failure = map(int, rng.choice(10, 2, replace=False))
    layout0 = _SOURCE_LAYOUT if rng.random() < .16 else _sample_layout(rng)
    board_mode = str(rng.choice(["shared", "rearrange", "fresh", "mixed"]))
    varying_layout = nt >= 4 and rng.random() < .6
    varying_palette = rng.random() < .5
    orientation0 = int(rng.integers(8))
    orientation_mode = (str(rng.choice(["fixed", "mixed", "held_out"], p=[.5, .35, .15]))
                        if nt >= 4 else "fixed")
    first_axis = str(rng.choice(["row", "col"]))
    other_axis = "col" if first_axis == "row" else "row"
    # With two demonstrations, do not demand an unwitnessed winning axis.
    failure_lesson = "near" if board_mode == "rearrange" or rng.random() < .5 else "partial"
    schedule = [first_axis, failure_lesson]
    if nt >= 3:
        schedule.append(other_axis)
    if nt >= 4:
        schedule.append("partial" if failure_lesson == "near" else "near")
    while len(schedule) < nt:
        schedule.append(str(rng.choice(["row", "col", "near", "scatter", "partial"])))
    empty_taught = "partial" in schedule
    full_failure_taught = any(k in ("near", "scatter") for k in schedule)
    edge_shift = nt >= 3 and rng.random() < .35
    train_edge = bool(rng.integers(2)) if edge_shift else None
    for _ in range(nq):
        if rng.random() < .5:
            schedule.append(str(rng.choice(["row", "col"])) if nt >= 3 else first_axis)
        else:
            kinds = (["near", "scatter"] if full_failure_taught else []) + (["partial"] if empty_taught else [])
            schedule.append(str(rng.choice(kinds)))
    order_lesson = board_mode == "shared" and nt >= 5 and rng.random() < .3
    used_selections, pairs = set(), []
    anchor = first_selected = first_query = fixed_colors = None
    for index, kind in enumerate(schedule):
        training = index < nt
        layout = layout0
        if varying_layout and index >= 2:
            inventory = layout0[:3] if board_mode in ("shared", "rearrange") else None
            different = layout0[0] if index == 2 and inventory is None else None
            layout = _sample_layout(rng, inventory=inventory, different_n=different)
        if order_lesson and index == 4:
            layout = layout0
        n = layout[0]
        reuse = (anchor is not None and layout[:3] == layout0[:3]
                 and (board_mode in ("shared", "rearrange")
                      or (board_mode == "mixed" and rng.random() < .5)))
        if reuse:
            bg, divider, inks = fixed_colors
            board = [(mask.copy(), ink) for mask, ink in anchor]
            if varying_palette and index >= 2 and not (order_lesson and index == 4):
                bg, divider, new_inks = _input_colors(rng, highlight, failure, len(inks))
                remap = dict(zip(inks, new_inks))
                board = [(mask, remap[ink]) for mask, ink in board]
                inks = new_inks
        else:
            if fixed_colors is not None and not varying_palette:
                bg, divider, inks = fixed_colors
            else:
                bg, divider, inks = _input_colors(rng, highlight, failure)
            # Ten translation-normalized shapes fit a 2x2 canvas.
            minimum_inks = 3 if layout[1:3] == (2, 2) and n == 5 else 2
            if len(inks) < minimum_inks:
                extras = [c for c in range(10) if c not in (bg, highlight) and c not in inks]
                inks = inks + list(map(int, rng.choice(extras, minimum_inks - len(inks), replace=False)))
            board = _make_board(rng, layout, inks,
                                ensure_confusable=index == 0 and board_mode in ("fresh", "mixed"))
        required = _confusable(board) if index == 0 else ()
        edge = (train_edge if training else not train_edge) if edge_shift else None
        selected = _selection(rng, n, kind, used_selections, required, edge=edge)
        if index == 0:
            board = _teach_color(rng, board, selected)
        if index == 1 and board_mode == "rearrange":
            board = _rearrange(rng, anchor, first_selected, selected)
        elif reuse and board_mode == "rearrange":
            board = _rearrange(rng, board)
        elif reuse and board_mode == "mixed" and rng.random() < .5:
            board = _rearrange(rng, board)
        query = _query_order(rng, selected, n)
        if order_lesson and index == 4:
            board = [(mask.copy(), ink) for mask, ink in anchor]
            bg, divider, inks = fixed_colors
            selected = first_selected
            query = first_query[1:] + first_query[:1]
        else:
            used_selections.add((n, tuple(sorted(selected))))
        if index == 0:
            anchor = [(mask.copy(), ink) for mask, ink in board]
            fixed_colors = bg, divider, list(inks)
            first_selected, first_query = list(selected), list(query)
        orientation = orientation0
        if orientation_mode == "mixed" and index >= 2:
            orientation = int(rng.integers(8))
        elif orientation_mode == "held_out":
            orientation = ((orientation0 + index % 2) % 4 if training else
                           (orientation0 + 2 + index % 2) % 4) + (orientation0 // 4) * 4
        if order_lesson and index == 4:
            orientation = orientation0
        pair = _render_pair(layout, board, query, bg, divider, highlight, failure, orientation)
        assert _digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS
        pairs.append(pair)
    # Neither split has a fixed outcome schedule in its returned order.
    puzzle = {"train": [pairs[int(i)] for i in rng.permutation(nt)],
              "test": [pairs[nt + int(i)] for i in rng.permutation(nq)]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _parse_canonical(grid):
    bg = int(grid[0, 0])
    dividers = [r for r in range(len(grid))
                if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])]
    assert len(dividers) == 2
    sh, end = dividers
    assert 4 <= sh <= 6 and int(grid[sh, 0]) == int(grid[end, 0])
    assert 1 <= len(grid) - end - 1 <= 3 and np.all(grid[end + 1:] == bg)
    choices = []
    for n, gap, margin in itertools.product((3, 4, 5), (1, 2), (1, 2)):
        if end != sh + 1 + gap + n * sh or (grid.shape[1] - 2 * margin) % n:
            continue
        sw = (grid.shape[1] - 2 * margin) // n
        if 4 <= sw <= 8:
            choices.append((n, gap, margin, sw))
    assert len(choices) == 1
    n, gap, margin, sw = choices[0]
    board0 = sh + 1 + gap
    expected = np.full(grid.shape, bg, np.int8)
    expected[sh], expected[end] = grid[sh], grid[end]
    board, lookup, trimmed = [], {}, set()
    for r, c in itertools.product(range(n), repeat=2):
        rs = slice(board0 + r * sh, board0 + (r + 1) * sh)
        cs = slice(margin + c * sw, margin + (c + 1) * sw)
        block = grid[rs, cs]
        assert np.all(block[[0, -1]] == bg) and np.all(block[:, [0, -1]] == bg)
        assert len(np.unique(block[block != bg])) == 1
        key, trimkey = _key(block), _key(_trim(block, block != bg))
        assert key not in lookup and trimkey not in trimmed
        lookup[key] = r * n + c
        trimmed.add(trimkey)
        board.append(block.copy())
        expected[rs, cs] = block
    order, empty = [], []
    for c in range(n):
        cs = slice(margin + c * sw, margin + (c + 1) * sw)
        block = grid[:sh, cs]
        if np.all(block == bg):
            empty.append(c)
        else:
            assert _key(block) in lookup
            order.append(lookup[_key(block)])
        expected[:sh, cs] = block
    assert 1 <= len(order) <= n and len(set(order)) == len(order)
    assert np.array_equal(expected, grid), "unexpected ink outside the slot lattice"
    return {"grid": grid, "bg": bg, "divider": int(grid[sh, 0]), "n": n,
            "sh": sh, "sw": sw, "margin": margin, "board0": board0, "end": end,
            "board": board, "order": order, "empty": empty, "line": _line_kind(order, n)}


def _analyze_input(grid):
    candidates = []
    for rotation in range(4):
        try:
            info = _parse_canonical(np.rot90(grid, rotation))
        except AssertionError:
            continue
        info["rotation"] = rotation
        candidates.append(info)
    assert len(candidates) == 1, "tray/board/status segmentation must be unambiguous"
    return candidates[0]


def _predict(info, highlight, failure):
    # Pixelwise reconstruction independent of the construction-time renderer.
    grid = info["grid"]
    result = grid.copy()
    n, sh, sw, margin = (info[k] for k in ("n", "sh", "sw", "margin"))
    selected = set(info["order"])
    win = any(all(r * n + c in selected for c in range(n)) for r in range(n)) or any(
        all(r * n + c in selected for r in range(n)) for c in range(n))
    assert len(selected) <= n
    for r, c in np.ndindex(grid.shape):
        if r > info["end"]:
            result[r, c] = highlight if win else failure
        elif grid[r, c] == info["bg"]:
            if r < sh and win:
                result[r, c] = highlight
            elif info["board0"] <= r < info["end"] and margin <= c < margin + n * sw:
                slot = ((r - info["board0"]) // sh) * n + (c - margin) // sw
                if slot in selected:
                    result[r, c] = highlight
    return np.rot90(result, -info["rotation"]).copy()


def _learn_colors(train_pairs):
    highlights, failures, wins = set(), set(), 0
    for pair in train_pairs:
        info = _analyze_input(pair["input"])
        out = np.rot90(pair["output"], info["rotation"])
        p = info["order"][0]
        r = info["board0"] + (p // info["n"]) * info["sh"]
        c = info["margin"] + (p % info["n"]) * info["sw"]
        highlights.add(int(out[r, c]))
        status = np.unique(out[info["end"] + 1:])
        assert len(status) == 1
        if info["line"]:
            assert int(status[0]) == int(out[r, c])
            wins += 1
        else:
            failures.add(int(status[0]))
    assert len(highlights) == 1 and wins >= 1 and len(failures) <= 1
    highlight = next(iter(highlights))
    failure = next(iter(failures)) if failures else None
    assert failure != highlight
    return highlight, failure


def solve_from_training(train_pairs, test_input):
    """Use demonstrations and the new input only, never its output label."""
    highlight, failure = _learn_colors(train_pairs)
    info = _analyze_input(test_input)
    assert info["line"] or failure is not None
    return _predict(info, highlight, failure)


def _grid_checks(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= dim <= 30 for dim in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))


def _digest(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.uint8).tobytes()).hexdigest()


_SOURCE_INPUT_DIGESTS = {
    "d897120ff21046f200bade8ebd94131e15bb728c3ce86648356559eda1351d2a",
    "56838f9b34b71cd3f8e5d3d48065a5eab7da6c58d23f8207fe5355b6516af8b0",
    "24eb3709ceddc634950480b1516ed6b90e2e11ab50828f567a9c014042d01899",
    "e0f44d7681fdc9ae8ad68c0b5ccd6497f10c6426a0acf7d1dbe3de3b5126191d",
    "1576d56f9fbf340b686a0ca1f735b80e64ed05bf1379b00dd01b50929d1f5ab0",
    "0002cac864a65004223ab4771edbaa1cd172d7757714159cbebb352f47436b9c",
    "6ba124a76a8ec0ef96a193398b62fee69b882d480db0c1ebaaf170cdd7362ca0",
}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, analyses = set(), []
    all_pairs = puzzle["train"] + puzzle["test"]
    for pair in all_pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        _grid_checks(inp)
        _grid_checks(out)
        assert inp.shape == out.shape and not np.array_equal(inp, out)
        digest = _digest(inp)
        assert digest not in seen and digest not in _SOURCE_INPUT_DIGESTS
        seen.add(digest)
        info = _analyze_input(inp)
        assert not _diagonal(info["order"], info["n"])
        analyses.append(info)
    highlight, failure = _learn_colors(puzzle["train"])
    assert failure is not None
    for pair, info in zip(all_pairs, analyses):
        assert highlight not in np.unique(pair["input"])
        assert info["bg"] not in (highlight, failure) and info["divider"] not in (highlight, failure)
        for block in info["board"]:
            rr, cc = np.where(block != info["bg"])
            assert int(rr.min()) == (len(block) - int(rr.max() - rr.min() + 1)) // 2
            assert abs(int(cc.min()) - (block.shape[1] - 1 - int(cc.max()))) <= 1
        assert np.array_equal(_predict(info, highlight, failure), pair["output"])
    train_info, test_info = analyses[:nt], analyses[nt:]
    if any(i["empty"] for i in test_info):
        assert any(i["empty"] for i in train_info), "empty query behavior must be taught"
    if any(not i["line"] and not i["empty"] for i in test_info):
        assert any(not i["line"] and not i["empty"] for i in train_info), "full-tray failure must be demonstrated"
    if len({i["n"] for i in analyses}) > 1:
        assert len({i["n"] for i in train_info}) > 1, "varying line length must be demonstrated"
    train_axes = {i["line"] for i in train_info if i["line"]}
    assert all(i["line"] in train_axes for i in test_info if i["line"])
    def color_relevant(info):
        shapes = [_shape_key(b, info["bg"]) for b in info["board"]]
        return any(shapes[p] == shapes[q] for p in info["order"]
                   for q in range(len(shapes)) if q not in info["order"])
    if any(color_relevant(i) for i in test_info):
        assert any(color_relevant(i) for i in train_info), "training must teach color-sensitive lookup"
    training_copies = [{k: v.copy() for k, v in pair.items()} for pair in puzzle["train"]]
    for pair in puzzle["test"]:
        predicted = solve_from_training(training_copies, pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    todo = [next(iter(cells))]
    seen = set(todo)
    while todo:
        r, c = todo.pop()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _has_hole(mask):
    background = ~np.pad(mask, 1)
    todo, seen = [(0, 0)], {(0, 0)}
    while todo:
        r, c = todo.pop()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            p = r + dr, c + dc
            if (0 <= p[0] < len(background) and 0 <= p[1] < background.shape[1]
                    and background[p] and p not in seen):
                seen.add(p)
                todo.append(p)
    return len(seen) != int(background.sum())


def measure_pair(pair):
    """Task features reconstructed from pixels, for self-tests and scratch QA."""
    info = _analyze_input(pair["input"])
    masks = [_trim(b != info["bg"], b != info["bg"]) for b in info["board"]]
    shapes = [_key(m) for m in masks]
    inks = [int(b[b != info["bg"]][0]) for b in info["board"]]
    selected, n = set(info["order"]), info["n"]
    near = (len(selected) == n and not info["line"] and
            max([sum(r * n + c in selected for c in range(n)) for r in range(n)] +
                [sum(r * n + c in selected for r in range(n)) for c in range(n)]) == n - 1)
    out = np.rot90(pair["output"], info["rotation"])
    fail = int(out[-1, 0]) if not info["line"] else None
    return {
        "n": n, "slot": [info["sh"], info["sw"]],
        "layout": [n, info["sh"], info["sw"], info["margin"],
                   info["board0"] - info["sh"] - 1, len(info["grid"]) - info["end"] - 1],
        "rotation": info["rotation"], "background": info["bg"], "divider": info["divider"],
        "ink_count": len(set(inks)), "unique_shapes": len(set(shapes)),
        "max_shape_multiplicity": max(shapes.count(s) for s in shapes),
        "shape_reuse_cells": sum(shapes.count(s) > 1 for s in shapes),
        "color_sensitive_selection": any(shapes[p] == shapes[q] for p in selected
                                         for q in range(len(shapes)) if q not in selected),
        "divider_ink": info["divider"] in inks, "failure_ink": fail in inks if fail is not None else None,
        "occupancy": len(selected), "empty": info["empty"], "ordered": info["order"] == sorted(selected),
        "selection": [n, sorted(selected)], "outcome": info["line"] or "failure",
        "edge_line": (min(selected) // n in (0, n - 1) if info["line"] == "row" else
                      min(selected) % n in (0, n - 1) if info["line"] == "col" else None),
        "near_miss": near, "disconnected": sum(not _connected(m) for m in masks),
        "holes": sum(_has_hole(m) for m in masks), "solid": sum(bool(np.all(m)) for m in masks),
        "dense": sum(float(m.mean()) >= .8 for m in masks),
        "narrow": sum(m.shape[0] < info["sh"] - 2 or m.shape[1] < info["sw"] - 2 for m in masks),
        "ink_cells": [int(m.sum()) for m in masks], "glyph_bounds": [list(m.shape) for m in masks],
        "board": hashlib.sha256(b"".join(b.tobytes() for b in info["board"])).hexdigest(),
        "inventory": sorted(hashlib.sha256(_trim(b, b != info["bg"]).tobytes() +
                                            bytes(_trim(b, b != info["bg"]).shape)).hexdigest()
                            for b in info["board"]),
    }


def _source_regression():
    # Transcribed original board; retain original small-glyph offsets. Hashes
    # cover every pixel of the seven original output grids, not just status.
    tokens = ('282828282', '088000080', '887887777', '989998999',
              '788777788', '111818111', '866886866', '000000000',
              '484484444', '228222828', '558588558', '181118181',
              '988999889', '666686686', '848444848', '222282222')
    queries = ((7, 5, 4, 6), (1, 5, 13, 9), (0, 8, 9, 15),
               (7, 9, 4, 6), (5, 11, 3, None), (7, 11, 3, 15), (0, 4, 9, 10))
    expected = (
        '74f43a5ffdc4befcd01d3f83f9da0567cd19bc18d2bc5799a7259fc6f576a081',
        'c5396ea065877fa6fcbc9fdbfd6dd414cbe24c96b23ba12fe91f088902d15501',
        '8cdd1a9ba0fb64320a669d05d4b03bf7035bbcc22e826ed08de20ec51e1e44dd',
        '127ce9fcbc0b77bab26b0aeaed11631851387034f4baae1f8e4f4975cbded06a',
        '5b756a5b71cb3d8853f6341845ffff39a1d485fd9c5a593798e8292e76817e01',
        '94536d3254887592cb2b9c55fad0c322476d47d1727374deea3492b3d8c09606',
        '0592408fb2ec7d82a62fb04b3c00c4df39400574a8b0587af2e94946fa101d92')
    arrays = [np.array(list(map(int, s)), np.int8).reshape(3, 3) for s in tokens]
    for query, digest in zip(queries, expected):
        inp = np.full((30, 22), 8, np.int8)
        inp[[5, 27]] = 6
        for p, token in enumerate(arrays):
            r, c = divmod(p, 4)
            inp[8 + r * 5:11 + r * 5, 2 + c * 5:5 + c * 5] = token
        for c, p in enumerate(query):
            if p is not None:
                inp[1:4, 2 + c * 5:5 + c * 5] = arrays[p]
        assert _digest(inp) in _SOURCE_INPUT_DIGESTS
        assert _digest(_predict(_analyze_input(inp), 3, 2)) == digest


def self_test() -> None:
    _source_regression()
    # Explicitly cover each formerly excluded source occupancy pattern.
    bank = {_shape_key(mask, False) for mask in _shape_bank(np.random.default_rng(91), 3, 3)}
    for bits in ('101010101', '101110111', '101110101', '111111111', '111101111'):
        mask = np.array([c == '1' for c in bits]).reshape(3, 3)
        assert _shape_key(mask, False) in bank
    records, puzzles, counts, conventions, puzzle_keys = [], [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        conventions.add(_learn_colors(puzzle["train"]))
        puzzle_keys.add(tuple(_digest(pair["input"]) for s in ("train", "test") for pair in puzzle[s]))
        rows = [(split, measure_pair(pair)) for split in ("train", "test") for pair in puzzle[split]]
        records.extend(rows)
        puzzles.append(rows)
        repeated = generate_puzzle(seed)
        assert all(np.array_equal(a[k], b[k]) for split in puzzle
                   for a, b in zip(puzzle[split], repeated[split]) for k in a)
    assert len({c[0] for c in counts}) >= 4 and len({c[1] for c in counts}) >= 3
    assert len(conventions) >= 20 and len(puzzle_keys) >= 40
    values = [r for _, r in records]
    tests = [r for s, r in records if s == "test"]
    assert {r["n"] for r in values} == {3, 4, 5}
    assert len({tuple(r["layout"]) for r in values}) >= 25
    assert len({r["board"] for r in values}) >= 80
    assert len({(r["selection"][0], tuple(r["selection"][1])) for r in tests}) >= 35
    assert {r["outcome"] for r in tests} == {"row", "col", "failure"}
    assert {r["occupancy"] for r in tests} >= {1, 2, 3, 4, 5}
    assert {r["ordered"] for r in tests} == {False, True}
    assert {r["edge_line"] for r in tests} >= {False, True}
    assert len({r["ink_count"] for r in values}) >= 6
    assert len({r["unique_shapes"] for r in values}) >= 10
    assert len({r["max_shape_multiplicity"] for r in values}) >= 4
    assert {r["divider_ink"] for r in values} == {False, True}
    assert {r["failure_ink"] for r in values} >= {False, True}
    for feature in ("disconnected", "holes", "solid", "dense", "narrow", "near_miss"):
        assert sum(bool(r[feature]) for r in values) >= 20, feature
    for feature in ("board", "layout", "background", "ink_count", "rotation"):
        assert sum(len({str(r[feature]) for _, r in rows}) > 1 for rows in puzzles) >= 5, feature
    assert any(len({r["board"] for _, r in rows}) == 1 for rows in puzzles)
    assert sum(all(len({str(r[k]) for _, r in rows}) > 1 for k in ("board", "background", "layout"))
               for rows in puzzles) >= 3
    assert len({(r["background"], r["divider"]) for r in values}) >= 20
    for n in (3, 4, 5):
        assert len({p for r in tests if r["n"] == n for p in r["empty"]}) >= 2
    for i in range(4):
        outcomes = set()
        for rows in puzzles:
            tests_here = [r for s, r in rows if s == "test"]
            if len(tests_here) > i:
                outcomes.add(tests_here[i]["outcome"] == "failure")
        assert len(outcomes) == 2
    # Reject real semantic damage, even when all schema and array bounds hold.
    baseline = generate_puzzle(123)
    for orientation in range(8):
        transformed = {s: [{k: _orient(v, orientation) for k, v in pair.items()}
                           for pair in baseline[s]] for s in baseline}
        validate_generated_puzzle(transformed)
    for defect in ("ink", "status", "divider", "duplicate"):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in baseline[s]] for s in baseline}
        pair = damaged["test"][0]
        info = _analyze_input(pair["input"])
        out = np.rot90(pair["output"], info["rotation"])
        if defect == "ink":
            p = info["order"][0]
            r, c = np.argwhere(info["board"][p] != info["bg"])[0]
            r += info["board0"] + (p // info["n"]) * info["sh"]
            c += info["margin"] + (p % info["n"]) * info["sw"]
            out[r, c] = (int(out[r, c]) + 1) % 10
        elif defect == "status":
            out[-1, 0] = (int(out[-1, 0]) + 1) % 10
        elif defect == "divider":
            out[info["sh"], 0] = info["bg"]
        else:
            damaged["train"][1] = {k: v.copy() for k, v in damaged["train"][0].items()}
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            continue
        raise AssertionError("validator accepted " + defect)
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1000 + (train or 0) * 10 + (test or 0), train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(123, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal count request was accepted")
