import numpy as np
from functools import lru_cache


_SOURCE_PAIR_ROWS = (
    (
        (
            "8888888888888",
            "8888848888888",
            "8888888888888",
            "8833833884888",
            "8838883888888",
            "8838883888888",
            "8883838888848",
            "8833833888888",
            "8888888888888",
            "8888888888888",
            "8811188868888",
            "8811188868888",
            "8888888888888",
        ),
        (
            "33433",
            "31113",
            "31113",
            "43634",
            "33633",
        ),
    ),
    (
        (
            "3333333333333",
            "3633333388833",
            "3633333338333",
            "3331113338353",
            "3313331333333",
            "3311311353333",
            "3331313333333",
            "3332323333433",
            "3322322333433",
            "3323332334443",
            "3332223333333",
            "3633333333533",
            "3633333533333",
        ),
        (
            "51115",
            "18881",
            "11811",
            "61816",
            "62426",
            "22422",
            "24442",
            "52225",
        ),
    ),
    (
        (
            "414444444444444",
            "444444444444444",
            "444448884444666",
            "443484448444464",
            "443448484414444",
            "443448484444444",
            "443448484444444",
            "444448484444444",
            "444484448444144",
            "444448884444444",
            "434444444444444",
            "434444494444444",
            "434444494446444",
            "434414444466644",
            "444444444444444",
        ),
        (
            "18881",
            "86668",
            "38683",
            "38983",
            "38983",
            "38683",
            "86668",
            "18881",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_cells(cells) -> tuple:
    cells = list(cells)
    _require(len(cells) > 0, "Cannot normalize an empty component")
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _cell_components(cells) -> list:
    remaining = set(cells)
    components = []

    while remaining:
        start = remaining.pop()
        stack = [start]
        component = []

        while stack:
            r, c = stack.pop()
            component.append((r, c))

            for neighbor in (
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components


def _mask_components(mask: np.ndarray) -> list:
    cells = {
        (int(r), int(c))
        for r, c in np.argwhere(mask)
    }
    return _cell_components(cells)


def _check_optional_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not minimum <= int(value) <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple:
    _check_optional_count(num_train, "num_train", 2, 10)
    _check_optional_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        test_count = 1 + seed % 4
        low = max(2, test_count + 1)
        span = 11 - low
        train_count = low + (seed // 4) % span
    elif num_train is None:
        test_count = int(num_test)
        low = max(2, test_count + 1)
        train_count = int(rng.integers(low, 11))
    elif num_test is None:
        train_count = int(num_train)
        high = min(4, train_count - 1)
        if high < 1:
            raise ValueError("No legal test count for num_train")
        test_count = int(rng.integers(1, high + 1))
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if not 2 <= train_count <= 10:
        raise ValueError("Illegal number of training pairs")
    if not 1 <= test_count <= 4:
        raise ValueError("Illegal number of test pairs")
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _array_key(array: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in array.shape),
        tuple(int(x) for x in array.ravel()),
    )


def _rows_key(array: np.ndarray) -> tuple:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in array
    )


def _is_source_duplicate(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    input_rows = _rows_key(input_grid)
    output_rows = _rows_key(output_grid)

    return any(
        input_rows == source_input
        and output_rows == source_output
        for source_input, source_output in _SOURCE_PAIR_ROWS
    )


class _SearchLimit(Exception):
    """A bounded proof search timed out; such a construction is rejected."""


def _assembly(board, background, pieces):
    """Exact cover by the supplied oriented pieces, without output symmetry.

    Equal pieces are counted rather than individually permuted. The memoized
    answers are colored grids, so exchanging same-color pieces is harmless.
    Two different colored answers suffice to disprove uniqueness.
    """
    h, w = board.shape
    holes = {(int(r), int(c)) for r, c in np.argwhere(board == background)}
    if sum(len(shape) for _, shape in pieces) != len(holes):
        return (), 0
    inventory = {}
    for color, shape in pieces:
        key = (int(color), _normalize_cells(shape))
        inventory[key] = inventory.get(key, 0) + 1
    types = sorted(inventory)
    positions = {cell: i for i, cell in enumerate(sorted(holes))}
    by_cell = [[] for _ in holes]
    for t, (color, shape) in enumerate(types):
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        for r in range(h - sh + 1):
            for c in range(w - sw + 1):
                cells = [(r + dr, c + dc) for dr, dc in shape]
                if not all(cell in holes for cell in cells):
                    continue
                indices = [positions[cell] for cell in cells]
                mask = sum(1 << i for i in indices)
                paint = sum((color + 1) << (4 * i) for i in indices)
                move = (t, mask, paint)
                for i in indices:
                    by_cell[i].append(move)
    nodes = 0

    @lru_cache(maxsize=None)
    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        if nodes > 12000:
            raise _SearchLimit()
        if not remaining:
            return (0,) if not any(counts) else ()
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            choices = [m for m in by_cell[i]
                       if counts[m[0]] and m[1] & remaining == m[1]]
            if not choices:
                return ()
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for t, mask, paint in best:
            rest = list(counts)
            rest[t] -= 1
            for suffix in search(remaining ^ mask, tuple(rest)):
                answers.add(paint | suffix)
                if len(answers) == 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search((1 << len(holes)) - 1,
                   tuple(inventory[t] for t in types))
    results = []
    for code in codes:
        result = board.copy()
        for cell, i in positions.items():
            result[cell] = ((code >> (4 * i)) & 15) - 1
        results.append(result)
    return tuple(results), nodes


def _board_candidates(inp):
    """Recover rectangles from complete color extents, not largest objects.

    The source has disjoint fixed/payload colors within each scene. Exact
    inventory conservation gives the rectangle's area directly from the input.
    Its fixed colors can comprise arbitrarily many disconnected components.
    """
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    foreground = inp != background
    area = int(foreground.sum())
    colors = [int(c) for c in colors if c != background]
    extents = []
    for color in colors:
        rr, cc = np.where(inp == color)
        extents.append((int(rr.min()), int(cc.min()),
                        int(rr.max()) + 1, int(cc.max()) + 1))
    seen = set()
    for subset in range(1, (1 << len(colors)) - 1):
        ids = [i for i in range(len(colors)) if subset & (1 << i)]
        top = min(extents[i][0] for i in ids)
        left = min(extents[i][1] for i in ids)
        bottom = max(extents[i][2] for i in ids)
        right = max(extents[i][3] for i in ids)
        bbox = (top, left, bottom - top, right - left)
        if bbox in seen or bbox[2] * bbox[3] != area or min(bbox[2:]) < 3:
            continue
        seen.add(bbox)
        board = inp[top:bottom, left:right]
        fixed_colors = set(int(c) for c in np.unique(board)) - {background}
        if not (np.array_equal(board, board[:, ::-1]) or
                np.array_equal(board, board[::-1, :])):
            continue
        outside = foreground.copy()
        outside[top:bottom, left:right] = False
        if fixed_colors & set(int(c) for c in inp[outside]):
            continue
        components = _mask_components(outside)
        if not components or any(len({int(inp[r, c]) for r, c in comp}) != 1
                                 for comp in components):
            continue
        pieces = [(int(inp[comp[0]]), _normalize_cells(comp)) for comp in components]
        yield background, bbox, board, pieces, components


@lru_cache(maxsize=768)
def _predict_key(key):
    inp = np.array(key[1], dtype=np.int8).reshape(key[0])
    feasible = []
    for background, bbox, board, pieces, components in _board_candidates(inp):
        answers, nodes = _assembly(board, background, pieces)
        if not answers:
            continue
        _require(len(answers) == 1, "Inventory admits different colored answers")
        feasible.append((answers[0], background, bbox, pieces, components, nodes))
    _require(len(feasible) == 1, "Input must have exactly one feasible board crop")
    return feasible[0]


def solve(train_pairs, test_input):
    """Predict from visible grids only; no seed, labels, or construction state.

    Training establishes exact inventory assembly and preservation. No literal
    palette, width, or orientation is a hidden convention needed for inference.
    """
    return _predict_key(_array_key(test_input))[0].copy()


def _random_shape(rng, h, w):
    family = int(rng.integers(0, 7))
    if family == 0:
        rh = int(rng.integers(1, min(3, h) + 1))
        rw = int(rng.integers(2, min(6, w) + 1))
        cells = {(r, c) for r in range(rh) for c in range(rw)}
    elif family == 1:
        length = int(rng.integers(2, min(7, max(h, w)) + 1))
        cells = {(0, c) for c in range(length)}
    elif family == 2:
        arm = int(rng.choice([3, 5])) if w >= 5 else 3
        stem = int(rng.integers(2, min(5, h) + 1))
        cells = {(0, c) for c in range(arm)} | {(r, arm // 2) for r in range(stem)}
    elif family == 3:
        a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        cells = {(r, 0) for r in range(a)} | {(a - 1, c) for c in range(b)}
    elif family == 4:
        cells = {(0, 0)}
    else:
        cells = {(0, 0)}
        for _ in range(int(rng.integers(2, 10))):
            neighbors = sorted({(r + dr, c + dc) for r, c in cells
                                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - cells)
            cells.add(neighbors[int(rng.integers(len(neighbors)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    return _normalize_cells(cells)


def _orbit(cells, h, w, dual):
    items = [frozenset(cells), frozenset((r, w - 1 - c) for r, c in cells)]
    if dual:
        items += [frozenset((h - 1 - r, c) for r, c in item) for item in items[:]]
    items = list(dict.fromkeys(items))
    used = set()
    for item in items:
        if used & item:
            return []
        used.update(item)
    return items


def _make_layout(rng, h, w, dual, easy):
    occupied = set()
    groups = []
    interior = h >= 6 and w >= 6 and rng.random() < 0.2
    target = float(rng.uniform(0.22, 0.43) if easy else rng.uniform(0.38, 0.65))
    if interior:
        target = min(target, 0.55 * (h - 2) * (w - 2) / (h * w))
    for _ in range(100):
        shape = _random_shape(rng, h, w)
        sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
        margin = int(interior)
        if sh > h - 2 * margin or sw > w - 2 * margin:
            continue
        r = int(rng.integers(margin, h - sh + 1 - margin))
        c = int(rng.integers(margin, w - sw + 1 - margin))
        orbit = _orbit({(r + dr, c + dc) for dr, dc in shape}, h, w, dual)
        if not orbit:
            continue
        cells = set().union(*orbit)
        if cells & occupied or len(cells | occupied) > h * w * 0.7:
            continue
        if easy and any((r + dr, c + dc) in occupied for r, c in cells
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
            continue
        remaining = {(r, c) for r in range(h) for c in range(w)} - cells - occupied
        if not remaining or len(_normalize_cells(remaining)) < 6:
            continue
        rr, cc = zip(*remaining)
        if min(rr) != 0 or max(rr) != h - 1 or min(cc) != 0 or max(cc) != w - 1:
            continue
        occupied.update(cells)
        groups.append(orbit)
        if len(occupied) >= target * h * w and len(groups) >= 2:
            break
        if sum(map(len, groups)) >= 14:
            break
    if len(groups) < 2 or sum(map(len, groups)) > 18:
        return None
    return groups, occupied


def _color_layout(rng, h, w, groups, occupied, palette, dual):
    background = int(palette[0])
    nf = int(rng.choice([1, 1, 2, 2, 3]))
    fixed_colors = list(map(int, palette[1:1 + nf]))
    # Shapes which are translations of one another must share a color: any
    # different-color swap would necessarily make the assembly ambiguous.
    parents = list(range(len(groups)))

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    owners = {}
    for i, group in enumerate(groups):
        for cells in group:
            sig = _normalize_cells(cells)
            if sig in owners:
                parents[root(i)] = root(owners[sig])
            owners[sig] = i
    roots = sorted({root(i) for i in range(len(groups))})
    if len(roots) < 2:
        return None
    npayload = int(rng.integers(2, min(5, len(roots)) + 1))
    payload = list(map(int, palette[1 + nf:1 + nf + npayload]))
    assignment = [payload[i % len(payload)] for i in range(len(roots))]
    rng.shuffle(assignment)
    colors = dict(zip(roots, assignment))
    board = np.full((h, w), background, dtype=np.int8)
    style = int(rng.integers(3))
    row_cuts = sorted(rng.choice(h - 1, size=min(nf - 1, h - 1), replace=False) + 1)
    col_cuts = sorted(rng.choice((w + 1) // 2, size=min(nf - 1, (w + 1) // 2), replace=False))
    for r in range(h):
        for c in range((w + 1) // 2):
            if (r, c) in occupied:
                continue
            rr = min(r, h - 1 - r) if dual else r
            if style == 0:
                index = sum(rr >= x for x in row_cuts) % nf
            elif style == 1:
                index = (sum(rr >= x for x in row_cuts) + sum(c >= x for x in col_cuts)) % nf
            else:
                index = int(rng.integers(nf))
            cells = {(r, c), (r, w - 1 - c)}
            if dual:
                cells |= {(h - 1 - r0, c0) for r0, c0 in list(cells)}
            for cell in cells:
                board[cell] = fixed_colors[index]
    output = board.copy()
    pieces = []
    for i, group in enumerate(groups):
        for cells in group:
            color = colors[root(i)]
            pieces.append((color, _normalize_cells(cells)))
            for cell in cells:
                output[cell] = color
    return board, output, pieces


def _scatter(rng, board, pieces, background):
    h, w = board.shape
    # The same canvas range serves both regimes, with no mandatory outer border.
    for attempt in range(12):
        low_h = max(12, h + 4)
        low_w = max(12, w + 4)
        ch = int(rng.integers(low_h, 31)) if attempt < 9 else 30
        cw = int(rng.integers(low_w, 31)) if attempt < 9 else 30
        br, bc = int(rng.integers(ch - h + 1)), int(rng.integers(cw - w + 1))
        canvas = np.full((ch, cw), background, dtype=np.int8)
        canvas[br:br + h, bc:bc + w] = board
        blocked = np.zeros((ch, cw), dtype=bool)
        blocked[max(0, br - 1):min(ch, br + h + 1),
                max(0, bc - 1):min(cw, bc + w + 1)] = True
        order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i][1]), float(rng.random())))
        for i in order:
            color, shape = pieces[i]
            sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
            candidates = [(r, c) for r in range(ch - sh + 1) for c in range(cw - sw + 1)
                          if all(not blocked[r + dr, c + dc] for dr, dc in shape)]
            if not candidates:
                break
            # Edge locations get occasional extra weight, but are never required.
            edge = [(r, c) for r, c in candidates if r == 0 or c == 0 or r + sh == ch or c + sw == cw]
            choices = edge if edge and rng.random() < 0.2 else candidates
            r, c = choices[int(rng.integers(len(choices)))]
            for dr, dc in shape:
                rr, cc = r + dr, c + dc
                canvas[rr, cc] = color
                for ar, ac in ((rr, cc), (rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= ar < ch and 0 <= ac < cw:
                        blocked[ar, ac] = True
        else:
            if np.count_nonzero(canvas == background) > canvas.size // 2:
                return canvas
    return None


def _generate_pair(rng, dimensions, rotation, palette, dual_chance, easy_chance):
    h, w = dimensions
    for _ in range(160):
        dual = bool(rng.random() < dual_chance)
        layout = _make_layout(rng, h, w, dual, rng.random() < easy_chance)
        if layout is None:
            continue
        colored = _color_layout(rng, h, w, *layout, palette, dual)
        if colored is None:
            continue
        board, output, pieces = colored
        try:
            answers, _ = _assembly(board, int(palette[0]), pieces)
        except _SearchLimit:
            continue
        if len(answers) != 1 or not np.array_equal(answers[0], output):
            continue
        canvas = _scatter(rng, board, pieces, int(palette[0]))
        if canvas is None:
            continue
        canvas, output = np.rot90(canvas, rotation).copy(), np.rot90(output, rotation).copy()
        try:
            predicted = solve([], canvas)
        except (AssertionError, _SearchLimit):
            continue
        if np.array_equal(predicted, output):
            return {"input": canvas, "output": output}
    # Bounded constructive fallback: two separated, differently sized bars in
    # a bilateral scaffold. Colors, dimensions, scattering and orientation vary.
    board = np.full((h, w), int(palette[1]), dtype=np.int8)
    pieces = []
    output = board.copy()
    for row, length, color in ((1, w - 2, int(palette[4])), (h - 2, w, int(palette[5]))):
        left = (w - length) // 2
        board[row, left:left + length] = int(palette[0])
        output[row, left:left + length] = color
        pieces.append((color, tuple((0, c) for c in range(length))))
    canvas = _scatter(rng, board, pieces, int(palette[0]))
    _require(canvas is not None, "Could not scatter fallback inventory")
    pair = {"input": np.rot90(canvas, rotation).copy(), "output": np.rot90(output, rotation).copy()}
    _require(np.array_equal(solve([], pair["input"]), pair["output"]), "Fallback must be uniquely solvable")
    return pair


def _analyze_pair(input_grid, output_grid):
    predicted, background, bbox, pieces, exterior, nodes = _predict_key(_array_key(input_grid))
    _require(np.array_equal(predicted, output_grid), "Output differs from unique input-only assembly")
    r, c, h, w = bbox
    board = input_grid[r:r + h, c:c + w]
    fixed = board != background
    _require(np.array_equal(board[fixed], output_grid[fixed]), "Fixed cells changed")
    _require(not np.any(output_grid == background), "Unfilled board cells")
    fixed_components = _mask_components(fixed)
    gaps = _mask_components(~fixed)
    counts = {}
    shapes_by_color = {}
    type_counts = {}
    for color, shape in pieces:
        counts[color] = counts.get(color, 0) + len(shape)
        shapes_by_color.setdefault(color, set()).add(shape)
        type_counts[color, shape] = type_counts.get((color, shape), 0) + 1
    _require(all(int(np.count_nonzero(output_grid[~fixed] == color)) == count
                 for color, count in counts.items()), "Inventory color counts changed")
    lr = np.array_equal(output_grid, output_grid[:, ::-1])
    tb = np.array_equal(output_grid, output_grid[::-1, :])
    _require(lr or tb, "Board lost bilateral organization")
    multicolor_fixed = len(set(map(int, board[fixed]))) > 1
    fixed_bands = multicolor_fixed and (
        all(len(set(map(int, row[row != background]))) <= 1 for row in board)
        or all(len(set(map(int, col[col != background]))) <= 1 for col in board.T))
    multicolor_gaps = sum(len({int(output_grid[cell]) for cell in comp}) > 1 for comp in gaps)
    def on_edge(cell, shape):
        rr, cc = cell
        return rr in (0, shape[0] - 1) or cc in (0, shape[1] - 1)
    return {
        "background": background,
        "bbox": bbox,
        "board_shape": (h, w),
        "canvas_shape": tuple(input_grid.shape),
        "fixed_colors": tuple(sorted(set(map(int, board[fixed])))),
        "payload_colors": tuple(sorted(counts)),
        "fixed_color_count": len(set(map(int, board[fixed]))),
        "payload_color_count": len(counts),
        "fixed_bands": fixed_bands,
        "fixed_blocks_or_mixed": multicolor_fixed and not fixed_bands,
        "fixed_components": len(fixed_components),
        "loose_piece_larger_than_fixed_component": max(map(len, exterior)) > max(map(len, fixed_components)),
        "piece_count": len(pieces),
        "piece_shapes": tuple(sorted({shape for _, shape in pieces})),
        "piece_sizes": tuple(sorted({len(shape) for _, shape in pieces})),
        "repeated_piece": any(n > 1 for n in type_counts.values()),
        "same_color_several_shapes": any(len(s) > 1 for s in shapes_by_color.values()),
        "multicolor_gaps": multicolor_gaps,
        "gap_count": len(gaps),
        "boundary_pieces": any(on_edge(tuple(cell), board.shape) for cell in np.argwhere(~fixed)),
        "edge_clues": any(on_edge(cell, input_grid.shape) for comp in exterior for cell in comp),
        "board_at_edge": r == 0 or c == 0 or r + h == input_grid.shape[0] or c + w == input_grid.shape[1],
        "symmetry": "both" if lr and tb else "lr" if lr else "tb",
        "solver_nodes": nodes,
        "layout_key": (board.shape, tuple(map(bool, fixed.ravel()))),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "Splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "Invalid pair counts")
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "Grid must be a 2D array")
                _require(np.issubdtype(grid.dtype, np.integer), "Grid must have integer dtype")
                _require(all(1 <= n <= 30 for n in grid.shape), "Grid exceeds ARC bounds")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "Invalid ARC color")
            ik, ok = _array_key(pair["input"]), _array_key(pair["output"])
            _require(ik not in inputs and (ik, ok) not in pairs, "Duplicate example")
            _require(ik != ok, "An assembly input must be larger than its output")
            _require(not _is_source_duplicate(pair["input"], pair["output"]), "Copied a source pair")
            inputs.add(ik)
            pairs.add((ik, ok))
            _analyze_pair(pair["input"], pair["output"])
    # Every demonstration instantiates the identical input-only assembly rule;
    # no learned lookup table, pair index, or test-only piece operation is used.
    training = [{k: v.copy() for k, v in p.items()} for p in puzzle["train"]]
    for pair in puzzle["test"]:
        prediction = solve(training, pair["input"].copy())
        _require(np.array_equal(prediction, pair["output"]), "Test is not recoverable from visible evidence")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, ne = _resolve_counts(seed, rng, num_train, num_test)
    # These puzzle-level sampling styles carry no hidden output convention.
    # All puzzles use preservation, translation, inventory conservation, crop.
    palette = rng.permutation(10)
    exchange_roles = bool(rng.random() < 0.7)
    orientation_style = int(rng.choice([0, 1, 2, 2]))
    dual_chance = float(rng.uniform(0.1, 0.3))
    size_center = int(rng.integers(6, 10))
    puzzle = {"train": [], "test": []}
    train_settings = []
    seen = set()
    for split, number in (("train", nt), ("test", ne)):
        for index in range(number):
            prng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            dims = tuple(int(x) for x in np.clip(prng.integers(size_center - 3, size_center + 4, size=2), 4, 12))
            rotation = orientation_style if orientation_style < 2 else int(prng.integers(2))
            if split == "train" and index == 1:
                if dims == train_settings[0][0]:
                    dims = (4 if dims[0] == 12 else dims[0] + 1, dims[1])
                if orientation_style == 2:
                    rotation = 1 - train_settings[0][1]
            if split == "test":
                regime = float(prng.random())
                known_dims, known_rotation = train_settings[int(prng.integers(nt))]
                if regime < 0.4:
                    dims, rotation = known_dims, known_rotation
                elif regime < 0.65:
                    dims = tuple(min(13, max(d[i] for d, _ in train_settings) + int(prng.integers(1, 3))) for i in range(2))
                elif regime < 0.8:
                    dims, rotation = known_dims, 1 - known_rotation
            else:
                train_settings.append((dims, rotation))
            pair_palette = prng.permutation(10) if exchange_roles else palette
            for _ in range(8):
                pair = _generate_pair(prng, dims, rotation, pair_palette, dual_chance,
                                      0.35 if split == "train" else 0.15)
                key = _array_key(pair["input"])
                if key not in seen and not _is_source_duplicate(pair["input"], pair["output"]):
                    break
            else:
                raise AssertionError("Could not produce a distinct assembly")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[k]) == len(second[k]) and
               all(np.array_equal(a[g], b[g]) for a, b in zip(first[k], second[k])
                   for g in ("input", "output")) for k in ("train", "test"))


def self_test() -> None:
    # Source inference is performed before comparing its result with the label.
    for input_rows, output_rows in _SOURCE_PAIR_ROWS:
        inp = np.array([[int(x) for x in row] for row in input_rows], dtype=np.int8)
        answer = solve([], inp)
        expected = np.array([[int(x) for x in row] for row in output_rows], dtype=np.int8)
        _require(np.array_equal(answer, expected), "Source assembly prediction failed")
    train_counts, test_counts, shapes, canvases, layouts, signatures = set(), set(), set(), set(), set(), set()
    backgrounds, fixed_counts, payload_counts, piece_counts, symmetries = set(), set(), set(), set(), set()
    measurements = []
    within = {k: 0 for k in ("board_shape", "piece_count", "fixed_color_count", "payload_color_count", "symmetry")}
    familiar = larger = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair["input"], pair["output"])
                measurements.append((split, a))
                local.append(a)
                shapes.add(a["board_shape"])
                canvases.add(a["canvas_shape"])
                layouts.add(a["layout_key"])
                signatures.update(a["piece_shapes"])
                backgrounds.add(a["background"])
                fixed_counts.add(len(a["fixed_colors"]))
                payload_counts.add(len(a["payload_colors"]))
                piece_counts.add(a["piece_count"])
                symmetries.add(a["symmetry"])
        for key in within:
            within[key] += len({a[key] for a in local}) > 1
        train_sizes = [p["output"].shape for p in puzzle["train"]]
        familiar += any(p["output"].shape in train_sizes for p in puzzle["test"])
        larger += any(p["output"].size > max(q["output"].size for q in puzzle["train"]) for p in puzzle["test"])
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "Counts do not vary")
    _require(len(shapes) >= 35 and len(canvases) >= 80 and len(layouts) >= 200, "Insufficient geometric diversity")
    _require({(5, 5), (8, 5)} <= shapes and any(w % 2 == 0 for h, w in shapes),
             "Missing source dimensions or even widths")
    _require(len(signatures) >= 60 and len(piece_counts) >= 9, "Insufficient piece diversity")
    _require(tuple((r, c) for r in range(2) for c in range(3)) in signatures,
             "Missing source 2 by 3 rectangle")
    _require(tuple((0, c) for c in range(4)) in signatures, "Missing four-cell bar")
    _require(((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)) in signatures, "Missing five-cell T")
    _require(any(len(s) >= 12 for s in signatures), "Missing thicker, longer pieces")
    _require(backgrounds == set(range(10)) and len(fixed_counts) >= 3 and len(payload_counts) >= 4,
             "Insufficient color-role/count diversity")
    _require(symmetries == {"lr", "tb", "both"}, "Missing symmetry class")
    _require(all(n >= 20 for k, n in within.items() if k != "symmetry") and within["symmetry"] >= 15,
             "Insufficient within-puzzle variation")
    _require(familiar >= 10 and larger >= 8, "Train/test regimes collapsed")
    for split in ("train", "test"):
        rows = [a for s, a in measurements if s == split]
        for flag in ("repeated_piece", "same_color_several_shapes", "boundary_pieces", "edge_clues"):
            _require(sum(bool(a[flag]) for a in rows) >= len(rows) // 5, "Missing review coverage: " + flag)
        _require(sum(a["fixed_components"] > 1 for a in rows) >= len(rows) // 4, "Too few disconnected scaffolds")
        _require(sum(a["multicolor_gaps"] > 0 for a in rows) >= len(rows) // 3, "Too few shared gaps")
        _require(sum(a["symmetry"] == "both" for a in rows) >= 5, "Double symmetry missing from split")
    _require(any(a["loose_piece_larger_than_fixed_component"] for _, a in measurements), "Largest-piece shortcut not broken")
    _require(sum(a["fixed_bands"] for _, a in measurements) >= 15 and
             sum(a["fixed_blocks_or_mixed"] for _, a in measurements) >= 15,
             "Fixed colors lack coherent bands and non-band arrangements")
    _require(any(a["board_at_edge"] for _, a in measurements), "Board placement requires a background border")
    _require(sum(not a["boundary_pieces"] for _, a in measurements) >= 12,
             "All layouts require boundary gaps")
    _require(any(a["fixed_components"] > 1 and a["multicolor_gaps"] and a["same_color_several_shapes"]
                 and a["symmetry"] == "tb" for _, a in measurements), "Review factors never combine")
    # Distinct colored assignments are ambiguous, but exchanges of identical
    # pieces are a single answer; a reflected T is not a legal translation.
    two_slots = np.array([[0, 8, 0]], dtype=np.int8)
    _require(len(_assembly(two_slots, 0, [(1, ((0, 0),)), (2, ((0, 0),))])[0]) == 2,
             "Did not detect a color-swap ambiguity")
    _require(len(_assembly(two_slots, 0, [(1, ((0, 0),)), (1, ((0, 0),))])[0]) == 1,
             "Identical-piece exchanges incorrectly count as ambiguity")
    t_slot = np.array([[0, 0, 0], [8, 0, 8]], dtype=np.int8)
    _require(not _assembly(t_slot, 0, [(1, ((0, 1), (1, 0), (1, 1), (1, 2)))])[0],
             "Solver illegally reflected a supplied piece")
    first = generate_puzzle(314159, 5, 2)
    _require(_puzzles_equal(first, generate_puzzle(314159, 5, 2)), "Nondeterministic generation")
    _require(not _puzzles_equal(first, generate_puzzle(314160, 5, 2)), "Seeds do not change puzzles")
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(200 + (a or 0) * 5 + (b or 0), a, b)
        validate_generated_puzzle(puzzle)
        _require(a is None or len(puzzle["train"]) == a, "Ignored train count")
        _require(b is None or len(puzzle["test"]) == b, "Ignored test count")
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted illegal pair counts")
    # Corrupt fixed cells and remove an inventory cell: both must be rejected.
    for kind in ("fixed", "missing"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        pair = bad["test"][0]
        _, bg, bbox, _, exterior, _ = _predict_key(_array_key(pair["input"]))
        if kind == "missing":
            pair["input"][exterior[0][0]] = bg
        else:
            r, c, h, w = bbox
            cell = tuple(np.argwhere(pair["input"][r:r + h, c:c + w] != bg)[0])
            pair["output"][cell] = bg
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Validator accepted an inventory/fixed-cell error")
