"""Colored glyph programs with a learnable, puzzle-specific stroke vocabulary."""

import numpy as np


_SOURCE_MASKS = (
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8),
    np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=np.int8),
    np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int8),
)
# (step down, step across, number of pixels). The first pixel is always
# directly below the previous endpoint, including for diagonal commands.
_SOURCE_STROKES = ((0, 1, 3), (0, -1, 2), (1, 0, 2), (0, -1, 4))
_SOURCE_COLORS = (1, 2, 6, 3)
_CATALOG = tuple((dr, dc, length)
                 for dr, dc in ((0, 1), (0, -1), (1, 0), (1, 1), (1, -1))
                 for length in range(2, 6 if dr == 0 else 5))
_SOURCE_PROGRAMS = (
    (4, 1, (1, 0, 0, 2, 2, 3, 2, 0)),
    (2, 3, (0, 2, 1, 3)),
    (3, 4, (1, 2, 1, 2, 0, 2)),
    (5, 2, (0, 1, 2, 3, 2, 0, 0, 2, 3, 1)),
)


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(rng, num_train, num_test):
    nt = None if num_train is None else _integer("num_train", num_train, 2, 10)
    nq = None if num_test is None else _integer("num_test", num_test, 1, 4)
    if nq is None:
        nq = int(rng.integers(1, min(4, nt - 1) + 1 if nt is not None else 5))
    if nt is None:
        nt = int(rng.integers(nq + 1, 11))
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    return nt, nq


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _mask_key(mask):
    return mask.shape[0], mask.shape[1], tuple(int(v) for v in mask.flat)


def _connected(mask):
    """Eight-connected ink, as in the source's diagonal glyph contacts."""
    cells = set(zip(*np.nonzero(mask)))
    reached = {min(cells)}
    pending = list(reached)
    while pending:
        r, c = pending.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (r + dy, c + dx)
                if q in cells and q not in reached:
                    reached.add(q)
                    pending.append(q)
    return reached == cells


def _vocabulary(rng, low_variation):
    if low_variation:
        return _SOURCE_MASKS, _SOURCE_STROKES
    gh, gw = (int(v) for v in rng.integers(2, 5, size=2))
    count = int(rng.integers(3, 6))
    masks = []

    def consider(bits):
        mask = np.array([(bits >> k) & 1 for k in range(gh * gw)],
                        dtype=np.int8).reshape(gh, gw)
        if (2 <= mask.sum() < gh * gw and np.all(mask.any(axis=0))
                and np.all(mask.any(axis=1)) and _connected(mask)
                and all(not np.array_equal(mask, m) for m in masks)):
            masks.append(mask)

    for _ in range(256):
        consider(int(rng.integers(1, 1 << (gh * gw))))
        if len(masks) == count:
            break
    if len(masks) < count:
        # Finite deterministic completion, including the six valid 2x2 masks.
        for bits in range(1, 1 << (gh * gw)):
            consider(bits)
            if len(masks) == count:
                break
    assert len(masks) == count
    strokes = [(0, 1, int(rng.integers(2, 6))),
               (0, -1, int(rng.integers(2, 6)))]
    third_direction = ((1, 0), (1, 1), (1, -1))[int(rng.integers(3))]
    strokes.append((*third_direction, int(rng.integers(2, 5))))
    remaining = [s for s in _CATALOG if s not in strokes]
    for index in rng.permutation(len(remaining))[:count - 3]:
        strokes.append(remaining[int(index)])
    return tuple(masks), tuple(strokes[int(i)] for i in rng.permutation(count))


def _reachability(strokes, width, distinct_neighbors=False):
    """Exact geometry feasibility, shared across candidate lengths and starts."""
    costs = [1 + dr * (n - 1) for dr, dc, n in strokes]
    shifts = [dc * (n - 1) for dr, dc, n in strokes]
    memo = {}

    def possible(left, col, room, missing, previous=-1):
        if left == 0:
            return missing == 0
        if left < missing.bit_count():
            return False
        extra = sum(costs[k] - 1 for k in range(len(strokes)) if missing & (1 << k))
        if room < left + extra:
            return False
        room = min(room, left * max(costs))
        if not distinct_neighbors:
            previous = -1
        state = left, col, room, missing, previous
        if state not in memo:
            memo[state] = any(
                (not distinct_neighbors or k != previous)
                and 0 <= col + shifts[k] < width and costs[k] <= room
                and possible(left - 1, col + shifts[k], room - costs[k],
                             missing & ~(1 << k), k)
                for k in range(len(strokes)))
        return memo[state]

    return possible


def _program(rng, strokes, length, width, budget, require_all, distinct_neighbors,
             forbidden):
    """Randomized finite search with feasibility pruning, never rejection sampling."""
    count = len(strokes)
    initial_missing = (1 << count) - 1 if require_all else 0
    costs = [1 + dr * (n - 1) for dr, dc, n in strokes]
    shifts = [dc * (n - 1) for dr, dc, n in strokes]
    order = [int(v) for v in rng.permutation(count)]
    failed = set()
    possible = _reachability(strokes, width, distinct_neighbors)

    def visit(left, col, room, missing, previous, prefix, start):
        if not possible(left, col, room, missing, previous):
            return None
        if left == 0:
            return prefix if not missing and (prefix, start) not in forbidden else None
        if left < missing.bit_count():
            return None
        extra = sum(costs[k] - 1 for k in range(count) if missing & (1 << k))
        if room < left + extra:
            return None
        # A forbidden complete program depends on its prefix; retain it in the
        # memo key rather than incorrectly pruning a different prefix.
        state = (left, col, room, missing, previous, prefix, start)
        if state in failed:
            return None
        choices = list(order)
        # Shuffle priorities along the path; all branches remain available.
        offset = int(rng.integers(count))
        choices = choices[offset:] + choices[:offset]
        if missing and left <= missing.bit_count() + 1:
            choices.sort(key=lambda k: not bool(missing & (1 << k)))
        for k in choices:
            if distinct_neighbors and k == previous:
                continue
            end = col + shifts[k]
            if not 0 <= end < width or costs[k] > room:
                continue
            answer = visit(left - 1, end, room - costs[k], missing & ~(1 << k),
                           k, prefix + (k,), start)
            if answer is not None:
                return answer
        failed.add(state)
        return None

    for start in rng.permutation(width):
        start = int(start)
        answer = visit(length, start, budget, initial_missing, -1, (), start)
        if answer is not None:
            forbidden.add((answer, start))
            return answer, start
    raise AssertionError("no feasible program in the chosen layout")


def _draw(height, width, background, marker, start, strokes, program, colors):
    output = np.full((height, width), background, dtype=np.int8)
    output[0, start] = marker
    r, c = 0, start
    for k, color in zip(program, colors):
        dr, dc, size = strokes[k]
        rr = r + 1 + dr * np.arange(size)
        cc = c + dc * np.arange(size)
        assert rr.max() < height and cc.min() >= 0 and cc.max() < width
        assert np.all(output[rr, cc] == background)
        output[rr, cc] = color
        r, c = int(rr[-1]), int(cc[-1])
    return output


def _compose(masks, strokes, program, colors, rows, lanes, ygap, xgap,
             height, width, roles, start):
    gh, gw = masks[0].shape
    background, separator, marker = roles
    panel_width = lanes * gw + (lanes - 1) * xgap
    inp = np.full((height, panel_width + 1 + width), background, dtype=np.int8)
    inp[:, panel_width] = separator
    inp[0, panel_width + 1 + start] = marker
    for index, (command, color) in enumerate(zip(program, colors)):
        lane, band = divmod(index, rows)
        top, left = band * (gh + ygap), lane * (gw + xgap)
        block = inp[top:top + gh, left:left + gw]
        block[masks[command].astype(bool)] = color
    out = _draw(height, width, background, marker, start, strokes, program, colors)
    return {"input": inp, "output": out}


def _layouts(masks, strokes):
    gh, gw = masks[0].shape
    extra = sum(dr * (n - 1) for dr, dc, n in strokes)
    cap = min(22, 29 - extra)
    minimum_width = {}
    for width in range(7, 19):
        possible = _reachability(strokes, width)
        for length in range(len(masks), cap + 1):
            if length not in minimum_width and any(
                    possible(length, start, 29, (1 << len(masks)) - 1)
                    for start in range(width)):
                minimum_width[length] = width
    choices = []
    for rows in range(2, 7):
        for lanes in range(2, 6):
            if not len(masks) <= rows * lanes <= cap:
                continue
            for ygap in (1, 2):
                for xgap in (1, 2):
                    panel_h = rows * gh + (rows - 1) * ygap
                    panel_w = lanes * gw + (lanes - 1) * xgap
                    minimum = minimum_width.get(rows * lanes, 31)
                    if panel_h <= 30 and panel_w + 1 + minimum <= 30:
                        choices.append((rows, lanes, ygap, xgap, minimum))
    return choices


def _roles(rng, previous):
    # These roles are visible from geometry and therefore need not stay fixed.
    backgrounds = [v for v in range(10) if previous is None or v != previous[0]]
    bg = int(rng.choice(backgrounds))
    separators = [v for v in range(10) if v != bg and
                  (previous is None or v != previous[1])]
    sep = int(rng.choice(separators))
    marker = int(rng.choice([v for v in range(10) if v not in (bg, sep)]))
    return bg, sep, marker


def _occurrence_colors(rng, program, roles, regime, index, first_colors):
    available = [v for v in range(10) if v not in roles]
    if regime == "source_train" and index == 0:
        # A seven-color cycle makes the traversal visible: there are at most
        # six rows and five lanes, so none of the seven other raster traversals
        # produces this same color sequence.
        palette = [int(v) for v in rng.permutation(available)]
        colors = [palette[j % 7] for j in range(len(program))]
    else:
        if regime == "source_train" and index == 1:
            available = [v for v in available if v not in set(first_colors.values())]
            palette = [int(v) for v in rng.choice(available, 2, replace=False)]
        else:
            palette = [int(v) for v in rng.choice(available, int(rng.integers(2, 6)),
                                                 replace=False)]
        colors = []
        for _ in program:
            candidates = [v for v in palette if not colors or
                          regime == "target_test" or v != colors[-1]]
            colors.append(int(rng.choice(candidates)))
    for k, color in zip(program, colors):
        first_colors.setdefault(k, color)
    return colors


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer("seed", seed, 0)
    rng = np.random.default_rng(seed)
    nt, nq = _counts(rng, num_train, num_test)
    low_variation = seed % 14 == 0
    masks, strokes = _vocabulary(rng, low_variation)
    choices = _layouts(masks, strokes) if not low_variation else []
    if not low_variation:
        lengths = sorted({t[0] * t[1] for t in choices})
        cutoffs = [n for n in lengths[1:-1] if n >= 8]
        cutoff = int(rng.choice(cutoffs))
        training_layouts = [t for t in choices if t[0] * t[1] <= cutoff]
        test_layouts = [t for t in choices if t[0] * t[1] > cutoff]
    forbidden = {(p, start) for _, start, p in _SOURCE_PROGRAMS} if low_variation else set()
    result = {"train": [], "test": []}
    previous_roles = None
    first_colors = {}
    first_layout = None
    for split, regime, count in (("train", "source_train", nt),
                                 ("test", "target_test", nq)):
        for index in range(count):
            if low_variation:
                rows = (2 if index == 0 else 4 if index == 1 else
                        int(rng.integers(2, 5))) if split == "train" else 5
                lanes, ygap, xgap, width = 2, 1, 1, 7
                height = 4 * rows - 1
                roles = (0, 4, 5)
                budget = height - 1
            else:
                pool = training_layouts if split == "train" else test_layouts
                if split == "train" and index == 1:
                    varied = [t for t in pool if t[0] * t[1] != first_layout[0] * first_layout[1]
                              and t[1] != first_layout[1]]
                    assert varied
                    pool = varied
                rows, lanes, ygap, xgap, min_width = pool[int(rng.integers(len(pool)))]
                if split == "train" and index == 0:
                    first_layout = (rows, lanes)
                panel_w = lanes * masks[0].shape[1] + (lanes - 1) * xgap
                width = int(rng.integers(min_width, min(18, 30 - panel_w - 1) + 1))
                roles = _roles(rng, previous_roles)
                previous_roles = roles
                budget = 29
            program, start = _program(rng, strokes, rows * lanes, width, budget,
                                      index < 2 or split == "test",
                                      low_variation and split == "train", forbidden)
            if low_variation:
                colors = [_SOURCE_COLORS[k] for k in program]
            else:
                colors = _occurrence_colors(rng, program, roles, regime, index, first_colors)
                path_h = 1 + sum(1 + strokes[k][0] * (strokes[k][2] - 1) for k in program)
                panel_h = rows * masks[0].shape[0] + (rows - 1) * ygap
                minimum = max(path_h, panel_h)
                height = int(rng.integers(minimum, min(30, minimum + 4) + 1))
            result[split].append(_compose(masks, strokes, program, colors, rows, lanes,
                                          ygap, xgap, height, width, roles, start))
    validate_generated_puzzle(result)
    return result


def _assert_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= d <= 30 for d in grid.shape)
    assert np.all((0 <= grid) & (grid <= 9))


def _runs(present):
    positions = [int(v) for v in np.flatnonzero(present)]
    groups = []
    for value in positions:
        if not groups or value != groups[-1][1]:
            groups.append([value, value + 1])
        else:
            groups[-1][1] += 1
    return groups


def _read_input(inp):
    """Recover every geometric/color role from the input, without metadata."""
    candidates = []
    for sep in range(2, inp.shape[1] - 2):
        if not np.all(inp[:, sep] == inp[0, sep]):
            continue
        pane = inp[:, sep + 1:]
        values, counts = np.unique(pane, return_counts=True)
        bg = int(values[np.argmax(counts)])
        marks = np.argwhere(pane != bg)
        if (int(inp[0, sep]) != bg and marks.shape == (1, 2)
                and marks[0, 0] == 0 and pane[tuple(marks[0])] != inp[0, sep]):
            candidates.append((sep, bg, int(marks[0, 1])))
    assert len(candidates) == 1, "ambiguous or absent pane separator"
    sep, bg, start = candidates[0]
    pane = inp[:, sep + 1:]
    marker = int(pane[0, start])
    panel = inp[:, :sep]
    row_runs = _runs(np.any(panel != bg, axis=1))
    col_runs = _runs(np.any(panel != bg, axis=0))
    assert 2 <= len(row_runs) <= 6 and 2 <= len(col_runs) <= 5
    assert row_runs[0][0] == col_runs[0][0] == 0 and col_runs[-1][1] == sep
    gh, gw = row_runs[0][1], col_runs[0][1]
    assert 2 <= gh <= 4 and 2 <= gw <= 4
    assert all(b - a == gh for a, b in row_runs)
    assert all(b - a == gw for a, b in col_runs)
    ygaps = {b[0] - a[1] for a, b in zip(row_runs, row_runs[1:])}
    xgaps = {b[0] - a[1] for a, b in zip(col_runs, col_runs[1:])}
    assert len(ygaps) == len(xgaps) == 1
    assert ygaps <= {1, 2} and xgaps <= {1, 2}
    tokens = []
    for left, right in col_runs:
        for top, bottom in row_runs:
            block = panel[top:bottom, left:right]
            mask = block != bg
            ink = np.unique(block[mask])
            assert len(ink) == 1
            assert int(ink[0]) not in (int(inp[0, sep]), marker)
            assert np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))
            assert 2 <= mask.sum() < gh * gw and _connected(mask)
            tokens.append((_mask_key(mask), int(ink[0])))
    return {"tokens": tokens, "rows": len(row_runs), "lanes": len(col_runs),
            "glyph_size": (gh, gw), "background": bg, "separator": int(inp[0, sep]),
            "marker": marker, "start": start, "pane": pane.copy(),
            "gaps": (next(iter(ygaps)), next(iter(xgaps)))}


def _ordered(info, order):
    major, reverse_lanes, reverse_rows = order
    lanes = list(range(info["lanes"]))[::(-1 if reverse_lanes else 1)]
    rows = list(range(info["rows"]))[::(-1 if reverse_rows else 1)]
    indices = ([(c, r) for c in lanes for r in rows] if major == 0 else
               [(c, r) for r in rows for c in lanes])
    return [info["tokens"][c * info["rows"] + r] for c, r in indices]


def _matches(out, bg, row, col, color, stroke):
    """Inspect whole output rows for a candidate stroke, independent of _draw."""
    dr, dc, size = stroke
    end_row, end_col = row + 1 + dr * (size - 1), col + dc * (size - 1)
    if end_row >= out.shape[0] or not 0 <= end_col < out.shape[1]:
        return False
    if dr == 0:
        expected = np.full(out.shape[1], bg, dtype=np.int8)
        expected[min(col, end_col):max(col, end_col) + 1] = color
        return np.array_equal(out[row + 1], expected)
    for offset in range(size):
        expected = np.full(out.shape[1], bg, dtype=np.int8)
        expected[col + dc * offset] = color
        if not np.array_equal(out[row + 1 + offset], expected):
            return False
    return True


def _learn(train_pairs, order=(0, False, False), limit=2):
    """Infer dictionaries using only training grids; never receive a test label."""
    examples = [(_read_input(p["input"]), p["output"]) for p in train_pairs]
    examples.sort(key=lambda item: len(item[0]["tokens"]))
    solutions = []
    signatures = set()

    def search(pair_index, index, row, col, mapping):
        if len(solutions) >= limit:
            return
        if pair_index == len(examples):
            signature = tuple(sorted(mapping.items()))
            if signature not in signatures:
                signatures.add(signature)
                solutions.append(dict(mapping))
            return
        info, out = examples[pair_index]
        tokens = _ordered(info, order)
        if out.shape != info["pane"].shape or not np.array_equal(out[0], info["pane"][0]):
            return
        if index == len(tokens):
            if not np.all(out[row + 1:] == info["background"]):
                return
            if pair_index + 1 < len(examples):
                next_col = examples[pair_index + 1][0]["start"]
            else:
                next_col = 0
            search(pair_index + 1, 0, 0, next_col, mapping)
            return
        mask, color = tokens[index]
        choices = (mapping[mask],) if mask in mapping else _CATALOG
        for stroke in choices:
            if _matches(out, info["background"], row, col, color, stroke):
                extended = dict(mapping)
                extended[mask] = stroke
                dr, dc, size = stroke
                search(pair_index, index + 1, row + 1 + dr * (size - 1),
                       col + dc * (size - 1), extended)

    search(0, 0, 0, examples[0][0]["start"], {})
    return solutions


def _predict(info, mapping):
    out = info["pane"].copy()
    row, col = 0, info["start"]
    for mask, color in info["tokens"]:
        assert mask in mapping, "test glyph absent from training"
        down, across, size = mapping[mask]
        row += 1
        for step in range(size):
            r, c = row + step * down, col + step * across
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            assert out[r, c] == info["background"]
            out[r, c] = color
        row += (size - 1) * down
        col += (size - 1) * across
    return out


def solve_from_training(train_pairs, test_input):
    maps = _learn(train_pairs)
    assert len(maps) == 1, "training does not uniquely identify the stroke dictionary"
    return _predict(_read_input(test_input), maps[0])


def _source_pairs():
    return [_compose(_SOURCE_MASKS, _SOURCE_STROKES, commands,
                     [_SOURCE_COLORS[k] for k in commands], rows, 2, 1, 1,
                     4 * rows - 1, 7, (0, 4, 5), start)
            for rows, start, commands in _SOURCE_PROGRAMS]


def _low_variation(infos, mapping):
    original = {_mask_key(m): s for m, s in zip(_SOURCE_MASKS, _SOURCE_STROKES)}
    return mapping == original and all(
        i["glyph_size"] == (3, 3) and i["lanes"] == 2 and i["gaps"] == (1, 1)
        and (i["background"], i["separator"], i["marker"]) == (0, 4, 5)
        and i["pane"].shape == (4 * i["rows"] - 1, 7)
        and all(color == _SOURCE_COLORS[list(original).index(mask)]
                for mask, color in i["tokens"])
        for i in infos)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    source = {(_key(p["input"]), _key(p["output"])) for p in _source_pairs()}
    infos = {"train": [], "test": []}
    for split in infos:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                _assert_grid(grid)
            ik, ok = _key(pair["input"]), _key(pair["output"])
            assert ik not in seen_inputs and (ik, ok) not in seen_pairs
            assert (ik, ok) not in source
            assert not np.array_equal(pair["input"], pair["output"])
            seen_inputs.add(ik)
            seen_pairs.add((ik, ok))
            info = _read_input(pair["input"])
            assert pair["output"].shape == info["pane"].shape
            infos[split].append(info)
    maps = _learn(puzzle["train"])
    assert len(maps) == 1, "ambiguous stroke dictionary"
    mapping = maps[0]
    assert 3 <= len(mapping) <= 5 and len(set(mapping.values())) == len(mapping)
    all_infos = infos["train"] + infos["test"]
    assert len({i["glyph_size"] for i in all_infos}) == 1
    for split in infos:
        for pair, info in zip(puzzle[split], infos[split]):
            assert np.array_equal(pair["output"], _predict(info, mapping))
    for major in (0, 1):
        for reverse_lanes in (False, True):
            for reverse_rows in (False, True):
                order = (major, reverse_lanes, reverse_rows)
                if order != (0, False, False):
                    assert not _learn(puzzle["train"], order, 1), "ambiguous traversal"
    lengths = {s: [len(i["tokens"]) for i in infos[s]] for s in infos}
    assert min(lengths["test"]) > max(lengths["train"])
    assert len(set(lengths["train"])) >= 2
    # All test commands are not just seen: each is covered by two demonstrations.
    for i in infos["train"][:2] + infos["test"]:
        assert {m for m, c in i["tokens"]} == set(mapping)
    if _low_variation(all_infos, mapping):
        assert all(i["rows"] in (2, 3, 4) for i in infos["train"])
        assert all(i["rows"] == 5 for i in infos["test"])
    else:
        assert len({i["background"] for i in infos["train"]}) >= 2
        assert len({i["separator"] for i in infos["train"]}) >= 2
        assert len({i["lanes"] for i in infos["train"]}) >= 2
        colors_by_mask = {m: set() for m in mapping}
        masks_by_color = {}
        for i in infos["train"]:
            for m, c in i["tokens"]:
                colors_by_mask[m].add(c)
                masks_by_color.setdefault(c, set()).add(m)
        assert all(len(cs) >= 2 for cs in colors_by_mask.values())
        assert any(len(ms) >= 2 for ms in masks_by_color.values())


def puzzle_features(puzzle):
    """Review measurements reconstructed from the returned grids."""
    infos = [_read_input(p["input"]) for p in puzzle["train"] + puzzle["test"]]
    mapping = _learn(puzzle["train"])[0]
    train = infos[:len(puzzle["train"])]
    colors = {m: {c for i in train for k, c in i["tokens"] if k == m} for m in mapping}
    return {"low_variation": _low_variation(infos, mapping),
            "glyph_size": infos[0]["glyph_size"],
            "vocabulary": tuple(sorted(mapping)), "dictionary": tuple(sorted(mapping.items())),
            "stroke_set": tuple(sorted(mapping.values())),
            "lanes": tuple(sorted({i["lanes"] for i in infos})),
            "rows": tuple(sorted({i["rows"] for i in infos})),
            "gaps": tuple(sorted({i["gaps"] for i in infos})),
            "backgrounds": tuple(sorted({i["background"] for i in train})),
            "separators": tuple(sorted({i["separator"] for i in train})),
            "min_glyph_training_colors": min(len(cs) for cs in colors.values()),
            "train_lengths": tuple(len(i["tokens"]) for i in train),
            "test_lengths": tuple(len(i["tokens"]) for i in infos[len(train):]),
            "pane_widths": tuple(sorted({i["pane"].shape[1] for i in infos})),
            "dimensions": tuple(sorted({p["input"].shape for p in puzzle["train"] + puzzle["test"]}))}


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    # Regression: the four required commands have net displacement +8.
    # A four-command demonstration therefore needs at least nine pane columns.
    unbalanced = ((1, 1, 3), (0, 1, 5), (1, 1, 4), (0, -1, 2))
    assert all(t[4] >= 9 for t in _layouts(_SOURCE_MASKS, unbalanced)
               if t[0] * t[1] == 4)
    a = generate_puzzle(12345, 7, 3)
    assert _puzzle_key(a) == _puzzle_key(generate_puzzle(12345, 7, 3))
    assert _puzzle_key(a) != _puzzle_key(generate_puzzle(12346, 7, 3))
    for index, (nt, nq) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                     (None, 4), (2, None), (10, None))):
        for seed in (1400, 2101 + index):
            p = generate_puzzle(seed, nt, nq)
            assert nt is None or len(p["train"]) == nt
            assert nq is None or len(p["test"]) == nq
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1)):
        try:
            generate_puzzle(17, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = [puzzle_features(p) for p in puzzles]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert sum(f["low_variation"] for f in features) == 3
    expanded = [f for f in features if not f["low_variation"]]
    assert len({f["glyph_size"] for f in expanded}) >= 8
    assert len({f["vocabulary"] for f in expanded}) >= 32
    assert len({f["dictionary"] for f in expanded}) >= 36
    assert len({f["stroke_set"] for f in expanded}) >= 30
    assert {c for f in expanded for c in f["lanes"]} == {2, 3, 4, 5}
    assert len({d for f in features for d in f["dimensions"]}) >= 60
    assert len({w for f in features for w in f["pane_widths"]}) >= 10
    assert {s[:2] for f in expanded for s in f["stroke_set"]} == {
        (0, 1), (0, -1), (1, 0), (1, 1), (1, -1)}
    assert {s[2] for f in expanded for s in f["stroke_set"]} == {2, 3, 4, 5}
    assert len({g for f in expanded for g in f["gaps"]}) == 4
    assert all(f["min_glyph_training_colors"] >= 2 and len(f["lanes"]) >= 2
               and len(f["backgrounds"]) >= 2 and len(f["separators"]) >= 2
               for f in expanded)
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    # Test prediction receives only copied training pairs and a test input.
    for p in puzzles:
        training = [{k: v.copy() for k, v in pair.items()} for pair in p["train"]]
        for pair in p["test"]:
            predicted = solve_from_training(training, pair["input"].copy())
            assert np.array_equal(predicted, pair["output"])
    # Mutation checks: ensure the validator rejects plausible semantic damage.
    for damage in ("output", "glyph"):
        changed = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]]
                   for s in ("train", "test")}
        if damage == "output":
            info = _read_input(changed["test"][0]["input"])
            changed["test"][0]["output"][1, info["start"]] = info["background"]
        else:
            info = _read_input(changed["test"][0]["input"])
            changed["test"][0]["input"][:info["glyph_size"][0],
                                           :info["glyph_size"][1]] = info["background"]
        try:
            validate_generated_puzzle(changed)
        except AssertionError:
            continue
        raise AssertionError("semantic mutation was accepted")
