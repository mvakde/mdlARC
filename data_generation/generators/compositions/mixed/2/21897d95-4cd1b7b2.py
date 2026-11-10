"""T-carried clues on an outlined order-four Latin board.

Only the standard library and NumPy are used. Construction and Latin solving
are separate: solve reads the visible board, glyphs and payloads afresh.
No files, parent modules, seeds or generation records are read by the solver.
The 4x4 board is logical: its framed territories are expanded into readable
pixel rectangles. T ink is erased into the final fill, never redrawn on top
of the Latin answer. See task_spec.md for the full input language.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
import hashlib
import json

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
BLANK = 0


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), "illegal ARC color")
    return grid.astype(np.int8, copy=True)


def _completions(clues, symbols, limit=2):
    """MRV search with early exit; there are only 576 order-four Latin squares."""
    symbols = set(map(int, symbols))
    work = np.asarray(clues, dtype=np.int8).copy()
    rows, columns = [set() for _ in range(4)], [set() for _ in range(4)]
    solutions = []
    for r, c in np.ndindex((4, 4)):
        v = int(work[r, c])
        if v == BLANK:
            continue
        if v not in symbols or v in rows[r] or v in columns[c]:
            return []
        rows[r].add(v)
        columns[c].add(v)

    def visit():
        if len(solutions) >= limit:
            return
        best, options = None, None
        for r, c in np.ndindex((4, 4)):
            if work[r, c] != BLANK:
                continue
            available = symbols - rows[r] - columns[c]
            if not available:
                return
            if options is None or len(available) < len(options):
                best, options = (r, c), available
                if len(options) == 1:
                    break
        if best is None:
            solutions.append(work.copy())
            return
        r, c = best
        for v in sorted(options):
            work[r, c] = v
            rows[r].add(v)
            columns[c].add(v)
            visit()
            rows[r].remove(v)
            columns[c].remove(v)
            work[r, c] = BLANK
            if len(solutions) >= limit:
                return

    visit()
    return solutions


def _intervals(lines, length):
    _require(len(lines) == 5 and lines[0] == 0 and lines[-1] == length - 1,
             "expected five complete frame lines")
    intervals = [(int(a) + 1, int(b)) for a, b in zip(lines, lines[1:])]
    _require(all(4 <= b - a <= 7 for a, b in intervals), "bad territory width")
    return intervals


def _stencils(height, width, direction):
    """All T junctions one pixel inward from the indicated territory edge."""
    dr, dc = direction
    if dr:
        centers = [(1 if dr < 0 else height - 2, c) for c in range(1, width - 1)]
    else:
        centers = [(r, 1 if dc < 0 else width - 2) for r in range(1, height - 1)]
    for r, c in centers:
        arms = ((r - dc, c + dr), (r + dc, c - dr))
        tip = (r + dr, c + dc)
        yield (r, c), arms, tip


def _read(input_grid):
    grid = _grid(input_grid)
    line = int(grid[0, 0])
    _require(line != BLANK, "frame cannot use the blank color")
    row_lines = np.flatnonzero(np.all(grid == line, axis=1)).tolist()
    col_lines = np.flatnonzero(np.all(grid == line, axis=0)).tolist()
    row_spans = _intervals(row_lines, grid.shape[0])
    col_spans = _intervals(col_lines, grid.shape[1])
    frame = np.zeros(grid.shape, dtype=bool)
    frame[row_lines, :] = True
    frame[:, col_lines] = True
    _require(np.array_equal(grid == line, frame), "frame color used in a territory")
    base = np.zeros((4, 4), dtype=np.int8)
    residual = Counter()
    patches = {}
    for r, c in np.ndindex((4, 4)):
        a, b = row_spans[r]
        x, y = col_spans[c]
        patch = grid[a:b, x:y]
        counts = Counter(map(int, patch.flat))
        fill, count = counts.most_common(1)[0]
        _require(count * 2 > patch.size, "territory lacks a strict majority fill")
        base[r, c] = fill
        patches[r, c] = patch
        residual.update(int(v) for v in patch.flat if v != fill)
    _require(bool(residual), "missing T annotations")
    # Each arrow contributes at least three ink pixels and at most one token.
    # Thus ink is the unique most frequent non-majority color in the board.
    ranked = residual.most_common()
    ink, count = ranked[0]
    _require(ink not in (BLANK, line) and count >= 3
             and (len(ranked) == 1 or count > ranked[1][1]), "ambiguous T ink")
    _require(ink not in base, "T ink used as a territory fill")
    arrows = []
    for r, c in np.ndindex((4, 4)):
        patch = patches[r, c]
        fill = int(base[r, c])
        if np.all(patch == fill):
            continue
        candidates = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 4 and 0 <= nc < 4):
                continue
            for center, arms, tip in _stencils(*patch.shape, (dr, dc)):
                if any(patch[p] != ink for p in (*arms, tip)):
                    continue
                junction = int(patch[center])
                explicit = junction != ink
                payload = junction if explicit else fill
                if payload in (BLANK, line, ink):
                    continue
                write = (center, *arms, tip)
                expected = np.full(patch.shape, fill, dtype=np.int8)
                for p in write:
                    expected[p] = ink
                expected[center] = junction
                if np.array_equal(patch, expected):
                    top, left = row_spans[r][0], col_spans[c][0]
                    candidates.append(dict(source=(r, c), target=(nr, nc),
                                           payload=payload, explicit=explicit,
                                           center=(top + center[0], left + center[1]),
                                           write=tuple((top + u, left + v) for u, v in write)))
        _require(len(candidates) == 1, "unrecognized or ambiguous T territory")
        arrows.append(candidates[0])
    symbols = set(map(int, base.flat)) - {BLANK}
    symbols.update(a["payload"] for a in arrows)
    _require(len(symbols) == 4 and not symbols & {BLANK, line, ink},
             "input must expose exactly four symbol colors")
    assignments = {}
    for arrow in arrows:
        target, payload = arrow["target"], arrow["payload"]
        _require(target not in assignments or assignments[target] == payload,
                 "conflicting incoming transfers")
        assignments[target] = payload
    clues = base.copy()
    for target, payload in assignments.items():
        clues[target] = payload
    return dict(grid=grid, line=line, ink=ink, rows=row_spans, cols=col_spans,
                base=base, symbols=symbols, arrows=arrows,
                assignments=assignments, clues=clues)


def _render(values, rows, cols, line):
    result = np.full((rows[-1][1] + 1, cols[-1][1] + 1), line, dtype=np.int8)
    for r, c in np.ndindex((4, 4)):
        a, b = rows[r]
        x, y = cols[c]
        result[a:b, x:y] = values[r, c]
    return result


def _analyze(input_grid):
    info = _read(input_grid)
    solutions = _completions(info["clues"], info["symbols"])
    _require(len(solutions) == 1, "transferred clues must have one Latin completion")
    info["answer"] = solutions[0]
    output = _render(solutions[0], info["rows"], info["cols"], info["line"])
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode simultaneous transfers, complete the Latin board, erase annotations.

    Blank is always 0. The frame and T ink are recovered from this input.
    Conflicting transfers and zero/multiple Latin completions raise ValueError.
    The caller's input is never modified.
    """
    return _analyze(input_grid)[0]


def _causal_witness(clues, targets, symbols, answer):
    """Without carried clues, at least one still-blank cell can change."""
    weakened = clues.copy()
    for target in targets:
        weakened[target] = BLANK
    holes = clues == BLANK
    return any(np.any(other[holes] != answer[holes])
               for other in _completions(weakened, symbols, limit=576))


def _simultaneous_witness(info):
    return any(not a["explicit"] and a["source"] in info["assignments"]
               and info["assignments"][a["source"]] != info["base"][a["source"]]
               for a in info["arrows"])


def _rectangle_mask(mask):
    return (int(mask.sum()) == 4
            and sorted(map(int, mask.sum(axis=0))) == [0, 0, 2, 2]
            and sorted(map(int, mask.sum(axis=1))) == [0, 0, 2, 2])


def _latin_square(rng, symbols):
    if int(rng.integers(2)):
        base = np.array([[r ^ c for c in range(4)] for r in range(4)], dtype=np.int8)
    else:
        base = np.array([[(r + c) % 4 for c in range(4)] for r in range(4)], dtype=np.int8)
    return np.asarray(symbols, dtype=np.int8)[rng.permutation(4)[
        base[np.ix_(rng.permutation(4), rng.permutation(4))]]]


def _cycle(rng, length):
    if length == 0:
        return []
    if length == 2:
        edges = [(p, (p[0] + dr, p[1] + dc)) for p in np.ndindex((4, 4))
                 for dr, dc in ((0, 1), (1, 0))
                 if p[0] + dr < 4 and p[1] + dc < 4]
        return list(edges[int(rng.integers(len(edges)))])
    r, c = map(int, rng.integers(0, 3, size=2))
    nodes = [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)]
    if rng.random() < .5:
        nodes.reverse()
    return nodes


def _transfer_plan(rng, square, cycle_length, extra):
    nodes = _cycle(rng, cycle_length)
    arrows, targets, sources = [], set(nodes), set(nodes)
    old_values = {}
    for i, source in enumerate(nodes):
        target = nodes[(i + 1) % len(nodes)]
        old_values[source] = int(square[target])
        arrows.append(dict(source=source, target=target, explicit=False))
    edges = [(p, (p[0] + dr, p[1] + dc)) for p in np.ndindex((4, 4))
             for dr, dc in DIRECTIONS
             if 0 <= p[0] + dr < 4 and 0 <= p[1] + dc < 4]
    rng.shuffle(edges)
    for source, target in edges:
        if extra == 0:
            break
        if source in sources or target in targets:
            continue
        arrows.append(dict(source=source, target=target, explicit=True))
        targets.add(target)
        sources.add(source)
        # Always expose at least one genuinely wrong given, rather than only
        # arrows that redundantly insert values into otherwise solvable holes.
        palette = ([BLANK] if old_values else []) + [
            int(v) for v in set(square.flat) if v != square[target]]
        old_values[target] = int(rng.choice(sorted(palette)))
        extra -= 1
    return None if extra else (arrows, targets, old_values)


def _mask(rng, square, targets, low, high, intersection):
    symbols = set(map(int, square.flat))

    def admissible(mask):
        clues = square.copy()
        clues[mask] = BLANK
        return (set(map(int, clues[~mask])) == symbols
                and len(_completions(clues, symbols)) == 1)

    if intersection:
        masks = []
        for rows in combinations(range(4), 2):
            for cols in combinations(range(4), 2):
                mask = np.zeros((4, 4), dtype=bool)
                mask[np.ix_(rows, cols)] = True
                if not any(mask[t] for t in targets) and admissible(mask):
                    masks.append(mask)
        rng.shuffle(masks)
    else:
        masks = []
        desired = int(rng.integers(low, high + 1))
        positions = [p for p in np.ndindex((4, 4)) if p not in targets]
        for _ in range(6):
            mask = np.zeros((4, 4), dtype=bool)
            rng.shuffle(positions)
            for p in positions:
                mask[p] = True
                if not admissible(mask):
                    mask[p] = False
                if int(mask.sum()) == desired:
                    break
            if low <= int(mask.sum()) <= high:
                masks.append(mask)
    for mask in masks:
        if max(mask.sum(axis=0)) < 2 or max(mask.sum(axis=1)) < 2:
            continue
        clues = square.copy()
        clues[mask] = BLANK
        if _causal_witness(clues, targets, symbols, square):
            return mask
    return None


def _spans(rng):
    widths = np.full(4, 4, dtype=int)
    for _ in range(int(rng.integers(10))):
        choices = np.flatnonzero(widths < 7)
        widths[int(rng.choice(choices))] += 1
    spans, start = [], 1
    for width in widths:
        spans.append((start, start + int(width)))
        start += int(width) + 1
    return spans


def _make_pair(rng, line, ink, symbols, split, witness):
    for _ in range(240):
        square = _latin_square(rng, symbols)
        intersection = witness == "intersection"
        if intersection and all(len(set(map(int, square[r:r + 2, c:c + 2].flat))) == 4
                                for r, c in ((0, 0), (0, 2), (2, 0), (2, 2))):
            continue
        if witness == "simultaneous":
            length, extra = 2, int(rng.integers(1, 3))
        elif intersection:
            length, extra = 0, int(rng.integers(2, 5))
        else:
            length = int(rng.choice((0, 2, 4)))
            extra = int(rng.integers(1 if length == 0 else 0, 4))
        planned = _transfer_plan(rng, square, length, extra)
        if planned is None:
            continue
        arrows, targets, old_values = planned
        low, high = (7, 9) if split == "test" else (4, 6)
        mask = _mask(rng, square, targets, low, high, intersection)
        if mask is None:
            continue
        clues = square.copy()
        clues[mask] = BLANK
        base = clues.copy()
        for target, value in old_values.items():
            base[target] = value
        rows, cols = _spans(rng), _spans(rng)
        inp = _render(base, rows, cols, line)
        for arrow in arrows:
            r, c = arrow["source"]
            nr, nc = arrow["target"]
            dr, dc = nr - r, nc - c
            top, bottom = rows[r]
            left, right = cols[c]
            options = list(_stencils(bottom - top, right - left, (dr, dc)))
            center, arms, tip = options[int(rng.integers(len(options)))]
            for u, v in (center, *arms, tip):
                inp[top + u, left + v] = ink
            if arrow["explicit"]:
                inp[top + center[0], left + center[1]] = square[nr, nc]
        output = _render(square, rows, cols, line)
        # Decode the rendered input, not the construction metadata.
        predicted, info = _analyze(inp)
        if not np.array_equal(predicted, output):
            continue
        if witness == "simultaneous" and not _simultaneous_witness(info):
            continue
        if witness == "simultaneous" and not any(
                a["explicit"] and a["payload"] != base[a["source"]] for a in info["arrows"]):
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("bounded pair construction exhausted")


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(np.arange(1, 10))
    line, ink = map(int, palette[:2])
    fixed_symbols = palette[2:6]
    variable_symbols = bool(rng.integers(2))
    available = [v for v in range(1, 10) if v not in (line, ink)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = ("simultaneous" if index == 0 else "intersection" if index == 1 else "")
            if split == "test":
                witness = ""
            for _ in range(8):
                symbols = rng.choice(available, 4, replace=False) if variable_symbols else fixed_symbols
                pair = _make_pair(rng, line, ink, symbols, split, witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded duplicate rejection exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    return puzzle


def _enumerated_unique(clues, symbols):
    """Validator's row-permutation check, independent of MRV search order."""
    candidates = []
    for r in range(4):
        candidates.append([p for p in permutations(sorted(symbols))
                           if all(clues[r, c] in (BLANK, p[c]) for c in range(4))])
    count, first = 0, None

    def append(rows):
        nonlocal count, first
        if count >= 2:
            return
        r = len(rows)
        if r == 4:
            count += 1
            first = np.array(rows, dtype=np.int8) if first is None else first
            return
        for row in candidates[r]:
            if all(all(previous[c] != row[c] for previous in rows) for c in range(4)):
                append(rows + [row])

    append([])
    return count, first


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, training = set(), []
    train_holes, test_holes = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            predicted, info = _analyze(inp)
            _require(np.array_equal(predicted, output), "wrong answer or annotation cleanup")
            answer, clues = info["answer"], info["clues"]
            holes = clues == BLANK
            for axis in (answer, answer.T):
                _require(all(set(map(int, row)) == info["symbols"] for row in axis), "non-Latin output")
            fixed = ~holes
            _require(np.array_equal(answer[fixed], clues[fixed]), "transferred clue changed")
            count, alternate = _enumerated_unique(clues, info["symbols"])
            _require(count == 1 and np.array_equal(alternate, answer), "nonunique Latin clues")
            _require(max(holes.sum(axis=0)) >= 2 and max(holes.sum(axis=1)) >= 2,
                     "one-axis alternatives lack a witness")
            _require(any(info["base"][t] != answer[t] and info["base"][t] != BLANK
                         for t in info["assignments"]), "omitting T does not contradict a clue")
            _require(_causal_witness(clues, info["assignments"], info["symbols"], answer),
                     "transfers do not determine any remaining blank")
            _require(info["ink"] not in output and BLANK not in output, "annotations or blanks survived")
            # A row-only completion is obtained by swapping two missing cells.
            # It keeps all clues and row permutations, but breaks a column.
            r = int(np.flatnonzero(holes.sum(axis=1) >= 2)[0])
            c, d = map(int, np.flatnonzero(holes[r])[:2])
            row_only = answer.copy()
            row_only[r, c], row_only[r, d] = answer[r, d], answer[r, c]
            _require(all(set(map(int, row)) == info["symbols"] for row in row_only)
                     and not np.array_equal(row_only, answer)
                     and any(len(set(map(int, col))) != 4 for col in row_only.T),
                     "bad row-only counterexample")
            c = int(np.flatnonzero(holes.sum(axis=0) >= 2)[0])
            r, s = map(int, np.flatnonzero(holes[:, c])[:2])
            column_only = answer.copy()
            column_only[r, c], column_only[s, c] = answer[s, c], answer[r, c]
            _require(all(set(map(int, col)) == info["symbols"] for col in column_only.T)
                     and any(len(set(map(int, row))) != 4 for row in column_only),
                     "bad column-only counterexample")
            if split == "train":
                training.append(info)
                train_holes.append(int(holes.sum()))
                _require(4 <= int(holes.sum()) <= 6, "training mask outside curriculum")
            else:
                test_holes.append(int(holes.sum()))
                _require(7 <= int(holes.sum()) <= 9, "test mask outside curriculum")
    _require(max(train_holes) < min(test_holes), "missing sparsity shift")
    _require(any(_simultaneous_witness(i) for i in training), "no simultaneous training witness")
    _require(any(a["explicit"] and a["payload"] != i["base"][a["source"]]
                 for i in training for a in i["arrows"]), "no explicit-junction contrast")
    _require(any(_rectangle_mask(i["clues"] == BLANK) for i in training),
             "no joint-intersection training case")
    _require(any(len(set(map(int, i["answer"][r:r + 2, c:c + 2].flat))) < 4
                 for i in training for r, c in ((0, 0), (0, 2), (2, 0), (2, 2))),
             "training fails to distinguish Latin constraints from Sudoku boxes")


def self_test():
    """42 sequential seeds; retain small signatures and aggregate counters only."""
    shapes, masks, latin_forms, palettes, arrow_counts, directions = (set() for _ in range(6))
    signatures, simultaneous, explicit_only, long_cycles, nonuniform = set(), 0, 0, 0, 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                output, i = _analyze(pair["input"])
                _require(np.array_equal(output, pair["output"]), "input-only solve failed")
                pairs += 1
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
                shapes.add(pair["input"].shape)
                masks.add((i["clues"] == BLANK).tobytes())
                mapping = {int(v): c for c, v in enumerate(i["answer"][0])}
                latin_forms.add(tuple(mapping[int(v)] for v in i["answer"].flat))
                palettes.add((i["line"], i["ink"], tuple(sorted(i["symbols"]))))
                arrow_counts.add(len(i["arrows"]))
                for arrow in i["arrows"]:
                    directions.add(tuple(b - a for a, b in zip(arrow["source"], arrow["target"])))
                simultaneous += _simultaneous_witness(i)
                implicit = sum(not a["explicit"] for a in i["arrows"])
                explicit_only += implicit == 0
                long_cycles += implicit == 4
                nonuniform += len({b - a for a, b in i["rows"] + i["cols"]}) > 1
        signatures.add(digest.hexdigest())
        if seed == 0:
            replica = generate_puzzle(seed)
            _require(all(_key(p[k]) == _key(q[k]) for s in ("train", "test")
                         for p, q in zip(puzzle[s], replica[s]) for k in ("input", "output")),
                     "nondeterministic puzzle")
            # Changing annotation ink preserves the answer when its new color
            # is reserved from the visible symbol/frame palette.
            sample = puzzle["train"][0]
            _, info = _analyze(sample["input"])
            replacement = next(c for c in range(1, 10)
                               if c not in info["symbols"] | {info["line"], info["ink"]})
            recolored = sample["input"].copy()
            recolored[recolored == info["ink"]] = replacement
            _require(np.array_equal(solve(recolored), sample["output"]), "ink identity leaked into rule")
            # Spatial symmetries move the actual arrows and frame together.
            # This checks new valid family members, not just sampled originals.
            for split in ("train", "test"):
                for sample in puzzle[split]:
                    before = sample["input"].copy()
                    for turn in range(4):
                        inp = np.rot90(before, turn)
                        out = np.rot90(sample["output"], turn)
                        _require(np.array_equal(solve(inp), out), "rotation changed the rule")
                        _require(np.array_equal(solve(np.fliplr(inp)), np.fliplr(out)),
                                 "reflection changed the rule")
                    _require(np.array_equal(sample["input"], before), "solver mutated input")
            # Invalid boards must not silently select the first completion.
            sample = puzzle["train"][0]
            _, info = _analyze(sample["input"])
            few_clues = np.zeros((4, 4), dtype=np.int8)
            few_clues[0] = info["answer"][0]
            ambiguous = _render(few_clues, info["rows"], info["cols"], info["line"])
            # Two explicit arrows can point to (0,0). The first is harmless;
            # adding the second with another payload is a transfer conflict.
            incoming_conflict = None
            for index, (source, direction) in enumerate((((1, 0), (-1, 0)),
                                                       ((0, 1), (0, -1)))):
                top, bottom = info["rows"][source[0]]
                left, right = info["cols"][source[1]]
                center, arms, tip = next(_stencils(bottom - top, right - left, direction))
                canvas = ambiguous if index == 0 else ambiguous.copy()
                for u, v in (center, *arms, tip):
                    canvas[top + u, left + v] = info["ink"]
                canvas[top + center[0], left + center[1]] = info["answer"][0, index]
                if index == 1:
                    incoming_conflict = canvas
                    agreeing = canvas.copy()
                    agreeing[top + center[0], left + center[1]] = info["answer"][0, 0]
                    agreeing_info = _read(agreeing)
                    _require(len(agreeing_info["arrows"]) == 2
                             and len(agreeing_info["assignments"]) == 1,
                             "consistent incoming transfers were rejected")
            conflicting = sample["input"].copy()
            # Choose an untargeted, unannotated cell and duplicate another
            # fixed clue from its row. This preserves the annotation language
            # but makes the post-transfer Latin constraints inconsistent.
            changed = False
            sources = {a["source"] for a in info["arrows"]}
            for r, c in np.ndindex((4, 4)):
                if (r, c) in sources or (r, c) in info["assignments"]:
                    continue
                candidates = [int(info["clues"][r, d]) for d in range(4)
                              if d != c and info["clues"][r, d] != BLANK]
                if candidates:
                    top, bottom = info["rows"][r]
                    left, right = info["cols"][c]
                    conflicting[top:bottom, left:right] = candidates[0]
                    changed = True
                    break
            _require(changed, "missing inconsistent-clue fixture")
            malformed_frame = sample["input"].copy()
            malformed_frame[0, 1] = BLANK
            _require(len(_completions(_read(ambiguous)["clues"], info["symbols"])) == 2,
                     "ambiguous fixture did not reach the Latin stage")
            for bad in (ambiguous, incoming_conflict, conflicting, malformed_frame):
                try:
                    solve(bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid board was accepted")
            _require(len(_completions(np.zeros((4, 4), dtype=np.int8),
                                     info["symbols"])) == 2,
                     "Latin uniqueness check failed to detect ambiguity")
    _require(len(signatures) == 42 and len(shapes) >= 30 and len(masks) >= 100,
             "collapsed structural variation")
    _require(len(latin_forms) >= 10 and len(palettes) >= 35, "collapsed Latin or palette variation")
    _require({2, 3, 4, 5} <= arrow_counts and directions == set(DIRECTIONS), "missing transfer variation")
    _require(simultaneous >= 42 and explicit_only >= 42 and long_cycles >= 5 and nonuniform >= 150,
             "missing graph or territory-size variation")
    # All thirty legal explicit count combinations, and malformed count types.
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(9000 + train * 10 + test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (None, 1), (4, None), (4, "1")):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                shapes=len(shapes), masks=len(masks), latin_forms=len(latin_forms),
                palettes=len(palettes), arrow_counts=sorted(arrow_counts),
                directions=sorted(directions), simultaneous_pairs=simultaneous,
                explicit_only_pairs=explicit_only, four_cycle_pairs=long_cycles,
                nonuniform_pairs=nonuniform)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
