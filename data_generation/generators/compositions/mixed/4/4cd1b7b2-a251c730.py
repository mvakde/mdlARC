"""Complete a four-symbol Latin board, then render its anchor dictionary.

Only standard-library modules and NumPy are used. No files are read or written
by generation, solving, validation, or the bounded self-test.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations
import hashlib

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside allowed range")
    return value


@lru_cache(maxsize=1)
def _latin_squares():
    """All 576 order-four Latin squares: mathematical candidates, not examples."""
    rows = list(permutations(range(4)))
    answers = []

    def visit(prefix):
        if len(prefix) == 4:
            answers.append(prefix)
            return
        for row in rows:
            if all(all(row[c] != earlier[c] for c in range(4)) for earlier in prefix):
                visit(prefix + [row])

    visit([])
    result = np.asarray(answers, dtype=np.int8)
    result.flags.writeable = False
    return result


def _complete(clues):
    """Clues use -1 for a missing identity, otherwise indices 0..3."""
    visible = clues >= 0
    candidates = _latin_squares()
    matches = candidates[np.all(candidates[:, visible] == clues[visible], axis=1)]
    if len(matches) != 1:
        raise ValueError("Latin clues must have exactly one row-and-column completion")
    return matches[0].copy()


def _unique_mask(square, mask):
    visible = ~mask
    if len(set(map(int, square[visible]))) != 4:
        return False
    return np.count_nonzero(np.all(_latin_squares()[:, visible] == square[visible], axis=1)) == 1


def _mask(rng, square, kind, target):
    if kind == "rectangle":
        options = []
        for rows in combinations(range(4), 2):
            for cols in combinations(range(4), 2):
                mask = np.zeros((4, 4), dtype=bool)
                mask[np.ix_(rows, cols)] = True
                if len(set(map(int, square[mask]))) == 4 and _unique_mask(square, mask):
                    options.append(mask)
        return _pick(rng, options).copy()
    best = None
    # At most 32 * 16 small candidate-filter operations; no retained retries.
    for _ in range(32):
        mask = np.zeros((4, 4), dtype=bool)
        if kind == "line":
            axis, index = int(rng.integers(2)), int(rng.integers(4))
            if axis == 0:
                mask[index, :] = True
            else:
                mask[:, index] = True
        for flat in rng.permutation(16):
            if int(mask.sum()) >= target:
                break
            r, c = divmod(int(flat), 4)
            mask[r, c] = True
            if not _unique_mask(square, mask):
                mask[r, c] = False
        if best is None or mask.sum() > best.sum():
            best = mask.copy()
        if int(mask.sum()) == target:
            return mask
    if int(best.sum()) >= 9 or target <= 8:
        return best
    # Exhaustive bounded recovery, only if greedy deletion missed nine holes.
    # Choosing seven givens has 11,440 possibilities. This is not a new rule.
    for kept in combinations(range(16), 7):
        mask = np.ones((4, 4), dtype=bool)
        mask.flat[list(kept)] = False
        if _unique_mask(square, mask):
            return mask
    raise RuntimeError("No nine-hole mask found")


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _D4:
                point = r + dr, c + dc
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(cells)
    return result


def _array(value):
    grid = np.asarray(value)
    if grid.ndim != 2 or not all(1 <= x <= 30 for x in grid.shape):
        raise ValueError("Grid dimensions must be 1..30")
    if not np.issubdtype(grid.dtype, np.integer) or np.any((grid < 0) | (grid > 9)):
        raise ValueError("Grid cells must be integers 0..9")
    return grid


def _crop(grid, box):
    r, c, h, w = box
    return grid[r:r+h, c:c+w]


def _frame_panels(grid):
    """The dedicated shared frame color has two separate components.

    A target component contains its five-by-five lattice lines. A reference
    component is exactly a rectangular outline. No output is consulted.
    """
    boards, references = [], []
    for color in map(int, np.unique(grid)):
        for cells in _components(grid == color):
            rr, cc = zip(*cells)
            box = min(rr), min(cc), max(rr)-min(rr)+1, max(cc)-min(cc)+1
            panel = _crop(grid, box)
            h, w = panel.shape
            if min(h, w) < 5:
                continue
            border = np.zeros((h, w), dtype=bool)
            border[[0, -1], :] = True
            border[:, [0, -1]] = True
            actual = panel == color
            if not np.all(actual[border]):
                continue
            if np.array_equal(actual, border):
                references.append((box, color))
            if (h-1) % 4 or (w-1) % 4:
                continue
            ch, cw = (h-1)//4-1, (w-1)//4-1
            if ch not in (3, 5) or cw not in (3, 5) or (ch, cw) == (5, 5):
                continue
            lattice = np.zeros((h, w), dtype=bool)
            lattice[::ch+1, :] = True
            lattice[:, ::cw+1] = True
            if np.array_equal(actual, lattice):
                boards.append((box, color, ch, cw))
    if len(boards) != 1:
        raise ValueError("Expected one framed four-by-four target lattice")
    board = boards[0]
    refs = [entry for entry in references if entry[1] == board[1]]
    if len(refs) != 1:
        raise ValueError("Expected one reference frame in the shared frame color")
    return board, refs[0]


def _decode(input_grid):
    grid = _array(input_grid)
    (target_box, frame, ch, cw), (reference_box, _) = _frame_panels(grid)
    tr, tc, th, tw = target_box
    rr, rc, rh, rw = reference_box
    if not (tr+th <= rr or rr+rh <= tr or tc+tw <= rc or rc+rw <= tc):
        raise ValueError("Panels must be disjoint")
    target, reference = _crop(grid, target_box), _crop(grid, reference_box)
    fill = int(target[1, 1])
    centers = [(1+r*(ch+1)+ch//2, 1+c*(cw+1)+cw//2)
               for r in range(4) for c in range(4)]
    literal = np.asarray([target[p] for p in centers], dtype=int).reshape(4, 4)
    keys = sorted(set(map(int, literal.flat)) - {fill})
    if len(keys) != 4 or not np.any(literal == fill):
        raise ValueError("Show four anchor identities and at least one blank")
    expected_input = np.full(target.shape, fill, dtype=target.dtype)
    expected_input[::ch+1, :] = frame
    expected_input[:, ::cw+1] = frame
    for point, value in zip(centers, literal.flat):
        expected_input[point] = value
    if not np.array_equal(target, expected_input):
        raise ValueError("Target cells must contain only central givens")
    mapping = {key: i for i, key in enumerate(keys)}
    clues = np.asarray([mapping.get(int(v), -1) for v in literal.flat], dtype=int).reshape(4, 4)
    completed = _complete(clues)
    rfill = int(reference[1, 1])
    inner = reference[1:-1, 1:-1]
    if not (np.all(inner[0] == rfill) and np.all(inner[-1] == rfill)
            and np.all(inner[:, 0] == rfill) and np.all(inner[:, -1] == rfill)):
        raise ValueError("Reference requires a clear fill rim")
    if fill in keys or rfill in keys or frame in keys or frame in (fill, rfill):
        raise ValueError("Invisible or ambiguous color roles")
    templates, repetitions = {}, {}
    for component in _components(inner != rfill):
        anchors = [(r, c) for r, c in component if int(inner[r, c]) in keys]
        if len(anchors) != 1 or len(component) < 3:
            raise ValueError("Each reference motif needs one identifiable anchor")
        ar, ac = anchors[0]
        key = int(inner[ar, ac])
        signature = tuple(sorted((r-ar, c-ac, int(inner[r, c])) for r, c in component))
        if any(not (-(ch//2) <= dr <= ch//2 and -(cw//2) <= dc <= cw//2)
               for dr, dc, _ in signature):
            raise ValueError("Template exceeds its centered target cell")
        if any(color in (fill, rfill, frame) or (color in keys and (dr, dc) != (0, 0))
               for dr, dc, color in signature):
            raise ValueError("Decoration roles must remain visible and distinct from anchors")
        if key in templates and templates[key] != signature:
            raise ValueError("Conflicting reference definitions")
        templates[key] = signature
        repetitions[key] = repetitions.get(key, 0) + 1
    if set(templates) != set(keys):
        raise ValueError("Every Latin identity needs a reference template")
    deco = [tuple((r, c, v) for r, c, v in templates[k] if (r, c) != (0, 0)) for k in keys]
    if len(set(deco)) != 4:
        raise ValueError("Four meaningfully different decoration templates required")
    # Exterior is a uniform, irrelevant fill, and never joins a frame.
    exterior = np.ones(grid.shape, dtype=bool)
    for r, c, h, w in (target_box, reference_box):
        exterior[r:r+h, c:c+w] = False
    if not exterior.any() or len(np.unique(grid[exterior])) != 1 or np.any(grid[exterior] == frame):
        raise ValueError("Expected a uniform non-frame exterior")
    return {"grid": grid, "target": target, "target_box": target_box,
            "reference_box": reference_box, "frame": frame, "fill": fill,
            "reference_fill": rfill, "cell_shape": (ch, cw), "centers": centers,
            "keys": keys, "clues": clues, "completed": completed,
            "templates": templates, "repetitions": repetitions}


def _render(info, *, complete=True, decorate=True):
    result = info["target"].copy()
    values = info["completed"] if complete else info["clues"]
    for (r, c), identity in zip(info["centers"], values.flat):
        if identity < 0:
            continue
        key = info["keys"][int(identity)]
        if decorate:
            for dr, dc, color in info["templates"][key]:
                result[r+dr, c+dc] = color
        else:
            result[r, c] = key
    return result


def _axis_counterexample(info, axis):
    """Keep one axis Latin and all clues, but violate the other axis.

    Swapping two missing entries of one completed line exhibits ambiguity of
    that axis alone, without relying on a particular greedy solver's choice.
    """
    mask = info["clues"] < 0
    for index in range(4):
        positions = np.flatnonzero(mask[index] if axis == 0 else mask[:, index])
        if len(positions) >= 2:
            a, b = map(int, positions[:2])
            alternative = info["completed"].copy()
            if axis == 0:
                alternative[index, a], alternative[index, b] = alternative[index, b], alternative[index, a]
            else:
                alternative[a, index], alternative[b, index] = alternative[b, index], alternative[a, index]
            changed = dict(info, completed=alternative)
            assert not np.array_equal(_render(changed), _render(info))
            return True
    return False


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only Latin completion followed by transparent keyed stamping."""
    return _render(_decode(input_grid))


def _shape(rng, ch, cw, require_witness=False):
    allowed = {(r, c) for r in range(-(ch//2), ch//2+1)
               for c in range(-(cw//2), cw//2+1)}
    for _ in range(64):
        cells = {(0, 0)}
        size = int(rng.integers(3, min(11, ch*cw)+1))
        while len(cells) < size:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4}
                              & allowed - cells)
            cells.add(_pick(rng, frontier))
        if not require_witness or _witness_shape(cells):
            return tuple(sorted(cells))
    # A connected nonrectangular asymmetric L, in every supported cell shape.
    return ((0, 0), (1, 0), (1, 1))


def _witness_shape(cells):
    cells = set(cells)
    rr, cc = zip(*cells)
    area = (max(rr)-min(rr)+1)*(max(cc)-min(cc)+1)
    if area <= len(cells):
        return False
    for reflect in (False, True):
        for turns in range(4):
            if not reflect and turns == 0:
                continue
            transformed = set()
            for r, c in cells:
                if reflect:
                    c = -c
                for _ in range(turns):
                    r, c = c, -r
                transformed.add((r, c))
            if cells == transformed:
                return False
    return True


def _make_templates(rng, keys, decorations, ch, cw):
    result, signatures = {}, set()
    for i, key in enumerate(keys):
        for _ in range(128):
            shape = _shape(rng, ch, cw, require_witness=(i == 0))
            offsets = [p for p in shape if p != (0, 0)]
            colors = [int(_pick(rng, decorations)) for _ in offsets]
            signature = tuple(sorted((r, c, v) for (r, c), v in zip(offsets, colors)))
            if signature not in signatures:
                signatures.add(signature)
                result[int(key)] = tuple(sorted(((0, 0, int(key)),) + signature))
                break
        else:
            raise RuntimeError("Could not draw distinct templates")
    return result


def _geometry(rng, count, ch, cw):
    target_shape = (4*(ch+1)+1, 4*(cw+1)+1)
    options = []
    for rows in range(1, count+1):
        cols = (count+rows-1)//rows
        for spacing in (1, 2):
            ref_shape = (rows*ch+(rows-1)*spacing+4, cols*cw+(cols-1)*spacing+4)
            for axis in (0, 1):
                for gap in (1, 2, 3):
                    dims = [max(a, b) for a, b in zip(target_shape, ref_shape)]
                    dims[axis] = target_shape[axis]+ref_shape[axis]+gap
                    if max(dims) <= 30:
                        options.append((rows, cols, spacing, ref_shape, axis, gap, dims))
    rows, cols, spacing, rshape, axis, gap, required = _pick(rng, options)
    shape = [int(rng.integers(n, 31)) for n in required]
    first = int(rng.integers(2))  # 0: reference first, 1: target first
    sizes = [rshape, target_shape]
    positions = [[0, 0], [0, 0]]
    start = int(rng.integers(shape[axis]-required[axis]+1))
    positions[first][axis] = start
    positions[1-first][axis] = start+sizes[first][axis]+gap
    other = 1-axis
    for i in (0, 1):
        positions[i][other] = int(rng.integers(shape[other]-sizes[i][other]+1))
    boxes = [tuple(positions[i])+tuple(sizes[i]) for i in (0, 1)]
    slots = [(2+r*(ch+spacing)+ch//2, 2+c*(cw+spacing)+cw//2)
             for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    return tuple(shape), boxes, slots[:count]


def _pair(rng, palette, kind, holes, first=False):
    ch, cw = _pick(rng, ((3, 3), (3, 5), (5, 3)))
    keys = sorted(map(int, palette[:4]))
    decorations = list(map(int, palette[4:4+int(rng.integers(1, 3))]))
    fill, rfill, frame = map(int, palette[6:9])
    if not first and rng.random() < .35:
        rfill = fill
    templates = _make_templates(rng, keys, decorations, ch, cw)
    if first:
        ids = keys*2
    else:
        reps = [int(rng.integers(1, 4)) for _ in keys]
        limit = 10 if (ch, cw) == (3, 3) else 8
        while sum(reps) > limit:
            choices = [i for i, n in enumerate(reps) if n > 1]
            reps[_pick(rng, choices)] -= 1
        ids = [key for key, n in zip(keys, reps) for _ in range(n)]
    shape, (rbox, tbox), slots = _geometry(rng, len(ids), ch, cw)
    rng.shuffle(ids)
    grid = np.full(shape, _pick(rng, (fill, rfill)), dtype=np.int8)
    for box, background in ((rbox, rfill), (tbox, fill)):
        panel = _crop(grid, box)
        panel[:] = frame
        panel[1:-1, 1:-1] = background
    ref, target = _crop(grid, rbox), _crop(grid, tbox)
    for key, (r, c) in zip(ids, slots):
        for dr, dc, color in templates[key]:
            ref[r+dr, c+dc] = color
    target[::ch+1, :] = frame
    target[:, ::cw+1] = frame
    square = _latin_squares()[int(rng.integers(576))]
    mask = _mask(rng, square, kind, holes)
    out = target.copy()
    for r in range(4):
        for c in range(4):
            ar, ac = 1+r*(ch+1)+ch//2, 1+c*(cw+1)+cw//2
            key = keys[int(square[r, c])]
            if not mask[r, c]:
                target[ar, ac] = key
            # Ground truth is constructed from the sampled complete square;
            # it is subsequently checked by input-only reconstruction.
            for dr, dc, color in templates[key]:
                out[ar+dr, ac+dc] = color
    return {"input": grid, "output": out}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shared_palette = rng.permutation(10)
    local_palette = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for _ in range(16):
                palette = rng.permutation(10) if local_palette else shared_palette
                if split == "train" and i == 0:
                    kind, holes = "rectangle", 4
                elif split == "train":
                    kind = _pick(rng, ("random", "random", "line"))
                    holes = int(rng.integers(4, 9))
                else:
                    kind, holes = "random", int(rng.integers(9, 12))
                pair = _pair(rng, palette, kind, holes, first=(split == "train" and i == 0))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible semantics, preservation, ablations, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, counts, teaching = set(), {"train": [], "test": []}, []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _decode(inp)
            assert np.array_equal(_render(info), out), "incorrect completed stamp board"
            assert out.shape == info["target"].shape
            completed, clues = info["completed"], info["clues"]
            mask = clues < 0
            counts[split].append(int(mask.sum()))
            assert np.array_equal(completed[~mask], clues[~mask]), "changed Latin givens"
            assert all(set(map(int, line)) == set(range(4)) for line in list(completed)+list(completed.T))
            ch, cw = info["cell_shape"]
            lattice = np.zeros(out.shape, dtype=bool)
            lattice[::ch+1, :] = True
            lattice[:, ::cw+1] = True
            assert np.array_equal(out[lattice], info["target"][lattice]), "changed frame or grid lines"
            occupied = np.zeros(out.shape, dtype=bool)
            for (r, c), v in zip(info["centers"], completed.flat):
                for dr, dc, _ in info["templates"][info["keys"][int(v)]]:
                    assert not occupied[r+dr, c+dc], "overlapping stamps"
                    assert not lattice[r+dr, c+dc], "stamp touches frame"
                    occupied[r+dr, c+dc] = True
            assert np.all(out[~occupied & ~lattice] == info["fill"]), "fill was not transparent"
            # Omitting A leaves blank-cell motifs absent; omitting B leaves bare symbols.
            assert not np.array_equal(_render(info, complete=False), out)
            assert not np.array_equal(_render(info, decorate=False), out)
            # Correct counts alone do not specify template positions: swapping any
            # two dictionary decorations would change at least one missing cell.
            assert len(set(tuple((r, c, v) for r, c, v in sig if (r, c) != (0, 0))
                           for sig in info["templates"].values())) == 4
            if split == "train":
                joint = _axis_counterexample(info, 0) and _axis_counterexample(info, 1)
                transparency = (info["fill"] != info["reference_fill"] and
                                any(_witness_shape((r, c) for r, c, _ in sig)
                                    for sig in info["templates"].values()))
                teaching.append((joint, transparency, min(info["repetitions"].values()) >= 2))
    assert any(all(t) for t in teaching), "training must expose joint constraints and transparent keyed copying"
    assert min(counts["test"]) > max(counts["train"]), "expected held-out clue sparsity"
    assert all(4 <= n <= 8 for n in counts["train"])
    assert all(9 <= n <= 11 for n in counts["test"])


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """42 seeds, streamed; only small signatures and counters are retained."""
    shapes, cells, masks, geometries, palettes, signatures, holes, reps, layouts = (set() for _ in range(9))
    multicolor = same_fill = missing_lines = row_ambiguous = col_ambiguous = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                mask = info["clues"] < 0
                holes.add(int(mask.sum()))
                masks.add(mask.tobytes())
                cells.add(info["cell_shape"])
                shapes.add((pair["input"].shape, pair["output"].shape))
                palettes.add((tuple(info["keys"]), info["fill"], info["reference_fill"], info["frame"]))
                reps.update(info["repetitions"].values())
                same_fill += info["fill"] == info["reference_fill"]
                missing_lines += bool(np.any(mask.sum(axis=0) == 4) or np.any(mask.sum(axis=1) == 4))
                row_ambiguous += bool(np.any(mask.sum(axis=1) >= 2))
                col_ambiguous += bool(np.any(mask.sum(axis=0) >= 2))
                for sig in info["templates"].values():
                    geometries.add(tuple((r, c) for r, c, _ in sig))
                    multicolor += len({v for r, c, v in sig if (r, c) != (0, 0)}) > 1
                r, c, h, w = info["reference_box"]
                y, x, k, l = info["target_box"]
                layouts.add("vertical" if r+h <= y or y+k <= r else "horizontal")
        # Whole-scene transformations move dictionary and target together.
        # These are extra solver checks, not added production sampling modes.
        pair = puzzle["test"][0]
        for k in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
        assert np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"]))
        color_map = np.random.default_rng(seed+4000).permutation(10)
        assert np.array_equal(solve(color_map[pair["input"]]), color_map[pair["output"]])
    assert len(signatures) == 42
    assert cells == {(3, 3), (3, 5), (5, 3)} and layouts == {"vertical", "horizontal"}
    assert len(shapes) >= 25 and len(masks) >= 100 and len(geometries) >= 100
    assert len(palettes) >= 40 and reps >= {1, 2, 3}
    assert holes >= {4, 5, 6, 7, 8, 9, 10} and multicolor >= 20 and same_fill >= 10
    assert missing_lines >= 5 and row_ambiguous >= 42 and col_ambiguous >= 42
    first = generate_puzzle(123456, 7, 3)
    assert _digest(first) == _digest(generate_puzzle(123456, 7, 3))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generated = generate_puzzle(1000+a*10+b, a, b)
            assert len(generated["train"]) == a and len(generated["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4.0, 1), (4, False), (4, None), (None, 1), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Negative semantics: corrupt labels, ambiguous Latin givens, and a
    # conflicting copy in a repeated dictionary must each be rejected.
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]]
              for s in ("train", "test")}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt frame label accepted")
    try:
        _complete(np.full((4, 4), -1, dtype=int))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous Latin clues accepted")
    original = generate_puzzle(0)["train"][0]["input"]
    info = _decode(original)
    bad = original.copy()
    ref = _crop(bad, info["reference_box"])
    key = info["keys"][0]
    occurrences = np.argwhere(ref == key)
    ar, ac = map(int, occurrences[0])
    dr, dc, _ = next(cell for cell in info["templates"][key] if cell[:2] != (0, 0))
    ref[ar+dr, ac+dc] = info["reference_fill"]
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting or broken reference accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shape_pairs": len(shapes),
            "cell_shapes": sorted(cells), "masks": len(masks), "motif_geometries": len(geometries),
            "palette_profiles": len(palettes), "blank_counts": sorted(holes),
            "reference_repetitions": sorted(reps), "multicolor_templates": multicolor,
            "equal_fill_pairs": same_fill, "missing_line_pairs": missing_lines}
