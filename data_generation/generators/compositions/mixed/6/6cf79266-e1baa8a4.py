"""Highlight physical empty 3x3 windows, then compress identical whole lines.

The solver has no generation state. The sampler constructs a binary noisy field
with repeated full lines, isolated qualifying windows, and protected near misses.
All experiments are in memory; self_test writes no files and retains no grids.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


BLANK = 0
HIGHLIGHT = 1
SELF_TEST_SEEDS = (
    0, 568596411, 574621495, 491597344, 656981596, 555714732,
    892231594, 383164714, 491550181, 288794616, 827608309, 254374623,
    858817384, 934458656, 876777147, 669068627, 433155144, 409789562,
    405784053, 365188120, 200113146, 937041399, 323664898, 516465764,
    488705356, 410275248, 791202177, 99564762, 777001809, 427098296,
    993693491, 859143049, 617869735, 897827614, 786443735, 684265046,
    20505198, 714876221, 641770222, 632106215, 938345022, 265993277,
)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside its permitted range")
    return value


def _grid(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("grid must be a rectangular integer array") from error
    if array.ndim != 2 or not all(1 <= x <= 30 for x in array.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(array.dtype, np.integer) or not np.all((0 <= array) & (array <= 9)):
        raise ValueError("grid cells must be integers in 0..9")
    return array.astype(np.int8, copy=False)


def _windows(array):
    """All qualifying starts on the ORIGINAL array, never on recolored cells."""
    height, width = array.shape
    if min(height, width) < 3:
        return []
    match = np.ones((height - 2, width - 2), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            match &= array[dr:height - 2 + dr, dc:width - 2 + dc] == BLANK
    return [tuple(map(int, point)) for point in np.argwhere(match)]


def _footprint(shape, starts):
    mask = np.zeros(shape, dtype=bool)
    for row, col in starts:
        mask[row:row + 3, col:col + 3] = True
    return mask


def _mark(array):
    marked = array.copy()
    marked[_footprint(array.shape, _windows(array))] = HIGHLIGHT
    return marked


def _line_starts(array, axis):
    changed = (np.any(array[1:] != array[:-1], axis=1) if axis == 0
               else np.any(array[:, 1:] != array[:, :-1], axis=0))
    return np.r_[0, np.flatnonzero(changed) + 1]


def _compress(array):
    return array[np.ix_(_line_starts(array, 0), _line_starts(array, 1))].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fixed roles: blank 0, new highlight 1, other colors copied unchanged."""
    array = _grid(input_grid)
    return _compress(_mark(array))


def _expand(logical, rows, cols):
    return np.repeat(np.repeat(logical, rows, axis=0), cols, axis=1)


def _intervals(runs, total):
    result = []
    for start in range(len(runs)):
        size = 0
        for stop in range(start + 1, len(runs) + 1):
            size += int(runs[stop - 1])
            if size == total:
                result.append((start, stop))
            if size >= total:
                break
    return result


def _runs(rng, length, wide=False):
    """Reserve partitions 1+1+1 and 3, and a 2-cell near-miss band."""
    runs = rng.choice(np.array([1, 2, 3]), size=length, p=[.60, .23, .17])
    triple = int(rng.integers(length - 2))
    reserved = set(range(triple, triple + 3))
    runs[triple:triple + 3] = 1
    available = [i for i in range(length) if i not in reserved]
    first, second = map(int, rng.choice(available, size=2, replace=False))
    runs[first], runs[second] = 3, 2
    reserved.update((first, second))
    if wide:
        available = [i for i in range(length) if i not in reserved]
        index = int(rng.choice(available))
        runs[index] = int(rng.integers(7, 11))
        reserved.add(index)
    # The protected assignments always fit: at length 14, even a wide run
    # would need at most 26 cells when all other runs are thin.
    for index in rng.permutation(length):
        if int(runs.sum()) <= 30:
            break
        index = int(index)
        if index not in reserved:
            runs[index] = 1
    if int(runs.sum()) > 30:
        raise RuntimeError("protected run schedule exceeds 30")
    return runs.astype(int)


def _rectangles(rows, cols, height, width):
    return [(a, b, c, d) for a, b in _intervals(rows, height)
            for c, d in _intervals(cols, width)]


def _stamp(logical, fixed, rect, texture):
    """Set a blank rectangle with a one-logical-band texture perimeter."""
    a, b, c, d = rect
    top, bottom = max(0, a - 1), min(logical.shape[0], b + 1)
    left, right = max(0, c - 1), min(logical.shape[1], d + 1)
    core_fixed = fixed[a:b, c:d]
    if np.any(core_fixed & (logical[a:b, c:d] != BLANK)):
        return False
    ring = np.zeros(logical.shape, dtype=bool)
    ring[top:bottom, left:right] = True
    ring[a:b, c:d] = False
    if np.any(ring & fixed & (logical == BLANK)):
        return False
    logical[ring] = texture
    fixed[ring] = True
    logical[a:b, c:d] = BLANK
    fixed[a:b, c:d] = True
    return True


def _targets(rng, logical, fixed, rows, cols, count, texture, evidence):
    options = _rectangles(rows, cols, 3, 3)
    selected = []
    for target in range(count):
        found = False
        for index in rng.permutation(len(options)):
            rect = options[int(index)]
            a, b, c, d = rect
            if rect in selected:
                continue
            if target == 0 and evidence == "attached":
                # A partly collapsed target, with room for a partial-side
                # blank attachment that does not create another window.
                if not (b - a <= 2 and d - c <= 2 and (b - a) * (d - c) > 1):
                    continue
            if target == 0 and evidence == "multiple":
                if b - a != 3 and d - c != 3:
                    continue
            if _stamp(logical, fixed, rect, texture):
                selected.append(rect)
                found = True
                break
        if not found:
            return None
    return selected


def _near_miss(rng, logical, fixed, rows, cols, texture):
    kinds = [(2, 3), (3, 2)]
    if rng.random() < .5:
        kinds.reverse()
    for height, width in kinds:
        options = _rectangles(rows, cols, height, width)
        for index in rng.permutation(len(options)):
            if _stamp(logical, fixed, options[int(index)], texture):
                return True
    return False


def _attach(rng, logical, fixed, rows, cols, targets):
    options = []
    for a, b, c, d in targets:
        for col in range(c, d):
            if cols[col] < 3:
                options.extend((r, col) for r in (a - 1, b)
                               if 0 <= r < logical.shape[0])
        for row in range(a, b):
            if rows[row] < 3:
                options.extend((row, c2) for c2 in (c - 1, d)
                               if 0 <= c2 < logical.shape[1])
    for index in rng.permutation(len(options)):
        point = options[int(index)]
        if logical[point] != BLANK:
            logical[point] = BLANK
            fixed[point] = True
            return True
    return False


def _recurrence(rng, logical, fixed, texture):
    """A separated full texture vector must survive; this is not global dedup."""
    axes = [0, 1]
    if rng.random() < .5:
        axes.reverse()
    for axis in axes:
        view, protection = (logical, fixed) if axis == 0 else (logical.T, fixed.T)
        free = [i for i in range(len(view))
                if not np.any(protection[i] & (view[i] == BLANK))]
        choices = [(a, b) for a, b in combinations(free, 2) if b - a >= 2]
        if choices:
            a, b = choices[int(rng.integers(len(choices)))]
            view[a] = texture
            view[b] = texture
            protection[a] = True
            protection[b] = True
            return True
    return False


def _repair(rng, logical, fixed, rows, cols, intended, texture):
    """Monotone deletion of accidental blank windows, at most n*m changes."""
    row_map = np.repeat(np.arange(len(rows)), rows)
    col_map = np.repeat(np.arange(len(cols)), cols)
    for _ in range(logical.size + 1):
        physical = _expand(logical, rows, cols)
        extras = [start for start in _windows(physical) if start not in intended]
        if not extras:
            return physical
        row, col = extras[0]
        candidates = [(int(r), int(c)) for r in np.unique(row_map[row:row + 3])
                      for c in np.unique(col_map[col:col + 3])
                      if not fixed[r, c] and logical[r, c] == BLANK]
        if not candidates:
            return None
        logical[candidates[int(rng.integers(len(candidates)))]] = texture
    return None


def _has_recurrence(array):
    return any(np.array_equal(view[a], view[b])
               for view in (array, array.T)
               for a in range(len(view)) for b in range(a + 2, len(view)))


def _attachment(array, mask):
    unmarked = (array == BLANK) & ~mask
    return bool(np.any(mask[1:] & unmarked[:-1]) or np.any(mask[:-1] & unmarked[1:])
                or np.any(mask[:, 1:] & unmarked[:, :-1])
                or np.any(mask[:, :-1] & unmarked[:, 1:]))


def _make_pair(rng, shape, count, texture, evidence="free", wide_axis=None):
    for _ in range(160):
        rows = _runs(rng, shape[0], wide_axis == 0)
        cols = _runs(rng, shape[1], wide_axis == 1)
        probability = float(rng.uniform(.25, .56))
        logical = np.where(rng.random(shape) < probability, BLANK, texture).astype(np.int8)
        fixed = np.zeros(shape, dtype=bool)
        targets = _targets(rng, logical, fixed, rows, cols, count, texture, evidence)
        if targets is None or not _near_miss(rng, logical, fixed, rows, cols, texture):
            continue
        attached = _attach(rng, logical, fixed, rows, cols, targets) if (
            evidence == "attached" or rng.random() < .5) else False
        if evidence == "attached" and not attached:
            continue
        if evidence == "multiple" and not _recurrence(rng, logical, fixed, texture):
            continue
        row_edges, col_edges = np.r_[0, np.cumsum(rows)], np.r_[0, np.cumsum(cols)]
        intended = {(int(row_edges[a]), int(col_edges[c])) for a, b, c, d in targets}
        physical = _repair(rng, logical, fixed, rows, cols, intended, texture)
        if physical is None:
            continue
        # Keep every logical boundary visible. Marking with a new color
        # preserves these boundaries as well as the within-band equalities.
        if len(_line_starts(logical, 0)) != shape[0] or len(_line_starts(logical, 1)) != shape[1]:
            continue
        output = solve(physical)
        mask = _footprint(physical.shape, sorted(intended))
        if evidence == "attached" and not _attachment(physical, mask):
            continue
        if evidence == "multiple" and not _has_recurrence(output):
            continue
        return {"input": physical, "output": output}
    raise RuntimeError("bounded sampler could not construct the requested evidence")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["height", "width", "both", "bands"]))
    textures = list(map(int, rng.permutation(np.arange(2, 10))))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, size in (("train", num_train), ("test", num_test)):
        for index in range(size):
            if split == "train" and index == 0:
                shape = (int(rng.integers(6, 9)), int(rng.integers(6, 9)))
                count, evidence, texture = 1, "attached", textures[0]
            elif split == "train" and index == 1:
                shape = (int(rng.integers(8, 11)), int(rng.integers(8, 11)))
                if shape == puzzle["train"][0]["output"].shape:
                    shape = (9, shape[1])
                count, evidence, texture = 2, "multiple", textures[1]
            else:
                count = int(rng.integers(1, 5))
                low = 9 if count >= 3 else 7
                shape = (int(rng.integers(low, 11)), int(rng.integers(low, 11)))
                evidence = "free"
                texture = int(rng.choice(textures))
            wide_axis = None
            if split == "test":
                if shift in ("height", "both"):
                    shape = (int(rng.integers(11, 15)), shape[1])
                if shift in ("width", "both"):
                    shape = (shape[0], int(rng.integers(11, 15)))
                if shift == "bands":
                    wide_axis = int(rng.integers(2))
            for _ in range(12):
                pair = _make_pair(rng, shape, count, texture, evidence, wide_axis)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded sampler produced duplicate inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _literal_mark(array):
    """Validation scan uses literal nine-cell tests instead of solver masks."""
    starts = [(r, c) for r in range(array.shape[0] - 2)
              for c in range(array.shape[1] - 2)
              if all(array[r + dr, c + dc] == BLANK
                     for dr in range(3) for dc in range(3))]
    marked = array.copy()
    for r, c in starts:
        marked[r:r + 3, c:c + 3] = HIGHLIGHT
    return marked, starts


def _literal_compress(array):
    rows = [0] + [i for i in range(1, len(array)) if list(array[i]) != list(array[i - 1])]
    cols = [0] + [j for j in range(1, array.shape[1])
                  if list(array[:, j]) != list(array[:, j - 1])]
    return array[np.ix_(rows, cols)]


def _near_misses(array, mask):
    for height, width in ((2, 3), (3, 2)):
        for row in range(array.shape[0] - height + 1):
            for col in range(array.shape[1] - width + 1):
                if (np.all(array[row:row + height, col:col + width] == BLANK)
                        and not np.any(mask[row:row + height, col:col + width])):
                    return True
    return False


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "illegal counts")
    seen = set()
    train_shapes, train_textures, train_counts = set(), set(), []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            colors = set(map(int, np.unique(inp)))
            _require(len(colors) == 2 and BLANK in colors and HIGHLIGHT not in colors,
                     "input must have blank 0 and one texture 2..9")
            texture = next(iter(colors - {BLANK}))
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            marked, starts = _literal_mark(inp)
            mask = marked == HIGHLIGHT
            _require(1 <= len(starts) <= 4, "expected 1..4 qualifying windows")
            _require(int(mask.sum()) == 9 * len(starts), "qualifying windows overlap")
            _require(all(abs(r - s) >= 4 or abs(c - d) >= 4
                         for (r, c), (s, d) in combinations(starts, 2)),
                     "target squares need a texture gap on at least one axis")
            _require(np.array_equal(out, _literal_compress(marked)), "incorrect ordered transformation")
            _require(np.array_equal(out, solve(inp)), "solver disagrees")
            _require(inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1],
                     "both axes must have nontrivial compression")
            _require(set(map(int, np.unique(out))) == {BLANK, HIGHLIGHT, texture},
                     "output must retain noise, untouched blank, and highlight")
            _require(_near_misses(inp, mask), "missing an unchanged 2x3 or 3x2 near miss")
            rs, cs = _line_starts(marked, 0), _line_starts(marked, 1)
            row_runs = np.diff(np.r_[rs, inp.shape[0]])
            col_runs = np.diff(np.r_[cs, inp.shape[1]])
            _require(np.array_equal(_expand(out, row_runs, col_runs), marked),
                     "compact output fails exact marked-grid reconstruction")
            _require(np.array_equal(rs, _line_starts(inp, 0))
                     and np.array_equal(cs, _line_starts(inp, 1)),
                     "new-color marking must preserve adjacent equality boundaries")
            _require(out.shape == _compress(out).shape, "output still has redundant adjacent lines")
            # Parent omissions must change every answer, not just a sampled pair.
            _require(not np.array_equal(_literal_compress(inp), out), "A is inert")
            _require(not np.array_equal(marked, out), "B is inert")
            if split == "train":
                train_shapes.add(out.shape)
                train_textures.add(texture)
                train_counts.append(len(starts))
                if index == 0:
                    _require(_attachment(inp, mask), "first demonstration needs attached blank noise")
                    _require(not np.array_equal(_mark(_compress(inp)), out),
                             "first demonstration does not distinguish reversed order")
                    # Painting the complete blank component is wrong: an
                    # unmarked neighbor survives in the same compact grid.
                    flood = mask.copy()
                    for _ in range(inp.size):
                        growth = flood.copy()
                        growth[1:] |= flood[:-1]
                        growth[:-1] |= flood[1:]
                        growth[:, 1:] |= flood[:, :-1]
                        growth[:, :-1] |= flood[:, 1:]
                        growth &= inp == BLANK
                        if np.array_equal(growth, flood):
                            break
                        flood = growth
                    component_mark = inp.copy()
                    component_mark[flood] = HIGHLIGHT
                    _require(not np.array_equal(_compress(component_mark), out),
                             "whole-blank-component shortcut survived")
                if index == 1:
                    _require(len(starts) >= 2 and _has_recurrence(out),
                             "second demonstration needs all targets and separated recurrence")
            else:
                large_logical = out.shape[0] > 10 or out.shape[1] > 10
                long_band = max(int(row_runs.max()), int(col_runs.max())) >= 7
                _require(large_logical or long_band, "missing structural test shift")
    _require(len(train_shapes) >= 2 and len(train_textures) >= 2, "training lacks geometry/palette variation")
    _require(train_counts[0] == 1 and train_counts[1] >= 2, "training lacks exhaustive-detection evidence")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                array = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(array.shape))
                digest.update(array.tobytes())
    return digest.hexdigest()


def self_test():
    """42 supplied seeds, bounded counters/hashes, all legal explicit counts."""
    shapes, output_shapes, textures, structures = set(), set(), set(), set()
    target_counts, target_layouts, schedules, highlight_shapes = set(), set(), set(), set()
    signatures = set()
    attached = edge = recurrence = reversed_wrong = pairs = 0
    height_shift = width_shift = both_shift = band_shift = 0
    for seed in SELF_TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                marked, starts = _literal_mark(inp)
                mask = marked == HIGHLIGHT
                pairs += 1
                shapes.add(inp.shape)
                output_shapes.add(out.shape)
                textures.add(tuple(map(int, np.unique(inp))))
                binary = (inp != BLANK).astype(np.int8)
                structures.add(hashlib.sha256(bytes(inp.shape) + binary.tobytes()).hexdigest())
                target_counts.add(len(starts))
                target_layouts.add((inp.shape, tuple(starts)))
                rs, cs = _line_starts(marked, 0), _line_starts(marked, 1)
                rr, cc = np.diff(np.r_[rs, inp.shape[0]]), np.diff(np.r_[cs, inp.shape[1]])
                schedules.add((tuple(map(int, rr)), tuple(map(int, cc))))
                for r, c in starts:
                    highlight_shapes.add((int(np.count_nonzero((rs >= r) & (rs < r + 3))),
                                          int(np.count_nonzero((cs >= c) & (cs < c + 3)))))
                attached += _attachment(inp, mask)
                recurrence += _has_recurrence(out)
                edge += any(r == 0 or c == 0 or r + 3 == inp.shape[0]
                            or c + 3 == inp.shape[1] for r, c in starts)
                reversed_wrong += not np.array_equal(_mark(_compress(inp)), out)
                if split == "test":
                    height_shift += out.shape[0] > 10 and out.shape[1] <= 10
                    width_shift += out.shape[1] > 10 and out.shape[0] <= 10
                    both_shift += out.shape[0] > 10 and out.shape[1] > 10
                    band_shift += max(int(rr.max()), int(cc.max())) >= 7
    for value, minimum, label in (
        (len(signatures), 42, "puzzles"), (len(shapes), 60, "input shapes"),
        (len(output_shapes), 35, "output shapes"), (len(textures), 8, "texture palettes"),
        (len(structures), 180, "color-normalized inputs"),
        (len(target_layouts), 180, "target layouts"), (len(target_counts), 4, "target counts"),
        (len(schedules), 150, "line schedules"), (len(highlight_shapes), 7, "compact target shapes"),
        (attached, 42, "attached blank geometry"), (recurrence, 42, "retained separated lines"),
        (edge, 1, "edge windows"), (reversed_wrong, 42, "order witnesses"),
        (height_shift, 1, "height shifts"), (width_shift, 1, "width shifts"),
        (both_shift, 1, "both-axis shifts"), (band_shift, 1, "long-band shifts"),
    ):
        _require(value >= minimum, f"insufficient {label}: {value}")
    _require(_digest(generate_puzzle(574621495)) == _digest(generate_puzzle(574621495)),
             "same seed is not reproducible")
    legal = 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(10000 + 10 * train + test, train, test)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "counts ignored")
            legal += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    # Solver extensions are deterministic and do not mutate even tiny inputs.
    for rows in ([[0]], [[0, 2, 2]], [[0, 0, 0]] * 3):
        array = np.array(rows, dtype=int)
        before = array.copy()
        solve(array)
        _require(np.array_equal(array, before), "solve mutated its input")
    return {"seeds": len(SELF_TEST_SEEDS), "pairs": pairs, "legal_count_combinations": legal,
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "texture_palettes": len(textures), "normalized_inputs": len(structures),
            "target_counts": sorted(target_counts), "compact_target_shapes": sorted(highlight_shapes),
            "target_layouts": len(target_layouts), "line_schedules": len(schedules),
            "attached_pairs": attached, "separated_recurrence_pairs": recurrence,
            "edge_pairs": edge, "reversed_order_differs": reversed_wrong,
            "test_shifts": {"height": height_shift, "width": width_shift,
                            "both": both_shift, "long_band": band_shift}}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
