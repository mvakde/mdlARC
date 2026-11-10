"""Pulse-category support tiling; standalone, input-only semantics.

The complete square drawing field is the virtual categorical input to the
support-mask expansion. No parent module or generation state is used by solve.
Only NumPy and the standard library are required. No artifacts are written here.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(num_train, num_test):
    tr = _integer(num_train, "num_train", 2, 10)
    te = _integer(num_test, "num_test", 1, 4)
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    return tr, te


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _analyze(input_grid):
    """Locate the square drawing half-plane, then read maximal source runs.

    An inactive lane makes every source column contain background. Consequently
    the separator is the only full-width nonbackground line. No longest-run or
    frequency heuristic, literal marker hue, or hidden direction is required.
    """
    grid = _grid(input_grid)
    if 0 in grid:
        raise ValueError("zero is reserved for output mismatches")
    n = min(grid.shape)
    if not 2 <= n <= 5 or max(grid.shape) < n + 2:
        raise ValueError("the lane count and drawing depth must both be 2..5")
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, rotation)
        if a.shape[0] != n:
            continue
        col = a.shape[1] - n - 1
        if col < 1:
            continue
        separator = int(a[0, col])
        background = int(a[0, col + 1])
        if separator == background or not np.all(a[:, col] == separator):
            continue
        if not np.all(a[:, col + 1:] == background):
            continue
        source = a[:, :col]
        if np.any(source == separator) or not np.any(source != background):
            continue
        if not np.any(np.all(source == background, axis=1)):
            continue
        lanes = []
        for row in range(n):
            runs, cursor = [], col - 1
            while cursor >= 0:
                color = int(a[row, cursor])
                if color == background:
                    cursor -= 1
                    continue
                near = col - cursor
                end = cursor
                while cursor >= 0 and a[row, cursor] == color:
                    cursor -= 1
                runs.append((color, end - cursor, near))
            lanes.append(runs)
        candidates.append({"n": n, "rotation": rotation, "canonical": a,
                           "col": col, "background": background,
                           "separator": separator, "lanes": lanes})
    if len(candidates) != 1:
        raise ValueError("input must have one square drawing field and separator")
    return candidates[0]


def _field(info, priority="near", phase="boundary", period_delta=0):
    """Compute winners cell by cell, then restore the square's screen orientation."""
    n, bg = info["n"], info["background"]
    field = np.full((n, n), bg, dtype=np.int8)
    for row, runs in enumerate(info["lanes"]):
        for t in range(n):
            firing = []
            for run in runs:
                color, length, near = run
                period = max(1, length + period_delta)
                # Deliberately wrong hypotheses are used only by teaching checks.
                offset = near - 1 if phase == "gap" else 0
                if (t - offset) % period == 0:
                    firing.append(run)
            if not firing:
                continue
            if priority == "far":
                winner = firing[-1]
            elif priority == "short":
                winner = min(firing, key=lambda b: (b[1], b[2]))
            elif priority == "long":
                winner = min(firing, key=lambda b: (-b[1], b[2]))
            else:
                winner = firing[0]
            field[row, t] = winner[0]
    return np.rot90(field, -info["rotation"]).copy()


def _expand(field):
    n = field.shape[0]
    selectors = np.repeat(np.repeat(field, n, axis=0), n, axis=1)
    reference = np.tile(field, (n, n))
    return np.where(selectors == reference, selectors, 0).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Emit the equality relation of the complete, inferred square pulse field."""
    return _expand(_field(_analyze(input_grid)))


def _visible_options(n, maximum, count):
    """Small arithmetic search: at most 4**4 candidate run-length tuples."""
    result = []
    for lengths in itertools.product(range(1, maximum + 1), repeat=count):
        winners = set()
        for t in range(n):
            hits = [i for i, length in enumerate(lengths) if t % length == 0]
            if hits:
                winners.add(hits[0])
        if len(winners) == count:
            result.append(lengths)
    return result


def _random_lengths(rng, maximum, max_runs):
    count = int(rng.choice(np.arange(1, max_runs + 1),
                           p=np.array([.44, .28, .18, .10][:max_runs])
                           / sum([.44, .28, .18, .10][:max_runs])))
    return [int(x) for x in rng.integers(1, maximum + 1, size=count)]


def _paint(rng, n, row_specs, rotation, palette, background, separator, shifted):
    """Construct from run specifications, independently of input parsing/solve.

    Each row spec is (nearest-first lengths, optional fixed colors). Gaps and
    margins do not enter the periodic painter. Bounds are computed before paint.
    """
    rows = [int(x) for x in rng.choice(n, len(row_specs), replace=False)]
    plans = []
    for index, (lengths, fixed) in enumerate(row_specs):
        if fixed is None:
            colors = [int(rng.choice(palette)) for _ in lengths]
        else:
            colors = list(fixed)
        # Equal-color neighbors must be separate maximal runs.
        gaps = [int(rng.integers(0, 3)) for _ in lengths]
        for i in range(1, len(lengths)):
            if colors[i] == colors[i - 1]:
                gaps[i] = max(1, gaps[i])
        if shifted and index == 0:
            # Gap one is a visible counterexample to source-offset phase.
            gaps[0] = 1
        plans.append((list(lengths), colors, gaps))
    minimum = max(sum(ls) + sum(gs) for ls, cs, gs in plans)
    limit = 29 - n
    _require(minimum <= limit, "source does not fit")
    source_width = int(rng.integers(minimum, min(limit, minimum + 7) + 1))
    inp = np.full((n, source_width + 1 + n), background, dtype=np.int8)
    inp[:, source_width] = separator
    field = np.full((n, n), background, dtype=np.int8)
    for row, (lengths, colors, gaps) in zip(rows, plans):
        spare = source_width - sum(lengths) - sum(gaps)
        # All remaining slack can become independent source-side gaps or margin.
        additions = rng.multinomial(spare, [1 / (len(gaps) + 1)] * (len(gaps) + 1))
        if shifted and row == rows[0]:
            additions[-1] += additions[0]
            additions[0] = 0
        gaps = [g + int(a) for g, a in zip(gaps, additions)]
        cursor = source_width
        for length, color, gap in zip(lengths, colors, gaps):
            cursor -= gap
            inp[row, cursor - length:cursor] = color
            cursor -= length
        for length, color in reversed(list(zip(lengths, colors))):
            field[row, ::length] = color
    inp = np.rot90(inp, rotation).copy()
    field = np.rot90(field, rotation).copy()
    # Construction uses individual color support masks, rather than _expand.
    output = np.zeros((n * n, n * n), dtype=np.int8)
    for color in np.unique(field):
        support = (field == color).astype(np.int8)
        output += np.kron(support, support) * color
    return {"input": inp, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent curriculum, with a visible size or direction shift."""
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["grow", "clip", "turn"]))
    maximum = 3 if shift == "grow" else int(rng.integers(3, 5))
    max_runs = maximum
    base = int(rng.integers(4))
    mixed = bool(rng.integers(2)) if shift != "turn" else False
    witness_colors = tuple(int(x) for x in rng.choice(np.arange(1, 10), 2, replace=False))
    puzzle, used = {"train": [], "test": []}, set()

    def add(split, n, specs, rotation, protected=(), shifted=False):
        for _ in range(64):
            available = [c for c in range(1, 10) if c not in protected]
            bg, sep = (int(c) for c in rng.choice(available, 2, replace=False))
            colors = [c for c in range(1, 10) if c not in (bg, sep)]
            count = int(rng.integers(max(2, len(protected)), len(colors) + 1))
            palette = list(protected)
            others = [c for c in colors if c not in protected]
            palette += [int(c) for c in rng.choice(others, count - len(palette), replace=False)]
            pair = _paint(rng, n, specs, rotation, palette, bg, sep, shifted)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in used:
                used.add(key)
                puzzle[split].append(pair)
                return
        raise RuntimeError("bounded distinct-input sampling exhausted")

    n0 = 4 if shift == "grow" else int(rng.integers(4, 6))
    a, b = witness_colors
    anchor = [([2, 3], [a, b]), ([3, 2], [b, a])]
    for _ in range(int(rng.integers(n0 - 2))):
        anchor.append((_random_lengths(rng, maximum, max_runs), None))
    add("train", n0, anchor, base, witness_colors, shifted=True)

    n1 = 5 if maximum == 4 else (4 if shift == "grow" else int(rng.integers(4, 6)))
    periods = list(range(2, maximum + 1))
    rng.shuffle(periods)
    lessons = [([p], None) for p in periods]
    rich_options = _visible_options(n1, maximum, max_runs)
    rich = rich_options[int(rng.integers(len(rich_options)))]
    # Distinct colors make every constituent of this richer mixture observable.
    rich_colors = [int(c) for c in rng.choice(np.arange(1, 10), max_runs, replace=False)]
    lessons.append((rich, rich_colors))
    # All fixed lesson colors must be protected from background/separator roles.
    direction1 = (base + 2) % 4 if mixed else base
    add("train", n1, lessons, direction1, rich_colors, shifted=True)

    for index in range(2, num_train):
        n = int(rng.integers(4, 6) if shift == "clip" else
                rng.integers(2, 5 if shift == "grow" else 6))
        count = int(rng.integers(1, n))
        specs = [(_random_lengths(rng, maximum, max_runs), None) for _ in range(count)]
        rotation = int(rng.integers(4)) if mixed else base
        add("train", n, specs, rotation)
    # Curriculum presentation order itself carries no semantic meaning.
    rng.shuffle(puzzle["train"])

    for index in range(num_test):
        n = 5 if shift == "grow" else (int(rng.integers(2, 4)) if shift == "clip"
                                      else int(rng.integers(4, 6)))
        count = int(rng.integers(1, n))
        specs = [(_random_lengths(rng, maximum, max_runs), None) for _ in range(count)]
        # Direction change in turn mode is perpendicular to every demonstration.
        rotation = (base + int(rng.choice([1, 3]))) % 4 if shift == "turn" else base
        add("test", n, specs, rotation)
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_field(info):
    """Independent far-to-near overwrite oracle for the validator."""
    n = info["n"]
    canonical = np.full((n, n), info["background"], dtype=np.int8)
    for row, runs in enumerate(info["lanes"]):
        for color, length, near in reversed(runs):
            for t in range(0, n, length):
                canonical[row, t] = color
    return np.rot90(canonical, -info["rotation"]).copy()


def _teaching_check(training):
    singles, short_near, long_near, edges = set(), False, False, set()
    shifted, global_repeat, asymmetric = False, False, False
    static_rejected, row_only_rejected = False, False
    rivals = {name: False for name in ("far", "short", "long", "gap", "minus", "plus")}
    for pair in training:
        info = _analyze(pair["input"])
        field = _audit_field(info)
        canonical = np.rot90(field, info["rotation"])
        n = info["n"]
        for row, runs in enumerate(info["lanes"]):
            if len(runs) == 1:
                color, length, near = runs[0]
                if length < n and canonical[row, 0] == canonical[row, length] == color:
                    singles.add(length)
                    shifted |= near == 2 and length > 1
            for i, first in enumerate(runs):
                for second in runs[i + 1:]:
                    if first[0] == second[0]:
                        continue
                    survives = any(t % second[1] == 0
                                   and all(t % closer[1] for closer in runs[:i + 1])
                                   and canonical[row, t] == second[0] for t in range(n))
                    if survives and canonical[row, 0] == first[0]:
                        short_near |= first[1] < second[1]
                        long_near |= first[1] > second[1]
                        edges.add((first[0], second[0]))
        global_repeat |= any(len(set(int(r) for r, c in np.argwhere(field == color))) >= 2
                             for color in np.unique(field) if color != info["background"])
        asymmetric |= not np.array_equal(field, field.T) and not np.array_equal(field, np.fliplr(field))
        for name in rivals:
            kwargs = ({"priority": name} if name in ("far", "short", "long") else
                      {"phase": "gap"} if name == "gap" else
                      {"period_delta": -1 if name == "minus" else 1})
            rivals[name] |= not np.array_equal(_expand(_field(info, **kwargs)), pair["output"])
        # Repetition of the whole field, solid substitution, and row-only masks
        # are all concrete B-omitting or B-weakening alternative rules.
        _require(not np.array_equal(np.tile(field, (n, n)), pair["output"]), "whole-field copying fits")
        _require(not np.array_equal(np.repeat(np.repeat(field, n, 0), n, 1), pair["output"]),
                 "solid tile substitution fits")
        row_only = np.zeros_like(pair["output"])
        for r in range(n):
            for c in range(n):
                color = field[r, c]
                row_only[r*n+r, c*n:(c+1)*n] = np.where(field[r] == color, color, 0)
        row_only_rejected |= not np.array_equal(row_only, pair["output"])
        static = np.full((n, n), info["background"], dtype=np.int8)
        for r, runs in enumerate(info["lanes"]):
            if runs:
                static[r, :] = runs[0][0]
        static = np.rot90(static, -info["rotation"])
        # At least one training example must need numerical pulse synthesis,
        # rather than a nearest-source color copied along a complete lane.
        static_rejected |= not np.array_equal(_expand(static), pair["output"])
    _require({2, 3} <= singles, "training must expose numerical recurrence")
    _require(short_near and long_near and any((b, a) in edges for a, b in edges),
             "training must refute period and fixed-color collision priorities")
    _require(shifted, "training must distinguish source-gap phase")
    _require(global_repeat and asymmetric, "training needs global masks and orientation evidence")
    _require(all(rivals.values()), "a simpler pulse hypothesis fits every demonstration")
    _require(static_rejected, "omitting pulse synthesis fits all demonstrations")
    _require(row_only_rejected, "row-local masks fit all demonstrations")
    return singles


def validate_generated_puzzle(puzzle) -> None:
    """Validate returned grids, algebraic invariants, and actual teaching evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, max_test_period = set(), 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(inp)
            n, field = info["n"], _audit_field(info)
            _require(out.shape == (n * n, n * n), "incorrect expansion size")
            _require(info["separator"] not in out, "separator leaked into categorical reference")
            _require(set(np.unique(out)) == set(np.unique(field)) | {0}, "wrong output palette")
            for r in range(n):
                for c in range(n):
                    color = int(field[r, c])
                    tile = out[r*n:(r+1)*n, c*n:(c+1)*n]
                    for u in range(n):
                        for v in range(n):
                            expected = color if color == field[u, v] else 0
                            _require(tile[u, v] == expected, "incorrect full-field equality mask")
                    _require(tile[r, c] == color, "reflexive diagonal must recover each pulse category")
                    _require(np.count_nonzero(tile) == np.count_nonzero(field == color), "wrong support size")
            # Equality of position pairs is symmetric under exchange of the two
            # positions, even though the displayed 2D grid need not be symmetric.
            tensor = out.reshape(n, n, n, n).transpose(0, 2, 1, 3)
            _require(np.array_equal(tensor, tensor.transpose(2, 3, 0, 1)), "relation is not symmetric")
            for runs in info["lanes"]:
                _require(len(runs) <= 4 and all(1 <= run[1] <= 4 for run in runs), "generated source bounds")
                if split == "test":
                    max_test_period = max([max_test_period] + [run[1] for run in runs])
            _require(np.array_equal(solve(inp), out), "input-only solve disagrees")
    taught = _teaching_check(puzzle["train"])
    _require(max_test_period <= max(taught), "test period exceeds taught recurrence range")


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded tests: 42 default and minimal curricula, fixtures, counts, mutations.

    Process puzzles one at a time; retain only small hashes/feature sets. At most
    14 pairs of <=900 input / <=625 output bytes live in any generated puzzle.
    No JSON, execution traces, images, journals, or backup payloads are stored.
    """
    # Hand-calculated support tiling, including the silent category.
    small = np.array([[2, 2, 9, 1, 1], [1, 1, 9, 1, 1]], dtype=np.int8)
    expected = np.array([[2, 0, 0, 1], [0, 0, 1, 1],
                         [0, 1, 0, 1], [1, 1, 1, 1]], dtype=np.int8)
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(small, turn)), np.rot90(expected, turn)), "small fixture")
    rows = ["133322191111", "122133391111", "111441191111", "111111191111"]
    fixture = np.array([list(map(int, row)) for row in rows], dtype=np.int8)
    field = np.array([[2, 1, 2, 3], [3, 1, 2, 3], [4, 1, 4, 1], [1, 1, 1, 1]], dtype=np.int8)
    for turn in range(4):
        _require(np.array_equal(_field(_analyze(np.rot90(fixture, turn))), np.rot90(field, turn)), "pulse fixture")
    # A period or priority edit changes the relation globally, not only one tile.
    far = _expand(_field(_analyze(fixture), priority="far"))
    _require(np.count_nonzero(far != solve(fixture)) > 16, "A ablation must affect multiple tiles")
    digests, structures, shapes, sizes, directions, backgrounds = set(), set(), set(), set(), set(), set()
    periods, active_counts, run_counts, palettes = set(), set(), set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _signature(puzzle)
        _require(digest == _signature(generate_puzzle(seed)), "nondeterminism")
        digests.add(digest)
        # Minimal curricula retain the same teaching guarantees, no hidden extra
        # examples are needed by tests with legal explicit counts.
        validate_generated_puzzle(generate_puzzle(seed, 2, 1))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                info = _analyze(pair["input"])
                p = _audit_field(info)
                mapping = {}
                structures.add((p.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in p.flat)))
                shapes.add(pair["input"].shape)
                sizes.add(info["n"])
                directions.add((-info["rotation"]) % 4)
                backgrounds.add(info["background"])
                active_counts.add(sum(bool(runs) for runs in info["lanes"]))
                palettes.add(tuple(int(c) for c in np.unique(p)))
                for runs in info["lanes"]:
                    run_counts.add(len(runs))
                    periods.update(run[1] for run in runs)
    _require(len(digests) == 42 and len(structures) >= 80 and len(shapes) >= 30, "insufficient structural variation")
    _require(sizes == {2, 3, 4, 5} and directions == {0, 1, 2, 3}, "missing geometry variation")
    _require(periods == {1, 2, 3, 4} and run_counts == {0, 1, 2, 3, 4}, "missing pulse variation")
    _require(active_counts == {1, 2, 3, 4} and len(backgrounds) >= 7 and len(palettes) >= 50, "missing density/palette variation")
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(1000 + tr * 10 + te, tr, te)
            _require(len(p["train"]) == tr and len(p["test"]) == te, "count error")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0, 2, 1)
    # Check that serialized nested-list pairs have the same public semantics.
    validate_generated_puzzle({s: [{r: a.tolist() for r, a in p.items()} for p in original[s]] for s in original})
    changed = {s: [{r: a.copy() for r, a in p.items()} for p in original[s]] for s in original}
    changed["test"][0]["output"][0, 0] = (int(changed["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError("output corruption accepted")
    # Geometry and category renaming are visible, equivariant conventions.
    permutation = np.array([0, 8, 7, 6, 5, 4, 3, 2, 1, 9], dtype=np.int8)
    for pair in original["train"] + original["test"]:
        for turn in range(4):
            _require(np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn)), "rotation equivariance")
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]), "color equivariance")
    return {"seeds": 42, "default_pairs": pairs, "distinct_puzzles": len(digests),
            "input_shapes": len(shapes), "categorical_structures": len(structures),
            "field_sizes": sorted(sizes), "directions": sorted(directions),
            "backgrounds": sorted(backgrounds), "palettes": len(palettes),
            "periods": sorted(periods), "active_counts": sorted(active_counts),
            "run_counts": sorted(run_counts)}
