"""Run-collapse ray branching. Only NumPy and the Python standard library.

The solver reads no files, RNG, examples, or generation metadata. Generation
works in memory; neither the solver nor self_test writes artifacts.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _bands(a):
    """Whole-line comparisons on the untouched input, independently by axis."""
    rows = np.r_[0, np.flatnonzero(np.any(a[1:] != a[:-1], axis=1)) + 1]
    cols = np.r_[0, np.flatnonzero(np.any(a[:, 1:] != a[:, :-1], axis=0)) + 1]
    return rows, cols, np.diff(np.r_[rows, a.shape[0]]), np.diff(np.r_[cols, a.shape[1]])


def _collapse(a):
    rows, cols, _, _ = _bands(a)
    return a[np.ix_(rows, cols)].copy()


def _recognize(a):
    """After collapse: modal background and a uniquely marked entry edge.

    All guide cells lie on one edge, away from its corners. That edge has no
    barriers. The other non-background color has at least one interior cell.
    Rotating a candidate by k quarter turns makes its rays point down.
    """
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("compact input needs three colors and a unique modal background")
    bg = int(colors[np.argmax(counts)])
    choices = []
    for k in range(4):
        c = np.rot90(a, k)
        h, w = c.shape
        if min(h, w) < 3:
            continue
        for g in map(int, colors):
            if g == bg:
                continue
            b = next(int(x) for x in colors if x not in (bg, g))
            entries = np.flatnonzero(c[0] == g)
            if (len(entries) and entries[0] > 0 and entries[-1] < w - 1
                    and not np.any(c[1:] == g) and not np.any(c[0] == b)
                    and np.any(c[1:-1, 1:-1] == b)):
                choices.append((k, bg, g, b))
    if len(choices) != 1:
        raise ValueError("guide color and entry edge must be uniquely visible")
    return choices[0]


def _runs(row, barrier):
    result = []
    c = 0
    while c < len(row):
        if row[c] != barrier:
            c += 1
            continue
        left = c
        while c + 1 < len(row) and row[c + 1] == barrier:
            c += 1
        result.append((left, c))
        c += 1
    return result


def _propagate(a, bg, g, b, mutant=None):
    """Simultaneous immutable-obstacle simulation, in downward coordinates.

    Mutants are finite counterexamples used for teaching checks, never labels.
    Statistics contain event counts, not saved execution traces.
    """
    h, w = a.shape
    out = a.copy()
    active = set(map(int, np.flatnonzero(a[0] == g)))
    stats = dict(events=0, simultaneous=0, multi_hit=0, two_flank=0,
                 clipped=0, singleton=0, interrupted=0, mergers=0, inert=0)
    for r in range(1, h):
        runs = _runs(a[r], b)
        hits = [(left, right, sum(left <= x <= right for x in active))
                for left, right in runs]
        struck = [(left, right, n) for left, right, n in hits if n]
        stats["inert"] += sum(n == 0 for _, _, n in hits)
        stats["simultaneous"] += len(struck) > 1
        following = {c for c in active if a[r, c] != b}
        for c in following:
            if a[r, c] == bg:
                out[r, c] = g
        selected = struck
        if mutant == "straight_stop":
            selected = []
        elif mutant == "one_per_layer":
            selected = struck[:1]
        elif mutant == "ignore_singletons":
            selected = [(l, u, n) for l, u, n in struck if l != u]
        elif mutant == "no_edge_detour":
            selected = [(l, u, n) for l, u, n in struck if l > 0 and u < w - 1]
        elif mutant == "all_barriers":
            selected = hits
        # Every struck run uses the same input row; no mutation can change the
        # interpretation of a later run at this depth. All rays are sets.
        for left, right, n in selected:
            stats["events"] += 1
            stats["multi_hit"] += n > 1
            stats["two_flank"] += left > 0 and right < w - 1
            stats["clipped"] += left == 0 or right == w - 1
            stats["singleton"] += left == right
            lo, hi = max(0, left - 1), min(w, right + 2)
            stats["interrupted"] += bool(np.any(a[r - 1, lo:hi] == b))
            for c in range(lo, hi):
                if mutant == "skip_entry_crosspiece" and r == 1:
                    continue
                if a[r - 1, c] == bg or mutant == "overwrite_crosspiece":
                    out[r - 1, c] = g
            for c in (left - 1, right + 1):
                if 0 <= c < w:
                    stats["mergers"] += c in following
                    following.add(c)
                    if a[r, c] == bg:
                        out[r, c] = g
        active = following
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Collapse first, simulate in compact coordinates, retain compact size."""
    compact = _collapse(_grid(input_grid))
    k, bg, g, b = _recognize(compact)
    out, _ = _propagate(np.rot90(compact, k), bg, g, b)
    # Do not collapse newly equal output lines and do not restore band widths.
    return np.rot90(out, -k).copy()


_MUTANTS = ("compress_only", "straight_stop", "one_per_layer", "ignore_singletons",
            "no_edge_detour", "all_barriers", "overwrite_crosspiece",
            "skip_entry_crosspiece",
            "global_unique", "recollapse_output", "raw_then_collapse", "raw_then_sample")


def _unique_lines(a):
    # Deliberately wrong: separated identical lines would also be discarded.
    rows, cols, seen_r, seen_c = [], [], set(), set()
    for i, row in enumerate(a):
        key = row.tobytes()
        if key not in seen_r:
            rows.append(i)
            seen_r.add(key)
    for i, col in enumerate(a.T):
        key = col.tobytes()
        if key not in seen_c:
            cols.append(i)
            seen_c.add(key)
    return a[np.ix_(rows, cols)]


def _alternative(a, name):
    compact = _collapse(a)
    k, bg, g, b = _recognize(compact)
    c = np.rot90(compact, k)
    if name == "compress_only":
        return compact
    if name == "global_unique":
        result, _ = _propagate(_unique_lines(c), bg, g, b)
    elif name.startswith("raw_then_"):
        raw = np.rot90(a, k)
        result, _ = _propagate(raw, bg, g, b)
        if name == "raw_then_collapse":
            result = _collapse(result)
        else:
            rows, cols, _, _ = _bands(raw)
            result = result[np.ix_(rows, cols)]
    elif name == "recollapse_output":
        result = _collapse(_propagate(c, bg, g, b)[0])
    else:
        result, _ = _propagate(c, bg, g, b, name)
    return np.rot90(result, -k).copy()


def _scene(rng, kind, large=False, quiet=False):
    """Compose input geometry, not complete parent puzzles or output templates.

    Unobservable adjacent scene lines are intentionally collapsed before they
    become the latent input. Thus no hidden intended blank-layer counts exist.
    """
    h = int(rng.integers(9, 19 if large else 14))
    w = int(rng.integers(11, 20 if large else 17))
    a = np.zeros((h, w), dtype=np.int8)
    left = int(rng.integers(2, max(3, w // 2 - 1)))
    right = int(rng.integers(max(left + 4, w // 2 + 1), w - 2))
    a[0, [left, right]] = 1
    if kind == "fork":
        # Separate simultaneous contacts, followed by a shared broad barrier.
        # An immediate contact can paint the entry line. This also makes some
        # formerly distinct neighboring columns equal only AFTER simulation;
        # those columns must still remain in the rendered compact frame.
        first = int(rng.integers(1, 3))
        a[first, left] = 2
        a[first, right:right + 2] = 2
        a[4, max(0, left - 1):right + 3] = 2
    elif kind == "join":
        # Multi-hit, interrupted upstream crosspiece, then a clipped detour.
        a[1, :left] = 2  # This prefix run is not reached by either entry.
        a[2, left:right + 1] = 2
        a[4, :left] = 2
        a[6, right + 1:] = 2
    elif kind == "edge":
        a[0] = 0
        a[0, left] = 1
        a[2, :left + 1] = 2
        a[4, left + 1:] = 2
    elif kind == "sparse":
        a[0] = 0
        a[0, left] = 1
        a[2, left:right] = 2
        a[5, :left - 1] = 2
    else:
        if rng.random() < .4:
            a[0, right] = 0
        if rng.random() < .25:
            middle = (left + right) // 2
            a[0, middle] = 1
        a[2, left:left + int(rng.integers(1, 4))] = 2
    if not quiet:
        # Bars, sparse singletons, and connected stems share one barrier role.
        count = int(rng.integers(1, 5 if large else 4))
        for _ in range(count):
            r = int(rng.integers(5 if kind in ("fork", "join") else 3, h))
            c = int(rng.integers(w))
            if rng.random() < .25:
                end = min(h, r + int(rng.integers(2, 5)))
                a[r:end, c] = 2
                if rng.random() < .5 and c + 1 < w:
                    a[end - 1, c:c + 2] = 2
            else:
                n = int(rng.integers(1, w + 1))
                c = min(c, w - n)
                a[r, c:c + n] = 2
    if rng.random() < .5:
        a = np.fliplr(a)
    return _collapse(a)


def _weights(rng, n, *, expand=True, stress=False, irregular=False):
    values = np.ones(n, dtype=int)
    if not expand:
        return values
    if stress:
        index = int(rng.integers(n))
        values[index] = int(rng.integers(6, min(10, 31 - n) + 1))
        budget = int(rng.integers(0, min(30 - int(values.sum()), n) + 1))
        for _ in range(budget):
            j = int(rng.integers(n))
            if j != index and values[j] < 3:
                values[j] += 1
        # Explicit thin neighbor rules out any global scaling factor.
        values[index - 1 if index else 1] = 1
        return values
    mode = int(rng.integers(4))
    if mode == 0 and not irregular and 2 * n <= 30:
        return np.full(n, 2, dtype=int)
    budget = int(rng.integers(1, min(30 - n, 2 * n) + 1))
    protected = int(rng.integers(n)) if irregular else -1
    for _ in range(budget):
        candidates = [i for i in range(n) if i != protected and values[i] < 3]
        if not candidates:
            break
        values[int(rng.choice(candidates))] += 1
    if values.max() == 1:
        values[0 if protected != 0 else 1] = 2
    if mode in (1, 2) and not irregular:
        values.sort()
        if mode == 2:
            values = values[::-1]
    return values


def _signature(a):
    return a.shape, a.tobytes()


def _make_pair(rng, kind, palette, rotation, split, index, seen,
               survivors=(), excluded_shape=None):
    for attempt in range(192):
        c = _scene(rng, kind, large=split == "test", quiet=attempt >= 128)
        try:
            _recognize(c)
        except ValueError:
            continue
        expected, stats = _propagate(c, 0, 1, 2)
        if np.array_equal(c, expected) or min(c.shape) < 4:
            continue
        if kind == "fork" and not (stats["simultaneous"] and stats["two_flank"]):
            continue
        if kind == "join" and not (stats["multi_hit"] and stats["clipped"]
                                   and stats["inert"] and stats["events"] >= 2):
            continue
        if excluded_shape == c.shape:
            continue
        if split == "test":
            axis = int(rng.integers(2))
            rows = _weights(rng, c.shape[0], stress=axis == 0, expand=axis == 0 or rng.random() < .8)
            cols = _weights(rng, c.shape[1], stress=axis == 1, expand=axis == 1 or rng.random() < .8)
        else:
            axis = int(rng.integers(3)) if index >= 2 else 2
            rows = _weights(rng, c.shape[0], expand=axis != 1, irregular=index < 2)
            cols = _weights(rng, c.shape[1], expand=axis != 0, irregular=index < 2)
        raw = np.repeat(np.repeat(c, rows, axis=0), cols, axis=1)
        mapping = np.asarray(palette, dtype=np.int8)
        raw = np.rot90(mapping[raw], rotation).copy()
        out = np.rot90(mapping[expected], rotation).copy()
        if _signature(raw) in seen:
            continue
        if split == "test" and any(not np.array_equal(_alternative(raw, m), out) for m in survivors):
            continue
        if split == "train" and index == 0:
            # Require a visible witness for coordinate dependence, not just a
            # smaller output whose shape alone would expose missing collapse.
            if np.array_equal(_alternative(raw, "raw_then_sample"), out):
                continue
            if np.array_equal(_alternative(raw, "one_per_layer"), out):
                continue
        return {"input": raw, "output": out}
    raise RuntimeError("bounded construction exhausted; no rule exceptions or cached pairs used")


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, size=3, replace=False)))
    rotation = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    survivors = set(_MUTANTS)
    first_shape = None
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            kind = ("fork" if i == 0 else "join") if split == "train" and i < 2 else str(
                rng.choice(("fork", "join", "edge", "sparse", "general")))
            pair = _make_pair(rng, kind, palette, rotation, split, i, seen,
                              sorted(survivors), first_shape if split == "train" and i == 1 else None)
            puzzle[split].append(pair)
            seen.add(_signature(pair["input"]))
            if split == "train":
                if i == 0:
                    # Compare shapes in canonical, rather than rotated, axes.
                    first_shape = np.rot90(_collapse(pair["input"]), -rotation).shape
                survivors = {m for m in survivors
                             if np.array_equal(_alternative(pair["input"], m), pair["output"])}
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "counts")
    seen, survivors, train_shapes, stats_train = set(), set(_MUTANTS), set(), []
    conventions, both_nonuniform, coordinate_witness = set(), False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            rows, cols, rh, cw = _bands(a)
            compact = a[np.ix_(rows, cols)]
            _require(compact.shape != a.shape, "missing repetition bands")
            _require(np.array_equal(np.repeat(np.repeat(compact, rh, axis=0), cw, axis=1), a),
                     "lossless band reconstruction")
            _require(np.array_equal(_collapse(compact), compact), "compact frame not stable")
            k, bg, g, b = _recognize(compact)
            conventions.add((k, bg, g, b))
            _require(out.shape == compact.shape, "restore size or collapse-after-drawing error")
            _require(np.array_equal(solve(a), out), "solver disagreement")
            _require(np.all(out[compact != bg] == compact[compact != bg]), "immutable pixels changed")
            changed = out != compact
            _require(np.any(changed) and np.all(out[changed] == g), "missing or invalid drawing")
            _require(set(map(int, np.unique(out))) == {bg, g, b}, "palette changed")
            _, stats = _propagate(np.rot90(compact, k), bg, g, b)
            if split == "train":
                train_shapes.add(compact.shape)
                stats_train.append(stats)
                both_nonuniform |= (max(rh) > 1 and max(cw) > 1
                                    and len(set(rh)) > 1 and len(set(cw)) > 1)
                coordinate_witness |= not np.array_equal(_alternative(a, "raw_then_sample"), out)
                _require(max(max(rh), max(cw)) <= 3, "training band schedule")
                survivors = {m for m in survivors if np.array_equal(_alternative(a, m), out)}
            else:
                _require(max(max(rh), max(cw)) >= 6, "missing resolution shift")
                _require(all(np.array_equal(_alternative(a, m), out) for m in survivors),
                         "a training-consistent tested shortcut changes test answer")
    _require(len(conventions) == 1, "puzzle palette/direction convention changed")
    _require(len(train_shapes) >= 2 and both_nonuniform and coordinate_witness,
             "training lacks compression/coordinate evidence")
    for feature in ("two_flank", "simultaneous", "multi_hit", "clipped", "inert"):
        _require(any(s[feature] for s in stats_train), f"training lacks {feature}")
    _require(any(s["events"] >= 2 for s in stats_train), "training lacks a cascade")
    _require("compress_only" not in survivors and "straight_stop" not in survivors,
             "training fails parent-A necessity")
    _require("raw_then_sample" not in survivors, "training fails parent-B coordinate necessity")
    _require("one_per_layer" not in survivors, "simultaneous witness is redundant")
    return True


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded: 42 default puzzles, 30 count cases, small metamorphic checks.

    Streaming metrics retain hashes and counters, not all generated payloads.
    These are generator-owned checks, not an independent semantic review.
    """
    def digits(text):
        return np.array([[int(c) for c in row] for row in text.split()], dtype=np.int8)

    # Hand-calculated regression: fork, edge detour, an inert singleton, and
    # repeated blank rows / nonadjacent identical columns which must survive.
    a = digits("0010000 0000000 0222000 0000000 2200020 0000000")
    expected = digits("0010000 1111100 1222100 1110100 2210120 0010100")
    _require(np.array_equal(solve(a), expected), "hand-calculated regression")
    expanded = np.repeat(np.repeat(a, [2, 1, 3, 2, 1, 2], axis=0), [1, 3, 1, 2, 1, 2, 1], axis=1)
    _require(np.array_equal(solve(expanded), expected), "resolution invariance")
    _require(_collapse(a).shape == a.shape and _unique_lines(a).shape != a.shape,
             "separated repeats were discarded")
    fingerprints, layouts, shapes, palettes, directions, entries, events = (set() for _ in range(7))
    features = {s: {f: 0 for f in ("simultaneous", "multi_hit", "clipped", "singleton",
                                  "interrupted", "mergers", "inert")} for s in ("train", "test")}
    repeated, one_axis, output_repeats, schedules = 0, 0, 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                raw, out = pair["input"], pair["output"]
                compact = _collapse(raw)
                k, bg, g, b = _recognize(compact)
                c = np.rot90(compact, k)
                normalized = np.zeros(c.shape, dtype=np.int8)
                normalized[c == g], normalized[c == b] = 1, 2
                layouts.add(_signature(normalized))
                shapes.add(compact.shape)
                palettes.add((bg, g, b))
                directions.add(k)
                entries.add(int(np.count_nonzero(c[0] == g)))
                _, stats = _propagate(c, bg, g, b)
                events.add(stats["events"])
                for feature in features[split]:
                    features[split][feature] += stats[feature] > 0
                _, _, rh, cw = _bands(raw)
                schedules.update((tuple(map(int, rh)), tuple(map(int, cw))))
                one_axis += (max(rh) == 1) != (max(cw) == 1)
                repeated += _unique_lines(compact).shape != compact.shape
                output_repeats += _collapse(out).shape != out.shape
                # Input-only purity, no mutation, rotation/color equivariance.
                before = raw.copy()
                answer = solve(raw)
                _require(np.array_equal(raw, before) and np.array_equal(answer, out), "purity")
                if split == "test":
                    mapping = np.array([9 - x for x in range(10)], dtype=np.int8)
                    _require(np.array_equal(solve(np.rot90(mapping[raw])), np.rot90(mapping[out])),
                             "rotation and palette equivariance")
                    # A different expansion of this same latent input has the
                    # same answer, even with no expansion on one axis.
                    fresh = np.repeat(compact, 2, axis=0) if 2 * compact.shape[0] <= 30 else compact
                    _require(np.array_equal(solve(fresh), out), "latent-frame invariance")
    _require(len(fingerprints) == 42 and len(layouts) >= 100, "structural diversity")
    _require(len(shapes) >= 20 and len(palettes) >= 30 and directions == set(range(4)), "shape/role variation")
    _require(len(entries) >= 2 and len(events) >= 4 and len(schedules) >= 80, "count/band variation")
    _require(repeated > 0 and one_axis > 0 and output_repeats > 0, "repetition edge cases")
    for split in features:
        _require(all(features[split][f] > 0 for f in features[split]), f"missing {split} event coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(8000 + 10 * nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_digest(generate_puzzle(0)) == _digest(generate_puzzle(0)), "determinism")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), normalized_latent_structures=len(layouts),
                latent_shapes=len(shapes), palettes=len(palettes), directions=len(directions),
                entry_counts=sorted(entries), collision_counts=sorted(events), band_schedules=len(schedules),
                separated_repeat_pairs=repeated, one_axis_expansion_pairs=int(one_axis),
                outputs_with_new_equal_adjacent_lines=output_repeats, features=features,
                semantic_verification="generator-owned invariants and hand-calculated regression only")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
