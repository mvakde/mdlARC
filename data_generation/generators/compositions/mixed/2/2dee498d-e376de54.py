"""Normalize an odd stroke family, then retain first-occurring whole columns.

Only NumPy and the standard library are used. Generation draws its labels from
sampled geometry; solve reconstructs geometry from input cells. No evidence files
are read at runtime. self_test streams its campaign and never saves artifacts.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


_AXES = {"V": (1, 0), "D+": (1, 1), "D-": (1, -1)}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid, size 1..30, colors 0..9")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.tobytes()


def _columns(g):
    return [tuple(int(x) for x in g[:, j]) for j in range(g.shape[1])]


def _dedup(g):
    seen, indices = set(), []
    for j, column in enumerate(_columns(g)):
        if column not in seen:
            seen.add(column)
            indices.append(j)
    return g[:, indices].copy()


def _components(mask):
    """Eight-neighbor components; separated strokes are individual components."""
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        pending, group = [first], []
        while pending:
            r, c = pending.pop()
            group.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in unseen:
                        unseen.remove(q)
                        pending.append(q)
        groups.append(sorted(group))
    return groups


def _analyze(g):
    values, counts = np.unique(g, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    if int(counts.max()) * 2 <= g.size:
        raise ValueError("background must occupy a strict majority")
    components = _components(g != background)
    n = len(components)
    if n < 3 or n % 2 != 1:
        raise ValueError("need an odd family of at least three separate strokes")
    candidates = []
    for axis, (_, dc) in _AXES.items():
        strokes = []
        for cells in components:
            r0, c0 = cells[0]
            color = int(g[r0, c0])
            # r is an axial coordinate for all three supported axes.
            if any(r != r0 + i or c != c0 + i * dc or int(g[r, c]) != color
                   for i, (r, c) in enumerate(cells)):
                break
            line = c0 if axis == "V" else r0 * dc - c0
            low = r0 + dc * c0
            r1, c1 = cells[-1]
            high = r1 + dc * c1
            strokes.append((line, cells, color, low, high))
        else:
            strokes.sort(key=lambda s: s[0])
            keys = [s[0] for s in strokes]
            gaps = [b - a for a, b in zip(keys, keys[1:])]
            if min(gaps) < (2 if axis == "V" else 4):
                continue
            common_low = len({s[3] for s in strokes}) == 1
            common_high = len({s[4] for s in strokes}) == 1
            if common_low == common_high:
                continue
            sign = 1 if common_low else -1
            lengths = [len(s[1]) for s in strokes]
            reference = lengths[n // 2]
            if reference < 2:
                continue
            anchors = [s[1][0] if sign == 1 else s[1][-1] for s in strokes]
            ends = [(r + sign * (reference - 1),
                     c + sign * dc * (reference - 1)) for r, c in anchors]
            if any(not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]) for r, c in ends):
                continue
            candidates.append({"background": background, "axis": axis, "sign": sign,
                               "anchors": anchors, "colors": [s[2] for s in strokes],
                               "lengths": lengths, "reference": reference,
                               "keys": keys, "gaps": gaps, "count": n})
    if len(candidates) != 1:
        raise ValueError("strokes must determine exactly one axis and common anchor side")
    return candidates[0]


def _normalize(g, a, reference=None):
    span = a["reference"] if reference is None else int(reference)
    result = np.full(g.shape, a["background"], dtype=np.int8)
    dr, dc = _AXES[a["axis"]]
    dr, dc = dr * a["sign"], dc * a["sign"]
    for (r, c), color in zip(a["anchors"], a["colors"]):
        for t in range(span):
            rr, cc = r + t * dr, c + t * dc
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                raise ValueError("normalized stroke leaves the canvas")
            result[rr, cc] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    return _dedup(_normalize(g, _analyze(g)))


def _geometric_index(a):
    distances = [abs(2 * k - a["keys"][0] - a["keys"][-1]) for k in a["keys"]]
    indices = [i for i, d in enumerate(distances) if d == min(distances)]
    return indices[0] if len(indices) == 1 else None


def _features(g, a=None, normalized=None):
    a = _analyze(g) if a is None else a
    b = _normalize(g, a) if normalized is None else normalized
    before, after = _columns(g), _columns(b)
    bg = (a["background"],) * g.shape[0]
    new_equal, nonadjacent, split_equal = False, False, False
    for i in range(g.shape[1]):
        for j in range(i + 1, g.shape[1]):
            if after[i] == after[j] and after[i] != bg:
                new_equal |= before[i] != before[j]
                nonadjacent |= j > i + 1
            split_equal |= before[i] == before[j] and after[i] != after[j]
    L = a["reference"]
    return {"new_equal": bool(new_equal), "nonadjacent": bool(nonadjacent),
            "split_equal": bool(split_equal),
            "extends": sum(v < L for v in a["lengths"]),
            "trims": sum(v > L for v in a["lengths"]),
            "nonmedian": sorted(a["lengths"])[a["count"] // 2] != L,
            "unique_columns": len(set(after))}


def _sample_lengths(rng, n, L, cap, witness):
    lengths = [L] * n
    others = [int(i) for i in rng.permutation(n) if i != n // 2]
    short = lambda: int(rng.integers(1, L))
    long = lambda: int(rng.integers(L + 1, cap + 1))
    if witness:
        mostly_short = bool(rng.integers(2))
        for i in others:
            lengths[i] = short() if mostly_short else long()
        lengths[others[0]] = long() if mostly_short else short()
    else:
        mode = str(rng.choice(("mixed", "extend", "trim", "one_extend", "one_trim")))
        if mode.startswith("one_"):
            lengths[others[0]] = short() if mode == "one_extend" else long()
        else:
            for i in others:
                if rng.random() < .20:
                    continue
                lengths[i] = short() if mode == "extend" or (mode == "mixed" and rng.random() < .5) else long()
            if mode == "mixed":
                lengths[others[0]], lengths[others[1]] = short(), long()
            elif lengths == [L] * n:
                lengths[others[0]] = short() if mode == "extend" else long()
    return lengths


def _vertical_gaps(rng, n, witness):
    gaps = [2] * (n - 1)
    if witness:
        extra = int(rng.choice((5, 7, 9)))
        gaps[0 if rng.integers(2) else -1] += extra
    elif rng.random() < .35:
        gap = int(rng.integers(2, min(5, 27 // (n - 1)) + 1))
        gaps = [gap] * (n - 1)
    else:
        budget = int(rng.integers(0, min(14, 27 - sum(gaps)) + 1))
        for _ in range(budget):
            gaps[int(rng.integers(n - 1))] += 1
    return gaps


def _diagonal_profiles(forbidden, training):
    profiles = []
    for n in (3, 5, 7):
        for gap in range(2, 6):
            span = (n - 1) * gap
            # These dense diagonal layouts have deliberate non-background
            # column coincidences; random lengths then disrupt them in input.
            for L in range(span + 1, span + 2 * gap):
                if L <= (12 if training else 24) and span + L <= 28 and L not in forbidden:
                    profiles.append((n, gap, L))
    return profiles


def _draw(rng, axis, sign, gaps, lengths, colors, L, background):
    dr, dc = _AXES[axis]
    positions = [0] + list(map(int, np.cumsum(gaps)))
    # Perpendicular baseline, then a directed stroke from each fixed anchor.
    bases = [(p * dc, -p * dr) for p in positions]
    direction = sign * dr, sign * dc
    union = [(r + t * direction[0], c + t * direction[1])
             for (r, c), length in zip(bases, lengths) for t in range(max(length, L))]
    low = [min(q[k] for q in union) for k in (0, 1)]
    box = [max(q[k] for q in union) - low[k] + 1 for k in (0, 1)]
    # Always include at least one full background column, at a variable side.
    candidates = [(h, w) for h in range(box[0], 31) for w in range(box[1] + 1, 31)
                  if h * w > 2 * max(sum(lengths), len(lengths) * L)]
    if not candidates:
        raise ValueError("sampled geometry cannot fit with a clear background")
    if rng.random() < .60:
        compact = [s for s in candidates if s[0] <= box[0] + 5 and s[1] <= box[1] + 5]
        candidates = compact or candidates
    shape = candidates[int(rng.integers(len(candidates)))]
    slack = [shape[k] - box[k] for k in (0, 1)]
    offsets = [int(rng.choice((0, x))) if rng.random() < .45 else int(rng.integers(x + 1)) for x in slack]
    anchors = [(r - low[0] + offsets[0], c - low[1] + offsets[1]) for r, c in bases]
    inp = np.full(shape, background, dtype=np.int8)
    norm = inp.copy()
    for (r, c), length, color in zip(anchors, lengths, colors):
        for grid, extent in ((inp, length), (norm, L)):
            for t in range(extent):
                rr, cc = r + t * direction[0], c + t * direction[1]
                _require(grid[rr, cc] == background, "generation overlap")
                grid[rr, cc] = color
    return inp, norm


def _make_pair(rng, axis, training, forbidden_refs, witness, seen, forbidden_bg=None):
    for _attempt in range(160):
        if axis == "V":
            n = int(rng.choice((5, 7) if witness else (3, 5, 7) if training else (3, 5, 7, 9, 11)))
            choices = [L for L in range(3 if witness else 2, 13 if training else 25) if L not in forbidden_refs]
            L = int(rng.choice(choices))
            gaps = _vertical_gaps(rng, n, witness)
            cap = int(rng.integers(max(L + 1, 8), 30))
        else:
            profiles = _diagonal_profiles(forbidden_refs, training)
            n, gap, L = profiles[int(rng.integers(len(profiles)))]
            gaps = [gap] * (n - 1)
            cap = 29 - sum(gaps)
        lengths = _sample_lengths(rng, n, L, cap, witness)
        background = int(rng.choice([c for c in range(10) if c != forbidden_bg]))
        palette = [int(c) for c in rng.permutation([c for c in range(10) if c != background])]
        if axis == "V":
            size = 3 if witness else int(rng.integers(1, min(6, n - 1) + 1))
            colors = [int(c) for c in rng.choice(palette[:size], n)]
            if witness:
                colors[0], colors[1], colors[2], colors[-2], colors[-1] = palette[0], palette[1], palette[0], palette[2], palette[1]
                # The reference's visible color has a noncentral counterpart.
                if colors.count(colors[n // 2]) == 1:
                    colors[0] = colors[n // 2]
        else:
            size = 1 if rng.random() < .45 else int(rng.integers(2, min(4, n) + 1))
            half = [int(c) for c in rng.choice(palette[:size], n // 2)]
            colors = half + [int(rng.choice(palette[:size]))] + half[::-1]
        sign = int(rng.choice((-1, 1)))
        try:
            inp, norm = _draw(rng, axis, sign, gaps, lengths, colors, L, background)
            a = _analyze(inp)
        except ValueError:
            continue
        f = _features(inp, a, norm)
        if not (f["new_equal"] and f["nonadjacent"]):
            continue
        out = _dedup(norm)
        if _key(inp) in seen or np.array_equal(_dedup(inp), out):
            continue
        if witness:
            geometric = _geometric_index(a)
            if (not f["nonmedian"] or geometric is None or geometric == n // 2
                    or a["lengths"][geometric] == L
                    or np.array_equal(norm[:, :out.shape[1]], out)):
                continue
        # Verify drawing and input interpretation agree before returning a label.
        _require(np.array_equal(_normalize(inp, a), norm), "draw/parser disagreement")
        seen.add(_key(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("bounded sampling exhausted while making an informative pair")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    diagonal = str(rng.choice(("D+", "D-")))
    seen = set()
    train = []
    # Even two demonstrations contain a positional-reference witness and a
    # diagonal whole-column witness. Later pairs freely recombine these axes.
    for i in range(num_train):
        axis = "V" if i == 0 else diagonal if i == 1 else str(rng.choice(("V", diagonal)))
        forbidden = {_analyze(train[0]["input"])["reference"]} if i == 1 else set()
        bg = _analyze(train[0]["input"])["background"] if i == 1 else None
        train.append(_make_pair(rng, axis, True, forbidden, i == 0, seen, bg))
    training_refs = {_analyze(pair["input"])["reference"] for pair in train}
    test = []
    for i in range(num_test):
        axis = str(rng.choice(("V", diagonal)))
        forbidden = training_refs | ({_analyze(test[0]["input"])["reference"]} if i == 1 else set())
        test.append(_make_pair(rng, axis, False, forbidden, False, seen))
    rng.shuffle(train)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible geometry, causal witnesses, labels, split shift and schema."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            norm = _normalize(inp, a)
            f = _features(inp, a, norm)
            after, target = _columns(norm), _columns(out)
            _require(target == list(dict.fromkeys(after)), "wrong first-occurrence column sequence")
            _require(out.shape[0] == inp.shape[0] and out.shape[1] < inp.shape[1], "wrong output dimensions")
            _require(len(set(target)) == len(target), "duplicate output columns")
            _require(f["new_equal"] and f["nonadjacent"], "missing causal/global dedup witness")
            _require(not np.array_equal(norm, out), "omitting A makes no difference")
            _require(not np.array_equal(_dedup(inp), out), "omitting B makes no difference")
            _require(set(np.unique(inp)) == set(np.unique(out)), "palette changed")
            # Confirm every retained prefix, anchor and the reference survive B;
            # only tails can be painted or erased before columns are compacted.
            dr, dc = _AXES[a["axis"]]
            for (r, c), old_length, color in zip(a["anchors"], a["lengths"], a["colors"]):
                for t in range(max(old_length, a["reference"])):
                    rr, cc = r + t * dr * a["sign"], c + t * dc * a["sign"]
                    _require(norm[rr, cc] == (color if t < a["reference"] else a["background"]), "tail edit error")
                    if t < min(old_length, a["reference"]):
                        _require(inp[rr, cc] == norm[rr, cc], "anchor/prefix moved")
            records[split].append((a, f, norm, out))
    train = records["train"]
    axes = {a["axis"] for a, _, _, _ in train}
    _require("V" in axes and bool(axes & {"D+", "D-"}), "training needs vertical and diagonal evidence")
    refs = {a["reference"] for a, _, _, _ in train}
    _require(len(refs) >= 2, "reference length must vary in training")
    _require(len({a["background"] for a, _, _, _ in train}) >= 2, "background role must vary")
    _require(any(_columns(out) != list(dict.fromkeys(_columns(norm)[::-1]))[::-1]
                 for _, _, norm, out in train), "first and last occurrence are indistinguishable")
    _require(any(a["axis"] != "V" and out.shape[1] > len(set(a["colors"])) + 1
                 for a, _, _, out in train), "missing whole-column versus color witness")
    _require(any(f["extends"] and f["trims"] and f["nonmedian"]
                 and (g := _geometric_index(a)) is not None and g != a["count"] // 2
                 and a["lengths"][g] != a["reference"]
                 and not np.array_equal(norm[:, :out.shape[1]], out)
                 for a, f, norm, out in train), "missing positional/crop counterexample")
    for a, _, _, _ in records["test"]:
        _require(a["axis"] in axes, "untrained orientation class")
        _require(a["reference"] not in refs, "test needs an unseen reference span")


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, streaming 42-seed checks; return descriptive compact metrics."""
    rng = np.random.default_rng(39481)
    fingerprints, shapes, structures, palettes = set(), set(), set(), set()
    directions, counts, refs, corrections, combinations = set(), set(), set(), set(), set()
    new_equal = split_equal = checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fp = _fingerprint(puzzle)
        _require(fp == _fingerprint(generate_puzzle(seed)), "nondeterministic generation")
        fingerprints.add(fp)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, out = pair["input"], pair["output"]
                a = _analyze(g)
                norm = _normalize(g, a)
                f = _features(g, a, norm)
                _require(np.array_equal(solve(g), out), "solve disagrees")
                directions.add((a["axis"], a["sign"]))
                counts.add(a["count"])
                refs.add(a["reference"])
                shapes.add(g.shape)
                palettes.add(tuple(map(int, np.unique(g))))
                mode = "mixed" if f["extends"] and f["trims"] else "extend" if f["extends"] else "trim"
                corrections.add(mode)
                combinations.add((a["axis"], a["count"], a["reference"], tuple(a["gaps"]), tuple(a["lengths"])))
                roles = {}
                canonical = bytes(roles.setdefault(int(c), len(roles)) for c in g.flat)
                structures.add((g.shape, hashlib.sha256(canonical).digest()))
                new_equal += f["new_equal"]
                split_equal += f["split_equal"]
                checked += 1
                permutation = rng.permutation(10).astype(np.int8)
                _require(np.array_equal(solve(permutation[g]), permutation[out]), "palette dependence")
                _require(np.array_equal(solve(g[::-1]), out[::-1]), "vertical reflection error")
                # The actual central span must affect the final compressed grid.
                # Shortening it leaves the anchors valid for every family member.
                if a["reference"] > 2:
                    altered = g.copy()
                    r, c = a["anchors"][a["count"] // 2]
                    dr, dc = _AXES[a["axis"]]
                    t = a["reference"] - 1
                    altered[r + t * dr * a["sign"], c + t * dc * a["sign"]] = a["background"]
                    try:
                        changed = solve(altered)
                    except ValueError:
                        # All other strokes can coincide with L-1 in one-edit
                        # examples; then anchor-side ambiguity is outside family.
                        _require(all(v == a["reference"] - 1 for i, v in enumerate(a["lengths"]) if i != a["count"] // 2), "unexpected perturbed-input ambiguity")
                    else:
                        _require(not np.array_equal(changed, out), "middle length has no effect")
    _require(len(fingerprints) == 42 and len(structures) >= 190, "insufficient structural variation")
    _require(len(shapes) >= 60 and len(palettes) >= 60, "insufficient shape/palette variation")
    _require(len(directions) == 6 and {3, 5, 7, 9, 11} <= counts, "insufficient geometry coverage")
    _require(len(refs) >= 15 and corrections == {"mixed", "extend", "trim"}, "insufficient length variation")
    _require(new_equal == checked and split_equal > 0, "causal equality variation missing")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(709, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (3.5, 1), (True, 1), (3, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Test validators on a label that violates exact full-column contents.
    broken = generate_puzzle(102, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged label")
    return {"seeds": 42, "pairs": checked, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "color_normalized_structures": len(structures),
            "palettes": len(palettes), "directed_axes": len(directions),
            "stroke_counts": sorted(counts), "references": sorted(refs),
            "correction_modes": sorted(corrections), "structural_combinations": len(combinations),
            "normalization_creates_equal_columns": new_equal,
            "normalization_splits_equal_columns": split_equal}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
