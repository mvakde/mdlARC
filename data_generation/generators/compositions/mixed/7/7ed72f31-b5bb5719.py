"""Marker-completed seed cascades. Standard library and NumPy only.

The upper row is a donor, the middle row selects reflection/half-turn, and
the lower input row is an unfinished seed. No training context is used by solve.
Nothing is written to disk by this module, including during self_test.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _read_input(value):
    """Read only visible geometry and colors; reject ambiguous conventions."""
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("The background must be the unique most frequent color")
    bg = int(colors[counts.argmax()])
    rows = np.flatnonzero(np.any(a != bg, axis=1))
    if len(rows) != 3 or not np.all(np.diff(rows) == 1):
        raise ValueError("Expected consecutive donor, marker, and seed-fragment rows")
    top, axis, row = map(int, rows)
    marker_cols = np.flatnonzero(a[axis] != bg)
    marker = int(a[axis, marker_cols[0]])
    if (not np.all(a[axis, marker_cols] == marker)
            or not np.all(np.diff(marker_cols) == 1)):
        raise ValueError("The marker is one contiguous horizontal line or singleton")
    kind = "P" if len(marker_cols) == 1 else "H"
    source = np.flatnonzero(a[top] != bg)
    receiver = np.flatnonzero(a[row] != bg)
    signals = tuple(sorted(set(map(int, np.unique(a[[top, row]]))) - {bg}))
    if len(signals) != 2 or marker in signals:
        raise ValueError("Exactly two signal colors must be visible, separate from marker/background")
    s0, s1 = int(source[0]), int(source[-1])
    b0, b1 = int(receiver[0]), int(receiver[-1])
    if s1 - s0 + 1 < 2 or b1 - b0 + 1 < 2:
        raise ValueError("Both fragments must have signal endpoints and at least two positions")
    if kind == "H":
        dest_cols = source.copy()
        d0, d1 = s0, s1
        if not set(map(int, source)).intersection(map(int, marker_cols)):
            raise ValueError("A line marker must overlap the donor in column")
    else:
        center = int(marker_cols[0])
        if not (s1 < center or s0 > center):
            raise ValueError("Point donor must lie strictly to one side of its center")
        dest_cols = 2 * center - source
        d0, d1 = 2 * center - s1, 2 * center - s0
    if d0 < 0 or d1 >= a.shape[1] or not (d1 < b0 or b1 < d0):
        raise ValueError("The completed fragment intervals must be disjoint and in bounds")
    if kind == "P":
        gap0, gap1 = (d1 + 1, b0 - 1) if d1 < b0 else (b1 + 1, d0 - 1)
        if not gap0 <= center <= gap1:
            raise ValueError("The point center must project into the gap between seed fragments")
    start, stop = min(d0, b0), max(d1, b1) + 1
    length = stop - start
    if length < 5 or length % 2 != 1:
        raise ValueError("The completed seed must have odd span, at least five positions")
    if row + length // 2 >= a.shape[0]:
        raise ValueError("The canvas must fit the final one-position row")
    completed = a[row].copy()
    completed[dest_cols] = a[top, source]
    seq = completed[start:stop].copy()
    return {"input": a, "background": bg, "marker": marker, "signals": signals,
            "kind": kind, "top": top, "axis": axis, "row": row,
            "source": source, "dest": dest_cols, "donor_start": s0,
            "donor_stop": s1 + 1, "receiver_start": b0, "receiver_stop": b1 + 1,
            "start": start, "length": length, "sequence": seq}


def _swap(sequence, signals):
    x, y = signals
    return np.where(sequence == x, y, np.where(sequence == y, x, sequence)).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete the lower seed first; then recursively drop two, swap, and inset."""
    p = _read_input(input_grid)
    out = p["input"].copy()
    current = p["sequence"].copy()
    row, col = p["row"], p["start"]
    out[row, col:col + len(current)] = current
    while len(current) > 1:
        current = _swap(current[:-2], p["signals"])
        row += 1
        col += 1
        out[row, col:col + len(current)] = current
    return out


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _fragment(rng, size, style, bg, signals):
    x, y = signals
    if style == "solid":
        result = np.full(size, int(rng.choice(signals)), dtype=np.int8)
    elif style == "alternating":
        phase = int(rng.integers(2))
        result = np.array([signals[(i + phase) % 2] for i in range(size)], dtype=np.int8)
    elif style == "runs":
        result = np.empty(size, dtype=np.int8)
        pos, phase = 0, int(rng.integers(2))
        while pos < size:
            stop = min(size, pos + int(rng.integers(1, 5)))
            result[pos:stop] = signals[phase]
            pos, phase = stop, 1 - phase
    else:
        gap = .48 if style == "sparse" else .22
        result = rng.choice([bg, x, y], size=size,
                            p=[gap, (1 - gap) / 2, (1 - gap) / 2]).astype(np.int8)
    if result[0] == bg:
        result[0] = int(rng.choice(signals))
    if result[-1] == bg:
        result[-1] = int(rng.choice(signals))
    return result


def _candidate(rng, palette, kind, length, diagnostic=False):
    bg, marker, x, y = palette
    signals = (x, y)
    min_part = 4 if diagnostic else 2
    gap_min = 1 if kind == "P" or diagnostic else 0
    gaps = [g for g in range(gap_min, 4) if length - g >= max(3, min_part) + min_part]
    gap = int(rng.choice(gaps))
    donor_size = int(rng.integers(max(3, min_part), min(11, length - gap - min_part) + 1))
    receiver_size = length - gap - donor_size
    donor_left = bool(rng.integers(2))
    if donor_left:
        d0, d1 = 0, donor_size
        b0, b1 = donor_size + gap, length
    else:
        b0, b1 = 0, receiver_size
        d0, d1 = receiver_size + gap, length
    styles = ("solid", "alternating", "runs", "dense", "sparse")
    donor_style, receiver_style = str(rng.choice(styles)), str(rng.choice(styles))
    seq = np.full(length, bg, dtype=np.int8)
    seq[d0:d1] = _fragment(rng, donor_size, donor_style, bg, signals)
    seq[b0:b1] = _fragment(rng, receiver_size, receiver_style, bg, signals)
    if diagnostic:
        # Two distinct retained phases, a retained left gap, and a trailing gap
        # distinguish prefix choice, color exchange, and fixed positional spans.
        first = int(rng.choice(signals))
        other = y if first == x else x
        seq[0], seq[1], seq[2], seq[-3] = first, bg, other, bg
        # A visibly asymmetric donor exposes point reversal even with only two demos.
        seq[d1 - 1] = y if seq[d0] == x else x
    if not {x, y} <= set(map(int, seq)):
        # The other phase is visibly encoded, never supplied by hidden puzzle state.
        seq[b0] = y if seq[d0] == x else x
    if kind == "H":
        source0, source1 = d0, d1
        donor = seq[d0:d1].copy()
        contacts = set(source0 + int(i) for i in np.flatnonzero(donor != bg))
        lines = [(a, b) for a in range(source0 - 2, source1 + 1)
                 for b in range(a + 2, source1 + 3)
                 if b - a <= donor_size + 2 and any(a <= c < b for c in contacts)]
        marker0, marker1 = lines[int(rng.integers(len(lines)))]
    else:
        gap0 = donor_size if donor_left else receiver_size
        center = gap0 + int(rng.integers(gap))
        source0, source1 = 2 * center - (d1 - 1), 2 * center - d0 + 1
        donor = seq[d0:d1][::-1].copy()
        marker0, marker1 = center, center + 1
    low, high = min(0, source0, marker0), max(length, source1, marker1)
    base_width = high - low
    if base_width > 30:
        raise AssertionError("Construction exceeded its bounded footprint")
    spare = min(8, 30 - base_width)
    extra = int(rng.integers(spare + 1))
    left_margin = int(rng.integers(extra + 1))
    offset = left_margin - low
    top = int(rng.integers(4))
    seed_row = top + 2
    height = seed_row + (length + 1) // 2 + int(rng.integers(5))
    inp = np.full((height, base_width + extra), bg, dtype=np.int8)
    inp[top, source0 + offset:source1 + offset] = donor
    inp[top + 1, marker0 + offset:marker1 + offset] = marker
    inp[seed_row, b0 + offset:b1 + offset] = seq[b0:b1]
    # Construction uses a closed form from the recipe; solve uses input parsing
    # and recurrence. No solver call creates the labels.
    out = inp.copy()
    for depth in range((length + 1) // 2):
        prefix = seq[:length - 2 * depth]
        values = _swap(prefix, signals) if depth % 2 else prefix
        out[seed_row + depth, offset + depth:offset + depth + len(values)] = values
    return {"input": inp, "output": out}


def _key(a):
    return (a.shape, a.tobytes())


def _diagnostic(p):
    seq, bg = p["sequence"], p["background"]
    donor = p["input"][p["top"], p["donor_start"]:p["donor_stop"]]
    return (len(seq) >= 7 and bg in seq[1:-2] and seq[-3] == bg
            and not np.array_equal(seq[:-2], seq[1:-1])
            and not np.array_equal(seq[:-2], seq[2:])
            and not np.array_equal(donor, donor[::-1]))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.permutation(10)[:4]))
    first_lengths = list(map(int, rng.choice(np.arange(11, 22, 2), size=2, replace=False)))
    jobs = [("H", first_lengths[0], True), ("P", first_lengths[1], True)]
    jobs += [(str(rng.choice(["H", "P"])), int(rng.choice(np.arange(7, 24, 2))), False)
             for _ in range(nt - 2)]
    rng.shuffle(jobs)
    max_train = max(length for _, length, _ in jobs)
    test_jobs = [(str(rng.choice(["H", "P"])), max_train + 2, False)]
    test_jobs += [(str(rng.choice(["H", "P"])), int(rng.choice(np.arange(7, 26, 2))), False)
                  for _ in range(ns - 1)]
    puzzle = {"train": [], "test": []}
    inputs, motifs = set(), set()
    for split, plans in (("train", jobs), ("test", test_jobs)):
        for kind, length, diagnostic in plans:
            for _ in range(64):
                pair = _candidate(rng, palette, kind, length, diagnostic)
                p = _read_input(pair["input"])
                key = _key(pair["input"])
                motif = tuple(map(int, p["sequence"]))
                if key not in inputs and motif not in motifs and (not diagnostic or _diagnostic(p)):
                    break
            else:
                raise RuntimeError("Bounded unique motif construction exhausted")
            puzzle[split].append(pair)
            inputs.add(key)
            motifs.add(motif)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    p = _read_input(inp)
    bg, signals = p["background"], p["signals"]
    # Independently read the two fragments in coordinate order. np.flip performs
    # the point reversal, rather than relying on the solver's cell mappings.
    raw = inp[p["top"], p["donor_start"]:p["donor_stop"]]
    donor = raw if p["kind"] == "H" else np.flip(raw)
    d0 = int(min(p["dest"]))
    complete = inp[p["row"]].copy()
    complete[d0:d0 + len(donor)] = donor
    seq = complete[p["start"]:p["start"] + p["length"]]
    assert np.array_equal(seq, p["sequence"])
    assert seq[0] in signals and seq[-1] in signals
    expected = inp.copy()
    footprint = np.zeros(inp.shape, dtype=bool)
    for depth in range((len(seq) + 1) // 2):
        rr, cc = p["row"] + depth, p["start"] + depth
        for index in range(len(seq) - 2 * depth):
            v = int(seq[index])
            if v != bg and depth % 2:
                v = signals[1] if v == signals[0] else signals[0]
            expected[rr, cc + index] = v
            footprint[rr, cc + index] = True
    assert np.array_equal(out, expected), "Completion/cascade cell mismatch"
    assert np.array_equal(out[~footprint], inp[~footprint]), "Outside cells changed"
    assert np.array_equal(out[inp != bg], inp[inp != bg]), "Existing signals/markers changed"
    assert not np.array_equal(inp, out)
    # Necessary causal link: a copied signal survives the first right-end cut.
    copied = p["dest"] - p["start"]
    assert np.any(copied < len(seq) - 2), "Symmetry would have no downstream influence"
    assert out[p["row"] + len(seq) // 2, p["start"] + len(seq) // 2] in signals
    # Neither parent alone supplies the answer; no-swap also changes a retained signal.
    only_symmetry = inp.copy()
    only_symmetry[p["row"]] = complete
    assert not np.array_equal(out, only_symmetry)
    first = out[p["row"] + 1, p["start"] + 1:p["start"] + len(seq) - 1]
    assert not np.array_equal(first, seq[:-2])
    without_a = inp[p["row"], p["start"]:p["start"] + len(seq)]
    assert not np.array_equal(first, _swap(without_a[:-2], signals))
    return p


def validate_generated_puzzle(puzzle):
    """Check labels, geometry, causal witnesses, coverage, diversity, and shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    inputs, motifs, palettes = set(), set(), set()
    props = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            p = _check_pair(pair)
            key, motif = _key(p["input"]), tuple(map(int, p["sequence"]))
            assert key not in inputs and motif not in motifs, "Repeated input or completed motif"
            inputs.add(key)
            motifs.add(motif)
            palettes.add((p["background"], p["marker"], p["signals"]))
            props[split].append(p)
    assert len(palettes) == 1, "Puzzle conventions must agree"
    assert {p["kind"] for p in props["train"] if _diagnostic(p)} == {"H", "P"}
    assert len({p["length"] for p in props["train"]}) >= 2
    assert max(p["length"] for p in props["test"]) > max(p["length"] for p in props["train"])
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _fixed_regression():
    # Hand-worked point example: upper 1201 rotates to the missing 1021.
    inp = np.array([[0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                    [0, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0],
                    *([[0] * 11] * 4)], dtype=np.int8)
    expected = inp.copy()
    expected[2] = [0, 2, 0, 1, 2, 0, 1, 0, 2, 1, 0]
    expected[3] = [0, 0, 1, 0, 2, 1, 0, 2, 0, 0, 0]
    expected[4] = [0, 0, 0, 2, 0, 1, 2, 0, 0, 0, 0]
    expected[5] = [0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0]
    expected[6] = [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
    assert np.array_equal(solve(inp), expected)
    # A line reflects to the same columns; its finite length need not be a ruler.
    line = inp.copy()
    line[1] = [0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    line[2] = [0, 0, 0, 0, 0, 0, 2, 0, 1, 2, 0]
    answer = solve(line)
    assert np.array_equal(answer[2], [0, 1, 2, 0, 1, 0, 2, 0, 1, 2, 0])
    assert np.array_equal(answer[3], [0, 0, 2, 1, 0, 2, 0, 1, 0, 0, 0])


def self_test():
    """42 streamed puzzles, all 30 legal counts, semantic and rejection checks."""
    _fixed_regression()
    signatures, shapes, palettes, motifs, lengths = set(), set(), set(), set(), set()
    kinds, sides, gaps, marker_lengths = set(), set(), set(), set()
    count_pairs = set()
    pairs = retained_gap_pairs = asymmetric_points = boundary_pairs = 0
    for seed in range(42):
        nt = 2 + seed % 9
        ns = 1 + (seed // 3) % min(4, nt - 1)
        puzzle = generate_puzzle(seed, nt, ns)
        signatures.add(_digest(puzzle))
        count_pairs.add((nt, ns))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _check_pair(pair)
                original = pair["input"].copy()
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert np.array_equal(pair["input"], original), "solve mutated its input"
                seq, bg = p["sequence"], p["background"]
                shapes.add(pair["input"].shape)
                palettes.add((bg, p["marker"], p["signals"]))
                roles = {bg: 0, p["signals"][0]: 1, p["signals"][1]: 2}
                motifs.add(tuple(roles[int(v)] for v in seq))
                lengths.add(len(seq))
                kinds.add(p["kind"])
                sides.add((p["kind"], int(min(p["dest"])) < p["receiver_start"]))
                marker_lengths.add(int(np.count_nonzero(original[p["axis"]] == p["marker"])))
                gaps.add(int(np.count_nonzero(seq == bg)))
                retained_gap_pairs += bg in seq[1:-2]
                if p["kind"] == "P":
                    donor = original[p["top"], p["donor_start"]:p["donor_stop"]]
                    asymmetric_points += not np.array_equal(donor, donor[::-1])
                boundary_pairs += (p["start"] == 0 or p["start"] + len(seq) == original.shape[1]
                                   or p["row"] + len(seq) // 2 == original.shape[0] - 1)
                pairs += 1
    assert len(signatures) == 42 and len(palettes) >= 30
    assert len(shapes) >= 40 and len(motifs) >= 150 and len(lengths) >= 8
    assert kinds == {"H", "P"} and len(sides) == 4
    assert len(marker_lengths) >= 5 and len(gaps) >= 7
    assert retained_gap_pairs >= 84 and asymmetric_points >= 42 and boundary_pairs >= 10
    legal_count_checks = 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1234567, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
            legal_count_checks += 1
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(7, 2, 1)
    assert _digest(p) == _digest(generate_puzzle(7, 2, 1))
    # Corrupt a retained phase, a seed-completion cell, and an outside cell.
    for region in ("cascade", "completion", "outside"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        q = _read_input(bad["train"][0]["input"])
        if region == "cascade":
            cell = (q["row"] + 1, q["start"] + 1)
        elif region == "completion":
            cell = (q["row"], int(q["dest"][0]))
        else:
            cell = (q["top"], int(q["source"][0]))
        out = bad["train"][0]["output"]
        out[cell] = (int(out[cell]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"Accepted corrupted {region}")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "palettes": len(palettes), "grid_shapes": len(shapes),
            "normalized_seed_motifs": len(motifs), "seed_lengths": sorted(lengths),
            "marker_lengths": sorted(marker_lengths), "gap_counts": sorted(gaps),
            "retained_gap_pairs": retained_gap_pairs, "asymmetric_point_pairs": asymmetric_points,
            "boundary_pairs": boundary_pairs, "legal_count_checks": legal_count_checks,
            "count_combinations_in_seed_sweep": len(count_pairs)}
