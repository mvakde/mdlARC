"""Restore a mirrored seed hem, then grow its shrinking, phase-swapped prefixes.

Only the standard library and NumPy are used. No files, examples, random state,
or generation parameters are consulted by solve(). See task_spec.md.
"""
from hashlib import sha256

import numpy as np


ERASE = 9
PORT = 8
STYLES = ("mixed", "runs", "alternating", "sparse", "gapped_mono", "solid")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _orbits(h, w, rs, cs):
    seen = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rows = {r, rs - r} if 0 <= rs - r < h else {r}
            cols = {c, cs - c} if 0 <= cs - c < w else {c}
            orbit = tuple(sorted((y, x) for y in rows for x in cols))
            seen.update(orbit)
            yield orbit


def _axis_candidates(field, dimension):
    a = field if dimension == 0 else field.T
    candidates = []
    for total in range(1, 2 * a.shape[0] - 2):
        left = np.array([i for i in range(a.shape[0])
                         if i < total - i < a.shape[0]], dtype=int)
        first, second = a[left], a[total - left]
        visible = (first != ERASE) & (second != ERASE)
        if visible.any() and np.all(first[visible] == second[visible]):
            candidates.append(total)
    return candidates


def _axes(field):
    rr, cc = _axis_candidates(field, 0), _axis_candidates(field, 1)
    if len(rr) != 1 or len(cc) != 1:
        raise ValueError("surviving evidence must identify exactly two folds")
    return rr[0], cc[0]


def _restore(field, axes):
    clean = field.copy()
    for orbit in _orbits(*field.shape, *axes):
        colors = {int(field[p]) for p in orbit if field[p] != ERASE}
        if len(colors) != 1:
            raise ValueError("an orbit lacks a unanimous surviving witness")
        color = next(iter(colors))
        for p in orbit:
            clean[p] = color
    return clean


def _decode(value):
    inp = _grid(value)
    ports = np.argwhere(inp == PORT)
    if ports.shape != (1, 2) or ports[0, 1] != 0:
        raise ValueError("one left-edge port must mark the seed row")
    row = int(ports[0, 0])
    if row < 3 or row >= inp.shape[0] - 1 or inp.shape[1] < 4:
        raise ValueError("the port needs a source field above and canvas below")
    bg = int(inp[-1, 0])
    if bg in (ERASE, PORT):
        raise ValueError("reserved marker used as background")
    margin = inp[:, 0].copy()
    margin[row] = bg
    if np.any(margin != bg) or np.any(inp[row + 1:] != bg):
        raise ValueError("the left gutter and lower canvas must be background")
    field = inp[:row + 1, 1:]
    signals = tuple(sorted(set(int(x) for x in np.unique(field)) - {bg, ERASE}))
    if len(signals) != 2:
        raise ValueError("both signal roles must be visible somewhere in the field")
    axes = _axes(field)
    clean = _restore(field, axes)
    occupied = np.flatnonzero(clean[-1] != bg)
    if len(occupied) < 2:
        raise ValueError("the restored hem has no seed interval")
    start, stop = int(occupied[0]), int(occupied[-1]) + 1
    sequence = clean[-1, start:stop].copy()
    if len(sequence) < 3 or len(sequence) % 2 != 1:
        raise ValueError("the seed interval must have odd length at least three")
    if row + len(sequence) // 2 >= inp.shape[0]:
        raise ValueError("insufficient room for the one-cell tip")
    return inp, row, bg, signals, axes, clean, start + 1, sequence


def _swap(sequence, signals):
    a, b = signals
    return np.where(sequence == a, b, np.where(sequence == b, a, sequence)).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer both folds, repair their orbits, and consume the repaired hem."""
    inp, row, bg, signals, axes, clean, start, sequence = _decode(input_grid)
    answer = inp.copy()
    answer[:row + 1, 1:] = clean
    depth = 0
    while len(sequence) > 1:
        sequence = _swap(sequence[:-2], signals)
        depth += 1
        answer[row + depth, start + depth:start + depth + len(sequence)] = sequence
    return answer


def _sequence(rng, length, style, bg, signals):
    a, b = (signals[int(i)] for i in rng.permutation(2))
    half = (length + 1) // 2
    if style in ("solid", "gapped_mono"):
        left = np.full(half, a, dtype=np.int8)
    elif style == "alternating":
        left = np.array([a if i % 2 == 0 else b for i in range(half)], dtype=np.int8)
    elif style == "runs":
        stride = int(rng.integers(2, 5))
        left = np.array([a if (i // stride) % 2 == 0 else b for i in range(half)], dtype=np.int8)
    else:
        p = 0.43 if style == "sparse" else 0.17
        left = rng.choice([bg, a, b], size=half, p=[p, (1 - p) / 2, (1 - p) / 2]).astype(np.int8)
    left[0] = a
    seq = np.concatenate((left, left[-2::-1]))
    gap = None
    if style in ("diagnostic", "mixed", "sparse", "gapped_mono"):
        # An even-index gap eventually reaches a retained prefix's right edge.
        choices = [i for i in range(2, length - 4, 2) if i != length // 2]
        gap = int(rng.choice(choices))
        seq[gap] = seq[length - 1 - gap] = bg
        if rng.random() < 0.45:
            eligible = [i for i in range(1, half) if i not in (gap, length - 1 - gap)]
            if eligible:
                q = int(rng.choice(eligible))
                seq[q] = seq[length - 1 - q] = bg
    if style != "solid" and style != "gapped_mono":
        # Force the other phase without removing the required retained gap.
        eligible = [i for i in range(1, half) if seq[i] != bg]
        if not eligible:
            eligible = [1]
        q = int(rng.choice(eligible))
        seq[q] = seq[length - 1 - q] = b
    return seq, gap


def _mask(rng, clean, axes, seed_left, sequence, gap, hard_count, signals):
    h, w = clean.shape
    rs, cs = axes
    family = int(rng.integers(4))
    mask = np.zeros((h, w), dtype=bool)
    if family == 0:
        mask = rng.random((h, w)) < rng.uniform(0.025, 0.15)
    elif family == 1:
        for _ in range(int(rng.integers(1, 4))):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            rh, cw = int(rng.integers(1, 4)), int(rng.integers(1, 5))
            mask[y:y + rh, x:x + cw] = True
    elif family == 2:
        y, x = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(3, max(4, h * w // 7)))):
            mask[y, x] = True
            dy, dx = ((0, 1), (0, -1), (1, 0), (-1, 0))[int(rng.integers(4))]
            y, x = int(np.clip(y + dy, 0, h - 1)), int(np.clip(x + dx, 0, w - 1))
    else:
        y = int(rng.integers(h))
        x = int(rng.integers(w))
        mask[y, max(0, x - 2):min(w, x + 4)] = True
    for orbit in _orbits(h, w, rs, cs):
        if all(mask[p] for p in orbit):
            mask[orbit[int(rng.integers(len(orbit)))]] = False
    # Every pair loses a signal that is used below the seed row.
    signal_indices = [i for i in range(len(sequence) - 2)
                      if int(sequence[i]) in signals and i != len(sequence) // 2]
    si = int(rng.choice(signal_indices))
    special = [si]
    if gap is not None:
        special.append(gap)
    elif hard_count > 1:
        special.append(next(i for i in signal_indices
                            if i not in (si, len(sequence) - 1 - si)))
    for index, local in enumerate(special):
        c = seed_left + local
        rr, cc = rs - (h - 1), cs - c
        orbit = ((h - 1, c), (h - 1, cc), (rr, c), (rr, cc))
        for p in orbit:
            mask[p] = False
        # Mixed diagnostic: a hard signal, a simply erased retained gap.
        # Held-out tests couple hard repair to both a signal and a gap.
        erase_points = orbit[:3] if index < hard_count else orbit[:1]
        for p in erase_points:
            mask[p] = True
    return mask


def _make_pair(rng, length, style, bg, signals, hard_count):
    for _ in range(160):
        left = int(rng.integers(0, min(5, 29 - length) + 1))
        right = int(rng.integers(0, min(5, 29 - length - left) + 1))
        width = left + length + right
        height = int(rng.integers(5, 11))
        fringe = int(rng.integers(0, min(3, height - 4) + 1))
        axes = height - 1 + fringe, 2 * left + length - 1
        sequence, gap = _sequence(rng, length, style, bg, signals)
        clean = np.empty((height, width), dtype=np.int8)
        # Independent orbit texture makes the two folds visible above the hem.
        for orbit in _orbits(height, width, *axes):
            color = int(rng.choice([bg, *signals]))
            for p in orbit:
                clean[p] = color
        hem = np.full(width, bg, dtype=np.int8)
        hem[left:left + length] = sequence
        clean[-1] = hem
        clean[fringe] = hem
        if _axis_candidates(clean, 0) != [axes[0]] or _axis_candidates(clean, 1) != [axes[1]]:
            continue
        mask = _mask(rng, clean, axes, left, sequence, gap, hard_count, signals)
        field = clean.copy()
        field[mask] = ERASE
        if set(int(x) for x in np.unique(field)) - {bg, ERASE} != set(signals):
            continue
        if _axis_candidates(field, 0) != [axes[0]] or _axis_candidates(field, 1) != [axes[1]]:
            continue
        total_height = height + length // 2 + int(rng.integers(0, 4))
        inp = np.full((total_height, width + 1), bg, dtype=np.int8)
        inp[:height, 1:] = field
        inp[height - 1, 0] = PORT
        out = inp.copy()
        out[:height, 1:] = clean
        current = sequence.copy()
        for depth in range(1, length // 2 + 1):
            current = _swap(current[:-2], signals)
            start = 1 + left + depth
            out[height - 1 + depth, start:start + len(current)] = current
        return {"input": inp, "output": out}
    raise RuntimeError("bounded source-field sampling exhausted")


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _motif(sequence, bg, signals):
    roles = {bg: 0, signals[0]: 1, signals[1]: 2}
    seq = tuple(roles[int(x)] for x in sequence)
    complement = tuple(3 - x if x else 0 for x in seq)
    return min(seq, complement)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        seed %= 1 << 64
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.permutation(8)[:3]]
    bg, signals = palette[0], tuple(palette[1:])
    diagnostic_length = int(rng.choice(np.arange(7, 20, 2)))
    mono_length = int(rng.choice([n for n in range(7, 20, 2) if n != diagnostic_length]))
    train_jobs = [("diagnostic", diagnostic_length), ("solid", mono_length)]
    train_jobs += [(str(rng.choice(STYLES)), int(rng.choice(np.arange(7, 22, 2))))
                   for _ in range(nt - 2)]
    rng.shuffle(train_jobs)
    test_jobs = [(str(rng.choice(("mixed", "sparse", "gapped_mono"))),
                  int(rng.choice(np.arange(7, 26, 2))))]
    test_jobs += [(str(rng.choice(STYLES)), int(rng.choice(np.arange(7, 26, 2))))
                  for _ in range(ns - 1)]
    rng.shuffle(test_jobs)
    result = {"train": [], "test": []}
    inputs, motifs = set(), set()
    for split, jobs in (("train", train_jobs), ("test", test_jobs)):
        for style, length in jobs:
            for attempt in range(80):
                if attempt >= 8:
                    length = int(rng.choice(np.arange(7, 22 if split == "train" else 26, 2)))
                hard = 2 if split == "test" else 1
                pair = _make_pair(rng, length, style, bg, signals, hard)
                data = _decode(pair["input"])
                motif = _motif(data[-1], bg, signals)
                key = _key(pair["input"])
                if key not in inputs and motif not in motifs:
                    inputs.add(key)
                    motifs.add(motif)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-pair sampling exhausted")
    validate_generated_puzzle(result)
    return result


def _features(data):
    inp, row, bg, signals, axes, clean, start, seq = data
    field = inp[:row + 1, 1:]
    rs, cs = axes
    hard_signal = hard_gap = erased_signal = erased_gap = 0
    for i in range(len(seq) - 2):
        c = start - 1 + i
        if field[row, c] != ERASE:
            continue
        is_signal = int(seq[i]) in signals
        erased_signal += is_signal
        erased_gap += not is_signal
        rr, cc = rs - row, cs - c
        hard = (0 <= rr < field.shape[0] and 0 <= cc < field.shape[1]
                and rr != row and cc != c and field[row, cc] == ERASE
                and field[rr, c] == ERASE and field[rr, cc] != ERASE)
        hard_signal += bool(hard and is_signal)
        hard_gap += bool(hard and not is_signal)
    return {"erased_signal": int(erased_signal), "erased_gap": int(erased_gap),
            "hard_signal": int(hard_signal), "hard_gap": int(hard_gap),
            "gaps": int(np.count_nonzero(seq == bg)),
            "mixed": set(int(x) for x in seq) - {bg} == set(signals),
            "solid": np.all(seq == seq[0]),
            "crop_differs": not np.array_equal(seq[:-2], seq[1:-1])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, motifs, palettes, train_lengths = set(), set(), set(), set()
    diagnostic = solid = test_joint = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            data = _decode(pair["input"])
            inp, row, bg, signals, axes, clean, start, sequence = data
            out = _grid(pair["output"])
            assert inp.shape == out.shape and not np.any(out == ERASE)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            motif = _motif(sequence, bg, signals)
            assert motif not in motifs, "seed motif repeated up to color phase exchange"
            motifs.add(motif)
            palettes.add((bg, signals))
            field = inp[:row + 1, 1:]
            # Independently check every erasure via the other three orbit members.
            rs, cs = axes
            for r, c in zip(*np.nonzero(field == ERASE)):
                partners = ((rs - r, c), (r, cs - c), (rs - r, cs - c))
                witnesses = {int(field[y, x]) for y, x in partners
                             if 0 <= y <= row and 0 <= x < field.shape[1]
                             and field[y, x] != ERASE}
                assert witnesses == {int(clean[r, c])}
            assert np.array_equal(out[:row + 1, 1:], clean)
            assert np.array_equal(field[field != ERASE], clean[field != ERASE])
            # Closed form from the original seed checks the recursive constructor.
            expected = inp.copy()
            expected[:row + 1, 1:] = clean
            written = np.zeros(inp.shape, dtype=bool)
            written[:row + 1, 1:] = field == ERASE
            for depth in range(1, len(sequence) // 2 + 1):
                prefix = sequence[:len(sequence) - 2 * depth]
                values = _swap(prefix, signals) if depth % 2 else prefix
                expected[row + depth, start + depth:start + depth + len(values)] = values
                written[row + depth, start + depth:start + depth + len(values)] = True
            assert np.array_equal(out, expected), "wrong repair, prefix, phase, placement, or exterior"
            assert np.array_equal(out[~written], inp[~written])
            assert np.array_equal(solve(inp), out)
            f = _features(data)
            assert f["erased_signal"] and f["hard_signal"], "reflection must feed a retained signal"
            # Even granting correct interval bounds, replacing missing seed signals
            # with background changes a lower row: A is causally necessary.
            altered = sequence.copy()
            lost = inp[row, start:start + len(sequence)] == ERASE
            altered[lost] = bg
            assert not np.array_equal(_swap(altered[:-2], signals), _swap(sequence[:-2], signals))
            # Restoring alone omits a non-background first cell on every lower row.
            assert out[row + 1, start + 1] != bg
            if split == "train":
                train_lengths.add(len(sequence))
                diagnostic |= bool(f["mixed"] and f["erased_gap"] and f["crop_differs"])
                solid |= bool(f["solid"])
            else:
                test_joint |= bool(f["hard_gap"] and f["hard_signal"])
    assert len(palettes) == 1, "background and signal roles change within a puzzle"
    assert len(train_lengths) >= 2, "training must expose multiple recurrence depths"
    assert diagnostic and solid, "training lacks a gap/prefix diagnostic or a one-phase demonstration"
    assert test_joint, "tests must couple hard reflection to a retained background gap"


def _digest(puzzle):
    h = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                a = np.asarray(pair[key], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed study, count boundaries, ablations and negative checks.

    No grids or traces are saved; only small digest/descriptor sets are retained.
    The returned summary is convenient for optional compact diagnostic reports.
    """
    puzzles, shapes, structures, palettes = set(), set(), set(), set()
    lengths, fringes, col_offsets, row_parities, starts, gap_counts = (set() for _ in range(6))
    hard_gaps = absent_phase = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.add(_digest(p))
        if seed in (0, 41):
            assert _digest(p) == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                data = _decode(pair["input"])
                inp, row, bg, signals, axes, clean, start, seq = data
                f = _features(data)
                shapes.add(inp.shape)
                palettes.add((bg, signals))
                lengths.add(len(seq))
                fringes.add(axes[0] - row)
                col_offsets.add(int(np.sign(axes[1] - clean.shape[1] + 1)))
                row_parities.add(axes[0] % 2)
                starts.add(start)
                gap_counts.add(f["gaps"])
                hard_gaps += f["hard_gap"] > 0
                absent_phase += len(set(int(x) for x in seq) - {bg}) == 1
                roles = {}
                canonical = bytes(roles.setdefault(int(v), len(roles)) for v in inp.flat)
                structures.add(sha256(bytes(inp.shape) + canonical).hexdigest())
    assert len(puzzles) == 42 and len(structures) >= 200
    assert len(shapes) >= 60 and len(palettes) >= 25
    assert len(lengths) >= 9 and fringes == {0, 1, 2, 3}
    assert col_offsets == {-1, 0, 1} and row_parities == {0, 1}
    assert len(starts) >= 5 and len(gap_counts) >= 5
    assert hard_gaps >= 42 and absent_phase >= 42
    for nt, ns in ((2, 1), (10, 4), (10, 1), (5, 4)):
        validate_generated_puzzle(generate_puzzle(1000 + nt, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    original = generate_puzzle(0, 2, 1)
    probe = original["test"][0]["input"]
    parsed = _decode(probe)
    # A missing orbit witness is a semantic error, not a background default.
    invalid = probe.copy()
    row, axes = parsed[1], parsed[4]
    for point in next(_orbits(row + 1, probe.shape[1] - 1, *axes)):
        invalid[point[0], point[1] + 1] = ERASE
    try:
        solve(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("witness-free orbit accepted")
    for mode in ("visible", "tip", "gap"):
        bad = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in original[s]]
               for s in ("train", "test")}
        q = bad["test"][0]
        inp, row, bg, signals, axes, clean, start, seq = _decode(q["input"])
        if mode == "visible":
            point = (row, 0)
        elif mode == "tip":
            point = (row + len(seq) // 2, start + len(seq) // 2)
        else:
            i = int(np.flatnonzero(seq[:-2] == bg)[0])
            point = (row + 1, start + 1 + i)
        q["output"][point] = (int(q["output"][point]) + 1) % 8
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(puzzles),
            "input_shapes": len(shapes), "normalized_inputs": len(structures),
            "palettes": len(palettes), "seed_lengths": sorted(lengths),
            "row_fringes": sorted(fringes), "column_offset_signs": sorted(col_offsets),
            "seed_starts": sorted(starts), "gap_counts": sorted(gap_counts),
            "pairs_with_hard_retained_gaps": int(hard_gaps),
            "one_phase_seeds": int(absent_phase)}
