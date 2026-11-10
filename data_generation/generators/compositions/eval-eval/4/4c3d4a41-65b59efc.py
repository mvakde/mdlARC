"""Contour-contact stamp press. Standard library and NumPy only.

The input contains its complete program: a keyed bitmap library, compact
top-to-bottom stamp sequences, and a one-pixel-wide probe in each chamber.
Generation uses coordinate placement; solve expands pixels and pushes them
one row at a time. Neither path imports parent code or reads any files.
"""

from __future__ import annotations

import hashlib
import numpy as np


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _frame(f):
    """Machine alone, with empty chambers and the installed comb."""
    n, d, m = f["n"], f["depth"], len(f["commands"])
    a = np.full((d + 2, m * (n + 1) + 1), f["background"], dtype=np.int8)
    s = f["structure"]
    a[0, :] = a[-1, :] = s
    a[:, ::n + 1] = s
    for j, (probe, height) in enumerate(zip(f["probes"], f["heights"])):
        a[d - height + 1:d + 1, 1 + j * (n + 1) + probe] = s
    return a


def _encode(f):
    """Canonical scene; outer background margins are added by the sampler."""
    n, k = f["n"], len(f["labels"])
    library_width = k * (n + 1) + 1
    machine = _frame(f)
    a = np.full((n + 4 + machine.shape[0], max(library_width, machine.shape[1])),
                f["background"], dtype=np.int8)
    s = f["structure"]
    a[0, :library_width] = a[n + 1, :library_width] = s
    a[:n + 2, :library_width:n + 1] = s
    for j, (label, mask, target) in enumerate(zip(f["labels"], f["masks"], f["targets"])):
        c = 1 + j * (n + 1)
        a[1:n + 1, c:c + n] = np.where(mask, label, f["background"])
        a[n + 2, c + n // 2] = target
    a[n + 4:, :machine.shape[1]] = machine
    for j, sequence in enumerate(f["commands"]):
        c = 1 + j * (n + 1)
        a[n + 5:n + 5 + len(sequence), c] = sequence
    return a


def _decode(input_grid):
    """Recover geometry and commands from input pixels, checking all metadata."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)
    rr, cc = np.nonzero(a != bg)
    assert len(rr)
    core = a[int(rr.min()):int(rr.max()) + 1, int(cc.min()):int(cc.max()) + 1]
    s = int(core[0, 0])
    assert s != bg
    run = 0
    while run < core.shape[0] and core[run, 0] == s:
        run += 1
    n = run - 2
    assert 2 <= n <= 5
    width = 0
    while width < core.shape[1] and core[0, width] == s:
        width += 1
    assert (width - 1) % (n + 1) == 0
    k = (width - 1) // (n + 1)
    assert 2 <= k <= 4
    labels, masks, targets = [], [], []
    for j in range(k):
        c = 1 + j * (n + 1)
        panel = core[1:n + 1, c:c + n]
        colors = np.unique(panel[panel != bg])
        assert len(colors) == 1
        label = int(colors[0])
        assert label != s and label not in labels
        labels.append(label)
        masks.append(panel == label)
        target = int(core[n + 2, c + n // 2])
        assert target != bg
        targets.append(target)
    roof = n + 4
    assert roof + 2 < core.shape[0]
    machine_width = 0
    while machine_width < core.shape[1] and core[roof, machine_width] == s:
        machine_width += 1
    assert (machine_width - 1) % (n + 1) == 0
    m = (machine_width - 1) // (n + 1)
    assert 2 <= m <= min(6, 27 // (n + 1))
    d = core.shape[0] - roof - 2
    assert n + 1 <= d <= 22 - n
    commands, probes, heights = [], [], []
    for j in range(m):
        c = 1 + j * (n + 1)
        tokens = core[roof + 1:roof + d + 1, c]
        occupied = np.flatnonzero(tokens != bg)
        assert 1 <= len(occupied) <= d // n
        assert np.array_equal(occupied, np.arange(len(occupied)))
        sequence = tuple(int(x) for x in tokens[:len(occupied)])
        assert set(sequence) <= set(labels)
        bottom_row = core[roof + d, c:c + n]
        tips = np.flatnonzero(bottom_row == s)
        assert len(tips) == 1 and 1 <= tips[0] < n
        probe = int(tips[0])
        height = 0
        while height < d and core[roof + d - height, c + probe] == s:
            height += 1
        assert 1 <= height <= d - 2
        commands.append(sequence)
        probes.append(probe)
        heights.append(height)
    f = dict(n=n, depth=d, background=bg, structure=s, labels=tuple(labels),
             masks=tuple(masks), targets=tuple(targets), commands=tuple(commands),
             probes=tuple(probes), heights=tuple(heights))
    assert np.array_equal(core, _encode(f)), "malformed library, rail, token or tooth"
    for j in range(m):
        bitmap = _ribbon(f, j)
        contact = np.flatnonzero(bitmap[:, probes[j]] != bg)
        assert len(contact) and contact[-1] >= d - heights[j] - 1, "unsupported contact gap"
    return f


def _ribbon(f, lane, mode="correct"):
    n, d, bg = f["n"], f["depth"], f["background"]
    a = np.full((d, n), bg, dtype=np.int8)
    sequence = f["commands"][lane]
    if mode == "reverse":
        sequence = sequence[::-1]
    for t, identity in enumerate(sequence):
        index = f["labels"].index(identity)
        if mode == "position":
            index = lane % len(f["labels"])
        mask = np.ones((n, n), dtype=bool) if mode == "solid" else f["masks"][index]
        color = f["labels"][index] if mode == "unkeyed" else f["targets"][index]
        a[t * n:(t + 1) * n] = np.where(mask, color, bg)
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Expand each rigid ribbon, push one pixel at a time, clip at the roof."""
    f = _decode(input_grid)
    result = _frame(f)
    n, d, bg = f["n"], f["depth"], f["background"]
    for j, probe in enumerate(f["probes"]):
        ribbon = _ribbon(f, j)
        tooth_top = d - f["heights"][j]
        while np.flatnonzero(ribbon[:, probe] != bg)[-1] >= tooth_top:
            ribbon[:-1] = ribbon[1:].copy()
            ribbon[-1] = bg
        assert np.flatnonzero(ribbon[:, probe] != bg)[-1] == tooth_top - 1
        c = 1 + j * (n + 1)
        region = result[1:d + 1, c:c + n]
        ink = ribbon != bg
        assert np.all(region[ink] == bg), "stamp intersects its tooth"
        region[ink] = ribbon[ink]
    return result


def _answer(f, mode="correct"):
    """Coordinate construction, also used for explicit shortcut counterfactuals."""
    result = _frame(f)
    n, d, bg = f["n"], f["depth"], f["background"]
    for j, probe in enumerate(f["probes"]):
        ribbon = _ribbon(f, j, mode)
        contacts = np.flatnonzero(np.any(ribbon != bg, axis=1) if mode == "envelope"
                                  else ribbon[:, probe] != bg)
        shift = max(0, int(contacts[-1]) - (d - f["heights"][j] - 1)) if len(contacts) else 0
        if mode == "unshifted":
            shift = 0
        elif mode == "rounded":
            shift = (shift // n) * n
        for r, c in zip(*np.nonzero(ribbon != bg)):
            destination = (int(r) - shift) % d if mode == "wrap" else int(r) - shift
            if 0 <= destination < d:
                result[destination + 1, 1 + j * (n + 1) + int(c)] = ribbon[r, c]
    return result


def _motif(rng, n):
    """No reserved silhouette: dense/sparse, outline, stripes, and irregular."""
    style = int(rng.integers(7))
    rr, cc = np.indices((n, n))
    if style == 0:
        a = np.ones((n, n), dtype=bool)
    elif style == 1:
        a = (rr == 0) | (cc == 0) | (rr == n - 1) | (cc == n - 1)
    elif style == 2:
        a = rr == cc
    elif style == 3:
        a = (rr + cc) % 2 == int(rng.integers(2))
    elif style == 4:
        a = cc == int(rng.integers(n))
    else:
        a = rng.random((n, n)) < float(rng.uniform(.18, .86))
    if not a.any():
        a.flat[int(rng.integers(n * n))] = True
    return a


def _library(rng, n, witness):
    bg, s = (int(x) for x in rng.choice(10, 2, replace=False))
    k = int(rng.integers(2, 5))
    labels = tuple(int(x) for x in rng.choice([x for x in range(10) if x not in (bg, s)],
                                            k, replace=False))
    masks = [_motif(rng, n) for _ in labels]
    if witness:
        # One blind probe column and one contour that reaches both its first
        # and last row. These are evidence conditions, not family-wide bans.
        masks[0][:, -1] = False
        masks[0][-1, 0] = True
        masks[1][0, -1] = masks[1][-1, -1] = True
        masks[1][0, 0] = False
        pool = [x for x in range(10) if x not in (bg, s)]
        targets = tuple(labels[(i + 1) % k] for i in range(k))
        # Usually use a fresh arbitrary, visible, distinct recoloring instead.
        for _ in range(12):
            candidate = tuple(int(x) for x in rng.choice(pool, k, replace=False))
            if all(a != b for a, b in zip(candidate, labels)):
                targets = candidate
                break
    else:
        # Identical geometry, shared targets and self-preserving keys are valid.
        if rng.random() < .2:
            masks[-1] = masks[0].copy()
        targets = tuple(int(x) for x in rng.choice([x for x in range(10) if x != bg], k))
    return dict(n=n, background=bg, structure=s, labels=labels, masks=tuple(masks), targets=targets)


def _random_lane(rng, f, depth):
    n = f["n"]
    length = int(rng.integers(1, min(6, depth // n) + 1))
    sequence = [int(x) for x in rng.choice(f["labels"], length)]
    # A tiny constructive repair guarantees a readable contact ray, even for
    # a library whose arbitrary motifs avoid every admissible probe column.
    last_index = f["labels"].index(sequence[-1])
    mask = f["masks"][last_index]
    candidates = [(c, (length - 1) * n + int(np.flatnonzero(mask[:, c])[-1]))
                  for c in range(1, n) if np.any(mask[:, c])]
    candidates = [(c, p) for c, p in candidates if p >= 1]
    if not candidates:
        # Change the chosen last command, not an already used template.
        for identity, other in zip(f["labels"], f["masks"]):
            candidates = [(c, (length - 1) * n + int(np.flatnonzero(other[:, c])[-1]))
                          for c in range(1, n) if np.any(other[:, c])]
            candidates = [(c, p) for c, p in candidates if p >= 1]
            if candidates:
                sequence[-1] = identity
                break
    assert candidates
    probe, p = candidates[int(rng.integers(len(candidates)))]
    # Choose the final contact coordinate directly: h = depth - 1 - q.
    if rng.random() < .22 and p <= depth - 2:
        q = p
    else:
        q = int(rng.integers(1, min(p, depth - 2) + 1))
    return tuple(sequence), probe, depth - 1 - q


def _witness(rng, n, m):
    f = _library(rng, n, True)
    depth = int(rng.integers(3 * n + 1, min(4 * n + 3, 22 - n) + 1))
    labels = f["labels"]
    sequence2 = (labels[2] if len(labels) >= 3 else labels[0],
                 labels[3] if len(labels) >= 4 else labels[0], labels[1])
    commands = [(labels[1], labels[1], labels[0]),
                (labels[0], labels[1], labels[1]), sequence2]
    probes = [n - 1] * 3
    h = depth - 2 * n + 1
    heights = [h, h, depth - 3 * n]
    for _ in range(m - 3):
        seq, probe, height = _random_lane(rng, f, depth)
        commands.append(seq)
        probes.append(probe)
        heights.append(height)
    f.update(depth=depth, commands=tuple(commands), probes=tuple(probes), heights=tuple(heights))
    return f


def _ordinary(rng, n, m):
    f = _library(rng, n, False)
    # Guarantee one available contact ray while leaving all other motifs free.
    if not any(np.any(mask[1:, 1:]) for mask in f["masks"]):
        f["masks"][int(rng.integers(len(f["masks"])))][-1, -1] = True
    depth = int(rng.integers(2 * n + 1, 22 - n + 1))
    lanes = [_random_lane(rng, f, depth) for _ in range(m)]
    f.update(depth=depth, commands=tuple(x[0] for x in lanes),
             probes=tuple(x[1] for x in lanes), heights=tuple(x[2] for x in lanes))
    return f


def _pair(rng, f):
    core = _encode(f)
    row_budget, col_budget = 30 - core.shape[0], 30 - core.shape[1]
    top = int(rng.integers(1, min(3, row_budget - 1) + 1))
    bottom = int(rng.integers(1, min(3, row_budget - top) + 1))
    left = int(rng.integers(1, min(3, col_budget - 1) + 1))
    right = int(rng.integers(1, min(3, col_budget - left) + 1))
    return {"input": np.pad(core, ((top, bottom), (left, right)),
                            constant_values=f["background"]), "output": _answer(f)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 6))
    cap = min(6, 27 // (n + 1))
    training_max = int(rng.integers(3, cap))
    first = _witness(rng, n, training_max)
    second = dict(first, heights=(first["heights"][0] + 1, *first["heights"][1:]))
    puzzle = {"train": [_pair(rng, first), _pair(rng, second)], "test": []}
    seen = {_grid_key(p["input"]) for p in puzzle["train"]}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for _ in range(count):
            for _attempt in range(64):
                m = int(rng.integers(2, training_max + 1) if split == "train"
                        else rng.integers(training_max + 1, cap + 1))
                candidate = _pair(rng, _ordinary(rng, n, m))
                key = _grid_key(candidate["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(candidate)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _hardware_key(f):
    return (f["n"], f["depth"], f["background"], f["structure"], f["labels"],
            tuple(m.tobytes() for m in f["masks"]), f["targets"], f["commands"], f["probes"])


def _features(f):
    shifts, skips, clipped, partial, overhang = [], [], 0, 0, 0
    n, d, bg = f["n"], f["depth"], f["background"]
    for j, (probe, h) in enumerate(zip(f["probes"], f["heights"])):
        ribbon = _ribbon(f, j)
        p = int(np.flatnonzero(ribbon[:, probe] != bg)[-1])
        q = d - h - 1
        shift = p - q
        shifts.append(shift)
        skips.append((len(f["commands"][j]) - 1) - p // n)
        clipped += int(np.count_nonzero(ribbon[:shift] != bg))
        partial += int(shift % n != 0 and np.any(ribbon[:shift] != bg))
        overhang += int(np.any(ribbon[p + 1:] != bg))
    return dict(shifts=tuple(shifts), skipped_tiles=tuple(skips), clipped_pixels=clipped,
                partial_clips=partial, overhang_lanes=overhang)


_SHORTCUTS = ("unshifted", "solid", "unkeyed", "envelope", "rounded", "reverse", "position", "wrap")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, readings = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            f = _decode(inp)
            assert out.shape == _frame(f).shape
            assert np.array_equal(out, solve(inp))
            assert np.array_equal(out, _answer(f))
            assert set(int(c) for c in np.unique(out)) <= {f["background"], f["structure"], *f["targets"]}
            assert _features(f)["clipped_pixels"] >= 0
            # All roof, base, dividers and installed teeth survive unchanged.
            metadata = _frame(f) == f["structure"]
            assert np.all(out[metadata] == f["structure"])
            readings[split].append(f)
    train, test = readings["train"], readings["test"]
    assert len({f["n"] for f in train + test}) == 1
    assert min(len(f["commands"]) for f in test) > max(len(f["commands"]) for f in train)
    assert any(_features(f)["partial_clips"] for f in train)
    assert any(max(_features(f)["skipped_tiles"]) >= 1 for f in train)
    assert any(0 in _features(f)["shifts"] and max(_features(f)["shifts"]) > 0 for f in train)
    # These compare actual answers, rather than trusting sampler mode flags.
    for mode in _SHORTCUTS:
        assert any(not np.array_equal(_answer(f), _answer(f, mode)) for f in train), mode
    controlled = False
    for i, a in enumerate(train):
        for b in train[i + 1:]:
            if _hardware_key(a) == _hardware_key(b):
                delta = np.asarray(a["heights"]) - np.asarray(b["heights"])
                if np.count_nonzero(delta) == 1 and int(np.abs(delta).sum()) == 1:
                    lane = int(np.flatnonzero(delta)[0])
                    aa, bb = _answer(a), _answer(b)
                    c = 1 + lane * (a["n"] + 1)
                    changed = aa != bb
                    assert changed.any()
                    changed[:, c:c + a["n"]] = False
                    assert not changed.any(), "a tooth can only change its own chamber"
                    controlled = True
    assert controlled, "missing controlled one-pixel tooth comparison"


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, legal counts, mutation checks; retains summaries, never traces."""
    dimensions, motifs, palettes, profiles, signatures, widths, depths, counts = (set() for _ in range(8))
    contact_skips = partial_clips = stationary = moving = shared_targets = 0
    scales = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_digest(p))
        for split in ("train", "test"):
            for pair in p[split]:
                f = _decode(pair["input"])
                features = _features(f)
                scales.add(f["n"])
                widths.add(len(f["commands"]))
                depths.add(f["depth"])
                counts.add(len(f["labels"]))
                dimensions.add((pair["input"].shape, pair["output"].shape))
                palettes.add((f["background"], f["structure"]))
                profiles.add((f["probes"], f["heights"]))
                motifs.update((m.shape, m.tobytes()) for m in f["masks"])
                contact_skips += sum(x > 0 for x in features["skipped_tiles"])
                partial_clips += features["partial_clips"]
                stationary += features["shifts"].count(0)
                moving += sum(x > 0 for x in features["shifts"])
                shared_targets += len(set(f["targets"])) < len(f["targets"])
    assert len(signatures) == 42 and scales == {2, 3, 4, 5}
    assert widths == {2, 3, 4, 5, 6} and counts == {2, 3, 4}
    assert len(dimensions) >= 60 and len(motifs) >= 80 and len(palettes) >= 35
    assert len(profiles) >= 100 and len(depths) >= 12
    assert min(contact_skips, partial_clips, stationary, moving, shared_targets) > 5
    assert _digest(generate_puzzle(816)) == _digest(generate_puzzle(816))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1147 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (3, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    # A missing tooth and a gap in a command stack must fail pixel parsing.
    f = _decode(p["train"][0]["input"])
    core = _encode(f)
    for row, col in ((f["n"] + 4 + f["depth"], 1 + f["probes"][0]),
                     (f["n"] + 5, 1)):
        bad = core.copy()
        bad[row, col] = f["background"]
        try:
            solve(np.pad(bad, 1, constant_values=f["background"]))
        except AssertionError:
            pass
        else:
            raise AssertionError("malformed input accepted")
    # Changing a bottom contour changes the motion, not just the printed ink.
    witness = _witness(np.random.default_rng(901), 3, 3)
    masks = tuple(mask.copy() for mask in witness["masks"])
    masks[0][-1, -1] = True
    altered = dict(witness, masks=masks)
    assert _features(altered)["shifts"][0] == _features(witness)["shifts"][0] + 3
    assert np.array_equal(solve(np.pad(_encode(altered), 1,
                                      constant_values=altered["background"])), _answer(altered))
    # A key changes colors but cannot change contact geometry or displacement.
    recolored = dict(witness, targets=tuple(witness["labels"]))
    assert _features(recolored)["shifts"] == _features(witness)["shifts"]
    # The solver also accepts legal long ribbons outside the sampler's six-token cap.
    long = _witness(np.random.default_rng(902), 2, 3)
    long = dict(long, depth=20, commands=((long["labels"][1],) * 10,) * 3,
                probes=(1, 1, 1), heights=(1, 2, 3))
    long_input = np.pad(_encode(long), 1, constant_values=long["background"])
    assert long_input.shape[0] == 30 and np.array_equal(solve(long_input), _answer(long))
    mapping = np.array([(3 * i + 7) % 10 for i in range(10)], dtype=np.int8)
    assert np.array_equal(solve(mapping[long_input]), mapping[_answer(long)])
    gap = dict(witness, heights=(1, *witness["heights"][1:]))
    try:
        solve(np.pad(_encode(gap), 1, constant_values=gap["background"]))
    except AssertionError:
        pass
    else:
        raise AssertionError("unsupported contact gap accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), shape_pairs=len(dimensions),
                motifs=len(motifs), role_palettes=len(palettes), profiles=len(profiles),
                scales=sorted(scales), chambers=sorted(widths), depths=sorted(depths),
                library_counts=sorted(counts), skipped_contact_lanes=contact_skips,
                partial_clips=partial_clips, stationary_lanes=stationary, moving_lanes=moving,
                pairs_with_shared_targets=shared_targets)
