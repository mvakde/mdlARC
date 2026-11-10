"""Clearance-ranked docking: transplant rod templates, dock, and crop at a roof.

Only NumPy and the standard library are used. No files are read by generation or
solving. The optional self-test returns small aggregate measurements, not traces.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 971707006, 952564774, 976346698, 952071240, 681972462,
              504084523, 406212726, 895923420, 251487695, 777313628, 644475862,
              390019868, 115500672, 262791281, 24777170, 193452632, 876475644,
              422697312, 585220911, 253460724, 354466997, 502389313, 75743317,
              908848310, 630916179, 395697609, 554908164, 740113239, 176347051,
              747794276, 178430657, 208582458, 557249090, 356431367, 4443635,
              362749247, 383629930, 829507676, 88585472, 476977877, 988992368)


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _orient(a, code):
    b = np.flipud(a) if code % 2 else a
    return (b.T if code >= 2 else b).copy()


def _unorient(a, code):
    b = a.T if code >= 2 else a
    return (np.flipud(b) if code % 2 else b).copy()


def _key(a):
    return a.shape, a.tobytes()


def _hardware(f):
    g = np.full(f["shape"], f["bg"], dtype=np.int8)
    t, r, left, right, s = (f[k] for k in ("roof", "rail", "left", "right", "structure"))
    g[t, left:right + 1] = s
    g[r, left:right + 1] = s
    g[t:r + 1, left] = g[t:r + 1, right] = s
    for x, w, c, capacity, _, _ in f["slots"]:
        g[t + 1 + capacity:r, x:x + w] = s
        g[r, x:x + w] = c
    return g


def _input(f):
    g = _hardware(f)
    r, s = f["rail"], f["structure"]
    for x, w, c, _, length, prefix in f["slots"]:
        g[r + 1:r + 1 + prefix, x:x + w] = s
        g[r + 1 + prefix:r + 1 + length, x:x + w] = c
    return g


def _ordered_templates(f, mode="rule"):
    slots = f["slots"]
    templates = [(v[4], v[5]) for v in slots]
    destinations = sorted(range(len(slots)), key=lambda i: (slots[i][3], i))
    if mode == "unsorted":
        return templates
    if mode in ("area", "payload"):
        order = sorted(range(len(slots)), key=lambda i:
                       (slots[i][4] * slots[i][1] if mode == "area" else slots[i][4] - slots[i][5], i))
        ordered = [templates[i] for i in order]
    else:
        ordered = sorted(templates)
    if mode == "spatial":
        destinations = list(range(len(slots)))
    elif mode == "reverse":
        ordered = ordered[::-1]
    elif mode == "tie_reverse":
        destinations = sorted(range(len(slots)), key=lambda i: (slots[i][3], -i))
    assigned = [None] * len(slots)
    for i, template in zip(destinations, ordered):
        assigned[i] = template
    if mode == "keep_prefix":
        assigned = [(length, min(slots[i][5], length - 1))
                    for i, (length, _) in enumerate(assigned)]
    elif mode == "solid":
        assigned = [(length, 0) for length, _ in assigned]
    return assigned


def _output(f, mode="rule"):
    """Constructor: direct placement, used also for named shortcut mutations."""
    g = _hardware(f)
    t, r, s = f["roof"], f["rail"], f["structure"]
    for slot, (length, prefix) in zip(f["slots"], _ordered_templates(f, mode)):
        x, w, color, capacity = slot[:4]
        end = t + capacity
        if mode == "ignore_teeth":
            end = r - 1
        start = end - length + 1
        for j in range(length):
            row = start + j
            if row > t:
                g[row, x:x + w] = s if j < prefix else color
    return g


def _read_canonical(g):
    bg = int(g[0, 0])
    assert np.all(g[0] == bg) and np.all(g[-1] == bg)
    assert np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)
    occupied = np.argwhere(g != bg)
    assert len(occupied)
    t, left = occupied.min(axis=0).tolist()
    _, right = occupied.max(axis=0).tolist()
    s = int(g[t, left])
    assert s != bg and np.all(g[t, left:right + 1] == s)
    wall = np.flatnonzero(g[:, left] != bg)
    r = int(wall[-1])
    assert r >= t + 4 and np.array_equal(wall, np.arange(t, r + 1))
    assert np.all(g[t:r + 1, left] == s) and np.all(g[t:r + 1, right] == s)
    assert np.all(g[r, left:right + 1] != bg)
    slots, x = [], left + 1
    while x < right:
        color = int(g[r, x])
        if color == s:
            x += 1
            continue
        stop = x + 1
        while stop < right and g[r, stop] == color:
            stop += 1
        assert x >= left + 2 and stop <= right - 1
        assert g[r, x - 1] == s and g[r, stop] == s
        top = r - 1
        while top >= t + 1 and np.all(g[top, x:stop] == s):
            top -= 1
        tooth_top = top + 1
        capacity = tooth_top - t - 1
        assert 1 <= capacity <= r - t - 2
        end = r + 1
        while end < len(g) and np.any(g[end, x:stop] != bg):
            end += 1
        seq = g[r + 1:end, x]
        length = len(seq)
        assert length >= 1
        assert np.all(g[r + 1:end, x:stop] == seq[:, None])
        prefix = 0
        while prefix < length and seq[prefix] == s:
            prefix += 1
        assert prefix < length and np.all(seq[prefix:] == color)
        slots.append((x, stop - x, color, capacity, length, prefix))
        x = stop
    assert 2 <= len(slots) <= 7
    prefix_by_length = {}
    for slot in slots:
        assert prefix_by_length.setdefault(slot[4], slot[5]) == slot[5], "ambiguous length tie"
    f = dict(shape=g.shape, bg=bg, structure=s, roof=t, rail=r,
             left=left, right=right, slots=slots)
    assert np.array_equal(_input(f), g), "unexplained cells or incomplete rods/teeth"
    return f


def _parse(input_grid):
    a = _grid(input_grid)
    readings = []
    for code in range(4):
        g = _unorient(a, code)
        try:
            f = _read_canonical(g)
        except (AssertionError, IndexError):
            continue
        readings.append((g, f, code))
    if len(readings) != 1:
        raise ValueError("input must have one readable rack orientation")
    return readings[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only input pixels; selection, translation, and clipping are local."""
    g, f, code = _parse(input_grid)
    result = g.copy()
    t, r, s, bg = (f[k] for k in ("roof", "rail", "structure", "bg"))
    # Collect actual source rows, not generator metadata or any stored examples.
    available = []
    for x, w, color, _, length, _ in f["slots"]:
        available.append(tuple(0 if int(v) == s else 1 for v in g[r + 1:r + 1 + length, x]))
        result[r + 1:, x:x + w] = bg
    remaining = list(range(len(f["slots"])))
    while remaining:
        i = min(remaining, key=lambda k: (f["slots"][k][3], k))
        remaining.remove(i)
        seq = min(available, key=len)
        available.remove(seq)
        x, w, color = f["slots"][i][:3]
        tooth_top = r - 1
        while result[tooth_top - 1, x] == s and tooth_top - 1 > t:
            tooth_top -= 1
        # Translate the complete ordered sequence so its last cell touches the
        # tooth. Check each destination against the fixed roof before copying.
        for offset, role in enumerate(reversed(seq), 1):
            row = tooth_top - offset
            if row > t:
                result[row, x:x + w] = s if role == 0 else color
    return _orient(result, code)


def _features(f):
    slots = f["slots"]
    assigned = _ordered_templates(f)
    losses = [max(0, length - v[3]) for v, (length, _) in zip(slots, assigned)]
    return dict(lanes=len(slots), depth=f["rail"] - f["roof"] - 1,
                widths=tuple(v[1] for v in slots), capacities=tuple(v[3] for v in slots),
                lengths=tuple(v[4] for v in slots), prefixes=tuple(v[5] for v in slots),
                ties=len({v[3] for v in slots}) < len(slots),
                length_ties=len({v[4] for v in slots}) < len(slots),
                clipped=any(losses), uncut=any(v == 0 for v in losses),
                retained_prefix=any(p > loss for (_, p), loss in zip(assigned, losses)),
                partial_prefix=any(0 < loss < p for (_, p), loss in zip(assigned, losses)),
                payload_clip=any(loss > p for (_, p), loss in zip(assigned, losses)))


def _sample(rng, config, split, witness=0, witness_count=None):
    shift = config["shift"]
    if witness:
        n = witness_count
        k = int(rng.integers(0, 2))
        if witness == 1:
            lengths = [2 + k, 3 + k, 7 + k] if n == 3 else [2 + k, 3 + k, 5 + k, 7 + k]
            capacities = [2 + k, 4 + k, 6 + k] if n == 3 else [2 + k, 4 + k, 5 + k, 6 + k]
            prefixes = {v: int(rng.integers(0, v)) for v in lengths}
            prefixes[max(lengths)] = 0  # Visible loss of payload at the roof.
            prefixes[3 + k] = 1
        else:
            c = 5 + k
            lengths = [2, 3, c + 2] if n == 3 else [2, 3, 4, c + 2]
            capacities = [2, c, c] if n == 3 else [2, 3, c, c]
            prefixes = {v: int(rng.integers(0, v)) for v in lengths}
            prefixes[c + 2] = 3  # Two prefix cells clip; one remains.
            prefixes[2] = 0
        depth = max(capacities) + int(rng.integers(1, 4))
    else:
        choices = (range(5, 8) if split == "test" and shift == "lanes" else
                   range(2, 5) if shift == "lanes" else
                   range(2, 6) if split == "test" and shift == "widths" else range(2, 8))
        n = int(rng.choice(list(choices)))
        depth = int(rng.integers(12, 16)) if split == "test" and shift == "depth" else int(rng.integers(5, 11))
        mode = int(rng.integers(6))
        capacities = rng.integers(1, depth, n).tolist()
        if mode == 0:
            capacities = [int(rng.integers(3, depth))] * n
        elif mode in (1, 2):
            capacities.sort(reverse=mode == 2)
        elif mode == 3:
            capacities[0] = capacities[-1] = max(capacities)
        capacities[int(rng.integers(n))] = max(3, max(capacities))
        lengths = None
    top, bottom_margin = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    rail = top + depth + 1
    max_length = min(12, 29 - bottom_margin - rail)
    if lengths is None:
        lengths = [1, int(rng.integers(3, max_length + 1))] + rng.integers(1, max_length + 1, n - 2).tolist()
        prefixes = {v: int(rng.integers(0, v)) for v in set(lengths)}
    assert max(lengths) <= max_length
    rng.shuffle(lengths)
    if config["fixed_roles"]:
        bg, s = config["roles"]
    else:
        bg, s = (int(v) for v in rng.choice(10, 2, replace=False))
    pool = [v for v in range(10) if v not in (bg, s)]
    palette_size = int(rng.integers(2 if witness else 1, min(5, n) + 1))
    palette = [int(v) for v in rng.choice(pool, palette_size, replace=False)]
    colors = palette + [int(v) for v in rng.choice(palette, n - palette_size)]
    rng.shuffle(colors)
    max_width = 4 if split == "test" and shift == "widths" else 3
    widths = rng.integers(1, max_width + 1, n).tolist()
    if split == "test" and shift == "widths":
        widths[int(rng.integers(n))] = 4
    if witness == 1:
        # Widths are attached to source slots, so create a strict area inversion.
        long_index, medium_index = lengths.index(max(lengths)), lengths.index(3 + k)
        widths[long_index], widths[medium_index] = 1, 3
    # Space every lane, including repeated colors. The full rack and margins fit.
    gaps = [1] * (n - 1)
    budget = 23 - sum(widths) - sum(gaps)
    while budget < 0:
        candidates = [i for i, w in enumerate(widths) if w > 1]
        i = max(candidates, key=lambda j: widths[j])
        widths[i] -= 1
        budget += 1
    extra = int(rng.integers(0, min(4, budget) + 1))
    for _ in range(extra):
        gaps[int(rng.integers(n - 1))] += 1
    left = int(rng.integers(1, 3))
    x = left + 2
    slots = []
    rng.shuffle(capacities)
    for i, (w, color, capacity, length) in enumerate(zip(widths, colors, capacities, lengths)):
        slots.append((x, w, color, capacity, length, prefixes[length]))
        x += w + (gaps[i] if i < n - 1 else 0)
    right = x + 1
    f = dict(shape=(rail + max(lengths) + 1 + bottom_margin, right + 2),
             bg=bg, structure=s, roof=top, rail=rail, left=left, right=right, slots=slots)
    return f


def _qualifies(f, witness):
    out = _output(f)
    required = ["unsorted", "ignore_teeth"]
    if witness == 1:
        required += ["area", "payload", "spatial", "reverse", "keep_prefix", "solid"]
    if witness == 2:
        required += ["tie_reverse"]
    return all(not np.array_equal(out, _output(f, mode)) for mode in required)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    config = dict(shift=str(rng.choice(["lanes", "depth", "widths"])),
                  fixed_roles=bool(rng.integers(2)),
                  roles=tuple(int(v) for v in rng.choice(10, 2, replace=False)))
    codes = [int(rng.integers(4))]
    if rng.random() < 0.4:
        codes.append(int(rng.choice([v for v in range(4) if v != codes[0]])))
    witness_counts = [3, 4]
    rng.shuffle(witness_counts)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = i + 1 if split == "train" and i < 2 else 0
            code = codes[i % len(codes)] if split == "train" else int(rng.choice(codes))
            for _ in range(128):
                f = _sample(rng, config, split, witness, witness_counts[i] if witness else None)
                if not _qualifies(f, witness):
                    continue
                inp, out = _orient(_input(f), code), _orient(_output(f), code)
                if _key(inp) not in seen:
                    break
            else:
                raise RuntimeError("bounded rack sampling exhausted")
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": out})
    rng.shuffle(puzzle["train"])
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    g, f, code = _parse(inp)
    actual = _unorient(out, code)
    expected = _hardware(f)
    # Relational suffix check: each output is the terminal capacity-sized suffix
    # of its ranked template, attached to the tooth, in the fixed slot's color.
    ranked = sorted((v[4], v[5]) for v in f["slots"])
    destinations = sorted(f["slots"], key=lambda v: (v[3], v[0]))
    for slot, (length, prefix) in zip(destinations, ranked):
        x, w, color, capacity = slot[:4]
        kept = min(length, capacity)
        kept_prefix = max(0, prefix - (length - kept))
        bottom = f["roof"] + capacity
        first = bottom - kept + 1
        expected[first:first + kept_prefix, x:x + w] = f["structure"]
        expected[first + kept_prefix:bottom + 1, x:x + w] = color
        assert actual[bottom, x] == color, "rod must contact the tooth with its payload end"
    assert np.array_equal(actual, expected), "rank, template, contact, clipping, or hardware mismatch"
    assert np.array_equal(solve(inp), out)
    return f, code


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train, test_codes = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            f, code = _check_pair(pair)
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert _qualifies(f, 0), "both inherited mechanisms must affect the answer"
            if split == "train":
                train.append((f, code))
            else:
                test_codes.add(code)
    assert test_codes <= {code for _, code in train}
    assert len({len(f["slots"]) for f, _ in train}) >= 2
    assert any(_qualifies(f, 1) for f, _ in train), "missing length/area/rank/prefix witness"
    assert any(_qualifies(f, 2) for f, _ in train), "missing capacity-tie witness"
    assert any(_features(f)["payload_clip"] for f, _ in train)
    assert any(_features(f)["partial_prefix"] for f, _ in train)
    assert any(_features(f)["uncut"] for f, _ in train)


def self_test():
    """42 seeds, all explicit counts, invariants, and shortcut rejection; no I/O."""
    # Hand-calculated regression, with no call to the direct output constructor.
    fixture = dict(shape=(20, 15), bg=0, structure=5, roof=1, rail=10, left=1, right=13,
                   slots=[(3, 3, 2, 6, 3, 1), (8, 1, 3, 2, 7, 3), (11, 1, 4, 4, 2, 0)])
    expected = _hardware(fixture)
    expected[2:4, 3:6] = 5
    expected[4:8, 3:6] = 2
    expected[2:4, 8] = 3
    expected[3, 11] = 5
    expected[4:6, 11] = 4
    for code in range(4):
        for mapping in (np.arange(10, dtype=np.int8), np.array([(3 * i + 7) % 10 for i in range(10)], dtype=np.int8)):
            assert np.array_equal(solve(_orient(mapping[_input(fixture)], code)),
                                  _orient(mapping[expected], code))
    # Semantic edge cases are legal even when generation's teaching filters
    # exclude their uninformative original-slot assignment.
    tied = dict(shape=(16, 13), bg=0, structure=5, roof=1, rail=7, left=1, right=11,
                slots=[(3, 2, 7, 3, 5, 4), (8, 1, 7, 3, 5, 4)])
    tied_expected = _hardware(tied)
    tied_expected[2:4, 3:5] = tied_expected[2:4, 8:9] = 5
    tied_expected[4, 3:5] = tied_expected[4, 8:9] = 7
    assert np.array_equal(solve(_input(tied)), tied_expected)
    single = dict(tied, slots=[(3, 2, 7, 1, 1, 0), (8, 1, 7, 4, 1, 0)])
    single_expected = _hardware(single)
    single_expected[2, 3:5] = single_expected[5, 8:9] = 7
    assert np.array_equal(solve(_input(single)), single_expected)
    signatures, shapes, structures, layouts, roles, codes = set(), set(), set(), set(), set(), set()
    counts, widths, depths, flags, total = set(), set(), set(), {}, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f, code = _check_pair(pair)
                total += 1
                features = _features(f)
                shapes.add(pair["input"].shape)
                counts.add(features["lanes"])
                widths.update(features["widths"])
                depths.add(features["depth"])
                roles.add((f["bg"], f["structure"]))
                codes.add(code)
                layouts.add((f["shape"], f["roof"], f["rail"],
                             tuple((v[0], v[1], v[3], v[4], v[5]) for v in f["slots"]), code))
                norm, mapping = [], {}
                for v in pair["input"].ravel():
                    norm.append(mapping.setdefault(int(v), len(mapping)))
                structures.add((pair["input"].shape, bytes(norm)))
                digest.update(repr(pair["input"].shape).encode())
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
                for name in ("ties", "length_ties", "clipped", "uncut", "retained_prefix", "partial_prefix", "payload_clip"):
                    flags[name] = flags.get(name, 0) + int(features[name])
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(structures) >= 180 and len(layouts) >= 180
    assert counts == set(range(2, 8)) and widths >= {1, 2, 3, 4}
    assert len(depths) >= 8 and codes == set(range(4)) and len(roles) >= 20
    assert all(value > 0 for value in flags.values())
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            a, b = generate_puzzle(0, nt, nq), generate_puzzle(0, nt, nq)
            validate_generated_puzzle(a)
            for split in a:
                for x, y in zip(a[split], b[split]):
                    assert all(np.array_equal(x[k], y[k]) for k in x)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Ensure the relational validator rejects actual faulty solved grids.
    demo = generate_puzzle(0)
    for pair in demo["train"]:
        f, code = _parse(pair["input"])[1:]
        for mode in ("unsorted", "area", "payload", "spatial", "reverse", "tie_reverse", "ignore_teeth", "keep_prefix", "solid"):
            mutant = _orient(_output(f, mode), code)
            if np.array_equal(mutant, pair["output"]):
                continue
            try:
                _check_pair({"input": pair["input"], "output": mutant})
            except AssertionError:
                pass
            else:
                raise AssertionError(f"accepted {mode} mutation")
    return dict(seeds=42, pairs=total, shapes=len(shapes), normalized_structures=len(structures),
                layouts=len(layouts), role_pairs=len(roles), orientations=sorted(codes),
                lane_counts=sorted(counts), widths=sorted(widths), depths=sorted(depths), flags=flags)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
