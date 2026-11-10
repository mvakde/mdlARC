"""Rank-folded tapes: a central span defines a two-reflection repair domain.

Only NumPy and the standard library are used. Solving parses the input afresh;
generation draws a clean rank/step table and damages its extents and witnesses.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0, 29144601, 517074528, 732197557, 910508303, 660898641,
              505464877, 153922660, 275498502, 940149167, 589841268,
              193062033, 336074142, 433263125, 679486733, 1748656,
              245115677, 730232018, 465504791, 508198711, 377536862,
              732590891, 760870155, 586488132, 226480949, 556948132,
              417525797, 460782703, 701138763, 807000877, 652811179,
              706535086, 162608360, 701576650, 536295370, 351346246,
              383559889, 859681179, 953791308, 764574049, 946428408, 638847786)
AXES = ((0, 1), (1, 0), (1, 1), (1, -1))


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid using colors 0..9")
    return a.astype(np.int8, copy=True)


def _coordinates(r, c, axis):
    if axis == (0, 1):
        return r, c, 1
    if axis == (1, 0):
        return c, r, 1
    if axis == (1, 1):
        return r - c, r + c, 2
    return r + c, r - c, 2


def _point(a, i, t):
    """t=-2,-1 are the two cap cells; t>=0 is patterned material."""
    r, c = a["starts"][i]
    dr, dc = a["direction"]
    return r + (t + 2) * dr, c + (t + 2) * dc


def _parse(grid):
    values, frequencies = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(frequencies)])
    if int(max(frequencies)) <= grid.size / 2:
        raise ValueError("background must occupy over half of the canvas")
    coords = np.argwhere(grid != background)
    candidates = []
    for axis in AXES:
        groups = {}
        for r, c in coords:
            key, x, step = _coordinates(int(r), int(c), axis)
            groups.setdefault(key, []).append((x, int(r), int(c)))
        n = len(groups)
        if n not in (3, 5, 7, 9):
            continue
        keys = sorted(groups)
        gaps = tuple(b - a for a, b in zip(keys, keys[1:]))
        unit = 2 if all(axis) else 1
        if min(gaps) < 2 * unit or any(g % unit for g in gaps):
            continue
        lanes = [sorted(groups[k]) for k in keys]
        if any(len(lane) < 3 or any(b[0] - a[0] != step
                   for a, b in zip(lane, lane[1:])) for lane in lanes):
            continue
        common_min = len({lane[0][0] for lane in lanes}) == 1
        common_max = len({lane[-1][0] for lane in lanes}) == 1
        if common_min == common_max:
            continue
        sign = 1 if common_min else -1
        ordered = lanes if sign == 1 else [lane[::-1] for lane in lanes]
        caps = {int(grid[r, c]) for lane in ordered for _, r, c in lane[:2]}
        if len(caps) != 1:
            continue
        marker = caps.pop()
        starts = [(lane[0][1], lane[0][2]) for lane in ordered]
        lengths = tuple(len(lane) - 2 for lane in ordered)
        length = lengths[n // 2]
        if length < 3:
            continue
        a = dict(background=background, marker=marker, n=n, length=length,
                 lengths=lengths, starts=starts, keys=tuple(keys), gaps=gaps,
                 direction=(sign * axis[0], sign * axis[1]), shape=grid.shape)
        if any(not (0 <= _point(a, i, length - 1)[0] < grid.shape[0]
                    and 0 <= _point(a, i, length - 1)[1] < grid.shape[1])
               for i in range(n)):
            continue
        candidates.append(a)
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous capped parallel array")
    return candidates[0]


def _orbits(n, length):
    for i in range((n + 1) // 2):
        for t in range((length + 1) // 2):
            yield tuple(sorted({(i, t), (n - 1 - i, t),
                                (i, length - 1 - t),
                                (n - 1 - i, length - 1 - t)}))


def _visible_table(grid, a):
    table = np.full((a["n"], a["length"]), -1, dtype=np.int8)
    for i, span in enumerate(a["lengths"]):
        for t in range(min(span, a["length"])):
            value = int(grid[_point(a, i, t)])
            if value != a["marker"]:
                table[i, t] = value
    return table


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer cap role, order, direction, central extent and orbit witnesses."""
    grid = _grid(input_grid)
    a = _parse(grid)
    visible = _visible_table(grid, a)
    restored = visible.copy()
    for orbit in _orbits(a["n"], a["length"]):
        witnesses = {int(visible[p]) for p in orbit if visible[p] >= 0}
        if len(witnesses) != 1:
            raise ValueError("an in-domain reflection orbit is empty or contradictory")
        value = witnesses.pop()
        for p in orbit:
            restored[p] = value
    output = np.full(grid.shape, a["background"], dtype=np.int8)
    for i in range(a["n"]):
        for t in (-2, -1):
            output[_point(a, i, t)] = a["marker"]
        for t in range(a["length"]):
            output[_point(a, i, t)] = restored[i, t]
    return output


def _clean_table(rng, n, length, payload):
    orbits = list(_orbits(n, length))
    k = len(payload)
    style = int(rng.integers(4))
    table = np.empty((n, length), dtype=np.int8)
    shift = int(rng.integers(k))
    for orbit in orbits:
        i, t = orbit[0]
        if style == 0:
            color = int(rng.choice(payload))
        elif style == 1:
            color = payload[(i + t // int(rng.integers(1, 4)) + shift) % k]
        elif style == 2:
            color = payload[(i + t + shift) % k]
        else:
            color = payload[(max(i, t // 2) + shift) % k]
        for p in orbit:
            table[p] = color
    # Explicitly distinguish local color continuation and copying the center.
    reserved = {(0, 0): payload[0], (0, 1): payload[1],
                (n // 2, 0): payload[1]}
    spare = [o for o in orbits if not any(p in reserved for p in o)]
    for color, orbit in zip(payload[2:], spare):
        for p in orbit:
            table[p] = color
    for orbit in orbits:
        hits = [reserved[p] for p in orbit if p in reserved]
        if hits:
            for p in orbit:
                table[p] = hits[0]
    return table


def _lengths(rng, n, length, capacity, mode):
    spans = [length] * n
    others = [int(i) for i in rng.permutation(n) if i != n // 2]
    short = lambda: int(rng.integers(1, length))
    long = lambda: int(rng.integers(length + 1, capacity + 1))
    if mode == "witness":
        majority_short = bool(rng.integers(2))
        for i in others:
            spans[i] = short() if majority_short else long()
        spans[others[0]] = long() if majority_short else short()
    elif mode in ("one_extend", "one_trim"):
        spans[others[0]] = short() if mode == "one_extend" else long()
    else:
        for i in others:
            if rng.random() < .25:
                continue
            spans[i] = short() if mode == "extend" or (
                mode == "mixed" and rng.random() < .5) else long()
        if mode == "mixed":
            spans[others[0]], spans[others[1]] = short(), long()
        elif spans == [length] * n:
            spans[others[0]] = short() if mode == "extend" else long()
    # Opposite prefixes must jointly reach the nearest half of the target.
    # Otherwise an entire central-step orbit could be absent before erasure.
    for i in range(n // 2):
        j = n - 1 - i
        if max(spans[i], spans[j]) < (length + 1) // 2:
            spans[int(rng.choice((i, j)))] = (length + 1) // 2
    return spans


def _generate_pair(rng, direction_id, n, witness=False, test=False,
                   avoid_lengths=(), equal=False):
    axis = AXES[direction_id // 2]
    sign = 1 if direction_id % 2 == 0 else -1
    dr, dc = sign * axis[0], sign * axis[1]
    gaps = [2] * (n - 1)
    if witness:
        gaps[int(rng.choice((0, n - 2)))] += int(rng.choice((5, 7)))
    elif equal:
        gaps = [int(rng.integers(2, min(4, 20 // (n - 1)) + 1))] * (n - 1)
    else:
        for _ in range(int(rng.integers(1, max(2, 22 - sum(gaps))))):
            gaps[int(rng.integers(n - 1))] += 1
        if len(set(gaps)) == 1:
            gaps[0] += 1
    span = sum(gaps)
    capacity = min(26, 28 - span if all(axis) else 26)
    choices = list(range(3, min(capacity - 1, 20 if test else 12) + 1))
    fresh = [x for x in choices if x not in avoid_lengths]
    length = int(rng.choice(fresh or choices))
    mode = "witness" if witness else str(rng.choice(
        ("mixed", "extend", "trim", "one_extend", "one_trim")))
    spans = _lengths(rng, n, length, capacity, mode)
    background, marker = (int(x) for x in rng.choice(10, 2, replace=False))
    pool = [c for c in range(10) if c not in (background, marker)]
    k = int(rng.integers(2, min(8, ((n + 1) // 2) * ((length + 1) // 2) - 1) + 1))
    payload = [int(x) for x in rng.choice(pool, k, replace=False)]
    clean = _clean_table(rng, n, length, payload)
    visible = np.array([[t < spans[i] for t in range(length)] for i in range(n)])
    # Block, band, walk and scattered erasures are expressed in rank coordinates.
    damage = np.zeros((n, length), dtype=bool)
    style = int(rng.integers(4))
    density = float(rng.uniform(.12, .38) if test else rng.uniform(.04, .20))
    if style == 0:
        damage = rng.random((n, length)) < density
    elif style == 1:
        r0, t0 = int(rng.integers(n)), int(rng.integers(length))
        damage[r0:min(n, r0 + int(rng.integers(1, 4))),
               t0:min(length, t0 + int(rng.integers(1, max(3, length // 2))))] = True
    elif style == 2:
        t = int(rng.integers(length))
        damage[:, t:min(length, t + int(rng.integers(1, 3)))] = True
    else:
        i, t = int(rng.integers(n)), int(rng.integers(length))
        for _ in range(int(rng.integers(2, max(3, n * length // 3)))):
            damage[i, t] = True
            di, dt = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            i, t = int(np.clip(i + di, 0, n - 1)), int(np.clip(t + dt, 0, length - 1))
    visible &= ~damage
    for orbit in _orbits(n, length):
        if not any(visible[p] for p in orbit):
            eligible = [p for p in orbit if p[1] < spans[p[0]]]
            assert eligible, "a positive prefix must witness each reflected step"
            visible[eligible[int(rng.integers(len(eligible)))]] = True
    # A single diagonal survivor forces both reflections. Some easier extra
    # training pairs retain direct-copy gaps, while every test has this witness.
    if witness or test or rng.random() < .35:
        orbit = ((0, 0), (n - 1, 0), (0, length - 1), (n - 1, length - 1))
        for p in orbit:
            visible[p] = False
        visible[(int(rng.choice((0, n - 1))), 0)] = True
    else:
        visible[0, 0] = False
        visible[n - 1, 0] = True
    positions = [0] + [int(x) for x in np.cumsum(gaps)]
    starts = [(p * dc, -p * dr) for p in positions]
    cells = [(r + s * dr, c + s * dc)
             for (r, c), m in zip(starts, spans) for s in range(max(m, length) + 2)]
    low = [min(p[d] for p in cells) for d in (0, 1)]
    box = [max(p[d] for p in cells) - low[d] + 1 for d in (0, 1)]
    occupied = max(sum(spans) + 2 * n, n * (length + 2))
    shapes = [(h, w) for h in range(box[0], 31) for w in range(box[1], 31)
              if h * w > 2 * occupied]
    assert shapes
    if rng.random() < .35:
        compact = [s for s in shapes if s[0] <= box[0] + 3 and s[1] <= box[1] + 3]
        shapes = compact or shapes
    shape = shapes[int(rng.integers(len(shapes)))]
    offsets = [int(rng.integers(shape[d] - box[d] + 1)) for d in (0, 1)]
    starts = [(r - low[0] + offsets[0], c - low[1] + offsets[1]) for r, c in starts]
    a = dict(starts=starts, direction=(dr, dc))
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for i, m in enumerate(spans):
        for t in (-2, -1):
            inp[_point(a, i, t)] = out[_point(a, i, t)] = marker
        for t in range(length):
            out[_point(a, i, t)] = clean[i, t]
        for t in range(m):
            if t < length:
                value = clean[i, t] if visible[i, t] else marker
            else:
                value = int(rng.choice(payload + [marker]))
            inp[_point(a, i, t)] = value
    return {"input": inp, "output": out}


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    directions = [int(x) for x in rng.permutation(8)]
    regime = int(rng.integers(3))
    puzzle, seen, train_lengths = {"train": [], "test": []}, set(), set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            witness = split == "train" and index == 0
            n = 5 if witness else int(rng.choice((3, 7) if split == "train" and index == 1
                                               else (3, 5, 7, 9)))
            if regime == 0:
                direction = directions[0]
            elif regime == 1:
                direction = directions[index % 8] if split == "train" else directions[(index + 2) % 8]
            else:
                direction = directions[0] if split == "train" else directions[0] ^ 1
            avoid = train_lengths if split == "test" or index == 1 else ()
            for _ in range(64):
                pair = _generate_pair(rng, direction, n, witness, split == "test", avoid,
                                      equal=split == "train" and index == 1)
                key = _signature(pair["input"])
                if key not in seen:
                    # This also rejects any accidentally ambiguous line parsing.
                    try:
                        predicted = solve(pair["input"])
                    except ValueError:
                        continue
                    if np.array_equal(predicted, pair["output"]):
                        break
            else:
                raise RuntimeError("distinct interpretable pair construction exhausted")
            puzzle[split].append(pair)
            seen.add(key)
            if split == "train":
                train_lengths.add(_parse(pair["input"])["length"])
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    a = _parse(inp)
    v = _visible_table(inp, a)
    n, length = v.shape
    diag_only = along_needed = rank_needed = 0
    for i, t in zip(*np.nonzero(v < 0)):
        across = v[n - 1 - i, t] >= 0
        along = v[i, length - 1 - t] >= 0
        diagonal = v[n - 1 - i, length - 1 - t] >= 0
        along_needed += not across
        rank_needed += not along
        diag_only += not across and not along and diagonal
    distances = [abs(2 * x - a["keys"][0] - a["keys"][-1]) for x in a["keys"]]
    nearest = [i for i, d in enumerate(distances) if d == min(distances)]
    a.update(unknown=int((v < 0).sum()), diagonal_only=int(diag_only),
             along_needed=int(along_needed), rank_needed=int(rank_needed),
             extends=sum(m < length for m in a["lengths"]),
             trims=sum(m > length for m in a["lengths"]),
             nonmedian=sorted(a["lengths"])[n // 2] != length,
             geometric_witness=len(nearest) == 1 and nearest[0] != n // 2
             and a["lengths"][nearest[0]] != length)
    return a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (_grid(pair[k]) for k in ("input", "output"))
            assert inp.shape == out.shape
            key = _signature(inp)
            assert key not in seen
            seen.add(key)
            a = _features(inp)
            visible = _visible_table(inp, a)
            table = np.array([[out[_point(a, i, t)] for t in range(a["length"])]
                              for i in range(a["n"])])
            allowed = np.zeros(out.shape, dtype=bool)
            for i in range(a["n"]):
                for t in (-2, -1):
                    p = _point(a, i, t)
                    assert inp[p] == out[p] == a["marker"]
                    allowed[p] = True
                for t in range(a["length"]):
                    allowed[_point(a, i, t)] = True
            assert np.all(out[~allowed] == a["background"]), "tails or background changed incorrectly"
            assert not np.any(np.isin(table, (a["background"], a["marker"])))
            assert np.array_equal(table, table[::-1]) and np.array_equal(table, table[:, ::-1])
            assert np.array_equal(table[visible >= 0], visible[visible >= 0]), "visible payload changed"
            for orbit in _orbits(a["n"], a["length"]):
                assert any(visible[p] >= 0 for p in orbit), "unwitnessed orbit"
            assert a["unknown"] and (a["extends"] or a["trims"])
            assert np.array_equal(solve(inp), out)
            if split == "train":
                training.append(a)
            else:
                assert a["diagonal_only"] > 0
    assert len({a["n"] for a in training}) >= 2
    assert len({a["length"] for a in training}) >= 2
    assert any(a["nonmedian"] and a["geometric_witness"] for a in training)
    assert any(a["extends"] and a["trims"] and a["diagonal_only"] for a in training)
    assert any(len(set(a["gaps"])) == 1 for a in training)


def self_test():
    """A bounded 42-seed campaign, counterexample checks and equivariance."""
    rng = np.random.default_rng(71093)
    shapes, structures, signatures, counts, lengths, directions = set(), set(), set(), set(), set(), set()
    backgrounds, markers, palette_sizes, gap_vectors = set(), set(), set(), set()
    operation_patterns, diagonal_cases = set(), 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        sig = tuple(_signature(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))
        assert sig == tuple(_signature(p[k]) for s in ("train", "test") for p in again[s] for k in ("input", "output"))
        signatures.add(sig)
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                inp, out = pair["input"], pair["output"]
                a = _features(inp)
                shapes.add(inp.shape)
                counts.add(a["n"])
                lengths.add(a["length"])
                directions.add(a["direction"])
                backgrounds.add(a["background"])
                markers.add(a["marker"])
                gap_vectors.add(a["gaps"])
                operation_patterns.add((a["extends"] > 0, a["trims"] > 0))
                diagonal_cases += a["diagonal_only"] > 0
                table = np.array([[out[_point(a, i, t)] for t in range(a["length"])] for i in range(a["n"])])
                palette_sizes.add(len(np.unique(table)))
                roles = {}
                norm = tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add((inp.shape, norm))
                if index == 0:
                    palette = rng.permutation(10).astype(np.int8)
                    assert np.array_equal(solve(palette[inp]), palette[out])
                    for turns in range(4):
                        for flip in (False, True):
                            left, right = np.rot90(inp, turns), np.rot90(out, turns)
                            if flip:
                                left, right = np.fliplr(left), np.fliplr(right)
                            assert np.array_equal(solve(left), right)
                # Removing A leaves unfilled colored material; removing B
                # leaves an incorrect domain, irrespective of the color rule.
                assert a["unknown"] > 0 and a["lengths"] != (a["length"],) * a["n"]
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 60 and counts == {3, 5, 7, 9}
    assert len(lengths) >= 10 and len(directions) == 8
    assert backgrounds == markers == set(range(10))
    assert len(gap_vectors) >= 50 and len(palette_sizes) >= 5
    assert operation_patterns == {(True, True), (True, False), (False, True)}
    assert diagonal_cases >= 84
    # Exercise the rule beyond the sampler's 20-cell reference ceiling.
    # A 28-cell body plus its cap spans an entire dimension of a 7x30 grid.
    clean = _clean_table(rng, 3, 28, [2, 3])
    edge_in = np.zeros((7, 30), dtype=np.int8)
    edge_out = edge_in.copy()
    for i, (row, span) in enumerate(zip((0, 3, 6), (1, 28, 14))):
        edge_in[row, :2] = edge_out[row, :2] = 8
        edge_in[row, 2:2 + span] = clean[i, :span]
        edge_out[row, 2:] = clean[i]
    edge_in[6, 2] = edge_in[3, 2] = 8
    assert np.array_equal(solve(edge_in), edge_out)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(9137, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(0)
    for kind in ("cap", "payload", "tail"):
        bad = {s: [{k: g.copy() for k, g in p.items()} for p in sample[s]] for s in sample}
        p = bad["test"][0]
        a = _parse(p["input"])
        if kind == "cap":
            point = _point(a, 0, -2)
        elif kind == "payload":
            point = _point(a, 0, 0)
        else:
            point = next((r, c) for r in range(p["output"].shape[0])
                         for c in range(p["output"].shape[1]) if p["output"][r, c] == a["background"])
        p["output"][point] = (int(p["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"validator accepted damaged {kind}")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures),
                grid_shapes=len(shapes), normalized_inputs=len(structures),
                tape_counts=sorted(counts), reference_lengths=sorted(lengths),
                directed_axes=len(directions), background_colors=sorted(backgrounds),
                marker_colors=sorted(markers), palette_sizes=sorted(palette_sizes),
                gap_vectors=len(gap_vectors), diagonal_only_pairs=diagonal_cases)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
