"""Unanimity rail: independently register colored tapes on their shared stripe.

Only NumPy and the Python standard library are used. All role and alignment
information is visible in the input; the solver does not consult demonstrations.
"""

from collections import Counter
import json

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or a.dtype.kind not in "iu" or
            not all(1 <= n <= 30 for n in a.shape) or
            np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, tuple(int(x) for x in a.flat)


def _canonical_parse(a):
    """Recognize upward tapes using only geometry and the boundary ticks."""
    h, w = a.shape
    if h < 6 or w < 5:
        return None
    bg = int(a[0, 0])
    if any(int(a[y, x]) != bg for y, x in ((0, w-1), (h-1, 0), (h-1, w-1))):
        return None
    border = np.zeros(a.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    ticks = np.argwhere(border & (a != bg))
    if len(ticks) != 2:
        return None
    (r1, c1), (r2, c2) = ticks.tolist()
    if r1 != r2 or c1 != 0 or c2 != w-1 or not 0 < r1 < h-1:
        return None
    marker = int(a[r1, 0])
    if a[r2, c2] != marker or np.count_nonzero(a == marker) != 2:
        return None
    raw = []
    for x in range(1, w-1):
        ys = np.flatnonzero(a[:, x] != bg)
        if not len(ys):
            continue
        if ys[0] <= r1 or not np.array_equal(ys, np.arange(ys[0], ys[-1]+1)):
            return None
        values = tuple(int(v) for v in a[ys, x])
        if len(values) < 3:
            return None
        raw.append((x, int(ys[0]), values))
    if len(raw) < 3:
        return None
    common = set(raw[0][2])
    for _, _, values in raw[1:]:
        common.intersection_update(values)
    if len(common) != 1:
        return None
    color = next(iter(common))
    lanes = []
    for x, start, values in raw:
        if values.count(color) != 1:
            return None
        rank = values.index(color)
        target = r1-rank
        if target < 1 or target+len(values) > h-1:
            return None
        lanes.append((x, start, values, rank))
    return {"bg": bg, "marker": marker, "datum": r1,
            "common": color, "lanes": tuple(lanes), "grid": a}


def _parse(input_grid):
    a = _grid(input_grid)
    candidates = []
    for turn in range(4):
        parsed = _canonical_parse(np.rot90(a, turn))
        if parsed is not None:
            parsed["turn"] = turn
            candidates.append(parsed)
    if len(candidates) != 1:
        raise ValueError("input must identify exactly one rail, motion side, and unanimous color")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _parse(input_grid)
    out = p["grid"].copy()
    # Clear first: overlapping old/new footprints must not erase another part
    # of the same translated tape.
    for x, start, values, _ in p["lanes"]:
        out[start:start+len(values), x] = p["bg"]
    for x, _, values, rank in p["lanes"]:
        start = p["datum"]-rank
        out[start:start+len(values), x] = values
    return np.rot90(out, -p["turn"]).copy()


def _make_tapes(rng, n, cap, palette, target, witness, long_tape):
    others = [int(c) for c in rng.permutation([c for c in palette if c != target])]
    near = 3 if witness else int(rng.integers(1, 4))
    omissions = {}
    missing = [0, n//2, n-1] if witness else rng.permutation(n).tolist()[:near]
    for color, lane in zip(others[:near], missing):
        omissions[color] = int(lane)
    for color in others[near:]:
        # Every distractor has at least one forbidden lane. Thus random fill
        # cannot accidentally introduce a second unanimous color.
        choices = [i for i in range(n)
                   if sum(v == i for v in omissions.values()) < len(others)-1]
        omissions[color] = int(rng.choice(choices))
    tapes = []
    for i in range(n):
        allowed = [c for c in others if omissions[c] != i]
        body = [c for c in others[:near] if omissions[c] != i]
        if witness and omissions[others[0]] != i:
            body.append(others[0])  # A frequent decoy defeats global frequency.
        minimum = max(3, len(body)+1)
        length = int(rng.integers(minimum, cap+1))
        if long_tape and i == 0:
            length = int(rng.integers(7, cap+1))
        body += [int(c) for c in rng.choice(allowed, size=length-1-len(body))]
        rng.shuffle(body)
        rank = int(rng.integers(0, length))
        if witness:
            if i == 0:
                rank = 0
            elif i == 1:
                rank = length-1
            elif i == 2:
                rank = int(rng.integers(1, length-1))
        body.insert(rank, int(target))
        tapes.append(tuple(body))
    return tapes


def _build_pair(rng, bg, marker, turn, mode, split, index, target):
    witness = split == "train" and index == 0
    n = int(rng.integers(6, 10)) if split == "test" and mode == "lanes" else int(rng.integers(3, 6))
    cap = 9 if split == "test" and mode == "length" else 6
    available = [c for c in range(10) if c not in (bg, marker, target)]
    palette = [target] + [int(c) for c in rng.choice(available, size=int(rng.integers(4, 8)), replace=False)]
    tapes = _make_tapes(rng, n, cap, palette, target, witness,
                        split == "test" and mode == "length")
    ranks = [t.index(target) for t in tapes]
    lengths = [len(t) for t in tapes]
    datum = max(ranks)+int(rng.integers(1, 4))
    minimum_h = datum+max(lengths)+3
    h = int(rng.integers(minimum_h, min(30, minimum_h+8)+1))
    w = int(rng.integers(n+2, min(30, n+16)+1))
    if witness:
        first = int(rng.integers(1, w-n))
        cols = [first, first+1] + sorted(int(c) for c in rng.choice(
            np.arange(first+2, w-1), size=n-2, replace=False))
    else:
        cols = sorted(int(c) for c in rng.choice(np.arange(1, w-1), size=n, replace=False))
    if split == "train" and index == 0:
        style = "top"
    elif split == "train" and index == 1:
        style = "bottom"
    elif mode == "stagger":
        style = "stagger" if split == "test" else str(rng.choice(("top", "bottom")))
    else:
        style = str(rng.choice(("top", "bottom", "stagger")))
    if style == "top":
        start = int(rng.integers(datum+1, h-max(lengths)))
        starts = [start]*n
    elif style == "bottom":
        end = int(rng.integers(datum+max(lengths), h-1))
        starts = [end-length+1 for length in lengths]
    else:
        for _ in range(100):
            starts = [int(rng.integers(datum+1, h-length)) for length in lengths]
            if len(set(starts)) > 1 and len({s+l for s, l in zip(starts, lengths)}) > 1:
                break
        else:
            raise RuntimeError("could not stagger the tapes")
    a = np.full((h, w), bg, dtype=int)
    a[datum, [0, w-1]] = marker
    b = a.copy()
    for x, start, tape, rank in zip(cols, starts, tapes, ranks):
        a[start:start+len(tape), x] = tape
        stop = datum-rank
        b[stop:stop+len(tape), x] = tape
    return {"input": np.rot90(a, -turn).copy(),
            "output": np.rot90(b, -turn).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    bg, marker = (int(c) for c in rng.choice(10, size=2, replace=False))
    turn = int(rng.integers(0, 4))
    mode = str(rng.choice(("stagger", "lanes", "length")))
    colors = [c for c in range(10) if c not in (bg, marker)]
    first_targets = [int(c) for c in rng.choice(colors, size=2, replace=False)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            target = first_targets[i] if split == "train" and i < 2 else int(rng.choice(colors))
            for _ in range(30):
                pair = _build_pair(rng, bg, marker, turn, mode, split, i, target)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    # Demonstration order carries no convention.
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(p):
    lanes = p["lanes"]
    sets = [set(values) for _, _, values, _ in lanes]
    all_colors = set.union(*sets)
    near_missing = set()
    for color in all_colors-{p["common"]}:
        missing = [i for i, s in enumerate(sets) if color not in s]
        if len(missing) == 1:
            i = missing[0]
            near_missing.add("first" if i == 0 else "last" if i == len(lanes)-1 else "interior")
    touching_unequal = False
    for left, right in zip(lanes, lanes[1:]):
        x, s, values, rank = left
        xx, ss, vv, rr = right
        if (xx == x+1 and max(s, ss) < min(s+len(values), ss+len(vv))
                and s+rank != ss+rr):
            touching_unequal = True
    counts = Counter(c for _, _, values, _ in lanes for c in values)
    return {"near_missing": near_missing, "touching_unequal": touching_unequal,
            "frequency_decoy": max(counts.values()) > counts[p["common"]],
            "interior": any(0 < rank < len(v)-1 for _, _, v, rank in lanes),
            "starts": {s for _, s, _, _ in lanes},
            "ends": {s+len(v) for _, s, v, _ in lanes},
            "max_length": max(len(v) for _, _, v, _ in lanes)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, roles = set(), set()
    parsed = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            p = _parse(inp)
            parsed[split].append(p)
            roles.add((p["bg"], p["marker"], p["turn"]))
            canonical_out = np.rot90(out, p["turn"])
            # Inspect returned tape sequences independently of solve's writes.
            occupied = np.zeros(p["grid"].shape, dtype=bool)
            occupied[p["datum"], [0, -1]] = True
            for x, start, values, rank in p["lanes"]:
                ys = np.flatnonzero(canonical_out[:, x] != p["bg"])
                _require(len(ys) == len(values), "per-lane length not conserved")
                _require(np.array_equal(ys, np.arange(ys[0], ys[0]+len(values))), "tape fragmented")
                _require(tuple(canonical_out[ys, x]) == values, "color sequence changed")
                _require(int(ys[0])+rank == p["datum"], "shared color not registered on rail")
                _require(int(ys[0]) < start, "tape did not move toward rail")
                occupied[ys, x] = True
                _require(canonical_out[p["datum"], x] == p["common"], "row lacks unanimity")
            _require(np.all(canonical_out[~occupied] == p["bg"]), "uncleared cells or extra objects")
            _require(np.all(canonical_out[p["datum"], [0, -1]] == p["marker"]), "ticks changed")
            _require(np.array_equal(solve(inp), out), "solve disagrees")
    _require(len(roles) == 1, "puzzle changed background, tick color, or direction")
    train = parsed["train"]
    features = [_features(p) for p in train]
    _require(set.union(*(f["near_missing"] for f in features)) == {"first", "interior", "last"},
             "missing whole-line agreement contrasts")
    _require(any(f["touching_unequal"] for f in features), "missing independent-motion witness")
    _require(any(f["frequency_decoy"] for f in features), "frequency shortcut not contradicted")
    _require(any(f["interior"] for f in features), "missing interior registration example")
    _require(len({p["common"] for p in train}) >= 2, "target identity appears fixed")
    # Each generated puzzle deliberately withholds one structural arrangement.
    tests = parsed["test"]
    more_lanes = all(len(p["lanes"]) > max(len(t["lanes"]) for t in train) for p in tests)
    longer = all(_features(p)["max_length"] > max(f["max_length"] for f in features) for p in tests)
    stagger = (all(len(f["starts"]) == 1 or len(f["ends"]) == 1 for f in features) and
               all(len(_features(p)["starts"]) > 1 and len(_features(p)["ends"]) > 1 for p in tests))
    _require(more_lanes or longer or stagger, "missing documented structural train/test shift")


def _same(left, right):
    return all(len(left[s]) == len(right[s]) and all(
        np.array_equal(p[k], q[k]) for p, q in zip(left[s], right[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded semantic, counterfactual, count, and 42-seed diversity checks."""
    shapes, roles, structures, targets, lane_counts, lengths = set(), set(), set(), set(), set(), set()
    fingerprints, shifts = set(), set()
    regimes = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p[k]) for s in ("train", "test")
                               for p in puzzle[s] for k in ("input", "output")))
        pp = {s: [_parse(p["input"]) for p in puzzle[s]] for s in ("train", "test")}
        tr, te = pp["train"], pp["test"]
        witness = next(p for p in tr if _features(p)["touching_unequal"] and
                       _features(p)["interior"])
        blank = np.full(witness["grid"].shape, witness["bg"], dtype=int)
        blank[witness["datum"], [0, -1]] = witness["marker"]
        no_equality = blank.copy()
        rigid = blank.copy()
        delta = min(start+rank-witness["datum"]
                    for _, start, _, rank in witness["lanes"])
        for x, start, values, _ in witness["lanes"]:
            no_equality[witness["datum"]:witness["datum"]+len(values), x] = values
            rigid[start-delta:start-delta+len(values), x] = values
        expected = np.rot90(solve(np.rot90(witness["grid"], -witness["turn"])), witness["turn"])
        _require(not np.array_equal(no_equality, expected), "omitting A did not change the answer")
        _require(not np.array_equal(rigid, expected), "omitting B did not change the answer")
        if len(te[0]["lanes"]) > max(len(p["lanes"]) for p in tr):
            regimes["lanes"] += 1
        elif _features(te[0])["max_length"] > max(_features(p)["max_length"] for p in tr):
            regimes["length"] += 1
        else:
            regimes["stagger"] += 1
        for s in ("train", "test"):
            for pair, p in zip(puzzle[s], pp[s]):
                shapes.add(pair["input"].shape)
                roles.add((p["bg"], p["marker"], p["turn"]))
                targets.add(p["common"])
                lane_counts.add(len(p["lanes"]))
                for x, start, values, rank in p["lanes"]:
                    lengths.add(len(values))
                    shifts.add(start+rank-p["datum"])
                mapping = {}
                normalized = tuple(mapping.setdefault(int(c), len(mapping)) for c in p["grid"].flat)
                structures.add((p["grid"].shape, normalized))
        # Equivariance: every role color can be permuted and every direction
        # rotated, without changing the choice of stripe or displacement.
        pair = puzzle["test"][0]
        perm = np.random.default_rng(seed+9000).permutation(10)
        for turn in range(4):
            inp = np.rot90(perm[pair["input"]], turn)
            expected = np.rot90(perm[pair["output"]], turn)
            _require(np.array_equal(solve(inp), expected), "rotation/palette equivariance failed")
    _require(len(fingerprints) == 42, "seed variation collapsed")
    _require(len(shapes) >= 65 and len(structures) >= 190, "insufficient geometric variation")
    _require(len(roles) >= 30 and len({r[2] for r in roles}) == 4, "insufficient role/direction variation")
    _require(len(targets) == 10 and {3, 4, 5, 6, 7, 8, 9} <= lane_counts, "insufficient consensus variation")
    _require({3, 4, 5, 6, 7, 8, 9} <= lengths and len(shifts) >= 10, "insufficient tape variation")
    _require(all(regimes[k] >= 5 for k in ("lanes", "length", "stagger")), "missing shift regime")
    # Exercise specified members beyond the sampler's usual size bands.
    wide = np.zeros((30, 15), dtype=int)
    expected = wide.copy()
    wide[10, [0, -1]] = expected[10, [0, -1]] = 8
    sequences = ([2]+[3]*12, [4]*8+[2]+[4]*4, [1]*5+[2]+[1]*7)
    for i in range(12):
        seq = sequences[i % 3]
        wide[15:28, i+1] = seq
        top = 10-seq.index(2)
        expected[top:top+13, i+1] = seq
    _require(np.array_equal(solve(wide.tolist()), expected), "long/wide family member failed")
    invalid = wide.copy()
    invalid[16, 1] = 2  # Two occurrences of the unanimous color in one tape.
    try:
        solve(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted ambiguous repeated registration color")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "explicit counts changed")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    _require(_same(generate_puzzle(730), generate_puzzle(730)), "nondeterminism")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "canonical_structures": len(structures), "role_direction_assignments": len(roles),
            "shift_regimes": dict(regimes), "lengths": sorted(lengths), "lane_counts": sorted(lane_counts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
