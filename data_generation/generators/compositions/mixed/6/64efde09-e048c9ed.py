"""Registered arithmetic emitters. All solving conventions are visible pixels.

Only standard library and NumPy are used. Generation keeps no external state,
writes no files, and constructs answers before the input-only decoder checks them.
"""

import hashlib
import numpy as np


HOST_SEEDS = (0, 522592028, 785381722, 934109888, 167463258, 210785587,
              182062913, 725316780, 383493213, 918938113, 240444671, 575638758,
              440007262, 126663896, 698687817, 830471077, 687889237, 782133181,
              304686501, 700568949, 520299615, 196190426, 198244478, 666663301,
              400088448, 889175626, 217111893, 363616684, 742211890, 721979996,
              831457529, 215662301, 105981784, 295682920, 847213696, 906902038,
              951745648, 713098996, 718139841, 210214844, 255982570, 948420472)
SQUARE_DIGITS = (0, 1, 4, 5, 6, 9)
STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _add(p, direction, scale=1):
    return p[0] + scale * direction[0], p[1] + scale * direction[1]


def _ray(start, direction, shape):
    result = []
    r, c = start
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        result.append((r, c))
        r += direction[0]
        c += direction[1]
    return tuple(result)


def _emitter(origin, u, v, length, shape):
    body = tuple(_add(origin, u, j) for j in range(length))
    flag = _add(origin, v)
    return {"origin": origin, "u": u, "v": v, "length": length,
            "body": body, "flag": flag, "cells": frozenset(body + (flag,)),
            "out": _ray(_add(origin, v, 2), v, shape)}


def _back(e, digit, shape):
    v = e["v"]
    return _ray(_add(e["body"][digit], v, -1), (-v[0], -v[1]), shape)


def _components(a, bg):
    unseen = set(map(tuple, np.argwhere(a != bg).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, component = [start], []
        while stack:
            p = stack.pop()
            component.append(p)
            for d in STEPS:
                q = _add(p, d)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(component)
    return result


def _read_emitter(a, component):
    """A repeated-color backbone and one different-color end flag."""
    colors = {}
    for p in component:
        colors.setdefault(int(a[p]), []).append(p)
    if len(colors) != 2:
        return None
    repeated = [(c, ps) for c, ps in colors.items() if 2 <= len(ps) <= 12]
    singles = [(c, ps[0]) for c, ps in colors.items() if len(ps) == 1]
    if len(repeated) != 1 or len(singles) != 1:
        return None
    body_color, body = repeated[0]
    flag_color, flag = singles[0]
    length = len(body)
    if len({p[0] for p in body}) == 1:
        u = (0, 1)
    elif len({p[1] for p in body}) == 1:
        u = (1, 0)
    else:
        return None
    body = sorted(body)
    if tuple(body) != tuple(_add(body[0], u, j) for j in range(length)):
        return None
    matches = []
    for origin, direction in ((body[0], u), (body[-1], (-u[0], -u[1]))):
        v = (flag[0] - origin[0], flag[1] - origin[1])
        if abs(v[0]) + abs(v[1]) == 1 and v[0]*u[0] + v[1]*u[1] == 0:
            matches.append(_emitter(origin, direction, v, length, a.shape))
    if len(matches) != 1:
        return None
    e = matches[0]
    e["roles"] = (body_color, flag_color)
    return e


def _decode(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    bg = int(values[np.argmax(counts)])
    emitters, clues = [], []
    for component in _components(a, bg):
        e = _read_emitter(a, component)
        if e is not None:
            emitters.append(e)
        else:
            for r, c in component:
                _require(r in (0, a.shape[0]-1) or c in (0, a.shape[1]-1),
                         "nonboundary clue or malformed emitter")
                clues.append(((r, c), int(a[r, c])))
    _require(2 <= len(emitters) <= 6 and clues, "need emitters and code clues")
    roles = {e["roles"] for e in emitters}
    _require(len(roles) == 1, "inconsistent registration colors")
    structural = set(next(iter(roles)))
    _require(all(c not in structural and c != bg for _, c in clues), "color role collision")
    occupied = set().union(*(e["cells"] for e in emitters))
    corridors = {}
    for index, e in enumerate(emitters):
        for digit in range(min(10, e["length"])):
            ray = _back(e, digit, a.shape)
            if len(ray) >= 2 and not occupied.intersection(ray):
                if all(int(a[p]) == bg for p in ray[:-1]):
                    corridors.setdefault(ray[-1], []).append((index, digit))
    code, observations = {}, []
    for p, color in clues:
        matches = corridors.get(p, [])
        _require(len(matches) == 1, "ambiguous or unregistered boundary clue")
        index, digit = matches[0]
        _require(digit not in code or code[digit] == color, "conflicting pooled entries")
        code[digit] = color
        observations.append((index, digit, p))
    used = set()
    for e in emitters:
        _require(len(e["out"]) >= 2, "empty active corridor")
        _require(not occupied.intersection(e["out"]) and not used.intersection(e["out"]),
                 "active rays intersect objects or each other")
        _require(all(int(a[p]) == bg for p in e["out"]), "active corridor is occupied")
        used.update(e["out"])
    return a, bg, emitters, code, observations


def _paint(a, emitters, code, rule="square"):
    out = a.copy()
    for e in emitters:
        length = e["length"]
        if rule == "square":
            digit = (length-1)**2 % 10
        elif rule == "length":
            digit = length % 10
        elif rule == "adjacency":
            digit = (length-1) % 10
        elif rule == "square_length":
            digit = length**2 % 10
        else:
            raise ValueError("unknown arithmetic ablation")
        if digit in code:
            for p in e["out"]:
                out[p] = code[digit]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pool back-side boundary slots, then address the pool by squared adjacency."""
    a, _, emitters, code, _ = _decode(input_grid)
    return _paint(a, emitters, code)


def _directions(index):
    u = STEPS[index % 4]
    sign = 1 if index < 4 else -1
    return u, (sign*u[1], -sign*u[0])


def _place(rng, lengths, shape, demonstration):
    """Bounded random geometry, with no retained candidate bank."""
    for _ in range(24):
        placed, occupied, outgoing = [], set(), set()
        for index, length in enumerate(lengths):
            for attempt in range(90):
                t = int(rng.integers(8))
                # First demonstration exposes reversed registration and an axis swap.
                if demonstration == 0 and index < 3:
                    t = (0, 6, 1)[index]
                u, v = _directions(t)
                local = ((0, 0), _add((0, 0), u, length-1), v)
                rlow, rhigh = -min(p[0] for p in local), shape[0]-max(p[0] for p in local)
                clow, chigh = -min(p[1] for p in local), shape[1]-max(p[1] for p in local)
                origin = (int(rng.integers(rlow, rhigh)), int(rng.integers(clow, chigh)))
                e = _emitter(origin, u, v, length, shape)
                if len(e["out"]) < 2:
                    continue
                if e["cells"] & (occupied | outgoing) or set(e["out"]) & (occupied | outgoing):
                    continue
                if any(_add(p, d) in occupied for p in e["cells"] for d in STEPS):
                    continue
                placed.append(e)
                occupied.update(e["cells"])
                outgoing.update(e["out"])
                break
            else:
                break
        if len(placed) == len(lengths):
            yield placed


def _clue_options(emitters, shape):
    occupied = set().union(*(e["cells"] for e in emitters))
    outgoing = set().union(*(set(e["out"]) for e in emitters))
    alignments = {}
    for index, e in enumerate(emitters):
        for digit in range(min(10, e["length"])):
            ray = _back(e, digit, shape)
            if len(ray) >= 2 and not occupied.intersection(ray):
                alignments.setdefault(ray[-1], []).append((index, digit))
    options = {}
    for p, matches in alignments.items():
        if len(matches) != 1 or p in outgoing:
            continue
        # Clues may touch one another but must not join an emitter component.
        if any(_add(p, d) in occupied for d in STEPS):
            continue
        index, digit = matches[0]
        options.setdefault(digit, []).append((index, p))
    return options


def _choose_observations(rng, options, code, demonstration):
    if not all(d in options for d in code):
        return None
    observations = []
    for digit in sorted(code):
        choices = options[digit]
        if demonstration == 0:
            choices = [entry for entry in choices if entry[0] == 0]
            if not choices:
                return None
        index, p = choices[int(rng.integers(len(choices)))]
        observations.append((index, digit, p))
    if demonstration != 0:
        # Make distributed pooling visible, rather than merely possible.
        if len({index for index, _, _ in observations}) == 1:
            old_index = observations[0][0]
            alternatives = [(i, d, p) for d in code for i, p in options[d] if i != old_index]
            if not alternatives:
                return None
            entry = alternatives[int(rng.integers(len(alternatives)))]
            observations = [x for x in observations if x[1] != entry[1]] + [entry]
        if rng.random() < .4:
            old = {(i, d) for i, d, _ in observations}
            extras = [(i, d, p) for d in code for i, p in options[d] if (i, d) not in old]
            if extras:
                observations.append(extras[int(rng.integers(len(extras)))])
    return observations


def _make_pair(rng, lengths, demonstration, omitted):
    for _ in range(28):
        # Padding/size distributions overlap across the split.
        shape = tuple(map(int, rng.integers(max(17, max(lengths)+4), 31, size=2)))
        if rng.random() < .2:
            shape = (shape[0], shape[0])
        palette = list(map(int, rng.permutation(10)))
        bg, body_color, flag_color = palette[:3]
        digits = set(SQUARE_DIGITS) - set(omitted)
        extras = list(map(int, rng.choice((2, 3, 7, 8), int(rng.integers(1, 3)), replace=False)))
        digits.update(extras)
        digits = sorted(digits)
        signals = palette[3:]
        # Distinct square-digit colors expose addressing; extra entries may repeat.
        square_order = list(map(int, rng.permutation(np.array(SQUARE_DIGITS))))
        mapping = {d: signals[i] for i, d in enumerate(square_order)}
        code = {d: mapping[d] if d in mapping else int(rng.choice(signals)) for d in digits}
        for emitters in _place(rng, lengths, shape, demonstration):
            options = _clue_options(emitters, shape)
            observations = _choose_observations(rng, options, code, demonstration)
            if observations is None:
                continue
            if demonstration == 1:
                # Distributed placement alone need not require remote lookup.
                # A selected entry must actually be absent from its querying bar.
                if not any((e["length"]-1)**2 % 10 in code and
                           not any(i == index and d == (e["length"]-1)**2 % 10
                                   for i, d, _ in observations)
                           for index, e in enumerate(emitters)):
                    continue
            inp = np.full(shape, bg, dtype=np.int8)
            for e in emitters:
                for p in e["body"]:
                    inp[p] = body_color
                inp[e["flag"]] = flag_color
            for _, digit, p in observations:
                inp[p] = code[digit]
            out = _paint(inp, emitters, code)
            # Check construction against a fresh pixel reconstruction.
            try:
                decoded = _decode(inp)
            except ValueError:
                continue
            if not np.array_equal(_paint(decoded[0], decoded[2], decoded[3]), out):
                continue
            if np.array_equal(inp, out):
                continue
            return {"input": inp, "output": out}
    raise RuntimeError("bounded layout search exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " out of range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    _require(nt > ne, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Every test length is taught even with the minimum two demonstrations.
    first = [int(rng.choice((10, 12))), 2, int(rng.choice((3, 9))),
             int(rng.choice((4, 8))), int(rng.choice((5, 7)))]
    second = [11, 6, 12-first[4], first[2]]  # 5 and 7 have the same square digit.
    learned = sorted(set(first + second))
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            demonstration = index if split == "train" and index < 2 else None
            if demonstration == 0:
                lengths, omitted = first, []
            elif demonstration == 1:
                lengths, omitted = second, [4]
            else:
                n = int(rng.integers(2, 7))
                carrier = int(rng.choice([x for x in learned if x >= 10]))
                lengths = [carrier] + list(map(int, rng.choice(learned, n-1)))
                if split == "test" and index == 0:
                    # Training omits 4; transfer the same absence rule to 0 or 1.
                    carrier_digit = (carrier-1)**2 % 10
                    alternatives = [x for x in learned if (x-1)**2 % 10 != carrier_digit]
                    lengths[1] = int(rng.choice(alternatives))
                query_digits = sorted({(x-1)**2 % 10 for x in lengths})
                omit_candidates = [d for d in query_digits if split == "test" or d not in (0, 1)]
                omit_count = int(rng.integers(0, min(2, len(omit_candidates), len(query_digits)-1)+1))
                omitted = list(map(int, rng.choice(omit_candidates, omit_count, replace=False)))
                if split == "test" and index == 0:
                    omitted = [carrier_digit]
            for _ in range(8):
                pair = _make_pair(rng, lengths, demonstration, omitted)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input search exhausted")
    validate_generated_puzzle(result)
    return result


def _features(pair):
    a, bg, es, code, obs = _decode(pair["input"])
    expected = _paint(a, es, code)
    _require(np.array_equal(expected, _grid(pair["output"])), "incorrect output")
    _require(not np.array_equal(a, expected), "identity pair")
    _require(all(expected[p] == a[p] for e in es for p in e["cells"]), "emitter changed")
    slots_by_emitter = [{d for i, d, _ in obs if i == j} for j in range(len(es))]
    return {"shape": a.shape, "background": bg, "roles": es[0]["roles"],
            "lengths": tuple(e["length"] for e in es), "slots": tuple(sorted(code)),
            "colors": tuple(code[d] for d in sorted(code)),
            "directions": tuple((e["u"], e["v"]) for e in es),
            "allocations": tuple(len(s) for s in slots_by_emitter),
            "redundant": len(obs) > len(code),
            "pooled": len({i for i, _, _ in obs}) > 1,
            "missing": sum((e["length"]-1)**2 % 10 not in code for e in es),
            "remote": any((e["length"]-1)**2 % 10 in code and
                          (e["length"]-1)**2 % 10 not in slots_by_emitter[j]
                          for j, e in enumerate(es)),
            "ablation_length": not np.array_equal(expected, _paint(a, es, code, "length")),
            "ablation_adjacency": not np.array_equal(expected, _paint(a, es, code, "adjacency")),
            "ablation_square_length": not np.array_equal(expected, _paint(a, es, code, "square_length")),
            "ablation_literal": any(code.get((e["length"]-1)**2 % 10,
                                              (e["length"]-1)**2 % 10) !=
                                    (e["length"]-1)**2 % 10
                                    for e in es if (e["length"]-1)**2 % 10 in code)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a = _grid(pair["input"])
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            features[split].append(_features(pair))
    train = features["train"]
    learned = set(x for f in train for x in f["lengths"])
    _require(set(SQUARE_DIGITS).issubset({(x-1)**2 % 10 for x in learned}), "untaught square digits")
    emitted_digits = set()
    absent_digits = set()
    for pair in puzzle["train"]:
        _, _, es, code, _ = _decode(pair["input"])
        for e in es:
            d = (e["length"]-1)**2 % 10
            (emitted_digits if d in code else absent_digits).add(d)
    _require(emitted_digits == set(SQUARE_DIGITS), "training must show every used square digit emitting")
    _require(any(f["pooled"] and f["remote"] for f in train), "pooling not demonstrated")
    _require(any(f["missing"] for f in train), "missing-entry behavior not demonstrated")
    _require(any(len(set(f["directions"])) > 1 for f in train), "registration not demonstrated")
    for ablation in ("ablation_length", "ablation_adjacency", "ablation_square_length", "ablation_literal"):
        _require(any(f[ablation] for f in train), "training permits shortcut: " + ablation)
    for f in features["test"]:
        _require(set(f["lengths"]).issubset(learned), "untaught test length")
    _, _, es, code, _ = _decode(puzzle["test"][0]["input"])
    _require(any((e["length"]-1)**2 % 10 not in code and
                 (e["length"]-1)**2 % 10 not in absent_digits for e in es),
             "first test must transfer absence to a newly missing digit")
    return None


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role])
                h.update(bytes(a.shape))
                h.update(a.astype(np.uint8).tobytes())
    return h.hexdigest()


def self_test():
    """42 supplied seeds, bounded streaming diagnostics, no file writes."""
    fingerprints, shapes, layouts, lengths, backgrounds, directions = set(), set(), set(), set(), set(), set()
    slot_sets, counts, flags = set(), set(), {k: 0 for k in ("pooled", "missing", "remote", "redundant")}
    redundant_pair = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                a = pair["input"]
                shapes.add(f["shape"])
                layouts.add(hashlib.sha256((a != f["background"]).tobytes()).hexdigest())
                lengths.update(f["lengths"])
                backgrounds.add(f["background"])
                directions.update(f["directions"])
                slot_sets.add(f["slots"])
                counts.add(len(f["lengths"]))
                for name in flags:
                    flags[name] += bool(f[name])
                if f["redundant"] and redundant_pair is None:
                    redundant_pair = pair
                _require(np.array_equal(solve(a), pair["output"]), "self-test solve mismatch")
        if seed == 0:
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterministic")
        # Minimum-count puzzles cannot rely on extra demonstrations to teach pooling.
        generate_puzzle(seed, 2, 1)
    _require(len(fingerprints) == 42 and len(shapes) >= 35 and len(layouts) >= 180, "insufficient geometry variation")
    _require(lengths == set(range(2, 13)) and len(backgrounds) == 10 and len(directions) == 8, "missing family variation")
    _require(counts == set(range(2, 7)) and len(slot_sets) >= 12, "insufficient count/code variation")
    _require(all(flags[k] > 5 for k in flags), "missing pooling/sparsity/redundancy variation")
    # Interventions on only the input code test its causal role.
    original = generate_puzzle(73, 2, 1)["train"][0]
    a, _, es, code, obs = _decode(original["input"])
    changed = a.copy()
    new_code = dict(code)
    new_code[1] = code[4]
    for _, d, p in obs:
        if d == 1:
            changed[p] = new_code[1]
    _require(np.array_equal(solve(changed), _paint(changed, es, new_code)), "lookup intervention failed")
    _require(not np.array_equal(solve(changed), original["output"]), "lookup has no causal effect")
    removed = a.copy()
    bg = _decode(a)[1]
    for _, d, p in obs:
        if d == 1:
            removed[p] = bg
    missing_code = {d: c for d, c in code.items() if d != 1}
    _require(np.array_equal(solve(removed), _paint(removed, es, missing_code)), "absence intervention failed")
    a, _, _, code, obs = _decode(redundant_pair["input"])
    repeated = next(d for d in code if sum(x[1] == d for x in obs) > 1)
    bad = a.copy()
    p = next(p for _, d, p in obs if d == repeated)
    bad[p] = next(c for c in code.values() if c != code[repeated])
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting pooled observations accepted")
    # All isometries and all color permutations preserve addressing, not digit values.
    sample = generate_puzzle(73, 2, 1)["train"][1]
    recolor = np.array([4, 7, 0, 9, 6, 2, 1, 5, 3, 8], dtype=np.int8)
    for t in range(8):
        inp, out = [np.rot90(np.fliplr(recolor[sample[k]]) if t >= 4 else recolor[sample[k]], t % 4)
                    for k in ("input", "output")]
        _require(np.array_equal(solve(inp), out), "isometry/recoloring mismatch")
    # The minimum and maximum counts and invalid types are checked explicitly.
    for nt, ne in ((2, 1), (10, 4), (7, 3)):
        p = generate_puzzle(99, nt, ne)
        _require(len(p["train"]) == nt and len(p["test"]) == ne, "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Output corruption must be detected by the semantic validator.
    p = generate_puzzle(83, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return None
