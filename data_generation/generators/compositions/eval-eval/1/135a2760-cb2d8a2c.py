"""Periodic signposts: error-correct a fence, then read it where the ray lands.

Only NumPy and the standard library are used. No parent code, files, examples,
randomness, or construction metadata enter solve(). All experiments are bounded.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


HOST_SEEDS = (0, 313443837, 223043364, 610635824, 968366400, 660990258,
              52512720, 391521070, 631211385, 945003670, 964078699,
              247688746, 58157504, 720917423, 699526074, 122722453,
              298943095, 687352806, 903352308, 454132761, 596687884,
              811831292, 180747355, 262864412, 5993899, 760713874,
              178341782, 61221509, 238048576, 793018490, 363540953,
              104898547, 652824264, 278389341, 23188099, 502230671,
              544556987, 91053919, 411662138, 404975857, 301812695,
              432563449)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _from_forward(a, direction):
    # 0 down, 1 left, 2 up, 3 right; genuine rotations preserve handedness.
    return np.rot90(a, -direction).copy()


def _to_forward(a, direction):
    return np.rot90(a, direction).copy()


def _primitive(word):
    return next(p for p in range(1, len(word) + 1)
                if len(word) % p == 0
                and all(word[i] == word[i % p] for i in range(len(word))))


def _canonical(word):
    word = tuple(word)
    return min(word[i:] + word[:i] for i in range(len(word)))


def _runs(word):
    """Cyclic marker runs (start, length), with at least two zeroes between."""
    word = tuple(map(int, word))
    if set(word) != {0, 1}:
        return []
    runs = []
    for i, bit in enumerate(word):
        if bit != 1 or word[i - 1] != 0:
            continue
        n = 0
        while word[(i + n) % len(word)] == 1:
            n += 1
        z = 0
        while word[(i + n + z) % len(word)] == 0:
            z += 1
        if not 1 <= n <= 3 or z < 2:
            return []
        runs.append((i, n))
    return runs


def _word_library():
    # Enumerated necklaces, not stored puzzle templates. At most 2**8 words.
    result = []
    for p in range(3, 9):
        for word in itertools.product((0, 1), repeat=p):
            if (_primitive(word) == p and _runs(word)
                    and word == _canonical(word)):
                result.append(word)
    return tuple(result)


_WORDS = _word_library()


def _recover_tape(tape, permanent, marker):
    """All supported majority repairs must agree; no geometry breaks a tie."""
    a = np.asarray(tape)
    if not set(map(int, a)) <= {permanent, marker}:
        raise ValueError("tape has an unrelated color")
    repairs = {}
    length = len(a)
    for p in range(3, min(8, length // 3) + 1):
        word, errors = [], 0
        for phase in range(p):
            values = a[phase::p]
            ones = int(np.count_nonzero(values == marker))
            zeroes = len(values) - ones
            if ones == zeroes:
                break
            word.append(int(ones > zeroes))
            errors += min(ones, zeroes)
        if len(word) != p or errors > 2 or not _runs(word):
            continue
        repaired = np.array([marker if word[i % p] else permanent
                             for i in range(length)], dtype=np.int8)
        key = repaired.tobytes()
        if key not in repairs:
            repairs[key] = (repaired, _primitive(word), tuple(word))
    if len(repairs) != 1:
        raise ValueError("periodic fence has no unique supported repair")
    return next(iter(repairs.values()))


def _read_command(tape, index, permanent, marker):
    """Contact a permanent cell and read the complete adjacent unary run."""
    if not 0 <= index < len(tape) or int(tape[index]) != permanent:
        raise ValueError("the incoming ray must contact a permanent symbol")
    left = index > 0 and int(tape[index - 1]) == marker
    right = index + 1 < len(tape) and int(tape[index + 1]) == marker
    if left == right:
        raise ValueError("contact must have markers on exactly one side")
    toward = -1 if left else 1
    cursor, n = index + toward, 0
    while 0 <= cursor < len(tape) and int(tape[cursor]) == marker:
        n += 1
        cursor += toward
    if not 0 <= cursor < len(tape) or not 1 <= n <= 3:
        raise ValueError("contacted marker run must be complete inside the tape")
    return -toward, n


def _decode(input_grid):
    """Input-only segmentation, periodic recovery, stateful readout and drawing."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    if not all(int(grid[r, c]) == bg
               for r, c in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("all corners must be background")
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 4 or counts[colors == bg][0] <= grid.size // 2:
        raise ValueError("four roles and a majority background are required")
    singletons = [int(c) for c, n in zip(colors, counts) if n == 1 and c != bg]
    if len(singletons) != 1:
        raise ValueError("one uniquely colored seed is required")
    route = singletons[0]
    r, c = map(int, np.argwhere(grid == route)[0])
    h, w = grid.shape
    directions = [d for d, yes in enumerate(
        (r == 0, c == w - 1, r == h - 1, c == 0)) if yes]
    if len(directions) != 1:
        raise ValueError("seed must be on one boundary, away from corners")
    direction = directions[0]
    a = _to_forward(grid, direction)
    h, w = a.shape
    current = int(np.argwhere(a == route)[0, 1])
    structure = (a != bg) & (a != route)
    rows = list(map(int, np.flatnonzero(structure.any(axis=1))))
    if not rows or rows[0] == 0 or rows[-1] == h - 1:
        raise ValueError("interior fences required")
    first = np.flatnonzero(structure[rows[0]])
    permanent = int(a[rows[0], first[0]])
    remaining = set(map(int, colors)) - {bg, route, permanent}
    if len(remaining) != 1:
        raise ValueError("fence caps must identify a separate permanent role")
    marker = remaining.pop()
    result, corrected = a.copy(), a.copy()
    path = np.zeros(a.shape, dtype=bool)
    start, previous = 0, -1
    bars = []
    for f in rows:
        occupied = np.flatnonzero(structure[f])
        lo, hi = int(occupied[0]), int(occupied[-1])
        if (len(occupied) != hi - lo + 1
                or a[f, lo] != permanent or a[f, hi] != permanent):
            raise ValueError("each fence needs a contiguous span and two intact caps")
        tape = a[f, lo + 1:hi]
        repaired, p, word = _recover_tape(tape, permanent, marker)
        j = current - lo - 1
        sign, n = _read_command(repaired, j, permanent, marker)
        d = n + 1
        target = hi + d if sign == 1 else lo - d
        turn = f - d
        if not (0 <= target < w and 1 <= turn and previous < turn and start <= turn):
            raise ValueError("routing command is out of bounds or reaches an earlier fence")
        path[start:turn + 1, current] = True
        path[turn, min(current, target):max(current, target) + 1] = True
        corrected[f, lo + 1:hi] = repaired
        result[f, lo:hi + 1] = permanent
        try:
            raw = _read_command(tape, j, permanent, marker)
        except ValueError:
            raw = None
        bars.append({"row": f, "lo": lo, "hi": hi, "period": p,
                     "word": word[:p], "runs": tuple(n for _, n in _runs(word[:p])),
                     "length": len(tape), "index": j, "incoming": current,
                     "sign": sign, "n": n, "target": target, "turn": turn,
                     "errors": int(np.count_nonzero(tape != repaired)),
                     "raw_differs": raw != (sign, n),
                     "raw_valid_differs": raw is not None and raw != (sign, n)})
        current, start, previous = target, turn, f
    path[start:, current] = True
    if np.any(path & structure):
        raise ValueError("route collides with a fence")
    result[path] = route
    return _from_forward(result, direction), {
        "direction": direction, "palette": (bg, route, permanent, marker),
        "bars": bars, "path": _from_forward(path, direction),
        "corrected": _from_forward(corrected, direction),
        "structure": _from_forward(structure, direction),
        "forward_shape": (h, w)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _decode(input_grid)[0]


def _spec(rng, n=None, *, small=False, multi=False, last=False):
    options = []
    for word in _WORDS:
        if (small and (len(word) > 5 or len(_runs(word)) != 1)
                or multi and (len(_runs(word)) != 2 or len(set(k for _, k in _runs(word))) < 2)
                or not last and len(word) > 7):
            continue
        for at, count in _runs(word):
            if n is None or n == count:
                options.append((word, at, count))
    word, at, n = options[int(rng.integers(len(options)))]
    # Long codes need room for subsequent detours; last/single fences can
    # also carry four or more repeats and longer incomplete tails.
    maximum = min(26, 4 * len(word) + 1) if last else min(24, 3 * len(word) + 2)
    length = int(rng.integers(3 * len(word), maximum + 1))
    return {"word": word, "at": at, "n": n, "length": length, "side": None}


def _span_options(width, current, spec):
    length, n = spec["length"], spec["n"]
    result = []
    for lo in range(width - length - 1):
        hi, j = lo + length + 1, current - lo - 1
        if not 0 <= j < length:
            continue
        for sign in ((-1, 1) if spec["side"] is None else (spec["side"],)):
            target = hi + n + 1 if sign == 1 else lo - n - 1
            if not 0 <= target < width:
                continue
            if sign == 1 and j < n + 1 or sign == -1 and j + n + 1 >= length:
                continue
            contact_phase = (spec["at"] + n) if sign == 1 else spec["at"] - 1
            phase = (contact_phase - j) % len(spec["word"])
            result.append((lo, hi, target, sign, phase))
    return result


def _plan_spans(rng, width, specs):
    failed = set()

    def search(index, current):
        if index == len(specs):
            return []
        key = index, current
        if key in failed:
            return None
        options = _span_options(width, current, specs[index])
        for raw_i in rng.permutation(len(options)):
            option = options[int(raw_i)]
            rest = search(index + 1, option[2])
            if rest is not None:
                return [option] + rest
        failed.add(key)
        return None

    for current in map(int, rng.permutation(np.arange(1, width - 1))):
        result = search(0, current)
        if result is not None:
            return current, result
    return None


def _corrupt(rng, clean, j, sign, n, p, permanent, marker, kind):
    # A required local fault is chosen on the contact symbol or its marker run.
    # Optional second errors occupy another phase, keeping every vote strict.
    primary = (j if kind == 0 else j - sign * n if kind == 2
               else j - sign * int(rng.integers(1, n + 1)))
    for attempt in range(24):
        noisy = clean.copy()
        flips = [primary]
        if kind != 2 and attempt < 16 and rng.random() < 0.35:
            choices = [i for i in range(len(clean)) if i % p != primary % p]
            flips.append(int(rng.choice(choices)))
        for i in flips:
            noisy[i] = marker if noisy[i] == permanent else permanent
        try:
            recovered = _recover_tape(noisy, permanent, marker)[0]
        except ValueError:
            continue
        if np.array_equal(recovered, clean):
            return noisy
    raise ValueError("no unambiguous sparse corruption for this tape")


def _make_pair(rng, palette, direction, count, obligation=None):
    bg, route, permanent, marker = palette
    for attempt in range(96):
        specs = [_spec(rng, last=i == count - 1) for i in range(count)]
        if obligation == "repeat":
            # Two modest codes leave room for a same-side run. Other bars vary.
            at = int(rng.integers(count - 1))
            side = int(rng.choice((-1, 1)))
            for offset, n in enumerate(rng.permutation((1, 2))):
                specs[at + offset] = _spec(rng, int(n), small=True)
                specs[at + offset]["side"] = side
        elif obligation == "contrast":
            # Two copies of one multirun program are addressed at different runs.
            # The seven-phase word fits repeated routing inside a 30-cell width.
            specs[-3] = _spec(rng, 3, small=True)
            word = next(w for w in _WORDS if len(w) == 7
                        and sorted(n for _, n in _runs(w)) == [1, 2])
            for offset, n in enumerate(rng.permutation((1, 2))):
                at = next(at for at, size in _runs(word) if size == n)
                specs[-2 + offset] = {"word": word, "at": at, "n": int(n),
                                      "length": int(rng.integers(21, 24)), "side": None}
            side = int(rng.choice((-1, 1)))
            for offset in range(3):
                specs[-3 + offset]["side"] = side if offset % 2 == 0 else -side
        width = int(rng.integers(16, 31))
        plan = None
        for candidate_width in dict.fromkeys((width, min(30, width + 5), 30)):
            plan = _plan_spans(rng, candidate_width, specs)
            if plan is not None:
                width = candidate_width
                break
        if plan is None:
            continue
        initial, spans = plan
        minimum = sum(s["n"] + 2 for s in specs) + 2
        height = int(rng.integers(minimum, 31))
        slack = rng.multinomial(height - minimum, np.full(count + 1, 1 / (count + 1)))
        a = np.full((height, width), bg, dtype=np.int8)
        b = a.copy()
        a[0, initial] = route
        current, start, previous = initial, 0, -1
        failed = False
        for i, (spec, span) in enumerate(zip(specs, spans)):
            lo, hi, target, sign, phase = span
            n, p = spec["n"], len(spec["word"])
            turn = (1 if i == 0 else previous + 1) + int(slack[i])
            f = turn + n + 1
            clean = np.array([marker if spec["word"][(k + phase) % p] else permanent
                              for k in range(spec["length"])], dtype=np.int8)
            # The repeat and contrast witnesses expose both flip directions.
            kind = (0 if obligation == "repeat" else 1) if i == 0 else int(rng.integers(2))
            if obligation == "contrast" and i == count - 3:
                kind = 2  # A valid raw count of two must be recovered as three.
            try:
                noisy = _corrupt(rng, clean, current - lo - 1, sign, n,
                                 p, permanent, marker, kind)
            except ValueError:
                failed = True
                break
            a[f, lo:hi + 1] = permanent
            a[f, lo + 1:hi] = noisy
            b[f, lo:hi + 1] = permanent
            b[start:turn + 1, current] = route
            b[turn, min(current, target):max(current, target) + 1] = route
            current, start, previous = target, turn, f
        if failed:
            continue
        b[start:, current] = route
        if np.count_nonzero(a == bg) <= a.size // 2:
            continue
        return {"input": _from_forward(a, direction), "output": _from_forward(b, direction)}
    raise RuntimeError("bounded geometric construction exhausted")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    directions = list(map(int, rng.permutation(4)))
    mixed = bool(rng.integers(2))
    train_dirs, test_dirs = (directions[:2], directions[2:]) if mixed else (directions[:1], directions[1:])
    longer = bool(rng.integers(2))
    train_counts = list(map(int, rng.integers(1, 4 if longer else 6, size=num_train)))
    repeat, contrast = map(int, rng.choice(num_train, 2, replace=False))
    train_counts[repeat] = max(2, train_counts[repeat])
    train_counts[contrast] = max(3, train_counts[contrast])
    puzzle, seen = {"train": [], "test": []}, set()
    for serial in range(num_train + num_test):
        split = "train" if serial < num_train else "test"
        if split == "train":
            count = train_counts[serial]
            direction = train_dirs[serial % len(train_dirs)] if serial < 2 else int(rng.choice(train_dirs))
            obligation = "repeat" if serial == repeat else "contrast" if serial == contrast else None
        else:
            count = int(rng.integers(4, 6) if longer else rng.integers(1, 6))
            direction, obligation = int(rng.choice(test_dirs)), None
        for _ in range(12):
            pair = _make_pair(rng, palette, direction, count, obligation)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("bounded distinct-input construction exhausted")
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, palettes = set(), set()
    features = {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            expected, info = _decode(a)
            assert np.array_equal(expected, b), "output disagrees with decoded commands"
            bg, route, permanent, marker = info["palette"]
            palettes.add(info["palette"])
            mask, path = info["structure"], info["path"]
            assert np.all(b[mask] == permanent) and not np.any(path & mask)
            assert np.all(b[path] == route) and np.all(b[~(path | mask)] == a[~(path | mask)])
            assert marker not in b and b[a == route].tolist() == [route]
            assert 1 <= len(info["bars"]) <= 5
            assert all(1 <= bar["errors"] <= 2 for bar in info["bars"])
            assert any(bar["raw_differs"] for bar in info["bars"]), "noise must affect a routing decision"
            # A-only repair leaves every background cell alone; B adds a route.
            assert np.any((a == bg) & (b == route))
            assert not np.array_equal(info["corrected"], b)
            features[split].append(info)
    assert len(palettes) == 1, "palette conventions must agree within a puzzle"
    train_bars = [bar for x in features["train"] for bar in x["bars"]]
    assert {b["n"] for b in train_bars} == {1, 2, 3}
    assert any(b["raw_valid_differs"] for b in train_bars)
    assert {b["sign"] for b in train_bars} == {-1, 1}
    assert any(len(set(b["runs"])) > 1 for b in train_bars)
    addressed = {}
    for bar in train_bars:
        addressed.setdefault(_canonical(bar["word"]), set()).add(bar["n"])
    assert any(len(ns) > 1 for word, ns in addressed.items()
               if len(set(n for _, n in _runs(word))) > 1)
    transitions = {(a["sign"], b["sign"]) for info in features["train"]
                   for a, b in zip(info["bars"], info["bars"][1:])}
    assert any(a == b for a, b in transitions) and any(a != b for a, b in transitions)
    assert ({i["direction"] for i in features["train"]}
            .isdisjoint({i["direction"] for i in features["test"]}))


def self_test():
    """Streaming 42-seed diversity, ablations, metamorphic and count checks."""
    shapes, periods, lengths, counts, directions, normalized = set(), set(), set(), set(), set(), set()
    signatures, transitions, role_colors = set(), set(), [set() for _ in range(4)]
    multirun = faults = partial = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in puzzle:
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                saved = a.copy()
                prediction, info = _decode(a)
                assert np.array_equal(prediction, b) and np.array_equal(a, saved)
                digest.update(str(a.shape).encode() + a.tobytes() + b.tobytes())
                shapes.add(a.shape)
                directions.add((info["direction"], split))
                counts.add(len(info["bars"]))
                norm = np.empty_like(a)
                for i, color in enumerate(info["palette"]):
                    norm[a == color] = i
                    role_colors[i].add(color)
                normalized.add(hashlib.sha256(norm.tobytes() + str(norm.shape).encode()).digest())
                bars = info["bars"]
                transitions.update((x["sign"], y["sign"]) for x, y in zip(bars, bars[1:]))
                for bar in bars:
                    periods.add(bar["period"])
                    lengths.add(bar["length"])
                    multirun += len(set(bar["runs"])) > 1
                    faults += bar["raw_differs"]
                    partial += bar["length"] % bar["period"] != 0
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(normalized) >= 200 and len(shapes) >= 40
    assert periods >= {3, 4, 5, 7} and len(lengths) >= 12 and counts == {1, 2, 3, 4, 5}
    assert len(directions) == 8 and len(transitions) == 4
    assert all(len(s) >= 9 for s in role_colors)
    assert multirun >= 42 and faults >= 210 and partial >= 100
    # Exact count contracts; only one puzzle is retained at a time.
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(79 + train * 7 + test, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(_key(x[role]) == _key(y[role]) for split in a
               for x, y in zip(a[split], b[split]) for role in x)
    permutation = np.array([6, 3, 9, 0, 7, 2, 1, 8, 5, 4], dtype=np.int8)
    for pair in a["train"] + a["test"]:
        for turns in range(4):
            query = permutation[np.rot90(pair["input"], turns)]
            expected = permutation[np.rot90(pair["output"], turns)]
            assert np.array_equal(solve(query), expected)
            assert np.array_equal(solve(query[:, ::-1]), expected[:, ::-1])
    # Explicit literal route regression: a period-four fence, one missing marker.
    x = np.zeros((9, 21), dtype=np.int8)
    x[0, 7] = 3
    x[5, 0:16] = 2
    x[5, 1:15] = [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1]
    x[5, 5] = 2
    y = np.zeros_like(x)
    y[5, 0:16] = 2
    y[0:3, 7] = 3
    y[2, 7:19] = 3
    y[2:, 18] = 3
    assert _read_command(x[5, 1:15], 6, 2, 1) == (1, 1)
    assert np.array_equal(solve(x), y)
    clean = x.copy()
    clean[5, 5] = 1
    assert np.array_equal(solve(clean), y), "clean repetitions obey the same rule"
    unrepairable = x.copy()
    unrepairable[5, 1:15] = 1
    try:
        solve(unrepairable)
    except ValueError:
        pass
    else:
        raise AssertionError("unrecoverable fence accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged["test"][0]["output"][:] = damaged["test"][0]["input"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a missing route")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "normalized_inputs": len(normalized), "periods": sorted(periods),
            "tape_lengths": sorted(lengths), "bar_counts": sorted(counts),
            "multirun_bars": multirun, "routing_relevant_faults": faults,
            "partial_tapes": partial, "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
