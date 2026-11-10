"""Dye-front ribbon: intrinsic distance, unary bands, and competing colors.

No files, parent generators, random state, or example labels are used by solve.
Generation and tests are bounded and keep their diagnostics in memory.
"""
from __future__ import annotations

import hashlib
from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid using colors 0..9")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _decode(input_grid):
    a = _grid(input_grid)
    hist = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(hist.argmax())
    if hist[bg] * 2 <= a.size:
        raise ValueError("background must have a strict majority")
    cells = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    if not 7 <= len(cells) <= 30:
        raise ValueError("the ribbon must contain 7..30 cells")
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in DIRS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    ends = sorted(p for p in cells if len(neighbors[p]) == 1)
    if (len(ends) != 2 or ends[0][0] == ends[1][0]
            or any(len(v) not in (1, 2) for v in neighbors.values())):
        raise ValueError("expected one induced chain with unequal endpoint rows")
    path, old, p = [], None, ends[0]
    while p is not None:
        if p in path:
            raise ValueError("cycle")
        path.append(p)
        nxt = [q for q in neighbors[p] if q != old]
        old, p = p, (nxt[0] if nxt else None)
    if len(path) != len(cells):
        raise ValueError("disconnected foreground")
    seq = [int(a[p]) for p in path]
    marker, thread = seq[0], seq[-1]
    if marker == thread or seq[1] != thread or seq[-2] != thread:
        raise ValueError("endpoints must key distinct marker/thread roles with buffers")
    bands = []
    i = 1
    while i < len(seq) - 1:
        if seq[i] == thread:
            i += 1
            continue
        start = i
        while i < len(seq) - 1 and seq[i] != thread:
            i += 1
        word = seq[start:i]
        bases = set(word) - {marker}
        if len(bases) != 1:
            raise ValueError("a band has one dye color and the marker color")
        base = bases.pop()
        n = word.count(marker)
        if not 1 <= n <= 4 or len(word) < 2 * n + 1:
            raise ValueError("a band needs 1..4 markers and an unmarked tail")
        pattern = [marker, base] * n + [base] * (len(word) - 2 * n)
        if word == pattern:
            sign = 1
        elif word == pattern[::-1]:
            sign = -1
        else:
            raise ValueError("markers must alternate from exactly one band end")
        if bands and bands[-1][4] == base:
            raise ValueError("neighboring bands must have different dyes")
        bands.append((start, i - 1, n, sign, base))
    if not bands:
        raise ValueError("a ribbon needs a marked band")
    return {"background": bg, "marker": marker, "thread": thread,
            "path": path, "sequence": seq, "bands": bands}


def _render(info, offset=1, flip=False, paint=True, last_wins=False, alternate=False):
    """Per-cell distance comparison, independent of input geometry."""
    seq, bands = info["sequence"], info["bands"]
    result = np.asarray(seq, dtype=np.int8).copy()
    distance = np.full(len(seq), 100, dtype=int)
    tied = np.zeros(len(seq), dtype=bool)
    for lo, hi, n, direction, color in bands:
        result[lo:hi + 1] = color
    if paint:
        for index, (lo, hi, n, direction, color) in enumerate(bands):
            if alternate:
                direction = bands[0][3] * (-1) ** index
            direction = -direction if flip else direction
            edge = hi if direction == 1 else lo
            for d in range(1, n + offset + 1):
                j = edge + direction * d
                if not 0 < j < len(seq) - 1 or seq[j] != info["thread"]:
                    break
                if last_wins or d < distance[j]:
                    distance[j], result[j], tied[j] = d, color, False
                elif d == distance[j] and result[j] != color:
                    tied[j] = True
        result[tied] = info["thread"]
    return result.reshape(-1, 1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Trace from the higher endpoint, decode bands, and compare dye distances."""
    return _render(_decode(input_grid))


def _sweep_orders(path):
    orders = [sorted(path), sorted(path, reverse=True)]
    columns = sorted({c for _, c in path})
    for cs in (columns, columns[::-1]):
        for reverse_last in (False, True):
            orders.append([p for c in cs for p in sorted(
                (p for p in path if p[1] == c),
                reverse=reverse_last and c == cs[-1])])
    return orders


def _defeats_sweeps(path, answer):
    colors = dict(zip(path, map(int, np.asarray(answer).ravel())))
    true = list(map(int, np.asarray(answer).ravel()))
    return all([colors[p] for p in order] != true for order in _sweep_orders(path))


def _geometry(path):
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return {"turns": sum(a != b for a, b in zip(steps, steps[1:])),
            "above": min(r for r, _ in path) < path[0][0],
            "reversal": (0, 1) in steps and (0, -1) in steps}


def _path(rng, length, answer, witness=False, above=False, need_above=False):
    for _ in range(100):
        path, used, previous = [(0, 0)], {(0, 0)}, None
        persistence = float(rng.choice((0.6, 2.0, 8.0, 25.0)))
        for _ in range(length - 1):
            options, weights = [], []
            for dr, dc in DIRS:
                q = (path[-1][0] + dr, path[-1][1] + dc)
                if q not in used and sum((q[0] + x, q[1] + y) in used for x, y in DIRS) == 1:
                    options.append((q, (dr, dc)))
                    weights.append(persistence if previous == (dr, dc) else 1.0)
            if not options:
                break
            weights = np.asarray(weights, dtype=float)
            q, previous = options[int(rng.choice(len(options), p=weights / weights.sum()))]
            path.append(q)
            used.add(q)
        if len(path) != length or path[0][0] == path[-1][0]:
            continue
        if path[0][0] > path[-1][0]:
            path.reverse()
        f = _geometry(path)
        if (f["above"] and not above) or (need_above and not f["above"]):
            continue
        if witness and not _defeats_sweeps(path, answer):
            continue
        return path
    # Constructive fallbacks are induced chains, not another random retry loop.
    if need_above:
        segments = [((-1, 0), 2), ((0, 1), 3), ((1, 0), length - 8), ((0, -1), 2)]
    elif witness:
        # The unique header is inside the horizontal span; the reversed lower
        # row starts with answer[4] (a dye), while the true sequence ends in thread.
        assert length >= 10 and int(answer[4, 0]) != int(answer[-1, 0])
        segments = [((0, 1), 2), ((1, 0), 2), ((0, -1), length - 5)]
    else:
        segments = [((1, 0), length - 1)]
    path = [(0, 0)]
    for (dr, dc), count in segments:
        for _ in range(count):
            path.append((path[-1][0] + dr, path[-1][1] + dc))
    assert len(path) == length
    assert not witness or _defeats_sweeps(path, answer)
    return path


def _embed(rng, path, sequence, background):
    r0, c0 = min(r for r, _ in path), min(c for _, c in path)
    path = [(r - r0, c - c0) for r, c in path]
    bh, bw = max(r for r, _ in path) + 1, max(c for _, c in path) + 1
    if rng.random() < 0.65:
        h, w = min(30, bh + int(rng.integers(0, 5))), min(30, bw + int(rng.integers(0, 5)))
    else:
        h, w = int(rng.integers(bh, 31)), int(rng.integers(bw, 31))
    while h * w <= 2 * len(path):
        if (h <= w and h < 30) or w == 30:
            h += 1
        else:
            w += 1
    top, left = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
    grid = np.full((h, w), background, dtype=np.int8)
    for (r, c), color in zip(path, sequence):
        grid[r + top, c + left] = color
    return grid


def _word_plan(rng, colors, ns, signs, gaps, tails):
    bg, marker, thread = colors[:3]
    dyes = colors[3:]
    sequence, bands = [marker] + [thread] * tails[0], []
    # Add unequal unmarked tails to bands only while the total stays <=30.
    spare = 30 - (2 + sum(tails) + sum(gaps) + sum(2 * n + 1 for n in ns))
    assert spare >= 0
    previous = None
    for i, (n, sign) in enumerate(zip(ns, signs)):
        base = int(rng.choice([c for c in dyes if c != previous]))
        extra = int(rng.integers(0, min(2, spare) + 1))
        spare -= extra
        word = [marker, base] * n + [base] * (1 + extra)
        if sign == -1:
            word.reverse()
        lo = len(sequence)
        sequence.extend(word)
        bands.append((lo, len(sequence) - 1, n, sign, base))
        sequence.extend([thread] * (gaps[i] if i < len(gaps) else tails[1]))
        previous = base
    sequence.append(thread)
    # Construction labels: chronological proposal arrival, rather than the
    # decoder's per-cell minimum-distance arrays. Only the first arrival wins;
    # simultaneous different arrivals leave a neutral seam.
    output = sequence.copy()
    for lo, hi, _, _, base in bands:
        output[lo:hi + 1] = [base] * (hi - lo + 1)
    reached = set()
    for tick in range(1, 6):
        arrivals = {}
        for lo, hi, n, sign, base in bands:
            edge = hi if sign == 1 else lo
            j = edge + sign * tick
            if tick <= n + 1 and 0 < j < len(sequence) - 1:
                corridor = range(min(edge + sign, j), max(edge + sign, j) + 1)
                if all(sequence[k] == thread for k in corridor):
                    arrivals.setdefault(j, set()).add(base)
        for j, proposals in arrivals.items():
            if j not in reached:
                output[j] = next(iter(proposals)) if len(proposals) == 1 else thread
                reached.add(j)
    return sequence, np.asarray(output, dtype=np.int8).reshape(-1, 1)


def _random_spec(rng, cap, max_bands):
    k = int(rng.integers(1, max_bands + 1))
    ns = list(map(int, rng.integers(1, cap + 1, size=k)))
    signs = list(map(int, rng.choice((-1, 1), size=k)))
    gaps = list(map(int, rng.integers(1, 6, size=k - 1)))
    tails = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
    while 2 + sum(tails) + sum(gaps) + sum(2 * n + 1 for n in ns) > 30:
        choices = [i for i, n in enumerate(ns) if n > 1]
        if choices:
            ns[int(rng.choice(choices))] -= 1
        elif max(gaps, default=0) > 1:
            gaps[int(np.argmax(gaps))] -= 1
        else:
            tails[int(np.argmax(tails))] -= 1
    return ns, signs, gaps, tails


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    cap = int(rng.integers(2, 5))
    mode = str(rng.choice(("gap", "count")))
    above = bool(rng.integers(2))
    shared = bool(rng.integers(2))
    palette = list(map(int, rng.permutation(10)))
    puzzle, seen = {"train": [], "test": []}, set()

    def add(split, spec, witness=False, need_above=False):
        for _ in range(32):
            colors = palette if shared else list(map(int, rng.permutation(10)))
            seq, out = _word_plan(rng, colors, *spec)
            path = _path(rng, len(seq), out, witness, above, need_above)
            a = _embed(rng, path, seq, colors[0])
            if _key(a) not in seen:
                seen.add(_key(a))
                puzzle[split].append({"input": a, "output": out})
                return
        raise RuntimeError("bounded uniqueness attempts exhausted")

    # Witnesses have variable colors, tails, band lengths, geometry and placement.
    ns = list(map(int, rng.permutation([1, cap])))
    add("train", (ns, [1, -1], [3], list(map(int, rng.integers(1, 4, size=2)))),
        witness=True, need_above=above)
    # Repeated directions are demonstrated even for num_train=2. The middle
    # band's cap+2 gap exposes its exact radius; outer bands expose radius two.
    # At cap=4 this construction fits exactly thirty cells.
    reach_gaps = [int(rng.integers(1, 6 - cap)), cap + 2]
    reach_ns, reach_signs = [1, cap, 1], [-1, 1, 1]
    if rng.integers(2):
        reach_ns, reach_signs, reach_gaps = reach_ns[::-1], [-x for x in reach_signs[::-1]], reach_gaps[::-1]
    add("train", (reach_ns, reach_signs, reach_gaps, [3, 3]))
    for _ in range(num_train - 2):
        add("train", _random_spec(rng, cap, 3))
    taught = sorted({b[2] for pair in puzzle["train"] for b in _decode(pair["input"])["bands"]})
    for i in range(num_test):
        if i == 0 and mode == "gap":
            ns = list(map(int, rng.permutation([1, cap])))
            spec = (ns, [1, -1], [int(rng.choice((7, 8, 9)))], [1, 1])
        elif i == 0:
            ns = list(map(int, rng.permutation([1, cap, 1, 1])))
            signs = list(map(int, rng.choice((-1, 1), size=4)))
            gaps = list(map(int, rng.integers(1, 5, size=3)))
            while 4 + sum(gaps) + sum(2 * n + 1 for n in ns) > 30:
                gaps[int(np.argmax(gaps))] -= 1
            spec = (ns, signs, gaps, [1, 1])
        else:
            spec = _random_spec(rng, cap, 3)
            spec = ([int(rng.choice(taught)) for _ in spec[0]], *spec[1:])
            # Resampling taught counts can grow a plan. Reduce gaps/counts by
            # rebuilding a safe plan with the already demonstrated extreme 1.
            if 2 + sum(spec[3]) + sum(spec[2]) + sum(2 * n + 1 for n in spec[0]) > 30:
                spec = ([1] * len(spec[0]), *spec[1:])
        add("test", spec, witness=i == 0, need_above=above and i == 0)
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, output):
    seq, bands = info["sequence"], info["bands"]
    ties, overlaps, exposed = 0, 0, set()
    claims = {}
    for lo, hi, n, sign, base in bands:
        edge, available = (hi if sign == 1 else lo), 0
        j = edge + sign
        while 0 < j < len(seq) - 1 and seq[j] == info["thread"]:
            available += 1
            j += sign
        if available >= n + 2:
            exposed.add(n)
        for d in range(1, min(available, n + 1) + 1):
            claims.setdefault(edge + sign * d, []).append((d, base))
    for candidates in claims.values():
        if len(candidates) > 1:
            overlaps += 1
            ties += candidates[0][0] == candidates[1][0]
    f = _geometry(info["path"])
    f.update({"length": len(seq), "count": len(bands), "ties": ties, "overlaps": overlaps,
              "exposed": exposed, "gaps": tuple(b[0] - a[1] - 1 for a, b in zip(bands, bands[1:])),
              "repeated_direction": any(a[3] == b[3] for a, b in zip(bands, bands[1:])),
              "sweep_failure": _defeats_sweeps(info["path"], output)})
    return f


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos, features = set(), {}, {}
    for split in ("train", "test"):
        infos[split], features[split] = [], []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _decode(a)
            assert b.shape == (len(info["sequence"]), 1)
            assert np.array_equal(b, _render(info))
            assert b[0, 0] == info["marker"] and b[-1, 0] == info["thread"]
            assert np.count_nonzero(b == info["marker"]) == 1
            infos[split].append(info)
            features[split].append(_features(info, b))
    taught = {b[2] for x in infos["train"] for b in x["bands"]}
    assert len(taught) >= 2
    assert {b[3] for x in infos["train"] for b in x["bands"]} == {-1, 1}
    assert {b[2] for x in infos["test"] for b in x["bands"]} <= taught
    assert len(set().union(*(f["exposed"] for f in features["train"]))) >= 2
    assert any(f["ties"] for f in features["train"])
    assert any(f["overlaps"] > f["ties"] for f in features["train"])
    assert any(f["repeated_direction"] for f in features["train"])
    for split in features:
        assert any(f["sweep_failure"] for f in features[split])
    if any(f["above"] for f in features["test"]):
        assert any(f["above"] for f in features["train"])
    train_gaps = {g for f in features["train"] for g in f["gaps"]}
    train_counts = {f["count"] for f in features["train"]}
    assert any(set(f["gaps"]) - train_gaps or f["count"] not in train_counts for f in features["test"])
    # Concrete omissions/shortcuts must change a demonstrated answer.
    for kwargs in ({"paint": False}, {"offset": 0}, {"flip": True}, {"last_wins": True}, {"alternate": True}):
        assert any(not np.array_equal(_render(x, **kwargs), pair["output"])
                   for x, pair in zip(infos["train"], puzzle["train"]))


def self_test():
    """42 seeds, streaming summaries, explicit boundaries and semantic mutations."""
    hashes, shapes, layouts, lengths, turns, counts, ns, roles = (set() for _ in range(8))
    observations = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in puzzle:
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                before = a.copy()
                assert np.array_equal(solve(a), b) and np.array_equal(a, before)
                info = _decode(a)
                f = _features(info, b)
                shapes.add(a.shape)
                lengths.add(f["length"])
                turns.add(f["turns"])
                counts.add(f["count"])
                ns.update(x[2] for x in info["bands"])
                roles.add((info["background"], info["marker"], info["thread"]))
                path = info["path"]
                layouts.add(tuple((r - path[0][0], c - path[0][1]) for r, c in path))
                for name in ("above", "reversal", "ties", "overlaps", "repeated_direction", "sweep_failure"):
                    observations[name] += bool(f[name])
                observations["border"] += any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1) for r, c in path)
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
                digest.update(b.tobytes())
        hashes.add(digest.hexdigest())
    assert len(hashes) == 42 and len(shapes) >= 60 and len(layouts) >= 100
    assert len(lengths) >= 12 and len(turns) >= 8 and counts == {1, 2, 3, 4}
    assert ns == {1, 2, 3, 4} and len(roles) >= 30
    assert all(observations[k] >= 10 for k in observations)
    a, b = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for tr, te in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(809, tr, te)
        assert (len(p["train"]), len(p["test"])) == (tr, te)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(0)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Literal word regression: a nearer dye wins, equal arrivals stay neutral.
    fixture = np.array([5, 1, 5, 2, 2, 1, 1, 1, 3, 3, 5, 3, 5, 1, 1], dtype=np.int8)
    info = {"sequence": fixture.tolist(), "thread": 1,
            "bands": [(2, 4, 1, 1, 2), (8, 12, 2, -1, 3)]}
    expected = np.array([5, 1, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 3, 1, 1])
    assert np.array_equal(_render(info).ravel(), expected)
    # Non-colliding n=1 paints exactly two cells; boundaries and keys stop dye.
    info = {"sequence": [5, 1, 5, 2, 2, 1, 1, 1, 1], "thread": 1,
            "bands": [(2, 4, 1, 1, 2)]}
    assert _render(info).ravel().tolist() == [5, 1, 2, 2, 2, 2, 2, 1, 1]
    # Valid family members outside the sampling ranges: q=8 and six bands.
    words = [[5, 1] + [5, 2] * 4 + [2] * 8 + [1, 1]]
    words.append([5, 1] + [x for i in range(6) for x in
                          ([5, 2 + i % 2, 2 + i % 2] + [1])] + [1])
    for word in words:
        grid = np.zeros((len(word), 3), dtype=np.int8)
        grid[:, 1] = word
        decoded = _decode(grid)
        assert solve(grid).shape == (len(word), 1)
        assert len(decoded["bands"]) == (6 if len(word) == 27 else 1)
    # Graph and role-key mutations must be rejected by the input decoder.
    malformed = []
    valid = np.zeros((9, 5), dtype=np.int8)
    valid[:, 2] = [5, 1, 5, 2, 2, 1, 1, 1, 1]
    branch = valid.copy()
    branch[4, 3] = 1
    malformed.append(branch)
    detached = valid.copy()
    detached[0, 4] = 2
    malformed.append(detached)
    bad_key = valid.copy()
    bad_key[0, 2] = 1
    malformed.append(bad_key)
    bad_band = valid.copy()
    bad_band[4, 2] = 5
    malformed.append(bad_band)
    tied_ends = np.zeros((7, 7), dtype=np.int8)
    for point, color in zip([(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3)],
                            [5, 1, 5, 2, 2, 1, 1]):
        tied_ends[point] = color
    malformed.append(tied_ends)
    for grid in malformed:
        try:
            solve(grid)
        except ValueError:
            continue
        raise AssertionError("invalid input accepted")
    # Direct fallback regressions across every attainable witness length.
    for length in range(15, 31):
        answer = np.ones((length, 1), dtype=np.int8)
        answer[0, 0], answer[4, 0] = 5, 2
        for above in (False, True):
            segments = ([((-1, 0), 2), ((0, 1), 3), ((1, 0), length - 8), ((0, -1), 2)]
                        if above else [((0, 1), 2), ((1, 0), 2), ((0, -1), length - 5)])
            path = [(0, 0)]
            for (dr, dc), n in segments:
                for _ in range(n):
                    path.append((path[-1][0] + dr, path[-1][1] + dc))
            assert _defeats_sweeps(path, answer)
            assert all(sum((r + dr, c + dc) in set(path) for dr, dc in DIRS) ==
                       (1 if i in (0, length - 1) else 2) for i, (r, c) in enumerate(path))
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(hashes),
            "input_shapes": len(shapes), "path_geometries": len(layouts),
            "lengths": sorted(lengths), "turn_counts": sorted(turns),
            "band_counts": sorted(counts), "marker_counts": sorted(ns),
            "role_assignments": len(roles), "observations": dict(observations)}
