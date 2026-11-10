"""Reconcile overlapping folded observations; display observation multiplicity.

Only NumPy and the Python standard library are used. All registration, direction,
and thickness information is recovered from the input pixels by solve().
"""
from collections import deque
import numpy as np

_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(int, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(int(x) for x in a.flat)


def _neighbors(p):
    return [(p[0] + dr, p[1] + dc) for dr, dc in _DIRECTIONS]


def _trace_component(cells, grid):
    adjacency = {p: [q for q in _neighbors(p) if q in cells] for p in cells}
    ends = sorted(p for p in cells if len(adjacency[p]) == 1)
    if (len(ends) != 2 or ends[0][0] == ends[1][0]
            or any(len(q) not in (1, 2) for q in adjacency.values())):
        raise ValueError("each observation must be a simple path with unequal endpoint rows")
    path, previous, current = [], None, ends[0]
    while current is not None:
        path.append(current)
        following = [q for q in adjacency[current] if q != previous]
        previous, current = current, following[0] if following else None
        if len(path) > len(cells):
            raise ValueError("cycle")
    if len(path) != len(cells) or not 4 <= len(path) <= 30:
        raise ValueError("incomplete or undersized observation")
    return path, tuple(int(grid[p]) for p in path)


def _extract(grid):
    values, counts = np.unique(grid, return_counts=True)
    if len(values) < 2 or int(counts.max()) * 2 <= grid.size:
        raise ValueError("background must have a strict majority")
    bg = int(values[counts.argmax()])
    unseen = {tuple(int(x) for x in p) for p in np.argwhere(grid != bg)}
    records = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        group, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for q in _neighbors(p):
                if q in unseen:
                    unseen.remove(q)
                    group.add(q)
                    stack.append(q)
        path, seq = _trace_component(group, grid)
        records.append({"path": path, "sequence": seq})
    if not 2 <= len(records) <= 8:
        raise ValueError("expected two to eight observations")
    return bg, records


def _words(sequence):
    words = {}
    for i in range(len(sequence) - 2):
        word = tuple(sequence[i:i + 3])
        if word in words:
            raise ValueError("a three-cell word occurs at two tape positions")
        words[word] = i
    return words


def _has_recurrence(sequence):
    runs = [c for i, c in enumerate(sequence) if i == 0 or c != sequence[i - 1]]
    return len(set(runs)) < len(runs)


def _pair_deltas(a, b, width):
    return {i - j for i in range(len(a) - width + 1)
            for j in range(len(b) - width + 1)
            if tuple(a[i:i + width]) == tuple(b[j:j + width])}


def _register(sequences):
    """Translate in path-index coordinates, using every shared ordered word."""
    words = [_words(seq) for seq in sequences]
    links = [[] for _ in sequences]
    edge_count = 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            shared = words[i].keys() & words[j].keys()
            if shared:
                deltas = {words[i][w] - words[j][w] for w in shared}
                if len(deltas) != 1:
                    raise ValueError("inconsistent repeated words")
                delta = next(iter(deltas))
                links[i].append((j, delta))
                links[j].append((i, -delta))
                edge_count += 1
    offsets, queue = {0: 0}, deque([0])
    while queue:
        i = queue.popleft()
        for j, delta in links[i]:
            candidate = offsets[i] + delta
            if j in offsets:
                if offsets[j] != candidate:
                    raise ValueError("inconsistent registration cycle")
            else:
                offsets[j] = candidate
                queue.append(j)
    if len(offsets) != len(sequences):
        raise ValueError("shared-word graph must connect all observations")
    origin = min(offsets.values())
    starts = [offsets[i] - origin for i in range(len(sequences))]
    length = max(s + len(seq) for s, seq in zip(starts, sequences))
    if not 4 <= length <= 30:
        raise ValueError("reconciled tape length is outside ARC bounds")
    tape, counts = [None] * length, [0] * length
    for start, sequence in zip(starts, sequences):
        for i, color in enumerate(sequence):
            position = start + i
            if tape[position] is not None and tape[position] != color:
                raise ValueError("colors disagree in an overlap")
            tape[position] = color
            counts[position] += 1
    if any(x is None for x in tape):
        raise ValueError("unobserved tape gap")
    _words(tape)  # Enforce global word uniqueness, not only within each view.
    return tape, counts, starts, edge_count


def _analyze(input_grid):
    grid = _grid(input_grid)
    bg, records = _extract(grid)
    tape, counts, starts, edges = _register([r["sequence"] for r in records])
    answer = np.full((len(tape), max(counts)), bg, dtype=int)
    for i, (color, count) in enumerate(zip(tape, counts)):
        answer[i, :count] = color
    return answer, {"background": bg, "records": records, "tape": tape,
                    "counts": counts, "starts": starts, "edges": edges}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only graph traversal, shared-word registration, and depth readout."""
    return _analyze(input_grid)[0]


def _sweeps(path, sequence):
    colors = dict(zip(path, sequence))
    columns = sorted({p[1] for p in path})
    orders = [sorted(path), sorted(path, key=lambda p: (p[1], p[0]))]
    for cs in (columns, columns[::-1]):
        for reverse_last in (False, True):
            orders.append([p for c in cs for p in sorted(
                [q for q in path if q[1] == c],
                reverse=reverse_last and c == cs[-1])])
    return any(tuple(colors[p] for p in order) == tuple(sequence) for order in orders)


def _walk(rng, length, above=False, witness=False, sequence=None):
    """Independent induced lattice walk; no physical shape is a registration key."""
    for _ in range(100):
        path, occupied, previous = [(0, 0)], {(0, 0)}, None
        persistence = float(rng.choice((0.5, 1.5, 5.0)))
        for _ in range(length - 1):
            options, weights = [], []
            for direction in _DIRECTIONS:
                q = (path[-1][0] + direction[0], path[-1][1] + direction[1])
                if q in occupied or sum(p in occupied for p in _neighbors(q)) != 1:
                    continue
                options.append((q, direction))
                weights.append(persistence if direction == previous else 1.)
            if not options:
                break
            probabilities = np.asarray(weights) / sum(weights)
            q, previous = options[int(rng.choice(len(options), p=probabilities))]
            path.append(q)
            occupied.add(q)
        if len(path) != length or path[0][0] == path[-1][0]:
            continue
        if path[0][0] > path[-1][0]:
            path.reverse()
        if max(p[0] for p in path) - min(p[0] for p in path) >= 13:
            continue
        if max(p[1] for p in path) - min(p[1] for p in path) >= 13:
            continue
        if above and min(p[0] for p in path) >= path[0][0]:
            continue
        if witness and (tuple(sequence) == tuple(sequence)[::-1] or _sweeps(path, sequence)):
            continue
        return path
    # Constructive induced paths with different row order and path order.
    for reflected in (False, True):
        if above:
            segments = [((-1, 0), 1), ((0, 1), 3), ((1, 0), length - 7), ((0, -1), 2)]
        elif witness:
            segments = [((0, 1), 2), ((1, 0), 2), ((0, -1), 2), ((1, 0), length - 7)]
        else:
            segments = [((1, 0), max(1, length // 2)), ((0, 1), length - 1 - max(1, length // 2))]
        path = [(0, 0)]
        for (dr, dc), count in segments:
            for _ in range(count):
                path.append((path[-1][0] + dr, path[-1][1] + dc))
        if reflected:
            path = [(r, -c) for r, c in path]
        if path[0][0] == path[-1][0]:
            continue
        if not witness or (tuple(sequence) != tuple(sequence)[::-1] and not _sweeps(path, sequence)):
            return path
    raise ValueError("could not construct distinguishing fold for this sequence")


def _sequence(rng, length, palette):
    # Greedy growth with bounded retries; independent of intervals and bends.
    for _ in range(40):
        seq, used = [], set()
        style = int(rng.integers(3))
        for i in range(length):
            options = [c for c in palette if i < 2 or tuple(seq[-2:] + [c]) not in used]
            if not options:
                break
            weights = np.ones(len(options))
            if seq and style == 1:
                weights = np.asarray([4. if c == seq[-1] else 1. for c in options])
            elif i >= 2 and style == 2:
                weights = np.asarray([4. if c == seq[-2] else 1. for c in options])
            c = int(rng.choice(options, p=weights / weights.sum()))
            seq.append(c)
            if i >= 2:
                used.add(tuple(seq[-3:]))
        if len(seq) == length and seq != seq[::-1]:
            return seq
    # Euler traversal of the order-three word graph provides a finite fallback.
    adjacency = {(a, b): list(rng.permutation(palette)) for a in palette for b in palette}
    stack, route = [(palette[0], palette[0])], []
    while stack:
        node = stack[-1]
        if adjacency[node]:
            c = int(adjacency[node].pop())
            stack.append((node[1], c))
        else:
            route.append(stack.pop())
    route.reverse()
    complete = list(route[0]) + [node[1] for node in route[1:]]
    start = int(rng.integers(len(complete) - length + 1))
    return complete[start:start + length]


def _intervals(rng, length, chains, extra, seam_view=False):
    for _ in range(40):
        minimum = 5 if chains == 2 else 4
        sizes = minimum + rng.multinomial(length - minimum * chains, np.ones(chains) / chains)
        cuts = [0] + list(np.cumsum(sizes))
        left = [0] + [int(rng.integers(1, 4)) for _ in range(chains - 1)]
        right = [int(rng.integers(2, 4)) for _ in range(chains - 1)] + [0]
        if seam_view:
            left[1], right[0] = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        result = [(max(0, int(cuts[i]) - left[i]), min(length, int(cuts[i + 1]) + right[i]))
                  for i in range(chains)]
        if max(b - a for a, b in result) <= 18:
            break
    else:
        raise ValueError("interval construction failed")
    for i in range(extra):
        if seam_view and i == 0:
            result.append((result[1][0], result[0][1]))
        elif rng.random() < .22:
            result.append(result[int(rng.integers(len(result)))])
        else:
            # Strictly contained in an existing view; arbitrary observation length.
            a, b = result[int(rng.integers(len(result)))]
            sublength = int(rng.integers(4, b - a + 1))
            start = int(rng.integers(a, b - sublength + 1))
            result.append((start, start + sublength))
    return result


def _pack(rng, paths, sequences, bg):
    parts = []
    for path, seq in zip(paths, sequences):
        r0, c0 = min(p[0] for p in path), min(p[1] for p in path)
        cells = {(r - r0, c - c0): color for (r, c), color in zip(path, seq)}
        parts.append((cells, max(p[0] for p in cells) + 1, max(p[1] for p in cells) + 1))
    total = sum(len(p[0]) for p in parts)
    min_h, min_w = max(p[1] for p in parts), max(p[2] for p in parts)
    for attempt in range(36):
        if attempt >= 28:
            h = w = 30
        else:
            h = int(rng.integers(max(min_h, 8), 31))
            w = int(rng.integers(max(min_w, 8), 31))
        if h * w <= 2 * total:
            continue
        grid = np.full((h, w), bg, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        order = list(rng.permutation(len(parts)))
        order.sort(key=lambda i: -len(parts[i][0]))
        for index in order:
            cells, ph, pw = parts[index]
            choices = []
            for top in range(h - ph + 1):
                for left in range(w - pw + 1):
                    if all(not blocked[r + top, c + left] for r, c in cells):
                        choices.append((top, left))
            if not choices:
                break
            top, left = choices[int(rng.integers(len(choices)))]
            for (r, c), color in cells.items():
                rr, cc = r + top, c + left
                grid[rr, cc] = color
                blocked[max(0, rr - 1):min(h, rr + 2), max(0, cc - 1):min(w, cc + 2)] = True
        else:
            return grid
    raise ValueError("observations did not fit")


def _pair(rng, target, index, fixed_bg):
    if not target and index == 0:
        length, chains, extra = int(rng.integers(11, 17)), 2, 0
    elif not target and index == 1:
        length, chains, extra = int(rng.integers(15, 20)), 2, 1
    elif target:
        length = int(rng.integers(20, 29))
        chains = int(rng.integers(3, 5))
        extra = int(rng.integers(1, 7 - chains))
    else:
        length = int(rng.integers(12, 24))
        chains = int(rng.integers(2, 4))
        extra = int(rng.integers(0, 3))
    bg = int(rng.integers(10)) if fixed_bg is None else fixed_bg
    palette_high = 6 if not target and index == 0 else 9
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != bg])[:int(rng.integers(3, palette_high))]]
    for _ in range(40):
        try:
            tape = _sequence(rng, length, palette)
            intervals = _intervals(rng, length, chains, extra, seam_view=(index == 1 and not target))
            snippets = [tape[a:b] for a, b in intervals]
            if (index == 0 or target) and not _has_recurrence(tape):
                continue
            if not target and index == 0 and len(_pair_deltas(snippets[0], snippets[1], 2)) < 2:
                continue
            longest = max(range(len(snippets)), key=lambda i: len(snippets[i]))
            need_above = not target and index == 1
            if need_above and len(snippets[longest]) < 9:
                continue
            paths = [_walk(rng, len(seq), above=(need_above and i == longest),
                           witness=(i == longest), sequence=seq) for i, seq in enumerate(snippets)]
            grid = _pack(rng, paths, snippets, bg)
            # Label is made from the construction intervals, then pixel reconstruction
            # must recover both the tape and each observation's coverage.
            depth = [sum(a <= i < b for a, b in intervals) for i in range(length)]
            output = np.full((length, max(depth)), bg, dtype=int)
            for i, color in enumerate(tape):
                output[i, :depth[i]] = color
            if not np.array_equal(solve(grid), output):
                continue
            return {"input": grid, "output": output}
        except ValueError:
            continue
    raise RuntimeError("bounded observation construction exhausted")


def _check_counts(num_train, num_test):
    for name, count, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not low <= count <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    fixed_bg = int(rng.integers(10)) if rng.random() < .5 else None
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(12):
                pair = _pair(rng, split == "test", index, fixed_bg)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            expected, info = _analyze(inp)
            assert np.array_equal(out, expected), "incorrect tape, overlap counts, or padding"
            assert out.shape[1] >= 2
            # Exposure conservation, unlike ordinary union or concatenation.
            assert sum(info["counts"]) == sum(len(r["sequence"]) for r in info["records"])
            assert len(info["tape"]) < sum(info["counts"]), "no registration effect"
            assert set(out.flat) - {info["background"]} == set(inp.flat) - {info["background"]}
            infos[split].append(info)
    # Training evidence applies even to minimal 2/1 puzzles.
    assert any(max(i["counts"]) >= 3 for i in infos["train"]), "containment/depth not demonstrated"
    assert any(_has_recurrence(i["tape"]) for i in infos["train"]), "separated repeated colors not taught"
    assert any(len(_pair_deltas(a["sequence"], b["sequence"], 2)) > 1
               for i in infos["train"] for n, a in enumerate(i["records"])
               for b in i["records"][n + 1:]), "two-symbol false registration not distinguished"
    for split in ("train", "test"):
        assert any(any(not _sweeps(r["path"], r["sequence"]) for r in i["records"]) for i in infos[split]), "traversal shortcut not distinguished"
    if any(min(p[0] for p in r["path"]) < r["path"][0][0] for i in infos["test"] for r in i["records"]):
        assert any(min(p[0] for p in r["path"]) < r["path"][0][0]
                   and r["sequence"] != r["sequence"][::-1]
                   for i in infos["train"] for r in i["records"]), "upper endpoint not taught"
    return None


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    pairs = [pair for p in puzzles for s in ("train", "test") for pair in p[s]]
    infos = [_analyze(pair["input"])[1] for pair in pairs]
    assert len({tuple(_key(x[k]) for s in ("train", "test") for x in p[s] for k in ("input", "output")) for p in puzzles}) == 42
    assert len({_key(x["input"]) for x in pairs}) == len(pairs)
    assert len({x["input"].shape for x in pairs}) >= 50
    assert len({x["output"].shape for x in pairs}) >= 25
    assert len({len(i["records"]) for i in infos}) >= 5
    assert len({i["background"] for i in infos}) == 10
    assert len({tuple(i["counts"]) for i in infos}) >= 100
    geometries = set()
    incompatible = 0
    for info in infos:
        for record in info["records"]:
            path = record["path"]
            origin = path[0]
            geometries.add(tuple((r - origin[0], c - origin[1]) for r, c in path))
        # A shared word often has different bends in its different observations.
        motifs = {}
        for record in info["records"]:
            path, seq = record["path"], record["sequence"]
            for j in range(len(seq) - 2):
                steps = tuple((path[t + 1][0] - path[t][0], path[t + 1][1] - path[t][1]) for t in (j, j + 1))
                motifs.setdefault(seq[j:j + 3], set()).add(steps)
        incompatible += any(len(shapes) > 1 for shapes in motifs.values())
    assert len(geometries) >= 200
    assert incompatible >= 150, "planar-registration shortcut insufficiently challenged"
    assert any(max(i["counts"]) >= 5 for i in infos)
    assert _key(generate_puzzle(17)["test"][0]["input"]) == _key(puzzles[17]["test"][0]["input"])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(901, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Registration unit fixture: a contained observation changes only thickness.
    tape = [1, 2, 1, 3, 2, 4, 3, 1, 4, 2]
    full, counts, _, _ = _register([tape[:7], tape[4:]])
    redundant, deeper, _, _ = _register([tape[:7], tape[4:], tape[3:7]])
    assert full == redundant == tape and sum(deeper) - sum(counts) == 4
    assert max(deeper) == 3
    for sequences in (([1, 2, 3, 4], [5, 6, 7, 8]), ([1, 1, 1, 1], [1, 1, 1, 2])):
        try:
            _register(sequences)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported ambiguous registration accepted")
    for cells in ([(1, 1), (1, 2), (1, 3), (1, 4)],
                  [(1, 2), (2, 1), (2, 2), (2, 3)],
                  [(1, 1), (1, 2), (2, 1), (2, 2)]):
        bad_grid = np.zeros((6, 6), dtype=int)
        for point in cells:
            bad_grid[point] = 1
        try:
            _extract(bad_grid)
        except ValueError:
            pass
        else:
            raise AssertionError("tied endpoints, branch, or cycle accepted")
    corrupted = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    corrupted["test"][0]["output"][0, 0] = (corrupted["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad output accepted")
    return None
